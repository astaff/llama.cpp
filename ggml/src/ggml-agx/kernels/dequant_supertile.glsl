// ============================================================================
// Pillar 2 Reusable Device Header: Branchless FTZ-Safe GPR Decoders &
// 128B Cache-Line-Aligned Super-Tile Buffer References for Apple AGX3B
// ============================================================================
// Shared by:
//   - kernels/nvfp4_vecmat.comp
//   - kernels/mxfp8_q8_vecmat.comp
//   - kernels/agx_mma_gemm.comp
//   - Pillar 3 (megakernel) & Pillar 4/5 (kv-moe) shaders via #include "dequant_supertile.glsl"
//
// Guarantees:
//   - ZERO shared-memory LUTs (zero bank conflicts)
//   - ZERO FP16 subnormals ever constructed (100% FTZ-safe on AGX3B ALU & f32matmad8x8)
//   - 0-ULP bit-exact match with agx-repack.cpp & ggml reference dequantizers

#ifndef AGX_DEQUANT_SUPERTILE_GLSL
#define AGX_DEQUANT_SUPERTILE_GLSL

// ----------------------------------------------------------------------------
// Raw 64-bit GPU VA buffer reference types (std430 / scalar, zero descriptors)
// ----------------------------------------------------------------------------

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer AgxBufF32R {
    float v[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer AgxBufF32RW {
    float v[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer AgxBufVec4R {
    vec4 v[];
};

layout(buffer_reference, scalar, buffer_reference_align = 2) readonly buffer AgxBufF16R {
    float16_t v[];
};

layout(buffer_reference, scalar, buffer_reference_align = 2) readonly buffer AgxBufU16R {
    uint16_t v[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer AgxBufU32R {
    uint v[];
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer AgxBufU32x4R {
    uvec4 v[];
};

layout(buffer_reference, scalar, buffer_reference_align = 1) readonly buffer AgxBufU8R {
    uint8_t v[];
};

// ----------------------------------------------------------------------------
// Repacked super-tile structs (exact binary match with agx-repack.h)
// ----------------------------------------------------------------------------

// AGX_NVFP4_32x16: 32 rows x 64 cols = 1152 B = 9 x 128B cache lines (0 bloat)
//   d[0..63]   : offset 0..127   (128B = 1 cache line: d[s*32 + lane])
//   qs[0..63]  : offset 128..1151 (1024B = 8 cache lines: qs[s*32 + lane] = uvec4)
struct AgxNvfp4Supertile {
    uint16_t d[64];
    uvec4    qs[64];
};

layout(buffer_reference, scalar, buffer_reference_align = 128) readonly buffer AgxBufNvfp4ST {
    AgxNvfp4Supertile st[];
};

// AGX_Q8_0_8BLK: 8 Q8_0 blocks (256 weights) = 272 B = 17 x 16B planar (0 bloat)
//   d[0..7]    : offset 0..15   (8 x FP16 scale)
//   qs[0..15]  : offset 16..271 (16 x uvec4 = 256 x int8)
struct AgxQ8_0_8Blk {
    uint16_t d[8];
    uvec4    qs[16];
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer AgxBufQ8_8Blk {
    AgxQ8_0_8Blk blk[];
};

// AGX_MXFP8_16BLK: 16 MXFP8 blocks (512 weights) = 528 B = 33 x 16B planar (0 bloat)
//   e[0..15]   : offset 0..15   (16 x UE8M0 power-of-two exponent scale)
//   qs[0..31]  : offset 16..527 (32 x uvec4 = 512 x E4M3 FP8)
struct AgxMxfp8_16Blk {
    uint8_t e[16];
    uvec4   qs[32];
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer AgxBufMxfp8_16Blk {
    AgxMxfp8_16Blk blk[];
};

// Raw GGML block fallbacks (for non-repacked AGX0 buffers)
struct AgxBlockNvfp4Raw {
    uint8_t d[4];
    uint8_t qs[32];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer AgxBufNvfp4Raw {
    AgxBlockNvfp4Raw blk[];
};

struct AgxBlockQ8_0Raw {
    uint16_t d;
    int8_t   qs[32];
};

layout(buffer_reference, scalar, buffer_reference_align = 2) readonly buffer AgxBufQ8_0Raw {
    AgxBlockQ8_0Raw blk[];
};

// ----------------------------------------------------------------------------
// 1. Branchless FTZ-Safe E2M1 -> FP16 Normal-Exponent Shift Decode
// ----------------------------------------------------------------------------
// Maps 4-bit E2M1 nibble (0..15) -> exact normal IEEE FP16:
//   {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} with sign bit from bit 3.
// NEVER constructs 0x0200 subnormal; 100% immune to FTZ on AGX3B coopmat.
float16_t agx_e2m1_to_f16(uint nib) {
    uint sign = (nib & 8u) << 12;
    uint mag  = nib & 7u;
    uint norm = 0x3800u + (mag << 9);
    uint fix1 = (mag == 1u) ? 0x3800u : norm;
    uint h    = (mag == 0u) ? 0u : fix1;
    return unpackFloat2x16(sign | h).x;
}

f16vec2 agx_e2m1_byte_to_f16vec2(uint b) {
    return f16vec2(agx_e2m1_to_f16(b & 0xFu), agx_e2m1_to_f16((b >> 4) & 0xFu));
}

f16vec4 agx_e2m1_u16_to_f16vec4(uint w16) {
    return f16vec4(
        agx_e2m1_to_f16( w16        & 0xFu),
        agx_e2m1_to_f16((w16 >> 4)  & 0xFu),
        agx_e2m1_to_f16((w16 >> 8)  & 0xFu),
        agx_e2m1_to_f16((w16 >> 12) & 0xFu)
    );
}

void agx_e2m1_u32_to_f16x8(uint w, out f16vec4 lo, out f16vec4 hi) {
    lo = agx_e2m1_u16_to_f16vec4(w & 0xFFFFu);
    hi = agx_e2m1_u16_to_f16vec4(w >> 16);
}

// ----------------------------------------------------------------------------
// 2. Branchless FTZ-Safe UE4M3 -> FP16 Decode (256/256 bit-exact, 0 LUT)
// ----------------------------------------------------------------------------
// Matches 2.0 * ggml_ue4m3_to_fp32(u) so that agx_e2m1_to_f16(n) * agx_ue4m3_to_f16(u)
// equals kvalues_fp4[n] * ggml_ue4m3_to_fp32(u) with 0 ULP error.
float16_t agx_ue4m3_to_f16(uint u) {
    uint u7 = u & 0x7Fu;
    float16_t sub_v  = unpackFloat2x16(0x2400u | (u7 << 7)).x - float16_t(0.015625);
    float16_t norm_v = unpackFloat2x16(0x2000u + (u7 << 7)).x;
    float16_t val    = (u7 < 8u) ? sub_v : norm_v;
    return (u == 0u || u == 0x7Fu) ? float16_t(0.0) : val;
}

f16vec2 agx_ue4m3_pair_to_f16vec2(uint16_t d_pair) {
    uint p = uint(d_pair);
    return f16vec2(agx_ue4m3_to_f16(p & 0xFFu), agx_ue4m3_to_f16(p >> 8));
}

// ----------------------------------------------------------------------------
// 3. Branchless Q8_0 Magic-Bias Decode (0 LUT, 0 denormals, 2 ops/weight)
// ----------------------------------------------------------------------------
// Decodes 4 packed signed int8 bytes in `w` -> exact f16vec4 via
//   as_f16(0x6400 | (b ^ 0x80)) - 1152.0hf
f16vec4 agx_q8_u32_magic_to_f16vec4(uint w) {
    uint u = w ^ 0x80808080u;
    f16vec2 p02 = unpackFloat2x16((u & 0x00FF00FFu) | 0x64006400u) - f16vec2(float16_t(1152.0));
    f16vec2 p13 = unpackFloat2x16(((u >> 8) & 0x00FF00FFu) | 0x64006400u) - f16vec2(float16_t(1152.0));
    return f16vec4(p02.x, p13.x, p02.y, p13.y);
}

// ----------------------------------------------------------------------------
// 4. Branchless OCP FP8 E4M3 -> FP16 + UE8M0 Power-of-Two Scale Decode
// ----------------------------------------------------------------------------
float16_t agx_e4m3_to_f16(uint b) {
    uint sign = (b & 0x80u) << 8;
    uint ax   = b & 0x7Fu;
    float16_t sub_v  = unpackFloat2x16(0x2400u | (ax << 7)).x - float16_t(0.015625);
    float16_t norm_v = unpackFloat2x16(0x2000u + (ax << 7)).x;
    float16_t mag_v  = (ax < 8u) ? sub_v : norm_v;
    uint mag_bits = (ax == 0u) ? 0u : ((ax == 0x7Fu) ? 0x7E00u : packFloat2x16(f16vec2(mag_v, float16_t(0.0))) & 0xFFFFu);
    return unpackFloat2x16(sign | mag_bits).x;
}

f16vec4 agx_e4m3_u32_to_f16vec4(uint w) {
    return f16vec4(
        agx_e4m3_to_f16( w        & 0xFFu),
        agx_e4m3_to_f16((w >> 8)  & 0xFFu),
        agx_e4m3_to_f16((w >> 16) & 0xFFu),
        agx_e4m3_to_f16((w >> 24) & 0xFFu)
    );
}

// UE8M0 power-of-two exponent -> exact FP32 (e==0 -> 2^-127, else 2^(e-127))
float agx_ue8m0_to_f32(uint e) {
    uint bits = (e == 0u) ? 0x00400000u : (e << 23);
    return uintBitsToFloat(bits);
}

// ----------------------------------------------------------------------------
// 5. BF16 -> FP32 Fast Bit-Shift Decode
// ----------------------------------------------------------------------------
float agx_bf16_to_f32(uint16_t b) {
    return uintBitsToFloat(uint(b) << 16);
}

vec2 agx_bf16_u32_to_vec2(uint w) {
    return vec2(uintBitsToFloat((w & 0xFFFFu) << 16), uintBitsToFloat(w & 0xFFFF0000u));
}

// ----------------------------------------------------------------------------
// 6. AGX3B 8x8 Cooperative Matrix Hardware Lane Coordinate Mapping
//    Matches mesa src/asahi/compiler/agx_nir_lower_cmat.c `layout_8` exactly:
//      row  = ((lane >> 1) & 3) | ((lane >> 2) & 4)
//      col0 = ((lane & 1) << 1) | ((lane >> 1) & 4)   (even: 0, 2, 4, 6)
//      col1 = col0 | 1                                (odd:  1, 3, 5, 7)
// ----------------------------------------------------------------------------
uint agx_cmat8_row(uint lane) {
    return ((lane >> 1) & 3u) | ((lane >> 2) & 4u);
}

uint agx_cmat8_col0(uint lane) {
    return ((lane & 1u) << 1) | ((lane >> 1) & 4u);
}

#endif // AGX_DEQUANT_SUPERTILE_GLSL
