// Shared 128B-aligned Planar SoA Tile layout constants & magic-bias decoders
// for Vulkan_Repack (Q4_K, Q5_K, Q6_K).
//
// Each 4-superblock (1024-weight) tile contains:
//   - Scales plane (128 B = 32 uint32 = 8 uvec4):
//       Q4_K / Q5_K: subblock s in 0..7 -> f16vec2(d*sc, -dmin*m) at k*8 + s
//       Q6_K:        subblock pair p in 0..7 -> f16vec2(d*sc_{2p}, d*sc_{2p+1}) at k*8 + p
//   - qs/ql interleaved plane (512 B = 128 uint32 = 32 uvec4):
//       lane tid = k*8 + itid8 (k=0..3, itid8=0..7):
//       uvec4(qs[itid8], qs[itid8+8], qs[itid8+16], qs[itid8+24])
//   - qh plane (starts at byte 640 = uint32 offset 160):
//       Q4_K: 0 B     (tile = 640 B = 160 u32 = 40 uvec4)
//       Q5_K: 128 B   (32 uint32: qh[itid8] at 160 + k*8 + itid8; tile = 768 B = 192 u32 = 48 uvec4)
//       Q6_K: 256 B   (32 uvec2: (qh[itid8], qh[itid8+8]) at 160 + 2*tid; tile = 896 B = 224 u32 = 56 uvec4)

#define REPACK_SCALES_OFF_U32 0u
#define REPACK_QS_OFF_U4      8u
#define REPACK_QS_OFF_U32     32u
#define REPACK_QH_OFF_U32     160u

#define REPACK_Q4_K_TILE_U4   40u
#define REPACK_Q4_K_TILE_U32  160u
#define REPACK_Q5_K_TILE_U4   48u
#define REPACK_Q5_K_TILE_U32  192u
#define REPACK_Q6_K_TILE_U4   56u
#define REPACK_Q6_K_TILE_U32  224u

vec4 repack_magic_u4(const uint q4) {
    const f16vec2 p02 = unpackFloat2x16((q4 & 0x000F000Fu) | 0x64006400u) - f16vec2(1024.0);
    const f16vec2 p13 = unpackFloat2x16(((q4 >> 8u) & 0x000F000Fu) | 0x64006400u) - f16vec2(1024.0);
    return vec4(float(p02.x), float(p13.x), float(p02.y), float(p13.y));
}

f16vec4 repack_magic_u4_h(const uint q4) {
    const f16vec2 p02 = unpackFloat2x16((q4 & 0x000F000Fu) | 0x64006400u) - f16vec2(1024.0);
    const f16vec2 p13 = unpackFloat2x16(((q4 >> 8u) & 0x000F000Fu) | 0x64006400u) - f16vec2(1024.0);
    return f16vec4(p02.x, p13.x, p02.y, p13.y);
}

vec4 repack_magic_u5(const uint q5_02, const uint q5_13) {
    const f16vec2 p02 = unpackFloat2x16((q5_02 & 0x001F001Fu) | 0x64006400u) - f16vec2(1024.0);
    const f16vec2 p13 = unpackFloat2x16((q5_13 & 0x001F001Fu) | 0x64006400u) - f16vec2(1024.0);
    return vec4(float(p02.x), float(p13.x), float(p02.y), float(p13.y));
}

f16vec4 repack_magic_u5_h(const uint q5_02, const uint q5_13) {
    const f16vec2 p02 = unpackFloat2x16((q5_02 & 0x001F001Fu) | 0x64006400u) - f16vec2(1024.0);
    const f16vec2 p13 = unpackFloat2x16((q5_13 & 0x001F001Fu) | 0x64006400u) - f16vec2(1024.0);
    return f16vec4(p02.x, p13.x, p02.y, p13.y);
}

vec4 repack_magic_q6(const uint q6_02, const uint q6_13) {
    const f16vec2 p02 = unpackFloat2x16((q6_02 & 0x003F003Fu) | 0x64006400u) - f16vec2(1056.0);
    const f16vec2 p13 = unpackFloat2x16((q6_13 & 0x003F003Fu) | 0x64006400u) - f16vec2(1056.0);
    return vec4(float(p02.x), float(p13.x), float(p02.y), float(p13.y));
}

f16vec4 repack_magic_q6_h(const uint q6_02, const uint q6_13) {
    const f16vec2 p02 = unpackFloat2x16((q6_02 & 0x003F003Fu) | 0x64006400u) - f16vec2(1056.0);
    const f16vec2 p13 = unpackFloat2x16((q6_13 & 0x003F003Fu) | 0x64006400u) - f16vec2(1056.0);
    return f16vec4(p02.x, p13.x, p02.y, p13.y);
}
