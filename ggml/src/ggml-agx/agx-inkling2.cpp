#include "agx-inkling2.h"
#include "agx-graph-exec.h"
#include "agx-gemm.h"
#include "agx-kv-hybrid.h"
#include "agx-moe.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "ggml-impl.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================================
// Inline Utility Shaders (Embedding + RMSNorm, Standalone RMSNorm, Copy)
// ============================================================================

static const char * k_glsl_embd_rmsnorm = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

#define WTYPE_F32       0u
#define WTYPE_F16       1u
#define WTYPE_BF16      2u
#define WTYPE_Q8_0_RAW  3u
#define WTYPE_Q8_0_8BLK 4u

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer ConstBufU16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) readonly buffer ConstBufI8 { int8_t v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t embd_va;
    uint64_t tok_ids_va;
    uint64_t norm_w_va;
    uint64_t dst_va;
    uint     n_embd;
    uint     n_tokens;
    uint     wtype;
    uint     row_bytes;
    float    eps;
    uint     pad0;
} pc;

shared float s_buf[3840];
shared float s_red[8];

float load_row_val(uint64_t row_va, uint wtype, uint k) {
    if (wtype == WTYPE_F32) {
        return ConstBufF32(row_va).v[k];
    } else if (wtype == WTYPE_F16) {
        return unpackHalf2x16(uint(ConstBufU16(row_va).v[k])).x;
    } else if (wtype == WTYPE_BF16) {
        return uintBitsToFloat(uint(ConstBufU16(row_va).v[k]) << 16u);
    } else if (wtype == WTYPE_Q8_0_RAW) {
        uint blk = k >> 5u;
        uint off = k & 31u;
        uint64_t bva = row_va + uint64_t(blk) * 34ul;
        float d = unpackHalf2x16(uint(ConstBufU16(bva).v[0])).x;
        return d * float(ConstBufI8(bva + 2ul).v[off]);
    } else {
        uint chunk = k >> 8u;
        uint rem   = k & 255u;
        uint blk   = rem >> 5u;
        uint64_t cva = row_va + uint64_t(chunk) * 272ul;
        float d = unpackHalf2x16(uint(ConstBufU16(cva).v[blk])).x;
        return d * float(ConstBufI8(cva + 16ul).v[rem]);
    }
}

void main() {
    const uint tok = gl_WorkGroupID.x;
    if (tok >= pc.n_tokens) return;
    const uint tid     = gl_LocalInvocationID.x;
    const uint sg_id   = tid >> 5u;
    const uint lane_id = tid & 31u;
    const uint D       = pc.n_embd;

    const int tid_val = max(ConstBufI32(pc.tok_ids_va).v[tok], 0);
    const uint64_t row_va = pc.embd_va + uint64_t(tid_val) * uint64_t(pc.row_bytes);
    const uint64_t dst_va = pc.dst_va + uint64_t(tok) * uint64_t(D) * 4ul;

    float ss = 0.0;
    for (uint i = tid; i < D; i += 256u) {
        float v = load_row_val(row_va, pc.wtype, i);
        if (pc.norm_w_va != 0ul) {
            s_buf[i] = v;
            ss += v * v;
        } else {
            BufF32(dst_va).v[i] = v;
        }
    }
    if (pc.norm_w_va == 0ul) return;

    float s = subgroupAdd(ss);
    if (lane_id == 0u) s_red[sg_id] = s;
    barrier();
    if (sg_id == 0u) {
        float v = (lane_id < 8u) ? s_red[lane_id] : 0.0;
        float tot = subgroupAdd(v);
        if (lane_id == 0u) s_red[0] = inversesqrt(tot / float(D) + pc.eps);
    }
    barrier();
    float inv_rms = s_red[0];
    for (uint i = tid; i < D; i += 256u) {
        BufF32(dst_va).v[i] = (s_buf[i] * inv_rms) * ConstBufF32(pc.norm_w_va).v[i];
    }
}
)GLSL";

static const char * k_glsl_rmsnorm_f32 = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufF16 { float16_t v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t w_va;
    uint64_t dst_va;
    uint     n_embd;
    uint     n_rows;
    float    eps;
    uint     pad0;
    uint64_t dst_f16_va;
} pc;

shared float s_red[8];

void main() {
    const uint row = gl_WorkGroupID.x;
    if (row >= pc.n_rows) return;
    const uint tid     = gl_LocalInvocationID.x;
    const uint sg_id   = tid >> 5u;
    const uint lane_id = tid & 31u;
    const uint D       = pc.n_embd;

    const uint64_t s_va   = pc.src_va + uint64_t(row) * uint64_t(D) * 4ul;
    const uint64_t d_va   = pc.dst_va + uint64_t(row) * uint64_t(D) * 4ul;
    const uint64_t d16_va = (pc.dst_f16_va != 0ul) ? (pc.dst_f16_va + uint64_t(row) * uint64_t(D) * 2ul) : 0ul;

    float ss = 0.0;
    for (uint i = tid; i < D; i += 256u) {
        float x = ConstBufF32(s_va).v[i];
        ss += x * x;
    }
    float s = subgroupAdd(ss);
    if (lane_id == 0u) s_red[sg_id] = s;
    barrier();
    if (sg_id == 0u) {
        float v = (lane_id < 8u) ? s_red[lane_id] : 0.0;
        float tot = subgroupAdd(v);
        if (lane_id == 0u) s_red[0] = inversesqrt(tot / float(D) + pc.eps);
    }
    barrier();
    float inv_rms = s_red[0];
    if (pc.w_va != 0ul) {
        for (uint i = tid; i < D; i += 256u) {
            float ov = (ConstBufF32(s_va).v[i] * inv_rms) * ConstBufF32(pc.w_va).v[i];
            BufF32(d_va).v[i] = ov;
            if (d16_va != 0ul) BufF16(d16_va).v[i] = float16_t(ov);
        }
    } else {
        for (uint i = tid; i < D; i += 256u) {
            float ov = ConstBufF32(s_va).v[i] * inv_rms;
            BufF32(d_va).v[i] = ov;
            if (d16_va != 0ul) BufF16(d16_va).v[i] = float16_t(ov);
        }
    }
}
)GLSL";

static const char * k_glsl_copy_f32 = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t dst_va;
    uint     n_elems;
    uint     pad0;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.n_elems) {
        BufF32(pc.dst_va).v[i] = ConstBufF32(pc.src_va).v[i];
    }
}
)GLSL";

struct embd_rmsnorm_push {
    uint64_t embd_va;
    uint64_t tok_ids_va;
    uint64_t norm_w_va;
    uint64_t dst_va;
    uint32_t n_embd;
    uint32_t n_tokens;
    uint32_t wtype;
    uint32_t row_bytes;
    float    eps;
    uint32_t pad0;
};

struct rmsnorm_f32_push {
    uint64_t src_va;
    uint64_t w_va;
    uint64_t dst_va;
    uint32_t n_embd;
    uint32_t n_rows;
    float    eps;
    uint32_t pad0;
    uint64_t dst_f16_va;
};
static_assert(sizeof(rmsnorm_f32_push) == 48, "rmsnorm_f32_push must be 48B");

struct copy_f32_push {
    uint64_t src_va;
    uint64_t dst_va;
    uint32_t n_elems;
    uint32_t pad0;
};

static const char * k_glsl_w4_router_topk_norm = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufI32 { int v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t logits_va;
    uint64_t bias_va;
    uint64_t ids_dst_va;
    uint64_t w_dst_va;
    uint     n_exp;
    uint     k_out;
    uint     n_tokens;
    uint     ids_nb1;
    float    norm_eps;
    float    w_scale;
} pc;

shared float s_sel[256];
shared float s_prob[256];
shared int   s_idx[256];

void main() {
    const uint tok = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    if (tok >= pc.n_tokens) return;

    const uint E = pc.n_exp;
    uint n_pad = 1u;
    while (n_pad < E) n_pad <<= 1u;
    const uint l_base = tok * E;

    for (uint i = tid; i < n_pad; i += 128u) {
        if (i < E) {
            float x = ConstBufF32(pc.logits_va).v[l_base + i];
            float p = 1.0 / (1.0 + exp(-x));
            float s = (pc.bias_va != 0ul) ? (p + ConstBufF32(pc.bias_va).v[i]) : p;
            s_prob[i] = p;
            s_sel[i]  = s;
            s_idx[i]  = int(i);
        } else {
            s_prob[i] = 0.0;
            s_sel[i]  = uintBitsToFloat(0xFF800000u);
            s_idx[i]  = 0x7FFFFFFF;
        }
    }
    barrier();

    const uint half_n = n_pad >> 1u;
    for (uint k = 2u; k <= n_pad; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            if (tid < half_n) {
                uint low_mask = j - 1u;
                uint lo = ((tid & ~low_mask) << 1u) | (tid & low_mask);
                uint hi = lo | j;
                bool asc_dir = ((lo & k) == 0u);
                float va = s_sel[lo]; int ia = s_idx[lo];
                float vb = s_sel[hi]; int ib = s_idx[hi];
                bool a_before = (va > vb) || (va == vb && ia < ib);
                if (asc_dir ? (!a_before) : a_before) {
                    s_sel[lo] = vb; s_idx[lo] = ib;
                    s_sel[hi] = va; s_idx[hi] = ia;
                }
            }
            barrier();
        }
    }

    const uint K = pc.k_out;
    if (tid == 0u) {
        float sum_p = 0.0;
        for (uint k = 0u; k < K; ++k) {
            sum_p += s_prob[uint(s_idx[k])];
        }
        float inv_s = 1.0 / (sum_p + pc.norm_eps);
        const uint id_base = tok * pc.ids_nb1;
        const uint w_base  = tok * K;
        for (uint k = 0u; k < K; ++k) {
            int eid = s_idx[k];
            BufI32(pc.ids_dst_va).v[id_base + k] = eid;
            BufF32(pc.w_dst_va).v[w_base + k]    = (s_prob[uint(eid)] * inv_s) * pc.w_scale;
        }
    }
}
)GLSL";

struct w4_router_topk_push {
    uint64_t logits_va;
    uint64_t bias_va;
    uint64_t ids_dst_va;
    uint64_t w_dst_va;
    uint32_t n_exp;
    uint32_t k_out;
    uint32_t n_tokens;
    uint32_t ids_nb1;
    float    norm_eps;
    float    w_scale;
};

static const char * k_glsl_w4_gather_f32 = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufF16 { float16_t v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t ids_va;
    uint64_t dst_va;
    uint64_t dst2_va;
    uint     n_embd;
    uint     n_out;
    uint64_t dst_f16_va;
} pc;

void main() {
    const uint col = gl_GlobalInvocationID.x;
    const uint row = gl_WorkGroupID.y;
    if (col >= pc.n_embd || row >= pc.n_out) return;
    const int src_row = max(ConstBufI32(pc.ids_va).v[row], 0);
    const float val = ConstBufF32(pc.src_va).v[uint(src_row) * pc.n_embd + col];
    BufF32(pc.dst_va).v[row * pc.n_embd + col] = val;
    if (pc.dst2_va != 0ul) {
        BufF32(pc.dst2_va).v[row * pc.n_embd + col] = val;
    }
    if (pc.dst_f16_va != 0ul) {
        BufF16(pc.dst_f16_va).v[row * pc.n_embd + col] = float16_t(val);
    }
}
)GLSL";

struct w4_gather_f32_push {
    uint64_t src_va;
    uint64_t ids_va;
    uint64_t dst_va;
    uint64_t dst2_va;
    uint32_t n_embd;
    uint32_t n_out;
    uint64_t dst_f16_va;
};
static_assert(sizeof(w4_gather_f32_push) == 48, "w4_gather_f32_push must be 48B");

// Wave 10: O(n_kv) Banded Rel-Bias Table Projection (64T = 2 SGs, 3 KB SRAM -> 100% coalesced 128B stores!)
// Loads rel_bias[e, 0..n_rel-1] ONCE per (e, tok), multiplies all 24 heads in registers -> rel_table[tok, h, e]!
static const char * k_glsl_w10_rel_bias_table = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4)  buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4)  readonly buffer ConstBufF32   { float    v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufVec4  { vec4     v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4)  readonly buffer ConstBufU32   { uint     v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8)  readonly buffer ConstBufUvec2 { uvec2    v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufUvec4 { uvec4    v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2)  readonly buffer ConstBufU16   { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1)  readonly buffer ConstBufI8    { int8_t   v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t r_va;
    uint64_t rel_bias_va;
    uint64_t dst_va;
    uint64_t sinks_src_va;
    uint64_t sinks_dst_va;
    uint     n_rel;
    uint     eff_rows;
    uint     n_tokens;
    uint     bias_wtype;
    uint     bias_nb1_B;
    uint     sinks_stride_f32;
} pc;

shared vec4 sh_r[24][8]; // 3,072 B: 24 heads x 32 floats (zero-padded for n_rel < 32)

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint tok = gl_WorkGroupID.y;
    if (tok >= pc.n_tokens) return;

    const uint n_rel = min(pc.n_rel, 32u);
    ConstBufF32 rb = ConstBufF32(pc.r_va);
    for (uint idx = tid; idx < 24u * 8u; idx += 64u) {
        const uint h = idx >> 3u;
        const uint g = idx & 7u;
        const uint k = g << 2u;
        const uint base = (tok * 24u + h) * pc.n_rel + k;
        vec4 rv = vec4(0.0);
        if (k + 0u < n_rel) rv.x = rb.v[base + 0u];
        if (k + 1u < n_rel) rv.y = rb.v[base + 1u];
        if (k + 2u < n_rel) rv.z = rb.v[base + 2u];
        if (k + 3u < n_rel) rv.w = rb.v[base + 3u];
        sh_r[h][g] = rv;
    }
    if (gl_WorkGroupID.x == 0u && tid < 24u && pc.sinks_dst_va != 0ul) {
        BufF32(pc.sinks_dst_va).v[tok * 24u + tid] =
            ConstBufF32(pc.sinks_src_va).v[tok * pc.sinks_stride_f32 + tid];
    }
    barrier();

    const uint e = gl_WorkGroupID.x * 64u + tid;
    if (e >= pc.eff_rows) return;

    vec4 w[8];
    [[unroll]] for (uint g = 0u; g < 8u; ++g) w[g] = vec4(0.0);

    const uint64_t b_va = pc.rel_bias_va + uint64_t(e) * uint64_t(pc.bias_nb1_B);
    const uint wt = pc.bias_wtype;
    if (wt == 2u) {
        // BF16 (Inkling-2-Mini default!): 4 x 16B uvec4 loads for n_rel = 32!
        const uint n8 = n_rel >> 3u;
        ConstBufUvec4 bb = ConstBufUvec4(b_va);
        for (uint p_i = 0u; p_i < n8; ++p_i) {
            uvec4 p = bb.v[p_i];
            w[p_i * 2u + 0u] = vec4(
                uintBitsToFloat((p.x & 0xFFFFu) << 16u),
                uintBitsToFloat(p.x & 0xFFFF0000u),
                uintBitsToFloat((p.y & 0xFFFFu) << 16u),
                uintBitsToFloat(p.y & 0xFFFF0000u));
            w[p_i * 2u + 1u] = vec4(
                uintBitsToFloat((p.z & 0xFFFFu) << 16u),
                uintBitsToFloat(p.z & 0xFFFF0000u),
                uintBitsToFloat((p.w & 0xFFFFu) << 16u),
                uintBitsToFloat(p.w & 0xFFFF0000u));
        }
        if ((n_rel & 7u) != 0u) {
            const uint rem0 = n8 * 8u;
            for (uint k = rem0; k < n_rel; ++k) {
                uint u = uint(ConstBufU16(b_va).v[k]);
                w[k >> 2u][k & 3u] = uintBitsToFloat(u << 16u);
            }
        }
    } else if (wt == 1u) {
        const uint n8 = n_rel >> 3u;
        ConstBufUvec4 bb = ConstBufUvec4(b_va);
        for (uint p_i = 0u; p_i < n8; ++p_i) {
            uvec4 p = bb.v[p_i];
            w[p_i * 2u + 0u] = vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y));
            w[p_i * 2u + 1u] = vec4(unpackHalf2x16(p.z), unpackHalf2x16(p.w));
        }
        if ((n_rel & 7u) != 0u) {
            const uint rem0 = n8 * 8u;
            for (uint k = rem0; k < n_rel; ++k) {
                uint u = uint(ConstBufU16(b_va).v[k]);
                w[k >> 2u][k & 3u] = unpackHalf2x16(u).x;
            }
        }
    } else if (wt == 0u) {
        const uint n4 = n_rel >> 2u;
        ConstBufVec4 bb = ConstBufVec4(b_va);
        for (uint g = 0u; g < n4; ++g) w[g] = bb.v[g];
        if ((n_rel & 3u) != 0u) {
            for (uint k = n4 * 4u; k < n_rel; ++k) w[k >> 2u][k & 3u] = ConstBufF32(b_va).v[k];
        }
    } else if (wt == 4u) {
        // Q8_0_8BLK (272B chunk: d[8] at +0, qs[256] at +16; n_rel <= 32 -> block 0 only!)
        float ds = unpackHalf2x16(uint(ConstBufU16(b_va).v[0])).x;
        for (uint g = 0u; g < 8u; ++g) {
            if (g * 4u < n_rel) {
                uint pk = ConstBufU32(b_va + 16ul).v[g];
                w[g] = vec4(
                    float(bitfieldExtract(int(pk),  0, 8)),
                    float(bitfieldExtract(int(pk),  8, 8)),
                    float(bitfieldExtract(int(pk), 16, 8)),
                    float(bitfieldExtract(int(pk), 24, 8))) * ds;
            }
        }
    } else {
        // Q8_0_RAW (34B block)
        float ds = unpackHalf2x16(uint(ConstBufU16(b_va).v[0])).x;
        for (uint k = 0u; k < n_rel; ++k) {
            w[k >> 2u][k & 3u] = float(ConstBufI8(b_va + 2ul).v[k]) * ds;
        }
    }

    BufF32 ob = BufF32(pc.dst_va);
    const uint n4_eff = (n_rel + 3u) >> 2u;
    const uint base_t = tok * 24u * pc.eff_rows + e;
    for (uint h = 0u; h < 24u; ++h) {
        float acc = 0.0;
        for (uint g = 0u; g < n4_eff; ++g) {
            acc += dot(w[g], sh_r[h][g]);
        }
        ob.v[base_t + h * pc.eff_rows] = acc;
    }
}
)GLSL";

// Wave 10: Lazy-compiled FWHT64-capable variant of agx_inkling2_qkv_prep_store.comp
// Matches struct agx_w4_qkv_prep_desc verbatim; adds QKV_FLAG_FWHT64_K (1<<9) & QKV_FLAG_FWHT64_V (1<<10)
// so -ctk q8_0 -ctv q8_0 (n_rot_k=128, n_rot_v=64) writes exact H_64-rotated V vectors into KV cache!
static const char * k_glsl_w10_qkv_prep_store_rot64 = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufVec4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer ConstBufI64 { int64_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufU32 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer ConstBufUvec2 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufF16 { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer ConstBufF16 { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufU16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer ConstBufU16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) readonly buffer ConstBufI8 { int8_t v[]; };

#define QKV_FLAG_IS_FULL_ATTN   (1u << 0)
#define QKV_FLAG_FWHT_K         (1u << 1)
#define QKV_FLAG_FWHT_V         (1u << 2)
#define QKV_FLAG_KV_Q8_0        (1u << 3)
#define QKV_FLAG_K_IDS_I64      (1u << 4)
#define QKV_FLAG_V_IDS_I64      (1u << 5)
#define QKV_FLAG_BASE_MASK_F16  (1u << 6)
#define QKV_FLAG_FUSE_REL_MASK  (1u << 7)
#define QKV_FLAG_PACK_QGATE     (1u << 8)
#define QKV_FLAG_FWHT64_K       (1u << 9)
#define QKV_FLAG_FWHT64_V       (1u << 10)
#define QKV_FLAG_BANDED_REL_PACK (1u << 11)

struct QkvPrepDesc {
    uint64_t q_raw_va;
    uint64_t k_raw_va;
    uint64_t v_raw_va;
    uint64_t r_raw_va;
    uint64_t sinks_va;
    uint64_t qgate_raw_va;
    uint64_t qgate_compact_va;
    uint64_t k_dyn_va;
    uint64_t v_dyn_va;

    uint64_t q_clip_va;
    uint64_t r_clip_va;
    uint64_t k_conv_w_va;
    uint64_t v_conv_w_va;
    uint64_t rel_bias_va;
    uint64_t inp_pos_va;

    uint64_t r_l_va;
    uint64_t s_copy_va;
    uint64_t k_cell_va;
    uint64_t v_cell_va;

    uint64_t k_store_va;
    uint64_t v_store_va;
    uint64_t k_idxs_va;
    uint64_t v_idxs_va;

    uint64_t q_out_va;
    uint64_t r_out_va;
    uint64_t rel_pos_va;
    uint64_t base_mask_va;
    uint64_t kq_mask_out_va;

    uint     q_raw_stride_f32;
    uint     k_raw_stride_f32;
    uint     v_raw_stride_f32;
    uint     r_raw_stride_f32;
    uint     sinks_stride_f32;
    uint     r_nb1_f32;
    uint     rollback_slots;
    uint     rollback_stride_f32;
    uint     k_store_nb1_B;
    uint     v_store_nb1_B;
    uint     k_ids_nb0_B;
    uint     v_ids_nb0_B;
    uint     n_tokens;
    uint     n_conv;
    uint     n_rel;
    uint     n_extent;
    uint     n_kv;
    uint     n_kv_mask_stride;
    uint     rpos_nb1_B;
    uint     bmask_nb1_B;
    uint     bias_wtype;
    uint     bias_nb1_B;
    uint     flags;
    uint     r_splits;
    float    conv_res_scale;
    float    qk_clip;
    float    rel_clip;
    float    rms_eps;
    float    log_scale_alpha;
    float    inv_log_floor;
};

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer DescBuf {
    QkvPrepDesc d;
};

layout(push_constant) uniform PushArgs {
    uint64_t desc_va;
} pc;

shared float s_r[32];

void fwht128_in_reg(inout float reg[4], uint lane) {
    const float sc = 0.08838834764831845;
    [[unroll]] for (uint k = 0u; k < 4u; ++k) reg[k] *= sc;
    [[unroll]] for (uint h = 1u; h < 32u; h <<= 1u) {
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            float a = reg[k];
            float b = subgroupShuffleXor(a, h);
            reg[k] = ((lane & h) == 0u) ? (a + b) : (b - a);
        }
    }
    float a0 = reg[0], a1 = reg[1], a2 = reg[2], a3 = reg[3];
    float b0 = a0 + a1, b1 = a0 - a1, b2 = a2 + a3, b3 = a2 - a3;
    reg[0] = b0 + b2; reg[1] = b1 + b3;
    reg[2] = b0 - b2; reg[3] = b1 - b3;
}

void fwht64_in_reg(inout float reg[4], uint lane) {
    const float sc = 0.125;
    [[unroll]] for (uint k = 0u; k < 4u; ++k) reg[k] *= sc;
    [[unroll]] for (uint h = 1u; h < 32u; h <<= 1u) {
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            float a = reg[k];
            float b = subgroupShuffleXor(a, h);
            reg[k] = ((lane & h) == 0u) ? (a + b) : (b - a);
        }
    }
    float a0 = reg[0], a1 = reg[1], a2 = reg[2], a3 = reg[3];
    reg[0] = a0 + a1; reg[1] = a0 - a1;
    reg[2] = a2 + a3; reg[3] = a2 - a3;
}

void store_kv_head(uint64_t row_va, uint head, float reg[4], uint lane, bool is_q8) {
    if (!is_q8) {
        uint64_t h_va = row_va + uint64_t(head * 128u) * 2ul;
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            BufF16(h_va).v[k * 32u + lane] = float16_t(reg[k]);
        }
    } else {
        uint64_t h_va = row_va + uint64_t(head * 4u) * 34ul;
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            float val  = reg[k];
            float amax = subgroupMax(abs(val));
            float d    = amax * (1.0 / 127.0);
            float id   = (d != 0.0) ? (1.0 / d) : 0.0;
            int   q    = clamp(int(round(val * id)), -127, 127);
            uint64_t bva = h_va + uint64_t(k) * 34ul;
            if (lane == 0u) BufF16(bva).v[0] = float16_t(d);
            int q_hi = subgroupShuffleXor(q, 1u);
            if ((lane & 1u) == 0u) {
                BufU16(bva + 2ul).v[lane >> 1u] =
                    uint16_t((uint(q) & 0xFFu) | ((uint(q_hi) & 0xFFu) << 8u));
            }
        }
    }
}

float eval_conv4_tap(uint64_t raw_va, uint raw_stride_f32, uint ch, uint t,
                     float p0, float p1, float p2, uint lag) {
    if (t >= lag) return ConstBufF32(raw_va).v[(t - lag) * raw_stride_f32 + ch];
    uint idx = 3u - (lag - t);
    return (idx == 0u) ? p0 : ((idx == 1u) ? p1 : p2);
}

#define Q_FLD(f) (DescBuf(pc.desc_va).d.f)

void process_kv_head_all_tokens(
        bool is_v, uint head, uint lane,
        bool is_full, bool do_fwht128, bool do_fwht64, bool kv_q8) {
    const uint64_t raw_va     = is_v ? Q_FLD(v_raw_va)         : Q_FLD(k_raw_va);
    const uint raw_stride     = is_v ? Q_FLD(v_raw_stride_f32) : Q_FLD(k_raw_stride_f32);
    const uint64_t conv_w_va  = is_v ? Q_FLD(v_conv_w_va)      : Q_FLD(k_conv_w_va);
    const uint64_t dyn_va     = is_v ? Q_FLD(v_dyn_va)         : Q_FLD(k_dyn_va);
    const uint64_t cell_va    = is_v ? Q_FLD(v_cell_va)        : Q_FLD(k_cell_va);
    const uint64_t store_va   = is_v ? Q_FLD(v_store_va)       : Q_FLD(k_store_va);
    const uint64_t idxs_va    = is_v ? Q_FLD(v_idxs_va)        : Q_FLD(k_idxs_va);
    const uint store_nb1_B    = is_v ? Q_FLD(v_store_nb1_B)    : Q_FLD(k_store_nb1_B);
    const uint ids_nb0_B      = is_v ? Q_FLD(v_ids_nb0_B)      : Q_FLD(k_ids_nb0_B);
    const uint flags          = Q_FLD(flags);
    const bool ids_i64        = is_v ? ((flags & QKV_FLAG_V_IDS_I64) != 0u)
                                     : ((flags & QKV_FLAG_K_IDS_I64) != 0u);
    const uint conv_ch_off    = is_v ? 1024u : 0u;
    const uint n_tok          = Q_FLD(n_tokens);
    const float conv_res_sc   = Q_FLD(conv_res_scale);
    const float qk_clip       = Q_FLD(qk_clip);
    const float rms_eps       = Q_FLD(rms_eps);

    const int src_cell = max(ConstBufI32(Q_FLD(s_copy_va)).v[0], 0);
    const uint64_t hist_cell_va = Q_FLD(r_l_va) + uint64_t(src_cell) * uint64_t(Q_FLD(r_nb1_f32)) * 4ul;

    for (uint t = 0u; t < n_tok; ++t) {
        float reg[4];
        float ss = 0.0;
        for (uint k = 0u; k < 4u; ++k) {
            const uint ch = head * 128u + k * 32u + lane;
            const uint64_t past_va = hist_cell_va + uint64_t((conv_ch_off + ch) * 3u) * 4ul;
            float p0 = ConstBufF32(past_va).v[0];
            float p1 = ConstBufF32(past_va).v[1];
            float p2 = ConstBufF32(past_va).v[2];

            float x3 = eval_conv4_tap(raw_va, raw_stride, ch, t, p0, p1, p2, 3u);
            float x2 = eval_conv4_tap(raw_va, raw_stride, ch, t, p0, p1, p2, 2u);
            float x1 = eval_conv4_tap(raw_va, raw_stride, ch, t, p0, p1, p2, 1u);
            float x0 = ConstBufF32(raw_va).v[t * raw_stride + ch];

            vec4 cw = ConstBufVec4(conv_w_va).v[ch];
            float csum = x3 * cw.x + x2 * cw.y + x1 * cw.z + x0 * cw.w;
            if (is_full && dyn_va != 0ul) {
                vec4 dw = ConstBufVec4(dyn_va).v[t * 8u + head];
                csum += x0 * dw.x + x1 * dw.y + x2 * dw.z + x3 * dw.w;
            }
            float v = (x0 + csum) * conv_res_sc;
            reg[k] = v;
            ss += v * v;
        }

        float sum_sq = subgroupAdd(ss);
        float sc = (!is_v)
            ? (qk_clip / max(sqrt(sum_sq), qk_clip))
            : inversesqrt(sum_sq * (1.0 / 128.0) + rms_eps);
        [[unroll]] for (uint k = 0u; k < 4u; ++k) reg[k] *= sc;
        if (do_fwht64)       fwht64_in_reg(reg, lane);
        else if (do_fwht128) fwht128_in_reg(reg, lane);

        const uint64_t id_va = idxs_va + uint64_t(t) * uint64_t(ids_nb0_B);
        int64_t dst_row = ids_i64 ? ConstBufI64(id_va).v[0] : int64_t(ConstBufI32(id_va).v[0]);
        if (dst_row >= 0l) {
            store_kv_head(store_va + uint64_t(dst_row) * uint64_t(store_nb1_B), head, reg, lane, kv_q8);
        }
    }

    if (cell_va != 0ul) {
        const uint Kw = min(Q_FLD(rollback_slots), n_tok);
        const uint rb_stride = Q_FLD(rollback_stride_f32);
        for (uint k = 0u; k < 4u; ++k) {
            const uint ch = head * 128u + k * 32u + lane;
            const uint64_t past_va = hist_cell_va + uint64_t((conv_ch_off + ch) * 3u) * 4ul;
            float p0 = ConstBufF32(past_va).v[0];
            float p1 = ConstBufF32(past_va).v[1];
            float p2 = ConstBufF32(past_va).v[2];
            for (uint w = 0u; w < Kw; ++w) {
                const uint slot  = (Kw - 1u) - w;
                const uint t_end = (n_tok - Kw) + w;
                const uint64_t dst_slot_va = cell_va +
                    uint64_t(slot) * uint64_t(rb_stride) * 4ul +
                    uint64_t(ch * 3u) * 4ul;
                BufF32(dst_slot_va).v[0] = eval_conv4_tap(raw_va, raw_stride, ch, t_end, p0, p1, p2, 2u);
                BufF32(dst_slot_va).v[1] = eval_conv4_tap(raw_va, raw_stride, ch, t_end, p0, p1, p2, 1u);
                BufF32(dst_slot_va).v[2] = ConstBufF32(raw_va).v[t_end * raw_stride + ch];
            }
        }
    }
}

float dot_rel_bias_row(uint64_t b_va, uint wtype, uint n_rel) {
    float acc = 0.0;
    if (wtype == 2u) {
        const uint n4 = n_rel >> 2u;
        for (uint g = 0u; g < n4; ++g) {
            uvec2 p = ConstBufUvec2(b_va).v[g];
            uint k = g << 2u;
            vec4 w = vec4(
                uintBitsToFloat((p.x & 0xFFFFu) << 16u),
                uintBitsToFloat(p.x & 0xFFFF0000u),
                uintBitsToFloat((p.y & 0xFFFFu) << 16u),
                uintBitsToFloat(p.y & 0xFFFF0000u));
            acc += dot(w, vec4(s_r[k], s_r[k + 1u], s_r[k + 2u], s_r[k + 3u]));
        }
    } else if (wtype == 1u) {
        const uint n4 = n_rel >> 2u;
        for (uint g = 0u; g < n4; ++g) {
            uvec2 p = ConstBufUvec2(b_va).v[g];
            uint k = g << 2u;
            vec4 w = vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y));
            acc += dot(w, vec4(s_r[k], s_r[k + 1u], s_r[k + 2u], s_r[k + 3u]));
        }
    } else if (wtype == 0u) {
        const uint n4 = n_rel >> 2u;
        for (uint g = 0u; g < n4; ++g) {
            uint k = g << 2u;
            acc += dot(ConstBufVec4(b_va).v[g], vec4(s_r[k], s_r[k + 1u], s_r[k + 2u], s_r[k + 3u]));
        }
    } else if (wtype == 4u) {
        const uint n_chunks = (n_rel + 255u) >> 8u;
        for (uint c = 0u; c < n_chunks; ++c) {
            uint64_t cva = b_va + uint64_t(c) * 272ul;
            uint blk_end = min(8u, (n_rel - c * 256u + 31u) >> 5u);
            for (uint b = 0u; b < blk_end; ++b) {
                float ds = unpackHalf2x16(uint(ConstBufU16(cva).v[b])).x;
                float p = 0.0;
                uint base_k = c * 256u + b * 32u;
                for (uint w = 0u; w < 8u; ++w) {
                    uint pk = ConstBufU32(cva + 16ul).v[b * 8u + w];
                    uint kk = base_k + w * 4u;
                    if (kk < n_rel) {
                        p += float(bitfieldExtract(int(pk),  0, 8)) * s_r[kk + 0u]
                           + float(bitfieldExtract(int(pk),  8, 8)) * s_r[kk + 1u]
                           + float(bitfieldExtract(int(pk), 16, 8)) * s_r[kk + 2u]
                           + float(bitfieldExtract(int(pk), 24, 8)) * s_r[kk + 3u];
                    }
                }
                acc += ds * p;
            }
        }
    } else {
        for (uint k = 0u; k < n_rel; k += 32u) {
            uint64_t blk_va = b_va + uint64_t(k >> 5u) * 34ul;
            float ds = unpackHalf2x16(uint(ConstBufU16(blk_va).v[0])).x;
            float p = 0.0;
            for (uint o = 0u; o < 32u && (k + o) < n_rel; ++o) {
                p += float(ConstBufI8(blk_va + 2ul).v[o]) * s_r[k + o];
            }
            acc += ds * p;
        }
    }
    return acc;
}

void main() {
    const uint wg  = gl_WorkGroupID.x;
    const uint tok = gl_WorkGroupID.y;
    const uint tid = gl_LocalInvocationID.x;
    const uint sg  = tid >> 5u;
    const uint ln  = tid & 31u;

    const uint n_tok = Q_FLD(n_tokens);
    if (tok >= n_tok) return;

    const uint flags    = Q_FLD(flags);
    const bool is_full  = (flags & QKV_FLAG_IS_FULL_ATTN) != 0u;
    const bool fwht_k   = (flags & QKV_FLAG_FWHT_K) != 0u;
    const bool fwht_v   = (flags & QKV_FLAG_FWHT_V) != 0u;
    const bool fwht64_k = (flags & QKV_FLAG_FWHT64_K) != 0u;
    const bool fwht64_v = (flags & QKV_FLAG_FWHT64_V) != 0u;
    const bool kv_q8    = (flags & QKV_FLAG_KV_Q8_0) != 0u;

    if (wg < 24u) {
        if (sg != 0u) return;
        const uint head = wg;
        const uint64_t q_raw_va  = Q_FLD(q_raw_va);
        const uint64_t q_clip_va = Q_FLD(q_clip_va);
        const uint q_stride      = Q_FLD(q_raw_stride_f32);
        float reg[4];
        float ss = 0.0;
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            const uint ch = head * 128u + k * 32u + ln;
            float x = ConstBufF32(q_raw_va).v[tok * q_stride + ch]
                    * ConstBufF32(q_clip_va).v[ch];
            reg[k] = x;
            ss += x * x;
        }
        const float qk_clip = Q_FLD(qk_clip);
        float sc = qk_clip / max(sqrt(subgroupAdd(ss)), qk_clip);
        if (is_full) {
            float pos_p1 = float(ConstBufI32(Q_FLD(inp_pos_va)).v[tok]) + 1.0;
            sc *= 1.0 + Q_FLD(log_scale_alpha) * log(max(1.0, pos_p1 * Q_FLD(inv_log_floor)));
        }
        [[unroll]] for (uint k = 0u; k < 4u; ++k) reg[k] *= sc;
        if (fwht64_k)     fwht64_in_reg(reg, ln);
        else if (fwht_k)  fwht128_in_reg(reg, ln);
        const uint q_base = (tok * 24u + head) * 128u;
        const uint64_t q_out_va = Q_FLD(q_out_va);
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            BufF32(q_out_va).v[q_base + k * 32u + ln] = reg[k];
        }
        if ((flags & QKV_FLAG_PACK_QGATE) != 0u && Q_FLD(qgate_compact_va) != 0ul) {
            const uint64_t qg_raw = Q_FLD(qgate_raw_va);
            const uint64_t qg_dst = Q_FLD(qgate_compact_va);
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                const uint ch = head * 128u + k * 32u + ln;
                BufF32(qg_dst).v[q_base + k * 32u + ln] =
                    ConstBufF32(qg_raw).v[tok * q_stride + ch];
            }
        }
        return;
    }

    if (wg < 32u) {
        if (sg != 0u || tok != 0u) return;
        process_kv_head_all_tokens(false, wg - 24u, ln, is_full, fwht_k, fwht64_k, kv_q8);
        return;
    }

    if (wg < 40u) {
        if (sg != 0u || tok != 0u) return;
        process_kv_head_all_tokens(true, wg - 32u, ln, is_full, fwht_v, fwht64_v, kv_q8);
        return;
    }

    const uint r_splits = max(Q_FLD(r_splits), 1u);
    const uint r_wg     = wg - 40u;
    const uint head     = r_wg / r_splits;
    const uint r_sp     = r_wg - head * r_splits;
    if (head >= 24u) return;

    const uint n_rel = Q_FLD(n_rel);
    if (sg == 0u) {
        float rv = 0.0;
        if (ln < n_rel) {
            const uint ch = head * n_rel + ln;
            rv = ConstBufF32(Q_FLD(r_raw_va)).v[tok * Q_FLD(r_raw_stride_f32) + ch] *
                 ConstBufF32(Q_FLD(r_clip_va)).v[ch];
        }
        const float rel_clip = Q_FLD(rel_clip);
        float sc = rel_clip / max(sqrt(subgroupAdd(rv * rv)), rel_clip);
        if (is_full) {
            float pos_p1 = float(ConstBufI32(Q_FLD(inp_pos_va)).v[tok]) + 1.0;
            sc *= 1.0 + Q_FLD(log_scale_alpha) * log(max(1.0, pos_p1 * Q_FLD(inv_log_floor)));
        }
        float out_r = rv * sc;
        if (ln < 32u) s_r[ln] = out_r;
        if ((flags & QKV_FLAG_FUSE_REL_MASK) == 0u && r_sp == 0u && ln < n_rel) {
            BufF32(Q_FLD(r_out_va)).v[(tok * 24u + head) * n_rel + ln] = out_r;
        }
    }

    if ((flags & QKV_FLAG_FUSE_REL_MASK) == 0u) return;
    barrier();

    const float sink = ConstBufF32(Q_FLD(sinks_va)).v[tok * Q_FLD(sinks_stride_f32) + head];
    const bool     banded_pack  = (flags & QKV_FLAG_BANDED_REL_PACK) != 0u;
    const uint64_t rpos_base_va = Q_FLD(rel_pos_va);
    const uint64_t rpos_row_va  = rpos_base_va + uint64_t(tok) * uint64_t(Q_FLD(rpos_nb1_B));
    const int      b_qp         = banded_pack ? ConstBufI32(rpos_base_va).v[Q_FLD(rpos_nb1_B) + tok] : 0;
    const uint64_t bmask_row_va = Q_FLD(base_mask_va) + uint64_t(tok) * uint64_t(Q_FLD(bmask_nb1_B));
    const uint64_t dst_row_va   = Q_FLD(kq_mask_out_va) +
        uint64_t((head * n_tok + tok) * Q_FLD(n_kv_mask_stride)) * 2ul;
    const bool bmask_f16 = (flags & QKV_FLAG_BASE_MASK_F16) != 0u;
    const uint n_kv      = Q_FLD(n_kv);
    const uint n_extent  = Q_FLD(n_extent);
    const uint64_t rel_bias_va = Q_FLD(rel_bias_va);
    const uint bias_wtype      = Q_FLD(bias_wtype);
    const uint bias_nb1_B      = Q_FLD(bias_nb1_B);

    const uint chunk = (n_kv + r_splits - 1u) / r_splits;
    const uint j_beg = r_sp * chunk;
    const uint j_end = min(j_beg + chunk, n_kv);

    for (uint j = j_beg + tid; j < j_end; j += 128u) {
        float bm = bmask_f16 ? float(ConstBufF16(bmask_row_va).v[j]) : ConstBufF32(bmask_row_va).v[j];
        if (!(bm > -1.0e20)) {
            BufF16(dst_row_va).v[j] = float16_t(bm);
            continue;
        }
        int d_idx = banded_pack ? (b_qp - ConstBufI32(rpos_base_va).v[j]) : ConstBufI32(rpos_row_va).v[j];
        float acc = 0.0;
        if (d_idx >= 0 && uint(d_idx) < n_extent) {
            uint64_t b_va = rel_bias_va + uint64_t(d_idx) * uint64_t(bias_nb1_B);
            acc = dot_rel_bias_row(b_va, bias_wtype, n_rel);
        }
        BufF16(dst_row_va).v[j] = float16_t((acc - sink) + bm);
    }
}
#undef Q_FLD
)GLSL";

static const char * k_glsl_w4_fill_f32 = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t dst_va;
    uint     n_elems;
    float    val;
} pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.n_elems) BufF32(pc.dst_va).v[i] = pc.val;
}
)GLSL";

struct w4_fill_f32_push {
    uint64_t dst_va;
    uint32_t n_elems;
    float    val;
};

static const char * k_glsl_w4_scatter_rows_f32 = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer ConstBufI64 { int64_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t ids_va;
    uint64_t dst_va;
    uint     n_embd;
    uint     n_tokens;
    uint     dst_nb1_B;
    uint     ids_is_i64;
} pc;

void main() {
    uint i   = gl_GlobalInvocationID.x;
    uint tok = gl_WorkGroupID.y;
    if (i >= pc.n_embd || tok >= pc.n_tokens) return;
    int64_t dst_row = (pc.ids_is_i64 != 0u) ? ConstBufI64(pc.ids_va).v[tok] : int64_t(ConstBufI32(pc.ids_va).v[tok]);
    if (dst_row < 0l) return;
    BufF32(pc.dst_va + uint64_t(dst_row) * uint64_t(pc.dst_nb1_B)).v[i] =
        ConstBufF32(pc.src_va).v[tok * pc.n_embd + i];
}
)GLSL";

struct w4_scatter_rows_f32_push {
    uint64_t src_va;
    uint64_t ids_va;
    uint64_t dst_va;
    uint32_t n_embd;
    uint32_t n_tokens;
    uint32_t dst_nb1_B;
    uint32_t ids_is_i64;
};

// Coalesced W_attn_in descriptor per physical layer (order: [wq, wqkv_gate, attn_rel, attn_sink_proj, wk, wv])
struct agx_w4_layer_coalesced {
    agx_coalesced_weight w_attn_in{};
    bool     valid             = false;
    uint32_t q_prefix_rows     = 0; // contiguous Q-group rows starting at row 0
    bool     in_coal[6]        = {};
    uint32_t coal_idx[6]       = {};
    uint32_t row_off[6]        = {};
    uint32_t row_cnt[6]        = {};
    size_t   byte_off[6]       = {};
    uint32_t total_coal_rows   = 0;
};

struct agx_w4_bound_model {
    bool     bound             = false;
    uint64_t tok_embd_va       = 0;
    uint64_t output_va         = 0;
    agx_w4_layer_coalesced layers[AGX_INKLING2_MAX_PHYS_LAYERS];
};

// Wave 7: DSpark Draft Megakernel max supported layers
#define AGX_W7_DSPARK_MAX_LAYERS 8u

// ============================================================================
// Engine Struct & SLC Scratch Layout
// ============================================================================

struct agx_inkling2_engine {
    agx_backend_context * ctx = nullptr;
    agx_inkling2_hparams  hp;
    uint32_t              max_tokens = 8;

    // Compiled AGX3B Pillar 3 kernels
    const agx_kernel * k_ctrl_gates_sram    = nullptr;
    const agx_kernel * k_prep_depth_mix     = nullptr;
    const agx_kernel * k_prep_conv_clip     = nullptr;
    const agx_kernel * k_prep_post_conv     = nullptr;
    const agx_kernel * k_prep_router_top8   = nullptr;
    const agx_kernel * k_prep_rel_bias      = nullptr;
    const agx_kernel * k_prep_sense_barrier = nullptr;
    const agx_kernel * k_softcap_trunc      = nullptr;
    const agx_kernel * k_softcap_argmax_s1  = nullptr;
    const agx_kernel * k_softcap_argmax_s2  = nullptr;
    const agx_kernel * k_dspark_inject_mix  = nullptr;
    const agx_kernel * k_dspark_joint_gru   = nullptr;
    const agx_kernel * k_dspark_w2_softcap  = nullptr;
    const agx_kernel * k_dspark_draw_pack   = nullptr;
    const agx_kernel * k_embd_rmsnorm       = nullptr;
    const agx_kernel * k_rmsnorm_f32        = nullptr;
    const agx_kernel * k_copy_f32           = nullptr;

    // Wave 4 Megakernel & DSpark fastpath kernels
    const agx_kernel * k_w4_qkv_prep_store  = nullptr;
    const agx_kernel * k_w4_stage_post_attn = nullptr;
    const agx_kernel * k_w4_stage_post_ffn  = nullptr;
    const agx_kernel * k_w4_rel_bias_mask   = nullptr;
    const agx_kernel * k_w4_fwht_qgate      = nullptr;
    const agx_kernel * k_w4_ctrl_norm_cat   = nullptr;
    const agx_kernel * k_w4_ctrl_mix_norm   = nullptr;
    const agx_kernel * k_w4_ctrl_up_mix     = nullptr;
    const agx_kernel * k_w4_batch_rmsnorm   = nullptr;
    const agx_kernel * k_w4_dspark_pack     = nullptr;
    const agx_kernel * k_w4_dspark_gru      = nullptr;
    const agx_kernel * k_w4_dspark_w2       = nullptr;
    const agx_kernel * k_w4_dspark_merge    = nullptr;
    const agx_kernel * k_w7_dspark_joint_gru = nullptr;
    const agx_kernel * k_w7_dspark_inp_block = nullptr;
    const agx_kernel * k_w7_dspark_qkv_rope  = nullptr;
    const agx_kernel * k_w7_dspark_post_op   = nullptr;
    const agx_kernel * k_w7_dspark_gate_up_glu    = nullptr;
    const agx_kernel * k_w7_dspark_gate_up_glu_m8 = nullptr;
    const agx_kernel * k_w8_dspark_prep           = nullptr;
    const agx_kernel * k_w8_dspark_inject_mix     = nullptr;
    const agx_kernel * k_w11_dspark_inject_mix_fused = nullptr;
    const agx_kernel * k_w8_dspark_inject_kv      = nullptr;
    const agx_kernel * k_w9_q8_gemv_m4_f16        = nullptr;
    const agx_kernel * k_w9_q8_gemv_m4_f32        = nullptr;
    const agx_kernel * k_w9_q8_gemv_m8_f16        = nullptr;
    const agx_kernel * k_w9_q8_gemv_m8_f32        = nullptr;
    const agx_kernel * k_w9_dspark_w2_q4          = nullptr;
    const agx_kernel * k_w4_router_topk     = nullptr;
    const agx_kernel * k_w4_gather_f32      = nullptr;
    const agx_kernel * k_w4_fill_f32        = nullptr;
    const agx_kernel * k_w4_scatter_rows_f32 = nullptr;
    const agx_kernel * k_w10_rel_bias_table  = nullptr;
    const agx_kernel * k_w10_qkv_prep_store_rot64 = nullptr;
    bool               w10_qkv_rot64_tried   = false;
    bool               w4_compiled           = false;

    // Wave 5 Persistent 40-WG SENSE_BARRIER B=1 Megakernel kernels
    // k_w5_persist_attn[kv_q8][has_loop]
    // k_w5_persist_moe[moe_idx: 0=NVFP4, 1=Q8_0][has_loop]
    // k_w5_persist_chunk[kv_q8][moe_idx: 0=NVFP4, 1=Q8_0][has_loop]
    const agx_kernel * k_w5_persist_attn[2][2]     = {};
    const agx_kernel * k_w5_persist_moe[2][2]      = {};
    const agx_kernel * k_w5_persist_chunk[2][2][2] = {};
    const agx_kernel * k_w5_attn_in_b1             = nullptr;
    const agx_kernel * k_w9_attn_in_m4             = nullptr;
    const agx_kernel * k_w9_attn_in_m8             = nullptr;
    const agx_kernel * k_w9_attn_in_m4_f16         = nullptr;
    const agx_kernel * k_w9_attn_in_m8_f16         = nullptr;
    const agx_kernel * k_w5_hybrid_qkv_flash[2]      = {};
    const agx_kernel * k_w5_hybrid_post_attn_rtr[2]  = {};
    const agx_kernel * k_w5_hybrid_post_ffn_ctrl     = nullptr;
    const agx_kernel * k_w5_hybrid_post_attn_rtr_m[2]= {};
    const agx_kernel * k_w5_hybrid_post_ffn_ctrl_m   = nullptr;
    const agx_kernel * k_w5_hybrid_post_ffn_nl_m     = nullptr;
    bool               w5_compiled                   = false;
    bool               w5_chunk_tried[2]             = {};
    bool               w5_chunk_ok[2]                = {};
    bool               w5_split_tried[2]             = {};
    bool               w5_split_ok[2]                = {};
    bool               w5_hybrid_tried[2]            = {};
    bool               w5_hybrid_ok[2]               = {};
    bool               w5_hybrid_m_tried             = false;
    bool               w5_hybrid_m_ok                = false;

    agx_w4_bound_model        w4_model;
    agx_bo *                  dspark_scratch_bo = nullptr;

    struct {
        agx_bo * w2_q4_bo                                 = nullptr; // 28.81 MB Q4_32 SLC-pinned!
        uint64_t w2_src_va                                = 0;
        uint32_t w2_n_vocab                               = 0;
        agx_bo * w_qkv_bo[AGX_W7_DSPARK_MAX_LAYERS]       = {};      // AGX_BO_SLC_STREAM coalesced [wq; wk; wv]
        uint64_t wq_src_va[AGX_W7_DSPARK_MAX_LAYERS]      = {};
        uint32_t w_qkv_rows[AGX_W7_DSPARK_MAX_LAYERS]     = {};
        bool     draft_bound                              = false;

        agx_bo * w_kv_ctx_bo[AGX_W7_DSPARK_MAX_LAYERS]    = {};      // AGX_BO_SLC_STREAM
        uint64_t wk_ctx_va[AGX_W7_DSPARK_MAX_LAYERS]      = {};
        uint32_t w_kv_ctx_rows[AGX_W7_DSPARK_MAX_LAYERS]  = {};
        bool     inject_bound                             = false;
    } dspark_bound;

    // Dedicated SLC-pinned activation & recurrent-state arena (AGX_BO_SLC_PIN)
    agx_bo * slc_bo   = nullptr;
    agx_bo * conv_bo  = nullptr; // 48 logical slots of shortconv history (SLC-pinned)

    // Partitioned GPU VAs inside slc_bo (all 128B aligned)
    uint64_t va_cur_a        = 0;
    uint64_t va_cur_b        = 0;
    uint64_t va_ffn_inp      = 0;
    uint64_t va_qkv_in[3]    = {};
    uint64_t va_qkv_norm[3]  = {};
    uint64_t va_messages     = 0; // [4, n_embd, max_tokens]
    uint64_t va_blocks[AGX_INKLING2_MAX_RES_BLOCKS] = {};
    uint64_t va_prelude      = 0;
    uint64_t va_bank_out     = 0;

    uint64_t va_q_raw        = 0;
    uint64_t va_q            = 0;
    uint64_t va_k_raw        = 0;
    uint64_t va_k            = 0;
    uint64_t va_v_raw        = 0;
    uint64_t va_v            = 0;
    uint64_t va_r_raw        = 0;
    uint64_t va_r            = 0;
    uint64_t va_sinks        = 0;
    uint64_t va_qgate        = 0;
    uint64_t va_k_dyn        = 0;
    uint64_t va_v_dyn        = 0;
    uint64_t va_rel_table    = 0;
    uint64_t va_attn_out     = 0;
    uint64_t va_wo_out       = 0;
    uint64_t va_ffn_pre      = 0;
    uint64_t va_ffn_norm     = 0;
    uint64_t va_moe_out      = 0;
    uint64_t va_top8_ids     = 0;
    uint64_t va_top8_w       = 0;

    uint64_t va_logits_pad   = 0;
    uint64_t va_logits_unpad = 0;
    uint64_t va_part_val     = 0;
    uint64_t va_part_idx     = 0;
    uint64_t va_greedy_tok   = 0;
    uint64_t va_tokens       = 0;
    uint64_t va_positions    = 0;

    // DSpark sidecar scratch inside slc_bo
    uint64_t va_dspark_state  = 0;
    uint64_t va_dspark_markov = 0;
    uint64_t va_dspark_prev   = 0;
    uint64_t va_dspark_col    = 0;

    // CPU pointers for host readback / upload
    int32_t * cpu_tokens       = nullptr;
    int32_t * cpu_positions    = nullptr;
    int32_t * cpu_greedy_tok   = nullptr;
    float *   cpu_logits_unpad = nullptr;

    uint64_t conv_slot_stride_bytes = 0;
};

void agx_inkling2_fill_default_hparams(agx_inkling2_hparams * hp) {
    if (!hp) return;
    *hp = agx_inkling2_hparams{};
    for (uint32_t i = 0; i < AGX_INKLING2_MAX_SLOTS; ++i) {
        hp->is_swa[i] = true;
    }
    // Physical layers 5, 11, 17, 23 are full-attention; looped slots 29, 35, 41, 47 mirror 11, 17
    const uint32_t full_slots[] = { 5, 11, 17, 23, 29, 35, 41, 47 };
    for (uint32_t s : full_slots) {
        if (s < AGX_INKLING2_MAX_SLOTS) {
            hp->is_swa[s] = false;
        }
    }
}

uint32_t agx_inkling2_tensor_wtype(const struct ggml_tensor * t, uint32_t * out_row_bytes) {
    if (!t) {
        if (out_row_bytes) *out_row_bytes = 0;
        return AGX_WTYPE_F32;
    }
    if (agx_tensor_is_repacked(t) && t->type == GGML_TYPE_Q8_0) {
        const uint32_t n_chunks = (uint32_t) ((t->ne[0] + AGX_Q8_0_CHUNK_K - 1) / AGX_Q8_0_CHUNK_K);
        if (out_row_bytes) *out_row_bytes = n_chunks * AGX_Q8_0_CHUNK_BYTES;
        return AGX_WTYPE_Q8_0_8BLK;
    }
    if (out_row_bytes) *out_row_bytes = (uint32_t) t->nb[1];
    switch (t->type) {
        case GGML_TYPE_F32:  return AGX_WTYPE_F32;
        case GGML_TYPE_F16:  return AGX_WTYPE_F16;
        case GGML_TYPE_BF16: return AGX_WTYPE_BF16;
        case GGML_TYPE_Q8_0: return AGX_WTYPE_Q8_0_RAW;
        default:             return AGX_WTYPE_F32;
    }
}

static bool compile_inkling2_kernels(agx_inkling2_engine * eng) {
    agx_compiler * comp = eng->ctx->compiler;
    if (!comp) return false;

    auto compile_mode = [&](const char * name, const char * file, const char * def) -> const agx_kernel * {
        agx_compile_opts opts;
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        if (def && def[0]) {
            opts.defines.emplace_back(def);
        }
        return agx_compiler_compile_file(comp, name, file, opts);
    };

    eng->k_ctrl_gates_sram    = compile_mode("agx_ctrl_gates_sram",    "fused_ctrl_gates_sram.comp",    nullptr);
    eng->k_prep_depth_mix     = compile_mode("agx_prep_depth_mix",     "fused_loop_prep_qkv_conv.comp", "PREP_MODE=0");
    eng->k_prep_conv_clip     = compile_mode("agx_prep_conv_clip",     "fused_loop_prep_qkv_conv.comp", "PREP_MODE=1");
    eng->k_prep_post_conv     = compile_mode("agx_prep_post_conv",     "fused_loop_prep_qkv_conv.comp", "PREP_MODE=2");
    eng->k_prep_router_top8   = compile_mode("agx_prep_router_top8",   "fused_loop_prep_qkv_conv.comp", "PREP_MODE=3");
    eng->k_prep_rel_bias      = compile_mode("agx_prep_rel_bias",      "fused_loop_prep_qkv_conv.comp", "PREP_MODE=4");
    eng->k_prep_sense_barrier = compile_mode("agx_prep_sense_barrier", "fused_loop_prep_qkv_conv.comp", "PREP_MODE=5");

    eng->k_softcap_trunc      = compile_mode("agx_softcap_trunc",      "fused_softcap_argmax.comp",     "SOFTCAP_MODE=0");
    eng->k_softcap_argmax_s1  = compile_mode("agx_softcap_argmax_s1",  "fused_softcap_argmax.comp",     "SOFTCAP_MODE=1");
    eng->k_softcap_argmax_s2  = compile_mode("agx_softcap_argmax_s2",  "fused_softcap_argmax.comp",     "SOFTCAP_MODE=2");

    eng->k_dspark_inject_mix  = compile_mode("agx_dspark_inject_mix",  "dspark_mtp.comp",               "DSPARK_MODE=0");
    eng->k_dspark_joint_gru   = compile_mode("agx_dspark_joint_gru",   "dspark_mtp.comp",               "DSPARK_MODE=1");
    eng->k_dspark_w2_softcap  = compile_mode("agx_dspark_w2_softcap",  "dspark_mtp.comp",               "DSPARK_MODE=2");
    eng->k_dspark_draw_pack   = compile_mode("agx_dspark_draw_pack",   "dspark_mtp.comp",               "DSPARK_MODE=3");

    agx_compile_opts util_opts;
    util_opts.is_hot_kernel     = true;
    util_opts.strict_zero_spill = true;
    eng->k_embd_rmsnorm = agx_compiler_compile_glsl(comp, "agx_inkling2_embd_rmsnorm", k_glsl_embd_rmsnorm, util_opts);
    eng->k_rmsnorm_f32  = agx_compiler_compile_glsl(comp, "agx_inkling2_rmsnorm_f32",  k_glsl_rmsnorm_f32,  util_opts);
    eng->k_copy_f32     = agx_compiler_compile_glsl(comp, "agx_inkling2_copy_f32",     k_glsl_copy_f32,     util_opts);

    return eng->k_ctrl_gates_sram && eng->k_prep_depth_mix && eng->k_prep_conv_clip &&
           eng->k_prep_post_conv && eng->k_prep_router_top8 && eng->k_prep_rel_bias &&
           eng->k_softcap_trunc && eng->k_softcap_argmax_s1 && eng->k_softcap_argmax_s2 &&
           eng->k_dspark_inject_mix && eng->k_dspark_joint_gru && eng->k_dspark_w2_softcap &&
           eng->k_dspark_draw_pack && eng->k_embd_rmsnorm && eng->k_rmsnorm_f32 && eng->k_copy_f32;
}

agx_inkling2_engine * agx_inkling2_engine_create(agx_backend_context *       ctx,
                                                 const agx_inkling2_hparams & hp,
                                                 uint32_t                    max_batch_tokens) {
    if (!ctx || !ctx->drm || !ctx->compiler) return nullptr;

    auto * eng = new agx_inkling2_engine();
    eng->ctx        = ctx;
    eng->hp         = hp;
    eng->max_tokens = std::max<uint32_t>(1u, max_batch_tokens);

    if (!compile_inkling2_kernels(eng)) {
        AGX_LOG_ERROR("agx_inkling2_engine_create: kernel compilation failed");
        delete eng;
        return nullptr;
    }

    // Partition ONE contiguous SLC-pinned scratch BO (128B aligned per region)
    const size_t M       = eng->max_tokens;
    const size_t D       = hp.n_embd;
    const size_t q_dim   = (size_t) hp.n_head * hp.head_dim;
    const size_t kv_dim  = (size_t) hp.n_head_kv * hp.head_dim;
    const size_t r_dim   = (size_t) hp.n_relative * hp.n_head;
    const size_t max_ext = std::max<size_t>(hp.n_swa, hp.n_relative_extent) + 1u;
    const size_t embd_B  = agx_align_up(M * D * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t q_B     = agx_align_up(M * q_dim * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t kv_B    = agx_align_up(M * kv_dim * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t r_B     = agx_align_up(M * r_dim * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t sink_B  = agx_align_up(M * hp.n_head * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t dyn_B   = agx_align_up(M * hp.n_shortconv_l_cache * hp.n_head_kv * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t rtbl_B  = agx_align_up(M * hp.n_head * max_ext * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t top8_B  = agx_align_up(M * hp.n_expert_used * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t lpad_B  = agx_align_up(M * hp.n_vocab_padded * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t lunp_B  = agx_align_up(M * hp.n_vocab_unpadded * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t part_B  = agx_align_up(M * 64u * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t tok_B   = agx_align_up(M * sizeof(int32_t), AGX_CACHE_LINE_BYTES);
    const size_t dsp_r_B = agx_align_up(M * 512u * sizeof(float), AGX_CACHE_LINE_BYTES);

    size_t cursor = 0;
    auto alloc_off = [&](size_t sz) -> size_t {
        size_t o = cursor;
        cursor += agx_align_up(sz, AGX_CACHE_LINE_BYTES);
        return o;
    };

    const size_t o_cur_a        = alloc_off(embd_B);
    const size_t o_cur_b        = alloc_off(embd_B);
    const size_t o_ffn_inp      = alloc_off(embd_B);
    size_t o_qkv_in[3], o_qkv_norm[3];
    for (int j = 0; j < 3; ++j) {
        o_qkv_in[j]   = alloc_off(embd_B);
        o_qkv_norm[j] = alloc_off(embd_B);
    }
    const size_t o_messages     = alloc_off(4u * embd_B);
    size_t o_blocks[AGX_INKLING2_MAX_RES_BLOCKS];
    for (uint32_t b = 0; b < AGX_INKLING2_MAX_RES_BLOCKS; ++b) {
        o_blocks[b] = alloc_off(embd_B);
    }
    const size_t o_prelude      = alloc_off(embd_B);
    const size_t o_bank_out     = alloc_off(embd_B);

    const size_t o_q_raw        = alloc_off(q_B);
    const size_t o_q            = alloc_off(q_B);
    const size_t o_k_raw        = alloc_off(kv_B);
    const size_t o_k            = alloc_off(kv_B);
    const size_t o_v_raw        = alloc_off(kv_B);
    const size_t o_v            = alloc_off(kv_B);
    const size_t o_r_raw        = alloc_off(r_B);
    const size_t o_r            = alloc_off(r_B);
    const size_t o_sinks        = alloc_off(sink_B);
    const size_t o_qgate        = alloc_off(q_B);
    const size_t o_k_dyn        = alloc_off(dyn_B);
    const size_t o_v_dyn        = alloc_off(dyn_B);
    const size_t o_rel_table    = alloc_off(rtbl_B);
    const size_t o_attn_out     = alloc_off(q_B);
    const size_t o_wo_out       = alloc_off(embd_B);
    const size_t o_ffn_pre      = alloc_off(embd_B);
    const size_t o_ffn_norm     = alloc_off(embd_B);
    const size_t o_moe_out      = alloc_off(embd_B);
    const size_t o_top8_ids     = alloc_off(top8_B);
    const size_t o_top8_w       = alloc_off(top8_B);

    const size_t o_logits_pad   = alloc_off(lpad_B);
    const size_t o_logits_unpad = alloc_off(lunp_B);
    const size_t o_part_val     = alloc_off(part_B);
    const size_t o_part_idx     = alloc_off(part_B);
    const size_t o_greedy_tok   = alloc_off(tok_B);
    const size_t o_tokens       = alloc_off(tok_B);
    const size_t o_positions    = alloc_off(tok_B);

    const size_t o_dspark_state  = alloc_off(dsp_r_B);
    const size_t o_dspark_markov = alloc_off(dsp_r_B);
    const size_t o_dspark_prev   = alloc_off(tok_B);
    const size_t o_dspark_col    = alloc_off(lunp_B);

    eng->slc_bo = agx_bo_alloc(ctx->drm, cursor,
                               AGX_BO_SLC_PIN | AGX_BO_WRITEBACK | AGX_BO_ZERO,
                               "inkling2-slc-scratch");
    if (!eng->slc_bo) {
        AGX_LOG_ERROR("agx_inkling2_engine_create: failed to allocate %zu B SLC scratch BO", cursor);
        delete eng;
        return nullptr;
    }

    const uint64_t base_va = eng->slc_bo->gpu_va;
    auto * base_cpu        = static_cast<uint8_t *>(eng->slc_bo->cpu_ptr);

    eng->va_cur_a        = base_va + o_cur_a;
    eng->va_cur_b        = base_va + o_cur_b;
    eng->va_ffn_inp      = base_va + o_ffn_inp;
    for (int j = 0; j < 3; ++j) {
        eng->va_qkv_in[j]   = base_va + o_qkv_in[j];
        eng->va_qkv_norm[j] = base_va + o_qkv_norm[j];
    }
    eng->va_messages     = base_va + o_messages;
    for (uint32_t b = 0; b < AGX_INKLING2_MAX_RES_BLOCKS; ++b) {
        eng->va_blocks[b] = base_va + o_blocks[b];
    }
    eng->va_prelude      = base_va + o_prelude;
    eng->va_bank_out     = base_va + o_bank_out;

    eng->va_q_raw        = base_va + o_q_raw;
    eng->va_q            = base_va + o_q;
    eng->va_k_raw        = base_va + o_k_raw;
    eng->va_k            = base_va + o_k;
    eng->va_v_raw        = base_va + o_v_raw;
    eng->va_v            = base_va + o_v;
    eng->va_r_raw        = base_va + o_r_raw;
    eng->va_r            = base_va + o_r;
    eng->va_sinks        = base_va + o_sinks;
    eng->va_qgate        = base_va + o_qgate;
    eng->va_k_dyn        = base_va + o_k_dyn;
    eng->va_v_dyn        = base_va + o_v_dyn;
    eng->va_rel_table    = base_va + o_rel_table;
    eng->va_attn_out     = base_va + o_attn_out;
    eng->va_wo_out       = base_va + o_wo_out;
    eng->va_ffn_pre      = base_va + o_ffn_pre;
    eng->va_ffn_norm     = base_va + o_ffn_norm;
    eng->va_moe_out      = base_va + o_moe_out;
    eng->va_top8_ids     = base_va + o_top8_ids;
    eng->va_top8_w       = base_va + o_top8_w;

    eng->va_logits_pad   = base_va + o_logits_pad;
    eng->va_logits_unpad = base_va + o_logits_unpad;
    eng->va_part_val     = base_va + o_part_val;
    eng->va_part_idx     = base_va + o_part_idx;
    eng->va_greedy_tok   = base_va + o_greedy_tok;
    eng->va_tokens       = base_va + o_tokens;
    eng->va_positions    = base_va + o_positions;

    eng->va_dspark_state  = base_va + o_dspark_state;
    eng->va_dspark_markov = base_va + o_dspark_markov;
    eng->va_dspark_prev   = base_va + o_dspark_prev;
    eng->va_dspark_col    = base_va + o_dspark_col;

    eng->cpu_tokens       = reinterpret_cast<int32_t *>(base_cpu + o_tokens);
    eng->cpu_positions    = reinterpret_cast<int32_t *>(base_cpu + o_positions);
    eng->cpu_greedy_tok   = reinterpret_cast<int32_t *>(base_cpu + o_greedy_tok);
    eng->cpu_logits_unpad = reinterpret_cast<float *>(base_cpu + o_logits_unpad);

    // Allocate SLC-pinned shortconv recurrent state for all 48 logical slots (~3.8 MB)
    const uint32_t n_hist = std::max<uint32_t>(1u, hp.n_shortconv_l_cache - 1u);
    eng->conv_slot_stride_bytes = agx_align_up((size_t) n_hist * hp.n_sconv_channels * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t conv_total = (size_t) hp.n_layer_all * eng->conv_slot_stride_bytes;
    eng->conv_bo = agx_bo_alloc(ctx->drm, conv_total,
                                AGX_BO_SLC_PIN | AGX_BO_WRITEBACK | AGX_BO_ZERO,
                                "inkling2-conv-state");
    if (!eng->conv_bo) {
        agx_bo_free(ctx->drm, eng->slc_bo);
        delete eng;
        return nullptr;
    }

    return eng;
}

void agx_inkling2_engine_destroy(agx_inkling2_engine * eng) {
    if (!eng) return;
    if (eng->ctx && eng->ctx->drm) {
        for (uint32_t i = 0; i < AGX_INKLING2_MAX_PHYS_LAYERS; ++i) {
            if (eng->w4_model.layers[i].valid) {
                agx_repack_coalesced_free(eng->ctx->drm, &eng->w4_model.layers[i].w_attn_in);
                eng->w4_model.layers[i].valid = false;
            }
        }
        if (eng->dspark_bound.w2_q4_bo) {
            agx_bo_free(eng->ctx->drm, eng->dspark_bound.w2_q4_bo);
        }
        for (uint32_t i = 0; i < AGX_W7_DSPARK_MAX_LAYERS; ++i) {
            if (eng->dspark_bound.w_qkv_bo[i]) {
                agx_bo_free(eng->ctx->drm, eng->dspark_bound.w_qkv_bo[i]);
            }
            if (eng->dspark_bound.w_kv_ctx_bo[i]) {
                agx_bo_free(eng->ctx->drm, eng->dspark_bound.w_kv_ctx_bo[i]);
            }
        }
        if (eng->dspark_scratch_bo) agx_bo_free(eng->ctx->drm, eng->dspark_scratch_bo);
        if (eng->conv_bo) agx_bo_free(eng->ctx->drm, eng->conv_bo);
        if (eng->slc_bo)  agx_bo_free(eng->ctx->drm, eng->slc_bo);
    }
    delete eng;
}

void agx_inkling2_engine_reset(agx_inkling2_engine * eng) {
    if (!eng) return;
    if (eng->conv_bo && eng->conv_bo->cpu_ptr) {
        std::memset(eng->conv_bo->cpu_ptr, 0, eng->conv_bo->size);
    }
}

// ============================================================================
// Direct Pillar 3 Fused Kernel Dispatchers
// ============================================================================

bool agx_inkling2_dispatch_depth_mix(agx_inkling2_engine * eng,
                                     agx_cdm_stream *      stream,
                                     const uint64_t *      state_vas,
                                     uint32_t              n_states,
                                     uint64_t              query_va,
                                     uint32_t              n_queries,
                                     uint64_t              norm_w_va,
                                     uint64_t              out_va,
                                     uint64_t              out_pre_va,
                                     uint32_t              n_embd,
                                     uint32_t              n_tokens,
                                     float                 eps_mix,
                                     float                 eps_norm,
                                     agx_barrier_mode      post_barrier) {
    if (!eng || !stream || !state_vas || n_states == 0 || n_states > 16 || n_queries == 0 || n_queries > 4) {
        return false;
    }
    void * ptrs_cpu = nullptr;
    uint64_t ptrs_va = agx_cdm_alloc_args(stream, n_states * sizeof(uint64_t), 16, &ptrs_cpu);
    if (!ptrs_va || !ptrs_cpu) return false;
    std::memcpy(ptrs_cpu, state_vas, n_states * sizeof(uint64_t));

    agx_depth_mix_push pc{};
    pc.states_ptrs_va = ptrs_va;
    pc.query_va       = query_va;
    pc.norm_w_va      = norm_w_va;
    pc.out_va         = out_va;
    pc.out_pre_va     = out_pre_va;
    pc.n_embd         = n_embd;
    pc.n_states       = n_states;
    pc.n_queries      = n_queries;
    pc.n_tokens       = n_tokens;
    pc.inv_sqrt_d     = 1.0f / std::sqrt((float) n_embd);
    pc.eps_mix        = eps_mix;
    pc.eps_norm       = eps_norm;

    return agx_cdm_dispatch(stream, eng->k_prep_depth_mix, n_tokens, 1, 1, &pc, sizeof(pc), post_barrier);
}

bool agx_inkling2_dispatch_ctrl_gates(agx_inkling2_engine *      eng,
                                      agx_cdm_stream *           stream,
                                      const agx_ctrl_gate_desc * descs,
                                      uint32_t                   n_ctrls,
                                      uint32_t                   n_embd,
                                      uint32_t                   rank,
                                      uint32_t                   n_tokens,
                                      float                      eps_ctrl,
                                      float                      eps_norm,
                                      agx_barrier_mode           post_barrier) {
    if (!eng || !stream || !descs || n_ctrls == 0 || n_embd > 3840 || rank > 256) {
        return false;
    }
    void * desc_cpu = nullptr;
    uint64_t desc_va = agx_cdm_alloc_args(stream, n_ctrls * sizeof(agx_ctrl_gate_desc), 16, &desc_cpu);
    if (!desc_va || !desc_cpu) return false;
    std::memcpy(desc_cpu, descs, n_ctrls * sizeof(agx_ctrl_gate_desc));

    agx_ctrl_gates_push pc{};
    pc.desc_va  = desc_va;
    pc.n_embd   = n_embd;
    pc.rank     = rank;
    pc.n_tokens = n_tokens;
    pc.n_ctrls  = n_ctrls;
    pc.eps_ctrl = eps_ctrl;
    pc.eps_norm = eps_norm;

    return agx_cdm_dispatch(stream, eng->k_ctrl_gates_sram, n_ctrls, n_tokens, 1, &pc, sizeof(pc), post_barrier);
}

bool agx_inkling2_dispatch_conv_clip_fwht(agx_inkling2_engine *           eng,
                                          agx_cdm_stream *                stream,
                                          const agx_conv_clip_fwht_push & args,
                                          agx_barrier_mode                post_barrier) {
    if (!eng || !stream || args.n_head == 0) return false;
    const uint32_t gx = (args.n_head + 3u) / 4u;
    return agx_cdm_dispatch(stream, eng->k_prep_conv_clip, gx, args.n_seq_tokens, args.n_seqs,
                            &args, sizeof(args), post_barrier);
}

bool agx_inkling2_dispatch_post_conv_res(agx_inkling2_engine *          eng,
                                         agx_cdm_stream *               stream,
                                         const agx_post_conv_res_push & args,
                                         agx_barrier_mode               post_barrier) {
    if (!eng || !stream || args.n_channels == 0) return false;
    const uint32_t gx = (args.n_channels + 255u) / 256u;
    return agx_cdm_dispatch(stream, eng->k_prep_post_conv, gx, args.n_seq_tokens, args.n_seqs,
                            &args, sizeof(args), post_barrier);
}

bool agx_inkling2_dispatch_router_top8(agx_inkling2_engine *        eng,
                                       agx_cdm_stream *             stream,
                                       const agx_router_top8_push & args,
                                       agx_barrier_mode             post_barrier) {
    if (!eng || !stream || args.n_tokens == 0) return false;
    return agx_cdm_dispatch(stream, eng->k_prep_router_top8, args.n_tokens, 1, 1,
                            &args, sizeof(args), post_barrier);
}

bool agx_inkling2_dispatch_rel_bias_table(agx_inkling2_engine *           eng,
                                          agx_cdm_stream *                stream,
                                          const agx_rel_bias_table_push & args,
                                          agx_barrier_mode                post_barrier) {
    if (!eng || !stream) return false;
    const uint32_t gx = (args.n_extent + 1u + 63u) / 64u;
    return agx_cdm_dispatch(stream, eng->k_prep_rel_bias, gx, args.n_head, args.n_tokens,
                            &args, sizeof(args), post_barrier);
}

bool agx_inkling2_dispatch_softcap_argmax(agx_inkling2_engine * eng,
                                          agx_cdm_stream *      stream,
                                          uint64_t              src_va,
                                          uint64_t              dst_logits_va,
                                          uint64_t              out_token_va,
                                          uint32_t              n_vocab_unpadded,
                                          uint32_t              src_stride_f32,
                                          uint32_t              n_tokens,
                                          float                 softcap,
                                          bool                  compute_argmax,
                                          agx_barrier_mode      post_barrier) {
    if (!eng || !stream || n_vocab_unpadded == 0 || n_tokens == 0) return false;

    agx_softcap_argmax_push pc{};
    pc.src_va           = src_va;
    pc.dst_logits_va    = dst_logits_va;
    pc.partial_val_va   = eng->va_part_val;
    pc.partial_idx_va   = eng->va_part_idx;
    pc.out_token_va     = out_token_va ? out_token_va : eng->va_greedy_tok;
    pc.n_vocab_unpadded = n_vocab_unpadded;
    pc.src_stride_f32   = src_stride_f32;
    pc.dst_stride_f32   = n_vocab_unpadded;
    pc.n_tokens         = n_tokens;
    pc.n_splits         = 40u; // 40 GPU cores on M4 Max
    pc.flags            = dst_logits_va ? AGX_SOFTCAP_FLAG_WRITE_LOGITS : 0u;
    pc.softcap          = softcap;
    pc.inv_softcap      = (softcap > 0.0f) ? (1.0f / softcap) : 0.0f;

    if (!compute_argmax) {
        const uint32_t gx = (n_vocab_unpadded + 255u) / 256u;
        return agx_cdm_dispatch(stream, eng->k_softcap_trunc, gx, n_tokens, 1, &pc, sizeof(pc), post_barrier);
    }

    if (!agx_cdm_dispatch(stream, eng->k_softcap_argmax_s1, pc.n_splits, n_tokens, 1,
                          &pc, sizeof(pc), AGX_BARRIER_STRONG_USC)) {
        return false;
    }
    return agx_cdm_dispatch(stream, eng->k_softcap_argmax_s2, n_tokens, 1, 1,
                            &pc, sizeof(pc), post_barrier);
}

// ============================================================================
// Fused DSpark MTP Sidecar Dispatchers
// ============================================================================

bool agx_dspark_dispatch_inject_ctx_mix(agx_inkling2_engine * eng,
                                        agx_cdm_stream *      stream,
                                        uint64_t              feats_va,
                                        uint64_t              mix_w_va,
                                        uint64_t              norm_w_va,
                                        uint64_t              out_va,
                                        uint32_t              n_embd,
                                        uint32_t              n_target,
                                        uint32_t              n_tokens,
                                        float                 rms_eps,
                                        agx_barrier_mode      post_barrier) {
    if (!eng || !stream) return false;
    agx_dspark_inject_mix_push pc{};
    pc.feats_va  = feats_va;
    pc.mix_w_va  = mix_w_va;
    pc.norm_w_va = norm_w_va;
    pc.out_va    = out_va;
    pc.n_embd    = n_embd;
    pc.n_target  = n_target;
    pc.n_tokens  = n_tokens;
    pc.rms_eps   = rms_eps;
    return agx_cdm_dispatch(stream, eng->k_dspark_inject_mix, n_tokens, 1, 1, &pc, sizeof(pc), post_barrier);
}

bool agx_dspark_dispatch_markov_head(agx_inkling2_engine *          eng,
                                     agx_cdm_stream *               stream,
                                     const agx_dspark_mtp_tensors & w,
                                     uint64_t                       anchor_toks_va,
                                     uint64_t                       base_logits_va,
                                     uint64_t                       hidden_va,
                                     uint64_t                       f_rnn_va,
                                     uint64_t                       rnn_state_va,
                                     uint64_t                       out_logits_va,
                                     uint64_t                       out_h_nextn_va,
                                     uint32_t                       n_blocks,
                                     uint32_t                       block_drafts,
                                     agx_barrier_mode               post_barrier) {
    if (!eng || !stream || !w.markov_w1 || !w.markov_w2 || n_blocks == 0 || block_drafts == 0) {
        return false;
    }

    const uint32_t i_beg = w.sample_from_anchor ? 0u : 1u;

    // Seed prev_tok from anchor tokens (copy on GPU — zero CPU roundtrip)
    {
        copy_f32_push cp{ anchor_toks_va, eng->va_dspark_prev, n_blocks, 0u };
        if (!agx_cdm_dispatch(stream, eng->k_copy_f32, (n_blocks + 255u) / 256u, 1, 1,
                              &cp, sizeof(cp), AGX_BARRIER_STRONG_USC)) {
            return false;
        }
    }

    // If bonus anchor slot 0 is passed through unbiased, copy block slot 0 logits directly
    if (!w.sample_from_anchor && out_logits_va != base_logits_va) {
        for (uint32_t b = 0; b < n_blocks; ++b) {
            uint64_t off = (uint64_t) b * block_drafts * w.n_vocab * sizeof(float);
            copy_f32_push cp{ base_logits_va + off, out_logits_va + off, w.n_vocab, 0u };
            if (!agx_cdm_dispatch(stream, eng->k_copy_f32, (w.n_vocab + 255u) / 256u, 1, 1,
                                  &cp, sizeof(cp), AGX_BARRIER_NONE)) {
                return false;
            }
        }
        agx_cdm_barrier_usc(stream);
    }

    uint32_t w1_rb = 0, joint_rb = 0, w2_rb = 0;
    const uint32_t w1_wt    = agx_inkling2_tensor_wtype(w.markov_w1,  &w1_rb);
    const uint32_t joint_wt = agx_inkling2_tensor_wtype(w.joint_proj, &joint_rb);
    const uint32_t w2_wt    = agx_inkling2_tensor_wtype(w.markov_w2,  &w2_rb);

    float w2_scale = 1.0f;
    if (w.markov_w2_s && w.markov_w2_s->data && w.markov_w2_s->type == GGML_TYPE_F32) {
        w2_scale = *static_cast<const float *>(w.markov_w2_s->data);
    }

    const uint32_t n_splits = 40u;
    const uint32_t n_cand   = std::min<uint32_t>(64u, w.n_vocab);

    for (uint32_t i = i_beg; i < block_drafts; ++i) {
        const bool is_last = (i + 1u == block_drafts);

        // 1. Fused concat4 + joint_proj + GRU state update (zero CPU sync!)
        if (w.joint_proj) {
            agx_dspark_joint_gru_push jpc{};
            jpc.prev_tok_va     = eng->va_dspark_prev;
            jpc.w1_va           = agx_tensor_gpu_va(w.markov_w1);
            jpc.hidden_va       = hidden_va;
            jpc.f_rnn_va        = f_rnn_va;
            jpc.state_va        = eng->va_dspark_state;
            jpc.markov_out_va   = eng->va_dspark_markov;
            jpc.joint_w_va      = agx_tensor_gpu_va(w.joint_proj);
            jpc.joint_b_va      = w.joint_proj_b ? agx_tensor_gpu_va(w.joint_proj_b) : 0ull;
            jpc.n_embd          = w.n_embd;
            jpc.rank            = w.rank;
            jpc.n_blocks        = n_blocks;
            jpc.block_drafts    = block_drafts;
            jpc.draft_pos       = i;
            jpc.w1_wtype        = w1_wt;
            jpc.w1_row_bytes    = w1_rb;
            jpc.joint_wtype     = joint_wt;
            jpc.joint_row_bytes = joint_rb;
            jpc.flags           = (w.has_f_rnn && f_rnn_va ? AGX_DSPARK_FLAG_HAS_F_RNN : 0u) |
                                  (i == i_beg ? AGX_DSPARK_FLAG_ZERO_INIT_STATE : 0u);

            if (!agx_cdm_dispatch(stream, eng->k_dspark_joint_gru, n_blocks, 1, 1,
                                  &jpc, sizeof(jpc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }

        // 2. Fused markov_w2 @ markov * scale + bias + base_i + softcap + 40-WG partial argmax
        const uint64_t col_dst = (n_blocks == 1u)
            ? (out_logits_va + (uint64_t) i * w.n_vocab * sizeof(float))
            : eng->va_dspark_col;

        agx_dspark_w2_softcap_push wpc{};
        wpc.markov_va    = eng->va_dspark_markov;
        wpc.w2_va        = agx_tensor_gpu_va(w.markov_w2);
        wpc.w2_bias_va   = w.markov_w2_b ? agx_tensor_gpu_va(w.markov_w2_b) : 0ull;
        wpc.base_va      = base_logits_va;
        wpc.out_col_va   = col_dst;
        wpc.part_val_va  = eng->va_part_val;
        wpc.part_idx_va  = eng->va_part_idx;
        wpc.rank         = w.rank;
        wpc.n_vocab      = w.n_vocab;
        wpc.n_blocks     = n_blocks;
        wpc.block_drafts = block_drafts;
        wpc.draft_pos    = i;
        wpc.n_splits     = n_splits;
        wpc.w2_wtype     = w2_wt;
        wpc.w2_row_bytes = w2_rb;
        wpc.flags        = (w.markov_w2_b ? AGX_DSPARK_FLAG_HAS_W2_BIAS : 0u) | AGX_DSPARK_FLAG_COMPUTE_ARGMAX;
        wpc.w2_scale     = w2_scale;
        wpc.softcap      = w.softcap;
        wpc.inv_softcap  = (w.softcap > 0.0f) ? (1.0f / w.softcap) : 0.0f;

        if (!agx_cdm_dispatch(stream, eng->k_dspark_w2_softcap, n_splits, n_blocks, 1,
                              &wpc, sizeof(wpc), AGX_BARRIER_STRONG_USC)) {
            return false;
        }

        // Scatter multi-block col into block-major out_logits if n_blocks > 1
        if (n_blocks > 1u) {
            for (uint32_t b = 0; b < n_blocks; ++b) {
                copy_f32_push cp{
                    eng->va_dspark_col + (uint64_t) b * w.n_vocab * sizeof(float),
                    out_logits_va + (uint64_t) (b * block_drafts + i) * w.n_vocab * sizeof(float),
                    w.n_vocab, 0u
                };
                if (!agx_cdm_dispatch(stream, eng->k_copy_f32, (w.n_vocab + 255u) / 256u, 1, 1,
                                      &cp, sizeof(cp), AGX_BARRIER_NONE)) {
                    return false;
                }
            }
            agx_cdm_barrier_usc(stream);
        }

        // 3. Draw next prev_tok (and pack top-64 candidates into h_nextn when rnn_state != 0)
        if (rnn_state_va != 0ull || !is_last) {
            agx_dspark_draw_pack_push dpc{};
            dpc.col_va        = col_dst;
            dpc.part_val_va   = eng->va_part_val;
            dpc.part_idx_va   = eng->va_part_idx;
            dpc.rnn_state_va  = rnn_state_va;
            dpc.prev_tok_va   = eng->va_dspark_prev;
            dpc.out_pack_va   = out_h_nextn_va;
            dpc.n_vocab       = w.n_vocab;
            dpc.n_embd        = w.n_embd;
            dpc.n_blocks      = n_blocks;
            dpc.block_drafts  = block_drafts;
            dpc.draft_pos     = i;
            dpc.n_splits      = n_splits;
            dpc.n_cand        = n_cand;
            dpc.has_rnn_state = (rnn_state_va != 0ull) ? 1u : 0u;

            if (!agx_cdm_dispatch(stream, eng->k_dspark_draw_pack, n_blocks, 1, 1,
                                  &dpc, sizeof(dpc),
                                  is_last ? post_barrier : AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Helper: Dispatch Dense Weight Projection via Pillar 2 (`agx-gemm`)
// ============================================================================

static bool dispatch_dense_mm(agx_backend_context *      ctx,
                              agx_cdm_stream *           stream,
                              const struct ggml_tensor * w,
                              uint64_t                   x_va,
                              uint64_t                   y_va,
                              uint32_t                   M,
                              agx_barrier_mode           post_barrier,
                              bool                       prec_f32 = false) {
    if (!w) return false;
    agx_weight_layout layout = AGX_WEIGHT_F32;
    if (!agx_gemm_infer_weight_layout(w, &layout)) {
        return false;
    }
    agx_gemm_desc d{};
    d.w_va         = agx_tensor_gpu_va(w);
    d.x_va         = x_va;
    d.y_va         = y_va;
    d.M            = M;
    d.N            = (uint32_t) w->ne[1];
    d.K            = (uint32_t) w->ne[0];
    d.layout       = layout;
    d.prec_f32     = prec_f32;
    d.post_barrier = post_barrier;
    return agx_gemm_dispatch(ctx, stream, d);
}

// ============================================================================
// Full 48-Pass Inkling 2 Mini Single-CDM-Stream Forward Execution
// ============================================================================

static bool encode_inkling2_layer(agx_inkling2_engine *             eng,
                                  agx_cdm_stream *                  stream,
                                  const agx_inkling2_model_tensors & W,
                                  agx_kv_hybrid_pool *              kv_pool,
                                  uint32_t                          il,
                                  uint32_t                          ip,
                                  bool                              has_loop,
                                  const agx_inkling2_loop_tensors * loop,
                                  const std::vector<uint64_t> &     bank_vas,
                                  uint64_t                          bank_out_va,
                                  uint64_t &                        cur_va,
                                  uint64_t &                        alt_va,
                                  std::vector<uint64_t> &           blocks,
                                  uint32_t &                        next_block_slot,
                                  uint32_t                          n_tokens,
                                  uint32_t                          n_kv_past) {
    const auto & hp    = eng->hp;
    const auto & layer = W.layers[ip];
    const bool is_swa  = hp.is_swa[il];
    const uint32_t D   = hp.n_embd;
    const uint32_t Hq  = hp.n_head;
    const uint32_t Hkv = hp.n_head_kv;
    const uint32_t Dh  = hp.head_dim;
    const uint32_t n_conv = hp.n_shortconv_l_cache;
    const uint64_t conv_slot_va = eng->conv_bo->gpu_va + (uint64_t) il * eng->conv_slot_stride_bytes;
    const uint32_t n_kv_active  = n_kv_past + n_tokens;

    const uint64_t inpSA_va = cur_va;

    // ------------------------------------------------------------------------
    // Stage 1: Depth-mix residual attention + optional cross-pass rank-64 FP32
    //          controller gates + RMSNorm + Q/K/V/R/sinks/qgate/dyn_conv +
    //          shortconv + exact L2 clip + tau + optional FWHT + KV append
    // ------------------------------------------------------------------------

    // 1a. Cross-pass message attention (if pass > 0 in looped core 6..17)
    if (has_loop && loop && loop->msg_query) {
        std::vector<uint64_t> msg_states;
        msg_states.reserve(bank_vas.size() + 1u);
        for (int s = (int) bank_vas.size() - 1; s >= 0; --s) {
            msg_states.push_back(bank_vas[s]);
        }
        msg_states.push_back(bank_out_va);

        if (!agx_inkling2_dispatch_depth_mix(eng, stream,
                msg_states.data(), (uint32_t) msg_states.size(),
                agx_tensor_gpu_va(loop->msg_query), 4u,
                0ull, eng->va_messages, 0ull,
                D, n_tokens, 1e-6f, hp.f_norm_rms_eps, AGX_BARRIER_NONE)) {
            return false;
        }
    }

    // 1b. Attention residual depth-mix + cross-pass Q/K/V controllers + input RMSNorm
    uint64_t inp_q_va = cur_va, inp_k_va = cur_va, inp_v_va = cur_va;
    if (layer.attn_res_query) {
        std::vector<uint64_t> res_states;
        res_states.reserve(blocks.size() + 1u);
        for (int s = (int) blocks.size() - 1; s >= 0; --s) {
            res_states.push_back(blocks[s]);
        }
        res_states.push_back(cur_va);

        const uint32_t n_q = is_swa ? 1u : 3u;
        const bool fuse_swa_norm = is_swa && !has_loop && layer.attn_norm != nullptr;
        const uint64_t mix_out_va = fuse_swa_norm ? eng->va_qkv_norm[0] : eng->va_qkv_in[0];

        if (!agx_inkling2_dispatch_depth_mix(eng, stream,
                res_states.data(), (uint32_t) res_states.size(),
                agx_tensor_gpu_va(layer.attn_res_query), n_q,
                fuse_swa_norm ? agx_tensor_gpu_va(layer.attn_norm) : 0ull,
                mix_out_va, 0ull,
                D, n_tokens, 1e-6f, hp.f_norm_rms_eps, AGX_BARRIER_STRONG_USC)) {
            return false;
        }

        if (fuse_swa_norm) {
            inp_q_va = inp_k_va = inp_v_va = eng->va_qkv_norm[0];
        } else if (is_swa) {
            inp_q_va = inp_k_va = inp_v_va = eng->va_qkv_in[0];
        } else {
            const uint64_t stride_B = (uint64_t) D * sizeof(float);
            // depth_mix writes [D, 3, M]; for M==1 entries are contiguous at +0, +D, +2D
            inp_q_va = eng->va_qkv_in[0];
            inp_k_va = eng->va_qkv_in[0] + stride_B;
            inp_v_va = eng->va_qkv_in[0] + 2u * stride_B;
        }
    } else if (has_loop) {
        // Ensure message depth_mix completed before controller gates read va_messages
        agx_cdm_barrier_usc(stream);
    }

    if (has_loop && loop) {
        // Batched 3-controller dispatch (Q, K, V) in native-FP32 32 KB SRAM + fused trailing RMSNorm!
        const uint64_t cur_vas[3] = { inp_q_va, inp_k_va, inp_v_va };
        const struct ggml_tensor * norms[3] = {
            is_swa ? layer.attn_norm : layer.attn_q_inp_norm,
            is_swa ? layer.attn_norm : layer.attn_k_inp_norm,
            is_swa ? layer.attn_norm : layer.attn_v_inp_norm,
        };
        agx_ctrl_gate_desc descs[3]{};
        for (int j = 0; j < 3; ++j) {
            descs[j].cur_va         = cur_vas[j];
            descs[j].msg_va         = eng->va_messages;
            descs[j].down_va        = agx_tensor_gpu_va(loop->ctrl_down[j]);
            descs[j].gain_va        = agx_tensor_gpu_va(loop->ctrl_gain[j]);
            descs[j].forget_va      = agx_tensor_gpu_va(loop->ctrl_forget[j]);
            descs[j].norm_w_va      = norms[j] ? agx_tensor_gpu_va(norms[j]) : 0ull;
            descs[j].out_va         = eng->va_qkv_norm[j];
            descs[j].down_wtype     = agx_inkling2_tensor_wtype(loop->ctrl_down[j],   &descs[j].down_row_bytes);
            descs[j].gain_wtype     = agx_inkling2_tensor_wtype(loop->ctrl_gain[j],   &descs[j].gain_row_bytes);
            descs[j].forget_wtype   = agx_inkling2_tensor_wtype(loop->ctrl_forget[j], &descs[j].forget_row_bytes);
            descs[j].cur_stride_f32 = (!is_swa && layer.attn_res_query) ? (3u * D) : D;
            descs[j].msg_stride_f32 = 4u * D;
            descs[j].msg_offset_f32 = (uint32_t) j * D;
            descs[j].out_stride_f32 = D;
        }
        if (!agx_inkling2_dispatch_ctrl_gates(eng, stream, descs, 3u, D, hp.gate_rank, n_tokens,
                                              1e-6f, hp.f_norm_rms_eps, AGX_BARRIER_STRONG_USC)) {
            return false;
        }
        inp_q_va = eng->va_qkv_norm[0];
        inp_k_va = eng->va_qkv_norm[1];
        inp_v_va = eng->va_qkv_norm[2];
    } else if (!is_swa) {
        // Full-attn pass 0: 3 independent RMSNorms (overlap with AGX_BARRIER_NONE!)
        const uint64_t srcs[3] = { inp_q_va, inp_k_va, inp_v_va };
        const struct ggml_tensor * norms[3] = { layer.attn_q_inp_norm, layer.attn_k_inp_norm, layer.attn_v_inp_norm };
        for (int j = 0; j < 3; ++j) {
            rmsnorm_f32_push np{ srcs[j], agx_tensor_gpu_va(norms[j]), eng->va_qkv_norm[j], D, n_tokens, hp.f_norm_rms_eps, 0u, 0ull };
            if (!agx_cdm_dispatch(stream, eng->k_rmsnorm_f32, n_tokens, 1, 1, &np, sizeof(np),
                                  j == 2 ? AGX_BARRIER_STRONG_USC : AGX_BARRIER_NONE)) {
                return false;
            }
        }
        inp_q_va = eng->va_qkv_norm[0];
        inp_k_va = eng->va_qkv_norm[1];
        inp_v_va = eng->va_qkv_norm[2];
    } else if (!layer.attn_res_query) {
        // Layer 0 SWA (no attn_res_query): single RMSNorm
        rmsnorm_f32_push np{ cur_va, agx_tensor_gpu_va(layer.attn_norm), eng->va_qkv_norm[0], D, n_tokens, hp.f_norm_rms_eps, 0u, 0ull };
        if (!agx_cdm_dispatch(stream, eng->k_rmsnorm_f32, n_tokens, 1, 1, &np, sizeof(np), AGX_BARRIER_STRONG_USC)) {
            return false;
        }
        inp_q_va = inp_k_va = inp_v_va = eng->va_qkv_norm[0];
    }

    // 1c. Concurrent dense attention projections (ZERO barriers between independent matvecs!)
    if (!dispatch_dense_mm(eng->ctx, stream, layer.wq,             inp_q_va, eng->va_q_raw, n_tokens, AGX_BARRIER_NONE)) return false;
    if (!dispatch_dense_mm(eng->ctx, stream, layer.wk,             inp_k_va, eng->va_k_raw, n_tokens, AGX_BARRIER_NONE)) return false;
    if (!dispatch_dense_mm(eng->ctx, stream, layer.wv,             inp_v_va, eng->va_v_raw, n_tokens, AGX_BARRIER_NONE)) return false;
    if (!dispatch_dense_mm(eng->ctx, stream, layer.attn_rel,       inp_q_va, eng->va_r_raw, n_tokens, AGX_BARRIER_NONE)) return false;
    if (!dispatch_dense_mm(eng->ctx, stream, layer.attn_sink_proj, inp_q_va, eng->va_sinks, n_tokens, AGX_BARRIER_NONE)) return false;
    if (!dispatch_dense_mm(eng->ctx, stream, layer.wqkv_gate,      inp_q_va, eng->va_qgate, n_tokens, AGX_BARRIER_NONE)) return false;
    if (!is_swa) {
        if (!dispatch_dense_mm(eng->ctx, stream, layer.attn_k_conv_proj, inp_k_va, eng->va_k_dyn, n_tokens, AGX_BARRIER_NONE)) return false;
        if (!dispatch_dense_mm(eng->ctx, stream, layer.attn_v_conv_proj, inp_v_va, eng->va_v_dyn, n_tokens, AGX_BARRIER_NONE)) return false;
    }
    agx_cdm_barrier_usc(stream);

    // 1d. Concurrent Q/K/V/R fused conv + exact L2 clip / RMSNorm + tau + optional FWHT128
    const float conv_scale = 1.0f / std::sqrt((float) ((is_swa ? 1u : 2u) * n_conv + 1u));
    const uint32_t kv_fwht_flag = hp.use_fwht128_kv ? AGX_PREP_FLAG_FWHT_128 : 0u;

    // Qcur: per-channel attn_q_clip + exact L2 clip + optional tau + optional FWHT128
    {
        agx_conv_clip_fwht_push qpc{};
        qpc.src_va          = eng->va_q_raw;
        qpc.dst_va          = eng->va_q;
        qpc.clip_scale_va   = agx_tensor_gpu_va(layer.attn_q_clip);
        qpc.inp_pos_va      = eng->va_positions;
        qpc.head_dim        = Dh;
        qpc.n_head          = Hq;
        qpc.n_seq_tokens    = n_tokens;
        qpc.n_seqs          = 1u;
        qpc.flags           = AGX_PREP_FLAG_PER_CHAN_SCALE | AGX_PREP_FLAG_L2_CLIP |
                              (!is_swa ? AGX_PREP_FLAG_TAU_SCALE : 0u) | kv_fwht_flag;
        qpc.clip_thresh     = hp.f_qk_clip;
        qpc.log_scale_alpha = hp.f_log_scale_alpha;
        qpc.inv_log_floor   = 1.0f / (float) hp.n_log_scale_floor;
        if (!agx_inkling2_dispatch_conv_clip_fwht(eng, stream, qpc, AGX_BARRIER_NONE)) return false;
    }
    // Kcur: static + optional dynamic shortconv + history update + exact L2 clip + optional FWHT128
    {
        agx_conv_clip_fwht_push kpc{};
        kpc.src_va          = eng->va_k_raw;
        kpc.dst_va          = eng->va_k;
        kpc.conv_w_va       = agx_tensor_gpu_va(layer.attn_k_conv);
        kpc.dyn_va          = is_swa ? 0ull : eng->va_k_dyn;
        kpc.history_va      = conv_slot_va;
        kpc.state_out_va    = conv_slot_va;
        kpc.head_dim        = Dh;
        kpc.n_head          = Hkv;
        kpc.n_seq_tokens    = n_tokens;
        kpc.n_seqs          = 1u;
        kpc.n_conv          = n_conv;
        kpc.n_sconv_ch      = hp.n_sconv_channels;
        kpc.conv_ch_offset  = 0u;
        kpc.flags           = AGX_PREP_FLAG_DO_CONV | AGX_PREP_FLAG_UPDATE_STATE |
                              AGX_PREP_FLAG_L2_CLIP | (!is_swa ? AGX_PREP_FLAG_DYNAMIC_CONV : 0u) | kv_fwht_flag;
        kpc.conv_res_scale  = conv_scale;
        kpc.clip_thresh     = hp.f_qk_clip;
        if (!agx_inkling2_dispatch_conv_clip_fwht(eng, stream, kpc, AGX_BARRIER_NONE)) return false;
    }
    // Vcur: static + optional dynamic shortconv + history update + RMSNorm + optional FWHT128
    {
        agx_conv_clip_fwht_push vpc{};
        vpc.src_va          = eng->va_v_raw;
        vpc.dst_va          = eng->va_v;
        vpc.conv_w_va       = agx_tensor_gpu_va(layer.attn_v_conv);
        vpc.dyn_va          = is_swa ? 0ull : eng->va_v_dyn;
        vpc.history_va      = conv_slot_va;
        vpc.state_out_va    = conv_slot_va;
        vpc.head_dim        = Dh;
        vpc.n_head          = Hkv;
        vpc.n_seq_tokens    = n_tokens;
        vpc.n_seqs          = 1u;
        vpc.n_conv          = n_conv;
        vpc.n_sconv_ch      = hp.n_sconv_channels;
        vpc.conv_ch_offset  = Hkv * Dh;
        vpc.flags           = AGX_PREP_FLAG_DO_CONV | AGX_PREP_FLAG_UPDATE_STATE |
                              AGX_PREP_FLAG_RMS_NORM | (!is_swa ? AGX_PREP_FLAG_DYNAMIC_CONV : 0u) | kv_fwht_flag;
        vpc.conv_res_scale  = conv_scale;
        vpc.rms_eps         = hp.f_norm_rms_eps;
        if (!agx_inkling2_dispatch_conv_clip_fwht(eng, stream, vpc, AGX_BARRIER_NONE)) return false;
    }
    // Rcur: per-channel attn_rel_clip + exact L2 clip + optional tau
    {
        agx_conv_clip_fwht_push rpc{};
        rpc.src_va          = eng->va_r_raw;
        rpc.dst_va          = eng->va_r;
        rpc.clip_scale_va   = agx_tensor_gpu_va(layer.attn_rel_clip);
        rpc.inp_pos_va      = eng->va_positions;
        rpc.head_dim        = hp.n_relative;
        rpc.n_head          = Hq;
        rpc.n_seq_tokens    = n_tokens;
        rpc.n_seqs          = 1u;
        rpc.flags           = AGX_PREP_FLAG_PER_CHAN_SCALE | AGX_PREP_FLAG_L2_CLIP |
                              (!is_swa ? AGX_PREP_FLAG_TAU_SCALE : 0u);
        rpc.clip_thresh     = hp.f_relative_clip;
        rpc.log_scale_alpha = hp.f_log_scale_alpha;
        rpc.inv_log_floor   = 1.0f / (float) hp.n_log_scale_floor;
        if (!agx_inkling2_dispatch_conv_clip_fwht(eng, stream, rpc, AGX_BARRIER_STRONG_USC)) return false;
    }

    // 1e. Concurrent: Append K/V into @kv-moe Hybrid KV pool + build rel_bias_table
    {
        agx_kv_append_args app{};
        app.logical_slot  = il;
        app.n_tokens      = n_tokens;
        app.kv_write_base = n_kv_past;
        app.k_new_va      = eng->va_k;
        app.v_new_va      = eng->va_v;
        app.pos_va        = eng->va_positions;
        app.src_is_f16    = false;
        if (!agx_kv_hybrid_append(kv_pool, stream, app, AGX_BARRIER_NONE)) return false;
    }
    {
        uint32_t rb_bytes = 0;
        agx_rel_bias_table_push tpc{};
        tpc.rcur_va      = eng->va_r;
        tpc.rel_bias_va  = agx_tensor_gpu_va(layer.attn_rel_bias);
        tpc.sinks_va     = eng->va_sinks;
        tpc.out_table_va = eng->va_rel_table;
        tpc.n_rel        = hp.n_relative;
        tpc.n_extent     = is_swa ? hp.n_swa : hp.n_relative_extent;
        tpc.n_head       = Hq;
        tpc.n_tokens     = n_tokens;
        tpc.bias_wtype   = agx_inkling2_tensor_wtype(layer.attn_rel_bias, &rb_bytes);
        if (!agx_inkling2_dispatch_rel_bias_table(eng, stream, tpc, AGX_BARRIER_STRONG_USC)) return false;
    }

    // ------------------------------------------------------------------------
    // Stage 2: @kv-moe Fused SWA Ring / Split-K Full Flash-Decode (+ rel_table
    //          + zero_sink + optional FWHT64 + sigmoid(qgate)) -> W_O ->
    //          attn_post_conv + block-res + ffn_res_mix + LOOP_CTRL_FFN +
    //          ffn_norm + 2304->128 router top-8 written to SLC!
    // ------------------------------------------------------------------------
    {
        agx_attn_stage_args attn{};
        attn.logical_slot  = il;
        attn.n_tokens      = n_tokens;
        attn.n_kv          = is_swa ? std::min<uint32_t>(n_kv_active, hp.n_swa) : n_kv_active;
        attn.n_head_q      = Hq;
        attn.n_head_kv     = Hkv;
        attn.head_dim      = Dh;
        attn.n_rel_extent  = is_swa ? hp.n_swa : hp.n_relative_extent;
        attn.pos_q_base    = n_kv_past;
        attn.flags         = AGX_ATTN_FLAG_CAUSAL |
                             (is_swa ? (AGX_ATTN_FLAG_SWA_WINDOW | AGX_ATTN_FLAG_SWA_RING_KV) : AGX_ATTN_FLAG_PAGED_KV) |
                             AGX_ATTN_FLAG_HAS_REL_TABLE | AGX_ATTN_FLAG_REL_TABLE_SINK |
                             AGX_ATTN_FLAG_QGATE_F32 |
                             (hp.use_fwht64_attn_out ? AGX_ATTN_FLAG_FWHT64 : 0u);
        attn.kq_scale      = 1.0f / std::sqrt((float) Dh);
        attn.q_va          = eng->va_q;
        attn.out_va        = eng->va_attn_out;
        attn.rel_table_va  = eng->va_rel_table;
        attn.q_gate_va     = eng->va_qgate;
        attn.pos_q_va      = eng->va_positions;
        if (!agx_kv_hybrid_dispatch_attn(kv_pool, stream, attn, AGX_BARRIER_STRONG_USC)) return false;
    }

    // W_O projection
    if (!dispatch_dense_mm(eng->ctx, stream, layer.wo, eng->va_attn_out, eng->va_wo_out, n_tokens, AGX_BARRIER_STRONG_USC)) {
        return false;
    }

    // attn_post_conv + residual block boundary (inkling2.cpp L402-406)
    const bool new_block = (ip % hp.attn_res_block_size == 0u);
    if (new_block) {
        // Save inpSA into a dedicated SLC block slot before overwriting cur_va!
        if (next_block_slot >= AGX_INKLING2_MAX_RES_BLOCKS) return false;
        const uint64_t slot_va = eng->va_blocks[next_block_slot++];
        copy_f32_push cp{ inpSA_va, slot_va, n_tokens * D, 0u };
        if (!agx_cdm_dispatch(stream, eng->k_copy_f32, (n_tokens * D + 255u) / 256u, 1, 1,
                              &cp, sizeof(cp), AGX_BARRIER_NONE)) {
            return false;
        }
        blocks.push_back(slot_va);
    }

    {
        agx_post_conv_res_push ppc{};
        ppc.src_va         = eng->va_wo_out;
        ppc.res_va         = new_block ? 0ull : inpSA_va;
        ppc.dst_va         = eng->va_ffn_inp;
        ppc.conv_w_va      = agx_tensor_gpu_va(layer.attn_post_conv);
        ppc.history_va     = conv_slot_va;
        ppc.state_out_va   = conv_slot_va;
        ppc.n_channels     = D;
        ppc.n_seq_tokens   = n_tokens;
        ppc.n_seqs         = 1u;
        ppc.n_conv         = n_conv;
        ppc.n_sconv_ch     = hp.n_sconv_channels;
        ppc.conv_ch_offset = 2u * Hkv * Dh;
        ppc.flags          = AGX_POST_FLAG_UPDATE_STATE | (new_block ? 0u : AGX_POST_FLAG_ADD_RESIDUAL);
        ppc.conv_res_scale = 1.0f / std::sqrt((float) (n_conv + 1u));
        if (!agx_inkling2_dispatch_post_conv_res(eng, stream, ppc, AGX_BARRIER_STRONG_USC)) return false;
    }

    // FFN depth-mix (+ fused ffn_norm when !has_loop, else followed by LOOP_CTRL_FFN + fused ffn_norm)
    {
        std::vector<uint64_t> ffn_states;
        ffn_states.reserve(blocks.size() + 1u);
        for (int s = (int) blocks.size() - 1; s >= 0; --s) {
            ffn_states.push_back(blocks[s]);
        }
        ffn_states.push_back(eng->va_ffn_inp);

        const bool fuse_norm = !has_loop;
        if (!agx_inkling2_dispatch_depth_mix(eng, stream,
                ffn_states.data(), (uint32_t) ffn_states.size(),
                agx_tensor_gpu_va(layer.ffn_res_query), 1u,
                fuse_norm ? agx_tensor_gpu_va(layer.ffn_norm) : 0ull,
                fuse_norm ? eng->va_ffn_norm : eng->va_ffn_pre, 0ull,
                D, n_tokens, 1e-6f, hp.f_norm_rms_eps, AGX_BARRIER_STRONG_USC)) {
            return false;
        }
    }

    if (has_loop && loop) {
        agx_ctrl_gate_desc fdesc{};
        fdesc.cur_va         = eng->va_ffn_pre;
        fdesc.msg_va         = eng->va_messages;
        fdesc.down_va        = agx_tensor_gpu_va(loop->ctrl_down[3]);
        fdesc.gain_va        = agx_tensor_gpu_va(loop->ctrl_gain[3]);
        fdesc.forget_va      = agx_tensor_gpu_va(loop->ctrl_forget[3]);
        fdesc.norm_w_va      = agx_tensor_gpu_va(layer.ffn_norm);
        fdesc.out_va         = eng->va_ffn_norm;
        fdesc.down_wtype     = agx_inkling2_tensor_wtype(loop->ctrl_down[3],   &fdesc.down_row_bytes);
        fdesc.gain_wtype     = agx_inkling2_tensor_wtype(loop->ctrl_gain[3],   &fdesc.gain_row_bytes);
        fdesc.forget_wtype   = agx_inkling2_tensor_wtype(loop->ctrl_forget[3], &fdesc.forget_row_bytes);
        fdesc.cur_stride_f32 = D;
        fdesc.msg_stride_f32 = 4u * D;
        fdesc.msg_offset_f32 = 3u * D; // LOOP_CTRL_FFN = 3
        fdesc.out_stride_f32 = D;
        if (!agx_inkling2_dispatch_ctrl_gates(eng, stream, &fdesc, 1u, D, hp.gate_rank, n_tokens,
                                              1e-6f, hp.f_norm_rms_eps, AGX_BARRIER_STRONG_USC)) {
            return false;
        }
    }

    // Fused 2304 -> 128 router + sigmoid + bias + top-8 + norm(1e-20) -> SLC
    {
        uint32_t r_rb = 0;
        agx_router_top8_push rpc{};
        rpc.x_va          = eng->va_ffn_norm;
        rpc.w_va          = agx_tensor_gpu_va(layer.ffn_gate_inp);
        rpc.bias_va       = layer.ffn_exp_probs_b ? agx_tensor_gpu_va(layer.ffn_exp_probs_b) : 0ull;
        rpc.out_ids_va    = eng->va_top8_ids;
        rpc.out_w_va      = eng->va_top8_w;
        rpc.n_embd        = D;
        rpc.n_experts     = hp.n_expert;
        rpc.top_k         = hp.n_expert_used;
        rpc.n_tokens      = n_tokens;
        rpc.wtype         = agx_inkling2_tensor_wtype(layer.ffn_gate_inp, &r_rb);
        rpc.w_row_bytes   = r_rb;
        rpc.norm_eps      = hp.expert_weights_norm_eps;
        rpc.weights_scale = hp.expert_weights_scale;
        if (!agx_inkling2_dispatch_router_top8(eng, stream, rpc, AGX_BARRIER_STRONG_USC)) return false;
    }

    // ------------------------------------------------------------------------
    // Stage 3: @kv-moe Fused Bitonic-Bucketed MoE (gate_up -> SiTU -> down)
    //          + ffn_post_conv + residual add -> next layer's cur_va in SLC
    // ------------------------------------------------------------------------
    {
        agx_weight_layout gu_layout = AGX_WEIGHT_F32, dn_layout = AGX_WEIGHT_F32;
        if (!agx_gemm_infer_weight_layout(layer.ffn_gate_up_exps, &gu_layout) ||
            !agx_gemm_infer_weight_layout(layer.ffn_down_exps,    &dn_layout)) {
            return false;
        }
        const uint64_t gu_slice = agx_tensor_is_repacked(layer.ffn_gate_up_exps)
            ? agx_repack_slice_stride(layer.ffn_gate_up_exps->type, layer.ffn_gate_up_exps->ne[0], layer.ffn_gate_up_exps->ne[1])
            : (uint64_t) layer.ffn_gate_up_exps->nb[2];
        const uint64_t dn_slice = agx_tensor_is_repacked(layer.ffn_down_exps)
            ? agx_repack_slice_stride(layer.ffn_down_exps->type, layer.ffn_down_exps->ne[0], layer.ffn_down_exps->ne[1])
            : (uint64_t) layer.ffn_down_exps->nb[2];

        if (!eng->ctx->moe_state) {
            eng->ctx->moe_state = agx_moe_create(eng->ctx->drm, eng->ctx->compiler);
            if (!eng->ctx->moe_state) return false;
        }
        auto * moe_runner = static_cast<agx_moe_runner *>(eng->ctx->moe_state);

        agx_moe_stage_args margs{};
        margs.n_tokens         = n_tokens;
        margs.n_embd           = D;
        margs.n_ff             = hp.n_ff_exp;
        margs.n_expert         = hp.n_expert;
        margs.top_k            = hp.n_expert_used;
        margs.act_op           = AGX_MOE_ACT_SITU;
        margs.situ_beta        = hp.situ_beta;
        margs.situ_linear_beta = hp.situ_linear_beta;
        margs.exp_scale0       = 1.0f; // router already applied expert_weights_scale
        margs.ffn_x_va         = eng->va_ffn_norm;
        margs.topk_ids_va      = eng->va_top8_ids;
        margs.topk_w_va        = eng->va_top8_w;
        margs.moe_out_va       = eng->va_moe_out;
        margs.gate_up_va       = agx_tensor_gpu_va(layer.ffn_gate_up_exps);
        margs.down_va          = agx_tensor_gpu_va(layer.ffn_down_exps);
        margs.gate_up_slice_B  = gu_slice;
        margs.down_slice_B     = dn_slice;
        margs.gate_up_layout   = gu_layout;
        margs.down_layout      = dn_layout;
        margs.gate_up_scale_va = layer.ffn_gate_up_exps_s ? agx_tensor_gpu_va(layer.ffn_gate_up_exps_s) : 0ull;
        margs.down_scale_va    = layer.ffn_down_exps_s    ? agx_tensor_gpu_va(layer.ffn_down_exps_s)    : 0ull;
        margs.x_is_f16         = false;
        margs.accumulate_out   = false;
        if (!agx_moe_dispatch(moe_runner, stream, margs, AGX_BARRIER_STRONG_USC)) return false;
    }

    // ffn_post_conv + residual add (`ffn_inp + conv_out` -> alt_va), then swap cur_va <-> alt_va
    {
        agx_post_conv_res_push fpc{};
        fpc.src_va         = eng->va_moe_out;
        fpc.res_va         = eng->va_ffn_inp;
        fpc.dst_va         = alt_va;
        fpc.conv_w_va      = agx_tensor_gpu_va(layer.ffn_post_conv);
        fpc.history_va     = conv_slot_va;
        fpc.state_out_va   = conv_slot_va;
        fpc.n_channels     = D;
        fpc.n_seq_tokens   = n_tokens;
        fpc.n_seqs         = 1u;
        fpc.n_conv         = n_conv;
        fpc.n_sconv_ch     = hp.n_sconv_channels;
        fpc.conv_ch_offset = 2u * Hkv * Dh + D;
        fpc.flags          = AGX_POST_FLAG_UPDATE_STATE | AGX_POST_FLAG_ADD_RESIDUAL;
        fpc.conv_res_scale = 1.0f / std::sqrt((float) (n_conv + 1u));
        if (!agx_inkling2_dispatch_post_conv_res(eng, stream, fpc, AGX_BARRIER_STRONG_USC)) return false;
    }

    std::swap(cur_va, alt_va);
    return true;
}

bool agx_inkling2_run_forward(agx_inkling2_engine *              eng,
                              const agx_inkling2_model_tensors & weights,
                              agx_kv_hybrid_pool *               kv_pool,
                              const int32_t *                    tokens,
                              const int32_t *                    positions,
                              uint32_t                           n_tokens,
                              uint32_t                           n_kv_past,
                              float *                            out_logits,
                              agx_inkling2_step_stats *          out_stats) {
    if (!eng || !kv_pool || !tokens || !positions || n_tokens == 0 || n_tokens > eng->max_tokens) {
        return false;
    }
    const auto & hp = eng->hp;
    agx_cdm_stream * stream = eng->ctx->stream;
    if (!stream) return false;

    if (!agx_kv_hybrid_ensure_full_capacity(kv_pool, n_kv_past + n_tokens)) {
        return false;
    }
    if (!agx_gemm_init(eng->ctx)) {
        return false;
    }

    // Upload tokens & positions directly into SLC-pinned BO (zero extra ioctls)
    std::memcpy(eng->cpu_tokens,    tokens,    n_tokens * sizeof(int32_t));
    std::memcpy(eng->cpu_positions, positions, n_tokens * sizeof(int32_t));

    // Begin ONE single CDM stream for all 48 passes + LM head + softcap/argmax!
    agx_cdm_stream_reset(stream);
    agx_cdm_stats st_before{};
    agx_cdm_get_stats(stream, &st_before);

    uint64_t cur_va = eng->va_cur_a;
    uint64_t alt_va = eng->va_cur_b;

    // 0. Embedding lookup + tok_norm
    {
        uint32_t emb_rb = 0;
        embd_rmsnorm_push ep{};
        ep.embd_va    = agx_tensor_gpu_va(weights.tok_embd);
        ep.tok_ids_va = eng->va_tokens;
        ep.norm_w_va  = weights.tok_norm ? agx_tensor_gpu_va(weights.tok_norm) : 0ull;
        ep.dst_va     = cur_va;
        ep.n_embd     = hp.n_embd;
        ep.n_tokens   = n_tokens;
        ep.wtype      = agx_inkling2_tensor_wtype(weights.tok_embd, &emb_rb);
        ep.row_bytes  = emb_rb;
        ep.eps        = hp.f_norm_rms_eps;
        if (!agx_cdm_dispatch(stream, eng->k_embd_rmsnorm, n_tokens, 1, 1, &ep, sizeof(ep), AGX_BARRIER_STRONG_USC)) {
            return false;
        }
    }

    std::vector<uint64_t> blocks;
    std::vector<uint64_t> empty_bank;
    uint32_t next_block_slot = 0u;
    uint32_t passes_done = 0u;

    // 1. Prelude: layers 0 .. core_start - 1 (0..5)
    for (uint32_t il = 0; il < hp.core_start; ++il) {
        if (!encode_inkling2_layer(eng, stream, weights, kv_pool, il, il,
                                   false, nullptr, empty_bank, 0ull,
                                   cur_va, alt_va, blocks, next_block_slot,
                                   n_tokens, n_kv_past)) {
            return false;
        }
        ++passes_done;
    }

    // Save prelude output & prelude_blocks (inkling2-loop.cpp L93-94)
    {
        copy_f32_push cp{ cur_va, eng->va_prelude, n_tokens * hp.n_embd, 0u };
        if (!agx_cdm_dispatch(stream, eng->k_copy_f32, (n_tokens * hp.n_embd + 255u) / 256u, 1, 1,
                              &cp, sizeof(cp), AGX_BARRIER_STRONG_USC)) {
            return false;
        }
    }
    const std::vector<uint64_t> prelude_blocks = blocks;
    const uint32_t prelude_slot_mark = next_block_slot;

    // 2. Looped Core: layers core_start .. core_end - 1 (6..17) x n_passes (3)
    for (uint32_t pass = 0; pass < hp.n_passes; ++pass) {
        std::vector<uint64_t> bank;
        uint64_t bank_out_va = 0ull;

        if (pass > 0u) {
            // bank = blocks[prelude_blocks.size() + 1 .. end], bank_out = cur (inkling2-loop.cpp L100-103)
            const size_t bank_beg = prelude_blocks.size() + 1u;
            if (bank_beg < blocks.size()) {
                bank.assign(blocks.begin() + bank_beg, blocks.end());
            }
            copy_f32_push cp_bank{ cur_va, eng->va_bank_out, n_tokens * hp.n_embd, 0u };
            copy_f32_push cp_prel{ eng->va_prelude, cur_va,  n_tokens * hp.n_embd, 0u };
            if (!agx_cdm_dispatch(stream, eng->k_copy_f32, (n_tokens * hp.n_embd + 255u) / 256u, 1, 1,
                                  &cp_bank, sizeof(cp_bank), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            if (!agx_cdm_dispatch(stream, eng->k_copy_f32, (n_tokens * hp.n_embd + 255u) / 256u, 1, 1,
                                  &cp_prel, sizeof(cp_prel), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            bank_out_va = eng->va_bank_out;
            blocks      = prelude_blocks;
            // Keep bank slots intact while writing new pass blocks in upper slots if needed
            if (next_block_slot + 4u >= AGX_INKLING2_MAX_RES_BLOCKS) {
                next_block_slot = prelude_slot_mark;
            }
        }

        for (uint32_t ip = hp.core_start; ip < hp.core_end; ++ip) {
            const uint32_t il = (pass == 0u)
                ? ip
                : (hp.n_phys + (pass - 1u) * (hp.core_end - hp.core_start) + (ip - hp.core_start));
            const agx_inkling2_loop_tensors * lt = (pass > 0u) ? &weights.loop_layers[ip - hp.core_start] : nullptr;

            if (!encode_inkling2_layer(eng, stream, weights, kv_pool, il, ip,
                                       pass > 0u, lt, bank, bank_out_va,
                                       cur_va, alt_va, blocks, next_block_slot,
                                       n_tokens, n_kv_past)) {
                return false;
            }
            ++passes_done;
        }
    }

    // 3. Coda: layers core_end .. n_phys - 1 (18..23)
    for (uint32_t il = hp.core_end; il < hp.n_phys; ++il) {
        if (!encode_inkling2_layer(eng, stream, weights, kv_pool, il, il,
                                   false, nullptr, empty_bank, 0ull,
                                   cur_va, alt_va, blocks, next_block_slot,
                                   n_tokens, n_kv_past)) {
            return false;
        }
        ++passes_done;
    }

    // 4. Final Output: depth-mix(blocks, cur, output_res_query) + fused output_norm
    {
        std::vector<uint64_t> out_states;
        out_states.reserve(blocks.size() + 1u);
        for (int s = (int) blocks.size() - 1; s >= 0; --s) {
            out_states.push_back(blocks[s]);
        }
        out_states.push_back(cur_va);

        if (!agx_inkling2_dispatch_depth_mix(eng, stream,
                out_states.data(), (uint32_t) out_states.size(),
                agx_tensor_gpu_va(weights.output_res_query), 1u,
                agx_tensor_gpu_va(weights.output_norm),
                alt_va, 0ull,
                hp.n_embd, n_tokens, 1e-6f, hp.f_norm_rms_eps, AGX_BARRIER_STRONG_USC)) {
            return false;
        }
    }

    // 5. LM Head (`2304 -> 201024`) + Fused Softcap / Truncation / 40-Core Parallel Argmax
    if (!dispatch_dense_mm(eng->ctx, stream, weights.output, alt_va, eng->va_logits_pad,
                           n_tokens, AGX_BARRIER_STRONG_USC)) {
        return false;
    }

    if (!agx_inkling2_dispatch_softcap_argmax(eng, stream,
            eng->va_logits_pad,
            out_logits ? eng->va_logits_unpad : 0ull,
            eng->va_greedy_tok,
            hp.n_vocab_unpadded, hp.n_vocab_padded, n_tokens,
            hp.f_final_logit_softcapping, true, AGX_BARRIER_MEMORY_HOST)) {
        return false;
    }

    // ONE SINGLE CDM SUBMISSION FOR ALL 48 PASSES (ZERO CPU WAKEUPS MID-TOKEN!)
    uint64_t gpu_ns = 0;
    if (agx_cdm_submit(stream, true, &gpu_ns) != 0) {
        AGX_LOG_ERROR("agx_inkling2_run_forward: CDM submit failed");
        return false;
    }

    if (out_logits) {
        std::memcpy(out_logits, eng->cpu_logits_unpad, (size_t) n_tokens * hp.n_vocab_unpadded * sizeof(float));
    }

    if (out_stats) {
        agx_cdm_stats st_after{};
        agx_cdm_get_stats(stream, &st_after);
        out_stats->passes_executed = passes_done;
        out_stats->cdm_dispatches  = (uint32_t) (st_after.dispatches - st_before.dispatches);
        out_stats->cdm_submits     = 1u;
        out_stats->gpu_time_ns     = gpu_ns;
        out_stats->greedy_token    = eng->cpu_greedy_tok[n_tokens - 1u];
    }
    return true;
}

// ============================================================================
// TLA+-Verified Sense-Reversing Atomic Barrier Hardware Probe
// ============================================================================

bool agx_inkling2_run_sense_barrier_probe(agx_inkling2_engine * eng,
                                          uint32_t              n_workgroups,
                                          uint32_t              n_iters,
                                          uint64_t *            out_gpu_ns,
                                          uint32_t *            out_epoch) {
    if (!eng || !eng->ctx || !eng->ctx->stream || !eng->k_prep_sense_barrier) return false;
    agx_cdm_stream * stream = eng->ctx->stream;

    agx_bo * scratch = agx_bo_alloc(eng->ctx->drm,
                                    agx_align_up((size_t) (n_workgroups + 1u) * sizeof(uint32_t), AGX_CACHE_LINE_BYTES),
                                    AGX_BO_SLC_PIN | AGX_BO_WRITEBACK | AGX_BO_ZERO,
                                    "sense-barrier-probe");
    if (!scratch) return false;

    agx_cdm_stream_reset(stream);
    agx_cdm_reset_sense_barrier(stream, n_workgroups);

    agx_sense_barrier_push pc{};
    pc.barrier_va   = agx_cdm_get_sense_barrier_va(stream);
    pc.scratch_va   = scratch->gpu_va;
    pc.n_workgroups = n_workgroups;
    pc.n_iters      = n_iters;

    bool ok = agx_cdm_dispatch(stream, eng->k_prep_sense_barrier, n_workgroups, 1, 1,
                               &pc, sizeof(pc), AGX_BARRIER_MEMORY_HOST);
    uint64_t gpu_ns = 0;
    if (ok) {
        ok = (agx_cdm_submit(stream, true, &gpu_ns) == 0);
    }

    const auto * bar = static_cast<const agx_sense_barrier_state *>(agx_cdm_get_sense_barrier_bo(stream)->cpu_ptr);
    const auto * sc  = static_cast<const uint32_t *>(scratch->cpu_ptr);
    const bool coherent = ok && (sc[n_workgroups] == 0u) && (bar->epoch == 2u * n_iters);

    if (out_gpu_ns) *out_gpu_ns = gpu_ns;
    if (out_epoch)  *out_epoch  = bar->epoch;

    agx_bo_free(eng->ctx->drm, scratch);
    return coherent;
}

// ============================================================================
// Wave 4 Pillar 3: Precompile, Fused DSpark MTP & 48-Pass Megakernel Engine
// ============================================================================

bool agx_inkling2_precompile_w4(agx_inkling2_engine * eng) {
    if (!eng || !eng->ctx || !eng->ctx->compiler) return false;
    if (eng->w4_compiled) return true;

    agx_compiler * comp = eng->ctx->compiler;
    auto compile_file = [&](const char * name, const char * file, const char * def) -> const agx_kernel * {
        agx_compile_opts opts;
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        if (def && def[0]) opts.defines.emplace_back(def);
        return agx_compiler_compile_file(comp, name, file, opts);
    };
    auto compile_inline = [&](const char * name, const char * glsl) -> const agx_kernel * {
        agx_compile_opts opts;
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        return agx_compiler_compile_glsl(comp, name, glsl, opts);
    };

    eng->k_w4_qkv_prep_store  = compile_file("agx_w4_qkv_prep_store",  "agx_inkling2_qkv_prep_store.comp", nullptr);
    eng->k_w4_stage_post_attn = compile_file("agx_w4_stage_post_attn", "agx_inkling2_stage_prep.comp",     "STAGE_MODE=0");
    eng->k_w4_stage_post_ffn  = compile_file("agx_w4_stage_post_ffn",  "agx_inkling2_stage_prep.comp",     "STAGE_MODE=1");
    eng->k_w4_rel_bias_mask   = compile_file("agx_w4_rel_bias_mask",   "agx_inkling2_stage_prep.comp",     "STAGE_MODE=2");
    eng->k_w4_fwht_qgate      = compile_file("agx_w4_fwht_qgate",      "agx_inkling2_stage_prep.comp",     "STAGE_MODE=3");
    eng->k_w4_ctrl_norm_cat   = compile_file("agx_w4_ctrl_norm_cat",   "agx_inkling2_stage_prep.comp",     "STAGE_MODE=4");
    eng->k_w4_ctrl_mix_norm   = compile_file("agx_w4_ctrl_mix_norm",   "agx_inkling2_stage_prep.comp",     "STAGE_MODE=5");
    eng->k_w4_ctrl_up_mix     = compile_file("agx_w4_ctrl_up_mix",     "agx_inkling2_stage_prep.comp",     "STAGE_MODE=6");
    eng->k_w4_batch_rmsnorm   = compile_file("agx_w4_batch_rmsnorm",   "agx_inkling2_stage_prep.comp",     "STAGE_MODE=7");
    eng->k_w4_dspark_pack      = compile_file("agx_w4_dspark_pack",      "agx_inkling2_dspark_mtp.comp",     "DSPARK_FUSED_MODE=0");
    eng->k_w4_dspark_gru       = compile_file("agx_w4_dspark_gru",       "agx_inkling2_dspark_mtp.comp",     "DSPARK_FUSED_MODE=3");
    eng->k_w4_dspark_w2        = compile_file("agx_w4_dspark_w2",        "agx_inkling2_dspark_mtp.comp",     "DSPARK_FUSED_MODE=1");
    eng->k_w4_dspark_merge     = compile_file("agx_w4_dspark_merge",     "agx_inkling2_dspark_mtp.comp",     "DSPARK_FUSED_MODE=2");
    eng->k_w7_dspark_joint_gru = compile_file("agx_w7_dspark_joint_gru", "agx_inkling2_dspark_mtp.comp",     "DSPARK_FUSED_MODE=4");
    eng->k_w7_dspark_inp_block = compile_file("agx_w7_dspark_inp_block", "agx_inkling2_dspark_mtp.comp",     "DSPARK_FUSED_MODE=5");
    eng->k_w7_dspark_qkv_rope  = compile_file("agx_w7_dspark_qkv_rope",  "agx_inkling2_dspark_mtp.comp",     "DSPARK_FUSED_MODE=6");
    eng->k_w7_dspark_post_op   = compile_file("agx_w7_dspark_post_op",   "agx_inkling2_dspark_mtp.comp",     "DSPARK_FUSED_MODE=7");
    auto compile_file_defs = [&](const char * name, const char * file,
                                 std::initializer_list<const char *> defs) -> const agx_kernel * {
        agx_compile_opts opts;
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        for (const char * d : defs) {
            if (d && d[0]) opts.defines.emplace_back(d);
        }
        return agx_compiler_compile_file(comp, name, file, opts);
    };
    eng->k_w7_dspark_gate_up_glu    = compile_file("agx_w7_dspark_gate_up_glu", "agx_inkling2_dspark_mtp.comp", "DSPARK_FUSED_MODE=8");
    eng->k_w7_dspark_gate_up_glu_m8 = compile_file_defs("agx_w7_dspark_gate_up_glu_m8", "agx_inkling2_dspark_mtp.comp",
                                                        { "DSPARK_FUSED_MODE=8", "BATCH_CAP=8" });
    eng->k_w8_dspark_prep      = compile_file("agx_w8_dspark_prep",      "agx_inkling2_dspark_mtp.comp",     "DSPARK_FUSED_MODE=9");
    eng->k_w8_dspark_inject_mix= compile_file("agx_w8_dspark_inject_mix","agx_inkling2_dspark_mtp.comp",     "DSPARK_FUSED_MODE=10");
    eng->k_w11_dspark_inject_mix_fused = compile_file("agx_w11_dspark_inject_mix_fused", "agx_inkling2_dspark_mtp.comp", "DSPARK_FUSED_MODE=14");
    eng->k_w8_dspark_inject_kv = compile_file("agx_w11_dspark_inject_kv", "agx_inkling2_dspark_mtp.comp",    "DSPARK_FUSED_MODE=11");
    eng->k_w9_q8_gemv_m4_f16   = compile_file_defs("agx_w9_q8_gemv_m4_f16", "agx_inkling2_dspark_mtp.comp",
                                                   { "DSPARK_FUSED_MODE=12", "BATCH_CAP=4", "X_IS_F16=1" });
    eng->k_w9_q8_gemv_m4_f32   = compile_file_defs("agx_w9_q8_gemv_m4_f32", "agx_inkling2_dspark_mtp.comp",
                                                   { "DSPARK_FUSED_MODE=12", "BATCH_CAP=4", "X_IS_F16=0" });
    eng->k_w9_q8_gemv_m8_f16   = compile_file_defs("agx_w9_q8_gemv_m8_f16", "agx_inkling2_dspark_mtp.comp",
                                                   { "DSPARK_FUSED_MODE=12", "BATCH_CAP=8", "X_IS_F16=1" });
    eng->k_w9_q8_gemv_m8_f32   = compile_file_defs("agx_w9_q8_gemv_m8_f32", "agx_inkling2_dspark_mtp.comp",
                                                   { "DSPARK_FUSED_MODE=12", "BATCH_CAP=8", "X_IS_F16=0" });
    eng->k_w9_dspark_w2_q4     = compile_file("agx_w9_dspark_w2_q4",     "agx_inkling2_dspark_mtp.comp",     "DSPARK_FUSED_MODE=13");
    eng->k_w4_router_topk      = compile_inline("agx_w4_router_topk",   k_glsl_w4_router_topk_norm);
    eng->k_w4_gather_f32       = compile_inline("agx_w4_gather_f32",    k_glsl_w4_gather_f32);
    eng->k_w4_fill_f32         = compile_inline("agx_w4_fill_f32",      k_glsl_w4_fill_f32);
    eng->k_w4_scatter_rows_f32 = compile_inline("agx_w4_scatter_rows_f32", k_glsl_w4_scatter_rows_f32);
    eng->k_w10_rel_bias_table       = compile_inline("agx_w10_rel_bias_table", k_glsl_w10_rel_bias_table);
    eng->k_w10_qkv_prep_store_rot64 = compile_inline("agx_w10_qkv_prep_store_rot64", k_glsl_w10_qkv_prep_store_rot64);
    eng->w10_qkv_rot64_tried        = true;

    eng->w4_compiled = eng->k_w4_qkv_prep_store && eng->k_w4_stage_post_attn &&
                       eng->k_w4_stage_post_ffn && eng->k_w4_rel_bias_mask &&
                       eng->k_w4_fwht_qgate && eng->k_w4_ctrl_norm_cat &&
                       eng->k_w4_ctrl_mix_norm && eng->k_w4_ctrl_up_mix &&
                       eng->k_w4_batch_rmsnorm && eng->k_w4_dspark_pack &&
                       eng->k_w4_dspark_gru && eng->k_w4_dspark_w2 &&
                       eng->k_w4_dspark_merge && eng->k_w7_dspark_joint_gru &&
                       eng->k_w7_dspark_inp_block && eng->k_w7_dspark_qkv_rope &&
                       eng->k_w7_dspark_post_op && eng->k_w7_dspark_gate_up_glu &&
                       eng->k_w8_dspark_prep && eng->k_w8_dspark_inject_mix &&
                       eng->k_w11_dspark_inject_mix_fused &&
                       eng->k_w8_dspark_inject_kv &&
                       eng->k_w4_router_topk &&
                       eng->k_w4_gather_f32 && eng->k_w4_fill_f32 &&
                       eng->k_w4_scatter_rows_f32 && eng->k_w10_rel_bias_table &&
                       eng->k_w10_qkv_prep_store_rot64;
    return eng->w4_compiled;
}

static const agx_kernel * ensure_w10_qkv_rot64_kernel(agx_inkling2_engine * eng) {
    if (!eng || !eng->ctx || !eng->ctx->compiler) return nullptr;
    if (!eng->w10_qkv_rot64_tried) {
        eng->w10_qkv_rot64_tried = true;
        agx_compile_opts opts;
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        eng->k_w10_qkv_prep_store_rot64 = agx_compiler_compile_glsl(
            eng->ctx->compiler, "agx_w10_qkv_prep_store_rot64", k_glsl_w10_qkv_prep_store_rot64, opts);
    }
    return eng->k_w10_qkv_prep_store_rot64;
}

static const agx_kernel * compile_w5_persist_kernel(
        agx_compiler * comp, const char * name,
        int persist_mode, int has_loop, int kv_q8, int moe_mode) {
    agx_compile_opts opts;
    opts.is_hot_kernel     = true;
    opts.strict_zero_spill = true;
    opts.defines.emplace_back("PERSIST_MODE=" + std::to_string(persist_mode));
    opts.defines.emplace_back("HAS_LOOP="     + std::to_string(has_loop));
    opts.defines.emplace_back("KV_Q8="        + std::to_string(kv_q8));
    opts.defines.emplace_back("MOE_MODE="     + std::to_string(moe_mode));
    return agx_compiler_compile_file(comp, name, "agx_inkling2_persist_b1.comp", opts);
}

static bool ensure_w5_chunk_kernels(agx_inkling2_engine * eng, uint32_t kv_q8) {
    if (!eng || !eng->ctx || !eng->ctx->compiler || kv_q8 > 1u) return false;
    if (eng->w5_chunk_tried[kv_q8]) return eng->w5_chunk_ok[kv_q8];
    eng->w5_chunk_tried[kv_q8] = true;
    agx_compiler * comp = eng->ctx->compiler;
    const char * kv_s = kv_q8 ? "q8" : "f16";
    char nm[64];
    std::snprintf(nm, sizeof(nm), "agx_w5_persist_chunk_nvfp4_%s_nl", kv_s);
    eng->k_w5_persist_chunk[kv_q8][0][0] = compile_w5_persist_kernel(comp, nm, 2, 0, (int) kv_q8, 0);
    std::snprintf(nm, sizeof(nm), "agx_w5_persist_chunk_nvfp4_%s_lp", kv_s);
    eng->k_w5_persist_chunk[kv_q8][0][1] = compile_w5_persist_kernel(comp, nm, 2, 1, (int) kv_q8, 0);
    std::snprintf(nm, sizeof(nm), "agx_w5_persist_chunk_q8_%s_nl", kv_s);
    eng->k_w5_persist_chunk[kv_q8][1][0] = compile_w5_persist_kernel(comp, nm, 2, 0, (int) kv_q8, 2);

    eng->w5_chunk_ok[kv_q8] = eng->k_w5_persist_chunk[kv_q8][0][0] &&
                              eng->k_w5_persist_chunk[kv_q8][0][1] &&
                              eng->k_w5_persist_chunk[kv_q8][1][0];
    return eng->w5_chunk_ok[kv_q8];
}

static bool ensure_w5_split_kernels(agx_inkling2_engine * eng, uint32_t kv_q8) {
    if (!eng || !eng->ctx || !eng->ctx->compiler || kv_q8 > 1u) return false;
    if (eng->w5_split_tried[kv_q8]) return eng->w5_split_ok[kv_q8];
    eng->w5_split_tried[kv_q8] = true;
    agx_compiler * comp = eng->ctx->compiler;
    const char * kv_s = kv_q8 ? "q8" : "f16";
    char nm[64];
    std::snprintf(nm, sizeof(nm), "agx_w5_persist_attn_%s_nl", kv_s);
    eng->k_w5_persist_attn[kv_q8][0] = compile_w5_persist_kernel(comp, nm, 0, 0, (int) kv_q8, 0);
    std::snprintf(nm, sizeof(nm), "agx_w5_persist_attn_%s_lp", kv_s);
    eng->k_w5_persist_attn[kv_q8][1] = compile_w5_persist_kernel(comp, nm, 0, 1, (int) kv_q8, 0);
    if (!eng->k_w5_persist_moe[0][0]) {
        eng->k_w5_persist_moe[0][0] = compile_w5_persist_kernel(comp, "agx_w5_persist_moe_nvfp4_nl", 1, 0, 0, 0);
        eng->k_w5_persist_moe[0][1] = compile_w5_persist_kernel(comp, "agx_w5_persist_moe_nvfp4_lp", 1, 1, 0, 0);
        eng->k_w5_persist_moe[1][0] = compile_w5_persist_kernel(comp, "agx_w5_persist_moe_q8_nl",    1, 0, 0, 2);
    }
    eng->w5_split_ok[kv_q8] = eng->k_w5_persist_attn[kv_q8][0] &&
                              eng->k_w5_persist_attn[kv_q8][1] &&
                              eng->k_w5_persist_moe[0][0] &&
                              eng->k_w5_persist_moe[0][1] &&
                              eng->k_w5_persist_moe[1][0];
    return eng->w5_split_ok[kv_q8];
}

static const agx_kernel * compile_w5_hybrid_kernel(
        agx_compiler * comp, const char * name,
        int hybrid_mode, int has_loop, int kv_q8) {
    agx_compile_opts opts;
    opts.is_hot_kernel     = true;
    opts.strict_zero_spill = true;
    opts.defines.emplace_back("HYBRID_MODE=" + std::to_string(hybrid_mode));
    opts.defines.emplace_back("HAS_LOOP="    + std::to_string(has_loop));
    opts.defines.emplace_back("KV_Q8="       + std::to_string(kv_q8));
    return agx_compiler_compile_file(comp, name, "agx_inkling2_hybrid_b1.comp", opts);
}

static bool ensure_w5_hybrid_kernels(agx_inkling2_engine * eng, uint32_t kv_q8) {
    if (!eng || !eng->ctx || !eng->ctx->compiler || kv_q8 > 1u) return false;
    if (eng->w5_hybrid_tried[kv_q8]) return eng->w5_hybrid_ok[kv_q8];
    eng->w5_hybrid_tried[kv_q8] = true;
    agx_compiler * comp = eng->ctx->compiler;
    const char * kv_s = kv_q8 ? "q8" : "f16";
    char nm[64];
    std::snprintf(nm, sizeof(nm), "agx_w5_hybrid_qkv_flash_%s", kv_s);
    eng->k_w5_hybrid_qkv_flash[kv_q8] = compile_w5_hybrid_kernel(comp, nm, 0, 0, (int) kv_q8);
    if (!eng->k_w5_attn_in_b1) {
        agx_compile_opts aopts;
        aopts.is_hot_kernel     = true;
        aopts.strict_zero_spill = true;
        eng->k_w5_attn_in_b1 = agx_compiler_compile_file(comp, "agx_w5_attn_in_b1", "agx_inkling2_attn_in_b1.comp", aopts);
    }
    if (!eng->k_w5_hybrid_post_attn_rtr[0]) {
        eng->k_w5_hybrid_post_attn_rtr[0] = compile_w5_hybrid_kernel(comp, "agx_w5_hybrid_post_attn_rtr_nl", 1, 0, 0);
        eng->k_w5_hybrid_post_attn_rtr[1] = compile_w5_hybrid_kernel(comp, "agx_w5_hybrid_post_attn_rtr_lp", 1, 1, 0);
        eng->k_w5_hybrid_post_ffn_ctrl    = compile_w5_hybrid_kernel(comp, "agx_w5_hybrid_post_ffn_ctrl",    2, 1, 0);
    }
    eng->w5_hybrid_ok[kv_q8] = eng->k_w5_hybrid_qkv_flash[kv_q8] &&
                               eng->k_w5_hybrid_post_attn_rtr[0] &&
                               eng->k_w5_hybrid_post_attn_rtr[1] &&
                               eng->k_w5_hybrid_post_ffn_ctrl;
    return eng->w5_hybrid_ok[kv_q8];
}

static bool ensure_w5_hybrid_m_kernels(agx_inkling2_engine * eng) {
    if (!eng || !eng->ctx || !eng->ctx->compiler) return false;
    if (eng->w5_hybrid_m_tried) return eng->w5_hybrid_m_ok;
    eng->w5_hybrid_m_tried = true;
    agx_compiler * comp = eng->ctx->compiler;
    eng->k_w5_hybrid_post_attn_rtr_m[0] = compile_w5_hybrid_kernel(comp, "agx_w5_hybrid_post_attn_rtr_m_nl", 3, 0, 0);
    eng->k_w5_hybrid_post_attn_rtr_m[1] = compile_w5_hybrid_kernel(comp, "agx_w5_hybrid_post_attn_rtr_m_lp", 3, 1, 0);
    eng->k_w5_hybrid_post_ffn_ctrl_m    = compile_w5_hybrid_kernel(comp, "agx_w5_hybrid_post_ffn_ctrl_m",    4, 1, 0);
    eng->k_w5_hybrid_post_ffn_nl_m      = compile_w5_hybrid_kernel(comp, "agx_w5_hybrid_post_ffn_nl_m",      4, 0, 0);
    {
        agx_compile_opts a4;
        a4.is_hot_kernel     = true;
        a4.strict_zero_spill = true;
        a4.defines.emplace_back("BATCH_M=4");
        eng->k_w9_attn_in_m4 = agx_compiler_compile_file(comp, "agx_w9_attn_in_m4", "agx_inkling2_attn_in_b1.comp", a4);

        agx_compile_opts a4_f16 = a4;
        a4_f16.defines.emplace_back("X_IS_F16=1");
        eng->k_w9_attn_in_m4_f16 = agx_compiler_compile_file(comp, "agx_w9_attn_in_m4_f16", "agx_inkling2_attn_in_b1.comp", a4_f16);

        agx_compile_opts a8;
        a8.is_hot_kernel     = true;
        a8.strict_zero_spill = true;
        a8.defines.emplace_back("BATCH_M=8");
        eng->k_w9_attn_in_m8 = agx_compiler_compile_file(comp, "agx_w9_attn_in_m8", "agx_inkling2_attn_in_b1.comp", a8);

        agx_compile_opts a8_f16 = a8;
        a8_f16.defines.emplace_back("X_IS_F16=1");
        eng->k_w9_attn_in_m8_f16 = agx_compiler_compile_file(comp, "agx_w9_attn_in_m8_f16", "agx_inkling2_attn_in_b1.comp", a8_f16);
    }
    eng->w5_hybrid_m_ok = eng->k_w5_hybrid_post_attn_rtr_m[0] &&
                          eng->k_w5_hybrid_post_attn_rtr_m[1] &&
                          eng->k_w5_hybrid_post_ffn_ctrl_m;
    return eng->w5_hybrid_m_ok;
}

bool agx_inkling2_precompile_w5(agx_inkling2_engine * eng) {
    if (!eng || !eng->ctx || !eng->ctx->compiler) return false;
    if (eng->w5_compiled) return true;
    const bool h0 = ensure_w5_hybrid_kernels(eng, 0u);
    const bool h1 = ensure_w5_hybrid_kernels(eng, 1u);
    const bool hm = ensure_w5_hybrid_m_kernels(eng);
    const bool s0 = ensure_w5_split_kernels(eng, 0u);
    const bool s1 = ensure_w5_split_kernels(eng, 1u);
    const bool c0 = ensure_w5_chunk_kernels(eng, 0u);
    const bool c1 = ensure_w5_chunk_kernels(eng, 1u);
    eng->w5_compiled = h0 && h1 && hm && s0 && s1 && c0 && c1;
    return eng->w5_compiled;
}

// Static SLC arena sub-layout across `slc->extra_scratch` (2.43 MiB) & `slc->sconv_state` (3.65 MiB)
struct agx_w4_slc_layout {
    uint64_t x_q;               // slc->ctrl_scratch + 0          (73,728 B)
    uint64_t x_k;               // slc->ctrl_scratch + 73,728     (73,728 B) -> also wo_out
    uint64_t x_v;               // slc->ctrl_scratch + 147,456    (73,728 B) -> dedicated next-pass x_v (NO moe_out alias!)
    uint64_t qkv_in_0;          // slc->ctrl_scratch + 221,184    (73,728 B) -> also ffn_pre
    uint64_t router_logits;     // slc->router_topk  + 8,192      (4,096 B)

    uint64_t moe_out;           // extra + ...                    (73,728 B, strictly disjoint from x_v for M=1..8 conv!)

    uint64_t q_out;             // extra + 0                      (98,304 B)
    uint64_t qgate_compact;     // extra + 98,304                 (98,304 B)
    uint64_t r_out;             // extra + 196,608                (24,576 B)
    uint64_t qkv_in_1;          // extra + 221,184                (73,728 B)
    uint64_t qkv_in_2;          // extra + 294,912                (73,728 B)
    uint64_t k_dyn;             // extra + 368,640                (4,096 B)
    uint64_t v_dyn;             // extra + 372,736                (4,096 B)
    uint64_t fallback_qkv;      // extra + 376,832                (287,488 B)
    uint64_t dspark_state;      // extra + ...                    (16,384 B)
    uint64_t dspark_markov;     // extra + ...                    (16,384 B)
    uint64_t dspark_prev;       // extra + ...                    (4,096 B)
    uint64_t x_q_f16;           // Wave 10: FP16 mirror of x_q    (36,864 B, 18 KB @ M=4 < 32 KB L1!)
    uint64_t x_k_f16;           // Wave 10: FP16 mirror of x_k    (36,864 B)
    uint64_t x_v_f16;           // Wave 10: FP16 mirror of x_v    (36,864 B)
    uint64_t sinks_compact;     // Wave 10: packed [M, 24] sinks  (1,024 B)

    // 40-core Loop Controller & DSpark scratch inside `slc->sconv_state` (3,833,856 B, unused by MK!)
    uint64_t ctrl_cat[3];       // 3 * (8 * 4608 * 4 = 147,456 B) = 442,368 B
    uint64_t ctrl_gate[3];      // 3 * (8 *  256 * 4 =   8,192 B) =  24,576 B
    uint64_t ctrl_gain[3];      // 3 * (8 * 2304 * 4 =  73,728 B) = 221,184 B
    uint64_t ctrl_forget[3];    // 3 * (8 * 2304 * 4 =  73,728 B) = 221,184 B
    uint64_t dspark_in;         // 4 * 6144 * 4                   =  98,304 B
    uint64_t dspark_joint_out;  // 4 * 1536 * 4                   =  24,576 B
    uint64_t dspark_cand_val;   // 4 * 160 * 64 * 4               = 163,840 B
    uint64_t dspark_cand_idx;   // 4 * 160 * 64 * 4               = 163,840 B
    uint64_t kq_mask_f16;       // remainder of sconv_state       (~2.47 MiB!)
    size_t   kq_mask_f16_bytes;
};

static agx_w4_slc_layout agx_w4_partition_slc(const agx_slc_arena * slc) {
    agx_w4_slc_layout L{};
    constexpr uint64_t kEmbdB = AGX_SLC_ARENA_H_CUR_BYTES; // 73,728 B
    L.x_q           = slc->ctrl_scratch;
    L.x_k           = slc->ctrl_scratch + kEmbdB;
    L.x_v           = slc->ctrl_scratch + 2u * kEmbdB;
    L.qkv_in_0      = slc->ctrl_scratch + 3u * kEmbdB;
    L.router_logits = slc->router_topk + 8192u;

    uint64_t cur = slc->extra_scratch;
    auto take_extra = [&](size_t sz) -> uint64_t {
        uint64_t va = cur;
        cur += agx_align_up(sz, AGX_CACHE_LINE_BYTES);
        return va;
    };
    L.q_out           = take_extra(98304u);
    L.qgate_compact   = take_extra(98304u);
    L.r_out           = take_extra(24576u);
    L.qkv_in_1        = take_extra(kEmbdB);
    L.qkv_in_2        = take_extra(kEmbdB);
    L.moe_out         = take_extra(kEmbdB);
    L.k_dyn           = take_extra(4096u);
    L.v_dyn           = take_extra(4096u);
    L.fallback_qkv    = take_extra(AGX_SLC_ARENA_QKV_BUF_BYTES);
    L.dspark_state    = take_extra(16384u);
    L.dspark_markov   = take_extra(16384u);
    L.dspark_prev     = take_extra(4096u);
    L.x_q_f16         = take_extra(kEmbdB / 2u);
    L.x_k_f16         = take_extra(kEmbdB / 2u);
    L.x_v_f16         = take_extra(kEmbdB / 2u);
    L.sinks_compact   = take_extra(1024u);

    uint64_t scur = slc->sconv_state;
    auto take_sc = [&](size_t sz) -> uint64_t {
        uint64_t va = scur;
        scur += agx_align_up(sz, AGX_CACHE_LINE_BYTES);
        return va;
    };
    for (int j = 0; j < 3; ++j) L.ctrl_cat[j]    = take_sc(2u * kEmbdB);
    for (int j = 0; j < 3; ++j) L.ctrl_gate[j]   = take_sc(8192u);
    for (int j = 0; j < 3; ++j) L.ctrl_gain[j]   = take_sc(kEmbdB);
    for (int j = 0; j < 3; ++j) L.ctrl_forget[j] = take_sc(kEmbdB);
    L.dspark_in        = take_sc(98304u);
    L.dspark_joint_out = take_sc(24576u);
    L.dspark_cand_val  = take_sc(163840u);
    L.dspark_cand_idx  = take_sc(163840u);
    L.kq_mask_f16      = scur;
    const uint64_t sc_end = slc->sconv_state + AGX_SLC_ARENA_SCONV_STATE_BYTES;
    L.kq_mask_f16_bytes   = (sc_end > scur) ? (size_t) (sc_end - scur) : 0u;
    return L;
}

// Wave 5 Theorem W5.1 & AgxPersistentW5.tla C++ static_assert mirror:
// Proves every sub-slice inside agx_w4_slc_layout is 128B-aligned, pairwise disjoint,
// fits strictly inside its parent arena region, and keeps `moe_out` disjoint from `x_v` (#132).
static constexpr bool agx_w5_verify_w4_sublayout() {
    constexpr uint64_t kEmbdB = AGX_SLC_ARENA_H_CUR_BYTES; // 73,728 B
    constexpr auto a128 = [](uint64_t x) constexpr -> uint64_t {
        return ((x + AGX_CACHE_LINE_BYTES - 1ULL) / AGX_CACHE_LINE_BYTES) * AGX_CACHE_LINE_BYTES;
    };
    // 1. ctrl_scratch sub-slices (x_q, x_k, x_v, qkv_in_0)
    constexpr uint64_t c_base = AGX_SLC_ARENA_OFF_CTRL_SCRATCH;
    constexpr uint64_t x_q = c_base + 0ULL * kEmbdB;
    constexpr uint64_t x_k = c_base + 1ULL * kEmbdB;
    constexpr uint64_t x_v = c_base + 2ULL * kEmbdB;
    constexpr uint64_t q0  = c_base + 3ULL * kEmbdB;
    if ((x_q % 128u) || (x_k % 128u) || (x_v % 128u) || (q0 % 128u)) return false;
    if (q0 + kEmbdB > c_base + AGX_SLC_ARENA_CTRL_BYTES) return false;

    // 2. router_topk sub-slices (ids +0, w +4096, logits +8192)
    if ((8192u % 128u) != 0u) return false;
    if (8192u + 4096u > AGX_SLC_ARENA_ROUTER_TOPK_BYTES) return false;

    // 3. extra_scratch sub-slices (including dedicated moe_out strictly disjoint from x_v!)
    constexpr uint64_t ex_sizes[] = {
        98304u, 98304u, 24576u, kEmbdB, kEmbdB, kEmbdB /* moe_out */,
        4096u, 4096u, AGX_SLC_ARENA_QKV_BUF_BYTES, 16384u, 16384u, 4096u,
        kEmbdB / 2u, kEmbdB / 2u, kEmbdB / 2u, 1024u
    };
    uint64_t cur = AGX_SLC_ARENA_OFF_EXTRA_SCRATCH;
    uint64_t moe_out_off = 0;
    for (size_t i = 0; i < sizeof(ex_sizes) / sizeof(ex_sizes[0]); ++i) {
        if ((cur % 128u) != 0u) return false;
        if (i == 5u) moe_out_off = cur;
        cur += a128(ex_sizes[i]);
    }
    if (cur > AGX_SLC_ARENA_TOTAL_BYTES) return false;
    // Explicit #132 proof: moe_out [moe_out_off, +kEmbdB) is strictly disjoint from x_v [x_v, +kEmbdB)
    if (!(moe_out_off >= x_v + kEmbdB || x_v >= moe_out_off + kEmbdB)) return false;

    // 4. sconv_state sub-slices (ctrl_cat/gate/gain/forget + dspark + kq_mask_f16 >= 2 MiB)
    constexpr uint64_t sc_sizes[] = {
        2u * kEmbdB, 2u * kEmbdB, 2u * kEmbdB,
        8192u, 8192u, 8192u,
        kEmbdB, kEmbdB, kEmbdB,
        kEmbdB, kEmbdB, kEmbdB,
        98304u, 24576u, 163840u, 163840u
    };
    uint64_t scur = AGX_SLC_ARENA_OFF_SCONV_STATE;
    for (uint64_t sz : sc_sizes) {
        if ((scur % 128u) != 0u) return false;
        scur += a128(sz);
    }
    if ((scur % 128u) != 0u) return false;
    constexpr uint64_t sc_end = AGX_SLC_ARENA_OFF_SCONV_STATE + AGX_SLC_ARENA_SCONV_STATE_BYTES;
    if (scur >= sc_end) return false;
    if ((sc_end - scur) < 2ULL * 1024ULL * 1024ULL) return false; // >= 2 MiB for kq_mask_f16
    return true;
}
static_assert(agx_w5_verify_w4_sublayout(),
              "Wave 5 Theorem W5.1: agx_w4_slc_layout sub-slices must be 128B-aligned, disjoint, and moe_out != x_v");

static bool agx_dspark_dispatch_markov_head_w7(agx_inkling2_engine *             eng,
                                               agx_cdm_stream *                  stream,
                                               const agx_dspark_mtp_tensors &    w,
                                               uint64_t                          anchor_toks_va,
                                               uint64_t                          base_logits_va,
                                               uint32_t                          base_stride_f32,
                                               uint64_t                          hidden_va,
                                               uint64_t                          f_rnn_va,
                                               uint64_t                          rnn_state_va,
                                               uint64_t                          out_logits_va,
                                               uint64_t                          out_h_nextn_va,
                                               uint32_t                          n_blocks,
                                               uint32_t                          block_drafts,
                                               agx_barrier_mode                  post_barrier,
                                               const agx_w7_dspark_topk_entry *  topk_table = nullptr,
                                               uint32_t                          n_topk_rows = 0u) {
    if (!eng || !stream || !w.markov_w1 || !w.markov_w2 || n_blocks == 0 || n_blocks > 4 || block_drafts == 0) {
        return false;
    }
    if (!agx_inkling2_precompile_w4(eng)) {
        return false;
    }
    agx_slc_arena * slc = agx_backend_get_slc_arena(eng->ctx);
    if (!slc) return false;
    const agx_w4_slc_layout L = agx_w4_partition_slc(slc);

    const uint32_t i_beg      = w.sample_from_anchor ? 0u : 1u;
    const uint32_t base_str   = (base_stride_f32 > 0u) ? base_stride_f32 : w.n_vocab;
    const bool     has_frnn   = (w.has_f_rnn && f_rnn_va != 0ull);

    uint64_t topk_table_va = 0ull;
    const bool has_topk_tbl = (topk_table != nullptr && n_topk_rows == n_blocks * block_drafts);
    if (has_topk_tbl) {
        void * tbl_cpu = nullptr;
        topk_table_va = agx_cdm_alloc_args(stream, (size_t) n_topk_rows * sizeof(agx_w7_dspark_topk_entry),
                                           16u, &tbl_cpu);
        if (!topk_table_va || !tbl_cpu) return false;
        std::memcpy(tbl_cpu, topk_table, (size_t) n_topk_rows * sizeof(agx_w7_dspark_topk_entry));
    }
    auto step_has_topk = [&](uint32_t step) -> bool {
        if (!has_topk_tbl) return false;
        for (uint32_t b = 0u; b < n_blocks; ++b) {
            if (topk_table[b * block_drafts + step].ids_va != 0ull) return true;
        }
        return false;
    };
    const bool need_slot0_topk = (!w.sample_from_anchor && step_has_topk(0u));

    uint32_t w1_rb = 0, w2_rb = 0;
    const uint32_t w1_wt = agx_inkling2_tensor_wtype(w.markov_w1, &w1_rb);
    const uint32_t w2_wt = agx_inkling2_tensor_wtype(w.markov_w2, &w2_rb);
    if (!agx_gemm_init(eng->ctx)) {
        return false;
    }

    const uint32_t n_splits = 40u;  // 1 WG/core x 512T = 16 SGs/core (Top-1 Pigeonhole + 2x smaller merge!)
    const uint32_t n_cand   = std::min<uint32_t>(64u, w.n_vocab);

    // 1-Dispatch Fused MARKOV_INIT: anchor_toks -> prev_tok + [slot-0 logit copy] + h_nextn[0] zero + step i_beg joint_in pack!
    const bool can_pack_init = (w.joint_proj != nullptr && i_beg < block_drafts);
    {
        agx_w7_dspark_post_op_push ipc{};
        ipc.va0 = anchor_toks_va;
        ipc.va1 = base_logits_va;
        ipc.va2 = out_logits_va;
        ipc.va3 = (rnn_state_va != 0ull) ? out_h_nextn_va : 0ull;
        ipc.va4 = L.dspark_prev;
        ipc.va5 = agx_tensor_gpu_va(w.markov_w1);
        ipc.va6 = hidden_va;
        ipc.va7 = f_rnn_va;
        ipc.va8 = L.dspark_in;
        ipc.op  = 2u; // MARKOV_INIT
        ipc.p0  = w.n_vocab;
        ipc.p1  = base_str;
        ipc.p2  = n_blocks;
        ipc.p3  = block_drafts;
        ipc.p4  = w.n_embd;
        ipc.p5  = w.rank;
        ipc.p6  = w1_wt;
        ipc.p7  = w1_rb;
        ipc.p8  = (w.sample_from_anchor ? 1u : 0u) |
                  (has_frnn ? 2u : 0u) |
                  (can_pack_init ? 4u : 0u) |
                  (need_slot0_topk ? 8u : 0u);
        const uint32_t init_wgs = need_slot0_topk ? 1u : 160u;
        const agx_barrier_mode init_bar = need_slot0_topk
            ? AGX_BARRIER_NONE
            : ((i_beg < block_drafts) ? AGX_BARRIER_STRONG_USC : post_barrier);
        if (!agx_cdm_dispatch(stream, eng->k_w7_dspark_post_op, init_wgs, n_blocks, 1u,
                              &ipc, sizeof(ipc), init_bar)) {
            return false;
        }
    }

    if (need_slot0_topk) {
        // Concurrent with 1-WG MARKOV_INIT: 160-WG copy slot 0 + emit top-64, then 1-WG merge -> top_k[0]!
        agx_w4_dspark_w2_push w0{};
        w0.base_va         = base_logits_va;
        w0.out_logits_va   = out_logits_va;
        w0.cand_val_va     = L.dspark_cand_val;
        w0.cand_idx_va     = L.dspark_cand_idx;
        w0.rank            = w.rank;
        w0.n_vocab         = w.n_vocab;
        w0.n_blocks        = n_blocks;
        w0.block_drafts    = block_drafts;
        w0.draft_pos       = 0u;
        w0.n_splits        = n_splits;
        w0.flags           = AGX_W4_DSPARK_F_BASE_ONLY | AGX_W4_DSPARK_F_EMIT_TOP64;
        w0.base_stride_f32 = base_str;
        if (!agx_cdm_dispatch(stream, eng->k_w4_dspark_w2, n_splits, n_blocks, 1u,
                              &w0, sizeof(w0), AGX_BARRIER_STRONG_USC)) {
            return false;
        }
        agx_w4_dspark_merge_push m0{};
        m0.cand_val_va       = L.dspark_cand_val;
        m0.cand_idx_va       = L.dspark_cand_idx;
        m0.out_topk_table_va = topk_table_va;
        m0.n_embd            = w.n_embd;
        m0.n_blocks          = n_blocks;
        m0.block_drafts      = block_drafts;
        m0.draft_pos         = 0u;
        m0.n_splits          = n_splits;
        m0.n_cand            = n_cand;
        if (!agx_cdm_dispatch(stream, eng->k_w4_dspark_merge, n_blocks, 1u, 1u,
                              &m0, sizeof(m0),
                              (i_beg < block_drafts) ? AGX_BARRIER_STRONG_USC : post_barrier)) {
            return false;
        }
    }

    float w2_scale = 1.0f;
    if (w.markov_w2_s && w.markov_w2_s->data && w.markov_w2_s->type == GGML_TYPE_F32) {
        w2_scale = *static_cast<const float *>(w.markov_w2_s->data);
    }

    uint32_t jp_rb = 0;
    const uint32_t jp_wt = w.joint_proj ? agx_inkling2_tensor_wtype(w.joint_proj, &jp_rb) : 0u;
    const bool use_fused_joint_gru =
        w.joint_proj != nullptr &&
        (w.joint_proj->type == GGML_TYPE_F32 || w.joint_proj->type == GGML_TYPE_F16 ||
         w.joint_proj->type == GGML_TYPE_BF16 || w.joint_proj->type == GGML_TYPE_Q8_0) &&
        (w.joint_proj->ne[0] % 8 == 0);
    const uint32_t n_joint = 2u * w.rank + (has_frnn ? 2u * w.n_embd : w.n_embd);

    bool joint_in_packed = can_pack_init;

    for (uint32_t i = i_beg; i < block_drafts; ++i) {
        const bool is_last   = (i + 1u == block_drafts);
        const bool cur_topk  = step_has_topk(i);
        const bool need_draw = (rnn_state_va != 0ull || !is_last || cur_topk);

        if (w.joint_proj) {
            if (!joint_in_packed) {
                agx_w4_dspark_pack_push ppc{};
                ppc.prev_tok_va  = L.dspark_prev;
                ppc.w1_va        = agx_tensor_gpu_va(w.markov_w1);
                ppc.hidden_va    = hidden_va;
                ppc.f_rnn_va     = f_rnn_va;
                ppc.state_va     = L.dspark_state;
                ppc.joint_in_va  = L.dspark_in;
                ppc.n_embd       = w.n_embd;
                ppc.rank         = w.rank;
                ppc.n_blocks     = n_blocks;
                ppc.block_drafts = block_drafts;
                ppc.draft_pos    = i;
                ppc.w1_wtype     = w1_wt;
                ppc.w1_row_bytes = w1_rb;
                ppc.flags        = (has_frnn ? AGX_W4_DSPARK_F_HAS_FRNN : 0u) |
                                   (i == i_beg ? AGX_W4_DSPARK_F_ZERO_STATE : 0u);
                if (!agx_cdm_dispatch(stream, eng->k_w4_dspark_pack, n_blocks, 1u, 1u,
                                      &ppc, sizeof(ppc), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
            }
            joint_in_packed = false;

            if (use_fused_joint_gru && eng->k_w7_dspark_joint_gru) {
                // 1 SINGLE 40-core dispatch reading joint_in ONCE across 3 rows + joint_b + GRU!
                agx_w7_dspark_joint_gru_push jpc{};
                jpc.joint_in_va   = L.dspark_in;
                jpc.joint_w_va    = agx_tensor_gpu_va(w.joint_proj);
                jpc.joint_b_va    = w.joint_proj_b ? agx_tensor_gpu_va(w.joint_proj_b) : 0ull;
                jpc.state_va      = L.dspark_state;
                jpc.markov_out_va = L.dspark_markov;
                jpc.n_joint       = n_joint;
                jpc.rank          = w.rank;
                jpc.n_blocks      = n_blocks;
                jpc.w_wtype       = jp_wt;
                jpc.w_row_bytes   = jp_rb;
                jpc.zero_state    = (i == i_beg) ? 1u : 0u;
                if (!agx_cdm_dispatch(stream, eng->k_w7_dspark_joint_gru, w.rank, n_blocks, 1u,
                                      &jpc, sizeof(jpc), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
            } else {
                if (!dispatch_dense_mm(eng->ctx, stream, w.joint_proj, L.dspark_in, L.dspark_joint_out,
                                       n_blocks, AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
                agx_w4_dspark_gru_push gpc{};
                gpc.joint_out_va  = L.dspark_joint_out;
                gpc.joint_b_va    = w.joint_proj_b ? agx_tensor_gpu_va(w.joint_proj_b) : 0ull;
                gpc.state_va      = L.dspark_state;
                gpc.markov_out_va = L.dspark_markov;
                gpc.rank          = w.rank;
                gpc.n_blocks      = n_blocks;
                gpc.zero_state    = (i == i_beg) ? 1u : 0u;
                if (!agx_cdm_dispatch(stream, eng->k_w4_dspark_gru, n_blocks, 1u, 1u,
                                      &gpc, sizeof(gpc), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
            }
        }

        const char * leg_mk_env = std::getenv("GGML_AGX_DSPARK_LEGACY_MK");
        const bool   legacy_mk  = (leg_mk_env && leg_mk_env[0] && leg_mk_env[0] != '0');
        bool use_w9_q4_mk = false;

        agx_w4_dspark_w2_push wpc{};
        wpc.markov_va       = L.dspark_markov;
        wpc.w2_va           = agx_tensor_gpu_va(w.markov_w2);
        wpc.w2_wtype        = w2_wt;
        wpc.w2_row_bytes    = w2_rb;
        if (eng->dspark_bound.w2_q4_bo &&
            eng->dspark_bound.w2_src_va == wpc.w2_va &&
            eng->dspark_bound.w2_n_vocab == w.n_vocab &&
            w.rank == 256u) {
            wpc.w2_va        = eng->dspark_bound.w2_q4_bo->gpu_va;
            wpc.w2_wtype     = AGX_W8_WTYPE_Q4_32;
            wpc.w2_row_bytes = AGX_W8_Q4_32_R256_ROW_BYTES;
            use_w9_q4_mk     = !legacy_mk && (eng->k_w9_dspark_w2_q4 != nullptr) && (w.n_vocab <= 204800u);
        }
        const uint32_t step_splits = use_w9_q4_mk ? 160u : n_splits;
        wpc.w2_bias_va      = w.markov_w2_b ? agx_tensor_gpu_va(w.markov_w2_b) : 0ull;
        wpc.base_va         = base_logits_va;
        wpc.out_logits_va   = out_logits_va;
        wpc.cand_val_va     = L.dspark_cand_val;
        wpc.cand_idx_va     = L.dspark_cand_idx;
        wpc.rank            = w.rank;
        wpc.n_vocab         = w.n_vocab;
        wpc.n_blocks        = n_blocks;
        wpc.block_drafts    = block_drafts;
        wpc.draft_pos       = i;
        wpc.n_splits        = step_splits;
        wpc.flags           = (w.markov_w2_b ? AGX_W4_DSPARK_F_HAS_W2_BIAS : 0u) |
                              (need_draw ? AGX_W4_DSPARK_F_EMIT_TOP64 : 0u);
        wpc.w2_scale        = w2_scale;
        wpc.softcap         = w.softcap;
        wpc.inv_softcap     = (w.softcap > 0.0f) ? (1.0f / w.softcap) : 0.0f;
        wpc.base_stride_f32 = base_str;

        const agx_kernel * k_w2 = use_w9_q4_mk ? eng->k_w9_dspark_w2_q4 : eng->k_w4_dspark_w2;
        if (!agx_cdm_dispatch(stream, k_w2, step_splits, n_blocks, 1u,
                              &wpc, sizeof(wpc),
                              need_draw ? AGX_BARRIER_STRONG_USC : (is_last ? post_barrier : AGX_BARRIER_STRONG_USC))) {
            return false;
        }

        if (need_draw) {
            const bool fuse_next_pack = (!is_last && w.joint_proj != nullptr);
            agx_w4_dspark_merge_push mpc{};
            mpc.cand_val_va       = L.dspark_cand_val;
            mpc.cand_idx_va       = L.dspark_cand_idx;
            mpc.rnn_state_va      = rnn_state_va;
            mpc.prev_tok_va       = L.dspark_prev;
            mpc.out_pack_va       = out_h_nextn_va;
            mpc.next_w1_va        = fuse_next_pack ? agx_tensor_gpu_va(w.markov_w1) : 0ull;
            mpc.next_hidden_va    = fuse_next_pack ? hidden_va : 0ull;
            mpc.next_f_rnn_va     = fuse_next_pack ? f_rnn_va : 0ull;
            mpc.next_state_va     = fuse_next_pack ? L.dspark_state : 0ull;
            mpc.next_joint_in_va  = fuse_next_pack ? L.dspark_in : 0ull;
            mpc.out_topk_table_va = cur_topk ? topk_table_va : 0ull;
            mpc.n_embd            = w.n_embd;
            mpc.n_blocks          = n_blocks;
            mpc.block_drafts      = block_drafts;
            mpc.draft_pos         = i;
            mpc.n_splits          = step_splits;
            mpc.n_cand            = n_cand;
            mpc.has_rnn_state     = (rnn_state_va != 0ull) ? 1u : 0u;
            mpc.next_rank         = w.rank;
            mpc.next_w1_wtype     = w1_wt;
            mpc.next_w1_row_bytes = w1_rb;
            mpc.next_flags        = has_frnn ? AGX_W4_DSPARK_F_HAS_FRNN : 0u;
            if (!agx_cdm_dispatch(stream, eng->k_w4_dspark_merge, n_blocks, 1u, 1u,
                                  &mpc, sizeof(mpc),
                                  is_last ? post_barrier : AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            joint_in_packed = fuse_next_pack;
        }
    }
    return true;
}

bool agx_dspark_dispatch_markov_head_w4(agx_inkling2_engine *          eng,
                                        agx_cdm_stream *               stream,
                                        const agx_dspark_mtp_tensors & w,
                                        uint64_t                       anchor_toks_va,
                                        uint64_t                       base_logits_va,
                                        uint64_t                       hidden_va,
                                        uint64_t                       f_rnn_va,
                                        uint64_t                       rnn_state_va,
                                        uint64_t                       out_logits_va,
                                        uint64_t                       out_h_nextn_va,
                                        uint32_t                       n_blocks,
                                        uint32_t                       block_drafts,
                                        agx_barrier_mode               post_barrier) {
    return agx_dspark_dispatch_markov_head_w7(eng, stream, w, anchor_toks_va, base_logits_va,
                                              w.n_vocab, hidden_va, f_rnn_va, rnn_state_va,
                                              out_logits_va, out_h_nextn_va, n_blocks, block_drafts,
                                              post_barrier);
}

// ============================================================================
// Wave 4 Graph Auto-Discovery & 48-Pass Inkling-2-Mini Megakernel Execution
// ============================================================================

namespace {

struct w4_layer_inp_store {
    uint64_t dst_va     = 0;
    uint64_t ids_va     = 0;
    uint32_t dst_nb1_B  = 0;
    bool     ids_is_i64 = true;
};

struct w4_pass_runtime {
    uint32_t il            = 0;  // logical cache slot 0..47
    uint32_t ip            = 0;  // physical layer 0..23
    bool     is_swa        = true;
    bool     has_loop      = false;
    bool     has_k_rot     = false;
    bool     has_v_rot     = false;
    uint32_t k_rot_dim     = 0;     // 0, 64, or 128
    uint32_t v_rot_dim     = 0;     // 0, 64, or 128
    bool     has_banded_rel = false; // ggml_flash_attn_ext_add_banded_rel_bias (src[6]/src[7] + 2D sinks)
    bool     kv_is_q8      = false;
    bool     k_ids_is_i64  = true;
    bool     v_ids_is_i64  = true;
    bool     bmask_is_f16  = true;

    // External KV cache (llama.cpp's k_l[il] / v_l[il])
    uint64_t k_va          = 0;
    uint64_t v_va          = 0;
    uint32_t k_nb1         = 0;
    uint32_t k_nb2         = 0;
    uint32_t v_nb1         = 0;
    uint32_t v_nb2         = 0;
    uint32_t n_kv          = 0;
    uint64_t k_store_va    = 0;
    uint64_t v_store_va    = 0;
    uint32_t k_store_nb1   = 0;
    uint32_t v_store_nb1   = 0;
    uint64_t k_idxs_va     = 0;
    uint64_t v_idxs_va     = 0;
    uint32_t k_ids_nb0     = 0;
    uint32_t v_ids_nb0     = 0;

    // Mask & relative pos (legacy kq_b OR O(n_kv) banded rel_table + rel_pos_pack)
    uint64_t rel_pos_va      = 0;
    uint32_t rpos_nb1        = 0;
    uint64_t base_mask_va    = 0;
    uint32_t bmask_nb1       = 0;
    uint64_t graph_mask_va   = 0; // fallback VA of fa->src[3] when n_kv is huge
    uint32_t graph_mask_nb1  = 0;
    uint64_t rel_table_va    = 0; // fa->src[6] [rel_table_len, 24, M] F32
    uint64_t rel_pos_pack_va = 0; // fa->src[7] packed I32 [cell_pos | q_pos | tile64_minmax]
    uint32_t rel_table_len   = 0; // eff_rows <= rel_extent
    uint32_t rel_extent      = 0; // fa->op_params[6]

    // Recurrent shortconv state (llama.cpp's r_l[il])
    uint64_t r_l_va        = 0;
    uint32_t r_nb1_f32     = 0;
    uint64_t cell_va[4]    = {}; // [0]=k, [1]=v, [2]=attn_post, [3]=ffn_post
    uint32_t rollback_slots      = 1;
    uint32_t rollback_stride_f32 = 0;
    uint64_t rs_zero_va    = 0;
    uint32_t rs_zero_elems = 0;

    // Optional DSpark target feature export (l_out of this pass: host OUTPUT or GPU layer_inp_dev SET_ROWS)
    uint64_t export_l_out_va = 0;
    std::vector<w4_layer_inp_store> layer_inp_stores;
};

struct w4_discovered_graph {
    agx_inkling2_hparams       hp;
    agx_inkling2_model_tensors W;
    w4_pass_runtime            passes[48];
    bool                       rel_bias_padded[24] = {};

    uint32_t n_tokens          = 0; // M (1..8)
    uint32_t n_out             = 0; // result_output->ne[1]
    uint64_t inp_tokens_va     = 0;
    uint64_t inp_pos_va        = 0;
    uint64_t s_copy_va         = 0;
    uint64_t inp_out_ids_va    = 0;
    uint64_t lm_head_pad_va    = 0;
    uint64_t result_output_va  = 0;
    uint64_t result_norm_va    = 0; // non-zero if result_norm is marked OUTPUT
    uint64_t export_tok_norm   = 0; // non-zero if t_layer_inp[0] is marked OUTPUT
    uint64_t export_out_norm   = 0; // non-zero if t_layer_inp[48] is marked OUTPUT
    std::vector<w4_layer_inp_store> tok_norm_stores; // DSpark GPU layer_inp_dev SET_ROWS from tok_norm_out
    std::vector<w4_layer_inp_store> out_norm_stores; // DSpark GPU layer_inp_dev SET_ROWS from out_norm_out
    bool     output_prec_f32   = true;
};

static const struct ggml_tensor * w4_strip_views(const struct ggml_tensor * t) {
    while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
                 t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE)) {
        t = t->src[0];
    }
    return t;
}

static void w4_read_scale_params(const struct ggml_tensor * t, float * s, float * b) {
    std::memcpy(s, t->op_params, sizeof(float));
    std::memcpy(b, reinterpret_cast<const char *>(t->op_params) + 4, sizeof(float));
}

static bool w4_tensor_in_va_range(const struct ggml_tensor * t, uint64_t base, uint64_t size) {
    const uint64_t va = agx_tensor_gpu_va(t);
    return va != 0 && va >= base && va < base + size;
}

static bool w4_parse_blk_name(const char * name, int * out_blk, const char ** out_suffix) {
    if (!name || std::strncmp(name, "blk.", 4) != 0) return false;
    char * endp = nullptr;
    long idx = std::strtol(name + 4, &endp, 10);
    if (!endp || *endp != '.' || idx < 0 || idx >= (long) AGX_INKLING2_MAX_PHYS_LAYERS) return false;
    *out_blk = (int) idx;
    *out_suffix = endp + 1;
    return true;
}

static void w4_register_weight_tensor(const struct ggml_tensor * t, agx_inkling2_model_tensors & W) {
    if (!t || !t->name[0] || agx_tensor_gpu_va(t) == 0) return;
    const char * nm = t->name;
    if (std::strcmp(nm, "token_embd.weight") == 0)        { W.tok_embd = t; return; }
    if (std::strcmp(nm, "token_embd_norm.weight") == 0 ||
        std::strcmp(nm, "blk.0.token_embd_norm.weight") == 0) { W.tok_norm = t; return; }
    if (std::strcmp(nm, "output.weight") == 0)            { W.output = t; return; }
    if (std::strcmp(nm, "output_norm.weight") == 0)       { W.output_norm = t; return; }
    if (std::strcmp(nm, "output_res_query.weight") == 0)  { W.output_res_query = t; return; }

    int blk = -1;
    const char * sfx = nullptr;
    if (!w4_parse_blk_name(nm, &blk, &sfx)) return;

    auto & L = W.layers[blk];
    if      (std::strcmp(sfx, "attn_q.weight") == 0)          L.wq = t;
    else if (std::strcmp(sfx, "attn_k.weight") == 0)          L.wk = t;
    else if (std::strcmp(sfx, "attn_v.weight") == 0)          L.wv = t;
    else if (std::strcmp(sfx, "attn_output.weight") == 0)     L.wo = t;
    else if (std::strcmp(sfx, "attn_gate.weight") == 0)       L.wqkv_gate = t;
    else if (std::strcmp(sfx, "attn_q_clip.weight") == 0)     L.attn_q_clip = t;
    else if (std::strcmp(sfx, "attn_sink_proj.weight") == 0)  L.attn_sink_proj = t;
    else if (std::strcmp(sfx, "attn_rel.weight") == 0)        L.attn_rel = t;
    else if (std::strcmp(sfx, "attn_rel_bias.weight") == 0)   L.attn_rel_bias = t;
    else if (std::strcmp(sfx, "attn_rel_clip.weight") == 0)   L.attn_rel_clip = t;
    else if (std::strcmp(sfx, "attn_k_conv.weight") == 0)     L.attn_k_conv = t;
    else if (std::strcmp(sfx, "attn_v_conv.weight") == 0)     L.attn_v_conv = t;
    else if (std::strcmp(sfx, "attn_post_conv.weight") == 0)  L.attn_post_conv = t;
    else if (std::strcmp(sfx, "ffn_post_conv.weight") == 0)   L.ffn_post_conv = t;
    else if (std::strcmp(sfx, "attn_norm.weight") == 0)       L.attn_norm = t;
    else if (std::strcmp(sfx, "attn_q_inp_norm.weight") == 0) L.attn_q_inp_norm = t;
    else if (std::strcmp(sfx, "attn_k_inp_norm.weight") == 0) L.attn_k_inp_norm = t;
    else if (std::strcmp(sfx, "attn_v_inp_norm.weight") == 0) L.attn_v_inp_norm = t;
    else if (std::strcmp(sfx, "attn_k_conv_proj.weight") == 0) L.attn_k_conv_proj = t;
    else if (std::strcmp(sfx, "attn_v_conv_proj.weight") == 0) L.attn_v_conv_proj = t;
    else if (std::strcmp(sfx, "attn_res_query.weight") == 0)  L.attn_res_query = t;
    else if (std::strcmp(sfx, "ffn_res_query.weight") == 0)   L.ffn_res_query = t;
    else if (std::strcmp(sfx, "ffn_norm.weight") == 0)        L.ffn_norm = t;
    else if (std::strcmp(sfx, "ffn_gate_inp.weight") == 0)    L.ffn_gate_inp = t;
    else if (std::strcmp(sfx, "exp_probs_b.bias") == 0 ||
             std::strcmp(sfx, "ffn_exp_probs_b.bias") == 0)    L.ffn_exp_probs_b = t;
    else if (std::strcmp(sfx, "ffn_gate_up_exps.weight") == 0) L.ffn_gate_up_exps   = t;
    else if (std::strcmp(sfx, "ffn_down_exps.weight") == 0)    L.ffn_down_exps      = t;
    else if (std::strcmp(sfx, "ffn_gate_up_exps.scale") == 0)  L.ffn_gate_up_exps_s = t;
    else if (std::strcmp(sfx, "ffn_down_exps.scale") == 0)     L.ffn_down_exps_s    = t;
    else if (blk >= 6 && blk < 18) {
        auto & LP = W.loop_layers[blk - 6];
        if      (std::strcmp(sfx, "loop_msg_query.weight") == 0)        LP.msg_query = t;
        else if (std::strcmp(sfx, "loop_q_ctrl_down.weight") == 0)      LP.ctrl_down[0] = t;
        else if (std::strcmp(sfx, "loop_q_ctrl_gain.weight") == 0)      LP.ctrl_gain[0] = t;
        else if (std::strcmp(sfx, "loop_q_ctrl_forget.weight") == 0)    LP.ctrl_forget[0] = t;
        else if (std::strcmp(sfx, "loop_k_ctrl_down.weight") == 0)      LP.ctrl_down[1] = t;
        else if (std::strcmp(sfx, "loop_k_ctrl_gain.weight") == 0)      LP.ctrl_gain[1] = t;
        else if (std::strcmp(sfx, "loop_k_ctrl_forget.weight") == 0)    LP.ctrl_forget[1] = t;
        else if (std::strcmp(sfx, "loop_v_ctrl_down.weight") == 0)      LP.ctrl_down[2] = t;
        else if (std::strcmp(sfx, "loop_v_ctrl_gain.weight") == 0)      LP.ctrl_gain[2] = t;
        else if (std::strcmp(sfx, "loop_v_ctrl_forget.weight") == 0)    LP.ctrl_forget[2] = t;
        else if (std::strcmp(sfx, "loop_ffn_ctrl_down.weight") == 0)    LP.ctrl_down[3] = t;
        else if (std::strcmp(sfx, "loop_ffn_ctrl_gain.weight") == 0)    LP.ctrl_gain[3] = t;
        else if (std::strcmp(sfx, "loop_ffn_ctrl_forget.weight") == 0)  LP.ctrl_forget[3] = t;
    }
}

static bool discover_inkling2_cgraph(const struct ggml_cgraph * cgraph, w4_discovered_graph & out) {
    if (!cgraph || cgraph->n_nodes < 512) return false;

    const char * mk_dbg_env = std::getenv("GGML_AGX_MK_DEBUG");
    const bool   mk_dbg     = (mk_dbg_env && mk_dbg_env[0] && mk_dbg_env[0] != '0');
    static int   mk_dbg_fail_prints = 0;
#define W4_MK_FAIL(...) do { \
        if (mk_dbg && mk_dbg_fail_prints < 8) { \
            ++mk_dbg_fail_prints; \
            std::fprintf(stderr, "[ggml-agx MK DISCOVER FAIL] " __VA_ARGS__); \
            std::fprintf(stderr, "\n"); \
        } \
        return false; \
    } while (0)

    agx_inkling2_fill_default_hparams(&out.hp);

    // 1. Scan leafs & nodes to harvest weights, inputs, 48 FLASH_ATTN_EXT, 48 RS GET_ROWS, and outputs
    for (int i = 0; i < cgraph->n_leafs; ++i) {
        w4_register_weight_tensor(cgraph->leafs[i], out.W);
    }

    const struct ggml_tensor * fa_nodes[48]      = {};
    int                        fa_node_idx[48]   = {};
    int                        n_fa              = 0;
    const struct ggml_tensor * rs_get_rows[48]   = {};
    int                        n_rs_gr           = 0;
    const struct ggml_tensor * l_out_nodes[48]   = {};
    const struct ggml_tensor * inp_tokens        = nullptr;
    const struct ggml_tensor * inp_pos           = nullptr;
    const struct ggml_tensor * rs_s_copy         = nullptr;
    const struct ggml_tensor * inp_out_ids       = nullptr;
    const struct ggml_tensor * tok_norm_out      = nullptr;
    const struct ggml_tensor * out_norm_out      = nullptr;
    const struct ggml_tensor * result_norm_node  = nullptr;
    const struct ggml_tensor * lm_head_node      = nullptr;
    const struct ggml_tensor * result_output     = nullptr;

    const struct ggml_tensor * rtr_div_node = nullptr;
    bool found_situ_params     = false;
    bool found_tau_params      = false;
    bool found_rtr_params      = false;
    bool found_qk_clip         = false;
    bool found_rel_clip        = false;
    bool cgraph_uses_exp_bias  = false;
    bool cgraph_uses_exp_scale = false;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * n = cgraph->nodes[i];
        if (!n) continue;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s]) {
                w4_register_weight_tensor(n->src[s], out.W);
                if (n->src[s]->name[0]) {
                    if (std::strstr(n->src[s]->name, "exp_probs_b") != nullptr) cgraph_uses_exp_bias = true;
                    if (std::strstr(n->src[s]->name, "_exps.scale") != nullptr) cgraph_uses_exp_scale = true;
                }
            }
        }
        if (n->op == GGML_OP_ADD && n->ne[0] == 128 && n->src[0] &&
            n->src[0]->op == GGML_OP_UNARY && ggml_get_unary_op(n->src[0]) == GGML_UNARY_OP_SIGMOID) {
            cgraph_uses_exp_bias = true;
        }

        if (n->op == GGML_OP_FLASH_ATTN_EXT) {
            if (n_fa >= 48) return false; // not 48-pass Inkling-2-Mini
            fa_nodes[n_fa]    = n;
            fa_node_idx[n_fa] = i;
            ++n_fa;
        } else if (n->op == GGML_OP_GET_ROWS && n->src[0] && n->src[1]) {
            if (std::strcmp(n->src[0]->name, "token_embd.weight") == 0) {
                inp_tokens = n->src[1];
            } else if (std::strncmp(n->src[1]->name, "rs_s_copy", 9) == 0) {
                rs_s_copy = n->src[1];
                if (n_rs_gr < 48) rs_get_rows[n_rs_gr++] = n;
            } else if (std::strncmp(n->src[1]->name, "out_ids", 7) == 0 ||
                       std::strncmp(n->src[1]->name, "inp_out_ids", 11) == 0) {
                inp_out_ids = n->src[1];
            }
        } else if (n->op == GGML_OP_MUL && n->src[0] && n->src[1]) {
            if (std::strcmp(n->src[1]->name, "token_embd_norm.weight") == 0 ||
                std::strcmp(n->src[1]->name, "blk.0.token_embd_norm.weight") == 0) {
                tok_norm_out = n;
                if (n->src[0]->op == GGML_OP_RMS_NORM) {
                    std::memcpy(&out.hp.f_norm_rms_eps, n->src[0]->op_params, sizeof(float));
                }
            } else if (std::strcmp(n->src[1]->name, "output_norm.weight") == 0) {
                out_norm_out = n;
            }
        } else if (n->op == GGML_OP_MUL_MAT && n->src[0] && std::strcmp(n->src[0]->name, "output.weight") == 0) {
            lm_head_node = n;
        }

        if (n->name[0]) {
            if (std::strcmp(n->name, "result_output") == 0) {
                result_output = n;
            } else if (std::strcmp(n->name, "result_norm") == 0) {
                result_norm_node = n;
                if (!inp_out_ids && n->op == GGML_OP_GET_ROWS && n->src[1]) {
                    inp_out_ids = n->src[1];
                }
            } else if (std::strncmp(n->name, "l_out-", 6) == 0) {
                int il = std::atoi(n->name + 6);
                // Prefer the primary residual ADD node so backend_sched split-copies never clobber l_out_nodes[il]
                if (il >= 0 && il < 48 && (!l_out_nodes[il] || n->op == GGML_OP_ADD)) {
                    l_out_nodes[il] = n;
                }
            }
        }

        // Structural (+ name-independent) SiTU extraction:
        //   situ(gate, up) = MUL(MUL(SCALE(TANH(SCALE(gate, 1/beta)), beta), SIGMOID(gate)), [SCALE(TANH(SCALE(up, 1/lb)), lb)])
        if (!found_situ_params && n->op == GGML_OP_MUL && n->src[0] && n->src[1]) {
            const struct ggml_tensor * act_mul = n->src[0];
            const struct ggml_tensor * up_node = n->src[1];
            if (act_mul->op == GGML_OP_MUL && act_mul->src[0] && act_mul->src[1] &&
                act_mul->src[0]->op == GGML_OP_SCALE &&
                act_mul->src[0]->src[0] && act_mul->src[0]->src[0]->op == GGML_OP_UNARY &&
                ggml_get_unary_op(act_mul->src[0]->src[0]) == GGML_UNARY_OP_TANH &&
                act_mul->src[1]->op == GGML_OP_UNARY &&
                ggml_get_unary_op(act_mul->src[1]) == GGML_UNARY_OP_SIGMOID) {
                float sb = 1.0f, bb = 0.0f;
                w4_read_scale_params(act_mul->src[0], &sb, &bb);
                out.hp.situ_beta = sb;
                if (up_node->op == GGML_OP_SCALE && up_node->src[0] && up_node->src[0]->op == GGML_OP_UNARY &&
                    ggml_get_unary_op(up_node->src[0]) == GGML_UNARY_OP_TANH) {
                    float slb = 1.0f, blb = 0.0f;
                    w4_read_scale_params(up_node, &slb, &blb);
                    out.hp.situ_linear_beta = slb;
                } else {
                    out.hp.situ_linear_beta = 0.0f;
                }
                found_situ_params = true;
            }
        }

        // Direct tau subgraph extraction from outer SCALE_BIAS(LOG(CLAMP(SCALE(SCALE(CPY(inp_pos))))))
        if (!found_tau_params && n->op == GGML_OP_SCALE && n->src[0] && n->src[0]->op == GGML_OP_LOG) {
            const struct ggml_tensor * lg = n->src[0];
            if (lg->src[0] && lg->src[0]->op == GGML_OP_CLAMP &&
                lg->src[0]->src[0] && lg->src[0]->src[0]->op == GGML_OP_SCALE &&
                lg->src[0]->src[0]->src[0] && lg->src[0]->src[0]->src[0]->op == GGML_OP_SCALE &&
                lg->src[0]->src[0]->src[0]->src[0] && lg->src[0]->src[0]->src[0]->src[0]->op == GGML_OP_CPY) {
                const struct ggml_tensor * pos_t = lg->src[0]->src[0]->src[0]->src[0]->src[0];
                if (pos_t && pos_t->type == GGML_TYPE_I32) {
                    inp_pos = pos_t;
                    float inv_floor = 0.0f, b0 = 0.0f;
                    w4_read_scale_params(lg->src[0]->src[0], &inv_floor, &b0);
                    if (inv_floor > 0.0f) {
                        out.hp.n_log_scale_floor = (uint32_t) std::llround(1.0 / (double) inv_floor);
                    }
                    float alpha = 0.0f, one = 0.0f;
                    w4_read_scale_params(n, &alpha, &one);
                    out.hp.f_log_scale_alpha = alpha;
                    found_tau_params = true;
                }
            }
        }

        // Capture Q/R clip thresholds from first Qcur / Rcur L2_NORM
        if (!found_qk_clip && n->op == GGML_OP_L2_NORM &&
            ((n->ne[0] == 128 && n->ne[1] == 24) ||
             (n->src[0] && n->src[0]->op == GGML_OP_MUL && n->src[0]->src[1] &&
              std::strstr(n->src[0]->src[1]->name, "attn_q_clip") != nullptr))) {
            std::memcpy(&out.hp.f_qk_clip, n->op_params, sizeof(float));
            found_qk_clip = true;
        }
        if (!found_rel_clip && n->op == GGML_OP_L2_NORM &&
            ((n->ne[0] <= 64 && n->ne[1] == 24) ||
             (n->src[0] && n->src[0]->op == GGML_OP_MUL && n->src[0]->src[1] &&
              std::strstr(n->src[0]->src[1]->name, "attn_rel_clip") != nullptr))) {
            std::memcpy(&out.hp.f_relative_clip, n->op_params, sizeof(float));
            found_rel_clip = true;
        }

        // Capture router norm_eps (from DIV) & weights_scale (from downstream SCALE through ggml_reshape_3d!)
        if (!found_rtr_params && n->op == GGML_OP_DIV && n->src[1] && n->src[1]->op == GGML_OP_SCALE &&
            n->src[1]->src[0] && n->src[1]->src[0]->op == GGML_OP_SUM_ROWS && n->ne[0] == 8) {
            float s1 = 1.0f, b1 = 0.0f;
            w4_read_scale_params(n->src[1], &s1, &b1);
            out.hp.expert_weights_norm_eps = b1;
            out.hp.expert_weights_scale    = 1.0f;
            rtr_div_node                   = n;
            found_rtr_params               = true;
        } else if (rtr_div_node && n->op == GGML_OP_SCALE && n->src[0] &&
                   w4_strip_views(n->src[0]) == rtr_div_node) {
            float ws = 1.0f, wb = 0.0f;
            w4_read_scale_params(n, &ws, &wb);
            if (wb == 0.0f && ws > 0.0f) {
                out.hp.expert_weights_scale = ws;
            }
            rtr_div_node = nullptr; // captured first layer's exact expert_weights_scale
        }
    }

    // Must be 48-pass Inkling-2-Mini decode with 1 <= M <= 8 and all hparams verified
    if (n_fa != 48) {
        return false; // not a 48-pass Inkling-2-Mini graph (e.g., DSpark, prefill split, other model)
    }
    if (n_rs_gr != 48 || !inp_tokens || !inp_pos || !rs_s_copy ||
        !tok_norm_out || !out_norm_out || !lm_head_node || !result_output ||
        !found_situ_params || !found_tau_params || !found_rtr_params || !found_qk_clip || !found_rel_clip) {
        W4_MK_FAIL("basic: n_fa=%d n_rs=%d tok=%d pos=%d scopy=%d tn=%d on=%d lm=%d ro=%d "
                       "situ=%d tau=%d rtr=%d qk=%d rel=%d",
                       n_fa, n_rs_gr, !!inp_tokens, !!inp_pos, !!rs_s_copy, !!tok_norm_out,
                       !!out_norm_out, !!lm_head_node, !!result_output,
                       (int) found_situ_params, (int) found_tau_params, (int) found_rtr_params,
                       (int) found_qk_clip, (int) found_rel_clip);
    }
    if (inp_tokens->type != GGML_TYPE_I32 || inp_tokens->ne[1] != 1 ||
        inp_tokens->ne[0] < 1 || inp_tokens->ne[0] > (int64_t) AGX_SLC_ARENA_MAX_M) {
        W4_MK_FAIL("M=%lld ne1=%lld type=%d (max_M=%u)",
                       (long long) inp_tokens->ne[0], (long long) inp_tokens->ne[1],
                       (int) inp_tokens->type, (unsigned) AGX_SLC_ARENA_MAX_M);
    }
    if (rs_s_copy->ne[0] != 1) {
        // Multi-sequence batch: fall back cleanly to Wave 3 generic graph
        W4_MK_FAIL("rs_s_copy->ne[0]=%lld != 1", (long long) rs_s_copy->ne[0]);
    }

    // Harvest attn_rel_bias from fa_nodes[v]->src[6] (banded) or src[3] (legacy) BEFORE weight check:
    // Because BF16 GGML_OP_PAD may run on CPU in split 0, the GPU split often sees the
    // already-padded [n_rel, n_extent+1] GPU split-copy as table->src[0] without the original weight name!
    for (uint32_t v = 0; v < 48; ++v) {
        uint32_t ip = (v < 18u) ? v : ((v < 30u) ? (6u + (v - 18u)) : ((v < 42u) ? (6u + (v - 30u)) : (18u + (v - 42u))));
        if (out.W.layers[ip].attn_rel_bias) continue;
        if (ggml_flash_attn_ext_has_banded_rel_bias(fa_nodes[v])) {
            const struct ggml_tensor * tbl = fa_nodes[v]->src[6];
            if (tbl) {
                if (tbl->op == GGML_OP_CONT) tbl = tbl->src[0];
                tbl = w4_strip_views(tbl);
                if (tbl && tbl->op == GGML_OP_MUL_MAT && tbl->src[0]) {
                    const struct ggml_tensor * rb = w4_strip_views(tbl->src[0]);
                    if (rb && agx_tensor_gpu_va(rb) != 0) {
                        out.W.layers[ip].attn_rel_bias = rb;
                        out.rel_bias_padded[ip]        = false;
                    }
                }
            }
            continue;
        }
        const struct ggml_tensor * m_in  = fa_nodes[v]->src[3];
        const struct ggml_tensor * add_m = (m_in && m_in->op == GGML_OP_CPY) ? m_in->src[0] : m_in;
        if (!add_m || add_m->op != GGML_OP_ADD || !add_m->src[0]) continue;
        const struct ggml_tensor * gr = add_m->src[0];
        if (gr->op == GGML_OP_CONT) gr = gr->src[0];
        gr = w4_strip_views(gr);
        if (!gr || gr->op != GGML_OP_GET_ROWS || !gr->src[0]) continue;
        const struct ggml_tensor * tbl = gr->src[0];
        if (tbl->op == GGML_OP_CONT) tbl = tbl->src[0];
        tbl = w4_strip_views(tbl);
        if (tbl && tbl->op == GGML_OP_SUB) tbl = tbl->src[0];
        if (tbl && tbl->op == GGML_OP_MUL_MAT && tbl->src[0]) {
            const struct ggml_tensor * rb = tbl->src[0];
            if (rb->op == GGML_OP_PAD && rb->src[0] && agx_tensor_gpu_va(rb->src[0]) != 0) {
                out.W.layers[ip].attn_rel_bias = rb->src[0];
                out.rel_bias_padded[ip]        = false;
            } else if (agx_tensor_gpu_va(rb) != 0) {
                out.W.layers[ip].attn_rel_bias = rb;
                out.rel_bias_padded[ip]        = true;
            }
        }
    }

    // Verify all 24 physical layers + 12 loop layers + core weights are present with supported dtypes
    auto is_wtype_supported = [](const struct ggml_tensor * t) -> bool {
        if (!t) return false;
        return t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 ||
               t->type == GGML_TYPE_BF16 || t->type == GGML_TYPE_Q8_0;
    };
    const auto & W = out.W;
    if (!W.tok_embd || !W.tok_norm || !W.output || !W.output_norm || !W.output_res_query) {
        W4_MK_FAIL("core weights missing: emb=%d tn=%d out=%d on=%d orq=%d",
                       !!W.tok_embd, !!W.tok_norm, !!W.output, !!W.output_norm, !!W.output_res_query);
    }
    if (W.tok_embd->ne[0] != 2304 || W.output->ne[0] != 2304) {
        W4_MK_FAIL("n_embd mismatch: emb=%lld out=%lld",
                       (long long) W.tok_embd->ne[0], (long long) W.output->ne[0]);
    }
    if (!is_wtype_supported(W.tok_embd) || W.tok_norm->type != GGML_TYPE_F32 ||
        W.output_norm->type != GGML_TYPE_F32 || W.output_res_query->type != GGML_TYPE_F32) {
        W4_MK_FAIL("core weight dtype unsupported");
    }

    out.n_tokens         = (uint32_t) inp_tokens->ne[0];
    out.n_out            = (uint32_t) result_output->ne[1];
    out.inp_tokens_va    = agx_tensor_gpu_va(inp_tokens);
    out.inp_pos_va       = agx_tensor_gpu_va(inp_pos);
    out.s_copy_va        = agx_tensor_gpu_va(rs_s_copy);
    out.inp_out_ids_va   = inp_out_ids ? agx_tensor_gpu_va(inp_out_ids) : 0ull;
    out.lm_head_pad_va   = agx_tensor_gpu_va(lm_head_node);
    out.result_output_va = agx_tensor_gpu_va(result_output);
    out.hp.n_vocab_padded   = (uint32_t) W.output->ne[1];
    out.hp.n_vocab_unpadded = (uint32_t) result_output->ne[0];
    if (!out.inp_tokens_va || !out.inp_pos_va || !out.s_copy_va || !out.lm_head_pad_va || !out.result_output_va) {
        W4_MK_FAIL("missing core GPU VA: tok=%llu pos=%llu sc=%llu lm=%llu ro=%llu",
                       (unsigned long long) out.inp_tokens_va, (unsigned long long) out.inp_pos_va,
                       (unsigned long long) out.s_copy_va, (unsigned long long) out.lm_head_pad_va,
                       (unsigned long long) out.result_output_va);
    }

    // Check final logit softcapping between lm_head_node and result_output
    out.hp.f_final_logit_softcapping = 0.0f;
    {
        const struct ggml_tensor * ro_src = result_output->src[0];
        if (ro_src && ro_src->op == GGML_OP_VIEW) ro_src = ro_src->src[0];
        if (ro_src && ro_src->op == GGML_OP_SCALE && ro_src->src[0] && ro_src->src[0]->op == GGML_OP_UNARY &&
            ggml_get_unary_op(ro_src->src[0]) == GGML_UNARY_OP_TANH) {
            float cap = 0.0f, b = 0.0f;
            w4_read_scale_params(ro_src, &cap, &b);
            if (cap > 0.0f && b == 0.0f) out.hp.f_final_logit_softcapping = cap;
        }
    }

    for (uint32_t ip = 0; ip < 24; ++ip) {
        const auto & L = W.layers[ip];
        const bool is_swa = (ip % 6u != 5u);
        if (!L.wq || !L.wk || !L.wv || !L.wo || !L.wqkv_gate ||
            !L.attn_q_clip || !L.attn_sink_proj || !L.attn_rel || !L.attn_rel_bias || !L.attn_rel_clip ||
            !L.attn_k_conv || !L.attn_v_conv || !L.attn_post_conv || !L.ffn_post_conv ||
            !L.ffn_res_query || !L.ffn_norm || !L.ffn_gate_inp || !L.ffn_gate_up_exps || !L.ffn_down_exps) {
            W4_MK_FAIL("layer %u missing weight (rb=%d wq=%d exp=%d)",
                           ip, !!L.attn_rel_bias, !!L.wq, !!L.ffn_gate_up_exps);
        }
        if (ip > 0 && !L.attn_res_query) W4_MK_FAIL("layer %u missing attn_res_query", ip);
        if (is_swa && !L.attn_norm) W4_MK_FAIL("layer %u missing attn_norm", ip);
        if (!is_swa && (!L.attn_q_inp_norm || !L.attn_k_inp_norm || !L.attn_v_inp_norm ||
                        !L.attn_k_conv_proj || !L.attn_v_conv_proj)) {
            W4_MK_FAIL("layer %u missing full-attn norm/conv_proj", ip);
        }
        if (L.attn_k_conv->ne[0] != 4 || L.attn_v_conv->ne[0] != 4 ||
            L.attn_post_conv->ne[0] != 4 || L.ffn_post_conv->ne[0] != 4) {
            W4_MK_FAIL("layer %u conv kernel != 4", ip);
        }
        if (!is_wtype_supported(L.attn_rel_bias) ||
            L.attn_k_conv->type != GGML_TYPE_F32 || L.attn_v_conv->type != GGML_TYPE_F32 ||
            L.attn_post_conv->type != GGML_TYPE_F32 || L.ffn_post_conv->type != GGML_TYPE_F32 ||
            L.attn_q_clip->type != GGML_TYPE_F32 || L.attn_rel_clip->type != GGML_TYPE_F32 ||
            L.ffn_res_query->type != GGML_TYPE_F32 || L.ffn_norm->type != GGML_TYPE_F32 ||
            (L.attn_res_query && L.attn_res_query->type != GGML_TYPE_F32) ||
            (L.attn_norm && L.attn_norm->type != GGML_TYPE_F32) ||
            (L.attn_q_inp_norm && L.attn_q_inp_norm->type != GGML_TYPE_F32) ||
            (L.attn_k_inp_norm && L.attn_k_inp_norm->type != GGML_TYPE_F32) ||
            (L.attn_v_inp_norm && L.attn_v_inp_norm->type != GGML_TYPE_F32)) {
            W4_MK_FAIL("layer %u unsupported dtype (rb=%d)", ip, (int) L.attn_rel_bias->type);
        }
        const uint32_t eff_ext = (uint32_t) (L.attn_rel_bias->ne[1] - (out.rel_bias_padded[ip] ? 1 : 0));
        if (ip == 0) {
            out.hp.n_relative = (uint32_t) L.attn_rel_bias->ne[0];
            out.hp.n_swa      = eff_ext;
            out.hp.n_expert   = (uint32_t) L.ffn_gate_inp->ne[1];
            out.hp.n_ff_exp   = (uint32_t) L.ffn_down_exps->ne[0];
        } else if (ip == 5) {
            out.hp.n_relative_extent = eff_ext;
        }
        // Wave 5 Fail-Loud Guardrails (#120 & #130 regression-proof):
        // Hard-fail loudly if router bias (exp_probs_b) or NVFP4 2nd-level scale (scale2) is missing!
        if (cgraph_uses_exp_bias && !L.ffn_exp_probs_b) {
            AGX_LOG_ERROR("FAIL-LOUD: discover_inkling2_cgraph layer %u missing required exp_probs_b.bias tensor!", ip);
            return false;
        }
        if ((L.ffn_gate_up_exps->type == GGML_TYPE_NVFP4 || cgraph_uses_exp_scale) &&
            L.ffn_gate_up_exps->type == GGML_TYPE_NVFP4 && !L.ffn_gate_up_exps_s) {
            AGX_LOG_ERROR("FAIL-LOUD: discover_inkling2_cgraph layer %u missing required NVFP4 ffn_gate_up_exps.scale!", ip);
            return false;
        }
        if ((L.ffn_down_exps->type == GGML_TYPE_NVFP4 || cgraph_uses_exp_scale) &&
            L.ffn_down_exps->type == GGML_TYPE_NVFP4 && !L.ffn_down_exps_s) {
            AGX_LOG_ERROR("FAIL-LOUD: discover_inkling2_cgraph layer %u missing required NVFP4 ffn_down_exps.scale!", ip);
            return false;
        }
    }
    // Wave 5 Fail-Loud AGX0_Repack Audit: verify repackable Inkling-2 weights are active when GGML_AGX_NO_REPACK is unset
    if (std::getenv("GGML_AGX_NO_REPACK") == nullptr) {
        uint32_t n_eligible = 0u, n_repacked = 0u;
        auto check_rep = [&](const struct ggml_tensor * t) {
            if (t && agx_repack_supports_tensor(t)) {
                ++n_eligible;
                if (agx_tensor_is_repacked(t)) ++n_repacked;
            }
        };
        check_rep(W.output);
        for (uint32_t ip = 0; ip < 24u; ++ip) {
            const auto & L = W.layers[ip];
            check_rep(L.wq); check_rep(L.wk); check_rep(L.wv); check_rep(L.wo);
            check_rep(L.wqkv_gate); check_rep(L.attn_rel);
            check_rep(L.ffn_gate_up_exps); check_rep(L.ffn_down_exps);
        }
        static bool logged_mk_repack = false;
        if (!logged_mk_repack) {
            logged_mk_repack = true;
            AGX_LOG_INFO("Inkling-2 AGX0_Repack audit: %u/%u repack-eligible weight tensors active",
                         n_repacked, n_eligible);
        }
        if (n_eligible > 0u && n_repacked == 0u) {
            AGX_LOG_ERROR("FAIL-LOUD: 0/%u Inkling-2 weight tensors repacked in AGX0_Repack (GGML_AGX_NO_REPACK unset)!",
                          n_eligible);
            return false;
        }
    }
    for (uint32_t l = 0; l < 12; ++l) {
        const auto & LP = W.loop_layers[l];
        if (!LP.msg_query) W4_MK_FAIL("loop %u missing msg_query", l);
        for (int c = 0; c < 4; ++c) {
            if (!LP.ctrl_down[c] || !LP.ctrl_gain[c] || !LP.ctrl_forget[c]) {
                W4_MK_FAIL("loop %u ctrl %d missing", l, c);
            }
        }
        if (l == 0) out.hp.gate_rank = (uint32_t) LP.ctrl_down[0]->ne[1];
    }

    // 2. Discover per-pass (visit 0..47) KV cache, mask, rel_pos, and recurrent conv state VAs
    for (uint32_t v = 0; v < 48; ++v) {
        uint32_t il = 0, ip = 0;
        bool has_loop = false;
        if (v < 6u) {
            il = v; ip = v;
        } else if (v < 18u) {
            il = v; ip = v;
        } else if (v < 30u) {
            il = 24u + (v - 18u); ip = 6u + (v - 18u); has_loop = true;
        } else if (v < 42u) {
            il = 36u + (v - 30u); ip = 6u + (v - 30u); has_loop = true;
        } else {
            il = 18u + (v - 42u); ip = 18u + (v - 42u);
        }

        auto & P = out.passes[v];
        P.il       = il;
        P.ip       = ip;
        P.is_swa   = (ip % 6u != 5u);
        P.has_loop = has_loop;

        const struct ggml_tensor * fa   = fa_nodes[v];
        const struct ggml_tensor * q_in = fa->src[0];
        const struct ggml_tensor * k_l  = fa->src[1];
        const struct ggml_tensor * v_l  = fa->src[2];
        const struct ggml_tensor * m_in = fa->src[3];
        if (!q_in || !k_l || !v_l || !m_in) W4_MK_FAIL("v=%u missing fa src", v);
        if (q_in->ne[0] != 128 || q_in->ne[1] != (int64_t) out.n_tokens ||
            q_in->ne[2] != 24 || q_in->ne[3] != 1 || k_l->ne[2] != 8 || v_l->ne[2] != 8) {
            W4_MK_FAIL("v=%u fa dims q=[%lld,%lld,%lld] k2=%lld v2=%lld", v,
                           (long long) q_in->ne[0], (long long) q_in->ne[1], (long long) q_in->ne[2],
                           (long long) k_l->ne[2], (long long) v_l->ne[2]);
        }
        if ((k_l->type != GGML_TYPE_F16 && k_l->type != GGML_TYPE_Q8_0) || k_l->type != v_l->type) {
            W4_MK_FAIL("v=%u kv type k=%d v=%d", v, (int) k_l->type, (int) v_l->type);
        }
        P.kv_is_q8 = (k_l->type == GGML_TYPE_Q8_0);
        P.k_va     = agx_tensor_gpu_va(k_l);
        P.v_va     = agx_tensor_gpu_va(v_l);
        P.k_nb1    = (uint32_t) k_l->nb[1];
        P.k_nb2    = (uint32_t) k_l->nb[2];
        P.v_nb1    = (uint32_t) v_l->nb[1];
        P.v_nb2    = (uint32_t) v_l->nb[2];
        P.n_kv     = (uint32_t) k_l->ne[1];
        P.graph_mask_va  = agx_tensor_gpu_va(m_in);
        P.graph_mask_nb1 = (uint32_t) m_in->nb[1];
        if (!P.k_va || !P.v_va || !P.graph_mask_va) W4_MK_FAIL("v=%u missing kv/mask VA", v);

        // Detect k_rot Hadamard on Q/K (exact block size: 64 or 128!)
        const struct ggml_tensor * q_root = w4_strip_views(q_in);
        if (q_root && q_root->op == GGML_OP_MUL_MAT &&
            ggml_get_op_params_i32(q_root, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
            const uint32_t rd = (q_root->src[0]) ? (uint32_t) q_root->src[0]->ne[0] : 128u;
            if (rd != 64u && rd != 128u) W4_MK_FAIL("v=%u unsupported k_rot_dim=%u", v, rd);
            P.has_k_rot = true;
            P.k_rot_dim = rd;
        }
        // Detect v_rot Hadamard on kqv_out right after fa_nodes[v] (exact block size: 64 or 128!)
        const int next_idx = fa_node_idx[v] + 1;
        for (int j = next_idx; j < std::min(cgraph->n_nodes, next_idx + 8); ++j) {
            const struct ggml_tensor * cand = cgraph->nodes[j];
            if (cand && cand->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(cand, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                const struct ggml_tensor * s1 = w4_strip_views(cand->src[1]);
                if (s1 == fa || (s1 && s1->op == GGML_OP_CONT && w4_strip_views(s1->src[0]) == fa)) {
                    const uint32_t rd = (cand->src[0]) ? (uint32_t) cand->src[0]->ne[0] : 128u;
                    if (rd != 64u && rd != 128u) W4_MK_FAIL("v=%u unsupported v_rot_dim=%u", v, rd);
                    P.has_v_rot = true;
                    P.v_rot_dim = rd;
                    break;
                }
            }
        }

        if (ggml_flash_attn_ext_has_banded_rel_bias(fa)) {
            // Wave 10: O(n_kv) banded rel-bias FA (src[3]=1-head base_mask, src[6]=rel_table, src[7]=rel_pos_pack, 2D sinks)
            const struct ggml_tensor * raw_m = (m_in->op == GGML_OP_CPY) ? m_in->src[0] : m_in;
            const struct ggml_tensor * rtab  = fa->src[6];
            const struct ggml_tensor * rpos  = fa->src[7];
            if (!raw_m || (raw_m->type != GGML_TYPE_F16 && raw_m->type != GGML_TYPE_F32) ||
                raw_m->ne[2] != 1 || !rtab || rtab->type != GGML_TYPE_F32 ||
                !rpos || rpos->type != GGML_TYPE_I32) {
                W4_MK_FAIL("v=%u banded fa invalid raw_m/rtab/rpos", v);
            }
            if (out.W.layers[ip].attn_rel_bias->ne[0] > 32) {
                W4_MK_FAIL("v=%u banded n_rel=%lld > 32", v, (long long) out.W.layers[ip].attn_rel_bias->ne[0]);
            }
            P.has_banded_rel  = true;
            P.base_mask_va    = agx_tensor_gpu_va(raw_m);
            P.bmask_nb1       = (uint32_t) raw_m->nb[1];
            P.bmask_is_f16    = (raw_m->type == GGML_TYPE_F16);
            P.rel_table_va    = agx_tensor_gpu_va(rtab);
            P.rel_pos_pack_va = agx_tensor_gpu_va(rpos);
            P.rel_table_len   = (uint32_t) rtab->ne[0];
            P.rel_extent      = (uint32_t) fa->op_params[6];
            P.rel_pos_va      = P.rel_pos_pack_va; // for B=1 SWA HYBRID_MODE=0 banded pack mode!
            P.rpos_nb1        = 0u;
            if (!P.base_mask_va || !P.rel_table_va || !P.rel_pos_pack_va || P.rel_table_len == 0u) {
                W4_MK_FAIL("v=%u banded missing VA/len", v);
            }
        } else {
            // Unwrap kq_mask_rel_pos -> ADD(kq_b, raw_mask) -> rel_pos + base_mask
            const struct ggml_tensor * add_m = (m_in->op == GGML_OP_CPY) ? m_in->src[0] : m_in;
            if (!add_m || add_m->op != GGML_OP_ADD || !add_m->src[0] || !add_m->src[1]) {
                W4_MK_FAIL("v=%u m_in unwrap failed (op=%d)", v, add_m ? (int) add_m->op : -1);
            }
            const struct ggml_tensor * kq_b_t = add_m->src[0];
            const struct ggml_tensor * raw_m  = add_m->src[1];
            if (raw_m->op == GGML_OP_CPY) raw_m = raw_m->src[0];
            if (!raw_m || (raw_m->type != GGML_TYPE_F16 && raw_m->type != GGML_TYPE_F32)) {
                W4_MK_FAIL("v=%u raw_m bad type=%d", v, raw_m ? (int) raw_m->type : -1);
            }
            P.base_mask_va = agx_tensor_gpu_va(raw_m);
            P.bmask_nb1    = (uint32_t) raw_m->nb[1];
            P.bmask_is_f16 = (raw_m->type == GGML_TYPE_F16);

            // kq_b_t = CONT(PERMUTE(RESHAPE(GET_ROWS(table, rel_pos))))
            const struct ggml_tensor * gr = kq_b_t;
            if (gr->op == GGML_OP_CONT) gr = gr->src[0];
            gr = w4_strip_views(gr);
            if (!gr || gr->op != GGML_OP_GET_ROWS || !gr->src[1] || gr->src[1]->type != GGML_TYPE_I32) {
                W4_MK_FAIL("v=%u kq_b gr unwrap failed (op=%d)", v, gr ? (int) gr->op : -1);
            }
            P.rel_pos_va = agx_tensor_gpu_va(gr->src[1]);
            P.rpos_nb1   = (uint32_t) gr->src[1]->nb[1];
            if (!P.base_mask_va || !P.rel_pos_va) W4_MK_FAIL("v=%u missing bmask/rpos VA", v);
        }

        // Recurrent state base r_l[il] from rs_get_rows[v]
        const struct ggml_tensor * r_src = rs_get_rows[v]->src[0];
        const struct ggml_tensor * r_raw = (r_src && r_src->op == GGML_OP_RESHAPE) ? r_src->src[0] : r_src;
        if (!r_raw || r_raw->type != GGML_TYPE_F32) W4_MK_FAIL("v=%u r_raw invalid", v);
        P.r_l_va    = agx_tensor_gpu_va(r_raw);
        P.r_nb1_f32 = (uint32_t) (r_raw->nb[1] / sizeof(float));
        if (!P.r_l_va || P.r_nb1_f32 < 6656u * 3u) {
            W4_MK_FAIL("v=%u r_l_va=%llu r_nb1_f32=%u", v, (unsigned long long) P.r_l_va, P.r_nb1_f32);
        }

        const uint64_t r_size_B = (uint64_t) ggml_nbytes(r_raw);
        const uint64_t k_size_B = (uint64_t) ggml_nbytes(k_l);
        const uint64_t v_size_B = (uint64_t) ggml_nbytes(v_l);

        // Scan pass window [win_beg, win_end) to find k/v SET_ROWS, rs_zero FILL, and 4 conv state stores
        const int win_beg = (v == 0u) ? 0 : fa_node_idx[v - 1];
        const int win_end = (v + 1u < 48u) ? fa_node_idx[v + 1] : cgraph->n_nodes;

        int n_conv_stores = 0;
        for (int j = win_beg; j < win_end; ++j) {
            const struct ggml_tensor * nd = cgraph->nodes[j];
            if (!nd) continue;
            if (j < fa_node_idx[v] && nd->op == GGML_OP_SET_ROWS && nd->src[0] && nd->src[1]) {
                if (!P.k_store_va && w4_tensor_in_va_range(nd, P.k_va, k_size_B) && nd->ne[0] == 1024) {
                    P.k_store_va   = agx_tensor_gpu_va(nd);
                    P.k_store_nb1  = (uint32_t) nd->nb[1];
                    P.k_idxs_va    = agx_tensor_gpu_va(nd->src[1]);
                    P.k_ids_is_i64 = (nd->src[1]->type == GGML_TYPE_I64);
                    P.k_ids_nb0    = (uint32_t) nd->src[1]->nb[0];
                } else if (!P.v_store_va && w4_tensor_in_va_range(nd, P.v_va, v_size_B) && nd->ne[0] == 1024) {
                    P.v_store_va   = agx_tensor_gpu_va(nd);
                    P.v_store_nb1  = (uint32_t) nd->nb[1];
                    P.v_idxs_va    = agx_tensor_gpu_va(nd->src[1]);
                    P.v_ids_is_i64 = (nd->src[1]->type == GGML_TYPE_I64);
                    P.v_ids_nb0    = (uint32_t) nd->src[1]->nb[0];
                }
            }
            if (j < fa_node_idx[v] && nd->op == GGML_OP_FILL && ggml_nelements(nd) > 0 &&
                w4_tensor_in_va_range(nd, P.r_l_va, r_size_B)) {
                P.rs_zero_va    = agx_tensor_gpu_va(nd);
                P.rs_zero_elems = (uint32_t) ggml_nelements(nd);
            }
            if (n_conv_stores < 4 && (nd->op == GGML_OP_CPY || nd->op == GGML_OP_SET_ROWS) &&
                w4_tensor_in_va_range(nd, P.r_l_va, r_size_B)) {
                // Skip zero-size extra-state copy (`states_extra` when n_rs == 1)
                if (ggml_nelements(nd) == 0) continue;
                const uint64_t cva = agx_tensor_gpu_va(nd);
                if (nd->op == GGML_OP_CPY) {
                    P.cell_va[n_conv_stores++] = cva;
                    P.rollback_slots           = 1u;
                    P.rollback_stride_f32      = 0u;
                } else if (nd->op == GGML_OP_SET_ROWS && nd->src[0]) {
                    P.cell_va[n_conv_stores++] = cva;
                    P.rollback_slots           = (uint32_t) nd->src[0]->ne[1];
                    P.rollback_stride_f32      = (uint32_t) (nd->nb[1] / sizeof(float));
                }
            }
        }
        if (!P.k_store_va || !P.v_store_va || !P.k_idxs_va || !P.v_idxs_va || n_conv_stores != 4) {
            W4_MK_FAIL("v=%u stores ks=%d vs=%d ki=%d vi=%d conv=%d",
                           v, !!P.k_store_va, !!P.v_store_va, !!P.k_idxs_va, !!P.v_idxs_va, n_conv_stores);
        }
        // Sanity check the 4 channel offsets in r_l[il]: 0, 1024*12, 2048*12, 4352*12
        if (P.cell_va[1] - P.cell_va[0] != 1024ULL * 3ULL * 4ULL ||
            P.cell_va[2] - P.cell_va[0] != 2048ULL * 3ULL * 4ULL ||
            P.cell_va[3] - P.cell_va[0] != 4352ULL * 3ULL * 4ULL) {
            W4_MK_FAIL("v=%u cell_va deltas=(%lld,%lld,%lld)", v,
                           (long long) (P.cell_va[1] - P.cell_va[0]),
                           (long long) (P.cell_va[2] - P.cell_va[0]),
                           (long long) (P.cell_va[3] - P.cell_va[0]));
        }
    }

    // 3. Audit EVERY GGML_TENSOR_FLAG_OUTPUT node in cgraph (handles DSpark t_layer_inp[0..48] & rejects unknown outputs!)
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * n = cgraph->nodes[i];
        if (!n || (n->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) continue;
        if (n == result_output) continue;
        if (n == result_norm_node) {
            out.result_norm_va = agx_tensor_gpu_va(n);
            continue;
        }
        if (n == tok_norm_out) {
            out.export_tok_norm = agx_tensor_gpu_va(n);
            continue;
        }
        if (n == out_norm_out) {
            out.export_out_norm = agx_tensor_gpu_va(n);
            continue;
        }
        bool matched_lout = false;
        for (uint32_t v = 0; v < 48; ++v) {
            if (l_out_nodes[out.passes[v].il] == n) {
                out.passes[v].export_l_out_va = agx_tensor_gpu_va(n);
                matched_lout = true;
                break;
            }
        }
        if (!matched_lout) {
            // Unknown output tensor requested by caller -> fall back cleanly to Wave 3 generic graph
            W4_MK_FAIL("unknown OUTPUT tensor '%s' op=%d", n->name, (int) n->op);
        }
    }

    // 4. Discover DSpark GPU layer_inp_dev SET_ROWS side-effect writes (ne[0]==2304, F32, src0 in t_layer_inp[0..48])
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * nd = cgraph->nodes[i];
        if (!nd || nd->op != GGML_OP_SET_ROWS || nd->ne[0] != 2304 || !nd->src[0] || !nd->src[1]) continue;
        if (nd->type != GGML_TYPE_F32 ||
            (nd->src[1]->type != GGML_TYPE_I64 && nd->src[1]->type != GGML_TYPE_I32)) {
            W4_MK_FAIL("unsupported 2304-wide SET_ROWS type=%d idx_type=%d", (int) nd->type, (int) nd->src[1]->type);
        }
        w4_layer_inp_store st{};
        st.dst_va     = agx_tensor_gpu_va(nd);
        st.ids_va     = agx_tensor_gpu_va(nd->src[1]);
        st.dst_nb1_B  = (uint32_t) nd->nb[1];
        st.ids_is_i64 = (nd->src[1]->type == GGML_TYPE_I64);
        if (!st.dst_va || !st.ids_va) {
            W4_MK_FAIL("2304-wide SET_ROWS missing GPU VA");
        }
        const struct ggml_tensor * s0 = w4_strip_views(nd->src[0]);
        while (s0 && (s0->op == GGML_OP_CONT || s0->op == GGML_OP_CPY || s0->op == GGML_OP_DUP)) {
            s0 = w4_strip_views(s0->src[0]);
        }
        if (s0 == tok_norm_out) {
            out.tok_norm_stores.push_back(st);
            continue;
        }
        if (s0 == out_norm_out) {
            out.out_norm_stores.push_back(st);
            continue;
        }
        bool matched = false;
        const int s0_lout_il = (s0 && s0->name[0] && std::strncmp(s0->name, "l_out-", 6) == 0)
            ? std::atoi(s0->name + 6) : -1;
        for (uint32_t v = 0; v < 48u; ++v) {
            if (l_out_nodes[out.passes[v].il] == s0 ||
                (s0_lout_il >= 0 && s0_lout_il == (int) out.passes[v].il)) {
                out.passes[v].layer_inp_stores.push_back(st);
                matched = true;
                break;
            }
        }
        if (!matched) {
            W4_MK_FAIL("unknown 2304-wide SET_ROWS src '%s' op=%d", s0 ? s0->name : "null", s0 ? (int) s0->op : -1);
        }
    }

#undef W4_MK_FAIL
    return true;
}

static bool ensure_w4_bound_model(agx_inkling2_engine * eng, const agx_inkling2_model_tensors & W) {
    const uint64_t tok_va = agx_tensor_gpu_va(W.tok_embd);
    const uint64_t out_va = agx_tensor_gpu_va(W.output);
    if (eng->w4_model.bound && eng->w4_model.tok_embd_va == tok_va && eng->w4_model.output_va == out_va) {
        return true;
    }

    // Free any previously bound model's coalesced W_attn_in BOs
    for (uint32_t ip = 0; ip < AGX_INKLING2_MAX_PHYS_LAYERS; ++ip) {
        if (eng->w4_model.layers[ip].valid) {
            agx_repack_coalesced_free(eng->ctx->drm, &eng->w4_model.layers[ip].w_attn_in);
            eng->w4_model.layers[ip] = agx_w4_layer_coalesced{};
        }
    }
    eng->w4_model.bound = false;

    for (uint32_t ip = 0; ip < 24u; ++ip) {
        const auto & L = W.layers[ip];
        // Exact order: [0]=wq, [1]=wqkv_gate, [2]=attn_rel, [3]=attn_sink_proj, [4]=wk, [5]=wv
        const struct ggml_tensor * mats_all[6] = {
            L.wq, L.wqkv_gate, L.attn_rel, L.attn_sink_proj, L.wk, L.wv
        };
        const struct ggml_tensor * q8_mats[6] = {};
        uint32_t q8_src_idx[6] = {};
        uint32_t n_q8 = 0;
        uint32_t q_prefix_cnt = 0;

        for (uint32_t m = 0; m < 6u; ++m) {
            if (mats_all[m] && mats_all[m]->type == GGML_TYPE_Q8_0 && mats_all[m]->ne[0] == 2304) {
                q8_mats[n_q8]    = mats_all[m];
                q8_src_idx[n_q8] = m;
                if (m < 4u) q_prefix_cnt++;
                n_q8++;
            }
        }

        auto & C = eng->w4_model.layers[ip];
        if (n_q8 >= 2u) {
            if (!agx_repack_coalesce_q8_0_matrices(eng->ctx->drm, q8_mats, n_q8, &C.w_attn_in)) {
                AGX_LOG_WARN("agx-inkling2: W_attn_in[%u] coalesce failed; falling back to per-mat GEMM", ip);
                C.valid = false;
            } else {
                C.valid           = true;
                C.total_coal_rows = C.w_attn_in.total_rows;
                C.q_prefix_rows   = 0u;
                for (uint32_t k = 0; k < n_q8; ++k) {
                    const uint32_t m = q8_src_idx[k];
                    C.in_coal[m]  = true;
                    C.coal_idx[m] = k;
                    C.row_off[m]  = C.w_attn_in.row_offsets[k];
                    C.row_cnt[m]  = C.w_attn_in.row_counts[k];
                    C.byte_off[m] = C.w_attn_in.byte_offsets[k];
                    if (k < q_prefix_cnt) {
                        C.q_prefix_rows += C.row_cnt[m];
                    }
                }
            }
        }
    }

    eng->w4_model.bound       = true;
    eng->w4_model.tok_embd_va = tok_va;
    eng->w4_model.output_va   = out_va;
    return true;
}

// ============================================================================
// Wave 7: Fused DSpark Draft Megakernel (3-Layer Q8_0 Body + LM Head + 4-Step Markov Head)
// Collapses ~230 generic dispatches (4.76 ms) -> ~42 fused wire-speed dispatches (~2.2 ms)!
// ============================================================================

struct w7_dspark_layer_desc {
    const struct ggml_tensor * attn_norm      = nullptr;
    const struct ggml_tensor * wq             = nullptr;
    const struct ggml_tensor * wk             = nullptr;
    const struct ggml_tensor * wv             = nullptr; // nullptr -> shared_kv
    const struct ggml_tensor * wo             = nullptr;
    const struct ggml_tensor * attn_q_norm    = nullptr;
    const struct ggml_tensor * attn_k_norm    = nullptr;
    const struct ggml_tensor * attn_post_norm = nullptr;
    const struct ggml_tensor * ffn_post_norm  = nullptr;
    const struct ggml_tensor * out_scale      = nullptr;
    const struct ggml_tensor * attn_sinks     = nullptr;
    const struct ggml_tensor * ffn_norm       = nullptr;
    const struct ggml_tensor * ffn_gate       = nullptr;
    const struct ggml_tensor * ffn_up         = nullptr;
    const struct ggml_tensor * ffn_down       = nullptr;

    // Attention & KV cache
    uint64_t k_va          = 0;
    uint64_t v_va          = 0;
    uint32_t k_nb1         = 0;
    uint32_t k_nb2         = 0;
    uint32_t v_nb1         = 0;
    uint32_t v_nb2         = 0;
    uint32_t n_kv          = 0;
    bool     kv_is_q8      = false;
    bool     has_k_rot     = false;
    bool     has_v_rot     = false;
    uint32_t k_rot_dim     = 0;
    uint32_t v_rot_dim     = 0;

    uint64_t k_store_va    = 0;
    uint64_t v_store_va    = 0;
    uint32_t k_store_nb1   = 0;
    uint32_t v_store_nb1   = 0;
    uint64_t k_idxs_va     = 0;
    uint64_t v_idxs_va     = 0;
    uint32_t k_ids_nb0     = 0;
    uint32_t v_ids_nb0     = 0;
    bool     k_ids_is_i64  = true;
    bool     v_ids_is_i64  = true;

    uint64_t kq_mask_va    = 0;
    uint32_t kq_mask_nb1   = 0;
    uint32_t kq_mask_nb2   = 0;
    uint32_t kq_mask_ne2   = 1;
    bool     kq_mask_f16   = true;
    uint64_t sinks_va      = 0;
    float    kq_scale      = 0.0f;
    float    max_bias      = 0.0f;
    float    logit_softcap = 0.0f;

    // RoPE
    bool     has_rope      = false;
    uint32_t rope_n_dims   = 0;
    uint32_t rope_mode     = 0;
    float    theta_scale   = 1.0f;
    float    freq_scale    = 1.0f;
    float    attn_factor   = 1.0f;
    uint64_t rope_freq_va  = 0;

    // GLU
    uint32_t glu_op        = 2u; // SWIGLU
    float    glu_alpha     = 1.702f;
    float    glu_limit     = 7.0f;
};

struct w7_dspark_draft_graph {
    uint32_t n_layers       = 0;
    uint32_t n_tokens       = 0;
    uint32_t n_blocks       = 1;
    uint32_t block_drafts   = 0;
    uint32_t n_embd         = 0;
    uint32_t n_head_q       = 0;
    uint32_t n_head_kv      = 0;
    uint32_t head_dim       = 0;
    uint32_t n_ff           = 0;
    uint32_t n_vocab        = 0;
    uint32_t n_vocab_padded = 0;
    float    rms_eps        = 1e-6f;
    float    embd_scale     = 0.0f;

    const struct ggml_tensor * tok_embd    = nullptr;
    const struct ggml_tensor * tok_norm    = nullptr;
    const struct ggml_tensor * mask_embd   = nullptr;
    const struct ggml_tensor * output_norm = nullptr;
    const struct ggml_tensor * output      = nullptr;

    uint64_t inp_tokens_va    = 0;
    uint64_t inp_pos_va       = 0;
    uint64_t rnn_state_va     = 0;
    uint64_t result_norm_va   = 0;
    uint64_t lm_head_pad_va   = 0;
    uint64_t final_out_va     = 0;
    uint64_t final_h_nextn_va = 0;
    bool     lm_pad_aliases_out = false;
    bool     lm_head_prec_f32   = false;

    w7_dspark_layer_desc                  layers[AGX_W7_DSPARK_MAX_LAYERS];
    agx_dspark_mtp_tensors                markov;
    std::vector<agx_w7_dspark_topk_entry> topk_rows;
    bool                                  has_any_topk = false;
};

static bool discover_dspark_draft_cgraph(const struct ggml_cgraph * cgraph, w7_dspark_draft_graph & out) {
    if (!cgraph || cgraph->n_nodes < 64) return false;

    const char * dbg_env = std::getenv("GGML_AGX_MK_DEBUG");
    const bool   mk_dbg  = (dbg_env && dbg_env[0] && dbg_env[0] != '0');
    static int   dsp_fail_prints = 0;
#define W7_DSP_FAIL(...) do { \
        if (mk_dbg && dsp_fail_prints < 8) { \
            ++dsp_fail_prints; \
            std::fprintf(stderr, "[ggml-agx DSPARK MK DISCOVER FAIL] " __VA_ARGS__); \
            std::fprintf(stderr, "\n"); \
        } \
        return false; \
    } while (0)

    // Quick presence check: must have markov_w1.weight, markov_w2.weight, joint_proj.weight
    const struct ggml_tensor * fa_nodes[AGX_W7_DSPARK_MAX_LAYERS] = {};
    int                        fa_idx[AGX_W7_DSPARK_MAX_LAYERS]   = {};
    uint32_t                   n_fa = 0u;

    const struct ggml_tensor * inp_tokens       = nullptr;
    const struct ggml_tensor * inp_pos          = nullptr;
    const struct ggml_tensor * base_t           = nullptr;
    const struct ggml_tensor * hid_t            = nullptr;
    const struct ggml_tensor * lm_head_node     = nullptr;
    const struct ggml_tensor * rnn_state        = nullptr;
    const struct ggml_tensor * final_out        = nullptr;
    const struct ggml_tensor * final_h_nextn    = nullptr;
    int                        base_idx         = -1;
    int                        first_w1_idx     = -1;

    auto reg_tensor = [&](const struct ggml_tensor * t) -> bool {
        if (!t || !t->name[0]) return true;
        const char * nm = t->name;
        if      (std::strcmp(nm, "markov_w1.weight")  == 0) out.markov.markov_w1   = t;
        else if (std::strcmp(nm, "markov_w2.weight")  == 0) out.markov.markov_w2   = t;
        else if (std::strcmp(nm, "markov_w2.scale")   == 0) out.markov.markov_w2_s = t;
        else if (std::strcmp(nm, "markov_w2.bias")    == 0) out.markov.markov_w2_b = t;
        else if (std::strcmp(nm, "joint_proj.weight") == 0) out.markov.joint_proj  = t;
        else if (std::strcmp(nm, "joint_proj.bias")   == 0) out.markov.joint_proj_b = t;
        else if (std::strcmp(nm, "token_embd.weight") == 0) out.tok_embd    = t;
        else if (std::strcmp(nm, "token_embd_norm.weight") == 0 ||
                 std::strcmp(nm, "blk.0.token_embd_norm.weight") == 0) out.tok_norm = t;
        else if (std::strcmp(nm, "mask_embd.weight")  == 0) out.mask_embd   = t;
        else if (std::strcmp(nm, "output_norm.weight")== 0) out.output_norm = t;
        else if (std::strcmp(nm, "output.weight")     == 0) out.output      = t;
        else if (std::strcmp(nm, "d2t") == 0 ||
                 std::strcmp(nm, "conf_proj.weight") == 0 ||
                 std::strstr(nm, "hc_attn_fn") != nullptr ||
                 std::strstr(nm, "attn_kv_a_mqa") != nullptr ||
                 std::strstr(nm, "attn_conv_proj") != nullptr ||
                 std::strstr(nm, "ffn_conv_proj") != nullptr) {
            return false; // unsupported DSpark variant -> clean fallback
        } else {
            int blk = -1;
            const char * sfx = nullptr;
            if (w4_parse_blk_name(nm, &blk, &sfx) && blk >= 0 && blk < (int) AGX_W7_DSPARK_MAX_LAYERS) {
                auto & L = out.layers[blk];
                if      (std::strcmp(sfx, "attn_norm.weight")           == 0) L.attn_norm      = t;
                else if (std::strcmp(sfx, "attn_q.weight")              == 0) L.wq             = t;
                else if (std::strcmp(sfx, "attn_k.weight")              == 0) L.wk             = t;
                else if (std::strcmp(sfx, "attn_v.weight")              == 0) L.wv             = t;
                else if (std::strcmp(sfx, "attn_output.weight")         == 0) L.wo             = t;
                else if (std::strcmp(sfx, "attn_q_norm.weight")         == 0) L.attn_q_norm    = t;
                else if (std::strcmp(sfx, "attn_k_norm.weight")         == 0) L.attn_k_norm    = t;
                else if (std::strcmp(sfx, "post_attention_norm.weight") == 0) L.attn_post_norm = t;
                else if (std::strcmp(sfx, "post_ffw_norm.weight")       == 0) L.ffn_post_norm  = t;
                else if (std::strcmp(sfx, "layer_output_scale.weight")  == 0) L.out_scale      = t;
                else if (std::strcmp(sfx, "attn_sinks.weight")          == 0) L.attn_sinks     = t;
                else if (std::strcmp(sfx, "ffn_norm.weight")            == 0) L.ffn_norm       = t;
                else if (std::strcmp(sfx, "ffn_gate.weight")            == 0) L.ffn_gate       = t;
                else if (std::strcmp(sfx, "ffn_up.weight")              == 0) L.ffn_up         = t;
                else if (std::strcmp(sfx, "ffn_down.weight")            == 0) L.ffn_down       = t;
            }
        }
        return true;
    };

    for (int i = 0; i < cgraph->n_leafs; ++i) {
        if (!reg_tensor(cgraph->leafs[i])) return false;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * n = cgraph->nodes[i];
        if (!n) continue;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s]) {
                if (!reg_tensor(n->src[s])) return false;
                if (first_w1_idx < 0 && n->src[s]->name[0] &&
                    std::strcmp(n->src[s]->name, "markov_w1.weight") == 0) {
                    first_w1_idx = i;
                }
            }
        }
        if (n->op == GGML_OP_FLASH_ATTN_EXT) {
            if (n_fa >= AGX_W7_DSPARK_MAX_LAYERS) return false;
            fa_nodes[n_fa] = n;
            fa_idx[n_fa]   = i;
            ++n_fa;
        }
        if (n->op == GGML_OP_MUL_MAT && n->src[0] && std::strcmp(n->src[0]->name, "output.weight") == 0) {
            lm_head_node = n;
        }
        if (n->name[0]) {
            if (std::strcmp(n->name, "result_output") == 0) { base_t = n; base_idx = i; }
            else if (std::strcmp(n->name, "result_norm") == 0) hid_t = n;
            else if (std::strcmp(n->name, "attn_out_scaled") == 0) return false; // attn_value_scale not supported
        }
        if (n->op == GGML_OP_RMS_NORM && out.rms_eps == 1e-6f) {
            float eps = 0.0f;
            std::memcpy(&eps, n->op_params, sizeof(float));
            if (eps > 0.0f) out.rms_eps = eps;
        }
    }

    if (!out.markov.markov_w1 || !out.markov.markov_w2 || !out.markov.joint_proj ||
        n_fa == 0u || !base_t || !hid_t || !lm_head_node || !out.tok_embd || !out.output || !out.output_norm ||
        base_idx < 0 || first_w1_idx <= base_idx) {
        return false;
    }

    // Extract n_blocks, block_drafts, inp_tokens, rnn_state across the ENTIRE Markov tail [base_idx+1 .. n_nodes-1]
    // (Note: ggml_build_forward_expand visits w1_prev BEFORE f_rnn, so f_rnn / rnn_state has node index > first_w1_idx!)
    const struct ggml_tensor * prev_cont = nullptr;
    for (int i = base_idx + 1; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * nd = cgraph->nodes[i];
        if (!nd) continue;
        if (!prev_cont && i < first_w1_idx && nd->op == GGML_OP_CONT && nd->type == GGML_TYPE_I32 &&
            nd->src[0] && nd->src[0]->op == GGML_OP_VIEW && nd->src[0]->view_src &&
            nd->src[0]->view_src->type == GGML_TYPE_I32) {
            prev_cont  = nd;
            inp_tokens = nd->src[0]->view_src;
        }
        if (!rnn_state && nd->op == GGML_OP_CONT && nd->src[0] && nd->src[0]->op == GGML_OP_VIEW &&
            nd->src[0]->view_src && nd->src[0]->view_src->type == GGML_TYPE_F32 &&
            nd->src[0]->view_src->ne[0] == hid_t->ne[0] &&
            nd->src[0]->view_src->ne[1] == base_t->ne[1] &&
            nd->src[0]->view_src != hid_t) {
            rnn_state = nd->src[0]->view_src;
        }
    }
    if (!inp_tokens || !prev_cont || prev_cont->ne[0] < 1 || prev_cont->ne[0] > 4) {
        W7_DSP_FAIL("inp_tokens/prev_cont missing or n_blocks=%lld", prev_cont ? (long long) prev_cont->ne[0] : -1LL);
    }

    out.n_layers       = n_fa;
    out.n_tokens       = (uint32_t) base_t->ne[1];
    out.n_blocks       = (uint32_t) prev_cont->ne[0];
    if (out.n_tokens == 0u || out.n_tokens > 16u || (out.n_tokens % out.n_blocks) != 0u ||
        base_t->ne[1] != hid_t->ne[1] || base_t->ne[1] != inp_tokens->ne[0]) {
        W7_DSP_FAIL("token/block shape mismatch M=%u blk=%u", out.n_tokens, out.n_blocks);
    }
    out.block_drafts   = out.n_tokens / out.n_blocks;
    out.n_embd         = (uint32_t) hid_t->ne[0];
    out.n_vocab        = (uint32_t) base_t->ne[0];
    out.n_vocab_padded = (uint32_t) lm_head_node->ne[0];
    if (out.block_drafts < 2u || out.n_embd > 3072u || (out.n_embd % 4u) != 0u) {
        W7_DSP_FAIL("unsupported dims drafts=%u D=%u", out.block_drafts, out.n_embd);
    }

    // Check optional embedding scale on inp_noise_embd
    for (int i = 0; i < fa_idx[0]; ++i) {
        const struct ggml_tensor * nd = cgraph->nodes[i];
        if (nd && nd->name[0] && std::strcmp(nd->name, "inp_noise_embd") == 0 && nd->op == GGML_OP_SCALE) {
            float s = 0.0f, b = 0.0f;
            w4_read_scale_params(nd, &s, &b);
            if (b != 0.0f) W7_DSP_FAIL("inp_noise_embd bias != 0");
            out.embd_scale = s;
        }
    }

    // Detect sample_from_anchor, final_out, final_h_nextn, and softcap in [base_idx+1 .. n_nodes-1]
    // NOTE: ggml_build_forward_expand expands res->t_h_nextn BEFORE res->t_logits, so scan ALL
    //       tail nodes for CONT(PERMUTE(...)) rather than assuming n_exec - 2!
    bool sample_from_anchor = true;
    float softcap = 0.0f;
    for (int i = base_idx + 1; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * nd = cgraph->nodes[i];
        if (!nd) continue;
        if (nd->op == GGML_OP_CONT && nd->src[0] && nd->src[0]->op == GGML_OP_VIEW &&
            nd->src[0]->view_src == base_t && nd->src[0]->view_offs == 0) {
            sample_from_anchor = false;
        }
        if (nd->op == GGML_OP_CONT && nd->type == GGML_TYPE_F32 &&
            nd->src[0] && nd->src[0]->op == GGML_OP_PERMUTE) {
            if (ggml_nelements(nd) == (int64_t) out.n_vocab * out.n_tokens) {
                final_out = nd;
            } else if (rnn_state && ggml_nelements(nd) == (int64_t) out.n_embd * out.n_tokens) {
                final_h_nextn = nd;
            }
        }
        if (softcap == 0.0f && i >= first_w1_idx && i + 2 < cgraph->n_nodes) {
            const struct ggml_tensor * s0 = cgraph->nodes[i];
            const struct ggml_tensor * u1 = cgraph->nodes[i + 1];
            const struct ggml_tensor * s2 = cgraph->nodes[i + 2];
            if (s0 && u1 && s2 &&
                s0->op == GGML_OP_SCALE && s0->ne[0] == (int64_t) out.n_vocab &&
                u1->op == GGML_OP_UNARY && ggml_get_unary_op(u1) == GGML_UNARY_OP_TANH &&
                s2->op == GGML_OP_SCALE && u1->src[0] == s0 && s2->src[0] == u1) {
                float cap = 0.0f, b = 0.0f;
                w4_read_scale_params(s2, &cap, &b);
                if (cap > 0.0f && b == 0.0f) softcap = cap;
            }
        }
    }
    if (!final_out || (rnn_state && !final_h_nextn)) {
        W7_DSP_FAIL("missing final_out=%d final_h_nextn=%d (rnn=%d)", !!final_out, !!final_h_nextn, !!rnn_state);
    }

    // Per-layer discovery: weights, RoPE, K/V SET_ROWS, FLASH_ATTN_EXT, GLU
    for (uint32_t il = 0u; il < n_fa; ++il) {
        auto & L = out.layers[il];
        if (!L.attn_norm || !L.wq || !L.wk || !L.wo || !L.attn_q_norm || !L.attn_k_norm ||
            !L.ffn_norm || !L.ffn_gate || !L.ffn_up || !L.ffn_down) {
            W7_DSP_FAIL("layer %u missing required weight", il);
        }
        if (L.attn_norm->type != GGML_TYPE_F32 || L.attn_q_norm->type != GGML_TYPE_F32 ||
            L.attn_k_norm->type != GGML_TYPE_F32 || L.ffn_norm->type != GGML_TYPE_F32 ||
            (L.attn_post_norm && L.attn_post_norm->type != GGML_TYPE_F32) ||
            (L.ffn_post_norm  && L.ffn_post_norm->type  != GGML_TYPE_F32) ||
            (L.out_scale      && L.out_scale->type      != GGML_TYPE_F32)) {
            W7_DSP_FAIL("layer %u non-F32 norm/scale", il);
        }

        const struct ggml_tensor * fa   = fa_nodes[il];
        const struct ggml_tensor * q_in = fa->src[0];
        const struct ggml_tensor * k_l  = fa->src[1];
        const struct ggml_tensor * v_l  = fa->src[2];
        const struct ggml_tensor * m_in = fa->src[3];
        const struct ggml_tensor * s_in = fa->src[4];
        if (!q_in || !k_l || !v_l) W7_DSP_FAIL("layer %u fa missing src", il);
        if ((k_l->type != GGML_TYPE_F16 && k_l->type != GGML_TYPE_Q8_0) || k_l->type != v_l->type) {
            W7_DSP_FAIL("layer %u kv type k=%d v=%d", il, (int) k_l->type, (int) v_l->type);
        }
        if (il == 0u) {
            out.head_dim  = (uint32_t) q_in->ne[0];
            out.n_head_q  = (uint32_t) q_in->ne[2];
            out.n_head_kv = (uint32_t) k_l->ne[2];
            out.n_ff      = (uint32_t) L.ffn_gate->ne[1];
            if (out.head_dim > 256u || (out.head_dim % 32u) != 0u || out.n_head_kv == 0u ||
                (out.n_head_q % out.n_head_kv) != 0u) {
                W7_DSP_FAIL("unsupported head dims hd=%u Hq=%u Hkv=%u", out.head_dim, out.n_head_q, out.n_head_kv);
            }
        } else if ((uint32_t) q_in->ne[0] != out.head_dim || (uint32_t) q_in->ne[2] != out.n_head_q ||
                   (uint32_t) k_l->ne[2] != out.n_head_kv || (uint32_t) L.ffn_gate->ne[1] != out.n_ff) {
            W7_DSP_FAIL("layer %u heterogeneous dims", il);
        }

        L.kv_is_q8 = (k_l->type == GGML_TYPE_Q8_0);
        L.k_va     = agx_tensor_gpu_va(k_l);
        L.v_va     = agx_tensor_gpu_va(v_l);
        L.k_nb1    = (uint32_t) k_l->nb[1];
        L.k_nb2    = (uint32_t) k_l->nb[2];
        L.v_nb1    = (uint32_t) v_l->nb[1];
        L.v_nb2    = (uint32_t) v_l->nb[2];
        L.n_kv     = (uint32_t) k_l->ne[1];
        if (!L.k_va || !L.v_va) W7_DSP_FAIL("layer %u missing kv VA", il);
        if (m_in) {
            if (m_in->type != GGML_TYPE_F16 && m_in->type != GGML_TYPE_F32) {
                W7_DSP_FAIL("layer %u unsupported mask type=%d", il, (int) m_in->type);
            }
            L.kq_mask_va  = agx_tensor_gpu_va(m_in);
            L.kq_mask_nb1 = (uint32_t) m_in->nb[1];
            L.kq_mask_nb2 = (m_in->ne[2] > 1) ? (uint32_t) m_in->nb[2] : 0u;
            L.kq_mask_ne2 = (uint32_t) std::max<int64_t>(1, m_in->ne[2]);
            L.kq_mask_f16 = (m_in->type == GGML_TYPE_F16);
            if (!L.kq_mask_va) W7_DSP_FAIL("layer %u missing mask VA", il);
        }
        if (s_in) {
            L.sinks_va = agx_tensor_gpu_va(s_in);
            if (!L.sinks_va) W7_DSP_FAIL("layer %u missing sinks VA", il);
        }
        std::memcpy(&L.kq_scale,      (const char *) fa->op_params + 0, sizeof(float));
        std::memcpy(&L.max_bias,      (const char *) fa->op_params + 4, sizeof(float));
        std::memcpy(&L.logit_softcap, (const char *) fa->op_params + 8, sizeof(float));

        // Hadamard rotation checks (exact block size: 64 or 128!)
        const struct ggml_tensor * q_root = w4_strip_views(q_in);
        if (q_root && q_root->op == GGML_OP_MUL_MAT &&
            ggml_get_op_params_i32(q_root, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
            const uint32_t rd = (q_root->src[0]) ? (uint32_t) q_root->src[0]->ne[0] : 128u;
            if ((rd != 64u && rd != 128u) || out.head_dim != 128u) {
                W7_DSP_FAIL("layer %u unsupported k_rot_dim=%u (hd=%u)", il, rd, out.head_dim);
            }
            L.has_k_rot = true;
            L.k_rot_dim = rd;
        }
        for (int j = fa_idx[il] + 1; j < std::min(cgraph->n_nodes, fa_idx[il] + 8); ++j) {
            const struct ggml_tensor * cand = cgraph->nodes[j];
            if (cand && cand->op == GGML_OP_MUL_MAT &&
                ggml_get_op_params_i32(cand, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                const struct ggml_tensor * s1 = w4_strip_views(cand->src[1]);
                if (s1 == fa || (s1 && s1->op == GGML_OP_CONT && w4_strip_views(s1->src[0]) == fa)) {
                    const uint32_t rd = (cand->src[0]) ? (uint32_t) cand->src[0]->ne[0] : 128u;
                    L.has_v_rot = true;
                    L.v_rot_dim = rd;
                    break;
                }
            }
        }
        if (L.has_v_rot && (L.v_rot_dim != 64u || out.head_dim != 128u)) {
            W7_DSP_FAIL("layer %u v_rot_dim=%u on draft unsupported (only 64 supported)", il, L.v_rot_dim);
        }

        // Scan window [win_beg, win_end) for RoPE, K/V SET_ROWS, and GLU
        const int win_beg = (il == 0u) ? 0 : fa_idx[il - 1u];
        const int win_end = (il + 1u < n_fa) ? fa_idx[il + 1u] : base_idx;
        const uint64_t k_size_B = (uint64_t) ggml_nbytes(k_l);
        const uint64_t v_size_B = (uint64_t) ggml_nbytes(v_l);
        const int64_t  kv_dim   = (int64_t) out.n_head_kv * out.head_dim;

        bool found_glu = false;
        for (int j = win_beg; j < win_end; ++j) {
            const struct ggml_tensor * nd = cgraph->nodes[j];
            if (!nd) continue;
            if (j < fa_idx[il] && nd->op == GGML_OP_ROPE && !L.has_rope) {
                const uint32_t mode = (uint32_t) ggml_get_op_params_i32(nd, 2);
                if (mode & 8u) W7_DSP_FAIL("layer %u M-RoPE unsupported", il);
                L.has_rope    = true;
                L.rope_n_dims = (uint32_t) ggml_get_op_params_i32(nd, 1);
                L.rope_mode   = mode;
                float freq_base = 10000.0f, freq_scale = 1.0f, attn_factor = 1.0f;
                std::memcpy(&freq_base,   (const char *) nd->op_params + 20, sizeof(float));
                std::memcpy(&freq_scale,  (const char *) nd->op_params + 24, sizeof(float));
                std::memcpy(&attn_factor, (const char *) nd->op_params + 32, sizeof(float));
                L.theta_scale  = std::pow(freq_base, -2.0f / (float) L.rope_n_dims);
                L.freq_scale   = freq_scale;
                L.attn_factor  = attn_factor;
                L.rope_freq_va = nd->src[2] ? agx_tensor_gpu_va(nd->src[2]) : 0ull;
                if (nd->src[1]) inp_pos = nd->src[1];
            }
            if (j < fa_idx[il] && nd->op == GGML_OP_SET_ROWS && nd->src[0] && nd->src[1]) {
                if (!L.k_store_va && w4_tensor_in_va_range(nd, L.k_va, k_size_B) && nd->ne[0] == kv_dim) {
                    L.k_store_va   = agx_tensor_gpu_va(nd);
                    L.k_store_nb1  = (uint32_t) nd->nb[1];
                    L.k_idxs_va    = agx_tensor_gpu_va(nd->src[1]);
                    L.k_ids_is_i64 = (nd->src[1]->type == GGML_TYPE_I64);
                    L.k_ids_nb0    = (uint32_t) nd->src[1]->nb[0];
                } else if (!L.v_store_va && w4_tensor_in_va_range(nd, L.v_va, v_size_B) && nd->ne[0] == kv_dim) {
                    L.v_store_va   = agx_tensor_gpu_va(nd);
                    L.v_store_nb1  = (uint32_t) nd->nb[1];
                    L.v_idxs_va    = agx_tensor_gpu_va(nd->src[1]);
                    L.v_ids_is_i64 = (nd->src[1]->type == GGML_TYPE_I64);
                    L.v_ids_nb0    = (uint32_t) nd->src[1]->nb[0];
                }
            }
            if (j > fa_idx[il] && !found_glu) {
                if (nd->op == GGML_OP_GLU && nd->ne[0] == (int64_t) out.n_ff) {
                    L.glu_op = (uint32_t) ggml_get_glu_op(nd);
                    if (ggml_get_op_params_i32(nd, 1) != 0) W7_DSP_FAIL("layer %u swapped GLU", il);
                    std::memcpy(&L.glu_alpha, (const char *) nd->op_params + 8,  sizeof(float));
                    std::memcpy(&L.glu_limit, (const char *) nd->op_params + 12, sizeof(float));
                    found_glu = true;
                } else if (nd->op == GGML_OP_MUL && nd->ne[0] == (int64_t) out.n_ff &&
                           nd->src[0] && nd->src[0]->op == GGML_OP_UNARY) {
                    const enum ggml_unary_op uop = ggml_get_unary_op(nd->src[0]);
                    if      (uop == GGML_UNARY_OP_SILU)       { L.glu_op = 2u; found_glu = true; }
                    else if (uop == GGML_UNARY_OP_GELU)       { L.glu_op = 1u; found_glu = true; }
                    else if (uop == GGML_UNARY_OP_GELU_QUICK) { L.glu_op = 5u; found_glu = true; }
                }
            }
        }
        if (!L.k_store_va || !L.v_store_va || !L.k_idxs_va || !L.v_idxs_va || !found_glu) {
            W7_DSP_FAIL("layer %u missing kv store or glu (ks=%d vs=%d glu=%d)",
                        il, !!L.k_store_va, !!L.v_store_va, (int) found_glu);
        }
    }

    // Verify no unexpected SET_ROWS or OUTPUT tensors exist in the graph.
    // Accept backend-sampling TOP_K + GET_ROWS (top_k_rows) OUTPUT pairs on rows of final_out (or PAD(final_out))!
    auto trace_final_out_row = [&](const struct ggml_tensor * t, uint32_t * out_row) -> bool {
        uint64_t total_offs = 0;
        const struct ggml_tensor * cur = t;
        while (cur && (cur->op == GGML_OP_RESHAPE || cur->op == GGML_OP_VIEW ||
                       cur->op == GGML_OP_PERMUTE || cur->op == GGML_OP_TRANSPOSE)) {
            if (cur->op == GGML_OP_VIEW) total_offs += (uint64_t) cur->view_offs;
            cur = cur->src[0];
        }
        if (!cur) return false;
        const bool is_final = (cur == final_out) ||
                              (cur->op == GGML_OP_PAD && w4_strip_views(cur->src[0]) == final_out);
        if (!is_final) return false;
        const uint64_t row_bytes = (uint64_t) out.n_vocab * sizeof(float);
        if (row_bytes == 0 || (total_offs % row_bytes) != 0) return false;
        const uint64_t r = total_offs / row_bytes;
        if (r >= (uint64_t) out.n_tokens) return false;
        *out_row = (uint32_t) r;
        return true;
    };

    out.topk_rows.assign(out.n_tokens, agx_w7_dspark_topk_entry{ 0ull, 0ull, 0ull });
    out.has_any_topk = false;
    std::vector<const struct ggml_tensor *> topk_nodes(out.n_tokens, nullptr);

    uint32_t total_set_rows = 0u;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * nd = cgraph->nodes[i];
        if (!nd) continue;
        if (nd->op == GGML_OP_SET_ROWS) ++total_set_rows;
        if ((nd->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            const struct ggml_tensor * root = w4_strip_views(nd);
            if (root == final_out || root == final_h_nextn || root == hid_t || root == base_t) {
                continue;
            }
            if (nd->op == GGML_OP_TOP_K && nd->type == GGML_TYPE_I32 &&
                nd->ne[0] >= 1 && nd->ne[0] <= 64 && ggml_nelements(nd) == nd->ne[0]) {
                uint32_t r = 0u;
                const uint64_t va = agx_tensor_gpu_va(nd);
                if (va != 0ull && trace_final_out_row(nd->src[0], &r) && topk_nodes[r] == nullptr) {
                    topk_nodes[r]         = nd;
                    out.topk_rows[r].ids_va = va;
                    out.topk_rows[r].k_u64  = (uint64_t) nd->ne[0];
                    out.has_any_topk      = true;
                    continue;
                }
            }
            if (nd->op == GGML_OP_GET_ROWS && nd->type == GGML_TYPE_F32 && nd->src[1]) {
                uint32_t r = 0u;
                const uint64_t va = agx_tensor_gpu_va(nd);
                if (va != 0ull && trace_final_out_row(nd->src[0], &r) &&
                    topk_nodes[r] != nullptr && w4_strip_views(nd->src[1]) == topk_nodes[r] &&
                    ggml_nelements(nd) == (int64_t) out.topk_rows[r].k_u64 &&
                    out.topk_rows[r].vals_va == 0ull) {
                    out.topk_rows[r].vals_va = va;
                    continue;
                }
            }
            W7_DSP_FAIL("unexpected OUTPUT '%s' op=%d", nd->name, (int) nd->op);
        }
    }
    if (total_set_rows != 2u * n_fa) {
        W7_DSP_FAIL("unexpected SET_ROWS count=%u != %u", total_set_rows, 2u * n_fa);
    }

    out.inp_tokens_va    = agx_tensor_gpu_va(inp_tokens);
    out.inp_pos_va       = inp_pos ? agx_tensor_gpu_va(inp_pos) : 0ull;
    out.rnn_state_va     = rnn_state ? agx_tensor_gpu_va(rnn_state) : 0ull;
    out.result_norm_va   = agx_tensor_gpu_va(hid_t);
    out.lm_head_pad_va   = agx_tensor_gpu_va(lm_head_node);
    out.final_out_va     = agx_tensor_gpu_va(final_out);
    out.final_h_nextn_va = final_h_nextn ? agx_tensor_gpu_va(final_h_nextn) : 0ull;
    if (!out.inp_tokens_va || !out.final_out_va || !out.lm_head_pad_va ||
        (rnn_state && (!out.rnn_state_va || !out.final_h_nextn_va))) {
        W7_DSP_FAIL("missing core GPU VA");
    }

    // Check if lm_head_pad_va overlaps final_out_va (gallocr reuse)
    const uint64_t lm_sz = (uint64_t) ggml_nbytes(lm_head_node);
    const uint64_t fo_sz = (uint64_t) ggml_nbytes(final_out);
    out.lm_pad_aliases_out =
        !(out.lm_head_pad_va + lm_sz <= out.final_out_va || out.final_out_va + fo_sz <= out.lm_head_pad_va) ||
        ((base_t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0);
    out.lm_head_prec_f32 = (ggml_prec_get_acc(lm_head_node) == GGML_PREC_F32 ||
                            ggml_prec_get_src(lm_head_node, 0) == GGML_PREC_F32 ||
                            ggml_prec_get_src(lm_head_node, 1) == GGML_PREC_F32);

    // Populate Markov head descriptor
    out.markov.n_embd             = out.n_embd;
    out.markov.rank               = (uint32_t) out.markov.markov_w1->ne[0];
    out.markov.n_vocab            = out.n_vocab;
    out.markov.block_size         = out.block_drafts;
    out.markov.sample_from_anchor = sample_from_anchor;
    out.markov.has_f_rnn          = (rnn_state != nullptr);
    out.markov.softcap            = softcap;
    out.markov.rms_eps            = out.rms_eps;
    if (out.markov.rank == 0u || out.markov.rank > 512u) {
        W7_DSP_FAIL("unsupported markov rank=%u", out.markov.rank);
    }

#undef W7_DSP_FAIL
    return true;
}

static void ensure_w8_dspark_draft_bound(agx_backend_context *         ctx,
                                         agx_inkling2_engine *         eng,
                                         const w7_dspark_draft_graph & G) {
    if (!ctx || !eng || !eng->k_w8_dspark_prep || G.n_layers == 0u) return;
    auto & B = eng->dspark_bound;
    const uint64_t cur_w2_va = G.markov.markov_w2 ? agx_tensor_gpu_va(G.markov.markov_w2) : 0ull;
    const uint64_t cur_wq0   = G.layers[0].wq ? agx_tensor_gpu_va(G.layers[0].wq) : 0ull;
    if (B.draft_bound && B.w2_src_va == cur_w2_va && B.w2_n_vocab == G.n_vocab && B.wq_src_va[0] == cur_wq0) {
        return;
    }

    if (B.w2_q4_bo) {
        agx_bo_free(ctx->drm, B.w2_q4_bo);
        B.w2_q4_bo = nullptr;
    }
    for (uint32_t i = 0; i < AGX_W7_DSPARK_MAX_LAYERS; ++i) {
        if (B.w_qkv_bo[i]) {
            agx_bo_free(ctx->drm, B.w_qkv_bo[i]);
            B.w_qkv_bo[i] = nullptr;
        }
        B.wq_src_va[i]  = 0ull;
        B.w_qkv_rows[i] = 0u;
    }
    B.w2_src_va   = cur_w2_va;
    B.w2_n_vocab  = G.n_vocab;
    B.draft_bound = true;

    struct prep_job {
        agx_w8_dspark_prep_push pc;
        uint32_t                grid_x;
    };
    std::vector<prep_job> jobs;

    // 1. SLC-Pinned Q4_32 markov_w2 (28.81 MB < 48 MiB SLC -> 1.64 TB/s across Markov steps!)
    const char * force_bf16 = std::getenv("GGML_AGX_DSPARK_W2_BF16");
    const bool want_q4 = !(force_bf16 && force_bf16[0] && force_bf16[0] != '0');
    uint32_t w2_rb = 0;
    const uint32_t w2_wt = agx_inkling2_tensor_wtype(G.markov.markov_w2, &w2_rb);
    if (want_q4 && G.markov.markov_w2 && G.markov.rank == 256u && w2_wt <= 2u && G.n_vocab > 0u) {
        const size_t w2_bytes = agx_align_up((size_t) G.n_vocab * AGX_W8_Q4_32_R256_ROW_BYTES, AGX_CACHE_LINE_BYTES);
        B.w2_q4_bo = agx_bo_alloc(ctx->drm, w2_bytes, AGX_BO_SLC_PIN, "dspark-w2-q4-slc");
        if (B.w2_q4_bo) {
            prep_job jb{};
            jb.pc.src0_va = cur_w2_va;
            jb.pc.dst_va  = B.w2_q4_bo->gpu_va;
            jb.pc.op      = 0u;
            jb.pc.n0      = G.n_vocab;
            jb.pc.n1      = w2_wt;
            jb.pc.n2      = w2_rb;
            jb.grid_x     = (G.n_vocab + 31u) / 32u;
            jobs.push_back(jb);
        }
    }

    // 2. Wave 9: Coalesced [wq; wk; wv] (or [wq; wk]) Q8_0_8BLK per draft layer (AGX_BO_SLC_STREAM -> never evicts w2!)
    const char * no_coal_qkv = std::getenv("GGML_AGX_DSPARK_NO_COAL_QKV");
    const bool want_coal_qkv = !(no_coal_qkv && no_coal_qkv[0] && no_coal_qkv[0] != '0') && ((G.n_embd % 256u) == 0u);
    if (want_coal_qkv) {
        const uint32_t row_bytes  = (G.n_embd / 256u) * AGX_Q8_0_CHUNK_BYTES;
        const uint32_t row_uvec4s = row_bytes / 16u;
        for (uint32_t il = 0u; il < G.n_layers && il < AGX_W7_DSPARK_MAX_LAYERS; ++il) {
            const auto & L = G.layers[il];
            B.wq_src_va[il] = L.wq ? agx_tensor_gpu_va(L.wq) : 0ull;
            const bool ok_qk = L.wq && L.wk &&
                               L.wq->type == GGML_TYPE_Q8_0 && agx_tensor_is_repacked(L.wq) &&
                               L.wk->type == GGML_TYPE_Q8_0 && agx_tensor_is_repacked(L.wk) &&
                               L.wq->ne[0] == (int64_t) G.n_embd && L.wk->ne[0] == (int64_t) G.n_embd;
            const bool ok_v  = (!L.wv) || (L.wv->type == GGML_TYPE_Q8_0 && agx_tensor_is_repacked(L.wv) &&
                                           L.wv->ne[0] == (int64_t) G.n_embd);
            if (!ok_qk || !ok_v) continue;
            const uint32_t n0  = (uint32_t) L.wq->ne[1];
            const uint32_t n1  = (uint32_t) L.wk->ne[1];
            const uint32_t n2  = L.wv ? (uint32_t) L.wv->ne[1] : 0u;
            const uint32_t tot = n0 + n1 + n2;
            const size_t   sz  = agx_align_up((size_t) tot * row_bytes, AGX_CACHE_LINE_BYTES);
            B.w_qkv_bo[il] = agx_bo_alloc(ctx->drm, sz, AGX_BO_SLC_STREAM, "dspark-wqkv-coal");
            if (B.w_qkv_bo[il]) {
                B.w_qkv_rows[il] = tot;
                prep_job jb{};
                jb.pc.src0_va    = agx_tensor_gpu_va(L.wq);
                jb.pc.src1_va    = agx_tensor_gpu_va(L.wk);
                jb.pc.src2_va    = L.wv ? agx_tensor_gpu_va(L.wv) : 0ull;
                jb.pc.dst_va     = B.w_qkv_bo[il]->gpu_va;
                jb.pc.op         = 1u;
                jb.pc.n0         = n0;
                jb.pc.n1         = n1;
                jb.pc.n2         = n2;
                jb.pc.row_uvec4s = row_uvec4s;
                jb.grid_x        = (tot * row_uvec4s + 255u) / 256u;
                jobs.push_back(jb);
            }
        }
    }

    if (!jobs.empty()) {
        agx_cdm_stream_reset(ctx->stream);
        bool ok = true;
        for (size_t j = 0; j < jobs.size(); ++j) {
            const agx_barrier_mode bar = (j + 1u == jobs.size()) ? AGX_BARRIER_MEMORY_HOST : AGX_BARRIER_NONE;
            if (!agx_cdm_dispatch(ctx->stream, eng->k_w8_dspark_prep, jobs[j].grid_x, 1u, 1u,
                                  &jobs[j].pc, sizeof(jobs[j].pc), bar)) {
                ok = false;
                break;
            }
        }
        if (!ok || agx_cdm_submit(ctx->stream, true, nullptr) != 0) {
            AGX_LOG_WARN("ensure_w8_dspark_draft_bound: GPU prep submit failed; falling back to uncoalesced/BF16");
            if (B.w2_q4_bo) {
                agx_bo_free(ctx->drm, B.w2_q4_bo);
                B.w2_q4_bo = nullptr;
            }
            for (uint32_t i = 0; i < AGX_W7_DSPARK_MAX_LAYERS; ++i) {
                if (B.w_qkv_bo[i]) {
                    agx_bo_free(ctx->drm, B.w_qkv_bo[i]);
                    B.w_qkv_bo[i] = nullptr;
                }
                B.w_qkv_rows[i] = 0u;
            }
        }
    }
}

// ============================================================================
// Wave 8: Fused 3-Stage DSpark Inject Graph (`build_inject_ctx_mix` -> 5 disp!)
// Kills the 28-dispatch 0.75 ms mystery submit between Target Verify & Draft!
// ============================================================================

struct w8_dspark_inject_layer {
    const struct ggml_tensor * ctx_mix    = nullptr;
    const struct ggml_tensor * ctx_norm   = nullptr;
    const struct ggml_tensor * wk_ctx     = nullptr;
    const struct ggml_tensor * wv_ctx     = nullptr;
    const struct ggml_tensor * k_ctx_norm = nullptr;

    uint64_t k_store_va   = 0;
    uint64_t v_store_va   = 0;
    uint32_t k_store_nb1  = 0;
    uint32_t v_store_nb1  = 0;
    uint64_t k_idxs_va    = 0;
    uint64_t v_idxs_va    = 0;
    uint32_t k_ids_nb0    = 0;
    uint32_t v_ids_nb0    = 0;
    bool     k_ids_is_i64 = true;
    bool     v_ids_is_i64 = true;
    bool     kv_is_q8     = false;
    bool     has_k_rot    = false;
    bool     has_v_rot    = false;
    uint32_t k_rot_dim    = 0;
    uint32_t v_rot_dim    = 0;
    bool     has_rope     = false;
};

struct w8_dspark_inject_graph {
    uint32_t n_layers     = 0;
    uint32_t n_tokens     = 0;
    uint32_t n_embd       = 0;
    uint32_t n_target     = 0;
    uint32_t n_head_kv    = 0;
    uint32_t head_dim     = 0;
    uint32_t feat_row_f32 = 0;
    float    rms_eps      = 1e-6f;

    uint32_t rope_n_dims  = 0;
    uint32_t rope_mode    = 0;
    float    theta_scale  = 1.0f;
    float    freq_scale   = 1.0f;
    float    attn_factor  = 1.0f;
    uint64_t rope_freq_va = 0;
    uint64_t inp_pos_va   = 0;

    uint64_t feat_src_va  = 0;
    uint64_t embd_rows_va = 0;
    uint64_t f_rnn_va     = 0;
    bool     has_f_rnn    = false;

    const struct ggml_tensor * markov_ctx_mix  = nullptr;
    const struct ggml_tensor * markov_ctx_norm = nullptr;
    w8_dspark_inject_layer     layers[AGX_W7_DSPARK_MAX_LAYERS];
};

static bool discover_dspark_inject_cgraph(const struct ggml_cgraph * cgraph, w8_dspark_inject_graph & out) {
    if (!cgraph || cgraph->n_nodes < 8 || cgraph->n_nodes > 128) return false;

    const char * dbg_env = std::getenv("GGML_AGX_MK_DEBUG");
    const bool   inj_dbg = (dbg_env && dbg_env[0] && dbg_env[0] != '0');
    static uint32_t inj_fail_logs = 0u;
    auto inj_fail = [&](const char * reason) -> bool {
        if (inj_dbg && inj_fail_logs < 8u) {
            ++inj_fail_logs;
            std::fprintf(stderr, "[ggml-agx DSPARK INJECT SKIP] n_nodes=%d M=%u reason=%s\n",
                         cgraph->n_nodes, out.n_tokens, reason);
        }
        return false;
    };

    uint32_t max_inject_m = 65535u;
    if (const char * max_m_env = std::getenv("GGML_AGX_DSPARK_INJECT_MAX_M")) {
        const int v = std::atoi(max_m_env);
        if (v > 0) max_inject_m = (uint32_t) v;
    }

    int max_blk = -1;
    const struct ggml_tensor * feats_node = nullptr;
    const struct ggml_tensor * f_rnn_node = nullptr;

    auto reg_t = [&](const struct ggml_tensor * t) -> bool {
        if (!t || !t->name[0]) return true;
        const char * nm = t->name;
        if      (std::strcmp(nm, "markov_ctx_mix.weight")  == 0) out.markov_ctx_mix  = t;
        else if (std::strcmp(nm, "markov_ctx_norm.weight") == 0) out.markov_ctx_norm = t;
        else {
            int blk = -1;
            const char * sfx = nullptr;
            if (w4_parse_blk_name(nm, &blk, &sfx)) {
                if (blk < 0 || blk >= (int) AGX_W7_DSPARK_MAX_LAYERS) {
                    if (std::strstr(sfx, "ctx_") != nullptr) return false;
                    return true;
                }
                auto & L = out.layers[blk];
                bool hit = true;
                if      (std::strcmp(sfx, "ctx_mix.weight")         == 0) L.ctx_mix    = t;
                else if (std::strcmp(sfx, "ctx_norm.weight")        == 0) L.ctx_norm   = t;
                else if (std::strcmp(sfx, "attn_k_ctx.weight")      == 0) L.wk_ctx     = t;
                else if (std::strcmp(sfx, "attn_v_ctx.weight")      == 0) L.wv_ctx     = t;
                else if (std::strcmp(sfx, "attn_k_ctx_norm.weight") == 0) L.k_ctx_norm = t;
                else hit = false;
                if (hit && blk > max_blk) max_blk = blk;
            }
        }
        return true;
    };

    for (int i = 0; i < cgraph->n_leafs; ++i) {
        if (!reg_t(cgraph->leafs[i])) return false;
    }
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * nd = cgraph->nodes[i];
        if (!nd) continue;
        if (nd->op == GGML_OP_FLASH_ATTN_EXT) return false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (nd->src[s] && !reg_t(nd->src[s])) return false;
        }
        if (nd->name[0]) {
            if (std::strcmp(nd->name, "inp_target_features") == 0) feats_node = nd;
            else if (std::strcmp(nd->name, "f_rnn") == 0)          f_rnn_node = nd;
        }
        if (nd->op == GGML_OP_RMS_NORM && out.rms_eps == 1e-6f) {
            float eps = 0.0f;
            std::memcpy(&eps, nd->op_params, sizeof(float));
            if (eps > 0.0f) out.rms_eps = eps;
        }
    }

    if (max_blk < 0 || !out.layers[0].ctx_mix || !out.layers[0].wk_ctx || !feats_node) {
        return false;
    }
    out.n_layers = (uint32_t) (max_blk + 1);
    if (out.n_layers > 3u) return inj_fail("n_layers > 3");

    if (!f_rnn_node && out.markov_ctx_norm) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const struct ggml_tensor * nd = cgraph->nodes[i];
            if (nd && nd->op == GGML_OP_MUL &&
                (nd->src[0] == out.markov_ctx_norm || nd->src[1] == out.markov_ctx_norm)) {
                f_rnn_node = nd;
                break;
            }
        }
    }

    // Unwrap inp_target_features: CONT(PERMUTE(RESHAPE_3D(feats_raw, n_embd, n_target, n_tokens), 1, 0, 2, 3))
    if (feats_node->op != GGML_OP_CONT || !feats_node->src[0] || feats_node->src[0]->op != GGML_OP_PERMUTE) {
        return inj_fail("feats_node not CONT(PERMUTE)");
    }
    const struct ggml_tensor * perm = feats_node->src[0];
    if (ggml_get_op_params_i32(perm, 0) != 1 || ggml_get_op_params_i32(perm, 1) != 0 ||
        ggml_get_op_params_i32(perm, 2) != 2 || ggml_get_op_params_i32(perm, 3) != 3) {
        return inj_fail("perm axes mismatch");
    }
    out.n_target = (uint32_t) feats_node->ne[0];
    out.n_embd   = (uint32_t) feats_node->ne[1];
    out.n_tokens = (uint32_t) feats_node->ne[2];
    if (out.n_tokens == 0u || out.n_tokens > max_inject_m ||
        out.n_target == 0u || out.n_target > 16u ||
        out.n_embd == 0u || out.n_embd > 3072u || (out.n_embd % 4u) != 0u) {
        return inj_fail("dims out of range");
    }

    const struct ggml_tensor * feats_raw = w4_strip_views(perm->src[0]);
    if (!feats_raw || feats_raw->type != GGML_TYPE_F32) return inj_fail("feats_raw not F32");
    if (feats_raw->op == GGML_OP_GET_ROWS) {
        const struct ggml_tensor * src_tbl = feats_raw->src[0];
        const struct ggml_tensor * rows_id = feats_raw->src[1];
        if (!src_tbl || !rows_id || src_tbl->type != GGML_TYPE_F32 || rows_id->type != GGML_TYPE_I32 ||
            rows_id->nb[0] != sizeof(int32_t) ||
            src_tbl->nb[0] != sizeof(float) || (src_tbl->nb[1] % 16u) != 0u ||
            src_tbl->ne[0] != (int64_t) out.n_embd * out.n_target || rows_id->ne[0] != (int64_t) out.n_tokens) {
            return inj_fail("feats GET_ROWS shape/stride");
        }
        out.feat_src_va  = agx_tensor_gpu_va(src_tbl);
        out.embd_rows_va = agx_tensor_gpu_va(rows_id);
        out.feat_row_f32 = (uint32_t) (src_tbl->nb[1] / sizeof(float));
        if (!out.feat_src_va || !out.embd_rows_va) return inj_fail("feats GET_ROWS null va");
    } else if (feats_raw->op == GGML_OP_NONE) {
        if (feats_raw->nb[0] != sizeof(float) || (feats_raw->nb[1] % 16u) != 0u ||
            feats_raw->ne[0] != (int64_t) out.n_embd * out.n_target || feats_raw->ne[1] != (int64_t) out.n_tokens) {
            return inj_fail("feats NONE shape/stride");
        }
        out.feat_src_va  = agx_tensor_gpu_va(feats_raw);
        out.embd_rows_va = 0ull;
        out.feat_row_f32 = (uint32_t) (feats_raw->nb[1] / sizeof(float));
        if (!out.feat_src_va) return inj_fail("feats NONE null va");
    } else {
        return inj_fail("feats_raw unexpected op");
    }

    if (out.markov_ctx_mix || out.markov_ctx_norm || f_rnn_node) {
        if (!out.markov_ctx_mix || !out.markov_ctx_norm || !f_rnn_node ||
            out.markov_ctx_norm->type != GGML_TYPE_F32 ||
            f_rnn_node->type != GGML_TYPE_F32 || !ggml_is_contiguous(f_rnn_node) ||
            ggml_nelements(f_rnn_node) != (int64_t) out.n_embd * (int64_t) out.n_tokens) {
            return inj_fail("f_rnn shape/type");
        }
        uint32_t rb_tmp = 0;
        if (agx_inkling2_tensor_wtype(out.markov_ctx_mix, &rb_tmp) > 2u) return inj_fail("markov_ctx_mix wtype");
        out.has_f_rnn = true;
        out.f_rnn_va  = agx_tensor_gpu_va(f_rnn_node);
        if (!out.f_rnn_va) return inj_fail("f_rnn null va");
    }

    for (uint32_t il = 0u; il < out.n_layers; ++il) {
        const auto & L = out.layers[il];
        if (!L.ctx_mix || !L.ctx_norm || !L.wk_ctx || !L.k_ctx_norm) return inj_fail("layer weights missing");
        if (L.ctx_norm->type != GGML_TYPE_F32 || L.k_ctx_norm->type != GGML_TYPE_F32) return inj_fail("layer norm not F32");
        uint32_t rb_tmp = 0;
        if (agx_inkling2_tensor_wtype(L.ctx_mix, &rb_tmp) > 2u) return inj_fail("ctx_mix wtype");
        const uint32_t hd = (uint32_t) L.k_ctx_norm->ne[0];
        if (hd == 0u || hd > 256u || (hd % 32u) != 0u || (L.wk_ctx->ne[1] % hd) != 0) return inj_fail("head_dim invalid");
        const uint32_t hkv = (uint32_t) (L.wk_ctx->ne[1] / hd);
        if (L.wv_ctx && L.wv_ctx->ne[1] != L.wk_ctx->ne[1]) return inj_fail("wv_ctx rows mismatch");
        if (il == 0u) {
            out.head_dim  = hd;
            out.n_head_kv = hkv;
        } else if (hd != out.head_dim || hkv != out.n_head_kv) {
            return inj_fail("layer head mismatch");
        }
    }

    // Trace all SET_ROWS nodes back to their originating blk.<il>.attn_{k,v}_ctx.weight!
    const int64_t kv_dim = (int64_t) out.n_head_kv * out.head_dim;
    bool have_global_rope = false;
    uint32_t total_set_rows = 0u;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * nd = cgraph->nodes[i];
        if (!nd) continue;
        if ((nd->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            if (nd->op != GGML_OP_SET_ROWS && w4_strip_views(nd) != f_rnn_node) {
                return inj_fail("unexpected OUTPUT node");
            }
        }
        if (nd->op != GGML_OP_SET_ROWS) continue;
        ++total_set_rows;
        if (!nd->src[0] || !nd->src[1] || nd->ne[0] != kv_dim) return inj_fail("SET_ROWS ne0 mismatch");
        if (nd->type != GGML_TYPE_F16 && nd->type != GGML_TYPE_Q8_0) return inj_fail("SET_ROWS dst type");
        if (nd->src[1]->type != GGML_TYPE_I64 && nd->src[1]->type != GGML_TYPE_I32) return inj_fail("SET_ROWS idxs type");

        bool has_rot = false;
        uint32_t rot_dim = 0u;
        const struct ggml_tensor * rope_nd = nullptr;
        const struct ggml_tensor * t = w4_strip_views(nd->src[0]);
        if (t && t->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(t, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
            if (out.head_dim != 128u) return inj_fail("FWHT head_dim != 128");
            rot_dim = (t->src[0]) ? (uint32_t) t->src[0]->ne[0] : 128u;
            if (rot_dim != 64u && rot_dim != 128u) return inj_fail("FWHT rot_dim");
            has_rot = true;
            t = w4_strip_views(t->src[1]);
        }
        if (t && t->op == GGML_OP_ROPE) {
            rope_nd = t;
            t = w4_strip_views(t->src[0]);
        }
        if (t && t->op == GGML_OP_MUL) {
            const struct ggml_tensor * r0 = w4_strip_views(t->src[0]);
            const struct ggml_tensor * r1 = w4_strip_views(t->src[1]);
            if (r0 && r0->op == GGML_OP_RMS_NORM)      t = w4_strip_views(r0->src[0]);
            else if (r1 && r1->op == GGML_OP_RMS_NORM) t = w4_strip_views(r1->src[0]);
        } else if (t && t->op == GGML_OP_RMS_NORM) {
            t = w4_strip_views(t->src[0]);
        }
        if (!t || t->op != GGML_OP_MUL_MAT || !t->src[0]) return inj_fail("SET_ROWS trace not MUL_MAT");

        int matched_il = -1;
        bool is_k = false;
        for (uint32_t il = 0u; il < out.n_layers; ++il) {
            if (t->src[0] == out.layers[il].wk_ctx && !out.layers[il].k_store_va) {
                matched_il = (int) il;
                is_k = true;
                break;
            }
            if (out.layers[il].wv_ctx && t->src[0] == out.layers[il].wv_ctx && !out.layers[il].v_store_va) {
                matched_il = (int) il;
                is_k = false;
                break;
            }
            if (!out.layers[il].wv_ctx && t->src[0] == out.layers[il].wk_ctx && !out.layers[il].v_store_va) {
                matched_il = (int) il;
                is_k = false;
                break;
            }
        }
        if (matched_il < 0) return inj_fail("SET_ROWS unmatched layer");

        auto & L = out.layers[matched_il];
        const bool is_q8 = (nd->type == GGML_TYPE_Q8_0);
        if (is_k) {
            L.k_store_va   = agx_tensor_gpu_va(nd);
            L.k_store_nb1  = (uint32_t) nd->nb[1];
            L.k_idxs_va    = agx_tensor_gpu_va(nd->src[1]);
            L.k_ids_is_i64 = (nd->src[1]->type == GGML_TYPE_I64);
            L.k_ids_nb0    = (uint32_t) nd->src[1]->nb[0];
            L.kv_is_q8     = is_q8;
            L.has_k_rot    = has_rot;
            L.k_rot_dim    = rot_dim;
            if (rope_nd) {
                const uint32_t mode = (uint32_t) ggml_get_op_params_i32(rope_nd, 2);
                if (mode & 8u) return inj_fail("rope mode & 8");
                const uint32_t rdims = (uint32_t) ggml_get_op_params_i32(rope_nd, 1);
                float freq_base = 10000.0f, freq_scale = 1.0f, attn_factor = 1.0f;
                std::memcpy(&freq_base,   (const char *) rope_nd->op_params + 20, sizeof(float));
                std::memcpy(&freq_scale,  (const char *) rope_nd->op_params + 24, sizeof(float));
                std::memcpy(&attn_factor, (const char *) rope_nd->op_params + 32, sizeof(float));
                const float    t_sc   = std::pow(freq_base, -2.0f / (float) rdims);
                const uint64_t rf_va  = rope_nd->src[2] ? agx_tensor_gpu_va(rope_nd->src[2]) : 0ull;
                const uint64_t pos_va = (rope_nd->src[1] && rope_nd->src[1]->nb[0] == sizeof(int32_t))
                    ? agx_tensor_gpu_va(rope_nd->src[1]) : 0ull;
                if (!pos_va) return inj_fail("rope pos_va invalid");
                L.has_rope = true;
                if (!have_global_rope) {
                    have_global_rope = true;
                    out.rope_n_dims  = rdims;
                    out.rope_mode    = mode;
                    out.theta_scale  = t_sc;
                    out.freq_scale   = freq_scale;
                    out.attn_factor  = attn_factor;
                    out.rope_freq_va = rf_va;
                    out.inp_pos_va   = pos_va;
                } else if (out.rope_n_dims != rdims || out.rope_mode != mode ||
                           out.rope_freq_va != rf_va || out.inp_pos_va != pos_va) {
                    return inj_fail("rope global mismatch");
                }
            }
        } else {
            L.v_store_va   = agx_tensor_gpu_va(nd);
            L.v_store_nb1  = (uint32_t) nd->nb[1];
            L.v_idxs_va    = agx_tensor_gpu_va(nd->src[1]);
            L.v_ids_is_i64 = (nd->src[1]->type == GGML_TYPE_I64);
            L.v_ids_nb0    = (uint32_t) nd->src[1]->nb[0];
            L.has_v_rot    = has_rot;
            L.v_rot_dim    = rot_dim;
            if (L.k_store_va && L.kv_is_q8 != is_q8) return inj_fail("K/V dtype mismatch");
            L.kv_is_q8     = is_q8;
        }
    }

    if (total_set_rows != 2u * out.n_layers) return inj_fail("total_set_rows != 2*n_layers");
    for (uint32_t il = 0u; il < out.n_layers; ++il) {
        const auto & L = out.layers[il];
        if (!L.k_store_va || !L.v_store_va || !L.k_idxs_va || !L.v_idxs_va) return inj_fail("layer store/idxs missing");
        if (il > 0u && (L.k_ids_nb0 != out.layers[0].k_ids_nb0 || L.v_ids_nb0 != out.layers[0].v_ids_nb0)) {
            return inj_fail("layer ids_nb0 mismatch");
        }
    }
    return true;
}

static void ensure_w8_dspark_inject_bound(agx_backend_context *          ctx,
                                          agx_inkling2_engine *          eng,
                                          const w8_dspark_inject_graph & G) {
    if (!ctx || !eng || !eng->k_w8_dspark_prep || G.n_layers == 0u) return;
    auto & B = eng->dspark_bound;
    const uint64_t cur_wk0 = G.layers[0].wk_ctx ? agx_tensor_gpu_va(G.layers[0].wk_ctx) : 0ull;
    if (B.inject_bound && B.wk_ctx_va[0] == cur_wk0) {
        return;
    }

    for (uint32_t i = 0; i < AGX_W7_DSPARK_MAX_LAYERS; ++i) {
        if (B.w_kv_ctx_bo[i]) {
            agx_bo_free(ctx->drm, B.w_kv_ctx_bo[i]);
            B.w_kv_ctx_bo[i] = nullptr;
        }
        B.wk_ctx_va[i]     = 0ull;
        B.w_kv_ctx_rows[i] = 0u;
    }
    B.inject_bound = true;

    const uint32_t D = G.n_embd;
    if ((D % 256u) != 0u) return;
    const uint32_t row_bytes  = (D / 256u) * AGX_Q8_0_CHUNK_BYTES;
    const uint32_t row_uvec4s = row_bytes / 16u;

    struct prep_job {
        agx_w8_dspark_prep_push pc;
        uint32_t                grid_x;
    };
    std::vector<prep_job> jobs;

    for (uint32_t il = 0u; il < G.n_layers; ++il) {
        const auto & L = G.layers[il];
        B.wk_ctx_va[il] = L.wk_ctx ? agx_tensor_gpu_va(L.wk_ctx) : 0ull;
        const bool ok = L.wk_ctx && L.wv_ctx &&
                        L.wk_ctx->type == GGML_TYPE_Q8_0 && agx_tensor_is_repacked(L.wk_ctx) &&
                        L.wv_ctx->type == GGML_TYPE_Q8_0 && agx_tensor_is_repacked(L.wv_ctx);
        if (!ok) continue;
        const uint32_t n0  = (uint32_t) L.wk_ctx->ne[1];
        const uint32_t n1  = (uint32_t) L.wv_ctx->ne[1];
        const uint32_t tot = n0 + n1;
        const size_t   sz  = agx_align_up((size_t) tot * row_bytes, AGX_CACHE_LINE_BYTES);
        // Read only 1x/round -> AGX_BO_SLC_STREAM so it NEVER evicts w2_q4_bo from 48 MiB SLC!
        B.w_kv_ctx_bo[il] = agx_bo_alloc(ctx->drm, sz, AGX_BO_SLC_STREAM, "dspark-wkv-ctx-coal");
        if (B.w_kv_ctx_bo[il]) {
            B.w_kv_ctx_rows[il] = tot;
            prep_job jb{};
            jb.pc.src0_va    = agx_tensor_gpu_va(L.wk_ctx);
            jb.pc.src1_va    = agx_tensor_gpu_va(L.wv_ctx);
            jb.pc.dst_va     = B.w_kv_ctx_bo[il]->gpu_va;
            jb.pc.op         = 1u;
            jb.pc.n0         = n0;
            jb.pc.n1         = n1;
            jb.pc.row_uvec4s = row_uvec4s;
            jb.grid_x        = (tot * row_uvec4s + 255u) / 256u;
            jobs.push_back(jb);
        }
    }

    if (!jobs.empty()) {
        agx_cdm_stream_reset(ctx->stream);
        bool ok = true;
        for (size_t j = 0; j < jobs.size(); ++j) {
            const agx_barrier_mode bar = (j + 1u == jobs.size()) ? AGX_BARRIER_MEMORY_HOST : AGX_BARRIER_NONE;
            if (!agx_cdm_dispatch(ctx->stream, eng->k_w8_dspark_prep, jobs[j].grid_x, 1u, 1u,
                                  &jobs[j].pc, sizeof(jobs[j].pc), bar)) {
                ok = false;
                break;
            }
        }
        if (!ok || agx_cdm_submit(ctx->stream, true, nullptr) != 0) {
            for (uint32_t i = 0; i < AGX_W7_DSPARK_MAX_LAYERS; ++i) {
                if (B.w_kv_ctx_bo[i]) {
                    agx_bo_free(ctx->drm, B.w_kv_ctx_bo[i]);
                    B.w_kv_ctx_bo[i] = nullptr;
                }
                B.w_kv_ctx_rows[i] = 0u;
            }
        }
    }
}

static bool execute_dspark_inject_cgraph(agx_backend_context *          ctx,
                                         const w8_dspark_inject_graph & G) {
    const auto t_enc_beg = std::chrono::steady_clock::now();
    agx_inkling2_engine * eng = agx_graph_exec_get_inkling2(ctx);
    if (!eng || !agx_inkling2_precompile_w4(eng) || !agx_gemm_init(ctx)) return false;

    const uint32_t M    = G.n_tokens;
    const uint32_t D    = G.n_embd;
    const uint32_t Hkv  = G.n_head_kv;
    const uint32_t hd   = G.head_dim;
    const uint32_t kv_d = Hkv * hd;

    // Wave 11: Keep M <= 64 decode/verify 100% single-chunk untouched; for large-M prefill (M > 64),
    // chunk into SLC-resident slices (default M_chunk=512 -> ~25.5 MiB SLC_PIN scratch) inside ONE CDM stream!
    uint32_t M_chunk = M;
    if (M > 64u) {
        uint32_t cfg_chunk = 512u;
        if (const char * cenv = std::getenv("GGML_AGX_DSPARK_INJECT_CHUNK")) {
            const int v = std::atoi(cenv);
            if (v > 0) cfg_chunk = std::clamp<uint32_t>((uint32_t) v, 64u, 4096u);
        }
        M_chunk = std::min<uint32_t>(M, cfg_chunk);
    }
    const uint32_t M_cap = std::max<uint32_t>(M_chunk, 8u);

    const char * no_mix14_env = std::getenv("GGML_AGX_DSPARK_NO_INJECT_MIX14");
    const bool   no_mix14     = (no_mix14_env && no_mix14_env[0] && no_mix14_env[0] != '0');

    const size_t sz_cur_l  = agx_align_up((size_t) M_cap * D * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t sz_kv_one = agx_align_up((size_t) M_cap * kv_d * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t sz_kv_l   = 2u * sz_kv_one;
    const size_t need_sz   = std::max<size_t>(G.n_layers * (sz_cur_l + sz_kv_l), 8u * 1024u * 1024u);

    if (!eng->dspark_scratch_bo || eng->dspark_scratch_bo->size < need_sz) {
        if (eng->dspark_scratch_bo) {
            agx_bo_free(ctx->drm, eng->dspark_scratch_bo);
            eng->dspark_scratch_bo = nullptr;
        }
        const uint32_t bo_flags = (need_sz <= 32u * 1024u * 1024u)
            ? (AGX_BO_SLC_PIN | AGX_BO_WRITEBACK)
            : (AGX_BO_SLC_STREAM | AGX_BO_WRITEBACK);
        eng->dspark_scratch_bo = agx_bo_alloc(ctx->drm, need_sz, bo_flags, "dspark-draft-scratch");
        if (!eng->dspark_scratch_bo) return false;
    }

    ensure_w8_dspark_inject_bound(ctx, eng, G);

    bool use_coal_kv = ((D % 256u) == 0u);
    for (uint32_t il = 0u; il < G.n_layers; ++il) {
        if (!eng->dspark_bound.w_kv_ctx_bo[il]) use_coal_kv = false;
    }

    const uint64_t sb        = eng->dspark_scratch_bo->gpu_va;
    const uint64_t va_cur_l0 = sb;
    const uint64_t va_kv_l0  = sb + (uint64_t) G.n_layers * sz_cur_l;

    agx_cdm_stream * stream = ctx->stream;
    agx_cdm_stream_reset(stream);

    agx_cdm_stats cdm_before{};
    const char * stats_env  = std::getenv("GGML_AGX_STATS");
    const bool   emit_stats = (M > 64u && stats_env && stats_env[0] && stats_env[0] != '0');
    if (emit_stats) {
        agx_cdm_get_stats(stream, &cdm_before);
    }

    for (uint32_t m0 = 0u; m0 < M; m0 += M_chunk) {
        const uint32_t mc      = std::min<uint32_t>(M_chunk, M - m0);
        const bool     is_last = (m0 + mc >= M);

        // Stage 1 (1 dispatch): Gather target features + all [f_rnn, cur_0..L-1] ctx_mix + RMSNorm*w!
        {
            agx_w8_dspark_inject_mix_push mpc{};
            if (G.embd_rows_va != 0ull) {
                mpc.feat_src_va  = G.feat_src_va;
                mpc.embd_rows_va = G.embd_rows_va + (uint64_t) m0 * sizeof(int32_t);
            } else {
                mpc.feat_src_va  = G.feat_src_va + (uint64_t) m0 * (uint64_t) G.feat_row_f32 * sizeof(float);
                mpc.embd_rows_va = 0ull;
            }
            mpc.n_embd       = D;
            mpc.n_target     = G.n_target;
            mpc.n_tokens     = mc;
            mpc.feat_row_f32 = G.feat_row_f32;
            mpc.rms_eps      = G.rms_eps;

            uint32_t slot   = 0u;
            uint32_t wtypes = 0u;
            uint32_t rb_tmp = 0u;
            if (G.has_f_rnn) {
                mpc.mix_w_va[slot]  = agx_tensor_gpu_va(G.markov_ctx_mix);
                mpc.norm_w_va[slot] = agx_tensor_gpu_va(G.markov_ctx_norm);
                mpc.out_va[slot]    = G.f_rnn_va + (uint64_t) m0 * (uint64_t) D * sizeof(float);
                wtypes |= (agx_inkling2_tensor_wtype(G.markov_ctx_mix, &rb_tmp) & 3u) << (slot * 2u);
                ++slot;
            }
            for (uint32_t il = 0u; il < G.n_layers; ++il) {
                mpc.mix_w_va[slot]  = agx_tensor_gpu_va(G.layers[il].ctx_mix);
                mpc.norm_w_va[slot] = agx_tensor_gpu_va(G.layers[il].ctx_norm);
                mpc.out_va[slot]    = va_cur_l0 + (uint64_t) il * sz_cur_l;
                wtypes |= (agx_inkling2_tensor_wtype(G.layers[il].ctx_mix, &rb_tmp) & 3u) << (slot * 2u);
                ++slot;
            }
            mpc.n_slots    = slot;
            mpc.mix_wtypes = wtypes;

            const bool use_mix14 = (!no_mix14 && mc > 32u && D <= 3072u && slot <= 4u &&
                                    eng->k_w11_dspark_inject_mix_fused != nullptr);
            if (use_mix14) {
                if (!agx_cdm_dispatch(stream, eng->k_w11_dspark_inject_mix_fused, mc, 1u, 1u,
                                      &mpc, sizeof(mpc), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
            } else {
                if (!agx_cdm_dispatch(stream, eng->k_w8_dspark_inject_mix, slot, mc, 1u,
                                      &mpc, sizeof(mpc), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
            }
        }

        // Stage 2 (3 concurrent Pillar-2 Q8_0 dispatches, 0 inter-layer barriers): [wk_ctx; wv_ctx] @ cur_il!
        for (uint32_t il = 0u; il < G.n_layers; ++il) {
            const auto & L          = G.layers[il];
            const uint64_t cur_va   = va_cur_l0 + (uint64_t) il * sz_cur_l;
            const uint64_t kv_il_va = va_kv_l0  + (uint64_t) il * sz_kv_l;
            const agx_barrier_mode last_bar = (il + 1u == G.n_layers) ? AGX_BARRIER_STRONG_USC : AGX_BARRIER_NONE;
            if (use_coal_kv) {
                const uint32_t tot_rows = eng->dspark_bound.w_kv_ctx_rows[il];
                if (!agx_gemm_dispatch_raw_q8_0(ctx, stream, eng->dspark_bound.w_kv_ctx_bo[il]->gpu_va,
                                                cur_va, kv_il_va, mc, tot_rows, D, last_bar, D, tot_rows)) {
                    return false;
                }
            } else {
                if (!dispatch_dense_mm(ctx, stream, L.wk_ctx, cur_va, kv_il_va, mc,
                                       L.wv_ctx ? AGX_BARRIER_NONE : last_bar)) {
                    return false;
                }
                if (L.wv_ctx) {
                    if (!dispatch_dense_mm(ctx, stream, L.wv_ctx, cur_va, kv_il_va + sz_kv_one, mc, last_bar)) {
                        return false;
                    }
                }
            }
        }

        // Stage 3 (1 dispatch across all layers!): k_ctx_norm + RoPE + [FWHT128] + K/V SET_ROWS!
        {
            agx_w8_dspark_inject_kv_push kpc{};
            kpc.k_raw_base_va        = va_kv_l0;
            kpc.v_raw_base_va        = use_coal_kv
                ? (va_kv_l0 + (uint64_t) kv_d * sizeof(float))
                : (va_kv_l0 + (G.layers[0].wv_ctx ? sz_kv_one : 0u));
            kpc.inp_pos_va           = G.inp_pos_va ? (G.inp_pos_va + (uint64_t) m0 * sizeof(int32_t)) : 0ull;
            kpc.rope_freq_va         = G.rope_freq_va;
            kpc.n_layers             = G.n_layers;
            kpc.n_head_kv            = Hkv;
            kpc.head_dim             = hd;
            kpc.n_tokens             = mc;
            kpc.layer_raw_stride_f32 = (uint32_t) (sz_kv_l / sizeof(float));
            kpc.k_raw_stride_f32     = use_coal_kv ? (2u * kv_d) : kv_d;
            kpc.v_raw_stride_f32     = use_coal_kv ? (2u * kv_d) : kv_d;
            kpc.k_ids_nb0_B          = G.layers[0].k_ids_nb0;
            kpc.v_ids_nb0_B          = G.layers[0].v_ids_nb0;
            kpc.rope_n_dims          = G.rope_n_dims;
            kpc.rms_eps              = G.rms_eps;
            kpc.theta_scale          = G.theta_scale;
            kpc.freq_scale           = G.freq_scale;
            kpc.attn_factor          = G.attn_factor;
            for (uint32_t il = 0u; il < G.n_layers; ++il) {
                const auto & L        = G.layers[il];
                kpc.k_norm_va[il]     = agx_tensor_gpu_va(L.k_ctx_norm);
                kpc.k_store_va[il]    = L.k_store_va;
                kpc.v_store_va[il]    = L.v_store_va;
                kpc.k_idxs_va[il]     = L.k_idxs_va + (uint64_t) m0 * (uint64_t) L.k_ids_nb0;
                kpc.v_idxs_va[il]     = L.v_idxs_va + (uint64_t) m0 * (uint64_t) L.v_ids_nb0;
                kpc.k_store_nb1_B[il] = L.k_store_nb1;
                kpc.v_store_nb1_B[il] = L.v_store_nb1;
                kpc.layer_flags[il]   = (!L.wv_ctx ? AGX_W7_DSPARK_QKV_SHARED_KV : 0u) |
                                        (L.has_rope ? AGX_W7_DSPARK_QKV_HAS_ROPE : 0u) |
                                        ((G.rope_mode & 2u) ? AGX_W7_DSPARK_QKV_ROPE_NEOX : 0u) |
                                        (L.k_rot_dim == 128u ? AGX_W7_DSPARK_QKV_FWHT_K : 0u) |
                                        (L.k_rot_dim ==  64u ? AGX_W7_DSPARK_QKV_FWHT64_K : 0u) |
                                        (L.v_rot_dim == 128u ? AGX_W7_DSPARK_QKV_FWHT_V : 0u) |
                                        (L.v_rot_dim ==  64u ? AGX_W7_DSPARK_QKV_FWHT64_V : 0u) |
                                        (L.kv_is_q8 ? AGX_W7_DSPARK_QKV_KV_Q8_0 : 0u) |
                                        (L.k_ids_is_i64 ? AGX_W7_DSPARK_QKV_K_IDS_I64 : 0u) |
                                        (L.v_ids_is_i64 ? AGX_W7_DSPARK_QKV_V_IDS_I64 : 0u);
            }
            const agx_barrier_mode s3_bar = is_last ? AGX_BARRIER_MEMORY_HOST : AGX_BARRIER_STRONG_USC;
            if (!agx_cdm_dispatch(stream, eng->k_w8_dspark_inject_kv, 2u * Hkv, mc, G.n_layers,
                                  &kpc, sizeof(kpc), s3_bar)) {
                return false;
            }
        }
    }

    const auto t_enc_end = std::chrono::steady_clock::now();
    const uint64_t cpu_enc_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t_enc_end - t_enc_beg).count();

    const char * async_inj_env = std::getenv("GGML_AGX_ASYNC_INJECT");
    const bool   async_inject  = (async_inj_env && async_inj_env[0] && async_inj_env[0] != '0');
    uint64_t gpu_ns = 0;
    if (agx_cdm_submit(stream, !async_inject, (!async_inject && emit_stats) ? &gpu_ns : nullptr) != 0) {
        AGX_LOG_ERROR("execute_dspark_inject_cgraph: CDM submit failed");
        return false;
    }

    if (emit_stats) {
        agx_cdm_stats cdm_after{};
        agx_cdm_get_stats(stream, &cdm_after);
        std::fprintf(stderr,
                     "[ggml-agx DSPARK INJECT STATS] M=%u chunk=%u layers=%u coal=%d mix14=%d "
                     "dispatches=%llu usc_bar=%llu cpu_enc=%.2fms gpu=%.2fms\n",
                     M, M_chunk, G.n_layers, use_coal_kv ? 1 : 0, (!no_mix14 && M_chunk > 32u && D <= 3072u) ? 1 : 0,
                     (unsigned long long) (cdm_after.dispatches          - cdm_before.dispatches),
                     (unsigned long long) (cdm_after.strong_usc_barriers - cdm_before.strong_usc_barriers),
                     (double) cpu_enc_ns * 1e-6, (double) gpu_ns * 1e-6);
    }
    return true;
}

static bool execute_dspark_draft_cgraph(agx_backend_context *         ctx,
                                        const w7_dspark_draft_graph & G) {
    agx_inkling2_engine * eng = agx_graph_exec_get_inkling2(ctx);
    if (!eng || !agx_inkling2_precompile_w4(eng)) return false;
    agx_slc_arena * slc = agx_backend_get_slc_arena(ctx);
    if (!slc || !agx_gemm_init(ctx)) return false;
    if (!ctx->kv_hybrid_state) {
        agx_kv_hybrid_config kcfg;
        kcfg.n_slots = 48;
        kcfg.max_full_tokens = 4096;
        ctx->kv_hybrid_state = agx_kv_hybrid_create(ctx->drm, ctx->compiler, kcfg);
        if (!ctx->kv_hybrid_state) return false;
    }

    const uint32_t M     = G.n_tokens;
    const uint32_t M_cap = std::max<uint32_t>(M, 8u);
    const uint32_t D     = G.n_embd;
    const uint32_t Hq    = G.n_head_q;
    const uint32_t Hkv   = G.n_head_kv;
    const uint32_t hd    = G.head_dim;
    const uint32_t F     = G.n_ff;

    // Partition dedicated scratch BO for DSpark draft activations (sized to >= 8 MiB so bo_epoch never bumps!)
    const size_t sz_embd     = agx_align_up((size_t) M_cap * D * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t sz_embd_f16 = agx_align_up((size_t) M_cap * D * sizeof(uint16_t), AGX_CACHE_LINE_BYTES);
    const size_t sz_q        = agx_align_up((size_t) M_cap * Hq * hd * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t sz_kv       = agx_align_up((size_t) M_cap * Hkv * hd * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t sz_qkv      = agx_align_up((size_t) M_cap * (Hq + 2u * Hkv) * hd * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t sz_gu       = agx_align_up((size_t) M_cap * 2u * F * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t sz_inter    = agx_align_up((size_t) M_cap * F * sizeof(float), AGX_CACHE_LINE_BYTES);
    const size_t sz_inter_f16= agx_align_up((size_t) M_cap * F * sizeof(uint16_t), AGX_CACHE_LINE_BYTES);
    const size_t sz_lm_pad   = G.lm_pad_aliases_out
        ? agx_align_up((size_t) M * G.n_vocab_padded * sizeof(float), AGX_CACHE_LINE_BYTES) : 0u;

    size_t cur_off = 0;
    auto take_off = [&](size_t sz) -> size_t {
        size_t o = cur_off;
        cur_off += agx_align_up(sz, AGX_CACHE_LINE_BYTES);
        return o;
    };
    const size_t o_cur_a     = take_off(sz_embd);
    const size_t o_cur_b     = take_off(sz_embd);
    const size_t o_norm      = take_off(sz_embd);
    const size_t o_norm_f16  = take_off(sz_embd_f16);
    const size_t o_wo_out    = take_off(sz_embd);
    const size_t o_qkv_raw   = take_off(sz_qkv);
    const size_t o_k_raw_fb  = take_off(sz_kv);
    const size_t o_v_raw_fb  = take_off(sz_kv);
    const size_t o_q_out     = take_off(sz_q);
    const size_t o_attn_out  = take_off(sz_q);
    const size_t o_gu_out    = take_off(sz_gu);
    const size_t o_inter     = take_off(sz_inter);
    const size_t o_inter_f16 = take_off(sz_inter_f16);
    const size_t o_lm_pad    = sz_lm_pad ? take_off(sz_lm_pad) : 0u;
    const size_t alloc_sz    = std::max<size_t>(cur_off, 8u * 1024u * 1024u);

    if (!eng->dspark_scratch_bo || eng->dspark_scratch_bo->size < alloc_sz) {
        if (eng->dspark_scratch_bo) {
            agx_bo_free(ctx->drm, eng->dspark_scratch_bo);
            eng->dspark_scratch_bo = nullptr;
        }
        eng->dspark_scratch_bo = agx_bo_alloc(ctx->drm, alloc_sz,
                                              AGX_BO_SLC_PIN | AGX_BO_WRITEBACK,
                                              "dspark-draft-scratch");
        if (!eng->dspark_scratch_bo) return false;
    }

    // Bind SLC-resident Q4_32 markov_w2 (28.81 MB < 48 MiB SLC) + coalesced [wq; wk; wv] once on round 1!
    ensure_w8_dspark_draft_bound(ctx, eng, G);

    const uint64_t sb          = eng->dspark_scratch_bo->gpu_va;
    const uint64_t va_cur_a    = sb + o_cur_a;
    const uint64_t va_cur_b    = sb + o_cur_b;
    const uint64_t va_norm     = sb + o_norm;
    const uint64_t va_norm_f16 = sb + o_norm_f16;
    const uint64_t va_wo_out   = sb + o_wo_out;
    const uint64_t va_qkv_raw  = sb + o_qkv_raw;
    const uint64_t va_k_raw_fb = sb + o_k_raw_fb;
    const uint64_t va_v_raw_fb = sb + o_v_raw_fb;
    const uint64_t va_q_out    = sb + o_q_out;
    const uint64_t va_attn_out = sb + o_attn_out;
    const uint64_t va_gu_out   = sb + o_gu_out;
    const uint64_t va_inter    = sb + o_inter;
    const uint64_t va_inter_f16= sb + o_inter_f16;
    const uint64_t va_lm_pad   = G.lm_pad_aliases_out ? (sb + o_lm_pad) : G.lm_head_pad_va;

    const char * leg_q8_env    = std::getenv("GGML_AGX_DSPARK_LEGACY_Q8");
    const bool   legacy_q8     = (leg_q8_env && leg_q8_env[0] && leg_q8_env[0] != '0');
    const char * q8_m8_env     = std::getenv("GGML_AGX_DSPARK_Q8_M8");
    const bool   force_q8_m8   = (q8_m8_env && q8_m8_env[0] && q8_m8_env[0] != '0');
    const char * no_cqkv_env   = std::getenv("GGML_AGX_DSPARK_NO_COAL_QKV");
    const bool   no_coal_qkv   = (no_cqkv_env && no_cqkv_env[0] && no_cqkv_env[0] != '0');

    auto * kv_pool = static_cast<agx_kv_hybrid_pool *>(ctx->kv_hybrid_state);
    agx_cdm_stream * stream = ctx->stream;

    auto try_w9_q8_gemv = [&](uint64_t w_va, uint64_t x_f32_va, uint64_t x_f16_va,
                              uint64_t y_va, uint32_t N, uint32_t K, uint32_t y_str,
                              bool allow_m8, agx_barrier_mode bar) -> bool {
        if (legacy_q8 || (K % 256u) != 0u || M == 0u || M > 8u) return false;
        if (M > 4u && !allow_m8 && !force_q8_m8) return false;
        const bool use_f16 = (x_f16_va != 0ull);
        const agx_kernel * k = (M <= 4u)
            ? (use_f16 ? eng->k_w9_q8_gemv_m4_f16 : eng->k_w9_q8_gemv_m4_f32)
            : (use_f16 ? eng->k_w9_q8_gemv_m8_f16 : eng->k_w9_q8_gemv_m8_f32);
        if (!k) return false;
        agx_w9_q8_gemv_m_push pc{};
        pc.w_va     = w_va;
        pc.x_va     = use_f16 ? x_f16_va : x_f32_va;
        pc.y_va     = y_va;
        pc.M        = M;
        pc.N        = N;
        pc.K        = K;
        pc.x_stride = K;
        pc.y_stride = y_str;
        return agx_cdm_dispatch(stream, k, (N + 7u) / 8u, 1u, 1u, &pc, sizeof(pc), bar);
    };

    const auto t_enc_beg = std::chrono::steady_clock::now();
    agx_cdm_stream_reset(stream);
    agx_cdm_stats cdm_before{};
    agx_cdm_get_stats(stream, &cdm_before);

    // Step 0: Fused Input Block (anchor embd + tok_norm + mask_embd + scale + blk.0.attn_norm) -> va_cur_a, va_norm & va_norm_f16
    {
        uint32_t emb_rb = 0, msk_rb = 0;
        agx_w7_dspark_inp_block_push ipc{};
        ipc.tok_embd_va     = agx_tensor_gpu_va(G.tok_embd);
        ipc.tok_ids_va      = G.inp_tokens_va;
        ipc.tok_norm_va     = G.tok_norm ? agx_tensor_gpu_va(G.tok_norm) : 0ull;
        ipc.mask_embd_va    = G.mask_embd ? agx_tensor_gpu_va(G.mask_embd) : 0ull;
        ipc.attn_norm0_va   = agx_tensor_gpu_va(G.layers[0].attn_norm);
        ipc.out_cur_va      = va_cur_a;
        ipc.out_norm_va     = va_norm;
        ipc.out_norm_f16_va = va_norm_f16;
        ipc.n_embd          = D;
        ipc.n_tokens        = M;
        ipc.block_drafts    = G.block_drafts;
        ipc.embd_wtype      = agx_inkling2_tensor_wtype(G.tok_embd, &emb_rb);
        ipc.embd_row_bytes  = emb_rb;
        ipc.mask_wtype      = G.mask_embd ? agx_inkling2_tensor_wtype(G.mask_embd, &msk_rb) : 0u;
        ipc.mask_row_bytes  = msk_rb;
        ipc.embd_scale      = G.embd_scale;
        ipc.rms_eps         = G.rms_eps;
        if (!agx_cdm_dispatch(stream, eng->k_w7_dspark_inp_block, M, 1u, 1u,
                              &ipc, sizeof(ipc), AGX_BARRIER_STRONG_USC)) {
            return false;
        }
    }

    uint64_t inp_l_va = va_cur_a;
    uint64_t alt_l_va = va_cur_b;

    // Step 1: 3-Layer Fused DSpark Draft Body
    for (uint32_t il = 0u; il < G.n_layers; ++il) {
        const auto & L = G.layers[il];

        // 1a. Wave 9 Coalesced [wq; wk; wv] 1-dispatch direct-L1 FP16 Q8 GEMV (or fallback)
        const bool use_coal_qkv = !no_coal_qkv && (D % 256u) == 0u &&
                                  il < AGX_W7_DSPARK_MAX_LAYERS &&
                                  eng->dspark_bound.w_qkv_bo[il] != nullptr &&
                                  eng->dspark_bound.wq_src_va[il] == agx_tensor_gpu_va(L.wq);
        uint64_t q_raw_va = va_qkv_raw;
        uint64_t k_raw_va = va_k_raw_fb;
        uint64_t v_raw_va = L.wv ? va_v_raw_fb : k_raw_va;
        uint32_t q_str = Hq * hd, k_str = Hkv * hd, v_str = Hkv * hd;
        if (use_coal_qkv) {
            const uint32_t tot_rows = eng->dspark_bound.w_qkv_rows[il];
            q_raw_va = va_qkv_raw;
            k_raw_va = va_qkv_raw + (uint64_t) (Hq * hd) * sizeof(float);
            v_raw_va = L.wv ? (k_raw_va + (uint64_t) (Hkv * hd) * sizeof(float)) : k_raw_va;
            q_str = k_str = v_str = tot_rows;
            if (!try_w9_q8_gemv(eng->dspark_bound.w_qkv_bo[il]->gpu_va, va_norm, va_norm_f16,
                                va_qkv_raw, tot_rows, D, tot_rows, false, AGX_BARRIER_STRONG_USC)) {
                if (!agx_gemm_dispatch_raw_q8_0(ctx, stream, eng->dspark_bound.w_qkv_bo[il]->gpu_va,
                                                va_norm, va_qkv_raw, M, tot_rows, D,
                                                AGX_BARRIER_STRONG_USC, D, tot_rows)) {
                    return false;
                }
            }
        } else {
            if (!dispatch_dense_mm(ctx, stream, L.wq, va_norm, q_raw_va, M, AGX_BARRIER_NONE)) return false;
            if (!dispatch_dense_mm(ctx, stream, L.wk, va_norm, k_raw_va, M, L.wv ? AGX_BARRIER_NONE : AGX_BARRIER_STRONG_USC)) return false;
            if (L.wv) {
                if (!dispatch_dense_mm(ctx, stream, L.wv, va_norm, v_raw_va, M, AGX_BARRIER_STRONG_USC)) return false;
            }
        }

        // 1b. Fused 1-SG/head Q/K RMSNorm + RoPE + [FWHT128] + KV-Store (F16/Q8_0)
        {
            agx_w7_dspark_qkv_rope_push qpc{};
            qpc.q_raw_va         = q_raw_va;
            qpc.k_raw_va         = k_raw_va;
            qpc.v_raw_va         = v_raw_va;
            qpc.q_norm_va        = agx_tensor_gpu_va(L.attn_q_norm);
            qpc.k_norm_va        = agx_tensor_gpu_va(L.attn_k_norm);
            qpc.inp_pos_va       = G.inp_pos_va;
            qpc.rope_freq_va     = L.rope_freq_va;
            qpc.q_out_va         = va_q_out;
            qpc.k_store_va       = L.k_store_va;
            qpc.v_store_va       = L.v_store_va;
            qpc.k_idxs_va        = L.k_idxs_va;
            qpc.v_idxs_va        = L.v_idxs_va;
            qpc.q_raw_stride_f32 = q_str;
            qpc.k_raw_stride_f32 = k_str;
            qpc.v_raw_stride_f32 = v_str;
            qpc.n_head_q         = Hq;
            qpc.n_head_kv        = Hkv;
            qpc.head_dim         = hd;
            qpc.n_tokens         = M;
            qpc.rope_n_dims      = L.rope_n_dims;
            qpc.k_store_nb1_B    = L.k_store_nb1;
            qpc.v_store_nb1_B    = L.v_store_nb1;
            qpc.k_ids_nb0_B      = L.k_ids_nb0;
            qpc.v_ids_nb0_B      = L.v_ids_nb0;
            qpc.flags            = (!L.wv ? AGX_W7_DSPARK_QKV_SHARED_KV : 0u) |
                                   (L.has_rope ? AGX_W7_DSPARK_QKV_HAS_ROPE : 0u) |
                                   ((L.rope_mode & 2u) ? AGX_W7_DSPARK_QKV_ROPE_NEOX : 0u) |
                                   (L.k_rot_dim == 128u ? AGX_W7_DSPARK_QKV_FWHT_K : 0u) |
                                   (L.k_rot_dim ==  64u ? AGX_W7_DSPARK_QKV_FWHT64_K : 0u) |
                                   (L.v_rot_dim == 128u ? AGX_W7_DSPARK_QKV_FWHT_V : 0u) |
                                   (L.v_rot_dim ==  64u ? AGX_W7_DSPARK_QKV_FWHT64_V : 0u) |
                                   (L.kv_is_q8 ? AGX_W7_DSPARK_QKV_KV_Q8_0 : 0u) |
                                   (L.k_ids_is_i64 ? AGX_W7_DSPARK_QKV_K_IDS_I64 : 0u) |
                                   (L.v_ids_is_i64 ? AGX_W7_DSPARK_QKV_V_IDS_I64 : 0u);
            qpc.rms_eps          = G.rms_eps;
            qpc.theta_scale      = L.theta_scale;
            qpc.freq_scale       = L.freq_scale;
            qpc.attn_factor      = L.attn_factor;
            if (!agx_cdm_dispatch(stream, eng->k_w7_dspark_qkv_rope, Hq + 2u * Hkv, M, 1u,
                                  &qpc, sizeof(qpc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }

        // 1c. FlashAttention
        {
            agx_attn_stage_args attn{};
            attn.n_tokens      = M;
            attn.n_kv          = L.n_kv;
            attn.n_head_q      = Hq;
            attn.n_head_kv     = Hkv;
            attn.head_dim      = hd;
            attn.kq_scale      = L.kq_scale;
            attn.max_bias      = L.max_bias;
            attn.logit_softcap = L.logit_softcap;
            attn.flags         = (L.kq_mask_va ? (L.kq_mask_f16 ? AGX_ATTN_FLAG_HAS_MASK_F16 : AGX_ATTN_FLAG_HAS_MASK_F32) : 0u) |
                                 (L.sinks_va ? AGX_ATTN_FLAG_SINK_PER_HEAD : 0u) |
                                 (L.v_rot_dim == 64u ? AGX_ATTN_FLAG_FWHT64 : 0u) |
                                 (L.max_bias > 0.0f ? AGX_ATTN_FLAG_ALIBI : 0u) |
                                 (L.logit_softcap > 0.0f ? AGX_ATTN_FLAG_SOFTCAP : 0u);
            attn.q_va          = va_q_out;
            attn.q_nb1         = Hq * hd * sizeof(float);
            attn.q_nb2         = hd * sizeof(float);
            attn.out_va        = va_attn_out;
            attn.sinks_va      = L.sinks_va;
            attn.kq_mask_va    = L.kq_mask_va;
            attn.ext_mask_nb1  = L.kq_mask_nb1;
            attn.ext_mask_nb2  = L.kq_mask_nb2;
            attn.ext_mask_ne2  = L.kq_mask_ne2;
            attn.ext_k_dtype   = L.kv_is_q8 ? AGX_KV_Q8_0_GGML : AGX_KV_F16;
            attn.ext_v_dtype   = L.kv_is_q8 ? AGX_KV_Q8_0_GGML : AGX_KV_F16;
            attn.ext_k_va      = L.k_va;
            attn.ext_v_va      = L.v_va;
            attn.ext_k_nb1     = L.k_nb1;
            attn.ext_k_nb2     = L.k_nb2;
            attn.ext_v_nb1     = L.v_nb1;
            attn.ext_v_nb2     = L.v_nb2;
            if (!agx_kv_hybrid_dispatch_full_attn(kv_pool, stream, attn, AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }

        // 1d. W_o GEMM -> va_wo_out (0-SRAM 0-barrier Q8 GEMV for M<=4!)
        if (!(L.wo->type == GGML_TYPE_Q8_0 && agx_tensor_is_repacked(L.wo) &&
              try_w9_q8_gemv(agx_tensor_gpu_va(L.wo), va_attn_out, 0ull, va_wo_out,
                             (uint32_t) L.wo->ne[1], (uint32_t) L.wo->ne[0],
                             (uint32_t) L.wo->ne[1], false, AGX_BARRIER_STRONG_USC))) {
            if (!dispatch_dense_mm(ctx, stream, L.wo, va_attn_out, va_wo_out, M, AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }

        // 1e. Fused [attn_post_norm] + ffn_inp = wo_out + inpL + RMSNorm(ffn_inp, ffn_norm) -> alt_l_va, va_norm & va_norm_f16
        {
            agx_w7_dspark_post_op_push rpc{};
            rpc.va0 = va_wo_out;
            rpc.va1 = inp_l_va;
            rpc.va2 = L.attn_post_norm ? agx_tensor_gpu_va(L.attn_post_norm) : 0ull;
            rpc.va4 = agx_tensor_gpu_va(L.ffn_norm);
            rpc.va5 = alt_l_va; // ffn_inp
            rpc.va6 = va_norm;  // ffn_norm_out
            rpc.va8 = va_norm_f16;
            rpc.op  = 0u; // RES_NORM
            rpc.p0  = D;
            rpc.p1  = M;
            rpc.f0  = G.rms_eps;
            if (!agx_cdm_dispatch(stream, eng->k_w7_dspark_post_op, M, 1u, 1u,
                                  &rpc, sizeof(rpc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }

        // 1f+1g. Fused Q8_0_8BLK M=1..8 Gate + Up + GLU (1x sh_x read, 0 gu_out traffic, FP16 mirror!) OR fallback
        const agx_kernel * k_gu = (M <= 4u) ? eng->k_w7_dspark_gate_up_glu : eng->k_w7_dspark_gate_up_glu_m8;
        if (M <= 8u && (D & 3u) == 0u && k_gu != nullptr &&
            L.ffn_gate->type == GGML_TYPE_Q8_0 && agx_tensor_is_repacked(L.ffn_gate) &&
            L.ffn_up->type == GGML_TYPE_Q8_0 && agx_tensor_is_repacked(L.ffn_up)) {
            agx_w7_dspark_gate_up_glu_push gupc{};
            gupc.wg_va     = agx_tensor_gpu_va(L.ffn_gate);
            gupc.wu_va     = agx_tensor_gpu_va(L.ffn_up);
            gupc.x_va      = va_norm;
            gupc.y_va      = va_inter;
            gupc.y_f16_va  = va_inter_f16;
            gupc.M         = M;
            gupc.N         = F;
            gupc.K         = D;
            gupc.glu_op    = L.glu_op;
            gupc.glu_alpha = L.glu_alpha;
            gupc.glu_limit = L.glu_limit;
            if (!agx_cdm_dispatch(stream, k_gu, (F + 15u) / 16u, 1u, 1u,
                                  &gupc, sizeof(gupc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        } else {
            const uint64_t g_va = va_gu_out;
            const uint64_t u_va = va_gu_out + (uint64_t) M * F * sizeof(float);
            const uint32_t g_str = F, u_str = F;
            if (!dispatch_dense_mm(ctx, stream, L.ffn_gate, va_norm, g_va, M, AGX_BARRIER_NONE)) return false;
            if (!dispatch_dense_mm(ctx, stream, L.ffn_up,   va_norm, u_va, M, AGX_BARRIER_STRONG_USC)) return false;

            agx_w7_dspark_post_op_push gpc{};
            gpc.va0 = g_va;
            gpc.va1 = u_va;
            gpc.va2 = va_inter;
            gpc.op  = 1u; // GLU
            gpc.p0  = F;
            gpc.p1  = M;
            gpc.p2  = g_str;
            gpc.p3  = u_str;
            gpc.p4  = L.glu_op;
            gpc.f0  = L.glu_alpha;
            gpc.f1  = L.glu_limit;
            if (!agx_cdm_dispatch(stream, eng->k_w7_dspark_post_op, (F + 255u) / 256u, M, 1u,
                                  &gpc, sizeof(gpc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }

        // 1h. FFN Down GEMM -> va_wo_out (direct-L1 FP16 Q8 GEMV when gate_up_glu emitted va_inter_f16!)
        const bool have_inter_f16 = (M <= 8u && (D & 3u) == 0u && k_gu != nullptr &&
                                     L.ffn_gate->type == GGML_TYPE_Q8_0 && agx_tensor_is_repacked(L.ffn_gate) &&
                                     L.ffn_up->type == GGML_TYPE_Q8_0 && agx_tensor_is_repacked(L.ffn_up));
        if (!(L.ffn_down->type == GGML_TYPE_Q8_0 && agx_tensor_is_repacked(L.ffn_down) &&
              try_w9_q8_gemv(agx_tensor_gpu_va(L.ffn_down), va_inter, have_inter_f16 ? va_inter_f16 : 0ull,
                             va_wo_out, (uint32_t) L.ffn_down->ne[1], (uint32_t) L.ffn_down->ne[0],
                             (uint32_t) L.ffn_down->ne[1], false, AGX_BARRIER_STRONG_USC))) {
            if (!dispatch_dense_mm(ctx, stream, L.ffn_down, va_inter, va_wo_out, M, AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }

        // 1i. Fused [ffn_post_norm] + l_out = ffn_out + ffn_inp + [out_scale] + next attn_norm / output_norm!
        const bool is_last_layer = (il + 1u == G.n_layers);
        {
            agx_w7_dspark_post_op_push rpc{};
            rpc.va0 = va_wo_out;
            rpc.va1 = alt_l_va; // ffn_inp
            rpc.va2 = L.ffn_post_norm ? agx_tensor_gpu_va(L.ffn_post_norm) : 0ull;
            rpc.va3 = L.out_scale     ? agx_tensor_gpu_va(L.out_scale)     : 0ull;
            rpc.va4 = is_last_layer ? agx_tensor_gpu_va(G.output_norm)
                                    : agx_tensor_gpu_va(G.layers[il + 1u].attn_norm);
            rpc.va5 = inp_l_va; // l_out -> next layer's inpL
            rpc.va6 = va_norm;  // next layer's attn_norm OR result_norm
            rpc.va7 = is_last_layer ? G.result_norm_va : 0ull;
            rpc.va8 = va_norm_f16;
            rpc.op  = 0u; // RES_NORM
            rpc.p0  = D;
            rpc.p1  = M;
            rpc.f0  = G.rms_eps;
            if (!agx_cdm_dispatch(stream, eng->k_w7_dspark_post_op, M, 1u, 1u,
                                  &rpc, sizeof(rpc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }
    }

    // Optional per-phase GPU profiler (GGML_AGX_DSPARK_PROFILE=1, 0 overhead when unset)
    static int dsp_prof_round = 0;
    const char * prof_env = std::getenv("GGML_AGX_DSPARK_PROFILE");
    const bool   do_prof  = (prof_env && prof_env[0] && prof_env[0] != '0' && ++dsp_prof_round == 3);
    uint64_t prof_body_ns = 0, prof_lm_ns = 0, prof_mk_ns = 0;
    if (do_prof) {
        if (agx_cdm_submit(stream, true, &prof_body_ns) != 0) return false;
        agx_cdm_stream_reset(stream);
    }

    // Step 2: Shared LM Head (0-SRAM 0-barrier direct-L1 FP16 Q8 GEMV -> va_lm_pad [n_vocab_padded, M]!)
    {
        agx_weight_layout lm_layout = AGX_WEIGHT_F32;
        if (!agx_gemm_infer_weight_layout(G.output, &lm_layout)) return false;
        const bool lm_allow_m8 = G.lm_head_prec_f32;
        if (!(lm_layout == AGX_WEIGHT_Q8_0_REPACKED &&
              try_w9_q8_gemv(agx_tensor_gpu_va(G.output), va_norm, va_norm_f16,
                             va_lm_pad, G.n_vocab_padded, D, G.n_vocab_padded,
                             lm_allow_m8, AGX_BARRIER_STRONG_USC))) {
            agx_gemm_desc d{};
            d.w_va         = agx_tensor_gpu_va(G.output);
            d.x_va         = va_norm;
            d.y_va         = va_lm_pad;
            d.M            = M;
            d.N            = G.n_vocab_padded;
            d.K            = D;
            d.layout       = lm_layout;
            d.prec_f32     = G.lm_head_prec_f32;
            d.post_barrier = AGX_BARRIER_STRONG_USC;
            if (!agx_gemm_dispatch(ctx, stream, d)) {
                return false;
            }
        }
    }
    if (do_prof) {
        if (agx_cdm_submit(stream, true, &prof_lm_ns) != 0) return false;
        agx_cdm_stream_reset(stream);
    }

    // Step 3: Fused 4-Step DSpark Markov Head (+ inline backend top_k/top_k_rows, reading padded va_lm_pad directly!)
    if (!agx_dspark_dispatch_markov_head_w7(eng, stream, G.markov,
                                            G.inp_tokens_va, va_lm_pad, G.n_vocab_padded,
                                            va_norm, G.rnn_state_va, G.rnn_state_va,
                                            G.final_out_va, G.final_h_nextn_va,
                                            G.n_blocks, G.block_drafts,
                                            AGX_BARRIER_MEMORY_HOST,
                                            G.has_any_topk ? G.topk_rows.data() : nullptr,
                                            G.has_any_topk ? (uint32_t) G.topk_rows.size() : 0u)) {
        return false;
    }

    const auto t_enc_end = std::chrono::steady_clock::now();
    const uint64_t cpu_enc_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t_enc_end - t_enc_beg).count();

    const char * stats_env  = std::getenv("GGML_AGX_STATS");
    const bool   emit_stats = (stats_env && stats_env[0] && stats_env[0] != '0');
    uint64_t gpu_ns = 0;
    // Only attach ts_bo when profiling or stats are requested (#289 §5: saves firmware timestamp DMA!)
    if (agx_cdm_submit(stream, true, (do_prof || emit_stats) ? &gpu_ns : nullptr) != 0) {
        AGX_LOG_ERROR("execute_dspark_draft_cgraph: CDM submit failed");
        return false;
    }
    if (do_prof) {
        prof_mk_ns = gpu_ns;
        uint32_t rb_tmp = 0;
        const uint32_t w2_wt = eng->dspark_bound.w2_q4_bo ? AGX_W8_WTYPE_Q4_32
                                                          : agx_inkling2_tensor_wtype(G.markov.markov_w2, &rb_tmp);
        const uint32_t jp_wt = G.markov.joint_proj ? agx_inkling2_tensor_wtype(G.markov.joint_proj, &rb_tmp) : 99u;
        const uint32_t wq_wt = agx_inkling2_tensor_wtype(G.layers[0].wq, &rb_tmp);
        std::fprintf(stderr,
                     "[ggml-agx DSPARK PROFILE] M=%u D=%u F=%u R=%u V=%u(pad=%u) wq_wt=%u w2_wt=%u jp_wt=%u lm_f32=%d "
                     "body=%.3fms lm=%.3fms markov=%.3fms sum=%.3fms\n",
                     M, D, F, G.markov.rank, G.n_vocab, G.n_vocab_padded, wq_wt, w2_wt, jp_wt, (int) G.lm_head_prec_f32,
                     (double) prof_body_ns * 1e-6, (double) prof_lm_ns * 1e-6, (double) prof_mk_ns * 1e-6,
                     (double) (prof_body_ns + prof_lm_ns + prof_mk_ns) * 1e-6);
    }

    if (emit_stats) {
        agx_cdm_stats cdm_after{};
        agx_cdm_get_stats(stream, &cdm_after);
        std::fprintf(stderr,
                     "[ggml-agx DSPARK MK STATS] layers=%u M=%u blk=%u drafts=%u kv=%s dispatches=%llu submits=%llu "
                     "usc_bar=%llu cpu_enc=%.2fms gpu=%.2fms\n",
                     G.n_layers, M, G.n_blocks, G.block_drafts, G.layers[0].kv_is_q8 ? "Q8_0" : "F16",
                     (unsigned long long) (cdm_after.dispatches          - cdm_before.dispatches),
                     (unsigned long long) (cdm_after.submits             - cdm_before.submits),
                     (unsigned long long) (cdm_after.strong_usc_barriers - cdm_before.strong_usc_barriers),
                     (double) cpu_enc_ns * 1e-6, (double) gpu_ns * 1e-6);
    }
    return true;
}

} // namespace

extern "C" bool agx_inkling2_try_execute(agx_backend_context * ctx, struct ggml_cgraph * cgraph) {
    if (!ctx || !ctx->drm || !ctx->stream || !ctx->compiler || !cgraph) {
        return false;
    }
    const char * no_mk = std::getenv("GGML_AGX_NO_MEGAKERNEL");
    if (no_mk && no_mk[0] && no_mk[0] != '0') {
        return false;
    }

    // 1. Discover Inkling-2-Mini 48-pass decode graph (M=1..8) OR DSpark Inject / Draft graph
    w4_discovered_graph G;
    if (!discover_inkling2_cgraph(cgraph, G)) {
        const char * no_dsp_mk = std::getenv("GGML_AGX_NO_DSPARK_MK");
        if (!(no_dsp_mk && no_dsp_mk[0] && no_dsp_mk[0] != '0')) {
            const char * no_inj = std::getenv("GGML_AGX_NO_DSPARK_INJECT_MK");
            if (!(no_inj && no_inj[0] && no_inj[0] != '0')) {
                w8_dspark_inject_graph IG;
                if (discover_dspark_inject_cgraph(cgraph, IG)) {
                    return execute_dspark_inject_cgraph(ctx, IG);
                }
            }
            w7_dspark_draft_graph DG;
            if (discover_dspark_draft_cgraph(cgraph, DG)) {
                return execute_dspark_draft_cgraph(ctx, DG);
            }
        }
        return false;
    }

    // 2. Ensure engine, Wave 4 kernels, SLC arena, GEMM, KV-hybrid & MoE runners are ready
    const char * dbg_env = std::getenv("GGML_AGX_MK_DEBUG");
    const bool   mk_dbg  = (dbg_env && dbg_env[0] && dbg_env[0] != '0');
    agx_inkling2_engine * eng = agx_graph_exec_get_inkling2(ctx);
    if (!eng || !agx_inkling2_precompile_w4(eng)) {
        if (mk_dbg) std::fprintf(stderr, "[ggml-agx MK FAIL] eng=%d precompile_w4 failed\n", !!eng);
        return false;
    }
    agx_slc_arena * slc = agx_backend_get_slc_arena(ctx);
    if (!slc || !agx_gemm_init(ctx)) {
        if (mk_dbg) std::fprintf(stderr, "[ggml-agx MK FAIL] slc=%d gemm_init failed\n", !!slc);
        return false;
    }
    if (!ctx->kv_hybrid_state) {
        agx_kv_hybrid_config kcfg;
        kcfg.n_slots = 48;
        kcfg.max_full_tokens = 4096;
        ctx->kv_hybrid_state = agx_kv_hybrid_create(ctx->drm, ctx->compiler, kcfg);
        if (!ctx->kv_hybrid_state) {
            if (mk_dbg) std::fprintf(stderr, "[ggml-agx MK FAIL] kv_hybrid_create failed\n");
            return false;
        }
    }
    if (!ctx->moe_state) {
        ctx->moe_state = agx_moe_create(ctx->drm, ctx->compiler);
        if (!ctx->moe_state) {
            if (mk_dbg) std::fprintf(stderr, "[ggml-agx MK FAIL] moe_create failed\n");
            return false;
        }
    }
    if (!ensure_w4_bound_model(eng, G.W)) {
        if (mk_dbg) std::fprintf(stderr, "[ggml-agx MK FAIL] ensure_w4_bound_model failed\n");
        return false;
    }

    auto * kv_pool    = static_cast<agx_kv_hybrid_pool *>(ctx->kv_hybrid_state);
    auto * moe_runner = static_cast<agx_moe_runner *>(ctx->moe_state);
    agx_cdm_stream * stream = ctx->stream;
    const agx_w4_slc_layout L = agx_w4_partition_slc(slc);

    const auto t_enc_beg = std::chrono::steady_clock::now();
    agx_cdm_stream_reset(stream);
    agx_cdm_stats cdm_before{};
    agx_cdm_get_stats(stream, &cdm_before);

    const uint32_t M  = G.n_tokens;
    const uint32_t D  = 2304u;
    const auto &   hp = G.hp;
    const auto &   W  = G.W;

    const char * no_coal_env = std::getenv("GGML_AGX_MK_NO_COAL");
    const bool   disable_coal = (no_coal_env && no_coal_env[0] && no_coal_env[0] != '0');
    const char * dump_env = std::getenv("GGML_AGX_MK_DUMP");
    static bool  did_dump = false;
    const bool   mk_dump  = (dump_env && dump_env[0] && dump_env[0] != '0' && !did_dump && M == 1u);
    if (mk_dump) did_dump = true;

    auto slc_dump_f32 = [&](const char * tag, uint64_t va, uint32_t n) {
        if (!mk_dump || !slc->cpu_base || va < slc->base_va || va + n * 4ull > slc->base_va + slc->total_bytes) return;
        agx_cdm_barrier_host(stream);
        agx_cdm_submit(stream, true, nullptr);
        agx_cdm_stream_reset(stream);
        const float * p = reinterpret_cast<const float *>(static_cast<const uint8_t *>(slc->cpu_base) + (va - slc->base_va));
        double ss = 0.0;
        for (uint32_t i = 0; i < n; ++i) ss += (double) p[i] * (double) p[i];
        std::fprintf(stderr, "[MK DUMP] %-16s n=%-5u l2=%10.5f v=[%9.5f %9.5f %9.5f %9.5f]\n",
                     tag, n, std::sqrt(ss), p[0], n > 1 ? p[1] : 0.0f, n > 2 ? p[2] : 0.0f, n > 3 ? p[3] : 0.0f);
    };
    static bool  printed_dbg = false;
    if (mk_dbg && !printed_dbg) {
        printed_dbg = true;
        std::fprintf(stderr,
                     "[ggml-agx MK DEBUG] M=%u n_out=%u kv=%s n_kv0=%u n_kv5=%u "
                     "eps=%.1e qk_clip=%.3f rel_clip=%.3f situ=(%.3f,%.3f) rtr_scale=%.4f rtr_eps=%.1e "
                     "tau=(floor=%u,alpha=%.4f) softcap=%.2f k_rot=%d v_rot=%d "
                     "dtypes: emb=%s wq=%s(r%d) rel_b=%s ctrl_d=%s exp0=%s(r%d,gs%d,ds%d,epb%d) exp20=%s(r%d) out=%s(r%d) coal0=%d\n",
                     M, G.n_out, G.passes[0].kv_is_q8 ? "Q8_0" : "F16", G.passes[0].n_kv, G.passes[5].n_kv,
                     hp.f_norm_rms_eps, hp.f_qk_clip, hp.f_relative_clip, hp.situ_beta, hp.situ_linear_beta,
                     hp.expert_weights_scale, hp.expert_weights_norm_eps,
                     hp.n_log_scale_floor, hp.f_log_scale_alpha, hp.f_final_logit_softcapping,
                     (int) G.passes[0].has_k_rot, (int) G.passes[0].has_v_rot,
                     ggml_type_name(W.tok_embd->type),
                     ggml_type_name(W.layers[0].wq->type), (int) agx_tensor_is_repacked(W.layers[0].wq),
                     ggml_type_name(W.layers[0].attn_rel_bias->type),
                     ggml_type_name(W.loop_layers[0].ctrl_down[0]->type),
                     ggml_type_name(W.layers[0].ffn_gate_up_exps->type),
                     (int) agx_tensor_is_repacked(W.layers[0].ffn_gate_up_exps),
                     !!W.layers[0].ffn_gate_up_exps_s, !!W.layers[0].ffn_down_exps_s, !!W.layers[0].ffn_exp_probs_b,
                     ggml_type_name(W.layers[20].ffn_gate_up_exps->type),
                     (int) agx_tensor_is_repacked(W.layers[20].ffn_gate_up_exps),
                     ggml_type_name(W.output->type), (int) agx_tensor_is_repacked(W.output),
                     (int) eng->w4_model.layers[0].valid);
    }

    // Static zero-copy slot mapping inside slc->res_blocks[0..7]:
    //   [0]=h_0, [1]=prelude, [2]=h_12_p0/p2, [3]=h_12_p1, [4]=bank_out, [5]=cur_p2, [6]=h_alt, [7]=ffn_inp
    const uint64_t va_h_0      = slc->res_blocks[0];
    const uint64_t va_prelude  = slc->res_blocks[1];
    const uint64_t va_h12_even = slc->res_blocks[2];
    const uint64_t va_h12_odd  = slc->res_blocks[3];
    const uint64_t va_bank_out = slc->res_blocks[4];
    const uint64_t va_cur_p2   = slc->res_blocks[5];
    const uint64_t va_h_alt    = slc->res_blocks[6];
    const uint64_t va_ffn_inp  = slc->res_blocks[7];

    auto alloc_block_ptrs = [&](const std::vector<uint64_t> & blk_vec) -> uint64_t {
        if (blk_vec.empty()) return 0ull;
        void * cpu_p = nullptr;
        uint64_t va = agx_cdm_alloc_args(stream, blk_vec.size() * sizeof(uint64_t), 16, &cpu_p);
        if (!va || !cpu_p) return 0ull;
        // Order: [b_{k-1}, ..., b_0] matching llama_model_inkling2::graph::mix_stack
        auto * dst = static_cast<uint64_t *>(cpu_p);
        for (size_t k = 0; k < blk_vec.size(); ++k) {
            dst[k] = blk_vec[blk_vec.size() - 1u - k];
        }
        return va;
    };

    // Helper: pick output VA for `l_out` of pass `v` so block/prelude/bank slots need ZERO copies!
    auto target_l_out_va = [&](uint32_t v, uint64_t cur_va) -> uint64_t {
        if (v == 5u)  return va_prelude;  // prelude output -> res_blocks[1]
        if (v == 11u) return va_h12_even; // ip=12 pass 0 input -> res_blocks[2]
        if (v == 17u) return va_bank_out; // pass 0 output -> res_blocks[4] (bank_out for pass 1)
        if (v == 23u) return va_h12_odd;  // ip=12 pass 1 input -> res_blocks[3]
        if (v == 29u) return va_bank_out; // pass 1 output -> res_blocks[4] (bank_out for pass 2)
        if (v == 35u) return va_h12_even; // ip=12 pass 2 input -> res_blocks[2]
        if (v == 41u) return va_cur_p2;   // pass 2 output -> res_blocks[5]
        return (cur_va == slc->h_cur) ? va_h_alt : slc->h_cur;
    };

    // ------------------------------------------------------------------------
    // Step 0: Embedding lookup + tok_norm -> va_h_0 (+ optional rs_zero FILLs)
    // ------------------------------------------------------------------------
    {
        uint32_t emb_rb = 0;
        embd_rmsnorm_push ep{};
        ep.embd_va    = agx_tensor_gpu_va(W.tok_embd);
        ep.tok_ids_va = G.inp_tokens_va;
        ep.norm_w_va  = agx_tensor_gpu_va(W.tok_norm);
        ep.dst_va     = va_h_0;
        ep.n_embd     = D;
        ep.n_tokens   = M;
        ep.wtype      = agx_inkling2_tensor_wtype(W.tok_embd, &emb_rb);
        ep.row_bytes  = emb_rb;
        ep.eps        = hp.f_norm_rms_eps;
        if (!agx_cdm_dispatch(stream, eng->k_embd_rmsnorm, M, 1u, 1u, &ep, sizeof(ep), AGX_BARRIER_NONE)) {
            return false;
        }
        // Emit any non-empty rs_zero FILLs concurrently with embd_rmsnorm!
        for (uint32_t v = 0; v < 48u; ++v) {
            if (G.passes[v].rs_zero_va && G.passes[v].rs_zero_elems > 0u) {
                w4_fill_f32_push fp{ G.passes[v].rs_zero_va, G.passes[v].rs_zero_elems, 0.0f };
                if (!agx_cdm_dispatch(stream, eng->k_w4_fill_f32, (fp.n_elems + 255u) / 256u, 1u, 1u,
                                      &fp, sizeof(fp), AGX_BARRIER_NONE)) {
                    return false;
                }
            }
        }
        if (G.export_tok_norm != 0ull) {
            copy_f32_push cp{ va_h_0, G.export_tok_norm, M * D, 0u };
            agx_cdm_barrier_usc(stream);
            if (!agx_cdm_dispatch(stream, eng->k_copy_f32, (M * D + 255u) / 256u, 1u, 1u,
                                  &cp, sizeof(cp), AGX_BARRIER_NONE)) {
                return false;
            }
        }
        agx_cdm_barrier_usc(stream);
    }

    auto emit_layer_inp_stores = [&](const std::vector<w4_layer_inp_store> & stores, uint64_t src_va) -> bool {
        for (const auto & st : stores) {
            w4_scatter_rows_f32_push sp{ src_va, st.ids_va, st.dst_va, D, M, st.dst_nb1_B, st.ids_is_i64 ? 1u : 0u };
            if (!agx_cdm_dispatch(stream, eng->k_w4_scatter_rows_f32, (D + 255u) / 256u, M, 1u,
                                  &sp, sizeof(sp), AGX_BARRIER_NONE)) {
                return false;
            }
        }
        return true;
    };
    if (!emit_layer_inp_stores(G.tok_norm_stores, va_h_0)) {
        return false;
    }

    // Layer 0 has no attn_res_query -> run single RMSNorm(va_h_0, blk.0.attn_norm) -> L.x_q (+ L.x_q_f16 for M>=2!)
    bool x_f16_ready = (M >= 2u);
    {
        rmsnorm_f32_push np{
            va_h_0, agx_tensor_gpu_va(W.layers[0].attn_norm), L.x_q,
            D, M, hp.f_norm_rms_eps, 0u, (M >= 2u) ? L.x_q_f16 : 0ull
        };
        if (!agx_cdm_dispatch(stream, eng->k_rmsnorm_f32, M, 1u, 1u, &np, sizeof(np), AGX_BARRIER_STRONG_USC)) {
            return false;
        }
    }

    std::vector<uint64_t> blocks;
    blocks.reserve(6);
    std::vector<uint64_t> bank;
    bank.reserve(2);
    uint64_t cur_va = va_h_0;
    bool attn_inputs_ready = true; // Layer 0 x_q already normalized above!

    // ------------------------------------------------------------------------
    // Wave 5 Persistent 40-WG SENSE_BARRIER B=1 Megakernel Path (M == 1)
    // Collapses ~700 STRONG_USC barriers -> ~12 (chunked) or ~99 (split)!
    // Instant A/B toggles:
    //   GGML_AGX_NO_PERSISTENT=1   : revert to Wave 4 per-stage dispatches
    //   GGML_AGX_PERSIST_SPLIT=1   : use split Attn + MoE 40-WG super-stages
    //   GGML_AGX_PERSIST_CHUNK=<N> : passes per persistent chunk (default 6)
    // ------------------------------------------------------------------------
    const char * no_persist_env = std::getenv("GGML_AGX_NO_PERSISTENT");
    const bool   want_persist   = (M == 1u) && !mk_dump && !disable_coal &&
                                  !(no_persist_env && no_persist_env[0] && no_persist_env[0] != '0');
    bool ran_w5_persistent = false;
    if (want_persist) {
        const char * split_env   = std::getenv("GGML_AGX_PERSIST_SPLIT");
        const char * chk_env     = std::getenv("GGML_AGX_PERSIST_CHUNK");
        const bool   force_split = (split_env && split_env[0] && split_env[0] != '0');
        const bool   force_chunk = (chk_env   && chk_env[0]   && chk_env[0]   != '0');
        uint32_t     max_chunk   = 6u;
        if (force_chunk) {
            int cv = std::atoi(chk_env);
            if (cv >= 1 && cv <= 18) max_chunk = (uint32_t) cv;
        }

        const uint32_t kv_q8_0 = G.passes[0].kv_is_q8 ? 1u : 0u;
        bool w5_ok = true;
        bool pass_use_b1_hybrid_fa[48] = {};
        bool any_pass_ext_fa = false;
        bool any_pass_banded_or_rot64 = false;
        uint32_t pass_moe_idx[48] = {};
        agx_w5_pass_desc descs[48];
        std::memset(descs, 0, sizeof(descs));

        std::vector<uint64_t> p_blocks;
        p_blocks.reserve(6);
        std::vector<uint64_t> p_bank;
        p_bank.reserve(2);
        uint64_t p_cur_va = va_h_0;
        bool     p_attn_ready = true;

        for (uint32_t v = 0; v < 48u && w5_ok; ++v) {
            const auto & P     = G.passes[v];
            const auto & layer = W.layers[P.ip];
            const auto & C     = eng->w4_model.layers[P.ip];
            const agx_inkling2_loop_tensors * loop = P.has_loop ? &W.loop_layers[P.ip - 6u] : nullptr;

            if ((P.kv_is_q8 ? 1u : 0u) != kv_q8_0) {
                if (mk_dbg) std::fprintf(stderr, "[ggml-agx W5 PERSIST FALLBACK] v=%u mixed kv_q8\n", v);
                w5_ok = false; break;
            }
            if (!C.valid || !C.in_coal[0] || !C.in_coal[1] || !C.in_coal[2] ||
                !C.in_coal[4] || !C.in_coal[5]) {
                if (mk_dbg) std::fprintf(stderr, "[ggml-agx W5 PERSIST FALLBACK] v=%u coal=%d in=[%d,%d,%d,%d,%d,%d]\n",
                                         v, (int) C.valid, (int) C.in_coal[0], (int) C.in_coal[1], (int) C.in_coal[2],
                                         (int) C.in_coal[3], (int) C.in_coal[4], (int) C.in_coal[5]);
                w5_ok = false; break;
            }
            agx_weight_layout wo_lay = AGX_WEIGHT_F32, gu_lay = AGX_WEIGHT_F32, dn_lay = AGX_WEIGHT_F32;
            if (!agx_gemm_infer_weight_layout(layer.wo,               &wo_lay) ||
                !agx_gemm_infer_weight_layout(layer.ffn_gate_up_exps, &gu_lay) ||
                !agx_gemm_infer_weight_layout(layer.ffn_down_exps,    &dn_lay) ||
                wo_lay != AGX_WEIGHT_Q8_0_REPACKED || gu_lay != dn_lay ||
                (gu_lay != AGX_WEIGHT_NVFP4_REPACKED && gu_lay != AGX_WEIGHT_Q8_0_REPACKED)) {
                if (mk_dbg) std::fprintf(stderr, "[ggml-agx W5 PERSIST FALLBACK] v=%u layouts wo=%u gu=%u dn=%u\n",
                                         v, (unsigned) wo_lay, (unsigned) gu_lay, (unsigned) dn_lay);
                w5_ok = false; break;
            }
            if (gu_lay == AGX_WEIGHT_Q8_0_REPACKED && P.has_loop) {
                if (mk_dbg) std::fprintf(stderr, "[ggml-agx W5 PERSIST FALLBACK] v=%u q8 moe + has_loop\n", v);
                w5_ok = false; break; // Q8_0 MoE only on non-looped coda passes
            }
            pass_moe_idx[v] = (gu_lay == AGX_WEIGHT_Q8_0_REPACKED) ? 1u : 0u;

            const uint32_t mask_row_stride = (P.n_kv + 7u) & ~7u;
            const size_t   mask_need_B     = (size_t) 24u * mask_row_stride * sizeof(uint16_t);
            // Wave 10: SWA KV cache is padded to n_swa + n_ubatch (up to 3072/4096 cells, only 1024 active!).
            // Keep ALL 40 SWA passes (<= 4096) and short Full passes (<= 1024) in ultra-lean <=56 GPR32 HYBRID_MODE=0;
            // route Full passes > 1024 straight to GQA_GRP=3 split-K PASS_MODE=0 (3x KV sharing, 120-240 WGs)!
            const bool     b1_hybrid_fa_ok = (mask_need_B <= L.kq_mask_f16_bytes) &&
                                             (P.is_swa ? (P.n_kv <= 4096u) : (P.n_kv <= 1024u));
            pass_use_b1_hybrid_fa[v] = b1_hybrid_fa_ok;
            if (!b1_hybrid_fa_ok) {
                any_pass_ext_fa = true;
            }
            if (P.has_banded_rel || P.k_rot_dim == 64u || P.v_rot_dim == 64u) {
                any_pass_banded_or_rot64 = true;
            }
            if (P.has_loop) {
                if (hp.gate_rank != 64u) {
                    if (mk_dbg) std::fprintf(stderr, "[ggml-agx W5 PERSIST FALLBACK] v=%u gate_rank=%u\n", v, hp.gate_rank);
                    w5_ok = false; break;
                }
                for (int c = 0; c < 4; ++c) {
                    if (loop->ctrl_down[c]->type   != GGML_TYPE_F32 ||
                        loop->ctrl_gain[c]->type   != GGML_TYPE_F32 ||
                        loop->ctrl_forget[c]->type != GGML_TYPE_F32) {
                        if (mk_dbg) std::fprintf(stderr, "[ggml-agx W5 PERSIST FALLBACK] v=%u ctrl[%d] non-F32\n", v, c);
                        w5_ok = false; break;
                    }
                }
                if (!w5_ok) break;
            }

            if (v == 18u || v == 30u) {
                p_bank.clear();
                p_bank.push_back((v == 18u) ? va_h12_even : va_h12_odd);
                p_blocks.clear();
                p_blocks.push_back(va_h_0);
                p_cur_va = va_prelude;
                p_attn_ready = false;
            }
            const bool need_qkv_mix = (P.has_loop && !p_attn_ready);
            if (need_qkv_mix && !P.is_swa) { w5_ok = false; break; }

            auto & d = descs[v];
            const bool same_qkv_inp = (P.is_swa && !P.has_loop);

            // 1. SLC Arena / Activation VAs
            d.x_q_va            = L.x_q;
            d.x_k_va            = same_qkv_inp ? L.x_q : L.x_k;
            d.x_v_va            = same_qkv_inp ? L.x_q : L.x_v;
            d.qkv_in_va[0]      = L.qkv_in_0;
            d.qkv_in_va[1]      = L.qkv_in_1;
            d.qkv_in_va[2]      = L.qkv_in_2;
            d.qkv_buf_va        = slc->qkv_buf;
            if (same_qkv_inp) {
                d.k_raw_va      = slc->qkv_buf + (uint64_t) C.row_off[4] * sizeof(float);
                d.v_raw_va      = slc->qkv_buf + (uint64_t) C.row_off[5] * sizeof(float);
            } else {
                const uint64_t k_off = agx_align_up((size_t) C.q_prefix_rows * sizeof(float), AGX_CACHE_LINE_BYTES);
                const uint64_t v_off = k_off + agx_align_up((size_t) C.row_cnt[4] * sizeof(float), AGX_CACHE_LINE_BYTES);
                d.k_raw_va      = slc->qkv_buf + k_off;
                d.v_raw_va      = slc->qkv_buf + v_off;
            }
            d.q_out_va          = L.q_out;
            d.attn_out_va       = slc->attn_out;
            d.wo_out_va         = L.x_k;
            d.ffn_inp_va        = va_ffn_inp;
            d.k_dyn_va          = P.is_swa ? 0ull : L.k_dyn;
            d.v_dyn_va          = P.is_swa ? 0ull : L.v_dyn;
            d.router_logits_va  = L.router_logits;
            d.router_topk_va    = slc->router_topk;
            d.router_topk_w_va  = slc->router_topk_w;
            d.moe_inter_va      = slc->moe_inter;
            d.moe_out_va        = L.moe_out;
            d.h_msg_va          = slc->h_msg;
            for (int j = 0; j < 3; ++j) {
                d.ctrl_cat_va[j]  = L.ctrl_cat[j];
                d.ctrl_gate_va[j] = L.ctrl_gate[j];
                d.ctrl_gain_va[j] = L.ctrl_gain[j];
            }

            // 2. Loop Controller Weights & Mix Inputs
            if (P.has_loop) {
                d.msg_state_va[0] = p_bank[0];
                d.msg_state_va[1] = va_bank_out;
                d.msg_query_va    = agx_tensor_gpu_va(loop->msg_query);
                if (need_qkv_mix) {
                    d.qkv_mix_cur_va   = p_cur_va;
                    d.qkv_mix_bptrs_va = alloc_block_ptrs(p_blocks);
                    d.qkv_mix_query_va = agx_tensor_gpu_va(layer.attn_res_query);
                    d.qkv_mix_n_blocks = (uint32_t) p_blocks.size();
                    if (!d.qkv_mix_bptrs_va) { w5_ok = false; break; }
                }
                for (int c = 0; c < 4; ++c) {
                    d.ctrl_down_w_va[c]   = agx_tensor_gpu_va(loop->ctrl_down[c]);
                    d.ctrl_gain_w_va[c]   = agx_tensor_gpu_va(loop->ctrl_gain[c]);
                    d.ctrl_forget_w_va[c] = agx_tensor_gpu_va(loop->ctrl_forget[c]);
                }
                d.attn_norm_w_va[0] = agx_tensor_gpu_va(P.is_swa ? layer.attn_norm : layer.attn_q_inp_norm);
                d.attn_norm_w_va[1] = agx_tensor_gpu_va(P.is_swa ? layer.attn_norm : layer.attn_k_inp_norm);
                d.attn_norm_w_va[2] = agx_tensor_gpu_va(P.is_swa ? layer.attn_norm : layer.attn_v_inp_norm);
            }
            d.ffn_norm_w_va = agx_tensor_gpu_va(layer.ffn_norm);

            // 3. Attention Weights, KV Cache & Mask
            d.w_attn_in_va = C.w_attn_in.gpu_va;
            d.wk_byte_off  = C.byte_off[4];
            d.wv_byte_off  = C.byte_off[5];
            if (C.in_coal[3]) {
                d.sink_proj_va    = 0ull;
                d.sinks_va        = slc->qkv_buf + (uint64_t) C.row_off[3] * sizeof(float);
                d.sink_proj_wtype = 0u;
                d.sink_proj_row_B = 0u;
            } else {
                d.sink_proj_va    = agx_tensor_gpu_va(layer.attn_sink_proj);
                d.sinks_va        = L.fallback_qkv;
                d.sink_proj_wtype = agx_inkling2_tensor_wtype(layer.attn_sink_proj, &d.sink_proj_row_B);
            }
            if (!P.is_swa) {
                d.k_conv_proj_va = agx_tensor_gpu_va(layer.attn_k_conv_proj);
                d.v_conv_proj_va = agx_tensor_gpu_va(layer.attn_v_conv_proj);
                d.k_proj_wtype   = agx_inkling2_tensor_wtype(layer.attn_k_conv_proj, &d.k_proj_row_B);
                d.v_proj_wtype   = agx_inkling2_tensor_wtype(layer.attn_v_conv_proj, &d.v_proj_row_B);
            }
            d.q_clip_va    = agx_tensor_gpu_va(layer.attn_q_clip);
            d.r_clip_va    = agx_tensor_gpu_va(layer.attn_rel_clip);
            d.k_conv_w_va  = agx_tensor_gpu_va(layer.attn_k_conv);
            d.v_conv_w_va  = agx_tensor_gpu_va(layer.attn_v_conv);
            d.rel_bias_va  = agx_tensor_gpu_va(layer.attn_rel_bias);
            d.inp_pos_va   = G.inp_pos_va;
            d.r_l_va       = P.r_l_va;
            d.s_copy_va    = G.s_copy_va;
            for (int c = 0; c < 4; ++c) d.cell_va[c] = P.cell_va[c];
            d.k_va         = P.k_va;
            d.v_va         = P.v_va;
            d.k_store_va   = P.k_store_va;
            d.v_store_va   = P.v_store_va;
            d.k_idxs_va    = P.k_idxs_va;
            d.v_idxs_va    = P.v_idxs_va;
            d.rel_pos_va   = P.rel_pos_va;
            d.base_mask_va = P.base_mask_va;
            d.kq_mask_va   = L.kq_mask_f16;
            d.wo_va        = agx_tensor_gpu_va(layer.wo);

            // 4. Post-Attn Stage Prep
            const bool new_block = (P.ip % hp.attn_res_block_size == 0u);
            if (new_block) p_blocks.push_back(p_cur_va);
            d.attn_post_conv_w_va  = agx_tensor_gpu_va(layer.attn_post_conv);
            d.attn_res_va          = new_block ? 0ull : p_cur_va;
            d.attn_bptrs_va        = alloc_block_ptrs(p_blocks);
            d.ffn_res_query_va     = agx_tensor_gpu_va(layer.ffn_res_query);
            d.post_attn_mix_out_va = !P.has_loop ? L.x_q : L.qkv_in_0;
            d.post_attn_n_blocks   = (uint32_t) p_blocks.size();
            if (!d.attn_bptrs_va) { w5_ok = false; break; }

            // 5. Router & MoE Weights
            d.router_w_va    = agx_tensor_gpu_va(layer.ffn_gate_inp);
            d.router_wtype   = agx_inkling2_tensor_wtype(layer.ffn_gate_inp, &d.router_row_B);
            d.exp_probs_b_va = layer.ffn_exp_probs_b ? agx_tensor_gpu_va(layer.ffn_exp_probs_b) : 0ull;
            d.gu_exps_va     = agx_tensor_gpu_va(layer.ffn_gate_up_exps);
            d.dn_exps_va     = agx_tensor_gpu_va(layer.ffn_down_exps);
            d.gu_slice_B     = agx_repack_slice_stride(layer.ffn_gate_up_exps->type,
                                                       layer.ffn_gate_up_exps->ne[0], layer.ffn_gate_up_exps->ne[1]);
            d.dn_slice_B     = agx_repack_slice_stride(layer.ffn_down_exps->type,
                                                       layer.ffn_down_exps->ne[0], layer.ffn_down_exps->ne[1]);
            d.gu_scale_va    = layer.ffn_gate_up_exps_s ? agx_tensor_gpu_va(layer.ffn_gate_up_exps_s) : 0ull;
            d.dn_scale_va    = layer.ffn_down_exps_s    ? agx_tensor_gpu_va(layer.ffn_down_exps_s)    : 0ull;

            // 6. Post-FFN Stage Prep
            const uint64_t next_cur_va = target_l_out_va(v, p_cur_va);
            const bool is_pass_reset   = (v == 17u || v == 29u);
            const bool is_final_pass   = (v == 47u);
            d.ffn_post_conv_w_va       = agx_tensor_gpu_va(layer.ffn_post_conv);
            d.next_cur_va              = next_cur_va;
            d.export_l_out_va          = P.export_l_out_va;

            bool post_ffn_do_mix  = false;
            bool post_ffn_do_norm = false;
            if (is_final_pass) {
                d.ffn_bptrs_va       = alloc_block_ptrs(p_blocks);
                d.next_attn_query_va = agx_tensor_gpu_va(W.output_res_query);
                d.next_norm_w_va[0]  = agx_tensor_gpu_va(W.output_norm);
                d.next_mix_out_va[0] = L.x_q;
                d.next_mix_export_va = G.export_out_norm;
                d.post_ffn_n_blocks  = (uint32_t) p_blocks.size();
                d.post_ffn_n_queries = 1u;
                post_ffn_do_mix      = true;
                post_ffn_do_norm     = true;
                if (!d.ffn_bptrs_va) { w5_ok = false; break; }
            } else if (!is_pass_reset) {
                const auto & Pn      = G.passes[v + 1u];
                const auto & Ln      = W.layers[Pn.ip];
                d.ffn_bptrs_va       = alloc_block_ptrs(p_blocks);
                d.next_attn_query_va = agx_tensor_gpu_va(Ln.attn_res_query);
                d.post_ffn_n_blocks  = (uint32_t) p_blocks.size();
                d.post_ffn_n_queries = Pn.is_swa ? 1u : 3u;
                post_ffn_do_mix      = true;
                if (!Pn.has_loop) {
                    post_ffn_do_norm     = true;
                    d.next_norm_w_va[0]  = agx_tensor_gpu_va(Pn.is_swa ? Ln.attn_norm : Ln.attn_q_inp_norm);
                    d.next_norm_w_va[1]  = !Pn.is_swa ? agx_tensor_gpu_va(Ln.attn_k_inp_norm) : 0ull;
                    d.next_norm_w_va[2]  = !Pn.is_swa ? agx_tensor_gpu_va(Ln.attn_v_inp_norm) : 0ull;
                    d.next_mix_out_va[0] = L.x_q;
                    d.next_mix_out_va[1] = L.x_k;
                    d.next_mix_out_va[2] = L.x_v;
                } else {
                    d.next_mix_out_va[0] = L.qkv_in_0;
                    d.next_mix_out_va[1] = L.qkv_in_1;
                    d.next_mix_out_va[2] = L.qkv_in_2;
                }
                p_attn_ready = true;
                if (!d.ffn_bptrs_va) { w5_ok = false; break; }
            }

            // 7. Scalars & Flags
            d.flags = (P.is_swa           ? AGX_W5_PFLAG_IS_SWA            : 0u) |
                      (same_qkv_inp       ? AGX_W5_PFLAG_SAME_QKV_INP      : 0u) |
                      (need_qkv_mix       ? AGX_W5_PFLAG_NEED_QKV_MIX      : 0u) |
                      (P.has_k_rot        ? AGX_W5_PFLAG_HAS_K_ROT         : 0u) |
                      (P.has_v_rot        ? AGX_W5_PFLAG_HAS_V_ROT         : 0u) |
                      (P.k_rot_dim == 64u ? AGX_W5_PFLAG_K_ROT_64          : 0u) |
                      (P.v_rot_dim == 64u ? AGX_W5_PFLAG_V_ROT_64          : 0u) |
                      (P.has_banded_rel   ? AGX_W5_PFLAG_BANDED_REL_PACK   : 0u) |
                      (P.k_ids_is_i64     ? AGX_W5_PFLAG_K_IDS_I64         : 0u) |
                      (P.v_ids_is_i64     ? AGX_W5_PFLAG_V_IDS_I64         : 0u) |
                      (P.bmask_is_f16     ? AGX_W5_PFLAG_BMASK_F16         : 0u) |
                      (!new_block         ? AGX_W5_PFLAG_POST_ATTN_ADD_RES : 0u) |
                      (!P.has_loop        ? AGX_W5_PFLAG_POST_ATTN_DO_NORM : 0u) |
                      (post_ffn_do_mix    ? AGX_W5_PFLAG_POST_FFN_DO_MIX   : 0u) |
                      (post_ffn_do_norm   ? AGX_W5_PFLAG_POST_FFN_DO_NORM  : 0u);
            d.rel_qp_off          = P.has_banded_rel ? P.n_kv : 0u;
            d.total_coal_rows     = C.total_coal_rows;
            d.q_prefix_rows       = C.q_prefix_rows;
            d.wk_rows             = C.row_cnt[4];
            d.wv_rows             = C.row_cnt[5];
            d.q_row_off           = C.row_off[0];
            d.qgate_row_off       = C.row_off[1];
            d.r_row_off           = C.row_off[2];
            d.r_nb1_f32           = P.r_nb1_f32;
            d.k_nb1               = P.k_nb1;
            d.k_nb2               = P.k_nb2;
            d.v_nb1               = P.v_nb1;
            d.v_nb2               = P.v_nb2;
            d.k_store_nb1         = P.k_store_nb1;
            d.v_store_nb1         = P.v_store_nb1;
            d.k_ids_nb0           = P.k_ids_nb0;
            d.v_ids_nb0           = P.v_ids_nb0;
            d.n_rel               = (uint32_t) layer.attn_rel_bias->ne[0];
            d.n_extent            = (uint32_t) (layer.attn_rel_bias->ne[1] - (G.rel_bias_padded[P.ip] ? 1 : 0));
            d.n_kv                = P.n_kv;
            d.kq_mask_stride      = mask_row_stride;
            d.bias_wtype          = agx_inkling2_tensor_wtype(layer.attn_rel_bias, &d.bias_nb1_B);
            d.n_expert            = hp.n_expert;
            d.n_ff_exp            = hp.n_ff_exp;
            d.kv_conv_res_scale   = 1.0f / std::sqrt((float) ((P.is_swa ? 1u : 2u) * 4u + 1u));
            d.post_conv_res_scale = 1.0f / std::sqrt(5.0f);
            d.qk_clip             = hp.f_qk_clip;
            d.rel_clip            = hp.f_relative_clip;
            d.rms_eps             = hp.f_norm_rms_eps;
            d.log_scale_alpha     = hp.f_log_scale_alpha;
            d.inv_log_floor       = 1.0f / (float) hp.n_log_scale_floor;
            d.router_norm_eps     = hp.expert_weights_norm_eps;
            d.router_w_scale      = (hp.expert_weights_scale != 0.0f) ? hp.expert_weights_scale : 1.0f;
            d.situ_beta           = hp.situ_beta;
            d.situ_linear_beta    = hp.situ_linear_beta;

            p_cur_va = next_cur_va;
        }

        if (w5_ok) {
            // Default = Wave 7 512T Hybrid (16 SGs/core across all 3 super-stages + 517 GB/s attn_in_b1 & Wo).
            // Instant A/B toggles:
            //   GGML_AGX_W7_FUSE_WO=1         : fuse WG-striped branchless Wo into HYBRID_MODE=1 (-48 disp/USC)
            //   GGML_AGX_PERSIST_CHUNK=<N>    : 14-dispatch chunked persistent mode
            //   GGML_AGX_PERSIST_SPLIT=1      : 101-dispatch split persistent mode
            bool hybrid_router_ok = true;
            for (uint32_t v = 0; v < 48u; ++v) {
                if (descs[v].router_wtype > 2u) { hybrid_router_ok = false; break; }
            }
            const bool use_hybrid = !force_chunk && !force_split && hybrid_router_ok &&
                                    ensure_w5_hybrid_kernels(eng, kv_q8_0);
            const bool use_chunk  = !use_hybrid && !force_split && !any_pass_ext_fa && !any_pass_banded_or_rot64 &&
                                    ensure_w5_chunk_kernels(eng, kv_q8_0);
            const bool use_split  = !use_hybrid && !use_chunk   && !any_pass_ext_fa && !any_pass_banded_or_rot64 &&
                                    ensure_w5_split_kernels(eng, kv_q8_0);
            if (use_hybrid || use_chunk || use_split) {
                void * descs_cpu = nullptr;
                const uint64_t descs_gpu_va = agx_cdm_alloc_args(stream, sizeof(descs), AGX_CACHE_LINE_BYTES, &descs_cpu);
                if (!descs_gpu_va || !descs_cpu) return false;
                std::memcpy(descs_cpu, descs, sizeof(descs));

                // Reset 128B SLC sense-reversing barrier for 40 co-resident WGs (replay resets via ggml-agx.cpp)
                agx_slc_arena_reset_sense_barrier(slc, 40u);

                if (use_hybrid) {
                    for (uint32_t v = 0u; v < 48u; ++v) {
                        const auto & P     = G.passes[v];
                        const auto & layer = W.layers[P.ip];
                        const auto & C     = eng->w4_model.layers[P.ip];
                        const auto & d     = descs[v];
                        const uint32_t lp_i = P.has_loop ? 1u : 0u;

                        // Op 1: Dedicated 517 GB/s 0B-SRAM agx_w5_attn_in_b1 (keeps HYBRID_MODE=0 <= 56 GPR32!)
                        {
                            const bool can_fuse_sink   = (!C.in_coal[3] && d.sink_proj_wtype <= 2u);
                            const bool can_fuse_dyn    = (!P.is_swa && d.k_proj_wtype <= 2u && d.v_proj_wtype <= 2u);
                            const bool need_extra_sink = (!C.in_coal[3] && !can_fuse_sink);
                            const bool need_extra_dyn  = (!P.is_swa && !can_fuse_dyn);
                            const uint32_t tail_rows   = (can_fuse_sink ? 24u : 0u) + (can_fuse_dyn ? 64u : 0u);
                            const uint32_t tot_rows    = C.total_coal_rows + tail_rows;
                            agx_w5_attn_in_push apc{};
                            apc.w_attn_in_va    = C.w_attn_in.gpu_va;
                            apc.x_q_va          = d.x_q_va;
                            apc.x_k_va          = d.x_k_va;
                            apc.x_v_va          = d.x_v_va;
                            apc.qkv_buf_va      = slc->qkv_buf;
                            apc.k_raw_va        = d.k_raw_va;
                            apc.v_raw_va        = d.v_raw_va;
                            apc.sink_proj_va    = can_fuse_sink ? d.sink_proj_va : 0ull;
                            apc.sinks_va        = d.sinks_va;
                            apc.k_conv_proj_va  = can_fuse_dyn ? d.k_conv_proj_va : 0ull;
                            apc.v_conv_proj_va  = can_fuse_dyn ? d.v_conv_proj_va : 0ull;
                            apc.k_dyn_va        = L.k_dyn;
                            apc.v_dyn_va        = L.v_dyn;
                            apc.q_prefix_rows   = C.q_prefix_rows;
                            apc.wk_rows         = C.row_cnt[4];
                            apc.wv_rows         = C.row_cnt[5];
                            apc.total_coal_rows = C.total_coal_rows;
                            apc.sink_wtype      = d.sink_proj_wtype;
                            apc.sink_row_B      = d.sink_proj_row_B;
                            apc.dyn_k_wtype     = d.k_proj_wtype;
                            apc.dyn_k_row_B     = d.k_proj_row_B;
                            apc.dyn_v_wtype     = d.v_proj_wtype;
                            apc.dyn_v_row_B     = d.v_proj_row_B;
                            const agx_barrier_mode b_kind = (!need_extra_sink && !need_extra_dyn)
                                ? AGX_BARRIER_STRONG_USC : AGX_BARRIER_NONE;
                            if (!agx_cdm_dispatch(stream, eng->k_w5_attn_in_b1,
                                                  (tot_rows + 7u) / 8u, 1u, 1u,
                                                  &apc, sizeof(apc), b_kind)) {
                                return false;
                            }
                            if (need_extra_sink) {
                                if (!dispatch_dense_mm(ctx, stream, layer.attn_sink_proj, d.x_q_va, d.sinks_va, 1u, AGX_BARRIER_NONE)) {
                                    return false;
                                }
                            }
                            if (need_extra_dyn) {
                                if (!dispatch_dense_mm(ctx, stream, layer.attn_k_conv_proj, d.x_k_va, L.k_dyn, 1u, AGX_BARRIER_NONE)) return false;
                                if (!dispatch_dense_mm(ctx, stream, layer.attn_v_conv_proj, d.x_v_va, L.v_dyn, 1u, AGX_BARRIER_NONE)) return false;
                            }
                            if (need_extra_sink || need_extra_dyn) {
                                agx_cdm_barrier_usc(stream);
                            }
                        }

                        // Op 2: Lean <= 56 GPR32 40-WG x 512T Fused qkv_prep_store + uvec4 RelBias + 16-SG FlashAttn + Qgate
                        // OR Wave 10 O(n_kv) Banded Split-K Flash-Decode for Full-Attn passes at n_kv = 2k..250K+!
                        agx_w5_persist_push ppc{};
                        ppc.descs_va   = descs_gpu_va + (uint64_t) v * sizeof(agx_w5_pass_desc);
                        ppc.barrier_va = slc->sense_barrier_va;
                        ppc.n_passes   = 1u;
                        ppc.n_wg       = 40u;
                        if (pass_use_b1_hybrid_fa[v]) {
                            if (!agx_cdm_dispatch(stream, eng->k_w5_hybrid_qkv_flash[kv_q8_0],
                                                  40u, 1u, 1u, &ppc, sizeof(ppc), AGX_BARRIER_STRONG_USC)) {
                                return false;
                            }
                        } else {
                            // Long-context Full-Attn pass (B=1): qkv_prep_store + banded rel_table + 40-core split-K PASS_MODE=0!
                            const uint64_t q_raw_b1     = slc->qkv_buf + (uint64_t) C.row_off[0] * sizeof(float);
                            const uint64_t qgate_raw_b1 = slc->qkv_buf + (uint64_t) C.row_off[1] * sizeof(float);
                            const uint64_t r_raw_b1     = slc->qkv_buf + (uint64_t) C.row_off[2] * sizeof(float);
                            const bool     need_rot64   = (P.k_rot_dim == 64u || P.v_rot_dim == 64u);
                            const agx_kernel * k_qkv    = need_rot64 ? ensure_w10_qkv_rot64_kernel(eng) : eng->k_w4_qkv_prep_store;
                            if (!k_qkv) return false;

                            agx_w4_qkv_prep_desc qd{};
                            qd.q_raw_va            = q_raw_b1;
                            qd.k_raw_va            = d.k_raw_va;
                            qd.v_raw_va            = d.v_raw_va;
                            qd.r_raw_va            = r_raw_b1;
                            qd.sinks_va            = d.sinks_va;
                            qd.qgate_raw_va        = qgate_raw_b1;
                            qd.k_dyn_va            = d.k_dyn_va;
                            qd.v_dyn_va            = d.v_dyn_va;
                            qd.q_clip_va           = d.q_clip_va;
                            qd.r_clip_va           = d.r_clip_va;
                            qd.k_conv_w_va         = d.k_conv_w_va;
                            qd.v_conv_w_va         = d.v_conv_w_va;
                            qd.rel_bias_va         = d.rel_bias_va;
                            qd.inp_pos_va          = G.inp_pos_va;
                            qd.r_l_va              = P.r_l_va;
                            qd.s_copy_va           = G.s_copy_va;
                            qd.k_cell_va           = P.cell_va[0];
                            qd.v_cell_va           = P.cell_va[1];
                            qd.k_store_va          = P.k_store_va;
                            qd.v_store_va          = P.v_store_va;
                            qd.k_idxs_va           = P.k_idxs_va;
                            qd.v_idxs_va           = P.v_idxs_va;
                            qd.q_out_va            = L.q_out;
                            qd.r_out_va            = L.r_out;
                            qd.rel_pos_va          = P.rel_pos_va;
                            qd.base_mask_va        = P.base_mask_va;
                            qd.kq_mask_out_va      = P.graph_mask_va;
                            qd.q_raw_stride_f32    = C.q_prefix_rows;
                            qd.k_raw_stride_f32    = C.row_cnt[4];
                            qd.v_raw_stride_f32    = C.row_cnt[5];
                            qd.r_raw_stride_f32    = C.q_prefix_rows;
                            qd.sinks_stride_f32    = 24u;
                            qd.r_nb1_f32           = P.r_nb1_f32;
                            qd.rollback_slots      = P.rollback_slots;
                            qd.rollback_stride_f32 = P.rollback_stride_f32;
                            qd.k_store_nb1_B       = P.k_store_nb1;
                            qd.v_store_nb1_B       = P.v_store_nb1;
                            qd.k_ids_nb0_B         = P.k_ids_nb0;
                            qd.v_ids_nb0_B         = P.v_ids_nb0;
                            qd.n_tokens            = 1u;
                            qd.n_conv              = 4u;
                            qd.n_rel               = d.n_rel;
                            qd.n_extent            = d.n_extent;
                            qd.n_kv                = P.n_kv;
                            qd.n_kv_mask_stride    = P.graph_mask_nb1 / 2u;
                            qd.rpos_nb1_B          = P.rpos_nb1;
                            qd.bmask_nb1_B         = P.bmask_nb1;
                            qd.bias_wtype          = d.bias_wtype;
                            qd.bias_nb1_B          = d.bias_nb1_B;
                            qd.flags               = (!P.is_swa ? AGX_W4_QKV_FLAG_IS_FULL_ATTN : 0u) |
                                                     (P.k_rot_dim == 128u ? AGX_W4_QKV_FLAG_FWHT_K : 0u) |
                                                     (P.k_rot_dim ==  64u ? AGX_W10_QKV_FLAG_FWHT64_K : 0u) |
                                                     (P.v_rot_dim == 128u ? AGX_W4_QKV_FLAG_FWHT_V : 0u) |
                                                     (P.v_rot_dim ==  64u ? AGX_W10_QKV_FLAG_FWHT64_V : 0u) |
                                                     (P.kv_is_q8 ? AGX_W4_QKV_FLAG_KV_Q8_0 : 0u) |
                                                     (P.k_ids_is_i64 ? AGX_W4_QKV_FLAG_K_IDS_I64 : 0u) |
                                                     (P.v_ids_is_i64 ? AGX_W4_QKV_FLAG_V_IDS_I64 : 0u) |
                                                     (P.bmask_is_f16 ? AGX_W4_QKV_FLAG_BASE_MASK_F16 : 0u);
                            qd.r_splits            = 1u;
                            qd.conv_res_scale      = d.kv_conv_res_scale;
                            qd.qk_clip             = d.qk_clip;
                            qd.rel_clip            = d.rel_clip;
                            qd.rms_eps             = d.rms_eps;
                            qd.log_scale_alpha     = d.log_scale_alpha;
                            qd.inv_log_floor       = d.inv_log_floor;
                            void * qd_cpu = nullptr;
                            uint64_t qd_va = agx_cdm_alloc_args(stream, sizeof(qd), 16, &qd_cpu);
                            if (!qd_va || !qd_cpu) return false;
                            std::memcpy(qd_cpu, &qd, sizeof(qd));
                            agx_w4_desc_va_push qpc{ qd_va };
                            if (!agx_cdm_dispatch(stream, k_qkv, 64u, 1u, 1u, &qpc, sizeof(qpc), AGX_BARRIER_STRONG_USC)) {
                                return false;
                            }

                            if (P.has_banded_rel) {
                                agx_w10_rel_table_push rtc{};
                                rtc.r_va             = L.r_out;
                                rtc.rel_bias_va      = d.rel_bias_va;
                                rtc.dst_va           = P.rel_table_va;
                                rtc.sinks_src_va     = 0ull;
                                rtc.sinks_dst_va     = 0ull;
                                rtc.n_rel            = d.n_rel;
                                rtc.eff_rows         = P.rel_table_len;
                                rtc.n_tokens         = 1u;
                                rtc.bias_wtype       = d.bias_wtype;
                                rtc.bias_nb1_B       = d.bias_nb1_B;
                                rtc.sinks_stride_f32 = 24u;
                                if (!agx_cdm_dispatch(stream, eng->k_w10_rel_bias_table,
                                                      (P.rel_table_len + 63u) / 64u, 1u, 1u,
                                                      &rtc, sizeof(rtc), AGX_BARRIER_STRONG_USC)) {
                                    return false;
                                }
                            } else {
                                agx_w4_rel_mask_push rmc{};
                                rmc.r_va             = L.r_out;
                                rmc.rel_bias_va      = d.rel_bias_va;
                                rmc.sinks_va         = d.sinks_va;
                                rmc.rel_pos_va       = P.rel_pos_va;
                                rmc.base_mask_va     = P.base_mask_va;
                                rmc.dst_va           = P.graph_mask_va;
                                rmc.n_rel            = d.n_rel;
                                rmc.n_extent         = d.n_extent;
                                rmc.n_kv             = P.n_kv;
                                rmc.n_kv_stride      = P.graph_mask_nb1 / 2u;
                                rmc.n_head           = 24u;
                                rmc.n_tokens         = 1u;
                                rmc.sinks_stride_f32 = 24u;
                                rmc.rpos_nb1_B       = P.rpos_nb1;
                                rmc.bmask_nb1_B      = P.bmask_nb1;
                                rmc.bias_wtype       = d.bias_wtype;
                                rmc.bias_nb1_B       = d.bias_nb1_B;
                                rmc.base_mask_is_f16 = P.bmask_is_f16 ? 1u : 0u;
                                if (!agx_cdm_dispatch(stream, eng->k_w4_rel_bias_mask,
                                                      (P.n_kv + 63u) / 64u, 24u, 1u,
                                                      &rmc, sizeof(rmc), AGX_BARRIER_STRONG_USC)) {
                                    return false;
                                }
                            }

                            agx_attn_stage_args attn{};
                            attn.n_tokens      = 1u;
                            attn.n_kv          = P.n_kv;
                            attn.n_head_q      = 24u;
                            attn.n_head_kv     = 8u;
                            attn.head_dim      = 128u;
                            attn.kq_scale      = 1.0f / std::sqrt(128.0f);
                            attn.flags         = (P.v_rot_dim == 64u ? AGX_ATTN_FLAG_FWHT64 : 0u) |
                                                 (P.v_rot_dim != 128u ? AGX_ATTN_FLAG_QGATE_F32 : 0u);
                            attn.q_va          = L.q_out;
                            attn.q_nb1         = 24u * 128u * sizeof(float);
                            attn.q_nb2         = 128u * sizeof(float);
                            attn.out_va        = slc->attn_out;
                            attn.q_gate_va     = (P.v_rot_dim != 128u) ? qgate_raw_b1 : 0ull;
                            attn.ext_k_dtype   = P.kv_is_q8 ? AGX_KV_Q8_0_GGML : AGX_KV_F16;
                            attn.ext_v_dtype   = P.kv_is_q8 ? AGX_KV_Q8_0_GGML : AGX_KV_F16;
                            attn.ext_k_va      = P.k_va;
                            attn.ext_v_va      = P.v_va;
                            attn.ext_k_nb1     = P.k_nb1;
                            attn.ext_k_nb2     = P.k_nb2;
                            attn.ext_v_nb1     = P.v_nb1;
                            attn.ext_v_nb2     = P.v_nb2;
                            if (P.has_banded_rel) {
                                attn.flags          |= AGX_ATTN_FLAG_BANDED_REL_BIAS |
                                                       AGX_ATTN_FLAG_SINK_PER_TOK_HEAD |
                                                       (P.bmask_is_f16 ? AGX_ATTN_FLAG_HAS_MASK_F16 : AGX_ATTN_FLAG_HAS_MASK_F32);
                                attn.n_rel_extent    = P.rel_extent;
                                attn.rel_table_len   = P.rel_table_len;
                                attn.rel_table_va    = P.rel_table_va;
                                attn.rel_pos_pack_va = P.rel_pos_pack_va;
                                attn.rel_cp_off      = 0u;
                                attn.rel_qp_off      = P.n_kv;
                                attn.rel_tm_off      = P.n_kv + 1u;
                                attn.sinks_va        = d.sinks_va;
                                attn.kq_mask_va      = P.base_mask_va;
                                attn.ext_mask_nb1    = P.bmask_nb1;
                                attn.ext_mask_nb2    = 0u;
                                attn.ext_mask_ne2    = 1u;
                            } else {
                                attn.flags          |= AGX_ATTN_FLAG_HAS_MASK_F16 | AGX_ATTN_FLAG_REL_TABLE_SINK;
                                attn.kq_mask_va      = P.graph_mask_va;
                                attn.ext_mask_nb1    = P.graph_mask_nb1;
                                attn.ext_mask_nb2    = P.graph_mask_nb1;
                                attn.ext_mask_ne2    = 24u;
                            }
                            if (!agx_kv_hybrid_dispatch_full_attn(kv_pool, stream, attn, AGX_BARRIER_STRONG_USC)) {
                                return false;
                            }
                            if (P.v_rot_dim == 128u) {
                                agx_w4_fwht_qgate_push fpc{};
                                fpc.attn_va          = slc->attn_out;
                                fpc.qgate_va         = qgate_raw_b1;
                                fpc.n_head           = 24u;
                                fpc.n_tokens         = 1u;
                                fpc.qgate_stride_f32 = C.q_prefix_rows;
                                fpc.do_fwht128       = 1u;
                                if (!agx_cdm_dispatch(stream, eng->k_w4_fwht_qgate, 24u, 1u, 1u,
                                                      &fpc, sizeof(fpc), AGX_BARRIER_STRONG_USC)) {
                                    return false;
                                }
                            }
                        }

                        // Op 3: Full-wire-speed 515 GB/s W_o
                        if (!dispatch_dense_mm(ctx, stream, layer.wo, slc->attn_out, L.x_k, 1u, AGX_BARRIER_STRONG_USC)) {
                            return false;
                        }

                        // Op 4: 40-WG x 512T (lean <= 52 GPR32!) Fused stage_post_attn + [FFN loop ctrl] + uvec4 router + Top-8
                        if (!agx_cdm_dispatch(stream, eng->k_w5_hybrid_post_attn_rtr[lp_i],
                                              40u, 1u, 1u, &ppc, sizeof(ppc), AGX_BARRIER_STRONG_USC)) {
                            return false;
                        }

                        // Op 5 & 6: Full-wire-speed 475 GB/s Fused 2-Stage MoE (gate_up+scale2+SiTU -> down+scale2+wsum)
                        {
                            agx_weight_layout gu_l = (pass_moe_idx[v] == 1u) ? AGX_WEIGHT_Q8_0_REPACKED : AGX_WEIGHT_NVFP4_REPACKED;
                            agx_moe_stage_args margs{};
                            margs.n_tokens         = 1u;
                            margs.n_embd           = D;
                            margs.n_ff             = hp.n_ff_exp;
                            margs.n_expert         = hp.n_expert;
                            margs.top_k            = hp.n_expert_used;
                            margs.act_op           = AGX_MOE_ACT_SITU;
                            margs.situ_beta        = hp.situ_beta;
                            margs.situ_linear_beta = hp.situ_linear_beta;
                            margs.exp_scale0       = 1.0f;
                            margs.ffn_x_va         = L.x_q;
                            margs.topk_ids_va      = slc->router_topk;
                            margs.ids_row_stride   = hp.n_expert_used;
                            margs.topk_w_va        = slc->router_topk_w;
                            margs.moe_out_va       = L.moe_out;
                            margs.gate_up_va       = d.gu_exps_va;
                            margs.down_va          = d.dn_exps_va;
                            margs.gate_up_slice_B  = d.gu_slice_B;
                            margs.down_slice_B     = d.dn_slice_B;
                            margs.gate_up_layout   = gu_l;
                            margs.down_layout      = gu_l;
                            margs.gate_up_scale_va = d.gu_scale_va;
                            margs.down_scale_va    = d.dn_scale_va;
                            margs.x_is_f16         = false;
                            margs.accumulate_out   = false;
                            margs.moe_inter_va     = slc->moe_inter;
                            margs.res_va           = 0ull;
                            if (!agx_moe_dispatch_fused_situ_down(moe_runner, stream, margs, AGX_BARRIER_STRONG_USC)) {
                                return false;
                            }
                        }

                        // Op 7: Fused stage_post_ffn[v] + pass[v+1] QKV loop ctrl (v=17..40) OR lean stage_post_ffn
                        if (v < 47u && G.passes[v + 1u].has_loop) {
                            if (!agx_cdm_dispatch(stream, eng->k_w5_hybrid_post_ffn_ctrl,
                                                  40u, 1u, 1u, &ppc, sizeof(ppc), AGX_BARRIER_STRONG_USC)) {
                                return false;
                            }
                        } else {
                            agx_w4_stage_prep_desc sd{};
                            sd.conv_src_va         = L.moe_out;
                            sd.res_va              = va_ffn_inp;
                            sd.conv_w_va           = d.ffn_post_conv_w_va;
                            sd.r_l_va              = P.r_l_va;
                            sd.s_copy_va           = G.s_copy_va;
                            sd.cell_va             = P.cell_va[3];
                            sd.cur_out_va          = d.next_cur_va;
                            sd.export_va           = d.export_l_out_va;
                            sd.block_ptrs_va       = d.ffn_bptrs_va;
                            sd.query_va            = d.next_attn_query_va;
                            for (int q = 0; q < 3; ++q) {
                                sd.norm_w_va[q]  = d.next_norm_w_va[q];
                                sd.mix_out_va[q] = d.next_mix_out_va[q];
                            }
                            sd.mix_out_export_va   = d.next_mix_export_va;
                            sd.n_embd              = D;
                            sd.n_tokens            = 1u;
                            sd.n_conv              = 4u;
                            sd.conv_ch_offset      = 4352u;
                            sd.r_nb1_f32           = P.r_nb1_f32;
                            sd.rollback_slots      = P.rollback_slots;
                            sd.rollback_stride_f32 = P.rollback_stride_f32;
                            sd.n_blocks            = d.post_ffn_n_blocks;
                            sd.n_queries           = d.post_ffn_n_queries;
                            sd.flags               = AGX_W4_SPREP_FLAG_DO_CONV | AGX_W4_SPREP_FLAG_ADD_RESIDUAL |
                                                     ((d.flags & AGX_W5_PFLAG_POST_FFN_DO_MIX)  ? AGX_W4_SPREP_FLAG_DO_DEPTH_MIX : 0u) |
                                                     ((d.flags & AGX_W5_PFLAG_POST_FFN_DO_NORM) ? AGX_W4_SPREP_FLAG_DO_NORM      : 0u);
                            sd.conv_res_scale      = d.post_conv_res_scale;
                            sd.inv_sqrt_d          = 1.0f / std::sqrt((float) D);
                            sd.eps_mix             = 1e-6f;
                            sd.eps_norm            = hp.f_norm_rms_eps;
                            void * sd_cpu = nullptr;
                            uint64_t sd_va = agx_cdm_alloc_args(stream, sizeof(sd), 16, &sd_cpu);
                            if (!sd_va || !sd_cpu) return false;
                            std::memcpy(sd_cpu, &sd, sizeof(sd));
                            agx_w4_desc_va_push spc{ sd_va };
                            if (!agx_cdm_dispatch(stream, eng->k_w4_stage_post_ffn, 1u, 1u, 1u,
                                                  &spc, sizeof(spc), AGX_BARRIER_STRONG_USC)) {
                                return false;
                            }
                        }

                        if (!P.layer_inp_stores.empty()) {
                            if (!emit_layer_inp_stores(P.layer_inp_stores, d.next_cur_va)) return false;
                        }
                    }
                    if (!G.out_norm_stores.empty()) {
                        if (!emit_layer_inp_stores(G.out_norm_stores, L.x_q)) return false;
                    }
                } else if (use_chunk) {
                    uint32_t v = 0u;
                    while (v < 48u) {
                        const uint32_t moe_i = pass_moe_idx[v];
                        const uint32_t lp_i  = G.passes[v].has_loop ? 1u : 0u;
                        uint32_t v_end = v + 1u;
                        while (v_end < 48u && (v_end - v) < max_chunk &&
                               pass_moe_idx[v_end] == moe_i &&
                               (G.passes[v_end].has_loop ? 1u : 0u) == lp_i &&
                               G.passes[v_end - 1u].layer_inp_stores.empty()) {
                            ++v_end;
                        }
                        agx_w5_persist_push ppc{};
                        ppc.descs_va   = descs_gpu_va + (uint64_t) v * sizeof(agx_w5_pass_desc);
                        ppc.barrier_va = slc->sense_barrier_va;
                        ppc.n_passes   = v_end - v;
                        ppc.n_wg       = 40u;
                        if (!agx_cdm_dispatch(stream, eng->k_w5_persist_chunk[kv_q8_0][moe_i][lp_i],
                                              40u, 1u, 1u, &ppc, sizeof(ppc), AGX_BARRIER_STRONG_USC)) {
                            return false;
                        }
                        if (!G.passes[v_end - 1u].layer_inp_stores.empty()) {
                            if (!emit_layer_inp_stores(G.passes[v_end - 1u].layer_inp_stores,
                                                       descs[v_end - 1u].next_cur_va)) {
                                return false;
                            }
                        }
                        v = v_end;
                    }
                    if (!G.out_norm_stores.empty()) {
                        if (!emit_layer_inp_stores(G.out_norm_stores, L.x_q)) return false;
                    }
                } else {
                    for (uint32_t v = 0u; v < 48u; ++v) {
                        const uint32_t moe_i = pass_moe_idx[v];
                        const uint32_t lp_i  = G.passes[v].has_loop ? 1u : 0u;
                        agx_w5_persist_push ppc{};
                        ppc.descs_va   = descs_gpu_va + (uint64_t) v * sizeof(agx_w5_pass_desc);
                        ppc.barrier_va = slc->sense_barrier_va;
                        ppc.n_passes   = 1u;
                        ppc.n_wg       = 40u;
                        if (!agx_cdm_dispatch(stream, eng->k_w5_persist_attn[kv_q8_0][lp_i],
                                              40u, 1u, 1u, &ppc, sizeof(ppc), AGX_BARRIER_STRONG_USC)) {
                            return false;
                        }
                        if (!agx_cdm_dispatch(stream, eng->k_w5_persist_moe[moe_i][lp_i],
                                              40u, 1u, 1u, &ppc, sizeof(ppc), AGX_BARRIER_STRONG_USC)) {
                            return false;
                        }
                        if (!G.passes[v].layer_inp_stores.empty()) {
                            if (!emit_layer_inp_stores(G.passes[v].layer_inp_stores, descs[v].next_cur_va)) {
                                return false;
                            }
                        }
                    }
                    if (!G.out_norm_stores.empty()) {
                        if (!emit_layer_inp_stores(G.out_norm_stores, L.x_q)) return false;
                    }
                }
                ran_w5_persistent = true;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Wave 6 + Wave 9: Multi-Token (M = 2..8 DSpark Verify & n=7!) 40-WG SENSE_BARRIER
    // Hybrid Super-Stages (`HYBRID_MODE = 3, 4` HAS_LOOP=0|1 -> cuts -336 disp & 24 serial post_ffn!)
    // ------------------------------------------------------------------------
    const char * m_max4_env = std::getenv("GGML_AGX_M_HYBRID_MAX4");
    const uint32_t m_hyb_max = (m_max4_env && m_max4_env[0] && m_max4_env[0] != '0') ? 4u : 8u;
    const char * no_mpffn_env = std::getenv("GGML_AGX_NO_M_POST_FFN_PAR");
    const bool   no_m_post_ffn_par = (no_mpffn_env && no_mpffn_env[0] && no_mpffn_env[0] != '0');
    const char * no_attnm_env = std::getenv("GGML_AGX_NO_ATTN_IN_M");
    const bool   no_attn_in_m = (no_attnm_env && no_attnm_env[0] && no_attnm_env[0] != '0');
    const char * v_f32_ain_env = std::getenv("GGML_AGX_VERIFY_F32_ATTN_IN");
    const bool   verify_f32_attn_in = (v_f32_ain_env && v_f32_ain_env[0] && v_f32_ain_env[0] != '0');
    const char * v_leg_wo_env  = std::getenv("GGML_AGX_VERIFY_LEGACY_WO");
    const bool   verify_legacy_wo   = (v_leg_wo_env && v_leg_wo_env[0] && v_leg_wo_env[0] != '0');
    const char * v_leg_lm_env  = std::getenv("GGML_AGX_VERIFY_LEGACY_LM");
    const bool   verify_legacy_lm   = (v_leg_lm_env && v_leg_lm_env[0] && v_leg_lm_env[0] != '0');

    uint64_t w5m_descs_gpu_va = 0ull;
    bool     use_w5_m_hybrid  = false;
    if (!ran_w5_persistent && M >= 2u && M <= m_hyb_max &&
        !(no_persist_env && no_persist_env[0] && no_persist_env[0] != '0') &&
        D == 2304u && hp.gate_rank == 64u && hp.n_expert <= 256u && hp.n_expert_used == 8u &&
        ensure_w5_hybrid_m_kernels(eng)) {
        agx_w5_pass_desc mdescs[48]{};
        bool m_ok = true;
        std::vector<uint64_t> m_blocks;
        m_blocks.reserve(6);
        std::vector<uint64_t> m_bank;
        m_bank.reserve(2);
        uint64_t m_cur_va = va_h_0;
        bool     m_attn_ready = true;

        for (uint32_t v = 0u; v < 48u && m_ok; ++v) {
            const auto & P     = G.passes[v];
            const auto & layer = W.layers[P.ip];
            const agx_inkling2_loop_tensors * loop = P.has_loop ? &W.loop_layers[P.ip - 6u] : nullptr;
            if (P.has_loop) {
                for (int c = 0; c < 4; ++c) {
                    if (loop->ctrl_down[c]->type   != GGML_TYPE_F32 ||
                        loop->ctrl_gain[c]->type   != GGML_TYPE_F32 ||
                        loop->ctrl_forget[c]->type != GGML_TYPE_F32) {
                        m_ok = false; break;
                    }
                }
                if (!m_ok) break;
            }
            if (v == 18u || v == 30u) {
                m_bank.clear();
                m_bank.push_back((v == 18u) ? va_h12_even : va_h12_odd);
                m_blocks.clear();
                m_blocks.push_back(va_h_0);
                m_cur_va = va_prelude;
                m_attn_ready = false;
            }
            const bool need_qkv_mix = (P.has_loop && !m_attn_ready);
            if (need_qkv_mix && !P.is_swa) { m_ok = false; break; }

            auto & d = mdescs[v];
            d.x_q_va           = L.x_q;
            d.x_k_va           = (P.is_swa && !P.has_loop) ? L.x_q : L.x_k;
            d.x_v_va           = (P.is_swa && !P.has_loop) ? L.x_q : L.x_v;
            d.qkv_in_va[0]     = L.qkv_in_0;
            d.qkv_in_va[1]     = L.qkv_in_1;
            d.qkv_in_va[2]     = L.qkv_in_2;
            d.wo_out_va        = L.x_k;
            d.ffn_inp_va       = va_ffn_inp;
            d.router_logits_va = L.router_logits;
            d.router_topk_va   = slc->router_topk;
            d.router_topk_w_va = slc->router_topk_w;
            d.moe_out_va       = L.moe_out;
            d.h_msg_va         = slc->h_msg;
            for (int j = 0; j < 3; ++j) {
                d.ctrl_cat_va[j]  = L.ctrl_cat[j];
                d.ctrl_gate_va[j] = L.ctrl_gate[j];
                d.ctrl_gain_va[j] = L.ctrl_gain[j];
            }
            if (P.has_loop) {
                d.msg_state_va[0] = m_bank[0];
                d.msg_state_va[1] = va_bank_out;
                d.msg_query_va    = agx_tensor_gpu_va(loop->msg_query);
                if (need_qkv_mix) {
                    d.qkv_mix_cur_va   = m_cur_va;
                    d.qkv_mix_bptrs_va = alloc_block_ptrs(m_blocks);
                    d.qkv_mix_query_va = agx_tensor_gpu_va(layer.attn_res_query);
                    d.qkv_mix_n_blocks = (uint32_t) m_blocks.size();
                    if (!d.qkv_mix_bptrs_va) { m_ok = false; break; }
                }
                for (int c = 0; c < 4; ++c) {
                    d.ctrl_down_w_va[c]   = agx_tensor_gpu_va(loop->ctrl_down[c]);
                    d.ctrl_gain_w_va[c]   = agx_tensor_gpu_va(loop->ctrl_gain[c]);
                    d.ctrl_forget_w_va[c] = agx_tensor_gpu_va(loop->ctrl_forget[c]);
                }
                d.attn_norm_w_va[0] = agx_tensor_gpu_va(P.is_swa ? layer.attn_norm : layer.attn_q_inp_norm);
                d.attn_norm_w_va[1] = agx_tensor_gpu_va(P.is_swa ? layer.attn_norm : layer.attn_k_inp_norm);
                d.attn_norm_w_va[2] = agx_tensor_gpu_va(P.is_swa ? layer.attn_norm : layer.attn_v_inp_norm);
            }
            d.ffn_norm_w_va = agx_tensor_gpu_va(layer.ffn_norm);
            d.r_l_va        = P.r_l_va;
            d.s_copy_va     = G.s_copy_va;
            for (int c = 0; c < 4; ++c) d.cell_va[c] = P.cell_va[c];

            const bool new_block = (P.ip % hp.attn_res_block_size == 0u);
            if (new_block) m_blocks.push_back(m_cur_va);
            d.attn_post_conv_w_va  = agx_tensor_gpu_va(layer.attn_post_conv);
            d.attn_res_va          = new_block ? 0ull : m_cur_va;
            d.attn_bptrs_va        = alloc_block_ptrs(m_blocks);
            d.ffn_res_query_va     = agx_tensor_gpu_va(layer.ffn_res_query);
            d.post_attn_mix_out_va = !P.has_loop ? L.x_q : L.qkv_in_0;
            d.post_attn_n_blocks   = (uint32_t) m_blocks.size();
            if (!d.attn_bptrs_va) { m_ok = false; break; }

            d.router_w_va    = agx_tensor_gpu_va(layer.ffn_gate_inp);
            d.router_wtype   = agx_inkling2_tensor_wtype(layer.ffn_gate_inp, &d.router_row_B);
            d.exp_probs_b_va = layer.ffn_exp_probs_b ? agx_tensor_gpu_va(layer.ffn_exp_probs_b) : 0ull;
            if (d.router_wtype > 2u) { m_ok = false; break; }

            const uint64_t next_cur_va = target_l_out_va(v, m_cur_va);
            const bool is_pass_reset   = (v == 17u || v == 29u);
            const bool is_final_pass   = (v == 47u);
            d.ffn_post_conv_w_va       = agx_tensor_gpu_va(layer.ffn_post_conv);
            d.next_cur_va              = next_cur_va;
            d.export_l_out_va          = P.export_l_out_va;

            bool post_ffn_do_mix  = false;
            bool post_ffn_do_norm = false;
            bool emit_x_f16 = P.has_loop;
            if (P.has_loop) {
                d.next_mix_f16_va[0] = L.x_q_f16;
                d.next_mix_f16_va[1] = L.x_k_f16;
                d.next_mix_f16_va[2] = L.x_v_f16;
            }
            if (is_final_pass) {
                d.ffn_bptrs_va       = alloc_block_ptrs(m_blocks);
                d.next_attn_query_va = agx_tensor_gpu_va(W.output_res_query);
                d.next_norm_w_va[0]  = agx_tensor_gpu_va(W.output_norm);
                d.next_mix_out_va[0] = L.x_q;
                d.next_mix_f16_va[0] = L.x_q_f16;
                d.next_mix_export_va = G.export_out_norm;
                d.post_ffn_n_blocks  = (uint32_t) m_blocks.size();
                d.post_ffn_n_queries = 1u;
                post_ffn_do_mix      = true;
                post_ffn_do_norm     = true;
                emit_x_f16           = true;
                if (!d.ffn_bptrs_va) { m_ok = false; break; }
            } else if (!is_pass_reset) {
                const auto & Pn      = G.passes[v + 1u];
                const auto & Ln      = W.layers[Pn.ip];
                d.ffn_bptrs_va       = alloc_block_ptrs(m_blocks);
                d.next_attn_query_va = agx_tensor_gpu_va(Ln.attn_res_query);
                d.post_ffn_n_blocks  = (uint32_t) m_blocks.size();
                d.post_ffn_n_queries = Pn.is_swa ? 1u : 3u;
                post_ffn_do_mix      = true;
                if (Pn.has_loop) {
                    d.next_mix_out_va[0] = L.qkv_in_0;
                    d.next_mix_out_va[1] = L.qkv_in_1;
                    d.next_mix_out_va[2] = L.qkv_in_2;
                } else {
                    d.next_norm_w_va[0]  = agx_tensor_gpu_va(Pn.is_swa ? Ln.attn_norm : Ln.attn_q_inp_norm);
                    d.next_norm_w_va[1]  = !Pn.is_swa ? agx_tensor_gpu_va(Ln.attn_k_inp_norm) : 0ull;
                    d.next_norm_w_va[2]  = !Pn.is_swa ? agx_tensor_gpu_va(Ln.attn_v_inp_norm) : 0ull;
                    d.next_mix_out_va[0] = L.x_q;
                    d.next_mix_out_va[1] = L.x_k;
                    d.next_mix_out_va[2] = L.x_v;
                    d.next_mix_f16_va[0] = L.x_q_f16;
                    d.next_mix_f16_va[1] = L.x_k_f16;
                    d.next_mix_f16_va[2] = L.x_v_f16;
                    post_ffn_do_norm     = true;
                    emit_x_f16           = true;
                }
                m_attn_ready = true;
                if (!d.ffn_bptrs_va) { m_ok = false; break; }
            }

            d.flags = (P.is_swa         ? AGX_W5_PFLAG_IS_SWA            : 0u) |
                      (need_qkv_mix     ? AGX_W5_PFLAG_NEED_QKV_MIX      : 0u) |
                      (!new_block       ? AGX_W5_PFLAG_POST_ATTN_ADD_RES : 0u) |
                      (!P.has_loop      ? AGX_W5_PFLAG_POST_ATTN_DO_NORM : 0u) |
                      (post_ffn_do_mix  ? AGX_W5_PFLAG_POST_FFN_DO_MIX   : 0u) |
                      (post_ffn_do_norm ? AGX_W5_PFLAG_POST_FFN_DO_NORM  : 0u) |
                      (emit_x_f16       ? AGX_W5_PFLAG_EMIT_X_F16        : 0u);
            d.r_nb1_f32           = P.r_nb1_f32;
            d.n_expert            = hp.n_expert;
            d.n_ff_exp            = hp.n_ff_exp;
            d.post_conv_res_scale = 1.0f / std::sqrt(5.0f);
            d.rms_eps             = hp.f_norm_rms_eps;
            d.router_norm_eps     = hp.expert_weights_norm_eps;
            d.router_w_scale      = (hp.expert_weights_scale != 0.0f) ? hp.expert_weights_scale : 1.0f;
            d.n_tokens            = M;
            d.rollback_slots      = P.rollback_slots;
            d.rollback_stride_f32 = P.rollback_stride_f32;

            m_cur_va = next_cur_va;
        }
        if (m_ok) {
            void * md_cpu = nullptr;
            w5m_descs_gpu_va = agx_cdm_alloc_args(stream, sizeof(mdescs), AGX_CACHE_LINE_BYTES, &md_cpu);
            if (w5m_descs_gpu_va && md_cpu) {
                std::memcpy(md_cpu, mdescs, sizeof(mdescs));
                agx_slc_arena_reset_sense_barrier(slc, 40u);
                use_w5_m_hybrid = true;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Main 48-Pass Loop (v = 0..47) — Wave 4 / Wave 6 M=2..8 Hybrid Path
    // ------------------------------------------------------------------------
    bool w5m_qkv_ctrl_done = false;
    for (uint32_t v = 0; !ran_w5_persistent && v < 48u; ++v) {
        const auto & P     = G.passes[v];
        const auto & layer = W.layers[P.ip];
        const auto & C     = eng->w4_model.layers[P.ip];
        const agx_inkling2_loop_tensors * loop = P.has_loop ? &W.loop_layers[P.ip - 6u] : nullptr;

        // Pass 1 (v==18) & Pass 2 (v==30) reset cur = va_prelude, blocks = {va_h_0}, bank = {prev h12}
        if (v == 18u || v == 30u) {
            bank.clear();
            bank.push_back((v == 18u) ? va_h12_even : va_h12_odd);
            blocks.clear();
            blocks.push_back(va_h_0);
            cur_va = va_prelude;
            attn_inputs_ready = false;
        }

        // ---- Stage 0a: Ensure Q/K/V inputs are ready (only needed for v==18,30 or looped ctrl_gates!) ----
        const bool skip_qkv_ctrl = w5m_qkv_ctrl_done;
        w5m_qkv_ctrl_done = false;
        if (P.has_loop && !skip_qkv_ctrl) {
            x_f16_ready = false;
            // Compute cross-pass messages = mix(bank, va_bank_out, loop->msg_query) -> slc->h_msg
            uint64_t msg_states[2] = { bank[0], va_bank_out };
            if (!agx_inkling2_dispatch_depth_mix(eng, stream, msg_states, 2u,
                    agx_tensor_gpu_va(loop->msg_query), 4u, 0ull, slc->h_msg, 0ull,
                    D, M, 1e-6f, hp.f_norm_rms_eps, AGX_BARRIER_NONE)) {
                return false;
            }
            if (!attn_inputs_ready) {
                // Only happens at v==18 and v==30 (start of Pass 1/2 where cur reset to va_prelude!)
                agx_w4_stage_prep_desc sd{};
                sd.cur_out_va    = cur_va;
                sd.block_ptrs_va = alloc_block_ptrs(blocks);
                sd.query_va      = agx_tensor_gpu_va(layer.attn_res_query);
                sd.mix_out_va[0] = L.qkv_in_0;
                sd.mix_out_va[1] = L.qkv_in_1;
                sd.mix_out_va[2] = L.qkv_in_2;
                sd.n_embd        = D;
                sd.n_tokens      = M;
                sd.n_blocks      = (uint32_t) blocks.size();
                sd.n_queries     = P.is_swa ? 1u : 3u;
                sd.flags         = AGX_W4_SPREP_FLAG_DO_DEPTH_MIX;
                sd.inv_sqrt_d    = 1.0f / std::sqrt((float) D);
                sd.eps_mix       = 1e-6f;
                sd.eps_norm      = hp.f_norm_rms_eps;
                void * d_cpu = nullptr;
                uint64_t d_va = agx_cdm_alloc_args(stream, sizeof(sd), 16, &d_cpu);
                if (!d_va || !d_cpu) return false;
                std::memcpy(d_cpu, &sd, sizeof(sd));
                agx_w4_desc_va_push pc{ d_va };
                if (!agx_cdm_dispatch(stream, eng->k_w4_stage_post_ffn, 1u, 1u, 1u,
                                      &pc, sizeof(pc), AGX_BARRIER_NONE)) {
                    return false;
                }
            }
            agx_cdm_barrier_usc(stream);

            // 40-Core Loop Controllers (Q, K, V): norm_cat -> 3x concurrent ctrl_down -> 6x concurrent ctrl_gain/forget -> mix_norm!
            // Eliminates the ~19 ms 1-3 WG single-core bandwidth wall on 226 MB of F32 controller weights!
            const uint64_t cur_vas[3] = {
                L.qkv_in_0,
                P.is_swa ? L.qkv_in_0 : L.qkv_in_1,
                P.is_swa ? L.qkv_in_0 : L.qkv_in_2
            };
            const struct ggml_tensor * norms[3] = {
                P.is_swa ? layer.attn_norm : layer.attn_q_inp_norm,
                P.is_swa ? layer.attn_norm : layer.attn_k_inp_norm,
                P.is_swa ? layer.attn_norm : layer.attn_v_inp_norm,
            };
            {
                agx_w4_ctrl_norm_cat_push ncp{};
                for (int j = 0; j < 3; ++j) {
                    ncp.cur_va[j] = cur_vas[j];
                    ncp.dst_va[j] = L.ctrl_cat[j];
                }
                ncp.msg_va   = slc->h_msg;
                ncp.n_embd   = D;
                ncp.n_tokens = M;
                ncp.n_ctrls  = 3u;
                ncp.msg_base = 0u;
                ncp.eps_ctrl = 1e-6f;
                if (!agx_cdm_dispatch(stream, eng->k_w4_ctrl_norm_cat, 3u, M, 1u,
                                      &ncp, sizeof(ncp), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
            }
            for (int j = 0; j < 3; ++j) {
                if (!dispatch_dense_mm(ctx, stream, loop->ctrl_down[j], L.ctrl_cat[j], L.ctrl_gate[j],
                                       M, (j == 2) ? AGX_BARRIER_STRONG_USC : AGX_BARRIER_NONE)) {
                    return false;
                }
            }
            const bool fused_qkv_up =
                loop->ctrl_gain[0]->type == GGML_TYPE_F32 && loop->ctrl_forget[0]->type == GGML_TYPE_F32 &&
                loop->ctrl_gain[1]->type == GGML_TYPE_F32 && loop->ctrl_forget[1]->type == GGML_TYPE_F32 &&
                loop->ctrl_gain[2]->type == GGML_TYPE_F32 && loop->ctrl_forget[2]->type == GGML_TYPE_F32 &&
                loop->ctrl_gain[0]->ne[0] == 64 && loop->ctrl_forget[0]->ne[0] == 64;
            const uint64_t out_vas[3] = { L.x_q, L.x_k, L.x_v };
            if (fused_qkv_up) {
                // 1 SINGLE 40-core coalesced dispatch replacing 6x K=64 GEMMs + mix!
                agx_w4_ctrl_up_mix_push ump{};
                for (int j = 0; j < 3; ++j) {
                    ump.cur_va[j]      = cur_vas[j];
                    ump.gate_va[j]     = L.ctrl_gate[j];
                    ump.gain_w_va[j]   = agx_tensor_gpu_va(loop->ctrl_gain[j]);
                    ump.forget_w_va[j] = agx_tensor_gpu_va(loop->ctrl_forget[j]);
                    ump.out_va[j]      = L.ctrl_gain[j];
                }
                ump.msg_va   = slc->h_msg;
                ump.n_embd   = D;
                ump.n_tokens = M;
                ump.n_ctrls  = 3u;
                ump.msg_base = 0u;
                if (!agx_cdm_dispatch(stream, eng->k_w4_ctrl_up_mix, (D + 15u) / 16u, 3u, M,
                                      &ump, sizeof(ump), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
                agx_w4_batch_rmsnorm_push brp{};
                for (int j = 0; j < 3; ++j) {
                    brp.src_va[j]    = L.ctrl_gain[j];
                    brp.norm_w_va[j] = agx_tensor_gpu_va(norms[j]);
                    brp.dst_va[j]    = out_vas[j];
                }
                brp.n_embd   = D;
                brp.n_tokens = M;
                brp.n_ctrls  = 3u;
                brp.eps_norm = hp.f_norm_rms_eps;
                if (!agx_cdm_dispatch(stream, eng->k_w4_batch_rmsnorm, 3u, M, 1u,
                                      &brp, sizeof(brp), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
            } else {
                for (int j = 0; j < 3; ++j) {
                    if (!dispatch_dense_mm(ctx, stream, loop->ctrl_gain[j],   L.ctrl_gate[j], L.ctrl_gain[j],
                                           M, AGX_BARRIER_NONE)) return false;
                    if (!dispatch_dense_mm(ctx, stream, loop->ctrl_forget[j], L.ctrl_gate[j], L.ctrl_forget[j],
                                           M, (j == 2) ? AGX_BARRIER_STRONG_USC : AGX_BARRIER_NONE)) return false;
                }
                agx_w4_ctrl_mix_norm_push mnp{};
                for (int j = 0; j < 3; ++j) {
                    mnp.cur_va[j]    = cur_vas[j];
                    mnp.gain_va[j]   = L.ctrl_gain[j];
                    mnp.forget_va[j] = L.ctrl_forget[j];
                    mnp.norm_w_va[j] = agx_tensor_gpu_va(norms[j]);
                    mnp.out_va[j]    = out_vas[j];
                }
                mnp.msg_va   = slc->h_msg;
                mnp.n_embd   = D;
                mnp.n_tokens = M;
                mnp.n_ctrls  = 3u;
                mnp.msg_base = 0u;
                mnp.eps_norm = hp.f_norm_rms_eps;
                if (!agx_cdm_dispatch(stream, eng->k_w4_ctrl_mix_norm, 3u, M, 1u,
                                      &mnp, sizeof(mnp), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
            }
        }

        const uint64_t x_q_va = L.x_q;
        const uint64_t x_k_va = (P.is_swa && !P.has_loop) ? L.x_q : L.x_k;
        const uint64_t x_v_va = (P.is_swa && !P.has_loop) ? L.x_q : L.x_v;

        // ---- Stage 0b: Coalesced W_attn_in GEMM -> slc->qkv_buf (1 dispatch for SWA !has_loop!) ----
        const struct ggml_tensor * mats_all[6] = {
            layer.wq, layer.wqkv_gate, layer.attn_rel, layer.attn_sink_proj, layer.wk, layer.wv
        };
        uint64_t raw_va[6]     = {};
        uint32_t raw_stride[6] = {};
        uint64_t fb_cur        = L.fallback_qkv;

        const bool use_coal     = C.valid && !disable_coal;
        const bool same_qkv_inp = (P.is_swa && !P.has_loop);
        bool ran_attn_in_m = false;
        if (!no_attn_in_m && M >= 2u && M <= 8u && use_coal &&
            C.in_coal[0] && C.in_coal[1] && C.in_coal[2] && C.in_coal[4] && C.in_coal[5]) {
            const agx_kernel * k_attn_f16 = (M <= 4u) ? eng->k_w9_attn_in_m4_f16 : eng->k_w9_attn_in_m8_f16;
            const agx_kernel * k_attn_f32 = (M <= 4u) ? eng->k_w9_attn_in_m4     : eng->k_w9_attn_in_m8;
            const bool use_f16_ain = x_f16_ready && !verify_f32_attn_in && (k_attn_f16 != nullptr);
            const agx_kernel * k_attn_m = use_f16_ain ? k_attn_f16 : k_attn_f32;
            uint32_t sink_rb = 0u, dk_rb = 0u, dv_rb = 0u;
            const uint32_t sink_wt = (!C.in_coal[3] && layer.attn_sink_proj)
                ? agx_inkling2_tensor_wtype(layer.attn_sink_proj, &sink_rb) : 0u;
            const uint32_t dk_wt   = (!P.is_swa && layer.attn_k_conv_proj)
                ? agx_inkling2_tensor_wtype(layer.attn_k_conv_proj, &dk_rb) : 0u;
            const uint32_t dv_wt   = (!P.is_swa && layer.attn_v_conv_proj)
                ? agx_inkling2_tensor_wtype(layer.attn_v_conv_proj, &dv_rb) : 0u;
            const size_t k_off = agx_align_up((size_t) M * C.q_prefix_rows * sizeof(float), AGX_CACHE_LINE_BYTES);
            const size_t v_off = k_off + agx_align_up((size_t) M * C.row_cnt[4] * sizeof(float), AGX_CACHE_LINE_BYTES);
            const size_t tot_B = v_off + agx_align_up((size_t) M * C.row_cnt[5] * sizeof(float), AGX_CACHE_LINE_BYTES);
            if (k_attn_m != nullptr && tot_B <= AGX_SLC_ARENA_QKV_BUF_BYTES &&
                (C.in_coal[3] || sink_wt <= 2u) && (P.is_swa || (dk_wt <= 2u && dv_wt <= 2u))) {
                const uint64_t k_va_out = slc->qkv_buf + k_off;
                const uint64_t v_va_out = slc->qkv_buf + v_off;
                for (uint32_t m = 0; m < 4u; ++m) {
                    if (C.in_coal[m]) {
                        raw_va[m]     = slc->qkv_buf + (uint64_t) C.row_off[m] * sizeof(float);
                        raw_stride[m] = C.q_prefix_rows;
                    }
                }
                if (!C.in_coal[3]) {
                    raw_va[3]     = fb_cur;
                    raw_stride[3] = 24u;
                    fb_cur += agx_align_up((size_t) M * 24u * sizeof(float), AGX_CACHE_LINE_BYTES);
                }
                raw_va[4]     = k_va_out;
                raw_stride[4] = C.row_cnt[4];
                raw_va[5]     = v_va_out;
                raw_stride[5] = C.row_cnt[5];

                agx_w5_attn_in_push apc{};
                apc.w_attn_in_va    = C.w_attn_in.gpu_va;
                apc.x_q_va          = x_q_va;
                apc.x_k_va          = x_k_va;
                apc.x_v_va          = x_v_va;
                apc.qkv_buf_va      = slc->qkv_buf;
                apc.k_raw_va        = k_va_out;
                apc.v_raw_va        = v_va_out;
                apc.sink_proj_va    = !C.in_coal[3] ? agx_tensor_gpu_va(layer.attn_sink_proj) : 0ull;
                apc.sinks_va        = !C.in_coal[3] ? raw_va[3] : 0ull;
                apc.k_conv_proj_va  = !P.is_swa ? agx_tensor_gpu_va(layer.attn_k_conv_proj) : 0ull;
                apc.v_conv_proj_va  = !P.is_swa ? agx_tensor_gpu_va(layer.attn_v_conv_proj) : 0ull;
                apc.k_dyn_va        = !P.is_swa ? L.k_dyn : 0ull;
                apc.v_dyn_va        = !P.is_swa ? L.v_dyn : 0ull;
                apc.q_prefix_rows   = C.q_prefix_rows;
                apc.wk_rows         = C.row_cnt[4];
                apc.wv_rows         = C.row_cnt[5];
                apc.total_coal_rows = C.total_coal_rows;
                apc.sink_wtype      = sink_wt;
                apc.sink_row_B      = sink_rb;
                apc.dyn_k_wtype     = dk_wt;
                apc.dyn_k_row_B     = dk_rb;
                apc.dyn_v_wtype     = dv_wt;
                apc.dyn_v_row_B     = dv_rb;
                apc.n_tokens        = M;
                apc.x_is_f16        = use_f16_ain ? 1u : 0u;
                apc.x_q_f16_va      = L.x_q_f16;
                apc.x_k_f16_va      = same_qkv_inp ? L.x_q_f16 : L.x_k_f16;
                apc.x_v_f16_va      = same_qkv_inp ? L.x_q_f16 : L.x_v_f16;
                const uint32_t tot_rows = C.total_coal_rows + (!C.in_coal[3] ? 24u : 0u) + (!P.is_swa ? 64u : 0u);
                if (!agx_cdm_dispatch(stream, k_attn_m, (tot_rows + 7u) / 8u, 1u, 1u,
                                      &apc, sizeof(apc), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
                ran_attn_in_m = true;
            }
        }
        if (!ran_attn_in_m) {
        if (use_coal && same_qkv_inp) {
            // 1 SINGLE DISPATCH for ALL coalesced Q8_0 attention projections reading x_q ONCE at 517 GB/s!
            if (!agx_gemm_dispatch_raw_q8_0(ctx, stream, C.w_attn_in.gpu_va, x_q_va, slc->qkv_buf,
                                            M, C.total_coal_rows, D, AGX_BARRIER_NONE, D, C.total_coal_rows)) {
                return false;
            }
            for (uint32_t m = 0; m < 6u; ++m) {
                if (C.in_coal[m]) {
                    raw_va[m]     = slc->qkv_buf + (uint64_t) C.row_off[m] * sizeof(float);
                    raw_stride[m] = C.total_coal_rows;
                } else {
                    const uint32_t N = (uint32_t) mats_all[m]->ne[1];
                    raw_va[m]     = fb_cur;
                    raw_stride[m] = N;
                    fb_cur += agx_align_up((size_t) M * N * sizeof(float), AGX_CACHE_LINE_BYTES);
                    if (!dispatch_dense_mm(ctx, stream, mats_all[m], x_q_va, raw_va[m], M, AGX_BARRIER_NONE)) {
                        return false;
                    }
                }
            }
        } else if (use_coal) {
            // Non-overlapping 128B-aligned slices inside slc->qkv_buf (eliminates AGX3B USC L1 false sharing!)
            const uint32_t pref_stride = C.q_prefix_rows;
            uint64_t qkv_cur = slc->qkv_buf;
            const uint64_t pref_va = qkv_cur;
            qkv_cur += agx_align_up((size_t) M * pref_stride * sizeof(float), AGX_CACHE_LINE_BYTES);
            // Prefix [0..3] (wq, wqkv_gate, attn_rel, attn_sink_proj) in 1 SINGLE DISPATCH on x_q!
            if (C.q_prefix_rows > 0u) {
                if (!agx_gemm_dispatch_raw_q8_0(ctx, stream, C.w_attn_in.gpu_va, x_q_va, pref_va,
                                                M, C.q_prefix_rows, D, AGX_BARRIER_NONE, D, pref_stride)) {
                    return false;
                }
            }
            // Suffix [4]=wk on x_k, [5]=wv on x_v (128B-aligned, 100% concurrent AGX_BARRIER_NONE!)
            for (uint32_t m = 0; m < 6u; ++m) {
                const uint64_t xin = (m < 4u) ? x_q_va : ((m == 4u) ? x_k_va : x_v_va);
                if (C.in_coal[m]) {
                    if (m < 4u) {
                        raw_va[m]     = pref_va + (uint64_t) C.row_off[m] * sizeof(float);
                        raw_stride[m] = pref_stride;
                    } else {
                        raw_va[m]     = qkv_cur;
                        raw_stride[m] = C.row_cnt[m];
                        qkv_cur += agx_align_up((size_t) M * C.row_cnt[m] * sizeof(float), AGX_CACHE_LINE_BYTES);
                        if (!agx_gemm_dispatch_raw_q8_0(ctx, stream, C.w_attn_in.gpu_va + C.byte_off[m],
                                                        xin, raw_va[m], M, C.row_cnt[m], D,
                                                        AGX_BARRIER_NONE, D, C.row_cnt[m])) {
                            return false;
                        }
                    }
                } else {
                    const uint32_t N = (uint32_t) mats_all[m]->ne[1];
                    raw_va[m]     = fb_cur;
                    raw_stride[m] = N;
                    fb_cur += agx_align_up((size_t) M * N * sizeof(float), AGX_CACHE_LINE_BYTES);
                    if (!dispatch_dense_mm(ctx, stream, mats_all[m], xin, raw_va[m], M, AGX_BARRIER_NONE)) {
                        return false;
                    }
                }
            }
        } else {
            for (uint32_t m = 0; m < 6u; ++m) {
                const uint64_t xin = (m < 4u) ? x_q_va : ((m == 4u) ? x_k_va : x_v_va);
                const uint32_t N   = (uint32_t) mats_all[m]->ne[1];
                raw_va[m]     = fb_cur;
                raw_stride[m] = N;
                fb_cur += agx_align_up((size_t) M * N * sizeof(float), AGX_CACHE_LINE_BYTES);
                if (!dispatch_dense_mm(ctx, stream, mats_all[m], xin, raw_va[m], M, AGX_BARRIER_NONE)) {
                    return false;
                }
            }
        }
        if (!P.is_swa) {
            if (!dispatch_dense_mm(ctx, stream, layer.attn_k_conv_proj, x_k_va, L.k_dyn, M, AGX_BARRIER_NONE)) return false;
            if (!dispatch_dense_mm(ctx, stream, layer.attn_v_conv_proj, x_v_va, L.v_dyn, M, AGX_BARRIER_NONE)) return false;
        }
        agx_cdm_barrier_usc(stream);
        }

        // ---- Stage 1a: 1-Dispatch Fused Q/K/V/R ShortConv + State-Store + Clip/Norm + Tau + FWHT(128/64) + KV-Store + [kq_mask_f16 | banded rel_table]! ----
        // Wave 10: SWA KV cache is padded to n_swa + n_ubatch (e.g. 1536..3072 cells) with only 1024 unmasked keys.
        // Inline-pack ALL 40 SWA layers into SLC L.kq_mask_f16 via QKV_FLAG_BANDED_REL_PACK (-40 disp, -40 STRONG_USC barriers, -15.7 MB DRAM!),
        // and reserve banded rel_table + PASS_MODE=4 tile64_minmax strictly for Full-Attn layers!
        const uint32_t mask_row_stride = (P.n_kv + 7u) & ~7u;
        const size_t   mask_need_B     = (size_t) 24u * M * mask_row_stride * sizeof(uint16_t);
        const bool     use_banded_swa_inline = P.has_banded_rel && P.is_swa && (P.n_kv <= 4096u) &&
                                               (mask_need_B <= L.kq_mask_f16_bytes);
        const bool     use_banded_table      = P.has_banded_rel && !use_banded_swa_inline;
        const bool     use_slc_mask    = (!use_banded_table) && (mask_need_B <= L.kq_mask_f16_bytes);
        const uint64_t kq_mask_va      = use_slc_mask ? L.kq_mask_f16 : P.graph_mask_va;
        const uint32_t kq_mask_stride  = use_slc_mask ? mask_row_stride : (P.graph_mask_nb1 / 2u);
        const bool     fuse_rel_inline = use_banded_swa_inline || ((!P.has_banded_rel) && (P.n_kv <= 2048u));
        const bool     need_pack_qgate = (M > 1u) && (raw_stride[1] != 3072u);
        const bool     need_pack_sinks = use_banded_table && (M > 1u) && (raw_stride[3] != 24u);
        const bool     need_w10_qkv    = (P.k_rot_dim == 64u || P.v_rot_dim == 64u || use_banded_swa_inline);
        const agx_kernel * k_qkv_prep  = need_w10_qkv ? ensure_w10_qkv_rot64_kernel(eng) : eng->k_w4_qkv_prep_store;
        if (!k_qkv_prep) return false;

        {
            agx_w4_qkv_prep_desc qd{};
            qd.q_raw_va         = raw_va[0];
            qd.k_raw_va         = raw_va[4];
            qd.v_raw_va         = raw_va[5];
            qd.r_raw_va         = raw_va[2];
            qd.sinks_va         = raw_va[3];
            qd.qgate_raw_va     = raw_va[1];
            qd.qgate_compact_va = need_pack_qgate ? L.qgate_compact : 0ull;
            qd.k_dyn_va         = P.is_swa ? 0ull : L.k_dyn;
            qd.v_dyn_va         = P.is_swa ? 0ull : L.v_dyn;

            qd.q_clip_va        = agx_tensor_gpu_va(layer.attn_q_clip);
            qd.r_clip_va        = agx_tensor_gpu_va(layer.attn_rel_clip);
            qd.k_conv_w_va      = agx_tensor_gpu_va(layer.attn_k_conv);
            qd.v_conv_w_va      = agx_tensor_gpu_va(layer.attn_v_conv);
            qd.rel_bias_va      = agx_tensor_gpu_va(layer.attn_rel_bias);
            qd.inp_pos_va       = G.inp_pos_va;

            qd.r_l_va           = P.r_l_va;
            qd.s_copy_va        = G.s_copy_va;
            qd.k_cell_va        = P.cell_va[0];
            qd.v_cell_va        = P.cell_va[1];

            qd.k_store_va       = P.k_store_va;
            qd.v_store_va       = P.v_store_va;
            qd.k_idxs_va        = P.k_idxs_va;
            qd.v_idxs_va        = P.v_idxs_va;

            qd.q_out_va         = L.q_out;
            qd.r_out_va         = L.r_out;
            qd.rel_pos_va       = use_banded_swa_inline ? P.rel_pos_pack_va : P.rel_pos_va;
            qd.base_mask_va     = P.base_mask_va;
            qd.kq_mask_out_va   = kq_mask_va;

            qd.q_raw_stride_f32    = raw_stride[0];
            qd.k_raw_stride_f32    = raw_stride[4];
            qd.v_raw_stride_f32    = raw_stride[5];
            qd.r_raw_stride_f32    = raw_stride[2];
            qd.sinks_stride_f32    = raw_stride[3];
            qd.r_nb1_f32           = P.r_nb1_f32;
            qd.rollback_slots      = P.rollback_slots;
            qd.rollback_stride_f32 = P.rollback_stride_f32;
            qd.k_store_nb1_B       = P.k_store_nb1;
            qd.v_store_nb1_B       = P.v_store_nb1;
            qd.k_ids_nb0_B         = P.k_ids_nb0;
            qd.v_ids_nb0_B         = P.v_ids_nb0;
            qd.n_tokens            = M;
            qd.n_conv              = 4u;
            qd.n_rel               = (uint32_t) layer.attn_rel_bias->ne[0];
            qd.n_extent            = (uint32_t) (layer.attn_rel_bias->ne[1] - (G.rel_bias_padded[P.ip] ? 1 : 0));
            qd.n_kv                = P.n_kv;
            qd.n_kv_mask_stride    = kq_mask_stride;
            qd.rpos_nb1_B          = use_banded_swa_inline ? P.n_kv : P.rpos_nb1;
            qd.bmask_nb1_B         = P.bmask_nb1;
            uint32_t rb_tmp = 0;
            qd.bias_wtype          = agx_inkling2_tensor_wtype(layer.attn_rel_bias, &rb_tmp);
            qd.bias_nb1_B          = rb_tmp;
            qd.flags               = (!P.is_swa ? AGX_W4_QKV_FLAG_IS_FULL_ATTN : 0u) |
                                     (P.k_rot_dim == 128u ? AGX_W4_QKV_FLAG_FWHT_K : 0u) |
                                     (P.k_rot_dim ==  64u ? AGX_W10_QKV_FLAG_FWHT64_K : 0u) |
                                     (P.v_rot_dim == 128u ? AGX_W4_QKV_FLAG_FWHT_V : 0u) |
                                     (P.v_rot_dim ==  64u ? AGX_W10_QKV_FLAG_FWHT64_V : 0u) |
                                     (use_banded_swa_inline ? AGX_W10_QKV_FLAG_BANDED_REL_PACK : 0u) |
                                     (P.kv_is_q8 ? AGX_W4_QKV_FLAG_KV_Q8_0 : 0u) |
                                     (P.k_ids_is_i64 ? AGX_W4_QKV_FLAG_K_IDS_I64 : 0u) |
                                     (P.v_ids_is_i64 ? AGX_W4_QKV_FLAG_V_IDS_I64 : 0u) |
                                     (P.bmask_is_f16 ? AGX_W4_QKV_FLAG_BASE_MASK_F16 : 0u) |
                                     (fuse_rel_inline ? AGX_W4_QKV_FLAG_FUSE_REL_MASK : 0u) |
                                     (need_pack_qgate ? AGX_W4_QKV_FLAG_PACK_QGATE : 0u);
            qd.r_splits            = fuse_rel_inline ? ((P.n_kv > 512u) ? 4u : ((P.n_kv > 128u) ? 2u : 1u)) : 1u;
            qd.conv_res_scale      = 1.0f / std::sqrt((float) ((P.is_swa ? 1u : 2u) * 4u + 1u));
            qd.qk_clip             = hp.f_qk_clip;
            qd.rel_clip            = hp.f_relative_clip;
            qd.rms_eps             = hp.f_norm_rms_eps;
            qd.log_scale_alpha     = hp.f_log_scale_alpha;
            qd.inv_log_floor       = 1.0f / (float) hp.n_log_scale_floor;

            void * d_cpu = nullptr;
            uint64_t d_va = agx_cdm_alloc_args(stream, sizeof(qd), 16, &d_cpu);
            if (!d_va || !d_cpu) return false;
            std::memcpy(d_cpu, &qd, sizeof(qd));
            agx_w4_desc_va_push pc{ d_va };
            if (!agx_cdm_dispatch(stream, k_qkv_prep, 40u + 24u * qd.r_splits, M, 1u,
                                  &pc, sizeof(pc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            if (mk_dump && v == 0u) {
                slc_dump_f32("v0.h_0",   va_h_0,   D);
                slc_dump_f32("v0.x_q",   x_q_va,   D);
                slc_dump_f32("v0.q_out", L.q_out,  3072u);
                slc_dump_f32("v0.qgate", raw_va[1], 3072u);
                slc_dump_f32("v0.sinks", raw_va[3], 24u);
            }
        }

        if (use_banded_table) {
            uint32_t rb_tmp = 0;
            const uint32_t rb_wt = agx_inkling2_tensor_wtype(layer.attn_rel_bias, &rb_tmp);
            agx_w10_rel_table_push rtc{};
            rtc.r_va             = L.r_out;
            rtc.rel_bias_va      = agx_tensor_gpu_va(layer.attn_rel_bias);
            rtc.dst_va           = P.rel_table_va;
            rtc.sinks_src_va     = need_pack_sinks ? raw_va[3] : 0ull;
            rtc.sinks_dst_va     = need_pack_sinks ? L.sinks_compact : 0ull;
            rtc.n_rel            = (uint32_t) layer.attn_rel_bias->ne[0];
            rtc.eff_rows         = P.rel_table_len;
            rtc.n_tokens         = M;
            rtc.bias_wtype       = rb_wt;
            rtc.bias_nb1_B       = rb_tmp;
            rtc.sinks_stride_f32 = raw_stride[3];
            if (!agx_cdm_dispatch(stream, eng->k_w10_rel_bias_table,
                                  (P.rel_table_len + 63u) / 64u, M, 1u,
                                  &rtc, sizeof(rtc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        } else if (!fuse_rel_inline) {
            agx_w4_rel_mask_push rmc{};
            rmc.r_va             = L.r_out;
            rmc.rel_bias_va      = agx_tensor_gpu_va(layer.attn_rel_bias);
            rmc.sinks_va         = raw_va[3];
            rmc.rel_pos_va       = P.rel_pos_va;
            rmc.base_mask_va     = P.base_mask_va;
            rmc.dst_va           = kq_mask_va;
            rmc.n_rel            = (uint32_t) layer.attn_rel_bias->ne[0];
            rmc.n_extent         = (uint32_t) (layer.attn_rel_bias->ne[1] - (G.rel_bias_padded[P.ip] ? 1 : 0));
            rmc.n_kv             = P.n_kv;
            rmc.n_kv_stride      = kq_mask_stride;
            rmc.n_head           = 24u;
            rmc.n_tokens         = M;
            rmc.sinks_stride_f32 = raw_stride[3];
            rmc.rpos_nb1_B       = P.rpos_nb1;
            rmc.bmask_nb1_B      = P.bmask_nb1;
            uint32_t rb_tmp = 0;
            rmc.bias_wtype       = agx_inkling2_tensor_wtype(layer.attn_rel_bias, &rb_tmp);
            rmc.bias_nb1_B       = rb_tmp;
            rmc.base_mask_is_f16 = P.bmask_is_f16 ? 1u : 0u;
            if (!agx_cdm_dispatch(stream, eng->k_w4_rel_bias_mask, (P.n_kv + 63u) / 64u, 24u, M,
                                  &rmc, sizeof(rmc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }

        // ---- Stage 1b: FlashAttention (+ fused FWHT64 + sigmoid(qgate) + optional OUT_F16!) -> slc->attn_out ----
        const uint64_t qgate_cont_va  = need_pack_qgate ? L.qgate_compact : raw_va[1];
        const agx_kernel * k_wo_m_f16 = (M <= 4u) ? eng->k_w9_q8_gemv_m4_f16 : eng->k_w9_q8_gemv_m8_f16;
        const bool use_w10_wo_f16 = !verify_legacy_wo && (M >= 2u && M <= 8u) && !mk_dump &&
                                    (P.v_rot_dim != 128u) && (k_wo_m_f16 != nullptr) &&
                                    (layer.wo->type == GGML_TYPE_Q8_0) && agx_tensor_is_repacked(layer.wo) &&
                                    (layer.wo->ne[0] == 3072) && (layer.wo->ne[1] == (int64_t) D);
        {
            agx_attn_stage_args attn{};
            attn.n_tokens      = M;
            attn.n_kv          = P.n_kv;
            attn.n_head_q      = 24u;
            attn.n_head_kv     = 8u;
            attn.head_dim      = 128u;
            attn.kq_scale      = 1.0f / std::sqrt(128.0f);
            attn.flags         = (P.v_rot_dim == 64u ? AGX_ATTN_FLAG_FWHT64 : 0u) |
                                 (P.v_rot_dim != 128u ? AGX_ATTN_FLAG_QGATE_F32 : 0u) |
                                 (use_w10_wo_f16 ? AGX_ATTN_FLAG_OUT_F16 : 0u);
            attn.q_va          = L.q_out;
            attn.q_nb1         = 24u * 128u * sizeof(float);
            attn.q_nb2         = 128u * sizeof(float);
            attn.out_va        = slc->attn_out;
            attn.q_gate_va     = (P.v_rot_dim != 128u) ? qgate_cont_va : 0ull;
            attn.ext_k_dtype   = P.kv_is_q8 ? AGX_KV_Q8_0_GGML : AGX_KV_F16;
            attn.ext_v_dtype   = P.kv_is_q8 ? AGX_KV_Q8_0_GGML : AGX_KV_F16;
            attn.ext_k_va      = P.k_va;
            attn.ext_v_va      = P.v_va;
            attn.ext_k_nb1     = P.k_nb1;
            attn.ext_k_nb2     = P.k_nb2;
            attn.ext_v_nb1     = P.v_nb1;
            attn.ext_v_nb2     = P.v_nb2;
            if (use_banded_table) {
                attn.flags          |= AGX_ATTN_FLAG_BANDED_REL_BIAS |
                                       AGX_ATTN_FLAG_SINK_PER_TOK_HEAD |
                                       (P.bmask_is_f16 ? AGX_ATTN_FLAG_HAS_MASK_F16 : AGX_ATTN_FLAG_HAS_MASK_F32);
                attn.n_rel_extent    = P.rel_extent;
                attn.rel_table_len   = P.rel_table_len;
                attn.rel_table_va    = P.rel_table_va;
                attn.rel_pos_pack_va = P.rel_pos_pack_va;
                attn.rel_cp_off      = 0u;
                attn.rel_qp_off      = P.n_kv;
                attn.rel_tm_off      = P.n_kv + M;
                attn.sinks_va        = need_pack_sinks ? L.sinks_compact : raw_va[3];
                attn.kq_mask_va      = P.base_mask_va;
                attn.ext_mask_nb1    = P.bmask_nb1;
                attn.ext_mask_nb2    = 0u;
                attn.ext_mask_ne2    = 1u;
            } else {
                attn.flags          |= AGX_ATTN_FLAG_HAS_MASK_F16 | AGX_ATTN_FLAG_REL_TABLE_SINK;
                attn.kq_mask_va      = kq_mask_va;
                attn.ext_mask_nb1    = kq_mask_stride * sizeof(uint16_t);
                attn.ext_mask_nb2    = M * kq_mask_stride * sizeof(uint16_t);
                attn.ext_mask_ne2    = 24u;
            }
            if (!agx_kv_hybrid_dispatch_full_attn(kv_pool, stream, attn, AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            if (mk_dump && v == 0u) {
                slc_dump_f32("v0.attn_out", slc->attn_out, 3072u);
            }
        }
        if (P.v_rot_dim == 128u) {
            agx_w4_fwht_qgate_push fpc{};
            fpc.attn_va          = slc->attn_out;
            fpc.qgate_va         = raw_va[1];
            fpc.n_head           = 24u;
            fpc.n_tokens         = M;
            fpc.qgate_stride_f32 = raw_stride[1];
            fpc.do_fwht128       = 1u;
            if (!agx_cdm_dispatch(stream, eng->k_w4_fwht_qgate, 24u, M, 1u,
                                  &fpc, sizeof(fpc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }

        // ---- Stage 1c: W_o GEMM -> L.x_k (direct-L1 FP16 0-SRAM 0-barrier Q8 GEMV for M=2..8!) ----
        if (use_w10_wo_f16) {
            agx_w9_q8_gemv_m_push wpc{};
            wpc.w_va     = agx_tensor_gpu_va(layer.wo);
            wpc.x_va     = slc->attn_out;
            wpc.y_va     = L.x_k;
            wpc.M        = M;
            wpc.N        = D;
            wpc.K        = 3072u;
            wpc.x_stride = 3072u;
            wpc.y_stride = D;
            if (!agx_cdm_dispatch(stream, k_wo_m_f16, (D + 7u) / 8u, 1u, 1u,
                                  &wpc, sizeof(wpc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        } else {
            if (!dispatch_dense_mm(ctx, stream, layer.wo, slc->attn_out, L.x_k, M, AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }

        // ---- Stage 2a + FFN Loop Ctrl + Stage 2b (Router): 1-dispatch M=2..8 Hybrid OR unfused ----
        const bool new_block = (P.ip % hp.attn_res_block_size == 0u);
        if (new_block) {
            // cur_va already resides in its permanent static res_blocks slot (va_h_0 / va_prelude / va_h12_* / va_cur_p2)!
            blocks.push_back(cur_va);
        }

        if (use_w5_m_hybrid) {
            agx_w5_persist_push ppc{};
            ppc.descs_va   = w5m_descs_gpu_va + (uint64_t) v * sizeof(agx_w5_pass_desc);
            ppc.barrier_va = slc->sense_barrier_va;
            ppc.n_passes   = 1u;
            ppc.n_wg       = 40u;
            if (!agx_cdm_dispatch(stream, eng->k_w5_hybrid_post_attn_rtr_m[P.has_loop ? 1u : 0u],
                                  40u, 1u, 1u, &ppc, sizeof(ppc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        } else {
        {
            agx_w4_stage_prep_desc sd{};
            sd.conv_src_va         = L.x_k;
            sd.res_va              = new_block ? 0ull : cur_va;
            sd.conv_w_va           = agx_tensor_gpu_va(layer.attn_post_conv);
            sd.r_l_va              = P.r_l_va;
            sd.s_copy_va           = G.s_copy_va;
            sd.cell_va             = P.cell_va[2];
            sd.cur_out_va          = va_ffn_inp;
            sd.block_ptrs_va       = alloc_block_ptrs(blocks);
            sd.query_va            = agx_tensor_gpu_va(layer.ffn_res_query);
            sd.norm_w_va[0]        = !P.has_loop ? agx_tensor_gpu_va(layer.ffn_norm) : 0ull;
            sd.mix_out_va[0]       = !P.has_loop ? L.x_q : L.qkv_in_0;
            sd.n_embd              = D;
            sd.n_tokens            = M;
            sd.n_conv              = 4u;
            sd.conv_ch_offset      = 2048u;
            sd.r_nb1_f32           = P.r_nb1_f32;
            sd.rollback_slots      = P.rollback_slots;
            sd.rollback_stride_f32 = P.rollback_stride_f32;
            sd.n_blocks            = (uint32_t) blocks.size();
            sd.n_queries           = 1u;
            sd.flags               = AGX_W4_SPREP_FLAG_DO_CONV | AGX_W4_SPREP_FLAG_DO_DEPTH_MIX |
                                     (!new_block ? AGX_W4_SPREP_FLAG_ADD_RESIDUAL : 0u) |
                                     (!P.has_loop ? AGX_W4_SPREP_FLAG_DO_NORM : 0u);
            sd.conv_res_scale      = 1.0f / std::sqrt(5.0f);
            sd.inv_sqrt_d          = 1.0f / std::sqrt((float) D);
            sd.eps_mix             = 1e-6f;
            sd.eps_norm            = hp.f_norm_rms_eps;

            void * d_cpu = nullptr;
            uint64_t d_va = agx_cdm_alloc_args(stream, sizeof(sd), 16, &d_cpu);
            if (!d_va || !d_cpu) return false;
            std::memcpy(d_cpu, &sd, sizeof(sd));
            agx_w4_desc_va_push pc{ d_va };
            if (!agx_cdm_dispatch(stream, eng->k_w4_stage_post_attn, 1u, 1u, 1u,
                                  &pc, sizeof(pc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            if (mk_dump && v == 0u) {
                slc_dump_f32("v0.wo_out",  L.x_k,      D);
                slc_dump_f32("v0.ffn_inp", va_ffn_inp, D);
                slc_dump_f32("v0.ffn_x",   L.x_q,      D);
            }
        }

        if (P.has_loop) {
            // 40-Core FFN Loop Controller: norm_cat -> ctrl_down -> concurrent ctrl_gain + ctrl_forget -> mix_norm!
            {
                agx_w4_ctrl_norm_cat_push ncp{};
                ncp.cur_va[0] = L.qkv_in_0;
                ncp.dst_va[0] = L.ctrl_cat[0];
                ncp.msg_va    = slc->h_msg;
                ncp.n_embd    = D;
                ncp.n_tokens  = M;
                ncp.n_ctrls   = 1u;
                ncp.msg_base  = 3u;
                ncp.eps_ctrl  = 1e-6f;
                if (!agx_cdm_dispatch(stream, eng->k_w4_ctrl_norm_cat, 1u, M, 1u,
                                      &ncp, sizeof(ncp), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
            }
            if (!dispatch_dense_mm(ctx, stream, loop->ctrl_down[3], L.ctrl_cat[0], L.ctrl_gate[0],
                                   M, AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            const bool fused_ffn_up =
                loop->ctrl_gain[3]->type == GGML_TYPE_F32 && loop->ctrl_forget[3]->type == GGML_TYPE_F32 &&
                loop->ctrl_gain[3]->ne[0] == 64 && loop->ctrl_forget[3]->ne[0] == 64;
            if (fused_ffn_up) {
                agx_w4_ctrl_up_mix_push ump{};
                ump.cur_va[0]      = L.qkv_in_0;
                ump.gate_va[0]     = L.ctrl_gate[0];
                ump.gain_w_va[0]   = agx_tensor_gpu_va(loop->ctrl_gain[3]);
                ump.forget_w_va[0] = agx_tensor_gpu_va(loop->ctrl_forget[3]);
                ump.out_va[0]      = L.ctrl_gain[0];
                ump.msg_va         = slc->h_msg;
                ump.n_embd         = D;
                ump.n_tokens       = M;
                ump.n_ctrls        = 1u;
                ump.msg_base       = 3u;
                if (!agx_cdm_dispatch(stream, eng->k_w4_ctrl_up_mix, (D + 15u) / 16u, 1u, M,
                                      &ump, sizeof(ump), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
                agx_w4_batch_rmsnorm_push brp{};
                brp.src_va[0]    = L.ctrl_gain[0];
                brp.norm_w_va[0] = agx_tensor_gpu_va(layer.ffn_norm);
                brp.dst_va[0]    = L.x_q;
                brp.n_embd       = D;
                brp.n_tokens     = M;
                brp.n_ctrls      = 1u;
                brp.eps_norm     = hp.f_norm_rms_eps;
                if (!agx_cdm_dispatch(stream, eng->k_w4_batch_rmsnorm, 1u, M, 1u,
                                      &brp, sizeof(brp), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
            } else {
                if (!dispatch_dense_mm(ctx, stream, loop->ctrl_gain[3], L.ctrl_gate[0], L.ctrl_gain[0],
                                       M, AGX_BARRIER_NONE)) {
                    return false;
                }
                if (!dispatch_dense_mm(ctx, stream, loop->ctrl_forget[3], L.ctrl_gate[0], L.ctrl_forget[0],
                                       M, AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
                agx_w4_ctrl_mix_norm_push mnp{};
                mnp.cur_va[0]    = L.qkv_in_0;
                mnp.gain_va[0]   = L.ctrl_gain[0];
                mnp.forget_va[0] = L.ctrl_forget[0];
                mnp.norm_w_va[0] = agx_tensor_gpu_va(layer.ffn_norm);
                mnp.out_va[0]    = L.x_q;
                mnp.msg_va       = slc->h_msg;
                mnp.n_embd       = D;
                mnp.n_tokens     = M;
                mnp.n_ctrls      = 1u;
                mnp.msg_base     = 3u;
                mnp.eps_norm     = hp.f_norm_rms_eps;
                if (!agx_cdm_dispatch(stream, eng->k_w4_ctrl_mix_norm, 1u, M, 1u,
                                      &mnp, sizeof(mnp), AGX_BARRIER_STRONG_USC)) {
                    return false;
                }
            }
        }

        // ---- Stage 2b: Router (40-core GEMM + bitonic Top-8 + norm -> slc->router_topk / router_topk_w) ----
        if (!dispatch_dense_mm(ctx, stream, layer.ffn_gate_inp, L.x_q, L.router_logits, M, AGX_BARRIER_STRONG_USC, true)) {
            return false;
        }
        {
            w4_router_topk_push rpc{};
            rpc.logits_va  = L.router_logits;
            rpc.bias_va    = layer.ffn_exp_probs_b ? agx_tensor_gpu_va(layer.ffn_exp_probs_b) : 0ull;
            rpc.ids_dst_va = slc->router_topk;
            rpc.w_dst_va   = slc->router_topk_w;
            rpc.n_exp      = hp.n_expert;
            rpc.k_out      = hp.n_expert_used;
            rpc.n_tokens   = M;
            rpc.ids_nb1    = hp.n_expert_used;
            rpc.norm_eps   = hp.expert_weights_norm_eps;
            rpc.w_scale    = (hp.expert_weights_scale != 0.0f) ? hp.expert_weights_scale : 1.0f;
            if (!agx_cdm_dispatch(stream, eng->k_w4_router_topk, M, 1u, 1u,
                                  &rpc, sizeof(rpc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }
        } // end else (!use_w5_m_hybrid)

        // ---- Stage 2c + Stage 3a: @w4-kv-moe Fused 2-Stage MoE (gate_up+SiTU -> slc->moe_inter -> down+wsum -> L.x_v) ----
        {
            agx_weight_layout gu_layout = AGX_WEIGHT_F32, dn_layout = AGX_WEIGHT_F32;
            if (!agx_gemm_infer_weight_layout(layer.ffn_gate_up_exps, &gu_layout) ||
                !agx_gemm_infer_weight_layout(layer.ffn_down_exps,    &dn_layout)) {
                return false;
            }
            const uint64_t gu_slice = agx_tensor_is_repacked(layer.ffn_gate_up_exps)
                ? agx_repack_slice_stride(layer.ffn_gate_up_exps->type, layer.ffn_gate_up_exps->ne[0], layer.ffn_gate_up_exps->ne[1])
                : (uint64_t) layer.ffn_gate_up_exps->nb[2];
            const uint64_t dn_slice = agx_tensor_is_repacked(layer.ffn_down_exps)
                ? agx_repack_slice_stride(layer.ffn_down_exps->type, layer.ffn_down_exps->ne[0], layer.ffn_down_exps->ne[1])
                : (uint64_t) layer.ffn_down_exps->nb[2];

            agx_moe_stage_args margs{};
            margs.n_tokens         = M;
            margs.n_embd           = D;
            margs.n_ff             = hp.n_ff_exp;
            margs.n_expert         = hp.n_expert;
            margs.top_k            = hp.n_expert_used;
            margs.act_op           = AGX_MOE_ACT_SITU;
            margs.situ_beta        = hp.situ_beta;
            margs.situ_linear_beta = hp.situ_linear_beta;
            margs.exp_scale0       = 1.0f; // already scaled in k_w4_router_topk
            margs.ffn_x_va         = L.x_q;
            margs.topk_ids_va      = slc->router_topk;
            margs.ids_row_stride   = hp.n_expert_used;
            margs.topk_w_va        = slc->router_topk_w;
            margs.moe_out_va       = L.moe_out;
            margs.gate_up_va       = agx_tensor_gpu_va(layer.ffn_gate_up_exps);
            margs.down_va          = agx_tensor_gpu_va(layer.ffn_down_exps);
            margs.gate_up_slice_B  = gu_slice;
            margs.down_slice_B     = dn_slice;
            margs.gate_up_layout   = gu_layout;
            margs.down_layout      = dn_layout;
            margs.gate_up_scale_va = layer.ffn_gate_up_exps_s ? agx_tensor_gpu_va(layer.ffn_gate_up_exps_s) : 0ull;
            margs.down_scale_va    = layer.ffn_down_exps_s    ? agx_tensor_gpu_va(layer.ffn_down_exps_s)    : 0ull;
            margs.x_is_f16         = false;
            margs.accumulate_out   = false;
            margs.moe_inter_va     = slc->moe_inter;
            margs.res_va           = 0ull; // ffn_post_conv runs between moe_out and + ffn_inp!

            if (!agx_moe_dispatch_fused_situ_down(moe_runner, stream, margs, AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            if (mk_dump && v == 0u) {
                slc_dump_f32("v0.rtr_w",     slc->router_topk_w, 8u);
                slc_dump_f32("v0.moe_inter", slc->moe_inter,     hp.n_ff_exp * hp.n_expert_used);
                slc_dump_f32("v0.moe_out",   L.moe_out,          D);
                const int32_t * ids = reinterpret_cast<const int32_t *>(
                    static_cast<const uint8_t *>(slc->cpu_base) + (slc->router_topk - slc->base_va));
                std::fprintf(stderr, "[MK DUMP] v0.rtr_ids       [%d %d %d %d %d %d %d %d] "
                             "gu=(lay=%u rep=%d s2=%d sl=%llu ne=%lldx%lld) dn=(lay=%u rep=%d s2=%d sl=%llu ne=%lldx%lld)\n",
                             ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], ids[6], ids[7],
                             (unsigned) gu_layout, (int) agx_tensor_is_repacked(layer.ffn_gate_up_exps),
                             !!layer.ffn_gate_up_exps_s,
                             (unsigned long long) gu_slice,
                             (long long) layer.ffn_gate_up_exps->ne[0], (long long) layer.ffn_gate_up_exps->ne[1],
                             (unsigned) dn_layout, (int) agx_tensor_is_repacked(layer.ffn_down_exps),
                             !!layer.ffn_down_exps_s,
                             (unsigned long long) dn_slice,
                             (long long) layer.ffn_down_exps->ne[0], (long long) layer.ffn_down_exps->ne[1]);
            }
        }

        // ---- Stage 3b -> NEXT PASS Stage 0a FUSED: 1-dispatch M=2..8 Hybrid (v=17..40) OR lean stage_post_ffn ----
        const uint64_t next_cur_va = target_l_out_va(v, cur_va);
        const bool is_pass_reset   = (v == 17u || v == 29u);
        const bool is_final_pass   = (v == 47u);

        if (use_w5_m_hybrid && v < 47u && G.passes[v + 1u].has_loop) {
            agx_w5_persist_push ppc{};
            ppc.descs_va   = w5m_descs_gpu_va + (uint64_t) v * sizeof(agx_w5_pass_desc);
            ppc.barrier_va = slc->sense_barrier_va;
            ppc.n_passes   = 1u;
            ppc.n_wg       = 40u;
            if (!agx_cdm_dispatch(stream, eng->k_w5_hybrid_post_ffn_ctrl_m,
                                  40u, 1u, 1u, &ppc, sizeof(ppc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            if (!is_pass_reset) attn_inputs_ready = true;
            w5m_qkv_ctrl_done = true;
            x_f16_ready       = true;
            if (!emit_layer_inp_stores(P.layer_inp_stores, next_cur_va)) {
                return false;
            }
        } else if (use_w5_m_hybrid && !no_m_post_ffn_par && eng->k_w5_hybrid_post_ffn_nl_m != nullptr) {
            // Wave 10: Lean (M+1)-WG parallel non-loop transition (overlapped conv rollback + FP16 mirror!)
            const uint32_t nl_wgs = M + 1u;
            agx_w5_persist_push ppc{};
            ppc.descs_va   = w5m_descs_gpu_va + (uint64_t) v * sizeof(agx_w5_pass_desc);
            ppc.barrier_va = slc->sense_barrier_va;
            ppc.n_passes   = 1u;
            ppc.n_wg       = nl_wgs;
            if (!agx_cdm_dispatch(stream, eng->k_w5_hybrid_post_ffn_nl_m,
                                  nl_wgs, 1u, 1u, &ppc, sizeof(ppc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            if (!is_final_pass && !is_pass_reset) attn_inputs_ready = true;
            x_f16_ready = true;
            if (!emit_layer_inp_stores(P.layer_inp_stores, next_cur_va)) {
                return false;
            }
            if (is_final_pass && !emit_layer_inp_stores(G.out_norm_stores, L.x_q)) {
                return false;
            }
        } else {
            x_f16_ready = false;
            agx_w4_stage_prep_desc sd{};
            sd.conv_src_va         = L.moe_out;
            sd.res_va              = va_ffn_inp;
            sd.conv_w_va           = agx_tensor_gpu_va(layer.ffn_post_conv);
            sd.r_l_va              = P.r_l_va;
            sd.s_copy_va           = G.s_copy_va;
            sd.cell_va             = P.cell_va[3];
            sd.cur_out_va          = next_cur_va;
            sd.export_va           = P.export_l_out_va;
            sd.n_embd              = D;
            sd.n_tokens            = M;
            sd.n_conv              = 4u;
            sd.conv_ch_offset      = 4352u;
            sd.r_nb1_f32           = P.r_nb1_f32;
            sd.rollback_slots      = P.rollback_slots;
            sd.rollback_stride_f32 = P.rollback_stride_f32;
            sd.flags               = AGX_W4_SPREP_FLAG_DO_CONV | AGX_W4_SPREP_FLAG_ADD_RESIDUAL;
            sd.conv_res_scale      = 1.0f / std::sqrt(5.0f);
            sd.inv_sqrt_d          = 1.0f / std::sqrt((float) D);
            sd.eps_mix             = 1e-6f;
            sd.eps_norm            = hp.f_norm_rms_eps;

            if (is_final_pass) {
                // Fuse final output_depth_mix + output_norm right here!
                sd.block_ptrs_va     = alloc_block_ptrs(blocks);
                sd.query_va          = agx_tensor_gpu_va(W.output_res_query);
                sd.norm_w_va[0]      = agx_tensor_gpu_va(W.output_norm);
                sd.mix_out_va[0]     = L.x_q;
                sd.mix_out_export_va = G.export_out_norm;
                sd.n_blocks          = (uint32_t) blocks.size();
                sd.n_queries         = 1u;
                sd.flags            |= AGX_W4_SPREP_FLAG_DO_DEPTH_MIX | AGX_W4_SPREP_FLAG_DO_NORM;
            } else if (!is_pass_reset) {
                // Fuse next pass's attn_depth_mix (+ fused 1x/3x RMSNorm if !next.has_loop)!
                const auto & Pn    = G.passes[v + 1u];
                const auto & Ln    = W.layers[Pn.ip];
                sd.block_ptrs_va   = alloc_block_ptrs(blocks);
                sd.query_va        = agx_tensor_gpu_va(Ln.attn_res_query);
                sd.n_blocks        = (uint32_t) blocks.size();
                sd.n_queries       = Pn.is_swa ? 1u : 3u;
                sd.flags          |= AGX_W4_SPREP_FLAG_DO_DEPTH_MIX;
                if (!Pn.has_loop) {
                    sd.flags        |= AGX_W4_SPREP_FLAG_DO_NORM;
                    sd.norm_w_va[0]  = agx_tensor_gpu_va(Pn.is_swa ? Ln.attn_norm : Ln.attn_q_inp_norm);
                    sd.norm_w_va[1]  = !Pn.is_swa ? agx_tensor_gpu_va(Ln.attn_k_inp_norm) : 0ull;
                    sd.norm_w_va[2]  = !Pn.is_swa ? agx_tensor_gpu_va(Ln.attn_v_inp_norm) : 0ull;
                    sd.mix_out_va[0] = L.x_q;
                    sd.mix_out_va[1] = L.x_k;
                    sd.mix_out_va[2] = L.x_v;
                } else {
                    sd.mix_out_va[0] = L.qkv_in_0;
                    sd.mix_out_va[1] = L.qkv_in_1;
                    sd.mix_out_va[2] = L.qkv_in_2;
                }
                attn_inputs_ready = true;
            }

            void * d_cpu = nullptr;
            uint64_t d_va = agx_cdm_alloc_args(stream, sizeof(sd), 16, &d_cpu);
            if (!d_va || !d_cpu) return false;
            std::memcpy(d_cpu, &sd, sizeof(sd));
            agx_w4_desc_va_push pc{ d_va };
            if (!agx_cdm_dispatch(stream, eng->k_w4_stage_post_ffn, 1u, 1u, 1u,
                                  &pc, sizeof(pc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            if (!emit_layer_inp_stores(P.layer_inp_stores, next_cur_va)) {
                return false;
            }
            if (is_final_pass && !emit_layer_inp_stores(G.out_norm_stores, L.x_q)) {
                return false;
            }
            if (mk_dump && (v == 0u || v == 1u || v == 5u || v == 6u || v == 17u || v == 18u || v == 29u || v == 30u || v == 41u || v == 47u)) {
                char tag[32];
                std::snprintf(tag, sizeof(tag), "v%u.l_out(il%u)", v, P.il);
                slc_dump_f32(tag, next_cur_va, D);
                if (v == 47u) slc_dump_f32("v47.out_norm", L.x_q, D);
            }
        }

        cur_va = next_cur_va;
    }

    // ------------------------------------------------------------------------
    // Final: Optional inp_out_ids Gather + Q8_0 LM Head + Softcap/Trunc -> result_output
    // ------------------------------------------------------------------------
    uint64_t lm_in_va     = L.x_q;
    uint64_t lm_in_f16_va = L.x_q_f16;
    bool     lm_f16_ready = (!ran_w5_persistent && x_f16_ready);
    if (G.inp_out_ids_va != 0ull) {
        const bool emit_lm_f16 = (G.n_out >= 2u && G.n_out <= 8u);
        w4_gather_f32_push gp{};
        gp.src_va     = L.x_q;
        gp.ids_va     = G.inp_out_ids_va;
        gp.dst_va     = L.x_k;
        gp.dst2_va    = G.result_norm_va;
        gp.n_embd     = D;
        gp.n_out      = G.n_out;
        gp.dst_f16_va = emit_lm_f16 ? L.x_k_f16 : 0ull;
        if (!agx_cdm_dispatch(stream, eng->k_w4_gather_f32, (D + 255u) / 256u, G.n_out, 1u,
                              &gp, sizeof(gp), AGX_BARRIER_STRONG_USC)) {
            return false;
        }
        lm_in_va     = L.x_k;
        lm_in_f16_va = L.x_k_f16;
        lm_f16_ready = emit_lm_f16;
    } else if (G.result_norm_va != 0ull) {
        copy_f32_push cp{ L.x_q, G.result_norm_va, G.n_out * D, 0u };
        if (!agx_cdm_dispatch(stream, eng->k_copy_f32, (G.n_out * D + 255u) / 256u, 1u, 1u,
                              &cp, sizeof(cp), AGX_BARRIER_NONE)) {
            return false;
        }
    }

    {
        agx_weight_layout lm_layout = AGX_WEIGHT_F32;
        if (!agx_gemm_infer_weight_layout(W.output, &lm_layout)) return false;
        const agx_kernel * k_lm_f16 = (G.n_out <= 4u) ? eng->k_w9_q8_gemv_m4_f16 : eng->k_w9_q8_gemv_m8_f16;
        if (!verify_legacy_lm && G.n_out >= 2u && G.n_out <= 8u &&
            lm_layout == AGX_WEIGHT_Q8_0_REPACKED && lm_f16_ready &&
            (D % 256u) == 0u && k_lm_f16 != nullptr) {
            agx_w9_q8_gemv_m_push lpc{};
            lpc.w_va     = agx_tensor_gpu_va(W.output);
            lpc.x_va     = lm_in_f16_va;
            lpc.y_va     = G.lm_head_pad_va;
            lpc.M        = G.n_out;
            lpc.N        = hp.n_vocab_padded;
            lpc.K        = D;
            lpc.x_stride = D;
            lpc.y_stride = hp.n_vocab_padded;
            if (!agx_cdm_dispatch(stream, k_lm_f16, (hp.n_vocab_padded + 7u) / 8u, 1u, 1u,
                                  &lpc, sizeof(lpc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        } else {
            agx_gemm_desc d{};
            d.w_va         = agx_tensor_gpu_va(W.output);
            d.x_va         = lm_in_va;
            d.y_va         = G.lm_head_pad_va;
            d.M            = G.n_out;
            d.N            = hp.n_vocab_padded;
            d.K            = D;
            d.layout       = lm_layout;
            d.prec_f32     = true; // matches prec_policy.set_acc(output, GGML_PREC_F32)
            d.post_barrier = AGX_BARRIER_STRONG_USC;
            if (!agx_gemm_dispatch(ctx, stream, d)) {
                return false;
            }
        }
    }

    if (!agx_inkling2_dispatch_softcap_argmax(eng, stream,
            G.lm_head_pad_va, G.result_output_va, 0ull,
            hp.n_vocab_unpadded, hp.n_vocab_padded, G.n_out,
            hp.f_final_logit_softcapping, false, AGX_BARRIER_MEMORY_HOST)) {
        return false;
    }

    const auto t_enc_end = std::chrono::steady_clock::now();
    const uint64_t cpu_enc_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t_enc_end - t_enc_beg).count();

    const char * stats_env  = std::getenv("GGML_AGX_STATS");
    const bool   emit_stats = (stats_env && stats_env[0] && stats_env[0] != '0');
    uint64_t gpu_ns = 0;
    if (agx_cdm_submit(stream, true, emit_stats ? &gpu_ns : nullptr) != 0) {
        AGX_LOG_ERROR("agx_inkling2_try_execute: CDM submit failed");
        return false;
    }

    if (emit_stats) {
        agx_cdm_stats cdm_after{};
        agx_cdm_get_stats(stream, &cdm_after);
        std::fprintf(stderr,
                     "[ggml-agx MEGAKERNEL STATS] M=%u n_out=%u kv=%s dispatches=%llu submits=%llu "
                     "usc_bar=%llu elided=%llu cpu_enc=%.2fms gpu=%.2fms\n",
                     M, G.n_out, G.passes[0].kv_is_q8 ? "Q8_0" : "F16",
                     (unsigned long long) (cdm_after.dispatches          - cdm_before.dispatches),
                     (unsigned long long) (cdm_after.submits             - cdm_before.submits),
                     (unsigned long long) (cdm_after.strong_usc_barriers - cdm_before.strong_usc_barriers),
                     (unsigned long long) (cdm_after.elided_barriers     - cdm_before.elided_barriers),
                     (double) cpu_enc_ns * 1e-6, (double) gpu_ns * 1e-6);
    }

    return true;
}
