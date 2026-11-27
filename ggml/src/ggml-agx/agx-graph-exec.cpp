#include "agx-graph-exec.h"
#include "agx-inkling2.h"
#include "agx-gemm.h"
#include "agx-kv-hybrid.h"
#include "agx-moe.h"
#include "ggml-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ============================================================================
// Pillar 3: Generic ggml_cgraph Lowering, Subgraph Fusion & Barrier Elision
// ============================================================================
// Every generic GLSL 450 shader below is compiled once via Borealis (`0:0`
// spills, `<= 48` GPRs) and launched over ONE `agx_cdm_stream` submission per
// `ggml_backend_graph_compute` call with interval-based RAW hazard tracking
// (TLA+-verified `AGX_BARRIER_NONE` vs `AGX_BARRIER_STRONG_USC`).

// ----------------------------------------------------------------------------
// 1. Generic 4D Strided Binary Elementwise Kernel (ADD, SUB, MUL, DIV)
// ----------------------------------------------------------------------------
static const char * k_glsl_bin_op = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t a_va;
    uint64_t b_va;
    uint64_t d_va;
    uint     ne0, ne1, ne2, ne3;
    uint     ne00, ne01, ne02, ne03;
    uint     ne10, ne11, ne12, ne13;
    uint     nb00, nb01, nb02, nb03; // in floats (bytes / 4)
    uint     nb10, nb11, nb12, nb13; // in floats
    uint     nb0,  nb1,  nb2,  nb3;  // in floats
    uint     op_kind; // 0=ADD, 1=SUB, 2=MUL, 3=DIV, 4=(a+b)*p0+p1, 5=a*sigmoid(b)
    uint     total;
    float    p0;
    float    p1;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.total) return;

    uint i0 = idx % pc.ne0;
    uint t1 = idx / pc.ne0;
    uint i1 = t1 % pc.ne1;
    uint t2 = t1 / pc.ne1;
    uint i2 = t2 % pc.ne2;
    uint i3 = t2 / pc.ne2;

    uint a_idx = (i0 % pc.ne00) * pc.nb00 + (i1 % pc.ne01) * pc.nb01 +
                 (i2 % pc.ne02) * pc.nb02 + (i3 % pc.ne03) * pc.nb03;
    uint b_idx = (i0 % pc.ne10) * pc.nb10 + (i1 % pc.ne11) * pc.nb11 +
                 (i2 % pc.ne12) * pc.nb12 + (i3 % pc.ne13) * pc.nb13;
    uint d_idx = i0 * pc.nb0 + i1 * pc.nb1 + i2 * pc.nb2 + i3 * pc.nb3;

    float a = ConstBufF32(pc.a_va).v[a_idx];
    float b = ConstBufF32(pc.b_va).v[b_idx];
    float r;
    if      (pc.op_kind == 0u) r = a + b;
    else if (pc.op_kind == 1u) r = a - b;
    else if (pc.op_kind == 2u) r = a * b;
    else if (pc.op_kind == 3u) r = a / b;
    else if (pc.op_kind == 4u) r = (a + b) * pc.p0 + pc.p1;
    else                       r = a / (1.0 + exp(-b));
    BufF32(pc.d_va).v[d_idx] = r;
}
)GLSL";

struct bin_op_push {
    uint64_t a_va, b_va, d_va;
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t ne00, ne01, ne02, ne03;
    uint32_t ne10, ne11, ne12, ne13;
    uint32_t nb00, nb01, nb02, nb03;
    uint32_t nb10, nb11, nb12, nb13;
    uint32_t nb0, nb1, nb2, nb3;
    uint32_t op_kind, total;
    float    p0, p1;
};
static_assert(sizeof(bin_op_push) <= AGX_MAX_PUSH_BYTES, "bin_op_push > 256B");

// ----------------------------------------------------------------------------
// 2. Generic 4D Strided Unary / Scale / Clamp / Fill Kernel
// ----------------------------------------------------------------------------
static const char * k_glsl_unary_op = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t a_va;
    uint64_t d_va;
    uint     ne0, ne1, ne2, ne3;
    uint     nb00, nb01, nb02, nb03;
    uint     nb0,  nb1,  nb2,  nb3;
    uint     op_kind;
    uint     total;
    float    p0;
    float    p1;
} pc;

// op_kind codes:
// 0=SCALE(x*p0+p1), 1=CLAMP(p0,p1), 2=SQR, 3=SQRT, 4=LOG, 5=SIN, 6=COS, 7=FILL(p0),
// 10=ABS, 11=SGN, 12=NEG, 13=STEP, 14=TANH, 15=ELU, 16=RELU, 17=SIGMOID,
// 18=GELU, 19=GELU_QUICK, 20=SILU, 21=EXP, 22=EXPM1, 23=SOFTPLUS, 24=GELU_ERF,
// 25=FLOOR, 26=CEIL, 27=ROUND, 28=TRUNC, 29=HARDSWISH, 30=HARDSIGMOID
float erf_approx(float x) {
    float s = (x < 0.0) ? -1.0 : 1.0;
    float a = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * a);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.total) return;

    uint i0 = idx % pc.ne0;
    uint t1 = idx / pc.ne0;
    uint i1 = t1 % pc.ne1;
    uint t2 = t1 / pc.ne1;
    uint i2 = t2 % pc.ne2;
    uint i3 = t2 / pc.ne2;

    uint d_idx = i0 * pc.nb0 + i1 * pc.nb1 + i2 * pc.nb2 + i3 * pc.nb3;
    if (pc.op_kind == 7u) {
        BufF32(pc.d_va).v[d_idx] = pc.p0;
        return;
    }
    if (pc.op_kind == 8u) {
        BufF32(pc.d_va).v[d_idx] = pc.p0 + float(idx) * pc.p1;
        return;
    }

    uint a_idx = i0 * pc.nb00 + i1 * pc.nb01 + i2 * pc.nb02 + i3 * pc.nb03;
    float x = ConstBufF32(pc.a_va).v[a_idx];
    float r = x;

    switch (pc.op_kind) {
        case 0u:  r = x * pc.p0 + pc.p1; break;
        case 1u:  r = clamp(x, pc.p0, pc.p1); break;
        case 2u:  r = x * x; break;
        case 3u:  r = sqrt(x); break;
        case 4u:  r = log(x); break;
        case 5u:  r = sin(x); break;
        case 6u:  r = cos(x); break;
        case 10u: r = abs(x); break;
        case 11u: r = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0); break;
        case 12u: r = -x; break;
        case 13u: r = (x > 0.0) ? 1.0 : 0.0; break;
        case 14u: r = tanh(x); break;
        case 15u: r = (x > 0.0) ? x : exp(x) - 1.0; break;
        case 16u: r = max(x, 0.0); break;
        case 17u: r = 1.0 / (1.0 + exp(-x)); break;
        case 18u: r = 0.5 * x * (1.0 + tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x))); break;
        case 19u: r = x / (1.0 + exp(-1.702 * x)); break;
        case 20u: r = x / (1.0 + exp(-x)); break;
        case 21u: r = exp(x); break;
        case 22u: r = exp(x) - 1.0; break;
        case 23u: {
            // Exact softplus = max(x, 0) + log1p(exp(-|x|)) without 1+t cancellation
            // (immune to fast-math folding (1+t)-1 -> t for x in [-16.6, -16.0]).
            float t = exp(-abs(x));
            float lp = (t < 0.05)
                ? (t * (1.0 + t * (-0.5 + t * (0.3333333333 + t * (-0.25 + t * 0.2)))))
                : log(1.0 + t);
            r = max(x, 0.0) + lp;
        } break;
        case 24u: r = 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865475)); break;
        case 25u: r = floor(x); break;
        case 26u: r = ceil(x); break;
        case 27u: r = roundEven(x); break;
        case 28u: r = trunc(x); break;
        case 29u: r = x * clamp(x + 3.0, 0.0, 6.0) / 6.0; break;
        case 30u: r = clamp(x + 3.0, 0.0, 6.0) / 6.0; break;
        default:  r = x; break;
    }
    BufF32(pc.d_va).v[d_idx] = r;
}
)GLSL";

struct unary_op_push {
    uint64_t a_va, d_va;
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t nb00, nb01, nb02, nb03;
    uint32_t nb0, nb1, nb2, nb3;
    uint32_t op_kind, total;
    float    p0, p1;
};
static_assert(sizeof(unary_op_push) <= AGX_MAX_PUSH_BYTES, "unary_op_push > 256B");

// ----------------------------------------------------------------------------
// 3. Generic Strided Copy / Cast / Contiguous Kernel (F32/F16/BF16/I32)
// ----------------------------------------------------------------------------
static const char * k_glsl_cpy_cast = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufI32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufU16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer ConstBufU16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufF16 { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer ConstBufF16 { float16_t v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t dst_va;
    uint     ne0, ne1, ne2, ne3;         // src logical shape
    uint     d_ne0, d_ne1, d_ne2, d_ne3; // dst logical shape
    uint     s_nb0, s_nb1, s_nb2, s_nb3; // byte strides
    uint     d_nb0, d_nb1, d_nb2, d_nb3; // byte strides
    uint     src_type; // 0=F32, 1=F16, 2=BF16, 3=I32
    uint     dst_type; // 0=F32, 1=F16, 2=BF16, 3=I32
    uint     total;
    uint     d_is_linear; // 1 = write linearly by idx
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.total) return;

    uint i0 = idx % pc.ne0;
    uint t1 = idx / pc.ne0;
    uint i1 = t1 % pc.ne1;
    uint t2 = t1 / pc.ne1;
    uint i2 = t2 % pc.ne2;
    uint i3 = t2 / pc.ne2;

    uint64_t s_off = uint64_t(i0) * pc.s_nb0 + uint64_t(i1) * pc.s_nb1 +
                     uint64_t(i2) * pc.s_nb2 + uint64_t(i3) * pc.s_nb3;
    uint64_t d_off;
    if (pc.d_is_linear != 0u) {
        d_off = uint64_t(idx) * uint64_t(pc.d_nb0);
    } else {
        uint d0 = idx % pc.d_ne0;
        uint u1 = idx / pc.d_ne0;
        uint d1 = u1 % pc.d_ne1;
        uint u2 = u1 / pc.d_ne1;
        uint d2 = u2 % pc.d_ne2;
        uint d3 = u2 / pc.d_ne2;
        d_off = uint64_t(d0) * pc.d_nb0 + uint64_t(d1) * pc.d_nb1 +
                uint64_t(d2) * pc.d_nb2 + uint64_t(d3) * pc.d_nb3;
    }

    // Exact bit-preserving copy when types match (I32/F16/BF16)
    if (pc.src_type == 3u && pc.dst_type == 3u) {
        BufI32(pc.dst_va + d_off).v[0] = ConstBufI32(pc.src_va + s_off).v[0];
        return;
    }
    if ((pc.src_type == 1u && pc.dst_type == 1u) || (pc.src_type == 2u && pc.dst_type == 2u)) {
        BufU16(pc.dst_va + d_off).v[0] = ConstBufU16(pc.src_va + s_off).v[0];
        return;
    }

    float val = 0.0;
    if      (pc.src_type == 0u) val = ConstBufF32(pc.src_va + s_off).v[0];
    else if (pc.src_type == 1u) val = float(ConstBufF16(pc.src_va + s_off).v[0]);
    else if (pc.src_type == 2u) val = uintBitsToFloat(uint(ConstBufU16(pc.src_va + s_off).v[0]) << 16u);
    else                        val = float(ConstBufI32(pc.src_va + s_off).v[0]);

    if      (pc.dst_type == 0u) BufF32(pc.dst_va + d_off).v[0] = val;
    else if (pc.dst_type == 1u) BufF16(pc.dst_va + d_off).v[0] = float16_t(val);
    else if (pc.dst_type == 2u) {
        uint bits = floatBitsToUint(val);
        uint rbit = 0x7FFFu + ((bits >> 16u) & 1u);
        BufU16(pc.dst_va + d_off).v[0] = uint16_t((bits + rbit) >> 16u);
    } else {
        BufI32(pc.dst_va + d_off).v[0] = int(val);
    }
}
)GLSL";

struct cpy_cast_push {
    uint64_t src_va, dst_va;
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t d_ne0, d_ne1, d_ne2, d_ne3;
    uint32_t s_nb0, s_nb1, s_nb2, s_nb3;
    uint32_t d_nb0, d_nb1, d_nb2, d_nb3;
    uint32_t src_type, dst_type, total, d_is_linear;
};
static_assert(sizeof(cpy_cast_push) <= AGX_MAX_PUSH_BYTES, "cpy_cast_push > 256B");

// ----------------------------------------------------------------------------
// 4. Generic Row-wise Norms & Reductions (RMS_NORM, NORM, L2_NORM, SUM_ROWS, MEAN, CUMSUM, ARGMAX)
// ----------------------------------------------------------------------------
static const char * k_glsl_row_reduce = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufI32 { int v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t a_va;
    uint64_t d_va;
    uint64_t w_va; // optional 1D [ne0] weight for fused RMS_NORM * w
    uint     ne0, ne1, ne2, ne3;
    uint     nb01, nb02, nb03; // floats
    uint     nb1,  nb2,  nb3;  // floats
    uint     op_kind; // 0=RMS_NORM, 1=LAYER_NORM, 2=L2_NORM(exact 1/max(||x||,eps)), 3=SUM_ROWS, 4=MEAN, 5=CUMSUM, 6=ARGMAX, 7=ROW_SUM_NORM(x/(sum+eps)*scale)
    float    eps;
    float    scale;
    uint     nb00;    // element stride along dim 0 in floats (0 or 1 = contiguous)
} pc;

shared float s_red[8];
shared int   s_idx[8];

float wg_sum(float val, uint sg_id, uint lane_id) {
    float s = subgroupAdd(val);
    if (lane_id == 0u) s_red[sg_id] = s;
    barrier();
    if (sg_id == 0u) {
        float v = (lane_id < 8u) ? s_red[lane_id] : 0.0;
        float t = subgroupAdd(v);
        if (lane_id == 0u) s_red[0] = t;
    }
    barrier();
    float res = s_red[0];
    barrier();
    return res;
}

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint n_rows = pc.ne1 * pc.ne2 * pc.ne3;
    if (row >= n_rows) return;

    const uint i1 = row % pc.ne1;
    const uint t2 = row / pc.ne1;
    const uint i2 = t2 % pc.ne2;
    const uint i3 = t2 / pc.ne2;

    const uint a_base = i1 * pc.nb01 + i2 * pc.nb02 + i3 * pc.nb03;
    const uint d_base = i1 * pc.nb1  + i2 * pc.nb2  + i3 * pc.nb3;

    const uint tid     = gl_LocalInvocationID.x;
    const uint sg_id   = tid >> 5u;
    const uint lane_id = tid & 31u;
    const uint D       = pc.ne0;

    if (pc.op_kind == 0u) {
        // RMS_NORM (with optional fused 1D weight multiply & optional dim-0 stride for CONT(VIEW_3D) absorption)
        const uint s0 = (pc.nb00 > 1u) ? pc.nb00 : 1u;
        float ss = 0.0;
        if (s0 == 1u) {
            for (uint i = tid; i < D; i += 256u) {
                float x = ConstBufF32(pc.a_va).v[a_base + i];
                ss += x * x;
            }
        } else {
            for (uint i = tid; i < D; i += 256u) {
                float x = ConstBufF32(pc.a_va).v[a_base + i * s0];
                ss += x * x;
            }
        }
        float sc = inversesqrt(wg_sum(ss, sg_id, lane_id) / float(D) + pc.eps);
        if (pc.w_va != 0ul) {
            if (s0 == 1u) {
                for (uint i = tid; i < D; i += 256u) {
                    BufF32(pc.d_va).v[d_base + i] = (ConstBufF32(pc.a_va).v[a_base + i] * sc) * ConstBufF32(pc.w_va).v[i];
                }
            } else {
                for (uint i = tid; i < D; i += 256u) {
                    BufF32(pc.d_va).v[d_base + i] = (ConstBufF32(pc.a_va).v[a_base + i * s0] * sc) * ConstBufF32(pc.w_va).v[i];
                }
            }
        } else {
            if (s0 == 1u) {
                for (uint i = tid; i < D; i += 256u) {
                    BufF32(pc.d_va).v[d_base + i] = ConstBufF32(pc.a_va).v[a_base + i] * sc;
                }
            } else {
                for (uint i = tid; i < D; i += 256u) {
                    BufF32(pc.d_va).v[d_base + i] = ConstBufF32(pc.a_va).v[a_base + i * s0] * sc;
                }
            }
        }
    } else if (pc.op_kind == 1u) {
        // LAYER_NORM
        float s = 0.0;
        for (uint i = tid; i < D; i += 256u) s += ConstBufF32(pc.a_va).v[a_base + i];
        float mean = wg_sum(s, sg_id, lane_id) / float(D);
        float ss = 0.0;
        for (uint i = tid; i < D; i += 256u) {
            float d = ConstBufF32(pc.a_va).v[a_base + i] - mean;
            ss += d * d;
        }
        float sc = inversesqrt(wg_sum(ss, sg_id, lane_id) / float(D) + pc.eps);
        for (uint i = tid; i < D; i += 256u) {
            BufF32(pc.d_va).v[d_base + i] = (ConstBufF32(pc.a_va).v[a_base + i] - mean) * sc;
        }
    } else if (pc.op_kind == 2u) {
        // Exact L2_NORM: x / max(||x||, eps) (no +eps bias when ||x|| == eps!)
        float ss = 0.0;
        for (uint i = tid; i < D; i += 256u) {
            float x = ConstBufF32(pc.a_va).v[a_base + i];
            ss += x * x;
        }
        float sc = 1.0 / max(sqrt(wg_sum(ss, sg_id, lane_id)), pc.eps);
        for (uint i = tid; i < D; i += 256u) {
            BufF32(pc.d_va).v[d_base + i] = ConstBufF32(pc.a_va).v[a_base + i] * sc;
        }
    } else if (pc.op_kind == 3u || pc.op_kind == 4u) {
        // SUM_ROWS / MEAN
        float s = 0.0;
        for (uint i = tid; i < D; i += 256u) s += ConstBufF32(pc.a_va).v[a_base + i];
        float tot = wg_sum(s, sg_id, lane_id);
        if (tid == 0u) {
            BufF32(pc.d_va).v[d_base] = (pc.op_kind == 4u) ? (tot / float(D)) : tot;
        }
    } else if (pc.op_kind == 5u) {
        // CUMSUM (exact sequential in FP32 on thread 0 — D is small, e.g. 64 in DSpark)
        if (tid == 0u) {
            float acc = 0.0;
            for (uint i = 0u; i < D; ++i) {
                acc += ConstBufF32(pc.a_va).v[a_base + i];
                BufF32(pc.d_va).v[d_base + i] = acc;
            }
        }
    } else if (pc.op_kind == 6u) {
        // ARGMAX (lower index wins ties)
        float bv = uintBitsToFloat(0xFF800000u);
        int   bi = 0x7FFFFFFF;
        for (uint i = tid; i < D; i += 256u) {
            float v = ConstBufF32(pc.a_va).v[a_base + i];
            int  ii = int(i);
            if (v > bv || (v == bv && ii < bi)) { bv = v; bi = ii; }
        }
        for (uint off = 16u; off > 0u; off >>= 1u) {
            float ov = subgroupShuffleXor(bv, off);
            int   oi = subgroupShuffleXor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane_id == 0u) { s_red[sg_id] = bv; s_idx[sg_id] = bi; }
        barrier();
        if (sg_id == 0u) {
            float v = (lane_id < 8u) ? s_red[lane_id] : uintBitsToFloat(0xFF800000u);
            int   i = (lane_id < 8u) ? s_idx[lane_id] : 0x7FFFFFFF;
            for (uint off = 16u; off > 0u; off >>= 1u) {
                float ov = subgroupShuffleXor(v, off);
                int   oi = subgroupShuffleXor(i, off);
                if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
            }
            if (lane_id == 0u) BufI32(pc.d_va).v[d_base] = i;
        }
    } else if (pc.op_kind == 7u) {
        // ROW_SUM_NORM: out[i] = a[i] * (scale / (sum(a) + eps))
        float s = 0.0;
        for (uint i = tid; i < D; i += 256u) s += ConstBufF32(pc.a_va).v[a_base + i];
        float inv = pc.scale / (wg_sum(s, sg_id, lane_id) + pc.eps);
        for (uint i = tid; i < D; i += 256u) {
            BufF32(pc.d_va).v[d_base + i] = ConstBufF32(pc.a_va).v[a_base + i] * inv;
        }
    }
}
)GLSL";

struct row_reduce_push {
    uint64_t a_va, d_va, w_va;
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t nb01, nb02, nb03;
    uint32_t nb1, nb2, nb3;
    uint32_t op_kind;
    float    eps;
    float    scale;
    uint32_t nb00;
};
static_assert(sizeof(row_reduce_push) <= AGX_MAX_PUSH_BYTES, "row_reduce_push > 256B");

// ----------------------------------------------------------------------------
// 5. Generic Softmax (with optional F16/F32 mask, scale, ALiBi max_bias, sinks)
// ----------------------------------------------------------------------------
static const char * k_glsl_softmax = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer ConstBufU16 { uint16_t v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t mask_va;
    uint64_t sinks_va;
    uint64_t dst_va;
    uint     ne0, ne1, ne2, ne3;
    uint     m_ne1, m_ne2, m_ne3;
    uint     mask_is_f16;
    float    scale;
    float    max_bias;
    float    m0, m1;
    uint     n_head_log2;
    uint     pad0;
} pc;

shared float s_red[8];

float wg_max(float val, uint sg, uint ln) {
    float m = subgroupMax(val);
    if (ln == 0u) s_red[sg] = m;
    barrier();
    if (sg == 0u) {
        float v = (ln < 8u) ? s_red[ln] : uintBitsToFloat(0xFF800000u);
        float t = subgroupMax(v);
        if (ln == 0u) s_red[0] = t;
    }
    barrier();
    float res = s_red[0];
    barrier();
    return res;
}

float wg_sum(float val, uint sg, uint ln) {
    float s = subgroupAdd(val);
    if (ln == 0u) s_red[sg] = s;
    barrier();
    if (sg == 0u) {
        float v = (ln < 8u) ? s_red[ln] : 0.0;
        float t = subgroupAdd(v);
        if (ln == 0u) s_red[0] = t;
    }
    barrier();
    float res = s_red[0];
    barrier();
    return res;
}

float load_mask(uint m_base, uint i) {
    if (pc.mask_va == 0ul) return 0.0;
    if (pc.mask_is_f16 != 0u) {
        return unpackHalf2x16(uint(ConstBufU16(pc.mask_va).v[m_base + i])).x;
    }
    return ConstBufF32(pc.mask_va).v[m_base + i];
}

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint n_rows = pc.ne1 * pc.ne2 * pc.ne3;
    if (row >= n_rows) return;

    const uint i1 = row % pc.ne1;
    const uint t2 = row / pc.ne1;
    const uint i2 = t2 % pc.ne2;
    const uint i3 = t2 / pc.ne2;

    const uint base = row * pc.ne0;
    const uint m_base = ((i3 % pc.m_ne3) * pc.m_ne2 + (i2 % pc.m_ne2)) * pc.m_ne1 * pc.ne0 + (i1 % pc.m_ne1) * pc.ne0;

    float slope = 1.0;
    if (pc.max_bias > 0.0) {
        uint h = i2;
        float b = (h < pc.n_head_log2) ? pc.m0 : pc.m1;
        uint  e = (h < pc.n_head_log2) ? (h + 1u) : (2u * (h - pc.n_head_log2) + 1u);
        slope = pow(b, float(e));
    }

    const uint tid = gl_LocalInvocationID.x;
    const uint sg  = tid >> 5u;
    const uint ln  = tid & 31u;

    float maxv = uintBitsToFloat(0xFF800000u);
    if (pc.sinks_va != 0ul) {
        maxv = ConstBufF32(pc.sinks_va).v[i2];
    }
    for (uint i = tid; i < pc.ne0; i += 256u) {
        float x = ConstBufF32(pc.src_va).v[base + i] * pc.scale + slope * load_mask(m_base, i);
        maxv = max(maxv, x);
    }
    maxv = wg_max(maxv, sg, ln);

    float sum_e = 0.0;
    for (uint i = tid; i < pc.ne0; i += 256u) {
        float x = ConstBufF32(pc.src_va).v[base + i] * pc.scale + slope * load_mask(m_base, i);
        float e = (x <= -1e30 || maxv <= -1e30) ? 0.0 : exp(x - maxv);
        BufF32(pc.dst_va).v[base + i] = e;
        sum_e += e;
    }
    sum_e = wg_sum(sum_e, sg, ln);
    if (pc.sinks_va != 0ul) {
        float sv = ConstBufF32(pc.sinks_va).v[i2];
        if (sv > -1e30 && maxv > -1e30) {
            sum_e += exp(sv - maxv);
        }
    }
    float inv_s = (sum_e > 0.0) ? (1.0 / sum_e) : 0.0;
    for (uint i = tid; i < pc.ne0; i += 256u) {
        BufF32(pc.dst_va).v[base + i] *= inv_s;
    }
}
)GLSL";

struct softmax_push {
    uint64_t src_va, mask_va, sinks_va, dst_va;
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t m_ne1, m_ne2, m_ne3, mask_is_f16;
    float    scale, max_bias, m0, m1;
    uint32_t n_head_log2, pad0;
};
static_assert(sizeof(softmax_push) <= AGX_MAX_PUSH_BYTES, "softmax_push > 256B");

// ----------------------------------------------------------------------------
// 6. Generic GET_ROWS / SET_ROWS / ADD_ID / CONCAT / PAD / REPEAT / SSM_CONV / ROPE / ARGSORT / TOP_K / GLU
// ----------------------------------------------------------------------------
static const char * k_glsl_get_rows = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer ConstBufU16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) readonly buffer ConstBufI8 { int8_t v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t ids_va;
    uint64_t dst_va;
    uint     ne0, ne1, ne2, ne3;       // dst shape
    uint     s_nb1, s_nb2, s_nb3;      // bytes
    uint     i_nb0, i_nb1, i_nb2;      // bytes
    uint     d_nb1, d_nb2, d_nb3;      // bytes
    uint     wtype;                    // 0=F32, 1=F16, 2=BF16, 3=Q8_0_RAW, 4=Q8_0_8BLK, 5=I32
    uint     pad0;
} pc;

void main() {
    uint col = gl_GlobalInvocationID.x;
    uint r1  = gl_WorkGroupID.y;
    uint r23 = gl_WorkGroupID.z;
    if (col >= pc.ne0 || r1 >= pc.ne1 || r23 >= pc.ne2 * pc.ne3) return;

    uint i2 = r23 % pc.ne2;
    uint i3 = r23 / pc.ne2;

    uint64_t id_off = uint64_t(r1) * pc.i_nb0 + uint64_t(i2) * pc.i_nb1 + uint64_t(i3) * pc.i_nb2;
    int row_id = ConstBufI32(pc.ids_va + id_off).v[0];
    uint64_t s_row = pc.src_va + uint64_t(max(row_id, 0)) * pc.s_nb1 + uint64_t(i2) * pc.s_nb2 + uint64_t(i3) * pc.s_nb3;
    uint64_t d_row = pc.dst_va + uint64_t(r1) * pc.d_nb1 + uint64_t(i2) * pc.d_nb2 + uint64_t(i3) * pc.d_nb3;

    float v = 0.0;
    if      (pc.wtype == 0u) v = ConstBufF32(s_row).v[col];
    else if (pc.wtype == 1u) v = unpackHalf2x16(uint(ConstBufU16(s_row).v[col])).x;
    else if (pc.wtype == 2u) v = uintBitsToFloat(uint(ConstBufU16(s_row).v[col]) << 16u);
    else if (pc.wtype == 3u) {
        uint blk = col >> 5u, off = col & 31u;
        uint64_t bva = s_row + uint64_t(blk) * 34ul;
        v = unpackHalf2x16(uint(ConstBufU16(bva).v[0])).x * float(ConstBufI8(bva + 2ul).v[off]);
    } else if (pc.wtype == 4u) {
        uint ch = col >> 8u, rem = col & 255u, blk = rem >> 5u;
        uint64_t cva = s_row + uint64_t(ch) * 272ul;
        v = unpackHalf2x16(uint(ConstBufU16(cva).v[blk])).x * float(ConstBufI8(cva + 16ul).v[rem]);
    } else {
        BufF32(d_row).v[col] = intBitsToFloat(ConstBufI32(s_row).v[col]);
        return;
    }
    BufF32(d_row).v[col] = v;
}
)GLSL";

struct get_rows_push {
    uint64_t src_va, ids_va, dst_va;
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t s_nb1, s_nb2, s_nb3;
    uint32_t i_nb0, i_nb1, i_nb2;
    uint32_t d_nb1, d_nb2, d_nb3;
    uint32_t wtype, pad0;
};
static_assert(sizeof(get_rows_push) <= AGX_MAX_PUSH_BYTES, "get_rows_push > 256B");

static const char * k_glsl_set_rows = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer ConstBufI64 { int64_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufF16 { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufU16 { uint16_t v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t ids_va;
    uint64_t dst_va;
    uint     ne0, ne1, ne2, ne3; // src shape
    uint     s_nb1, s_nb2, s_nb3;
    uint     i_nb0, i_nb1, i_nb2;
    uint     i_ne1, i_ne2;
    uint     d_nb1, d_nb2, d_nb3;
    uint     dst_type;   // 0=F32, 1=F16, 2=Q8_0
    uint     ids_is_i64; // 1=I64, 0=I32
} pc;

void main() {
    uint col = gl_GlobalInvocationID.x;
    uint r1  = gl_WorkGroupID.y;
    uint r23 = gl_WorkGroupID.z;
    if (col >= pc.ne0 || r1 >= pc.ne1 || r23 >= pc.ne2 * pc.ne3) return;

    uint i2 = r23 % pc.ne2;
    uint i3 = r23 / pc.ne2;
    uint64_t id_off = uint64_t(r1) * pc.i_nb0 + uint64_t(i2 % pc.i_ne1) * pc.i_nb1 + uint64_t(i3 % pc.i_ne2) * pc.i_nb2;
    int64_t dst_row = (pc.ids_is_i64 != 0u)
        ? ConstBufI64(pc.ids_va + id_off).v[0]
        : int64_t(ConstBufI32(pc.ids_va + id_off).v[0]);
    if (dst_row < 0l) return;

    uint64_t s_row = pc.src_va + uint64_t(r1) * pc.s_nb1 + uint64_t(i2) * pc.s_nb2 + uint64_t(i3) * pc.s_nb3;
    uint64_t d_row = pc.dst_va + uint64_t(dst_row) * pc.d_nb1 + uint64_t(i2) * pc.d_nb2 + uint64_t(i3) * pc.d_nb3;
    float val = ConstBufF32(s_row).v[col];
    if (pc.dst_type == 0u) {
        BufF32(d_row).v[col] = val;
    } else if (pc.dst_type == 1u) {
        BufF16(d_row).v[col] = float16_t(val);
    } else {
        // GGML_TYPE_Q8_0: 32-lane subgroup block quantization (ne0 % 32 == 0 => uniform per simdgroup)
        float amax = subgroupMax(abs(val));
        float d    = amax * (1.0 / 127.0);
        float id   = (d != 0.0) ? (1.0 / d) : 0.0;
        int   q    = clamp(int(round(val * id)), -127, 127);
        uint  blk  = col >> 5u;
        uint  lane = col & 31u;
        uint64_t bva = d_row + uint64_t(blk) * 34ul;
        if (lane == 0u) {
            BufF16(bva).v[0] = float16_t(d);
        }
        int q_hi = subgroupShuffleXor(q, 1u);
        if ((lane & 1u) == 0u) {
            BufU16(bva + 2ul).v[lane >> 1u] =
                uint16_t((uint(q) & 0xFFu) | ((uint(q_hi) & 0xFFu) << 8u));
        }
    }
}
)GLSL";

struct set_rows_push {
    uint64_t src_va, ids_va, dst_va;
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t s_nb1, s_nb2, s_nb3;
    uint32_t i_nb0, i_nb1, i_nb2;
    uint32_t i_ne1, i_ne2;
    uint32_t d_nb1, d_nb2, d_nb3;
    uint32_t dst_type, ids_is_i64;
};
static_assert(sizeof(set_rows_push) <= AGX_MAX_PUSH_BYTES, "set_rows_push > 256B");

static const char * k_glsl_concat_pad = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufU32 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufU32 { uint v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t a_va;
    uint64_t b_va;
    uint64_t d_va;
    uint     ne0, ne1, ne2, ne3; // dst shape
    uint     a_ne0, a_ne1, a_ne2, a_ne3;
    uint     a_nb0, a_nb1, a_nb2, a_nb3; // in 32-bit words
    uint     b_nb0, b_nb1, b_nb2, b_nb3; // in 32-bit words
    uint     mode; // 0=CONCAT(dim=param0), 1=PAD(lp0=param0,lp1=param1,lp2=param2,lp3=param3), 2=REPEAT
    uint     p0, p1, p2, p3;
    uint     total;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.total) return;

    uint i0 = idx % pc.ne0;
    uint t1 = idx / pc.ne0;
    uint i1 = t1 % pc.ne1;
    uint t2 = t1 / pc.ne1;
    uint i2 = t2 % pc.ne2;
    uint i3 = t2 / pc.ne2;

    if (pc.mode == 0u) {
        // CONCAT along dim pc.p0
        uint c[4] = uint[4](i0, i1, i2, i3);
        uint an[4] = uint[4](pc.a_ne0, pc.a_ne1, pc.a_ne2, pc.a_ne3);
        if (c[pc.p0] < an[pc.p0]) {
            uint off = c[0] * pc.a_nb0 + c[1] * pc.a_nb1 + c[2] * pc.a_nb2 + c[3] * pc.a_nb3;
            BufU32(pc.d_va).v[idx] = ConstBufU32(pc.a_va).v[off];
        } else {
            c[pc.p0] -= an[pc.p0];
            uint off = c[0] * pc.b_nb0 + c[1] * pc.b_nb1 + c[2] * pc.b_nb2 + c[3] * pc.b_nb3;
            BufU32(pc.d_va).v[idx] = ConstBufU32(pc.b_va).v[off];
        }
    } else if (pc.mode == 1u) {
        // PAD with left-pad (p0,p1,p2,p3)
        if (i0 >= pc.p0 && i0 < pc.p0 + pc.a_ne0 &&
            i1 >= pc.p1 && i1 < pc.p1 + pc.a_ne1 &&
            i2 >= pc.p2 && i2 < pc.p2 + pc.a_ne2 &&
            i3 >= pc.p3 && i3 < pc.p3 + pc.a_ne3) {
            uint off = (i0 - pc.p0) * pc.a_nb0 + (i1 - pc.p1) * pc.a_nb1 +
                       (i2 - pc.p2) * pc.a_nb2 + (i3 - pc.p3) * pc.a_nb3;
            BufU32(pc.d_va).v[idx] = ConstBufU32(pc.a_va).v[off];
        } else {
            BufU32(pc.d_va).v[idx] = 0u;
        }
    } else {
        // REPEAT
        uint off = (i0 % pc.a_ne0) * pc.a_nb0 + (i1 % pc.a_ne1) * pc.a_nb1 +
                   (i2 % pc.a_ne2) * pc.a_nb2 + (i3 % pc.a_ne3) * pc.a_nb3;
        BufU32(pc.d_va).v[idx] = ConstBufU32(pc.a_va).v[off];
    }
}
)GLSL";

struct concat_pad_push {
    uint64_t a_va, b_va, d_va;
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t a_ne0, a_ne1, a_ne2, a_ne3;
    uint32_t a_nb0, a_nb1, a_nb2, a_nb3;
    uint32_t b_nb0, b_nb1, b_nb2, b_nb3;
    uint32_t mode, p0, p1, p2, p3, total;
};
static_assert(sizeof(concat_pad_push) <= AGX_MAX_PUSH_BYTES, "concat_pad_push > 256B");

static const char * k_glsl_ssm_conv = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 32, local_size_y = 8, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src0_va;
    uint64_t src1_va;
    uint64_t dst_va;
    uint64_t res_va;     // optional residual tensor for fused (res + conv) * res_scale + res_bias
    uint     nb01, nb02; // floats
    uint     nb11;       // floats
    uint     d_nb1, d_nb2; // floats
    uint     nc, nr, n_t, n_s;
    float    res_scale;
    float    res_bias;
} pc;

void main() {
    uint i1 = gl_GlobalInvocationID.x;
    uint i2 = gl_GlobalInvocationID.y;
    uint i3 = gl_WorkGroupID.z;
    if (i1 >= pc.nr || i2 >= pc.n_t || i3 >= pc.n_s) return;

    uint s0 = i3 * pc.nb02 + i1 * pc.nb01 + i2;
    uint s1 = i1 * pc.nb11;
    float sum = 0.0;
    for (uint i0 = 0u; i0 < pc.nc; ++i0) {
        sum += ConstBufF32(pc.src0_va).v[s0 + i0] * ConstBufF32(pc.src1_va).v[s1 + i0];
    }
    uint d_idx = i3 * pc.d_nb2 + i2 * pc.d_nb1 + i1;
    if (pc.res_va != 0ul) {
        sum = (ConstBufF32(pc.res_va).v[d_idx] + sum) * pc.res_scale + pc.res_bias;
    }
    BufF32(pc.dst_va).v[d_idx] = sum;
}
)GLSL";

struct ssm_conv_push {
    uint64_t src0_va, src1_va, dst_va, res_va;
    uint32_t nb01, nb02, nb11, d_nb1, d_nb2;
    uint32_t nc, nr, n_t, n_s;
    float    res_scale, res_bias;
};
static_assert(sizeof(ssm_conv_push) <= AGX_MAX_PUSH_BYTES, "ssm_conv_push > 256B");

static const char * k_glsl_rope = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t pos_va;
    uint64_t freq_va;
    uint64_t dst_va;
    uint     ne0, ne1, ne2, ne3;
    uint     s_nb1, s_nb2, s_nb3; // floats
    uint     d_nb1, d_nb2, d_nb3; // floats
    uint     n_dims;
    uint     mode; // bit 1 = NEOX
    float    theta_scale;
    float    freq_scale;
    float    attn_factor;
    uint     pad0;
} pc;

void main() {
    uint pair = gl_GlobalInvocationID.x;
    uint h    = gl_WorkGroupID.y;
    uint t3   = gl_WorkGroupID.z;
    uint half_d = pc.ne0 >> 1u;
    if (pair >= half_d || h >= pc.ne1 || t3 >= pc.ne2 * pc.ne3) return;

    uint i2 = t3 % pc.ne2;
    uint i3 = t3 / pc.ne2;
    uint s_base = h * pc.s_nb1 + i2 * pc.s_nb2 + i3 * pc.s_nb3;
    uint d_base = h * pc.d_nb1 + i2 * pc.d_nb2 + i3 * pc.d_nb3;

    bool is_neox = (pc.mode & 2u) != 0u;
    uint i0 = is_neox ? pair : (2u * pair);
    uint i1 = is_neox ? (pair + (pc.n_dims >> 1u)) : (2u * pair + 1u);

    if (pair * 2u >= pc.n_dims) {
        if (is_neox) {
            uint k0 = pair * 2u;
            BufF32(pc.dst_va).v[d_base + k0]      = ConstBufF32(pc.src_va).v[s_base + k0];
            BufF32(pc.dst_va).v[d_base + k0 + 1u] = ConstBufF32(pc.src_va).v[s_base + k0 + 1u];
        } else {
            BufF32(pc.dst_va).v[d_base + i0] = ConstBufF32(pc.src_va).v[s_base + i0];
            BufF32(pc.dst_va).v[d_base + i1] = ConstBufF32(pc.src_va).v[s_base + i1];
        }
        return;
    }

    float pos   = float(ConstBufI32(pc.pos_va).v[i2]);
    float freq  = pow(pc.theta_scale, float(pair)) * pc.freq_scale;
    if (pc.freq_va != 0ul) {
        freq /= ConstBufF32(pc.freq_va).v[pair];
    }
    float theta = pos * freq;
    float ct    = cos(theta) * pc.attn_factor;
    float st    = sin(theta) * pc.attn_factor;

    float x0 = ConstBufF32(pc.src_va).v[s_base + i0];
    float x1 = ConstBufF32(pc.src_va).v[s_base + i1];
    BufF32(pc.dst_va).v[d_base + i0] = x0 * ct - x1 * st;
    BufF32(pc.dst_va).v[d_base + i1] = x0 * st + x1 * ct;
}
)GLSL";

struct rope_push {
    uint64_t src_va, pos_va, freq_va, dst_va;
    uint32_t ne0, ne1, ne2, ne3;
    uint32_t s_nb1, s_nb2, s_nb3;
    uint32_t d_nb1, d_nb2, d_nb3;
    uint32_t n_dims, mode;
    float    theta_scale, freq_scale, attn_factor;
    uint32_t pad0;
};
static_assert(sizeof(rope_push) <= AGX_MAX_PUSH_BYTES, "rope_push > 256B");

static const char * k_glsl_topk_argsort = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufI32 { int v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t dst_va;
    uint     ne0;      // row length
    uint     k_out;    // top-k count (<= 256)
    uint     n_rows;
    uint     desc;     // 1 = descending, 0 = ascending
} pc;

shared float s_val[512];
shared int   s_idx[512];
shared float s_red_v[8];
shared int   s_red_i[8];
shared uint  s_cnt;

bool comes_before(float va, int ia, float vb, int ib, bool is_desc) {
    return is_desc ? (va > vb || (va == vb && ia < ib))
                   : (va < vb || (va == vb && ia < ib));
}

// Parallel in-SRAM bitonic sort of s_val[0..n-1] / s_idx[0..n-1] (n = power-of-2 <= 512).
void bitonic_sort_sram(uint n, uint tid, bool is_desc) {
    const uint half_n = n >> 1u;
    for (uint k = 2u; k <= n; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            if (tid < half_n) {
                uint low_mask = j - 1u;
                uint lo = ((tid & ~low_mask) << 1u) | (tid & low_mask);
                uint hi = lo | j;
                bool asc_dir = ((lo & k) == 0u);
                float va = s_val[lo];
                int   ia = s_idx[lo];
                float vb = s_val[hi];
                int   ib = s_idx[hi];
                bool a_before_b = comes_before(va, ia, vb, ib, is_desc);
                bool need_swap  = asc_dir ? (!a_before_b) : a_before_b;
                if (need_swap) {
                    s_val[lo] = vb; s_idx[lo] = ib;
                    s_val[hi] = va; s_idx[hi] = ia;
                }
            }
            barrier();
        }
    }
}

void main() {
    const uint row = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    if (row >= pc.n_rows || pc.k_out == 0u || pc.ne0 == 0u) return;

    const uint s_base   = row * pc.ne0;
    const uint d_base   = row * pc.k_out;
    const uint K        = pc.k_out;
    const bool is_desc  = (pc.desc != 0u);
    const float sentinel = is_desc ? uintBitsToFloat(0xFF800000u) : uintBitsToFloat(0x7F800000u);

    // Fast Path A: ne0 <= 512 (covers Inkling 2 Mini 128-expert router & ARGSORT)
    // 1 global read pass + 7-stage parallel bitonic sort in SRAM (~1.5 us vs 2,700 us!)
    if (pc.ne0 <= 512u) {
        uint n_pad = 1u;
        while (n_pad < pc.ne0) n_pad <<= 1u;
        if (tid < n_pad) {
            s_val[tid] = (tid < pc.ne0) ? ConstBufF32(pc.src_va).v[s_base + tid] : sentinel;
            s_idx[tid] = (tid < pc.ne0) ? int(tid) : 0x7FFFFFFF;
        }
        if (tid + 256u < n_pad) {
            uint i2 = tid + 256u;
            s_val[i2] = (i2 < pc.ne0) ? ConstBufF32(pc.src_va).v[s_base + i2] : sentinel;
            s_idx[i2] = (i2 < pc.ne0) ? int(i2) : 0x7FFFFFFF;
        }
        barrier();

        bitonic_sort_sram(n_pad, tid, is_desc);

        if (tid < K) {
            BufI32(pc.dst_va).v[d_base + tid] = s_idx[tid];
        }
        if (tid + 256u < K) {
            BufI32(pc.dst_va).v[d_base + tid + 256u] = s_idx[tid + 256u];
        }
        return;
    }

    // Fast Path B: ne0 > 512, k_out <= 256 (e.g. DSpark TOP_K on 200,058 logits)
    // Pass 1: per-thread local best -> sort 256 distinct winners -> guaranteed threshold `thresh`.
    float local_best = sentinel;
    int   local_bi   = 0x7FFFFFFF;
    for (uint i = tid; i < pc.ne0; i += 256u) {
        float v = ConstBufF32(pc.src_va).v[s_base + i];
        if (comes_before(v, int(i), local_best, local_bi, is_desc)) {
            local_best = v;
            local_bi   = int(i);
        }
    }
    s_val[tid] = local_best;
    s_idx[tid] = local_bi;
    if (tid == 0u) s_cnt = 0u;
    barrier();

    bitonic_sort_sram(256u, tid, is_desc);
    float thresh = s_val[min(K - 1u, 255u)];
    barrier();

    // Pass 2: collect all elements passing `thresh` into s_val[0..511]
    for (uint i = tid; i < pc.ne0; i += 256u) {
        float v = ConstBufF32(pc.src_va).v[s_base + i];
        bool pass = is_desc ? (v >= thresh) : (v <= thresh);
        if (pass) {
            uint slot = atomicAdd(s_cnt, 1u);
            if (slot < 512u) {
                s_val[slot] = v;
                s_idx[slot] = int(i);
            }
        }
    }
    barrier();

    uint total_cands = s_cnt;
    if (total_cands <= 512u) {
        if (tid >= total_cands) {
            s_val[tid] = sentinel;
            s_idx[tid] = 0x7FFFFFFF;
        }
        if (tid + 256u >= total_cands) {
            s_val[tid + 256u] = sentinel;
            s_idx[tid + 256u] = 0x7FFFFFFF;
        }
        barrier();

        bitonic_sort_sram(512u, tid, is_desc);
        if (tid < K) {
            BufI32(pc.dst_va).v[d_base + tid] = s_idx[tid];
        }
        return;
    }

    // Extreme-tie fallback (>512 elements tied at `thresh`):
    // Exact O(K * N / 256) monotonic (last_v, last_i) selection with zero s_picked[] scan.
    const uint sg = tid >> 5u;
    const uint ln = tid & 31u;
    float last_v  = is_desc ? uintBitsToFloat(0x7F800000u) : uintBitsToFloat(0xFF800000u);
    int   last_i  = -1;

    for (uint k = 0u; k < K; ++k) {
        float bv = sentinel;
        int   bi = 0x7FFFFFFF;
        for (uint i = tid; i < pc.ne0; i += 256u) {
            int   ii = int(i);
            float v  = ConstBufF32(pc.src_va).v[s_base + i];
            if (k > 0u) {
                bool already = is_desc ? (v > last_v || (v == last_v && ii <= last_i))
                                       : (v < last_v || (v == last_v && ii <= last_i));
                if (already) continue;
            }
            if (comes_before(v, ii, bv, bi, is_desc)) {
                bv = v;
                bi = ii;
            }
        }
        for (uint off = 16u; off > 0u; off >>= 1u) {
            float ov = subgroupShuffleXor(bv, off);
            int   oi = subgroupShuffleXor(bi, off);
            if (comes_before(ov, oi, bv, bi, is_desc)) { bv = ov; bi = oi; }
        }
        if (ln == 0u) { s_red_v[sg] = bv; s_red_i[sg] = bi; }
        barrier();
        if (sg == 0u) {
            float v = (ln < 8u) ? s_red_v[ln] : sentinel;
            int   i = (ln < 8u) ? s_red_i[ln] : 0x7FFFFFFF;
            for (uint off = 16u; off > 0u; off >>= 1u) {
                float ov = subgroupShuffleXor(v, off);
                int   oi = subgroupShuffleXor(i, off);
                if (comes_before(ov, oi, v, i, is_desc)) { v = ov; i = oi; }
            }
            if (ln == 0u) {
                s_red_v[0] = v;
                s_red_i[0] = i;
                BufI32(pc.dst_va).v[d_base + k] = i;
            }
        }
        barrier();
        last_v = s_red_v[0];
        last_i = s_red_i[0];
        barrier();
    }
}
)GLSL";

struct topk_argsort_push {
    uint64_t src_va, dst_va;
    uint32_t ne0, k_out, n_rows, desc;
};
static_assert(sizeof(topk_argsort_push) <= AGX_MAX_PUSH_BYTES, "topk_argsort_push > 256B");

static const char * k_glsl_glu = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t g_va;
    uint64_t u_va;
    uint64_t d_va;
    uint     ne0, n_rows;
    uint     g_nb1, u_nb1, d_nb1; // floats
    uint     glu_op; // 0=REGLU, 1=GEGLU, 2=SWIGLU, 3=SWIGLU_OAI, 6=SWIGLU_CLAMP
    uint     swapped;
    float    alpha, limit;
} pc;

void main() {
    uint col = gl_GlobalInvocationID.x;
    uint row = gl_WorkGroupID.y;
    if (col >= pc.ne0 || row >= pc.n_rows) return;

    float g = ConstBufF32(pc.g_va).v[row * pc.g_nb1 + col];
    float u = ConstBufF32(pc.u_va).v[row * pc.u_nb1 + col];
    if (pc.swapped != 0u) { float tmp = g; g = u; u = tmp; }

    float r = 0.0;
    if (pc.glu_op == 0u) {
        r = max(g, 0.0) * u;
    } else if (pc.glu_op == 1u) {
        float act = 0.5 * g * (1.0 + tanh(0.7978845608028654 * g * (1.0 + 0.044715 * g * g)));
        r = act * u;
    } else if (pc.glu_op == 2u) {
        r = (g / (1.0 + exp(-g))) * u;
    } else if (pc.glu_op == 3u) {
        float cg = min(g, pc.limit);
        float cu = clamp(u, -pc.limit, pc.limit);
        r = (cg / (1.0 + exp(-pc.alpha * cg))) * (cu + 1.0);
    } else if (pc.glu_op == 99u) {
        // Inkling 2 SiTU: beta*tanh(g/beta)*sigmoid(g) * (lb > 0 ? lb*tanh(u/lb) : u)
        float act = (pc.alpha * tanh(g / pc.alpha)) * (1.0 / (1.0 + exp(-g)));
        float lin = (pc.limit > 0.0) ? (pc.limit * tanh(u / pc.limit)) : u;
        r = act * lin;
    } else {
        float cg = min(g, pc.limit);
        float cu = clamp(u, -pc.limit, pc.limit);
        r = (cg / (1.0 + exp(-cg))) * cu;
    }
    BufF32(pc.d_va).v[row * pc.d_nb1 + col] = r;
}
)GLSL";

struct glu_push {
    uint64_t g_va, u_va, d_va;
    uint32_t ne0, n_rows, g_nb1, u_nb1, d_nb1;
    uint32_t glu_op, swapped;
    float    alpha, limit;
};
static_assert(sizeof(glu_push) <= AGX_MAX_PUSH_BYTES, "glu_push > 256B");

static const char * k_glsl_add_id = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t a_va;
    uint64_t b_va;
    uint64_t ids_va;
    uint64_t d_va;
    uint     ne0, ne1, ne2;
    uint     a_nb1, a_nb2; // floats
    uint     b_nb1;        // floats
    uint     i_nb1;        // ints
    uint     d_nb1, d_nb2; // floats
} pc;

void main() {
    uint i0 = gl_GlobalInvocationID.x;
    uint i1 = gl_WorkGroupID.y;
    uint i2 = gl_WorkGroupID.z;
    if (i0 >= pc.ne0 || i1 >= pc.ne1 || i2 >= pc.ne2) return;

    int eid = ConstBufI32(pc.ids_va).v[i2 * pc.i_nb1 + i1];
    float a = ConstBufF32(pc.a_va).v[i2 * pc.a_nb2 + i1 * pc.a_nb1 + i0];
    float b = (eid < 0) ? 0.0 : ConstBufF32(pc.b_va).v[uint(eid) * pc.b_nb1 + i0];
    BufF32(pc.d_va).v[i2 * pc.d_nb2 + i1 * pc.d_nb1 + i0] = a + b;
}
)GLSL";

struct add_id_push {
    uint64_t a_va, b_va, ids_va, d_va;
    uint32_t ne0, ne1, ne2;
    uint32_t a_nb1, a_nb2, b_nb1, i_nb1, d_nb1, d_nb2;
};
static_assert(sizeof(add_id_push) <= AGX_MAX_PUSH_BYTES, "add_id_push > 256B");

// Fused Inkling 2 controller residual mix + optional trailing weighted RMSNorm (B < 16 fastpath):
//   y[i] = (cur[i] - tanh(forget[i]) * cur[i]) + tanh(gain[i]) * msg[i]
//   if norm_w_va != 0: out[i] = (y[i] * rsqrt(mean(y^2) + eps)) * norm_w[i]
static const char * k_glsl_ctrl_mix = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t cur_va;
    uint64_t msg_va;
    uint64_t forget_va;
    uint64_t gain_va;
    uint64_t norm_w_va;
    uint64_t out_va;
    uint     n_embd;
    uint     n_tokens;
    float    eps_norm;
    uint     mode;    // 0=full(cur-tanh(f)*cur+tanh(g)*msg), 1=forget_only(cur-tanh(f)*cur), 2=gain_only(cur+tanh(g)*msg)
    uint     cur_nb0; // floats (1 = contiguous, Q = strided CONT(VIEW_3D) absorption)
    uint     cur_nb1; // floats (n_embd = contiguous, Q*n_embd = strided)
    uint     msg_nb0; // floats
    uint     msg_nb1; // floats
} pc;

shared float s_y[4096];
shared float s_red[8];

float wg_sum(float val, uint sg, uint ln) {
    float s = subgroupAdd(val);
    if (ln == 0u) s_red[sg] = s;
    barrier();
    if (sg == 0u) {
        float v = (ln < 8u) ? s_red[ln] : 0.0;
        float t = subgroupAdd(v);
        if (ln == 0u) s_red[0] = t;
    }
    barrier();
    float res = s_red[0];
    barrier();
    return res;
}

void main() {
    const uint tok = gl_WorkGroupID.x;
    if (tok >= pc.n_tokens) return;

    const uint tid  = gl_LocalInvocationID.x;
    const uint sg   = tid >> 5u;
    const uint ln   = tid & 31u;
    const uint D    = pc.n_embd;
    const uint base = tok * D;
    const uint c_nb0 = (pc.cur_nb0 > 0u) ? pc.cur_nb0 : 1u;
    const uint c_nb1 = (pc.cur_nb1 > 0u) ? pc.cur_nb1 : D;
    const uint m_nb0 = (pc.msg_nb0 > 0u) ? pc.msg_nb0 : 1u;
    const uint m_nb1 = (pc.msg_nb1 > 0u) ? pc.msg_nb1 : D;
    const uint c_base = tok * c_nb1;
    const uint m_base = tok * m_nb1;

    if (pc.mode == 1u) {
        for (uint i = tid; i < D; i += 256u) {
            float c = ConstBufF32(pc.cur_va).v[c_base + i * c_nb0];
            float f = tanh(ConstBufF32(pc.forget_va).v[base + i]);
            BufF32(pc.out_va).v[base + i] = c - f * c;
        }
        return;
    }

    if (pc.norm_w_va == 0ul) {
        for (uint i = tid; i < D; i += 256u) {
            float c = ConstBufF32(pc.cur_va).v[c_base + i * c_nb0];
            float m = ConstBufF32(pc.msg_va).v[m_base + i * m_nb0];
            float g = tanh(ConstBufF32(pc.gain_va).v[base + i]);
            float y = (pc.mode == 2u)
                ? (c + g * m)
                : ((c - tanh(ConstBufF32(pc.forget_va).v[base + i]) * c) + g * m);
            BufF32(pc.out_va).v[base + i] = y;
        }
        return;
    }

    float ss = 0.0;
    for (uint i = tid; i < D; i += 256u) {
        float c = ConstBufF32(pc.cur_va).v[c_base + i * c_nb0];
        float m = ConstBufF32(pc.msg_va).v[m_base + i * m_nb0];
        float g = tanh(ConstBufF32(pc.gain_va).v[base + i]);
        float y = (pc.mode == 2u)
            ? (c + g * m)
            : ((c - tanh(ConstBufF32(pc.forget_va).v[base + i]) * c) + g * m);
        s_y[i] = y;
        ss += y * y;
    }
    float sc = inversesqrt(wg_sum(ss, sg, ln) / float(D) + pc.eps_norm);
    for (uint i = tid; i < D; i += 256u) {
        BufF32(pc.out_va).v[base + i] = (s_y[i] * sc) * ConstBufF32(pc.norm_w_va).v[i];
    }
}
)GLSL";

struct ctrl_mix_push {
    uint64_t cur_va, msg_va, forget_va, gain_va, norm_w_va, out_va;
    uint32_t n_embd, n_tokens;
    float    eps_norm;
    uint32_t mode;
    uint32_t cur_nb0, cur_nb1, msg_nb0, msg_nb1;
};
static_assert(sizeof(ctrl_mix_push) <= AGX_MAX_PUSH_BYTES, "ctrl_mix_push > 256B");

// Fused CONCAT(cur, msg, dim=1) + per-row RMS_NORM(eps) for B < 16 control() fastpath.
// Supports strided CONT(VIEW_3D) absorption on cur and msg.
// Grid: (2, n_tokens, 1), Local: (256, 1, 1).
static const char * k_glsl_ctrl_norm_cat = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t cur_va;
    uint64_t msg_va;
    uint64_t dst_va;
    uint     n_embd;
    uint     n_tokens;
    float    eps;
    uint     cur_nb0;
    uint     cur_nb1;
    uint     msg_nb0;
    uint     msg_nb1;
} pc;

shared float s_red[8];

float wg_sum(float val, uint sg, uint ln) {
    float s = subgroupAdd(val);
    if (ln == 0u) s_red[sg] = s;
    barrier();
    if (sg == 0u) {
        float v = (ln < 8u) ? s_red[ln] : 0.0;
        float t = subgroupAdd(v);
        if (ln == 0u) s_red[0] = t;
    }
    barrier();
    float res = s_red[0];
    barrier();
    return res;
}

void main() {
    const uint row_id = gl_WorkGroupID.x; // 0 = cur, 1 = msg
    const uint tok    = gl_WorkGroupID.y;
    if (row_id >= 2u || tok >= pc.n_tokens) return;

    const uint tid = gl_LocalInvocationID.x;
    const uint sg  = tid >> 5u;
    const uint ln  = tid & 31u;
    const uint D   = pc.n_embd;
    const uint64_t src_va = (row_id == 0u) ? pc.cur_va : pc.msg_va;
    const uint raw_nb0 = (row_id == 0u) ? pc.cur_nb0 : pc.msg_nb0;
    const uint raw_nb1 = (row_id == 0u) ? pc.cur_nb1 : pc.msg_nb1;
    const uint s_nb0   = (raw_nb0 > 0u) ? raw_nb0 : 1u;
    const uint s_nb1   = (raw_nb1 > 0u) ? raw_nb1 : D;
    const uint s_base  = tok * s_nb1;
    const uint d_base  = (tok * 2u + row_id) * D;

    float ss = 0.0;
    if (s_nb0 == 1u) {
        for (uint i = tid; i < D; i += 256u) {
            float x = ConstBufF32(src_va).v[s_base + i];
            ss += x * x;
        }
    } else {
        for (uint i = tid; i < D; i += 256u) {
            float x = ConstBufF32(src_va).v[s_base + i * s_nb0];
            ss += x * x;
        }
    }
    float sc = inversesqrt(wg_sum(ss, sg, ln) / float(D) + pc.eps);
    if (s_nb0 == 1u) {
        for (uint i = tid; i < D; i += 256u) {
            BufF32(pc.dst_va).v[d_base + i] = ConstBufF32(src_va).v[s_base + i] * sc;
        }
    } else {
        for (uint i = tid; i < D; i += 256u) {
            BufF32(pc.dst_va).v[d_base + i] = ConstBufF32(src_va).v[s_base + i * s_nb0] * sc;
        }
    }
}
)GLSL";

struct ctrl_norm_cat_push {
    uint64_t cur_va, msg_va, dst_va;
    uint32_t n_embd, n_tokens;
    float    eps;
    uint32_t cur_nb0, cur_nb1, msg_nb0, msg_nb1;
};
static_assert(sizeof(ctrl_norm_cat_push) <= AGX_MAX_PUSH_BYTES, "ctrl_norm_cat_push > 256B");

// Fused Inkling 2 `mix()` depth-mix attention (+ optional CONCAT & inline RMS_NORM on extra state):
//   if extra_va != 0: state S-1 comes from extra_va (optionally RMS-normalized inline with extra_eps),
//                     while states 0..S-2 come from states_va (eliminating CONCAT + RMS_NORM!).
//   scores = softmax((states @ queries) * scale)
//   out[q, d, tok] = sum_s scores[s, q, tok] * states[d, s, tok]
// Grid: (n_queries, n_tokens, 1), Local: (256, 1, 1), n_states <= 32.
static const char * k_glsl_mix_attn = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t states_va;
    uint64_t extra_va;     // optional last state (eliminates CONCAT(blocks, norm_cur, dim=1))
    uint64_t queries_va;
    uint64_t out_va;
    uint     n_embd;
    uint     n_states;     // total states (including extra if extra_va != 0)
    uint     n_queries;
    uint     n_tokens;
    uint     q_ne2;
    uint     extra_do_rms; // 1 = RMS_NORM(extra_va, extra_eps) inline
    float    scale;
    float    extra_eps;
} pc;

shared float s_w[32];
shared float s_red[8];
shared float s_extra_inv_rms;

void main() {
    const uint q   = gl_WorkGroupID.x;
    const uint tok = gl_WorkGroupID.y;
    if (q >= pc.n_queries || tok >= pc.n_tokens) return;

    const uint tid = gl_LocalInvocationID.x;
    const uint sg  = tid >> 5u;
    const uint ln  = tid & 31u;
    const uint D   = pc.n_embd;
    const uint S   = pc.n_states;
    const uint Q   = pc.n_queries;
    const bool has_extra = (pc.extra_va != 0ul);
    const uint S_base = has_extra ? (S - 1u) : S;

    const uint q_tok  = tok % pc.q_ne2;
    const uint q_base = (q_tok * Q + q) * D;
    const uint s_tok  = tok * S_base * D;
    const uint e_tok  = tok * D;

    // 0. Optional inline RMS_NORM scale for extra_va (state S-1)
    if (has_extra) {
        if (pc.extra_do_rms != 0u) {
            float ss = 0.0;
            for (uint d = tid; d < D; d += 256u) {
                float x = ConstBufF32(pc.extra_va).v[e_tok + d];
                ss += x * x;
            }
            float sg_s = subgroupAdd(ss);
            if (ln == 0u) s_red[sg] = sg_s;
            barrier();
            if (sg == 0u) {
                float v = (ln < 8u) ? s_red[ln] : 0.0;
                float tot = subgroupAdd(v);
                if (ln == 0u) s_extra_inv_rms = inversesqrt(tot / float(D) + pc.extra_eps);
            }
            barrier();
        } else if (tid == 0u) {
            s_extra_inv_rms = 1.0;
        }
        barrier();
    }

    // 1. 8 simdgroups compute dot(states[s], query[q]) concurrently across states
    for (uint s = sg; s < S; s += 8u) {
        float acc = 0.0;
        if (has_extra && s == S_base) {
            for (uint d = ln; d < D; d += 32u) {
                acc += ConstBufF32(pc.extra_va).v[e_tok + d] * ConstBufF32(pc.queries_va).v[q_base + d];
            }
            float dot_val = subgroupAdd(acc) * s_extra_inv_rms;
            if (ln == 0u) s_w[s] = dot_val * pc.scale;
        } else {
            const uint sb = s_tok + s * D;
            for (uint d = ln; d < D; d += 32u) {
                acc += ConstBufF32(pc.states_va).v[sb + d] * ConstBufF32(pc.queries_va).v[q_base + d];
            }
            float dot_val = subgroupAdd(acc);
            if (ln == 0u) s_w[s] = dot_val * pc.scale;
        }
    }
    barrier();

    // 2. Single-simdgroup exact softmax over S <= 32 states
    if (sg == 0u) {
        float v  = (ln < S) ? s_w[ln] : uintBitsToFloat(0xFF800000u);
        float mx = subgroupMax(v);
        float e  = (ln < S) ? exp(v - mx) : 0.0;
        float sm = subgroupAdd(e);
        if (ln < S) {
            float p = (sm > 0.0) ? (e / sm) : 0.0;
            if (has_extra && ln == S_base) {
                p *= s_extra_inv_rms;
            }
            s_w[ln] = p;
        }
    }
    barrier();

    // 3. Weighted sum -> out[(tok * D + d) * Q + q]
    const uint out_tok = tok * D * Q;
    for (uint d = tid; d < D; d += 256u) {
        float acc = 0.0;
        for (uint s = 0u; s < S_base; ++s) {
            acc += s_w[s] * ConstBufF32(pc.states_va).v[s_tok + s * D + d];
        }
        if (has_extra) {
            acc += s_w[S_base] * ConstBufF32(pc.extra_va).v[e_tok + d];
        }
        BufF32(pc.out_va).v[out_tok + d * Q + q] = acc;
    }
}
)GLSL";

struct mix_attn_push {
    uint64_t states_va, extra_va, queries_va, out_va;
    uint32_t n_embd, n_states, n_queries, n_tokens, q_ne2, extra_do_rms;
    float    scale, extra_eps;
};
static_assert(sizeof(mix_attn_push) <= AGX_MAX_PUSH_BYTES, "mix_attn_push > 256B");

// Fused MoE expert weighted sum: out[tok, d] = sum_{k=0..n_exp-1} experts[tok, k, d] * weights[tok, k]
// Replaces MUL(experts, weights) + (n_exp-1) serial ADDs with 1 dispatch.
static const char * k_glsl_moe_wsum = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t exp_va;
    uint64_t w_va;
    uint64_t out_va;
    uint     D;
    uint     n_exp;
    uint     n_tokens;
    uint     pad0;
} pc;

void main() {
    const uint d   = gl_GlobalInvocationID.x;
    const uint tok = gl_WorkGroupID.y;
    if (d >= pc.D || tok >= pc.n_tokens || pc.n_exp == 0u) return;

    const uint e_base = tok * pc.n_exp * pc.D + d;
    const uint w_base = tok * pc.n_exp;
    // Exact sequential MUL + ADD matching unfused ggml_mul + (K-1) ggml_add (no FMA contraction)
    precise float acc = ConstBufF32(pc.exp_va).v[e_base] * ConstBufF32(pc.w_va).v[w_base];
    for (uint k = 1u; k < pc.n_exp; ++k) {
        precise float prod = ConstBufF32(pc.exp_va).v[e_base + k * pc.D] * ConstBufF32(pc.w_va).v[w_base + k];
        acc = acc + prod;
    }
    BufF32(pc.out_va).v[tok * pc.D + d] = acc;
}
)GLSL";

struct moe_wsum_push {
    uint64_t exp_va, w_va, out_va;
    uint32_t D, n_exp, n_tokens, pad0;
};
static_assert(sizeof(moe_wsum_push) <= AGX_MAX_PUSH_BYTES, "moe_wsum_push > 256B");

// ----------------------------------------------------------------------------
// 7. Wave 3 Fused Kernels:
//    - k_glsl_prep_norm_fwht_store: [MUL(scale)] -> (L2_CLIP | RMS_NORM) -> [MUL(tau)] -> [FWHT64/128] -> (F32 | SET_ROWS(F32/F16/Q8_0))
//    - k_glsl_full_shortconv:       CONCAT + CPY(state) + SSM_CONV + [dynamic conv] + ADD + SCALE [+ post ADD]
//    - k_glsl_router_topk_norm:     SIGMOID + [ADD(bias)] + ARGSORT(DESC) + VIEW(K) + GET_ROWS + NORM(1e-20)
//    - k_glsl_rel_bias_gather:      PAD + MUL_MAT(Rcur) + SUB(sinks) + CONT(PERM) + GET_ROWS(rel_pos) + CONT(PERM)
// ----------------------------------------------------------------------------
static const char * k_glsl_prep_norm_fwht_store = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require

layout(local_size_x = 32, local_size_y = 4, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer ConstBufI64 { int64_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufF16 { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufU16 { uint16_t v[]; };

// flags: bit0=L2_CLIP, bit1=RMS_NORM, bit2=FWHT_64, bit3=FWHT_128
// store_mode: 0=F32_CONT, 1=SET_ROWS_F32, 2=SET_ROWS_F16, 3=SET_ROWS_Q8_0
layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t scale_va;   // optional [head_dim, n_head] F32
    uint64_t tau_va;     // optional [n_tokens] F32
    uint64_t ids_va;     // optional SET_ROWS indices
    uint64_t dst_va;
    uint     head_dim;   // <= 128
    uint     n_head;
    uint     n_tokens;
    uint     flags;
    uint     store_mode;
    uint     ids_is_i64;
    uint     i_nb0;      // bytes
    uint     d_nb1;      // bytes (SET_ROWS row stride)
    float    param;      // clip_thresh (L2_CLIP) or eps (RMS_NORM)
    uint     s_nb1;      // floats
    uint     s_nb2;      // floats
} pc;

void main() {
    const uint lane = gl_LocalInvocationID.x;
    const uint head = gl_WorkGroupID.x * 4u + gl_LocalInvocationID.y;
    const uint tok  = gl_WorkGroupID.y;
    if (head >= pc.n_head || tok >= pc.n_tokens) return;

    const uint D = pc.head_dim;
    const uint s_base = tok * pc.s_nb2 + head * pc.s_nb1;

    float reg[4];
    [[unroll]] for (uint k = 0u; k < 4u; ++k) {
        const uint d = k * 32u + lane;
        if (d < D) {
            float v = ConstBufF32(pc.src_va).v[s_base + d];
            if (pc.scale_va != 0ul) {
                v *= ConstBufF32(pc.scale_va).v[head * D + d];
            }
            reg[k] = v;
        } else {
            reg[k] = 0.0;
        }
    }

    if ((pc.flags & 3u) != 0u) {
        // Match k_glsl_row_reduce's exact 32-lane block reduction order (0..31, 32..63, 64..95, 96..127)
        float s0 = subgroupAdd(reg[0] * reg[0]);
        float s1 = (D > 32u) ? subgroupAdd(reg[1] * reg[1]) : 0.0;
        float s2 = (D > 64u) ? subgroupAdd(reg[2] * reg[2]) : 0.0;
        float s3 = (D > 96u) ? subgroupAdd(reg[3] * reg[3]) : 0.0;
        float sum_sq = (s0 + s1) + (s2 + s3);
        if ((pc.flags & 1u) != 0u) {
            // Exact ggml L2_NORM(x, t) followed by SCALE(t): (x * (1 / max(||x||, t))) * t
            float inv_norm = 1.0 / max(sqrt(sum_sq), pc.param);
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                precise float t = reg[k] * inv_norm;
                reg[k] = t * pc.param;
            }
        } else {
            float sc = inversesqrt(sum_sq / float(D) + pc.param);
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                reg[k] *= sc;
            }
        }
    }

    if (pc.tau_va != 0ul) {
        float tv = ConstBufF32(pc.tau_va).v[tok];
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            reg[k] *= tv;
        }
    }

    if ((pc.flags & 12u) != 0u) {
        const bool is_128 = (pc.flags & 8u) != 0u;
        const float fsc   = is_128 ? 0.08838834764831845 : 0.125;
        const uint el_w   = (D + 31u) >> 5u;

        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            if (k < el_w) reg[k] *= fsc;
        }
        [[unroll]] for (uint h = 1u; h < 32u; h <<= 1u) {
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                if (k < el_w) {
                    float a = reg[k];
                    float b = subgroupShuffleXor(a, h);
                    reg[k] = ((lane & h) == 0u) ? (a + b) : (b - a);
                }
            }
        }
        // Stage h = 32
        {
            float a0 = reg[0], a1 = reg[1];
            reg[0] = a0 + a1;
            reg[1] = a0 - a1;
        }
        if (el_w >= 4u) {
            float a2 = reg[2], a3 = reg[3];
            reg[2] = a2 + a3;
            reg[3] = a2 - a3;
        }
        // Stage h = 64 (only for FWHT_128)
        if (is_128) {
            float b0 = reg[0], b1 = reg[1], b2 = reg[2], b3 = reg[3];
            reg[0] = b0 + b2;
            reg[1] = b1 + b3;
            reg[2] = b0 - b2;
            reg[3] = b1 - b3;
        }
    }

    if (pc.store_mode == 0u) {
        const uint d_base = (tok * pc.n_head + head) * D;
        [[unroll]] for (uint k = 0u; k < 4u; ++k) {
            const uint d = k * 32u + lane;
            if (d < D) {
                BufF32(pc.dst_va).v[d_base + d] = reg[k];
            }
        }
    } else {
        int64_t dst_row = (pc.ids_is_i64 != 0u)
            ? ConstBufI64(pc.ids_va + uint64_t(tok) * pc.i_nb0).v[0]
            : int64_t(ConstBufI32(pc.ids_va + uint64_t(tok) * pc.i_nb0).v[0]);
        if (dst_row < 0l) return;

        uint64_t d_row = pc.dst_va + uint64_t(dst_row) * pc.d_nb1;
        if (pc.store_mode == 1u) {
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                const uint d = k * 32u + lane;
                if (d < D) BufF32(d_row).v[head * D + d] = reg[k];
            }
        } else if (pc.store_mode == 2u) {
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                const uint d = k * 32u + lane;
                if (d < D) BufF16(d_row).v[head * D + d] = float16_t(reg[k]);
            }
        } else {
            // GGML_TYPE_Q8_0: each k in [0..D/32-1] is one exact 32-lane block
            const uint n_blks_head = D >> 5u;
            const uint blk_base    = head * n_blks_head;
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                if (k < n_blks_head) {
                    float v    = reg[k];
                    float amax = subgroupMax(abs(v));
                    float sc   = amax * (1.0 / 127.0);
                    float id   = (sc != 0.0) ? (1.0 / sc) : 0.0;
                    int   q    = clamp(int(round(v * id)), -127, 127);
                    uint64_t bva = d_row + uint64_t(blk_base + k) * 34ul;
                    if (lane == 0u) {
                        BufF16(bva).v[0] = float16_t(sc);
                    }
                    int q_hi = subgroupShuffleXor(q, 1u);
                    if ((lane & 1u) == 0u) {
                        BufU16(bva + 2ul).v[lane >> 1u] =
                            uint16_t((uint(q) & 0xFFu) | ((uint(q_hi) & 0xFFu) << 8u));
                    }
                }
            }
        }
    }
}
)GLSL";

struct prep_norm_fwht_store_push {
    uint64_t src_va, scale_va, tau_va, ids_va, dst_va;
    uint32_t head_dim, n_head, n_tokens, flags, store_mode, ids_is_i64, i_nb0, d_nb1;
    float    param;
    uint32_t s_nb1, s_nb2;
};
static_assert(sizeof(prep_norm_fwht_store_push) <= AGX_MAX_PUSH_BYTES, "prep_norm_fwht_store_push > 256B");

static const char * k_glsl_full_shortconv = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t past_va;     // history view [n_hist, n_ch, n_s]
    uint64_t cur_va;      // input [n_ch, n_t, n_s] contiguous
    uint64_t w_va;        // static conv weight [n_conv, n_ch]
    uint64_t dyn_va;      // optional dynamic taps [n_conv, n_heads, n_t * n_s]
    uint64_t cell_va;     // optional state store cell [n_hist * n_ch, n_s]
    uint64_t post_add_va; // optional trailing residual [n_ch, n_t * n_s]
    uint64_t dst_va;      // output [n_ch, n_t * n_s]
    uint     past_nb1;    // floats
    uint     past_nb2;    // floats
    uint     w_nb1;       // floats
    uint     cell_nb1;    // floats
    uint     n_conv;
    uint     n_ch;
    uint     n_t;
    uint     n_s;
    uint     n_heads;
    uint     head_dim;
    float    res_scale;
    float    res_bias;
} pc;

void main() {
    const uint ch  = gl_GlobalInvocationID.x;
    const uint t   = gl_WorkGroupID.y;
    const uint seq = gl_WorkGroupID.z;
    if (ch >= pc.n_ch || t >= pc.n_t || seq >= pc.n_s) return;

    const uint tok    = seq * pc.n_t + t;
    const uint n_conv = pc.n_conv;
    const uint n_hist = n_conv - 1u;
    const uint p_base = seq * pc.past_nb2 + ch * pc.past_nb1;
    const uint w_base = ch * pc.w_nb1;
    const uint head   = (pc.dyn_va != 0ul) ? (ch / pc.head_dim) : 0u;
    const uint d_base = (tok * pc.n_heads + head) * n_conv;

    float x_cur = ConstBufF32(pc.cur_va).v[tok * pc.n_ch + ch];
    float sum   = 0.0;
    for (uint k = 0u; k < n_conv; ++k) {
        uint lag = n_conv - 1u - k;
        float x_k = (t >= lag)
            ? ConstBufF32(pc.cur_va).v[(tok - lag) * pc.n_ch + ch]
            : ConstBufF32(pc.past_va).v[p_base + (n_hist - (lag - t))];
        sum += x_k * ConstBufF32(pc.w_va).v[w_base + k];
    }
    if (pc.dyn_va != 0ul) {
        for (uint lag = 0u; lag < n_conv; ++lag) {
            float x_lag = (t >= lag)
                ? ConstBufF32(pc.cur_va).v[(tok - lag) * pc.n_ch + ch]
                : ConstBufF32(pc.past_va).v[p_base + (n_hist - (lag - t))];
            sum += x_lag * ConstBufF32(pc.dyn_va).v[d_base + lag];
        }
    }

    // Update recurrent conv state (last n_hist raw inputs, oldest -> newest)
    if (pc.cell_va != 0ul && t == pc.n_t - 1u) {
        float new_h[8];
        uint lim = min(n_hist, 8u);
        for (uint s = 0u; s < lim; ++s) {
            uint lag = (n_hist - 1u) - s;
            if (t >= lag) {
                new_h[s] = ConstBufF32(pc.cur_va).v[(tok - lag) * pc.n_ch + ch];
            } else {
                new_h[s] = ConstBufF32(pc.past_va).v[p_base + (n_hist - (lag - t))];
            }
        }
        uint c_base = seq * pc.cell_nb1 + ch * n_hist;
        for (uint s = 0u; s < lim; ++s) {
            BufF32(pc.cell_va).v[c_base + s] = new_h[s];
        }
    }

    float out_v = (x_cur + sum) * pc.res_scale + pc.res_bias;
    if (pc.post_add_va != 0ul) {
        out_v += ConstBufF32(pc.post_add_va).v[tok * pc.n_ch + ch];
    }
    BufF32(pc.dst_va).v[tok * pc.n_ch + ch] = out_v;
}
)GLSL";

struct full_shortconv_push {
    uint64_t past_va, cur_va, w_va, dyn_va, cell_va, post_add_va, dst_va;
    uint32_t past_nb1, past_nb2, w_nb1, cell_nb1;
    uint32_t n_conv, n_ch, n_t, n_s, n_heads, head_dim;
    float    res_scale, res_bias;
};
static_assert(sizeof(full_shortconv_push) <= AGX_MAX_PUSH_BYTES, "full_shortconv_push > 256B");

static const char * k_glsl_router_topk_norm = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

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

shared float s_prob[512];
shared float s_val[512];
shared int   s_idx[512];

void main() {
    const uint tok = gl_WorkGroupID.x;
    const uint tid = gl_LocalInvocationID.x;
    if (tok >= pc.n_tokens || pc.n_exp == 0u || pc.k_out == 0u) return;

    const float neg_inf = uintBitsToFloat(0xFF800000u);
    uint n_pad = 1u;
    while (n_pad < pc.n_exp) n_pad <<= 1u;

    const uint l_base = tok * pc.n_exp;
    if (tid < n_pad) {
        if (tid < pc.n_exp) {
            float p = 1.0 / (1.0 + exp(-ConstBufF32(pc.logits_va).v[l_base + tid]));
            float s = (pc.bias_va != 0ul) ? (p + ConstBufF32(pc.bias_va).v[tid]) : p;
            s_prob[tid] = p;
            s_val[tid]  = s;
            s_idx[tid]  = int(tid);
        } else {
            s_prob[tid] = 0.0;
            s_val[tid]  = neg_inf;
            s_idx[tid]  = 0x7FFFFFFF;
        }
    }
    if (tid + 256u < n_pad) {
        uint e2 = tid + 256u;
        if (e2 < pc.n_exp) {
            float p = 1.0 / (1.0 + exp(-ConstBufF32(pc.logits_va).v[l_base + e2]));
            float s = (pc.bias_va != 0ul) ? (p + ConstBufF32(pc.bias_va).v[e2]) : p;
            s_prob[e2] = p;
            s_val[e2]  = s;
            s_idx[e2]  = int(e2);
        } else {
            s_prob[e2] = 0.0;
            s_val[e2]  = neg_inf;
            s_idx[e2]  = 0x7FFFFFFF;
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
                float va = s_val[lo]; int ia = s_idx[lo];
                float vb = s_val[hi]; int ib = s_idx[hi];
                bool a_before_b = (va > vb || (va == vb && ia < ib));
                if (asc_dir ? (!a_before_b) : a_before_b) {
                    s_val[lo] = vb; s_idx[lo] = ib;
                    s_val[hi] = va; s_idx[hi] = ia;
                }
            }
            barrier();
        }
    }

    if (tid < 32u) {
        int   eid = (tid < pc.k_out) ? s_idx[tid] : 0;
        float p   = (tid < pc.k_out) ? s_prob[uint(eid)] : 0.0;
        float sp  = subgroupAdd(p);
        float inv = pc.w_scale / (sp + pc.norm_eps);
        if (tid < pc.k_out) {
            BufI32(pc.ids_dst_va).v[tok * pc.ids_nb1 + tid] = eid;
            BufF32(pc.w_dst_va).v[tok * pc.k_out + tid]     = p * inv;
        }
    }
}
)GLSL";

struct router_topk_norm_push {
    uint64_t logits_va, bias_va, ids_dst_va, w_dst_va;
    uint32_t n_exp, k_out, n_tokens, ids_nb1;
    float    norm_eps, w_scale;
};
static_assert(sizeof(router_topk_norm_push) <= AGX_MAX_PUSH_BYTES, "router_topk_norm_push > 256B");

static const char * k_glsl_rel_bias_gather = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer ConstBufU16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) readonly buffer ConstBufI8 { int8_t v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t rcur_va;
    uint64_t bias_va;
    uint64_t sinks_va;
    uint64_t rel_pos_va;
    uint64_t dst_va;
    uint     n_rel;
    uint     n_extent;
    uint     n_kv;
    uint     n_head;
    uint     n_t;
    uint     n_stream;
    uint     bias_wtype; // 0=F32, 1=F16, 2=BF16, 3=Q8_0_RAW
    uint     bias_nb1;   // bytes
    uint     rpos_nb0;   // bytes
    uint     rpos_nb1;   // bytes
} pc;

shared float s_r[64];

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint kv  = gl_GlobalInvocationID.x;
    const uint h   = gl_WorkGroupID.y;
    const uint tok = gl_WorkGroupID.z;
    const uint n_tok = pc.n_t * pc.n_stream;
    if (h >= pc.n_head || tok >= n_tok) return;

    if (tid < pc.n_rel) {
        s_r[tid] = ConstBufF32(pc.rcur_va).v[(tok * pc.n_head + h) * pc.n_rel + tid];
    }
    barrier();

    if (kv >= pc.n_kv) return;

    int dist = ConstBufI32(pc.rel_pos_va + uint64_t(tok) * pc.rpos_nb1 + uint64_t(kv) * pc.rpos_nb0).v[0];
    float sink = ConstBufF32(pc.sinks_va).v[tok * pc.n_head + h];
    float acc  = 0.0;

    if (dist >= 0 && uint(dist) < pc.n_extent) {
        uint64_t row_va = pc.bias_va + uint64_t(dist) * uint64_t(pc.bias_nb1);
        if (pc.bias_wtype == 0u) {
            for (uint k = 0u; k < pc.n_rel; ++k) {
                acc += ConstBufF32(row_va).v[k] * s_r[k];
            }
        } else if (pc.bias_wtype == 1u) {
            for (uint k = 0u; k < pc.n_rel; ++k) {
                acc += unpackHalf2x16(uint(ConstBufU16(row_va).v[k])).x * s_r[k];
            }
        } else if (pc.bias_wtype == 2u) {
            for (uint k = 0u; k < pc.n_rel; ++k) {
                acc += uintBitsToFloat(uint(ConstBufU16(row_va).v[k]) << 16u) * s_r[k];
            }
        } else {
            // Q8_0_RAW: 34B per 32 elements
            uint n_blks = pc.n_rel >> 5u;
            for (uint b = 0u; b < n_blks; ++b) {
                uint64_t bva = row_va + uint64_t(b) * 34ul;
                float d = unpackHalf2x16(uint(ConstBufU16(bva).v[0])).x;
                float p = 0.0;
                uint bk = b << 5u;
                for (uint j = 0u; j < 32u; ++j) {
                    p += float(ConstBufI8(bva + 2ul).v[j]) * s_r[bk + j];
                }
                acc += d * p;
            }
        }
    }

    const uint m = tok % pc.n_t;
    const uint s = tok / pc.n_t;
    const uint out_idx = ((s * pc.n_head + h) * pc.n_t + m) * pc.n_kv + kv;
    BufF32(pc.dst_va).v[out_idx] = acc - sink;
}
)GLSL";

struct rel_bias_gather_push {
    uint64_t rcur_va, bias_va, sinks_va, rel_pos_va, dst_va;
    uint32_t n_rel, n_extent, n_kv, n_head, n_t, n_stream, bias_wtype, bias_nb1, rpos_nb0, rpos_nb1;
};
static_assert(sizeof(rel_bias_gather_push) <= AGX_MAX_PUSH_BYTES, "rel_bias_gather_push > 256B");

// ============================================================================
// Graph Executor State & Lazy Kernel Compilation
// ============================================================================

struct agx_va_range {
    uint64_t beg;
    uint64_t end;
};

static inline bool va_range_overlap(agx_va_range a, agx_va_range b) {
    return a.beg < a.end && b.beg < b.end && a.beg < b.end && b.beg < a.end;
}

struct agx_ptr_count_map {
    std::vector<const struct ggml_tensor *> keys;
    std::vector<uint32_t>                   vals;
    size_t                                  mask = 0;

    void reset(size_t expected_entries) {
        size_t cap = 64u;
        while (cap < expected_entries * 2u) cap <<= 1u;
        if (keys.size() != cap) {
            keys.assign(cap, nullptr);
            vals.assign(cap, 0u);
        } else {
            std::fill(keys.begin(), keys.end(), nullptr);
            std::fill(vals.begin(), vals.end(), 0u);
        }
        mask = cap - 1u;
    }

    static inline size_t hash_ptr(const struct ggml_tensor * p) {
        uintptr_t x = reinterpret_cast<uintptr_t>(p);
        x ^= x >> 16;
        x *= 0x45d9f3bu;
        x ^= x >> 16;
        return (size_t) x;
    }

    void inc(const struct ggml_tensor * p) {
        if (!p) return;
        size_t idx = hash_ptr(p) & mask;
        while (keys[idx] != nullptr && keys[idx] != p) {
            idx = (idx + 1u) & mask;
        }
        keys[idx] = p;
        vals[idx]++;
    }

    uint32_t get(const struct ggml_tensor * p) const {
        if (!p || keys.empty()) return 0u;
        size_t idx = hash_ptr(p) & mask;
        while (keys[idx] != nullptr) {
            if (keys[idx] == p) return vals[idx];
            idx = (idx + 1u) & mask;
        }
        return 0u;
    }
};

struct agx_ptr_node_map {
    std::vector<const struct ggml_tensor *> keys;
    std::vector<struct ggml_tensor *>       vals;
    size_t                                  mask = 0;

    void reset(size_t expected_entries) {
        size_t cap = 64u;
        while (cap < expected_entries * 2u) cap <<= 1u;
        if (keys.size() != cap) {
            keys.assign(cap, nullptr);
            vals.assign(cap, nullptr);
        } else {
            std::fill(keys.begin(), keys.end(), nullptr);
            std::fill(vals.begin(), vals.end(), nullptr);
        }
        mask = cap - 1u;
    }

    void set_if_empty(const struct ggml_tensor * k, struct ggml_tensor * v) {
        if (!k || !v) return;
        size_t idx = agx_ptr_count_map::hash_ptr(k) & mask;
        while (keys[idx] != nullptr && keys[idx] != k) {
            idx = (idx + 1u) & mask;
        }
        if (keys[idx] == nullptr) {
            keys[idx] = k;
            vals[idx] = v;
        }
    }

    struct ggml_tensor * get(const struct ggml_tensor * k) const {
        if (!k || keys.empty()) return nullptr;
        size_t idx = agx_ptr_count_map::hash_ptr(k) & mask;
        while (keys[idx] != nullptr) {
            if (keys[idx] == k) return vals[idx];
            idx = (idx + 1u) & mask;
        }
        return nullptr;
    }
};

struct agx_ptr_idx_map {
    std::vector<const struct ggml_tensor *> keys;
    std::vector<int32_t>                    vals;
    size_t                                  mask = 0;

    void reset(size_t expected_entries) {
        size_t cap = 64u;
        while (cap < expected_entries * 2u) cap <<= 1u;
        if (keys.size() != cap) {
            keys.assign(cap, nullptr);
            vals.assign(cap, -1);
        } else {
            std::fill(keys.begin(), keys.end(), nullptr);
            std::fill(vals.begin(), vals.end(), -1);
        }
        mask = cap - 1u;
    }

    void set(const struct ggml_tensor * k, int32_t v) {
        if (!k) return;
        size_t idx = agx_ptr_count_map::hash_ptr(k) & mask;
        while (keys[idx] != nullptr && keys[idx] != k) {
            idx = (idx + 1u) & mask;
        }
        keys[idx] = k;
        vals[idx] = v;
    }

    int32_t get(const struct ggml_tensor * k) const {
        if (!k || keys.empty()) return -1;
        size_t idx = agx_ptr_count_map::hash_ptr(k) & mask;
        while (keys[idx] != nullptr) {
            if (keys[idx] == k) return vals[idx];
            idx = (idx + 1u) & mask;
        }
        return -1;
    }
};

enum agx_exec_kind : uint8_t {
    ITEM_GENERIC_NODE = 0,
    ITEM_PREP_STORE,
    ITEM_GLU,
    ITEM_SOFTCAP,
    ITEM_CTRL_SRAM,
    ITEM_CTRL_NORM_CAT,
    ITEM_CTRL_MIX,
    ITEM_ROW_REDUCE,
    ITEM_BIN_OP,
    ITEM_MIX_ATTN,
    ITEM_MOE_WSUM,
    ITEM_SSM_CONV,
    ITEM_FULL_SHORTCONV,
    ITEM_ROUTER_TOPK,
    ITEM_REL_BIAS,
};

struct agx_exec_item {
    agx_exec_kind        kind              = ITEM_GENERIC_NODE;
    struct ggml_tensor * node              = nullptr;
    uint32_t             gx = 1, gy = 1, gz = 1;
    uint32_t             nodes_covered     = 1;
    agx_va_range         reads[8]          = {};
    uint8_t              n_reads           = 0;
    agx_va_range         writes[4]         = {};
    uint8_t              n_writes          = 0;
    bool                 uses_moe_scratch  = false;
    bool                 uses_attn_scratch = false;
    alignas(8) uint8_t   push[160]         = {};
    uint32_t             push_bytes        = 0;
    agx_ctrl_gate_desc   ctrl_desc         = {};
    uint64_t             va0 = 0, va1 = 0;
    uint32_t             u0 = 0, u1 = 0, u2 = 0;
    float                f0 = 0.0f, f1 = 0.0f;

    void add_read(agx_va_range r) {
        if (r.beg < r.end && n_reads < 8u) reads[n_reads++] = r;
    }
    void add_write(agx_va_range w) {
        if (w.beg < w.end && n_writes < 4u) writes[n_writes++] = w;
    }
    template <typename T>
    void set_push(const T & p) {
        static_assert(sizeof(T) <= sizeof(push), "push payload exceeds 160B");
        std::memcpy(push, &p, sizeof(T));
        push_bytes = (uint32_t) sizeof(T);
    }
};

struct agx_graph_exec_state {
    const agx_kernel * k_bin_op              = nullptr;
    const agx_kernel * k_unary_op            = nullptr;
    const agx_kernel * k_cpy_cast            = nullptr;
    const agx_kernel * k_row_reduce          = nullptr;
    const agx_kernel * k_softmax             = nullptr;
    const agx_kernel * k_get_rows            = nullptr;
    const agx_kernel * k_set_rows            = nullptr;
    const agx_kernel * k_concat_pad          = nullptr;
    const agx_kernel * k_ssm_conv            = nullptr;
    const agx_kernel * k_rope                = nullptr;
    const agx_kernel * k_topk_argsort        = nullptr;
    const agx_kernel * k_glu                 = nullptr;
    const agx_kernel * k_add_id              = nullptr;
    const agx_kernel * k_ctrl_mix            = nullptr;
    const agx_kernel * k_ctrl_norm_cat       = nullptr;
    const agx_kernel * k_mix_attn            = nullptr;
    const agx_kernel * k_moe_wsum            = nullptr;
    const agx_kernel * k_prep_norm_fwht_store = nullptr;
    const agx_kernel * k_full_shortconv      = nullptr;
    const agx_kernel * k_router_topk_norm    = nullptr;
    const agx_kernel * k_rel_bias_gather     = nullptr;

    agx_ptr_count_map           use_map;
    agx_ptr_node_map            first_use;
    agx_ptr_idx_map             exec_idx_map;
    std::vector<int>            exec_nodes;
    std::vector<agx_exec_item>  items;
    std::vector<agx_exec_item>  ordered_items;
    agx_inkling2_engine *       inkling2    = nullptr;
    agx_graph_exec_stats        stats       = {};
    bool                        initialized = false;
};

static agx_graph_exec_state * get_exec_state(agx_backend_context * ctx) {
    if (!ctx) return nullptr;
    if (!ctx->graph_exec_state) {
        ctx->graph_exec_state = new agx_graph_exec_state();
    }
    auto * st = static_cast<agx_graph_exec_state *>(ctx->graph_exec_state);
    if (!st->initialized && ctx->compiler) {
        agx_compile_opts opts;
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        st->k_bin_op               = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_bin_op",               k_glsl_bin_op,               opts);
        st->k_unary_op             = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_unary_op",             k_glsl_unary_op,             opts);
        st->k_cpy_cast             = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_cpy_cast",             k_glsl_cpy_cast,             opts);
        st->k_row_reduce           = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_row_reduce",           k_glsl_row_reduce,           opts);
        st->k_softmax              = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_softmax",              k_glsl_softmax,              opts);
        st->k_get_rows             = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_get_rows",             k_glsl_get_rows,             opts);
        st->k_set_rows             = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_set_rows",             k_glsl_set_rows,             opts);
        st->k_concat_pad           = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_concat_pad",           k_glsl_concat_pad,           opts);
        st->k_ssm_conv             = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_ssm_conv",             k_glsl_ssm_conv,             opts);
        st->k_rope                 = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_rope",                 k_glsl_rope,                 opts);
        st->k_topk_argsort         = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_topk_argsort",         k_glsl_topk_argsort,         opts);
        st->k_glu                  = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_glu",                  k_glsl_glu,                  opts);
        st->k_add_id               = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_add_id",               k_glsl_add_id,               opts);
        st->k_ctrl_mix             = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_ctrl_mix",             k_glsl_ctrl_mix,             opts);
        st->k_ctrl_norm_cat        = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_ctrl_norm_cat",        k_glsl_ctrl_norm_cat,        opts);
        st->k_mix_attn             = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_mix_attn",             k_glsl_mix_attn,             opts);
        st->k_moe_wsum             = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_moe_wsum",             k_glsl_moe_wsum,             opts);
        st->k_prep_norm_fwht_store = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_prep_norm_fwht_store", k_glsl_prep_norm_fwht_store, opts);
        st->k_full_shortconv       = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_full_shortconv",       k_glsl_full_shortconv,       opts);
        st->k_router_topk_norm     = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_router_topk_norm",     k_glsl_router_topk_norm,     opts);
        st->k_rel_bias_gather      = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_rel_bias_gather",      k_glsl_rel_bias_gather,      opts);
        st->initialized            = (st->k_bin_op && st->k_unary_op && st->k_cpy_cast && st->k_row_reduce &&
                                      st->k_softmax && st->k_get_rows && st->k_set_rows && st->k_concat_pad &&
                                      st->k_ssm_conv && st->k_rope && st->k_topk_argsort && st->k_glu &&
                                      st->k_add_id && st->k_ctrl_mix && st->k_ctrl_norm_cat &&
                                      st->k_mix_attn && st->k_moe_wsum && st->k_prep_norm_fwht_store &&
                                      st->k_full_shortconv && st->k_router_topk_norm && st->k_rel_bias_gather);
    }
    return st;
}

agx_inkling2_engine * agx_graph_exec_get_inkling2(agx_backend_context * ctx) {
    auto * st = get_exec_state(ctx);
    if (!st) return nullptr;
    if (!st->inkling2) {
        agx_inkling2_hparams hp;
        agx_inkling2_fill_default_hparams(&hp);
        st->inkling2 = agx_inkling2_engine_create(ctx, hp, 8u);
    }
    return st->inkling2;
}

void agx_graph_exec_get_stats(const agx_backend_context * ctx, agx_graph_exec_stats * out) {
    if (!out) return;
    if (!ctx || !ctx->graph_exec_state) {
        std::memset(out, 0, sizeof(*out));
        return;
    }
    *out = static_cast<const agx_graph_exec_state *>(ctx->graph_exec_state)->stats;
}

void agx_graph_exec_reset_stats(agx_backend_context * ctx) {
    if (ctx && ctx->graph_exec_state) {
        static_cast<agx_graph_exec_state *>(ctx->graph_exec_state)->stats = {};
    }
}

// ============================================================================
// Helper Utilities for TLA+-Verified Hazard Tracking & Type Mapping
// ============================================================================

static bool is_view_only_op(enum ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool cpy_type_code(enum ggml_type t, uint32_t * out) {
    switch (t) {
        case GGML_TYPE_F32:  *out = 0u; return true;
        case GGML_TYPE_F16:  *out = 1u; return true;
        case GGML_TYPE_BF16: *out = 2u; return true;
        case GGML_TYPE_I32:  *out = 3u; return true;
        default: return false;
    }
}

static size_t tensor_byte_span(const struct ggml_tensor * x) {
    size_t span = ggml_nbytes(x);
    if (x->ne[0] > 0 && x->ne[1] > 0 && x->ne[2] > 0 && x->ne[3] > 0) {
        const size_t blk = ggml_blck_size(x->type);
        const size_t n0_blks = ((size_t) x->ne[0] + blk - 1u) / blk;
        const size_t last =
            (n0_blks > 0 ? (n0_blks - 1u) : 0u) * x->nb[0] +
            (size_t) (x->ne[1] - 1) * x->nb[1] +
            (size_t) (x->ne[2] - 1) * x->nb[2] +
            (size_t) (x->ne[3] - 1) * x->nb[3] +
            ggml_type_size(x->type);
        if (last > span) span = last;
    }
    return span ? span : 4u;
}

static agx_va_range tensor_va_range(const struct ggml_tensor * t) {
    if (!t || ggml_nelements(t) == 0) return { 0, 0 };
    uint64_t va = agx_tensor_gpu_va(t);
    if (!va) return { 0, 0 };
    return { va, va + tensor_byte_span(t) };
}

static bool ranges_overlap(const std::vector<agx_va_range> & dirty, agx_va_range q) {
    if (q.beg == q.end) return false;
    for (const auto & r : dirty) {
        if (q.beg < r.end && r.beg < q.end) return true;
    }
    return false;
}

static uint32_t map_unary_op_code(enum ggml_unary_op u) {
    switch (u) {
        case GGML_UNARY_OP_ABS:         return 10u;
        case GGML_UNARY_OP_SGN:         return 11u;
        case GGML_UNARY_OP_NEG:         return 12u;
        case GGML_UNARY_OP_STEP:        return 13u;
        case GGML_UNARY_OP_TANH:        return 14u;
        case GGML_UNARY_OP_ELU:         return 15u;
        case GGML_UNARY_OP_RELU:        return 16u;
        case GGML_UNARY_OP_SIGMOID:     return 17u;
        case GGML_UNARY_OP_GELU:        return 18u;
        case GGML_UNARY_OP_GELU_QUICK:  return 19u;
        case GGML_UNARY_OP_SILU:        return 20u;
        case GGML_UNARY_OP_EXP:         return 21u;
        case GGML_UNARY_OP_EXPM1:       return 22u;
        case GGML_UNARY_OP_SOFTPLUS:    return 23u;
        case GGML_UNARY_OP_GELU_ERF:    return 24u;
        case GGML_UNARY_OP_FLOOR:       return 25u;
        case GGML_UNARY_OP_CEIL:        return 26u;
        case GGML_UNARY_OP_ROUND:       return 27u;
        case GGML_UNARY_OP_TRUNC:       return 28u;
        case GGML_UNARY_OP_HARDSWISH:   return 29u;
        case GGML_UNARY_OP_HARDSIGMOID: return 30u;
        default:                        return 0xFFFFFFFFu;
    }
}

// ============================================================================
// Cross-Module Entrypoints (`agx-common.h` extern "C")
// ============================================================================

extern "C" {

bool agx_supports_op(const agx_drm_device * dev, const struct ggml_tensor * op) {
    (void) dev;
    if (!op) return false;

    const struct ggml_tensor * s0 = op->src[0];
    const struct ggml_tensor * s1 = op->src[1];
    uint32_t tc0 = 0, tc1 = 0;

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return s0 && s1 && s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;

        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_CUMSUM:
            return s0 && s0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   s0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float);

        case GGML_OP_FILL:
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32;

        case GGML_OP_ARGMAX:
            return s0 && s0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 &&
                   s0->nb[0] == sizeof(float) && op->nb[0] == sizeof(int32_t);

        case GGML_OP_UNARY:
            return s0 && s0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   map_unary_op_code(ggml_get_unary_op(op)) != 0xFFFFFFFFu;

        case GGML_OP_GLU: {
            const enum ggml_glu_op gop = ggml_get_glu_op(op);
            return s0 && s0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   (!s1 || (s1->type == GGML_TYPE_F32 && ggml_is_contiguous(s1))) &&
                   ggml_is_contiguous(s0) && ggml_is_contiguous(op) &&
                   (gop == GGML_GLU_OP_REGLU || gop == GGML_GLU_OP_GEGLU ||
                    gop == GGML_GLU_OP_SWIGLU || gop == GGML_GLU_OP_SWIGLU_OAI ||
                    gop == GGML_GLU_OP_SWIGLU_CLAMP);
        }

        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            return s0 && cpy_type_code(s0->type, &tc0) && cpy_type_code(op->type, &tc1);

        case GGML_OP_REPEAT:
            return s0 && (s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_I32) &&
                   op->type == s0->type && ggml_is_contiguous(op);

        case GGML_OP_PAD:
            return s0 && (s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_I32) &&
                   op->type == s0->type && ggml_is_contiguous(op) &&
                   ggml_get_op_params_i32(op, 8) == 0; // non-circular pad

        case GGML_OP_CONCAT:
            return s0 && s1 && (s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_I32) &&
                   s1->type == s0->type && op->type == s0->type && ggml_is_contiguous(op);

        case GGML_OP_SOFT_MAX:
            return s0 && s0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(s0) && ggml_is_contiguous(op) &&
                   (!s1 || ((s1->type == GGML_TYPE_F16 || s1->type == GGML_TYPE_F32) && ggml_is_contiguous(s1))) &&
                   (!op->src[2] || (op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2])));

        case GGML_OP_GET_ROWS:
            return s0 && s1 && s1->type == GGML_TYPE_I32 &&
                   (op->type == GGML_TYPE_F32 || (op->type == GGML_TYPE_I32 && s0->type == GGML_TYPE_I32)) &&
                   (s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_F16 ||
                    s0->type == GGML_TYPE_BF16 || s0->type == GGML_TYPE_Q8_0 || s0->type == GGML_TYPE_I32) &&
                   s0->nb[0] == ggml_type_size(s0->type) && op->nb[0] == ggml_type_size(op->type);

        case GGML_OP_SET_ROWS:
            return s0 && s1 && s0->type == GGML_TYPE_F32 &&
                   (s1->type == GGML_TYPE_I64 || s1->type == GGML_TYPE_I32) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 ||
                    (op->type == GGML_TYPE_Q8_0 && (s0->ne[0] % 32) == 0)) &&
                   s0->nb[0] == sizeof(float) && op->nb[0] == ggml_type_size(op->type);

        case GGML_OP_ADD_ID:
            return s0 && s1 && op->src[2] && s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_F32 &&
                   op->src[2]->type == GGML_TYPE_I32 && op->type == GGML_TYPE_F32 &&
                   s0->nb[0] == sizeof(float) && s1->nb[0] == sizeof(float) &&
                   op->src[2]->nb[0] == sizeof(int32_t) && op->nb[0] == sizeof(float) && op->ne[3] == 1;

        case GGML_OP_SSM_CONV:
            return s0 && s1 && s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 && ggml_is_contiguous(s0) && ggml_is_contiguous(s1) &&
                   ggml_is_contiguous(op) && s0->ne[3] == 1 && op->ne[3] == 1;

        case GGML_OP_ROPE: {
            float ext_factor = 0.0f;
            std::memcpy(&ext_factor, (const char *) op->op_params + 28, sizeof(float));
            const int n_dims = ggml_get_op_params_i32(op, 1);
            return s0 && s1 && s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32 && s0->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
                   ggml_is_contiguous(s1) &&
                   (!op->src[2] || (op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]))) &&
                   (ggml_get_op_params_i32(op, 2) & ~2) == 0 && ext_factor == 0.0f &&
                   n_dims > 0 && n_dims <= s0->ne[0] && (n_dims & 1) == 0;
        }

        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return s0 && s0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_I32 &&
                   ggml_is_contiguous(s0) && ggml_is_contiguous(op) &&
                   s0->ne[0] <= 262144 && op->ne[0] <= 256;

        case GGML_OP_MUL_MAT:
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                s1 && s1->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(s1) && (s1->ne[0] == 64 || s1->ne[0] == 128)) {
                return true;
            }
            return agx_gemm_supports_mul_mat(op);

        case GGML_OP_MUL_MAT_ID:
            return agx_moe_supports_mul_mat_id(op);

        case GGML_OP_FLASH_ATTN_EXT:
            return agx_kv_hybrid_supports_ggml_flash_attn(op);

        default:
            return false;
    }
}

void agx_graph_optimize(agx_backend_context *                       ctx,
                        struct ggml_cgraph *                        cgraph,
                        struct ggml_backend_graph_optimize_params * params) {
    (void) ctx;
    (void) cgraph;
    (void) params;
}

void agx_graph_exec_cleanup(agx_backend_context * ctx) {
    if (!ctx) return;
    if (ctx->graph_exec_state) {
        auto * st = static_cast<agx_graph_exec_state *>(ctx->graph_exec_state);
        if (st->inkling2) {
            agx_inkling2_engine_destroy(st->inkling2);
            st->inkling2 = nullptr;
        }
        delete st;
        ctx->graph_exec_state = nullptr;
    }
    if (ctx->kv_hybrid_state) {
        agx_kv_hybrid_destroy(static_cast<agx_kv_hybrid_pool *>(ctx->kv_hybrid_state));
        ctx->kv_hybrid_state = nullptr;
    }
    if (ctx->moe_state) {
        agx_moe_destroy(static_cast<agx_moe_runner *>(ctx->moe_state));
        ctx->moe_state = nullptr;
    }
    if (ctx->gemm_state) {
        agx_gemm_cleanup(ctx);
    }
}

// ============================================================================
// Single-CDM-Stream Generic + DAG-Fused + Wave-Scheduled ggml_cgraph Execution
// ============================================================================

struct agx_interval_lvl {
    uint64_t beg;
    uint64_t end;
    uint32_t lvl;
};

static void insert_interval_lvl(std::vector<agx_interval_lvl> & vec, agx_va_range q, uint32_t lv) {
    if (q.beg >= q.end) return;
    size_t i = 0;
    while (i < vec.size()) {
        auto & e = vec[i];
        if (e.beg == q.beg && e.end == q.end) {
            if (lv > e.lvl) e.lvl = lv;
            return;
        }
        if (e.beg <= q.beg && q.end <= e.end && e.lvl >= lv) {
            return; // already dominated
        }
        if (q.beg <= e.beg && e.end <= q.end && e.lvl <= lv) {
            vec[i] = vec.back();
            vec.pop_back();
            continue;
        }
        ++i;
    }
    vec.push_back({ q.beg, q.end, lv });
}

static uint32_t query_interval_max_lvl(const std::vector<agx_interval_lvl> & vec, agx_va_range q, bool * found) {
    uint32_t mx = 0u;
    bool hit = false;
    if (q.beg >= q.end) {
        *found = false;
        return 0u;
    }
    for (const auto & e : vec) {
        if (q.beg < e.end && e.beg < q.end) {
            if (!hit || e.lvl > mx) mx = e.lvl;
            hit = true;
        }
    }
    *found = hit;
    return mx;
}

static void prune_contained_intervals(std::vector<agx_interval_lvl> & vec, agx_va_range w, uint32_t lv) {
    size_t i = 0;
    while (i < vec.size()) {
        const auto & e = vec[i];
        if (w.beg <= e.beg && e.end <= w.end && e.lvl <= lv) {
            vec[i] = vec.back();
            vec.pop_back();
            continue;
        }
        ++i;
    }
}

enum ggml_status agx_graph_compute(agx_backend_context * ctx, struct ggml_cgraph * cgraph) {
    if (!ctx || !ctx->drm || !ctx->stream || !cgraph) {
        return GGML_STATUS_FAILED;
    }
    if (cgraph->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    auto * st = get_exec_state(ctx);
    if (!st || !st->initialized) {
        AGX_LOG_ERROR("agx_graph_compute: failed to initialize generic compute kernels");
        return GGML_STATUS_FAILED;
    }

    const auto t_enc_beg = std::chrono::steady_clock::now();

    agx_cdm_stream * stream = ctx->stream;
    agx_cdm_stream_reset(stream);

    agx_cdm_stats cdm_before{};
    agx_cdm_get_stats(stream, &cdm_before);

    const bool force_barrier = (std::getenv("GGML_AGX_FORCE_BARRIER") != nullptr &&
                                std::getenv("GGML_AGX_FORCE_BARRIER")[0] == '1');
    const bool no_fusion     = (std::getenv("GGML_AGX_NO_FUSION") != nullptr &&
                                std::getenv("GGML_AGX_NO_FUSION")[0] == '1');
    const bool no_wave_sched = (std::getenv("GGML_AGX_NO_WAVE_SCHED") != nullptr &&
                                std::getenv("GGML_AGX_NO_WAVE_SCHED")[0] == '1');
    const bool print_stats   = (std::getenv("GGML_AGX_STATS") != nullptr &&
                                std::getenv("GGML_AGX_STATS")[0] == '1');
    uint32_t fusion_mask     = no_fusion ? 0u : 0xFFFFFFFFu;
    if (!no_fusion) {
        if (const char * fm = std::getenv("GGML_AGX_FUSION_MASK")) {
            fusion_mask = (uint32_t) std::strtoul(fm, nullptr, 0);
        }
    }

    auto va_disjoint = [](const struct ggml_tensor * a, const struct ggml_tensor * b) -> bool {
        if (!a || !b) return true;
        return !va_range_overlap(tensor_va_range(a), tensor_va_range(b));
    };
    auto va_same_or_disjoint = [&](const struct ggml_tensor * a, const struct ggml_tensor * b) -> bool {
        if (!a || !b) return true;
        return agx_tensor_gpu_va(a) == agx_tensor_gpu_va(b) || va_disjoint(a, b);
    };

    // ------------------------------------------------------------------------
    // Phase 0: Precompute O(1) refcounts, first-consumer map, exec_nodes, & VA table
    // ------------------------------------------------------------------------
    st->use_map.reset((size_t) cgraph->n_nodes + 16u);
    st->first_use.reset((size_t) cgraph->n_nodes + 16u);
    st->exec_nodes.clear();
    st->exec_nodes.reserve((size_t) cgraph->n_nodes);

    for (int k = 0; k < cgraph->n_nodes; ++k) {
        struct ggml_tensor * n = cgraph->nodes[k];
        if (!n) continue;
        bool view_counted = false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s]) {
                st->use_map.inc(n->src[s]);
                st->first_use.set_if_empty(n->src[s], n);
                if (n->src[s] == n->view_src) view_counted = true;
            }
        }
        if (n->view_src && !view_counted && !is_view_only_op(n->op)) {
            st->use_map.inc(n->view_src);
            st->first_use.set_if_empty(n->view_src, n);
        }
        if (is_view_only_op(n->op)) {
            st->stats.view_nodes_elided++;
        } else if (ggml_nelements(n) > 0) {
            st->exec_nodes.push_back(k);
        }
    }
    if (cgraph->n_nodes > 0 && cgraph->nodes[cgraph->n_nodes - 1]) {
        st->use_map.inc(cgraph->nodes[cgraph->n_nodes - 1]); // pin graph output
    }

    const size_t n_exec = st->exec_nodes.size();
    st->exec_idx_map.reset(n_exec + 16u);

    // Precompute exact VA ranges & validate non-null GPU VAs up front
    struct raw_node_vas {
        agx_va_range w;
        agx_va_range r[GGML_MAX_SRC];
        uint8_t      nr;
    };
    std::vector<raw_node_vas> raw_vas(n_exec);
    for (size_t e = 0; e < n_exec; ++e) {
        const int i = st->exec_nodes[e];
        struct ggml_tensor * node = cgraph->nodes[i];
        if (agx_tensor_gpu_va(node) == 0) {
            AGX_LOG_ERROR("agx_graph_compute: node %d (%s) dst has null GPU VA", i, ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
        st->exec_idx_map.set(node, (int32_t) e);
        raw_vas[e].w  = tensor_va_range(node);
        raw_vas[e].nr = 0u;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->op == GGML_OP_CPY && s == 1) continue; // src[1] is write-only dst alias
            if (node->src[s] && ggml_nelements(node->src[s]) > 0) {
                if (agx_tensor_gpu_va(node->src[s]) == 0) {
                    AGX_LOG_ERROR("agx_graph_compute: node %d (%s) src[%d] has null GPU VA",
                                  i, ggml_op_name(node->op), s);
                    return GGML_STATUS_FAILED;
                }
                const agx_va_range sr = tensor_va_range(node->src[s]);
                if (sr.beg < sr.end) {
                    raw_vas[e].r[raw_vas[e].nr++] = sr;
                }
            }
        }
    }

    // Helper lambdas for graph pattern matching & exact VA non-interference checks
    auto strip_reshape = [](const struct ggml_tensor * t) -> const struct ggml_tensor * {
        while (t && t->op == GGML_OP_RESHAPE && t->src[0]) {
            t = t->src[0];
        }
        return t;
    };

    auto strip_reshape_fullview = [](const struct ggml_tensor * t) -> const struct ggml_tensor * {
        while (t) {
            if (t->op == GGML_OP_RESHAPE && t->src[0]) {
                t = t->src[0];
            } else if (t->op == GGML_OP_VIEW && (t->src[0] || t->view_src) && t->view_offs == 0 &&
                       ggml_is_contiguous(t)) {
                const struct ggml_tensor * base = t->src[0] ? t->src[0] : t->view_src;
                if (ggml_nelements(t) == ggml_nelements(base)) {
                    t = base;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        return t;
    };

    auto chain_uses_exact = [&](const struct ggml_tensor * leaf,
                                const struct ggml_tensor * target,
                                uint32_t target_uses = 1u) -> bool {
        const struct ggml_tensor * p = leaf;
        while (p && p != target) {
            if (p->op != GGML_OP_RESHAPE || !p->src[0]) return false;
            if ((p->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return false;
            if (st->use_map.get(p) != 1u) return false;
            p = p->src[0];
        }
        if (p != target) return false;
        if ((target->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return false;
        return st->use_map.get(target) == target_uses;
    };

    auto next_single_use_compute = [&](const struct ggml_tensor * t, bool allow_full_view) -> struct ggml_tensor * {
        const struct ggml_tensor * cur = t;
        while (cur) {
            if ((cur->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return nullptr;
            if (st->use_map.get(cur) != 1u) return nullptr;
            struct ggml_tensor * u = st->first_use.get(cur);
            if (!u) return nullptr;
            if (u->op == GGML_OP_RESHAPE && u->src[0] == cur) {
                cur = u;
                continue;
            }
            if (allow_full_view && u->op == GGML_OP_VIEW &&
                (u->src[0] == cur || u->view_src == cur) &&
                u->view_offs == 0 && ggml_is_contiguous(u) &&
                ggml_nelements(u) == ggml_nelements(cur)) {
                cur = u;
                continue;
            }
            return is_view_only_op(u->op) ? nullptr : u;
        }
        return nullptr;
    };

    auto read_scale_params = [](const struct ggml_tensor * sc_node, float * s, float * b) {
        std::memcpy(s, sc_node->op_params, sizeof(float));
        std::memcpy(b, (const char *) sc_node->op_params + 4, sizeof(float));
    };

    auto is_ctrl_wtype_ok = [](const struct ggml_tensor * w) -> bool {
        if (!w || !ggml_is_contiguous(w) || agx_tensor_gpu_va(w) == 0) return false;
        return w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 ||
               w->type == GGML_TYPE_BF16 || w->type == GGML_TYPE_Q8_0;
    };

    // ------------------------------------------------------------------------
    // Phase A: Non-Adjacent DAG Fusion Pass -> st->items + anchor_slices
    // ------------------------------------------------------------------------
    struct anchor_slice {
        uint32_t beg = 0u;
        uint32_t cnt = 0u;
    };
    std::vector<uint8_t>      consumed(n_exec, 0u);
    std::vector<anchor_slice> anchors(n_exec);
    st->items.clear();
    st->items.reserve(n_exec);

    auto in_idx_set = [](size_t j, const int32_t * set_arr, int set_n) -> bool {
        for (int k = 0; k < set_n; ++k) {
            if ((size_t) set_arr[k] == j) return true;
        }
        return false;
    };

    // Exact VA non-interference check for placing fused op at e_min or e_max.
    // `all_ins_prod[r]` is the exec index that produces `all_ins[r]` (-1 if external/input/weight),
    // so placing at e_max allows `j == all_ins_prod[r]` (its legitimate producer in (e_min, e_max))
    // while still rejecting any clobbering write at j > all_ins_prod[r]!
    auto find_safe_anchor = [&](const int32_t * set_arr, int set_n,
                                const agx_va_range * all_ins, const int32_t * all_ins_prod, int n_all_ins,
                                const agx_va_range * extra_ins_for_first, int n_extra_first,
                                const agx_va_range * outs, int n_outs) -> int32_t {
        if (set_n <= 0) return -1;
        size_t e_min = (size_t) set_arr[0];
        size_t e_max = (size_t) set_arr[0];
        for (int k = 0; k < set_n; ++k) {
            if (set_arr[k] < 0 || (size_t) set_arr[k] >= n_exec) return -1;
            if (consumed[(size_t) set_arr[k]]) return -1;
            if ((size_t) set_arr[k] < e_min) e_min = (size_t) set_arr[k];
            if ((size_t) set_arr[k] > e_max) e_max = (size_t) set_arr[k];
        }
        if (e_min == e_max) return (int32_t) e_min;

        // Try placing at e_min first (optimal for K/V SET_ROWS since src is guaranteed live at e_min!)
        bool ok_first = true;
        for (size_t j = e_min + 1u; j <= e_max && ok_first; ++j) {
            if (in_idx_set(j, set_arr, set_n)) continue;
            for (int r = 0; r < n_extra_first; ++r) {
                if (va_range_overlap(raw_vas[j].w, extra_ins_for_first[r])) { ok_first = false; break; }
            }
            for (int w = 0; ok_first && w < n_outs; ++w) {
                if (va_range_overlap(raw_vas[j].w, outs[w])) { ok_first = false; break; }
                for (uint8_t jr = 0; jr < raw_vas[j].nr; ++jr) {
                    if (va_range_overlap(raw_vas[j].r[jr], outs[w])) { ok_first = false; break; }
                }
            }
        }
        if (ok_first) return (int32_t) e_min;

        // Fall back to placing at e_max (e.g. when an extra input like sinks/dyn is computed in (e_min, e_max))
        bool ok_last = true;
        for (int r = 0; r < n_all_ins; ++r) {
            if (all_ins_prod && all_ins_prod[r] >= (int32_t) e_max) { ok_last = false; break; }
        }
        for (size_t j = e_min; j < e_max && ok_last; ++j) {
            if (in_idx_set(j, set_arr, set_n)) continue;
            for (int r = 0; r < n_all_ins; ++r) {
                if (all_ins_prod && all_ins_prod[r] >= 0 && (int32_t) j <= all_ins_prod[r]) continue;
                if (va_range_overlap(raw_vas[j].w, all_ins[r])) { ok_last = false; break; }
            }
            for (int w = 0; ok_last && w < n_outs; ++w) {
                if (va_range_overlap(raw_vas[j].w, outs[w])) { ok_last = false; break; }
                for (uint8_t jr = 0; jr < raw_vas[j].nr; ++jr) {
                    if (va_range_overlap(raw_vas[j].r[jr], outs[w])) { ok_last = false; break; }
                }
            }
        }
        return ok_last ? (int32_t) e_max : -1;
    };

    auto tensor_prod_idx = [&](const struct ggml_tensor * t) -> int32_t {
        if (!t) return -1;
        const struct ggml_tensor * r = t;
        while (r->view_src) r = r->view_src;
        return st->exec_idx_map.get(r);
    };

    // Check if an already-emitted single-node generic op at e_prev < e_cur can be backward-absorbed
    auto can_absorb_prev = [&](int32_t e_prev, size_t e_cur, const agx_va_range * new_ins, int n_new_ins) -> bool {
        if (e_prev < 0 || (size_t) e_prev >= e_cur) return false;
        if (consumed[(size_t) e_prev]) return false;
        const auto & sl = anchors[(size_t) e_prev];
        if (sl.cnt != 1u || st->items[sl.beg].kind != ITEM_GENERIC_NODE) return false;
        for (size_t j = (size_t) e_prev + 1u; j < e_cur; ++j) {
            for (int r = 0; r < n_new_ins; ++r) {
                if (va_range_overlap(raw_vas[j].w, new_ins[r])) return false;
            }
        }
        return true;
    };

    auto commit_single_item = [&](int32_t anchor_e, const int32_t * set_arr, int set_n, const agx_exec_item & it) {
        const uint32_t idx = (uint32_t) st->items.size();
        st->items.push_back(it);
        anchors[(size_t) anchor_e] = { idx, 1u };
        for (int k = 0; k < set_n; ++k) {
            consumed[(size_t) set_arr[k]] = 1u;
        }
    };

    // Helper to recognize strided CONT(VIEW_3D(src, 1, n_embd, n_tokens, nb1, nb2, off))
    auto match_cont_view3d = [&](const struct ggml_tensor * t, int64_t n_embd, int64_t n_tokens,
                                 const struct ggml_tensor ** out_view, uint32_t * out_nb0, uint32_t * out_nb1,
                                 int32_t * out_e_cont) -> bool {
        if (!t || t->op != GGML_OP_CONT || (t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return false;
        if (st->use_map.get(t) != 1u) return false;
        const struct ggml_tensor * v = t->src[0];
        if (!v || v->op != GGML_OP_VIEW || v->type != GGML_TYPE_F32) return false;
        if (v->ne[0] != 1 || v->ne[1] != n_embd || v->ne[2] != n_tokens || v->ne[3] != 1) return false;
        if ((v->nb[1] & 3u) != 0 || (v->nb[2] & 3u) != 0) return false;
        if (agx_tensor_gpu_va(v) == 0) return false;
        int32_t ec = st->exec_idx_map.get(t);
        if (ec < 0) return false;
        *out_view   = v;
        *out_nb0    = (uint32_t) (v->nb[1] / 4u);
        *out_nb1    = (uint32_t) (v->nb[2] / 4u);
        *out_e_cont = ec;
        return true;
    };

    uint64_t g_l2_clip = 0, g_situ = 0, g_softcap = 0, g_ctrl = 0, g_rms_mul = 0, g_add_sc = 0, g_sig_mul = 0;
    uint64_t g_mix = 0, g_moe_wsum = 0, g_ssm_res = 0, g_moe_norm = 0;
    uint64_t g_prep_store = 0, g_full_sconv = 0, g_router = 0, g_rel_bias = 0, g_cont_abs = 0;

    for (size_t e = 0; e < n_exec; ++e) {
        if (consumed[e]) continue;
        struct ggml_tensor * node = cgraph->nodes[st->exec_nodes[e]];

        // --------------------------------------------------------------------
        // Fusion 1 (Wave 3 Upgraded): Q/K/V/R Prep + FWHT + SET_ROWS(F32/F16/Q8_0)
        //   [MUL(clip_scale)] -> (L2_NORM+SCALE | RMS_NORM) -> [MUL(tau)] -> [FWHT64/128] -> [SET_ROWS]
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 0)) &&
            (node->op == GGML_OP_MUL || node->op == GGML_OP_L2_NORM || node->op == GGML_OP_RMS_NORM)) {
            int32_t fset[8];
            int fset_n = 0;

            const struct ggml_tensor * raw_src   = nullptr;
            const struct ggml_tensor * clip_sc_t = nullptr;
            struct ggml_tensor *       norm_n    = nullptr;
            struct ggml_tensor *       tail_n    = nullptr;
            bool is_l2_clip = false;
            float norm_param = 0.0f;
            bool head_ok = false;

            if (node->op == GGML_OP_MUL && node->type == GGML_TYPE_F32 &&
                node->ne[0] > 0 && node->ne[0] <= 128 && node->ne[1] > 0 && node->ne[2] > 0 && node->ne[3] == 1) {
                struct ggml_tensor * l2n = next_single_use_compute(node, false);
                if (l2n && l2n->op == GGML_OP_L2_NORM) {
                    struct ggml_tensor * scn = next_single_use_compute(l2n, false);
                    if (scn && scn->op == GGML_OP_SCALE) {
                        float t0 = 0.0f, s0 = 0.0f, b0 = 0.0f;
                        std::memcpy(&t0, l2n->op_params, sizeof(float));
                        read_scale_params(scn, &s0, &b0);
                        const struct ggml_tensor * a = node->src[0];
                        const struct ggml_tensor * b = node->src[1];
                        if (b0 == 0.0f && t0 > 0.0f && std::fabs(t0 - s0) < 1e-6f && a && b &&
                            a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
                            ggml_is_contiguous(a) && ggml_is_contiguous(b)) {
                            const int64_t D = node->ne[0], H = node->ne[1], M = node->ne[2];
                            if (b->ne[0] == D && b->ne[1] == H && b->ne[2] == 1 && b->ne[3] == 1 &&
                                a->ne[0] == D && a->ne[1] == H && a->ne[2] == M && a->ne[3] == 1) {
                                raw_src = a; clip_sc_t = b;
                            } else if (a->ne[0] == D && a->ne[1] == H && a->ne[2] == 1 && a->ne[3] == 1 &&
                                       b->ne[0] == D && b->ne[1] == H && b->ne[2] == M && b->ne[3] == 1) {
                                raw_src = b; clip_sc_t = a;
                            }
                            if (raw_src && clip_sc_t) {
                                norm_n = l2n; tail_n = scn;
                                is_l2_clip = true; norm_param = t0;
                                fset[fset_n++] = (int32_t) e;
                                fset[fset_n++] = st->exec_idx_map.get(l2n);
                                fset[fset_n++] = st->exec_idx_map.get(scn);
                                head_ok = true;
                            }
                        }
                    }
                }
            } else if (node->op == GGML_OP_L2_NORM && node->ne[0] > 0 && node->ne[0] <= 128 &&
                       node->ne[1] > 0 && node->ne[2] > 0 && node->ne[3] == 1) {
                struct ggml_tensor * scn = next_single_use_compute(node, false);
                if (scn && scn->op == GGML_OP_SCALE) {
                    float t0 = 0.0f, s0 = 0.0f, b0 = 0.0f;
                    std::memcpy(&t0, node->op_params, sizeof(float));
                    read_scale_params(scn, &s0, &b0);
                    if (b0 == 0.0f && t0 > 0.0f && std::fabs(t0 - s0) < 1e-6f &&
                        node->src[0] && ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(scn)) {
                        raw_src = node->src[0];
                        norm_n = node; tail_n = scn;
                        is_l2_clip = true; norm_param = t0;
                        fset[fset_n++] = (int32_t) e;
                        fset[fset_n++] = st->exec_idx_map.get(scn);
                        head_ok = true;
                    }
                }
            } else if (node->op == GGML_OP_RMS_NORM && node->ne[0] > 0 && node->ne[0] <= 128 &&
                       node->ne[1] > 0 && node->ne[2] > 0 && node->ne[3] == 1 &&
                       node->src[0] && ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(node)) {
                raw_src = node->src[0];
                norm_n = node; tail_n = node;
                is_l2_clip = false;
                std::memcpy(&norm_param, node->op_params, sizeof(float));
                fset[fset_n++] = (int32_t) e;
                head_ok = true;
            }

            if (head_ok) {
                const int64_t D = norm_n->ne[0];
                const int64_t H = norm_n->ne[1];
                const int64_t M = norm_n->ne[2];

                // Optional trailing MUL(tau) where tau is [1, 1, M, 1]
                const struct ggml_tensor * tau_t = nullptr;
                if (struct ggml_tensor * u = next_single_use_compute(tail_n, false)) {
                    if (u->op == GGML_OP_MUL && u->type == GGML_TYPE_F32 && ggml_is_contiguous(u) &&
                        u->ne[0] == D && u->ne[1] == H && u->ne[2] == M && u->ne[3] == 1) {
                        const struct ggml_tensor * other = nullptr;
                        if (strip_reshape(u->src[0]) == tail_n) other = u->src[1];
                        else if (strip_reshape(u->src[1]) == tail_n) other = u->src[0];
                        if (other && other->type == GGML_TYPE_F32 && ggml_is_contiguous(other) &&
                            other->ne[0] == 1 && other->ne[1] == 1 && other->ne[2] == M && other->ne[3] == 1 &&
                            agx_tensor_gpu_va(other) != 0) {
                            tau_t  = other;
                            tail_n = u;
                            fset[fset_n++] = st->exec_idx_map.get(u);
                        }
                    }
                }

                // Optional trailing MUL_MAT(FWHT64/128)
                uint32_t fwht_flag = 0u;
                if (struct ggml_tensor * u = next_single_use_compute(tail_n, false)) {
                    if (u->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(u, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                        u->src[1] && strip_reshape(u->src[1]) == tail_n &&
                        u->type == GGML_TYPE_F32 && ggml_is_contiguous(u) &&
                        ggml_nelements(u) == D * H * M) {
                        const int64_t fd = u->src[1]->ne[0];
                        if ((fd == 128 && D == 128) || (fd == 64 && (D == 64 || D == 128))) {
                            fwht_flag = (fd == 128) ? 8u : 4u;
                            tail_n    = u;
                            fset[fset_n++] = st->exec_idx_map.get(u);
                        }
                    }
                }

                // Optional trailing SET_ROWS(F32 / F16 / Q8_0)
                struct ggml_tensor * sr_n = nullptr;
                uint32_t store_mode = 0u;
                if (struct ggml_tensor * u = next_single_use_compute(tail_n, true)) {
                    if (u->op == GGML_OP_SET_ROWS && u->src[0] && u->src[1] &&
                        strip_reshape_fullview(u->src[0]) == tail_n &&
                        u->src[0]->ne[0] == D * H && u->src[0]->ne[1] == M &&
                        u->src[0]->ne[2] == 1 && u->src[0]->ne[3] == 1 &&
                        u->src[1]->ne[0] == M && u->src[1]->ne[1] == 1 && u->src[1]->ne[2] == 1 &&
                        u->ne[2] == 1 && u->ne[3] == 1 &&
                        (u->src[1]->type == GGML_TYPE_I64 || u->src[1]->type == GGML_TYPE_I32)) {
                        if (u->type == GGML_TYPE_F32) store_mode = 1u;
                        else if (u->type == GGML_TYPE_F16) store_mode = 2u;
                        else if (u->type == GGML_TYPE_Q8_0 && (D % 32) == 0) store_mode = 3u;
                        if (store_mode != 0u) {
                            sr_n   = u;
                            tail_n = u;
                            fset[fset_n++] = st->exec_idx_map.get(u);
                        }
                    }
                }

                // Require at least 2 fused ops (so plain RMS_NORM falls through to Fusion 5 or generic!)
                if (fset_n >= 2) {
                    struct ggml_tensor * final_dst = tail_n;
                    const struct ggml_tensor * ids_t = sr_n ? sr_n->src[1] : nullptr;
                    const bool va_ok = (store_mode == 0u)
                        ? (va_same_or_disjoint(final_dst, raw_src) && va_disjoint(final_dst, clip_sc_t) && va_disjoint(final_dst, tau_t))
                        : (va_disjoint(final_dst, raw_src) && va_disjoint(final_dst, clip_sc_t) &&
                           va_disjoint(final_dst, tau_t) && va_disjoint(final_dst, ids_t));
                    if (va_ok && agx_tensor_gpu_va(final_dst) != 0) {
                        agx_va_range all_ins[4];
                        int32_t      all_prods[4];
                        int n_all = 0;
                        all_ins[n_all] = tensor_va_range(raw_src);   all_prods[n_all++] = tensor_prod_idx(raw_src);
                        if (clip_sc_t) { all_ins[n_all] = tensor_va_range(clip_sc_t); all_prods[n_all++] = tensor_prod_idx(clip_sc_t); }
                        if (tau_t)     { all_ins[n_all] = tensor_va_range(tau_t);     all_prods[n_all++] = tensor_prod_idx(tau_t); }
                        if (ids_t)     { all_ins[n_all] = tensor_va_range(ids_t);     all_prods[n_all++] = tensor_prod_idx(ids_t); }

                        agx_va_range extra_ins[3];
                        int n_extra = 0;
                        if (tau_t) extra_ins[n_extra++] = tensor_va_range(tau_t);
                        if (ids_t) extra_ins[n_extra++] = tensor_va_range(ids_t);

                        agx_va_range outs[1] = { tensor_va_range(final_dst) };
                        const int32_t anc = find_safe_anchor(fset, fset_n, all_ins, all_prods, n_all, extra_ins, n_extra, outs, 1);
                        if (anc >= 0) {
                            prep_norm_fwht_store_push pc{};
                            pc.src_va     = agx_tensor_gpu_va(raw_src);
                            pc.scale_va   = clip_sc_t ? agx_tensor_gpu_va(clip_sc_t) : 0ull;
                            pc.tau_va     = tau_t     ? agx_tensor_gpu_va(tau_t)     : 0ull;
                            pc.ids_va     = ids_t     ? agx_tensor_gpu_va(ids_t)     : 0ull;
                            pc.dst_va     = agx_tensor_gpu_va(final_dst);
                            pc.head_dim   = (uint32_t) D;
                            pc.n_head     = (uint32_t) H;
                            pc.n_tokens   = (uint32_t) M;
                            pc.flags      = (is_l2_clip ? 1u : 2u) | fwht_flag;
                            pc.store_mode = store_mode;
                            pc.ids_is_i64 = (ids_t && ids_t->type == GGML_TYPE_I64) ? 1u : 0u;
                            pc.i_nb0      = ids_t ? (uint32_t) ids_t->nb[0] : 0u;
                            pc.d_nb1      = sr_n  ? (uint32_t) sr_n->nb[1]  : 0u;
                            pc.param      = norm_param;
                            pc.s_nb1      = (uint32_t) D;
                            pc.s_nb2      = (uint32_t) (D * H);

                            agx_exec_item it{};
                            it.kind          = ITEM_PREP_STORE;
                            it.gx            = (pc.n_head + 3u) / 4u;
                            it.gy            = pc.n_tokens;
                            it.gz            = 1u;
                            it.nodes_covered = (uint32_t) fset_n;
                            for (int r = 0; r < n_all; ++r) it.add_read(all_ins[r]);
                            it.add_write(outs[0]);
                            it.set_push(pc);
                            commit_single_item(anc, fset, fset_n, it);

                            st->stats.fused_prep_store_hits++;
                            g_prep_store++;
                            if (is_l2_clip) { st->stats.fused_l2_clip_hits++; g_l2_clip++; }
                            if (fwht_flag)  { st->stats.fused_fwht_hits++; }
                            continue;
                        }
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 2: Inkling 2 MoE SiTU Activation (9-op DAG -> 1 k_glu dispatch)
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 1)) && node->op == GGML_OP_SCALE && e + 8u < n_exec) {
            struct ggml_tensor * n1 = cgraph->nodes[st->exec_nodes[e + 1u]];
            struct ggml_tensor * n2 = cgraph->nodes[st->exec_nodes[e + 2u]];
            struct ggml_tensor * n3 = cgraph->nodes[st->exec_nodes[e + 3u]];
            struct ggml_tensor * n4 = cgraph->nodes[st->exec_nodes[e + 4u]];
            struct ggml_tensor * n5 = cgraph->nodes[st->exec_nodes[e + 5u]];
            struct ggml_tensor * n6 = cgraph->nodes[st->exec_nodes[e + 6u]];
            struct ggml_tensor * n7 = cgraph->nodes[st->exec_nodes[e + 7u]];
            struct ggml_tensor * n8 = cgraph->nodes[st->exec_nodes[e + 8u]];
            bool any_c = false;
            for (size_t k = 1u; k <= 8u; ++k) if (consumed[e + k]) any_c = true;
            if (!any_c &&
                n1->op == GGML_OP_UNARY && ggml_get_unary_op(n1) == GGML_UNARY_OP_TANH &&
                n2->op == GGML_OP_SCALE &&
                n3->op == GGML_OP_UNARY && ggml_get_unary_op(n3) == GGML_UNARY_OP_SIGMOID &&
                n4->op == GGML_OP_MUL &&
                n5->op == GGML_OP_SCALE &&
                n6->op == GGML_OP_UNARY && ggml_get_unary_op(n6) == GGML_UNARY_OP_TANH &&
                n7->op == GGML_OP_SCALE &&
                n8->op == GGML_OP_MUL &&
                n3->src[0] == node->src[0] &&
                chain_uses_exact(n1->src[0], node) &&
                chain_uses_exact(n2->src[0], n1) &&
                ((chain_uses_exact(n4->src[0], n2) && chain_uses_exact(n4->src[1], n3)) ||
                 (chain_uses_exact(n4->src[1], n2) && chain_uses_exact(n4->src[0], n3))) &&
                chain_uses_exact(n6->src[0], n5) &&
                chain_uses_exact(n7->src[0], n6) &&
                ((chain_uses_exact(n8->src[0], n4) && chain_uses_exact(n8->src[1], n7)) ||
                 (chain_uses_exact(n8->src[1], n4) && chain_uses_exact(n8->src[0], n7)))) {
                float s0 = 0, b0 = 0, s2 = 0, b2 = 0, s5 = 0, b5 = 0, s7 = 0, b7 = 0;
                read_scale_params(node, &s0, &b0);
                read_scale_params(n2,   &s2, &b2);
                read_scale_params(n5,   &s5, &b5);
                read_scale_params(n7,   &s7, &b7);
                const struct ggml_tensor * g = node->src[0];
                const struct ggml_tensor * u = n5->src[0];
                if (b0 == 0.0f && b2 == 0.0f && b5 == 0.0f && b7 == 0.0f &&
                    s2 > 0.0f && s7 > 0.0f &&
                    std::fabs(s0 * s2 - 1.0f) < 1e-4f && std::fabs(s5 * s7 - 1.0f) < 1e-4f &&
                    g && u && g->type == GGML_TYPE_F32 && u->type == GGML_TYPE_F32 && n8->type == GGML_TYPE_F32 &&
                    ggml_are_same_shape(g, u) && ggml_are_same_shape(g, n8) &&
                    g->nb[0] == sizeof(float) && u->nb[0] == sizeof(float) && ggml_is_contiguous(n8) &&
                    g->ne[3] == 1 && u->ne[3] == 1 &&
                    (g->ne[2] == 1 || g->nb[2] == (size_t) g->ne[1] * g->nb[1]) &&
                    (u->ne[2] == 1 || u->nb[2] == (size_t) u->ne[1] * u->nb[1]) &&
                    agx_tensor_gpu_va(u) != 0 && agx_tensor_gpu_va(n8) != 0 &&
                    va_disjoint(g, n8) && va_disjoint(u, n8)) {
                    glu_push pc{};
                    pc.g_va    = agx_tensor_gpu_va(g);
                    pc.u_va    = agx_tensor_gpu_va(u);
                    pc.d_va    = agx_tensor_gpu_va(n8);
                    pc.ne0     = (uint32_t) n8->ne[0];
                    pc.n_rows  = (uint32_t) ggml_nrows(n8);
                    pc.g_nb1   = (uint32_t) (g->nb[1] / 4);
                    pc.u_nb1   = (uint32_t) (u->nb[1] / 4);
                    pc.d_nb1   = (uint32_t) (n8->nb[1] / 4);
                    pc.glu_op  = 99u;
                    pc.swapped = 0u;
                    pc.alpha   = s2;
                    pc.limit   = s7;

                    int32_t fset[9];
                    for (int k = 0; k < 9; ++k) fset[k] = (int32_t) (e + (size_t) k);
                    agx_exec_item it{};
                    it.kind          = ITEM_GLU;
                    it.gx            = (pc.ne0 + 255u) / 256u;
                    it.gy            = pc.n_rows;
                    it.gz            = 1u;
                    it.nodes_covered = 9u;
                    it.add_read(tensor_va_range(g));
                    it.add_read(tensor_va_range(u));
                    it.add_write(tensor_va_range(n8));
                    it.set_push(pc);
                    commit_single_item((int32_t) e, fset, 9, it);
                    st->stats.fused_situ_hits++;
                    g_situ++;
                    continue;
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 3: Inkling 2 Logit Softcap (`SCALE(1/c) -> TANH -> SCALE(c)`)
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 2)) && node->op == GGML_OP_SCALE && e + 2u < n_exec &&
            !consumed[e + 1u] && !consumed[e + 2u]) {
            struct ggml_tensor * u_node  = cgraph->nodes[st->exec_nodes[e + 1u]];
            struct ggml_tensor * s2_node = cgraph->nodes[st->exec_nodes[e + 2u]];
            if (u_node->op == GGML_OP_UNARY && ggml_get_unary_op(u_node) == GGML_UNARY_OP_TANH &&
                s2_node->op == GGML_OP_SCALE &&
                chain_uses_exact(u_node->src[0], node) &&
                chain_uses_exact(s2_node->src[0], u_node)) {
                float s1 = 0.0f, b1 = 0.0f, s2 = 0.0f, b2 = 0.0f;
                read_scale_params(node,    &s1, &b1);
                read_scale_params(s2_node, &s2, &b2);
                if (b1 == 0.0f && b2 == 0.0f && s2 > 0.0f && std::fabs(s1 * s2 - 1.0f) < 1e-4f &&
                    ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(s2_node) &&
                    agx_tensor_gpu_va(s2_node) != 0) {
                    int32_t fset[3] = { (int32_t) e, (int32_t) (e + 1u), (int32_t) (e + 2u) };
                    agx_exec_item it{};
                    it.kind          = ITEM_SOFTCAP;
                    it.nodes_covered = 3u;
                    it.va0           = agx_tensor_gpu_va(node->src[0]);
                    it.va1           = agx_tensor_gpu_va(s2_node);
                    it.u0            = (uint32_t) s2_node->ne[0];
                    it.u1            = (uint32_t) ggml_nrows(s2_node);
                    it.f0            = s2;
                    it.add_read(tensor_va_range(node->src[0]));
                    it.add_write(tensor_va_range(s2_node));
                    commit_single_item((int32_t) e, fset, 3, it);
                    st->stats.fused_softcap_hits++;
                    g_softcap++;
                    continue;
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 4 (Wave 3 Upgraded): Inkling 2 Cross-Pass Rank-64 Controller Gate
        //   + backward CONT(VIEW_3D) absorption on cur/msg
        //   + wave-schedulable B<16 sub-items so Q/K/V controllers run in lockstep waves!
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 3)) && node->op == GGML_OP_CONCAT && e + 10u < n_exec &&
            ggml_get_op_params_i32(node, 0) == 1) {
            bool any_c = false;
            for (size_t k = 1u; k <= 10u; ++k) if (consumed[e + k]) any_c = true;
            if (!any_c) {
                struct ggml_tensor * n1  = cgraph->nodes[st->exec_nodes[e + 1u]];
                struct ggml_tensor * n2  = cgraph->nodes[st->exec_nodes[e + 2u]];
                struct ggml_tensor * n3  = cgraph->nodes[st->exec_nodes[e + 3u]];
                struct ggml_tensor * n4  = cgraph->nodes[st->exec_nodes[e + 4u]];
                struct ggml_tensor * n5  = cgraph->nodes[st->exec_nodes[e + 5u]];
                struct ggml_tensor * n6  = cgraph->nodes[st->exec_nodes[e + 6u]];
                struct ggml_tensor * n7  = cgraph->nodes[st->exec_nodes[e + 7u]];
                struct ggml_tensor * n8  = cgraph->nodes[st->exec_nodes[e + 8u]];
                struct ggml_tensor * n9  = cgraph->nodes[st->exec_nodes[e + 9u]];
                struct ggml_tensor * n10 = cgraph->nodes[st->exec_nodes[e + 10u]];
                const struct ggml_tensor * cur = strip_reshape(node->src[0]);
                const struct ggml_tensor * msg = strip_reshape(node->src[1]);
                if (cur && msg && cur != msg &&
                    n1->op == GGML_OP_RMS_NORM &&
                    n2->op == GGML_OP_MUL_MAT &&
                    n3->op == GGML_OP_MUL_MAT &&
                    n4->op == GGML_OP_UNARY && ggml_get_unary_op(n4) == GGML_UNARY_OP_TANH &&
                    n5->op == GGML_OP_MUL &&
                    n6->op == GGML_OP_SUB &&
                    n7->op == GGML_OP_MUL_MAT &&
                    n8->op == GGML_OP_UNARY && ggml_get_unary_op(n8) == GGML_UNARY_OP_TANH &&
                    n9->op == GGML_OP_MUL &&
                    n10->op == GGML_OP_ADD &&
                    chain_uses_exact(n1->src[0], node) &&
                    chain_uses_exact(n2->src[1], n1) &&
                    chain_uses_exact(n3->src[1], n2, 2u) &&
                    chain_uses_exact(n7->src[1], n2, 2u) &&
                    chain_uses_exact(n4->src[0], n3) &&
                    ((chain_uses_exact(n5->src[0], n4) && strip_reshape(n5->src[1]) == cur) ||
                     (chain_uses_exact(n5->src[1], n4) && strip_reshape(n5->src[0]) == cur)) &&
                    strip_reshape(n6->src[0]) == cur && chain_uses_exact(n6->src[1], n5) &&
                    chain_uses_exact(n8->src[0], n7) &&
                    ((chain_uses_exact(n9->src[0], n8) && strip_reshape(n9->src[1]) == msg) ||
                     (chain_uses_exact(n9->src[1], n8) && strip_reshape(n9->src[0]) == msg)) &&
                    ((chain_uses_exact(n10->src[0], n6) && chain_uses_exact(n10->src[1], n9)) ||
                     (chain_uses_exact(n10->src[1], n6) && chain_uses_exact(n10->src[0], n9)))) {

                    const struct ggml_tensor * w_down   = n2->src[0];
                    const struct ggml_tensor * w_forget = n3->src[0];
                    const struct ggml_tensor * w_gain   = n7->src[0];
                    const int64_t n_embd   = node->ne[0];
                    const int64_t n_tokens = node->ne[2];
                    const int64_t rank     = w_down ? w_down->ne[1] : 0;

                    if (node->ne[1] == 2 && node->ne[3] == 1 &&
                        cur->type == GGML_TYPE_F32 && msg->type == GGML_TYPE_F32 && n10->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(cur) && ggml_is_contiguous(msg) && ggml_is_contiguous(n10) &&
                        ggml_nelements(cur) == n_embd * n_tokens &&
                        ggml_nelements(msg) == n_embd * n_tokens &&
                        ggml_nelements(n10) == n_embd * n_tokens &&
                        n_embd > 0 && n_embd <= 3840 && n_tokens > 0 && rank > 0 && rank <= 256 &&
                        w_down && w_forget && w_gain &&
                        w_down->ne[0] == 2 * n_embd && w_forget->ne[0] == rank && w_forget->ne[1] == n_embd &&
                        w_gain->ne[0] == rank && w_gain->ne[1] == n_embd &&
                        agx_tensor_gpu_va(n10) != 0) {

                        struct ggml_tensor * final_dst = n10;
                        const struct ggml_tensor * norm_w = nullptr;
                        float eps_ctrl = 1e-6f, eps_norm = 1e-6f;
                        std::memcpy(&eps_ctrl, n1->op_params, sizeof(float));
                        size_t advance = 10u;

                        if (e + 12u < n_exec && !consumed[e + 11u] && !consumed[e + 12u]) {
                            struct ggml_tensor * n11 = cgraph->nodes[st->exec_nodes[e + 11u]];
                            struct ggml_tensor * n12 = cgraph->nodes[st->exec_nodes[e + 12u]];
                            if (n11->op == GGML_OP_RMS_NORM && n12->op == GGML_OP_MUL &&
                                chain_uses_exact(n11->src[0], n10) && ggml_is_contiguous(n12) &&
                                agx_tensor_gpu_va(n12) != 0) {
                                const struct ggml_tensor * nw = nullptr;
                                if (chain_uses_exact(n12->src[0], n11)) nw = n12->src[1];
                                else if (chain_uses_exact(n12->src[1], n11)) nw = n12->src[0];
                                if (nw && nw->type == GGML_TYPE_F32 && ggml_is_contiguous(nw) &&
                                    nw->ne[0] == n_embd && nw->ne[1] == 1 && nw->ne[2] == 1 && nw->ne[3] == 1 &&
                                    agx_tensor_gpu_va(nw) != 0) {
                                    std::memcpy(&eps_norm, n11->op_params, sizeof(float));
                                    norm_w    = nw;
                                    final_dst = n12;
                                    advance   = 12u;
                                }
                            }
                        }

                        if (n_tokens >= 16 &&
                            va_same_or_disjoint(final_dst, cur) && va_same_or_disjoint(final_dst, msg) &&
                            is_ctrl_wtype_ok(w_down) && is_ctrl_wtype_ok(w_forget) && is_ctrl_wtype_ok(w_gain)) {
                            int32_t fset[13];
                            for (size_t k = 0; k <= advance; ++k) fset[k] = (int32_t) (e + k);
                            agx_exec_item it{};
                            it.kind          = ITEM_CTRL_SRAM;
                            it.nodes_covered = (uint32_t) (advance + 1u);
                            it.ctrl_desc.cur_va         = agx_tensor_gpu_va(cur);
                            it.ctrl_desc.msg_va         = agx_tensor_gpu_va(msg);
                            it.ctrl_desc.down_va        = agx_tensor_gpu_va(w_down);
                            it.ctrl_desc.gain_va        = agx_tensor_gpu_va(w_gain);
                            it.ctrl_desc.forget_va      = agx_tensor_gpu_va(w_forget);
                            it.ctrl_desc.norm_w_va      = norm_w ? agx_tensor_gpu_va(norm_w) : 0ull;
                            it.ctrl_desc.out_va         = agx_tensor_gpu_va(final_dst);
                            it.ctrl_desc.down_wtype     = agx_inkling2_tensor_wtype(w_down,   &it.ctrl_desc.down_row_bytes);
                            it.ctrl_desc.gain_wtype     = agx_inkling2_tensor_wtype(w_gain,   &it.ctrl_desc.gain_row_bytes);
                            it.ctrl_desc.forget_wtype   = agx_inkling2_tensor_wtype(w_forget, &it.ctrl_desc.forget_row_bytes);
                            it.ctrl_desc.cur_stride_f32 = (uint32_t) n_embd;
                            it.ctrl_desc.msg_stride_f32 = (uint32_t) n_embd;
                            it.ctrl_desc.msg_offset_f32 = 0u;
                            it.ctrl_desc.out_stride_f32 = (uint32_t) n_embd;
                            it.u0 = (uint32_t) n_embd;
                            it.u1 = (uint32_t) rank;
                            it.u2 = (uint32_t) n_tokens;
                            it.f0 = eps_ctrl;
                            it.f1 = eps_norm;
                            it.add_read(tensor_va_range(cur));
                            it.add_read(tensor_va_range(msg));
                            it.add_read(tensor_va_range(w_down));
                            it.add_read(tensor_va_range(w_forget));
                            it.add_read(tensor_va_range(w_gain));
                            if (norm_w) it.add_read(tensor_va_range(norm_w));
                            it.add_write(tensor_va_range(final_dst));
                            commit_single_item((int32_t) e, fset, (int) (advance + 1u), it);
                            st->stats.fused_ctrl_gate_hits++;
                            g_ctrl++;
                            continue;
                        } else if (n_tokens < 16 &&
                                   agx_tensor_gpu_va(n1) != 0 && agx_tensor_gpu_va(n2) != 0 &&
                                   agx_tensor_gpu_va(n3) != 0 && agx_tensor_gpu_va(n6) != 0 &&
                                   agx_tensor_gpu_va(n7) != 0 &&
                                   agx_gemm_supports_mul_mat(n2) && agx_gemm_supports_mul_mat(n3) &&
                                   agx_gemm_supports_mul_mat(n7)) {
                            const struct ggml_tensor * cur_act = cur;
                            const struct ggml_tensor * msg_act = msg;
                            uint32_t cur_nb0 = 1u, cur_nb1 = (uint32_t) n_embd;
                            uint32_t msg_nb0 = 1u, msg_nb1 = (uint32_t) n_embd;
                            int32_t ec_cur = -1, ec_msg = -1;

                            auto try_abs_ctrl_cont = [&](const struct ggml_tensor * t, uint32_t expected_uses,
                                                         const struct ggml_tensor ** out_act,
                                                         uint32_t * out_nb0, uint32_t * out_nb1, int32_t * out_ec) {
                                if (!t || t->op != GGML_OP_CONT || (t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return;
                                const uint32_t uc = st->use_map.get(t);
                                if (uc != 1u && uc != expected_uses) return;
                                if (uc == 1u) {
                                    struct ggml_tensor * u = st->first_use.get(t);
                                    if (!u || u->op != GGML_OP_RESHAPE || (u->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
                                        st->use_map.get(u) != expected_uses) return;
                                }
                                const struct ggml_tensor * v = t->src[0];
                                if (!v || v->op != GGML_OP_VIEW || v->type != GGML_TYPE_F32) return;
                                if (v->ne[0] != 1 || v->ne[1] != n_embd || v->ne[2] != n_tokens || v->ne[3] != 1) return;
                                if ((v->nb[1] & 3u) != 0 || (v->nb[2] & 3u) != 0 || agx_tensor_gpu_va(v) == 0) return;
                                int32_t ec = st->exec_idx_map.get(t);
                                agx_va_range vr = tensor_va_range(v);
                                if (!can_absorb_prev(ec, e, &vr, 1)) return;
                                *out_act = v;
                                *out_nb0 = (uint32_t) (v->nb[1] / 4u);
                                *out_nb1 = (uint32_t) (v->nb[2] / 4u);
                                *out_ec  = ec;
                            };

                            try_abs_ctrl_cont(cur, 3u, &cur_act, &cur_nb0, &cur_nb1, &ec_cur);
                            try_abs_ctrl_cont(msg, 2u, &msg_act, &msg_nb0, &msg_nb1, &ec_msg);

                            auto dst_ok_with = [&](const struct ggml_tensor * d, const struct ggml_tensor * s, uint32_t nb0) -> bool {
                                return (nb0 > 1u) ? va_disjoint(d, s) : va_same_or_disjoint(d, s);
                            };

                            const bool va_safe_abs =
                                va_disjoint(n1, cur_act) && va_disjoint(n1, msg_act) &&
                                va_disjoint(n2, cur_act) && va_disjoint(n2, msg_act) &&
                                va_disjoint(n3, cur_act) && va_disjoint(n3, msg_act) &&
                                dst_ok_with(n6, cur_act, cur_nb0) && va_disjoint(n6, msg_act) && va_same_or_disjoint(n6, n3) &&
                                va_disjoint(n7, n6) && va_disjoint(n7, msg_act) &&
                                va_same_or_disjoint(final_dst, n6) && dst_ok_with(final_dst, msg_act, msg_nb0) &&
                                va_same_or_disjoint(final_dst, n7);

                            if (!va_safe_abs && (ec_cur >= 0 || ec_msg >= 0)) {
                                cur_act = cur; cur_nb0 = 1u; cur_nb1 = (uint32_t) n_embd; ec_cur = -1;
                                msg_act = msg; msg_nb0 = 1u; msg_nb1 = (uint32_t) n_embd; ec_msg = -1;
                            }

                            const bool va_safe =
                                va_disjoint(n1, cur_act) && va_disjoint(n1, msg_act) &&
                                va_disjoint(n2, cur_act) && va_disjoint(n2, msg_act) &&
                                va_disjoint(n3, cur_act) && va_disjoint(n3, msg_act) &&
                                dst_ok_with(n6, cur_act, cur_nb0) && va_disjoint(n6, msg_act) && va_same_or_disjoint(n6, n3) &&
                                va_disjoint(n7, n6) && va_disjoint(n7, msg_act) &&
                                va_same_or_disjoint(final_dst, n6) && dst_ok_with(final_dst, msg_act, msg_nb0) &&
                                va_same_or_disjoint(final_dst, n7);

                            if (va_safe) {
                                if (ec_cur >= 0) {
                                    anchors[(size_t) ec_cur].cnt = 0u;
                                    consumed[(size_t) ec_cur]    = 1u;
                                    st->stats.fused_cont_absorbed++;
                                    g_cont_abs++;
                                }
                                if (ec_msg >= 0) {
                                    anchors[(size_t) ec_msg].cnt = 0u;
                                    consumed[(size_t) ec_msg]    = 1u;
                                    st->stats.fused_cont_absorbed++;
                                    g_cont_abs++;
                                }

                                const uint32_t slice_beg = (uint32_t) st->items.size();
                                const uint32_t extra_cov = (ec_cur >= 0 ? 1u : 0u) + (ec_msg >= 0 ? 1u : 0u);

                                // Sub-item 1: k_ctrl_norm_cat -> n1
                                {
                                    ctrl_norm_cat_push ncpc{};
                                    ncpc.cur_va   = agx_tensor_gpu_va(cur_act);
                                    ncpc.msg_va   = agx_tensor_gpu_va(msg_act);
                                    ncpc.dst_va   = agx_tensor_gpu_va(n1);
                                    ncpc.n_embd   = (uint32_t) n_embd;
                                    ncpc.n_tokens = (uint32_t) n_tokens;
                                    ncpc.eps      = eps_ctrl;
                                    ncpc.cur_nb0  = cur_nb0;
                                    ncpc.cur_nb1  = cur_nb1;
                                    ncpc.msg_nb0  = msg_nb0;
                                    ncpc.msg_nb1  = msg_nb1;
                                    agx_exec_item it{};
                                    it.kind          = ITEM_CTRL_NORM_CAT;
                                    it.gx            = 2u;
                                    it.gy            = ncpc.n_tokens;
                                    it.gz            = 1u;
                                    it.nodes_covered = 2u + extra_cov;
                                    it.add_read(tensor_va_range(cur_act));
                                    it.add_read(tensor_va_range(msg_act));
                                    it.add_write(tensor_va_range(n1));
                                    it.set_push(ncpc);
                                    st->items.push_back(it);
                                }
                                // Sub-item 2: W_down (n2)
                                {
                                    agx_exec_item it{};
                                    it.kind          = ITEM_GENERIC_NODE;
                                    it.node          = n2;
                                    it.nodes_covered = 1u;
                                    it.add_read(tensor_va_range(n2->src[0]));
                                    it.add_read(tensor_va_range(n2->src[1]));
                                    it.add_write(tensor_va_range(n2));
                                    st->items.push_back(it);
                                }

                                const bool can_overlap_fg =
                                    va_disjoint(n3, n7) && va_disjoint(n7, cur_act) &&
                                    dst_ok_with(final_dst, cur_act, cur_nb0) && va_same_or_disjoint(final_dst, n3);
                                if (can_overlap_fg) {
                                    {
                                        agx_exec_item it{};
                                        it.kind = ITEM_GENERIC_NODE; it.node = n3; it.nodes_covered = 1u;
                                        it.add_read(tensor_va_range(n3->src[0]));
                                        it.add_read(tensor_va_range(n3->src[1]));
                                        it.add_write(tensor_va_range(n3));
                                        st->items.push_back(it);
                                    }
                                    {
                                        agx_exec_item it{};
                                        it.kind = ITEM_GENERIC_NODE; it.node = n7; it.nodes_covered = 1u;
                                        it.add_read(tensor_va_range(n7->src[0]));
                                        it.add_read(tensor_va_range(n7->src[1]));
                                        it.add_write(tensor_va_range(n7));
                                        st->items.push_back(it);
                                    }
                                    {
                                        ctrl_mix_push mpc{};
                                        mpc.cur_va    = agx_tensor_gpu_va(cur_act);
                                        mpc.msg_va    = agx_tensor_gpu_va(msg_act);
                                        mpc.forget_va = agx_tensor_gpu_va(n3);
                                        mpc.gain_va   = agx_tensor_gpu_va(n7);
                                        mpc.norm_w_va = norm_w ? agx_tensor_gpu_va(norm_w) : 0ull;
                                        mpc.out_va    = agx_tensor_gpu_va(final_dst);
                                        mpc.n_embd    = (uint32_t) n_embd;
                                        mpc.n_tokens  = (uint32_t) n_tokens;
                                        mpc.eps_norm  = eps_norm;
                                        mpc.mode      = 0u;
                                        mpc.cur_nb0   = cur_nb0; mpc.cur_nb1 = cur_nb1;
                                        mpc.msg_nb0   = msg_nb0; mpc.msg_nb1 = msg_nb1;
                                        agx_exec_item it{};
                                        it.kind = ITEM_CTRL_MIX;
                                        it.gx = mpc.n_tokens; it.gy = 1u; it.gz = 1u;
                                        it.nodes_covered = (uint32_t) (advance + 1u - 5u);
                                        it.add_read(tensor_va_range(cur_act));
                                        it.add_read(tensor_va_range(msg_act));
                                        it.add_read(tensor_va_range(n3));
                                        it.add_read(tensor_va_range(n7));
                                        if (norm_w) it.add_read(tensor_va_range(norm_w));
                                        it.add_write(tensor_va_range(final_dst));
                                        it.set_push(mpc);
                                        st->items.push_back(it);
                                    }
                                } else {
                                    {
                                        agx_exec_item it{};
                                        it.kind = ITEM_GENERIC_NODE; it.node = n3; it.nodes_covered = 1u;
                                        it.add_read(tensor_va_range(n3->src[0]));
                                        it.add_read(tensor_va_range(n3->src[1]));
                                        it.add_write(tensor_va_range(n3));
                                        st->items.push_back(it);
                                    }
                                    {
                                        ctrl_mix_push fpc{};
                                        fpc.cur_va    = agx_tensor_gpu_va(cur_act);
                                        fpc.forget_va = agx_tensor_gpu_va(n3);
                                        fpc.out_va    = agx_tensor_gpu_va(n6);
                                        fpc.n_embd    = (uint32_t) n_embd;
                                        fpc.n_tokens  = (uint32_t) n_tokens;
                                        fpc.mode      = 1u;
                                        fpc.cur_nb0   = cur_nb0; fpc.cur_nb1 = cur_nb1;
                                        fpc.msg_nb0   = 1u;      fpc.msg_nb1 = (uint32_t) n_embd;
                                        agx_exec_item it{};
                                        it.kind = ITEM_CTRL_MIX;
                                        it.gx = fpc.n_tokens; it.gy = 1u; it.gz = 1u;
                                        it.nodes_covered = 3u;
                                        it.add_read(tensor_va_range(cur_act));
                                        it.add_read(tensor_va_range(n3));
                                        it.add_write(tensor_va_range(n6));
                                        it.set_push(fpc);
                                        st->items.push_back(it);
                                    }
                                    {
                                        agx_exec_item it{};
                                        it.kind = ITEM_GENERIC_NODE; it.node = n7; it.nodes_covered = 1u;
                                        it.add_read(tensor_va_range(n7->src[0]));
                                        it.add_read(tensor_va_range(n7->src[1]));
                                        it.add_write(tensor_va_range(n7));
                                        st->items.push_back(it);
                                    }
                                    {
                                        ctrl_mix_push gpc{};
                                        gpc.cur_va    = agx_tensor_gpu_va(n6);
                                        gpc.msg_va    = agx_tensor_gpu_va(msg_act);
                                        gpc.gain_va   = agx_tensor_gpu_va(n7);
                                        gpc.norm_w_va = norm_w ? agx_tensor_gpu_va(norm_w) : 0ull;
                                        gpc.out_va    = agx_tensor_gpu_va(final_dst);
                                        gpc.n_embd    = (uint32_t) n_embd;
                                        gpc.n_tokens  = (uint32_t) n_tokens;
                                        gpc.eps_norm  = eps_norm;
                                        gpc.mode      = 2u;
                                        gpc.cur_nb0   = 1u;      gpc.cur_nb1 = (uint32_t) n_embd;
                                        gpc.msg_nb0   = msg_nb0; gpc.msg_nb1 = msg_nb1;
                                        agx_exec_item it{};
                                        it.kind = ITEM_CTRL_MIX;
                                        it.gx = gpc.n_tokens; it.gy = 1u; it.gz = 1u;
                                        it.nodes_covered = (uint32_t) (advance + 1u - 8u);
                                        it.add_read(tensor_va_range(n6));
                                        it.add_read(tensor_va_range(msg_act));
                                        it.add_read(tensor_va_range(n7));
                                        if (norm_w) it.add_read(tensor_va_range(norm_w));
                                        it.add_write(tensor_va_range(final_dst));
                                        it.set_push(gpc);
                                        st->items.push_back(it);
                                    }
                                }

                                anchors[e] = { slice_beg, (uint32_t) (st->items.size() - slice_beg) };
                                for (size_t k = 0; k <= advance; ++k) consumed[e + k] = 1u;
                                st->stats.fused_ctrl_gate_hits++;
                                g_ctrl++;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 5 (Wave 3 Upgraded): Weighted RMSNorm (`[CONT(VIEW_3D)] -> RMS_NORM -> MUL(1D w)`)
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 4)) && node->op == GGML_OP_RMS_NORM && e + 1u < n_exec && !consumed[e + 1u]) {
            struct ggml_tensor * mul_node = cgraph->nodes[st->exec_nodes[e + 1u]];
            if (mul_node->op == GGML_OP_MUL && mul_node->type == GGML_TYPE_F32 &&
                mul_node->nb[0] == sizeof(float) && agx_tensor_gpu_va(mul_node) != 0) {
                const struct ggml_tensor * w = nullptr;
                if (chain_uses_exact(mul_node->src[0], node))      w = mul_node->src[1];
                else if (chain_uses_exact(mul_node->src[1], node)) w = mul_node->src[0];
                if (w && w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) &&
                    w->ne[0] == node->ne[0] && w->ne[1] == 1 && w->ne[2] == 1 && w->ne[3] == 1 &&
                    ggml_are_same_shape(node, mul_node) && agx_tensor_gpu_va(w) != 0) {
                    const struct ggml_tensor * a = node->src[0];
                    uint32_t nb00 = 1u;
                    uint32_t nb01 = (uint32_t) (a->nb[1] / 4u);
                    uint32_t nb02 = (uint32_t) (a->nb[2] / 4u);
                    uint32_t nb03 = (uint32_t) (a->nb[3] / 4u);
                    const struct ggml_tensor * a_act = a;
                    int32_t ec_cont = -1;

                    if (node->ne[2] == 1 && node->ne[3] == 1) {
                        const struct ggml_tensor * a_cont = strip_reshape(a);
                        const struct ggml_tensor * v = nullptr;
                        uint32_t v_nb0 = 1u, v_nb1 = 0u;
                        int32_t ec = -1;
                        if (match_cont_view3d(a_cont, node->ne[0], node->ne[1], &v, &v_nb0, &v_nb1, &ec)) {
                            agx_va_range vr = tensor_va_range(v);
                            if (can_absorb_prev(ec, e, &vr, 1) && (v_nb0 == 1u ? va_same_or_disjoint(mul_node, v) : va_disjoint(mul_node, v))) {
                                a_act = v;
                                nb00  = v_nb0;
                                nb01  = v_nb1;
                                nb02  = 0u;
                                nb03  = 0u;
                                ec_cont = ec;
                            }
                        }
                    }

                    if (ec_cont >= 0) {
                        anchors[(size_t) ec_cont].cnt = 0u;
                        consumed[(size_t) ec_cont]    = 1u;
                        st->stats.fused_cont_absorbed++;
                        g_cont_abs++;
                    }

                    row_reduce_push pc{};
                    pc.a_va = agx_tensor_gpu_va(a_act);
                    pc.d_va = agx_tensor_gpu_va(mul_node);
                    pc.w_va = agx_tensor_gpu_va(w);
                    pc.ne0  = (uint32_t) node->ne[0]; pc.ne1 = (uint32_t) node->ne[1];
                    pc.ne2  = (uint32_t) node->ne[2]; pc.ne3 = (uint32_t) node->ne[3];
                    pc.nb01 = nb01; pc.nb02 = nb02; pc.nb03 = nb03;
                    pc.nb1  = (uint32_t) (mul_node->nb[1] / 4u);
                    pc.nb2  = (uint32_t) (mul_node->nb[2] / 4u);
                    pc.nb3  = (uint32_t) (mul_node->nb[3] / 4u);
                    std::memcpy(&pc.eps, node->op_params, sizeof(float));
                    pc.op_kind = 0u;
                    pc.nb00    = nb00;

                    int32_t fset[2] = { (int32_t) e, (int32_t) (e + 1u) };
                    agx_exec_item it{};
                    it.kind          = ITEM_ROW_REDUCE;
                    it.gx            = pc.ne1 * pc.ne2 * pc.ne3;
                    it.gy            = 1u;
                    it.gz            = 1u;
                    it.nodes_covered = 2u + (ec_cont >= 0 ? 1u : 0u);
                    it.add_read(tensor_va_range(a_act));
                    it.add_read(tensor_va_range(w));
                    it.add_write(tensor_va_range(mul_node));
                    it.set_push(pc);
                    commit_single_item((int32_t) e, fset, 2, it);
                    st->stats.fused_rms_mul_hits++;
                    g_rms_mul++;
                    continue;
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 6: Residual + Scale (`ADD -> SCALE`)
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 5)) && node->op == GGML_OP_ADD && e + 1u < n_exec && !consumed[e + 1u]) {
            struct ggml_tensor * sc_node = cgraph->nodes[st->exec_nodes[e + 1u]];
            if (sc_node->op == GGML_OP_SCALE && chain_uses_exact(sc_node->src[0], node) &&
                ggml_are_same_shape(node, sc_node) && agx_tensor_gpu_va(sc_node) != 0) {
                const auto * a = node->src[0];
                const auto * b = node->src[1];
                bin_op_push pc{};
                pc.a_va = agx_tensor_gpu_va(a);
                pc.b_va = agx_tensor_gpu_va(b);
                pc.d_va = agx_tensor_gpu_va(sc_node);
                pc.ne0  = (uint32_t) sc_node->ne[0]; pc.ne1  = (uint32_t) sc_node->ne[1];
                pc.ne2  = (uint32_t) sc_node->ne[2]; pc.ne3  = (uint32_t) sc_node->ne[3];
                pc.ne00 = (uint32_t) a->ne[0];       pc.ne01 = (uint32_t) a->ne[1];
                pc.ne02 = (uint32_t) a->ne[2];       pc.ne03 = (uint32_t) a->ne[3];
                pc.ne10 = (uint32_t) b->ne[0];       pc.ne11 = (uint32_t) b->ne[1];
                pc.ne12 = (uint32_t) b->ne[2];       pc.ne13 = (uint32_t) b->ne[3];
                pc.nb00 = (uint32_t) (a->nb[0] / 4); pc.nb01 = (uint32_t) (a->nb[1] / 4);
                pc.nb02 = (uint32_t) (a->nb[2] / 4); pc.nb03 = (uint32_t) (a->nb[3] / 4);
                pc.nb10 = (uint32_t) (b->nb[0] / 4); pc.nb11 = (uint32_t) (b->nb[1] / 4);
                pc.nb12 = (uint32_t) (b->nb[2] / 4); pc.nb13 = (uint32_t) (b->nb[3] / 4);
                pc.nb0  = (uint32_t) (sc_node->nb[0] / 4); pc.nb1 = (uint32_t) (sc_node->nb[1] / 4);
                pc.nb2  = (uint32_t) (sc_node->nb[2] / 4); pc.nb3 = (uint32_t) (sc_node->nb[3] / 4);
                pc.op_kind = 4u;
                pc.total   = (uint32_t) ggml_nelements(sc_node);
                read_scale_params(sc_node, &pc.p0, &pc.p1);

                int32_t fset[2] = { (int32_t) e, (int32_t) (e + 1u) };
                agx_exec_item it{};
                it.kind = ITEM_BIN_OP;
                it.gx = (pc.total + 255u) / 256u; it.gy = 1u; it.gz = 1u;
                it.nodes_covered = 2u;
                it.add_read(tensor_va_range(a));
                it.add_read(tensor_va_range(b));
                it.add_write(tensor_va_range(sc_node));
                it.set_push(pc);
                commit_single_item((int32_t) e, fset, 2, it);
                st->stats.fused_add_scale_hits++;
                g_add_sc++;
                continue;
            }
        }

        // --------------------------------------------------------------------
        // Fusion 11A (Wave 3 NEW): Full MoE Router
        //   SIGMOID -> [ADD(bias)] -> ARGSORT(DESC) -> VIEW(K) -> GET_ROWS -> SUM_ROWS -> SCALE_BIAS -> DIV [+ SCALE]
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 11)) &&
            node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID &&
            node->src[0] && node->src[0]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(node) &&
            node->ne[0] > 0 && node->ne[0] <= 512 && node->ne[2] == 1 && node->ne[3] == 1 &&
            st->use_map.get(node) == 2u && (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
            const int64_t n_exp    = node->ne[0];
            const int64_t n_tokens = node->ne[1];
            const struct ggml_tensor * logits = node->src[0];

            struct ggml_tensor * add_b   = nullptr;
            const struct ggml_tensor * bias_t = nullptr;
            struct ggml_tensor * argsort = nullptr;
            struct ggml_tensor * get_r   = nullptr;
            const size_t w_end = std::min(n_exec, e + 12u);
            for (size_t j = e + 1u; j < w_end; ++j) {
                if (consumed[j]) continue;
                struct ggml_tensor * cand = cgraph->nodes[st->exec_nodes[j]];
                if (!add_b && !argsort && cand->op == GGML_OP_ADD &&
                    (cand->src[0] == node || cand->src[1] == node)) {
                    const struct ggml_tensor * ob = (cand->src[0] == node) ? cand->src[1] : cand->src[0];
                    if (ob && ob->type == GGML_TYPE_F32 && ggml_is_contiguous(ob) &&
                        ob->ne[0] == n_exp && ob->ne[1] == 1 && ob->ne[2] == 1 && ob->ne[3] == 1) {
                        add_b  = cand;
                        bias_t = ob;
                    }
                } else if (!argsort && cand->op == GGML_OP_ARGSORT &&
                           ggml_get_op_params_i32(cand, 0) == GGML_SORT_ORDER_DESC &&
                           cand->src[0] == (add_b ? add_b : node) &&
                           (!add_b || chain_uses_exact(cand->src[0], add_b))) {
                    argsort = cand;
                } else if (argsort && !get_r && cand->op == GGML_OP_GET_ROWS &&
                           cand->src[0] && strip_reshape(cand->src[0]) == node &&
                           cand->src[1] && cand->src[1]->op == GGML_OP_VIEW &&
                           cand->src[1]->view_src == argsort && cand->src[1]->view_offs == 0 &&
                           st->use_map.get(argsort) == 1u && (argsort->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                    get_r = cand;
                    break;
                }
            }
            if (argsort && get_r) {
                const struct ggml_tensor * topk_view = get_r->src[1];
                const int64_t k_out = topk_view->ne[0];
                if (k_out > 0 && k_out <= 32 && topk_view->ne[1] == n_tokens &&
                    topk_view->ne[2] == 1 && topk_view->ne[3] == 1 &&
                    st->use_map.get(get_r) == 1u && (get_r->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                    struct ggml_tensor * w_2d = st->first_use.get(get_r);
                    const struct ggml_tensor * w_in = (w_2d && w_2d->op == GGML_OP_RESHAPE &&
                                                       (w_2d->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                                       st->use_map.get(w_2d) == 2u)
                                                      ? w_2d : (const struct ggml_tensor *) get_r;
                    int32_t e_gr = st->exec_idx_map.get(get_r);
                    struct ggml_tensor * sum_n = nullptr;
                    struct ggml_tensor * sb_n  = nullptr;
                    struct ggml_tensor * div_n = nullptr;
                    if (e_gr >= 0 && (size_t) e_gr + 3u < n_exec) {
                        struct ggml_tensor * c1 = cgraph->nodes[st->exec_nodes[(size_t) e_gr + 1u]];
                        struct ggml_tensor * c2 = cgraph->nodes[st->exec_nodes[(size_t) e_gr + 2u]];
                        struct ggml_tensor * c3 = cgraph->nodes[st->exec_nodes[(size_t) e_gr + 3u]];
                        if (c1->op == GGML_OP_SUM_ROWS && c1->src[0] == w_in &&
                            c2->op == GGML_OP_SCALE && chain_uses_exact(c2->src[0], c1) &&
                            c3->op == GGML_OP_DIV && c3->src[0] == w_in && chain_uses_exact(c3->src[1], c2)) {
                            sum_n = c1; sb_n = c2; div_n = c3;
                        }
                    }
                    if (sum_n && sb_n && div_n) {
                        float s1 = 0.0f, b1 = 0.0f;
                        read_scale_params(sb_n, &s1, &b1);
                        if (s1 == 1.0f && b1 >= 0.0f && ggml_is_contiguous(div_n) &&
                            ggml_nelements(div_n) == k_out * n_tokens) {
                            struct ggml_tensor * final_w = div_n;
                            float w_scale = 1.0f;
                            struct ggml_tensor * sc_n = nullptr;
                            if (struct ggml_tensor * u = next_single_use_compute(div_n, false)) {
                                if (u->op == GGML_OP_SCALE && ggml_is_contiguous(u) &&
                                    ggml_nelements(u) == k_out * n_tokens) {
                                    float s3 = 0.0f, b3 = 0.0f;
                                    read_scale_params(u, &s3, &b3);
                                    if (b3 == 0.0f) {
                                        sc_n    = u;
                                        final_w = u;
                                        w_scale = s3;
                                    }
                                }
                            }
                            if (va_disjoint(logits, argsort) && va_disjoint(logits, final_w) &&
                                va_disjoint(bias_t, argsort) && va_disjoint(bias_t, final_w) &&
                                va_disjoint(argsort, final_w)) {
                                int32_t fset[8];
                                int fn = 0;
                                fset[fn++] = (int32_t) e;
                                if (add_b) fset[fn++] = st->exec_idx_map.get(add_b);
                                fset[fn++] = st->exec_idx_map.get(argsort);
                                fset[fn++] = e_gr;
                                fset[fn++] = st->exec_idx_map.get(sum_n);
                                fset[fn++] = st->exec_idx_map.get(sb_n);
                                fset[fn++] = st->exec_idx_map.get(div_n);
                                if (sc_n) fset[fn++] = st->exec_idx_map.get(sc_n);

                                agx_va_range all_ins[2];
                                int32_t      all_prods[2];
                                int n_ins = 0;
                                all_ins[n_ins] = tensor_va_range(logits); all_prods[n_ins++] = tensor_prod_idx(logits);
                                if (bias_t) { all_ins[n_ins] = tensor_va_range(bias_t); all_prods[n_ins++] = tensor_prod_idx(bias_t); }
                                agx_va_range extra_ins[1];
                                int n_extra = 0;
                                if (bias_t) extra_ins[n_extra++] = tensor_va_range(bias_t);
                                agx_va_range outs[2] = { tensor_va_range(argsort), tensor_va_range(final_w) };

                                const int32_t anc = find_safe_anchor(fset, fn, all_ins, all_prods, n_ins, extra_ins, n_extra, outs, 2);
                                if (anc >= 0) {
                                    router_topk_norm_push pc{};
                                    pc.logits_va  = agx_tensor_gpu_va(logits);
                                    pc.bias_va    = bias_t ? agx_tensor_gpu_va(bias_t) : 0ull;
                                    pc.ids_dst_va = agx_tensor_gpu_va(argsort);
                                    pc.w_dst_va   = agx_tensor_gpu_va(final_w);
                                    pc.n_exp      = (uint32_t) n_exp;
                                    pc.k_out      = (uint32_t) k_out;
                                    pc.n_tokens   = (uint32_t) n_tokens;
                                    pc.ids_nb1    = (uint32_t) (argsort->nb[1] / 4u);
                                    pc.norm_eps   = b1;
                                    pc.w_scale    = w_scale;

                                    agx_exec_item it{};
                                    it.kind          = ITEM_ROUTER_TOPK;
                                    it.gx            = pc.n_tokens;
                                    it.gy            = 1u;
                                    it.gz            = 1u;
                                    it.nodes_covered = (uint32_t) fn;
                                    for (int r = 0; r < n_ins; ++r) it.add_read(all_ins[r]);
                                    it.add_write(outs[0]);
                                    it.add_write(outs[1]);
                                    it.set_push(pc);
                                    commit_single_item(anc, fset, fn, it);
                                    st->stats.fused_router_hits++;
                                    g_router++;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 7: Sigmoid Gate (`SIGMOID(b) -> MUL(a, sig_b)`)
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 6)) && node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID &&
            e + 1u < n_exec && !consumed[e + 1u]) {
            struct ggml_tensor * mul_node = cgraph->nodes[st->exec_nodes[e + 1u]];
            if (mul_node->op == GGML_OP_MUL && mul_node->type == GGML_TYPE_F32 &&
                agx_tensor_gpu_va(mul_node) != 0) {
                const struct ggml_tensor * a = nullptr;
                if (mul_node->src[1] == node && chain_uses_exact(mul_node->src[1], node))      a = mul_node->src[0];
                else if (mul_node->src[0] == node && chain_uses_exact(mul_node->src[0], node)) a = mul_node->src[1];
                const struct ggml_tensor * b = node->src[0];
                if (a && b && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
                    ggml_are_same_shape(b, node) && agx_tensor_gpu_va(a) != 0) {
                    bin_op_push pc{};
                    pc.a_va = agx_tensor_gpu_va(a);
                    pc.b_va = agx_tensor_gpu_va(b);
                    pc.d_va = agx_tensor_gpu_va(mul_node);
                    pc.ne0  = (uint32_t) mul_node->ne[0]; pc.ne1  = (uint32_t) mul_node->ne[1];
                    pc.ne2  = (uint32_t) mul_node->ne[2]; pc.ne3  = (uint32_t) mul_node->ne[3];
                    pc.ne00 = (uint32_t) a->ne[0];        pc.ne01 = (uint32_t) a->ne[1];
                    pc.ne02 = (uint32_t) a->ne[2];        pc.ne03 = (uint32_t) a->ne[3];
                    pc.ne10 = (uint32_t) b->ne[0];        pc.ne11 = (uint32_t) b->ne[1];
                    pc.ne12 = (uint32_t) b->ne[2];        pc.ne13 = (uint32_t) b->ne[3];
                    pc.nb00 = (uint32_t) (a->nb[0] / 4);  pc.nb01 = (uint32_t) (a->nb[1] / 4);
                    pc.nb02 = (uint32_t) (a->nb[2] / 4);  pc.nb03 = (uint32_t) (a->nb[3] / 4);
                    pc.nb10 = (uint32_t) (b->nb[0] / 4);  pc.nb11 = (uint32_t) (b->nb[1] / 4);
                    pc.nb12 = (uint32_t) (b->nb[2] / 4);  pc.nb13 = (uint32_t) (b->nb[3] / 4);
                    pc.nb0  = (uint32_t) (mul_node->nb[0] / 4); pc.nb1 = (uint32_t) (mul_node->nb[1] / 4);
                    pc.nb2  = (uint32_t) (mul_node->nb[2] / 4); pc.nb3 = (uint32_t) (mul_node->nb[3] / 4);
                    pc.op_kind = 5u;
                    pc.total   = (uint32_t) ggml_nelements(mul_node);

                    int32_t fset[2] = { (int32_t) e, (int32_t) (e + 1u) };
                    agx_exec_item it{};
                    it.kind = ITEM_BIN_OP;
                    it.gx = (pc.total + 255u) / 256u; it.gy = 1u; it.gz = 1u;
                    it.nodes_covered = 2u;
                    it.add_read(tensor_va_range(a));
                    it.add_read(tensor_va_range(b));
                    it.add_write(tensor_va_range(mul_node));
                    it.set_push(pc);
                    commit_single_item((int32_t) e, fset, 2, it);
                    st->stats.fused_sig_mul_hits++;
                    g_sig_mul++;
                    continue;
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 8 (Wave 3 Upgraded): Inkling 2 `mix()` Depth-Mix Attention
        //   + backward absorption of CONCAT(blocks_stack, norm_cur, dim=1) & RMS_NORM(cur, 1e-6)!
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 7)) && (node->op == GGML_OP_REPEAT || node->op == GGML_OP_MUL_MAT)) {
            const size_t off = (node->op == GGML_OP_REPEAT) ? 1u : 0u;
            if (e + off + 3u < n_exec) {
                bool any_c = false;
                for (size_t k = 1u; k <= off + 3u; ++k) if (consumed[e + k]) any_c = true;
                if (!any_c) {
                    struct ggml_tensor * mm1 = cgraph->nodes[st->exec_nodes[e + off]];
                    struct ggml_tensor * sm  = cgraph->nodes[st->exec_nodes[e + off + 1u]];
                    struct ggml_tensor * cnt = cgraph->nodes[st->exec_nodes[e + off + 2u]];
                    struct ggml_tensor * mm2 = cgraph->nodes[st->exec_nodes[e + off + 3u]];
                    if (mm1->op == GGML_OP_MUL_MAT &&
                        sm->op  == GGML_OP_SOFT_MAX && !sm->src[1] && !sm->src[2] &&
                        cnt->op == GGML_OP_CONT && cnt->src[0] && cnt->src[0]->op == GGML_OP_TRANSPOSE &&
                        mm2->op == GGML_OP_MUL_MAT &&
                        (off == 0u || chain_uses_exact(mm1->src[1], node)) &&
                        chain_uses_exact(sm->src[0], mm1) &&
                        cnt->src[0]->src[0] == mm1->src[0] &&
                        chain_uses_exact(mm2->src[0], sm) &&
                        chain_uses_exact(mm2->src[1], cnt)) {
                        const struct ggml_tensor * states = mm1->src[0];
                        const struct ggml_tensor * orig_q = (off == 1u) ? node->src[0] : mm1->src[1];
                        float sm_scale = 1.0f, sm_max_bias = 0.0f;
                        std::memcpy(&sm_scale,    sm->op_params, sizeof(float));
                        std::memcpy(&sm_max_bias, (const char *) sm->op_params + 4, sizeof(float));
                        if (sm_max_bias == 0.0f && states && orig_q &&
                            states->type == GGML_TYPE_F32 && orig_q->type == GGML_TYPE_F32 && mm2->type == GGML_TYPE_F32 &&
                            ggml_is_contiguous(states) && ggml_is_contiguous(orig_q) && ggml_is_contiguous(mm2) &&
                            states->ne[3] == 1 && orig_q->ne[3] == 1 && mm2->ne[3] == 1 &&
                            states->ne[1] > 0 && states->ne[1] <= 32 &&
                            orig_q->ne[0] == states->ne[0] && orig_q->ne[1] > 0 &&
                            (orig_q->ne[2] == 1 || orig_q->ne[2] == states->ne[2]) &&
                            mm2->ne[0] == orig_q->ne[1] && mm2->ne[1] == states->ne[0] && mm2->ne[2] == states->ne[2] &&
                            agx_tensor_gpu_va(states) != 0 && agx_tensor_gpu_va(orig_q) != 0 && agx_tensor_gpu_va(mm2) != 0 &&
                            va_disjoint(states, mm2) && va_disjoint(orig_q, mm2)) {

                            const struct ggml_tensor * base_states = states;
                            const struct ggml_tensor * extra_t     = nullptr;
                            uint32_t extra_do_rms = 0u;
                            float    extra_eps    = 1e-6f;
                            int32_t  e_abs_cat    = -1;
                            int32_t  e_abs_rms    = -1;

                            if (states->op == GGML_OP_CONCAT && ggml_get_op_params_i32(states, 0) == 1 &&
                                (states->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                st->use_map.get(states) == 2u && states->src[0] && states->src[1]) {
                                const struct ggml_tensor * bs = states->src[0];
                                const struct ggml_tensor * ex = states->src[1];
                                if (bs->type == GGML_TYPE_F32 && ex->type == GGML_TYPE_F32 &&
                                    ggml_is_contiguous(bs) && ggml_is_contiguous(ex) &&
                                    bs->ne[0] == states->ne[0] && bs->ne[1] == states->ne[1] - 1 &&
                                    bs->ne[2] == states->ne[2] && bs->ne[3] == 1 &&
                                    ex->ne[0] == states->ne[0] && ex->ne[1] == 1 &&
                                    ex->ne[2] == states->ne[2] && ex->ne[3] == 1 &&
                                    va_disjoint(bs, mm2) && va_disjoint(ex, mm2)) {
                                    int32_t e_cat = st->exec_idx_map.get(states);
                                    agx_va_range cat_ins[2] = { tensor_va_range(bs), tensor_va_range(ex) };
                                    if (can_absorb_prev(e_cat, e, cat_ins, 2)) {
                                        e_abs_cat   = e_cat;
                                        base_states = bs;
                                        extra_t     = ex;
                                        const struct ggml_tensor * ex_r = strip_reshape(ex);
                                        if (ex_r && ex_r->op == GGML_OP_RMS_NORM &&
                                            (ex_r->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                            st->use_map.get(ex_r) == 1u &&
                                            ex_r->src[0] && ex_r->src[0]->type == GGML_TYPE_F32 &&
                                            ggml_is_contiguous(ex_r->src[0]) &&
                                            ggml_nelements(ex_r->src[0]) == ggml_nelements(ex) &&
                                            va_disjoint(ex_r->src[0], mm2)) {
                                            int32_t e_rms = st->exec_idx_map.get(ex_r);
                                            agx_va_range rms_in = tensor_va_range(ex_r->src[0]);
                                            if (can_absorb_prev(e_rms, e, &rms_in, 1)) {
                                                e_abs_rms    = e_rms;
                                                extra_t      = ex_r->src[0];
                                                extra_do_rms = 1u;
                                                std::memcpy(&extra_eps, ex_r->op_params, sizeof(float));
                                            }
                                        }
                                    }
                                }
                            } else if (states->ne[1] == 1 && states->op == GGML_OP_RMS_NORM &&
                                       (states->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                       st->use_map.get(states) == 2u &&
                                       states->src[0] && states->src[0]->type == GGML_TYPE_F32 &&
                                       ggml_is_contiguous(states->src[0]) &&
                                       va_disjoint(states->src[0], mm2)) {
                                int32_t e_rms = st->exec_idx_map.get(states);
                                agx_va_range rms_in = tensor_va_range(states->src[0]);
                                if (can_absorb_prev(e_rms, e, &rms_in, 1)) {
                                    e_abs_rms    = e_rms;
                                    base_states  = nullptr;
                                    extra_t      = states->src[0];
                                    extra_do_rms = 1u;
                                    std::memcpy(&extra_eps, states->op_params, sizeof(float));
                                }
                            }

                            if (e_abs_cat >= 0) { anchors[(size_t) e_abs_cat].cnt = 0u; consumed[(size_t) e_abs_cat] = 1u; }
                            if (e_abs_rms >= 0) { anchors[(size_t) e_abs_rms].cnt = 0u; consumed[(size_t) e_abs_rms] = 1u; }

                            mix_attn_push mpc{};
                            mpc.states_va    = base_states ? agx_tensor_gpu_va(base_states) : 0ull;
                            mpc.extra_va     = extra_t     ? agx_tensor_gpu_va(extra_t)     : 0ull;
                            mpc.queries_va   = agx_tensor_gpu_va(orig_q);
                            mpc.out_va       = agx_tensor_gpu_va(mm2);
                            mpc.n_embd       = (uint32_t) states->ne[0];
                            mpc.n_states     = (uint32_t) states->ne[1];
                            mpc.n_queries    = (uint32_t) orig_q->ne[1];
                            mpc.n_tokens     = (uint32_t) states->ne[2];
                            mpc.q_ne2        = (uint32_t) orig_q->ne[2];
                            mpc.extra_do_rms = extra_do_rms;
                            mpc.scale        = sm_scale;
                            mpc.extra_eps    = extra_eps;

                            int32_t fset[5];
                            for (size_t k = 0; k < off + 4u; ++k) fset[k] = (int32_t) (e + k);
                            agx_exec_item it{};
                            it.kind          = ITEM_MIX_ATTN;
                            it.gx            = mpc.n_queries;
                            it.gy            = mpc.n_tokens;
                            it.gz            = 1u;
                            it.nodes_covered = (uint32_t) (off + 4u) + (e_abs_cat >= 0 ? 1u : 0u) + (e_abs_rms >= 0 ? 1u : 0u);
                            if (base_states) it.add_read(tensor_va_range(base_states));
                            if (extra_t)     it.add_read(tensor_va_range(extra_t));
                            it.add_read(tensor_va_range(orig_q));
                            it.add_write(tensor_va_range(mm2));
                            it.set_push(mpc);
                            commit_single_item((int32_t) e, fset, (int) (off + 4u), it);
                            st->stats.fused_mix_hits++;
                            g_mix++;
                            continue;
                        }
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 9: MoE Expert Weighted Sum (`MUL(experts, weights) + (K-1)x ADD(view_k)`)
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 8)) && node->op == GGML_OP_MUL && node->type == GGML_TYPE_F32 &&
            (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && ggml_is_contiguous(node) && node->ne[3] == 1) {
            const struct ggml_tensor * exp_t = node->src[0];
            const struct ggml_tensor * w_t   = node->src[1];
            if (exp_t && w_t && exp_t->ne[0] == 1 && w_t->ne[0] > 1) {
                std::swap(exp_t, w_t);
            }
            const int64_t D = exp_t ? exp_t->ne[0] : 0;
            const int64_t K = exp_t ? exp_t->ne[1] : 0;
            const int64_t T = exp_t ? exp_t->ne[2] : 0;
            if (exp_t && w_t && D > 1 && K >= 2 && K <= 32 && T > 0 &&
                exp_t->type == GGML_TYPE_F32 && w_t->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(exp_t) && ggml_is_contiguous(w_t) &&
                exp_t->ne[3] == 1 && w_t->ne[0] == 1 && w_t->ne[1] == K && w_t->ne[2] == T && w_t->ne[3] == 1 &&
                node->ne[0] == D && node->ne[1] == K && node->ne[2] == T &&
                st->use_map.get(node) == (uint32_t) K &&
                e + (size_t) (K - 1) < n_exec) {
                auto is_exp_view = [&](const struct ggml_tensor * v, int64_t k_idx) -> bool {
                    return v && v->op == GGML_OP_VIEW && v->view_src == node &&
                           v->view_offs == (size_t) k_idx * node->nb[1] &&
                           v->ne[0] == D && v->ne[1] == T && v->ne[2] == 1 && v->ne[3] == 1 &&
                           v->nb[1] == node->nb[2] &&
                           st->use_map.get(v) == 1u && (v->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
                };
                bool match = true;
                struct ggml_tensor * prev_add = nullptr;
                for (int64_t k = 1; k < K; ++k) {
                    if (consumed[e + (size_t) k]) { match = false; break; }
                    struct ggml_tensor * add_k = cgraph->nodes[st->exec_nodes[e + (size_t) k]];
                    if (add_k->op != GGML_OP_ADD || add_k->type != GGML_TYPE_F32) { match = false; break; }
                    if (k == 1) {
                        if (!((is_exp_view(add_k->src[0], 0) && is_exp_view(add_k->src[1], 1)) ||
                              (is_exp_view(add_k->src[1], 0) && is_exp_view(add_k->src[0], 1)))) {
                            match = false; break;
                        }
                    } else {
                        if (!((chain_uses_exact(add_k->src[0], prev_add) && is_exp_view(add_k->src[1], k)) ||
                              (chain_uses_exact(add_k->src[1], prev_add) && is_exp_view(add_k->src[0], k)))) {
                            match = false; break;
                        }
                    }
                    prev_add = add_k;
                }
                if (match && prev_add && ggml_is_contiguous(prev_add) &&
                    prev_add->ne[0] == D && prev_add->ne[1] == T && prev_add->ne[2] == 1 && prev_add->ne[3] == 1 &&
                    agx_tensor_gpu_va(prev_add) != 0 &&
                    (va_disjoint(exp_t, prev_add) || (T == 1 && agx_tensor_gpu_va(prev_add) == agx_tensor_gpu_va(exp_t))) &&
                    va_disjoint(w_t, prev_add)) {
                    moe_wsum_push mpc{};
                    mpc.exp_va   = agx_tensor_gpu_va(exp_t);
                    mpc.w_va     = agx_tensor_gpu_va(w_t);
                    mpc.out_va   = agx_tensor_gpu_va(prev_add);
                    mpc.D        = (uint32_t) D;
                    mpc.n_exp    = (uint32_t) K;
                    mpc.n_tokens = (uint32_t) T;

                    int32_t fset[32];
                    for (int64_t k = 0; k < K; ++k) fset[k] = (int32_t) (e + (size_t) k);
                    agx_exec_item it{};
                    it.kind          = ITEM_MOE_WSUM;
                    it.gx            = (mpc.D + 255u) / 256u;
                    it.gy            = mpc.n_tokens;
                    it.gz            = 1u;
                    it.nodes_covered = (uint32_t) K;
                    it.add_read(tensor_va_range(exp_t));
                    it.add_read(tensor_va_range(w_t));
                    it.add_write(tensor_va_range(prev_add));
                    it.set_push(mpc);
                    commit_single_item((int32_t) e, fset, (int) K, it);
                    st->stats.fused_moe_wsum_hits++;
                    g_moe_wsum++;
                    continue;
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 10A (Wave 3 NEW): Full Static/Dynamic ShortConv + History Concat + State Store + Res + Scale [+ Post-Add]
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 9)) && node->op == GGML_OP_CONCAT && ggml_get_op_params_i32(node, 0) == 0 &&
            (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && node->ne[3] == 1 &&
            node->src[0] && node->src[1] && node->src[0]->type == GGML_TYPE_F32 &&
            node->src[0]->nb[0] == sizeof(float) && node->src[0]->ne[0] >= 1 && node->src[0]->ne[0] <= 8 &&
            node->src[1]->op == GGML_OP_TRANSPOSE && node->src[1]->src[0]) {
            const struct ggml_tensor * past   = node->src[0];
            const struct ggml_tensor * cur_3d = node->src[1]->src[0];
            const struct ggml_tensor * cur    = strip_reshape(cur_3d);
            const int64_t n_hist = past->ne[0];
            const int64_t n_conv = n_hist + 1;
            const int64_t n_ch   = past->ne[1];
            const int64_t n_s    = past->ne[2];
            const int64_t n_t    = node->ne[0] - n_hist;

            if (cur && cur->type == GGML_TYPE_F32 && ggml_is_contiguous(cur) &&
                n_t > 0 && n_ch > 0 && n_s > 0 &&
                cur_3d->ne[0] == n_ch && cur_3d->ne[1] == n_t && cur_3d->ne[2] == n_s && cur_3d->ne[3] == 1 &&
                ggml_nelements(cur) == n_ch * n_t * n_s &&
                (past->nb[1] & 3u) == 0 && (past->nb[2] & 3u) == 0) {
                struct ggml_tensor * cpy_n = nullptr;
                struct ggml_tensor * ssm_n = nullptr;
                const size_t w_end = std::min(n_exec, e + 48u);
                for (size_t j = e + 1u; j < w_end; ++j) {
                    if (consumed[j]) continue;
                    struct ggml_tensor * cand = cgraph->nodes[st->exec_nodes[j]];
                    if (!cpy_n && cand->op == GGML_OP_CPY && cand->type == GGML_TYPE_F32 &&
                        cand->src[0] && cand->src[0]->op == GGML_OP_VIEW && cand->src[0]->view_src == node &&
                        cand->src[0]->view_offs == (size_t) n_t * sizeof(float) &&
                        cand->src[0]->ne[0] == n_hist && cand->src[0]->ne[1] == n_ch &&
                        cand->src[0]->ne[2] == n_s && cand->src[0]->ne[3] == 1 &&
                        cand->ne[0] == n_hist * n_ch && cand->ne[1] == n_s &&
                        cand->ne[2] == 1 && cand->ne[3] == 1 &&
                        cand->nb[0] == sizeof(float) && (cand->nb[1] & 3u) == 0 &&
                        st->use_map.get(cand->src[0]) == 1u && st->use_map.get(cand) == 0u &&
                        (cand->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                        cpy_n = cand;
                    } else if (!ssm_n && cand->op == GGML_OP_SSM_CONV && cand->src[0] == node &&
                               cand->src[1] && cand->src[1]->ne[0] == n_conv && cand->src[1]->ne[1] == n_ch) {
                        ssm_n = cand;
                    }
                    if (cpy_n && ssm_n) break;
                }

                if (ssm_n) {
                    int32_t fset[40];
                    int fn = 0;
                    fset[fn++] = (int32_t) e;
                    if (cpy_n) fset[fn++] = st->exec_idx_map.get(cpy_n);
                    fset[fn++] = st->exec_idx_map.get(ssm_n);

                    const struct ggml_tensor * dyn_t = nullptr;
                    int64_t n_heads = 0;
                    struct ggml_tensor * cur_out = ssm_n;
                    bool dyn_ok = false;

                    if (struct ggml_tensor * first_add = next_single_use_compute(ssm_n, false)) {
                        if (first_add->op == GGML_OP_ADD &&
                            strip_reshape(first_add->src[0]) != cur && strip_reshape(first_add->src[1]) != cur) {
                            int32_t dyn_fset[32];
                            int dyn_fn = 0;
                            struct ggml_tensor * lag_out = ssm_n;
                            bool all_lags_ok = true;
                            for (int64_t lag = 0; lag < n_conv && all_lags_ok; ++lag) {
                                struct ggml_tensor * add_k = next_single_use_compute(lag_out, false);
                                if (!add_k || add_k->op != GGML_OP_ADD) { all_lags_ok = false; break; }
                                const struct ggml_tensor * other = (strip_reshape(add_k->src[0]) == lag_out)
                                    ? strip_reshape(add_k->src[1]) : strip_reshape(add_k->src[0]);
                                if (!other || other->op != GGML_OP_MUL || !chain_uses_exact(
                                        (strip_reshape(add_k->src[0]) == lag_out) ? add_k->src[1] : add_k->src[0], other)) {
                                    all_lags_ok = false; break;
                                }
                                const struct ggml_tensor * m0 = other->src[0];
                                const struct ggml_tensor * m1 = other->src[1];
                                const struct ggml_tensor * tap = nullptr;
                                const struct ggml_tensor * win_c = nullptr;
                                const struct ggml_tensor * win_leaf = nullptr;
                                if (m0 && m1) {
                                    const struct ggml_tensor * s0 = strip_reshape(m0);
                                    const struct ggml_tensor * s1 = strip_reshape(m1);
                                    if (s0 && s0->op == GGML_OP_CONT && m1->op == GGML_OP_VIEW) { win_c = s0; win_leaf = m0; tap = m1; }
                                    else if (s1 && s1->op == GGML_OP_CONT && m0->op == GGML_OP_VIEW) { win_c = s1; win_leaf = m1; tap = m0; }
                                }
                                if (!tap || !win_c || !chain_uses_exact(win_leaf, win_c) ||
                                    !win_c->src[0] || win_c->src[0]->op != GGML_OP_TRANSPOSE ||
                                    !win_c->src[0]->src[0] || win_c->src[0]->src[0]->op != GGML_OP_VIEW) {
                                    all_lags_ok = false; break;
                                }
                                const struct ggml_tensor * win_v = win_c->src[0]->src[0];
                                if (win_v->view_src != node ||
                                    win_v->view_offs != (size_t) (n_conv - 1 - lag) * node->nb[0] ||
                                    win_v->ne[0] != n_t || win_v->ne[1] != n_ch || win_v->ne[2] != n_s ||
                                    st->use_map.get(win_v) != 1u || st->use_map.get(win_c->src[0]) != 1u) {
                                    all_lags_ok = false; break;
                                }
                                // Note: ggml_view_impl short-circuits tap->view_src through RESHAPE_3D to the 2D dyn_root!
                                const struct ggml_tensor * dyn_root = strip_reshape(tap->view_src);
                                const int64_t tap_heads = tap->ne[1];
                                if (!dyn_root || dyn_root->type != GGML_TYPE_F32 ||
                                    !ggml_is_contiguous(dyn_root) ||
                                    tap->type != GGML_TYPE_F32 ||
                                    tap->ne[0] != 1 || tap_heads <= 0 ||
                                    tap->ne[2] != n_t * n_s || tap->ne[3] != 1 ||
                                    tap->nb[1] != (size_t) n_conv * sizeof(float) ||
                                    tap->nb[2] != (size_t) (n_conv * tap_heads) * sizeof(float) ||
                                    tap->view_offs != (size_t) lag * sizeof(float) ||
                                    ggml_nelements(dyn_root) != n_conv * tap_heads * n_t * n_s ||
                                    (n_ch % tap_heads) != 0) {
                                    all_lags_ok = false; break;
                                }
                                if (lag == 0) {
                                    dyn_t   = dyn_root;
                                    n_heads = tap_heads;
                                } else if (dyn_root != dyn_t || tap_heads != n_heads) {
                                    all_lags_ok = false; break;
                                }
                                dyn_fset[dyn_fn++] = st->exec_idx_map.get(win_c);
                                dyn_fset[dyn_fn++] = st->exec_idx_map.get(other);
                                dyn_fset[dyn_fn++] = st->exec_idx_map.get(add_k);
                                lag_out = add_k;
                            }
                            if (all_lags_ok && dyn_t) {
                                for (int k = 0; k < dyn_fn; ++k) fset[fn++] = dyn_fset[k];
                                cur_out = lag_out;
                                dyn_ok  = true;
                            }
                        }
                    }

                    const uint32_t expected_x_uses = 1u + (cpy_n ? 1u : 0u) + (dyn_ok ? (uint32_t) n_conv : 0u);
                    if (st->use_map.get(node) == expected_x_uses) {
                        struct ggml_tensor * add_res = next_single_use_compute(cur_out, false);
                        struct ggml_tensor * sc_res  = add_res ? next_single_use_compute(add_res, false) : nullptr;
                        if (add_res && add_res->op == GGML_OP_ADD &&
                            sc_res  && sc_res->op  == GGML_OP_SCALE &&
                            ((strip_reshape(add_res->src[0]) == cur && strip_reshape(add_res->src[1]) == cur_out) ||
                             (strip_reshape(add_res->src[1]) == cur && strip_reshape(add_res->src[0]) == cur_out)) &&
                            ggml_is_contiguous(sc_res) && ggml_nelements(sc_res) == n_ch * n_t * n_s) {
                            fset[fn++] = st->exec_idx_map.get(add_res);
                            fset[fn++] = st->exec_idx_map.get(sc_res);

                            struct ggml_tensor * final_dst = sc_res;
                            const struct ggml_tensor * post_add_t = nullptr;
                            if (struct ggml_tensor * pa = next_single_use_compute(sc_res, false)) {
                                if (pa->op == GGML_OP_ADD && pa->type == GGML_TYPE_F32 &&
                                    ggml_is_contiguous(pa) && ggml_are_same_shape(pa, sc_res)) {
                                    const struct ggml_tensor * other = (strip_reshape(pa->src[0]) == sc_res)
                                        ? pa->src[1] : pa->src[0];
                                    if (other && other->type == GGML_TYPE_F32 && ggml_is_contiguous(other) &&
                                        ggml_are_same_shape(other, sc_res) &&
                                        va_same_or_disjoint(pa, other)) {
                                        post_add_t = other;
                                        final_dst  = pa;
                                        fset[fn++] = st->exec_idx_map.get(pa);
                                    }
                                }
                            }

                            const struct ggml_tensor * w = ssm_n->src[1];
                            const bool dst_va_ok =
                                va_disjoint(past, final_dst) && va_disjoint(w, final_dst) &&
                                va_disjoint(dyn_t, final_dst) && va_disjoint(cpy_n, final_dst) &&
                                (n_t == 1 ? va_same_or_disjoint(cur, final_dst) : va_disjoint(cur, final_dst)) &&
                                va_same_or_disjoint(post_add_t, final_dst);
                            const bool cell_va_ok = (!cpy_n) || (
                                va_disjoint(cur, cpy_n) && va_disjoint(w, cpy_n) &&
                                va_disjoint(dyn_t, cpy_n) && va_disjoint(post_add_t, cpy_n) &&
                                (n_t == 1 || va_disjoint(past, cpy_n)));

                            if (dst_va_ok && cell_va_ok) {
                                agx_va_range all_ins[5];
                                int32_t      all_prods[5];
                                int n_ins = 0;
                                all_ins[n_ins] = tensor_va_range(past); all_prods[n_ins++] = tensor_prod_idx(past);
                                all_ins[n_ins] = tensor_va_range(cur);  all_prods[n_ins++] = tensor_prod_idx(cur);
                                all_ins[n_ins] = tensor_va_range(w);    all_prods[n_ins++] = tensor_prod_idx(w);
                                if (dyn_t)      { all_ins[n_ins] = tensor_va_range(dyn_t);      all_prods[n_ins++] = tensor_prod_idx(dyn_t); }
                                if (post_add_t) { all_ins[n_ins] = tensor_va_range(post_add_t); all_prods[n_ins++] = tensor_prod_idx(post_add_t); }

                                agx_va_range extra_ins[3];
                                int n_extra = 0;
                                extra_ins[n_extra++] = tensor_va_range(w);
                                if (dyn_t)      extra_ins[n_extra++] = tensor_va_range(dyn_t);
                                if (post_add_t) extra_ins[n_extra++] = tensor_va_range(post_add_t);

                                agx_va_range outs[2];
                                int n_outs = 0;
                                outs[n_outs++] = tensor_va_range(final_dst);
                                if (cpy_n) outs[n_outs++] = tensor_va_range(cpy_n);

                                const int32_t anc = find_safe_anchor(fset, fn, all_ins, all_prods, n_ins, extra_ins, n_extra, outs, n_outs);
                                if (anc >= 0) {
                                    full_shortconv_push pc{};
                                    pc.past_va     = agx_tensor_gpu_va(past);
                                    pc.cur_va      = agx_tensor_gpu_va(cur);
                                    pc.w_va        = agx_tensor_gpu_va(w);
                                    pc.dyn_va      = dyn_t      ? agx_tensor_gpu_va(dyn_t)      : 0ull;
                                    pc.cell_va     = cpy_n      ? agx_tensor_gpu_va(cpy_n)      : 0ull;
                                    pc.post_add_va = post_add_t ? agx_tensor_gpu_va(post_add_t) : 0ull;
                                    pc.dst_va      = agx_tensor_gpu_va(final_dst);
                                    pc.past_nb1    = (uint32_t) (past->nb[1] / 4u);
                                    pc.past_nb2    = (uint32_t) (past->nb[2] / 4u);
                                    pc.w_nb1       = (uint32_t) (w->nb[1] / 4u);
                                    pc.cell_nb1    = cpy_n ? (uint32_t) (cpy_n->nb[1] / 4u) : 0u;
                                    pc.n_conv      = (uint32_t) n_conv;
                                    pc.n_ch        = (uint32_t) n_ch;
                                    pc.n_t         = (uint32_t) n_t;
                                    pc.n_s         = (uint32_t) n_s;
                                    pc.n_heads     = (uint32_t) (dyn_ok ? n_heads : 1);
                                    pc.head_dim    = (uint32_t) (dyn_ok ? (n_ch / n_heads) : n_ch);
                                    read_scale_params(sc_res, &pc.res_scale, &pc.res_bias);

                                    agx_exec_item it{};
                                    it.kind          = ITEM_FULL_SHORTCONV;
                                    it.gx            = (pc.n_ch + 255u) / 256u;
                                    it.gy            = pc.n_t;
                                    it.gz            = pc.n_s;
                                    it.nodes_covered = (uint32_t) fn;
                                    for (int r = 0; r < n_ins; ++r)    it.add_read(all_ins[r]);
                                    for (int ow = 0; ow < n_outs; ++ow) it.add_write(outs[ow]);
                                    it.set_push(pc);
                                    commit_single_item(anc, fset, fn, it);
                                    st->stats.fused_full_sconv_hits++;
                                    g_full_sconv++;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 10B (Fallback when state store is SET_ROWS, e.g. DSpark n_rs_seq > 0):
        //   `SSM_CONV -> ADD(res, out) -> SCALE(s, b)`
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 9)) && node->op == GGML_OP_SSM_CONV && e + 2u < n_exec &&
            !consumed[e + 1u] && !consumed[e + 2u]) {
            struct ggml_tensor * add_n = cgraph->nodes[st->exec_nodes[e + 1u]];
            struct ggml_tensor * sc_n  = cgraph->nodes[st->exec_nodes[e + 2u]];
            if (add_n->op == GGML_OP_ADD && sc_n->op == GGML_OP_SCALE &&
                chain_uses_exact(sc_n->src[0], add_n)) {
                const struct ggml_tensor * res = nullptr;
                if (chain_uses_exact(add_n->src[0], node))      res = add_n->src[1];
                else if (chain_uses_exact(add_n->src[1], node)) res = add_n->src[0];
                if (res && res->type == GGML_TYPE_F32 && ggml_is_contiguous(res) &&
                    ggml_is_contiguous(sc_n) &&
                    ggml_nelements(res) == ggml_nelements(node) &&
                    ggml_nelements(sc_n) == ggml_nelements(node) &&
                    agx_tensor_gpu_va(res) != 0 && agx_tensor_gpu_va(sc_n) != 0 &&
                    va_disjoint(node->src[0], sc_n) && va_disjoint(node->src[1], sc_n)) {
                    const auto * s0 = node->src[0];
                    const auto * s1 = node->src[1];
                    ssm_conv_push pc{};
                    pc.src0_va = agx_tensor_gpu_va(s0);
                    pc.src1_va = agx_tensor_gpu_va(s1);
                    pc.dst_va  = agx_tensor_gpu_va(sc_n);
                    pc.res_va  = agx_tensor_gpu_va(res);
                    pc.nb01    = (uint32_t) (s0->nb[1] / 4);
                    pc.nb02    = (uint32_t) (s0->nb[2] / 4);
                    pc.nb11    = (uint32_t) (s1->nb[1] / 4);
                    pc.d_nb1   = (uint32_t) (node->nb[1] / 4);
                    pc.d_nb2   = (uint32_t) (node->nb[2] / 4);
                    pc.nc      = (uint32_t) s1->ne[0];
                    pc.nr      = (uint32_t) s0->ne[1];
                    pc.n_t     = (uint32_t) node->ne[1];
                    pc.n_s     = (uint32_t) node->ne[2];
                    read_scale_params(sc_n, &pc.res_scale, &pc.res_bias);

                    int32_t fset[3] = { (int32_t) e, (int32_t) (e + 1u), (int32_t) (e + 2u) };
                    agx_exec_item it{};
                    it.kind          = ITEM_SSM_CONV;
                    it.gx            = (pc.nr + 31u) / 32u;
                    it.gy            = (pc.n_t + 7u) / 8u;
                    it.gz            = pc.n_s;
                    it.nodes_covered = 3u;
                    it.add_read(tensor_va_range(s0));
                    it.add_read(tensor_va_range(s1));
                    it.add_read(tensor_va_range(res));
                    it.add_write(tensor_va_range(sc_n));
                    it.set_push(pc);
                    commit_single_item((int32_t) e, fset, 3, it);
                    st->stats.fused_ssm_res_hits++;
                    g_ssm_res++;
                    continue;
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 11 (Fallback): MoE Router Weights Norm (`SUM_ROWS -> SCALE_BIAS(1, eps) -> DIV [-> SCALE]`)
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 10)) && node->op == GGML_OP_SUM_ROWS && e + 2u < n_exec &&
            !consumed[e + 1u] && !consumed[e + 2u]) {
            struct ggml_tensor * sb_n  = cgraph->nodes[st->exec_nodes[e + 1u]];
            struct ggml_tensor * div_n = cgraph->nodes[st->exec_nodes[e + 2u]];
            if (sb_n->op == GGML_OP_SCALE && div_n->op == GGML_OP_DIV &&
                chain_uses_exact(sb_n->src[0], node) &&
                chain_uses_exact(div_n->src[1], sb_n) &&
                div_n->src[0] == node->src[0]) {
                float s1 = 0.0f, b1 = 0.0f;
                read_scale_params(sb_n, &s1, &b1);
                const struct ggml_tensor * a = node->src[0];
                if (s1 == 1.0f && b1 >= 0.0f && a && a->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(a) && ggml_are_same_shape(a, div_n) &&
                    ggml_is_contiguous(div_n) && agx_tensor_gpu_va(div_n) != 0) {
                    struct ggml_tensor * final_dst = div_n;
                    float w_scale = 1.0f;
                    size_t adv = 2u;
                    if (e + 3u < n_exec && !consumed[e + 3u]) {
                        struct ggml_tensor * sc_n = cgraph->nodes[st->exec_nodes[e + 3u]];
                        if (sc_n->op == GGML_OP_SCALE && chain_uses_exact(sc_n->src[0], div_n) &&
                            ggml_is_contiguous(sc_n) && ggml_nelements(sc_n) == ggml_nelements(div_n) &&
                            agx_tensor_gpu_va(sc_n) != 0) {
                            float s3 = 0.0f, b3 = 0.0f;
                            read_scale_params(sc_n, &s3, &b3);
                            if (b3 == 0.0f) {
                                final_dst = sc_n;
                                w_scale   = s3;
                                adv       = 3u;
                            }
                        }
                    }
                    row_reduce_push pc{};
                    pc.a_va    = agx_tensor_gpu_va(a);
                    pc.d_va    = agx_tensor_gpu_va(final_dst);
                    pc.ne0     = (uint32_t) a->ne[0]; pc.ne1 = (uint32_t) a->ne[1];
                    pc.ne2     = (uint32_t) a->ne[2]; pc.ne3 = (uint32_t) a->ne[3];
                    pc.nb01    = (uint32_t) (a->nb[1] / 4); pc.nb02 = (uint32_t) (a->nb[2] / 4); pc.nb03 = (uint32_t) (a->nb[3] / 4);
                    pc.nb1     = pc.nb01; pc.nb2 = pc.nb02; pc.nb3 = pc.nb03;
                    pc.op_kind = 7u;
                    pc.eps     = b1;
                    pc.scale   = w_scale;
                    pc.nb00    = 1u;

                    int32_t fset[4];
                    for (size_t k = 0; k <= adv; ++k) fset[k] = (int32_t) (e + k);
                    agx_exec_item it{};
                    it.kind          = ITEM_ROW_REDUCE;
                    it.gx            = pc.ne1 * pc.ne2 * pc.ne3;
                    it.gy            = 1u;
                    it.gz            = 1u;
                    it.nodes_covered = (uint32_t) (adv + 1u);
                    it.add_read(tensor_va_range(a));
                    it.add_write(tensor_va_range(final_dst));
                    it.set_push(pc);
                    commit_single_item((int32_t) e, fset, (int) (adv + 1u), it);
                    st->stats.fused_moe_norm_hits++;
                    g_moe_norm++;
                    continue;
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 12 (Wave 3 NEW): Direct Relative-Position Bias Gather
        //   PAD(rel_bias, 0,1,0,0) -> MUL_MAT(Rcur) -> SUB(sinks) -> CONT(PERM 1,0,2,3) -> GET_ROWS(rel_pos) -> CONT(PERM 2,0,1,3)
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 12)) && node->op == GGML_OP_PAD && node->src[0] &&
            ggml_get_op_params_i32(node, 0) == 0 && ggml_get_op_params_i32(node, 2) == 0 &&
            ggml_get_op_params_i32(node, 4) == 0 && ggml_get_op_params_i32(node, 6) == 0 &&
            node->ne[0] == node->src[0]->ne[0] && node->ne[1] == node->src[0]->ne[1] + 1 &&
            node->ne[2] == 1 && node->ne[3] == 1 && node->ne[0] > 0 && node->ne[0] <= 64) {
            const struct ggml_tensor * orig_bias = node->src[0];
            struct ggml_tensor * mm  = next_single_use_compute(node, false);
            struct ggml_tensor * sub = (mm && mm->op == GGML_OP_MUL_MAT && mm->src[0] == node) ? next_single_use_compute(mm, false) : nullptr;
            if (mm && sub && sub->op == GGML_OP_SUB && sub->src[0] == mm && sub->src[1]) {
                struct ggml_tensor * p1 = (st->use_map.get(sub) == 1u && (sub->flags & GGML_TENSOR_FLAG_OUTPUT) == 0)
                                          ? st->first_use.get(sub) : nullptr;
                struct ggml_tensor * c1 = (p1 && p1->op == GGML_OP_PERMUTE && st->use_map.get(p1) == 1u &&
                                           (p1->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                           p1->nb[0] == sub->nb[1] && p1->nb[1] == sub->nb[0] && p1->nb[2] == sub->nb[2])
                                          ? st->first_use.get(p1) : nullptr;
                struct ggml_tensor * gr = (c1 && c1->op == GGML_OP_CONT) ? next_single_use_compute(c1, false) : nullptr;
                if (c1 && gr && gr->op == GGML_OP_GET_ROWS && gr->src[0] == c1 && gr->src[1]) {
                    const struct ggml_tensor * r4 = (st->use_map.get(gr) == 1u && (gr->flags & GGML_TENSOR_FLAG_OUTPUT) == 0)
                                                    ? st->first_use.get(gr) : nullptr;
                    const struct ggml_tensor * p2_in = (r4 && r4->op == GGML_OP_RESHAPE &&
                                                        st->use_map.get(r4) == 1u &&
                                                        (r4->flags & GGML_TENSOR_FLAG_OUTPUT) == 0)
                                                       ? r4 : (const struct ggml_tensor *) gr;
                    struct ggml_tensor * p2 = (p2_in && st->use_map.get(p2_in) == 1u) ? st->first_use.get(p2_in) : nullptr;
                    struct ggml_tensor * c2 = (p2 && p2->op == GGML_OP_PERMUTE && st->use_map.get(p2) == 1u &&
                                               (p2->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                               p2->nb[0] == p2_in->nb[1] && p2->nb[1] == p2_in->nb[2] &&
                                               p2->nb[2] == p2_in->nb[0] && p2->nb[3] == p2_in->nb[3])
                                              ? st->first_use.get(p2) : nullptr;
                    if (c2 && c2->op == GGML_OP_CONT && ggml_is_contiguous(c2)) {
                        const struct ggml_tensor * rcur    = mm->src[1];
                        const struct ggml_tensor * sinks   = strip_reshape(sub->src[1]);
                        const struct ggml_tensor * rel_pos = gr->src[1];
                        const int64_t n_rel    = orig_bias->ne[0];
                        const int64_t n_extent = orig_bias->ne[1];
                        const int64_t n_head   = rcur ? rcur->ne[1] : 0;
                        const int64_t n_tokens = rcur ? rcur->ne[2] : 0;
                        const int64_t n_kv     = rel_pos->ne[0];
                        const int64_t n_t      = c2->ne[1];
                        const int64_t n_stream = c2->ne[3];
                        uint32_t rb = 0u;
                        const uint32_t bwtype = agx_inkling2_tensor_wtype(orig_bias, &rb);
                        const bool bias_dtype_ok =
                            orig_bias->type == GGML_TYPE_F32 || orig_bias->type == GGML_TYPE_F16 ||
                            orig_bias->type == GGML_TYPE_BF16 ||
                            (orig_bias->type == GGML_TYPE_Q8_0 && bwtype == 3u);
                        if (rcur && sinks && bias_dtype_ok &&
                            rcur->type == GGML_TYPE_F32 && sinks->type == GGML_TYPE_F32 &&
                            rel_pos->type == GGML_TYPE_I32 && c2->type == GGML_TYPE_F32 &&
                            ggml_is_contiguous(orig_bias) && ggml_is_contiguous(rcur) && ggml_is_contiguous(sinks) &&
                            rcur->ne[0] == n_rel && rcur->ne[3] == 1 &&
                            sub->src[1]->ne[0] == 1 && sub->src[1]->ne[1] == n_head && sub->src[1]->ne[2] == n_tokens &&
                            ggml_nelements(sinks) == n_head * n_tokens &&
                            rel_pos->ne[1] == n_tokens && rel_pos->ne[2] == 1 && rel_pos->ne[3] == 1 &&
                            c2->ne[0] == n_kv && c2->ne[2] == n_head && n_t * n_stream == n_tokens &&
                            (n_kv <= n_extent + 1 || n_tokens <= 8) &&
                            va_disjoint(orig_bias, c2) && va_disjoint(rcur, c2) &&
                            va_disjoint(sinks, c2) && va_disjoint(rel_pos, c2)) {
                            int32_t fset[6] = {
                                (int32_t) e,
                                st->exec_idx_map.get(mm),
                                st->exec_idx_map.get(sub),
                                st->exec_idx_map.get(c1),
                                st->exec_idx_map.get(gr),
                                st->exec_idx_map.get(c2),
                            };
                            agx_va_range all_ins[4] = {
                                tensor_va_range(orig_bias), tensor_va_range(rcur),
                                tensor_va_range(sinks),     tensor_va_range(rel_pos),
                            };
                            int32_t all_prods[4] = {
                                tensor_prod_idx(orig_bias), tensor_prod_idx(rcur),
                                tensor_prod_idx(sinks),     tensor_prod_idx(rel_pos),
                            };
                            agx_va_range extra_ins[3] = {
                                tensor_va_range(rcur), tensor_va_range(sinks), tensor_va_range(rel_pos),
                            };
                            agx_va_range outs[1] = { tensor_va_range(c2) };
                            const int32_t anc = find_safe_anchor(fset, 6, all_ins, all_prods, 4, extra_ins, 3, outs, 1);
                            if (anc >= 0) {
                                rel_bias_gather_push pc{};
                                pc.rcur_va    = agx_tensor_gpu_va(rcur);
                                pc.bias_va    = agx_tensor_gpu_va(orig_bias);
                                pc.sinks_va   = agx_tensor_gpu_va(sinks);
                                pc.rel_pos_va = agx_tensor_gpu_va(rel_pos);
                                pc.dst_va     = agx_tensor_gpu_va(c2);
                                pc.n_rel      = (uint32_t) n_rel;
                                pc.n_extent   = (uint32_t) n_extent;
                                pc.n_kv       = (uint32_t) n_kv;
                                pc.n_head     = (uint32_t) n_head;
                                pc.n_t        = (uint32_t) n_t;
                                pc.n_stream   = (uint32_t) n_stream;
                                pc.bias_wtype = bwtype;
                                pc.bias_nb1   = (uint32_t) orig_bias->nb[1];
                                pc.rpos_nb0   = (uint32_t) rel_pos->nb[0];
                                pc.rpos_nb1   = (uint32_t) rel_pos->nb[1];

                                agx_exec_item it{};
                                it.kind          = ITEM_REL_BIAS;
                                it.gx            = (pc.n_kv + 63u) / 64u;
                                it.gy            = pc.n_head;
                                it.gz            = (uint32_t) n_tokens;
                                it.nodes_covered = 6u;
                                for (int r = 0; r < 4; ++r) it.add_read(all_ins[r]);
                                it.add_write(outs[0]);
                                it.set_push(pc);
                                commit_single_item(anc, fset, 6, it);
                                st->stats.fused_rel_bias_hits++;
                                g_rel_bias++;
                                continue;
                            }
                        }
                    }
                }
            }
        }

        // Default: emit plain single-node generic item at anchor `e`
        {
            agx_exec_item it{};
            it.kind              = ITEM_GENERIC_NODE;
            it.node              = node;
            it.nodes_covered     = 1u;
            it.uses_moe_scratch  = (node->op == GGML_OP_MUL_MAT_ID);
            it.uses_attn_scratch = (node->op == GGML_OP_FLASH_ATTN_EXT);
            for (uint8_t r = 0; r < raw_vas[e].nr; ++r) it.add_read(raw_vas[e].r[r]);
            it.add_write(raw_vas[e].w);
            const uint32_t idx = (uint32_t) st->items.size();
            st->items.push_back(it);
            anchors[e] = { idx, 1u };
        }
    }

    // Compact items in original anchor program order
    st->ordered_items.clear();
    st->ordered_items.reserve(st->items.size());
    for (size_t e = 0; e < n_exec; ++e) {
        const auto & sl = anchors[e];
        for (uint32_t k = 0; k < sl.cnt; ++k) {
            st->ordered_items.push_back(st->items[sl.beg + k]);
        }
    }
    const size_t n_items = st->ordered_items.size();

    // ------------------------------------------------------------------------
    // Phase B: Exact VA-Dependency Wave Scheduler (RAW + WAR + WAW + Scratch!)
    //   Preserves 100% of ggml_gallocr VA reuse while batching independent ops
    //   into waves -> 1 STRONG_USC barrier per wave instead of per node!
    //   Toggleable via GGML_AGX_NO_WAVE_SCHED=1 for instant A/B verification.
    // ------------------------------------------------------------------------
    std::vector<uint32_t> sched_order(n_items);
    uint32_t num_waves = 0u;

    if (no_wave_sched || force_barrier || n_items <= 2u) {
        for (size_t u = 0; u < n_items; ++u) sched_order[u] = (uint32_t) u;
        num_waves = (uint32_t) n_items;
    } else {
        // 1. Build sorted disjoint union of all written VA ranges so read-only weights are skipped in O(log W)
        std::vector<agx_va_range> all_w;
        all_w.reserve(n_items + 16u);
        for (size_t u = 0; u < n_items; ++u) {
            const auto & it = st->ordered_items[u];
            for (uint8_t w = 0; w < it.n_writes; ++w) {
                all_w.push_back(it.writes[w]);
            }
        }
        std::sort(all_w.begin(), all_w.end(), [](const agx_va_range & a, const agx_va_range & b) {
            return a.beg < b.beg || (a.beg == b.beg && a.end < b.end);
        });
        std::vector<agx_va_range> w_union;
        w_union.reserve(64u);
        for (const auto & r : all_w) {
            if (!w_union.empty() && r.beg <= w_union.back().end) {
                if (r.end > w_union.back().end) w_union.back().end = r.end;
            } else {
                w_union.push_back(r);
            }
        }
        auto overlaps_any_write = [&](agx_va_range q) -> bool {
            if (q.beg >= q.end || w_union.empty()) return false;
            size_t lo = 0, hi = w_union.size();
            while (lo < hi) {
                size_t mid = lo + ((hi - lo) >> 1u);
                if (w_union[mid].end <= q.beg) lo = mid + 1u;
                else hi = mid;
            }
            return lo < w_union.size() && w_union[lo].beg < q.end;
        };

        // 2. Exact ASAP level computation over RAW + WAR + WAW + internal-scratch DAG
        std::vector<uint32_t> item_lvl(n_items, 0u);
        std::vector<agx_interval_lvl> active_writes;
        std::vector<agx_interval_lvl> active_reads;
        active_writes.reserve(128u);
        active_reads.reserve(128u);
        int32_t last_moe_lvl  = -1;
        int32_t last_attn_lvl = -1;
        uint32_t max_lvl = 0u;

        for (size_t u = 0; u < n_items; ++u) {
            const auto & it = st->ordered_items[u];
            uint32_t lv = 0u;
            if (it.uses_moe_scratch  && last_moe_lvl  >= 0) lv = std::max(lv, (uint32_t) last_moe_lvl  + 1u);
            if (it.uses_attn_scratch && last_attn_lvl >= 0) lv = std::max(lv, (uint32_t) last_attn_lvl + 1u);

            // RAW: any dynamic read overlapping an active write
            for (uint8_t r = 0; r < it.n_reads; ++r) {
                if (!overlaps_any_write(it.reads[r])) continue;
                bool hit = false;
                uint32_t pl = query_interval_max_lvl(active_writes, it.reads[r], &hit);
                if (hit) lv = std::max(lv, pl + 1u);
            }
            // WAW & WAR: any write overlapping an active write or active read
            for (uint8_t w = 0; w < it.n_writes; ++w) {
                bool hit = false;
                uint32_t pl = query_interval_max_lvl(active_writes, it.writes[w], &hit);
                if (hit) lv = std::max(lv, pl + 1u);
                uint32_t rl = query_interval_max_lvl(active_reads, it.writes[w], &hit);
                if (hit) lv = std::max(lv, rl + 1u);
            }

            item_lvl[u] = lv;
            if (lv > max_lvl) max_lvl = lv;

            if (it.uses_moe_scratch)  last_moe_lvl  = (int32_t) lv;
            if (it.uses_attn_scratch) last_attn_lvl = (int32_t) lv;

            for (uint8_t r = 0; r < it.n_reads; ++r) {
                if (!overlaps_any_write(it.reads[r])) continue;
                insert_interval_lvl(active_reads, it.reads[r], lv);
            }
            for (uint8_t w = 0; w < it.n_writes; ++w) {
                insert_interval_lvl(active_writes, it.writes[w], lv);
                prune_contained_intervals(active_reads, it.writes[w], lv);
            }
        }

        // 3. Stable counting sort into wave order
        num_waves = max_lvl + 1u;
        std::vector<uint32_t> bucket_off(num_waves + 1u, 0u);
        for (size_t u = 0; u < n_items; ++u) bucket_off[item_lvl[u] + 1u]++;
        for (uint32_t w = 1u; w <= num_waves; ++w) bucket_off[w] += bucket_off[w - 1u];
        for (size_t u = 0; u < n_items; ++u) {
            sched_order[bucket_off[item_lvl[u]]++] = (uint32_t) u;
        }
        st->stats.wave_sched_waves += num_waves;
    }

    // ------------------------------------------------------------------------
    // Phase C: Single-CDM-Stream Dispatch Loop with Live Hazard Safety Net
    // ------------------------------------------------------------------------
    std::vector<agx_va_range> dirty_vas;
    std::vector<agx_va_range> read_vas;
    dirty_vas.reserve(64);
    read_vas.reserve(128);
    bool has_dispatched    = false;
    bool used_moe_scratch  = false;
    bool used_attn_scratch = false;

    auto ensure_item_ready = [&](const agx_exec_item & it) {
        bool hazard = force_barrier;
        if (!hazard && has_dispatched) {
            if ((it.uses_moe_scratch && used_moe_scratch) ||
                (it.uses_attn_scratch && used_attn_scratch)) {
                hazard = true;
            }
            for (uint8_t r = 0; !hazard && r < it.n_reads; ++r) {
                if (ranges_overlap(dirty_vas, it.reads[r])) hazard = true;
            }
            for (uint8_t w = 0; !hazard && w < it.n_writes; ++w) {
                if (ranges_overlap(dirty_vas, it.writes[w]) || ranges_overlap(read_vas, it.writes[w])) {
                    hazard = true;
                }
            }
        }
        if (hazard && has_dispatched) {
            agx_cdm_barrier_usc(stream);
            dirty_vas.clear();
            read_vas.clear();
            used_moe_scratch  = false;
            used_attn_scratch = false;
        } else if (has_dispatched) {
            st->stats.barriers_elided++;
        }
    };

    auto record_item_dispatch = [&](const agx_exec_item & it) {
        has_dispatched = true;
        for (uint8_t r = 0; r < it.n_reads; ++r)  read_vas.push_back(it.reads[r]);
        for (uint8_t w = 0; w < it.n_writes; ++w) dirty_vas.push_back(it.writes[w]);
        if (it.uses_moe_scratch)  used_moe_scratch  = true;
        if (it.uses_attn_scratch) used_attn_scratch = true;
    };

    for (size_t idx = 0; idx < n_items; ++idx) {
        const agx_exec_item & it = st->ordered_items[sched_order[idx]];
        ensure_item_ready(it);

        bool ok = false;
        switch (it.kind) {
            case ITEM_PREP_STORE:
                ok = agx_cdm_dispatch(stream, st->k_prep_norm_fwht_store, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_GLU:
                ok = agx_cdm_dispatch(stream, st->k_glu, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_SOFTCAP: {
                auto * eng = agx_graph_exec_get_inkling2(ctx);
                ok = eng && agx_inkling2_dispatch_softcap_argmax(eng, stream, it.va0, it.va1, 0ull,
                                                                 it.u0, it.u0, it.u1, it.f0, false, AGX_BARRIER_NONE);
            } break;

            case ITEM_CTRL_SRAM: {
                auto * eng = agx_graph_exec_get_inkling2(ctx);
                ok = eng && agx_inkling2_dispatch_ctrl_gates(eng, stream, &it.ctrl_desc, 1u,
                                                             it.u0, it.u1, it.u2, it.f0, it.f1, AGX_BARRIER_NONE);
            } break;

            case ITEM_CTRL_NORM_CAT:
                ok = agx_cdm_dispatch(stream, st->k_ctrl_norm_cat, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_CTRL_MIX:
                ok = agx_cdm_dispatch(stream, st->k_ctrl_mix, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_ROW_REDUCE:
                ok = agx_cdm_dispatch(stream, st->k_row_reduce, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_BIN_OP:
                ok = agx_cdm_dispatch(stream, st->k_bin_op, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_MIX_ATTN:
                ok = agx_cdm_dispatch(stream, st->k_mix_attn, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_MOE_WSUM:
                ok = agx_cdm_dispatch(stream, st->k_moe_wsum, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_SSM_CONV:
                ok = agx_cdm_dispatch(stream, st->k_ssm_conv, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_FULL_SHORTCONV:
                ok = agx_cdm_dispatch(stream, st->k_full_shortconv, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_ROUTER_TOPK:
                ok = agx_cdm_dispatch(stream, st->k_router_topk_norm, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_REL_BIAS:
                ok = agx_cdm_dispatch(stream, st->k_rel_bias_gather, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_GENERIC_NODE: {
                struct ggml_tensor * node = it.node;
                switch (node->op) {
                    case GGML_OP_ADD:
                    case GGML_OP_SUB:
                    case GGML_OP_MUL:
                    case GGML_OP_DIV: {
                        const auto * a = node->src[0];
                        const auto * b = node->src[1];
                        bin_op_push pc{};
                        pc.a_va = agx_tensor_gpu_va(a);
                        pc.b_va = agx_tensor_gpu_va(b);
                        pc.d_va = agx_tensor_gpu_va(node);
                        pc.ne0  = (uint32_t) node->ne[0]; pc.ne1  = (uint32_t) node->ne[1];
                        pc.ne2  = (uint32_t) node->ne[2]; pc.ne3  = (uint32_t) node->ne[3];
                        pc.ne00 = (uint32_t) a->ne[0];    pc.ne01 = (uint32_t) a->ne[1];
                        pc.ne02 = (uint32_t) a->ne[2];    pc.ne03 = (uint32_t) a->ne[3];
                        pc.ne10 = (uint32_t) b->ne[0];    pc.ne11 = (uint32_t) b->ne[1];
                        pc.ne12 = (uint32_t) b->ne[2];    pc.ne13 = (uint32_t) b->ne[3];
                        pc.nb00 = (uint32_t) (a->nb[0] / 4); pc.nb01 = (uint32_t) (a->nb[1] / 4);
                        pc.nb02 = (uint32_t) (a->nb[2] / 4); pc.nb03 = (uint32_t) (a->nb[3] / 4);
                        pc.nb10 = (uint32_t) (b->nb[0] / 4); pc.nb11 = (uint32_t) (b->nb[1] / 4);
                        pc.nb12 = (uint32_t) (b->nb[2] / 4); pc.nb13 = (uint32_t) (b->nb[3] / 4);
                        pc.nb0  = (uint32_t) (node->nb[0] / 4); pc.nb1 = (uint32_t) (node->nb[1] / 4);
                        pc.nb2  = (uint32_t) (node->nb[2] / 4); pc.nb3 = (uint32_t) (node->nb[3] / 4);
                        pc.op_kind = (node->op == GGML_OP_ADD) ? 0u :
                                     (node->op == GGML_OP_SUB) ? 1u :
                                     (node->op == GGML_OP_MUL) ? 2u : 3u;
                        pc.total = (uint32_t) ggml_nelements(node);
                        ok = agx_cdm_dispatch(stream, st->k_bin_op, (pc.total + 255u) / 256u, 1, 1,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_SCALE:
                    case GGML_OP_CLAMP:
                    case GGML_OP_SQR:
                    case GGML_OP_SQRT:
                    case GGML_OP_LOG:
                    case GGML_OP_SIN:
                    case GGML_OP_COS:
                    case GGML_OP_FILL:
                    case GGML_OP_ARANGE:
                    case GGML_OP_UNARY: {
                        const auto * a = node->src[0];
                        unary_op_push pc{};
                        pc.a_va = a ? agx_tensor_gpu_va(a) : 0ull;
                        pc.d_va = agx_tensor_gpu_va(node);
                        pc.ne0  = (uint32_t) node->ne[0]; pc.ne1 = (uint32_t) node->ne[1];
                        pc.ne2  = (uint32_t) node->ne[2]; pc.ne3 = (uint32_t) node->ne[3];
                        if (a) {
                            pc.nb00 = (uint32_t) (a->nb[0] / 4); pc.nb01 = (uint32_t) (a->nb[1] / 4);
                            pc.nb02 = (uint32_t) (a->nb[2] / 4); pc.nb03 = (uint32_t) (a->nb[3] / 4);
                        }
                        pc.nb0 = (uint32_t) (node->nb[0] / 4); pc.nb1 = (uint32_t) (node->nb[1] / 4);
                        pc.nb2 = (uint32_t) (node->nb[2] / 4); pc.nb3 = (uint32_t) (node->nb[3] / 4);
                        pc.total = (uint32_t) ggml_nelements(node);

                        if (node->op == GGML_OP_SCALE) {
                            pc.op_kind = 0u;
                            std::memcpy(&pc.p0, node->op_params, sizeof(float));
                            std::memcpy(&pc.p1, (const char *) node->op_params + 4, sizeof(float));
                        } else if (node->op == GGML_OP_CLAMP) {
                            pc.op_kind = 1u;
                            std::memcpy(&pc.p0, node->op_params, sizeof(float));
                            std::memcpy(&pc.p1, (const char *) node->op_params + 4, sizeof(float));
                        } else if (node->op == GGML_OP_SQR)  { pc.op_kind = 2u; }
                        else if   (node->op == GGML_OP_SQRT) { pc.op_kind = 3u; }
                        else if   (node->op == GGML_OP_LOG)  { pc.op_kind = 4u; }
                        else if   (node->op == GGML_OP_SIN)  { pc.op_kind = 5u; }
                        else if   (node->op == GGML_OP_COS)  { pc.op_kind = 6u; }
                        else if   (node->op == GGML_OP_FILL) {
                            pc.op_kind = 7u;
                            std::memcpy(&pc.p0, node->op_params, sizeof(float));
                        } else if (node->op == GGML_OP_ARANGE) {
                            pc.op_kind = 8u;
                            std::memcpy(&pc.p0, node->op_params, sizeof(float));
                            std::memcpy(&pc.p1, (const char *) node->op_params + 8, sizeof(float));
                        } else {
                            pc.op_kind = map_unary_op_code(ggml_get_unary_op(node));
                        }
                        ok = agx_cdm_dispatch(stream, st->k_unary_op, (pc.total + 255u) / 256u, 1, 1,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_CPY:
                    case GGML_OP_CONT:
                    case GGML_OP_DUP: {
                        const auto * s = node->src[0];
                        cpy_cast_push pc{};
                        pc.src_va = agx_tensor_gpu_va(s);
                        pc.dst_va = agx_tensor_gpu_va(node);
                        pc.ne0    = (uint32_t) s->ne[0]; pc.ne1 = (uint32_t) s->ne[1];
                        pc.ne2    = (uint32_t) s->ne[2]; pc.ne3 = (uint32_t) s->ne[3];
                        pc.d_ne0  = (uint32_t) node->ne[0]; pc.d_ne1 = (uint32_t) node->ne[1];
                        pc.d_ne2  = (uint32_t) node->ne[2]; pc.d_ne3 = (uint32_t) node->ne[3];
                        pc.s_nb0  = (uint32_t) s->nb[0]; pc.s_nb1 = (uint32_t) s->nb[1];
                        pc.s_nb2  = (uint32_t) s->nb[2]; pc.s_nb3 = (uint32_t) s->nb[3];
                        pc.d_nb0  = (uint32_t) node->nb[0]; pc.d_nb1 = (uint32_t) node->nb[1];
                        pc.d_nb2  = (uint32_t) node->nb[2]; pc.d_nb3 = (uint32_t) node->nb[3];
                        cpy_type_code(s->type,    &pc.src_type);
                        cpy_type_code(node->type, &pc.dst_type);
                        pc.total = (uint32_t) ggml_nelements(s);
                        pc.d_is_linear = ggml_is_contiguous(node) ? 1u : 0u;
                        ok = agx_cdm_dispatch(stream, st->k_cpy_cast, (pc.total + 255u) / 256u, 1, 1,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_RMS_NORM:
                    case GGML_OP_NORM:
                    case GGML_OP_L2_NORM:
                    case GGML_OP_SUM_ROWS:
                    case GGML_OP_MEAN:
                    case GGML_OP_CUMSUM:
                    case GGML_OP_ARGMAX: {
                        const auto * a = node->src[0];
                        row_reduce_push pc{};
                        pc.a_va = agx_tensor_gpu_va(a);
                        pc.d_va = agx_tensor_gpu_va(node);
                        pc.ne0  = (uint32_t) a->ne[0]; pc.ne1 = (uint32_t) a->ne[1];
                        pc.ne2  = (uint32_t) a->ne[2]; pc.ne3 = (uint32_t) a->ne[3];
                        pc.nb01 = (uint32_t) (a->nb[1] / 4); pc.nb02 = (uint32_t) (a->nb[2] / 4); pc.nb03 = (uint32_t) (a->nb[3] / 4);
                        if (node->op == GGML_OP_ARGMAX) {
                            pc.nb1 = (uint32_t) (node->nb[0] / 4);
                            pc.nb2 = (uint32_t) (node->nb[1] / 4);
                            pc.nb3 = (uint32_t) (node->nb[2] / 4);
                        } else {
                            pc.nb1 = (uint32_t) (node->nb[1] / 4);
                            pc.nb2 = (uint32_t) (node->nb[2] / 4);
                            pc.nb3 = (uint32_t) (node->nb[3] / 4);
                        }
                        if (node->op == GGML_OP_RMS_NORM || node->op == GGML_OP_NORM || node->op == GGML_OP_L2_NORM) {
                            std::memcpy(&pc.eps, node->op_params, sizeof(float));
                        }
                        pc.op_kind = (node->op == GGML_OP_RMS_NORM) ? 0u :
                                     (node->op == GGML_OP_NORM)     ? 1u :
                                     (node->op == GGML_OP_L2_NORM)  ? 2u :
                                     (node->op == GGML_OP_SUM_ROWS) ? 3u :
                                     (node->op == GGML_OP_MEAN)     ? 4u :
                                     (node->op == GGML_OP_CUMSUM)   ? 5u : 6u;
                        pc.nb00 = 1u;
                        const uint32_t n_rows = pc.ne1 * pc.ne2 * pc.ne3;
                        ok = agx_cdm_dispatch(stream, st->k_row_reduce, n_rows, 1, 1,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_SOFT_MAX: {
                        const auto * s     = node->src[0];
                        const auto * mask  = node->src[1];
                        const auto * sinks = node->src[2];
                        softmax_push pc{};
                        pc.src_va      = agx_tensor_gpu_va(s);
                        pc.mask_va     = mask  ? agx_tensor_gpu_va(mask)  : 0ull;
                        pc.sinks_va    = sinks ? agx_tensor_gpu_va(sinks) : 0ull;
                        pc.dst_va      = agx_tensor_gpu_va(node);
                        pc.ne0         = (uint32_t) s->ne[0]; pc.ne1 = (uint32_t) s->ne[1];
                        pc.ne2         = (uint32_t) s->ne[2]; pc.ne3 = (uint32_t) s->ne[3];
                        pc.m_ne1       = mask ? (uint32_t) mask->ne[1] : 1u;
                        pc.m_ne2       = mask ? (uint32_t) mask->ne[2] : 1u;
                        pc.m_ne3       = mask ? (uint32_t) mask->ne[3] : 1u;
                        pc.mask_is_f16 = (mask && mask->type == GGML_TYPE_F16) ? 1u : 0u;
                        std::memcpy(&pc.scale,    node->op_params, sizeof(float));
                        std::memcpy(&pc.max_bias, (const char *) node->op_params + 4, sizeof(float));
                        const uint32_t n_head = pc.ne2;
                        const uint32_t n_log2 = 1u << (uint32_t) std::floor(std::log2((double) std::max<uint32_t>(n_head, 1u)));
                        pc.n_head_log2 = n_log2;
                        pc.m0 = (pc.max_bias > 0.0f) ? std::pow(2.0f, -pc.max_bias / (float) n_log2) : 1.0f;
                        pc.m1 = (pc.max_bias > 0.0f) ? std::pow(2.0f, -pc.max_bias / (float) (2u * n_log2)) : 1.0f;
                        const uint32_t n_rows = pc.ne1 * pc.ne2 * pc.ne3;
                        ok = agx_cdm_dispatch(stream, st->k_softmax, n_rows, 1, 1, &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_GET_ROWS: {
                        const auto * s   = node->src[0];
                        const auto * ids = node->src[1];
                        uint32_t rb = 0;
                        get_rows_push pc{};
                        pc.src_va = agx_tensor_gpu_va(s);
                        pc.ids_va = agx_tensor_gpu_va(ids);
                        pc.dst_va = agx_tensor_gpu_va(node);
                        pc.ne0    = (uint32_t) node->ne[0]; pc.ne1 = (uint32_t) node->ne[1];
                        pc.ne2    = (uint32_t) node->ne[2]; pc.ne3 = (uint32_t) node->ne[3];
                        pc.wtype  = (s->type == GGML_TYPE_I32) ? 5u : agx_inkling2_tensor_wtype(s, &rb);
                        pc.s_nb1  = (pc.wtype == AGX_WTYPE_Q8_0_8BLK) ? rb : (uint32_t) s->nb[1];
                        pc.s_nb2  = (uint32_t) s->nb[2];
                        pc.s_nb3  = (uint32_t) s->nb[3];
                        pc.i_nb0  = (uint32_t) ids->nb[0]; pc.i_nb1 = (uint32_t) ids->nb[1]; pc.i_nb2 = (uint32_t) ids->nb[2];
                        pc.d_nb1  = (uint32_t) node->nb[1]; pc.d_nb2 = (uint32_t) node->nb[2]; pc.d_nb3 = (uint32_t) node->nb[3];
                        ok = agx_cdm_dispatch(stream, st->k_get_rows, (pc.ne0 + 255u) / 256u, pc.ne1, pc.ne2 * pc.ne3,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_SET_ROWS: {
                        const auto * s   = node->src[0];
                        const auto * ids = node->src[1];
                        set_rows_push pc{};
                        pc.src_va     = agx_tensor_gpu_va(s);
                        pc.ids_va     = agx_tensor_gpu_va(ids);
                        pc.dst_va     = agx_tensor_gpu_va(node);
                        pc.ne0        = (uint32_t) s->ne[0]; pc.ne1 = (uint32_t) s->ne[1];
                        pc.ne2        = (uint32_t) s->ne[2]; pc.ne3 = (uint32_t) s->ne[3];
                        pc.s_nb1      = (uint32_t) s->nb[1]; pc.s_nb2 = (uint32_t) s->nb[2]; pc.s_nb3 = (uint32_t) s->nb[3];
                        pc.i_nb0      = (uint32_t) ids->nb[0]; pc.i_nb1 = (uint32_t) ids->nb[1]; pc.i_nb2 = (uint32_t) ids->nb[2];
                        pc.i_ne1      = (uint32_t) ids->ne[1]; pc.i_ne2 = (uint32_t) ids->ne[2];
                        pc.d_nb1      = (uint32_t) node->nb[1]; pc.d_nb2 = (uint32_t) node->nb[2]; pc.d_nb3 = (uint32_t) node->nb[3];
                        pc.dst_type   = (node->type == GGML_TYPE_Q8_0) ? 2u :
                                        (node->type == GGML_TYPE_F16)  ? 1u : 0u;
                        pc.ids_is_i64 = (ids->type == GGML_TYPE_I64) ? 1u : 0u;
                        ok = agx_cdm_dispatch(stream, st->k_set_rows, (pc.ne0 + 255u) / 256u, pc.ne1, pc.ne2 * pc.ne3,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_ADD_ID: {
                        const auto * a   = node->src[0];
                        const auto * b   = node->src[1];
                        const auto * ids = node->src[2];
                        add_id_push pc{};
                        pc.a_va   = agx_tensor_gpu_va(a);
                        pc.b_va   = agx_tensor_gpu_va(b);
                        pc.ids_va = agx_tensor_gpu_va(ids);
                        pc.d_va   = agx_tensor_gpu_va(node);
                        pc.ne0    = (uint32_t) node->ne[0]; pc.ne1 = (uint32_t) node->ne[1]; pc.ne2 = (uint32_t) node->ne[2];
                        pc.a_nb1  = (uint32_t) (a->nb[1] / 4); pc.a_nb2 = (uint32_t) (a->nb[2] / 4);
                        pc.b_nb1  = (uint32_t) (b->nb[1] / 4);
                        pc.i_nb1  = (uint32_t) (ids->nb[1] / 4);
                        pc.d_nb1  = (uint32_t) (node->nb[1] / 4); pc.d_nb2 = (uint32_t) (node->nb[2] / 4);
                        ok = agx_cdm_dispatch(stream, st->k_add_id, (pc.ne0 + 255u) / 256u, pc.ne1, pc.ne2,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_CONCAT:
                    case GGML_OP_PAD:
                    case GGML_OP_REPEAT: {
                        const auto * a = node->src[0];
                        const auto * b = node->src[1];
                        concat_pad_push pc{};
                        pc.a_va  = agx_tensor_gpu_va(a);
                        pc.b_va  = b ? agx_tensor_gpu_va(b) : 0ull;
                        pc.d_va  = agx_tensor_gpu_va(node);
                        pc.ne0   = (uint32_t) node->ne[0]; pc.ne1   = (uint32_t) node->ne[1];
                        pc.ne2   = (uint32_t) node->ne[2]; pc.ne3   = (uint32_t) node->ne[3];
                        pc.a_ne0 = (uint32_t) a->ne[0];    pc.a_ne1 = (uint32_t) a->ne[1];
                        pc.a_ne2 = (uint32_t) a->ne[2];    pc.a_ne3 = (uint32_t) a->ne[3];
                        pc.a_nb0 = (uint32_t) (a->nb[0] / 4); pc.a_nb1 = (uint32_t) (a->nb[1] / 4);
                        pc.a_nb2 = (uint32_t) (a->nb[2] / 4); pc.a_nb3 = (uint32_t) (a->nb[3] / 4);
                        if (b) {
                            pc.b_nb0 = (uint32_t) (b->nb[0] / 4); pc.b_nb1 = (uint32_t) (b->nb[1] / 4);
                            pc.b_nb2 = (uint32_t) (b->nb[2] / 4); pc.b_nb3 = (uint32_t) (b->nb[3] / 4);
                        }
                        if (node->op == GGML_OP_CONCAT) {
                            pc.mode = 0u;
                            pc.p0   = (uint32_t) ggml_get_op_params_i32(node, 0);
                        } else if (node->op == GGML_OP_PAD) {
                            pc.mode = 1u;
                            pc.p0   = (uint32_t) ggml_get_op_params_i32(node, 0);
                            pc.p1   = (uint32_t) ggml_get_op_params_i32(node, 2);
                            pc.p2   = (uint32_t) ggml_get_op_params_i32(node, 4);
                            pc.p3   = (uint32_t) ggml_get_op_params_i32(node, 6);
                        } else {
                            pc.mode = 2u;
                        }
                        pc.total = (uint32_t) ggml_nelements(node);
                        ok = agx_cdm_dispatch(stream, st->k_concat_pad, (pc.total + 255u) / 256u, 1, 1,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_SSM_CONV: {
                        const auto * s0 = node->src[0];
                        const auto * s1 = node->src[1];
                        ssm_conv_push pc{};
                        pc.src0_va = agx_tensor_gpu_va(s0);
                        pc.src1_va = agx_tensor_gpu_va(s1);
                        pc.dst_va  = agx_tensor_gpu_va(node);
                        pc.nb01    = (uint32_t) (s0->nb[1] / 4);
                        pc.nb02    = (uint32_t) (s0->nb[2] / 4);
                        pc.nb11    = (uint32_t) (s1->nb[1] / 4);
                        pc.d_nb1   = (uint32_t) (node->nb[1] / 4);
                        pc.d_nb2   = (uint32_t) (node->nb[2] / 4);
                        pc.nc      = (uint32_t) s1->ne[0];
                        pc.nr      = (uint32_t) s0->ne[1];
                        pc.n_t     = (uint32_t) node->ne[1];
                        pc.n_s     = (uint32_t) node->ne[2];
                        ok = agx_cdm_dispatch(stream, st->k_ssm_conv,
                                              (pc.nr + 31u) / 32u, (pc.n_t + 7u) / 8u, pc.n_s,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_ROPE: {
                        const auto * s    = node->src[0];
                        const auto * pos  = node->src[1];
                        const auto * freq = node->src[2];
                        rope_push pc{};
                        pc.src_va  = agx_tensor_gpu_va(s);
                        pc.pos_va  = agx_tensor_gpu_va(pos);
                        pc.freq_va = freq ? agx_tensor_gpu_va(freq) : 0ull;
                        pc.dst_va  = agx_tensor_gpu_va(node);
                        pc.ne0     = (uint32_t) s->ne[0]; pc.ne1 = (uint32_t) s->ne[1];
                        pc.ne2     = (uint32_t) s->ne[2]; pc.ne3 = (uint32_t) s->ne[3];
                        pc.s_nb1   = (uint32_t) (s->nb[1] / 4); pc.s_nb2 = (uint32_t) (s->nb[2] / 4); pc.s_nb3 = (uint32_t) (s->nb[3] / 4);
                        pc.d_nb1   = (uint32_t) (node->nb[1] / 4); pc.d_nb2 = (uint32_t) (node->nb[2] / 4); pc.d_nb3 = (uint32_t) (node->nb[3] / 4);
                        pc.n_dims  = (uint32_t) ggml_get_op_params_i32(node, 1);
                        pc.mode    = (uint32_t) ggml_get_op_params_i32(node, 2);
                        float freq_base = 10000.0f, freq_scale = 1.0f, attn_factor = 1.0f;
                        std::memcpy(&freq_base,   (const char *) node->op_params + 20, sizeof(float));
                        std::memcpy(&freq_scale,  (const char *) node->op_params + 24, sizeof(float));
                        std::memcpy(&attn_factor, (const char *) node->op_params + 32, sizeof(float));
                        pc.theta_scale = std::pow(freq_base, -2.0f / (float) pc.n_dims);
                        pc.freq_scale  = freq_scale;
                        pc.attn_factor = attn_factor;
                        ok = agx_cdm_dispatch(stream, st->k_rope,
                                              ((pc.ne0 >> 1u) + 63u) / 64u, pc.ne1, pc.ne2 * pc.ne3,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_ARGSORT:
                    case GGML_OP_TOP_K: {
                        const auto * s = node->src[0];
                        topk_argsort_push pc{};
                        pc.src_va = agx_tensor_gpu_va(s);
                        pc.dst_va = agx_tensor_gpu_va(node);
                        pc.ne0    = (uint32_t) s->ne[0];
                        pc.k_out  = (uint32_t) node->ne[0];
                        pc.n_rows = (uint32_t) ggml_nrows(s);
                        pc.desc   = (node->op == GGML_OP_TOP_K) ? 1u :
                                    (ggml_get_op_params_i32(node, 0) == GGML_SORT_ORDER_DESC ? 1u : 0u);
                        ok = agx_cdm_dispatch(stream, st->k_topk_argsort, pc.n_rows, 1, 1,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_GLU: {
                        const auto * s0 = node->src[0];
                        const auto * s1 = node->src[1];
                        glu_push pc{};
                        pc.g_va   = agx_tensor_gpu_va(s0);
                        pc.u_va   = s1 ? agx_tensor_gpu_va(s1) : (pc.g_va + (uint64_t) node->ne[0] * sizeof(float));
                        pc.d_va   = agx_tensor_gpu_va(node);
                        pc.ne0    = (uint32_t) node->ne[0];
                        pc.n_rows = (uint32_t) ggml_nrows(node);
                        pc.g_nb1  = (uint32_t) (s0->nb[1] / 4);
                        pc.u_nb1  = s1 ? (uint32_t) (s1->nb[1] / 4) : pc.g_nb1;
                        pc.d_nb1  = (uint32_t) (node->nb[1] / 4);
                        pc.glu_op  = (uint32_t) ggml_get_glu_op(node);
                        pc.swapped = (uint32_t) ggml_get_op_params_i32(node, 1);
                        std::memcpy(&pc.alpha, (const char *) node->op_params + 8,  sizeof(float));
                        std::memcpy(&pc.limit, (const char *) node->op_params + 12, sizeof(float));
                        ok = agx_cdm_dispatch(stream, st->k_glu, (pc.ne0 + 255u) / 256u, pc.n_rows, 1,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_MUL_MAT: {
                        if (!no_fusion && ggml_get_op_params_i32(node, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                            node->src[1] && node->src[1]->type == GGML_TYPE_F32 &&
                            ggml_is_contiguous(node->src[1]) &&
                            (node->src[1]->ne[0] == 64 || node->src[1]->ne[0] == 128)) {
                            auto * eng = agx_graph_exec_get_inkling2(ctx);
                            if (eng) {
                                agx_conv_clip_fwht_push fpc{};
                                fpc.src_va       = agx_tensor_gpu_va(node->src[1]);
                                fpc.dst_va       = agx_tensor_gpu_va(node);
                                fpc.head_dim     = (uint32_t) node->src[1]->ne[0];
                                fpc.n_head       = (uint32_t) ggml_nrows(node->src[1]);
                                fpc.n_seq_tokens = 1u;
                                fpc.n_seqs       = 1u;
                                fpc.flags        = (fpc.head_dim == 128u) ? AGX_PREP_FLAG_FWHT_128 : AGX_PREP_FLAG_FWHT_64;
                                ok = agx_inkling2_dispatch_conv_clip_fwht(eng, stream, fpc, AGX_BARRIER_NONE);
                                if (ok) st->stats.fused_fwht_hits++;
                                break;
                            }
                        }
                        if (!agx_gemm_init(ctx)) return GGML_STATUS_FAILED;
                        ok = agx_gemm_mul_mat(ctx, stream, node, AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_MUL_MAT_ID: {
                        if (!ctx->moe_state) {
                            ctx->moe_state = agx_moe_create(ctx->drm, ctx->compiler);
                            if (!ctx->moe_state) return GGML_STATUS_FAILED;
                        }
                        ok = agx_moe_dispatch_mul_mat_id(static_cast<agx_moe_runner *>(ctx->moe_state),
                                                         stream, node, AGX_BARRIER_NONE);
                    } break;

                    case GGML_OP_FLASH_ATTN_EXT: {
                        if (!ctx->kv_hybrid_state) {
                            agx_kv_hybrid_config kcfg;
                            kcfg.n_slots = 48;
                            kcfg.max_full_tokens = 4096;
                            ctx->kv_hybrid_state = agx_kv_hybrid_create(ctx->drm, ctx->compiler, kcfg);
                            if (!ctx->kv_hybrid_state) return GGML_STATUS_FAILED;
                        }
                        ok = agx_kv_hybrid_dispatch_ggml_flash_attn(static_cast<agx_kv_hybrid_pool *>(ctx->kv_hybrid_state),
                                                                    stream, node, AGX_BARRIER_NONE);
                    } break;

                    default:
                        AGX_LOG_ERROR("agx_graph_compute: unsupported op %s (%d)", ggml_op_name(node->op), (int) node->op);
                        return GGML_STATUS_FAILED;
                }
            } break;
        }

        if (!ok) {
            AGX_LOG_ERROR("agx_graph_compute: dispatch failed for item %zu (kind=%u)", idx, (unsigned) it.kind);
            return GGML_STATUS_FAILED;
        }
        record_item_dispatch(it);
        st->stats.nodes_executed += it.nodes_covered;
    }

    const auto t_enc_end = std::chrono::steady_clock::now();
    const uint64_t cpu_encode_ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t_enc_end - t_enc_beg).count();
    st->stats.total_cpu_encode_ns += cpu_encode_ns;

    uint64_t gpu_ns = 0;
    if (has_dispatched) {
        // Single memory-scope host visibility barrier at end of entire graph!
        agx_cdm_barrier_host(stream);
        if (agx_cdm_submit(stream, true, &gpu_ns) != 0) {
            AGX_LOG_ERROR("agx_graph_compute: CDM submit failed");
            return GGML_STATUS_FAILED;
        }
        st->stats.total_gpu_ns += gpu_ns;
    }

    if (print_stats && n_exec >= 64u) {
        agx_cdm_stats cdm_after{};
        agx_cdm_get_stats(stream, &cdm_after);
        const uint64_t d_disp  = cdm_after.dispatches          - cdm_before.dispatches;
        const uint64_t d_usc   = cdm_after.strong_usc_barriers - cdm_before.strong_usc_barriers;
        const uint64_t d_elide = cdm_after.elided_barriers     - cdm_before.elided_barriers;
        const uint64_t d_sub   = cdm_after.submits             - cdm_before.submits;
        std::fprintf(stderr,
                     "[ggml-agx STATS] nodes=%d exec=%zu items=%zu waves=%u wave_sched=%d dispatches=%llu submits=%llu usc_bar=%llu elided=%llu "
                     "cpu_enc=%.2fms gpu=%.2fms fusions(prep=%llu sconv=%llu rel=%llu rtr=%llu cont=%llu rms=%llu situ=%llu ctrl=%llu mix=%llu wsum=%llu ssm=%llu mnorm=%llu clip=%llu sc=%llu addsc=%llu sigm=%llu)\n",
                     cgraph->n_nodes, n_exec, n_items, num_waves, no_wave_sched ? 0 : 1,
                     (unsigned long long) d_disp, (unsigned long long) d_sub,
                     (unsigned long long) d_usc, (unsigned long long) d_elide,
                     (double) cpu_encode_ns * 1e-6, (double) gpu_ns * 1e-6,
                     (unsigned long long) g_prep_store, (unsigned long long) g_full_sconv,
                     (unsigned long long) g_rel_bias, (unsigned long long) g_router,
                     (unsigned long long) g_cont_abs,
                     (unsigned long long) g_rms_mul, (unsigned long long) g_situ,
                     (unsigned long long) g_ctrl, (unsigned long long) g_mix,
                     (unsigned long long) g_moe_wsum, (unsigned long long) g_ssm_res,
                     (unsigned long long) g_moe_norm, (unsigned long long) g_l2_clip,
                     (unsigned long long) g_softcap, (unsigned long long) g_add_sc,
                     (unsigned long long) g_sig_mul);
    }

    st->stats.graphs_executed++;
    return GGML_STATUS_SUCCESS;
}

} // extern "C"
