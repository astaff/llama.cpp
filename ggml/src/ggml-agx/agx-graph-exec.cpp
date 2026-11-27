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
    uint     op_kind; // 0=RMS_NORM, 1=LAYER_NORM, 2=L2_NORM(exact 1/max(||x||,eps)), 3=SUM_ROWS, 4=MEAN, 5=CUMSUM, 6=ARGMAX
    float    eps;
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
        // RMS_NORM (with optional fused 1D weight multiply)
        float ss = 0.0;
        for (uint i = tid; i < D; i += 256u) {
            float x = ConstBufF32(pc.a_va).v[a_base + i];
            ss += x * x;
        }
        float sc = inversesqrt(wg_sum(ss, sg_id, lane_id) / float(D) + pc.eps);
        if (pc.w_va != 0ul) {
            for (uint i = tid; i < D; i += 256u) {
                BufF32(pc.d_va).v[d_base + i] = (ConstBufF32(pc.a_va).v[a_base + i] * sc) * ConstBufF32(pc.w_va).v[i];
            }
        } else {
            for (uint i = tid; i < D; i += 256u) {
                BufF32(pc.d_va).v[d_base + i] = ConstBufF32(pc.a_va).v[a_base + i] * sc;
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

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer ConstBufI64 { int64_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufF16 { float16_t v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t ids_va;
    uint64_t dst_va;
    uint     ne0, ne1, ne2, ne3; // src shape
    uint     s_nb1, s_nb2, s_nb3;
    uint     i_nb0, i_nb1, i_nb2;
    uint     i_ne1, i_ne2;
    uint     d_nb1, d_nb2, d_nb3;
    uint     dst_type;   // 0=F32, 1=F16
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
    } else {
        BufF16(d_row).v[col] = float16_t(val);
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
    uint     nb01, nb02; // floats
    uint     nb11;       // floats
    uint     d_nb1, d_nb2; // floats
    uint     nc, nr, n_t, n_s;
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
    BufF32(pc.dst_va).v[i3 * pc.d_nb2 + i2 * pc.d_nb1 + i1] = sum;
}
)GLSL";

struct ssm_conv_push {
    uint64_t src0_va, src1_va, dst_va;
    uint32_t nb01, nb02, nb11, d_nb1, d_nb2;
    uint32_t nc, nr, n_t, n_s;
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
    uint     pad0;
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

    if (pc.norm_w_va == 0ul) {
        for (uint i = tid; i < D; i += 256u) {
            float c = ConstBufF32(pc.cur_va).v[base + i];
            float m = ConstBufF32(pc.msg_va).v[base + i];
            float f = tanh(ConstBufF32(pc.forget_va).v[base + i]);
            float g = tanh(ConstBufF32(pc.gain_va).v[base + i]);
            BufF32(pc.out_va).v[base + i] = (c - f * c) + g * m;
        }
        return;
    }

    float ss = 0.0;
    for (uint i = tid; i < D; i += 256u) {
        float c = ConstBufF32(pc.cur_va).v[base + i];
        float m = ConstBufF32(pc.msg_va).v[base + i];
        float f = tanh(ConstBufF32(pc.forget_va).v[base + i]);
        float g = tanh(ConstBufF32(pc.gain_va).v[base + i]);
        float y = (c - f * c) + g * m;
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
    uint32_t pad0;
};
static_assert(sizeof(ctrl_mix_push) <= AGX_MAX_PUSH_BYTES, "ctrl_mix_push > 256B");

// ============================================================================
// Graph Executor State & Lazy Kernel Compilation
// ============================================================================

struct agx_va_range {
    uint64_t beg;
    uint64_t end;
};

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

struct agx_graph_exec_state {
    const agx_kernel * k_bin_op       = nullptr;
    const agx_kernel * k_unary_op     = nullptr;
    const agx_kernel * k_cpy_cast     = nullptr;
    const agx_kernel * k_row_reduce   = nullptr;
    const agx_kernel * k_softmax      = nullptr;
    const agx_kernel * k_get_rows     = nullptr;
    const agx_kernel * k_set_rows     = nullptr;
    const agx_kernel * k_concat_pad   = nullptr;
    const agx_kernel * k_ssm_conv     = nullptr;
    const agx_kernel * k_rope         = nullptr;
    const agx_kernel * k_topk_argsort = nullptr;
    const agx_kernel * k_glu          = nullptr;
    const agx_kernel * k_add_id       = nullptr;
    const agx_kernel * k_ctrl_mix     = nullptr;

    agx_ptr_count_map     use_map;
    std::vector<int>      exec_nodes;
    agx_inkling2_engine * inkling2    = nullptr;
    agx_graph_exec_stats  stats       = {};
    bool                  initialized = false;
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
        st->k_bin_op       = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_bin_op",       k_glsl_bin_op,       opts);
        st->k_unary_op     = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_unary_op",     k_glsl_unary_op,     opts);
        st->k_cpy_cast     = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_cpy_cast",     k_glsl_cpy_cast,     opts);
        st->k_row_reduce   = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_row_reduce",   k_glsl_row_reduce,   opts);
        st->k_softmax      = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_softmax",      k_glsl_softmax,      opts);
        st->k_get_rows     = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_get_rows",     k_glsl_get_rows,     opts);
        st->k_set_rows     = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_set_rows",     k_glsl_set_rows,     opts);
        st->k_concat_pad   = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_concat_pad",   k_glsl_concat_pad,   opts);
        st->k_ssm_conv     = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_ssm_conv",     k_glsl_ssm_conv,     opts);
        st->k_rope         = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_rope",         k_glsl_rope,         opts);
        st->k_topk_argsort = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_topk_argsort", k_glsl_topk_argsort, opts);
        st->k_glu          = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_glu",          k_glsl_glu,          opts);
        st->k_add_id       = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_add_id",       k_glsl_add_id,       opts);
        st->k_ctrl_mix     = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_ctrl_mix",     k_glsl_ctrl_mix,     opts);
        st->initialized    = (st->k_bin_op && st->k_unary_op && st->k_cpy_cast && st->k_row_reduce &&
                              st->k_softmax && st->k_get_rows && st->k_set_rows && st->k_concat_pad &&
                              st->k_ssm_conv && st->k_rope && st->k_topk_argsort && st->k_glu &&
                              st->k_add_id && st->k_ctrl_mix);
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
    if (!t) return { 0, 0 };
    uint64_t va = agx_tensor_gpu_va(t);
    if (!va) return { 0, 0 };
    // Walk multi-hop view chains (e.g. RESHAPE(VIEW(PERMUTE(x)))) to the root allocation
    const struct ggml_tensor * root = t;
    while (root->view_src) {
        root = root->view_src;
    }
    const uint64_t t_beg = va;
    const uint64_t t_end = va + tensor_byte_span(t);
    if (root != t) {
        const uint64_t rva = agx_tensor_gpu_va(root);
        if (rva) {
            const uint64_t r_beg = rva;
            const uint64_t r_end = rva + tensor_byte_span(root);
            return { std::min(t_beg, r_beg), std::max(t_end, r_end) };
        }
    }
    return { t_beg, t_end };
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
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
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
// Single-CDM-Stream Generic + Fused ggml_cgraph Execution
// ============================================================================

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
    const bool print_stats   = (std::getenv("GGML_AGX_STATS") != nullptr &&
                                std::getenv("GGML_AGX_STATS")[0] == '1');

    // Precompute O(1) reference counts & compact non-view compute node list (~15 us for 11k nodes)
    st->use_map.reset((size_t) cgraph->n_nodes + 16u);
    st->exec_nodes.clear();
    st->exec_nodes.reserve((size_t) cgraph->n_nodes);
    for (int k = 0; k < cgraph->n_nodes; ++k) {
        struct ggml_tensor * n = cgraph->nodes[k];
        if (!n) continue;
        if (n->view_src) st->use_map.inc(n->view_src);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s]) st->use_map.inc(n->src[s]);
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

    auto strip_reshape = [](const struct ggml_tensor * t) -> const struct ggml_tensor * {
        while (t && t->op == GGML_OP_RESHAPE && t->src[0]) {
            t = t->src[0];
        }
        return t;
    };

    // Verifies `leaf` reaches `target` solely through 0+ single-use non-output RESHAPE nodes,
    // and that `target` itself has exactly `target_uses` references and is not a graph output.
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

    auto read_scale_params = [](const struct ggml_tensor * sc_node, float * s, float * b) {
        std::memcpy(s, sc_node->op_params, sizeof(float));
        std::memcpy(b, (const char *) sc_node->op_params + 4, sizeof(float));
    };

    auto is_ctrl_wtype_ok = [](const struct ggml_tensor * w) -> bool {
        if (!w || !ggml_is_contiguous(w) || agx_tensor_gpu_va(w) == 0) return false;
        return w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16 ||
               w->type == GGML_TYPE_BF16 || w->type == GGML_TYPE_Q8_0;
    };

    std::vector<agx_va_range> dirty_vas;
    std::vector<agx_va_range> read_vas;
    dirty_vas.reserve(64);
    read_vas.reserve(128);
    bool has_dispatched    = false;
    bool used_moe_scratch  = false;
    bool used_attn_scratch = false;

    auto ensure_node_ready = [&](const struct ggml_tensor * const * srcs, int n_srcs,
                                 const struct ggml_tensor * dst,
                                 bool needs_moe_scratch, bool needs_attn_scratch) {
        bool hazard = force_barrier;
        if (!hazard && has_dispatched) {
            if ((needs_moe_scratch && used_moe_scratch) ||
                (needs_attn_scratch && used_attn_scratch)) {
                hazard = true;
            }
            // RAW: any input overlaps an unflushed write
            for (int s = 0; !hazard && s < n_srcs; ++s) {
                if (!srcs[s]) continue;
                if (ranges_overlap(dirty_vas, tensor_va_range(srcs[s]))) {
                    hazard = true;
                }
            }
            // WAW & WAR: output overlaps an unflushed write OR an unflushed read
            if (!hazard && dst) {
                const agx_va_range dr = tensor_va_range(dst);
                if (ranges_overlap(dirty_vas, dr) || ranges_overlap(read_vas, dr)) {
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

    auto record_dispatch = [&](const struct ggml_tensor * const * srcs, int n_srcs,
                               const struct ggml_tensor * dst,
                               bool uses_moe_scratch, bool uses_attn_scratch) {
        has_dispatched = true;
        for (int s = 0; s < n_srcs; ++s) {
            if (!srcs[s]) continue;
            const agx_va_range sr = tensor_va_range(srcs[s]);
            if (sr.beg != sr.end) read_vas.push_back(sr);
        }
        if (dst) {
            const agx_va_range dr = tensor_va_range(dst);
            if (dr.beg != dr.end) dirty_vas.push_back(dr);
        }
        if (uses_moe_scratch)  used_moe_scratch  = true;
        if (uses_attn_scratch) used_attn_scratch = true;
    };

    const size_t n_exec = st->exec_nodes.size();
    uint64_t g_l2_clip = 0, g_situ = 0, g_softcap = 0, g_ctrl = 0, g_rms_mul = 0, g_add_sc = 0, g_sig_mul = 0;

    for (size_t e = 0; e < n_exec; ++e) {
        const int i = st->exec_nodes[e];
        struct ggml_tensor * node = cgraph->nodes[i];
        // Never submit a kernel referencing a null / non-AGX GPU VA (prevents G17 page faults)
        if (agx_tensor_gpu_va(node) == 0) {
            AGX_LOG_ERROR("agx_graph_compute: node %d (%s) dst has null GPU VA", i, ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] && ggml_nelements(node->src[s]) > 0 && agx_tensor_gpu_va(node->src[s]) == 0) {
                AGX_LOG_ERROR("agx_graph_compute: node %d (%s) src[%d] has null GPU VA", i, ggml_op_name(node->op), s);
                return GGML_STATUS_FAILED;
            }
        }

        // --------------------------------------------------------------------
        // Fusion 1: Inkling 2 Exact L2 Clip (`L2_NORM(t) -> SCALE(t)`)
        // --------------------------------------------------------------------
        if (!no_fusion && node->op == GGML_OP_L2_NORM && e + 1u < n_exec) {
            struct ggml_tensor * sc_node = cgraph->nodes[st->exec_nodes[e + 1u]];
            if (sc_node->op == GGML_OP_SCALE && chain_uses_exact(sc_node->src[0], node)) {
                float thresh = 0.0f, sc = 0.0f, bias = 0.0f;
                std::memcpy(&thresh, node->op_params, sizeof(float));
                read_scale_params(sc_node, &sc, &bias);
                if (bias == 0.0f && std::fabs(thresh - sc) < 1e-6f && thresh > 0.0f &&
                    ggml_is_contiguous(node->src[0]) && ggml_is_contiguous(sc_node) &&
                    agx_tensor_gpu_va(sc_node) != 0 && node->src[0]->ne[0] <= 128) {
                    auto * eng = agx_graph_exec_get_inkling2(ctx);
                    if (eng) {
                        const struct ggml_tensor * srcs[1] = { node->src[0] };
                        ensure_node_ready(srcs, 1, sc_node, false, false);

                        agx_conv_clip_fwht_push cpc{};
                        cpc.src_va       = agx_tensor_gpu_va(node->src[0]);
                        cpc.dst_va       = agx_tensor_gpu_va(sc_node);
                        cpc.head_dim     = (uint32_t) node->src[0]->ne[0];
                        cpc.n_head       = (uint32_t) node->src[0]->ne[1];
                        cpc.n_seq_tokens = (uint32_t) (node->src[0]->ne[2] * node->src[0]->ne[3]);
                        cpc.n_seqs       = 1u;
                        cpc.flags        = AGX_PREP_FLAG_L2_CLIP;
                        cpc.clip_thresh  = thresh;
                        if (!agx_inkling2_dispatch_conv_clip_fwht(eng, stream, cpc, AGX_BARRIER_NONE)) {
                            return GGML_STATUS_FAILED;
                        }
                        record_dispatch(srcs, 1, sc_node, false, false);
                        st->stats.fused_l2_clip_hits++;
                        g_l2_clip++;
                        st->stats.nodes_executed += 2;
                        e += 1u;
                        continue;
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 2: Inkling 2 MoE SiTU Activation (9-op DAG -> 1 k_glu dispatch)
        //   SCALE(g,1/b) -> TANH -> SCALE(b) -> SIGMOID(g) -> MUL
        //   -> SCALE(u,1/lb) -> TANH -> SCALE(lb) -> MUL
        // --------------------------------------------------------------------
        if (!no_fusion && node->op == GGML_OP_SCALE && e + 8u < n_exec) {
            struct ggml_tensor * n1 = cgraph->nodes[st->exec_nodes[e + 1u]];
            struct ggml_tensor * n2 = cgraph->nodes[st->exec_nodes[e + 2u]];
            struct ggml_tensor * n3 = cgraph->nodes[st->exec_nodes[e + 3u]];
            struct ggml_tensor * n4 = cgraph->nodes[st->exec_nodes[e + 4u]];
            struct ggml_tensor * n5 = cgraph->nodes[st->exec_nodes[e + 5u]];
            struct ggml_tensor * n6 = cgraph->nodes[st->exec_nodes[e + 6u]];
            struct ggml_tensor * n7 = cgraph->nodes[st->exec_nodes[e + 7u]];
            struct ggml_tensor * n8 = cgraph->nodes[st->exec_nodes[e + 8u]];
            if (n1->op == GGML_OP_UNARY && ggml_get_unary_op(n1) == GGML_UNARY_OP_TANH &&
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
                    agx_tensor_gpu_va(u) != 0 && agx_tensor_gpu_va(n8) != 0) {
                    const struct ggml_tensor * srcs[2] = { g, u };
                    ensure_node_ready(srcs, 2, n8, false, false);

                    glu_push pc{};
                    pc.g_va    = agx_tensor_gpu_va(g);
                    pc.u_va    = agx_tensor_gpu_va(u);
                    pc.d_va    = agx_tensor_gpu_va(n8);
                    pc.ne0     = (uint32_t) n8->ne[0];
                    pc.n_rows  = (uint32_t) ggml_nrows(n8);
                    pc.g_nb1   = (uint32_t) (g->nb[1] / 4);
                    pc.u_nb1   = (uint32_t) (u->nb[1] / 4);
                    pc.d_nb1   = (uint32_t) (n8->nb[1] / 4);
                    pc.glu_op  = 99u; // SiTU
                    pc.swapped = 0u;
                    pc.alpha   = s2;  // situ_beta
                    pc.limit   = s7;  // situ_linear_beta
                    if (!agx_cdm_dispatch(stream, st->k_glu, (pc.ne0 + 255u) / 256u, pc.n_rows, 1,
                                          &pc, sizeof(pc), AGX_BARRIER_NONE)) {
                        return GGML_STATUS_FAILED;
                    }
                    record_dispatch(srcs, 2, n8, false, false);
                    st->stats.fused_situ_hits++;
                    g_situ++;
                    st->stats.nodes_executed += 9;
                    e += 8u;
                    continue;
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 3: Inkling 2 Logit Softcap (`SCALE(1/c) -> TANH -> SCALE(c)`)
        // --------------------------------------------------------------------
        if (!no_fusion && node->op == GGML_OP_SCALE && e + 2u < n_exec) {
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
                    auto * eng = agx_graph_exec_get_inkling2(ctx);
                    if (eng) {
                        const struct ggml_tensor * srcs[1] = { node->src[0] };
                        ensure_node_ready(srcs, 1, s2_node, false, false);

                        const uint32_t ne0 = (uint32_t) s2_node->ne[0];
                        const uint32_t nr  = (uint32_t) ggml_nrows(s2_node);
                        if (!agx_inkling2_dispatch_softcap_argmax(eng, stream,
                                agx_tensor_gpu_va(node->src[0]),
                                agx_tensor_gpu_va(s2_node),
                                0ull, ne0, ne0, nr, s2, false, AGX_BARRIER_NONE)) {
                            return GGML_STATUS_FAILED;
                        }
                        record_dispatch(srcs, 1, s2_node, false, false);
                        st->stats.fused_softcap_hits++;
                        g_softcap++;
                        st->stats.nodes_executed += 3;
                        e += 2u;
                        continue;
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 4: Inkling 2 Cross-Pass Rank-64 Controller Gate (`control()` + optional RMSNorm*w)
        //   e+0..e+10 (+ optional e+11..e+12):
        //   n_tokens >= 16 -> 1 dispatch via `fused_ctrl_gates_sram` (native FP32 32KB SRAM)
        //   n_tokens <  16 -> 40-core overlapped W_forget/W_gain GEMVs + 1 `k_ctrl_mix` dispatch
        // --------------------------------------------------------------------
        if (!no_fusion && node->op == GGML_OP_CONCAT && e + 10u < n_exec &&
            ggml_get_op_params_i32(node, 0) == 1) {
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
                const int64_t n_embd   = cur->ne[0];
                const int64_t n_tokens = cur->ne[1];
                const int64_t rank     = w_down ? w_down->ne[1] : 0;

                if (cur->type == GGML_TYPE_F32 && msg->type == GGML_TYPE_F32 && n10->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(cur) && ggml_is_contiguous(msg) && ggml_is_contiguous(n10) &&
                    cur->ne[0] == msg->ne[0] && cur->ne[1] == msg->ne[1] &&
                    cur->ne[2] == 1 && cur->ne[3] == 1 && msg->ne[2] == 1 && msg->ne[3] == 1 &&
                    n_embd > 0 && n_embd <= 3840 && rank > 0 && rank <= 256 &&
                    w_down && w_forget && w_gain &&
                    w_down->ne[0] == 2 * n_embd && w_forget->ne[0] == rank && w_forget->ne[1] == n_embd &&
                    w_gain->ne[0] == rank && w_gain->ne[1] == n_embd &&
                    agx_tensor_gpu_va(n10) != 0) {

                    // Check optional trailing `RMS_NORM -> MUL(norm_w)` (e+11, e+12)
                    struct ggml_tensor * final_dst = n10;
                    const struct ggml_tensor * norm_w = nullptr;
                    float eps_ctrl = 1e-6f, eps_norm = 1e-6f;
                    std::memcpy(&eps_ctrl, n1->op_params, sizeof(float));
                    size_t advance = 10u;

                    if (e + 12u < n_exec) {
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
                        is_ctrl_wtype_ok(w_down) && is_ctrl_wtype_ok(w_forget) && is_ctrl_wtype_ok(w_gain)) {
                        auto * eng = agx_graph_exec_get_inkling2(ctx);
                        if (eng) {
                            const struct ggml_tensor * srcs[6] = { cur, msg, w_down, w_forget, w_gain, norm_w };
                            ensure_node_ready(srcs, 6, final_dst, false, false);

                            agx_ctrl_gate_desc desc{};
                            desc.cur_va         = agx_tensor_gpu_va(cur);
                            desc.msg_va         = agx_tensor_gpu_va(msg);
                            desc.down_va        = agx_tensor_gpu_va(w_down);
                            desc.gain_va        = agx_tensor_gpu_va(w_gain);
                            desc.forget_va      = agx_tensor_gpu_va(w_forget);
                            desc.norm_w_va      = norm_w ? agx_tensor_gpu_va(norm_w) : 0ull;
                            desc.out_va         = agx_tensor_gpu_va(final_dst);
                            desc.down_wtype     = agx_inkling2_tensor_wtype(w_down,   &desc.down_row_bytes);
                            desc.gain_wtype     = agx_inkling2_tensor_wtype(w_gain,   &desc.gain_row_bytes);
                            desc.forget_wtype   = agx_inkling2_tensor_wtype(w_forget, &desc.forget_row_bytes);
                            desc.cur_stride_f32 = (uint32_t) (cur->nb[1] / 4);
                            desc.msg_stride_f32 = (uint32_t) (msg->nb[1] / 4);
                            desc.msg_offset_f32 = 0u;
                            desc.out_stride_f32 = (uint32_t) (final_dst->nb[1] / 4);
                            if (!agx_inkling2_dispatch_ctrl_gates(eng, stream, &desc, 1u,
                                    (uint32_t) n_embd, (uint32_t) rank, (uint32_t) n_tokens,
                                    eps_ctrl, eps_norm, AGX_BARRIER_NONE)) {
                                return GGML_STATUS_FAILED;
                            }
                            record_dispatch(srcs, 6, final_dst, false, false);
                            st->stats.fused_ctrl_gate_hits++;
                            g_ctrl++;
                            st->stats.nodes_executed += (uint64_t) (advance + 1u);
                            e += advance;
                            continue;
                        }
                    } else if (n_tokens < 16 &&
                               agx_tensor_gpu_va(n1) != 0 && agx_tensor_gpu_va(n2) != 0 &&
                               agx_tensor_gpu_va(n3) != 0 && agx_tensor_gpu_va(n7) != 0 &&
                               agx_gemm_supports_mul_mat(n2) && agx_gemm_supports_mul_mat(n3) &&
                               agx_gemm_supports_mul_mat(n7) && agx_gemm_init(ctx)) {
                        // B < 16: preserve full 40-core DRAM bandwidth on W_down/W_forget/W_gain,
                        // overlap W_forget & W_gain concurrently, then fuse 6..8 tail ops into 1 k_ctrl_mix!
                        const struct ggml_tensor * srcs_cat[2] = { cur, msg };
                        ensure_node_ready(srcs_cat, 2, node, false, false);
                        {
                            const auto * a = node->src[0];
                            const auto * b = node->src[1];
                            concat_pad_push cpc{};
                            cpc.a_va  = agx_tensor_gpu_va(a); cpc.b_va = agx_tensor_gpu_va(b);
                            cpc.d_va  = agx_tensor_gpu_va(node);
                            cpc.ne0   = (uint32_t) node->ne[0]; cpc.ne1 = (uint32_t) node->ne[1];
                            cpc.ne2   = (uint32_t) node->ne[2]; cpc.ne3 = (uint32_t) node->ne[3];
                            cpc.a_ne0 = (uint32_t) a->ne[0];    cpc.a_ne1 = (uint32_t) a->ne[1];
                            cpc.a_ne2 = (uint32_t) a->ne[2];    cpc.a_ne3 = (uint32_t) a->ne[3];
                            cpc.a_nb0 = (uint32_t) (a->nb[0]/4); cpc.a_nb1 = (uint32_t) (a->nb[1]/4);
                            cpc.a_nb2 = (uint32_t) (a->nb[2]/4); cpc.a_nb3 = (uint32_t) (a->nb[3]/4);
                            cpc.b_nb0 = (uint32_t) (b->nb[0]/4); cpc.b_nb1 = (uint32_t) (b->nb[1]/4);
                            cpc.b_nb2 = (uint32_t) (b->nb[2]/4); cpc.b_nb3 = (uint32_t) (b->nb[3]/4);
                            cpc.mode  = 0u; cpc.p0 = 1u; cpc.total = (uint32_t) ggml_nelements(node);
                            if (!agx_cdm_dispatch(stream, st->k_concat_pad, (cpc.total + 255u) / 256u, 1, 1,
                                                  &cpc, sizeof(cpc), AGX_BARRIER_STRONG_USC)) {
                                return GGML_STATUS_FAILED;
                            }
                        }
                        {
                            const auto * a = n1->src[0];
                            row_reduce_push rpc{};
                            rpc.a_va = agx_tensor_gpu_va(a); rpc.d_va = agx_tensor_gpu_va(n1);
                            rpc.ne0  = (uint32_t) a->ne[0]; rpc.ne1 = (uint32_t) a->ne[1];
                            rpc.ne2  = (uint32_t) a->ne[2]; rpc.ne3 = (uint32_t) a->ne[3];
                            rpc.nb01 = (uint32_t) (a->nb[1]/4); rpc.nb02 = (uint32_t) (a->nb[2]/4); rpc.nb03 = (uint32_t) (a->nb[3]/4);
                            rpc.nb1  = (uint32_t) (n1->nb[1]/4); rpc.nb2 = (uint32_t) (n1->nb[2]/4); rpc.nb3 = (uint32_t) (n1->nb[3]/4);
                            rpc.op_kind = 0u; rpc.eps = eps_ctrl;
                            if (!agx_cdm_dispatch(stream, st->k_row_reduce, rpc.ne1 * rpc.ne2 * rpc.ne3, 1, 1,
                                                  &rpc, sizeof(rpc), AGX_BARRIER_STRONG_USC)) {
                                return GGML_STATUS_FAILED;
                            }
                        }
                        if (!agx_gemm_mul_mat(ctx, stream, n2, AGX_BARRIER_STRONG_USC)) return GGML_STATUS_FAILED;
                        // Overlap W_forget and W_gain across all 40 cores (no barrier between them!)
                        if (!agx_gemm_mul_mat(ctx, stream, n3, AGX_BARRIER_NONE))       return GGML_STATUS_FAILED;
                        st->stats.barriers_elided++;
                        if (!agx_gemm_mul_mat(ctx, stream, n7, AGX_BARRIER_STRONG_USC)) return GGML_STATUS_FAILED;

                        ctrl_mix_push mpc{};
                        mpc.cur_va    = agx_tensor_gpu_va(cur);
                        mpc.msg_va    = agx_tensor_gpu_va(msg);
                        mpc.forget_va = agx_tensor_gpu_va(n3);
                        mpc.gain_va   = agx_tensor_gpu_va(n7);
                        mpc.norm_w_va = norm_w ? agx_tensor_gpu_va(norm_w) : 0ull;
                        mpc.out_va    = agx_tensor_gpu_va(final_dst);
                        mpc.n_embd    = (uint32_t) n_embd;
                        mpc.n_tokens  = (uint32_t) n_tokens;
                        mpc.eps_norm  = eps_norm;
                        if (!agx_cdm_dispatch(stream, st->k_ctrl_mix, mpc.n_tokens, 1, 1,
                                              &mpc, sizeof(mpc), AGX_BARRIER_NONE)) {
                            return GGML_STATUS_FAILED;
                        }
                        dirty_vas.clear();
                        read_vas.clear();
                        used_moe_scratch  = false;
                        used_attn_scratch = false;
                        const struct ggml_tensor * mix_srcs[5] = { cur, msg, n3, n7, norm_w };
                        record_dispatch(mix_srcs, 5, final_dst, false, false);
                        st->stats.fused_ctrl_gate_hits++;
                        g_ctrl++;
                        st->stats.nodes_executed += (uint64_t) (advance + 1u);
                        e += advance;
                        continue;
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 5: Weighted RMSNorm (`RMS_NORM -> MUL(1D weight)`)
        // --------------------------------------------------------------------
        if (!no_fusion && node->op == GGML_OP_RMS_NORM && e + 1u < n_exec) {
            struct ggml_tensor * mul_node = cgraph->nodes[st->exec_nodes[e + 1u]];
            if (mul_node->op == GGML_OP_MUL && mul_node->type == GGML_TYPE_F32 &&
                mul_node->nb[0] == sizeof(float) && agx_tensor_gpu_va(mul_node) != 0) {
                const struct ggml_tensor * w = nullptr;
                if (chain_uses_exact(mul_node->src[0], node))      w = mul_node->src[1];
                else if (chain_uses_exact(mul_node->src[1], node)) w = mul_node->src[0];
                if (w && w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) &&
                    w->ne[0] == node->ne[0] && w->ne[1] == 1 && w->ne[2] == 1 && w->ne[3] == 1 &&
                    ggml_are_same_shape(node, mul_node) && agx_tensor_gpu_va(w) != 0) {
                    const auto * a = node->src[0];
                    const struct ggml_tensor * srcs[2] = { a, w };
                    ensure_node_ready(srcs, 2, mul_node, false, false);

                    row_reduce_push pc{};
                    pc.a_va = agx_tensor_gpu_va(a);
                    pc.d_va = agx_tensor_gpu_va(mul_node);
                    pc.w_va = agx_tensor_gpu_va(w);
                    pc.ne0  = (uint32_t) a->ne[0]; pc.ne1 = (uint32_t) a->ne[1];
                    pc.ne2  = (uint32_t) a->ne[2]; pc.ne3 = (uint32_t) a->ne[3];
                    pc.nb01 = (uint32_t) (a->nb[1] / 4); pc.nb02 = (uint32_t) (a->nb[2] / 4); pc.nb03 = (uint32_t) (a->nb[3] / 4);
                    pc.nb1  = (uint32_t) (mul_node->nb[1] / 4); pc.nb2 = (uint32_t) (mul_node->nb[2] / 4); pc.nb3 = (uint32_t) (mul_node->nb[3] / 4);
                    std::memcpy(&pc.eps, node->op_params, sizeof(float));
                    pc.op_kind = 0u;
                    const uint32_t n_rows = pc.ne1 * pc.ne2 * pc.ne3;
                    if (!agx_cdm_dispatch(stream, st->k_row_reduce, n_rows, 1, 1,
                                          &pc, sizeof(pc), AGX_BARRIER_NONE)) {
                        return GGML_STATUS_FAILED;
                    }
                    record_dispatch(srcs, 2, mul_node, false, false);
                    st->stats.fused_rms_mul_hits++;
                    g_rms_mul++;
                    st->stats.nodes_executed += 2;
                    e += 1u;
                    continue;
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 6: Residual + Scale (`ADD -> SCALE`, 192x/tok in Inkling 2 shortconv)
        // --------------------------------------------------------------------
        if (!no_fusion && node->op == GGML_OP_ADD && e + 1u < n_exec) {
            struct ggml_tensor * sc_node = cgraph->nodes[st->exec_nodes[e + 1u]];
            if (sc_node->op == GGML_OP_SCALE && chain_uses_exact(sc_node->src[0], node) &&
                ggml_are_same_shape(node, sc_node) && agx_tensor_gpu_va(sc_node) != 0) {
                const auto * a = node->src[0];
                const auto * b = node->src[1];
                const struct ggml_tensor * srcs[2] = { a, b };
                ensure_node_ready(srcs, 2, sc_node, false, false);

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
                if (!agx_cdm_dispatch(stream, st->k_bin_op, (pc.total + 255u) / 256u, 1, 1,
                                      &pc, sizeof(pc), AGX_BARRIER_NONE)) {
                    return GGML_STATUS_FAILED;
                }
                record_dispatch(srcs, 2, sc_node, false, false);
                st->stats.fused_add_scale_hits++;
                g_add_sc++;
                st->stats.nodes_executed += 2;
                e += 1u;
                continue;
            }
        }

        // --------------------------------------------------------------------
        // Fusion 7: Sigmoid Gate (`SIGMOID(b) -> MUL(a, sig_b)`, 48x/tok Q-gate)
        // --------------------------------------------------------------------
        if (!no_fusion && node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID &&
            e + 1u < n_exec) {
            struct ggml_tensor * mul_node = cgraph->nodes[st->exec_nodes[e + 1u]];
            if (mul_node->op == GGML_OP_MUL && mul_node->type == GGML_TYPE_F32 &&
                agx_tensor_gpu_va(mul_node) != 0) {
                const struct ggml_tensor * a = nullptr;
                if (mul_node->src[1] == node && chain_uses_exact(mul_node->src[1], node))      a = mul_node->src[0];
                else if (mul_node->src[0] == node && chain_uses_exact(mul_node->src[0], node)) a = mul_node->src[1];
                const struct ggml_tensor * b = node->src[0];
                if (a && b && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
                    ggml_are_same_shape(b, node) && agx_tensor_gpu_va(a) != 0) {
                    const struct ggml_tensor * srcs[2] = { a, b };
                    ensure_node_ready(srcs, 2, mul_node, false, false);

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
                    if (!agx_cdm_dispatch(stream, st->k_bin_op, (pc.total + 255u) / 256u, 1, 1,
                                          &pc, sizeof(pc), AGX_BARRIER_NONE)) {
                        return GGML_STATUS_FAILED;
                    }
                    record_dispatch(srcs, 2, mul_node, false, false);
                    st->stats.fused_sig_mul_hits++;
                    g_sig_mul++;
                    st->stats.nodes_executed += 2;
                    e += 1u;
                    continue;
                }
            }
        }

        // Standard RAW + WAR + WAW + internal-scratch hazard check across inputs & dst
        const bool is_moe_op  = (node->op == GGML_OP_MUL_MAT_ID);
        const bool is_attn_op = (node->op == GGML_OP_FLASH_ATTN_EXT);
        ensure_node_ready(node->src, GGML_MAX_SRC, node, is_moe_op, is_attn_op);

        bool ok = false;
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
                    // ggml_argmax drops dim 0: node->ne = [a->ne[1], a->ne[2], a->ne[3], 1]
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
                pc.dst_type   = (node->type == GGML_TYPE_F16) ? 1u : 0u;
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
                // Fastpath: in-register FWHT_{64,128} when GGML_HINT_SRC0_IS_HADAMARD is set!
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

        if (!ok) {
            AGX_LOG_ERROR("agx_graph_compute: dispatch failed for node %d (%s)", i, ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
        record_dispatch(node->src, GGML_MAX_SRC, node, is_moe_op, is_attn_op);
        st->stats.nodes_executed++;
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
                     "[ggml-agx STATS] nodes=%d exec=%zu dispatches=%llu submits=%llu usc_bar=%llu elided=%llu "
                     "cpu_enc=%.2fms gpu=%.2fms fusions(rms=%llu situ=%llu ctrl=%llu clip=%llu sc=%llu addsc=%llu sigm=%llu)\n",
                     cgraph->n_nodes, n_exec,
                     (unsigned long long) d_disp, (unsigned long long) d_sub,
                     (unsigned long long) d_usc, (unsigned long long) d_elide,
                     (double) cpu_encode_ns * 1e-6, (double) gpu_ns * 1e-6,
                     (unsigned long long) g_rms_mul, (unsigned long long) g_situ,
                     (unsigned long long) g_ctrl, (unsigned long long) g_l2_clip,
                     (unsigned long long) g_softcap, (unsigned long long) g_add_sc,
                     (unsigned long long) g_sig_mul);
    }

    st->stats.graphs_executed++;
    return GGML_STATUS_SUCCESS;
}

} // extern "C"
