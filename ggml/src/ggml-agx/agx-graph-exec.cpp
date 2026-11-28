#include "agx-graph-exec.h"
#include "agx-inkling2.h"
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
layout(buffer_reference, std430, buffer_reference_align = 16) buffer BufVec4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufVec4 { vec4 v[]; };

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
    uint     op_kind; // low 8 bits: 0=ADD,1=SUB,2=MUL,3=DIV,4=(a+b)*p0+p1,5=a*sigmoid(b); 0x100=flat vec4; 0x200=3D bcast vec4
    uint     total;
    float    p0;
    float    p1;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if ((pc.op_kind & 0x300u) != 0u) {
        vec4 a, b;
        uint out_off4;
        if ((pc.op_kind & 0x100u) != 0u) {
            if (idx >= (pc.total >> 2u)) return;
            a = ConstBufVec4(pc.a_va).v[idx];
            b = ConstBufVec4(pc.b_va).v[idx];
            out_off4 = idx;
        } else {
            const uint ne0_4 = pc.ne0 >> 2u;
            const uint i1    = gl_WorkGroupID.y;
            const uint i2    = gl_WorkGroupID.z;
            if (idx >= ne0_4 || i1 >= pc.ne1 || i2 >= pc.ne2) return;
            out_off4 = (i2 * pc.ne1 + i1) * ne0_4 + idx;
            a = ConstBufVec4(pc.a_va).v[out_off4];
            const uint bi1 = (pc.ne11 > 1u) ? i1 : 0u;
            const uint bi2 = (pc.ne12 > 1u) ? i2 : 0u;
            if (pc.ne10 > 1u) {
                b = ConstBufVec4(pc.b_va).v[(bi2 * pc.ne11 + bi1) * ne0_4 + idx];
            } else {
                b = vec4(ConstBufF32(pc.b_va).v[bi2 * pc.ne11 + bi1]);
            }
        }
        uint op = pc.op_kind & 0xFFu;
        vec4 r;
        if      (op == 0u) r = a + b;
        else if (op == 1u) r = a - b;
        else if (op == 2u) r = a * b;
        else if (op == 3u) r = a / b;
        else if (op == 4u) r = (a + b) * pc.p0 + pc.p1;
        else               r = a / (vec4(1.0) + exp(-b));
        BufVec4(pc.d_va).v[out_off4] = r;
        return;
    }
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
layout(buffer_reference, std430, buffer_reference_align = 8) buffer BufUvec2 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer ConstBufUvec2 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer BufVec4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufVec4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer BufUvec4 { uvec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufUvec4 { uvec4 v[]; };

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
    uint     d_is_linear; // 1=linear; 2=flat uvec4 copy; 3=flat F16->F32 vec4; 4=flat F32->F16 vec4
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (pc.d_is_linear >= 2u) {
        if (idx >= (pc.total >> 2u)) return;
        if (pc.d_is_linear == 2u) {
            BufUvec4(pc.dst_va).v[idx] = ConstBufUvec4(pc.src_va).v[idx];
        } else if (pc.d_is_linear == 3u) {
            uvec2 p = ConstBufUvec2(pc.src_va).v[idx];
            BufVec4(pc.dst_va).v[idx] = vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y));
        } else {
            vec4 v = ConstBufVec4(pc.src_va).v[idx];
            BufUvec2(pc.dst_va).v[idx] = uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
        }
        return;
    }
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
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer BufVec4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufVec4 { vec4 v[]; };
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
        const bool v4_ok = (s0 == 1u) && ((D & 3u) == 0u) && (((a_base | d_base) & 3u) == 0u) &&
                           (((pc.a_va | pc.d_va | pc.w_va) & 15ul) == 0ul);
        if (v4_ok) {
            const uint D4 = D >> 2u;
            const uint a4 = a_base >> 2u;
            const uint d4 = d_base >> 2u;
            ConstBufVec4 ab = ConstBufVec4(pc.a_va);
            BufVec4      db = BufVec4(pc.d_va);
            if (D4 <= 1024u) {
                vec4 rx[4];
                float ss = 0.0;
                [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                    uint i = tid + k * 256u;
                    vec4 x = (i < D4) ? ab.v[a4 + i] : vec4(0.0);
                    rx[k] = x;
                    ss += dot(x, x);
                }
                float sc = inversesqrt(wg_sum(ss, sg_id, lane_id) / float(D) + pc.eps);
                if (pc.w_va != 0ul) {
                    ConstBufVec4 wb = ConstBufVec4(pc.w_va);
                    [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                        uint i = tid + k * 256u;
                        if (i < D4) db.v[d4 + i] = (rx[k] * sc) * wb.v[i];
                    }
                } else {
                    [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                        uint i = tid + k * 256u;
                        if (i < D4) db.v[d4 + i] = rx[k] * sc;
                    }
                }
                return;
            }
            float ss = 0.0;
            for (uint i = tid; i < D4; i += 256u) {
                vec4 x = ab.v[a4 + i];
                ss += dot(x, x);
            }
            float sc = inversesqrt(wg_sum(ss, sg_id, lane_id) / float(D) + pc.eps);
            if (pc.w_va != 0ul) {
                ConstBufVec4 wb = ConstBufVec4(pc.w_va);
                for (uint i = tid; i < D4; i += 256u) {
                    db.v[d4 + i] = (ab.v[a4 + i] * sc) * wb.v[i];
                }
            } else {
                for (uint i = tid; i < D4; i += 256u) {
                    db.v[d4 + i] = ab.v[a4 + i] * sc;
                }
            }
            return;
        }
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
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer ConstBufUvec2 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufVec4 { vec4 v[]; };
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
layout(buffer_reference, std430, buffer_reference_align = 16) buffer BufUvec4 { uvec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufUvec4 { uvec4 v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t a_va;
    uint64_t b_va;
    uint64_t d_va;
    uint     ne0, ne1, ne2, ne3; // dst shape
    uint     a_ne0, a_ne1, a_ne2, a_ne3;
    uint     a_nb0, a_nb1, a_nb2, a_nb3; // in 32-bit words
    uint     b_nb0, b_nb1, b_nb2, b_nb3; // in 32-bit words
    uint     mode; // 0=CONCAT(dim=p0), 1=PAD, 2=REPEAT, 3=fast 3D uvec4 CONCAT dim=1 (0 divides!)
    uint     p0, p1, p2, p3;
    uint     total;
} pc;

void main() {
    if (pc.mode == 3u) {
        const uint i0_4  = gl_GlobalInvocationID.x;
        const uint ne0_4 = pc.ne0 >> 2u;
        const uint i1    = gl_WorkGroupID.y;
        const uint i2    = gl_WorkGroupID.z;
        if (i0_4 >= ne0_4 || i1 >= pc.ne1 || i2 >= pc.ne2) return;
        const uint d_off4 = (i2 * pc.ne1 + i1) * ne0_4 + i0_4;
        if (i1 < pc.a_ne1) {
            BufUvec4(pc.d_va).v[d_off4] = ConstBufUvec4(pc.a_va).v[(i2 * pc.a_ne1 + i1) * ne0_4 + i0_4];
        } else {
            const uint b_ne1 = pc.ne1 - pc.a_ne1;
            BufUvec4(pc.d_va).v[d_off4] = ConstBufUvec4(pc.b_va).v[(i2 * b_ne1 + (i1 - pc.a_ne1)) * ne0_4 + i0_4];
        }
        return;
    }

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
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer BufVec4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufVec4 { vec4 v[]; };

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

    const bool v4_c = (c_nb0 == 1u) && ((c_base & 3u) == 0u) && ((pc.cur_va & 15ul) == 0ul);
    const bool v4_m = (m_nb0 == 1u) && ((m_base & 3u) == 0u) && ((pc.msg_va & 15ul) == 0ul);
    const bool v4_base = ((D & 3u) == 0u) && ((base & 3u) == 0u) &&
                         (((pc.out_va | pc.forget_va | pc.gain_va | pc.norm_w_va) & 15ul) == 0ul);

    if (pc.mode == 1u) {
        if (v4_base && v4_c) {
            const uint D4 = D >> 2u;
            const uint b4 = base >> 2u;
            const uint c4 = c_base >> 2u;
            ConstBufVec4 cb = ConstBufVec4(pc.cur_va);
            ConstBufVec4 fb = ConstBufVec4(pc.forget_va);
            BufVec4      ob = BufVec4(pc.out_va);
            for (uint i = tid; i < D4; i += 256u) {
                vec4 c = cb.v[c4 + i];
                vec4 f = tanh(fb.v[b4 + i]);
                ob.v[b4 + i] = c - f * c;
            }
            return;
        }
        for (uint i = tid; i < D; i += 256u) {
            float c = ConstBufF32(pc.cur_va).v[c_base + i * c_nb0];
            float f = tanh(ConstBufF32(pc.forget_va).v[base + i]);
            BufF32(pc.out_va).v[base + i] = c - f * c;
        }
        return;
    }

    if (pc.norm_w_va == 0ul) {
        if (v4_base && v4_c && v4_m) {
            const uint D4 = D >> 2u;
            const uint b4 = base >> 2u;
            const uint c4 = c_base >> 2u;
            const uint m4 = m_base >> 2u;
            ConstBufVec4 cb = ConstBufVec4(pc.cur_va);
            ConstBufVec4 mb = ConstBufVec4(pc.msg_va);
            ConstBufVec4 gb = ConstBufVec4(pc.gain_va);
            ConstBufVec4 fb = ConstBufVec4(pc.forget_va);
            BufVec4      ob = BufVec4(pc.out_va);
            for (uint i = tid; i < D4; i += 256u) {
                vec4 c = cb.v[c4 + i];
                vec4 m = mb.v[m4 + i];
                vec4 g = tanh(gb.v[b4 + i]);
                vec4 y = (pc.mode == 2u) ? (c + g * m) : ((c - tanh(fb.v[b4 + i]) * c) + g * m);
                ob.v[b4 + i] = y;
            }
            return;
        }
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

    if (v4_base && v4_c && v4_m) {
        const uint D4 = D >> 2u;
        const uint b4 = base >> 2u;
        const uint c4 = c_base >> 2u;
        const uint m4 = m_base >> 2u;
        ConstBufVec4 cb  = ConstBufVec4(pc.cur_va);
        ConstBufVec4 mb  = ConstBufVec4(pc.msg_va);
        ConstBufVec4 gb  = ConstBufVec4(pc.gain_va);
        ConstBufVec4 fb  = ConstBufVec4(pc.forget_va);
        ConstBufVec4 nwb = ConstBufVec4(pc.norm_w_va);
        BufVec4      ob  = BufVec4(pc.out_va);
        if (D4 <= 1024u) {
            vec4 ry[4];
            float ss = 0.0;
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                uint i = tid + k * 256u;
                vec4 y = vec4(0.0);
                if (i < D4) {
                    vec4 c = cb.v[c4 + i];
                    vec4 m = mb.v[m4 + i];
                    vec4 g = tanh(gb.v[b4 + i]);
                    y = (pc.mode == 2u) ? (c + g * m) : ((c - tanh(fb.v[b4 + i]) * c) + g * m);
                }
                ry[k] = y;
                ss += dot(y, y);
            }
            float sc = inversesqrt(wg_sum(ss, sg, ln) / float(D) + pc.eps_norm);
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                uint i = tid + k * 256u;
                if (i < D4) ob.v[b4 + i] = (ry[k] * sc) * nwb.v[i];
            }
            return;
        }
        float ss = 0.0;
        for (uint i = tid; i < D4; i += 256u) {
            vec4 c = cb.v[c4 + i];
            vec4 m = mb.v[m4 + i];
            vec4 g = tanh(gb.v[b4 + i]);
            vec4 y = (pc.mode == 2u) ? (c + g * m) : ((c - tanh(fb.v[b4 + i]) * c) + g * m);
            ob.v[b4 + i] = y;
            ss += dot(y, y);
        }
        float sc = inversesqrt(wg_sum(ss, sg, ln) / float(D) + pc.eps_norm);
        for (uint i = tid; i < D4; i += 256u) {
            ob.v[b4 + i] = (ob.v[b4 + i] * sc) * nwb.v[i];
        }
        return;
    }

    ConstBufF32 cbf  = ConstBufF32(pc.cur_va);
    ConstBufF32 mbf  = ConstBufF32(pc.msg_va);
    ConstBufF32 gbf  = ConstBufF32(pc.gain_va);
    ConstBufF32 fbf  = ConstBufF32(pc.forget_va);
    ConstBufF32 nwbf = ConstBufF32(pc.norm_w_va);
    BufF32      obf  = BufF32(pc.out_va);
    if (D <= 2560u) {
        float ry[10];
        float ss = 0.0;
        [[unroll]] for (uint k = 0u; k < 10u; ++k) {
            uint i = tid + k * 256u;
            float y = 0.0;
            if (i < D) {
                float c = cbf.v[c_base + i * c_nb0];
                float m = mbf.v[m_base + i * m_nb0];
                float g = tanh(gbf.v[base + i]);
                y = (pc.mode == 2u) ? (c + g * m) : ((c - tanh(fbf.v[base + i]) * c) + g * m);
            }
            ry[k] = y;
            ss += y * y;
        }
        float sc = inversesqrt(wg_sum(ss, sg, ln) / float(D) + pc.eps_norm);
        [[unroll]] for (uint k = 0u; k < 10u; ++k) {
            uint i = tid + k * 256u;
            if (i < D) obf.v[base + i] = (ry[k] * sc) * nwbf.v[i];
        }
        return;
    }
    float ss = 0.0;
    for (uint i = tid; i < D; i += 256u) {
        float c = cbf.v[c_base + i * c_nb0];
        float m = mbf.v[m_base + i * m_nb0];
        float g = tanh(gbf.v[base + i]);
        float y = (pc.mode == 2u) ? (c + g * m) : ((c - tanh(fbf.v[base + i]) * c) + g * m);
        obf.v[base + i] = y;
        ss += y * y;
    }
    float sc = inversesqrt(wg_sum(ss, sg, ln) / float(D) + pc.eps_norm);
    for (uint i = tid; i < D; i += 256u) {
        obf.v[base + i] = (obf.v[base + i] * sc) * nwbf.v[i];
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

// Fused CONCAT(cur, msg, dim=1) + per-row RMS_NORM(eps) for control() MMA pipeline.
// Supports strided CONT(VIEW_3D) absorption on cur and msg + vec4 contiguous fastpath.
// Grid: (2, n_tokens, 1), Local: (256, 1, 1).
static const char * k_glsl_ctrl_norm_cat = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer BufVec4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufVec4 { vec4 v[]; };

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

    if (s_nb0 == 1u && ((D & 3u) == 0u) && (((s_base | d_base) & 3u) == 0u) &&
        (((src_va | pc.dst_va) & 15ul) == 0ul)) {
        const uint D4 = D >> 2u;
        const uint s4 = s_base >> 2u;
        const uint d4 = d_base >> 2u;
        ConstBufVec4 sb = ConstBufVec4(src_va);
        BufVec4      db = BufVec4(pc.dst_va);
        if (D4 <= 1024u) {
            vec4 rx[4];
            float ss = 0.0;
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                uint i = tid + k * 256u;
                vec4 x = (i < D4) ? sb.v[s4 + i] : vec4(0.0);
                rx[k] = x;
                ss += dot(x, x);
            }
            float sc = inversesqrt(wg_sum(ss, sg, ln) / float(D) + pc.eps);
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                uint i = tid + k * 256u;
                if (i < D4) db.v[d4 + i] = rx[k] * sc;
            }
            return;
        }
        float ss = 0.0;
        for (uint i = tid; i < D4; i += 256u) {
            vec4 x = sb.v[s4 + i];
            ss += dot(x, x);
        }
        float sc = inversesqrt(wg_sum(ss, sg, ln) / float(D) + pc.eps);
        for (uint i = tid; i < D4; i += 256u) {
            db.v[d4 + i] = sb.v[s4 + i] * sc;
        }
        return;
    }

    ConstBufF32 sbf = ConstBufF32(src_va);
    BufF32      dbf = BufF32(pc.dst_va);
    if (D <= 2560u) {
        // Single-pass register-cached path: halves DRAM traffic on strided 72 MiB messages!
        float rx[10];
        float ss = 0.0;
        [[unroll]] for (uint k = 0u; k < 10u; ++k) {
            uint i = tid + k * 256u;
            float x = (i < D) ? sbf.v[s_base + i * s_nb0] : 0.0;
            rx[k] = x;
            ss += x * x;
        }
        float sc = inversesqrt(wg_sum(ss, sg, ln) / float(D) + pc.eps);
        [[unroll]] for (uint k = 0u; k < 10u; ++k) {
            uint i = tid + k * 256u;
            if (i < D) dbf.v[d_base + i] = rx[k] * sc;
        }
        return;
    }

    float ss = 0.0;
    for (uint i = tid; i < D; i += 256u) {
        float x = sbf.v[s_base + i * s_nb0];
        ss += x * x;
    }
    float sc = inversesqrt(wg_sum(ss, sg, ln) / float(D) + pc.eps);
    for (uint i = tid; i < D; i += 256u) {
        dbf.v[d_base + i] = sbf.v[s_base + i * s_nb0] * sc;
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
layout(buffer_reference, std430, buffer_reference_align = 16) buffer BufVec4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufVec4 { vec4 v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t states_va;
    uint64_t extra_va;     // optional last state (eliminates CONCAT(blocks, norm_cur, dim=1))
    uint64_t queries_va;
    uint64_t out_va;
    uint64_t out1_va;      // optional Q>=2 separate contiguous [D, M] output
    uint64_t out2_va;      // optional Q>=3 separate contiguous [D, M] output
    uint64_t norm_w0_va;   // optional trailing RMS_NORM weight for query 0
    uint64_t norm_w1_va;   // optional trailing RMS_NORM weight for query 1
    uint64_t norm_w2_va;   // optional trailing RMS_NORM weight for query 2
    uint     n_embd;
    uint     n_states;     // total states (including extra if extra_va != 0)
    uint     n_queries;
    uint     n_tokens;
    uint     q_ne2;
    uint     extra_do_rms; // 1 = RMS_NORM(extra_va, extra_eps) inline
    float    scale;
    float    extra_eps;
    float    norm_eps;
    uint     norm_flags;   // bit0: separate [D, M] outs, bit1: trailing RMS_NORM * norm_w
} pc;

shared float s_w[4][32];
shared float s_red[8];
shared float s_extra_inv_rms;

void main() {
    const uint tok = gl_WorkGroupID.y;
    if (tok >= pc.n_tokens) return;

    const uint tid = gl_LocalInvocationID.x;
    const uint sg  = tid >> 5u;
    const uint ln  = tid & 31u;
    const uint D   = pc.n_embd;
    const uint S   = pc.n_states;
    const uint Q   = pc.n_queries;
    const bool has_extra = (pc.extra_va != 0ul);
    const uint S_base = has_extra ? (S - 1u) : S;
    const bool fused_q4 = (Q <= 4u) && ((D & 3u) == 0u) &&
                          (((pc.states_va | pc.extra_va | pc.queries_va) & 15ul) == 0ul);

    const uint q_tok  = tok % pc.q_ne2;
    const uint s_tok  = tok * S_base * D;
    const uint e_tok  = tok * D;

    if (fused_q4) {
        // Fast Wave 6/8 multi-query vec4 path: 1 WG computes ALL Q <= 4 queries, reading states ONCE!
        const uint D4 = D >> 2u;
        if (has_extra) {
            if (pc.extra_do_rms != 0u) {
                ConstBufVec4 eb = ConstBufVec4(pc.extra_va + uint64_t(e_tok) * 4ul);
                float ss = 0.0;
                for (uint i4 = tid; i4 < D4; i4 += 256u) {
                    vec4 x = eb.v[i4];
                    ss += dot(x, x);
                }
                float sg_s = subgroupAdd(ss);
                if (ln == 0u) s_red[sg] = sg_s;
                barrier();
                if (sg == 0u) {
                    float tot = subgroupAdd((ln < 8u) ? s_red[ln] : 0.0);
                    if (ln == 0u) s_extra_inv_rms = inversesqrt(tot / float(D) + pc.extra_eps);
                }
                barrier();
            } else if (tid == 0u) {
                s_extra_inv_rms = 1.0;
            }
            barrier();
        }

        const uint64_t q_tok_va = pc.queries_va + uint64_t(q_tok * Q * D) * 4ul;
        ConstBufVec4 qb = ConstBufVec4(q_tok_va);
        for (uint s = sg; s < S; s += 8u) {
            const bool is_ex = has_extra && (s == S_base);
            ConstBufVec4 xb = is_ex
                ? ConstBufVec4(pc.extra_va + uint64_t(e_tok) * 4ul)
                : ConstBufVec4(pc.states_va + uint64_t(s_tok + s * D) * 4ul);
            vec4 acc = vec4(0.0);
            for (uint i4 = ln; i4 < D4; i4 += 32u) {
                vec4 xv = xb.v[i4];
                acc.x += dot(xv, qb.v[i4]);
                if (Q >= 2u) acc.y += dot(xv, qb.v[D4 + i4]);
                if (Q >= 3u) acc.z += dot(xv, qb.v[2u * D4 + i4]);
                if (Q >= 4u) acc.w += dot(xv, qb.v[3u * D4 + i4]);
            }
            float sc = pc.scale * (is_ex ? s_extra_inv_rms : 1.0);
            float d0 = subgroupAdd(acc.x);
            float d1 = (Q >= 2u) ? subgroupAdd(acc.y) : 0.0;
            float d2 = (Q >= 3u) ? subgroupAdd(acc.z) : 0.0;
            float d3 = (Q >= 4u) ? subgroupAdd(acc.w) : 0.0;
            if (ln == 0u) {
                s_w[0][s] = d0 * sc;
                if (Q >= 2u) s_w[1][s] = d1 * sc;
                if (Q >= 3u) s_w[2][s] = d2 * sc;
                if (Q >= 4u) s_w[3][s] = d3 * sc;
            }
        }
        barrier();

        if (sg < Q) {
            float v  = (ln < S) ? s_w[sg][ln] : uintBitsToFloat(0xFF800000u);
            float mx = subgroupMax(v);
            float e  = (ln < S) ? exp(v - mx) : 0.0;
            float sm = subgroupAdd(e);
            if (ln < S) {
                float p = (sm > 0.0) ? (e / sm) : 0.0;
                if (has_extra && ln == S_base) p *= s_extra_inv_rms;
                s_w[sg][ln] = p;
            }
        }
        barrier();

        // Wave 8 Fused Trailing `[CONT(VIEW_3D)] -> RMS_NORM -> MUL(norm_w)` Epilogue!
        if (Q == 1u && (pc.norm_flags & 2u) != 0u) {
            ConstBufVec4 nw0 = ConstBufVec4(pc.norm_w0_va);
            BufVec4      ob0 = BufVec4(pc.out_va + uint64_t(e_tok) * 4ul);
            if (D4 <= 1024u) {
                vec4 ro[4];
                float ss = 0.0;
                for (uint k = 0u; k < 4u; ++k) {
                    const uint i4 = tid + k * 256u;
                    vec4 o0 = vec4(0.0);
                    if (i4 < D4) {
                        for (uint s = 0u; s < S_base; ++s) {
                            o0 += s_w[0][s] * ConstBufVec4(pc.states_va + uint64_t(s_tok + s * D) * 4ul).v[i4];
                        }
                        if (has_extra) {
                            o0 += s_w[0][S_base] * ConstBufVec4(pc.extra_va + uint64_t(e_tok) * 4ul).v[i4];
                        }
                    }
                    ro[k] = o0;
                    ss += dot(o0, o0);
                }
                float sg_s = subgroupAdd(ss);
                if (ln == 0u) s_red[sg] = sg_s;
                barrier();
                if (sg == 0u) {
                    float tot = subgroupAdd((ln < 8u) ? s_red[ln] : 0.0);
                    if (ln == 0u) s_red[0] = inversesqrt(tot / float(D) + pc.norm_eps);
                }
                barrier();
                float inv_rms = s_red[0];
                for (uint k = 0u; k < 4u; ++k) {
                    const uint i4 = tid + k * 256u;
                    if (i4 < D4) {
                        ob0.v[i4] = (ro[k] * inv_rms) * nw0.v[i4];
                    }
                }
                return;
            }
            float ss = 0.0;
            for (uint i4 = tid; i4 < D4; i4 += 256u) {
                vec4 o0 = vec4(0.0);
                for (uint s = 0u; s < S_base; ++s) {
                    o0 += s_w[0][s] * ConstBufVec4(pc.states_va + uint64_t(s_tok + s * D) * 4ul).v[i4];
                }
                if (has_extra) {
                    o0 += s_w[0][S_base] * ConstBufVec4(pc.extra_va + uint64_t(e_tok) * 4ul).v[i4];
                }
                ob0.v[i4] = o0;
                ss += dot(o0, o0);
            }
            float sg_s = subgroupAdd(ss);
            if (ln == 0u) s_red[sg] = sg_s;
            barrier();
            if (sg == 0u) {
                float tot = subgroupAdd((ln < 8u) ? s_red[ln] : 0.0);
                if (ln == 0u) s_red[0] = inversesqrt(tot / float(D) + pc.norm_eps);
            }
            barrier();
            float inv_rms = s_red[0];
            for (uint i4 = tid; i4 < D4; i4 += 256u) {
                ob0.v[i4] = (ob0.v[i4] * inv_rms) * nw0.v[i4];
            }
            return;
        }

        if (Q >= 2u && Q <= 3u && (pc.norm_flags & 3u) != 0u) {
            BufVec4 ob0 = BufVec4(pc.out_va  + uint64_t(e_tok) * 4ul);
            BufVec4 ob1 = BufVec4(pc.out1_va + uint64_t(e_tok) * 4ul);
            BufVec4 ob2 = BufVec4(((Q >= 3u) ? pc.out2_va : pc.out_va) + uint64_t(e_tok) * 4ul);
            float ss0 = 0.0, ss1 = 0.0, ss2 = 0.0;
            for (uint i4 = tid; i4 < D4; i4 += 256u) {
                vec4 o0 = vec4(0.0), o1 = vec4(0.0), o2 = vec4(0.0);
                for (uint s = 0u; s < S_base; ++s) {
                    vec4 xv = ConstBufVec4(pc.states_va + uint64_t(s_tok + s * D) * 4ul).v[i4];
                    o0 += s_w[0][s] * xv;
                    o1 += s_w[1][s] * xv;
                    if (Q >= 3u) o2 += s_w[2][s] * xv;
                }
                if (has_extra) {
                    vec4 xv = ConstBufVec4(pc.extra_va + uint64_t(e_tok) * 4ul).v[i4];
                    o0 += s_w[0][S_base] * xv;
                    o1 += s_w[1][S_base] * xv;
                    if (Q >= 3u) o2 += s_w[2][S_base] * xv;
                }
                ob0.v[i4] = o0;
                ob1.v[i4] = o1;
                if (Q >= 3u) ob2.v[i4] = o2;
                ss0 += dot(o0, o0);
                ss1 += dot(o1, o1);
                if (Q >= 3u) ss2 += dot(o2, o2);
            }
            if ((pc.norm_flags & 2u) != 0u) {
                float r0 = subgroupAdd(ss0);
                float r1 = subgroupAdd(ss1);
                float r2 = subgroupAdd(ss2);
                barrier(); // s_w reads complete -> reuse s_w[0..2][0..7] for 3-way RMS reduction!
                if (ln == 0u) {
                    s_w[0][sg] = r0;
                    s_w[1][sg] = r1;
                    if (Q >= 3u) s_w[2][sg] = r2;
                }
                barrier();
                if (sg < Q) {
                    float tot = subgroupAdd((ln < 8u) ? s_w[sg][ln] : 0.0);
                    if (ln == 0u) s_red[sg] = inversesqrt(tot / float(D) + pc.norm_eps);
                }
                barrier();
                float inv0 = s_red[0];
                float inv1 = s_red[1];
                float inv2 = (Q >= 3u) ? s_red[2] : 0.0;
                ConstBufVec4 nw0 = ConstBufVec4(pc.norm_w0_va);
                ConstBufVec4 nw1 = ConstBufVec4(pc.norm_w1_va);
                ConstBufVec4 nw2 = ConstBufVec4((Q >= 3u) ? pc.norm_w2_va : pc.norm_w0_va);
                for (uint i4 = tid; i4 < D4; i4 += 256u) {
                    ob0.v[i4] = (ob0.v[i4] * inv0) * nw0.v[i4];
                    ob1.v[i4] = (ob1.v[i4] * inv1) * nw1.v[i4];
                    if (Q >= 3u) ob2.v[i4] = (ob2.v[i4] * inv2) * nw2.v[i4];
                }
            }
            return;
        }

        const uint out_tok = tok * D * Q;
        BufF32 ob = BufF32(pc.out_va);
        for (uint i4 = tid; i4 < D4; i4 += 256u) {
            vec4 o0 = vec4(0.0), o1 = vec4(0.0), o2 = vec4(0.0), o3 = vec4(0.0);
            for (uint s = 0u; s < S_base; ++s) {
                vec4 xv = ConstBufVec4(pc.states_va + uint64_t(s_tok + s * D) * 4ul).v[i4];
                o0 += s_w[0][s] * xv;
                if (Q >= 2u) o1 += s_w[1][s] * xv;
                if (Q >= 3u) o2 += s_w[2][s] * xv;
                if (Q >= 4u) o3 += s_w[3][s] * xv;
            }
            if (has_extra) {
                vec4 xv = ConstBufVec4(pc.extra_va + uint64_t(e_tok) * 4ul).v[i4];
                o0 += s_w[0][S_base] * xv;
                if (Q >= 2u) o1 += s_w[1][S_base] * xv;
                if (Q >= 3u) o2 += s_w[2][S_base] * xv;
                if (Q >= 4u) o3 += s_w[3][S_base] * xv;
            }
            if (Q == 1u && ((pc.out_va & 15ul) == 0ul)) {
                BufVec4(pc.out_va + uint64_t(out_tok) * 4ul).v[i4] = o0;
            } else {
                const uint d0 = i4 << 2u;
                ob.v[out_tok + (d0 + 0u) * Q + 0u] = o0.x;
                ob.v[out_tok + (d0 + 1u) * Q + 0u] = o0.y;
                ob.v[out_tok + (d0 + 2u) * Q + 0u] = o0.z;
                ob.v[out_tok + (d0 + 3u) * Q + 0u] = o0.w;
                if (Q >= 2u) {
                    ob.v[out_tok + (d0 + 0u) * Q + 1u] = o1.x;
                    ob.v[out_tok + (d0 + 1u) * Q + 1u] = o1.y;
                    ob.v[out_tok + (d0 + 2u) * Q + 1u] = o1.z;
                    ob.v[out_tok + (d0 + 3u) * Q + 1u] = o1.w;
                }
                if (Q >= 3u) {
                    ob.v[out_tok + (d0 + 0u) * Q + 2u] = o2.x;
                    ob.v[out_tok + (d0 + 1u) * Q + 2u] = o2.y;
                    ob.v[out_tok + (d0 + 2u) * Q + 2u] = o2.z;
                    ob.v[out_tok + (d0 + 3u) * Q + 2u] = o2.w;
                }
                if (Q >= 4u) {
                    ob.v[out_tok + (d0 + 0u) * Q + 3u] = o3.x;
                    ob.v[out_tok + (d0 + 1u) * Q + 3u] = o3.y;
                    ob.v[out_tok + (d0 + 2u) * Q + 3u] = o3.z;
                    ob.v[out_tok + (d0 + 3u) * Q + 3u] = o3.w;
                }
            }
        }
        return;
    }

    const uint q = gl_WorkGroupID.x;
    if (q >= Q) return;
    const uint q_base = (q_tok * Q + q) * D;

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

    for (uint s = sg; s < S; s += 8u) {
        float acc = 0.0;
        if (has_extra && s == S_base) {
            for (uint d = ln; d < D; d += 32u) {
                acc += ConstBufF32(pc.extra_va).v[e_tok + d] * ConstBufF32(pc.queries_va).v[q_base + d];
            }
            float dot_val = subgroupAdd(acc) * s_extra_inv_rms;
            if (ln == 0u) s_w[0][s] = dot_val * pc.scale;
        } else {
            const uint sb = s_tok + s * D;
            for (uint d = ln; d < D; d += 32u) {
                acc += ConstBufF32(pc.states_va).v[sb + d] * ConstBufF32(pc.queries_va).v[q_base + d];
            }
            float dot_val = subgroupAdd(acc);
            if (ln == 0u) s_w[0][s] = dot_val * pc.scale;
        }
    }
    barrier();

    if (sg == 0u) {
        float v  = (ln < S) ? s_w[0][ln] : uintBitsToFloat(0xFF800000u);
        float mx = subgroupMax(v);
        float e  = (ln < S) ? exp(v - mx) : 0.0;
        float sm = subgroupAdd(e);
        if (ln < S) {
            float p = (sm > 0.0) ? (e / sm) : 0.0;
            if (has_extra && ln == S_base) {
                p *= s_extra_inv_rms;
            }
            s_w[0][ln] = p;
        }
    }
    barrier();

    const uint out_tok = tok * D * Q;
    for (uint d = tid; d < D; d += 256u) {
        float acc = 0.0;
        for (uint s = 0u; s < S_base; ++s) {
            acc += s_w[0][s] * ConstBufF32(pc.states_va).v[s_tok + s * D + d];
        }
        if (has_extra) {
            acc += s_w[0][S_base] * ConstBufF32(pc.extra_va).v[e_tok + d];
        }
        BufF32(pc.out_va).v[out_tok + d * Q + q] = acc;
    }
}
)GLSL";

struct mix_attn_push {
    uint64_t states_va, extra_va, queries_va, out_va;
    uint64_t out1_va, out2_va;
    uint64_t norm_w0_va, norm_w1_va, norm_w2_va;
    uint32_t n_embd, n_states, n_queries, n_tokens, q_ne2, extra_do_rms;
    float    scale, extra_eps;
    float    norm_eps;
    uint32_t norm_flags;
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

    const uint tok     = seq * pc.n_t + t;
    const uint n_conv  = pc.n_conv;
    const uint n_hist  = n_conv - 1u;
    const uint p_base  = seq * pc.past_nb2 + ch * pc.past_nb1;
    const uint w_base  = ch * pc.w_nb1;
    const bool has_dyn = (pc.dyn_va != 0ul);
    uint d_base = 0u;
    if (has_dyn) {
        const uint head = ((pc.head_dim & (pc.head_dim - 1u)) == 0u)
            ? (ch >> findLSB(pc.head_dim))
            : (ch / pc.head_dim);
        d_base = (tok * pc.n_heads + head) * n_conv;
    }

    float x_cur = ConstBufF32(pc.cur_va).v[tok * pc.n_ch + ch];
    float sum   = 0.0;
    for (uint lag = 0u; lag < n_conv; ++lag) {
        float x_lag = (lag == 0u) ? x_cur : (
            (t >= lag)
                ? ConstBufF32(pc.cur_va).v[(tok - lag) * pc.n_ch + ch]
                : ConstBufF32(pc.past_va).v[p_base + (n_hist - (lag - t))]);
        sum += x_lag * ConstBufF32(pc.w_va).v[w_base + (n_hist - lag)];
        if (has_dyn) {
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

// Wave 8 Fused [K_conv / V_conv + L2_CLIP / RMS_NORM + [tau] + [FWHT64/128] + [SET_ROWS(F32/F16/Q8_0)]]!
// Eliminates 96 dispatches, 48-96 strong_usc barriers, and 1.61 GB F32 DRAM traffic on pp2048!
static const char * k_glsl_sconv_prep_store = R"GLSL(
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

layout(push_constant) uniform PushArgs {
    uint64_t past_va;
    uint64_t cur_va;
    uint64_t w_va;
    uint64_t dyn_va;
    uint64_t cell_va;
    uint64_t scale_va;
    uint64_t tau_va;
    uint64_t ids_va;
    uint64_t dst_va;
    uint     past_nb1;
    uint     past_nb2;
    uint     w_nb1;
    uint     cell_nb1;
    uint     n_conv;
    uint     head_dim;
    uint     n_head;
    uint     n_t;
    uint     n_s;
    uint     dyn_heads;
    uint     flags;      // bit0=L2_CLIP, bit1=RMS_NORM, bit2=FWHT_64, bit3=FWHT_128
    uint     store_mode; // 0=F32_CONT, 1=SET_ROWS_F32, 2=SET_ROWS_F16, 3=SET_ROWS_Q8_0
    uint     ids_is_i64;
    uint     i_nb0;
    uint     d_nb1;
    float    res_scale;
    float    res_bias;
    float    param;
} pc;

void main() {
    const uint lane     = gl_LocalInvocationID.x;
    const uint head     = gl_WorkGroupID.x * 4u + gl_LocalInvocationID.y;
    const uint tok      = gl_WorkGroupID.y;
    const uint n_tokens = pc.n_t * pc.n_s;
    if (head >= pc.n_head || tok >= n_tokens) return;

    const uint D       = pc.head_dim;
    const uint n_ch    = pc.n_head * D;
    const uint seq     = (pc.n_s <= 1u) ? 0u  : (tok / pc.n_t);
    const uint t       = (pc.n_s <= 1u) ? tok : (tok % pc.n_t);
    const uint n_conv  = pc.n_conv;
    const uint n_hist  = n_conv - 1u;
    const bool has_dyn = (pc.dyn_va != 0ul);
    const uint dyn_dim = has_dyn ? (n_ch / pc.dyn_heads) : n_ch;

    float reg[4];
    [[unroll]] for (uint k = 0u; k < 4u; ++k) {
        const uint d = k * 32u + lane;
        if (d < D) {
            const uint ch     = head * D + d;
            const uint p_base = seq * pc.past_nb2 + ch * pc.past_nb1;
            const uint w_base = ch * pc.w_nb1;
            const uint d_head = (pc.dyn_heads == pc.n_head) ? head : (ch / dyn_dim);
            const uint d_base = has_dyn ? ((tok * pc.dyn_heads + d_head) * n_conv) : 0u;

            float x_cur = ConstBufF32(pc.cur_va).v[tok * n_ch + ch];
            float sum   = 0.0;
            for (uint lag = 0u; lag < n_conv; ++lag) {
                float x_lag = (lag == 0u) ? x_cur : (
                    (t >= lag)
                        ? ConstBufF32(pc.cur_va).v[(tok - lag) * n_ch + ch]
                        : ConstBufF32(pc.past_va).v[p_base + (n_hist - (lag - t))]);
                sum += x_lag * ConstBufF32(pc.w_va).v[w_base + (n_hist - lag)];
                if (has_dyn) {
                    sum += x_lag * ConstBufF32(pc.dyn_va).v[d_base + lag];
                }
            }

            if (pc.cell_va != 0ul && t == pc.n_t - 1u) {
                uint lim    = min(n_hist, 8u);
                uint c_base = seq * pc.cell_nb1 + ch * n_hist;
                for (uint s = 0u; s < lim; ++s) {
                    uint lag = (n_hist - 1u) - s;
                    float hv = (t >= lag)
                        ? ConstBufF32(pc.cur_va).v[(tok - lag) * n_ch + ch]
                        : ConstBufF32(pc.past_va).v[p_base + (n_hist - (lag - t))];
                    BufF32(pc.cell_va).v[c_base + s] = hv;
                }
            }

            float v = (x_cur + sum) * pc.res_scale + pc.res_bias;
            if (pc.scale_va != 0ul) {
                v *= ConstBufF32(pc.scale_va).v[ch];
            }
            reg[k] = v;
        } else {
            reg[k] = 0.0;
        }
    }

    if ((pc.flags & 3u) != 0u) {
        float s0 = subgroupAdd(reg[0] * reg[0]);
        float s1 = (D > 32u) ? subgroupAdd(reg[1] * reg[1]) : 0.0;
        float s2 = (D > 64u) ? subgroupAdd(reg[2] * reg[2]) : 0.0;
        float s3 = (D > 96u) ? subgroupAdd(reg[3] * reg[3]) : 0.0;
        float sum_sq = (s0 + s1) + (s2 + s3);
        if ((pc.flags & 1u) != 0u) {
            float inv_norm = 1.0 / max(sqrt(sum_sq), pc.param);
            [[unroll]] for (uint k = 0u; k < 4u; ++k) {
                precise float tv = reg[k] * inv_norm;
                reg[k] = tv * pc.param;
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

struct sconv_prep_store_push {
    uint64_t past_va, cur_va, w_va, dyn_va, cell_va, scale_va, tau_va, ids_va, dst_va;
    uint32_t past_nb1, past_nb2, w_nb1, cell_nb1;
    uint32_t n_conv, head_dim, n_head, n_t, n_s, dyn_heads;
    uint32_t flags, store_mode, ids_is_i64, i_nb0, d_nb1;
    float    res_scale, res_bias, param;
};
static_assert(sizeof(sconv_prep_store_push) <= AGX_MAX_PUSH_BYTES, "sconv_prep_store_push > 256B");

// Wave 10 Pre-Scan of Exact Active [kv_beg, kv_end) Tile Bounds per q_br (16 or 32) Query Tile
// Runs ONCE per unique 1-head mask in Wave 0 (2 dispatches total = 0.04 ms for all 48 Inkling 2 layers),
// with unlimited-n_kv span_no_holes detection (2048-KV chunk reduction -> works up to 1M+ ctx, eliminating
// 2 barriers/tile + 100% redundant mask reads at 32k/250K+)!
static const char * k_glsl_fa_mask_bounds = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 2) readonly  buffer ConstBufF16   { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly  buffer ConstBufF32   { float     v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly  buffer ConstBufUvec2 { uvec2     v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufVec4  { vec4      v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) writeonly buffer BufUvec2W     { uvec2     v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t mask_va;
    uint64_t bounds_va;
    uint     n_kv;
    uint     n_tokens;
    uint     mask_nb1;
    uint     is_f16;
    uint     q_br;
    uint     _pad0;
} pc;

shared uint s_min[4];
shared uint s_max[4];
shared uint s_bits[4];

void main() {
    const uint q_br   = (pc.q_br > 16u) ? 32u : 16u;
    const uint q_tile = gl_WorkGroupID.x;
    const uint q0     = q_tile * q_br;
    const uint tid    = gl_LocalInvocationID.x;
    const uint sg     = tid >> 5u;
    const uint ql0    = tid >> 3u;       // 0..15
    const uint kl4    = (tid & 7u) * 4u; // 0, 4, 8, 12, 16, 20, 24, 28 (32 KV/iter)

    uint my_min          = 0xFFFFFFFFu;
    uint my_max          = 0u;
    uint total_valid_t64 = 0u;

    const bool vec_f16  = (pc.is_f16 != 0u) && ((pc.mask_va & 7ul) == 0ul) && ((pc.mask_nb1 & 7u) == 0u);
    const bool vec_f32  = (pc.is_f16 == 0u) && ((pc.mask_va & 15ul) == 0ul) && ((pc.mask_nb1 & 15u) == 0u);
    const uint r_passes = (q_br > 16u) ? 2u : 1u;

    // 2048-KV chunks (32 x 64-KV tiles per chunk -> 1 uint32 bitmask, exact span_no_holes up to 1M+ ctx!)
    for (uint chunk0 = 0u; chunk0 < pc.n_kv; chunk0 += 2048u) {
        const uint chunk_end = min(chunk0 + 2048u, pc.n_kv);
        uint my_bits = 0u;
        for (uint rp = 0u; rp < r_passes; ++rp) {
            const uint q_row = q0 + ql0 + rp * 16u;
            if (q_row < pc.n_tokens) {
                const uint64_t rva = pc.mask_va + uint64_t(q_row) * uint64_t(pc.mask_nb1);
                for (uint kv0 = chunk0; kv0 < chunk_end; kv0 += 32u) {
                    const uint rem = pc.n_kv - kv0;
                    bool ok = false;
                    if (rem >= 32u && vec_f16) {
                        uvec2 p = ConstBufUvec2(rva + uint64_t(kv0 + kl4) * 2ul).v[0];
                        vec4 mv = vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y));
                        ok = (mv.x > -1.0e30) || (mv.y > -1.0e30) || (mv.z > -1.0e30) || (mv.w > -1.0e30);
                    } else if (rem >= 32u && vec_f32) {
                        vec4 mv = ConstBufVec4(rva + uint64_t(kv0 + kl4) * 4ul).v[0];
                        ok = (mv.x > -1.0e30) || (mv.y > -1.0e30) || (mv.z > -1.0e30) || (mv.w > -1.0e30);
                    } else {
                        for (uint c = 0u; c < 4u; ++c) {
                            if (kl4 + c < rem) {
                                float v = (pc.is_f16 != 0u) ? float(ConstBufF16(rva).v[kv0 + kl4 + c])
                                                            : ConstBufF32(rva).v[kv0 + kl4 + c];
                                if (v > -1.0e30) ok = true;
                            }
                        }
                    }
                    if (ok) {
                        my_min = min(my_min, kv0);
                        my_max = max(my_max, min(kv0 + 32u, pc.n_kv));
                        my_bits |= (1u << ((kv0 - chunk0) >> 6u));
                    }
                }
            }
        }
        uint b_or = subgroupOr(my_bits);
        if ((tid & 31u) == 0u) {
            s_bits[sg] = b_or;
        }
        barrier();
        if (tid == 0u) {
            uint u = s_bits[0] | s_bits[1] | s_bits[2] | s_bits[3];
            total_valid_t64 += uint(bitCount(u));
        }
        if (chunk_end < pc.n_kv) {
            barrier();
        }
    }

    uint sg_min = subgroupMin(my_min);
    uint sg_max = subgroupMax(my_max);
    if ((tid & 31u) == 0u) {
        s_min[sg] = sg_min;
        s_max[sg] = sg_max;
    }
    barrier();

    if (tid == 0u) {
        uint gmin = min(min(s_min[0], s_min[1]), min(s_min[2], s_min[3]));
        uint gmax = max(max(s_max[0], s_max[1]), max(s_max[2], s_max[3]));
        if (gmin >= gmax) {
            gmin = pc.n_kv;
            gmax = 0u;
        } else {
            const uint span_tiles = ((gmax + 63u) >> 6u) - (gmin >> 6u);
            if (total_valid_t64 == span_tiles) {
                gmax |= 0x80000000u; // contiguous active 64-KV span (zero -INF hole tiles!)
            }
        }
        BufUvec2W(pc.bounds_va).v[q_tile] = uvec2(gmin, gmax);
    }
}
)GLSL";

struct fa_mask_bounds_push {
    uint64_t mask_va, bounds_va;
    uint32_t n_kv, n_tokens, mask_nb1, is_f16, q_br, _pad0;
};
static_assert(sizeof(fa_mask_bounds_push) <= AGX_MAX_PUSH_BYTES, "fa_mask_bounds_push > 256B");

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
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer BufF16 { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer BufUvec2W { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufI32 { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufIvec4 { ivec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer ConstBufU16 { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) readonly buffer ConstBufF16 { float16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer ConstBufUvec2 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer ConstBufVec4 { vec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) readonly buffer ConstBufI8 { int8_t v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t rcur_va;
    uint64_t bias_va;
    uint64_t sinks_va;
    uint64_t rel_pos_va;
    uint64_t mask_va;    // optional [n_kv, n_t, mask_ne2, mask_ne3] F16/F32
    uint64_t dst_va;
    uint64_t bounds_va;  // optional Wave-0 uvec2[n_q_tiles] active [kv_beg, kv_end) bounds!
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
    uint     mask_ne2;   // 1 or n_head
    uint     mask_ne3;   // 1 or n_stream
    uint     flags;      // bit0=mask_is_f16, bit1=dst_is_f16, bit2=vec4_f16_fastpath
    uint     pad0;
} pc;

shared float s_r[64];
shared vec4  s_r4[16];
shared float s_sink;

float eval_rel_dot(int dist) {
    if (dist < 0 || uint(dist) >= pc.n_extent) return 0.0;
    uint64_t row_va = pc.bias_va + uint64_t(dist) * uint64_t(pc.bias_nb1);
    const bool vec4_ok = ((pc.n_rel & 3u) == 0u) && ((row_va & 7ul) == 0ul);
    float acc = 0.0;
    if (pc.bias_wtype == 0u) {
        if (vec4_ok && ((row_va & 15ul) == 0ul)) {
            const uint n4 = pc.n_rel >> 2u;
            ConstBufVec4 rb = ConstBufVec4(row_va);
            for (uint g = 0u; g < n4; ++g) {
                acc += dot(rb.v[g], s_r4[g]);
            }
        } else {
            for (uint k = 0u; k < pc.n_rel; ++k) {
                acc += ConstBufF32(row_va).v[k] * s_r[k];
            }
        }
    } else if (pc.bias_wtype == 1u) {
        if (vec4_ok) {
            const uint n4 = pc.n_rel >> 2u;
            ConstBufUvec2 rb = ConstBufUvec2(row_va);
            for (uint g = 0u; g < n4; ++g) {
                uvec2 p = rb.v[g];
                acc += dot(vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y)), s_r4[g]);
            }
        } else {
            for (uint k = 0u; k < pc.n_rel; ++k) {
                acc += unpackHalf2x16(uint(ConstBufU16(row_va).v[k])).x * s_r[k];
            }
        }
    } else if (pc.bias_wtype == 2u) {
        if (vec4_ok) {
            const uint n4 = pc.n_rel >> 2u;
            ConstBufUvec2 rb = ConstBufUvec2(row_va);
            for (uint g = 0u; g < n4; ++g) {
                uvec2 p = rb.v[g];
                vec4 w = vec4(
                    uintBitsToFloat((p.x & 0xFFFFu) << 16u),
                    uintBitsToFloat(p.x & 0xFFFF0000u),
                    uintBitsToFloat((p.y & 0xFFFFu) << 16u),
                    uintBitsToFloat(p.y & 0xFFFF0000u));
                acc += dot(w, s_r4[g]);
            }
        } else {
            for (uint k = 0u; k < pc.n_rel; ++k) {
                acc += uintBitsToFloat(uint(ConstBufU16(row_va).v[k]) << 16u) * s_r[k];
            }
        }
    } else {
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
    return acc;
}

void main() {
    const uint tid   = gl_LocalInvocationID.x;
    const uint kv    = gl_GlobalInvocationID.x;
    const uint h     = gl_WorkGroupID.y;
    const uint tok   = gl_WorkGroupID.z;
    const uint n_tok = pc.n_t * pc.n_stream;
    if (h >= pc.n_head || tok >= n_tok) return;

    // Uniform WG early-exit when entire 512-KV / 128-KV WG falls outside PASS_MODE=3's [kv_beg, kv_end)
    // (eliminates 9.22 GB of dead 0xFC00FC00u DRAM writes + ~700K WG barriers across 40 SWA + 8 Full layers!)
    uint bnd_beg = 0u;
    uint bnd_end = pc.n_kv;
    if (pc.bounds_va != 0ul) {
        uvec2 bnd = ConstBufUvec2(pc.bounds_va).v[tok >> 4u];
        bnd_beg = bnd.x;
        bnd_end = bnd.y & 0x7FFFFFFFu;
        const uint wg_span = ((pc.flags & 4u) != 0u) ? 512u : 128u;
        const uint wg_kv0  = gl_WorkGroupID.x * wg_span;
        if (bnd_beg >= bnd_end || wg_kv0 + wg_span <= bnd_beg || wg_kv0 >= bnd_end) {
            return;
        }
    }

    const uint r_base = (tok * pc.n_head + h) * pc.n_rel;
    if (tid < pc.n_rel) {
        s_r[tid] = ConstBufF32(pc.rcur_va).v[r_base + tid];
    }
    if (tid < (pc.n_rel >> 2u)) {
        const uint rb = r_base + (tid << 2u);
        ConstBufF32 rcf = ConstBufF32(pc.rcur_va);
        s_r4[tid] = vec4(rcf.v[rb], rcf.v[rb + 1u], rcf.v[rb + 2u], rcf.v[rb + 3u]);
    }
    if (tid == 0u) {
        s_sink = ConstBufF32(pc.sinks_va).v[tok * pc.n_head + h];
    }
    barrier();

    const uint m = (pc.n_stream == 1u) ? tok : (tok % pc.n_t);
    const uint s = (pc.n_stream == 1u) ? 0u  : (tok / pc.n_t);

    if ((pc.flags & 4u) != 0u) {
        const uint n_kv4 = pc.n_kv >> 2u;
        if (kv >= n_kv4 || (kv << 2u) + 4u <= bnd_beg || (kv << 2u) >= bnd_end) return;
        const uint out4 = (((s * pc.n_head + h) * pc.n_t + m) * n_kv4) + kv;
        const uint hm = (pc.mask_ne2 > 1u) ? h : 0u;
        const uint sm = (pc.mask_ne3 > 1u) ? s : 0u;
        const uint m4 = (((sm * pc.mask_ne2 + hm) * pc.n_t + m) * n_kv4) + kv;
        vec4 mv;
        if ((pc.flags & 1u) != 0u) {
            uvec2 mp = ConstBufUvec2(pc.mask_va).v[m4];
            mv = vec4(unpackHalf2x16(mp.x), unpackHalf2x16(mp.y));
        } else {
            mv = ConstBufVec4(pc.mask_va).v[m4];
        }
        if (mv.x < -1.0e30 && mv.y < -1.0e30 && mv.z < -1.0e30 && mv.w < -1.0e30) {
            BufUvec2W(pc.dst_va).v[out4] = uvec2(0xFC00FC00u, 0xFC00FC00u);
            return;
        }
        ivec4 d4 = ConstBufIvec4(pc.rel_pos_va + uint64_t(tok) * pc.rpos_nb1).v[kv];
        vec4 ov = mv;
        for (uint j = 0u; j < 4u; ++j) {
            float m_j = mv[j];
            if (m_j >= -1.0e30) {
                ov[j] = (eval_rel_dot(d4[j]) - s_sink) + m_j;
            }
        }
        BufUvec2W(pc.dst_va).v[out4] = uvec2(packHalf2x16(ov.xy), packHalf2x16(ov.zw));
        return;
    }

    if (kv >= pc.n_kv || kv < bnd_beg || kv >= bnd_end) return;

    const uint out_idx = ((s * pc.n_head + h) * pc.n_t + m) * pc.n_kv + kv;
    const bool dst_f16 = (pc.flags & 2u) != 0u;

    float mask_v = 0.0;
    if (pc.mask_va != 0ul) {
        const uint hm = (pc.mask_ne2 > 1u) ? h : 0u;
        const uint sm = (pc.mask_ne3 > 1u) ? s : 0u;
        const uint m_idx = ((sm * pc.mask_ne2 + hm) * pc.n_t + m) * pc.n_kv + kv;
        mask_v = ((pc.flags & 1u) != 0u)
            ? float(ConstBufF16(pc.mask_va).v[m_idx])
            : ConstBufF32(pc.mask_va).v[m_idx];
        // Exact -inf early exit: finite + (-inf) == -inf (skips 50%-62.5% of causal/SWA matrix!)
        if (mask_v < -1.0e30) {
            if (dst_f16) BufF16(pc.dst_va).v[out_idx] = float16_t(mask_v);
            else         BufF32(pc.dst_va).v[out_idx] = mask_v;
            return;
        }
    }

    int dist = (pc.rpos_nb0 == 4u)
        ? ConstBufI32(pc.rel_pos_va + uint64_t(tok) * pc.rpos_nb1).v[kv]
        : ConstBufI32(pc.rel_pos_va + uint64_t(tok) * pc.rpos_nb1 + uint64_t(kv) * pc.rpos_nb0).v[0];
    float acc = eval_rel_dot(dist);

    float out_v = (acc - s_sink) + mask_v;
    if (dst_f16) BufF16(pc.dst_va).v[out_idx] = float16_t(out_v);
    else         BufF32(pc.dst_va).v[out_idx] = out_v;
}
)GLSL";

struct rel_bias_gather_push {
    uint64_t rcur_va, bias_va, sinks_va, rel_pos_va, mask_va, dst_va, bounds_va;
    uint32_t n_rel, n_extent, n_kv, n_head, n_t, n_stream, bias_wtype, bias_nb1, rpos_nb0, rpos_nb1;
    uint32_t mask_ne2, mask_ne3, flags, pad0;
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
    ITEM_SCONV_PREP_STORE,
    ITEM_ROUTER_TOPK,
    ITEM_REL_BIAS,
    ITEM_REL_SAVE_TMP,
    ITEM_DSPARK_MARKOV_W4,
    ITEM_FUSED_MOE,
    ITEM_FA_MASK_BOUNDS,
    ITEM_FLASH_ATTN,
    ITEM_GEMM_REDIRECT,
};

struct agx_dspark_w4_item_args {
    agx_dspark_mtp_tensors w;
    uint64_t anchor_toks_va = 0;
    uint64_t base_logits_va = 0;
    uint64_t hidden_va      = 0;
    uint64_t f_rnn_va       = 0;
    uint64_t rnn_state_va   = 0;
    uint64_t out_logits_va  = 0;
    uint64_t out_h_nextn_va = 0;
    uint32_t n_blocks       = 1;
    uint32_t block_drafts   = 1;
};

struct agx_exec_item {
    agx_exec_kind        kind               = ITEM_GENERIC_NODE;
    struct ggml_tensor * node               = nullptr;
    uint32_t             gx = 1, gy = 1, gz = 1;
    uint32_t             nodes_covered      = 1;
    agx_va_range         reads[12]          = {};
    uint8_t              n_reads            = 0;
    agx_va_range         writes[4]          = {};
    uint8_t              n_writes           = 0;
    bool                 uses_moe_scratch   = false;
    bool                 uses_attn_scratch  = false;
    bool                 uses_alias_scratch = false;
    alignas(8) uint8_t   push[160]          = {};
    uint32_t             push_bytes         = 0;
    agx_ctrl_gate_desc   ctrl_desc          = {};
    uint64_t             va0 = 0, va1 = 0;
    uint32_t             u0 = 0, u1 = 0, u2 = 0;
    float                f0 = 0.0f, f1 = 0.0f;
    agx_dspark_w4_item_args dspark_w4       = {};
    agx_moe_stage_args      moe_args        = {};
    agx_ggml_flash_attn_epi fa_epi          = {};
    size_t                  alias_tmp_need_B = 0;
    uint32_t                alias_tmp_off_B  = 0; // byte offset in alias_tmp_bo (e.g. sinks slot)
    uint32_t                alias_floats0    = 0; // element count for alias copy 0 (or dst copy)
    uint32_t                alias_floats1    = 0; // element count for alias copy 1 (sinks)
    uint8_t                 alias_flags      = 0; // item-specific alias trampoline bitmask

    void add_read(agx_va_range r) {
        if (r.beg < r.end && n_reads < 12u) reads[n_reads++] = r;
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
    const agx_kernel * k_sconv_prep_store    = nullptr;
    const agx_kernel * k_router_topk_norm    = nullptr;
    const agx_kernel * k_rel_bias_gather     = nullptr;
    const agx_kernel * k_fa_mask_bounds      = nullptr;

    agx_ptr_count_map           use_map;
    agx_ptr_node_map            first_use;
    agx_ptr_idx_map             exec_idx_map;
    std::vector<int>            exec_nodes;
    std::vector<agx_exec_item>  items;
    std::vector<agx_exec_item>  ordered_items;
    agx_inkling2_engine *       inkling2       = nullptr;
    agx_bo *                    alias_tmp_bo   = nullptr;
    size_t                      alias_tmp_cap_B = 0;
    agx_bo *                    fa_bounds_bo   = nullptr;
    size_t                      fa_bounds_cap_B = 0;
    agx_bo *                    redirect_tmp_bo = nullptr;
    size_t                      redirect_tmp_cap_B = 0;
    agx_bo *                    ctrl_pipe_bo   = nullptr;
    size_t                      ctrl_pipe_cap_B = 0;
    agx_graph_exec_stats        stats          = {};
    bool                        initialized    = false;
    bool                        precompiled_llm = false;
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
        st->k_sconv_prep_store     = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_sconv_prep_store",     k_glsl_sconv_prep_store,     opts);
        st->k_router_topk_norm     = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_router_topk_norm",     k_glsl_router_topk_norm,     opts);
        st->k_rel_bias_gather      = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_rel_bias_gather_w9",   k_glsl_rel_bias_gather,      opts);
        st->k_fa_mask_bounds       = agx_compiler_compile_glsl(ctx->compiler, "agx_ge_fa_mask_bounds_w10",   k_glsl_fa_mask_bounds,       opts);
        st->initialized            = (st->k_bin_op && st->k_unary_op && st->k_cpy_cast && st->k_row_reduce &&
                                      st->k_softmax && st->k_get_rows && st->k_set_rows && st->k_concat_pad &&
                                      st->k_ssm_conv && st->k_rope && st->k_topk_argsort && st->k_glu &&
                                      st->k_add_id && st->k_ctrl_mix && st->k_ctrl_norm_cat &&
                                      st->k_mix_attn && st->k_moe_wsum && st->k_prep_norm_fwht_store &&
                                      st->k_full_shortconv && st->k_sconv_prep_store &&
                                      st->k_router_topk_norm && st->k_rel_bias_gather && st->k_fa_mask_bounds);
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
    (void) params;
    if (!ctx || !ctx->drm || !ctx->compiler || !cgraph || cgraph->n_nodes < 256) {
        return;
    }
    const char * no_pre = std::getenv("GGML_AGX_NO_PRECOMPILE");
    if (no_pre && std::atoi(no_pre) != 0) {
        return;
    }

    auto * st = get_exec_state(ctx);
    if (!st) {
        return;
    }

    // 1. Pre-compile ALL hot prefill/decode shaders ONCE during llama_context::graph_reserve()
    //    so the first timed prompt (e.g. Code 1k after a 32-token warmup) pays ZERO cold compile tax!
    if (!st->precompiled_llm) {
        st->precompiled_llm = true;
        agx_gemm_precompile_all(ctx);

        bool has_moe = false;
        struct fa_dt_key {
            uint32_t     head_dim;
            agx_kv_dtype k_dt;
            agx_kv_dtype v_dt;
        };
        fa_dt_key fa_keys[4] = {};
        int       n_fa_keys  = 0;

        auto map_kv_dt = [](enum ggml_type t, agx_kv_dtype * out) -> bool {
            switch (t) {
                case GGML_TYPE_F8_E4M3: *out = AGX_KV_F8_E4M3;   return true;
                case GGML_TYPE_F16:     *out = AGX_KV_F16;       return true;
                case GGML_TYPE_F32:     *out = AGX_KV_F32;       return true;
                case GGML_TYPE_Q8_0:    *out = AGX_KV_Q8_0_GGML; return true;
                default:                return false;
            }
        };

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const struct ggml_tensor * nd = cgraph->nodes[i];
            if (!nd) continue;
            if (nd->op == GGML_OP_MUL_MAT_ID) {
                has_moe = true;
            } else if (nd->op == GGML_OP_FLASH_ATTN_EXT && nd->src[0] && nd->src[1] && nd->src[2]) {
                const uint32_t d = static_cast<uint32_t>(nd->src[0]->ne[0]);
                agx_kv_dtype k_dt, v_dt;
                if ((d == 64u || d == 128u) &&
                    map_kv_dt(nd->src[1]->type, &k_dt) &&
                    map_kv_dt(nd->src[2]->type, &v_dt)) {
                    bool seen = false;
                    for (int k = 0; k < n_fa_keys; ++k) {
                        if (fa_keys[k].head_dim == d && fa_keys[k].k_dt == k_dt && fa_keys[k].v_dt == v_dt) {
                            seen = true;
                            break;
                        }
                    }
                    if (!seen && n_fa_keys < 4) {
                        fa_keys[n_fa_keys++] = { d, k_dt, v_dt };
                    }
                }
            }
        }

        if (has_moe) {
            if (!ctx->moe_state) {
                ctx->moe_state = agx_moe_create(ctx->drm, ctx->compiler);
            }
            if (ctx->moe_state) {
                agx_moe_precompile_inkling2(static_cast<agx_moe_runner *>(ctx->moe_state));
            }
        }

        if (n_fa_keys > 0) {
            if (!ctx->kv_hybrid_state) {
                agx_kv_hybrid_config kcfg;
                kcfg.n_slots = 48;
                kcfg.max_full_tokens = 4096;
                ctx->kv_hybrid_state = agx_kv_hybrid_create(ctx->drm, ctx->compiler, kcfg);
            }
            if (ctx->kv_hybrid_state) {
                auto * pool = static_cast<agx_kv_hybrid_pool *>(ctx->kv_hybrid_state);
                for (int k = 0; k < n_fa_keys; ++k) {
                    agx_kv_hybrid_precompile_ext_dtype(pool, fa_keys[k].head_dim, fa_keys[k].k_dt, fa_keys[k].v_dt);
                }
            }
        }
    }

    // 2. Pre-size scratch BOs (fa_bounds_bo, redirect_tmp_bo, ctrl_pipe_bo) to the reserved
    //    n_ubatch peak graph so Code 1k..32k pay ZERO BO free/alloc/VM-bind stalls!
    constexpr size_t k_fa_bounds_pool_B = 131072u;
    if (!st->fa_bounds_bo || st->fa_bounds_cap_B < k_fa_bounds_pool_B) {
        if (st->fa_bounds_bo) agx_bo_free(ctx->drm, st->fa_bounds_bo);
        st->fa_bounds_bo = agx_bo_alloc(ctx->drm, k_fa_bounds_pool_B,
                                        AGX_BO_SLC_PIN | AGX_BO_WRITEBACK, "ge-fa-bounds");
        st->fa_bounds_cap_B = st->fa_bounds_bo ? k_fa_bounds_pool_B : 0u;
    }

    const bool no_fusion    = (std::getenv("GGML_AGX_NO_FUSION")    != nullptr && std::atoi(std::getenv("GGML_AGX_NO_FUSION"))    != 0);
    const bool no_redirect  = (std::getenv("GGML_AGX_NO_REDIRECT")  != nullptr && std::atoi(std::getenv("GGML_AGX_NO_REDIRECT"))  != 0);
    const bool no_ctrl_pipe = (std::getenv("GGML_AGX_NO_CTRL_PIPE") != nullptr && std::atoi(std::getenv("GGML_AGX_NO_CTRL_PIPE")) != 0);
    const bool ctrl_sram    = (std::getenv("GGML_AGX_CTRL_SRAM")    != nullptr && std::atoi(std::getenv("GGML_AGX_CTRL_SRAM"))    != 0);

    if (!no_fusion && !no_redirect) {
        size_t max_redir_elem_B = 0u;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const struct ggml_tensor * nd = cgraph->nodes[i];
            if (nd && nd->type == GGML_TYPE_F32 && nd->ne[0] > 0 && nd->ne[0] <= 8192 &&
                (nd->op == GGML_OP_MUL_MAT || nd->op == GGML_OP_MUL_MAT_ID ||
                 nd->op == GGML_OP_ADD || nd->op == GGML_OP_MUL)) {
                const size_t nb = ggml_nbytes(nd);
                if (nb <= 32u * 1024u * 1024u && nb > max_redir_elem_B) {
                    max_redir_elem_B = nb;
                }
            }
        }
        if (max_redir_elem_B > 0u) {
            const size_t cand_slot_B  = agx_align_up(max_redir_elem_B, AGX_PAGE_SIZE);
            const size_t total_need_B = cand_slot_B * 4u;
            if (!st->redirect_tmp_bo || st->redirect_tmp_cap_B < total_need_B) {
                if (st->redirect_tmp_bo) agx_bo_free(ctx->drm, st->redirect_tmp_bo);
                const uint32_t rflags = (total_need_B <= 32u * 1024u * 1024u)
                    ? (AGX_BO_SLC_PIN | AGX_BO_WRITEBACK)
                    : (AGX_BO_SLC_STREAM | AGX_BO_WRITEBACK);
                st->redirect_tmp_bo    = agx_bo_alloc(ctx->drm, total_need_B, rflags, "ge-redir-tmp");
                st->redirect_tmp_cap_B = st->redirect_tmp_bo ? total_need_B : 0u;
            }
        }
    }

    if (!no_fusion && !no_ctrl_pipe && !ctrl_sram) {
        std::vector<const struct ggml_tensor *> exec_nd;
        exec_nd.reserve((size_t) cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const struct ggml_tensor * nd = cgraph->nodes[i];
            if (nd && !is_view_only_op(nd->op)) {
                exec_nd.push_back(nd);
            }
        }
        size_t n_ctrl_cand = 0u;
        size_t m1 = 0u, m2 = 0u, m3 = 0u, m7 = 0u;
        for (size_t e = 0; e + 10u < exec_nd.size(); ++e) {
            const struct ggml_tensor * nd = exec_nd[e];
            if (nd->op != GGML_OP_CONCAT || nd->ne[1] != 2 || nd->ne[3] != 1) continue;
            const struct ggml_tensor * c1 = exec_nd[e + 1u];
            const struct ggml_tensor * c2 = exec_nd[e + 2u];
            const struct ggml_tensor * c3 = exec_nd[e + 3u];
            const struct ggml_tensor * c7 = exec_nd[e + 7u];
            if (c1->op == GGML_OP_RMS_NORM && c2->op == GGML_OP_MUL_MAT &&
                c3->op == GGML_OP_MUL_MAT && c7->op == GGML_OP_MUL_MAT) {
                n_ctrl_cand++;
                m1 = std::max(m1, ggml_nbytes(c1));
                m2 = std::max(m2, ggml_nbytes(c2));
                m3 = std::max(m3, ggml_nbytes(c3));
                m7 = std::max(m7, ggml_nbytes(c7));
            }
        }
        if (n_ctrl_cand >= 2u && m1 > 0u && m2 > 0u && m3 > 0u && m7 > 0u) {
            const size_t bank_B  = agx_align_up(m1, AGX_PAGE_SIZE) + agx_align_up(m2, AGX_PAGE_SIZE)
                                 + agx_align_up(m3, AGX_PAGE_SIZE) + agx_align_up(m7, AGX_PAGE_SIZE);
            const size_t total_B = bank_B * 3u;
            if (total_B <= 384u * 1024u * 1024u && (!st->ctrl_pipe_bo || st->ctrl_pipe_cap_B < total_B)) {
                if (st->ctrl_pipe_bo) agx_bo_free(ctx->drm, st->ctrl_pipe_bo);
                const uint32_t cflags = (total_B <= 32u * 1024u * 1024u)
                    ? (AGX_BO_SLC_PIN | AGX_BO_WRITEBACK)
                    : (AGX_BO_SLC_STREAM | AGX_BO_WRITEBACK);
                st->ctrl_pipe_bo    = agx_bo_alloc(ctx->drm, total_B, cflags, "ge-ctrl-pipe");
                st->ctrl_pipe_cap_B = st->ctrl_pipe_bo ? total_B : 0u;
            }
        }
    }
}

void agx_graph_exec_cleanup(agx_backend_context * ctx) {
    if (!ctx) return;
    if (ctx->graph_exec_state) {
        auto * st = static_cast<agx_graph_exec_state *>(ctx->graph_exec_state);
        if (st->inkling2) {
            agx_inkling2_engine_destroy(st->inkling2);
            st->inkling2 = nullptr;
        }
        if (st->alias_tmp_bo) {
            agx_bo_free(ctx->drm, st->alias_tmp_bo);
            st->alias_tmp_bo = nullptr;
            st->alias_tmp_cap_B = 0;
        }
        if (st->fa_bounds_bo) {
            agx_bo_free(ctx->drm, st->fa_bounds_bo);
            st->fa_bounds_bo = nullptr;
            st->fa_bounds_cap_B = 0;
        }
        if (st->redirect_tmp_bo) {
            agx_bo_free(ctx->drm, st->redirect_tmp_bo);
            st->redirect_tmp_bo = nullptr;
            st->redirect_tmp_cap_B = 0;
        }
        if (st->ctrl_pipe_bo) {
            agx_bo_free(ctx->drm, st->ctrl_pipe_bo);
            st->ctrl_pipe_bo = nullptr;
            st->ctrl_pipe_cap_B = 0;
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

static bool can_gemm_redirect(const struct ggml_tensor * dst) {
    return dst && dst->op == GGML_OP_MUL_MAT &&
           (dst->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
           ggml_get_op_params_i32(dst, 1) != GGML_HINT_SRC0_IS_HADAMARD &&
           agx_gemm_supports_mul_mat(dst);
}

static bool dispatch_gemm_to_va(agx_backend_context *      ctx,
                                agx_cdm_stream *           stream,
                                const struct ggml_tensor * dst,
                                uint64_t                   out_va_override,
                                uint64_t                   x_va_override,
                                agx_barrier_mode           post_barrier) {
    if (!can_gemm_redirect(dst) || (out_va_override == 0ull && x_va_override == 0ull)) {
        return false;
    }
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    agx_weight_layout layout;
    if (!agx_gemm_infer_weight_layout(src0, &layout)) {
        return false;
    }
    const uint64_t w_va   = agx_tensor_gpu_va(src0);
    const uint64_t x_va   = (x_va_override   != 0ull) ? x_va_override   : agx_tensor_gpu_va(src1);
    const uint64_t out_va = (out_va_override != 0ull) ? out_va_override : agx_tensor_gpu_va(dst);
    if (w_va == 0ull || x_va == 0ull || out_va == 0ull) {
        return false;
    }
    const int64_t K    = src0->ne[0];
    const int64_t N    = src0->ne[1];
    const int64_t M    = src1->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t r2   = ne12 / ne02;
    const int64_t r3   = ne13 / ne03;

    uint64_t w_nb2 = 0ull, w_nb3 = 0ull;
    if (layout == AGX_WEIGHT_NVFP4_REPACKED || layout == AGX_WEIGHT_Q8_0_REPACKED) {
        w_nb2 = static_cast<uint64_t>(agx_repack_slice_stride(src0->type, K, N));
        w_nb3 = w_nb2 * static_cast<uint64_t>(ne02);
    } else {
        w_nb2 = static_cast<uint64_t>(src0->nb[2]);
        w_nb3 = static_cast<uint64_t>(src0->nb[3]);
    }
    const enum ggml_prec p_src0 = ggml_prec_get_src(dst, 0);
    const enum ggml_prec p_src1 = ggml_prec_get_src(dst, 1);

    agx_gemm_desc desc{};
    desc.M              = static_cast<uint32_t>(M);
    desc.N              = static_cast<uint32_t>(N);
    desc.K              = static_cast<uint32_t>(K);
    desc.x_stride_elems = static_cast<uint32_t>(src1->nb[1] / sizeof(float));
    desc.y_stride_elems = static_cast<uint32_t>(dst->nb[1] / sizeof(float));
    desc.w_slice_bytes  = static_cast<uint32_t>(w_nb2);
    desc.x_slice_elems  = static_cast<uint32_t>(src1->nb[2] / sizeof(float));
    desc.y_slice_elems  = static_cast<uint32_t>(dst->nb[2] / sizeof(float));
    desc.batch_count    = static_cast<uint32_t>(ne12);
    desc.w_bcast_div    = static_cast<uint32_t>(r2);
    desc.layout         = layout;
    desc.prec_f32       = (p_src1 == GGML_PREC_F32 || p_src1 == GGML_PREC_BF16 ||
                           p_src0 == GGML_PREC_F32 || p_src0 == GGML_PREC_BF16 ||
                           src0->type == GGML_TYPE_F32);
    if (ne13 == 1) {
        desc.w_va         = w_va;
        desc.x_va         = x_va;
        desc.y_va         = out_va;
        desc.post_barrier = post_barrier;
        return agx_gemm_dispatch(ctx, stream, desc);
    }
    for (int64_t i3 = 0; i3 < ne13; ++i3) {
        const int64_t w_i3 = i3 / r3;
        desc.w_va         = w_va + static_cast<uint64_t>(w_i3) * w_nb3;
        desc.x_va         = x_va + static_cast<uint64_t>(i3)   * static_cast<uint64_t>(src1->nb[3]);
        desc.y_va         = out_va + static_cast<uint64_t>(i3) * static_cast<uint64_t>(dst->nb[3]);
        desc.post_barrier = (i3 + 1 == ne13) ? post_barrier : AGX_BARRIER_NONE;
        if (!agx_gemm_dispatch(ctx, stream, desc)) {
            return false;
        }
    }
    return true;
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
    const bool no_wave_sched  = (std::getenv("GGML_AGX_NO_WAVE_SCHED") != nullptr &&
                                 std::getenv("GGML_AGX_NO_WAVE_SCHED")[0] == '1');
    const bool ctrl_use_sram  = (std::getenv("GGML_AGX_CTRL_SRAM") != nullptr &&
                                 std::getenv("GGML_AGX_CTRL_SRAM")[0] == '1');
    const bool no_fuse_fa_epi     = (std::getenv("GGML_AGX_NO_FUSE_FA_EPI") != nullptr &&
                                     std::getenv("GGML_AGX_NO_FUSE_FA_EPI")[0] == '1');
    const bool no_fuse_mix_rms    = (std::getenv("GGML_AGX_NO_FUSE_MIX_RMS") != nullptr &&
                                     std::getenv("GGML_AGX_NO_FUSE_MIX_RMS")[0] == '1');
    const bool no_fuse_sconv_prep = (std::getenv("GGML_AGX_NO_FUSE_SCONV_PREP") != nullptr &&
                                     std::getenv("GGML_AGX_NO_FUSE_SCONV_PREP")[0] == '1');
    const bool no_fa_bounds       = (std::getenv("GGML_AGX_NO_FA_BOUNDS") != nullptr &&
                                     std::getenv("GGML_AGX_NO_FA_BOUNDS")[0] == '1');
    const bool no_rel_skip        = (std::getenv("GGML_AGX_NO_REL_SKIP") != nullptr &&
                                     std::getenv("GGML_AGX_NO_REL_SKIP")[0] == '1');
    const bool no_redirect        = (std::getenv("GGML_AGX_NO_REDIRECT") != nullptr &&
                                     std::getenv("GGML_AGX_NO_REDIRECT")[0] == '1');
    const bool no_ctrl_pipe       = (std::getenv("GGML_AGX_NO_CTRL_PIPE") != nullptr &&
                                     std::getenv("GGML_AGX_NO_CTRL_PIPE")[0] == '1');
    const bool no_weak_war        = (std::getenv("GGML_AGX_NO_WEAK_WAR") != nullptr &&
                                     std::getenv("GGML_AGX_NO_WEAK_WAR")[0] == '1');
    const bool print_stats    = (std::getenv("GGML_AGX_STATS") != nullptr &&
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
            if (n->op == GGML_OP_CPY && s == 1 && n->src[s] == n) continue; // ggml_cast self-ref
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
    // while still rejecting any clobbering write at j > all_ins_prod[r].
    // `outs_exec[w]` is the original exec index that produces `outs[w]`:
    //   - Placing at e_min moves write earlier (outs_exec[w] -> e_min): check j in (e_min, outs_exec[w]).
    //   - Placing at e_max moves write later (outs_exec[w] -> e_max): check j in (outs_exec[w], e_max).
    //     When outs_exec[w] == e_max (every single-output fusion!), (e_max, e_max) is empty so dead
    //     intermediates before e_max whose VA was reused by outs[w] never falsely block fusion!
    auto find_safe_anchor = [&](const int32_t * set_arr, int set_n,
                                const agx_va_range * all_ins, const int32_t * all_ins_prod, int n_all_ins,
                                const agx_va_range * extra_ins_for_first, int n_extra_first,
                                const agx_va_range * outs, const int32_t * outs_exec, int n_outs,
                                bool require_last = false, int32_t e_mid = -1) -> int32_t {
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
        if (!require_last) {
            bool ok_first = true;
            for (size_t j = e_min + 1u; j <= e_max && ok_first; ++j) {
                if (in_idx_set(j, set_arr, set_n)) continue;
                for (int r = 0; r < n_extra_first; ++r) {
                    if (va_range_overlap(raw_vas[j].w, extra_ins_for_first[r])) { ok_first = false; break; }
                }
                for (int w = 0; ok_first && w < n_outs; ++w) {
                    if (outs_exec && outs_exec[w] >= 0 && (int32_t) j >= outs_exec[w]) continue;
                    if (va_range_overlap(raw_vas[j].w, outs[w])) { ok_first = false; break; }
                    for (uint8_t jr = 0; jr < raw_vas[j].nr; ++jr) {
                        if (va_range_overlap(raw_vas[j].r[jr], outs[w])) { ok_first = false; break; }
                    }
                }
            }
            if (ok_first) return (int32_t) e_min;
        }

        // Optional interior anchor e_mid (e.g. sc_res for dynamic K/V ShortConv + Prep + SET_ROWS,
        // where dyn_t is computed after e_min and cur is reused by wv before e_max!)
        if (!require_last && e_mid > (int32_t) e_min && e_mid < (int32_t) e_max &&
            in_idx_set((size_t) e_mid, set_arr, set_n)) {
            bool ok_mid = true;
            for (int r = 0; r < n_all_ins; ++r) {
                if (all_ins_prod && all_ins_prod[r] >= e_mid) { ok_mid = false; break; }
            }
            for (size_t j = e_min; j <= e_max && ok_mid; ++j) {
                if (in_idx_set(j, set_arr, set_n)) continue;
                if (j < (size_t) e_mid) {
                    for (int r = 0; r < n_all_ins; ++r) {
                        if (all_ins_prod && all_ins_prod[r] >= 0 && (int32_t) j <= all_ins_prod[r]) continue;
                        if (va_range_overlap(raw_vas[j].w, all_ins[r])) { ok_mid = false; break; }
                    }
                }
                for (int w = 0; ok_mid && w < n_outs; ++w) {
                    const int32_t ox = (outs_exec && outs_exec[w] >= 0) ? outs_exec[w] : (int32_t) e_max;
                    const bool in_move_span = (ox > e_mid)
                        ? ((int32_t) j > e_mid && (int32_t) j < ox)
                        : ((int32_t) j > ox    && (int32_t) j < e_mid);
                    if (!in_move_span) continue;
                    if (va_range_overlap(raw_vas[j].w, outs[w])) { ok_mid = false; break; }
                    for (uint8_t jr = 0; jr < raw_vas[j].nr; ++jr) {
                        if (va_range_overlap(raw_vas[j].r[jr], outs[w])) { ok_mid = false; break; }
                    }
                }
            }
            if (ok_mid) return e_mid;
        }

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
                if (outs_exec && outs_exec[w] >= 0 && (int32_t) j <= outs_exec[w]) continue;
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
    // (optionally ignoring writes from other to-be-absorbed nodes skip0..skip2).
    auto can_absorb_prev = [&](int32_t e_prev, size_t e_cur, const agx_va_range * new_ins, int n_new_ins,
                               int32_t skip0 = -1, int32_t skip1 = -1, int32_t skip2 = -1) -> bool {
        if (e_prev < 0 || (size_t) e_prev >= e_cur) return false;
        if (consumed[(size_t) e_prev]) return false;
        const auto & sl = anchors[(size_t) e_prev];
        if (sl.cnt != 1u || st->items[sl.beg].kind != ITEM_GENERIC_NODE) return false;
        for (size_t j = (size_t) e_prev + 1u; j < e_cur; ++j) {
            if ((int32_t) j == skip0 || (int32_t) j == skip1 || (int32_t) j == skip2) continue;
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
    uint64_t g_prep_store = 0, g_full_sconv = 0, g_router = 0, g_rel_bias = 0, g_cont_abs = 0, g_fused_moe = 0;
    uint64_t g_sprep = 0, g_fa_epi = 0, g_mix_rms = 0, g_rel_bskip = 0;
    uint64_t g_ctrl_ovlp = 0, g_redir = 0, g_ctrl_pipe = 0, g_wfi_bar = 0;
    std::vector<uint8_t> alias_scratch_locked(n_exec, 0u);
    std::vector<agx_exec_item> pre_items;

    // 4-slot SLC-writeback zero-copy GEMM/MoE/FA output redirect pool:
    // Eliminates gallocr VA-reuse fusion blocks on faepi (wqkv_gate & tail_n->wo),
    // and post-ShortConv (wo_out / moe_out) alias trampolines without any emit_f32_copy!
    constexpr size_t k_redir_slots = 4u;
    size_t redir_slot_cap_B = 0u;
    std::vector<uint8_t> redir_slot_locked[k_redir_slots];
    size_t redir_rr = 0u;
    if (!no_fusion && !no_redirect && n_exec >= 16u) {
        size_t max_redir_elem_B = 0u;
        for (size_t e = 0; e < n_exec; ++e) {
            const struct ggml_tensor * nd = cgraph->nodes[st->exec_nodes[e]];
            if (nd && nd->type == GGML_TYPE_F32 && nd->ne[0] > 0 && nd->ne[0] <= 8192 &&
                (nd->op == GGML_OP_MUL_MAT || nd->op == GGML_OP_MUL_MAT_ID ||
                 nd->op == GGML_OP_ADD || nd->op == GGML_OP_MUL)) {
                const size_t nb = ggml_nbytes(nd);
                if (nb <= 32u * 1024u * 1024u && nb > max_redir_elem_B) max_redir_elem_B = nb;
            }
        }
        if (max_redir_elem_B > 0u) {
            const size_t cand_slot_B = agx_align_up(max_redir_elem_B, AGX_PAGE_SIZE);
            const size_t total_need_B = cand_slot_B * k_redir_slots;
            if (!st->redirect_tmp_bo || st->redirect_tmp_cap_B < total_need_B) {
                if (st->redirect_tmp_bo) agx_bo_free(ctx->drm, st->redirect_tmp_bo);
                const uint32_t rflags = (total_need_B <= 32u * 1024u * 1024u)
                    ? (AGX_BO_SLC_PIN | AGX_BO_WRITEBACK)
                    : (AGX_BO_SLC_STREAM | AGX_BO_WRITEBACK);
                st->redirect_tmp_bo    = agx_bo_alloc(ctx->drm, total_need_B, rflags, "ge-redir-tmp");
                st->redirect_tmp_cap_B = st->redirect_tmp_bo ? total_need_B : 0u;
            }
            if (st->redirect_tmp_bo) {
                redir_slot_cap_B = cand_slot_B;
                for (size_t s = 0; s < k_redir_slots; ++s) {
                    redir_slot_locked[s].assign(n_exec, 0u);
                }
            }
        }
    }

    auto try_reserve_redir_slot = [&](size_t e_lo, size_t e_hi, size_t need_B,
                                      agx_va_range * out_r, int * out_slot = nullptr) -> bool {
        *out_r = { 0, 0 };
        if (out_slot) *out_slot = -1;
        if (no_redirect || !st->redirect_tmp_bo || need_B == 0u || need_B > redir_slot_cap_B ||
            e_lo > e_hi || e_hi >= n_exec) {
            return false;
        }
        for (size_t off = 0; off < k_redir_slots; ++off) {
            const size_t s = (redir_rr + off) % k_redir_slots;
            bool free_ok = true;
            for (size_t j = e_lo; j <= e_hi; ++j) {
                if (redir_slot_locked[s][j]) { free_ok = false; break; }
            }
            if (free_ok) {
                for (size_t j = e_lo; j <= e_hi; ++j) {
                    redir_slot_locked[s][j] = 1u;
                }
                redir_rr = (s + 1u) % k_redir_slots;
                const uint64_t bva = st->redirect_tmp_bo->gpu_va + s * redir_slot_cap_B;
                *out_r = { bva, bva + need_B };
                if (out_slot) *out_slot = (int) s;
                return true;
            }
        }
        return false;
    };

    auto unlock_redir_slot = [&](int slot, size_t e_lo, size_t e_hi) {
        if (slot < 0 || (size_t) slot >= k_redir_slots || e_lo > e_hi || e_hi >= n_exec) return;
        for (size_t j = e_lo; j <= e_hi; ++j) {
            redir_slot_locked[(size_t) slot][j] = 0u;
        }
    };

    // Wave 9: 3-bank zero-copy scratch BO for Inkling 2 `control()` streams (Fusion 4):
    // Breaks gallocr VA sharing across the 3 independent Q/K/V `control()` streams per looped layer
    // so all 3 execute in 4 lockstep waves (3x NORM_CAT -> 3x W_down -> 6x W_fg -> 3x CTRL_MIX),
    // eliminating up to ~192 strong_usc barriers on pp512/pp2048!
    constexpr size_t k_ctrl_pipe_banks = 3u;
    size_t ctrl_n1_B = 0u, ctrl_n2_B = 0u, ctrl_n3_B = 0u, ctrl_n7_B = 0u, ctrl_bank_B = 0u;
    size_t ctrl_pipe_rr = 0u;
    if (!no_fusion && !no_ctrl_pipe && !(ctrl_use_sram) && n_exec >= 32u) {
        size_t n_ctrl_cand = 0u;
        size_t m1 = 0u, m2 = 0u, m3 = 0u, m7 = 0u;
        for (size_t e = 0; e + 10u < n_exec; ++e) {
            const struct ggml_tensor * nd = cgraph->nodes[st->exec_nodes[e]];
            if (!nd || nd->op != GGML_OP_CONCAT || nd->ne[1] != 2 || nd->ne[3] != 1) continue;
            const struct ggml_tensor * c1 = cgraph->nodes[st->exec_nodes[e + 1u]];
            const struct ggml_tensor * c2 = cgraph->nodes[st->exec_nodes[e + 2u]];
            const struct ggml_tensor * c3 = cgraph->nodes[st->exec_nodes[e + 3u]];
            const struct ggml_tensor * c7 = cgraph->nodes[st->exec_nodes[e + 7u]];
            if (c1->op == GGML_OP_RMS_NORM && c2->op == GGML_OP_MUL_MAT &&
                c3->op == GGML_OP_MUL_MAT && c7->op == GGML_OP_MUL_MAT) {
                n_ctrl_cand++;
                m1 = std::max(m1, ggml_nbytes(c1));
                m2 = std::max(m2, ggml_nbytes(c2));
                m3 = std::max(m3, ggml_nbytes(c3));
                m7 = std::max(m7, ggml_nbytes(c7));
            }
        }
        if (n_ctrl_cand >= 2u && m1 > 0u && m2 > 0u && m3 > 0u && m7 > 0u) {
            const size_t s1 = agx_align_up(m1, AGX_PAGE_SIZE);
            const size_t s2 = agx_align_up(m2, AGX_PAGE_SIZE);
            const size_t s3 = agx_align_up(m3, AGX_PAGE_SIZE);
            const size_t s7 = agx_align_up(m7, AGX_PAGE_SIZE);
            const size_t bank_B  = s1 + s2 + s3 + s7;
            const size_t total_B = bank_B * k_ctrl_pipe_banks;
            if (total_B <= 384u * 1024u * 1024u) {
                if (!st->ctrl_pipe_bo || st->ctrl_pipe_cap_B < total_B) {
                    if (st->ctrl_pipe_bo) agx_bo_free(ctx->drm, st->ctrl_pipe_bo);
                    const uint32_t cflags = (total_B <= 32u * 1024u * 1024u)
                        ? (AGX_BO_SLC_PIN | AGX_BO_WRITEBACK)
                        : (AGX_BO_SLC_STREAM | AGX_BO_WRITEBACK);
                    st->ctrl_pipe_bo    = agx_bo_alloc(ctx->drm, total_B, cflags, "ge-ctrl-pipe");
                    st->ctrl_pipe_cap_B = st->ctrl_pipe_bo ? total_B : 0u;
                }
                if (st->ctrl_pipe_bo) {
                    ctrl_n1_B = s1; ctrl_n2_B = s2; ctrl_n3_B = s3; ctrl_n7_B = s7;
                    ctrl_bank_B = bank_B;
                }
            }
        }
    }

    // Map `Fusion 12` kq_b_rel_pos output tensor -> underlying 1-head base mask tensor
    // so `ITEM_REL_BIAS` & `PASS_MODE=3` FlashAttn share 1 Wave-0 [kv_beg, kv_end) bounds scan!
    struct rel_mask_entry {
        const struct ggml_tensor * kq_b_out;
        const struct ggml_tensor * base_mask;
        uint64_t                   bounds_va;
        agx_va_range               bounds_range;
    };
    std::vector<rel_mask_entry> rel_mask_map;
    rel_mask_map.reserve(64u);

    struct fa_bounds_slot {
        uint64_t mask_va;
        uint32_t n_kv;
        uint32_t n_tokens;
        uint32_t mask_nb1;
        uint32_t is_f16;
        uint32_t q_br;
        uint64_t bounds_va;
        agx_va_range bounds_range;
    };
    std::vector<fa_bounds_slot> fa_bounds_slots;
    fa_bounds_slots.reserve(8u);
    size_t fa_bounds_used_B = 0u;
    const size_t k_fa_bounds_pool_B = 131072u; // 128 KiB fixed VA pool (covers >= 16 unique masks at Br=16/32 up to M=16384)

    auto get_or_alloc_fa_bounds = [&](const struct ggml_tensor * bm,
                                      uint32_t n_kv_u,
                                      uint32_t n_tok_u,
                                      uint32_t q_br_u,
                                      uint64_t * out_va,
                                      agx_va_range * out_r) -> bool {
        *out_va = 0ull;
        *out_r  = { 0, 0 };
        if (no_fa_bounds || !bm || tensor_prod_idx(bm) >= 0) return false;
        if (bm->ne[0] < (int64_t) n_kv_u || bm->ne[1] < (int64_t) n_tok_u || bm->ne[2] != 1 || bm->ne[3] != 1) return false;
        if ((bm->type != GGML_TYPE_F16 && bm->type != GGML_TYPE_F32) || bm->nb[0] != ggml_type_size(bm->type)) return false;
        const uint64_t bm_va   = agx_tensor_gpu_va(bm);
        const uint32_t m_nb1_u = (uint32_t) bm->nb[1];
        const uint32_t is_f16  = (bm->type == GGML_TYPE_F16) ? 1u : 0u;
        const uint32_t br_u    = (q_br_u > 16u) ? 32u : 16u;
        if (bm_va == 0ull) return false;
        for (const auto & sl : fa_bounds_slots) {
            if (sl.mask_va == bm_va && sl.n_kv == n_kv_u && sl.n_tokens == n_tok_u &&
                sl.mask_nb1 == m_nb1_u && sl.is_f16 == is_f16 && sl.q_br == br_u) {
                *out_va = sl.bounds_va;
                *out_r  = sl.bounds_range;
                return true;
            }
        }
        const uint32_t n_qt = (n_tok_u + br_u - 1u) / br_u;
        const size_t need_B = agx_align_up((size_t) n_qt * 8u, 256u);
        if (fa_bounds_used_B + need_B > k_fa_bounds_pool_B) return false;
        if (!st->fa_bounds_bo || st->fa_bounds_cap_B < k_fa_bounds_pool_B) {
            if (st->fa_bounds_bo) agx_bo_free(ctx->drm, st->fa_bounds_bo);
            st->fa_bounds_bo = agx_bo_alloc(ctx->drm, k_fa_bounds_pool_B, AGX_BO_SLC_PIN | AGX_BO_WRITEBACK, "ge-fa-bounds");
            st->fa_bounds_cap_B = st->fa_bounds_bo ? k_fa_bounds_pool_B : 0u;
        }
        if (!st->fa_bounds_bo) return false;
        const uint64_t bva = st->fa_bounds_bo->gpu_va + fa_bounds_used_B;
        const agx_va_range br = { bva, bva + need_B };
        fa_bounds_used_B += need_B;
        fa_bounds_slots.push_back({ bm_va, n_kv_u, n_tok_u, m_nb1_u, is_f16, br_u, bva, br });

        fa_mask_bounds_push bpc{};
        bpc.mask_va   = bm_va;
        bpc.bounds_va = bva;
        bpc.n_kv      = n_kv_u;
        bpc.n_tokens  = n_tok_u;
        bpc.mask_nb1  = m_nb1_u;
        bpc.is_f16    = is_f16;
        bpc.q_br      = br_u;
        bpc._pad0     = 0u;

        agx_exec_item bit{};
        bit.kind          = ITEM_FA_MASK_BOUNDS;
        bit.gx            = n_qt;
        bit.gy            = 1u;
        bit.gz            = 1u;
        bit.nodes_covered = 0u;
        bit.add_read(tensor_va_range(bm));
        bit.add_write(br);
        bit.set_push(bpc);
        pre_items.push_back(bit);

        *out_va = bva;
        *out_r  = br;
        return true;
    };

    // ------------------------------------------------------------------------
    // Wave 4 Fusion 12: DSpark Markov Head Tail Fusion (~320 ops -> 21 high-BW dispatches!)
    // Matches src/models/dspark.cpp:build_dspark_markov_head (RNN + Top-64 CDF draw + softcap)
    // ------------------------------------------------------------------------
    if ((fusion_mask & (1u << 13)) && n_exec >= 64u) {
        const struct ggml_tensor * w1_t = nullptr, * w2_t = nullptr, * w2_s_t = nullptr, * w2_b_t = nullptr;
        const struct ggml_tensor * jp_t = nullptr, * jp_b_t = nullptr;
        const struct ggml_tensor * base_t = nullptr, * hid_t = nullptr;
        bool has_d2t_or_conf = false;
        int32_t e_base = -1, e_hid = -1, e_first_w1 = -1;

        for (size_t e = 0; e < n_exec; ++e) {
            const struct ggml_tensor * nd = cgraph->nodes[st->exec_nodes[e]];
            if (nd->name[0]) {
                if (std::strcmp(nd->name, "result_output") == 0) { base_t = nd; e_base = (int32_t) e; }
                else if (std::strcmp(nd->name, "result_norm") == 0) { hid_t = nd; e_hid = (int32_t) e; }
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const struct ggml_tensor * src = nd->src[s];
                if (!src || !src->name[0]) continue;
                if      (std::strcmp(src->name, "markov_w1.weight") == 0) { w1_t = src; if (e_first_w1 < 0) e_first_w1 = (int32_t) e; }
                else if (std::strcmp(src->name, "markov_w2.weight") == 0) w2_t = src;
                else if (std::strcmp(src->name, "markov_w2.scale") == 0)  w2_s_t = src;
                else if (std::strcmp(src->name, "markov_w2.bias") == 0)   w2_b_t = src;
                else if (std::strcmp(src->name, "joint_proj.weight") == 0) jp_t = src;
                else if (std::strcmp(src->name, "joint_proj.bias") == 0)  jp_b_t = src;
                else if (std::strcmp(src->name, "d2t") == 0 ||
                         std::strcmp(src->name, "conf_proj.weight") == 0) has_d2t_or_conf = true;
            }
        }

        if (w1_t && w2_t && jp_t && base_t && hid_t && !has_d2t_or_conf &&
            e_base >= 0 && e_hid >= 0 && e_first_w1 > e_base && e_first_w1 > e_hid) {
            // Find the start of build_dspark_markov_head: CONT(VIEW_2D(inp_tokens, 1, n_blocks))
            const int32_t markov_beg = e_base + 1;
            const struct ggml_tensor * prev_cont = cgraph->nodes[st->exec_nodes[(size_t) markov_beg]];
            const struct ggml_tensor * inp_toks  = (prev_cont->op == GGML_OP_CONT && prev_cont->src[0] &&
                                                    prev_cont->src[0]->op == GGML_OP_VIEW)
                                                   ? prev_cont->src[0]->view_src : nullptr;
            const struct ggml_tensor * final_out = cgraph->nodes[st->exec_nodes[n_exec - 1u]];

            if (inp_toks && inp_toks->type == GGML_TYPE_I32 &&
                prev_cont->type == GGML_TYPE_I32 && prev_cont->ne[0] >= 1 && prev_cont->ne[0] <= 4 &&
                final_out->op == GGML_OP_CONT && final_out->type == GGML_TYPE_F32 &&
                base_t->type == GGML_TYPE_F32 && hid_t->type == GGML_TYPE_F32 &&
                base_t->ne[1] == hid_t->ne[1] && base_t->ne[1] == inp_toks->ne[0] &&
                (base_t->ne[1] % prev_cont->ne[0]) == 0 &&
                ggml_nelements(final_out) == ggml_nelements(base_t)) {
                const uint32_t n_blocks     = (uint32_t) prev_cont->ne[0];
                const uint32_t block_drafts = (uint32_t) (base_t->ne[1] / prev_cont->ne[0]);
                const uint32_t n_vocab      = (uint32_t) base_t->ne[0];
                const uint32_t n_embd       = (uint32_t) hid_t->ne[0];
                const uint32_t rank         = (uint32_t) w1_t->ne[0];

                // Locate optional rnn_state (from f_rnn CONT(VIEW_2D(rnn_state, n_embd, n_blocks)) right after prev_cont)
                const struct ggml_tensor * rnn_state = nullptr;
                const struct ggml_tensor * final_h_nextn = nullptr;
                bool sample_from_anchor = true;
                float softcap = 0.0f;

                for (size_t e = (size_t) markov_beg + 1u; e < (size_t) e_first_w1; ++e) {
                    const struct ggml_tensor * nd = cgraph->nodes[st->exec_nodes[e]];
                    if (!rnn_state && nd->op == GGML_OP_CONT && nd->src[0] && nd->src[0]->op == GGML_OP_VIEW &&
                        nd->src[0]->view_src && nd->src[0]->view_src->type == GGML_TYPE_F32 &&
                        nd->src[0]->view_src->ne[0] == (int64_t) n_embd &&
                        nd->src[0]->view_src->ne[1] == base_t->ne[1] &&
                        nd->src[0]->view_src != hid_t) {
                        rnn_state = nd->src[0]->view_src;
                    }
                    if (nd->op == GGML_OP_CONT && nd->src[0] && nd->src[0]->op == GGML_OP_VIEW &&
                        nd->src[0]->view_src == base_t && nd->src[0]->view_offs == 0) {
                        sample_from_anchor = false;
                    }
                }
                if (rnn_state && n_exec >= 2u) {
                    // final_h_nextn is the CONT(PERMUTE(cat_draw)) right before final_out
                    const struct ggml_tensor * cand_h = cgraph->nodes[st->exec_nodes[n_exec - 2u]];
                    if (cand_h->op == GGML_OP_CONT && cand_h->type == GGML_TYPE_F32 &&
                        ggml_nelements(cand_h) == (int64_t) n_embd * base_t->ne[1]) {
                        final_h_nextn = cand_h;
                    }
                }
                // Detect softcap on first col: SCALE(1/c) -> TANH -> SCALE(c) with ne[0] == n_vocab
                for (size_t e = (size_t) e_first_w1; e + 2u < n_exec; ++e) {
                    const struct ggml_tensor * s0 = cgraph->nodes[st->exec_nodes[e]];
                    const struct ggml_tensor * u1 = cgraph->nodes[st->exec_nodes[e + 1u]];
                    const struct ggml_tensor * s2 = cgraph->nodes[st->exec_nodes[e + 2u]];
                    if (s0->op == GGML_OP_SCALE && s0->ne[0] == (int64_t) n_vocab &&
                        u1->op == GGML_OP_UNARY && ggml_get_unary_op(u1) == GGML_UNARY_OP_TANH &&
                        s2->op == GGML_OP_SCALE && u1->src[0] == s0 && s2->src[0] == u1) {
                        float cap = 0.0f, b = 0.0f;
                        read_scale_params(s2, &cap, &b);
                        if (cap > 0.0f && b == 0.0f) { softcap = cap; break; }
                    }
                }

                // Verify NO external side effects (SET_ROWS / unexpected OUTPUT) inside [markov_beg .. n_exec-1]
                bool tail_clean = (block_drafts >= 2u && rank <= 512u && n_embd <= 2304u) &&
                                  (!rnn_state || final_h_nextn != nullptr) &&
                                  va_disjoint(base_t, final_out) &&
                                  (!final_h_nextn || va_disjoint(rnn_state, final_h_nextn));
                for (size_t e = (size_t) markov_beg; tail_clean && e < n_exec; ++e) {
                    const struct ggml_tensor * nd = cgraph->nodes[st->exec_nodes[e]];
                    if (nd->op == GGML_OP_SET_ROWS) tail_clean = false;
                    if ((nd->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 &&
                        nd != final_out && nd != final_h_nextn) {
                        tail_clean = false;
                    }
                }

                if (tail_clean) {
                    agx_exec_item it{};
                    it.kind          = ITEM_DSPARK_MARKOV_W4;
                    it.nodes_covered = (uint32_t) (n_exec - (size_t) markov_beg);
                    it.add_read(tensor_va_range(inp_toks));
                    it.add_read(tensor_va_range(base_t));
                    it.add_read(tensor_va_range(hid_t));
                    if (rnn_state) it.add_read(tensor_va_range(rnn_state));
                    it.add_write(tensor_va_range(final_out));
                    if (final_h_nextn) it.add_write(tensor_va_range(final_h_nextn));

                    auto & da = it.dspark_w4;
                    da.w.markov_w1          = w1_t;
                    da.w.markov_w2          = w2_t;
                    da.w.markov_w2_s        = w2_s_t;
                    da.w.markov_w2_b        = w2_b_t;
                    da.w.joint_proj         = jp_t;
                    da.w.joint_proj_b       = jp_b_t;
                    da.w.n_embd             = n_embd;
                    da.w.rank               = rank;
                    da.w.n_vocab            = n_vocab;
                    da.w.block_size         = block_drafts;
                    da.w.sample_from_anchor = sample_from_anchor;
                    da.w.has_f_rnn          = (rnn_state != nullptr);
                    da.w.softcap            = softcap;
                    da.anchor_toks_va       = agx_tensor_gpu_va(inp_toks);
                    da.base_logits_va       = agx_tensor_gpu_va(base_t);
                    da.hidden_va            = agx_tensor_gpu_va(hid_t);
                    da.f_rnn_va             = rnn_state ? agx_tensor_gpu_va(rnn_state) : 0ull;
                    da.rnn_state_va         = rnn_state ? agx_tensor_gpu_va(rnn_state) : 0ull;
                    da.out_logits_va        = agx_tensor_gpu_va(final_out);
                    da.out_h_nextn_va       = final_h_nextn ? agx_tensor_gpu_va(final_h_nextn) : 0ull;
                    da.n_blocks             = n_blocks;
                    da.block_drafts         = block_drafts;

                    const uint32_t idx = (uint32_t) st->items.size();
                    st->items.push_back(it);
                    anchors[(size_t) markov_beg] = { idx, 1u };
                    for (size_t e = (size_t) markov_beg; e < n_exec; ++e) {
                        consumed[e] = 1u;
                    }
                    st->stats.fused_dspark_hits++;
                }
            }
        }
    }

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

                        agx_va_range outs[1]    = { tensor_va_range(final_dst) };
                        int32_t      outs_ex[1] = { st->exec_idx_map.get(final_dst) };
                        const int32_t anc = find_safe_anchor(fset, fset_n, all_ins, all_prods, n_all, extra_ins, n_extra, outs, outs_ex, 1);
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

                        // Wave 7: Prefer wave-schedulable 8x8 MMA + CONT(VIEW_3D)-absorbing pipeline for ALL M!
                        // (At M=512..2048, scalar 1-WG/core ITEM_CTRL_SRAM burned 232 GFLOPs without MMA and left
                        //  168 strided CONT(VIEW_3D) copies unfused; set GGML_AGX_CTRL_SRAM=1 to A/B legacy SRAM path.)
                        bool ctrl_emitted = false;
                        if (!(ctrl_use_sram && n_tokens >= 16) &&
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

                            auto match_ctrl_cont_view = [&](const struct ggml_tensor * t, uint32_t expected_uses,
                                                            const struct ggml_tensor ** out_v, int32_t * out_ec) -> bool {
                                *out_v  = nullptr;
                                *out_ec = -1;
                                if (!t || t->op != GGML_OP_CONT || (t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return false;
                                const uint32_t uc = st->use_map.get(t);
                                if (uc != 1u && uc != expected_uses) return false;
                                if (uc == 1u) {
                                    struct ggml_tensor * u = st->first_use.get(t);
                                    if (!u || u->op != GGML_OP_RESHAPE || (u->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
                                        st->use_map.get(u) != expected_uses) return false;
                                }
                                const struct ggml_tensor * v = t->src[0];
                                if (!v || v->op != GGML_OP_VIEW || v->type != GGML_TYPE_F32) return false;
                                if (v->ne[0] != 1 || v->ne[1] != n_embd || v->ne[2] != n_tokens || v->ne[3] != 1) return false;
                                if ((v->nb[1] & 3u) != 0 || (v->nb[2] & 3u) != 0 || agx_tensor_gpu_va(v) == 0) return false;
                                int32_t ec = st->exec_idx_map.get(t);
                                if (ec < 0) return false;
                                *out_v  = v;
                                *out_ec = ec;
                                return true;
                            };

                            const struct ggml_tensor * cv_cand = nullptr, * mv_cand = nullptr;
                            int32_t ec_c0 = -1, ec_m0 = -1;
                            const bool has_cv = match_ctrl_cont_view(cur, 3u, &cv_cand, &ec_c0);
                            const bool has_mv = match_ctrl_cont_view(msg, 2u, &mv_cand, &ec_m0);
                            if (has_cv && has_mv) {
                                agx_va_range vr_c = tensor_va_range(cv_cand);
                                agx_va_range vr_m = tensor_va_range(mv_cand);
                                if (can_absorb_prev(ec_c0, e, &vr_c, 1, ec_m0) &&
                                    can_absorb_prev(ec_m0, e, &vr_m, 1, ec_c0)) {
                                    cur_act = cv_cand; cur_nb0 = (uint32_t) (cv_cand->nb[1] / 4u); cur_nb1 = (uint32_t) (cv_cand->nb[2] / 4u); ec_cur = ec_c0;
                                    msg_act = mv_cand; msg_nb0 = (uint32_t) (mv_cand->nb[1] / 4u); msg_nb1 = (uint32_t) (mv_cand->nb[2] / 4u); ec_msg = ec_m0;
                                } else {
                                    if (can_absorb_prev(ec_c0, e, &vr_c, 1)) {
                                        cur_act = cv_cand; cur_nb0 = (uint32_t) (cv_cand->nb[1] / 4u); cur_nb1 = (uint32_t) (cv_cand->nb[2] / 4u); ec_cur = ec_c0;
                                    }
                                    if (can_absorb_prev(ec_m0, e, &vr_m, 1)) {
                                        msg_act = mv_cand; msg_nb0 = (uint32_t) (mv_cand->nb[1] / 4u); msg_nb1 = (uint32_t) (mv_cand->nb[2] / 4u); ec_msg = ec_m0;
                                    }
                                }
                            } else if (has_cv) {
                                agx_va_range vr_c = tensor_va_range(cv_cand);
                                if (can_absorb_prev(ec_c0, e, &vr_c, 1)) {
                                    cur_act = cv_cand; cur_nb0 = (uint32_t) (cv_cand->nb[1] / 4u); cur_nb1 = (uint32_t) (cv_cand->nb[2] / 4u); ec_cur = ec_c0;
                                }
                            } else if (has_mv) {
                                agx_va_range vr_m = tensor_va_range(mv_cand);
                                if (can_absorb_prev(ec_m0, e, &vr_m, 1)) {
                                    msg_act = mv_cand; msg_nb0 = (uint32_t) (mv_cand->nb[1] / 4u); msg_nb1 = (uint32_t) (mv_cand->nb[2] / 4u); ec_msg = ec_m0;
                                }
                            }

                            auto dst_ok_with = [&](const struct ggml_tensor * d, const struct ggml_tensor * s, uint32_t nb0) -> bool {
                                return (nb0 > 1u) ? va_disjoint(d, s) : va_same_or_disjoint(d, s);
                            };

                            const size_t n1_bytes = ggml_nbytes(n1);
                            const size_t n2_bytes = ggml_nbytes(n2);
                            const size_t n3_bytes = ggml_nbytes(n3);
                            const size_t n7_bytes = ggml_nbytes(n7);
                            const bool can_ctrl_pipe =
                                !no_ctrl_pipe && st->ctrl_pipe_bo != nullptr && ctrl_bank_B > 0u &&
                                n1_bytes > 0u && n1_bytes <= ctrl_n1_B &&
                                n2_bytes > 0u && n2_bytes <= ctrl_n2_B &&
                                n3_bytes > 0u && n3_bytes <= ctrl_n3_B &&
                                n7_bytes > 0u && n7_bytes <= ctrl_n7_B &&
                                can_gemm_redirect(n2) && can_gemm_redirect(n3) && can_gemm_redirect(n7) &&
                                n2->src[1] && ggml_is_contiguous(n2->src[1]) && n2->src[1]->view_offs == 0 &&
                                n3->src[1] && ggml_is_contiguous(n3->src[1]) && n3->src[1]->view_offs == 0 &&
                                n7->src[1] && ggml_is_contiguous(n7->src[1]) && n7->src[1]->view_offs == 0;

                            bool can_redir_n7 = !no_redirect && st->redirect_tmp_bo &&
                                                n7_bytes > 0u && n7_bytes <= redir_slot_cap_B &&
                                                st->use_map.get(n7) == 1u && can_gemm_redirect(n7);
                            if (can_redir_n7) {
                                bool any_free = false;
                                for (size_t s = 0; s < k_redir_slots; ++s) {
                                    if (!redir_slot_locked[s][e]) { any_free = true; break; }
                                }
                                can_redir_n7 = any_free;
                            }

                            auto check_ctrl_va = [&](const struct ggml_tensor * ca, uint32_t c_nb0,
                                                     const struct ggml_tensor * ma, uint32_t m_nb0) -> bool {
                                if (can_ctrl_pipe &&
                                    dst_ok_with(final_dst, ca, c_nb0) &&
                                    dst_ok_with(final_dst, ma, m_nb0)) {
                                    return true;
                                }
                                const bool base_ok =
                                    va_disjoint(n1, ca) && va_disjoint(n1, ma) &&
                                    va_disjoint(n2, ca) && va_disjoint(n2, ma) &&
                                    va_disjoint(n3, ca) && va_disjoint(n3, ma);
                                if (!base_ok) return false;
                                const bool ovlp_redir_ok = can_redir_n7 &&
                                    dst_ok_with(final_dst, ca, c_nb0) &&
                                    dst_ok_with(final_dst, ma, m_nb0) &&
                                    va_same_or_disjoint(final_dst, n3);
                                if (ovlp_redir_ok) return true;
                                return dst_ok_with(n6, ca, c_nb0) && va_disjoint(n6, ma) && va_same_or_disjoint(n6, n3) &&
                                       va_disjoint(n7, n6) && va_disjoint(n7, ma) &&
                                       va_same_or_disjoint(final_dst, n6) && dst_ok_with(final_dst, ma, m_nb0) &&
                                       va_same_or_disjoint(final_dst, n7);
                            };

                            if (!check_ctrl_va(cur_act, cur_nb0, msg_act, msg_nb0) && (ec_cur >= 0 || ec_msg >= 0)) {
                                // Try dropping only the conflicting CONT absorption(s) before giving up both
                                if (ec_cur >= 0 && ec_msg >= 0) {
                                    if (check_ctrl_va(cur, 1u, msg_act, msg_nb0)) {
                                        cur_act = cur; cur_nb0 = 1u; cur_nb1 = (uint32_t) n_embd; ec_cur = -1;
                                    } else if (check_ctrl_va(cur_act, cur_nb0, msg, 1u)) {
                                        msg_act = msg; msg_nb0 = 1u; msg_nb1 = (uint32_t) n_embd; ec_msg = -1;
                                    } else {
                                        cur_act = cur; cur_nb0 = 1u; cur_nb1 = (uint32_t) n_embd; ec_cur = -1;
                                        msg_act = msg; msg_nb0 = 1u; msg_nb1 = (uint32_t) n_embd; ec_msg = -1;
                                    }
                                } else {
                                    cur_act = cur; cur_nb0 = 1u; cur_nb1 = (uint32_t) n_embd; ec_cur = -1;
                                    msg_act = msg; msg_nb0 = 1u; msg_nb1 = (uint32_t) n_embd; ec_msg = -1;
                                }
                            }

                            const bool va_safe = check_ctrl_va(cur_act, cur_nb0, msg_act, msg_nb0);

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
                                const bool use_ctrl_pipe =
                                    can_ctrl_pipe &&
                                    dst_ok_with(final_dst, cur_act, cur_nb0) &&
                                    dst_ok_with(final_dst, msg_act, msg_nb0);

                                if (use_ctrl_pipe) {
                                    const size_t bank = ctrl_pipe_rr++ % k_ctrl_pipe_banks;
                                    const uint64_t b0 = st->ctrl_pipe_bo->gpu_va + bank * ctrl_bank_B;
                                    const uint64_t va_n1 = b0;
                                    const uint64_t va_n2 = va_n1 + ctrl_n1_B;
                                    const uint64_t va_n3 = va_n2 + ctrl_n2_B;
                                    const uint64_t va_n7 = va_n3 + ctrl_n3_B;
                                    const agx_va_range r_n1 = { va_n1, va_n1 + n1_bytes };
                                    const agx_va_range r_n2 = { va_n2, va_n2 + n2_bytes };
                                    const agx_va_range r_n3 = { va_n3, va_n3 + n3_bytes };
                                    const agx_va_range r_n7 = { va_n7, va_n7 + n7_bytes };
                                    g_ctrl_pipe++;
                                    g_ctrl_ovlp++;

                                    // Sub-item 1: k_ctrl_norm_cat -> va_n1
                                    {
                                        ctrl_norm_cat_push ncpc{};
                                        ncpc.cur_va   = agx_tensor_gpu_va(cur_act);
                                        ncpc.msg_va   = agx_tensor_gpu_va(msg_act);
                                        ncpc.dst_va   = va_n1;
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
                                        it.add_write(r_n1);
                                        it.set_push(ncpc);
                                        st->items.push_back(it);
                                    }
                                    // Sub-item 2: W_down (n2): va_n1 -> va_n2
                                    {
                                        agx_exec_item it{};
                                        it.kind          = ITEM_GEMM_REDIRECT;
                                        it.node          = n2;
                                        it.va0           = va_n2;
                                        it.va1           = va_n1;
                                        it.nodes_covered = 1u;
                                        it.add_read(tensor_va_range(n2->src[0]));
                                        it.add_read(r_n1);
                                        it.add_write(r_n2);
                                        st->items.push_back(it);
                                    }
                                    // Sub-item 3: W_forget (n3): va_n2 -> va_n3
                                    {
                                        agx_exec_item it{};
                                        it.kind          = ITEM_GEMM_REDIRECT;
                                        it.node          = n3;
                                        it.va0           = va_n3;
                                        it.va1           = va_n2;
                                        it.nodes_covered = 1u;
                                        it.add_read(tensor_va_range(n3->src[0]));
                                        it.add_read(r_n2);
                                        it.add_write(r_n3);
                                        st->items.push_back(it);
                                    }
                                    // Sub-item 4: W_gain (n7): va_n2 -> va_n7
                                    {
                                        agx_exec_item it{};
                                        it.kind          = ITEM_GEMM_REDIRECT;
                                        it.node          = n7;
                                        it.va0           = va_n7;
                                        it.va1           = va_n2;
                                        it.nodes_covered = 1u;
                                        it.add_read(tensor_va_range(n7->src[0]));
                                        it.add_read(r_n2);
                                        it.add_write(r_n7);
                                        st->items.push_back(it);
                                    }
                                    // Sub-item 5: k_ctrl_mix (reads va_n3, va_n7 -> writes final_dst)
                                    {
                                        ctrl_mix_push mpc{};
                                        mpc.cur_va    = agx_tensor_gpu_va(cur_act);
                                        mpc.msg_va    = agx_tensor_gpu_va(msg_act);
                                        mpc.forget_va = va_n3;
                                        mpc.gain_va   = va_n7;
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
                                        it.add_read(r_n3);
                                        it.add_read(r_n7);
                                        if (norm_w) it.add_read(tensor_va_range(norm_w));
                                        it.add_write(tensor_va_range(final_dst));
                                        it.set_push(mpc);
                                        st->items.push_back(it);
                                    }
                                } else {
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

                                const bool ovlp_tail_ok =
                                    dst_ok_with(final_dst, cur_act, cur_nb0) &&
                                    dst_ok_with(final_dst, msg_act, msg_nb0) &&
                                    va_same_or_disjoint(final_dst, n3);
                                const bool native_overlap_fg =
                                    ovlp_tail_ok && va_disjoint(n3, n7) &&
                                    va_disjoint(n7, cur_act) && va_disjoint(n7, msg_act) &&
                                    va_same_or_disjoint(final_dst, n7);
                                agx_va_range n7_redir_r = { 0, 0 };
                                const bool use_n7_redir = !native_overlap_fg && ovlp_tail_ok && can_redir_n7 &&
                                                          try_reserve_redir_slot(e, e, n7_bytes, &n7_redir_r);
                                const bool can_overlap_fg = native_overlap_fg || use_n7_redir;
                                if (can_overlap_fg) {
                                    g_ctrl_ovlp++;
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
                                        it.kind = use_n7_redir ? ITEM_GEMM_REDIRECT : ITEM_GENERIC_NODE;
                                        it.node = n7;
                                        it.va0  = use_n7_redir ? n7_redir_r.beg : 0ull;
                                        it.nodes_covered = 1u;
                                        it.add_read(tensor_va_range(n7->src[0]));
                                        it.add_read(tensor_va_range(n7->src[1]));
                                        it.add_write(use_n7_redir ? n7_redir_r : tensor_va_range(n7));
                                        st->items.push_back(it);
                                        if (use_n7_redir) g_redir++;
                                    }
                                    {
                                        ctrl_mix_push mpc{};
                                        mpc.cur_va    = agx_tensor_gpu_va(cur_act);
                                        mpc.msg_va    = agx_tensor_gpu_va(msg_act);
                                        mpc.forget_va = agx_tensor_gpu_va(n3);
                                        mpc.gain_va   = use_n7_redir ? n7_redir_r.beg : agx_tensor_gpu_va(n7);
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
                                        it.add_read(use_n7_redir ? n7_redir_r : tensor_va_range(n7));
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
                                }

                                anchors[e] = { slice_beg, (uint32_t) (st->items.size() - slice_beg) };
                                for (size_t k = 0; k <= advance; ++k) consumed[e + k] = 1u;
                                st->stats.fused_ctrl_gate_hits++;
                                g_ctrl++;
                                ctrl_emitted = true;
                            }
                        }
                        if (ctrl_emitted) continue;

                        if (va_same_or_disjoint(final_dst, cur) && va_same_or_disjoint(final_dst, msg) &&
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
                const bool vec4_ok = ggml_are_same_shape(a, b) && ggml_are_same_shape(a, sc_node) &&
                                     ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(sc_node) &&
                                     (pc.total & 3u) == 0u && ((pc.a_va | pc.b_va | pc.d_va) & 15ull) == 0ull;
                if (vec4_ok) pc.op_kind |= 0x100u;

                int32_t fset[2] = { (int32_t) e, (int32_t) (e + 1u) };
                agx_exec_item it{};
                it.kind = ITEM_BIN_OP;
                it.gx = vec4_ok ? (((pc.total >> 2u) + 255u) / 256u) : ((pc.total + 255u) / 256u);
                it.gy = 1u; it.gz = 1u;
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
                                agx_va_range outs[2]    = { tensor_va_range(argsort), tensor_va_range(final_w) };
                                int32_t      outs_ex[2] = { st->exec_idx_map.get(argsort), st->exec_idx_map.get(final_w) };

                                const int32_t anc = find_safe_anchor(fset, fn, all_ins, all_prods, n_ins, extra_ins, n_extra, outs, outs_ex, 2);
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
                    const bool vec4_ok = ggml_are_same_shape(a, b) && ggml_are_same_shape(a, mul_node) &&
                                         ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(mul_node) &&
                                         (pc.total & 3u) == 0u && ((pc.a_va | pc.b_va | pc.d_va) & 15ull) == 0ull;
                    if (vec4_ok) pc.op_kind |= 0x100u;

                    int32_t fset[2] = { (int32_t) e, (int32_t) (e + 1u) };
                    agx_exec_item it{};
                    it.kind = ITEM_BIN_OP;
                    it.gx = vec4_ok ? (((pc.total >> 2u) + 255u) / 256u) : ((pc.total + 255u) / 256u);
                    it.gy = 1u; it.gz = 1u;
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
        // Fusion 8 (Wave 7 Upgraded): Inkling 2 `mix()` / `msg_mix()` Depth-Mix Attention
        //   - Non-adjacent DAG matching (`mm1 -> sm -> mm2` + `cnt = CONT(TRANSPOSE(states))`)
        //   - Backward absorption of `REPEAT(query)`, `CONT(TRANSPOSE)`, `CONCAT`, & `RMS_NORM(cur, 1e-6)`
        //   - Skips to-be-absorbed `e_rep`/`e_cnt`/`e_cat` in `can_absorb_prev` & supports `alias_tmp_bo`
        //     trampoline when `mm2` aliases `base_states`/`extra_t` at block boundaries (`mix=120/120`!)
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 7)) && node->op == GGML_OP_MUL_MAT) {
            struct ggml_tensor * mm1 = node;
            struct ggml_tensor * sm  = next_single_use_compute(mm1, false);
            struct ggml_tensor * mm2 = (sm && sm->op == GGML_OP_SOFT_MAX && !sm->src[1] && !sm->src[2])
                                       ? next_single_use_compute(sm, false) : nullptr;
            if (sm && mm2 && mm2->op == GGML_OP_MUL_MAT &&
                chain_uses_exact(mm2->src[0], sm) && mm2->src[1]) {
                struct ggml_tensor * cnt = mm2->src[1];
                while (cnt && cnt->op == GGML_OP_RESHAPE && cnt->src[0]) cnt = cnt->src[0];
                if (cnt && cnt->op == GGML_OP_CONT && cnt->src[0] &&
                    cnt->src[0]->op == GGML_OP_TRANSPOSE &&
                    cnt->src[0]->src[0] == mm1->src[0] &&
                    chain_uses_exact(mm2->src[1], cnt)) {
                    const struct ggml_tensor * states = mm1->src[0];
                    const struct ggml_tensor * q_in   = mm1->src[1];
                    const struct ggml_tensor * rep_n  = nullptr;
                    const struct ggml_tensor * orig_q = q_in;
                    if (q_in && q_in->op == GGML_OP_REPEAT && q_in->src[0] &&
                        q_in->ne[0] == q_in->src[0]->ne[0] && q_in->ne[1] == q_in->src[0]->ne[1] &&
                        q_in->ne[2] == states->ne[2] && q_in->ne[3] == 1) {
                        rep_n  = q_in;
                        orig_q = q_in->src[0];
                    }
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
                        agx_tensor_gpu_va(states) != 0 && agx_tensor_gpu_va(orig_q) != 0 && agx_tensor_gpu_va(mm2) != 0) {

                        const int32_t e_sm  = st->exec_idx_map.get(sm);
                        const int32_t e_cnt = st->exec_idx_map.get(cnt);
                        const int32_t e_mm2 = st->exec_idx_map.get(mm2);
                        const int32_t e_rep = rep_n ? st->exec_idx_map.get(rep_n) : -1;

                        int32_t fset[16];
                        int n_fset = 0;
                        fset[n_fset++] = (int32_t) e;
                        fset[n_fset++] = e_sm;
                        if (e_cnt > (int32_t) e) fset[n_fset++] = e_cnt;
                        fset[n_fset++] = e_mm2;

                        // Pre-check backward-absorbable REPEAT(query) and pre-emitted CONT(TRANSPOSE)
                        // so their dead writes do not falsely block absorbing CONCAT / RMS_NORM!
                        agx_va_range orig_q_r = tensor_va_range(orig_q);
                        int32_t e_abs_rep = -1;
                        if (rep_n && e_rep >= 0 && e_rep < (int32_t) e &&
                            st->use_map.get(rep_n) == 1u && (rep_n->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                            can_absorb_prev(e_rep, e, &orig_q_r, 1)) {
                            e_abs_rep = e_rep;
                        }
                        int32_t e_abs_cnt = -1;
                        if (e_cnt >= 0 && e_cnt < (int32_t) e && !consumed[(size_t) e_cnt] &&
                            anchors[(size_t) e_cnt].cnt == 1u &&
                            st->items[anchors[(size_t) e_cnt].beg].kind == ITEM_GENERIC_NODE) {
                            e_abs_cnt = e_cnt;
                        }

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
                                ex->ne[2] == states->ne[2] && ex->ne[3] == 1) {
                                int32_t e_cat = st->exec_idx_map.get(states);
                                agx_va_range cat_ins[2] = { tensor_va_range(bs), tensor_va_range(ex) };
                                if (can_absorb_prev(e_cat, e, cat_ins, 2, e_abs_rep, e_abs_cnt)) {
                                    e_abs_cat   = e_cat;
                                    base_states = bs;
                                    extra_t     = ex;
                                    const struct ggml_tensor * ex_r = strip_reshape(ex);
                                    if (ex_r && ex_r->op == GGML_OP_RMS_NORM &&
                                        (ex_r->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                        st->use_map.get(ex_r) == 1u &&
                                        ex_r->src[0] && ex_r->src[0]->type == GGML_TYPE_F32 &&
                                        ggml_is_contiguous(ex_r->src[0]) &&
                                        ggml_nelements(ex_r->src[0]) == ggml_nelements(ex)) {
                                        int32_t e_rms = st->exec_idx_map.get(ex_r);
                                        agx_va_range rms_in = tensor_va_range(ex_r->src[0]);
                                        if (can_absorb_prev(e_rms, e, &rms_in, 1, e_abs_cat, e_abs_rep, e_abs_cnt)) {
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
                                   ggml_is_contiguous(states->src[0])) {
                            int32_t e_rms = st->exec_idx_map.get(states);
                            agx_va_range rms_in = tensor_va_range(states->src[0]);
                            if (can_absorb_prev(e_rms, e, &rms_in, 1, e_abs_rep, e_abs_cnt)) {
                                e_abs_rms    = e_rms;
                                base_states  = nullptr;
                                extra_t      = states->src[0];
                                extra_do_rms = 1u;
                                std::memcpy(&extra_eps, states->op_params, sizeof(float));
                            }
                        }

                        // Wave 8: Try fusing trailing `[CONT(VIEW_3D)] -> RMS_NORM -> MUL(1D w)` (56/58 of rms=58!)
                        const int64_t D_mix = states->ne[0];
                        const int64_t Q_mix = orig_q->ne[1];
                        const int64_t M_mix = states->ne[2];
                        const uint64_t bs_va = base_states ? agx_tensor_gpu_va(base_states) : 0ull;
                        const uint64_t ex_va = extra_t     ? agx_tensor_gpu_va(extra_t)     : 0ull;
                        const uint64_t oq_va = agx_tensor_gpu_va(orig_q);
                        const bool q4_in_ok  = (Q_mix >= 1 && Q_mix <= 3) && ((D_mix & 3) == 0) &&
                                               (((bs_va | ex_va | oq_va) & 15ull) == 0ull);

                        auto match_mix_rms_mul = [&](const struct ggml_tensor * src_t,
                                                     float * out_eps,
                                                     const struct ggml_tensor ** out_w,
                                                     struct ggml_tensor ** out_rms,
                                                     struct ggml_tensor ** out_mul) -> bool {
                            struct ggml_tensor * rn = next_single_use_compute(src_t, false);
                            if (!rn || rn->op != GGML_OP_RMS_NORM ||
                                rn->ne[0] != D_mix || rn->ne[1] != M_mix || rn->ne[2] != 1 || rn->ne[3] != 1) {
                                return false;
                            }
                            struct ggml_tensor * mn = next_single_use_compute(rn, false);
                            if (!mn || mn->op != GGML_OP_MUL || mn->type != GGML_TYPE_F32 ||
                                !ggml_is_contiguous(mn) || !ggml_are_same_shape(rn, mn)) {
                                return false;
                            }
                            const uint64_t mva = agx_tensor_gpu_va(mn);
                            if (mva == 0ull || (mva & 15ull) != 0ull) return false;
                            const struct ggml_tensor * w = nullptr;
                            if (chain_uses_exact(mn->src[0], rn))      w = mn->src[1];
                            else if (chain_uses_exact(mn->src[1], rn)) w = mn->src[0];
                            if (!w || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) ||
                                w->ne[0] != D_mix || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1) {
                                return false;
                            }
                            const uint64_t wva = agx_tensor_gpu_va(w);
                            if (wva == 0ull || (wva & 15ull) != 0ull || !va_disjoint(mn, w)) return false;
                            std::memcpy(out_eps, rn->op_params, sizeof(float));
                            *out_w   = w;
                            *out_rms = rn;
                            *out_mul = mn;
                            return true;
                        };

                        uint32_t mix_norm_flags = 0u;
                        float    mix_norm_eps   = 1e-6f;
                        uint32_t mix_rms_cnt    = 0u;
                        uint32_t mix_cont_cnt   = 0u;
                        struct ggml_tensor *       mix_outs_t[3] = { mm2, nullptr, nullptr };
                        const struct ggml_tensor * mix_nw_t[3]   = { nullptr, nullptr, nullptr };
                        int32_t extra_fset[9];
                        int     n_extra_fset = 0;

                        if (!no_fuse_mix_rms && (fusion_mask & (1u << 4)) && q4_in_ok &&
                            (mm2->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && e_mm2 > (int32_t) e) {
                            if (Q_mix == 1 && st->use_map.get(mm2) == 1u) {
                                struct ggml_tensor * u0 = st->first_use.get(mm2);
                                struct ggml_tensor * c0 = nullptr;
                                if (u0 && u0->op == GGML_OP_VIEW && (u0->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                    u0->view_src == mm2 && u0->view_offs == 0u &&
                                    u0->ne[0] == 1 && u0->ne[1] == D_mix && u0->ne[2] == M_mix && u0->ne[3] == 1 &&
                                    u0->nb[1] == mm2->nb[1] && u0->nb[2] == mm2->nb[2] &&
                                    st->use_map.get(u0) == 1u) {
                                    struct ggml_tensor * cand_c = st->first_use.get(u0);
                                    if (cand_c && cand_c->op == GGML_OP_CONT && ggml_is_contiguous(cand_c) &&
                                        cand_c->type == GGML_TYPE_F32 && ggml_nelements(cand_c) == D_mix * M_mix) {
                                        c0 = cand_c;
                                    }
                                }
                                float eps0 = 1e-6f;
                                const struct ggml_tensor * nw0 = nullptr;
                                struct ggml_tensor * rn0 = nullptr, * mn0 = nullptr;
                                if (c0 && match_mix_rms_mul(c0, &eps0, &nw0, &rn0, &mn0)) {
                                    extra_fset[n_extra_fset++] = st->exec_idx_map.get(c0);
                                    extra_fset[n_extra_fset++] = st->exec_idx_map.get(rn0);
                                    extra_fset[n_extra_fset++] = st->exec_idx_map.get(mn0);
                                    mix_outs_t[0]  = mn0;
                                    mix_nw_t[0]    = nw0;
                                    mix_norm_eps   = eps0;
                                    mix_norm_flags = 3u;
                                    mix_rms_cnt    = 1u;
                                    mix_cont_cnt   = 1u;
                                } else if (!c0 && match_mix_rms_mul(mm2, &eps0, &nw0, &rn0, &mn0)) {
                                    extra_fset[n_extra_fset++] = st->exec_idx_map.get(rn0);
                                    extra_fset[n_extra_fset++] = st->exec_idx_map.get(mn0);
                                    mix_outs_t[0]  = mn0;
                                    mix_nw_t[0]    = nw0;
                                    mix_norm_eps   = eps0;
                                    mix_norm_flags = 2u;
                                    mix_rms_cnt    = 1u;
                                }
                            } else if ((Q_mix == 2 || Q_mix == 3) && st->use_map.get(mm2) == (uint32_t) Q_mix) {
                                struct ggml_tensor * c_arr[3] = {}, * r_arr[3] = {}, * m_arr[3] = {};
                                const struct ggml_tensor * w_arr[3] = {};
                                float eps_arr[3] = {};
                                int found_mask = 0;
                                const size_t scan_end = std::min(n_exec, (size_t) e_mm2 + 32u);
                                for (size_t j = (size_t) e_mm2 + 1u; j < scan_end; ++j) {
                                    if (consumed[j]) continue;
                                    struct ggml_tensor * cand = cgraph->nodes[st->exec_nodes[j]];
                                    if (cand->op != GGML_OP_CONT || !cand->src[0] || cand->src[0]->op != GGML_OP_VIEW) continue;
                                    const struct ggml_tensor * v = cand->src[0];
                                    if (v->view_src != mm2 || (v->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
                                        st->use_map.get(v) != 1u ||
                                        v->ne[0] != 1 || v->ne[1] != D_mix || v->ne[2] != M_mix || v->ne[3] != 1 ||
                                        v->nb[1] != mm2->nb[1] || v->nb[2] != mm2->nb[2]) continue;
                                    if ((v->view_offs % sizeof(float)) != 0u) continue;
                                    const int64_t qj = (int64_t) (v->view_offs / sizeof(float));
                                    if (qj < 0 || qj >= Q_mix || (found_mask & (1 << qj)) != 0) continue;
                                    if (match_mix_rms_mul(cand, &eps_arr[qj], &w_arr[qj], &r_arr[qj], &m_arr[qj])) {
                                        c_arr[qj] = cand;
                                        found_mask |= (1 << qj);
                                    }
                                }
                                if (found_mask == ((1 << Q_mix) - 1)) {
                                    bool multi_ok = true;
                                    for (int64_t qj = 0; qj < Q_mix && multi_ok; ++qj) {
                                        if (std::fabs(eps_arr[qj] - eps_arr[0]) > 1e-9f) multi_ok = false;
                                        for (int64_t qk = 0; qk < Q_mix; ++qk) {
                                            if (!va_disjoint(w_arr[qk], m_arr[qj])) multi_ok = false;
                                            if (qk < qj && !va_disjoint(m_arr[qk], m_arr[qj])) multi_ok = false;
                                        }
                                    }
                                    if (multi_ok) {
                                        for (int64_t qj = 0; qj < Q_mix; ++qj) {
                                            extra_fset[n_extra_fset++] = st->exec_idx_map.get(c_arr[qj]);
                                            extra_fset[n_extra_fset++] = st->exec_idx_map.get(r_arr[qj]);
                                            extra_fset[n_extra_fset++] = st->exec_idx_map.get(m_arr[qj]);
                                            mix_outs_t[qj] = m_arr[qj];
                                            mix_nw_t[qj]   = w_arr[qj];
                                        }
                                        mix_norm_eps   = eps_arr[0];
                                        mix_norm_flags = 3u;
                                        mix_rms_cnt    = (uint32_t) Q_mix;
                                        mix_cont_cnt   = (uint32_t) Q_mix;
                                    }
                                }
                            }
                        }

                        // Attempt fused mix + trailing rms_norm first; fall back cleanly to plain mix if anchor is blocked!
                        bool mix_committed = false;
                        for (int pass = (mix_norm_flags != 0u ? 0 : 1); pass < 2 && !mix_committed; ++pass) {
                            const bool with_rms = (pass == 0);
                            int cur_fset_n = n_fset;
                            if (with_rms) {
                                for (int k = 0; k < n_extra_fset; ++k) fset[cur_fset_n++] = extra_fset[k];
                            }

                            agx_va_range all_ins[6];
                            int32_t      all_prods[6];
                            int          n_all_ins = 0;
                            if (base_states) {
                                all_ins[n_all_ins]   = tensor_va_range(base_states);
                                all_prods[n_all_ins] = tensor_prod_idx(base_states);
                                n_all_ins++;
                            }
                            if (extra_t) {
                                all_ins[n_all_ins]   = tensor_va_range(extra_t);
                                all_prods[n_all_ins] = tensor_prod_idx(extra_t);
                                n_all_ins++;
                            }
                            all_ins[n_all_ins]   = orig_q_r;
                            all_prods[n_all_ins] = tensor_prod_idx(orig_q);
                            n_all_ins++;

                            agx_va_range extra_first[4];
                            int          n_extra_first = 0;
                            extra_first[n_extra_first++] = orig_q_r;

                            int n_outs = 1;
                            agx_va_range outs_r[3];
                            int32_t      outs_ex[3];
                            if (with_rms) {
                                n_outs = (Q_mix >= 2) ? (int) Q_mix : 1;
                                for (int qj = 0; qj < n_outs; ++qj) {
                                    outs_r[qj]  = tensor_va_range(mix_outs_t[qj]);
                                    outs_ex[qj] = st->exec_idx_map.get(mix_outs_t[qj]);
                                    all_ins[n_all_ins]       = tensor_va_range(mix_nw_t[qj]);
                                    all_prods[n_all_ins++]   = tensor_prod_idx(mix_nw_t[qj]);
                                    extra_first[n_extra_first++] = tensor_va_range(mix_nw_t[qj]);
                                }
                            } else {
                                outs_r[0]  = tensor_va_range(mm2);
                                outs_ex[0] = e_mm2;
                            }

                            struct ggml_tensor * out0_t = with_rms ? mix_outs_t[0] : mm2;
                            uint8_t mix_amask = 0u;
                            uint32_t n_alias_slots = 0u;
                            if (with_rms && n_outs >= 2) {
                                for (int qj = 0; qj < n_outs; ++qj) {
                                    const struct ggml_tensor * oq = mix_outs_t[qj];
                                    if ((base_states && !va_disjoint(base_states, oq)) ||
                                        (extra_t     && !va_disjoint(extra_t, oq)) ||
                                        !va_disjoint(orig_q, oq)) {
                                        mix_amask |= (uint8_t) (2u << qj);
                                        n_alias_slots++;
                                    }
                                }
                            } else {
                                if ((base_states && !va_disjoint(base_states, out0_t)) ||
                                    (extra_t     && !va_disjoint(extra_t, out0_t)) ||
                                    !va_disjoint(orig_q, out0_t) ||
                                    (with_rms && mix_nw_t[0] && !va_disjoint(mix_nw_t[0], out0_t))) {
                                    mix_amask     = 1u;
                                    n_alias_slots = 1u;
                                }
                            }
                            const bool mix_alias = (mix_amask != 0u);

                            const int32_t anc = find_safe_anchor(fset, cur_fset_n, all_ins, all_prods, n_all_ins,
                                                                 extra_first, n_extra_first, outs_r, outs_ex, n_outs,
                                                                 false, e_mm2);
                            if (anc >= 0 && (!mix_alias || !alias_scratch_locked[(size_t) anc])) {
                                if (e_abs_cat >= 0) { anchors[(size_t) e_abs_cat].cnt = 0u; consumed[(size_t) e_abs_cat] = 1u; }
                                if (e_abs_rms >= 0) { anchors[(size_t) e_abs_rms].cnt = 0u; consumed[(size_t) e_abs_rms] = 1u; }
                                if (e_abs_rep >= 0) { anchors[(size_t) e_abs_rep].cnt = 0u; consumed[(size_t) e_abs_rep] = 1u; }
                                if (e_abs_cnt >= 0) { anchors[(size_t) e_abs_cnt].cnt = 0u; consumed[(size_t) e_abs_cnt] = 1u; }

                                mix_attn_push mpc{};
                                mpc.states_va    = bs_va;
                                mpc.extra_va     = ex_va;
                                mpc.queries_va   = oq_va;
                                mpc.out_va       = agx_tensor_gpu_va(out0_t);
                                mpc.out1_va      = (with_rms && n_outs >= 2) ? agx_tensor_gpu_va(mix_outs_t[1]) : 0ull;
                                mpc.out2_va      = (with_rms && n_outs >= 3) ? agx_tensor_gpu_va(mix_outs_t[2]) : 0ull;
                                mpc.norm_w0_va   = (with_rms && mix_nw_t[0]) ? agx_tensor_gpu_va(mix_nw_t[0])   : 0ull;
                                mpc.norm_w1_va   = (with_rms && mix_nw_t[1]) ? agx_tensor_gpu_va(mix_nw_t[1])   : 0ull;
                                mpc.norm_w2_va   = (with_rms && mix_nw_t[2]) ? agx_tensor_gpu_va(mix_nw_t[2])   : 0ull;
                                mpc.n_embd       = (uint32_t) D_mix;
                                mpc.n_states     = (uint32_t) states->ne[1];
                                mpc.n_queries    = (uint32_t) Q_mix;
                                mpc.n_tokens     = (uint32_t) M_mix;
                                mpc.q_ne2        = (uint32_t) orig_q->ne[2];
                                mpc.extra_do_rms = extra_do_rms;
                                mpc.scale        = sm_scale;
                                mpc.extra_eps    = extra_eps;
                                mpc.norm_eps     = with_rms ? mix_norm_eps   : 1e-6f;
                                mpc.norm_flags   = with_rms ? mix_norm_flags : 0u;

                                agx_exec_item it{};
                                it.kind          = ITEM_MIX_ATTN;
                                const bool fused_q4 = (mpc.n_queries <= 4u) && ((mpc.n_embd & 3u) == 0u) &&
                                                      (((mpc.states_va | mpc.extra_va | mpc.queries_va) & 15ull) == 0ull);
                                it.gx            = fused_q4 ? 1u : mpc.n_queries;
                                it.gy            = mpc.n_tokens;
                                it.gz            = 1u;
                                it.nodes_covered = (uint32_t) cur_fset_n
                                                 + (e_abs_cat >= 0 ? 1u : 0u) + (e_abs_rms >= 0 ? 1u : 0u)
                                                 + (e_abs_rep >= 0 ? 1u : 0u) + (e_abs_cnt >= 0 ? 1u : 0u);
                                if (mix_alias) {
                                    const uint32_t one_floats = (with_rms ? 1u : mpc.n_queries) * mpc.n_embd * mpc.n_tokens;
                                    const size_t   slot_B     = agx_align_up((size_t) one_floats * sizeof(float), 256u);
                                    it.uses_alias_scratch = true;
                                    it.alias_flags        = mix_amask;
                                    it.alias_floats0      = one_floats;
                                    it.alias_tmp_off_B    = (uint32_t) slot_B;
                                    it.alias_tmp_need_B   = agx_align_up(slot_B * n_alias_slots, AGX_PAGE_SIZE);
                                }
                                for (int r = 0; r < n_all_ins; ++r) it.add_read(all_ins[r]);
                                for (int w = 0; w < n_outs; ++w)    it.add_write(outs_r[w]);
                                it.set_push(mpc);
                                commit_single_item(anc, fset, cur_fset_n, it);
                                st->stats.fused_mix_hits++;
                                g_mix++;
                                if (with_rms) {
                                    st->stats.fused_rms_mul_hits += mix_rms_cnt;
                                    g_rms_mul                    += mix_rms_cnt;
                                    g_mix_rms                    += mix_rms_cnt;
                                    st->stats.fused_cont_absorbed += mix_cont_cnt;
                                    g_cont_abs                    += mix_cont_cnt;
                                }
                                mix_committed = true;
                            }
                        }
                        if (mix_committed) continue;
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 13 (Wave 6 NEW): Full Fused MoE Block (`M = 1..8192`)
        //   MUL_MAT_ID(gate_up) -> [NVFP4 scale2] -> SiTU / SwiGLU / GeGLU ->
        //   MUL_MAT_ID(down)    -> [NVFP4 scale2] -> MUL(router_w) -> (K-1)x ADD
        //   Collapsed into ONE `agx_moe_dispatch_fused_situ_down` call!
        //   Eliminates ~150+ ms/ubatch of unfused scale2/SiTU/wsum DRAM traffic on pp512/pp2048!
        // --------------------------------------------------------------------
        if ((fusion_mask & (1u << 14)) && node->op == GGML_OP_MUL_MAT_ID &&
            (node->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && node->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(node) && agx_moe_supports_mul_mat_id(node)) {
            struct ggml_tensor * mm_gu = node;
            const struct ggml_tensor * w_gu = mm_gu->src[0];
            const struct ggml_tensor * x_in = mm_gu->src[1];
            const struct ggml_tensor * ids  = mm_gu->src[2];
            agx_weight_layout gu_layout = AGX_WEIGHT_F32;
            const int64_t n_embd   = w_gu->ne[0];
            const int64_t n_ff     = w_gu->ne[1] / 2;
            const int64_t n_expert = w_gu->ne[2];
            const int64_t top_k    = ids->ne[0];
            const int64_t M        = ids->ne[1];
            if (x_in->ne[1] == 1 && (w_gu->ne[1] & 1) == 0 && n_ff > 0 && n_embd > 0 &&
                M >= 1 && M <= 8192 && top_k >= 2 && top_k <= 32 &&
                n_expert > 0 && n_expert <= (int64_t) AGX_MOE_MAX_EXPERTS &&
                agx_gemm_infer_weight_layout(w_gu, &gu_layout) &&
                (gu_layout != AGX_WEIGHT_NVFP4_REPACKED || (n_ff & 15) == 0)) {

                struct mmid_scale2_match {
                    struct ggml_tensor *       out = nullptr;
                    const struct ggml_tensor * w_s = nullptr;
                    struct ggml_tensor *       rep = nullptr;
                    struct ggml_tensor *       gr  = nullptr;
                    struct ggml_tensor *       mul = nullptr;
                };
                auto match_mmid_scale2 = [&](struct ggml_tensor * mm_n) -> mmid_scale2_match {
                    mmid_scale2_match m;
                    m.out = mm_n;
                    if (st->use_map.get(mm_n) != 1u) return m;
                    struct ggml_tensor * cand = st->first_use.get(mm_n);
                    if (!cand || cand->op != GGML_OP_MUL || (cand->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
                        cand->type != GGML_TYPE_F32 || !ggml_is_contiguous(cand)) return m;
                    struct ggml_tensor * s_gr = (cand->src[0] == mm_n) ? cand->src[1]
                                              : ((cand->src[1] == mm_n) ? cand->src[0] : nullptr);
                    if (!s_gr || s_gr->op != GGML_OP_GET_ROWS || (s_gr->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
                        st->use_map.get(s_gr) != 1u || s_gr->src[1] != ids || !s_gr->src[0]) return m;
                    struct ggml_tensor * s_rep = nullptr;
                    const struct ggml_tensor * w_s = nullptr;
                    if (s_gr->src[0]->op == GGML_OP_REPEAT &&
                        (s_gr->src[0]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                        st->use_map.get(s_gr->src[0]) == 1u && s_gr->src[0]->src[0]) {
                        s_rep = s_gr->src[0];
                        w_s   = strip_reshape(s_rep->src[0]);
                    } else {
                        w_s   = strip_reshape(s_gr->src[0]);
                    }
                    if (!w_s || w_s->type != GGML_TYPE_F32 || !ggml_is_contiguous(w_s) ||
                        ggml_nelements(w_s) != n_expert || agx_tensor_gpu_va(w_s) == 0) return m;
                    m.out = cand;
                    m.w_s = w_s;
                    m.rep = s_rep;
                    m.gr  = s_gr;
                    m.mul = cand;
                    return m;
                };

                mmid_scale2_match gu_m = match_mmid_scale2(mm_gu);
                struct ggml_tensor * gu_out = gu_m.out;

                // Locate downstream `mm_dn = MUL_MAT_ID(down_exps, act_out, ids)` in window
                struct ggml_tensor * mm_dn = nullptr;
                const size_t w_dn_end = std::min(n_exec, e + 48u);
                for (size_t j = e + 1u; j < w_dn_end; ++j) {
                    if (consumed[j]) continue;
                    struct ggml_tensor * cand = cgraph->nodes[st->exec_nodes[j]];
                    if (cand->op == GGML_OP_MUL_MAT_ID && cand->src[2] == ids &&
                        cand->src[0] && cand->src[0]->ne[0] == n_ff &&
                        cand->src[0]->ne[1] == n_embd && cand->src[0]->ne[2] == n_expert &&
                        (cand->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && cand->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(cand) && agx_moe_supports_mul_mat_id(cand)) {
                        mm_dn = cand;
                        break;
                    }
                }

                agx_weight_layout dn_layout = AGX_WEIGHT_F32;
                if (mm_dn && mm_dn->src[1] &&
                    st->use_map.get(gu_out) == 2u &&
                    st->use_map.get(mm_dn->src[1]) == 1u &&
                    (mm_dn->src[1]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                    agx_gemm_infer_weight_layout(mm_dn->src[0], &dn_layout)) {

                    auto is_gu_half_view = [&](const struct ggml_tensor * v, size_t off_B) -> bool {
                        return v && v->op == GGML_OP_VIEW && v->view_src == gu_out && v->view_offs == off_B &&
                               v->ne[0] == n_ff && v->ne[1] == top_k && v->ne[2] == M && v->ne[3] == 1 &&
                               v->nb[0] == sizeof(float) && v->nb[1] == gu_out->nb[1] && v->nb[2] == gu_out->nb[2];
                    };

                    struct ggml_tensor * act_out = mm_dn->src[1];
                    struct ggml_tensor * act_nodes[10] = {};
                    int n_act_nodes = 0;
                    bool act_ok = false;
                    agx_moe_act_op act_op = AGX_MOE_ACT_SITU;
                    float situ_beta = 0.0f, situ_lb = 0.0f;

                    if (act_out->op == GGML_OP_MUL && act_out->src[0] && act_out->src[1]) {
                        auto try_situ_order = [&](struct ggml_tensor * m1, struct ggml_tensor * up_br) -> bool {
                            if (!m1 || m1->op != GGML_OP_MUL || st->use_map.get(m1) != 1u ||
                                (m1->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || !m1->src[0] || !m1->src[1]) return false;
                            // Match gate branch: m1 = MUL(SCALE(TANH(SCALE(gate_v, 1/b)), b), SIGMOID(gate_v))
                            struct ggml_tensor * s2 = nullptr, * sg = nullptr;
                            if (m1->src[0]->op == GGML_OP_SCALE && m1->src[1]->op == GGML_OP_UNARY) {
                                s2 = m1->src[0]; sg = m1->src[1];
                            } else if (m1->src[1]->op == GGML_OP_SCALE && m1->src[0]->op == GGML_OP_UNARY) {
                                s2 = m1->src[1]; sg = m1->src[0];
                            } else {
                                return false;
                            }
                            if (ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID ||
                                st->use_map.get(s2) != 1u || (s2->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
                                st->use_map.get(sg) != 1u || (sg->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return false;
                            struct ggml_tensor * t1 = s2->src[0];
                            if (!t1 || t1->op != GGML_OP_UNARY || ggml_get_unary_op(t1) != GGML_UNARY_OP_TANH ||
                                st->use_map.get(t1) != 1u || (t1->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return false;
                            struct ggml_tensor * s1 = t1->src[0];
                            if (!s1 || s1->op != GGML_OP_SCALE ||
                                st->use_map.get(s1) != 1u || (s1->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return false;
                            const struct ggml_tensor * gate_v = s1->src[0];
                            if (!gate_v || sg->src[0] != gate_v ||
                                !is_gu_half_view(gate_v, 0u) || st->use_map.get(gate_v) != 2u) return false;
                            float k1 = 0, b1 = 0, k2 = 0, b2 = 0;
                            read_scale_params(s1, &k1, &b1);
                            read_scale_params(s2, &k2, &b2);
                            if (b1 != 0.0f || b2 != 0.0f || k2 <= 0.0f || std::fabs(k1 * k2 - 1.0f) >= 1e-4f) return false;

                            // Match up branch: either SCALE(TANH(SCALE(up_v, 1/lb)), lb) or raw up_v
                            int na = 0;
                            act_nodes[na++] = s1;
                            act_nodes[na++] = t1;
                            act_nodes[na++] = s2;
                            act_nodes[na++] = sg;
                            act_nodes[na++] = m1;
                            float lb = 0.0f;
                            if (up_br && up_br->op == GGML_OP_SCALE) {
                                struct ggml_tensor * s4 = up_br;
                                if (st->use_map.get(s4) != 1u || (s4->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return false;
                                struct ggml_tensor * t2 = s4->src[0];
                                if (!t2 || t2->op != GGML_OP_UNARY || ggml_get_unary_op(t2) != GGML_UNARY_OP_TANH ||
                                    st->use_map.get(t2) != 1u || (t2->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return false;
                                struct ggml_tensor * s3 = t2->src[0];
                                if (!s3 || s3->op != GGML_OP_SCALE ||
                                    st->use_map.get(s3) != 1u || (s3->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) return false;
                                const struct ggml_tensor * up_v = s3->src[0];
                                if (!is_gu_half_view(up_v, (size_t) n_ff * sizeof(float)) || st->use_map.get(up_v) != 1u) return false;
                                float k3 = 0, b3 = 0, k4 = 0, b4 = 0;
                                read_scale_params(s3, &k3, &b3);
                                read_scale_params(s4, &k4, &b4);
                                if (b3 != 0.0f || b4 != 0.0f || k4 <= 0.0f || std::fabs(k3 * k4 - 1.0f) >= 1e-4f) return false;
                                lb = k4;
                                act_nodes[na++] = s3;
                                act_nodes[na++] = t2;
                                act_nodes[na++] = s4;
                            } else if (is_gu_half_view(up_br, (size_t) n_ff * sizeof(float)) && st->use_map.get(up_br) == 1u) {
                                lb = 0.0f;
                            } else {
                                return false;
                            }
                            act_nodes[na++] = act_out;
                            n_act_nodes = na;
                            act_op    = AGX_MOE_ACT_SITU;
                            situ_beta = k2;
                            situ_lb   = lb;
                            return true;
                        };
                        act_ok = try_situ_order(act_out->src[0], act_out->src[1]) ||
                                 try_situ_order(act_out->src[1], act_out->src[0]);
                    } else if (act_out->op == GGML_OP_GLU && act_out->src[0] && act_out->src[1] &&
                               ggml_get_op_params_i32(act_out, 1) == 0 &&
                               is_gu_half_view(act_out->src[0], 0u) && st->use_map.get(act_out->src[0]) == 1u &&
                               is_gu_half_view(act_out->src[1], (size_t) n_ff * sizeof(float)) && st->use_map.get(act_out->src[1]) == 1u) {
                        const auto gop = ggml_get_glu_op(act_out);
                        if (gop == GGML_GLU_OP_SWIGLU || gop == GGML_GLU_OP_GEGLU) {
                            act_op = (gop == GGML_GLU_OP_SWIGLU) ? AGX_MOE_ACT_SILU : AGX_MOE_ACT_GELU;
                            act_nodes[0] = act_out;
                            n_act_nodes  = 1;
                            act_ok       = true;
                        }
                    }

                    if (act_ok) {
                        mmid_scale2_match dn_m = match_mmid_scale2(mm_dn);
                        struct ggml_tensor * dn_out = dn_m.out;
                        struct ggml_tensor * mul_w  = next_single_use_compute(dn_out, false);
                        const struct ggml_tensor * weights = nullptr;
                        if (mul_w && mul_w->op == GGML_OP_MUL && mul_w->type == GGML_TYPE_F32 &&
                            (mul_w->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && ggml_is_contiguous(mul_w) &&
                            mul_w->ne[0] == n_embd && mul_w->ne[1] == top_k && mul_w->ne[2] == M && mul_w->ne[3] == 1 &&
                            st->use_map.get(mul_w) == (uint32_t) top_k) {
                            weights = (mul_w->src[0] == dn_out) ? mul_w->src[1]
                                    : ((mul_w->src[1] == dn_out) ? mul_w->src[0] : nullptr);
                        }
                        if (weights && weights->type == GGML_TYPE_F32 && ggml_is_contiguous(weights) &&
                            weights->ne[0] == 1 && weights->ne[1] == top_k && weights->ne[2] == M && weights->ne[3] == 1 &&
                            agx_tensor_gpu_va(weights) != 0) {
                            const int32_t e_mul_w = st->exec_idx_map.get(mul_w);
                            auto is_w_view = [&](const struct ggml_tensor * v, int64_t k_idx) -> bool {
                                return v && v->op == GGML_OP_VIEW && v->view_src == mul_w &&
                                       v->view_offs == (size_t) k_idx * mul_w->nb[1] &&
                                       v->ne[0] == n_embd && v->ne[1] == M && v->ne[2] == 1 && v->ne[3] == 1 &&
                                       v->nb[1] == mul_w->nb[2] &&
                                       st->use_map.get(v) == 1u && (v->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
                            };
                            bool adds_ok = (e_mul_w > (int32_t) e && (size_t) e_mul_w + (size_t) (top_k - 1) < n_exec);
                            struct ggml_tensor * prev_add = nullptr;
                            int32_t add_idxs[32];
                            int     n_adds = 0;
                            for (int64_t k = 1; adds_ok && k < top_k; ++k) {
                                const size_t ek = (size_t) e_mul_w + (size_t) k;
                                if (consumed[ek]) { adds_ok = false; break; }
                                struct ggml_tensor * ak = cgraph->nodes[st->exec_nodes[ek]];
                                if (ak->op != GGML_OP_ADD || ak->type != GGML_TYPE_F32) { adds_ok = false; break; }
                                if (k == 1) {
                                    if (!((is_w_view(ak->src[0], 0) && is_w_view(ak->src[1], 1)) ||
                                          (is_w_view(ak->src[1], 0) && is_w_view(ak->src[0], 1)))) {
                                        adds_ok = false; break;
                                    }
                                } else {
                                    if (!((chain_uses_exact(ak->src[0], prev_add) && is_w_view(ak->src[1], k)) ||
                                          (chain_uses_exact(ak->src[1], prev_add) && is_w_view(ak->src[0], k)))) {
                                        adds_ok = false; break;
                                    }
                                }
                                prev_add = ak;
                                add_idxs[n_adds++] = (int32_t) ek;
                            }

                            if (adds_ok && prev_add && ggml_is_contiguous(prev_add) &&
                                prev_add->ne[0] == n_embd && prev_add->ne[1] == M &&
                                prev_add->ne[2] == 1 && prev_add->ne[3] == 1 &&
                                agx_tensor_gpu_va(prev_add) != 0) {
                                // Build fset[] (all forward exec nodes >= e) and optional bset[] (scale2 helpers < e)
                                int32_t fset[64];
                                int     n_fset = 0;
                                int32_t bset[8];
                                int     n_bset = 0;
                                auto add_exec_node = [&](struct ggml_tensor * t) -> bool {
                                    if (!t) return true;
                                    int32_t ei = st->exec_idx_map.get(t);
                                    if (ei < 0) return false;
                                    if (ei < (int32_t) e) {
                                        for (int b = 0; b < n_bset; ++b) if (bset[b] == ei) return true;
                                        if (n_bset < 8) bset[n_bset++] = ei;
                                        return true;
                                    }
                                    for (int f = 0; f < n_fset; ++f) if (fset[f] == ei) return true;
                                    if (n_fset >= 64) return false;
                                    fset[n_fset++] = ei;
                                    return true;
                                };
                                bool set_ok = add_exec_node(mm_gu) &&
                                              add_exec_node(gu_m.rep) && add_exec_node(gu_m.gr) && add_exec_node(gu_m.mul) &&
                                              add_exec_node(mm_dn) &&
                                              add_exec_node(dn_m.rep) && add_exec_node(dn_m.gr) && add_exec_node(dn_m.mul) &&
                                              add_exec_node(mul_w);
                                for (int k = 0; set_ok && k < n_act_nodes; ++k) set_ok = add_exec_node(act_nodes[k]);
                                for (int k = 0; set_ok && k < n_adds; ++k) {
                                    if (n_fset >= 64) { set_ok = false; break; }
                                    fset[n_fset++] = add_idxs[k];
                                }

                                if (set_ok) {
                                    agx_va_range all_ins[7];
                                    int32_t      all_prods[7];
                                    int          n_ins = 0;
                                    auto push_in = [&](const struct ggml_tensor * t) {
                                        if (!t) return;
                                        all_ins[n_ins]   = tensor_va_range(t);
                                        all_prods[n_ins] = tensor_prod_idx(t);
                                        n_ins++;
                                    };
                                    push_in(w_gu);
                                    push_in(x_in);
                                    push_in(ids);
                                    push_in(mm_dn->src[0]);
                                    push_in(weights);
                                    push_in(gu_m.w_s);
                                    push_in(dn_m.w_s);

                                    agx_va_range extra_ins[4];
                                    int          n_extra = 0;
                                    extra_ins[n_extra++] = tensor_va_range(mm_dn->src[0]);
                                    extra_ins[n_extra++] = tensor_va_range(weights);
                                    if (gu_m.w_s) extra_ins[n_extra++] = tensor_va_range(gu_m.w_s);
                                    if (dn_m.w_s) extra_ins[n_extra++] = tensor_va_range(dn_m.w_s);
                                    agx_va_range out_r  = tensor_va_range(prev_add);
                                    int32_t      out_ex = st->exec_idx_map.get(prev_add);

                                    const bool moe_alias = !va_disjoint(x_in, prev_add) ||
                                                           !va_disjoint(ids, prev_add) ||
                                                           !va_disjoint(weights, prev_add);
                                    const int32_t anc = find_safe_anchor(fset, n_fset, all_ins, all_prods, n_ins,
                                                                         extra_ins, n_extra, &out_r, &out_ex, 1);
                                    if (anc >= 0 && (!moe_alias || !alias_scratch_locked[(size_t) anc])) {
                                        uint32_t abs_cnt = 0u;
                                        for (int b = 0; b < n_bset; ++b) {
                                            if (can_absorb_prev(bset[b], e, all_ins, n_ins)) {
                                                anchors[(size_t) bset[b]].cnt = 0u;
                                                consumed[(size_t) bset[b]]    = 1u;
                                                abs_cnt++;
                                            }
                                        }
                                        const struct ggml_tensor * w_dn = mm_dn->src[0];
                                        agx_moe_stage_args ma{};
                                        ma.n_tokens         = (uint32_t) M;
                                        ma.n_embd           = (uint32_t) n_embd;
                                        ma.n_ff             = (uint32_t) n_ff;
                                        ma.n_expert         = (uint32_t) n_expert;
                                        ma.top_k            = (uint32_t) top_k;
                                        ma.act_op           = act_op;
                                        ma.situ_beta        = situ_beta;
                                        ma.situ_linear_beta = situ_lb;
                                        ma.exp_scale0       = 1.0f;
                                        ma.ffn_x_va         = agx_tensor_gpu_va(x_in);
                                        ma.topk_ids_va      = agx_tensor_gpu_va(ids);
                                        ma.ids_row_stride   = (uint32_t) (ids->nb[1] / sizeof(int32_t));
                                        ma.topk_w_va        = agx_tensor_gpu_va(weights);
                                        ma.moe_out_va       = agx_tensor_gpu_va(prev_add);
                                        ma.gate_up_va       = agx_tensor_gpu_va(w_gu);
                                        ma.down_va          = agx_tensor_gpu_va(w_dn);
                                        ma.gate_up_slice_B  = agx_tensor_is_repacked(w_gu)
                                            ? (uint64_t) agx_repack_slice_stride(w_gu->type, w_gu->ne[0], w_gu->ne[1])
                                            : (uint64_t) w_gu->nb[2];
                                        ma.down_slice_B     = agx_tensor_is_repacked(w_dn)
                                            ? (uint64_t) agx_repack_slice_stride(w_dn->type, w_dn->ne[0], w_dn->ne[1])
                                            : (uint64_t) w_dn->nb[2];
                                        ma.gate_up_layout   = gu_layout;
                                        ma.down_layout      = dn_layout;
                                        ma.gate_up_scale_va = gu_m.w_s ? agx_tensor_gpu_va(gu_m.w_s) : 0ull;
                                        ma.down_scale_va    = dn_m.w_s ? agx_tensor_gpu_va(dn_m.w_s) : 0ull;

                                        agx_exec_item it{};
                                        it.kind             = ITEM_FUSED_MOE;
                                        it.nodes_covered    = (uint32_t) n_fset + abs_cnt;
                                        it.uses_moe_scratch = true;
                                        it.moe_args         = ma;
                                        if (moe_alias) {
                                            const uint32_t out_floats = ma.n_tokens * ma.n_embd;
                                            it.uses_alias_scratch = true;
                                            it.alias_flags        = 1u;
                                            it.alias_floats0      = out_floats;
                                            it.alias_tmp_need_B   = agx_align_up((size_t) out_floats * sizeof(float), AGX_PAGE_SIZE);
                                        }
                                        for (int r = 0; r < n_ins; ++r) it.add_read(all_ins[r]);
                                        it.add_write(out_r);
                                        commit_single_item(anc, fset, n_fset, it);
                                        st->stats.fused_moe_wsum_hits++;
                                        g_fused_moe++;
                                        continue;
                                    }
                                }
                            }
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

                            const struct ggml_tensor * w = ssm_n->src[1];

                            // Wave 8 High-ROI Fusion 1: Fuse [K_conv / V_conv + L2_CLIP / RMS_NORM + [tau] + [FWHT] + [SET_ROWS]]!
                            if (!no_fuse_sconv_prep && (fusion_mask & (1u << 0))) {
                                if (struct ggml_tensor * u0 = next_single_use_compute(sc_res, false)) {
                                    const struct ggml_tensor * clip_sc_t = nullptr;
                                    struct ggml_tensor *       norm_n    = nullptr;
                                    struct ggml_tensor *       tail_n    = nullptr;
                                    bool  is_l2_clip = false;
                                    float norm_param = 0.0f;
                                    bool  p_head_ok  = false;
                                    int32_t p_fset[48];
                                    int p_fn = 0;
                                    for (int k = 0; k < fn; ++k) p_fset[p_fn++] = fset[k];

                                    if (u0->op == GGML_OP_MUL && u0->type == GGML_TYPE_F32 &&
                                        u0->ne[0] > 0 && u0->ne[0] <= 128 && u0->ne[1] > 0 &&
                                        u0->ne[0] * u0->ne[1] == n_ch && u0->ne[2] == n_t * n_s && u0->ne[3] == 1) {
                                        struct ggml_tensor * l2n = next_single_use_compute(u0, false);
                                        struct ggml_tensor * scn = (l2n && l2n->op == GGML_OP_L2_NORM)
                                            ? next_single_use_compute(l2n, false) : nullptr;
                                        if (l2n && scn && scn->op == GGML_OP_SCALE) {
                                            float t0 = 0.0f, s0 = 0.0f, b0 = 0.0f;
                                            std::memcpy(&t0, l2n->op_params, sizeof(float));
                                            read_scale_params(scn, &s0, &b0);
                                            const struct ggml_tensor * a = u0->src[0];
                                            const struct ggml_tensor * b = u0->src[1];
                                            const int64_t Dp = u0->ne[0], Hp = u0->ne[1], Mp = u0->ne[2];
                                            if (b0 == 0.0f && t0 > 0.0f && std::fabs(t0 - s0) < 1e-6f && a && b &&
                                                a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
                                                ggml_is_contiguous(a) && ggml_is_contiguous(b)) {
                                                if (strip_reshape(a) == sc_res && a->ne[0] == Dp && a->ne[1] == Hp && a->ne[2] == Mp &&
                                                    b->ne[0] == Dp && b->ne[1] == Hp && b->ne[2] == 1 && b->ne[3] == 1) {
                                                    clip_sc_t = b;
                                                } else if (strip_reshape(b) == sc_res && b->ne[0] == Dp && b->ne[1] == Hp && b->ne[2] == Mp &&
                                                           a->ne[0] == Dp && a->ne[1] == Hp && a->ne[2] == 1 && a->ne[3] == 1) {
                                                    clip_sc_t = a;
                                                }
                                                if (clip_sc_t) {
                                                    norm_n = l2n; tail_n = scn;
                                                    is_l2_clip = true; norm_param = t0;
                                                    p_fset[p_fn++] = st->exec_idx_map.get(u0);
                                                    p_fset[p_fn++] = st->exec_idx_map.get(l2n);
                                                    p_fset[p_fn++] = st->exec_idx_map.get(scn);
                                                    p_head_ok = true;
                                                }
                                            }
                                        }
                                    } else if (u0->op == GGML_OP_L2_NORM &&
                                               u0->ne[0] > 0 && u0->ne[0] <= 128 && u0->ne[1] > 0 &&
                                               u0->ne[0] * u0->ne[1] == n_ch && u0->ne[2] == n_t * n_s && u0->ne[3] == 1 &&
                                               u0->src[0] && strip_reshape(u0->src[0]) == sc_res) {
                                        struct ggml_tensor * scn = next_single_use_compute(u0, false);
                                        if (scn && scn->op == GGML_OP_SCALE && ggml_is_contiguous(scn)) {
                                            float t0 = 0.0f, s0 = 0.0f, b0 = 0.0f;
                                            std::memcpy(&t0, u0->op_params, sizeof(float));
                                            read_scale_params(scn, &s0, &b0);
                                            if (b0 == 0.0f && t0 > 0.0f && std::fabs(t0 - s0) < 1e-6f) {
                                                norm_n = u0; tail_n = scn;
                                                is_l2_clip = true; norm_param = t0;
                                                p_fset[p_fn++] = st->exec_idx_map.get(u0);
                                                p_fset[p_fn++] = st->exec_idx_map.get(scn);
                                                p_head_ok = true;
                                            }
                                        }
                                    } else if (u0->op == GGML_OP_RMS_NORM &&
                                               u0->ne[0] > 0 && u0->ne[0] <= 128 && u0->ne[1] > 0 &&
                                               u0->ne[0] * u0->ne[1] == n_ch && u0->ne[2] == n_t * n_s && u0->ne[3] == 1 &&
                                               u0->src[0] && strip_reshape(u0->src[0]) == sc_res && ggml_is_contiguous(u0)) {
                                        norm_n = u0; tail_n = u0;
                                        is_l2_clip = false;
                                        std::memcpy(&norm_param, u0->op_params, sizeof(float));
                                        p_fset[p_fn++] = st->exec_idx_map.get(u0);
                                        p_head_ok = true;
                                    }

                                    if (p_head_ok) {
                                        const int64_t Dp = norm_n->ne[0];
                                        const int64_t Hp = norm_n->ne[1];
                                        const int64_t Mp = norm_n->ne[2];

                                        const struct ggml_tensor * tau_t = nullptr;
                                        if (struct ggml_tensor * ut = next_single_use_compute(tail_n, false)) {
                                            if (ut->op == GGML_OP_MUL && ut->type == GGML_TYPE_F32 && ggml_is_contiguous(ut) &&
                                                ut->ne[0] == Dp && ut->ne[1] == Hp && ut->ne[2] == Mp && ut->ne[3] == 1) {
                                                const struct ggml_tensor * other = nullptr;
                                                if (strip_reshape(ut->src[0]) == tail_n) other = ut->src[1];
                                                else if (strip_reshape(ut->src[1]) == tail_n) other = ut->src[0];
                                                if (other && other->type == GGML_TYPE_F32 && ggml_is_contiguous(other) &&
                                                    other->ne[0] == 1 && other->ne[1] == 1 && other->ne[2] == Mp && other->ne[3] == 1 &&
                                                    agx_tensor_gpu_va(other) != 0) {
                                                    tau_t  = other;
                                                    tail_n = ut;
                                                    p_fset[p_fn++] = st->exec_idx_map.get(ut);
                                                }
                                            }
                                        }

                                        uint32_t fwht_flag = 0u;
                                        if (struct ggml_tensor * uf = next_single_use_compute(tail_n, false)) {
                                            if (uf->op == GGML_OP_MUL_MAT &&
                                                ggml_get_op_params_i32(uf, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                                                uf->src[1] && strip_reshape(uf->src[1]) == tail_n &&
                                                uf->type == GGML_TYPE_F32 && ggml_is_contiguous(uf) &&
                                                ggml_nelements(uf) == Dp * Hp * Mp) {
                                                const int64_t fd = uf->src[1]->ne[0];
                                                if ((fd == 128 && Dp == 128) || (fd == 64 && (Dp == 64 || Dp == 128))) {
                                                    fwht_flag = (fd == 128) ? 8u : 4u;
                                                    tail_n    = uf;
                                                    p_fset[p_fn++] = st->exec_idx_map.get(uf);
                                                }
                                            }
                                        }

                                        struct ggml_tensor * sr_n = nullptr;
                                        uint32_t store_mode = 0u;
                                        if (struct ggml_tensor * us = next_single_use_compute(tail_n, true)) {
                                            if (us->op == GGML_OP_SET_ROWS && us->src[0] && us->src[1] &&
                                                strip_reshape_fullview(us->src[0]) == tail_n &&
                                                us->src[0]->ne[0] == Dp * Hp && us->src[0]->ne[1] == Mp &&
                                                us->src[0]->ne[2] == 1 && us->src[0]->ne[3] == 1 &&
                                                us->src[1]->ne[0] == Mp && us->src[1]->ne[1] == 1 && us->src[1]->ne[2] == 1 &&
                                                us->ne[2] == 1 && us->ne[3] == 1 &&
                                                (us->src[1]->type == GGML_TYPE_I64 || us->src[1]->type == GGML_TYPE_I32)) {
                                                if (us->type == GGML_TYPE_F32) store_mode = 1u;
                                                else if (us->type == GGML_TYPE_F16) store_mode = 2u;
                                                else if (us->type == GGML_TYPE_Q8_0 && (Dp % 32) == 0) store_mode = 3u;
                                                if (store_mode != 0u) {
                                                    sr_n   = us;
                                                    tail_n = us;
                                                    p_fset[p_fn++] = st->exec_idx_map.get(us);
                                                }
                                            }
                                        }

                                        const struct ggml_tensor * ids_t = sr_n ? sr_n->src[1] : nullptr;
                                        const bool sp_dst_ok =
                                            agx_tensor_gpu_va(tail_n) != 0ull &&
                                            va_disjoint(past, tail_n) && va_disjoint(w, tail_n) &&
                                            va_disjoint(dyn_t, tail_n) && va_disjoint(cpy_n, tail_n) &&
                                            va_disjoint(clip_sc_t, tail_n) && va_disjoint(tau_t, tail_n) &&
                                            va_disjoint(ids_t, tail_n) &&
                                            ((store_mode != 0u || n_t > 1) ? va_disjoint(cur, tail_n) : va_same_or_disjoint(cur, tail_n));
                                        const bool sp_cell_ok = (!cpy_n) || (
                                            va_disjoint(cur, cpy_n) && va_disjoint(w, cpy_n) &&
                                            va_disjoint(dyn_t, cpy_n) && va_disjoint(clip_sc_t, cpy_n) &&
                                            va_disjoint(tau_t, cpy_n) && va_disjoint(ids_t, cpy_n) &&
                                            (n_t == 1 || va_disjoint(past, cpy_n)));

                                        if (sp_dst_ok && sp_cell_ok) {
                                            agx_va_range sp_ins[7];
                                            int32_t      sp_prods[7];
                                            int n_sp_ins = 0;
                                            sp_ins[n_sp_ins] = tensor_va_range(past); sp_prods[n_sp_ins++] = tensor_prod_idx(past);
                                            sp_ins[n_sp_ins] = tensor_va_range(cur);  sp_prods[n_sp_ins++] = tensor_prod_idx(cur);
                                            sp_ins[n_sp_ins] = tensor_va_range(w);    sp_prods[n_sp_ins++] = tensor_prod_idx(w);
                                            if (dyn_t)     { sp_ins[n_sp_ins] = tensor_va_range(dyn_t);     sp_prods[n_sp_ins++] = tensor_prod_idx(dyn_t); }
                                            if (clip_sc_t) { sp_ins[n_sp_ins] = tensor_va_range(clip_sc_t); sp_prods[n_sp_ins++] = tensor_prod_idx(clip_sc_t); }
                                            if (tau_t)     { sp_ins[n_sp_ins] = tensor_va_range(tau_t);     sp_prods[n_sp_ins++] = tensor_prod_idx(tau_t); }
                                            if (ids_t)     { sp_ins[n_sp_ins] = tensor_va_range(ids_t);     sp_prods[n_sp_ins++] = tensor_prod_idx(ids_t); }

                                            agx_va_range sp_extra[5];
                                            int n_sp_extra = 0;
                                            sp_extra[n_sp_extra++] = tensor_va_range(w);
                                            if (dyn_t)     sp_extra[n_sp_extra++] = tensor_va_range(dyn_t);
                                            if (clip_sc_t) sp_extra[n_sp_extra++] = tensor_va_range(clip_sc_t);
                                            if (tau_t)     sp_extra[n_sp_extra++] = tensor_va_range(tau_t);
                                            if (ids_t)     sp_extra[n_sp_extra++] = tensor_va_range(ids_t);

                                            agx_va_range sp_outs[2];
                                            int32_t      sp_outs_ex[2];
                                            int n_sp_outs = 0;
                                            sp_outs[n_sp_outs] = tensor_va_range(tail_n); sp_outs_ex[n_sp_outs++] = st->exec_idx_map.get(tail_n);
                                            if (cpy_n) { sp_outs[n_sp_outs] = tensor_va_range(cpy_n); sp_outs_ex[n_sp_outs++] = st->exec_idx_map.get(cpy_n); }

                                            const int32_t sp_anc = find_safe_anchor(p_fset, p_fn, sp_ins, sp_prods, n_sp_ins,
                                                                                    sp_extra, n_sp_extra, sp_outs, sp_outs_ex, n_sp_outs,
                                                                                    false, st->exec_idx_map.get(sc_res));
                                            if (sp_anc >= 0) {
                                                sconv_prep_store_push spc{};
                                                spc.past_va    = agx_tensor_gpu_va(past);
                                                spc.cur_va     = agx_tensor_gpu_va(cur);
                                                spc.w_va       = agx_tensor_gpu_va(w);
                                                spc.dyn_va     = dyn_t     ? agx_tensor_gpu_va(dyn_t)     : 0ull;
                                                spc.cell_va    = cpy_n     ? agx_tensor_gpu_va(cpy_n)     : 0ull;
                                                spc.scale_va   = clip_sc_t ? agx_tensor_gpu_va(clip_sc_t) : 0ull;
                                                spc.tau_va     = tau_t     ? agx_tensor_gpu_va(tau_t)     : 0ull;
                                                spc.ids_va     = ids_t     ? agx_tensor_gpu_va(ids_t)     : 0ull;
                                                spc.dst_va     = agx_tensor_gpu_va(tail_n);
                                                spc.past_nb1   = (uint32_t) (past->nb[1] / 4u);
                                                spc.past_nb2   = (uint32_t) (past->nb[2] / 4u);
                                                spc.w_nb1      = (uint32_t) (w->nb[1] / 4u);
                                                spc.cell_nb1   = cpy_n ? (uint32_t) (cpy_n->nb[1] / 4u) : 0u;
                                                spc.n_conv     = (uint32_t) n_conv;
                                                spc.head_dim   = (uint32_t) Dp;
                                                spc.n_head     = (uint32_t) Hp;
                                                spc.n_t        = (uint32_t) n_t;
                                                spc.n_s        = (uint32_t) n_s;
                                                spc.dyn_heads  = (uint32_t) (dyn_ok ? n_heads : Hp);
                                                spc.flags      = (is_l2_clip ? 1u : 2u) | fwht_flag;
                                                spc.store_mode = store_mode;
                                                spc.ids_is_i64 = (ids_t && ids_t->type == GGML_TYPE_I64) ? 1u : 0u;
                                                spc.i_nb0      = ids_t ? (uint32_t) ids_t->nb[0] : 0u;
                                                spc.d_nb1      = sr_n  ? (uint32_t) sr_n->nb[1]  : 0u;
                                                read_scale_params(sc_res, &spc.res_scale, &spc.res_bias);
                                                spc.param      = norm_param;

                                                agx_exec_item it{};
                                                it.kind          = ITEM_SCONV_PREP_STORE;
                                                it.gx            = (spc.n_head + 3u) / 4u;
                                                it.gy            = (uint32_t) Mp;
                                                it.gz            = 1u;
                                                it.nodes_covered = (uint32_t) p_fn;
                                                for (int r = 0; r < n_sp_ins; ++r)   it.add_read(sp_ins[r]);
                                                for (int ow = 0; ow < n_sp_outs; ++ow) it.add_write(sp_outs[ow]);
                                                it.set_push(spc);
                                                commit_single_item(sp_anc, p_fset, p_fn, it);
                                                st->stats.fused_full_sconv_hits++;
                                                st->stats.fused_prep_store_hits++;
                                                g_full_sconv++;
                                                g_prep_store++;
                                                g_sprep++;
                                                if (is_l2_clip) { st->stats.fused_l2_clip_hits++; g_l2_clip++; }
                                                if (fwht_flag)  { st->stats.fused_fwht_hits++; }
                                                continue;
                                            }
                                        }
                                    }
                                }
                            }

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

                            const bool sconv_alias_dst =
                                !va_disjoint(past, final_dst) || !va_disjoint(dyn_t, final_dst) ||
                                (n_t > 1 ? !va_disjoint(cur, final_dst) : !va_same_or_disjoint(cur, final_dst)) ||
                                !va_same_or_disjoint(post_add_t, final_dst);
                            const bool dst_va_ok = va_disjoint(w, final_dst) && va_disjoint(cpy_n, final_dst);
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
                                int32_t      outs_ex[2];
                                int n_outs = 0;
                                outs[n_outs] = tensor_va_range(final_dst); outs_ex[n_outs++] = st->exec_idx_map.get(final_dst);
                                if (cpy_n) { outs[n_outs] = tensor_va_range(cpy_n); outs_ex[n_outs++] = st->exec_idx_map.get(cpy_n); }

                                const int32_t anc = find_safe_anchor(fset, fn, all_ins, all_prods, n_ins,
                                                                     extra_ins, n_extra, outs, outs_ex, n_outs);
                                if (anc >= 0) {
                                    bool use_sconv_alias = sconv_alias_dst;
                                    agx_va_range cur_redir_r = { 0, 0 };
                                    if (use_sconv_alias && !no_redirect && st->redirect_tmp_bo &&
                                        va_disjoint(past, final_dst) && va_disjoint(dyn_t, final_dst) &&
                                        va_same_or_disjoint(post_add_t, final_dst) && !dyn_ok &&
                                        (cur->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                        st->use_map.get(cur) == 2u &&
                                        (cur_3d == cur || (cur_3d->op == GGML_OP_RESHAPE &&
                                                           (cur_3d->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                                                           st->use_map.get(cur_3d) == 1u))) {
                                        const int32_t e_cur_p = st->exec_idx_map.get(cur);
                                        int32_t anc_cur = -1;
                                        if (e_cur_p >= 0 && e_cur_p < anc) {
                                            if (anchors[(size_t) e_cur_p].cnt == 1u &&
                                                (st->items[anchors[(size_t) e_cur_p].beg].kind == ITEM_GENERIC_NODE ||
                                                 (st->items[anchors[(size_t) e_cur_p].beg].kind == ITEM_GEMM_REDIRECT &&
                                                  st->items[anchors[(size_t) e_cur_p].beg].va0 == 0ull)) &&
                                                st->items[anchors[(size_t) e_cur_p].beg].node == cur &&
                                                can_gemm_redirect(cur)) {
                                                anc_cur = e_cur_p;
                                            } else {
                                                const uint64_t cur_va_orig = agx_tensor_gpu_va(cur);
                                                const size_t j_lo = (e_cur_p > 48) ? (size_t) (e_cur_p - 48) : 0u;
                                                for (size_t j = (size_t) e_cur_p + 1u; j-- > j_lo; ) {
                                                    if (anchors[j].cnt == 1u &&
                                                        st->items[anchors[j].beg].kind == ITEM_FUSED_MOE &&
                                                        st->items[anchors[j].beg].moe_args.moe_out_va == cur_va_orig) {
                                                        anc_cur = (int32_t) j;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        if (anc_cur >= 0 && anc_cur < anc &&
                                            try_reserve_redir_slot((size_t) anc_cur, (size_t) anc, ggml_nbytes(cur), &cur_redir_r)) {
                                            auto & prod_it = st->items[anchors[(size_t) anc_cur].beg];
                                            if (prod_it.kind == ITEM_GENERIC_NODE || prod_it.kind == ITEM_GEMM_REDIRECT) {
                                                prod_it.kind      = ITEM_GEMM_REDIRECT;
                                                prod_it.va0       = cur_redir_r.beg;
                                                prod_it.writes[0] = cur_redir_r;
                                                prod_it.n_writes  = 1u;
                                            } else {
                                                prod_it.moe_args.moe_out_va = cur_redir_r.beg;
                                                prod_it.writes[0]           = cur_redir_r;
                                                prod_it.n_writes            = 1u;
                                                prod_it.uses_alias_scratch  = false;
                                                prod_it.alias_flags         = 0u;
                                                prod_it.alias_floats0       = 0u;
                                                prod_it.alias_tmp_need_B    = 0u;
                                            }
                                            all_ins[1]      = cur_redir_r;
                                            use_sconv_alias = false;
                                            g_redir++;
                                        }
                                    }

                                    if (!use_sconv_alias || !alias_scratch_locked[(size_t) anc]) {
                                        full_shortconv_push pc{};
                                        pc.past_va     = agx_tensor_gpu_va(past);
                                        pc.cur_va      = (cur_redir_r.beg != 0ull) ? cur_redir_r.beg : agx_tensor_gpu_va(cur);
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
                                        if (use_sconv_alias) {
                                            const uint32_t out_floats = pc.n_ch * pc.n_t * pc.n_s;
                                            it.uses_alias_scratch = true;
                                            it.alias_flags        = 1u;
                                            it.alias_floats0      = out_floats;
                                            it.alias_tmp_need_B   = agx_align_up((size_t) out_floats * sizeof(float), AGX_PAGE_SIZE);
                                        }
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
        // Fusion 12 (Wave 7 Upgraded): Direct Relative-Position Bias Gather
        //   PAD(rel_bias, 0,1,0,0) -> MUL_MAT(Rcur) -> SUB(sinks) -> CONT(PERM 1,0,2,3) ->
        //   GET_ROWS(rel_pos) -> CONT(PERM 2,0,1,3) [-> ADD([CAST(mask,F32)]) -> CAST(F16)]
        //   Supports `alias_tmp_bo` trampoline (both inline and pre-saved at e_mm/e_sub) so
        //   48/48 layers fuse even at `-ub 2048` where 192-384 MiB output reuses rcur/sinks!
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
                            ((n_kv + 127) / 128) <= 65535 && n_head <= 65535 && n_tokens <= 65535 &&
                            (uint64_t) n_tokens * (uint64_t) n_head * (uint64_t) n_kv <= 0xFFFFFFFFull) {

                            // Optionally absorb trailing `[CAST(mask_f16 -> F32)] -> ADD(c2, mask) -> [CAST(F16)]`
                            // (eliminates 73.6 GB DRAM + 96.6B divides on -ub 2048 & enables exact -inf early-exit!)
                            struct ggml_tensor *       final_dst    = c2;
                            struct ggml_tensor *       add_m        = nullptr;
                            struct ggml_tensor *       cast_m       = nullptr;
                            struct ggml_tensor *       cast_out     = nullptr;
                            const struct ggml_tensor * mask_t       = nullptr;
                            bool                       mask_is_f16  = false;
                            bool                       dst_is_f16   = false;
                            int32_t                    e_abs_cast_m = -1;

                            if (struct ggml_tensor * u_add = next_single_use_compute(c2, false)) {
                                if (u_add->op == GGML_OP_ADD && u_add->type == GGML_TYPE_F32 &&
                                    ggml_is_contiguous(u_add) && ggml_are_same_shape(u_add, c2) &&
                                    u_add->src[0] && u_add->src[1]) {
                                    struct ggml_tensor * other = (u_add->src[0] == c2) ? u_add->src[1]
                                                               : ((u_add->src[1] == c2) ? u_add->src[0] : nullptr);
                                    const struct ggml_tensor * cand_m = nullptr;
                                    bool cand_f16 = false;
                                    struct ggml_tensor * cand_cast_m = nullptr;
                                    int32_t cand_abs_cm = -1;
                                    if (other && other->op == GGML_OP_CPY && other->type == GGML_TYPE_F32 &&
                                        other->src[0] && other->src[0]->type == GGML_TYPE_F16 &&
                                        ggml_is_contiguous(other->src[0]) && ggml_are_same_shape(other, other->src[0]) &&
                                        st->use_map.get(other) == 1u && (other->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                                        int32_t e_cm = st->exec_idx_map.get(other);
                                        agx_va_range mr = tensor_va_range(other->src[0]);
                                        if (e_cm > (int32_t) e) {
                                            cand_m      = other->src[0];
                                            cand_f16    = true;
                                            cand_cast_m = other;
                                        } else if (e_cm >= 0 && can_absorb_prev(e_cm, e, &mr, 1)) {
                                            cand_m      = other->src[0];
                                            cand_f16    = true;
                                            cand_abs_cm = e_cm;
                                        }
                                    }
                                    if (!cand_m && other && other->type == GGML_TYPE_F32 && ggml_is_contiguous(other)) {
                                        cand_m   = other;
                                        cand_f16 = false;
                                    }
                                    if (cand_m && agx_tensor_gpu_va(cand_m) != 0 &&
                                        cand_m->ne[0] == n_kv && cand_m->ne[1] == n_t &&
                                        (cand_m->ne[2] == 1 || cand_m->ne[2] == n_head) &&
                                        (cand_m->ne[3] == 1 || cand_m->ne[3] == n_stream)) {
                                        add_m        = u_add;
                                        cast_m       = cand_cast_m;
                                        e_abs_cast_m = cand_abs_cm;
                                        mask_t       = cand_m;
                                        mask_is_f16  = cand_f16;
                                        final_dst    = u_add;
                                        if (struct ggml_tensor * u_co = next_single_use_compute(u_add, false)) {
                                            if (u_co->op == GGML_OP_CPY && u_co->src[0] == u_add &&
                                                u_co->type == GGML_TYPE_F16 && ggml_is_contiguous(u_co) &&
                                                ggml_are_same_shape(u_co, u_add) && agx_tensor_gpu_va(u_co) != 0) {
                                                cast_out   = u_co;
                                                final_dst  = u_co;
                                                dst_is_f16 = true;
                                            }
                                        }
                                    }
                                }
                            }

                            if (va_disjoint(orig_bias, final_dst) && va_disjoint(rel_pos, final_dst) &&
                                va_disjoint(mask_t, final_dst)) {
                                const int32_t e_mm  = st->exec_idx_map.get(mm);
                                const int32_t e_sub = st->exec_idx_map.get(sub);
                                int32_t fset[9];
                                int fn = 0;
                                fset[fn++] = (int32_t) e;
                                fset[fn++] = e_mm;
                                fset[fn++] = e_sub;
                                fset[fn++] = st->exec_idx_map.get(c1);
                                fset[fn++] = st->exec_idx_map.get(gr);
                                fset[fn++] = st->exec_idx_map.get(c2);
                                if (cast_m)   fset[fn++] = st->exec_idx_map.get(cast_m);
                                if (add_m)    fset[fn++] = st->exec_idx_map.get(add_m);
                                if (cast_out) fset[fn++] = st->exec_idx_map.get(cast_out);

                                size_t e_min_r = (size_t) fset[0], e_max_r = (size_t) fset[0];
                                bool fset_valid = true;
                                for (int k = 0; k < fn; ++k) {
                                    if (fset[k] < 0 || (size_t) fset[k] >= n_exec || consumed[(size_t) fset[k]]) {
                                        fset_valid = false; break;
                                    }
                                    if ((size_t) fset[k] < e_min_r) e_min_r = (size_t) fset[k];
                                    if ((size_t) fset[k] > e_max_r) e_max_r = (size_t) fset[k];
                                }

                                if (fset_valid) {
                                    const agx_va_range rcur_r  = tensor_va_range(rcur);
                                    const agx_va_range sinks_r = tensor_va_range(sinks);
                                    const int32_t p_rcur       = tensor_prod_idx(rcur);
                                    const int32_t p_sinks      = tensor_prod_idx(sinks);

                                    // Check if any non-fset node in (prod, e_max_r) clobbers rcur or sinks
                                    bool rcur_clob  = false;
                                    bool sinks_clob = false;
                                    for (size_t j = e_min_r; j < e_max_r; ++j) {
                                        if (in_idx_set(j, fset, fn)) continue;
                                        if ((p_rcur  < 0 || (int32_t) j > p_rcur)  && va_range_overlap(raw_vas[j].w, rcur_r))  rcur_clob  = true;
                                        if ((p_sinks < 0 || (int32_t) j > p_sinks) && va_range_overlap(raw_vas[j].w, sinks_r)) sinks_clob = true;
                                    }

                                    auto presave_window_ok = [&](int32_t e_save, agx_va_range r, int32_t prod) -> bool {
                                        if (e_save <= (int32_t) e_min_r || e_save >= (int32_t) e_max_r) return false;
                                        if (prod >= e_save) return false;
                                        for (size_t j = e_min_r; j < (size_t) e_save; ++j) {
                                            if (in_idx_set(j, fset, fn)) continue;
                                            if ((prod < 0 || (int32_t) j > prod) && va_range_overlap(raw_vas[j].w, r)) return false;
                                        }
                                        for (size_t j = (size_t) e_save; j <= e_max_r; ++j) {
                                            if (alias_scratch_locked[j]) return false;
                                            if (j > (size_t) e_save && j < e_max_r && anchors[j].cnt > 0u) {
                                                for (uint32_t k = 0; k < anchors[j].cnt; ++k) {
                                                    if (st->items[anchors[j].beg + k].uses_alias_scratch) return false;
                                                }
                                            }
                                        }
                                        return true;
                                    };

                                    const bool rcur_presave  = rcur_clob  && presave_window_ok(e_mm,  rcur_r,  p_rcur);
                                    const bool sinks_presave = sinks_clob && presave_window_ok(e_sub, sinks_r, p_sinks);
                                    if ((!rcur_clob || rcur_presave) && (!sinks_clob || sinks_presave)) {
                                        agx_va_range all_ins[5];
                                        int32_t      all_prods[5];
                                        int n_all = 0;
                                        all_ins[n_all] = tensor_va_range(orig_bias); all_prods[n_all++] = tensor_prod_idx(orig_bias);
                                        if (!rcur_presave)  { all_ins[n_all] = rcur_r;  all_prods[n_all++] = p_rcur; }
                                        if (!sinks_presave) { all_ins[n_all] = sinks_r; all_prods[n_all++] = p_sinks; }
                                        all_ins[n_all] = tensor_va_range(rel_pos);   all_prods[n_all++] = tensor_prod_idx(rel_pos);
                                        if (mask_t) { all_ins[n_all] = tensor_va_range(mask_t); all_prods[n_all++] = tensor_prod_idx(mask_t); }

                                        agx_va_range extra_ins[4];
                                        int n_extra = 0;
                                        if (!rcur_presave)  extra_ins[n_extra++] = rcur_r;
                                        if (!sinks_presave) extra_ins[n_extra++] = sinks_r;
                                        extra_ins[n_extra++] = tensor_va_range(rel_pos);
                                        if (mask_t) extra_ins[n_extra++] = tensor_va_range(mask_t);

                                        agx_va_range outs[1]    = { tensor_va_range(final_dst) };
                                        int32_t      outs_ex[1] = { st->exec_idx_map.get(final_dst) };
                                        const bool need_last    = rcur_presave || sinks_presave;
                                        const int32_t anc = find_safe_anchor(fset, fn, all_ins, all_prods, n_all,
                                                                             extra_ins, n_extra, outs, outs_ex, 1, need_last);
                                        uint8_t rel_amask = 0u;
                                        if (rcur_presave)                        rel_amask |= 4u;
                                        else if (!va_disjoint(rcur, final_dst))  rel_amask |= 1u;
                                        if (sinks_presave)                       rel_amask |= 8u;
                                        else if (!va_disjoint(sinks, final_dst)) rel_amask |= 2u;

                                        if (anc >= 0 && (rel_amask == 0u || !alias_scratch_locked[(size_t) anc])) {
                                            if (e_abs_cast_m >= 0) {
                                                anchors[(size_t) e_abs_cast_m].cnt = 0u;
                                                consumed[(size_t) e_abs_cast_m]    = 1u;
                                            }
                                            const uint32_t n_rcur_f    = (uint32_t) (n_rel * n_head * n_tokens);
                                            const uint32_t n_sinks_f   = (uint32_t) (n_head * n_tokens);
                                            const uint32_t rcur_slot_B = (uint32_t) agx_align_up((size_t) n_rcur_f * sizeof(float), 256u);
                                            const uint32_t tot_slot_B  = rcur_slot_B + (uint32_t) agx_align_up((size_t) n_sinks_f * sizeof(float), 256u);

                                            rel_bias_gather_push pc{};
                                            pc.rcur_va    = agx_tensor_gpu_va(rcur);
                                            pc.bias_va    = agx_tensor_gpu_va(orig_bias);
                                            pc.sinks_va   = agx_tensor_gpu_va(sinks);
                                            pc.rel_pos_va = agx_tensor_gpu_va(rel_pos);
                                            pc.mask_va    = mask_t ? agx_tensor_gpu_va(mask_t) : 0ull;
                                            pc.dst_va     = agx_tensor_gpu_va(final_dst);
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
                                            pc.mask_ne2   = mask_t ? (uint32_t) mask_t->ne[2] : 1u;
                                            pc.mask_ne3   = mask_t ? (uint32_t) mask_t->ne[3] : 1u;
                                            const bool vec4_f16 = dst_is_f16 && (mask_t != nullptr) && ((pc.n_kv & 3u) == 0u) &&
                                                                  (pc.rpos_nb0 == 4u) && ((pc.rpos_nb1 & 15u) == 0u) &&
                                                                  ((pc.dst_va & 7ull) == 0ull) && ((pc.rel_pos_va & 15ull) == 0ull) &&
                                                                  ((pc.mask_va & (mask_is_f16 ? 7ull : 15ull)) == 0ull);
                                            uint64_t     rel_bounds_va = 0ull;
                                            agx_va_range rel_bounds_r  = { 0, 0 };
                                            if (!no_rel_skip && mask_t && n_stream == 1 && n_all < 8 &&
                                                st->use_map.get(final_dst) == 1u &&
                                                (final_dst->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                                                struct ggml_tensor * fa_u = st->first_use.get(final_dst);
                                                if (fa_u && fa_u->op == GGML_OP_FLASH_ATTN_EXT && fa_u->src[3] == final_dst &&
                                                    agx_kv_hybrid_supports_ggml_flash_attn(fa_u)) {
                                                    const struct ggml_tensor * fq = fa_u->src[0];
                                                    const struct ggml_tensor * fk = fa_u->src[1];
                                                    const struct ggml_tensor * fv = fa_u->src[2];
                                                    if (fq->ne[0] == 128 && fq->ne[1] == n_tokens && fq->ne[1] >= 64 &&
                                                        fk->ne[1] == n_kv && fk->ne[1] >= 128 && fq->ne[3] == 1 &&
                                                        fk->type != GGML_TYPE_F32 && fv->type != GGML_TYPE_F32) {
                                                        get_or_alloc_fa_bounds(mask_t, (uint32_t) n_kv, (uint32_t) n_tokens,
                                                                               16u, &rel_bounds_va, &rel_bounds_r);
                                                    }
                                                }
                                            }
                                            pc.bounds_va  = rel_bounds_va;
                                            pc.flags      = (mask_is_f16 ? 1u : 0u) | (dst_is_f16 ? 2u : 0u) | (vec4_f16 ? 4u : 0u);

                                            agx_exec_item it{};
                                            it.kind          = ITEM_REL_BIAS;
                                            it.gx            = vec4_f16 ? (((pc.n_kv >> 2u) + 127u) / 128u) : ((pc.n_kv + 127u) / 128u);
                                            it.gy            = pc.n_head;
                                            it.gz            = (uint32_t) n_tokens;
                                            it.nodes_covered = (uint32_t) fn + (e_abs_cast_m >= 0 ? 1u : 0u);
                                            if (rel_amask != 0u) {
                                                it.uses_alias_scratch = true;
                                                it.alias_flags        = rel_amask;
                                                it.alias_tmp_off_B    = rcur_slot_B;
                                                it.alias_floats0      = n_rcur_f;
                                                it.alias_floats1      = n_sinks_f;
                                                it.alias_tmp_need_B   = agx_align_up((size_t) tot_slot_B, AGX_PAGE_SIZE);
                                            }
                                            for (int r = 0; r < n_all; ++r) it.add_read(all_ins[r]);
                                            if (rel_bounds_va != 0ull) {
                                                it.add_read(rel_bounds_r);
                                                g_rel_bskip++;
                                            }
                                            it.add_write(outs[0]);
                                            it.set_push(pc);
                                            commit_single_item(anc, fset, fn, it);

                                            if (rcur_presave) {
                                                agx_exec_item sv{};
                                                sv.kind               = ITEM_REL_SAVE_TMP;
                                                sv.nodes_covered      = 0u;
                                                sv.uses_alias_scratch = true;
                                                sv.va0                = agx_tensor_gpu_va(rcur);
                                                sv.alias_tmp_off_B    = 0u;
                                                sv.alias_floats0      = n_rcur_f;
                                                sv.alias_tmp_need_B   = agx_align_up((size_t) tot_slot_B, AGX_PAGE_SIZE);
                                                sv.add_read(rcur_r);
                                                const uint32_t sv_idx = (uint32_t) st->items.size();
                                                st->items.push_back(sv);
                                                anchors[(size_t) e_mm] = { sv_idx, 1u };
                                                for (size_t j = (size_t) e_mm; j <= (size_t) anc; ++j) alias_scratch_locked[j] = 1u;
                                            }
                                            if (sinks_presave) {
                                                agx_exec_item sv{};
                                                sv.kind               = ITEM_REL_SAVE_TMP;
                                                sv.nodes_covered      = 0u;
                                                sv.uses_alias_scratch = true;
                                                sv.va0                = agx_tensor_gpu_va(sinks);
                                                sv.alias_tmp_off_B    = rcur_slot_B;
                                                sv.alias_floats0      = n_sinks_f;
                                                sv.alias_tmp_need_B   = agx_align_up((size_t) tot_slot_B, AGX_PAGE_SIZE);
                                                sv.add_read(sinks_r);
                                                const uint32_t sv_idx = (uint32_t) st->items.size();
                                                st->items.push_back(sv);
                                                anchors[(size_t) e_sub] = { sv_idx, 1u };
                                                for (size_t j = (size_t) e_sub; j <= (size_t) anc; ++j) alias_scratch_locked[j] = 1u;
                                            }

                                            if (mask_t && n_stream == 1 && mask_t->ne[2] == 1 && mask_t->ne[3] == 1 &&
                                                (mask_t->type == GGML_TYPE_F16 || mask_t->type == GGML_TYPE_F32) &&
                                                mask_t->nb[0] == ggml_type_size(mask_t->type)) {
                                                rel_mask_map.push_back({ final_dst, mask_t, rel_bounds_va, rel_bounds_r });
                                            }
                                            st->stats.fused_rel_bias_hits++;
                                            g_rel_bias++;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // --------------------------------------------------------------------
        // Fusion 14 (Wave 8 NEW): FlashAttn + [FWHT64(v_rot)] + [SIGMOID(q_gate) * cur]
        //   + Shared Wave-0 [kv_beg, kv_end) SWA/Causal Tile Bounds Pre-Scan!
        //   Eliminates 48-96 dispatches, 48-96 strong_usc barriers, & 3.6-6.0 GB F32 DRAM RW!
        // --------------------------------------------------------------------
        if (node->op == GGML_OP_FLASH_ATTN_EXT && agx_kv_hybrid_supports_ggml_flash_attn(node)) {
            const struct ggml_tensor * q         = node->src[0];
            const struct ggml_tensor * k         = node->src[1];
            const struct ggml_tensor * v         = node->src[2];
            const struct ggml_tensor * mask      = node->src[3];
            const struct ggml_tensor * sinks     = node->src[4];
            const struct ggml_tensor * rel_table = ggml_flash_attn_ext_has_banded_rel_bias(node) ? node->src[6] : nullptr;
            const struct ggml_tensor * rel_pos   = ggml_flash_attn_ext_has_banded_rel_bias(node) ? node->src[7] : nullptr;
            const int64_t D  = q->ne[0];
            const int64_t N  = q->ne[1];
            const int64_t Hq = q->ne[2];
            const int64_t B  = q->ne[3];

            // Check if PASS_MODE=3 prefill fastpath can use 1-head base mask [kv_beg, kv_end) bounds!
            agx_va_range bounds_r  = { 0, 0 };
            uint64_t     bounds_va = 0ull;
            if (!no_fa_bounds && D == 128 && N >= 32 && k->ne[1] >= 128 && B == 1 &&
                k->type != GGML_TYPE_F32 && v->type != GGML_TYPE_F32 && mask != nullptr) {
                const uint32_t fa_q_br = agx_kv_hybrid_prefill_br_for_tokens((uint32_t) N);
                for (const auto & rm : rel_mask_map) {
                    if (rm.kq_b_out == mask) {
                        if (rm.bounds_va != 0ull && fa_q_br == 16u) {
                            bounds_va = rm.bounds_va;
                            bounds_r  = rm.bounds_range;
                        } else {
                            get_or_alloc_fa_bounds(rm.base_mask, (uint32_t) k->ne[1], (uint32_t) N,
                                                   fa_q_br, &bounds_va, &bounds_r);
                        }
                        break;
                    }
                }
                if (bounds_va == 0ull && mask->ne[2] == 1 && mask->ne[3] == 1) {
                    get_or_alloc_fa_bounds(mask, (uint32_t) k->ne[1], (uint32_t) N, fa_q_br, &bounds_va, &bounds_r);
                }
            }

            int32_t epi_fset[4] = { (int32_t) e, -1, -1, -1 };
            int     epi_fn      = 1;
            struct ggml_tensor *       epi_tail  = node;
            const struct ggml_tensor * q_gate_t  = nullptr;
            int32_t                    e_abs_sig = -1;
            uint32_t                   only_epi  = 0u;
            bool                       did_fwht  = false;
            bool                       did_qgate = false;

            if (!no_fuse_fa_epi && (fusion_mask & (1u << 6)) &&
                node->ne[0] == D && node->ne[1] == Hq && node->ne[2] == N && node->ne[3] == B) {
                if (struct ggml_tensor * uf = next_single_use_compute(epi_tail, false)) {
                    if (uf->op == GGML_OP_MUL_MAT &&
                        ggml_get_op_params_i32(uf, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
                        uf->src[1] && strip_reshape(uf->src[1]) == epi_tail &&
                        uf->src[1]->ne[0] == 64 && (D == 64 || D == 128) &&
                        uf->type == GGML_TYPE_F32 && ggml_is_contiguous(uf) &&
                        ggml_nelements(uf) == D * Hq * N * B) {
                        only_epi |= AGX_ATTN_FLAG_FWHT64;
                        epi_tail  = uf;
                        epi_fset[epi_fn++] = st->exec_idx_map.get(uf);
                        did_fwht  = true;
                    }
                }
                if (struct ggml_tensor * um = next_single_use_compute(epi_tail, false)) {
                    if (um->op == GGML_OP_MUL && um->type == GGML_TYPE_F32 && ggml_is_contiguous(um) &&
                        ggml_nelements(um) == D * Hq * N * B) {
                        const struct ggml_tensor * other_arg = nullptr;
                        if (strip_reshape(um->src[0]) == epi_tail && chain_uses_exact(um->src[0], epi_tail)) {
                            other_arg = um->src[1];
                        } else if (strip_reshape(um->src[1]) == epi_tail && chain_uses_exact(um->src[1], epi_tail)) {
                            other_arg = um->src[0];
                        }
                        const struct ggml_tensor * s_cand = strip_reshape(other_arg);
                        if (s_cand && s_cand->op == GGML_OP_UNARY &&
                            ggml_get_unary_op(s_cand) == GGML_UNARY_OP_SIGMOID &&
                            chain_uses_exact(other_arg, s_cand) &&
                            s_cand->src[0] && ggml_is_contiguous(s_cand->src[0]) &&
                            ggml_nelements(s_cand->src[0]) == D * Hq * N * B &&
                            (s_cand->src[0]->type == GGML_TYPE_F32 || s_cand->src[0]->type == GGML_TYPE_F16) &&
                            (B == 1 || s_cand->src[0]->ne[3] == B ||
                             (s_cand->src[0]->ne[2] == B && s_cand->src[0]->ne[3] == 1))) {
                            const int32_t es = st->exec_idx_map.get(s_cand);
                            const agx_va_range gr = tensor_va_range(s_cand->src[0]);
                            if (es > (int32_t) e && !consumed[(size_t) es]) {
                                q_gate_t   = s_cand->src[0];
                                epi_fset[epi_fn++] = es;
                                epi_fset[epi_fn++] = st->exec_idx_map.get(um);
                                epi_tail   = um;
                                did_qgate  = true;
                                only_epi  |= (q_gate_t->type == GGML_TYPE_F16)
                                    ? AGX_ATTN_FLAG_QGATE_F16 : AGX_ATTN_FLAG_QGATE_F32;
                            } else if (es >= 0 && es < (int32_t) e && can_absorb_prev(es, e, &gr, 1)) {
                                e_abs_sig  = es;
                                q_gate_t   = s_cand->src[0];
                                epi_fset[epi_fn++] = st->exec_idx_map.get(um);
                                epi_tail   = um;
                                did_qgate  = true;
                                only_epi  |= (q_gate_t->type == GGML_TYPE_F16)
                                    ? AGX_ATTN_FLAG_QGATE_F16 : AGX_ATTN_FLAG_QGATE_F32;
                            }
                        }
                    }
                }
            }

            // Try with epilogue first; if blocked, still emit bounds-only ITEM_FLASH_ATTN so bounds_va is guaranteed active!
            bool fa_committed = false;
            for (int fpass = (only_epi != 0u ? 0 : 1); fpass < 2 && !fa_committed; ++fpass) {
                const bool with_epi = (fpass == 0);
                const uint32_t cur_flags = (with_epi ? only_epi : 0u) | (bounds_va != 0ull ? AGX_ATTN_FLAG_KV_BOUNDS : 0u);
                if (cur_flags == 0u) break;

                struct ggml_tensor * tail_n = with_epi ? epi_tail : node;
                const struct ggml_tensor * cur_qg = with_epi ? q_gate_t : nullptr;
                const int fn = with_epi ? epi_fn : 1;

                const bool fa_base_va_ok =
                    agx_tensor_gpu_va(tail_n) != 0ull &&
                    va_disjoint(tail_n, q) && va_disjoint(tail_n, k) && va_disjoint(tail_n, v) &&
                    va_disjoint(tail_n, mask) && va_disjoint(tail_n, sinks) &&
                    va_disjoint(tail_n, rel_table) && va_disjoint(tail_n, rel_pos);
                const bool fa_va_ok = fa_base_va_ok && va_same_or_disjoint(tail_n, cur_qg);

                if (fa_va_ok) {
                    agx_va_range fa_ins[10];
                    int32_t      fa_prods[10];
                    int n_fa_ins = 0;
                    fa_ins[n_fa_ins] = tensor_va_range(q); fa_prods[n_fa_ins++] = tensor_prod_idx(q);
                    fa_ins[n_fa_ins] = tensor_va_range(k); fa_prods[n_fa_ins++] = tensor_prod_idx(k);
                    fa_ins[n_fa_ins] = tensor_va_range(v); fa_prods[n_fa_ins++] = tensor_prod_idx(v);
                    if (mask)      { fa_ins[n_fa_ins] = tensor_va_range(mask);      fa_prods[n_fa_ins++] = tensor_prod_idx(mask); }
                    if (sinks)     { fa_ins[n_fa_ins] = tensor_va_range(sinks);     fa_prods[n_fa_ins++] = tensor_prod_idx(sinks); }
                    if (rel_table) { fa_ins[n_fa_ins] = tensor_va_range(rel_table); fa_prods[n_fa_ins++] = tensor_prod_idx(rel_table); }
                    if (rel_pos)   { fa_ins[n_fa_ins] = tensor_va_range(rel_pos);   fa_prods[n_fa_ins++] = tensor_prod_idx(rel_pos); }
                    if (cur_qg)    { fa_ins[n_fa_ins] = tensor_va_range(cur_qg);    fa_prods[n_fa_ins++] = tensor_prod_idx(cur_qg); }

                    agx_va_range fa_extra[1];
                    int n_fa_extra = 0;
                    if (cur_qg) fa_extra[n_fa_extra++] = tensor_va_range(cur_qg);

                    agx_va_range fa_out_r  = tensor_va_range(tail_n);
                    int32_t      fa_out_ex = st->exec_idx_map.get(tail_n);
                    const int32_t fa_anc = find_safe_anchor(epi_fset, fn, fa_ins, fa_prods, n_fa_ins,
                                                            fa_extra, n_fa_extra, &fa_out_r, &fa_out_ex, 1);
                    if (fa_anc >= 0) {
                        if (with_epi && e_abs_sig >= 0) {
                            anchors[(size_t) e_abs_sig].cnt = 0u;
                            consumed[(size_t) e_abs_sig]    = 1u;
                        }
                        const size_t elem_B    = sizeof(float);
                        const size_t slice_B   = (size_t) (D * Hq * N) * elem_B;
                        const size_t qg_elem_B = cur_qg ? ggml_type_size(cur_qg->type) : 4u;
                        const size_t qg_slice  = (size_t) (D * Hq * N) * qg_elem_B;

                        agx_exec_item it{};
                        it.kind              = ITEM_FLASH_ATTN;
                        it.node              = node;
                        it.nodes_covered     = (uint32_t) fn + ((with_epi && e_abs_sig >= 0) ? 1u : 0u);
                        it.uses_attn_scratch = agx_kv_hybrid_ggml_flash_attn_uses_scratch(node);
                        it.fa_epi.extra_flags  = cur_flags;
                        it.fa_epi.out_va       = agx_tensor_gpu_va(tail_n);
                        it.fa_epi.out_nb3      = (uint64_t) slice_B;
                        it.fa_epi.q_gate_va    = cur_qg ? agx_tensor_gpu_va(cur_qg) : 0ull;
                        it.fa_epi.q_gate_nb3   = (uint64_t) qg_slice;
                        it.fa_epi.kv_bounds_va = bounds_va;
                        for (int r = 0; r < n_fa_ins; ++r) it.add_read(fa_ins[r]);
                        if (bounds_va != 0ull) it.add_read(bounds_r);
                        it.add_write(fa_out_r);
                        commit_single_item(fa_anc, epi_fset, fn, it);
                        if (with_epi) {
                            g_fa_epi++;
                            if (did_fwht)  st->stats.fused_fwht_hits++;
                            if (did_qgate) { st->stats.fused_sig_mul_hits++; g_sig_mul++; }
                        }
                        fa_committed = true;
                    }
                }

                // Wave 9 Dual-VA Redirect Rescue (`faepi: 12 -> 48/48`):
                // Rescues missed faepi when (a) `tail_n` (`um`) reused `Qcur`/`kq_b`/`rel_table` freed VA
                // (`!fa_base_va_ok`, 36/48 visits!) via `out_redir_r` + `wo` `ITEM_GEMM_REDIRECT(va1=out_redir_r.beg)`,
                // and/or (b) `wqkv_gate` (`eqg > e`) reused freed VA via `qg_redir_r`!
                if (!fa_committed && with_epi && !no_redirect && st->redirect_tmp_bo &&
                    agx_tensor_gpu_va(tail_n) != 0ull) {
                    const int32_t eqg_cand = (did_qgate && e_abs_sig < 0 && cur_qg &&
                                              st->use_map.get(cur_qg) == 1u && can_gemm_redirect(cur_qg))
                                           ? st->exec_idx_map.get(cur_qg) : -1;
                    const bool need_qg_redir = (eqg_cand > (int32_t) e && !consumed[(size_t) eqg_cand]);

                    struct ggml_tensor * wo_cand = next_single_use_compute(tail_n, false);
                    int32_t e_wo = -1;
                    bool can_out_redir = false;
                    if (wo_cand && can_gemm_redirect(wo_cand) && wo_cand->src[1] &&
                        strip_reshape(wo_cand->src[1]) == tail_n &&
                        chain_uses_exact(wo_cand->src[1], tail_n) &&
                        ggml_is_contiguous(wo_cand->src[1]) &&
                        wo_cand->src[1]->view_offs == 0 &&
                        ggml_nbytes(tail_n) <= redir_slot_cap_B) {
                        e_wo = st->exec_idx_map.get(wo_cand);
                        if (e_wo > (int32_t) e && !consumed[(size_t) e_wo]) {
                            can_out_redir = true;
                        }
                    }

                    for (int opass = (fa_base_va_ok && need_qg_redir ? 0 : 1); opass < 2 && !fa_committed; ++opass) {
                        const bool use_out_redir = (opass == 1);
                        if (use_out_redir && !can_out_redir) break;
                        if (!use_out_redir && !need_qg_redir) continue;

                        int32_t r_fset[5];
                        int r_fn = 0;
                        for (int k = 0; k < fn; ++k) r_fset[r_fn++] = epi_fset[k];
                        if (need_qg_redir) r_fset[r_fn++] = eqg_cand;

                        agx_va_range r_ins[11];
                        int32_t      r_prods[11];
                        int n_r_ins = 0;
                        r_ins[n_r_ins] = tensor_va_range(q);               r_prods[n_r_ins++] = tensor_prod_idx(q);
                        r_ins[n_r_ins] = tensor_va_range(k);               r_prods[n_r_ins++] = tensor_prod_idx(k);
                        r_ins[n_r_ins] = tensor_va_range(v);               r_prods[n_r_ins++] = tensor_prod_idx(v);
                        if (mask)      { r_ins[n_r_ins] = tensor_va_range(mask);      r_prods[n_r_ins++] = tensor_prod_idx(mask); }
                        if (sinks)     { r_ins[n_r_ins] = tensor_va_range(sinks);     r_prods[n_r_ins++] = tensor_prod_idx(sinks); }
                        if (rel_table) { r_ins[n_r_ins] = tensor_va_range(rel_table); r_prods[n_r_ins++] = tensor_prod_idx(rel_table); }
                        if (rel_pos)   { r_ins[n_r_ins] = tensor_va_range(rel_pos);   r_prods[n_r_ins++] = tensor_prod_idx(rel_pos); }

                        agx_va_range r_extra[2];
                        int n_r_extra = 0;
                        if (need_qg_redir) {
                            r_ins[n_r_ins] = tensor_va_range(cur_qg->src[0]); r_prods[n_r_ins++] = tensor_prod_idx(cur_qg->src[0]);
                            r_ins[n_r_ins] = tensor_va_range(cur_qg->src[1]); r_prods[n_r_ins++] = tensor_prod_idx(cur_qg->src[1]);
                            r_extra[n_r_extra++] = tensor_va_range(cur_qg->src[0]);
                            r_extra[n_r_extra++] = tensor_va_range(cur_qg->src[1]);
                        } else if (cur_qg) {
                            r_ins[n_r_ins] = tensor_va_range(cur_qg); r_prods[n_r_ins++] = tensor_prod_idx(cur_qg);
                            r_extra[n_r_extra++] = tensor_va_range(cur_qg);
                        }

                        agx_va_range fa_out_r  = tensor_va_range(tail_n);
                        int32_t      fa_out_ex = st->exec_idx_map.get(tail_n);
                        const int32_t r_anc = find_safe_anchor(
                            r_fset, r_fn, r_ins, r_prods, n_r_ins,
                            r_extra, n_r_extra,
                            use_out_redir ? nullptr : &fa_out_r,
                            use_out_redir ? nullptr : &fa_out_ex,
                            use_out_redir ? 0 : 1);
                        if (r_anc < 0) continue;
                        if (use_out_redir && r_anc >= e_wo) continue;

                        agx_va_range out_redir_r = { 0, 0 };
                        agx_va_range qg_redir_r  = { 0, 0 };
                        int out_slot = -1;
                        bool slots_ok = true;
                        if (use_out_redir) {
                            slots_ok = try_reserve_redir_slot((size_t) r_anc, (size_t) e_wo,
                                                              ggml_nbytes(tail_n), &out_redir_r, &out_slot);
                        }
                        if (slots_ok && need_qg_redir) {
                            slots_ok = try_reserve_redir_slot((size_t) r_anc, (size_t) r_anc,
                                                              ggml_nbytes(cur_qg), &qg_redir_r);
                            if (!slots_ok && out_slot >= 0) {
                                unlock_redir_slot(out_slot, (size_t) r_anc, (size_t) e_wo);
                            }
                        }
                        if (!slots_ok) continue;

                        if (!need_qg_redir && e_abs_sig >= 0) {
                            anchors[(size_t) e_abs_sig].cnt = 0u;
                            consumed[(size_t) e_abs_sig]    = 1u;
                        }
                        const size_t slice_B   = (size_t) (D * Hq * N) * sizeof(float);
                        const size_t qg_elem_B = cur_qg ? ggml_type_size(cur_qg->type) : 4u;
                        const size_t qg_slice  = (size_t) (D * Hq * N) * qg_elem_B;
                        const uint32_t sbeg    = (uint32_t) st->items.size();
                        uint32_t scnt = 0u;

                        if (need_qg_redir) {
                            agx_exec_item git{};
                            git.kind          = ITEM_GEMM_REDIRECT;
                            git.node          = const_cast<struct ggml_tensor *>(cur_qg);
                            git.va0           = qg_redir_r.beg;
                            git.nodes_covered = 1u;
                            git.add_read(tensor_va_range(cur_qg->src[0]));
                            git.add_read(tensor_va_range(cur_qg->src[1]));
                            git.add_write(qg_redir_r);
                            st->items.push_back(git);
                            scnt++;
                        }

                        agx_exec_item it{};
                        it.kind              = ITEM_FLASH_ATTN;
                        it.node              = node;
                        it.nodes_covered     = (uint32_t) fn + ((!need_qg_redir && e_abs_sig >= 0) ? 1u : 0u);
                        it.uses_attn_scratch = agx_kv_hybrid_ggml_flash_attn_uses_scratch(node);
                        it.fa_epi.extra_flags  = cur_flags;
                        it.fa_epi.out_va       = use_out_redir ? out_redir_r.beg : agx_tensor_gpu_va(tail_n);
                        it.fa_epi.out_nb3      = (uint64_t) slice_B;
                        it.fa_epi.q_gate_va    = need_qg_redir ? qg_redir_r.beg : (cur_qg ? agx_tensor_gpu_va(cur_qg) : 0ull);
                        it.fa_epi.q_gate_nb3   = (uint64_t) qg_slice;
                        it.fa_epi.kv_bounds_va = bounds_va;
                        it.add_read(tensor_va_range(q));
                        it.add_read(tensor_va_range(k));
                        it.add_read(tensor_va_range(v));
                        if (mask)      it.add_read(tensor_va_range(mask));
                        if (sinks)     it.add_read(tensor_va_range(sinks));
                        if (rel_table) it.add_read(tensor_va_range(rel_table));
                        if (rel_pos)   it.add_read(tensor_va_range(rel_pos));
                        if (need_qg_redir)  it.add_read(qg_redir_r);
                        else if (cur_qg)    it.add_read(tensor_va_range(cur_qg));
                        if (bounds_va != 0ull) it.add_read(bounds_r);
                        it.add_write(use_out_redir ? out_redir_r : fa_out_r);
                        st->items.push_back(it);
                        scnt++;

                        anchors[(size_t) r_anc] = { sbeg, scnt };
                        for (int k = 0; k < r_fn; ++k) consumed[(size_t) r_fset[k]] = 1u;

                        if (use_out_redir) {
                            agx_exec_item wo_it{};
                            wo_it.kind          = ITEM_GEMM_REDIRECT;
                            wo_it.node          = wo_cand;
                            wo_it.va0           = 0ull;
                            wo_it.va1           = out_redir_r.beg;
                            wo_it.nodes_covered = 1u;
                            wo_it.add_read(tensor_va_range(wo_cand->src[0]));
                            wo_it.add_read(out_redir_r);
                            wo_it.add_write(tensor_va_range(wo_cand));
                            const uint32_t wo_idx = (uint32_t) st->items.size();
                            st->items.push_back(wo_it);
                            anchors[(size_t) e_wo]  = { wo_idx, 1u };
                            consumed[(size_t) e_wo] = 1u;
                            g_redir++;
                        }
                        if (need_qg_redir) g_redir++;
                        g_fa_epi++;
                        if (did_fwht)  st->stats.fused_fwht_hits++;
                        if (did_qgate) { st->stats.fused_sig_mul_hits++; g_sig_mul++; }
                        fa_committed = true;
                    }
                }
            }
            if (fa_committed) continue;
        }

        // Default: emit plain single-node generic item at anchor `e`
        {
            agx_exec_item it{};
            it.kind              = ITEM_GENERIC_NODE;
            it.node              = node;
            it.nodes_covered     = 1u;
            it.uses_moe_scratch  = (node->op == GGML_OP_MUL_MAT_ID);
            it.uses_attn_scratch = (node->op == GGML_OP_FLASH_ATTN_EXT) &&
                                   agx_kv_hybrid_ggml_flash_attn_uses_scratch(node);
            for (uint8_t r = 0; r < raw_vas[e].nr; ++r) it.add_read(raw_vas[e].r[r]);
            it.add_write(raw_vas[e].w);
            const uint32_t idx = (uint32_t) st->items.size();
            st->items.push_back(it);
            anchors[e] = { idx, 1u };
        }
    }

    // Compact items in original anchor program order (with Wave-0 mask bounds pre-items first)
    st->ordered_items.clear();
    st->ordered_items.reserve(pre_items.size() + st->items.size());
    for (const auto & pit : pre_items) {
        st->ordered_items.push_back(pit);
    }
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
        int32_t last_moe_lvl   = -1;
        int32_t last_attn_lvl  = -1;
        int32_t last_alias_lvl = -1;
        uint32_t max_lvl = 0u;

        for (size_t u = 0; u < n_items; ++u) {
            const auto & it = st->ordered_items[u];
            uint32_t lv = 0u;
            if (it.uses_moe_scratch   && last_moe_lvl   >= 0) lv = std::max(lv, (uint32_t) last_moe_lvl   + 1u);
            if (it.uses_attn_scratch  && last_attn_lvl  >= 0) lv = std::max(lv, (uint32_t) last_attn_lvl  + 1u);
            if (it.uses_alias_scratch && last_alias_lvl >= 0) lv = std::max(lv, (uint32_t) last_alias_lvl + 1u);

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

            if (it.uses_moe_scratch)   last_moe_lvl   = (int32_t) lv;
            if (it.uses_attn_scratch)  last_attn_lvl  = (int32_t) lv;
            if (it.uses_alias_scratch) last_alias_lvl = (int32_t) lv;

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
    // Pre-size shared alias-trampoline BO ONCE before encoding the CDM stream so
    // its GPU VA never moves mid-submission!
    size_t max_alias_tmp_B = 0u;
    for (size_t u = 0; u < n_items; ++u) {
        if (st->ordered_items[u].alias_tmp_need_B > max_alias_tmp_B) {
            max_alias_tmp_B = st->ordered_items[u].alias_tmp_need_B;
        }
    }
    if (max_alias_tmp_B > 0u && (!st->alias_tmp_bo || max_alias_tmp_B > st->alias_tmp_cap_B)) {
        if (st->alias_tmp_bo) agx_bo_free(ctx->drm, st->alias_tmp_bo);
        const uint32_t aflags = (max_alias_tmp_B <= 24u * 1024u * 1024u)
            ? (AGX_BO_SLC_PIN | AGX_BO_WRITEBACK)
            : (AGX_BO_SLC_STREAM | AGX_BO_WRITEBACK);
        st->alias_tmp_bo = agx_bo_alloc(ctx->drm, max_alias_tmp_B, aflags, "ge-alias-tmp");
        if (!st->alias_tmp_bo) { st->alias_tmp_cap_B = 0; return GGML_STATUS_FAILED; }
        st->alias_tmp_cap_B = max_alias_tmp_B;
    }

    auto emit_f32_copy = [&](uint64_t src_va, uint64_t dst_va, uint32_t n_floats, agx_barrier_mode bm) -> bool {
        cpy_cast_push cpc{};
        cpc.src_va = src_va;
        cpc.dst_va = dst_va;
        cpc.ne0    = n_floats; cpc.ne1 = 1u; cpc.ne2 = 1u; cpc.ne3 = 1u;
        cpc.d_ne0  = n_floats; cpc.d_ne1 = 1u; cpc.d_ne2 = 1u; cpc.d_ne3 = 1u;
        cpc.s_nb0  = 4u; cpc.s_nb1 = n_floats * 4u; cpc.s_nb2 = cpc.s_nb1; cpc.s_nb3 = cpc.s_nb1;
        cpc.d_nb0  = 4u; cpc.d_nb1 = n_floats * 4u; cpc.d_nb2 = cpc.d_nb1; cpc.d_nb3 = cpc.d_nb1;
        cpc.src_type = 0u;
        cpc.dst_type = 0u;
        cpc.total    = n_floats;
        const bool v4_cp = ((n_floats & 3u) == 0u) && (((src_va | dst_va) & 15ull) == 0ull);
        cpc.d_is_linear  = v4_cp ? 2u : 1u;
        return agx_cdm_dispatch(stream, st->k_cpy_cast,
                                v4_cp ? (((n_floats >> 2u) + 255u) / 256u) : ((n_floats + 255u) / 256u),
                                1u, 1u, &cpc, sizeof(cpc), bm);
    };

    std::vector<agx_va_range> dirty_vas;
    std::vector<agx_va_range> read_vas;
    dirty_vas.reserve(64);
    read_vas.reserve(128);
    bool has_dispatched     = false;
    bool used_moe_scratch   = false;
    bool used_attn_scratch  = false;
    bool used_alias_scratch = false;

    auto ensure_item_ready = [&](const agx_exec_item & it) {
        bool need_strong   = force_barrier;
        bool need_weak_war = false;
        if (!need_strong && has_dispatched) {
            if ((it.uses_moe_scratch   && used_moe_scratch) ||
                (it.uses_attn_scratch  && used_attn_scratch) ||
                (it.uses_alias_scratch && used_alias_scratch)) {
                need_strong = true;
            }
            for (uint8_t r = 0; !need_strong && r < it.n_reads; ++r) {
                if (ranges_overlap(dirty_vas, it.reads[r])) need_strong = true;
            }
            for (uint8_t w = 0; !need_strong && w < it.n_writes; ++w) {
                if (ranges_overlap(dirty_vas, it.writes[w])) {
                    need_strong = true;
                } else if (ranges_overlap(read_vas, it.writes[w])) {
                    need_weak_war = true;
                }
            }
        }
        if (need_strong && has_dispatched) {
            agx_cdm_barrier_usc(stream);
            dirty_vas.clear();
            read_vas.clear();
            used_moe_scratch   = false;
            used_attn_scratch  = false;
            used_alias_scratch = false;
        } else if (need_weak_war && has_dispatched) {
            if (no_weak_war) {
                agx_cdm_barrier_usc(stream);
                dirty_vas.clear();
                read_vas.clear();
                used_moe_scratch   = false;
                used_attn_scratch  = false;
                used_alias_scratch = false;
            } else {
                // Pure WAR hazard (zero RAW / WAW / scratch conflict): 0x60000160 WFI ordering-only
                // barrier ensures all prior readers finish before this write; dirty_vas & scratch
                // stay intact so every future RAW/WAW still receives a full STRONG_USC cache flush!
                agx_cdm_barrier_wfi(stream);
                read_vas.clear();
                g_wfi_bar++;
            }
        } else if (has_dispatched) {
            st->stats.barriers_elided++;
        }
    };

    auto record_item_dispatch = [&](const agx_exec_item & it) {
        has_dispatched = true;
        for (uint8_t r = 0; r < it.n_reads; ++r)  read_vas.push_back(it.reads[r]);
        for (uint8_t w = 0; w < it.n_writes; ++w) dirty_vas.push_back(it.writes[w]);
        if (it.uses_moe_scratch)   used_moe_scratch   = true;
        if (it.uses_attn_scratch)  used_attn_scratch  = true;
        if (it.uses_alias_scratch) used_alias_scratch = true;
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
                if (it.alias_flags != 0u) {
                    mix_attn_push mpc{};
                    std::memcpy(&mpc, it.push, sizeof(mpc));
                    if ((it.alias_flags & 1u) != 0u) {
                        const uint64_t final_out_va = mpc.out_va;
                        mpc.out_va = st->alias_tmp_bo->gpu_va;
                        ok = agx_cdm_dispatch(stream, st->k_mix_attn, it.gx, it.gy, it.gz,
                                              &mpc, sizeof(mpc), AGX_BARRIER_STRONG_USC);
                        if (ok) {
                            ok = emit_f32_copy(st->alias_tmp_bo->gpu_va, final_out_va, it.alias_floats0, AGX_BARRIER_NONE);
                        }
                    } else {
                        uint64_t orig_vas[3] = { mpc.out_va, mpc.out1_va, mpc.out2_va };
                        uint64_t tmp_vas[3]  = {};
                        uint32_t slot = 0u;
                        for (uint32_t qj = 0u; qj < 3u; ++qj) {
                            if ((it.alias_flags & (2u << qj)) != 0u) {
                                tmp_vas[qj] = st->alias_tmp_bo->gpu_va + (uint64_t) slot * it.alias_tmp_off_B;
                                slot++;
                                if      (qj == 0u) mpc.out_va  = tmp_vas[qj];
                                else if (qj == 1u) mpc.out1_va = tmp_vas[qj];
                                else               mpc.out2_va = tmp_vas[qj];
                            }
                        }
                        ok = agx_cdm_dispatch(stream, st->k_mix_attn, it.gx, it.gy, it.gz,
                                              &mpc, sizeof(mpc), AGX_BARRIER_STRONG_USC);
                        for (uint32_t qj = 0u; ok && qj < 3u; ++qj) {
                            if ((it.alias_flags & (2u << qj)) != 0u) {
                                ok = emit_f32_copy(tmp_vas[qj], orig_vas[qj], it.alias_floats0, AGX_BARRIER_NONE);
                            }
                        }
                    }
                } else {
                    ok = agx_cdm_dispatch(stream, st->k_mix_attn, it.gx, it.gy, it.gz,
                                          it.push, it.push_bytes, AGX_BARRIER_NONE);
                }
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
                if (it.alias_flags != 0u) {
                    full_shortconv_push pc{};
                    std::memcpy(&pc, it.push, sizeof(pc));
                    const uint64_t final_dst_va = pc.dst_va;
                    pc.dst_va = st->alias_tmp_bo->gpu_va;
                    ok = agx_cdm_dispatch(stream, st->k_full_shortconv, it.gx, it.gy, it.gz,
                                          &pc, sizeof(pc), AGX_BARRIER_STRONG_USC);
                    if (ok) {
                        ok = emit_f32_copy(st->alias_tmp_bo->gpu_va, final_dst_va, it.alias_floats0, AGX_BARRIER_NONE);
                    }
                } else {
                    ok = agx_cdm_dispatch(stream, st->k_full_shortconv, it.gx, it.gy, it.gz,
                                          it.push, it.push_bytes, AGX_BARRIER_NONE);
                }
                break;

            case ITEM_SCONV_PREP_STORE:
                ok = agx_cdm_dispatch(stream, st->k_sconv_prep_store, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_FA_MASK_BOUNDS:
                ok = agx_cdm_dispatch(stream, st->k_fa_mask_bounds, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_FLASH_ATTN: {
                if (!ctx->kv_hybrid_state) {
                    agx_kv_hybrid_config kcfg;
                    kcfg.n_slots = 48;
                    kcfg.max_full_tokens = 4096;
                    ctx->kv_hybrid_state = agx_kv_hybrid_create(ctx->drm, ctx->compiler, kcfg);
                    if (!ctx->kv_hybrid_state) return GGML_STATUS_FAILED;
                }
                ok = agx_kv_hybrid_dispatch_ggml_flash_attn_ext(
                    static_cast<agx_kv_hybrid_pool *>(ctx->kv_hybrid_state),
                    stream, it.node, &it.fa_epi, AGX_BARRIER_NONE);
            } break;

            case ITEM_GEMM_REDIRECT: {
                if (!agx_gemm_init(ctx)) return GGML_STATUS_FAILED;
                ok = dispatch_gemm_to_va(ctx, stream, it.node, it.va0, it.va1, AGX_BARRIER_NONE);
            } break;

            case ITEM_ROUTER_TOPK:
                ok = agx_cdm_dispatch(stream, st->k_router_topk_norm, it.gx, it.gy, it.gz,
                                      it.push, it.push_bytes, AGX_BARRIER_NONE);
                break;

            case ITEM_REL_SAVE_TMP:
                ok = emit_f32_copy(it.va0, st->alias_tmp_bo->gpu_va + it.alias_tmp_off_B,
                                   it.alias_floats0, AGX_BARRIER_NONE);
                break;

            case ITEM_REL_BIAS:
                if (it.alias_flags != 0u) {
                    rel_bias_gather_push pc{};
                    std::memcpy(&pc, it.push, sizeof(pc));
                    const uint64_t tmp_base = st->alias_tmp_bo->gpu_va;
                    ok = true;
                    bool did_copy = false;
                    if (it.alias_flags & 1u) {
                        ok = ok && emit_f32_copy(pc.rcur_va, tmp_base, it.alias_floats0, AGX_BARRIER_NONE);
                        pc.rcur_va = tmp_base;
                        did_copy = true;
                    } else if (it.alias_flags & 4u) {
                        pc.rcur_va = tmp_base;
                    }
                    if (it.alias_flags & 2u) {
                        ok = ok && emit_f32_copy(pc.sinks_va, tmp_base + it.alias_tmp_off_B, it.alias_floats1, AGX_BARRIER_NONE);
                        pc.sinks_va = tmp_base + it.alias_tmp_off_B;
                        did_copy = true;
                    } else if (it.alias_flags & 8u) {
                        pc.sinks_va = tmp_base + it.alias_tmp_off_B;
                    }
                    if (ok && did_copy) {
                        agx_cdm_barrier_usc(stream);
                    }
                    if (ok) {
                        ok = agx_cdm_dispatch(stream, st->k_rel_bias_gather, it.gx, it.gy, it.gz,
                                              &pc, sizeof(pc), AGX_BARRIER_NONE);
                    }
                } else {
                    ok = agx_cdm_dispatch(stream, st->k_rel_bias_gather, it.gx, it.gy, it.gz,
                                          it.push, it.push_bytes, AGX_BARRIER_NONE);
                }
                break;

            case ITEM_DSPARK_MARKOV_W4: {
                auto * eng = agx_graph_exec_get_inkling2(ctx);
                const auto & da = it.dspark_w4;
                ok = eng && agx_dspark_dispatch_markov_head_w4(eng, stream, da.w,
                                                               da.anchor_toks_va, da.base_logits_va,
                                                               da.hidden_va, da.f_rnn_va, da.rnn_state_va,
                                                               da.out_logits_va, da.out_h_nextn_va,
                                                               da.n_blocks, da.block_drafts, AGX_BARRIER_NONE);
            } break;

            case ITEM_FUSED_MOE: {
                if (!ctx->moe_state) {
                    ctx->moe_state = agx_moe_create(ctx->drm, ctx->compiler);
                    if (!ctx->moe_state) return GGML_STATUS_FAILED;
                }
                auto * mr = static_cast<agx_moe_runner *>(ctx->moe_state);
                agx_moe_stage_args ma = it.moe_args;
                if (it.alias_flags != 0u) {
                    const uint64_t final_out_va = ma.moe_out_va;
                    ma.moe_out_va = st->alias_tmp_bo->gpu_va;
                    ok = agx_moe_dispatch_fused_situ_down(mr, stream, ma, AGX_BARRIER_STRONG_USC);
                    if (ok) {
                        ok = emit_f32_copy(st->alias_tmp_bo->gpu_va, final_out_va, it.alias_floats0, AGX_BARRIER_NONE);
                    }
                } else {
                    ok = agx_moe_dispatch_fused_situ_down(mr, stream, ma, AGX_BARRIER_NONE);
                }
            } break;

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
                        const bool vec4_ok = ggml_are_same_shape(a, b) && ggml_are_same_shape(a, node) &&
                                             ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(node) &&
                                             (pc.total & 3u) == 0u && ((pc.a_va | pc.b_va | pc.d_va) & 15ull) == 0ull;
                        const bool bcast3d_ok = !vec4_ok && ggml_are_same_shape(a, node) &&
                                                ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(node) &&
                                                node->ne[3] == 1 && b->ne[3] == 1 &&
                                                (pc.ne0 & 3u) == 0u && pc.ne1 <= 65535u && pc.ne2 <= 65535u &&
                                                (pc.ne10 == pc.ne0 || pc.ne10 == 1u) &&
                                                (pc.ne11 == pc.ne1 || pc.ne11 == 1u) &&
                                                (pc.ne12 == pc.ne2 || pc.ne12 == 1u) &&
                                                ((pc.a_va | pc.d_va) & 15ull) == 0ull &&
                                                (pc.ne10 == 1u || (pc.b_va & 15ull) == 0ull);
                        if (vec4_ok) {
                            pc.op_kind |= 0x100u;
                            ok = agx_cdm_dispatch(stream, st->k_bin_op, ((pc.total >> 2u) + 255u) / 256u,
                                                  1, 1, &pc, sizeof(pc), AGX_BARRIER_NONE);
                        } else if (bcast3d_ok) {
                            pc.op_kind |= 0x200u;
                            ok = agx_cdm_dispatch(stream, st->k_bin_op, ((pc.ne0 >> 2u) + 255u) / 256u,
                                                  pc.ne1, pc.ne2, &pc, sizeof(pc), AGX_BARRIER_NONE);
                        } else {
                            ok = agx_cdm_dispatch(stream, st->k_bin_op, (pc.total + 255u) / 256u,
                                                  1, 1, &pc, sizeof(pc), AGX_BARRIER_NONE);
                        }
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
                        const bool contig_v4 = ggml_is_contiguous(s) && ggml_is_contiguous(node) && ((pc.total & 3u) == 0u);
                        const bool v4_cp     = contig_v4 && (s->type == node->type) &&
                                               (pc.src_type == 0u || pc.src_type == 3u) &&
                                               (((pc.src_va | pc.dst_va) & 15ull) == 0ull);
                        const bool v4_f16_to_f32 = contig_v4 && (pc.src_type == 1u && pc.dst_type == 0u) &&
                                                   ((pc.src_va & 7ull) == 0ull) && ((pc.dst_va & 15ull) == 0ull);
                        const bool v4_f32_to_f16 = contig_v4 && (pc.src_type == 0u && pc.dst_type == 1u) &&
                                                   ((pc.src_va & 15ull) == 0ull) && ((pc.dst_va & 7ull) == 0ull);
                        const bool v4_any = v4_cp || v4_f16_to_f32 || v4_f32_to_f16;
                        pc.d_is_linear = v4_cp ? 2u : (v4_f16_to_f32 ? 3u : (v4_f32_to_f16 ? 4u : (ggml_is_contiguous(node) ? 1u : 0u)));
                        ok = agx_cdm_dispatch(stream, st->k_cpy_cast,
                                              v4_any ? (((pc.total >> 2u) + 255u) / 256u) : ((pc.total + 255u) / 256u),
                                              1, 1, &pc, sizeof(pc), AGX_BARRIER_NONE);
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
                        const bool fast_cat1 = (node->op == GGML_OP_CONCAT) && (pc.p0 == 1u) && b &&
                                               (node->ne[3] == 1) && (a->ne[3] == 1) && (b->ne[3] == 1) &&
                                               ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(node) &&
                                               (pc.ne0 & 3u) == 0u && (((pc.a_va | pc.b_va | pc.d_va) & 15ull) == 0ull) &&
                                               pc.ne1 <= 65535u && pc.ne2 <= 65535u;
                        if (fast_cat1) {
                            pc.mode = 3u;
                            ok = agx_cdm_dispatch(stream, st->k_concat_pad,
                                                  ((pc.ne0 >> 2u) + 255u) / 256u, pc.ne1, pc.ne2,
                                                  &pc, sizeof(pc), AGX_BARRIER_NONE);
                        } else {
                            ok = agx_cdm_dispatch(stream, st->k_concat_pad, (pc.total + 255u) / 256u, 1, 1,
                                                  &pc, sizeof(pc), AGX_BARRIER_NONE);
                        }
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

    const char * nomk_dump = std::getenv("GGML_AGX_NOMK_DUMP");
    static bool  did_nomk_dump = false;
    if (nomk_dump && nomk_dump[0] && nomk_dump[0] != '0' && !did_nomk_dump && cgraph->n_nodes > 512) {
        // Trigger on first M=1 decode graph (skip prefill where result_output->ne[1] > 1 or token_embd ne[1] > 1)
        bool is_m1 = false;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const struct ggml_tensor * nd = cgraph->nodes[i];
            if (nd && nd->op == GGML_OP_GET_ROWS && nd->src[0] &&
                std::strcmp(nd->src[0]->name, "token_embd.weight") == 0 && nd->ne[1] == 1) {
                is_m1 = true;
                break;
            }
        }
        if (is_m1) {
            did_nomk_dump = true;
            const char * wanted[] = {
                "Qcur-0", "kqv_out-0", "attn_out-0", "ffn_inp-0",
                "ffn_moe_topk-0", "ffn_moe_weights_scaled-0", "ffn_out-0",
                "l_out-0", "l_out-1", "l_out-5", "l_out-6", "l_out-17",
                "l_out-24", "l_out-35", "l_out-36", "l_out-47", "l_out-23", "result_norm"
            };
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const struct ggml_tensor * nd = cgraph->nodes[i];
                if (!nd || !nd->data || !nd->name[0]) continue;
                for (const char * wn : wanted) {
                    if (std::strcmp(nd->name, wn) != 0) continue;
                    const size_t ne = (size_t) ggml_nelements(nd);
                    if (nd->type == GGML_TYPE_I32 && ne >= 8) {
                        const int32_t * ip = static_cast<const int32_t *>(nd->data);
                        std::fprintf(stderr, "[NOMK DUMP] %-24s [%d %d %d %d %d %d %d %d]\n",
                                     nd->name, ip[0], ip[1], ip[2], ip[3], ip[4], ip[5], ip[6], ip[7]);
                    } else if (nd->type == GGML_TYPE_F32 && ne > 0) {
                        const float * fp = static_cast<const float *>(nd->data);
                        double ss = 0.0;
                        for (size_t k = 0; k < ne; ++k) ss += (double) fp[k] * (double) fp[k];
                        std::fprintf(stderr, "[NOMK DUMP] %-24s n=%-5zu l2=%10.5f v=[%9.5f %9.5f %9.5f %9.5f]\n",
                                     nd->name, ne, std::sqrt(ss), fp[0], ne > 1 ? fp[1] : 0.0f,
                                     ne > 2 ? fp[2] : 0.0f, ne > 3 ? fp[3] : 0.0f);
                    }
                }
            }
        }
    }

    if (print_stats && n_exec >= 64u) {
        agx_cdm_stats cdm_after{};
        agx_cdm_get_stats(stream, &cdm_after);
        const uint64_t d_disp  = cdm_after.dispatches          - cdm_before.dispatches;
        const uint64_t d_usc   = cdm_after.strong_usc_barriers - cdm_before.strong_usc_barriers;
        const uint64_t d_elide = cdm_after.elided_barriers     - cdm_before.elided_barriers;
        const uint64_t d_sub   = cdm_after.submits             - cdm_before.submits;
        std::fprintf(stderr,
                     "[ggml-agx STATS] nodes=%d exec=%zu items=%zu waves=%u wave_sched=%d dispatches=%llu submits=%llu usc_bar=%llu wfi_bar=%llu elided=%llu "
                     "cpu_enc=%.2fms gpu=%.2fms fusions(prep=%llu sconv=%llu rel=%llu rtr=%llu cont=%llu rms=%llu situ=%llu ctrl=%llu mix=%llu moe=%llu wsum=%llu ssm=%llu mnorm=%llu clip=%llu sc=%llu addsc=%llu sigm=%llu sprep=%llu faepi=%llu mrms=%llu relb=%llu covlp=%llu redir=%llu cpipe=%llu)\n",
                     cgraph->n_nodes, n_exec, n_items, num_waves, no_wave_sched ? 0 : 1,
                     (unsigned long long) d_disp, (unsigned long long) d_sub,
                     (unsigned long long) d_usc, (unsigned long long) g_wfi_bar, (unsigned long long) d_elide,
                     (double) cpu_encode_ns * 1e-6, (double) gpu_ns * 1e-6,
                     (unsigned long long) g_prep_store, (unsigned long long) g_full_sconv,
                     (unsigned long long) g_rel_bias, (unsigned long long) g_router,
                     (unsigned long long) g_cont_abs,
                     (unsigned long long) g_rms_mul, (unsigned long long) g_situ,
                     (unsigned long long) g_ctrl, (unsigned long long) g_mix,
                     (unsigned long long) g_fused_moe,
                     (unsigned long long) g_moe_wsum, (unsigned long long) g_ssm_res,
                     (unsigned long long) g_moe_norm, (unsigned long long) g_l2_clip,
                     (unsigned long long) g_softcap, (unsigned long long) g_add_sc,
                     (unsigned long long) g_sig_mul,
                     (unsigned long long) g_sprep, (unsigned long long) g_fa_epi,
                     (unsigned long long) g_mix_rms, (unsigned long long) g_rel_bskip,
                     (unsigned long long) g_ctrl_ovlp, (unsigned long long) g_redir,
                     (unsigned long long) g_ctrl_pipe);
    }

    st->stats.graphs_executed++;
    return GGML_STATUS_SUCCESS;
}

} // extern "C"
