#include "agx-inkling2.h"
#include "agx-gemm.h"
#include "agx-kv-hybrid.h"
#include "agx-moe.h"

#include <algorithm>
#include <cmath>
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
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer BufF32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer ConstBufF32 { float v[]; };

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t w_va;
    uint64_t dst_va;
    uint     n_embd;
    uint     n_rows;
    float    eps;
} pc;

shared float s_red[8];

void main() {
    const uint row = gl_WorkGroupID.x;
    if (row >= pc.n_rows) return;
    const uint tid     = gl_LocalInvocationID.x;
    const uint sg_id   = tid >> 5u;
    const uint lane_id = tid & 31u;
    const uint D       = pc.n_embd;

    const uint64_t s_va = pc.src_va + uint64_t(row) * uint64_t(D) * 4ul;
    const uint64_t d_va = pc.dst_va + uint64_t(row) * uint64_t(D) * 4ul;

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
            BufF32(d_va).v[i] = (ConstBufF32(s_va).v[i] * inv_rms) * ConstBufF32(pc.w_va).v[i];
        }
    } else {
        for (uint i = tid; i < D; i += 256u) {
            BufF32(d_va).v[i] = ConstBufF32(s_va).v[i] * inv_rms;
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
};

struct copy_f32_push {
    uint64_t src_va;
    uint64_t dst_va;
    uint32_t n_elems;
    uint32_t pad0;
};

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
                              agx_barrier_mode           post_barrier) {
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
            rmsnorm_f32_push np{ srcs[j], agx_tensor_gpu_va(norms[j]), eng->va_qkv_norm[j], D, n_tokens, hp.f_norm_rms_eps };
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
        rmsnorm_f32_push np{ cur_va, agx_tensor_gpu_va(layer.attn_norm), eng->va_qkv_norm[0], D, n_tokens, hp.f_norm_rms_eps };
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
