#pragma once

#include <cstddef>
#include <cstdint>

// ============================================================================
// Wave 4 Pillar 3: C++ Push/Descriptor Structs Matching:
//   - kernels/agx_inkling2_qkv_prep_store.comp
//   - kernels/agx_inkling2_stage_prep.comp (STAGE_MODE=0..5)
//   - kernels/agx_inkling2_dspark_mtp.comp (DSPARK_FUSED_MODE=0..3)
// ============================================================================

#define AGX_W4_QKV_FLAG_IS_FULL_ATTN   (1u << 0)
#define AGX_W4_QKV_FLAG_FWHT_K         (1u << 1)
#define AGX_W4_QKV_FLAG_FWHT_V         (1u << 2)
#define AGX_W4_QKV_FLAG_KV_Q8_0        (1u << 3)
#define AGX_W4_QKV_FLAG_K_IDS_I64      (1u << 4)
#define AGX_W4_QKV_FLAG_V_IDS_I64      (1u << 5)
#define AGX_W4_QKV_FLAG_BASE_MASK_F16  (1u << 6)
#define AGX_W4_QKV_FLAG_FUSE_REL_MASK  (1u << 7)
#define AGX_W4_QKV_FLAG_PACK_QGATE     (1u << 8)

struct alignas(16) agx_w4_qkv_prep_desc {
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

    uint32_t q_raw_stride_f32;
    uint32_t k_raw_stride_f32;
    uint32_t v_raw_stride_f32;
    uint32_t r_raw_stride_f32;
    uint32_t sinks_stride_f32;
    uint32_t r_nb1_f32;
    uint32_t rollback_slots;
    uint32_t rollback_stride_f32;
    uint32_t k_store_nb1_B;
    uint32_t v_store_nb1_B;
    uint32_t k_ids_nb0_B;
    uint32_t v_ids_nb0_B;
    uint32_t n_tokens;
    uint32_t n_conv;
    uint32_t n_rel;
    uint32_t n_extent;
    uint32_t n_kv;
    uint32_t n_kv_mask_stride;
    uint32_t rpos_nb1_B;
    uint32_t bmask_nb1_B;
    uint32_t bias_wtype;
    uint32_t bias_nb1_B;
    uint32_t flags;
    uint32_t r_splits; // WGs per R head for inline RelBias mask (1..4)
    float    conv_res_scale;
    float    qk_clip;
    float    rel_clip;
    float    rms_eps;
    float    log_scale_alpha;
    float    inv_log_floor;
    uint32_t pad1;
    uint32_t pad2;
};
static_assert(sizeof(agx_w4_qkv_prep_desc) == 28 * 8 + 32 * 4, "agx_w4_qkv_prep_desc std430 alignment");

#define AGX_W4_SPREP_FLAG_ADD_RESIDUAL   (1u << 0)
#define AGX_W4_SPREP_FLAG_DO_DEPTH_MIX   (1u << 1)
#define AGX_W4_SPREP_FLAG_DO_NORM        (1u << 2)
#define AGX_W4_SPREP_FLAG_DO_CONV        (1u << 3)

struct alignas(16) agx_w4_stage_prep_desc {
    uint64_t conv_src_va;
    uint64_t res_va;
    uint64_t conv_w_va;
    uint64_t r_l_va;
    uint64_t s_copy_va;
    uint64_t cell_va;

    uint64_t cur_out_va;
    uint64_t export_va;

    uint64_t block_ptrs_va;
    uint64_t query_va;
    uint64_t norm_w_va[3];
    uint64_t mix_out_va[3];
    uint64_t mix_out_export_va;

    uint32_t n_embd;
    uint32_t n_tokens;
    uint32_t n_conv;
    uint32_t conv_ch_offset;
    uint32_t r_nb1_f32;
    uint32_t rollback_slots;
    uint32_t rollback_stride_f32;
    uint32_t n_blocks;
    uint32_t n_queries;
    uint32_t flags;
    float    conv_res_scale;
    float    inv_sqrt_d;
    float    eps_mix;
    float    eps_norm;
};
static_assert(sizeof(agx_w4_stage_prep_desc) == 17 * 8 + 14 * 4, "agx_w4_stage_prep_desc std430 alignment");

struct agx_w4_desc_va_push {
    uint64_t desc_va;
};

struct agx_w4_rel_mask_push {
    uint64_t r_va;
    uint64_t rel_bias_va;
    uint64_t sinks_va;
    uint64_t rel_pos_va;
    uint64_t base_mask_va;
    uint64_t dst_va;
    uint32_t n_rel;
    uint32_t n_extent;
    uint32_t n_kv;
    uint32_t n_kv_stride;
    uint32_t n_head;
    uint32_t n_tokens;
    uint32_t sinks_stride_f32;
    uint32_t rpos_nb1_B;
    uint32_t bmask_nb1_B;
    uint32_t bias_wtype;
    uint32_t bias_nb1_B;
    uint32_t base_mask_is_f16;
};
static_assert(sizeof(agx_w4_rel_mask_push) <= 256, "agx_w4_rel_mask_push > 256B");

struct agx_w4_fwht_qgate_push {
    uint64_t attn_va;
    uint64_t qgate_va;
    uint32_t n_head;
    uint32_t n_tokens;
    uint32_t qgate_stride_f32;
    uint32_t do_fwht128;
};
static_assert(sizeof(agx_w4_fwht_qgate_push) <= 256, "agx_w4_fwht_qgate_push > 256B");

// STAGE_MODE == 4: Batched [RMSNorm(cur) || RMSNorm(msg)] -> ctrl_cat[0..2] for 40-core agx_gemm
struct agx_w4_ctrl_norm_cat_push {
    uint64_t cur_va[3];
    uint64_t msg_va;
    uint64_t dst_va[3];
    uint32_t n_embd;
    uint32_t n_tokens;
    uint32_t n_ctrls;
    uint32_t msg_base;
    float    eps_ctrl;
    uint32_t pad0;
};
static_assert(sizeof(agx_w4_ctrl_norm_cat_push) <= 256, "agx_w4_ctrl_norm_cat_push > 256B");

// STAGE_MODE == 5: Batched (cur - tanh(forget)*cur) + tanh(gain)*msg + fused RMSNorm * norm_w
struct agx_w4_ctrl_mix_norm_push {
    uint64_t cur_va[3];
    uint64_t msg_va;
    uint64_t gain_va[3];
    uint64_t forget_va[3];
    uint64_t norm_w_va[3];
    uint64_t out_va[3];
    uint32_t n_embd;
    uint32_t n_tokens;
    uint32_t n_ctrls;
    uint32_t msg_base;
    float    eps_norm;
    uint32_t pad0;
};
static_assert(sizeof(agx_w4_ctrl_mix_norm_push) <= 256, "agx_w4_ctrl_mix_norm_push > 256B");

// STAGE_MODE == 6: 40-Core Fused F32 (K=64) ctrl_gain + ctrl_forget + tanh + residual mix -> out_va
struct agx_w4_ctrl_up_mix_push {
    uint64_t cur_va[3];
    uint64_t gate_va[3];
    uint64_t gain_w_va[3];
    uint64_t forget_w_va[3];
    uint64_t out_va[3];
    uint64_t msg_va;
    uint32_t n_embd;
    uint32_t n_tokens;
    uint32_t n_ctrls;
    uint32_t msg_base;
};
static_assert(sizeof(agx_w4_ctrl_up_mix_push) <= 256, "agx_w4_ctrl_up_mix_push > 256B");

// STAGE_MODE == 7: Batched vec4 RMSNorm * norm_w for n_ctrls in [1..3]
struct agx_w4_batch_rmsnorm_push {
    uint64_t src_va[3];
    uint64_t norm_w_va[3];
    uint64_t dst_va[3];
    uint32_t n_embd;
    uint32_t n_tokens;
    uint32_t n_ctrls;
    float    eps_norm;
};
static_assert(sizeof(agx_w4_batch_rmsnorm_push) <= 256, "agx_w4_batch_rmsnorm_push > 256B");

// DSpark Fused Markov Head (agx_inkling2_dspark_mtp.comp)
#define AGX_W4_DSPARK_F_HAS_FRNN     (1u << 0)
#define AGX_W4_DSPARK_F_ZERO_STATE   (1u << 1)
#define AGX_W4_DSPARK_F_HAS_W2_BIAS  (1u << 2)
#define AGX_W4_DSPARK_F_EMIT_TOP64   (1u << 3)

struct agx_w4_dspark_pack_push {
    uint64_t prev_tok_va;
    uint64_t w1_va;
    uint64_t hidden_va;
    uint64_t f_rnn_va;
    uint64_t state_va;
    uint64_t joint_in_va;
    uint32_t n_embd;
    uint32_t rank;
    uint32_t n_blocks;
    uint32_t block_drafts;
    uint32_t draft_pos;
    uint32_t w1_wtype;
    uint32_t w1_row_bytes;
    uint32_t flags;
};
static_assert(sizeof(agx_w4_dspark_pack_push) <= 256, "agx_w4_dspark_pack_push > 256B");

struct agx_w4_dspark_gru_push {
    uint64_t joint_out_va;
    uint64_t joint_b_va;
    uint64_t state_va;
    uint64_t markov_out_va;
    uint32_t rank;
    uint32_t n_blocks;
    uint32_t zero_state;
    uint32_t pad0;
};
static_assert(sizeof(agx_w4_dspark_gru_push) <= 256, "agx_w4_dspark_gru_push > 256B");

struct agx_w4_dspark_w2_push {
    uint64_t markov_va;
    uint64_t w2_va;
    uint64_t w2_bias_va;
    uint64_t base_va;
    uint64_t out_logits_va;
    uint64_t cand_val_va;
    uint64_t cand_idx_va;
    uint32_t rank;
    uint32_t n_vocab;
    uint32_t n_blocks;
    uint32_t block_drafts;
    uint32_t draft_pos;
    uint32_t n_splits;
    uint32_t w2_wtype;
    uint32_t w2_row_bytes;
    uint32_t flags;
    float    w2_scale;
    float    softcap;
    float    inv_softcap;
};
static_assert(sizeof(agx_w4_dspark_w2_push) <= 256, "agx_w4_dspark_w2_push > 256B");

struct agx_w4_dspark_merge_push {
    uint64_t cand_val_va;
    uint64_t cand_idx_va;
    uint64_t rnn_state_va;
    uint64_t prev_tok_va;
    uint64_t out_pack_va;
    uint32_t n_embd;
    uint32_t n_blocks;
    uint32_t block_drafts;
    uint32_t draft_pos;
    uint32_t n_splits;
    uint32_t n_cand;
    uint32_t has_rnn_state;
};
static_assert(sizeof(agx_w4_dspark_merge_push) <= 256, "agx_w4_dspark_merge_push > 256B");
