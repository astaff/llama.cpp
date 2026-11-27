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

// ============================================================================
// Wave 5 Pillar 3: Persistent 40-WG SENSE_BARRIER B=1 Megakernel Descriptors
// Matches kernels/agx_inkling2_persist_b1.comp (PERSIST_MODE=0,1,2)
// Formally verified in formal/AgxPersistentW5.tla & Wave5Roofline.lean (W5.1/W5.4)
// ============================================================================

#define AGX_W5_PFLAG_IS_SWA            (1u << 0)
#define AGX_W5_PFLAG_SAME_QKV_INP      (1u << 1)
#define AGX_W5_PFLAG_NEED_QKV_MIX      (1u << 2)
#define AGX_W5_PFLAG_HAS_K_ROT         (1u << 3)
#define AGX_W5_PFLAG_HAS_V_ROT         (1u << 4)
#define AGX_W5_PFLAG_K_IDS_I64         (1u << 5)
#define AGX_W5_PFLAG_V_IDS_I64         (1u << 6)
#define AGX_W5_PFLAG_BMASK_F16         (1u << 7)
#define AGX_W5_PFLAG_POST_ATTN_ADD_RES (1u << 8)
#define AGX_W5_PFLAG_POST_ATTN_DO_NORM (1u << 9)
#define AGX_W5_PFLAG_POST_FFN_DO_MIX   (1u << 10)
#define AGX_W5_PFLAG_POST_FFN_DO_NORM  (1u << 11)
#define AGX_W5_PFLAG_NEXT_IS_SWA       (1u << 12)

struct alignas(16) agx_w5_pass_desc {
    // 1. SLC Arena / Activation VAs (30 x u64)
    uint64_t x_q_va;
    uint64_t x_k_va;
    uint64_t x_v_va;
    uint64_t qkv_in_va[3];
    uint64_t qkv_buf_va;
    uint64_t k_raw_va;
    uint64_t v_raw_va;
    uint64_t q_out_va;
    uint64_t attn_out_va;
    uint64_t wo_out_va;
    uint64_t ffn_inp_va;
    uint64_t k_dyn_va;
    uint64_t v_dyn_va;
    uint64_t router_logits_va;
    uint64_t router_topk_va;
    uint64_t router_topk_w_va;
    uint64_t moe_inter_va;
    uint64_t moe_out_va;
    uint64_t h_msg_va;
    uint64_t ctrl_cat_va[3];
    uint64_t ctrl_gate_va[3];
    uint64_t ctrl_gain_va[3];

    // 2. Loop Controller Weights & Mix Inputs (22 x u64 -> 52)
    uint64_t msg_state_va[2];
    uint64_t msg_query_va;
    uint64_t qkv_mix_cur_va;
    uint64_t qkv_mix_bptrs_va;
    uint64_t qkv_mix_query_va;
    uint64_t ctrl_down_w_va[4];
    uint64_t ctrl_gain_w_va[4];
    uint64_t ctrl_forget_w_va[4];
    uint64_t attn_norm_w_va[3];
    uint64_t ffn_norm_w_va;

    // 3. Attention Weights, KV Cache & Mask (29 x u64 -> 81)
    uint64_t w_attn_in_va;
    uint64_t wk_byte_off;
    uint64_t wv_byte_off;
    uint64_t sink_proj_va;   // non-zero when attn_sink_proj is not coalesced into w_attn_in (e.g. BF16)
    uint64_t sinks_va;       // output/read VA for sinks[0..23]
    uint64_t k_conv_proj_va;
    uint64_t v_conv_proj_va;
    uint64_t q_clip_va;
    uint64_t r_clip_va;
    uint64_t k_conv_w_va;
    uint64_t v_conv_w_va;
    uint64_t rel_bias_va;
    uint64_t inp_pos_va;
    uint64_t r_l_va;
    uint64_t s_copy_va;
    uint64_t cell_va[4];
    uint64_t k_va;
    uint64_t v_va;
    uint64_t k_store_va;
    uint64_t v_store_va;
    uint64_t k_idxs_va;
    uint64_t v_idxs_va;
    uint64_t rel_pos_va;
    uint64_t base_mask_va;
    uint64_t kq_mask_va;
    uint64_t wo_va;

    // 4. Post-Attn Stage Prep (5 x u64 -> 84)
    uint64_t attn_post_conv_w_va;
    uint64_t attn_res_va;
    uint64_t attn_bptrs_va;
    uint64_t ffn_res_query_va;
    uint64_t post_attn_mix_out_va;

    // 5. Router & MoE Weights (8 x u64 -> 92)
    uint64_t router_w_va;
    uint64_t exp_probs_b_va;
    uint64_t gu_exps_va;
    uint64_t dn_exps_va;
    uint64_t gu_slice_B;
    uint64_t dn_slice_B;
    uint64_t gu_scale_va;
    uint64_t dn_scale_va;

    // 6. Post-FFN Stage Prep (12 x u64 -> 104)
    uint64_t ffn_post_conv_w_va;
    uint64_t next_cur_va;
    uint64_t export_l_out_va;
    uint64_t ffn_bptrs_va;
    uint64_t next_attn_query_va;
    uint64_t next_norm_w_va[3];
    uint64_t next_mix_out_va[3];
    uint64_t next_mix_export_va;

    // 7. Scalar 32-bit Parameters (48 x u32/f32 = 192 B -> total 1024 B = 8 x 128B lines)
    uint32_t flags;
    uint32_t total_coal_rows;
    uint32_t q_prefix_rows;
    uint32_t wk_rows;
    uint32_t wv_rows;
    uint32_t q_row_off;
    uint32_t qgate_row_off;
    uint32_t r_row_off;
    uint32_t sink_proj_wtype;
    uint32_t qkv_mix_n_blocks;
    uint32_t k_proj_wtype;
    uint32_t k_proj_row_B;
    uint32_t v_proj_wtype;
    uint32_t v_proj_row_B;
    uint32_t r_nb1_f32;
    uint32_t k_nb1;
    uint32_t k_nb2;
    uint32_t v_nb1;
    uint32_t v_nb2;
    uint32_t k_store_nb1;
    uint32_t v_store_nb1;
    uint32_t k_ids_nb0;
    uint32_t v_ids_nb0;
    uint32_t n_rel;
    uint32_t n_extent;
    uint32_t n_kv;
    uint32_t kq_mask_stride;
    uint32_t bias_wtype;
    uint32_t bias_nb1_B;
    uint32_t post_attn_n_blocks;
    uint32_t post_ffn_n_blocks;
    uint32_t post_ffn_n_queries;
    uint32_t router_wtype;
    uint32_t router_row_B;
    uint32_t n_expert;
    uint32_t n_ff_exp;
    float    kv_conv_res_scale;
    float    post_conv_res_scale;
    float    qk_clip;
    float    rel_clip;
    float    rms_eps;
    float    log_scale_alpha;
    float    inv_log_floor;
    float    router_norm_eps;
    float    router_w_scale;
    float    situ_beta;
    float    situ_linear_beta;
    uint32_t sink_proj_row_B;
    uint32_t reserved[28]; // pad to 1152 B = exact 9 x 128B cache lines
};
static_assert(sizeof(agx_w5_pass_desc) == 1152, "agx_w5_pass_desc must be 1152B (9 x 128B cache lines)");

struct agx_w5_persist_push {
    uint64_t descs_va;
    uint64_t barrier_va;
    uint32_t n_passes;
    uint32_t n_wg;
};
static_assert(sizeof(agx_w5_persist_push) == 24, "agx_w5_persist_push must be 24B");

struct agx_w5_attn_in_push {
    uint64_t w_attn_in_va;
    uint64_t x_q_va;
    uint64_t x_k_va;
    uint64_t x_v_va;
    uint64_t qkv_buf_va;
    uint64_t k_raw_va;
    uint64_t v_raw_va;
    uint64_t sink_proj_va;
    uint64_t sinks_va;
    uint64_t k_conv_proj_va;
    uint64_t v_conv_proj_va;
    uint64_t k_dyn_va;
    uint64_t v_dyn_va;
    uint32_t q_prefix_rows;
    uint32_t wk_rows;
    uint32_t wv_rows;
    uint32_t total_coal_rows;
    uint32_t sink_wtype;
    uint32_t sink_row_B;
    uint32_t dyn_k_wtype;
    uint32_t dyn_k_row_B;
    uint32_t dyn_v_wtype;
    uint32_t dyn_v_row_B;
    uint32_t pad0;
};
static_assert(sizeof(agx_w5_attn_in_push) == 152, "agx_w5_attn_in_push must be 152B (<= 256B)");
