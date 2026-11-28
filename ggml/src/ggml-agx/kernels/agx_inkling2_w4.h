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
#define AGX_W4_DSPARK_F_BASE_ONLY    (1u << 4)

struct agx_w7_dspark_topk_entry {
    uint64_t ids_va;
    uint64_t vals_va;
    uint64_t k_u64;
};
static_assert(sizeof(agx_w7_dspark_topk_entry) == 24, "agx_w7_dspark_topk_entry must be 24B");

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
    uint32_t base_stride_f32; // 0 -> n_vocab (non-zero reads padded LM head directly!)
    uint32_t pad0;
};
static_assert(sizeof(agx_w4_dspark_w2_push) <= 256, "agx_w4_dspark_w2_push > 256B");

struct agx_w4_dspark_merge_push {
    uint64_t cand_val_va;
    uint64_t cand_idx_va;
    uint64_t rnn_state_va;
    uint64_t prev_tok_va;
    uint64_t out_pack_va;
    // Optional inline next-step JOINT_PACK (active when next_w1_va != 0)
    uint64_t next_w1_va;
    uint64_t next_hidden_va;
    uint64_t next_f_rnn_va;
    uint64_t next_state_va;
    uint64_t next_joint_in_va;
    uint64_t out_topk_table_va; // optional [n_tokens] table of agx_w7_dspark_topk_entry
    uint32_t n_embd;
    uint32_t n_blocks;
    uint32_t block_drafts;
    uint32_t draft_pos;
    uint32_t n_splits;
    uint32_t n_cand;
    uint32_t has_rnn_state;
    uint32_t next_rank;
    uint32_t next_w1_wtype;
    uint32_t next_w1_row_bytes;
    uint32_t next_flags;
    uint32_t pad0;
};
static_assert(sizeof(agx_w4_dspark_merge_push) <= 256, "agx_w4_dspark_merge_push > 256B");

// Wave 7 DSpark Fused Accelerator push structs (agx_inkling2_dspark_mtp.comp DSPARK_FUSED_MODE=4..7)
struct agx_w7_dspark_joint_gru_push {
    uint64_t joint_in_va;
    uint64_t joint_w_va;
    uint64_t joint_b_va;
    uint64_t state_va;
    uint64_t markov_out_va;
    uint32_t n_joint;
    uint32_t rank;
    uint32_t n_blocks;
    uint32_t w_wtype;
    uint32_t w_row_bytes;
    uint32_t zero_state;
};
static_assert(sizeof(agx_w7_dspark_joint_gru_push) <= 256, "agx_w7_dspark_joint_gru_push > 256B");

struct agx_w7_dspark_inp_block_push {
    uint64_t tok_embd_va;
    uint64_t tok_ids_va;
    uint64_t tok_norm_va;
    uint64_t mask_embd_va;
    uint64_t attn_norm0_va;
    uint64_t out_cur_va;
    uint64_t out_norm_va;
    uint32_t n_embd;
    uint32_t n_tokens;
    uint32_t block_drafts;
    uint32_t embd_wtype;
    uint32_t embd_row_bytes;
    uint32_t mask_wtype;
    uint32_t mask_row_bytes;
    float    embd_scale;
    float    rms_eps;
    uint32_t pad0;
    uint64_t out_norm_f16_va; // Wave 9: optional FP16 mirror of out_norm_va for direct-L1 Q8 GEMV
};
static_assert(sizeof(agx_w7_dspark_inp_block_push) <= 256, "agx_w7_dspark_inp_block_push > 256B");

#define AGX_W7_DSPARK_QKV_SHARED_KV   (1u << 0)
#define AGX_W7_DSPARK_QKV_HAS_ROPE    (1u << 1)
#define AGX_W7_DSPARK_QKV_ROPE_NEOX   (1u << 2)
#define AGX_W7_DSPARK_QKV_FWHT_K      (1u << 3)
#define AGX_W7_DSPARK_QKV_FWHT_V      (1u << 4)
#define AGX_W7_DSPARK_QKV_KV_Q8_0     (1u << 5)
#define AGX_W7_DSPARK_QKV_K_IDS_I64   (1u << 6)
#define AGX_W7_DSPARK_QKV_V_IDS_I64   (1u << 7)

struct agx_w7_dspark_qkv_rope_push {
    uint64_t q_raw_va;
    uint64_t k_raw_va;
    uint64_t v_raw_va;
    uint64_t q_norm_va;
    uint64_t k_norm_va;
    uint64_t inp_pos_va;
    uint64_t rope_freq_va;
    uint64_t q_out_va;
    uint64_t k_store_va;
    uint64_t v_store_va;
    uint64_t k_idxs_va;
    uint64_t v_idxs_va;
    uint32_t q_raw_stride_f32;
    uint32_t k_raw_stride_f32;
    uint32_t v_raw_stride_f32;
    uint32_t n_head_q;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_tokens;
    uint32_t rope_n_dims;
    uint32_t k_store_nb1_B;
    uint32_t v_store_nb1_B;
    uint32_t k_ids_nb0_B;
    uint32_t v_ids_nb0_B;
    uint32_t flags;
    float    rms_eps;
    float    theta_scale;
    float    freq_scale;
    float    attn_factor;
    uint32_t pad0;
};
static_assert(sizeof(agx_w7_dspark_qkv_rope_push) <= 256, "agx_w7_dspark_qkv_rope_push > 256B");

// DSPARK_FUSED_MODE == 7: op=0 RES_NORM, op=1 GLU, op=2 MARKOV_INIT
struct agx_w7_dspark_post_op_push {
    uint64_t va0; // op0: cur_in   | op1: gate_in  | op2: anchor_toks
    uint64_t va1; // op0: res_in   | op1: up_in    | op2: base_logits
    uint64_t va2; // op0: post_nw  | op1: out_va   | op2: out_logits
    uint64_t va3; // op0: out_sc   | op1: 0        | op2: out_h_nextn
    uint64_t va4; // op0: next_nw  | op1: 0        | op2: prev_tok_va
    uint64_t va5; // op0: out_cur  | op1: 0        | op2: w1_va
    uint64_t va6; // op0: out_norm | op1: 0        | op2: hidden_va
    uint64_t va7; // op0: export_n | op1: 0        | op2: f_rnn_va
    uint64_t va8; // op0: 0        | op1: 0        | op2: joint_in_va
    uint32_t op;  // 0=RES_NORM, 1=GLU, 2=MARKOV_INIT
    uint32_t p0;  // op0: n_embd   | op1: n_ff     | op2: n_vocab
    uint32_t p1;  // op0: n_tokens | op1: n_tokens | op2: base_stride_f32
    uint32_t p2;  // op0: flags    | op1: g_stride | op2: n_blocks
    uint32_t p3;  // op0: 0        | op1: u_stride | op2: block_drafts
    uint32_t p4;  // op0: 0        | op1: glu_op   | op2: n_embd
    uint32_t p5;  // op0: 0        | op1: 0        | op2: rank
    uint32_t p6;  // op0: 0        | op1: 0        | op2: w1_wtype
    uint32_t p7;  // op0: 0        | op1: 0        | op2: w1_row_bytes
    uint32_t p8;  // op0: 0        | op1: 0        | op2: flags (sample_from_anchor, has_frnn, i_beg)
    float    f0;  // op0: rms_eps  | op1: alpha    | op2: 0
    float    f1;  // op0: 0        | op1: limit    | op2: 0
};
static_assert(sizeof(agx_w7_dspark_post_op_push) <= 256, "agx_w7_dspark_post_op_push > 256B");

// DSPARK_FUSED_MODE == 8: Fused Q8_0_8BLK M=1..8 Gate + Up + GLU (1x sh_x read, 0 va_gu_out traffic!)
struct agx_w7_dspark_gate_up_glu_push {
    uint64_t wg_va;
    uint64_t wu_va;
    uint64_t x_va;
    uint64_t y_va;
    uint32_t M;       // 1..8
    uint32_t N;       // n_ff (e.g. 9216)
    uint32_t K;       // n_embd (e.g. 2304, multiple of 4)
    uint32_t glu_op;  // 0=REGLU, 1=GEGLU, 2=SWIGLU, 3=SWIGLU_OAI, 5=GEGLU_QUICK
    float    glu_alpha;
    float    glu_limit;
    uint64_t y_f16_va; // Wave 9: optional FP16 mirror of y_va for direct-L1 ffn_down Q8 GEMV
};
static_assert(sizeof(agx_w7_dspark_gate_up_glu_push) <= 256, "agx_w7_dspark_gate_up_glu_push > 256B");

// Wave 9: Zero-SRAM, zero-barrier 128T (32 SGs/core) M=1..8 Q8_0_8BLK GEMV (DSPARK_FUSED_MODE=12)
struct agx_w9_q8_gemv_m_push {
    uint64_t w_va;
    uint64_t x_va;      // F16 [M, x_stride] when X_IS_F16=1, else F32 [M, x_stride]
    uint64_t y_va;      // F32 [M, y_stride]
    uint32_t M;         // 1..8
    uint32_t N;
    uint32_t K;         // multiple of 256 (or 16)
    uint32_t x_stride;  // elements
    uint32_t y_stride;  // elements
    uint32_t pad0;
};
static_assert(sizeof(agx_w9_q8_gemv_m_push) == 48, "agx_w9_q8_gemv_m_push must be 48B");

// Wave 8 DSpark MTP Silicon-Ceiling Accelerators (agx_inkling2_dspark_mtp.comp DSPARK_FUSED_MODE=9..11)
#define AGX_W8_WTYPE_Q4_32          5u
#define AGX_W8_Q4_32_R256_ROW_BYTES 144u // 8x FP16 scales (16B) + 8x uvec4 nibbles (128B) -> 28.81 MB < 48 MiB SLC!

// DSPARK_FUSED_MODE == 9: One-shot GPU weight binder (op=0: w2 -> Q4_32 SLC_PIN, op=1: Q8_0_8BLK row coalescer)
struct agx_w8_dspark_prep_push {
    uint64_t src0_va;
    uint64_t src1_va;
    uint64_t src2_va;
    uint64_t dst_va;
    uint32_t op;          // 0 = quantize R=256 w2 -> Q4_32 (144B/row), 1 = coalesce [src0; src1; src2] uvec4 rows
    uint32_t n0;          // op0: n_vocab      | op1: rows0
    uint32_t n1;          // op0: src_wtype    | op1: rows1
    uint32_t n2;          // op0: src_row_B    | op1: rows2
    uint32_t row_uvec4s;  // op1: 16B uvec4 count per row ((K+255)/256 * 17)
    uint32_t pad0;
};
static_assert(sizeof(agx_w8_dspark_prep_push) <= 256, "agx_w8_dspark_prep_push > 256B");

// DSPARK_FUSED_MODE == 10: 1-Dispatch Fused Target-Feature Gather + All-[f_rnn, cur_0..2] ctx_mix + RMSNorm*w
struct agx_w8_dspark_inject_mix_push {
    uint64_t feat_src_va;   // layer_inp_src (when embd_rows_va != 0) or inp_embd
    uint64_t embd_rows_va;  // I32 [n_tokens] or 0
    uint64_t mix_w_va[4];   // [0]=f_rnn_mix (optional), [1..3]=layer[0..2].dflash_ctx_mix
    uint64_t norm_w_va[4];  // [0]=f_rnn_norm (optional), [1..3]=layer[0..2].dflash_ctx_norm
    uint64_t out_va[4];     // [0]=final_h_nextn_va (optional), [1..3]=cur[0..2]
    uint32_t n_embd;
    uint32_t n_target;
    uint32_t n_tokens;
    uint32_t n_slots;       // <= 4
    uint32_t feat_row_f32;  // stride between rows in feat_src_va (in floats)
    uint32_t mix_wtypes;    // 2 bits per slot (0=F32, 1=F16, 2=BF16)
    float    rms_eps;
    uint32_t pad0;
};
static_assert(sizeof(agx_w8_dspark_inject_mix_push) <= 256, "agx_w8_dspark_inject_mix_push > 256B");

// DSPARK_FUSED_MODE == 11: 1-Dispatch Batched Multi-Layer Inject k_ctx_norm + RoPE + [FWHT128] + K/V SET_ROWS
struct agx_w8_dspark_inject_kv_push {
    uint64_t k_raw_base_va;
    uint64_t v_raw_base_va;
    uint64_t k_norm_va[3];
    uint64_t k_store_va[3];
    uint64_t v_store_va[3];
    uint64_t k_idxs_va[3];
    uint64_t v_idxs_va[3];
    uint64_t inp_pos_va;
    uint64_t rope_freq_va;
    uint32_t n_layers;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_tokens;
    uint32_t layer_raw_stride_f32;
    uint32_t k_raw_stride_f32;
    uint32_t v_raw_stride_f32;
    uint32_t k_store_nb1_B[3];
    uint32_t v_store_nb1_B[3];
    uint32_t k_ids_nb0_B;
    uint32_t v_ids_nb0_B;
    uint32_t rope_n_dims;
    uint32_t layer_flags[3];
    float    rms_eps;
    float    theta_scale;
    float    freq_scale;
    float    attn_factor;
    uint32_t pad0;
};
static_assert(sizeof(agx_w8_dspark_inject_kv_push) <= 256, "agx_w8_dspark_inject_kv_push > 256B");

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
#define AGX_W5_PFLAG_FUSE_ATTN_IN      (1u << 13)
#define AGX_W5_PFLAG_FUSE_WO           (1u << 14)
#define AGX_W5_PFLAG_FUSE_SINK_PROJ    (1u << 15)
#define AGX_W5_PFLAG_FUSE_DYN_CONV     (1u << 16)

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
    uint32_t n_tokens;
    uint32_t rollback_slots;
    uint32_t rollback_stride_f32;
    uint32_t reserved[25]; // pad to 1152 B = exact 9 x 128B cache lines
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
    uint32_t n_tokens;       // 1 for B=1, 2..8 for Wave 9 M=2..8 coalesced attn_in!
};
static_assert(sizeof(agx_w5_attn_in_push) == 152, "agx_w5_attn_in_push must be 152B (<= 256B)");
