#pragma once

#include <cstdint>

// ============================================================================
// Pillar 3: Fused Inkling 2 Mini Loop-Prep, QKV/Conv/Clip/FWHT & Router
// ============================================================================
// Compiled into lean specialized AGX3B kernels via -DPREP_MODE=0..5 so every
// kernel stays <= 64 32-bit GPRs with 0:0 spills.

#define AGX_PREP_MODE_DEPTH_MIX         0u // mix(blocks, cur, query) + optional RMSNorm
#define AGX_PREP_MODE_CONV_CLIP_FWHT    1u // shortconv + dyn conv + L2 clip / RMSNorm + tau + FWHT_{64,128}
#define AGX_PREP_MODE_POST_CONV_RES     2u // attn_post_conv / ffn_post_conv + residual add + history update
#define AGX_PREP_MODE_ROUTER_TOP8       3u // 2304->128 router + sigmoid + bias + top-8 + norm(1e-20) -> SLC
#define AGX_PREP_MODE_REL_BIAS_TABLE    4u // Rcur @ pad(attn_rel_bias) - sinks -> rel_table[n_extent+1, Hq, M]
#define AGX_PREP_MODE_SENSE_BARRIER     5u // TLA+-verified 40-WG sense-reversing atomic barrier probe

// Flags for Mode 1 (CONV_CLIP_FWHT)
#define AGX_PREP_FLAG_DO_CONV           (1u << 0)
#define AGX_PREP_FLAG_DYNAMIC_CONV      (1u << 1)
#define AGX_PREP_FLAG_PER_CHAN_SCALE    (1u << 2)
#define AGX_PREP_FLAG_L2_CLIP           (1u << 3)
#define AGX_PREP_FLAG_RMS_NORM          (1u << 4)
#define AGX_PREP_FLAG_TAU_SCALE         (1u << 5)
#define AGX_PREP_FLAG_FWHT_64           (1u << 6)
#define AGX_PREP_FLAG_FWHT_128          (1u << 7)
#define AGX_PREP_FLAG_UPDATE_STATE      (1u << 8)

// Flags for Mode 2 (POST_CONV_RES)
#define AGX_POST_FLAG_ADD_RESIDUAL      (1u << 0)
#define AGX_POST_FLAG_UPDATE_STATE      (1u << 1)

// Mode 0 push args: depth-mix residual attention
struct agx_depth_mix_push {
    uint64_t states_ptrs_va; // GPU VA of uint64_t[n_states]: [b_{n-1}, ..., b_0, cur]
    uint64_t query_va;       // [n_embd, n_queries] F32
    uint64_t norm_w_va;      // optional [n_embd] F32 (only used when n_queries == 1 and non-zero)
    uint64_t out_va;         // [n_embd, n_queries, n_tokens] F32
    uint64_t out_pre_va;     // optional pre-norm out when norm_w_va != 0
    uint32_t n_embd;         // 2304
    uint32_t n_states;       // <= 16
    uint32_t n_queries;      // 1, 3, or 4
    uint32_t n_tokens;       // M
    float    inv_sqrt_d;     // 1.0f / sqrtf(n_embd)
    float    eps_mix;        // 1e-6f
    float    eps_norm;       // model f_norm_rms_eps
    uint32_t pad0;
};
static_assert(sizeof(agx_depth_mix_push) <= 256, "agx_depth_mix_push exceeds 256B");

// Mode 1 push args: fused K/V conv + Q/K/R exact L2 clip + V RMSNorm + tau + FWHT_{64,128}
struct agx_conv_clip_fwht_push {
    uint64_t src_va;         // [head_dim * n_head, n_tokens] F32
    uint64_t dst_va;         // [head_dim, n_head, n_tokens] F32
    uint64_t conv_w_va;      // [n_conv, head_dim * n_head] F32 (if DO_CONV)
    uint64_t dyn_va;         // [n_conv, n_head, n_tokens] F32 (if DYNAMIC_CONV)
    uint64_t history_va;     // [n_conv - 1, n_sconv_channels, n_seqs] F32
    uint64_t state_out_va;   // [n_conv - 1, n_sconv_channels, n_seqs] F32 (may alias history_va for B=1)
    uint64_t clip_scale_va;  // [head_dim, n_head] F32 (if PER_CHAN_SCALE)
    uint64_t inp_pos_va;     // [n_tokens] I32 (if TAU_SCALE)
    uint32_t head_dim;       // 128 (Q/K/V) or n_relative (R) or 64 (FWHT64)
    uint32_t n_head;         // 24 (Q/R) or 8 (K/V)
    uint32_t n_seq_tokens;   // tokens per sequence
    uint32_t n_seqs;         // sequences in ubatch
    uint32_t n_conv;         // e.g. 4
    uint32_t n_sconv_ch;     // total channels in conv history buffer
    uint32_t conv_ch_offset; // 0 for K, n_embd_k_gqa for V
    uint32_t flags;          // AGX_PREP_FLAG_*
    float    conv_res_scale; // 1.0f / sqrtf((dyn ? 2 : 1) * n_conv + 1)
    float    clip_thresh;    // f_qk_clip or f_relative_clip
    float    rms_eps;        // f_norm_rms_eps
    float    log_scale_alpha;// f_log_scale_alpha
    float    inv_log_floor;  // 1.0f / float(n_log_scale_floor)
    uint32_t pad0;
};
static_assert(sizeof(agx_conv_clip_fwht_push) <= 256, "agx_conv_clip_fwht_push exceeds 256B");

// Mode 2 push args: attn_post_conv / ffn_post_conv + residual add
struct agx_post_conv_res_push {
    uint64_t src_va;         // [n_channels, n_tokens] F32
    uint64_t res_va;         // [n_channels, n_tokens] F32 (if ADD_RESIDUAL)
    uint64_t dst_va;         // [n_channels, n_tokens] F32
    uint64_t conv_w_va;      // [n_conv, n_channels] F32
    uint64_t history_va;     // [n_conv - 1, n_sconv_ch, n_seqs] F32
    uint64_t state_out_va;   // [n_conv - 1, n_sconv_ch, n_seqs] F32
    uint32_t n_channels;     // 2304
    uint32_t n_seq_tokens;
    uint32_t n_seqs;
    uint32_t n_conv;
    uint32_t n_sconv_ch;
    uint32_t conv_ch_offset; // 2*n_embd_k_gqa (attn_post) or 2*n_embd_k_gqa + n_embd (ffn_post)
    uint32_t flags;          // AGX_POST_FLAG_*
    float    conv_res_scale; // 1.0f / sqrtf(n_conv + 1)
};
static_assert(sizeof(agx_post_conv_res_push) <= 256, "agx_post_conv_res_push exceeds 256B");

// Mode 3 push args: fused 2304 -> 128 router + sigmoid + bias + top-8 + norm -> SLC
struct agx_router_top8_push {
    uint64_t x_va;           // [n_embd, n_tokens] F32 (ffn_norm output)
    uint64_t w_va;           // [n_embd, n_experts] (Q8_0_8BLK / Q8_0_RAW / F32 / F16 / BF16)
    uint64_t bias_va;        // optional [n_experts] F32 (ffn_exp_probs_b, 0 if none)
    uint64_t out_ids_va;     // [top_k, n_tokens] I32 in SLC
    uint64_t out_w_va;       // [top_k, n_tokens] F32 in SLC
    uint32_t n_embd;         // 2304
    uint32_t n_experts;      // 128
    uint32_t top_k;          // 8
    uint32_t n_tokens;       // M
    uint32_t wtype;          // AGX_WTYPE_*
    uint32_t w_row_bytes;    // bytes per expert row
    float    norm_eps;       // 1e-20f (Inkling 2 exact expert_weights_norm_eps)
    float    weights_scale;  // hparams.expert_weights_scale (0 or 1 => unscaled)
};
static_assert(sizeof(agx_router_top8_push) <= 256, "agx_router_top8_push exceeds 256B");

// Mode 4 push args: Rcur @ pad(attn_rel_bias, +1 zero row) - sinks -> rel_table
struct agx_rel_bias_table_push {
    uint64_t rcur_va;        // [n_rel, n_head, n_tokens] F32
    uint64_t rel_bias_va;    // [n_rel, n_extent] F32 (or F16/BF16 via wtype)
    uint64_t sinks_va;       // [n_head, n_tokens] F32
    uint64_t out_table_va;   // [n_extent + 1, n_head, n_tokens] F32
    uint32_t n_rel;
    uint32_t n_extent;
    uint32_t n_head;
    uint32_t n_tokens;
    uint32_t bias_wtype;     // AGX_WTYPE_F32 / F16 / BF16
    uint32_t pad0;
};
static_assert(sizeof(agx_rel_bias_table_push) <= 256, "agx_rel_bias_table_push exceeds 256B");

// Mode 5 push args: TLA+-verified sense-reversing barrier test / persistent driver
struct agx_sense_barrier_push {
    uint64_t barrier_va;     // agx_sense_barrier_state (128B SLC-pinned)
    uint64_t scratch_va;     // [n_workgroups] U32 in SLC for coherence check
    uint32_t n_workgroups;   // 40 on M4 Max
    uint32_t n_iters;        // e.g. 48 passes * 3 stages = 144 barriers
};
static_assert(sizeof(agx_sense_barrier_push) <= 256, "agx_sense_barrier_push exceeds 256B");
