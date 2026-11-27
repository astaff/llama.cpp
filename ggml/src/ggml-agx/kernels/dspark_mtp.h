#pragma once

#include <cstdint>

// ============================================================================
// Pillar 3: Fused DSpark MTP Sidecar Kernels (Sharing Same AGX3 Context)
// ============================================================================
// Exact match to `src/models/dspark.cpp` & `src/models/dflash.cpp` (Inkling 2 DSpark):
//   Mode 0: INJECT_CTX_MIX
//           rms_norm(sum_{k=0..n_target-1} mix_w[k] * target[k]) * norm_w
//   Mode 1: CONCAT4_JOINT_GRU
//           w1_prev = markov_w1[prev_tok]
//           joint   = joint_proj @ [state, w1_prev, hidden_i, (f_rnn)] + joint_proj_b
//           gate    = sigmoid(joint[0..R-1])
//           cand    = tanh(joint[R..2R-1])
//           state   = cand + gate * (state - cand)
//           markov  = tanh(joint[2R..3R-1])
//   Mode 2: MARKOV_W2_BIAS_SOFTCAP
//           col[v]  = softcap(base_i[v] + (markov_w2[v] . markov) * w2_scale + w2_bias[v])
//           + 40-WG partial argmax in SLC
//   Mode 3: DRAW_ARGMAX_OR_TOP64_PACK
//           Greedy argmax (inv_temp == 0) or top-64 descending sort + softmax(vals*inv_temp)
//           + CDF draw against uniform u -> writes next prev_tok & packs [tok, k, ids, probs]

#define AGX_DSPARK_MODE_INJECT_CTX_MIX     0u
#define AGX_DSPARK_MODE_CONCAT4_JOINT_GRU  1u
#define AGX_DSPARK_MODE_MARKOV_W2_SOFTCAP  2u
#define AGX_DSPARK_MODE_DRAW_PACK          3u

#define AGX_DSPARK_FLAG_HAS_F_RNN          (1u << 0)
#define AGX_DSPARK_FLAG_ZERO_INIT_STATE    (1u << 1)
#define AGX_DSPARK_FLAG_HAS_W2_BIAS        (1u << 2)
#define AGX_DSPARK_FLAG_COMPUTE_ARGMAX     (1u << 3)

struct agx_dspark_inject_mix_push {
    uint64_t feats_va;      // [n_embd, n_target, n_tokens] F32 (contiguous per-token taps)
    uint64_t mix_w_va;      // [n_target] F32
    uint64_t norm_w_va;     // [n_embd] F32
    uint64_t out_va;        // [n_embd, n_tokens] F32
    uint32_t n_embd;        // 2304
    uint32_t n_target;      // e.g. 5..12 target taps
    uint32_t n_tokens;      // M
    float    rms_eps;       // 1e-6f
};
static_assert(sizeof(agx_dspark_inject_mix_push) <= 256, "agx_dspark_inject_mix_push exceeds 256B");

struct agx_dspark_joint_gru_push {
    uint64_t prev_tok_va;   // [n_blocks] I32 (current prev token per block)
    uint64_t w1_va;         // [rank, n_vocab] (BF16 / Q8_0 / F16 / F32)
    uint64_t hidden_va;     // [n_embd, block_drafts, n_blocks] F32
    uint64_t f_rnn_va;      // optional [n_embd, n_blocks] F32 (if HAS_F_RNN)
    uint64_t state_va;      // [rank, n_blocks] F32 (in/out)
    uint64_t markov_out_va; // [rank, n_blocks] F32 (out)
    uint64_t joint_w_va;    // [n_joint, 3 * rank] (BF16 / Q8_0 / F16 / F32)
    uint64_t joint_b_va;    // [3 * rank] F32 (optional, 0 if none)
    uint32_t n_embd;        // 2304
    uint32_t rank;          // markov_rank (e.g. 64..512)
    uint32_t n_blocks;
    uint32_t block_drafts;
    uint32_t draft_pos;     // current i in [i_draft_beg .. block_drafts-1]
    uint32_t w1_wtype;      // AGX_WTYPE_*
    uint32_t w1_row_bytes;
    uint32_t joint_wtype;   // AGX_WTYPE_*
    uint32_t joint_row_bytes;
    uint32_t flags;         // AGX_DSPARK_FLAG_*
};
static_assert(sizeof(agx_dspark_joint_gru_push) <= 256, "agx_dspark_joint_gru_push exceeds 256B");

struct agx_dspark_w2_softcap_push {
    uint64_t markov_va;     // [rank, n_blocks] F32
    uint64_t w2_va;         // [rank, n_vocab] (BF16 / Q8_0 / F16 / F32)
    uint64_t w2_bias_va;    // optional [n_vocab] F32
    uint64_t base_va;       // [n_vocab, block_drafts, n_blocks] F32
    uint64_t out_col_va;    // [n_vocab, n_blocks] F32 (or directly into block_drafts slot)
    uint64_t part_val_va;   // [n_splits, n_blocks] F32 (partial argmax)
    uint64_t part_idx_va;   // [n_splits, n_blocks] I32 (partial argmax)
    uint32_t rank;
    uint32_t n_vocab;
    uint32_t n_blocks;
    uint32_t block_drafts;
    uint32_t draft_pos;
    uint32_t n_splits;      // e.g. 40
    uint32_t w2_wtype;
    uint32_t w2_row_bytes;
    uint32_t flags;
    float    w2_scale;      // 1.0f if no scale tensor
    float    softcap;       // 0.0f = disabled
    float    inv_softcap;   // 1.0f / softcap
};
static_assert(sizeof(agx_dspark_w2_softcap_push) <= 256, "agx_dspark_w2_softcap_push exceeds 256B");

struct agx_dspark_draw_pack_push {
    uint64_t col_va;        // [n_vocab, n_blocks] F32
    uint64_t part_val_va;   // [n_splits, n_blocks] F32 (from Mode 2 argmax)
    uint64_t part_idx_va;   // [n_splits, n_blocks] I32
    uint64_t rnn_state_va;  // optional [n_embd, block_drafts, n_blocks] F32 ([uniform, inv_temp])
    uint64_t prev_tok_va;   // [n_blocks] I32 (updated in-place for next draft step!)
    uint64_t out_pack_va;   // optional [n_embd, block_drafts, n_blocks] F32 (h_nextn)
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_blocks;
    uint32_t block_drafts;
    uint32_t draft_pos;
    uint32_t n_splits;
    uint32_t n_cand;        // min(64, n_vocab)
    uint32_t has_rnn_state; // 1 = top-64 CDF draw + pack; 0 = greedy argmax only
};
static_assert(sizeof(agx_dspark_draw_pack_push) <= 256, "agx_dspark_draw_pack_push exceeds 256B");
