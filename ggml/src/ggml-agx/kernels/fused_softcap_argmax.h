#pragma once

#include <cstdint>

// ============================================================================
// Pillar 3: Fused Final Logit Softcap + Vocab Truncation + Parallel Argmax
// ============================================================================
// Matches `llama_model_inkling2::graph::build_output` (inkling2.cpp L459-469):
//   if (cap > 0) y = cap * tanh(x / cap)
//   truncate [n_vocab_padded, M] -> [n_vocab_unpadded, M]
// Optional zero-host-roundtrip 2-stage parallel argmax (40 WGs -> 1 WG) writing
// exact greedy token IDs directly to SLC.

#define AGX_SOFTCAP_MODE_TRUNC_ONLY    0u // softcap + truncate padded vocab
#define AGX_SOFTCAP_MODE_ARGMAX_STAGE1 1u // optional softcap + optional write + 40-WG partial argmax
#define AGX_SOFTCAP_MODE_ARGMAX_STAGE2 2u // 1-WG reduce partial argmax -> exact token ID

#define AGX_SOFTCAP_FLAG_WRITE_LOGITS  (1u << 0)

struct agx_softcap_argmax_push {
    uint64_t src_va;           // [n_vocab_padded, n_tokens] F32
    uint64_t dst_logits_va;    // optional [n_vocab_unpadded, n_tokens] F32
    uint64_t partial_val_va;   // [n_splits, n_tokens] F32 (SLC scratch)
    uint64_t partial_idx_va;   // [n_splits, n_tokens] I32 (SLC scratch)
    uint64_t out_token_va;     // [n_tokens] I32 (exact greedy argmax token ID)
    uint32_t n_vocab_unpadded; // 200,058
    uint32_t src_stride_f32;   // 201,024 (padded vocab stride)
    uint32_t dst_stride_f32;   // 200,058
    uint32_t n_tokens;         // M
    uint32_t n_splits;         // e.g. 40 (M4 Max GPU cores)
    uint32_t flags;
    float    softcap;          // 0.0f = disabled; else cap * tanh(x / cap)
    float    inv_softcap;      // 1.0f / softcap
};
static_assert(sizeof(agx_softcap_argmax_push) <= 256, "agx_softcap_argmax_push exceeds 256B limit");
