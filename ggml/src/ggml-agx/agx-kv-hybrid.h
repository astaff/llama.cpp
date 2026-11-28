#pragma once

#include "agx-common.h"
#include "agx-drm.h"
#include "agx-cdm.h"
#include "agx-compiler.h"

#include <cstddef>
#include <cstdint>

// ============================================================================
// Pillar 4: Hybrid 1M-Context KV Cache (40 SWA-1024 Rings + 8 Paged Full-Attn)
//           & Fused AGX3B Attention Kernels (SWA Ring + Split-K Flash-Decode)
// ============================================================================
//
// Inkling 2 Mini (24 physical layers, 3x looped core 6..17 -> 48 logical passes):
//   - 40 SWA-1024 passes: fixed-size circular ring buffers in a single SLC-pinned
//     BO (~80.0 MiB at F8_E4M3, ~85.0 MiB at Q8_0 planar, ~160.0 MiB at F16).
//   - 8 Full-attention passes (logical slots {5, 11, 17, 23, 29, 35, 41, 47}):
//     demand-grown 64-token paged pool in SLC-streaming BOs (16.0 GiB at 1M ctx
//     in F8_E4M3, 17.0 GiB in Q8_0 planar), indexed by a 1 MiB SLC-pinned
//     page table (`uint64_t page_va[8][16384]`).
//   - Also supports zero-copy external contiguous `ggml_tensor` K/V views for
//     generic `GGML_OP_FLASH_ATTN_EXT` and `test-backend-ops`.

#define AGX_KV_MAX_SLOTS        64u
#define AGX_KV_DEFAULT_SWA_WIN  1024u
#define AGX_KV_PAGE_TOKENS      64u
#define AGX_KV_PAGE_SHIFT       6u
#define AGX_KV_PAGE_MASK        (AGX_KV_PAGE_TOKENS - 1u)
#define AGX_KV_MAX_FULL_TOKENS  1048576u
#define AGX_KV_MAX_FULL_PAGES   (AGX_KV_MAX_FULL_TOKENS / AGX_KV_PAGE_TOKENS) // 16384
#define AGX_KV_MAX_SPLITS       64u

enum agx_kv_dtype : uint32_t {
    AGX_KV_F8_E4M3      = 0, // 1 B/elem (128 B/head -> 1 cache line)
    AGX_KV_Q8_0_PLANAR  = 1, // planar 128B int8 qs + 8B fp16 scales = 136 B/head
    AGX_KV_F16          = 2, // 2 B/elem (256 B/head -> 2 cache lines)
    AGX_KV_F32          = 3, // 4 B/elem (external ggml_tensor F32 K/V)
    AGX_KV_Q8_0_GGML    = 4, // external ggml block_q8_0 (34 B per 32 elems)
};

enum agx_attn_flags : uint32_t {
    AGX_ATTN_FLAG_NONE              = 0u,
    AGX_ATTN_FLAG_CAUSAL            = 1u << 0, // mask keys with pos_k > pos_q
    AGX_ATTN_FLAG_SWA_WINDOW        = 1u << 1, // mask keys with pos_q - pos_k >= swa_window
    AGX_ATTN_FLAG_HAS_REL_TABLE     = 1u << 2, // rel_table_va [M, Hq, n_extent+1] F32
    AGX_ATTN_FLAG_REL_TABLE_SINK    = 1u << 3, // rel_table already has -sink subtracted (zero_sink=0)
    AGX_ATTN_FLAG_SINK_PER_HEAD     = 1u << 4, // sinks_va [Hq] F32 (ggml FLASH_ATTN_EXT)
    AGX_ATTN_FLAG_SINK_PER_TOK_HEAD = 1u << 5, // sinks_va [M, Hq] F32 (unscaled Inkling sink)
    AGX_ATTN_FLAG_HAS_MASK_F16      = 1u << 6, // kq_mask_va [M, n_kv] F16 (ggml mask)
    AGX_ATTN_FLAG_HAS_MASK_F32      = 1u << 7, // kq_mask_va [M, n_kv] F32
    AGX_ATTN_FLAG_FWHT64            = 1u << 8, // in-register orthonormal FWHT64(kqv_out) before gate
    AGX_ATTN_FLAG_QGATE_F32         = 1u << 9, // multiply output by sigmoid(q_gate) [M, Hq*D] F32
    AGX_ATTN_FLAG_QGATE_F16         = 1u << 10,// multiply output by sigmoid(q_gate) [M, Hq*D] F16
    AGX_ATTN_FLAG_SOFTCAP           = 1u << 11,// logit_softcap * tanh(s) (ggml convention)
    AGX_ATTN_FLAG_ALIBI             = 1u << 12,// max_bias ALiBi slope on mask
    AGX_ATTN_FLAG_PAGED_KV          = 1u << 13,// full-attn paged pool via page_table_va
    AGX_ATTN_FLAG_SWA_RING_KV       = 1u << 14,// SWA-1024 circular ring pool
    AGX_ATTN_FLAG_Q_F16             = 1u << 15,// Q input is F16 instead of F32
    AGX_ATTN_FLAG_OUT_F16           = 1u << 16,// attn_out is F16 instead of F32
    AGX_ATTN_FLAG_KV_BOUNDS         = 1u << 17,// partial_va holds uvec2[n_q_tiles] [kv_beg, kv_end) in PASS_MODE=3
};

struct agx_kv_hybrid_config {
    uint32_t     n_slots           = 48;    // logical passes (Inkling 2 Mini = 48)
    uint32_t     n_head_kv         = 8;
    uint32_t     head_dim          = 128;
    uint32_t     swa_window        = AGX_KV_DEFAULT_SWA_WIN; // must be power of 2
    uint32_t     max_full_tokens   = 32768; // initial page-table limit (up to 1,048,576)
    uint32_t     page_chunk_pages  = 64;    // 64 pages = 4096 tokens per demand chunk
    agx_kv_dtype swa_dtype         = AGX_KV_F8_E4M3;
    agx_kv_dtype full_dtype        = AGX_KV_F8_E4M3;
    bool         slot_is_full[AGX_KV_MAX_SLOTS] = {};
    bool         use_inkling2_default_pattern   = true;
};

struct agx_kv_hybrid_stats {
    uint32_t n_swa_slots;
    uint32_t n_full_slots;
    uint32_t swa_window;
    uint32_t full_allocated_pages;
    uint32_t full_max_pages;
    uint64_t swa_ring_bytes;         // SLC-pinned bytes
    uint64_t full_pool_bytes;        // SLC-streaming bytes currently allocated
    uint64_t full_projected_1m_bytes;// projected bytes at 1,048,576 tokens
    uint64_t metadata_slc_bytes;     // page table + pos tables + split-K scratch
};

struct agx_kv_hybrid_pool;

// Fill Inkling 2 Mini's 48-slot pattern: 8 full-attn slots {5,11,17,23,29,35,41,47}
// and 40 SWA-1024 ring slots.
void agx_kv_hybrid_fill_inkling2_pattern(bool slot_is_full[AGX_KV_MAX_SLOTS], uint32_t n_slots = 48);

// Create/destroy the hybrid KV pool + compile fused attention & KV-append kernels.
agx_kv_hybrid_pool * agx_kv_hybrid_create(agx_drm_device *            dev,
                                          agx_compiler *              compiler,
                                          const agx_kv_hybrid_config & cfg);
void                 agx_kv_hybrid_destroy(agx_kv_hybrid_pool * pool);

// Reset all sequence positions and ring slots to empty (does not free demand pages unless release_pages=true).
void agx_kv_hybrid_reset(agx_kv_hybrid_pool * pool, bool release_pages = false);

// Ensure full-attention pages exist for `[0, required_tokens)` across all 8 full-attn slots.
bool agx_kv_hybrid_ensure_full_capacity(agx_kv_hybrid_pool * pool, uint32_t required_tokens);

// Query pool stats & slot metadata
void     agx_kv_hybrid_get_stats(const agx_kv_hybrid_pool * pool, agx_kv_hybrid_stats * out);
bool     agx_kv_hybrid_slot_is_full(const agx_kv_hybrid_pool * pool, uint32_t logical_slot);
uint32_t agx_kv_hybrid_slot_pool_idx(const agx_kv_hybrid_pool * pool, uint32_t logical_slot);
uint64_t agx_kv_hybrid_swa_slot_va(const agx_kv_hybrid_pool * pool, uint32_t logical_slot);
uint64_t agx_kv_hybrid_swa_pos_va(const agx_kv_hybrid_pool * pool, uint32_t logical_slot);
uint64_t agx_kv_hybrid_full_page_table_va(const agx_kv_hybrid_pool * pool, uint32_t logical_slot);
uint64_t agx_kv_hybrid_full_pos_va(const agx_kv_hybrid_pool * pool);

// Stage 1 -> KV Append: quantize & store `k_new`, `v_new` ([D, Hkv, M] F32 or F16)
// into either the SWA-1024 ring (`pos & (W-1)`) or the demand-paged full-attn pool
// (`kv_idx >> 6`, `kv_idx & 63`) and record token positions on GPU.
struct agx_kv_append_args {
    uint32_t logical_slot;
    uint32_t n_tokens;         // M
    uint32_t kv_write_base;    // starting logical KV index (full-attn) or base pos
    uint64_t k_new_va;         // [M, Hkv, D]
    uint64_t v_new_va;         // [M, Hkv, D]
    uint64_t pos_va;           // optional [M] int32_t positions; if 0, uses kv_write_base + t
    bool     src_is_f16;
};

bool agx_kv_hybrid_append(agx_kv_hybrid_pool *     pool,
                          agx_cdm_stream *         stream,
                          const agx_kv_append_args & args,
                          agx_barrier_mode         post_barrier = AGX_BARRIER_STRONG_USC);

// Stage 2: Fused Attention Launch Contract (called by @megakernel Stage 2 and generic FLASH_ATTN_EXT)
struct agx_attn_stage_args {
    uint32_t     logical_slot;      // 0..47 (ignored if external contiguous KV)
    uint32_t     n_tokens;          // M (1 for decode, 5 for DSpark, up to 8192 prefill)
    uint32_t     n_kv;              // total active KV tokens (for SWA: min(total_seen, swa_window) or swa_window)
    uint32_t     n_head_q;          // 24
    uint32_t     n_head_kv;         // 8
    uint32_t     head_dim;          // 128
    uint32_t     n_rel_extent;      // 1024 for SWA, hparams.n_relative_extent for full-attn
    uint32_t     pos_q_base;        // used when pos_q_va == 0: pos_q[t] = pos_q_base + t
    uint32_t     flags;             // agx_attn_flags bitmask
    float        kq_scale;          // 1.0f / sqrtf(head_dim)
    float        max_bias;          // ALiBi max_bias (0.0f for Inkling 2)
    float        logit_softcap;     // 0.0f for Inkling 2

    uint64_t     q_va;              // [M, Hq, D] F32 (or F16 if AGX_ATTN_FLAG_Q_F16)
    uint64_t     out_va;            // [M, Hq, D] F32 (or F16 if AGX_ATTN_FLAG_OUT_F16)
    uint64_t     rel_table_va;      // optional [M, Hq, n_rel_extent + 1] F32
    uint64_t     sinks_va;          // optional [Hq] or [M, Hq] F32
    uint64_t     q_gate_va;         // optional [M, Hq, D] F32/F16 (wqkv_gate * x)
    uint64_t     pos_q_va;          // optional [M] int32_t query positions
    uint64_t     kq_mask_va;        // optional [M, n_kv] F16/F32 mask (generic ggml)

    // External contiguous KV override (used when neither SWA_RING_KV nor PAGED_KV is set):
    agx_kv_dtype ext_k_dtype;
    agx_kv_dtype ext_v_dtype;
    uint64_t     ext_k_va;
    uint64_t     ext_v_va;
    uint32_t     ext_k_nb1;         // byte stride between KV tokens
    uint32_t     ext_k_nb2;         // byte stride between KV heads
    uint32_t     ext_v_nb1;
    uint32_t     ext_v_nb2;
    uint32_t     ext_mask_nb1;      // byte stride between mask rows (queries)
    uint32_t     ext_mask_nb2;      // byte stride between mask heads (0 = broadcast across heads)
    uint32_t     ext_mask_ne2;      // mask head count for modulo broadcast (h_q % ext_mask_ne2)
    uint32_t     q_nb1;             // optional Q token byte stride (0 -> n_head_q * head_dim * elem)
    uint32_t     q_nb2;             // optional Q head byte stride  (0 -> head_dim * elem)
    uint64_t     kv_bounds_va;      // optional uvec2[n_q_tiles] active [kv_beg, kv_end) bounds (PASS_MODE=3)
};

// Dispatch fused SWA-1024 ring attention (single-wave 40-core, zero split-K reduce).
bool agx_kv_hybrid_dispatch_swa_attn(agx_kv_hybrid_pool *        pool,
                                     agx_cdm_stream *            stream,
                                     const agx_attn_stage_args & args,
                                     agx_barrier_mode            post_barrier = AGX_BARRIER_STRONG_USC);

// Dispatch fused full-attention split-K Flash-Decoding (Phase 1 split-K + Phase 2 reduce
// with single-application sink, optional FWHT64, and optional sigmoid(Qgate)).
bool agx_kv_hybrid_dispatch_full_attn(agx_kv_hybrid_pool *        pool,
                                      agx_cdm_stream *            stream,
                                      const agx_attn_stage_args & args,
                                      agx_barrier_mode            post_barrier = AGX_BARRIER_STRONG_USC);

// Unified dispatcher: picks SWA ring vs full-attn Flash-Decode from `logical_slot` (or flags).
bool agx_kv_hybrid_dispatch_attn(agx_kv_hybrid_pool *        pool,
                                 agx_cdm_stream *            stream,
                                 const agx_attn_stage_args & args,
                                 agx_barrier_mode            post_barrier = AGX_BARRIER_STRONG_USC);

// Optional epilogue & tile-bounds fusion parameters for generic `GGML_OP_FLASH_ATTN_EXT`.
struct agx_ggml_flash_attn_epi {
    uint32_t extra_flags  = 0u;   // e.g. AGX_ATTN_FLAG_FWHT64 | AGX_ATTN_FLAG_QGATE_F32 | AGX_ATTN_FLAG_KV_BOUNDS
    uint64_t out_va       = 0ull; // override dst GPU VA when fused (0 -> agx_tensor_gpu_va(dst))
    uint64_t out_nb3      = 0ull; // batch byte stride for fused out_va (0 -> dst->nb[3])
    uint64_t q_gate_va    = 0ull; // optional [D*Hq*N, B] F32/F16 pre-sigmoid gate
    uint64_t q_gate_nb3   = 0ull; // batch byte stride for q_gate_va
    uint64_t kv_bounds_va = 0ull; // optional uvec2[n_q_tiles] active [kv_beg, kv_end) bounds (B == 1)
};

// Generic ggml `GGML_OP_FLASH_ATTN_EXT` support check & dispatcher for `agx-graph-exec.cpp`.
bool agx_kv_hybrid_supports_ggml_flash_attn(const struct ggml_tensor * dst);
bool agx_kv_hybrid_ggml_flash_attn_uses_scratch(const struct ggml_tensor * dst);
bool agx_kv_hybrid_dispatch_ggml_flash_attn_ext(agx_kv_hybrid_pool *            pool,
                                                agx_cdm_stream *                stream,
                                                const struct ggml_tensor *      dst,
                                                const agx_ggml_flash_attn_epi * epi,
                                                agx_barrier_mode                post_barrier = AGX_BARRIER_STRONG_USC);
bool agx_kv_hybrid_dispatch_ggml_flash_attn(agx_kv_hybrid_pool *     pool,
                                            agx_cdm_stream *         stream,
                                            const struct ggml_tensor * dst,
                                            agx_barrier_mode         post_barrier = AGX_BARRIER_STRONG_USC);

// Host-side reference FP8 E4M3FN & FWHT64 helpers (for verification / tests).
uint8_t agx_fp32_to_e4m3fn(float x);
float   agx_e4m3fn_to_fp32(uint8_t b);
void    agx_fwht64_f32_ref(float * x, uint32_t n_blocks);
