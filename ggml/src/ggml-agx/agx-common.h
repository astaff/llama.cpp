#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

// ============================================================================
// Apple Silicon AGX3 (M4 Max / J616c / T6041, HAL200 / AGX3B) Constants
// ============================================================================

#define AGX_SIMD_WIDTH          32u
#define AGX_CACHE_LINE_BYTES    128u
#define AGX_PAGE_SIZE           16384u
#define AGX_MAX_TG_SRAM_BYTES   32768u
#define AGX_SLC_BYTES           (48ULL * 1024ULL * 1024ULL)
#define AGX_MAX_PUSH_BYTES      256u
#define AGX_HOT_GPR32_BUDGET    96u
#define AGX_HOT_GPR16_BUDGET    (AGX_HOT_GPR32_BUDGET * 2u)
#define AGX_USC_WINDOW_BASE     0x10000000000ULL  // Bit 40 MUST be set on G17P/M4 Max
#define AGX_USC_WINDOW_SIZE     0x100000000ULL    // 4 GiB USC executable window

static inline size_t agx_align_up(size_t v, size_t align) {
    return (v + align - 1u) & ~(align - 1u);
}

static inline uint64_t agx_align_up_u64(uint64_t v, uint64_t align) {
    return (v + align - 1ULL) & ~(align - 1ULL);
}

#define AGX_LOG_INFO(...)  do { std::fprintf(stderr, "[ggml-agx] "); std::fprintf(stderr, __VA_ARGS__); std::fputc('\n', stderr); } while (0)
#define AGX_LOG_WARN(...)  do { std::fprintf(stderr, "[ggml-agx WARN] "); std::fprintf(stderr, __VA_ARGS__); std::fputc('\n', stderr); } while (0)
#define AGX_LOG_ERROR(...) do { std::fprintf(stderr, "[ggml-agx ERROR] "); std::fprintf(stderr, __VA_ARGS__); std::fputc('\n', stderr); } while (0)

// ============================================================================
// Pillar 4: SLC Residency & VM Cache Policy Partitioning
// ============================================================================

enum agx_cache_policy : uint32_t {
    // Hot working set: activations, cross-pass residuals, SWA-1024 KV rings (~84 MB),
    // router weights, DSpark MTP weights, sense-reversing barrier scratchpad.
    // Mapped CPU writeback + DRM_ASAHI_GEM_RESIDENCY_KEEP in low-fragmentation hot VA arena.
    AGX_CACHE_SLC_PIN    = 0,

    // Cold streaming buffers: 14.7 GB MoE expert weights, demand-paged full-attention KV pools.
    // Mapped write-combine (no residency pin) in separate streaming VA arena; read-only bind for weights.
    AGX_CACHE_SLC_STREAM = 1,

    // USC executable code & Ioto state-loader arena inside [0x10000000000, 0x10100000000).
    AGX_CACHE_EXEC       = 2,

    // CDM command streams & push-constant / root-descriptor upload rings.
    AGX_CACHE_CMDSTREAM  = 3,
};

enum agx_bo_flags : uint32_t {
    AGX_BO_NONE       = 0u,
    AGX_BO_SLC_PIN    = 1u << 0,
    AGX_BO_SLC_STREAM = 1u << 1,
    AGX_BO_EXEC       = 1u << 2,
    AGX_BO_CMDSTREAM  = 1u << 3,
    AGX_BO_READONLY   = 1u << 4,
    AGX_BO_WRITEBACK  = 1u << 5,
    AGX_BO_ZERO       = 1u << 6,
    AGX_BO_TIMESTAMPS = 1u << 7,
};

// ============================================================================
// Pillar 6: TLA+-Verified AGX3 CDM Barrier Modes & Sense-Reversing Barrier
// ============================================================================

enum agx_barrier_mode : uint32_t {
    // Elide trailing weak barrier between independent dispatches so work overlaps across 40 cores.
    AGX_BARRIER_NONE        = 0,
    // CDM wait_for_idle (0x60000160): execution ordering only, no cache flush.
    AGX_BARRIER_WEAK_WFI    = 1,
    // Strong intra-VM USC/SLC visibility flush (0x60001560, flags=5): same-VA coherent in SLC, no DRAM writeback.
    AGX_BARRIER_STRONG_USC  = 2,
    // Memory-scope flush (0x6000157c, flags=5 | FLUSH_MEMORY): visible to host CPU & cross-alias VAs.
    AGX_BARRIER_MEMORY_HOST = 3,
};

// 128-byte cache-line-aligned SLC-pinned state for persistent-megakernel device-scope
// sense-reversing barriers (formally verified in formal/AgxBarrier.tla).
struct alignas(AGX_CACHE_LINE_BYTES) agx_sense_barrier_state {
    uint32_t arrived;       // atomic counter [0..n_workgroups]
    uint32_t sense;         // toggling sense bit (0 or 1)
    uint32_t epoch;         // monotonic completed barrier count
    uint32_t n_workgroups;  // participating workgroup count (e.g., 40 on M4 Max)
    uint32_t stage;         // current Inkling stage (0=prep_qkv, 1=attn_out, 2=moe_down)
    uint32_t pass;          // current pass index [0..47]
    uint32_t reserved[26];
};
static_assert(sizeof(agx_sense_barrier_state) == AGX_CACHE_LINE_BYTES,
              "agx_sense_barrier_state must occupy exactly one 128B cache line");

// ============================================================================
// Forward Declarations & Buffer Context
// ============================================================================

struct agx_drm_device;
struct agx_bo;
struct agx_cdm_stream;
struct agx_compiler;
struct agx_kernel;
struct agx_slc_arena;
struct agx_backend_context;

// ============================================================================
// Wave 4: Static SLC-Pinned Activation Arena (Theorem W4.2, Wave4Megakernel.lean)
// ============================================================================
// Single lazily-allocated 8 MiB AGX_BO_SLC_PIN BO (< 17% of M4 Max's 48 MiB SLC;
// active working set 3.90 MiB for M=1 / 5.16 MiB for M=8) with fixed, strictly
// non-aliasing 128B-aligned GPU VAs and an isolated 128B sense-reversing barrier
// cache line (formally verified in formal/AgxMegakernelW4.tla).

#define AGX_SLC_ARENA_TOTAL_BYTES       (8ULL * 1024ULL * 1024ULL) // 8 MiB
#define AGX_SLC_ARENA_MAX_M             8u
#define AGX_SLC_ARENA_RES_BLOCKS        8u
#define AGX_SLC_ARENA_SCONV_SLOTS       48u
#define AGX_SLC_ARENA_SCONV_TAPS        3u
#define AGX_SLC_ARENA_SCONV_CHANNELS    6656u
#define AGX_SLC_ARENA_ATTN_IN_ROWS      8984u  // 3072+3072+768+1024+1024+24 (Theorem W4.1)

// Fixed 128B-aligned byte sizes per arena slot (for M <= 8)
#define AGX_SLC_ARENA_SENSE_BYTES       4096u      // 128B agx_sense_barrier_state in isolated 4 KiB guard
#define AGX_SLC_ARENA_H_CUR_BYTES       73728u     // 8 * 2304 * 4
#define AGX_SLC_ARENA_H_MSG_BYTES       294912u    // 4 * 8 * 2304 * 4 (up to 4 cross-pass msg slots)
#define AGX_SLC_ARENA_RES_BLOCK_BYTES   73728u     // 8 * 2304 * 4 per block (x8 = 589,824 B)
#define AGX_SLC_ARENA_QKV_BUF_BYTES     287488u    // 8 * 8984 * 4 (coalesced W_attn_in output)
#define AGX_SLC_ARENA_ATTN_OUT_BYTES    98304u     // 8 * 3072 * 4
#define AGX_SLC_ARENA_CTRL_BYTES        294912u    // 4 * 8 * 2304 * 4 (>= 8 * 3072 * 4)
#define AGX_SLC_ARENA_MOE_INTER_BYTES   294912u    // 8 * 8 * 1152 * 4 (M=8 x top_k=8 x n_ff=1152)
#define AGX_SLC_ARENA_ROUTER_TOPK_BYTES 65536u     // topk_ids (+0), topk_w (+4096), router scratch
#define AGX_SLC_ARENA_ROUTER_W_OFFSET   4096u      // byte offset of router_topk_w within router_topk
#define AGX_SLC_ARENA_SCONV_STATE_BYTES 3833856u   // 48 * 3 * 6656 * 4

// Fixed 128B-aligned byte offsets inside the 8 MiB SLC_PIN BO
#define AGX_SLC_ARENA_OFF_SENSE_BARRIER (0ULL)
#define AGX_SLC_ARENA_OFF_H_CUR         (AGX_SLC_ARENA_OFF_SENSE_BARRIER + AGX_SLC_ARENA_SENSE_BYTES)
#define AGX_SLC_ARENA_OFF_H_MSG         (AGX_SLC_ARENA_OFF_H_CUR + AGX_SLC_ARENA_H_CUR_BYTES)
#define AGX_SLC_ARENA_OFF_RES_BLOCKS    (AGX_SLC_ARENA_OFF_H_MSG + AGX_SLC_ARENA_H_MSG_BYTES)
#define AGX_SLC_ARENA_OFF_QKV_BUF       (AGX_SLC_ARENA_OFF_RES_BLOCKS + AGX_SLC_ARENA_RES_BLOCKS * AGX_SLC_ARENA_RES_BLOCK_BYTES)
#define AGX_SLC_ARENA_OFF_ATTN_OUT      (AGX_SLC_ARENA_OFF_QKV_BUF + AGX_SLC_ARENA_QKV_BUF_BYTES)
#define AGX_SLC_ARENA_OFF_CTRL_SCRATCH  (AGX_SLC_ARENA_OFF_ATTN_OUT + AGX_SLC_ARENA_ATTN_OUT_BYTES)
#define AGX_SLC_ARENA_OFF_MOE_INTER     (AGX_SLC_ARENA_OFF_CTRL_SCRATCH + AGX_SLC_ARENA_CTRL_BYTES)
#define AGX_SLC_ARENA_OFF_ROUTER_TOPK   (AGX_SLC_ARENA_OFF_MOE_INTER + AGX_SLC_ARENA_MOE_INTER_BYTES)
#define AGX_SLC_ARENA_OFF_SCONV_STATE   (AGX_SLC_ARENA_OFF_ROUTER_TOPK + AGX_SLC_ARENA_ROUTER_TOPK_BYTES)
#define AGX_SLC_ARENA_OFF_EXTRA_SCRATCH (AGX_SLC_ARENA_OFF_SCONV_STATE + AGX_SLC_ARENA_SCONV_STATE_BYTES)
#define AGX_SLC_ARENA_USED_BYTES        AGX_SLC_ARENA_OFF_EXTRA_SCRATCH

static_assert((AGX_SLC_ARENA_OFF_SENSE_BARRIER % AGX_CACHE_LINE_BYTES) == 0, "sense_barrier 128B align");
static_assert((AGX_SLC_ARENA_OFF_H_CUR         % AGX_CACHE_LINE_BYTES) == 0, "h_cur 128B align");
static_assert((AGX_SLC_ARENA_OFF_H_MSG         % AGX_CACHE_LINE_BYTES) == 0, "h_msg 128B align");
static_assert((AGX_SLC_ARENA_OFF_RES_BLOCKS    % AGX_CACHE_LINE_BYTES) == 0, "res_blocks 128B align");
static_assert((AGX_SLC_ARENA_RES_BLOCK_BYTES   % AGX_CACHE_LINE_BYTES) == 0, "res_block stride 128B align");
static_assert((AGX_SLC_ARENA_OFF_QKV_BUF       % AGX_CACHE_LINE_BYTES) == 0, "qkv_buf 128B align");
static_assert((AGX_SLC_ARENA_OFF_ATTN_OUT      % AGX_CACHE_LINE_BYTES) == 0, "attn_out 128B align");
static_assert((AGX_SLC_ARENA_OFF_CTRL_SCRATCH  % AGX_CACHE_LINE_BYTES) == 0, "ctrl_scratch 128B align");
static_assert((AGX_SLC_ARENA_OFF_MOE_INTER     % AGX_CACHE_LINE_BYTES) == 0, "moe_inter 128B align");
static_assert((AGX_SLC_ARENA_OFF_ROUTER_TOPK   % AGX_CACHE_LINE_BYTES) == 0, "router_topk 128B align");
static_assert((AGX_SLC_ARENA_ROUTER_W_OFFSET   % AGX_CACHE_LINE_BYTES) == 0, "router_topk_w 128B align");
static_assert((AGX_SLC_ARENA_OFF_SCONV_STATE   % AGX_CACHE_LINE_BYTES) == 0, "sconv_state 128B align");
static_assert((AGX_SLC_ARENA_OFF_EXTRA_SCRATCH % AGX_CACHE_LINE_BYTES) == 0, "extra_scratch 128B align");
static_assert(AGX_SLC_ARENA_USED_BYTES <= AGX_SLC_ARENA_TOTAL_BYTES,         "arena fits inside 8 MiB BO");
static_assert(AGX_SLC_ARENA_TOTAL_BYTES < (AGX_SLC_BYTES / 2ULL),            "arena strictly < 24 MiB (< 50% SLC)");

struct agx_slc_arena {
    agx_bo * bo          = nullptr;
    uint64_t base_va     = 0;
    void *   cpu_base    = nullptr;
    size_t   total_bytes = 0;       // AGX_SLC_ARENA_TOTAL_BYTES (8 MiB)
    size_t   used_bytes  = 0;       // AGX_SLC_ARENA_USED_BYTES (5,837,568 B)
    uint32_t max_m       = AGX_SLC_ARENA_MAX_M;

    // Fixed, non-aliasing, 128B-aligned GPU VAs inside `bo`
    uint64_t sense_barrier_va = 0;
    uint64_t h_cur            = 0;
    uint64_t h_msg            = 0;
    uint64_t res_blocks[AGX_SLC_ARENA_RES_BLOCKS] = {};
    uint64_t qkv_buf          = 0;
    uint64_t attn_out         = 0;
    uint64_t ctrl_scratch     = 0;
    uint64_t moe_inter        = 0;
    uint64_t router_topk      = 0;
    uint64_t router_topk_w    = 0;  // router_topk + AGX_SLC_ARENA_ROUTER_W_OFFSET
    uint64_t sconv_state      = 0;
    uint64_t extra_scratch    = 0;  // Remaining tail of 8 MiB BO (2,551,040 B)
    size_t   extra_scratch_bytes = 0;

    // Direct CPU-coherent pointer to the 128B sense-reversing barrier cache line
    agx_sense_barrier_state * sense_barrier_cpu = nullptr;
};

// Context stored in ggml_backend_buffer::context for AGX0 and AGX0_Repack buffers.
struct agx_buffer_ctx {
    agx_drm_device * dev;
    agx_bo *         bo;
    uint64_t         gpu_va;
    void *           cpu_ptr;
    size_t           size;
    agx_cache_policy policy;
    bool             is_repacked;
    bool             is_host_import;
};

// Resolve GPU VA of any ggml_tensor backed by an AGX0 / AGX0_Repack / AGX_Host buffer.
// Handles tensor views, slices, and non-zero byte offsets cleanly.
static inline uint64_t agx_tensor_gpu_va(const struct ggml_tensor * t) {
    if (!t || !t->buffer || !t->buffer->context || !t->data) {
        return 0;
    }
    const auto * bctx = static_cast<const agx_buffer_ctx *>(t->buffer->context);
    const uintptr_t base = reinterpret_cast<uintptr_t>(bctx->cpu_ptr);
    const uintptr_t ptr  = reinterpret_cast<uintptr_t>(t->data);
    if (ptr < base || (ptr - base) > bctx->size) {
        return 0;
    }
    return bctx->gpu_va + static_cast<uint64_t>(ptr - base);
}

static inline void * agx_tensor_cpu_ptr(const struct ggml_tensor * t) {
    return t ? t->data : nullptr;
}

static inline bool agx_tensor_is_repacked(const struct ggml_tensor * t) {
    if (!t || !t->buffer || !t->buffer->context) {
        return false;
    }
    const auto * bctx = static_cast<const agx_buffer_ctx *>(t->buffer->context);
    return bctx->is_repacked;
}

// ============================================================================
// Backend Context & Cross-Module Hooks (Pillar 2 / Pillar 3 / Pillar 4-5)
// ============================================================================

struct agx_backend_context {
    int              dev_num;
    agx_drm_device * drm;
    agx_compiler *   compiler;
    agx_cdm_stream * stream;
    agx_slc_arena *  slc_arena;         // lazily allocated 8 MiB AGX_BO_SLC_PIN arena (Wave 4)
    void *           graph_exec_state;  // owned by agx-graph-exec.cpp
    void *           kv_hybrid_state;   // owned by agx-kv-hybrid.cpp
    void *           moe_state;         // owned by agx-moe.cpp
    void *           gemm_state;        // owned by agx-gemm.cpp
    bool             debug;
    bool             profile;
    bool             last_replay_megakernel;
};

// Cross-module entrypoints implemented by Pillar 2 (`agx-repack.cpp`) and
// Pillar 3 (`agx-graph-exec.cpp` / `agx-inkling2.cpp`). `ggml-agx.cpp` provides
// weak fallback stubs so Pillar 1/4/6 builds and runs standalone before all
// modules are linked.
extern "C" {

// Pillar 2 (`agx-repack.cpp`): 128B cache-line-aligned super-tile repacking
bool   agx_repack_supports_tensor(const struct ggml_tensor * tensor);
size_t agx_repack_get_alloc_size(const struct ggml_tensor * tensor);
void   agx_repack_set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void   agx_repack_get_tensor(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

// Pillar 3 (`agx-graph-exec.cpp` / `agx-inkling2.cpp`): Graph execution, fusion & Inkling 2 megakernel dispatch
bool             agx_supports_op(const agx_drm_device * dev, const struct ggml_tensor * op);
bool             agx_inkling2_try_execute(agx_backend_context * ctx, struct ggml_cgraph * cgraph);
enum ggml_status agx_graph_compute(agx_backend_context * ctx, struct ggml_cgraph * cgraph);
void             agx_graph_optimize(agx_backend_context * ctx, struct ggml_cgraph * cgraph,
                                    struct ggml_backend_graph_optimize_params * params);
void             agx_graph_exec_cleanup(agx_backend_context * ctx);

} // extern "C"
