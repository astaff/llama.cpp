#include "ggml-agx.h"
#include "agx-common.h"
#include "agx-drm.h"
#include "agx-cdm.h"
#include "agx-compiler.h"

#include "ggml-backend-impl.h"
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "ggml-impl.h"
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// ============================================================================
// Weak Fallback Stubs for Pillar 2 (agx-repack.cpp) & Pillar 3 (agx-graph-exec.cpp)
// ============================================================================
// Strong definitions in agx-repack.cpp and agx-graph-exec.cpp automatically
// override these when linked into libggml-agx.

#if defined(__GNUC__) || defined(__clang__)
#    define AGX_WEAK __attribute__((weak))
#else
#    define AGX_WEAK
#endif

extern "C" {

AGX_WEAK bool agx_repack_supports_tensor(const struct ggml_tensor * tensor) {
    (void) tensor;
    return false;
}

AGX_WEAK size_t agx_repack_get_alloc_size(const struct ggml_tensor * tensor) {
    return tensor ? ggml_nbytes(tensor) : 0;
}

AGX_WEAK void agx_repack_set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (tensor && tensor->data && data && size > 0) {
        std::memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
    }
}

AGX_WEAK void agx_repack_get_tensor(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (tensor && tensor->data && data && size > 0) {
        std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
    }
}

AGX_WEAK bool agx_supports_op(const agx_drm_device * dev, const struct ggml_tensor * op) {
    (void) dev;
    (void) op;
    return false;
}

AGX_WEAK bool agx_inkling2_try_execute(agx_backend_context * ctx, struct ggml_cgraph * cgraph) {
    (void) ctx;
    (void) cgraph;
    return false;
}

AGX_WEAK enum ggml_status agx_graph_compute(agx_backend_context * ctx, struct ggml_cgraph * cgraph) {
    (void) ctx;
    (void) cgraph;
    return GGML_STATUS_FAILED;
}

AGX_WEAK void agx_graph_optimize(agx_backend_context * ctx, struct ggml_cgraph * cgraph,
                                 struct ggml_backend_graph_optimize_params * params) {
    (void) ctx;
    (void) cgraph;
    (void) params;
}

AGX_WEAK void agx_graph_exec_cleanup(agx_backend_context * ctx) {
    (void) ctx;
}

} // extern "C"

// ============================================================================
// Shared Device Singleton (One DRM VM shared across all AGX0 buffers & streams)
// ============================================================================

namespace {

// Threshold above which single allocations default to cold streaming VA/WC policy
// (e.g. 14.7 GB MoE expert blobs) to keep the 48 MB SLC free for hot working sets.
constexpr size_t kAgxSlcPinMaxBufferBytes    = 256ULL * 1024ULL * 1024ULL;
constexpr size_t kAgxRepackSlcPinMaxBytes    = AGX_SLC_BYTES; // 48 MiB

struct agx_device_state {
    int                    dev_num = 0;
    std::mutex             mtx;
    agx_drm_device *       drm     = nullptr;
    agx_compiler *         compiler = nullptr;
    std::string            name;
    std::string            description;
    agx_drm_device_info    info{};
    bool                   probed  = false;
    bool                   available = false;

    ggml_backend_device      dev{};
    ggml_backend_buffer_type buft_default{};
    ggml_backend_buffer_type buft_repack{};
    ggml_backend_buffer_type buft_host{};
    ggml_backend_buffer_type_t extra_bufts[2] = { nullptr, nullptr };
};

static agx_device_state * g_agx_dev0 = nullptr;
static std::mutex         g_agx_global_mtx;

static agx_drm_device * agx_dev_ensure_open(agx_device_state * st) {
    if (!st) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(st->mtx);
    if (st->drm) {
        return st->drm;
    }
    st->drm = agx_drm_open(st->dev_num);
    if (st->drm) {
        st->info = agx_drm_get_info(st->drm);
        st->available = true;
    }
    return st->drm;
}

static agx_compiler * agx_dev_ensure_compiler(agx_device_state * st) {
    agx_drm_device * drm = agx_dev_ensure_open(st);
    if (!drm) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(st->mtx);
    if (!st->compiler) {
        const char * kdir = std::getenv("GGML_AGX_KERNEL_DIR");
#ifdef GGML_AGX_KERNEL_DIR
        if (!kdir || !kdir[0]) {
            kdir = GGML_AGX_KERNEL_DIR;
        }
#endif
        const char * cdir = std::getenv("GGML_AGX_CACHE_DIR");
        st->compiler = agx_compiler_create(drm, kdir, cdir);
    }
    return st->compiler;
}

static ggml_guid_t agx_backend_guid() {
    static ggml_guid guid = {
        0x41, 0x47, 0x58, 0x33, 0x42, 0x2d, 0x44, 0x52,
        0x4d, 0x2d, 0x43, 0x44, 0x4d, 0x30, 0x30, 0x31
    };
    return &guid;
}

// ============================================================================
// Buffer Interface (AGX0 / AGX0_Repack / AGX_Host)
// ============================================================================

static void agx_buf_free(ggml_backend_buffer_t buffer) {
    auto * bctx = static_cast<agx_buffer_ctx *>(buffer->context);
    if (bctx) {
        if (bctx->bo) {
            agx_bo_free(bctx->dev, bctx->bo);
        }
        delete bctx;
        buffer->context = nullptr;
    }
}

static void * agx_buf_get_base(ggml_backend_buffer_t buffer) {
    auto * bctx = static_cast<agx_buffer_ctx *>(buffer->context);
    return bctx ? bctx->cpu_ptr : nullptr;
}

static enum ggml_status agx_buf_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    (void) buffer;
    (void) tensor;
    return GGML_STATUS_SUCCESS;
}

static void agx_buf_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                  uint8_t value, size_t offset, size_t size) {
    (void) buffer;
    if (tensor && tensor->data && size > 0) {
        std::memset(static_cast<uint8_t *>(tensor->data) + offset, value, size);
    }
}

static void agx_buf_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                               const void * data, size_t offset, size_t size) {
    auto * bctx = static_cast<agx_buffer_ctx *>(buffer->context);
    if (!tensor || !tensor->data || !data || size == 0) {
        return;
    }
    if (bctx && bctx->is_repacked && agx_repack_supports_tensor(tensor)) {
        agx_repack_set_tensor(tensor, data, offset, size);
        return;
    }
    std::memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
}

static void agx_buf_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                               void * data, size_t offset, size_t size) {
    auto * bctx = static_cast<agx_buffer_ctx *>(buffer->context);
    if (!tensor || !tensor->data || !data || size == 0) {
        return;
    }
    if (bctx && bctx->is_repacked && agx_repack_supports_tensor(tensor)) {
        agx_repack_get_tensor(tensor, data, offset, size);
        return;
    }
    std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
}

static bool agx_buf_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    auto * dst_ctx = static_cast<agx_buffer_ctx *>(buffer->context);
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }
    const bool dst_repacked = dst_ctx && dst_ctx->is_repacked && agx_repack_supports_tensor(dst);
    const bool src_repacked = agx_tensor_is_repacked(src) && agx_repack_supports_tensor(src);

    if (dst_repacked && !src_repacked) {
        if (ggml_backend_buffer_is_host(src->buffer) || agx_tensor_gpu_va(src) != 0) {
            agx_repack_set_tensor(dst, src->data, 0, ggml_nbytes(src));
            return true;
        }
        return false;
    }
    if (!dst_repacked && src_repacked) {
        if (ggml_backend_buffer_is_host(dst->buffer) || agx_tensor_gpu_va(dst) != 0) {
            agx_repack_get_tensor(src, dst->data, 0, ggml_nbytes(dst));
            return true;
        }
        return false;
    }
    if (dst_repacked && src_repacked) {
        const size_t nbytes = agx_repack_get_alloc_size(dst);
        if (nbytes == agx_repack_get_alloc_size(src)) {
            std::memcpy(dst->data, src->data, nbytes);
            return true;
        }
        return false;
    }
    // Unified memory: both src and dst are CPU-mapped coherent BOs or host memory.
    if (ggml_backend_buffer_is_host(src->buffer) || agx_tensor_gpu_va(src) != 0) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void agx_buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * bctx = static_cast<agx_buffer_ctx *>(buffer->context);
    if (bctx && bctx->cpu_ptr && bctx->size > 0) {
        std::memset(bctx->cpu_ptr, value, bctx->size);
    }
}

static ggml_backend_buffer_i g_agx_buf_iface = {
    /* .free_buffer   = */ agx_buf_free,
    /* .get_base      = */ agx_buf_get_base,
    /* .init_tensor   = */ agx_buf_init_tensor,
    /* .memset_tensor = */ agx_buf_memset_tensor,
    /* .set_tensor    = */ agx_buf_set_tensor,
    /* .get_tensor    = */ agx_buf_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ agx_buf_cpy_tensor,
    /* .clear         = */ agx_buf_clear,
    /* .reset         = */ nullptr,
    /* .set_tensor_n  = */ nullptr,
    /* .get_tensor_n  = */ nullptr,
    /* .cpy_tensor_n  = */ nullptr,
};

// ============================================================================
// Buffer Types: AGX0 (SLC-partitioned), AGX0_Repack (128B super-tile), AGX_Host
// ============================================================================

struct agx_buft_ctx {
    agx_device_state * dev_state;
    bool               is_repack;
    bool               is_host;
};

static const char * agx_buft_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<agx_buft_ctx *>(buft->context);
    if (ctx->is_host)   return "AGX_Host";
    if (ctx->is_repack) return "AGX0_Repack";
    return "AGX0";
}

static ggml_backend_buffer_t agx_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = static_cast<agx_buft_ctx *>(buft->context);
    agx_drm_device * drm = agx_dev_ensure_open(ctx->dev_state);
    if (!drm) {
        AGX_LOG_ERROR("alloc_buffer: failed to open AGX DRM device");
        return nullptr;
    }

    // Zero-size probe buffers (used by llama-model-loader::weight_buft_supported and
    // llama-model::buft_supported) carry only buft/repack metadata with zero DRM ioctls.
    if (size == 0) {
        auto * bctx = new agx_buffer_ctx();
        bctx->dev            = drm;
        bctx->bo             = nullptr;
        bctx->gpu_va         = 0;
        bctx->cpu_ptr        = nullptr;
        bctx->size           = 0;
        bctx->policy         = ctx->is_repack ? AGX_CACHE_SLC_STREAM : AGX_CACHE_SLC_PIN;
        bctx->is_repacked    = ctx->is_repack;
        bctx->is_host_import = false;
        return ggml_backend_buffer_init(buft, g_agx_buf_iface, bctx, 0);
    }

    const size_t alloc_size = std::max<size_t>(size, AGX_CACHE_LINE_BYTES);

    // Pillar 4: Partition BO/VM cache policy between 48 MB SLC-pinned hot working sets
    // and cold streaming weight/KV allocations.
    uint32_t bo_flags = 0;
    agx_cache_policy policy = AGX_CACHE_SLC_PIN;
    if (ctx->is_host) {
        policy   = AGX_CACHE_SLC_PIN;
        bo_flags = AGX_BO_SLC_PIN | AGX_BO_WRITEBACK;
    } else if (ctx->is_repack) {
        if (alloc_size > kAgxRepackSlcPinMaxBytes) {
            policy   = AGX_CACHE_SLC_STREAM;
            bo_flags = AGX_BO_SLC_STREAM | AGX_BO_READONLY;
        } else {
            policy   = AGX_CACHE_SLC_PIN;
            bo_flags = AGX_BO_SLC_PIN | AGX_BO_WRITEBACK | AGX_BO_READONLY;
        }
    } else {
        if (alloc_size > kAgxSlcPinMaxBufferBytes) {
            policy   = AGX_CACHE_SLC_STREAM;
            bo_flags = AGX_BO_SLC_STREAM;
        } else {
            policy   = AGX_CACHE_SLC_PIN;
            bo_flags = AGX_BO_SLC_PIN | AGX_BO_WRITEBACK;
        }
    }

    const char * label = ctx->is_host ? "agx_host" : (ctx->is_repack ? "agx_repack" : "agx_buf");
    agx_bo * bo = agx_bo_alloc(drm, alloc_size, bo_flags, label);
    if (!bo) {
        AGX_LOG_ERROR("alloc_buffer(%s, %zu bytes, policy=%u) failed", label, alloc_size, policy);
        return nullptr;
    }

    auto * bctx = new agx_buffer_ctx();
    bctx->dev            = drm;
    bctx->bo             = bo;
    bctx->gpu_va         = bo->gpu_va;
    bctx->cpu_ptr        = bo->cpu_ptr;
    bctx->size           = bo->size;
    bctx->policy         = policy;
    bctx->is_repacked    = ctx->is_repack;
    bctx->is_host_import = false;

    return ggml_backend_buffer_init(buft, g_agx_buf_iface, bctx, size);
}

static size_t agx_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void) buft;
    // Always 128-byte cache-line aligned for AGX3 SLC & super-tile loads.
    return AGX_CACHE_LINE_BYTES;
}

static size_t agx_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<agx_buft_ctx *>(buft->context);
    constexpr size_t kMaxBoBytes = 4ULL * 1024ULL * 1024ULL * 1024ULL; // 4 GiB per DRM BO
    if (ctx && ctx->dev_state && ctx->dev_state->info.client_memory_limit > 0) {
        return std::min<size_t>(ctx->dev_state->info.client_memory_limit, kMaxBoBytes);
    }
    return kMaxBoBytes;
}

static size_t agx_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    auto * ctx = static_cast<agx_buft_ctx *>(buft->context);
    if (ctx && ctx->is_repack && agx_repack_supports_tensor(tensor)) {
        return agx_align_up(agx_repack_get_alloc_size(tensor), AGX_CACHE_LINE_BYTES);
    }
    return ggml_nbytes(tensor);
}

static bool agx_buft_is_host(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<agx_buft_ctx *>(buft->context);
    return ctx ? ctx->is_host : false;
}

static ggml_backend_buffer_type_i g_agx_buft_iface = {
    /* .get_name         = */ agx_buft_get_name,
    /* .alloc_buffer     = */ agx_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ agx_buft_get_alignment,
    /* .get_max_size     = */ agx_buft_get_max_size,
    /* .get_alloc_size   = */ agx_buft_get_alloc_size,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ agx_buft_is_host,
};

// ============================================================================
// Backend Stream Interface (ggml_backend_i)
// ============================================================================

static const char * agx_backend_get_name(ggml_backend_t backend) {
    (void) backend;
    return "AGX0";
}

static void agx_backend_free(ggml_backend_t backend) {
    auto * ctx = static_cast<agx_backend_context *>(backend->context);
    if (ctx) {
        if (ctx->stream) {
            agx_cdm_wait(ctx->stream, nullptr);
        }
        agx_graph_exec_cleanup(ctx);
        if (ctx->slc_arena) {
            agx_slc_arena_destroy(ctx->drm, ctx->slc_arena);
            ctx->slc_arena = nullptr;
        }
        if (ctx->stream) {
            agx_cdm_stream_destroy(ctx->stream);
            ctx->stream = nullptr;
        }
        delete ctx;
    }
    delete backend;
}

static void agx_backend_synchronize(ggml_backend_t backend) {
    auto * ctx = static_cast<agx_backend_context *>(backend->context);
    if (ctx && ctx->stream) {
        agx_cdm_wait(ctx->stream, nullptr);
    }
}

static void agx_backend_set_tensor_async(ggml_backend_t backend, struct ggml_tensor * tensor,
                                         const void * data, size_t offset, size_t size) {
    agx_backend_synchronize(backend);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

static void agx_backend_get_tensor_async(ggml_backend_t backend, const struct ggml_tensor * tensor,
                                         void * data, size_t offset, size_t size) {
    agx_backend_synchronize(backend);
    ggml_backend_tensor_get(tensor, data, offset, size);
}

static bool agx_backend_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst,
                                         const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_is_agx(backend_src)) {
        agx_backend_synchronize(backend_src);
    }
    if (ggml_backend_is_agx(backend_dst) && backend_dst != backend_src) {
        agx_backend_synchronize(backend_dst);
    }
    return agx_buf_cpy_tensor(dst->buffer, src, dst);
}

static bool agx_megakernel_enabled() {
    const char * v = std::getenv("GGML_AGX_NO_MEGAKERNEL");
    return !(v && v[0] && v[0] != '0');
}

static enum ggml_status agx_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<agx_backend_context *>(backend->context);
    if (!ctx || !ctx->drm || !ctx->stream) {
        return GGML_STATUS_FAILED;
    }
    if (!cgraph || cgraph->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    const bool use_megakernel = agx_megakernel_enabled();
    if (ctx->last_replay_megakernel != use_megakernel) {
        agx_cdm_stream_invalidate_replay(ctx->stream);
    }

    const bool can_replay = !ctx->debug && !ctx->profile &&
                            cgraph->uid != 0 &&
                            agx_cdm_replay_enabled();
    if (can_replay) {
        const void * first_node = cgraph->nodes[0];
        const void * last_node  = cgraph->nodes[cgraph->n_nodes - 1];
        const uint64_t bo_epoch = agx_drm_get_bo_epoch(ctx->drm);
        if (ctx->slc_arena) {
            agx_slc_arena_prepare_sense_barrier_replay(ctx->slc_arena);
        }
        int replay_ret = 0;
        if (agx_cdm_stream_try_replay(ctx->stream, cgraph->uid, cgraph->n_nodes,
                                      first_node, last_node, bo_epoch,
                                      /*wait=*/true, /*out_gpu_ns=*/nullptr, &replay_ret)) {
            if (replay_ret != 0) {
                AGX_LOG_ERROR("agx_backend_graph_compute: CDM replay submit failed (%d)", replay_ret);
                return GGML_STATUS_FAILED;
            }
            return GGML_STATUS_SUCCESS;
        }
    }

    const uint64_t submits_before = agx_cdm_stream_submit_count(ctx->stream);
    enum ggml_status st = GGML_STATUS_FAILED;
    bool handled = false;
    if (use_megakernel) {
        if (agx_inkling2_try_execute(ctx, cgraph)) {
            handled = true;
            st = (agx_drm_vm_error(ctx->drm) == 0) ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
        } else if (agx_drm_vm_error(ctx->drm) != 0) {
            handled = true;
            st = GGML_STATUS_FAILED;
        }
    }
    if (!handled) {
        st = agx_graph_compute(ctx, cgraph);
    }

    if (st == GGML_STATUS_SUCCESS && can_replay) {
        const uint64_t submits_after = agx_cdm_stream_submit_count(ctx->stream);
        if (submits_after == submits_before + 1ULL &&
            agx_cdm_stream_is_replayable(ctx->stream) &&
            agx_drm_vm_error(ctx->drm) == 0) {
            const uint64_t bo_epoch_after = agx_drm_get_bo_epoch(ctx->drm);
            agx_cdm_stream_tag_replay(ctx->stream, cgraph->uid, cgraph->n_nodes,
                                      cgraph->nodes[0], cgraph->nodes[cgraph->n_nodes - 1],
                                      bo_epoch_after);
            ctx->last_replay_megakernel = use_megakernel;
        } else {
            agx_cdm_stream_invalidate_replay(ctx->stream);
        }
    } else {
        agx_cdm_stream_invalidate_replay(ctx->stream);
    }
    return st;
}

static void agx_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph,
                                       struct ggml_backend_graph_optimize_params * params) {
    auto * ctx = static_cast<agx_backend_context *>(backend->context);
    if (ctx) {
        if (ctx->stream) {
            agx_cdm_stream_invalidate_replay(ctx->stream);
        }
        agx_graph_optimize(ctx, cgraph, params);
    }
}

static void agx_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    agx_backend_synchronize(backend);
    if (event && event->context) {
        auto * done = static_cast<std::atomic<bool> *>(event->context);
        done->store(true, std::memory_order_release);
    }
}

static void agx_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    (void) backend;
    (void) event;
}

static ggml_backend_i g_agx_backend_iface = {
    /* .get_name            = */ agx_backend_get_name,
    /* .free                = */ agx_backend_free,
    /* .set_tensor_async    = */ agx_backend_set_tensor_async,
    /* .get_tensor_async    = */ agx_backend_get_tensor_async,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ agx_backend_cpy_tensor_async,
    /* .synchronize         = */ agx_backend_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ agx_backend_graph_compute,
    /* .event_record        = */ agx_backend_event_record,
    /* .event_wait          = */ agx_backend_event_wait,
    /* .graph_optimize      = */ agx_backend_graph_optimize,
};

// ============================================================================
// Device Interface (ggml_backend_device_i)
// ============================================================================

static const char * agx_dev_get_name(ggml_backend_dev_t dev) {
    auto * st = static_cast<agx_device_state *>(dev->context);
    return st->name.c_str();
}

static const char * agx_dev_get_description(ggml_backend_dev_t dev) {
    auto * st = static_cast<agx_device_state *>(dev->context);
    return st->description.c_str();
}

static void agx_dev_get_memory(ggml_backend_dev_t dev, size_t * free_bytes, size_t * total_bytes) {
    auto * st = static_cast<agx_device_state *>(dev->context);
    agx_drm_device * drm = agx_dev_ensure_open(st);
    if (drm) {
        agx_drm_memory_stats mstats{};
        agx_drm_get_memory_stats(drm, &mstats);
        if (free_bytes)  *free_bytes  = static_cast<size_t>(mstats.free_estimate_bytes);
        if (total_bytes) *total_bytes = static_cast<size_t>(mstats.memory_limit_bytes);
        return;
    }
    if (free_bytes)  *free_bytes  = static_cast<size_t>(st->info.client_memory_limit);
    if (total_bytes) *total_bytes = static_cast<size_t>(st->info.client_memory_limit);
}

static enum ggml_backend_dev_type agx_dev_get_type(ggml_backend_dev_t dev) {
    (void) dev;
    // Matches ggml-metal so llama.cpp selects AGX0 as primary GPU on M4 Max.
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void agx_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = agx_dev_get_name(dev);
    props->description = agx_dev_get_description(dev);
    props->type        = agx_dev_get_type(dev);
    agx_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {};
    props->caps.async                = true;
    props->caps.host_buffer          = true;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events               = true;
}

static ggml_backend_t agx_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void) params;
    auto * st = static_cast<agx_device_state *>(dev->context);
    agx_drm_device * drm = agx_dev_ensure_open(st);
    if (!drm) {
        AGX_LOG_ERROR("init_backend: failed to open drm_asahi device %d", st->dev_num);
        return nullptr;
    }
    agx_compiler * compiler = agx_dev_ensure_compiler(st);
    agx_cdm_stream * stream = agx_cdm_stream_create(drm);
    if (!stream) {
        AGX_LOG_ERROR("init_backend: failed to create CDM command stream");
        return nullptr;
    }

    auto * ctx = new agx_backend_context();
    std::memset(ctx, 0, sizeof(*ctx));
    ctx->dev_num  = st->dev_num;
    ctx->drm      = drm;
    ctx->compiler = compiler;
    ctx->stream   = stream;
#ifdef GGML_AGX_DEBUG
    ctx->debug    = true;
#else
    ctx->debug    = (std::getenv("GGML_AGX_DEBUG") != nullptr);
#endif
    ctx->profile  = (std::getenv("GGML_AGX_PROFILE") != nullptr);

    auto * backend = new ggml_backend();
    backend->guid    = agx_backend_guid();
    backend->iface   = g_agx_backend_iface;
    backend->device  = dev;
    backend->context = ctx;
    return backend;
}

static ggml_backend_buffer_type_t agx_dev_get_buffer_type(ggml_backend_dev_t dev) {
    auto * st = static_cast<agx_device_state *>(dev->context);
    return &st->buft_default;
}

static ggml_backend_buffer_type_t agx_dev_get_host_buffer_type(ggml_backend_dev_t dev) {
    auto * st = static_cast<agx_device_state *>(dev->context);
    return &st->buft_host;
}

static ggml_backend_buffer_t agx_dev_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr,
                                                          size_t size, size_t max_tensor_size) {
    (void) max_tensor_size;
    auto * st = static_cast<agx_device_state *>(dev->context);
    agx_drm_device * drm = agx_dev_ensure_open(st);
    if (!drm || !ptr || size == 0) {
        return nullptr;
    }
    agx_bo * bo = agx_bo_import_userptr(drm, ptr, size, true, "agx_userptr");
    if (!bo) {
        return nullptr;
    }
    auto * bctx = new agx_buffer_ctx();
    bctx->dev            = drm;
    bctx->bo             = bo;
    bctx->gpu_va         = bo->gpu_va;
    bctx->cpu_ptr        = ptr;
    bctx->size           = size;
    bctx->policy         = AGX_CACHE_SLC_STREAM;
    bctx->is_repacked    = false;
    bctx->is_host_import = true;
    return ggml_backend_buffer_init(&st->buft_default, g_agx_buf_iface, bctx, size);
}

static bool agx_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    auto * st = static_cast<agx_device_state *>(dev->context);
    agx_drm_device * drm = agx_dev_ensure_open(st);
    if (!drm || !op) {
        return false;
    }
    if (!agx_supports_op(drm, op)) {
        return false;
    }

    // During model weight placement (`llama-model-loader.cpp:weight_buft_supported`),
    // `make_gpu_buft_list` probes `AGX0` (default) before `AGX0_Repack` (extra_bufts[0])
    // using a zero-size dummy buffer (`ggml_backend_buft_alloc_buffer(buft, 0)` short-circuits
    // with `buf->size == 0` and `buf->context == nullptr`). For repackable NVFP4/Q8_0
    // weight matrices used by MUL_MAT / MUL_MAT_ID, decline the default probe so
    // `select_weight_buft` advances directly to `AGX0_Repack` (128B super-tiles + Pillar 4
    // SLC streaming). Unallocated `test-backend-ops` probes (`w->buffer == nullptr`) and
    // real allocated buffers (`buf->size > 0 && bctx->bo != nullptr`) are unaffected.
    if ((op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) &&
        op->src[0] && op->src[0]->buffer &&
        std::getenv("GGML_AGX_NO_REPACK") == nullptr) {
        const ggml_backend_buffer_t buf = op->src[0]->buffer;
        const auto * bctx = static_cast<const agx_buffer_ctx *>(buf->context);
        const bool is_zero_probe = (buf->size == 0 && (bctx == nullptr || bctx->bo == nullptr));
        if (is_zero_probe &&
            buf->buft == &st->buft_default &&
            agx_repack_supports_tensor(op->src[0])) {
            return false;
        }
    }

    return true;
}

static bool agx_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!dev || !buft) {
        return false;
    }
    auto * st = static_cast<agx_device_state *>(dev->context);
    return buft == &st->buft_default ||
           buft == &st->buft_repack  ||
           buft == &st->buft_host;
}

static bool agx_dev_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (!op) {
        return false;
    }
    // Offload expensive matrix/attention/MoE ops to AGX0.
    const int min_batch_size = 1;
    return (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_FLASH_ATTN_EXT) &&
           ggml_nelements(op) >= min_batch_size &&
           agx_dev_supports_op(dev, op);
}

static ggml_backend_event_t agx_dev_event_new(ggml_backend_dev_t dev) {
    auto * ev = new ggml_backend_event();
    ev->device  = dev;
    ev->context = new std::atomic<bool>(false);
    return ev;
}

static void agx_dev_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    (void) dev;
    if (event) {
        delete static_cast<std::atomic<bool> *>(event->context);
        delete event;
    }
}

static void agx_dev_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    (void) dev;
    (void) event;
}

static ggml_backend_device_i g_agx_dev_iface = {
    /* .get_name             = */ agx_dev_get_name,
    /* .get_description      = */ agx_dev_get_description,
    /* .get_memory           = */ agx_dev_get_memory,
    /* .get_type             = */ agx_dev_get_type,
    /* .get_props            = */ agx_dev_get_props,
    /* .init_backend         = */ agx_dev_init_backend,
    /* .get_buffer_type      = */ agx_dev_get_buffer_type,
    /* .get_host_buffer_type = */ agx_dev_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ agx_dev_buffer_from_host_ptr,
    /* .supports_op          = */ agx_dev_supports_op,
    /* .supports_buft        = */ agx_dev_supports_buft,
    /* .offload_op           = */ agx_dev_offload_op,
    /* .event_new            = */ agx_dev_event_new,
    /* .event_free           = */ agx_dev_event_free,
    /* .event_synchronize    = */ agx_dev_event_synchronize,
};

// ============================================================================
// Backend Registry (ggml_backend_reg_i)
// ============================================================================

static agx_device_state * agx_get_or_init_dev0(ggml_backend_reg_t reg);

static const char * agx_reg_get_name(ggml_backend_reg_t reg) {
    (void) reg;
    return "AGX";
}

static size_t agx_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * st = agx_get_or_init_dev0(reg);
    return (st && st->available) ? 1u : 0u;
}

static ggml_backend_dev_t agx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * st = agx_get_or_init_dev0(reg);
    if (index == 0 && st && st->available) {
        return &st->dev;
    }
    return nullptr;
}

static ggml_backend_buffer_type_t * agx_dev_get_extra_bufts(ggml_backend_dev_t dev) {
    if (!dev || !dev->context) {
        return nullptr;
    }
    auto * st = static_cast<agx_device_state *>(dev->context);
    return st->extra_bufts;
}

static ggml_backend_feature * agx_reg_get_features(ggml_backend_reg_t reg) {
    (void) reg;
    static ggml_backend_feature features[] = {
        { "DRM_ASAHI",   "1" },
        { "AGX3_HAL200", "1" },
        { "SLC_POLICY",  "1" },
        { "REPACK_128B", "1" },
        { nullptr,       nullptr },
    };
    return features;
}

static void * agx_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    (void) reg;
    if (!name) {
        return nullptr;
    }
    if (std::strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        ggml_backend_dev_get_extra_bufts_t fn = agx_dev_get_extra_bufts;
        void * p = nullptr;
        static_assert(sizeof(p) == sizeof(fn), "function pointer size mismatch");
        std::memcpy(&p, &fn, sizeof(p));
        return p;
    }
    if (std::strcmp(name, "ggml_backend_get_features") == 0) {
        ggml_backend_get_features_t fn = agx_reg_get_features;
        void * p = nullptr;
        static_assert(sizeof(p) == sizeof(fn), "function pointer size mismatch");
        std::memcpy(&p, &fn, sizeof(p));
        return p;
    }
    return nullptr;
}

static ggml_backend_reg_i g_agx_reg_iface = {
    /* .get_name         = */ agx_reg_get_name,
    /* .get_device_count = */ agx_reg_get_device_count,
    /* .get_device       = */ agx_reg_get_device,
    /* .get_proc_address = */ agx_reg_get_proc_address,
};

static ggml_backend_reg g_agx_reg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ g_agx_reg_iface,
    /* .context     = */ nullptr,
};

static agx_device_state * agx_get_or_init_dev0(ggml_backend_reg_t reg) {
    std::lock_guard<std::mutex> lock(g_agx_global_mtx);
    if (g_agx_dev0) {
        return g_agx_dev0;
    }
    auto * st = new agx_device_state();
    st->dev_num = 0;
    st->name    = "AGX0";
    st->probed  = true;

    const int n = agx_drm_get_device_count();
    if (n > 0) {
        // Open device once so description & memory are accurate immediately.
        st->drm = agx_drm_open(0);
        if (st->drm) {
            st->info      = agx_drm_get_info(st->drm);
            st->available = true;
            char desc[256];
            std::snprintf(desc, sizeof(desc),
                          "Apple AGX3 (chip 0x%x, %u cores, HAL%u, %u MHz, direct drm_asahi)",
                          st->info.chip_id, st->info.num_cores, st->info.gpu_hal_generation,
                          st->info.max_frequency_khz / 1000u);
            st->description = desc;
        }
    }
    if (!st->available) {
        st->description = "Apple AGX3 (unavailable)";
    }

    st->dev.iface   = g_agx_dev_iface;
    st->dev.reg     = reg ? reg : &g_agx_reg;
    st->dev.context = st;

    auto * c_def = new agx_buft_ctx{ st, false, false };
    auto * c_rep = new agx_buft_ctx{ st, true,  false };
    auto * c_hst = new agx_buft_ctx{ st, false, true  };

    st->buft_default = { g_agx_buft_iface, &st->dev, c_def };
    st->buft_repack  = { g_agx_buft_iface, &st->dev, c_rep };
    st->buft_host    = { g_agx_buft_iface, &st->dev, c_hst };

    st->extra_bufts[0] = &st->buft_repack;
    st->extra_bufts[1] = nullptr;

    g_agx_dev0 = st;
    return g_agx_dev0;
}

#ifdef GGML_BACKEND_DL
static int agx_backend_score() {
    if (std::getenv("GGML_DISABLE_AGX") != nullptr) {
        return 0;
    }
    return agx_drm_get_device_count() > 0 ? 100 : 0;
}
#endif

} // namespace

// ============================================================================
// Public C API (ggml/include/ggml-agx.h)
// ============================================================================

extern "C" {

ggml_backend_t ggml_backend_agx_init(size_t dev_num) {
    if (dev_num != 0) {
        return nullptr;
    }
    auto * st = agx_get_or_init_dev0(&g_agx_reg);
    if (!st || !st->available) {
        return nullptr;
    }
    return agx_dev_init_backend(&st->dev, nullptr);
}

bool ggml_backend_is_agx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, agx_backend_guid());
}

ggml_backend_buffer_type_t ggml_backend_agx_buffer_type(size_t dev_num) {
    if (dev_num != 0) {
        return nullptr;
    }
    auto * st = agx_get_or_init_dev0(&g_agx_reg);
    return (st && st->available) ? &st->buft_default : nullptr;
}

ggml_backend_buffer_type_t ggml_backend_agx_repack_buffer_type(size_t dev_num) {
    if (dev_num != 0) {
        return nullptr;
    }
    auto * st = agx_get_or_init_dev0(&g_agx_reg);
    return (st && st->available) ? &st->buft_repack : nullptr;
}

ggml_backend_buffer_type_t ggml_backend_agx_host_buffer_type(void) {
    auto * st = agx_get_or_init_dev0(&g_agx_reg);
    return (st && st->available) ? &st->buft_host : nullptr;
}

int ggml_backend_agx_get_device_count(void) {
    return agx_drm_get_device_count();
}

void ggml_backend_agx_get_device_description(int device, char * description, size_t description_size) {
    if (!description || description_size == 0) {
        return;
    }
    if (device != 0) {
        description[0] = '\0';
        return;
    }
    auto * st = agx_get_or_init_dev0(&g_agx_reg);
    std::snprintf(description, description_size, "%s", st ? st->description.c_str() : "");
}

void ggml_backend_agx_get_device_memory(int device, size_t * free_bytes, size_t * total_bytes) {
    if (device != 0) {
        if (free_bytes)  *free_bytes  = 0;
        if (total_bytes) *total_bytes = 0;
        return;
    }
    auto * st = agx_get_or_init_dev0(&g_agx_reg);
    if (st && st->available) {
        agx_dev_get_memory(&st->dev, free_bytes, total_bytes);
    } else {
        if (free_bytes)  *free_bytes  = 0;
        if (total_bytes) *total_bytes = 0;
    }
}

ggml_backend_reg_t ggml_backend_agx_reg(void) {
    agx_get_or_init_dev0(&g_agx_reg);
    return &g_agx_reg;
}

} // extern "C"

GGML_BACKEND_DL_IMPL(ggml_backend_agx_reg)
GGML_BACKEND_DL_SCORE_IMPL(agx_backend_score)
