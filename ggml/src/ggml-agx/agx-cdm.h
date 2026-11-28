#pragma once

#include "agx-common.h"
#include "agx-drm.h"

#include <cstddef>
#include <cstdint>

struct agx_kernel;

// ============================================================================
// AGX3 HAL200/G17P CDM Command Stream & Ioto Compute Program Encoder
// ============================================================================

// Exact 6-u32 indirect dispatch geometry record consumed by HAL200/G17P
// mode-2 compact 24-byte CDM indirect launch (three global THREAD counts
// followed by three local workgroup dimensions).
struct agx_cdm_indirect_grid {
    uint32_t global_x;
    uint32_t global_y;
    uint32_t global_z;
    uint32_t local_x;
    uint32_t local_y;
    uint32_t local_z;
};
static_assert(sizeof(agx_cdm_indirect_grid) == 24, "HAL200 indirect grid must be 24 bytes");

struct agx_cdm_stats {
    uint64_t dispatches;
    uint64_t indirect_dispatches;
    uint64_t elided_barriers;
    uint64_t weak_wfi_barriers;
    uint64_t strong_usc_barriers;
    uint64_t memory_host_barriers;
    uint64_t submits;
    uint64_t last_gpu_ns;
    uint64_t total_gpu_ns;
    uint64_t replays;
};

struct agx_cdm_stream;

// Create/destroy a bare-metal CDM compute command stream with its own:
// - CDM command buffer BO (`AGX_BO_CMDSTREAM`)
// - USC Ioto state-loader arena BO (`AGX_BO_EXEC`, bit 40 set!)
// - Root descriptor + push-constant upload ring BO (`AGX_BO_CMDSTREAM`)
// - GPU timestamp BO (`AGX_BO_TIMESTAMPS`)
// - SLC-pinned 128B sense-reversing atomic barrier scratchpad BO (`AGX_BO_SLC_PIN`)
agx_cdm_stream * agx_cdm_stream_create(agx_drm_device * dev,
                                       size_t           cmd_bytes  = 2u  * 1024u * 1024u,
                                       size_t           ioto_bytes = 24u * 1024u * 1024u,
                                       size_t           arg_bytes  = 32u * 1024u * 1024u);
void             agx_cdm_stream_destroy(agx_cdm_stream * stream);

// Reset stream cursors for recording a fresh submission (waits if in-flight).
void             agx_cdm_stream_reset(agx_cdm_stream * stream);

// Allocate temporary GPU-visible argument / indirect-grid storage in the stream's
// SLC-pinned arg ring (valid until the next agx_cdm_stream_reset).
uint64_t         agx_cdm_alloc_args(agx_cdm_stream * stream,
                                    size_t           bytes,
                                    size_t           align,
                                    void **          out_cpu_ptr);

// Direct dispatch by WORKGROUP counts (grid_x, grid_y, grid_z).
// Multiplies by kernel->local_size to form HAL200 global thread counts.
// `push_args` (<= 256 B) is copied into the root descriptor table's push window.
// `post_barrier` controls TLA+-verified CDM barrier emission:
//   - AGX_BARRIER_NONE: elides the preceding dispatch's weak tail so independent
//     dispatches execute concurrently; still appends a minimal weak tail on the
//     final dispatch (upgraded or elided by subsequent calls).
//   - AGX_BARRIER_WEAK_WFI: 0x60000160 ordering-only barrier.
//   - AGX_BARRIER_STRONG_USC: 0x60001560 intra-VM SLC/USC visibility flush.
//   - AGX_BARRIER_MEMORY_HOST: 0x6000157c memory-scope flush.
bool             agx_cdm_dispatch(agx_cdm_stream *   stream,
                                  const agx_kernel * kernel,
                                  uint32_t           grid_x,
                                  uint32_t           grid_y,
                                  uint32_t           grid_z,
                                  const void *       push_args,
                                  size_t             push_bytes,
                                  agx_barrier_mode   post_barrier = AGX_BARRIER_STRONG_USC);

// Direct dispatch by GLOBAL THREAD counts (global_x, global_y, global_z).
bool             agx_cdm_dispatch_threads(agx_cdm_stream *   stream,
                                          const agx_kernel * kernel,
                                          uint32_t           global_x,
                                          uint32_t           global_y,
                                          uint32_t           global_z,
                                          const void *       push_args,
                                          size_t             push_bytes,
                                          agx_barrier_mode   post_barrier = AGX_BARRIER_STRONG_USC);

// GPU-indirect dispatch (24-byte HAL200 mode-2 record reading `agx_cdm_indirect_grid`
// from `geometry_va` in GPU memory; zero CPU round-trip).
bool             agx_cdm_dispatch_indirect(agx_cdm_stream *   stream,
                                           const agx_kernel * kernel,
                                           uint64_t           geometry_va,
                                           const void *       push_args,
                                           size_t             push_bytes,
                                           agx_barrier_mode   post_barrier = AGX_BARRIER_STRONG_USC);

// Explicit barrier control: strengthens the previous launch's trailing barrier
// in place (never emits two adjacent CDM barrier records, per HAL200 rules).
bool             agx_cdm_barrier(agx_cdm_stream * stream, agx_barrier_mode mode);
static inline bool agx_cdm_barrier_wfi(agx_cdm_stream * s)  { return agx_cdm_barrier(s, AGX_BARRIER_WEAK_WFI); }
static inline bool agx_cdm_barrier_usc(agx_cdm_stream * s)  { return agx_cdm_barrier(s, AGX_BARRIER_STRONG_USC); }
static inline bool agx_cdm_barrier_host(agx_cdm_stream * s) { return agx_cdm_barrier(s, AGX_BARRIER_MEMORY_HOST); }

// Terminate and submit the recorded CDM stream directly via DRM_IOCTL_ASAHI_SUBMIT.
// If `wait == true`, waits on the timeline syncobj, checks VM sticky error, and
// populates `*out_gpu_ns` from hardware GPU timestamps if non-null.
int              agx_cdm_submit(agx_cdm_stream * stream,
                                bool             wait       = true,
                                uint64_t *       out_gpu_ns = nullptr);

// Re-submit an already-recorded, already-terminated CDM stream (e.g. 48-pass
// pre-recorded Inkling 2 decode graph) with zero CPU stream re-encoding.
int              agx_cdm_submit_recorded(agx_cdm_stream * stream,
                                         bool             wait       = true,
                                         uint64_t *       out_gpu_ns = nullptr);

// Wait for the most recent async submission on this stream.
int              agx_cdm_wait(agx_cdm_stream * stream, uint64_t * out_gpu_ns = nullptr);

// Access the stream's SLC-pinned 128B sense-reversing barrier scratchpad BO.
agx_bo *         agx_cdm_get_sense_barrier_bo(agx_cdm_stream * stream);
uint64_t         agx_cdm_get_sense_barrier_va(agx_cdm_stream * stream);
void             agx_cdm_zero_sense_barrier(agx_cdm_stream * stream);
void             agx_cdm_reset_sense_barrier(agx_cdm_stream * stream, uint32_t n_workgroups = 0);

// Diagnostics & counters
void             agx_cdm_get_stats(const agx_cdm_stream * stream, agx_cdm_stats * out);
void             agx_cdm_reset_stats(agx_cdm_stream * stream);
bool             agx_cdm_is_empty(const agx_cdm_stream * stream);

// Steady-state CDM graph replay (toggleable via GGML_AGX_NO_REPLAY=1).
// When a ggml_cgraph repeats with identical uid, topology, and device BO epoch,
// `agx_cdm_stream_try_replay` re-submits the already-recorded CDM/Ioto/Arg BOs
// in place with zero CPU re-encoding.
bool             agx_cdm_replay_enabled();
uint64_t         agx_cdm_stream_record_gen(const agx_cdm_stream * stream);
uint64_t         agx_cdm_stream_submit_count(const agx_cdm_stream * stream);
bool             agx_cdm_stream_is_replayable(const agx_cdm_stream * stream);
void             agx_cdm_stream_set_replayable(agx_cdm_stream * stream, bool replayable);
bool             agx_cdm_stream_try_replay(agx_cdm_stream * stream,
                                           uint64_t         graph_uid,
                                           int              n_nodes,
                                           const void *     first_node,
                                           const void *     last_node,
                                           uint64_t         bo_epoch,
                                           bool             wait,
                                           uint64_t *       out_gpu_ns,
                                           int *            out_ret);
void             agx_cdm_stream_tag_replay(agx_cdm_stream * stream,
                                           uint64_t         graph_uid,
                                           int              n_nodes,
                                           const void *     first_node,
                                           const void *     last_node,
                                           uint64_t         bo_epoch);
void             agx_cdm_stream_invalidate_replay(agx_cdm_stream * stream);

// Low-level HAL200/G17P CDM packet encoders (also callable from unit tests)
uint32_t         agx_cdm_encode_barrier_word(agx_barrier_mode mode);
bool             agx_cdm_pack_direct_launch(uint64_t       program_va,
                                            const uint32_t global_size[3],
                                            const uint32_t local_size[3],
                                            uint32_t       ti_spill_bytes,
                                            uint32_t       ti_ipr_bytes,
                                            uint8_t        out[40]);
bool             agx_cdm_pack_indirect_launch(uint64_t program_va,
                                              uint64_t geometry_va,
                                              uint32_t ti_spill_bytes,
                                              uint32_t ti_ipr_bytes,
                                              uint8_t  out[24]);
