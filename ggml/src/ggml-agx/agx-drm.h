#pragma once

#include "agx-common.h"

#include <cstddef>
#include <cstdint>

// ============================================================================
// Direct Linux drm_asahi (/dev/dri/renderD*) UAPI Runtime
// ============================================================================
// Self-contained: vendors exact drm_asahi ioctl definitions matching
// linux/drivers/gpu/drm/asahi & mesa include/drm-uapi/asahi_drm.h with zero
// external libdrm/kernel-header build requirements.

struct agx_drm_device_info {
    char     render_node[64];
    char     driver_name[32];
    char     device_name[96];

    uint64_t features;
    uint32_t gpu_generation;       // 16 for M4 (G16C)
    uint32_t gpu_variant;          // 'C' for Max
    uint32_t gpu_revision;         // BCD e.g. 0x11
    uint32_t chip_id;              // 0x6041 for T6041 (M4 Max)
    uint32_t num_dies;
    uint32_t num_clusters;         // 4 on M4 Max
    uint32_t cores_per_cluster;    // 10 on M4 Max
    uint32_t num_cores;            // 40 on M4 Max
    uint32_t max_frequency_khz;    // 1578000 on M4 Max

    uint32_t usc_generation;       // 3 (AGX3)
    uint32_t gpu_hal_generation;   // 200 (HAL200 -> G17P command stream)
    uint32_t max_commands_per_sub; // 64
    uint32_t max_attachments;      // 16
    uint64_t timestamp_freq_hz;    // 1000000000 (1 ns ticks)

    uint64_t vm_start;
    uint64_t vm_end;
    uint64_t vm_kernel_min_size;
    uint64_t memory_limit;
    uint64_t client_memory_limit;

    bool     has_vm_status;
    bool     has_vm_status_mirror;
    bool     has_gem_residency;
    bool     has_compute_timeout;
    bool     has_userptr;
    bool     has_userptr_no_memlock;
    bool     has_soft_faults;
    bool     has_cumulative_signals;
};

struct agx_drm_memory_stats {
    uint64_t slc_pin_bytes;
    uint64_t slc_stream_bytes;
    uint64_t exec_bytes;
    uint64_t cmdstream_bytes;
    uint64_t total_allocated_bytes;
    uint64_t memory_limit_bytes;
    uint64_t free_estimate_bytes;
};

struct agx_bo {
    agx_drm_device * dev;
    uint32_t         handle;         // GEM handle
    uint32_t         ts_obj_handle;  // non-zero if bound for DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS
    uint64_t         gpu_va;         // 16 KiB aligned GPU virtual address
    void *           cpu_ptr;        // CPU mapping (mmap or userptr)
    size_t           size;           // requested byte size
    size_t           alloc_size;     // 16 KiB rounded backing size
    uint32_t         flags;          // agx_bo_flags bitmask
    agx_cache_policy policy;
    bool             residency_kept;
    bool             is_userptr;
    char             label[64];
};

// Device discovery & lifecycle
int                        agx_drm_get_device_count();
agx_drm_device *           agx_drm_open(int device_index = 0);
void                       agx_drm_close(agx_drm_device * dev);
const agx_drm_device_info & agx_drm_get_info(const agx_drm_device * dev);
int                        agx_drm_get_fd(const agx_drm_device * dev);
uint32_t                   agx_drm_get_vm_id(const agx_drm_device * dev);
uint32_t                   agx_drm_get_queue_id(const agx_drm_device * dev);

// Buffer Object allocation & Pillar 4 SLC cache-policy controls
agx_bo * agx_bo_alloc(agx_drm_device * dev, size_t size, uint32_t flags, const char * label = nullptr);
agx_bo * agx_bo_import_userptr(agx_drm_device * dev, void * ptr, size_t size, bool read_only, const char * label = nullptr);
void     agx_bo_free(agx_drm_device * dev, agx_bo * bo);
int      agx_bo_set_residency(agx_drm_device * dev, agx_bo * bo, bool keep);
int      agx_bo_madvise(agx_drm_device * dev, agx_bo * bo, bool dontneed, bool * retained = nullptr);
void     agx_drm_get_memory_stats(const agx_drm_device * dev, agx_drm_memory_stats * out);

// Direct CDM compute command submission & timeline synchronization
int      agx_drm_submit_cdm(agx_drm_device * dev,
                            uint64_t         cdm_start_va,
                            uint64_t         cdm_end_va,
                            uint32_t         ts_obj_handle,
                            uint32_t         ts_start_offset_B,
                            uint32_t         ts_end_offset_B,
                            uint64_t *       out_timeline_point,
                            uint32_t         compute_flags = 0);

int      agx_drm_wait_timeline(agx_drm_device * dev,
                               uint64_t         timeline_point,
                               uint64_t         timeout_ns = UINT64_MAX);
void     agx_drm_sync_all(agx_drm_device * dev);

// Sticky VM error, BO mutation epoch & GPU hardware timestamp queries
int32_t  agx_drm_vm_error(agx_drm_device * dev);
uint64_t agx_drm_get_bo_epoch(const agx_drm_device * dev);
uint64_t agx_drm_get_gpu_time_ns(agx_drm_device * dev);
uint64_t agx_drm_ticks_to_ns(const agx_drm_device * dev, uint64_t ticks);
