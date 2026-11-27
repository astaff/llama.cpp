#include "agx-drm.h"

#include <linux/ioctl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

// ============================================================================
// Self-contained Linux DRM + drm_asahi UAPI Definitions
// (Exact ABI match to linux/include/uapi/drm/drm.h & asahi_drm.h)
// ============================================================================

namespace {

#ifndef DRM_IOCTL_BASE
#define DRM_IOCTL_BASE 'd'
#endif
#ifndef DRM_COMMAND_BASE
#define DRM_COMMAND_BASE 0x40
#endif

#define AGX_DRM_IO(nr)          _IO(DRM_IOCTL_BASE, nr)
#define AGX_DRM_IOR(nr, type)   _IOR(DRM_IOCTL_BASE, nr, type)
#define AGX_DRM_IOW(nr, type)   _IOW(DRM_IOCTL_BASE, nr, type)
#define AGX_DRM_IOWR(nr, type)  _IOWR(DRM_IOCTL_BASE, nr, type)

struct agx_drm_version {
    int    version_major;
    int    version_minor;
    int    version_patchlevel;
    size_t name_len;
    char * name;
    size_t date_len;
    char * date;
    size_t desc_len;
    char * desc;
};

struct agx_drm_gem_close {
    uint32_t handle;
    uint32_t pad;
};

struct agx_drm_syncobj_create {
    uint32_t handle;
    uint32_t flags;
};

struct agx_drm_syncobj_destroy {
    uint32_t handle;
    uint32_t pad;
};

struct agx_drm_syncobj_timeline_wait {
    uint64_t handles;
    uint64_t points;
    int64_t  timeout_nsec;
    uint32_t count_handles;
    uint32_t flags;
    uint32_t first_signaled;
    uint32_t pad;
};

constexpr uint32_t AGX_DRM_SYNCOBJ_WAIT_ALL        = (1u << 0);
constexpr uint32_t AGX_DRM_SYNCOBJ_WAIT_FOR_SUBMIT = (1u << 1);

constexpr unsigned long AGX_IOCTL_DRM_VERSION               = AGX_DRM_IOWR(0x00, struct agx_drm_version);
constexpr unsigned long AGX_IOCTL_DRM_GEM_CLOSE             = AGX_DRM_IOW(0x09, struct agx_drm_gem_close);
constexpr unsigned long AGX_IOCTL_DRM_SYNCOBJ_CREATE        = AGX_DRM_IOWR(0xBF, struct agx_drm_syncobj_create);
constexpr unsigned long AGX_IOCTL_DRM_SYNCOBJ_DESTROY       = AGX_DRM_IOWR(0xC0, struct agx_drm_syncobj_destroy);
constexpr unsigned long AGX_IOCTL_DRM_SYNCOBJ_TIMELINE_WAIT = AGX_DRM_IOWR(0xCA, struct agx_drm_syncobj_timeline_wait);

// ---- drm_asahi UAPI ----

constexpr uint32_t ASAHI_MAX_CLUSTERS = 64u;

enum asahi_ioctl_id : uint32_t {
    ASAHI_GET_PARAMS = 0,
    ASAHI_GET_TIME,
    ASAHI_VM_CREATE,
    ASAHI_VM_DESTROY,
    ASAHI_VM_BIND,
    ASAHI_GEM_CREATE,
    ASAHI_GEM_MMAP_OFFSET,
    ASAHI_GEM_BIND_OBJECT,
    ASAHI_QUEUE_CREATE,
    ASAHI_QUEUE_DESTROY,
    ASAHI_SUBMIT,
    ASAHI_GEM_MADVISE,
    ASAHI_GEM_USERPTR,
    ASAHI_GEM_RESIDENCY,
};

constexpr uint64_t ASAHI_FEAT_SOFT_FAULTS        = (1ULL << 0);
constexpr uint64_t ASAHI_FEAT_VM_STATUS          = (1ULL << 1);
constexpr uint64_t ASAHI_FEAT_VM_STATUS_MIRROR   = (1ULL << 11);
constexpr uint64_t ASAHI_FEAT_CUMULATIVE_SIGNALS = (1ULL << 29);
constexpr uint64_t ASAHI_FEAT_GEM_RESIDENCY      = (1ULL << 35);
constexpr uint64_t ASAHI_FEAT_COMPUTE_TIMEOUT    = (1ULL << 36);
constexpr uint64_t ASAHI_FEAT_USERPTR            = (1ULL << 37);
constexpr uint64_t ASAHI_FEAT_USERPTR_NO_MEMLOCK = (1ULL << 43);

constexpr uint32_t ASAHI_PARAM_GROUP_GLOBAL          = 0u;
constexpr uint32_t ASAHI_PARAM_GROUP_VM_STATUS       = 1u;
constexpr uint32_t ASAHI_PARAM_GROUP_VM_STATUS_MIRROR = 3u;

constexpr uint32_t ASAHI_GEM_WRITEBACK         = (1u << 0);
constexpr uint32_t ASAHI_GEM_VM_PRIVATE        = (1u << 1);
constexpr uint32_t ASAHI_GEM_USERPTR_READ_ONLY = (1u << 2);

constexpr uint32_t ASAHI_GEM_MADV_WILLNEED = 0u;
constexpr uint32_t ASAHI_GEM_MADV_DONTNEED = 1u;

constexpr uint32_t ASAHI_GEM_RESIDENCY_KEEP = (1u << 0);

constexpr uint32_t ASAHI_BIND_UNBIND = (1u << 0);
constexpr uint32_t ASAHI_BIND_READ   = (1u << 1);
constexpr uint32_t ASAHI_BIND_WRITE  = (1u << 2);

constexpr uint32_t ASAHI_BIND_OBJECT_OP_BIND       = 0u;
constexpr uint32_t ASAHI_BIND_OBJECT_OP_UNBIND     = 1u;
constexpr uint32_t ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS = (1u << 0);

constexpr uint32_t ASAHI_PRIORITY_LOW    = 0u;
constexpr uint32_t ASAHI_PRIORITY_MEDIUM = 1u;
constexpr uint32_t ASAHI_PRIORITY_HIGH   = 2u;

constexpr uint16_t ASAHI_CMD_COMPUTE  = 1u;
constexpr uint16_t ASAHI_BARRIER_NONE = 0xFFFFu;

constexpr uint32_t ASAHI_SYNC_TIMELINE_SYNCOBJ            = 1u;
constexpr uint32_t ASAHI_SYNC_TIMELINE_SYNCOBJ_CUMULATIVE = 2u;

struct asahi_params_global {
    uint64_t features;
    uint32_t gpu_generation;
    uint32_t gpu_variant;
    uint32_t gpu_revision;
    uint32_t chip_id;
    uint32_t num_dies;
    uint32_t num_clusters_total;
    uint32_t num_cores_per_cluster;
    uint32_t max_frequency_khz;
    uint64_t core_masks[ASAHI_MAX_CLUSTERS];
    uint64_t vm_start;
    uint64_t vm_end;
    uint64_t vm_kernel_min_size;
    uint32_t max_commands_per_submission;
    uint32_t max_attachments;
    uint64_t command_timestamp_frequency_hz;
    uint32_t usc_generation;
    uint32_t gpu_hal_generation;
    uint64_t memory_limit;
    uint64_t client_memory_limit;
};

struct asahi_get_params {
    uint32_t param_group;
    uint32_t pad;
    uint64_t pointer;
    uint64_t size;
};

struct asahi_vm_status {
    uint32_t vm_id;
    uint32_t flags;
    int32_t  error;
    uint32_t pad;
};

struct asahi_vm_status_mirror {
    uint32_t vm_id;
    uint32_t handle;
    uint32_t offset;
    uint32_t flags;
};

struct asahi_get_time {
    uint64_t flags;
    uint64_t gpu_timestamp;
};

struct asahi_vm_create {
    uint64_t kernel_start;
    uint64_t kernel_end;
    uint32_t vm_id;
    uint32_t pad;
};

struct asahi_vm_destroy {
    uint32_t vm_id;
    uint32_t pad;
};

struct asahi_gem_bind_op {
    uint32_t flags;
    uint32_t handle;
    uint64_t offset;
    uint64_t range;
    uint64_t addr;
};

struct asahi_vm_bind {
    uint32_t vm_id;
    uint32_t num_binds;
    uint32_t stride;
    uint32_t pad;
    uint64_t userptr;
};

struct asahi_gem_create {
    uint64_t size;
    uint32_t flags;
    uint32_t vm_id;
    uint32_t handle;
    uint32_t pad;
};

struct asahi_gem_userptr {
    uint64_t userptr;
    uint64_t size;
    uint32_t flags;
    uint32_t vm_id;
    uint32_t handle;
    uint32_t pad;
};

struct asahi_gem_mmap_offset {
    uint32_t handle;
    uint32_t flags;
    uint64_t offset;
};

struct asahi_gem_madvise {
    uint32_t handle;
    uint32_t madv;
    uint32_t retained;
    uint32_t pad;
};

struct asahi_gem_residency {
    uint32_t handle;
    uint32_t flags;
};

struct asahi_gem_bind_object {
    uint32_t op;
    uint32_t flags;
    uint32_t handle;
    uint32_t vm_id;
    uint64_t offset;
    uint64_t range;
    uint32_t object_handle;
    uint32_t pad;
};

struct asahi_queue_create {
    uint32_t flags;
    uint32_t vm_id;
    uint32_t priority;
    uint32_t queue_id;
    uint64_t usc_exec_base;
    uint64_t render_scs_layout[2];
    uint32_t compute_timeout_ms;
    uint32_t pad;
};

struct asahi_queue_destroy {
    uint32_t queue_id;
    uint32_t pad;
};

struct asahi_sync {
    uint32_t sync_type;
    uint32_t handle;
    uint64_t timeline_value;
};

struct asahi_cmd_header {
    uint16_t cmd_type;
    uint16_t size;
    uint16_t vdm_barrier;
    uint16_t cdm_barrier;
};

struct asahi_timestamp {
    uint32_t handle;
    uint32_t offset;
};

struct asahi_timestamps {
    struct asahi_timestamp start;
    struct asahi_timestamp end;
};

struct asahi_helper_program {
    uint32_t binary;
    uint32_t cfg;
    uint64_t data;
};

struct asahi_cmd_compute {
    uint32_t                    flags;
    uint32_t                    sampler_count;
    uint64_t                    cdm_ctrl_stream_base;
    uint64_t                    cdm_ctrl_stream_end;
    uint64_t                    sampler_heap;
    struct asahi_helper_program helper;
    struct asahi_timestamps     ts;
    struct asahi_timestamp      ts_end_copies[8];
};

struct asahi_submit {
    uint64_t syncs;
    uint64_t cmdbuf;
    uint32_t flags;
    uint32_t queue_id;
    uint32_t in_sync_count;
    uint32_t out_sync_count;
    uint32_t cmdbuf_size;
    uint32_t pad;
};

#define AGX_ASAHI_IOCTL(access, id, type) \
    AGX_DRM_IO##access(DRM_COMMAND_BASE + (id), struct asahi_##type)

constexpr unsigned long IOCTL_ASAHI_GET_PARAMS      = AGX_DRM_IOW(DRM_COMMAND_BASE + ASAHI_GET_PARAMS, struct asahi_get_params);
constexpr unsigned long IOCTL_ASAHI_GET_TIME        = AGX_DRM_IOWR(DRM_COMMAND_BASE + ASAHI_GET_TIME, struct asahi_get_time);
constexpr unsigned long IOCTL_ASAHI_VM_CREATE       = AGX_DRM_IOWR(DRM_COMMAND_BASE + ASAHI_VM_CREATE, struct asahi_vm_create);
constexpr unsigned long IOCTL_ASAHI_VM_DESTROY      = AGX_DRM_IOW(DRM_COMMAND_BASE + ASAHI_VM_DESTROY, struct asahi_vm_destroy);
constexpr unsigned long IOCTL_ASAHI_VM_BIND         = AGX_DRM_IOW(DRM_COMMAND_BASE + ASAHI_VM_BIND, struct asahi_vm_bind);
constexpr unsigned long IOCTL_ASAHI_GEM_CREATE      = AGX_DRM_IOWR(DRM_COMMAND_BASE + ASAHI_GEM_CREATE, struct asahi_gem_create);
constexpr unsigned long IOCTL_ASAHI_GEM_MMAP_OFFSET = AGX_DRM_IOWR(DRM_COMMAND_BASE + ASAHI_GEM_MMAP_OFFSET, struct asahi_gem_mmap_offset);
constexpr unsigned long IOCTL_ASAHI_GEM_BIND_OBJECT = AGX_DRM_IOWR(DRM_COMMAND_BASE + ASAHI_GEM_BIND_OBJECT, struct asahi_gem_bind_object);
constexpr unsigned long IOCTL_ASAHI_QUEUE_CREATE    = AGX_DRM_IOWR(DRM_COMMAND_BASE + ASAHI_QUEUE_CREATE, struct asahi_queue_create);
constexpr unsigned long IOCTL_ASAHI_QUEUE_DESTROY   = AGX_DRM_IOW(DRM_COMMAND_BASE + ASAHI_QUEUE_DESTROY, struct asahi_queue_destroy);
constexpr unsigned long IOCTL_ASAHI_SUBMIT          = AGX_DRM_IOW(DRM_COMMAND_BASE + ASAHI_SUBMIT, struct asahi_submit);
constexpr unsigned long IOCTL_ASAHI_GEM_MADVISE     = AGX_DRM_IOWR(DRM_COMMAND_BASE + ASAHI_GEM_MADVISE, struct asahi_gem_madvise);
constexpr unsigned long IOCTL_ASAHI_GEM_USERPTR     = AGX_DRM_IOWR(DRM_COMMAND_BASE + ASAHI_GEM_USERPTR, struct asahi_gem_userptr);
constexpr unsigned long IOCTL_ASAHI_GEM_RESIDENCY   = AGX_DRM_IOW(DRM_COMMAND_BASE + ASAHI_GEM_RESIDENCY, struct asahi_gem_residency);

static int drm_ioctl_retry(int fd, unsigned long req, void * arg) {
    int ret;
    do {
        ret = ioctl(fd, req, arg);
    } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
    return ret;
}

static bool is_asahi_driver_name(const char * name) {
    return name && (std::strcmp(name, "aurora") == 0 || std::strcmp(name, "asahi") == 0);
}

static bool probe_render_node(const char * path, char driver_out[32]) {
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }

    char name_buf[32] = {};
    struct agx_drm_version ver = {};
    ver.name_len = sizeof(name_buf) - 1;
    ver.name     = name_buf;

    bool ok = false;
    if (drm_ioctl_retry(fd, AGX_IOCTL_DRM_VERSION, &ver) == 0) {
        name_buf[std::min(ver.name_len, sizeof(name_buf) - 1)] = '\0';
        if (is_asahi_driver_name(name_buf)) {
            ok = true;
            if (driver_out) {
                std::snprintf(driver_out, 32, "%s", name_buf);
            }
        }
    }
    close(fd);
    return ok;
}

static int64_t monotonic_now_ns() {
    struct timespec ts = {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + static_cast<int64_t>(ts.tv_nsec);
}

static uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t r = a % b;
        a = b;
        b = r;
    }
    return a ? a : 1;
}

// Simple coalescing free-list GPU virtual address allocator (16 KiB granularity).
struct agx_va_arena {
    struct range {
        uint64_t start;
        uint64_t size;
    };

    std::vector<range> free_list;
    uint64_t base = 0;
    uint64_t size = 0;

    void init(uint64_t base_va, uint64_t size_bytes) {
        base = base_va;
        size = size_bytes;
        free_list.clear();
        if (size_bytes > 0) {
            free_list.push_back({ base_va, size_bytes });
        }
    }

    uint64_t alloc(uint64_t bytes, uint64_t align) {
        if (bytes == 0) {
            return 0;
        }
        align = std::max<uint64_t>(align, AGX_PAGE_SIZE);
        bytes = agx_align_up_u64(bytes, AGX_PAGE_SIZE);
        // Reserve one guard page after every allocation to catch out-of-bounds cleanly.
        const uint64_t total = bytes + AGX_PAGE_SIZE;

        for (size_t i = 0; i < free_list.size(); ++i) {
            const uint64_t r_start = free_list[i].start;
            const uint64_t r_end   = r_start + free_list[i].size;
            const uint64_t aligned = agx_align_up_u64(r_start, align);
            if (aligned < r_end && (r_end - aligned) >= total) {
                const uint64_t alloc_end = aligned + total;
                if (aligned == r_start && alloc_end == r_end) {
                    free_list.erase(free_list.begin() + static_cast<ptrdiff_t>(i));
                } else if (aligned == r_start) {
                    free_list[i].start = alloc_end;
                    free_list[i].size  = r_end - alloc_end;
                } else if (alloc_end == r_end) {
                    free_list[i].size = aligned - r_start;
                } else {
                    free_list[i].size = aligned - r_start;
                    free_list.insert(free_list.begin() + static_cast<ptrdiff_t>(i + 1),
                                     { alloc_end, r_end - alloc_end });
                }
                return aligned;
            }
        }
        return 0;
    }

    void free(uint64_t va, uint64_t bytes) {
        if (va == 0 || bytes == 0) {
            return;
        }
        const uint64_t total = agx_align_up_u64(bytes, AGX_PAGE_SIZE) + AGX_PAGE_SIZE;
        auto it = std::lower_bound(free_list.begin(), free_list.end(), va,
                                   [](const range & r, uint64_t addr) { return r.start < addr; });
        it = free_list.insert(it, { va, total });

        // Merge with next
        if (it + 1 != free_list.end() && it->start + it->size == (it + 1)->start) {
            it->size += (it + 1)->size;
            free_list.erase(it + 1);
        }
        // Merge with prev
        if (it != free_list.begin() && (it - 1)->start + (it - 1)->size == it->start) {
            (it - 1)->size += it->size;
            free_list.erase(it);
        }
    }
};

} // namespace

// ============================================================================
// agx_drm_device Implementation
// ============================================================================

struct agx_drm_device {
    int                 fd       = -1;
    uint32_t            vm_id    = 0;
    uint32_t            queue_id = 0;
    uint32_t            syncobj  = 0;
    agx_drm_device_info info     = {};

    uint64_t ts_mul = 1;
    uint64_t ts_div = 1;

    // Partitioned VA arenas (Pillar 4)
    std::mutex   va_mutex;
    agx_va_arena va_slc_pin;     // [0x4000000000, 0x4800000000) (32 GiB)
    agx_va_arena va_slc_stream;  // [0x5000000000, 0x8000000000) (192 GiB)
    agx_va_arena va_usc_exec;    // [0x10000008000, 0x10100000000) (~4 GiB, bit 40 set!)

    // Fixed hardware BOs required by G17P / HAL200
    agx_bo * usc_base_bo       = nullptr;
    agx_bo * render_context_bo = nullptr;
    agx_bo * zero_page_bo      = nullptr;
    agx_bo * vm_mirror_bo      = nullptr;
    volatile int32_t * vm_mirror_word = nullptr;

    // Submission serialization + monotonic timeline
    std::mutex submit_mutex;
    uint64_t   timeline_value = 0;

    // Memory accounting
    std::atomic<uint64_t> slc_pin_bytes{ 0 };
    std::atomic<uint64_t> slc_stream_bytes{ 0 };
    std::atomic<uint64_t> exec_bytes{ 0 };
    std::atomic<uint64_t> cmdstream_bytes{ 0 };
};

static int vm_bind_single(agx_drm_device * dev, uint32_t handle, uint64_t va,
                          uint64_t range, uint64_t offset, uint32_t flags) {
    struct asahi_gem_bind_op op = {};
    op.flags  = flags;
    op.handle = handle;
    op.offset = offset;
    op.range  = range;
    op.addr   = va;

    struct asahi_vm_bind bind = {};
    bind.vm_id     = dev->vm_id;
    bind.num_binds = 1;
    bind.stride    = sizeof(op);
    bind.userptr   = reinterpret_cast<uint64_t>(&op);

    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_VM_BIND, &bind) != 0) {
        AGX_LOG_ERROR("DRM_IOCTL_ASAHI_VM_BIND failed (va=0x%llx range=0x%llx handle=%u flags=0x%x): %s",
                      (unsigned long long) va, (unsigned long long) range, handle, flags, std::strerror(errno));
        return -errno;
    }
    return 0;
}

static constexpr uint64_t AGX_G17P_RENDER_CONTEXT_BASE = 0x1000000000ULL;
static constexpr uint64_t AGX_G17P_RENDER_CONTEXT_SIZE = 0x00080000ULL;

static constexpr struct {
    uint64_t offset;
    uint64_t size;
} k_g17p_context_low_slots[6] = {
    { 0x18000ULL, 0x8000ULL },
    { 0x28000ULL, 0x8000ULL },
    { 0x38000ULL, 0x8000ULL },
    { 0x48000ULL, 0x8000ULL },
    { 0x58000ULL, 0x8000ULL },
    { 0x68000ULL, 0xc000ULL },
};

// Exact G17P/HAL200 USC window base page builder (mirrors Mesa agx_linker.c:agx3_build_usc_base_page).
static void g17p_build_usc_base_page(uint8_t * page, size_t size,
                                     uint32_t cores_per_cluster, uint32_t num_clusters) {
    static constexpr uint32_t kLeadSize = 0x340u;
    static constexpr uint8_t  kHelperTerminal[16] = {
        0xf7, 0x03, 0xaa, 0x00, 0x8f, 0x02, 0x54, 0x01,
        0x06, 0x00, 0x06, 0x00, 0x06, 0x00, 0x06, 0x00,
    };

    std::memset(page, 0, size);
    const uint32_t lead = kLeadSize;
    std::memcpy(page, &lead, sizeof(lead));
    for (size_t off = 0x40; off < kLeadSize; off += 2) {
        page[off]     = 0x06;
        page[off + 1] = 0x00;
    }

    const uint32_t slots = std::min<uint32_t>(14u, std::max<uint32_t>(6u, cores_per_cluster));
    auto fill_table = [&](size_t base) {
        for (uint32_t i = 0; i < slots; ++i) {
            uint8_t * rec = page + base + i * 0x10u;
            std::memset(rec, 0, 0x10u);
            rec[0] = 0x0f;
            rec[2] = 0x54;
            rec[3] = static_cast<uint8_t>((slots - i) * 0x10u);
            for (size_t h = 10; h < 16; h += 2) {
                rec[h]     = 0x06;
                rec[h + 1] = 0x00;
            }
        }
        std::memcpy(page + base + slots * 0x10u, kHelperTerminal, 16u);
        std::memcpy(page + base + (slots + 1u) * 0x10u, kHelperTerminal, 16u);
    };
    fill_table(0x100u);
    fill_table(0x200u);

    // Multi-cluster parts (e.g. M4 Max = 4 clusters) fill every slot past the tables
    // and past the zero chain terminator at +0x340 with helper return terminals.
    if (num_clusters > 2u) {
        for (size_t off = 0x200u + (slots + 2u) * 0x10u; off + 16u <= kLeadSize; off += 16u) {
            std::memcpy(page + off, kHelperTerminal, 16u);
        }
        for (size_t off = kLeadSize + 0x10u; off + 16u <= size; off += 16u) {
            std::memcpy(page + off, kHelperTerminal, 16u);
        }
    }
}

static agx_bo * alloc_fixed_bo(agx_drm_device * dev, size_t size, uint64_t fixed_va,
                               uint32_t bind_flags, bool writeback, const char * label,
                               bool shared = false) {
    const size_t alloc_size = agx_align_up(size, AGX_PAGE_SIZE);

    struct asahi_gem_create req = {};
    req.size  = alloc_size;
    req.flags = (shared ? 0u : ASAHI_GEM_VM_PRIVATE) | (writeback ? ASAHI_GEM_WRITEBACK : 0u);
    req.vm_id = shared ? 0u : dev->vm_id;
    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GEM_CREATE, &req) != 0) {
        AGX_LOG_ERROR("GEM_CREATE(%s, %zu) failed: %s", label ? label : "", alloc_size, std::strerror(errno));
        return nullptr;
    }

    struct asahi_gem_mmap_offset mm = {};
    mm.handle = req.handle;
    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GEM_MMAP_OFFSET, &mm) != 0) {
        struct agx_drm_gem_close cl = { req.handle, 0 };
        drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);
        return nullptr;
    }

    void * cpu_ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
                          static_cast<off_t>(mm.offset));
    if (cpu_ptr == MAP_FAILED) {
        struct agx_drm_gem_close cl = { req.handle, 0 };
        drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);
        return nullptr;
    }
    std::memset(cpu_ptr, 0, alloc_size);

    if (fixed_va != 0 && vm_bind_single(dev, req.handle, fixed_va, alloc_size, 0, bind_flags) != 0) {
        munmap(cpu_ptr, alloc_size);
        struct agx_drm_gem_close cl = { req.handle, 0 };
        drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);
        return nullptr;
    }

    auto * bo = new agx_bo{};
    bo->dev        = dev;
    bo->handle     = req.handle;
    bo->gpu_va     = fixed_va;
    bo->cpu_ptr    = cpu_ptr;
    bo->size       = size;
    bo->alloc_size = alloc_size;
    bo->flags      = writeback ? AGX_BO_WRITEBACK : AGX_BO_NONE;
    bo->policy     = AGX_CACHE_SLC_PIN;
    if (label) {
        std::snprintf(bo->label, sizeof(bo->label), "%s", label);
    }
    return bo;
}

static void free_fixed_bo(agx_drm_device * dev, agx_bo * bo) {
    if (!bo) {
        return;
    }
    if (bo->gpu_va != 0) {
        vm_bind_single(dev, 0, bo->gpu_va, bo->alloc_size, 0, ASAHI_BIND_UNBIND);
    }
    if (bo->cpu_ptr) {
        munmap(bo->cpu_ptr, bo->alloc_size);
    }
    struct agx_drm_gem_close cl = { bo->handle, 0 };
    drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);
    delete bo;
}

int agx_drm_get_device_count() {
    int count = 0;
    for (int i = 128; i < 144; ++i) {
        char path[64];
        std::snprintf(path, sizeof(path), "/dev/dri/renderD%d", i);
        if (probe_render_node(path, nullptr)) {
            ++count;
        }
    }
    return count;
}

agx_drm_device * agx_drm_open(int device_index) {
    char chosen_path[64] = {};
    char driver_name[32] = {};
    int seen = 0;

    for (int i = 128; i < 144; ++i) {
        char path[64];
        std::snprintf(path, sizeof(path), "/dev/dri/renderD%d", i);
        char drv[32] = {};
        if (probe_render_node(path, drv)) {
            if (seen == device_index) {
                std::snprintf(chosen_path, sizeof(chosen_path), "%s", path);
                std::snprintf(driver_name, sizeof(driver_name), "%s", drv);
                break;
            }
            ++seen;
        }
    }

    if (chosen_path[0] == '\0') {
        AGX_LOG_ERROR("No drm_asahi/aurora render node found for device index %d", device_index);
        return nullptr;
    }

    int fd = open(chosen_path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        AGX_LOG_ERROR("Failed to open %s: %s", chosen_path, std::strerror(errno));
        return nullptr;
    }

    struct asahi_params_global params = {};
    struct asahi_get_params gp = {};
    gp.param_group = ASAHI_PARAM_GROUP_GLOBAL;
    gp.pointer     = reinterpret_cast<uint64_t>(&params);
    gp.size        = sizeof(params);
    if (drm_ioctl_retry(fd, IOCTL_ASAHI_GET_PARAMS, &gp) != 0) {
        AGX_LOG_ERROR("DRM_IOCTL_ASAHI_GET_PARAMS failed on %s: %s", chosen_path, std::strerror(errno));
        close(fd);
        return nullptr;
    }

    if (params.usc_generation != 3 || params.gpu_hal_generation != 200) {
        AGX_LOG_WARN("Unexpected GPU profile on %s (USC=%u HAL=%u, expected AGX3/HAL200); proceeding carefully",
                     chosen_path, params.usc_generation, params.gpu_hal_generation);
    }

    auto * dev = new agx_drm_device{};
    dev->fd = fd;

    auto & info = dev->info;
    std::snprintf(info.render_node, sizeof(info.render_node), "%s", chosen_path);
    std::snprintf(info.driver_name, sizeof(info.driver_name), "%s", driver_name);
    info.features             = params.features;
    info.gpu_generation       = params.gpu_generation;
    info.gpu_variant          = params.gpu_variant;
    info.gpu_revision         = params.gpu_revision;
    info.chip_id              = params.chip_id;
    info.num_dies             = params.num_dies;
    info.num_clusters         = params.num_clusters_total;
    info.cores_per_cluster    = params.num_cores_per_cluster;
    info.max_frequency_khz    = params.max_frequency_khz;
    info.usc_generation       = params.usc_generation;
    info.gpu_hal_generation   = params.gpu_hal_generation;
    info.max_commands_per_sub = params.max_commands_per_submission;
    info.max_attachments      = params.max_attachments;
    info.timestamp_freq_hz    = params.command_timestamp_frequency_hz ? params.command_timestamp_frequency_hz : 1000000000ULL;
    info.vm_start             = params.vm_start;
    info.vm_end               = params.vm_end;
    info.vm_kernel_min_size   = params.vm_kernel_min_size;
    info.memory_limit         = params.memory_limit;
    info.client_memory_limit  = params.client_memory_limit;

    uint32_t active_cores = 0;
    for (uint32_t c = 0; c < std::min(params.num_clusters_total, ASAHI_MAX_CLUSTERS); ++c) {
        active_cores += static_cast<uint32_t>(__builtin_popcountll(params.core_masks[c]));
    }
    info.num_cores = active_cores ? active_cores : (params.num_clusters_total * params.num_cores_per_cluster);

    info.has_vm_status          = (params.features & ASAHI_FEAT_VM_STATUS) != 0;
    info.has_vm_status_mirror   = (params.features & ASAHI_FEAT_VM_STATUS_MIRROR) != 0;
    info.has_gem_residency      = (params.features & ASAHI_FEAT_GEM_RESIDENCY) != 0;
    info.has_compute_timeout    = (params.features & ASAHI_FEAT_COMPUTE_TIMEOUT) != 0;
    info.has_userptr            = (params.features & ASAHI_FEAT_USERPTR) != 0;
    info.has_userptr_no_memlock = (params.features & ASAHI_FEAT_USERPTR_NO_MEMLOCK) != 0;
    info.has_soft_faults        = (params.features & ASAHI_FEAT_SOFT_FAULTS) != 0;
    info.has_cumulative_signals = (params.features & ASAHI_FEAT_CUMULATIVE_SIGNALS) != 0;

    const char * variant_str = "";
    switch (params.gpu_variant) {
        case 'G': variant_str = "";       break;
        case 'S': variant_str = " Pro";   break;
        case 'C': variant_str = " Max";   break;
        case 'D': variant_str = " Ultra"; break;
        default:  variant_str = "";       break;
    }
    std::snprintf(info.device_name, sizeof(info.device_name),
                  "Apple M%u%s (G%u%c %02X, %uc AGX3, drm_asahi)",
                  params.gpu_generation >= 12 ? params.gpu_generation - 12 : params.gpu_generation,
                  variant_str, params.gpu_generation,
                  (char)(params.gpu_variant ? params.gpu_variant : '?'),
                  params.gpu_revision + 0xA0u, info.num_cores);

    // Timestamp conversion ratio
    const uint64_t g = gcd_u64(info.timestamp_freq_hz, 1000000000ULL);
    dev->ts_mul = 1000000000ULL / g;
    dev->ts_div = info.timestamp_freq_hz / g;

    // Create GPU VM with 32 GiB kernel reservation at top of [vm_start, vm_end)
    const uint64_t kernel_size = std::max<uint64_t>(params.vm_kernel_min_size, 32ULL << 30);
    struct asahi_vm_create vmc = {};
    vmc.kernel_start = params.vm_end - kernel_size;
    vmc.kernel_end   = params.vm_end;
    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_VM_CREATE, &vmc) != 0) {
        AGX_LOG_ERROR("DRM_IOCTL_ASAHI_VM_CREATE failed: %s", std::strerror(errno));
        agx_drm_close(dev);
        return nullptr;
    }
    dev->vm_id = vmc.vm_id;

    // Initialize partitioned Pillar 4 VA arenas strictly below 0x7000000000
    // (kernel G17 runtime-graph hole [0x7000000000, 0x7100000000)) and above
    // 0x1000000000 render-context aperture.
    const uint64_t slc_pin_base    = 0x4000000000ULL;  // 256 GiB
    const uint64_t slc_pin_size    = 0x0800000000ULL;  // 32 GiB hot working set
    const uint64_t slc_stream_base = 0x5000000000ULL;  // 320 GiB
    const uint64_t slc_stream_size = 0x2000000000ULL;  // 128 GiB cold streaming pool (< 0x7000000000)
    // USC executable window MUST have bit 40 set on G17P / M4 Max (0x10000000000).
    // Start above the kernel-reserved G17P render aux alias [0x10002000000, 0x10002400000).
    dev->va_slc_pin.init(slc_pin_base, slc_pin_size);
    dev->va_slc_stream.init(slc_stream_base, slc_stream_size);
    dev->va_usc_exec.init(AGX_USC_WINDOW_BASE + 0x4000000ULL, AGX_USC_WINDOW_SIZE - 0x4000000ULL);

    // Bind mandatory resident USC base page at 0x10000000000 (HAL200/G17P requirement)
    // and populate its helper tables + multi-cluster return terminals.
    dev->usc_base_bo = alloc_fixed_bo(dev, AGX_PAGE_SIZE, AGX_USC_WINDOW_BASE,
                                      ASAHI_BIND_READ | ASAHI_BIND_WRITE, true, "usc-base-page");
    if (!dev->usc_base_bo) {
        AGX_LOG_ERROR("Failed to bind G17P USC window base page at 0x%llx",
                      (unsigned long long) AGX_USC_WINDOW_BASE);
        agx_drm_close(dev);
        return nullptr;
    }
    g17p_build_usc_base_page(static_cast<uint8_t *>(dev->usc_base_bo->cpu_ptr),
                             dev->usc_base_bo->alloc_size,
                             info.cores_per_cluster, info.num_clusters);

    // Bind mandatory G17P/HAL200 context views at 0x1000000000 + six low identity slots
    // (required by drm_asahi mmu::Vm::install_job_context_aliases before any compute/render kick).
    if (info.gpu_hal_generation == 200u) {
        dev->render_context_bo = alloc_fixed_bo(dev, AGX_G17P_RENDER_CONTEXT_SIZE,
                                                AGX_G17P_RENDER_CONTEXT_BASE,
                                                ASAHI_BIND_READ | ASAHI_BIND_WRITE,
                                                true, "g17p-context", /*shared=*/true);
        if (!dev->render_context_bo) {
            AGX_LOG_ERROR("Failed to bind G17P context BO at 0x%llx",
                          (unsigned long long) AGX_G17P_RENDER_CONTEXT_BASE);
            agx_drm_close(dev);
            return nullptr;
        }
        for (const auto & slot : k_g17p_context_low_slots) {
            if (vm_bind_single(dev, dev->render_context_bo->handle, slot.offset, slot.size,
                               slot.offset, ASAHI_BIND_READ | ASAHI_BIND_WRITE) != 0) {
                AGX_LOG_ERROR("Failed to bind G17P context low slot 0x%llx",
                              (unsigned long long) slot.offset);
                agx_drm_close(dev);
                return nullptr;
            }
        }
    }

    // Bind read-only zero page at 2^32 for safe robustness/null references
    dev->zero_page_bo = alloc_fixed_bo(dev, 65536u, 1ULL << 32, ASAHI_BIND_READ, false, "zero-page");
    if (!dev->zero_page_bo) {
        AGX_LOG_WARN("Failed to bind zero page at 0x100000000; continuing without it");
    }

    // Register zero-ioctl VM sticky-error mirror if kernel supports it
    if (info.has_vm_status_mirror) {
        dev->vm_mirror_bo = alloc_fixed_bo(dev, AGX_PAGE_SIZE, 0, 0, true, "vm-status-mirror");
        if (dev->vm_mirror_bo) {
            struct asahi_vm_status_mirror mirror = {};
            mirror.vm_id  = dev->vm_id;
            mirror.handle = dev->vm_mirror_bo->handle;
            mirror.offset = 0;
            mirror.flags  = 0;

            struct asahi_get_params mp = {};
            mp.param_group = ASAHI_PARAM_GROUP_VM_STATUS_MIRROR;
            mp.pointer     = reinterpret_cast<uint64_t>(&mirror);
            mp.size        = sizeof(mirror);
            if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GET_PARAMS, &mp) == 0) {
                dev->vm_mirror_word = reinterpret_cast<volatile int32_t *>(dev->vm_mirror_bo->cpu_ptr);
            } else {
                free_fixed_bo(dev, dev->vm_mirror_bo);
                dev->vm_mirror_bo = nullptr;
            }
        }
    }

    // Create dedicated compute command queue with bit-40 USC base and extended watchdog
    struct asahi_queue_create qc = {};
    qc.flags              = 0;
    qc.vm_id              = dev->vm_id;
    qc.priority           = ASAHI_PRIORITY_MEDIUM;
    qc.usc_exec_base      = AGX_USC_WINDOW_BASE;
    qc.compute_timeout_ms = info.has_compute_timeout ? 30000u : 0u;
    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_QUEUE_CREATE, &qc) != 0) {
        AGX_LOG_ERROR("DRM_IOCTL_ASAHI_QUEUE_CREATE failed: %s", std::strerror(errno));
        agx_drm_close(dev);
        return nullptr;
    }
    dev->queue_id = qc.queue_id;

    // Create timeline syncobj for queue completion tracking
    struct agx_drm_syncobj_create sc = {};
    if (drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_SYNCOBJ_CREATE, &sc) != 0) {
        AGX_LOG_ERROR("DRM_IOCTL_SYNCOBJ_CREATE failed: %s", std::strerror(errno));
        agx_drm_close(dev);
        return nullptr;
    }
    dev->syncobj = sc.handle;

    AGX_LOG_INFO("Opened %s (%s): %u cores (%u clusters x %u), %u MHz, USC=%u HAL=%u, "
                 "limit=%.2f GiB (client=%.2f GiB), residency=%d timeout=%ums",
                 info.device_name, info.render_node, info.num_cores, info.num_clusters,
                 info.cores_per_cluster, info.max_frequency_khz / 1000u,
                 info.usc_generation, info.gpu_hal_generation,
                 (double) info.memory_limit / (1024.0 * 1024.0 * 1024.0),
                 (double) info.client_memory_limit / (1024.0 * 1024.0 * 1024.0),
                 (int) info.has_gem_residency, qc.compute_timeout_ms);

    return dev;
}

void agx_drm_close(agx_drm_device * dev) {
    if (!dev) {
        return;
    }
    if (dev->syncobj != 0) {
        if (dev->timeline_value > 0) {
            agx_drm_wait_timeline(dev, dev->timeline_value, 2000000000ULL);
        }
        struct agx_drm_syncobj_destroy sd = { dev->syncobj, 0 };
        drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_SYNCOBJ_DESTROY, &sd);
        dev->syncobj = 0;
    }
    if (dev->queue_id != 0) {
        struct asahi_queue_destroy qd = { dev->queue_id, 0 };
        drm_ioctl_retry(dev->fd, IOCTL_ASAHI_QUEUE_DESTROY, &qd);
        dev->queue_id = 0;
    }
    free_fixed_bo(dev, dev->vm_mirror_bo);
    free_fixed_bo(dev, dev->zero_page_bo);
    if (dev->render_context_bo) {
        for (const auto & slot : k_g17p_context_low_slots) {
            vm_bind_single(dev, 0, slot.offset, slot.size, 0, ASAHI_BIND_UNBIND);
        }
        free_fixed_bo(dev, dev->render_context_bo);
    }
    free_fixed_bo(dev, dev->usc_base_bo);
    dev->vm_mirror_bo      = nullptr;
    dev->zero_page_bo      = nullptr;
    dev->render_context_bo = nullptr;
    dev->usc_base_bo       = nullptr;

    if (dev->vm_id != 0) {
        struct asahi_vm_destroy vmd = { dev->vm_id, 0 };
        drm_ioctl_retry(dev->fd, IOCTL_ASAHI_VM_DESTROY, &vmd);
        dev->vm_id = 0;
    }
    if (dev->fd >= 0) {
        close(dev->fd);
        dev->fd = -1;
    }
    delete dev;
}

const agx_drm_device_info & agx_drm_get_info(const agx_drm_device * dev) {
    return dev->info;
}

int agx_drm_get_fd(const agx_drm_device * dev) {
    return dev ? dev->fd : -1;
}

uint32_t agx_drm_get_vm_id(const agx_drm_device * dev) {
    return dev ? dev->vm_id : 0;
}

uint32_t agx_drm_get_queue_id(const agx_drm_device * dev) {
    return dev ? dev->queue_id : 0;
}

// ============================================================================
// BO Allocation & Pillar 4 Cache Policy Partitioning
// ============================================================================

static agx_cache_policy policy_from_flags(uint32_t flags) {
    if (flags & AGX_BO_EXEC) {
        return AGX_CACHE_EXEC;
    }
    if (flags & AGX_BO_CMDSTREAM) {
        return AGX_CACHE_CMDSTREAM;
    }
    if (flags & AGX_BO_SLC_STREAM) {
        return AGX_CACHE_SLC_STREAM;
    }
    return AGX_CACHE_SLC_PIN;
}

static agx_va_arena & arena_for_policy(agx_drm_device * dev, agx_cache_policy policy) {
    switch (policy) {
        case AGX_CACHE_EXEC:       return dev->va_usc_exec;
        case AGX_CACHE_SLC_STREAM: return dev->va_slc_stream;
        case AGX_CACHE_CMDSTREAM:
        case AGX_CACHE_SLC_PIN:
        default:                   return dev->va_slc_pin;
    }
}

static std::atomic<uint64_t> & counter_for_policy(agx_drm_device * dev, agx_cache_policy policy) {
    switch (policy) {
        case AGX_CACHE_EXEC:       return dev->exec_bytes;
        case AGX_CACHE_SLC_STREAM: return dev->slc_stream_bytes;
        case AGX_CACHE_CMDSTREAM:  return dev->cmdstream_bytes;
        case AGX_CACHE_SLC_PIN:
        default:                   return dev->slc_pin_bytes;
    }
}

agx_bo * agx_bo_alloc(agx_drm_device * dev, size_t size, uint32_t flags, const char * label) {
    if (!dev || size == 0) {
        return nullptr;
    }

    const agx_cache_policy policy = policy_from_flags(flags);
    const size_t alloc_size = agx_align_up(size, AGX_PAGE_SIZE);

    // Pillar 4 policy:
    // - SLC_PIN / EXEC / CMDSTREAM: CPU writeback (coherent unified memory) + RESIDENCY_KEEP
    // - SLC_STREAM: write-combine (unless explicit AGX_BO_WRITEBACK), no residency pin,
    //               read-only GPU bind when AGX_BO_READONLY is set.
    const bool use_writeback = (flags & AGX_BO_WRITEBACK) ||
                               (policy != AGX_CACHE_SLC_STREAM) ||
                               (flags & AGX_BO_TIMESTAMPS);
    const bool keep_resident = (policy == AGX_CACHE_SLC_PIN ||
                                policy == AGX_CACHE_EXEC ||
                                policy == AGX_CACHE_CMDSTREAM);

    const bool is_timestamps = (flags & AGX_BO_TIMESTAMPS) != 0;
    struct asahi_gem_create req = {};
    req.size  = alloc_size;
    // drm_asahi requires timestamp buffers to be SHARED (not VM_PRIVATE) for GEM_BIND_OBJECT.
    req.flags = (is_timestamps ? 0u : ASAHI_GEM_VM_PRIVATE) |
                ((use_writeback || is_timestamps) ? ASAHI_GEM_WRITEBACK : 0u);
    req.vm_id = is_timestamps ? 0u : dev->vm_id;

    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GEM_CREATE, &req) != 0) {
        AGX_LOG_ERROR("GEM_CREATE(%s, %zu B, policy=%u) failed: %s",
                      label ? label : "", alloc_size, (unsigned) policy, std::strerror(errno));
        return nullptr;
    }

    // Obtain CPU mapping
    struct asahi_gem_mmap_offset mm = {};
    mm.handle = req.handle;
    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GEM_MMAP_OFFSET, &mm) != 0) {
        AGX_LOG_ERROR("GEM_MMAP_OFFSET(%s) failed: %s", label ? label : "", std::strerror(errno));
        struct agx_drm_gem_close cl = { req.handle, 0 };
        drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);
        return nullptr;
    }

    void * cpu_ptr = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
                          static_cast<off_t>(mm.offset));
    if (cpu_ptr == MAP_FAILED) {
        AGX_LOG_ERROR("mmap(%s, %zu) failed: %s", label ? label : "", alloc_size, std::strerror(errno));
        struct agx_drm_gem_close cl = { req.handle, 0 };
        drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);
        return nullptr;
    }

    if (flags & AGX_BO_ZERO) {
        std::memset(cpu_ptr, 0, alloc_size);
    }

    uint64_t gpu_va = 0;
    uint32_t ts_obj_handle = 0;

    if (flags & AGX_BO_TIMESTAMPS) {
        struct asahi_gem_bind_object bobj = {};
        bobj.op     = ASAHI_BIND_OBJECT_OP_BIND;
        bobj.flags  = ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS;
        bobj.handle = req.handle;
        bobj.vm_id  = 0;
        bobj.offset = 0;
        bobj.range  = alloc_size;
        if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GEM_BIND_OBJECT, &bobj) != 0) {
            AGX_LOG_ERROR("GEM_BIND_OBJECT(TIMESTAMPS) failed: %s", std::strerror(errno));
            munmap(cpu_ptr, alloc_size);
            struct agx_drm_gem_close cl = { req.handle, 0 };
            drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);
            return nullptr;
        }
        ts_obj_handle = bobj.object_handle;
    } else {
        // Allocate GPU VA in the policy-appropriate arena
        {
            std::lock_guard<std::mutex> lock(dev->va_mutex);
            gpu_va = arena_for_policy(dev, policy).alloc(alloc_size, AGX_PAGE_SIZE);
        }
        if (gpu_va == 0) {
            AGX_LOG_ERROR("GPU VA exhaustion in policy=%u arena for %zu bytes (%s)",
                          (unsigned) policy, alloc_size, label ? label : "");
            munmap(cpu_ptr, alloc_size);
            struct agx_drm_gem_close cl = { req.handle, 0 };
            drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);
            return nullptr;
        }

        const uint32_t bind_flags = ASAHI_BIND_READ |
                                    ((flags & AGX_BO_READONLY) ? 0u : ASAHI_BIND_WRITE);
        if (vm_bind_single(dev, req.handle, gpu_va, alloc_size, 0, bind_flags) != 0) {
            {
                std::lock_guard<std::mutex> lock(dev->va_mutex);
                arena_for_policy(dev, policy).free(gpu_va, alloc_size);
            }
            munmap(cpu_ptr, alloc_size);
            struct agx_drm_gem_close cl = { req.handle, 0 };
            drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);
            return nullptr;
        }
    }

    auto * bo = new agx_bo{};
    bo->dev           = dev;
    bo->handle        = req.handle;
    bo->ts_obj_handle = ts_obj_handle;
    bo->gpu_va        = gpu_va;
    bo->cpu_ptr       = cpu_ptr;
    bo->size          = size;
    bo->alloc_size    = alloc_size;
    bo->flags         = flags;
    bo->policy        = policy;
    bo->is_userptr    = false;
    if (label) {
        std::snprintf(bo->label, sizeof(bo->label), "%s", label);
    }

    if (keep_resident && dev->info.has_gem_residency) {
        agx_bo_set_residency(dev, bo, true);
    }

    counter_for_policy(dev, policy).fetch_add(alloc_size, std::memory_order_relaxed);
    return bo;
}

agx_bo * agx_bo_import_userptr(agx_drm_device * dev, void * ptr, size_t size,
                               bool read_only, const char * label) {
    if (!dev || !dev->info.has_userptr || !ptr || size == 0) {
        return nullptr;
    }
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if ((addr & (AGX_PAGE_SIZE - 1u)) != 0 || (size & (AGX_PAGE_SIZE - 1u)) != 0) {
        return nullptr;
    }

    struct asahi_gem_userptr req = {};
    req.userptr = static_cast<uint64_t>(addr);
    req.size    = size;
    req.flags   = ASAHI_GEM_VM_PRIVATE | (read_only ? ASAHI_GEM_USERPTR_READ_ONLY : 0u);
    req.vm_id   = dev->vm_id;

    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GEM_USERPTR, &req) != 0) {
        return nullptr;
    }

    const agx_cache_policy policy = read_only ? AGX_CACHE_SLC_STREAM : AGX_CACHE_SLC_PIN;
    uint64_t gpu_va = 0;
    {
        std::lock_guard<std::mutex> lock(dev->va_mutex);
        gpu_va = arena_for_policy(dev, policy).alloc(size, AGX_PAGE_SIZE);
    }
    if (gpu_va == 0) {
        struct agx_drm_gem_close cl = { req.handle, 0 };
        drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);
        return nullptr;
    }

    const uint32_t bind_flags = ASAHI_BIND_READ | (read_only ? 0u : ASAHI_BIND_WRITE);
    if (vm_bind_single(dev, req.handle, gpu_va, size, 0, bind_flags) != 0) {
        {
            std::lock_guard<std::mutex> lock(dev->va_mutex);
            arena_for_policy(dev, policy).free(gpu_va, size);
        }
        struct agx_drm_gem_close cl = { req.handle, 0 };
        drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);
        return nullptr;
    }

    auto * bo = new agx_bo{};
    bo->dev        = dev;
    bo->handle     = req.handle;
    bo->gpu_va     = gpu_va;
    bo->cpu_ptr    = ptr;
    bo->size       = size;
    bo->alloc_size = size;
    bo->flags      = (read_only ? (AGX_BO_SLC_STREAM | AGX_BO_READONLY) : AGX_BO_SLC_PIN);
    bo->policy     = policy;
    bo->is_userptr = true;
    if (label) {
        std::snprintf(bo->label, sizeof(bo->label), "%s", label);
    }

    counter_for_policy(dev, policy).fetch_add(size, std::memory_order_relaxed);
    return bo;
}

void agx_bo_free(agx_drm_device * dev, agx_bo * bo) {
    if (!dev || !bo) {
        return;
    }

    if (bo->ts_obj_handle != 0) {
        struct asahi_gem_bind_object unbind = {};
        unbind.op            = ASAHI_BIND_OBJECT_OP_UNBIND;
        unbind.object_handle = bo->ts_obj_handle;
        drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GEM_BIND_OBJECT, &unbind);
    }
    if (bo->gpu_va != 0) {
        vm_bind_single(dev, 0, bo->gpu_va, bo->alloc_size, 0, ASAHI_BIND_UNBIND);
        std::lock_guard<std::mutex> lock(dev->va_mutex);
        arena_for_policy(dev, bo->policy).free(bo->gpu_va, bo->alloc_size);
    }
    if (bo->cpu_ptr && !bo->is_userptr) {
        munmap(bo->cpu_ptr, bo->alloc_size);
    }

    struct agx_drm_gem_close cl = { bo->handle, 0 };
    drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_GEM_CLOSE, &cl);

    counter_for_policy(dev, bo->policy).fetch_sub(bo->alloc_size, std::memory_order_relaxed);
    delete bo;
}

int agx_bo_set_residency(agx_drm_device * dev, agx_bo * bo, bool keep) {
    if (!dev || !bo || !dev->info.has_gem_residency || bo->is_userptr) {
        return 0;
    }
    struct asahi_gem_residency req = {};
    req.handle = bo->handle;
    req.flags  = keep ? ASAHI_GEM_RESIDENCY_KEEP : 0u;
    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GEM_RESIDENCY, &req) != 0) {
        return -errno;
    }
    bo->residency_kept = keep;
    return 0;
}

int agx_bo_madvise(agx_drm_device * dev, agx_bo * bo, bool dontneed, bool * retained) {
    if (!dev || !bo || bo->is_userptr) {
        return -EINVAL;
    }
    struct asahi_gem_madvise req = {};
    req.handle = bo->handle;
    req.madv   = dontneed ? ASAHI_GEM_MADV_DONTNEED : ASAHI_GEM_MADV_WILLNEED;
    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GEM_MADVISE, &req) != 0) {
        return -errno;
    }
    if (retained) {
        *retained = (req.retained != 0);
    }
    return 0;
}

void agx_drm_get_memory_stats(const agx_drm_device * dev, agx_drm_memory_stats * out) {
    if (!dev || !out) {
        return;
    }
    out->slc_pin_bytes    = dev->slc_pin_bytes.load(std::memory_order_relaxed);
    out->slc_stream_bytes = dev->slc_stream_bytes.load(std::memory_order_relaxed);
    out->exec_bytes       = dev->exec_bytes.load(std::memory_order_relaxed);
    out->cmdstream_bytes  = dev->cmdstream_bytes.load(std::memory_order_relaxed);
    out->total_allocated_bytes =
        out->slc_pin_bytes + out->slc_stream_bytes + out->exec_bytes + out->cmdstream_bytes;

    const uint64_t lim = dev->info.client_memory_limit
                             ? dev->info.client_memory_limit
                             : (dev->info.memory_limit ? dev->info.memory_limit : (48ULL << 30));
    out->memory_limit_bytes  = lim;
    out->free_estimate_bytes = (lim > out->total_allocated_bytes) ? (lim - out->total_allocated_bytes) : 0;
}

// ============================================================================
// Direct CDM Compute Submission & Timeline Synchronization
// ============================================================================

int agx_drm_submit_cdm(agx_drm_device * dev,
                       uint64_t         cdm_start_va,
                       uint64_t         cdm_end_va,
                       uint32_t         ts_obj_handle,
                       uint32_t         ts_start_offset_B,
                       uint32_t         ts_end_offset_B,
                       uint64_t *       out_timeline_point,
                       uint32_t         compute_flags) {
    if (!dev || cdm_start_va == 0 || cdm_end_va <= cdm_start_va) {
        return -EINVAL;
    }

    std::lock_guard<std::mutex> lock(dev->submit_mutex);

    struct {
        struct asahi_cmd_header  hdr;
        struct asahi_cmd_compute cmd;
    } packet = {};

    packet.hdr.cmd_type    = ASAHI_CMD_COMPUTE;
    packet.hdr.size        = sizeof(struct asahi_cmd_compute);
    packet.hdr.vdm_barrier = ASAHI_BARRIER_NONE;
    packet.hdr.cdm_barrier = (dev->timeline_value == 0) ? ASAHI_BARRIER_NONE : 0u;

    packet.cmd.flags                = compute_flags;
    packet.cmd.cdm_ctrl_stream_base = cdm_start_va;
    packet.cmd.cdm_ctrl_stream_end  = cdm_end_va;
    if (ts_obj_handle != 0) {
        packet.cmd.ts.start.handle = ts_obj_handle;
        packet.cmd.ts.start.offset = ts_start_offset_B;
        packet.cmd.ts.end.handle   = ts_obj_handle;
        packet.cmd.ts.end.offset   = ts_end_offset_B;
    }

    const uint64_t next_point = dev->timeline_value + 1;
    struct asahi_sync out_sync = {};
    out_sync.sync_type      = dev->info.has_cumulative_signals
                                  ? ASAHI_SYNC_TIMELINE_SYNCOBJ_CUMULATIVE
                                  : ASAHI_SYNC_TIMELINE_SYNCOBJ;
    out_sync.handle         = dev->syncobj;
    out_sync.timeline_value = next_point;

    struct asahi_submit sub = {};
    sub.syncs          = reinterpret_cast<uint64_t>(&out_sync);
    sub.cmdbuf         = reinterpret_cast<uint64_t>(&packet);
    sub.flags          = 0;
    sub.queue_id       = dev->queue_id;
    sub.in_sync_count  = 0;
    sub.out_sync_count = 1;
    sub.cmdbuf_size    = sizeof(packet);

    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_SUBMIT, &sub) != 0) {
        const int err = -errno;
        const int32_t vm_err = agx_drm_vm_error(dev);
        AGX_LOG_ERROR("DRM_IOCTL_ASAHI_SUBMIT failed (cdm=[0x%llx,0x%llx)): %s (vm_error=%d)",
                      (unsigned long long) cdm_start_va, (unsigned long long) cdm_end_va,
                      std::strerror(-err), vm_err);
        return err;
    }

    dev->timeline_value = next_point;
    if (out_timeline_point) {
        *out_timeline_point = next_point;
    }
    return 0;
}

int agx_drm_wait_timeline(agx_drm_device * dev, uint64_t timeline_point, uint64_t timeout_ns) {
    if (!dev || timeline_point == 0) {
        return 0;
    }

    int64_t deadline = INT64_MAX;
    if (timeout_ns != UINT64_MAX) {
        const int64_t now = monotonic_now_ns();
        if (timeout_ns < static_cast<uint64_t>(INT64_MAX - now)) {
            deadline = now + static_cast<int64_t>(timeout_ns);
        }
    }

    uint32_t handle = dev->syncobj;
    uint64_t point  = timeline_point;
    struct agx_drm_syncobj_timeline_wait wait = {};
    wait.handles       = reinterpret_cast<uint64_t>(&handle);
    wait.points        = reinterpret_cast<uint64_t>(&point);
    wait.timeout_nsec  = deadline;
    wait.count_handles = 1;
    wait.flags         = AGX_DRM_SYNCOBJ_WAIT_ALL | AGX_DRM_SYNCOBJ_WAIT_FOR_SUBMIT;

    if (drm_ioctl_retry(dev->fd, AGX_IOCTL_DRM_SYNCOBJ_TIMELINE_WAIT, &wait) != 0) {
        return -errno;
    }

    // Per drm_asahi UAPI contract: a signalled fence does NOT imply execution success;
    // always check sticky VM status before trusting GPU outputs.
    const int32_t vm_err = agx_drm_vm_error(dev);
    if (vm_err != 0) {
        AGX_LOG_ERROR("GPU VM %u sticky error after timeline %llu: %d (%s)",
                      dev->vm_id, (unsigned long long) timeline_point, vm_err, std::strerror(-vm_err));
        return vm_err;
    }
    return 0;
}

void agx_drm_sync_all(agx_drm_device * dev) {
    if (!dev) {
        return;
    }
    uint64_t pt = 0;
    {
        std::lock_guard<std::mutex> lock(dev->submit_mutex);
        pt = dev->timeline_value;
    }
    if (pt > 0) {
        agx_drm_wait_timeline(dev, pt, UINT64_MAX);
    }
}

int32_t agx_drm_vm_error(agx_drm_device * dev) {
    if (!dev) {
        return -EINVAL;
    }
    if (dev->vm_mirror_word) {
        return __atomic_load_n(dev->vm_mirror_word, __ATOMIC_ACQUIRE);
    }
    if (!dev->info.has_vm_status) {
        return 0;
    }
    struct asahi_vm_status st = {};
    st.vm_id = dev->vm_id;
    struct asahi_get_params gp = {};
    gp.param_group = ASAHI_PARAM_GROUP_VM_STATUS;
    gp.pointer     = reinterpret_cast<uint64_t>(&st);
    gp.size        = sizeof(st);
    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GET_PARAMS, &gp) != 0) {
        return 0;
    }
    return st.error;
}

uint64_t agx_drm_get_gpu_time_ns(agx_drm_device * dev) {
    if (!dev) {
        return 0;
    }
    struct asahi_get_time req = {};
    if (drm_ioctl_retry(dev->fd, IOCTL_ASAHI_GET_TIME, &req) != 0) {
        return 0;
    }
    return req.gpu_timestamp;
}

uint64_t agx_drm_ticks_to_ns(const agx_drm_device * dev, uint64_t ticks) {
    if (!dev || dev->ts_div == 0) {
        return ticks;
    }
    return (ticks * dev->ts_mul) / dev->ts_div;
}
