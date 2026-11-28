#include "agx-cdm.h"
#include "agx-compiler.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#    include <sched.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#    define AGX_LIKELY(x)   __builtin_expect(!!(x), 1)
#    define AGX_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#    define AGX_LIKELY(x)   (x)
#    define AGX_UNLIKELY(x) (x)
#endif

struct agx_ioto_reloc {
    uint16_t byte_off;
    uint16_t dest_half;
    uint32_t root_delta;
    uint64_t dest_mask_lo;
    uint64_t dest_mask_hi;
};

struct agx_ioto_template {
    uint64_t                    shader_va          = 0;
    uint64_t                    rodata_va          = 0;
    uint64_t                    const_data_va      = 0;
    uint32_t                    max_push_dma_bytes = 0;
    uint32_t                    push_offset_bytes  = 0;
    uint32_t                    alloc_root_min     = 64u;
    uint32_t                    launch_w0_1d       = 0;
    uint32_t                    launch_w0_nd       = 0;
    uint32_t                    launch_w1          = 0;
    uint32_t                    local_size[3]      = { 1u, 1u, 1u };
    bool                        uses_root_or_dma   = false;
    bool                        needs_full_zero    = false;
    bool                        local_size_valid   = false;
    uint8_t                     reloc_count        = 0;
    agx_ioto_reloc              reloc0             = { 0, 0, 0, 0, 0 };
    std::vector<uint8_t>        bytes;
    std::vector<agx_ioto_reloc> relocs;
};

struct agx_ioto_fast_entry {
    const agx_kernel *  kernel      = nullptr;
    uint64_t            shader_va   = 0;
    uint64_t            rodata_va   = 0;
    agx_ioto_template * tmpl        = nullptr;
    bool                is_indirect = false;
};

struct agx_cdm_replay_tag {
    bool         valid      = false;
    uint64_t     graph_uid  = 0;
    int          n_nodes    = 0;
    const void * first_node = nullptr;
    const void * last_node  = nullptr;
    uint64_t     bo_epoch   = 0;
    uint64_t     record_gen = 0;
    uint64_t     hit_count  = 0;
};

namespace {

// ============================================================================
// Exact HAL200 / G17P (M4 Max / AGX3B / GPU0) Command Stream & Ioto Constants
// Verified byte-for-byte against Aurora Mesa AGX3 encoders
// ============================================================================

constexpr uint32_t AGX3_CDM_DIRECT_LAUNCH_BYTES   = 40u;
constexpr uint32_t AGX3_CDM_INDIRECT_LAUNCH_BYTES = 24u;
constexpr uint32_t AGX3_CDM_BARRIER_BYTES         = 4u;
constexpr uint32_t AGX3_CDM_TERMINATE_BYTES       = 4u;
constexpr uint32_t AGX3_CDM_TERMINATE_WORD        = 0x40000000u;
constexpr uint32_t AGX3_CDM_INDIRECT_MODE2        = 0x10000000u;

constexpr uint32_t AGX3_CDM_BARRIER_WEAK_WORD   = 0x60000160u;
constexpr uint32_t AGX3_CDM_BARRIER_STRONG_WORD = 0x60001560u;
constexpr uint32_t AGX3_CDM_BARRIER_MEMORY_WORD = 0x6000157cu;

constexpr uint32_t AGX_IOTO_PROGRAM_ALIGN       = 64u;
constexpr uint32_t AGX_IOTO_SPECLM_SIZE         = 8u;
constexpr uint32_t AGX_IOTO_LDSHDR_SIZE         = 10u;
constexpr uint32_t AGX_IOTO_SPECTPR_SIZE        = 10u;
constexpr uint32_t AGX_IOTO_STATE_TAIL_SIZE     = 28u;
constexpr uint32_t AGX_IOTO_TERMINATE_SIZE      = 4u;

constexpr uint8_t AGX_IOTO_LDSHDR_F1_DIRECT   = 0u;
constexpr uint8_t AGX_IOTO_LDSHDR_F1_INDIRECT = 3u;

static const uint8_t k_e4[4] = { 2, 0, 1, 3 };

static const uint8_t k_g17p_profile_prefix[48] = {
    0x1c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x14, 0x81, 0x11, 0x06, 0x00, 0x00, 0x00, 0x00,
    0x0c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x08, 0xa8,
    0x10, 0x05, 0x1c, 0x80, 0x02, 0x00, 0x04, 0x00,
    0x00, 0x00, 0x0f, 0x12, 0x54, 0x00, 0x4c, 0x00,
};

static const uint8_t k_g17p_profile_suffix[10] = {
    0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x60, 0x00,
};

static inline void store_le16(uint8_t * dst, uint16_t v) {
    dst[0] = static_cast<uint8_t>(v);
    dst[1] = static_cast<uint8_t>(v >> 8);
}

static inline void store_le32(uint8_t * dst, uint32_t v) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(dst, &v, sizeof(v));
#else
    dst[0] = static_cast<uint8_t>(v);
    dst[1] = static_cast<uint8_t>(v >> 8);
    dst[2] = static_cast<uint8_t>(v >> 16);
    dst[3] = static_cast<uint8_t>(v >> 24);
#endif
}

static inline void store_le64(uint8_t * dst, uint64_t v) {
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(dst, &v, sizeof(v));
#else
    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<uint8_t>(v >> (8 * i));
    }
#endif
}

static inline uint32_t load_le32(const uint8_t * src) {
    return static_cast<uint32_t>(src[0]) |
           (static_cast<uint32_t>(src[1]) << 8) |
           (static_cast<uint32_t>(src[2]) << 16) |
           (static_cast<uint32_t>(src[3]) << 24);
}

// ============================================================================
// G17P / GPU0 Ioto Record Packers
// ============================================================================

static bool pack_g17p_state_tail(uint64_t shader_va,
                                 uint8_t  ldshdr_f1,
                                 uint32_t threadgroup_bytes,
                                 uint32_t scratch_bytes,
                                 uint32_t staging_halfs,
                                 uint8_t  out[AGX_IOTO_STATE_TAIL_SIZE]) {
    if (shader_va == 0 || (shader_va & 0x3fULL) != 0 || shader_va > 0xffffffffffffULL) {
        return false;
    }
    if (ldshdr_f1 > 3 || scratch_bytes > (65535u * 4u)) {
        return false;
    }

    // 1. SPECLM (8 bytes)
    uint16_t speclm[4] = { 0x0077u, 0x412au, 0x0000u, 0x0000u };
    if (threadgroup_bytes > 0) {
        uint32_t tg = std::max<uint32_t>(256u, static_cast<uint32_t>(agx_align_up(threadgroup_bytes, 64u)));
        if (tg > AGX_MAX_TG_SRAM_BYTES) {
            return false;
        }
        speclm[2] = static_cast<uint16_t>((1u << 7) | (((tg >> 6) & 0xffu) << 8));
        speclm[3] = static_cast<uint16_t>((tg >> 14) & 0x3u);
    }
    for (int i = 0; i < 4; ++i) {
        store_le16(out + 2 * i, speclm[i]);
    }

    // 2. LDSHDR (10 bytes)
    const uint16_t ldshdr[5] = {
        0x0177u,
        static_cast<uint16_t>(0x002au | (((shader_va >> 6) & 0x1ffu) << 7)),
        static_cast<uint16_t>(shader_va >> 15),
        static_cast<uint16_t>(shader_va >> 31),
        static_cast<uint16_t>(((shader_va >> 47) & 1u) | (static_cast<uint16_t>(k_e4[ldshdr_f1]) << 1)),
    };
    for (int i = 0; i < 5; ++i) {
        store_le16(out + AGX_IOTO_SPECLM_SIZE + 2 * i, ldshdr[i]);
    }

    // 3. SPECTPR (10 bytes)
    const uint32_t compiler_d  = staging_halfs * 2u;
    const uint16_t spill_quads = static_cast<uint16_t>((scratch_bytes + 3u) >> 2);
    const uint16_t d_rounded   = static_cast<uint16_t>(((((uint64_t) compiler_d + 3u) >> 2) + 1u) & 0xffeu);
    const uint16_t spectpr[5]  = {
        0x00f7u,
        static_cast<uint16_t>(0x002au | ((d_rounded & 2u) << 14)),
        static_cast<uint16_t>(((d_rounded >> 2) & 0x3ffu) | ((spill_quads & 0x3fu) << 10)),
        static_cast<uint16_t>(((spill_quads >> 6) & 0x3ffu) | ((spill_quads & 0x3fu) << 10)),
        static_cast<uint16_t>((spill_quads >> 6) & 0x3ffu),
    };
    for (int i = 0; i < 5; ++i) {
        store_le16(out + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE + 2 * i, spectpr[i]);
    }
    return true;
}

static inline uint64_t ldimm_dest_mask(uint16_t dest_half) {
    return static_cast<uint64_t>(0x800cu | (((dest_half >> 1) & 0xfu) << 4)) |
           (static_cast<uint64_t>(0x0002u | ((dest_half & 1u) << 2) |
                                  (((dest_half >> 5) & 3u) << 6) |
                                  (((dest_half >> 7) & 1u) << 3)) << 16);
}

static inline uint64_t pack_ldimm_val(uint32_t imm) {
    return (static_cast<uint64_t>(imm & 0x7fu) << 8) |
           (static_cast<uint64_t>((imm >> 25) & 0x7fu) << 25) |
           (static_cast<uint64_t>((imm >> 7) & 0xfu) << 33) |
           (static_cast<uint64_t>((imm >> 11) & 3u) << 42) |
           (static_cast<uint64_t>((imm >> 13) & 0xfffu) << 48);
}

static inline void pack_ldimm_long(uint16_t dest_half, uint32_t imm, uint8_t out[8]) {
    store_le64(out, ldimm_dest_mask(dest_half) | pack_ldimm_val(imm));
}

static bool pack_gpu0_lddev14(uint16_t data_half, uint16_t base_half,
                              int16_t byte_off, uint8_t count_dwords, uint8_t out[14]) {
    if (data_half >= 192 || base_half > 188 || (base_half & 1u) != 0 ||
        count_dwords < 1 || count_dwords > 4) {
        return false;
    }
    static const uint8_t k_masks[4] = { 0x51, 0x59, 0x5d, 0x57 };
    const uint16_t off = static_cast<uint16_t>(byte_off);
    out[0]  = 0x67;
    out[1]  = 0x00;
    out[2]  = 0x54;
    out[3]  = static_cast<uint8_t>(data_half);
    out[4]  = static_cast<uint8_t>(base_half >> 1);
    out[5]  = 0x00;
    out[6]  = 0x00;
    out[7]  = 0x00;
    out[8]  = k_masks[count_dwords - 1];
    out[9]  = static_cast<uint8_t>(off << 5);
    out[10] = static_cast<uint8_t>(off >> 3);
    out[11] = static_cast<uint8_t>(0x40u | (off >> 11));
    out[12] = 0x22;
    out[13] = 0x00;
    return true;
}

static bool pack_gpu0_copy(uint16_t dest_half, uint16_t src_half,
                           uint8_t out[10], uint8_t * out_len) {
    if (dest_half >= 256 || src_half >= 256) {
        return false;
    }
    if (dest_half < 128) {
        out[0] = static_cast<uint8_t>(0x0bu | ((dest_half & 0x1eu) << 3));
        out[1] = static_cast<uint8_t>(src_half & 0x7fu);
        out[2] = static_cast<uint8_t>(0x09u | ((dest_half & 0x60u) << 1));
        out[3] = static_cast<uint8_t>(0x04u | (dest_half & 1u) | ((src_half & 0x80u) >> 3));
        *out_len = 4;
        return true;
    }
    out[0] = static_cast<uint8_t>(0x0bu | ((dest_half & 0x1eu) << 3));
    out[1] = static_cast<uint8_t>(0x01u | (src_half & 0x7eu));
    out[2] = static_cast<uint8_t>(0x0fu | ((dest_half & 0x60u) << 1));
    out[3] = 0x00;
    out[4] = static_cast<uint8_t>(0x40u | ((dest_half & 1u) << 2) |
                                  ((src_half & 1u) << 3) | ((dest_half & 0x80u) >> 6));
    out[5] = static_cast<uint8_t>(0x10u | ((src_half & 0x80u) >> 7));
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    *out_len = 10;
    return true;
}

struct uniform_range {
    uint16_t start_halfs;
    uint16_t size_halfs;
    uint64_t address;
    bool     is_root_rel;
    uint32_t root_delta;
};

struct queued_load {
    uint16_t base_half;
    uint16_t data_half;
    int16_t  byte_off;
    uint8_t  count_dwords;
    uint16_t dest_uniform_half;
};

static bool compose_g17p_ioto_template(const agx_kernel *    kernel,
                                       const uniform_range * ranges,
                                       size_t                range_count,
                                       bool                  is_indirect,
                                       agx_ioto_template &   out_tmpl) {
    out_tmpl.shader_va = kernel->shader_va;
    out_tmpl.rodata_va = (kernel->rodata_size_16 != 0 && kernel->code_bo)
                             ? (kernel->code_bo->gpu_va + kernel->rodata_offset)
                             : 0ULL;
    out_tmpl.bytes.clear();
    out_tmpl.relocs.clear();

    uint8_t tail[AGX_IOTO_STATE_TAIL_SIZE];
    if (!pack_g17p_state_tail(kernel->shader_va,
                              is_indirect ? AGX_IOTO_LDSHDR_F1_INDIRECT : AGX_IOTO_LDSHDR_F1_DIRECT,
                              kernel->threadgroup_bytes,
                              kernel->scratch_bytes,
                              kernel->staging_halfs,
                              tail)) {
        return false;
    }

    // No-resource fast path (exact 90-byte G17P compute program)
    if (range_count == 0) {
        const size_t total = sizeof(tail) + sizeof(k_g17p_profile_prefix) +
                             sizeof(k_g17p_profile_suffix) + AGX_IOTO_TERMINATE_SIZE;
        out_tmpl.bytes.resize(total);
        size_t off = 0;
        std::memcpy(out_tmpl.bytes.data() + off, tail, sizeof(tail));
        off += sizeof(tail);
        std::memcpy(out_tmpl.bytes.data() + off, k_g17p_profile_prefix, sizeof(k_g17p_profile_prefix));
        off += sizeof(k_g17p_profile_prefix);
        std::memcpy(out_tmpl.bytes.data() + off, k_g17p_profile_suffix, sizeof(k_g17p_profile_suffix));
        off += sizeof(k_g17p_profile_suffix);
        out_tmpl.bytes[off + 0] = 0x0e;
        out_tmpl.bytes[off + 1] = 0x00;
        out_tmpl.bytes[off + 2] = 0x00;
        out_tmpl.bytes[off + 3] = 0x00;
        return true;
    }

    // Build pre-copy body (LDIMM + LDDEV + 4 round NOPs) and final COPY suffix.
    // Supports automatic multi-round rollover if > 97 GPR dwords are needed.
    std::vector<uint8_t> pre_copy;
    std::vector<uint8_t> final_copies;
    pre_copy.reserve(512);
    final_copies.reserve(256);

    std::vector<queued_load> pending;
    uint8_t  cur_gpr_dwords      = 0;
    uint64_t cached_base_va      = 0;
    uint16_t cached_base_half    = 0;
    bool     has_cached_base     = false;
    bool     cached_base_is_root = false;

    auto flush_round = [&](bool into_final_copies) -> bool {
        for (const auto & ld : pending) {
            uint8_t rec[14];
            if (!pack_gpu0_lddev14(ld.data_half, ld.base_half, ld.byte_off, ld.count_dwords, rec)) {
                return false;
            }
            pre_copy.insert(pre_copy.end(), rec, rec + sizeof(rec));
        }
        // 4 GPU0 NOPs: first waits on physical slot 2 (hazard_for_value[1] = 2 -> bit 11 = 0x0800 | 0x0006)
        uint8_t nops[8];
        store_le16(nops + 0, pending.empty() ? 0x0006u : 0x0806u);
        store_le16(nops + 2, 0x0006u);
        store_le16(nops + 4, 0x0006u);
        store_le16(nops + 6, 0x0006u);
        pre_copy.insert(pre_copy.end(), nops, nops + 8);

        auto & copy_dst = into_final_copies ? final_copies : pre_copy;
        for (const auto & ld : pending) {
            for (uint8_t d = 0; d < ld.count_dwords; ++d) {
                uint8_t cp[10];
                uint8_t cp_len = 0;
                if (!pack_gpu0_copy(static_cast<uint16_t>(ld.dest_uniform_half + 2u * d),
                                    static_cast<uint16_t>(ld.data_half + 2u * d),
                                    cp, &cp_len)) {
                    return false;
                }
                copy_dst.insert(copy_dst.end(), cp, cp + cp_len);
            }
        }
        pending.clear();
        cur_gpr_dwords      = 0;
        has_cached_base     = false;
        cached_base_is_root = false;
        return true;
    };

    for (size_t r = 0; r < range_count; ++r) {
        const auto & rng = ranges[r];
        if (rng.size_halfs == 0 || (rng.start_halfs & 1u) != 0 || (rng.size_halfs & 1u) != 0 ||
            (rng.address & 3u) != 0) {
            return false;
        }
        const uint32_t total_dwords = rng.size_halfs / 2u;
        uint32_t dw = 0;
        while (dw < total_dwords) {
            const uint8_t  cnt = static_cast<uint8_t>(std::min<uint32_t>(4u, total_dwords - dw));
            const uint64_t addr = rng.address + static_cast<uint64_t>(dw) * 4u;
            const uint32_t root_delta = rng.root_delta + dw * 4u;
            const uint8_t  reserve_dwords = static_cast<uint8_t>(cnt + 2u); // 2 dwords base + cnt data

            // GPU0 limit is 97 GPR dwords per round
            if (static_cast<unsigned>(cur_gpr_dwords) + reserve_dwords >= 97u) {
                if (!flush_round(false)) {
                    return false;
                }
            }

            const uint16_t base_temp_half = static_cast<uint16_t>(cur_gpr_dwords * 2u);
            const uint16_t data_temp_half = static_cast<uint16_t>(base_temp_half + 4u);
            cur_gpr_dwords = static_cast<uint8_t>(cur_gpr_dwords + reserve_dwords);

            uint16_t use_base_half = base_temp_half;
            int16_t  delta = 0;
            if (has_cached_base && (rng.is_root_rel == cached_base_is_root) &&
                addr >= cached_base_va && (addr - cached_base_va) <= 32767ULL) {
                use_base_half = cached_base_half;
                delta = static_cast<int16_t>(addr - cached_base_va);
            } else {
                const size_t ldimm_off = pre_copy.size();
                if (rng.is_root_rel) {
                    if (ldimm_off > UINT16_MAX) {
                        return false;
                    }
                    out_tmpl.relocs.push_back({
                        static_cast<uint16_t>(ldimm_off),
                        base_temp_half,
                        root_delta,
                        ldimm_dest_mask(base_temp_half),
                        ldimm_dest_mask(static_cast<uint16_t>(base_temp_half + 2u)),
                    });
                }
                uint8_t lo[8], hi[8];
                pack_ldimm_long(base_temp_half, static_cast<uint32_t>(addr), lo);
                pack_ldimm_long(static_cast<uint16_t>(base_temp_half + 2u),
                                static_cast<uint32_t>(addr >> 32), hi);
                pre_copy.insert(pre_copy.end(), lo, lo + 8);
                pre_copy.insert(pre_copy.end(), hi, hi + 8);
                cached_base_va      = addr;
                cached_base_half    = base_temp_half;
                has_cached_base     = true;
                cached_base_is_root = rng.is_root_rel;
            }

            pending.push_back({
                use_base_half,
                data_temp_half,
                delta,
                cnt,
                static_cast<uint16_t>(rng.start_halfs + 2u * dw),
            });
            dw += cnt;
        }
    }

    if (!flush_round(true)) {
        return false;
    }

    const size_t total = pre_copy.size() + sizeof(tail) + sizeof(k_g17p_profile_prefix) +
                         final_copies.size() + sizeof(k_g17p_profile_suffix) + AGX_IOTO_TERMINATE_SIZE;
    out_tmpl.bytes.resize(total);

    size_t off = 0;
    std::memcpy(out_tmpl.bytes.data() + off, pre_copy.data(), pre_copy.size());
    off += pre_copy.size();
    std::memcpy(out_tmpl.bytes.data() + off, tail, sizeof(tail));
    off += sizeof(tail);
    std::memcpy(out_tmpl.bytes.data() + off, k_g17p_profile_prefix, sizeof(k_g17p_profile_prefix));
    off += sizeof(k_g17p_profile_prefix);
    std::memcpy(out_tmpl.bytes.data() + off, final_copies.data(), final_copies.size());
    off += final_copies.size();
    std::memcpy(out_tmpl.bytes.data() + off, k_g17p_profile_suffix, sizeof(k_g17p_profile_suffix));
    off += sizeof(k_g17p_profile_suffix);
    out_tmpl.bytes[off + 0] = 0x0e;
    out_tmpl.bytes[off + 1] = 0x00;
    out_tmpl.bytes[off + 2] = 0x00;
    out_tmpl.bytes[off + 3] = 0x00;
    return true;
}

static bool g_agx_stats_enabled() {
    static const bool enabled = []() {
        const char * v = std::getenv("GGML_AGX_STATS");
        return v && v[0] && v[0] != '0';
    }();
    return enabled;
}

} // namespace

// ============================================================================
// Low-level HAL200 / G17P CDM Packet Encoders
// ============================================================================

uint32_t agx_cdm_encode_barrier_word(agx_barrier_mode mode) {
    switch (mode) {
        case AGX_BARRIER_MEMORY_HOST: return AGX3_CDM_BARRIER_MEMORY_WORD;
        case AGX_BARRIER_STRONG_USC:  return AGX3_CDM_BARRIER_STRONG_WORD;
        case AGX_BARRIER_WEAK_WFI:
        case AGX_BARRIER_NONE:
        default:                      return AGX3_CDM_BARRIER_WEAK_WORD;
    }
}

static void build_template_words(uint32_t ti_spill_bytes, uint32_t ti_ipr_bytes,
                                 bool is_1d, uint32_t * out_w0, uint32_t * out_w1) {
    const uint32_t w1_quads = (ti_spill_bytes + 3u) >> 2;
    const uint32_t w0_quads = (((ti_ipr_bytes + 3u) >> 2) + 1u) & ~1u;
    uint32_t w0 = ((w0_quads & 0xfffu) << 3) | (1u << 19); // subkernel_x_target_workitems = 1
    if (is_1d) {
        w0 &= 0xfff1ffffu;
    }
    *out_w0 = w0;
    *out_w1 = (w1_quads & 0x3ffffu) << 3;
}

bool agx_cdm_pack_direct_launch(uint64_t       program_va,
                                const uint32_t global_size[3],
                                const uint32_t local_size[3],
                                uint32_t       ti_spill_bytes,
                                uint32_t       ti_ipr_bytes,
                                uint8_t        out[40]) {
    if (!out || program_va == 0 || (program_va & 0x3fULL) != 0 ||
        program_va >= (1ULL << 48)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (global_size[i] == 0 || local_size[i] == 0 || local_size[i] > 0x7ffu) {
            return false;
        }
    }

    const bool is_1d = (global_size[1] < 2 && global_size[2] < 2);
    uint32_t w0 = 0, w1 = 0;
    build_template_words(ti_spill_bytes, ti_ipr_bytes, is_1d, &w0, &w1);

    store_le32(out + 0,  w0);
    store_le32(out + 4,  static_cast<uint32_t>(((program_va >> 38) & 0x3ffu) << 22) | w1);
    store_le32(out + 8,  static_cast<uint32_t>(program_va >> 6));
    store_le32(out + 12, 0x40000000u | static_cast<uint32_t>((program_va >> 40) & 0xffu));

    for (int i = 0; i < 3; ++i) {
        store_le32(out + 16 + i * 4, global_size[i]);
        store_le32(out + 28 + i * 4, local_size[i]);
    }
    return true;
}

bool agx_cdm_pack_indirect_launch(uint64_t program_va,
                                  uint64_t geometry_va,
                                  uint32_t ti_spill_bytes,
                                  uint32_t ti_ipr_bytes,
                                  uint8_t  out[24]) {
    if (!out || program_va == 0 || (program_va & 0x3fULL) != 0 || program_va >= (1ULL << 48) ||
        geometry_va == 0 || (geometry_va & 3ULL) != 0 || geometry_va >= (1ULL << 48)) {
        return false;
    }

    uint32_t w0 = 0, w1 = 0;
    build_template_words(ti_spill_bytes, ti_ipr_bytes, false, &w0, &w1);

    store_le32(out + 0,  w0 | AGX3_CDM_INDIRECT_MODE2);
    store_le32(out + 4,  static_cast<uint32_t>(((program_va >> 38) & 0x3ffu) << 22) | w1);
    store_le32(out + 8,  static_cast<uint32_t>(program_va >> 6));
    store_le32(out + 12, 0x40000000u | static_cast<uint32_t>((program_va >> 40) & 0xffu));
    store_le32(out + 16, static_cast<uint32_t>(geometry_va >> 32));
    store_le32(out + 20, static_cast<uint32_t>(geometry_va));
    return true;
}

// ============================================================================
// agx_cdm_stream Implementation
// ============================================================================

struct agx_cdm_stream {
    agx_drm_device * dev              = nullptr;
    agx_bo *         cmd_bo           = nullptr;
    agx_bo *         ioto_bo          = nullptr;
    agx_bo *         arg_bo           = nullptr;
    agx_bo *         ts_bo            = nullptr;
    agx_bo *         sense_barrier_bo = nullptr;

    size_t           cmd_offset       = 0;
    size_t           ioto_offset      = 0;
    size_t           arg_offset       = 0;
    bool             arg_bo_single_hi32 = false;

    // Barrier tracking (enforces HAL200 rule: never two adjacent CDM barrier records)
    size_t           last_barrier_off = SIZE_MAX;
    agx_barrier_mode last_barrier_mode = AGX_BARRIER_NONE;
    bool             last_launch_elidable = false;
    bool             terminated       = false;
    bool             rings_dirty      = false;
    bool             has_external_args = false;
    bool             last_submit_has_ts = false;
    bool             allow_replay     = true;

    uint64_t         in_flight_point  = 0;
    uint64_t         record_gen       = 0;
    uint64_t         submits_since_reset = 0;
    agx_cdm_stats    stats            = {};

    // Per-(shader_va, is_indirect) pre-templated G17P Ioto programs (zero heap allocs/dispatch!)
    std::unordered_map<uint64_t, agx_ioto_template> ioto_templates;
    agx_ioto_fast_entry fast_cache[256] = {};

    // Steady-state graph replay state
    agx_cdm_replay_tag replay{};
    uint64_t           last_rec_disp    = 0;
    uint64_t           last_rec_ind     = 0;
    uint64_t           last_rec_usc     = 0;
    uint64_t           last_rec_elid    = 0;

    // Lightweight GGML_AGX_STATS=1 telemetry (CPU encode vs GPU execute split)
    std::chrono::steady_clock::time_point reset_tp{};
    agx_cdm_stats       stats_at_reset{};
    bool                has_reset_tp = false;

    // Wave 9 Breakthrough A: 4-tier cmd_offset EMA + P-core clock-gated WFE wait
    // Tiers: 0 (<1KB Inject), 1 (<8KB Draft), 2 (<48KB Verify/Decode), 3 (>=48KB Prefill)
    double              ema_wait_us[4]  = { 0.0, 0.0, 0.0, 0.0 };
    uint32_t            last_tier       = 0;
    std::chrono::steady_clock::time_point submit_tp{};
    double              last_ioctl_ms   = 0.0;
};

agx_cdm_stream * agx_cdm_stream_create(agx_drm_device * dev,
                                       size_t           cmd_bytes,
                                       size_t           ioto_bytes,
                                       size_t           arg_bytes) {
    if (!dev) {
        return nullptr;
    }

    auto * s = new agx_cdm_stream{};
    s->dev = dev;

    s->cmd_bo = agx_bo_alloc(dev, std::max<size_t>(cmd_bytes, 65536u),
                             AGX_BO_CMDSTREAM | AGX_BO_ZERO, "cdm-cmdstream");
    s->ioto_bo = agx_bo_alloc(dev, std::max<size_t>(ioto_bytes, 256u * 1024u),
                              AGX_BO_EXEC | AGX_BO_ZERO, "cdm-ioto-arena");
    s->arg_bo = agx_bo_alloc(dev, std::max<size_t>(arg_bytes, 512u * 1024u),
                             AGX_BO_CMDSTREAM | AGX_BO_ZERO, "cdm-arg-ring");
    s->ts_bo = agx_bo_alloc(dev, AGX_PAGE_SIZE, AGX_BO_TIMESTAMPS | AGX_BO_ZERO, "cdm-timestamps");
    s->sense_barrier_bo = agx_bo_alloc(dev, AGX_PAGE_SIZE,
                                       AGX_BO_SLC_PIN | AGX_BO_ZERO, "cdm-sense-barrier");

    if (!s->cmd_bo || !s->ioto_bo || !s->arg_bo || !s->sense_barrier_bo) {
        AGX_LOG_ERROR("Failed to allocate CDM stream backing BOs");
        agx_cdm_stream_destroy(s);
        return nullptr;
    }
    const uint64_t arg_beg_hi = s->arg_bo->gpu_va >> 32;
    const uint64_t arg_end_hi = (s->arg_bo->gpu_va + s->arg_bo->alloc_size - 1u) >> 32;
    s->arg_bo_single_hi32 = (arg_beg_hi == arg_end_hi);
    return s;
}

void agx_cdm_stream_destroy(agx_cdm_stream * s) {
    if (!s) {
        return;
    }
    if (s->in_flight_point != 0) {
        agx_drm_wait_timeline(s->dev, s->in_flight_point, 2000000000ULL);
    }
    agx_bo_free(s->dev, s->sense_barrier_bo);
    agx_bo_free(s->dev, s->ts_bo);
    agx_bo_free(s->dev, s->arg_bo);
    agx_bo_free(s->dev, s->ioto_bo);
    agx_bo_free(s->dev, s->cmd_bo);
    delete s;
}

void agx_cdm_stream_reset(agx_cdm_stream * s) {
    if (!s) {
        return;
    }
    if (s->in_flight_point != 0) {
        agx_drm_wait_timeline(s->dev, s->in_flight_point, UINT64_MAX);
        s->in_flight_point = 0;
    }
    s->cmd_offset           = 0;
    s->ioto_offset          = 0;
    s->arg_offset           = 0;
    s->last_barrier_off     = SIZE_MAX;
    s->last_barrier_mode    = AGX_BARRIER_NONE;
    s->last_launch_elidable = false;
    s->terminated           = false;
    s->has_external_args    = false;
    s->allow_replay         = true;
    s->submits_since_reset  = 0;
    s->record_gen++;
    s->replay.valid         = false;
    if (g_agx_stats_enabled()) {
        s->reset_tp       = std::chrono::steady_clock::now();
        s->stats_at_reset = s->stats;
        s->has_reset_tp   = true;
    }
}

static uint64_t alloc_args_internal(agx_cdm_stream * s, size_t bytes, size_t align, void ** out_cpu_ptr) {
    if (!s || !s->arg_bo || bytes == 0) {
        return 0;
    }
    align = std::max<size_t>(align, 8u);
    const size_t off = agx_align_up(s->arg_offset, align);
    if (off + bytes > s->arg_bo->alloc_size) {
        AGX_LOG_ERROR("CDM arg ring overflow (need %zu at %zu / %zu)", bytes, off, s->arg_bo->alloc_size);
        return 0;
    }
    s->arg_offset = off + bytes;
    if (out_cpu_ptr) {
        *out_cpu_ptr = static_cast<uint8_t *>(s->arg_bo->cpu_ptr) + off;
    }
    return s->arg_bo->gpu_va + off;
}

uint64_t agx_cdm_alloc_args(agx_cdm_stream * s, size_t bytes, size_t align, void ** out_cpu_ptr) {
    const uint64_t va = alloc_args_internal(s, bytes, align, out_cpu_ptr);
    if (va != 0) {
        s->has_external_args = true;
    }
    return va;
}

static bool maybe_auto_flush_chunk(agx_cdm_stream * s,
                                   size_t           need_cmd,
                                   size_t           need_ioto,
                                   size_t           need_arg) {
    if (!s || s->terminated || s->cmd_offset == 0 || s->has_external_args) {
        return true;
    }
    const size_t cmd_need  = s->cmd_offset + need_cmd + AGX3_CDM_BARRIER_BYTES + AGX3_CDM_TERMINATE_BYTES;
    const size_t ioto_need = agx_align_up(s->ioto_offset, AGX_IOTO_PROGRAM_ALIGN) + need_ioto;
    const size_t arg_need  = agx_align_up(s->arg_offset, 16u) + need_arg;
    if (cmd_need <= s->cmd_bo->alloc_size &&
        ioto_need <= s->ioto_bo->alloc_size &&
        arg_need <= s->arg_bo->alloc_size) {
        return true;
    }
    if (agx_cdm_submit(s, true, nullptr) != 0) {
        AGX_LOG_ERROR("CDM auto-flush chunk submit failed");
        return false;
    }
    agx_cdm_stream_reset(s);
    return true;
}

static const agx_ioto_template * get_or_build_ioto_template(agx_cdm_stream *   s,
                                                            const agx_kernel * kernel,
                                                            bool               is_indirect) {
    const uint64_t rodata_va = (AGX_UNLIKELY(kernel->rodata_size_16 != 0) && kernel->code_bo)
                                   ? (kernel->code_bo->gpu_va + kernel->rodata_offset)
                                   : 0ULL;

    // 256-slot direct-mapped L1 pointer cache with Fibonacci hash (~99.8% hit rate across 55+ kernels)
    const size_t fidx = static_cast<size_t>(
        ((static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kernel)) * 0x9e3779b97f4a7c15ULL) >> 56) ^
        (is_indirect ? 0x55u : 0u)) & 255u;
    agx_ioto_fast_entry & fe = s->fast_cache[fidx];
    if (AGX_LIKELY(fe.kernel == kernel && fe.is_indirect == is_indirect && fe.tmpl != nullptr &&
                   fe.shader_va == kernel->shader_va && fe.rodata_va == rodata_va)) {
        return fe.tmpl;
    }

    const uint64_t key = (kernel->shader_va << 1) | (is_indirect ? 1ULL : 0ULL);
    auto it = s->ioto_templates.find(key);
    if (it != s->ioto_templates.end() &&
        it->second.shader_va == kernel->shader_va &&
        it->second.rodata_va == rodata_va) {
        fe = { kernel, kernel->shader_va, rodata_va, &it->second, is_indirect };
        return fe.tmpl;
    }

    // Build Ioto uniform range list in optimal base-sharing order:
    //   1. `uses_root` (root_va + 0)
    //   2. `uniform_dma` spans sorted by `offset_B` ascending (all reuse root_va base -> 1 LDIMM pair!)
    //   3. `rodata` (static code_bo VA baked into template -> 0 runtime relocations!)
    // Bake `arg_bo`'s upper 32 bits into the template so hot-path patching only writes lo32!
    const uint64_t kDummyRootVa = (s->arg_bo_single_hi32 && s->arg_bo)
                                      ? (s->arg_bo->gpu_va & ~0xffffffffULL)
                                      : 0x4000000000ULL;
    uniform_range ranges[32];
    size_t range_count = 0;

    if (kernel->uses_root) {
        ranges[range_count++] = {
            kernel->root_uniform_half,
            4u,
            kDummyRootVa,
            true,
            0u,
        };
    }

    uint32_t max_push_dma_bytes = 0;
    bool     needs_full_zero    = false;

    const size_t dma_start = range_count;
    for (const auto & span : kernel->uniform_dma) {
        if (span.set != -1 || range_count >= 31) {
            continue;
        }
        const uint16_t aligned_halfs = static_cast<uint16_t>((span.halfs + 1u) & ~1u);
        const uint32_t span_bytes    = static_cast<uint32_t>(aligned_halfs) * 2u;
        const uint32_t span_beg      = span.offset_B;
        const uint32_t span_end      = span_beg + span_bytes;
        if (span_beg >= kernel->push_offset_bytes &&
            span_end <= kernel->push_offset_bytes + AGX_MAX_PUSH_BYTES) {
            max_push_dma_bytes = std::max(max_push_dma_bytes, span_end - kernel->push_offset_bytes);
        } else if (!((span_beg == 0u && span_end <= 8u) ||
                     (span_beg >= 0x88u && span_end <= 0xa8u))) {
            needs_full_zero = true;
        }
        ranges[range_count++] = {
            span.dest_half,
            aligned_halfs,
            kDummyRootVa + span.offset_B,
            true,
            span.offset_B,
        };
    }
    if (range_count > dma_start + 1u) {
        std::sort(ranges + dma_start, ranges + range_count,
                  [](const uniform_range & a, const uniform_range & b) {
                      return a.root_delta < b.root_delta;
                  });
    }

    if (kernel->rodata_size_16 != 0 && kernel->code_bo) {
        const uint16_t aligned_halfs = static_cast<uint16_t>((kernel->rodata_size_16 + 1u) & ~1u);
        const size_t ro_bytes = static_cast<size_t>(aligned_halfs) * 2u;
        if (static_cast<size_t>(kernel->rodata_offset) + ro_bytes > kernel->code_bo->alloc_size) {
            AGX_LOG_ERROR("Kernel '%s' rodata [0x%x + %zu] exceeds code_bo (%zu B)",
                          kernel->name.c_str(), kernel->rodata_offset, ro_bytes, kernel->code_bo->alloc_size);
            return nullptr;
        }
        ranges[range_count++] = {
            kernel->rodata_base_half,
            aligned_halfs,
            rodata_va,
            false,
            0u,
        };
    }

    agx_ioto_template tmpl;
    if (!compose_g17p_ioto_template(kernel, ranges, range_count, is_indirect, tmpl)) {
        AGX_LOG_ERROR("Failed to compose G17P Ioto template for kernel '%s'", kernel->name.c_str());
        return nullptr;
    }
    tmpl.const_data_va      = kernel->const_data_va;
    tmpl.max_push_dma_bytes = max_push_dma_bytes;
    tmpl.push_offset_bytes  = kernel->push_offset_bytes;
    tmpl.alloc_root_min     = std::max<uint32_t>(kernel->root_size_bytes, 64u);
    tmpl.uses_root_or_dma   = kernel->uses_root || !kernel->uniform_dma.empty();
    tmpl.needs_full_zero    = needs_full_zero;
    tmpl.local_size[0]      = kernel->local_size[0];
    tmpl.local_size[1]      = kernel->local_size[1];
    tmpl.local_size[2]      = kernel->local_size[2];
    tmpl.local_size_valid   =
        (kernel->local_size[0] > 0 && kernel->local_size[0] <= 0x7ffu) &&
        (kernel->local_size[1] > 0 && kernel->local_size[1] <= 0x7ffu) &&
        (kernel->local_size[2] > 0 && kernel->local_size[2] <= 0x7ffu);
    build_template_words(kernel->ti_spill_bytes, kernel->ti_ipr_bytes, true,
                         &tmpl.launch_w0_1d, &tmpl.launch_w1);
    uint32_t dummy_w1 = 0;
    build_template_words(kernel->ti_spill_bytes, kernel->ti_ipr_bytes, false,
                         &tmpl.launch_w0_nd, &dummy_w1);
    tmpl.reloc_count = static_cast<uint8_t>(std::min<size_t>(tmpl.relocs.size(), 255u));
    if (!tmpl.relocs.empty()) {
        tmpl.reloc0 = tmpl.relocs[0];
    }

    auto & slot = s->ioto_templates[key];
    slot = std::move(tmpl);
    fe = { kernel, kernel->shader_va, rodata_va, &slot, is_indirect };
    return fe.tmpl;
}

static inline bool prepare_launch_ioto_with_tmpl(agx_cdm_stream *          s,
                                                 const agx_ioto_template * tmpl,
                                                 const void *              push_args,
                                                 size_t                    push_bytes,
                                                 size_t                    alloc_root,
                                                 uint64_t *                out_program_va) {
    // Upload hk_root_descriptor_table slice with push constants at push_offset_bytes.
    // Avoids full 1,088B memset per dispatch (~8x less CPU/SLC write traffic) by explicitly
    // writing every live field (`root_va`, `const_data_va`, `group_count`/`base_group`, `push[]` + tail).
    uint64_t root_va = 0;
    if (AGX_LIKELY(tmpl->uses_root_or_dma || push_bytes > 0)) {
        const size_t off = agx_align_up(s->arg_offset, 16u);
        if (AGX_UNLIKELY(off + alloc_root > s->arg_bo->alloc_size)) {
            AGX_LOG_ERROR("CDM arg ring overflow (need %zu at %zu / %zu)",
                          alloc_root, off, s->arg_bo->alloc_size);
            return false;
        }
        s->arg_offset = off + alloc_root;
        root_va = s->arg_bo->gpu_va + off;
        auto * rbytes = static_cast<uint8_t *>(s->arg_bo->cpu_ptr) + off;

        if (AGX_UNLIKELY(tmpl->needs_full_zero)) {
            std::memset(rbytes, 0, alloc_root);
            store_le64(rbytes, root_va);
            if (tmpl->const_data_va != 0 && alloc_root >= 144u) {
                store_le64(rbytes + 136u, tmpl->const_data_va);
            }
            if (push_args && push_bytes > 0) {
                std::memcpy(rbytes + tmpl->push_offset_bytes, push_args, push_bytes);
            }
        } else {
            // Field 0 of hk_root_descriptor_table is always its own GPU VA (reflection)
            store_le64(rbytes, root_va);
            if (AGX_LIKELY(alloc_root >= 0xa8u)) {
                // Write cs.constant_data_addr (0x88) and zero group_count/base_group[3] (0x90..0xa8)
                store_le64(rbytes + 0x88u, tmpl->const_data_va);
                store_le64(rbytes + 0x90u, 0ULL);
                store_le64(rbytes + 0x98u, 0ULL);
                store_le64(rbytes + 0xa0u, 0ULL);
            } else if (alloc_root >= 144u) {
                store_le64(rbytes + 136u, tmpl->const_data_va);
            }
            if (AGX_LIKELY(push_args && push_bytes > 0)) {
                std::memcpy(rbytes + tmpl->push_offset_bytes, push_args, push_bytes);
            }
            if (AGX_UNLIKELY(tmpl->max_push_dma_bytes > push_bytes)) {
                std::memset(rbytes + tmpl->push_offset_bytes + push_bytes, 0,
                            tmpl->max_push_dma_bytes - push_bytes);
            }
        }
    }

    // Reserve 64B-aligned slot in the stream's USC Ioto arena (bit 40 set!)
    const size_t ioto_off = agx_align_up(s->ioto_offset, AGX_IOTO_PROGRAM_ALIGN);
    const size_t written  = tmpl->bytes.size();
    if (AGX_UNLIKELY(ioto_off + written > s->ioto_bo->alloc_size)) {
        AGX_LOG_ERROR("CDM USC Ioto arena overflow (%zu + %zu / %zu)",
                      ioto_off, written, s->ioto_bo->alloc_size);
        return false;
    }

    uint8_t * dst = static_cast<uint8_t *>(s->ioto_bo->cpu_ptr) + ioto_off;
    std::memcpy(dst, tmpl->bytes.data(), written);
    if (AGX_LIKELY(tmpl->reloc_count == 1u)) {
        const uint64_t addr = root_va + tmpl->reloc0.root_delta;
        store_le64(dst + tmpl->reloc0.byte_off,
                   tmpl->reloc0.dest_mask_lo | pack_ldimm_val(static_cast<uint32_t>(addr)));
        if (AGX_UNLIKELY(!s->arg_bo_single_hi32)) {
            store_le64(dst + tmpl->reloc0.byte_off + 8u,
                       tmpl->reloc0.dest_mask_hi | pack_ldimm_val(static_cast<uint32_t>(addr >> 32)));
        }
    } else if (tmpl->reloc_count > 1u) {
        if (AGX_LIKELY(s->arg_bo_single_hi32)) {
            for (const auto & rel : tmpl->relocs) {
                const uint64_t addr = root_va + rel.root_delta;
                store_le64(dst + rel.byte_off,
                           rel.dest_mask_lo | pack_ldimm_val(static_cast<uint32_t>(addr)));
            }
        } else {
            for (const auto & rel : tmpl->relocs) {
                const uint64_t addr = root_va + rel.root_delta;
                store_le64(dst + rel.byte_off,
                           rel.dest_mask_lo | pack_ldimm_val(static_cast<uint32_t>(addr)));
                store_le64(dst + rel.byte_off + 8u,
                           rel.dest_mask_hi | pack_ldimm_val(static_cast<uint32_t>(addr >> 32)));
            }
        }
    }

    s->ioto_offset = ioto_off + written;
    *out_program_va = s->ioto_bo->gpu_va + ioto_off;
    return true;
}

static bool prepare_launch_ioto(agx_cdm_stream *   s,
                                const agx_kernel * kernel,
                                const void *       push_args,
                                size_t             push_bytes,
                                bool               is_indirect,
                                uint64_t *         out_program_va) {
    if (!s || !kernel || kernel->shader_va == 0) {
        return false;
    }
    if (push_bytes > AGX_MAX_PUSH_BYTES) {
        AGX_LOG_ERROR("Push constant size %zu exceeds AGX_MAX_PUSH_BYTES (%u)",
                      push_bytes, AGX_MAX_PUSH_BYTES);
        return false;
    }

    const agx_ioto_template * tmpl = get_or_build_ioto_template(s, kernel, is_indirect);
    if (!tmpl) {
        return false;
    }
    const size_t push_end = static_cast<size_t>(tmpl->push_offset_bytes) + agx_align_up(push_bytes, 16u);
    const size_t alloc_root = std::max<size_t>(tmpl->alloc_root_min, push_end);
    return prepare_launch_ioto_with_tmpl(s, tmpl, push_args, push_bytes, alloc_root, out_program_va);
}

static inline void maybe_elide_previous_barrier(agx_cdm_stream * s) {
    // Pillar 6 barrier elision: if the previous launch requested AGX_BARRIER_NONE
    // and nothing subsequently strengthened its trailing weak barrier, overwrite
    // that trailing 4-byte record directly with the next launch so independent
    // dispatches execute concurrently across all 40 cores.
    if (s->last_launch_elidable &&
        s->last_barrier_off != SIZE_MAX &&
        s->last_barrier_off + AGX3_CDM_BARRIER_BYTES == s->cmd_offset &&
        s->last_barrier_mode == AGX_BARRIER_WEAK_WFI) {
        s->cmd_offset = s->last_barrier_off;
        s->last_barrier_off = SIZE_MAX;
        s->last_barrier_mode = AGX_BARRIER_NONE;
        s->last_launch_elidable = false;
        s->stats.elided_barriers++;
    }
}

static inline bool dispatch_threads_impl(agx_cdm_stream *   s,
                                         const agx_kernel * kernel,
                                         uint32_t           global_x,
                                         uint32_t           global_y,
                                         uint32_t           global_z,
                                         const void *       push_args,
                                         size_t             push_bytes,
                                         agx_barrier_mode   post_barrier) {
    if (AGX_UNLIKELY(!s || !kernel || s->terminated || kernel->shader_va == 0)) {
        return false;
    }
    if (AGX_UNLIKELY(global_x == 0 || global_y == 0 || global_z == 0)) {
        return true;
    }
    if (AGX_UNLIKELY(push_bytes > AGX_MAX_PUSH_BYTES)) {
        AGX_LOG_ERROR("Push constant size %zu exceeds AGX_MAX_PUSH_BYTES (%u)",
                      push_bytes, AGX_MAX_PUSH_BYTES);
        return false;
    }

    const agx_ioto_template * tmpl = get_or_build_ioto_template(s, kernel, false);
    if (AGX_UNLIKELY(!tmpl || !tmpl->local_size_valid)) {
        return false;
    }

    const size_t push_end   = static_cast<size_t>(tmpl->push_offset_bytes) + agx_align_up(push_bytes, 16u);
    const size_t alloc_root = std::max<size_t>(tmpl->alloc_root_min, push_end);
    const size_t ioto_bytes = tmpl->bytes.size();

    // Fast single-branch capacity check before touching rings; auto-flush only on overflow.
    if (AGX_UNLIKELY(s->cmd_offset + AGX3_CDM_DIRECT_LAUNCH_BYTES + AGX3_CDM_BARRIER_BYTES + AGX3_CDM_TERMINATE_BYTES > s->cmd_bo->alloc_size ||
                     agx_align_up(s->ioto_offset, AGX_IOTO_PROGRAM_ALIGN) + ioto_bytes > s->ioto_bo->alloc_size ||
                     agx_align_up(s->arg_offset, 16u) + alloc_root > s->arg_bo->alloc_size)) {
        if (!maybe_auto_flush_chunk(s, AGX3_CDM_DIRECT_LAUNCH_BYTES, ioto_bytes, alloc_root)) {
            return false;
        }
        if (s->cmd_offset + AGX3_CDM_DIRECT_LAUNCH_BYTES + AGX3_CDM_BARRIER_BYTES + AGX3_CDM_TERMINATE_BYTES >
            s->cmd_bo->alloc_size) {
            AGX_LOG_ERROR("CDM command buffer overflow (%zu / %zu)", s->cmd_offset, s->cmd_bo->alloc_size);
            return false;
        }
    }

    uint64_t program_va = 0;
    if (AGX_UNLIKELY(!prepare_launch_ioto_with_tmpl(s, tmpl, push_args, push_bytes, alloc_root, &program_va))) {
        return false;
    }
    if (AGX_UNLIKELY(program_va == 0 || (program_va & 0x3fULL) != 0 || program_va >= (1ULL << 48))) {
        AGX_LOG_ERROR("Invalid Ioto program VA 0x%llx for kernel '%s'",
                      (unsigned long long) program_va, kernel->name.c_str());
        return false;
    }

    maybe_elide_previous_barrier(s);

    const bool is_1d = (global_y < 2u && global_z < 2u);
    const agx_barrier_mode emit_mode =
        (post_barrier == AGX_BARRIER_NONE) ? AGX_BARRIER_WEAK_WFI : post_barrier;
    const uint32_t bword = agx_cdm_encode_barrier_word(emit_mode);

    // Emit 40-byte HAL200 direct launch + 4-byte trailing barrier in one compact 44B store burst
    uint8_t * dst = static_cast<uint8_t *>(s->cmd_bo->cpu_ptr) + s->cmd_offset;
    store_le32(dst + 0,  is_1d ? tmpl->launch_w0_1d : tmpl->launch_w0_nd);
    store_le32(dst + 4,  static_cast<uint32_t>(((program_va >> 38) & 0x3ffu) << 22) | tmpl->launch_w1);
    store_le32(dst + 8,  static_cast<uint32_t>(program_va >> 6));
    store_le32(dst + 12, 0x40000000u | static_cast<uint32_t>((program_va >> 40) & 0xffu));
    store_le32(dst + 16, global_x);
    store_le32(dst + 20, global_y);
    store_le32(dst + 24, global_z);
    store_le32(dst + 28, tmpl->local_size[0]);
    store_le32(dst + 32, tmpl->local_size[1]);
    store_le32(dst + 36, tmpl->local_size[2]);
    store_le32(dst + 40, bword);

    s->last_barrier_off     = s->cmd_offset + AGX3_CDM_DIRECT_LAUNCH_BYTES;
    s->last_barrier_mode    = emit_mode;
    s->last_launch_elidable = (post_barrier == AGX_BARRIER_NONE);
    s->cmd_offset          += AGX3_CDM_DIRECT_LAUNCH_BYTES + AGX3_CDM_BARRIER_BYTES;

    s->stats.dispatches++;
    if (emit_mode == AGX_BARRIER_WEAK_WFI) {
        s->stats.weak_wfi_barriers++;
    } else if (emit_mode == AGX_BARRIER_STRONG_USC) {
        s->stats.strong_usc_barriers++;
    } else if (emit_mode == AGX_BARRIER_MEMORY_HOST) {
        s->stats.memory_host_barriers++;
    }
    return true;
}

bool agx_cdm_dispatch_threads(agx_cdm_stream *   s,
                              const agx_kernel * kernel,
                              uint32_t           global_x,
                              uint32_t           global_y,
                              uint32_t           global_z,
                              const void *       push_args,
                              size_t             push_bytes,
                              agx_barrier_mode   post_barrier) {
    return dispatch_threads_impl(s, kernel, global_x, global_y, global_z,
                                 push_args, push_bytes, post_barrier);
}

bool agx_cdm_dispatch(agx_cdm_stream *   s,
                      const agx_kernel * kernel,
                      uint32_t           grid_x,
                      uint32_t           grid_y,
                      uint32_t           grid_z,
                      const void *       push_args,
                      size_t             push_bytes,
                      agx_barrier_mode   post_barrier) {
    if (AGX_UNLIKELY(!kernel)) {
        return false;
    }
    const uint64_t gx = static_cast<uint64_t>(grid_x) * kernel->local_size[0];
    const uint64_t gy = static_cast<uint64_t>(grid_y) * kernel->local_size[1];
    const uint64_t gz = static_cast<uint64_t>(grid_z) * kernel->local_size[2];
    if (AGX_UNLIKELY((gx | gy | gz) > UINT32_MAX)) {
        return false;
    }
    return dispatch_threads_impl(s, kernel,
                                 static_cast<uint32_t>(gx),
                                 static_cast<uint32_t>(gy),
                                 static_cast<uint32_t>(gz),
                                 push_args, push_bytes, post_barrier);
}

bool agx_cdm_dispatch_indirect(agx_cdm_stream *   s,
                               const agx_kernel * kernel,
                               uint64_t           geometry_va,
                               const void *       push_args,
                               size_t             push_bytes,
                               agx_barrier_mode   post_barrier) {
    if (!s || !kernel || s->terminated || geometry_va == 0) {
        return false;
    }
    const size_t est_root_bytes = std::max<size_t>(
        kernel->root_size_bytes,
        static_cast<size_t>(kernel->push_offset_bytes) + agx_align_up(push_bytes, 16u));
    const bool geom_in_arg_bo = s->arg_bo &&
        geometry_va >= s->arg_bo->gpu_va &&
        geometry_va < s->arg_bo->gpu_va + s->arg_bo->alloc_size;
    if (!geom_in_arg_bo &&
        !maybe_auto_flush_chunk(s, AGX3_CDM_INDIRECT_LAUNCH_BYTES, 4096u, std::max<size_t>(est_root_bytes, 64u))) {
        return false;
    }
    if (s->cmd_offset + AGX3_CDM_INDIRECT_LAUNCH_BYTES + AGX3_CDM_BARRIER_BYTES + AGX3_CDM_TERMINATE_BYTES >
        s->cmd_bo->alloc_size) {
        AGX_LOG_ERROR("CDM command buffer overflow on indirect launch");
        return false;
    }

    uint64_t program_va = 0;
    if (!prepare_launch_ioto(s, kernel, push_args, push_bytes, true, &program_va)) {
        return false;
    }

    // Indirect launches do not elide the preceding barrier (the geometry producer must settle).
    uint8_t * cmd = static_cast<uint8_t *>(s->cmd_bo->cpu_ptr);
    if (!agx_cdm_pack_indirect_launch(program_va, geometry_va,
                                      kernel->ti_spill_bytes, kernel->ti_ipr_bytes,
                                      cmd + s->cmd_offset)) {
        return false;
    }
    s->cmd_offset += AGX3_CDM_INDIRECT_LAUNCH_BYTES;

    const agx_barrier_mode emit_mode =
        (post_barrier == AGX_BARRIER_NONE) ? AGX_BARRIER_WEAK_WFI : post_barrier;
    s->last_barrier_off     = s->cmd_offset;
    s->last_barrier_mode    = emit_mode;
    s->last_launch_elidable = false;
    store_le32(cmd + s->cmd_offset, agx_cdm_encode_barrier_word(emit_mode));
    s->cmd_offset += AGX3_CDM_BARRIER_BYTES;

    s->stats.dispatches++;
    s->stats.indirect_dispatches++;
    if (emit_mode == AGX_BARRIER_WEAK_WFI) {
        s->stats.weak_wfi_barriers++;
    } else if (emit_mode == AGX_BARRIER_STRONG_USC) {
        s->stats.strong_usc_barriers++;
    } else if (emit_mode == AGX_BARRIER_MEMORY_HOST) {
        s->stats.memory_host_barriers++;
    }
    return true;
}

bool agx_cdm_barrier(agx_cdm_stream * s, agx_barrier_mode mode) {
    if (!s || s->terminated || mode == AGX_BARRIER_NONE) {
        return true;
    }
    if (s->cmd_offset == 0 || s->last_barrier_off == SIZE_MAX) {
        return true;
    }

    // Upgrade the trailing barrier in place (monotonic: WEAK < STRONG < MEMORY).
    // HAL200/G17P never emits two adjacent CDM barrier records.
    s->last_launch_elidable = false;
    if (mode > s->last_barrier_mode) {
        uint8_t * cmd = static_cast<uint8_t *>(s->cmd_bo->cpu_ptr);
        store_le32(cmd + s->last_barrier_off, agx_cdm_encode_barrier_word(mode));
        s->last_barrier_mode = mode;
        if (mode == AGX_BARRIER_STRONG_USC) {
            s->stats.strong_usc_barriers++;
        } else if (mode == AGX_BARRIER_MEMORY_HOST) {
            s->stats.memory_host_barriers++;
        }
    }
    return true;
}

static inline uint32_t agx_cmd_offset_tier(size_t cmd_bytes) {
    if (cmd_bytes < 1024u)  return 0u; // tiny (e.g. DSpark Inject MK ~0.3 KB)
    if (cmd_bytes < 8192u)  return 1u; // small (e.g. DSpark Draft MK ~2 KB)
    if (cmd_bytes < 49152u) return 2u; // medium (e.g. Target Decode/Verify ~12-28 KB)
    return 3u;                         // large (e.g. Prefill >= 48 KB)
}

static inline void agx_update_asymmetric_ema(double & ema, double sample_us) {
    if (sample_us <= 0.0) {
        return;
    }
    if (ema <= 0.0) {
        ema = sample_us;
    } else if (sample_us < ema) {
        ema = 0.50 * ema + 0.50 * sample_us;
    } else {
        ema = 0.85 * ema + 0.15 * sample_us;
    }
}

#if defined(__linux__) && defined(__aarch64__)
static void agx_pin_decode_thread_once() {
    static const bool no_pin = []() {
        const char * v = std::getenv("GGML_AGX_NO_PIN_DECODE");
        return v && v[0] && v[0] != '0';
    }();
    static thread_local bool tl_pinned = false;
    if (tl_pinned || no_pin) {
        return;
    }
    tl_pinned = true;

    cpu_set_t cur;
    CPU_ZERO(&cur);
    if (sched_getaffinity(0, sizeof(cur), &cur) != 0 || CPU_COUNT(&cur) <= 1) {
        return;
    }

    // Pick the first allowed CPU with the highest cpufreq max (P-core, e.g. cpu4/cpu5 on M4 Max),
    // keeping other P-cores dark and never pinning to an E-core!
    int      best_cpu  = -1;
    uint64_t best_freq = 0;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &cur)) {
            continue;
        }
        char path[96];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        uint64_t freq = 0;
        if (FILE * f = std::fopen(path, "r")) {
            unsigned long long khz = 0;
            if (std::fscanf(f, "%llu", &khz) == 1) {
                freq = static_cast<uint64_t>(khz);
            }
            std::fclose(f);
        }
        if (best_cpu < 0 || freq > best_freq) {
            best_cpu  = c;
            best_freq = freq;
        }
    }
    // Fallback if cpufreq sysfs is unavailable: prefer first allowed CPU >= 4, else >= 1
    if (best_freq == 0) {
        best_cpu = -1;
        for (int c = 4; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &cur)) { best_cpu = c; break; }
        }
        if (best_cpu < 0) {
            for (int c = 1; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &cur)) { best_cpu = c; break; }
            }
        }
    }
    if (best_cpu >= 0) {
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(best_cpu, &one);
        (void) sched_setaffinity(0, sizeof(one), &one);
    }
}
#endif

static int agx_cdm_wait_timeline_adaptive(agx_cdm_stream * s, uint64_t pt) {
    // GGML_AGX_WAIT_MODE:
    //   wfe   (default on Linux aarch64): ARM64 clock-gated `sevl; wfe; wfe` + tight tail + prefill sleep
    //         Prevents Linux schedutil PELT P-core clock collapse (4.104 -> 1.260 GHz) after >5 ms sleep!
    //   block : original blocking DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT
    //   spin  : pure non-blocking yield poll
    static const int wait_mode = []() {
        const char * m = std::getenv("GGML_AGX_WAIT_MODE");
        if (m && m[0]) {
            if (std::strcmp(m, "block") == 0) return 1;
            if (std::strcmp(m, "spin")  == 0) return 2;
            if (std::strcmp(m, "wfe")   == 0) return 0;
        }
#if defined(__linux__) && defined(__aarch64__)
        return 0;
#else
        return 1;
#endif
    }();

    if (wait_mode == 1) {
        return agx_drm_wait_timeline(s->dev, pt, UINT64_MAX);
    }

    auto is_not_ready = [](int r) -> bool {
        return r == -ETIME || r == -ETIMEDOUT;
    };

    if (wait_mode == 2) {
        for (;;) {
            const int r = agx_drm_wait_timeline(s->dev, pt, 0u);
            if (!is_not_ready(r)) {
                return r;
            }
#if defined(__aarch64__)
            for (int y = 0; y < 32; ++y) {
                __asm__ __volatile__("yield" ::: "memory");
            }
#endif
        }
    }

#if defined(__linux__) && defined(__aarch64__)
    static const int64_t sleep_min_est_us = []() -> int64_t {
        const char * v = std::getenv("GGML_AGX_SLEEP_MIN_EST_US");
        return (v && v[0]) ? std::atoll(v) : 100000;
    }();
    static const int64_t warm_margin_us = []() -> int64_t {
        const char * v = std::getenv("GGML_AGX_WARM_MARGIN_US");
        return (v && v[0]) ? std::atoll(v) : 6000;
    }();
    static const int64_t tight_tail_us = []() -> int64_t {
        const char * v = std::getenv("GGML_AGX_TIGHT_TAIL_US");
        return (v && v[0]) ? std::atoll(v) : 350;
    }();
    static const int64_t subms_spin_us = []() -> int64_t {
        const char * v = std::getenv("GGML_AGX_SUBMS_SPIN_US");
        return (v && v[0]) ? std::atoll(v) : 600;
    }();

    agx_pin_decode_thread_once();

    const uint32_t tier   = s->last_tier;
    const double   est_us = s->ema_wait_us[tier];

    auto elapsed_since_submit_us = [&]() -> int64_t {
        const auto now = std::chrono::steady_clock::now();
        return std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::microseconds>(now - s->submit_tp).count());
    };

    // Cold huge prefill (tier 3, no EMA yet): block cleanly instead of burning 102 ms in WFE.
    if (est_us <= 0.0 && tier == 3u) {
        return agx_drm_wait_timeline(s->dev, pt, UINT64_MAX);
    }

    // Warm long prefill (> 100 ms): sleep in kernel up to (est_us - 6 ms), then wake into WFE so P-core ramps to 4.1+ GHz!
    if (est_us > static_cast<double>(sleep_min_est_us)) {
        const int64_t sleep_us = static_cast<int64_t>(est_us) - elapsed_since_submit_us() - warm_margin_us;
        if (sleep_us > 1000) {
            const int r = agx_drm_wait_timeline(s->dev, pt, static_cast<uint64_t>(sleep_us) * 1000ULL);
            if (!is_not_ready(r)) {
                return r;
            }
        }
    }

    // Sub-ms submits (e.g. DSpark Inject 0.08 ms): immediate low-latency yield poll (< 5 us completion catch!)
    if (tier == 0u || (est_us > 0.0 && est_us <= static_cast<double>(subms_spin_us))) {
        for (uint32_t tp = 0u; tp < 4096u; ++tp) {
            const int r = agx_drm_wait_timeline(s->dev, pt, 0u);
            if (!is_not_ready(r)) {
                return r;
            }
            for (int y = 0; y < 32; ++y) {
                __asm__ __volatile__("yield" ::: "memory");
            }
        }
    }

    // Clock-gated ARM64 double-WFE (~100 us arch-timer event stream, <1% sys, keeps P-core pinned at 4.1+ GHz!)
    // ERET from ioctl sets evReg=1; SEVL+WFE clears it so the 2nd WFE genuinely clock-gates the core!
    for (uint32_t wfe_poll = 0u; wfe_poll < 1024u; ++wfe_poll) {
        const int r = agx_drm_wait_timeline(s->dev, pt, 0u);
        if (!is_not_ready(r)) {
            return r;
        }
        if (est_us > 0.0 && elapsed_since_submit_us() + tight_tail_us >= static_cast<int64_t>(est_us)) {
            for (uint32_t tp = 0u; tp < 4096u; ++tp) {
                const int rt = agx_drm_wait_timeline(s->dev, pt, 0u);
                if (!is_not_ready(rt)) {
                    return rt;
                }
                for (int y = 0; y < 64; ++y) {
                    __asm__ __volatile__("yield" ::: "memory");
                }
            }
        }
        __asm__ __volatile__("sevl; wfe; wfe" ::: "memory");
    }

    // Safety fallback for unexpectedly long graphs (> 102 ms)
    return agx_drm_wait_timeline(s->dev, pt, UINT64_MAX);
#else
    return agx_drm_wait_timeline(s->dev, pt, UINT64_MAX);
#endif
}

static inline void agx_clean_dcache_range(const void * ptr, size_t bytes) {
    if (!ptr || bytes == 0u) {
        return;
    }
#if defined(__aarch64__)
    static const bool no_clean = []() {
        const char * v = std::getenv("GGML_AGX_NO_DC_CVAC");
        return v && v[0] && v[0] != '0';
    }();
    if (no_clean) {
        return;
    }
    static const size_t dline = []() -> size_t {
        uint64_t ctr = 0;
        __asm__ __volatile__("mrs %0, ctr_el0" : "=r"(ctr));
        const uint64_t dmin = (ctr >> 16) & 0xFu;
        const size_t sz = static_cast<size_t>(4u) << dmin;
        return (sz >= 16u && sz <= 256u) ? sz : 64u;
    }();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr) & ~(dline - 1u);
    const uintptr_t end = reinterpret_cast<uintptr_t>(ptr) + bytes;
    for (; addr < end; addr += dline) {
        __asm__ __volatile__("dc cvac, %0" :: "r"(addr) : "memory");
    }
#else
    (void) ptr;
    (void) bytes;
#endif
}

static inline void agx_dsb_sy() {
#if defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

int agx_cdm_submit_recorded(agx_cdm_stream * s, bool wait, uint64_t * out_gpu_ns) {
    if (!s || !s->terminated || s->cmd_offset == 0) {
        return -EINVAL;
    }

    // Only attach kernel/firmware timestamp object when timestamps are requested or stats enabled.
    const bool want_ts = ((out_gpu_ns != nullptr) || g_agx_stats_enabled()) && (s->ts_bo != nullptr);
    if (want_ts) {
        auto * ts = static_cast<uint64_t *>(s->ts_bo->cpu_ptr);
        ts[0] = 0;
        ts[1] = 0;
        agx_clean_dcache_range(ts, 2u * sizeof(uint64_t));
    }
    s->last_submit_has_ts = want_ts;
    s->last_tier          = agx_cmd_offset_tier(s->cmd_offset);
    s->submit_tp          = std::chrono::steady_clock::now();

    // Clean dirty ASAHI_GEM_WRITEBACK command/ioto/arg BO lines to PoC before ringing the GPU doorbell
    // (matches Mesa hk_queue.c / cache_ops_aarch64.c for G17P AGX_CAP_CACHED_CMD_MEMORY).
    if (s->rings_dirty) {
        if (s->cmd_bo && s->cmd_bo->cpu_ptr && s->cmd_offset > 0u) {
            agx_clean_dcache_range(s->cmd_bo->cpu_ptr, s->cmd_offset);
        }
        if (s->ioto_bo && s->ioto_bo->cpu_ptr && s->ioto_offset > 0u) {
            agx_clean_dcache_range(s->ioto_bo->cpu_ptr, s->ioto_offset);
        }
        if (s->arg_bo && s->arg_bo->cpu_ptr && s->arg_offset > 0u) {
            agx_clean_dcache_range(s->arg_bo->cpu_ptr, s->arg_offset);
        }
        s->rings_dirty = false;
    }
    agx_dsb_sy();

    uint64_t pt = 0;
    const int ret = agx_drm_submit_cdm(
        s->dev,
        s->cmd_bo->gpu_va,
        s->cmd_bo->gpu_va + s->cmd_offset,
        want_ts ? s->ts_bo->ts_obj_handle : 0u,
        0u,
        sizeof(uint64_t),
        &pt,
        0u);
    if (g_agx_stats_enabled()) {
        const auto t_after_ioctl = std::chrono::steady_clock::now();
        s->last_ioctl_ms = std::chrono::duration<double, std::milli>(t_after_ioctl - s->submit_tp).count();
    }
    if (ret != 0) {
        return ret;
    }

    s->in_flight_point = pt;
    s->submits_since_reset++;
    s->stats.submits++;

    if (wait) {
        return agx_cdm_wait(s, out_gpu_ns);
    }
    return 0;
}

int agx_cdm_submit(agx_cdm_stream * s, bool wait, uint64_t * out_gpu_ns) {
    if (!s) {
        return -EINVAL;
    }
    if (s->cmd_offset == 0) {
        if (out_gpu_ns) {
            *out_gpu_ns = 0;
        }
        return 0;
    }

    if (!s->terminated) {
        // Reserve full host-memory visibility (0x6000157c) for host-synced submits;
        // async GPU-only submits only need intra-VM USC visibility (0x60001560).
        agx_cdm_barrier(s, wait ? AGX_BARRIER_MEMORY_HOST : AGX_BARRIER_STRONG_USC);

        uint8_t * cmd = static_cast<uint8_t *>(s->cmd_bo->cpu_ptr);
        store_le32(cmd + s->cmd_offset, AGX3_CDM_TERMINATE_WORD);
        s->cmd_offset += AGX3_CDM_TERMINATE_BYTES;
        s->terminated = true;
        s->rings_dirty = true;
    }

    const bool emit_stats = g_agx_stats_enabled();
    const auto t_sub = emit_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    uint64_t gpu_ns_local = 0;
    const int ret = agx_cdm_submit_recorded(s, wait, (emit_stats || out_gpu_ns) ? &gpu_ns_local : nullptr);
    if (out_gpu_ns) {
        *out_gpu_ns = gpu_ns_local;
    }

    if (ret == 0) {
        s->last_rec_disp = s->stats.dispatches          - s->stats_at_reset.dispatches;
        s->last_rec_ind  = s->stats.indirect_dispatches - s->stats_at_reset.indirect_dispatches;
        s->last_rec_usc  = s->stats.strong_usc_barriers - s->stats_at_reset.strong_usc_barriers;
        s->last_rec_elid = s->stats.elided_barriers     - s->stats_at_reset.elided_barriers;
    }

    if (emit_stats && ret == 0) {
        const auto t_done = std::chrono::steady_clock::now();
        const double encode_ms = s->has_reset_tp
            ? std::chrono::duration<double, std::milli>(t_sub - s->reset_tp).count()
            : 0.0;
        const double wait_ms = std::chrono::duration<double, std::milli>(t_done - t_sub).count();
        const double gpu_ms  = static_cast<double>(gpu_ns_local) * 1e-6;
        AGX_LOG_INFO("stats submit=#%llu | cpu_encode=%.3fms ioctl=%.3fms gpu=%.3fms wait=%.3fms (drm_oh=%.3fms) | "
                     "disp=%llu (ind=%llu) strong_usc=%llu elided=%llu | ring(cmd=%.1fK ioto=%.1fK arg=%.1fK tmpl=%zu)",
                     (unsigned long long) s->stats.submits,
                     encode_ms, s->last_ioctl_ms, gpu_ms, wait_ms, std::max(0.0, wait_ms - gpu_ms),
                     (unsigned long long) s->last_rec_disp, (unsigned long long) s->last_rec_ind,
                     (unsigned long long) s->last_rec_usc, (unsigned long long) s->last_rec_elid,
                     static_cast<double>(s->cmd_offset) / 1024.0,
                     static_cast<double>(s->ioto_offset) / 1024.0,
                     static_cast<double>(s->arg_offset) / 1024.0,
                     s->ioto_templates.size());
    }
    return ret;
}

int agx_cdm_wait(agx_cdm_stream * s, uint64_t * out_gpu_ns) {
    if (!s) {
        return -EINVAL;
    }
    if (s->in_flight_point == 0) {
        if (out_gpu_ns) {
            *out_gpu_ns = s->stats.last_gpu_ns;
        }
        return 0;
    }

    const int ret = agx_cdm_wait_timeline_adaptive(s, s->in_flight_point);
    s->in_flight_point = 0;
    if (ret != 0) {
        return ret;
    }

    const auto done_tp = std::chrono::steady_clock::now();
    const double wall_us = std::chrono::duration<double, std::micro>(done_tp - s->submit_tp).count();
    agx_update_asymmetric_ema(s->ema_wait_us[s->last_tier], wall_us);

    uint64_t gpu_ns = 0;
    if (s->last_submit_has_ts && s->ts_bo) {
        const auto * ts = static_cast<const uint64_t *>(s->ts_bo->cpu_ptr);
        if (ts[1] >= ts[0]) {
            gpu_ns = agx_drm_ticks_to_ns(s->dev, ts[1] - ts[0]);
        }
    }
    s->stats.last_gpu_ns = gpu_ns;
    s->stats.total_gpu_ns += gpu_ns;
    if (out_gpu_ns) {
        *out_gpu_ns = gpu_ns;
    }
    return 0;
}

agx_bo * agx_cdm_get_sense_barrier_bo(agx_cdm_stream * s) {
    return s ? s->sense_barrier_bo : nullptr;
}

uint64_t agx_cdm_get_sense_barrier_va(agx_cdm_stream * s) {
    return (s && s->sense_barrier_bo) ? s->sense_barrier_bo->gpu_va : 0;
}

void agx_cdm_zero_sense_barrier(agx_cdm_stream * s) {
    if (!s || !s->sense_barrier_bo) {
        return;
    }
    auto * st = static_cast<agx_sense_barrier_state *>(s->sense_barrier_bo->cpu_ptr);
    std::memset(st, 0, sizeof(*st));
    agx_clean_dcache_range(st, sizeof(*st));
    agx_dsb_sy();
}

void agx_cdm_reset_sense_barrier(agx_cdm_stream * s, uint32_t n_workgroups) {
    if (!s || !s->sense_barrier_bo) {
        return;
    }
    auto * st = static_cast<agx_sense_barrier_state *>(s->sense_barrier_bo->cpu_ptr);
    std::memset(st, 0, sizeof(*st));
    st->n_workgroups = n_workgroups;
    agx_clean_dcache_range(st, sizeof(*st));
    agx_dsb_sy();
}

void agx_cdm_get_stats(const agx_cdm_stream * s, agx_cdm_stats * out) {
    if (s && out) {
        *out = s->stats;
    }
}

void agx_cdm_reset_stats(agx_cdm_stream * s) {
    if (s) {
        s->stats = {};
    }
}

bool agx_cdm_is_empty(const agx_cdm_stream * s) {
    return !s || s->cmd_offset == 0;
}

bool agx_cdm_replay_enabled() {
    const char * v = std::getenv("GGML_AGX_NO_REPLAY");
    return !(v && v[0] && v[0] != '0');
}

uint64_t agx_cdm_stream_record_gen(const agx_cdm_stream * s) {
    return s ? s->record_gen : 0ULL;
}

uint64_t agx_cdm_stream_submit_count(const agx_cdm_stream * s) {
    return s ? s->stats.submits : 0ULL;
}

bool agx_cdm_stream_is_replayable(const agx_cdm_stream * s) {
    return s && s->allow_replay && s->terminated && s->cmd_offset > 0 && s->submits_since_reset >= 1u;
}

void agx_cdm_stream_set_replayable(agx_cdm_stream * s, bool replayable) {
    if (s) {
        s->allow_replay = replayable;
        if (!replayable) {
            s->replay.valid = false;
        }
    }
}

void agx_cdm_stream_invalidate_replay(agx_cdm_stream * s) {
    if (s) {
        s->replay.valid = false;
    }
}

void agx_cdm_stream_tag_replay(agx_cdm_stream * s,
                               uint64_t         graph_uid,
                               int              n_nodes,
                               const void *     first_node,
                               const void *     last_node,
                               uint64_t         bo_epoch) {
    if (!s || graph_uid == 0 || n_nodes <= 0 || !agx_cdm_stream_is_replayable(s)) {
        agx_cdm_stream_invalidate_replay(s);
        return;
    }
    s->replay.valid      = true;
    s->replay.graph_uid  = graph_uid;
    s->replay.n_nodes    = n_nodes;
    s->replay.first_node = first_node;
    s->replay.last_node  = last_node;
    s->replay.bo_epoch   = bo_epoch;
    s->replay.record_gen = s->record_gen;
    s->replay.hit_count  = 0;
}

bool agx_cdm_stream_try_replay(agx_cdm_stream * s,
                               uint64_t         graph_uid,
                               int              n_nodes,
                               const void *     first_node,
                               const void *     last_node,
                               uint64_t         bo_epoch,
                               bool             wait,
                               uint64_t *       out_gpu_ns,
                               int *            out_ret) {
    if (AGX_UNLIKELY(!s || graph_uid == 0 || n_nodes <= 0)) {
        return false;
    }
    const bool emit_stats = g_agx_stats_enabled();
    const auto t_enter = emit_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    if (!s->replay.valid ||
        !s->allow_replay ||
        s->replay.graph_uid != graph_uid ||
        s->replay.n_nodes != n_nodes ||
        s->replay.first_node != first_node ||
        s->replay.last_node != last_node ||
        s->replay.bo_epoch != bo_epoch ||
        s->replay.record_gen != s->record_gen ||
        !s->terminated ||
        s->cmd_offset == 0 ||
        !agx_cdm_replay_enabled() ||
        agx_drm_vm_error(s->dev) != 0) {
        return false;
    }

    // Reset stream sense-reversing barrier counters (arrived=0, sense=0, epoch=0)
    // while preserving n_workgroups so replaying an odd-phase persistent megakernel
    // never races against a leftover sense=1 from the previous submission.
    if (s->sense_barrier_bo && s->sense_barrier_bo->cpu_ptr) {
        auto * st = static_cast<agx_sense_barrier_state *>(s->sense_barrier_bo->cpu_ptr);
        st->arrived = 0u;
        st->sense   = 0u;
        st->epoch   = 0u;
        st->stage   = 0u;
        st->pass    = 0u;
        agx_clean_dcache_range(st, sizeof(*st));
    }

    const auto t_sub = emit_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    uint64_t gpu_ns_local = 0;
    const int ret = agx_cdm_submit_recorded(s, wait, (emit_stats || out_gpu_ns) ? &gpu_ns_local : nullptr);
    if (out_gpu_ns) {
        *out_gpu_ns = gpu_ns_local;
    }
    if (out_ret) {
        *out_ret = ret;
    }
    if (ret != 0) {
        s->replay.valid = false;
        return true;
    }

    s->replay.hit_count++;
    s->stats.replays++;
    if (emit_stats) {
        const auto t_done = std::chrono::steady_clock::now();
        const double encode_ms = std::chrono::duration<double, std::milli>(t_sub - t_enter).count();
        const double wait_ms   = std::chrono::duration<double, std::milli>(t_done - t_sub).count();
        const double gpu_ms    = static_cast<double>(gpu_ns_local) * 1e-6;
        AGX_LOG_INFO("stats submit=#%llu [REPLAY #%llu] | cpu_encode=%.3fms ioctl=%.3fms gpu=%.3fms wait=%.3fms (drm_oh=%.3fms) | "
                     "disp=%llu (ind=%llu) strong_usc=%llu elided=%llu | ring(cmd=%.1fK ioto=%.1fK arg=%.1fK tmpl=%zu)",
                     (unsigned long long) s->stats.submits,
                     (unsigned long long) s->replay.hit_count,
                     encode_ms, s->last_ioctl_ms, gpu_ms, wait_ms, std::max(0.0, wait_ms - gpu_ms),
                     (unsigned long long) s->last_rec_disp, (unsigned long long) s->last_rec_ind,
                     (unsigned long long) s->last_rec_usc, (unsigned long long) s->last_rec_elid,
                     static_cast<double>(s->cmd_offset) / 1024.0,
                     static_cast<double>(s->ioto_offset) / 1024.0,
                     static_cast<double>(s->arg_offset) / 1024.0,
                     s->ioto_templates.size());
    }
    return true;
}
