#include "agx-cdm.h"
#include "agx-compiler.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

struct agx_ioto_reloc {
    uint16_t byte_off;
    uint16_t dest_half;
    uint32_t root_delta;
};

struct agx_ioto_template {
    uint64_t                    shader_va = 0;
    uint64_t                    rodata_va = 0;
    std::vector<uint8_t>        bytes;
    std::vector<agx_ioto_reloc> relocs;
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
    dst[0] = static_cast<uint8_t>(v);
    dst[1] = static_cast<uint8_t>(v >> 8);
    dst[2] = static_cast<uint8_t>(v >> 16);
    dst[3] = static_cast<uint8_t>(v >> 24);
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

static void pack_ldimm_long(uint16_t dest_half, uint32_t imm, uint8_t out[8]) {
    const uint64_t dest =
        static_cast<uint64_t>(0x800cu | (((dest_half >> 1) & 0xfu) << 4)) |
        (static_cast<uint64_t>(0x0002u | ((dest_half & 1u) << 2) |
                               (((dest_half >> 5) & 3u) << 6) |
                               (((dest_half >> 7) & 1u) << 3)) << 16);
    const uint64_t val =
        static_cast<uint64_t>((imm & 0x7fu) << 8) |
        (static_cast<uint64_t>(((imm >> 25) & 0x7fu) << 9) << 16) |
        (static_cast<uint64_t>((((imm >> 7) & 0xfu) << 1) | (((imm >> 11) & 3u) << 10)) << 32) |
        (static_cast<uint64_t>(((imm >> 13) & 0xfu) | (((imm >> 17) & 0xffu) << 4)) << 48);
    const uint64_t word = dest | val;
    for (int i = 0; i < 8; ++i) {
        out[i] = static_cast<uint8_t>(word >> (8 * i));
    }
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

    // Barrier tracking (enforces HAL200 rule: never two adjacent CDM barrier records)
    size_t           last_barrier_off = SIZE_MAX;
    agx_barrier_mode last_barrier_mode = AGX_BARRIER_NONE;
    bool             last_launch_elidable = false;
    bool             terminated       = false;
    bool             has_external_args = false;

    uint64_t         in_flight_point  = 0;
    agx_cdm_stats    stats            = {};

    // Per-(shader_va, is_indirect) pre-templated G17P Ioto programs (zero heap allocs/dispatch!)
    std::unordered_map<uint64_t, agx_ioto_template> ioto_templates;
    uint64_t            last_tmpl_key = 0;
    agx_ioto_template * last_tmpl_ptr = nullptr;

    // Lightweight GGML_AGX_STATS=1 telemetry (CPU encode vs GPU execute split)
    std::chrono::steady_clock::time_point reset_tp{};
    agx_cdm_stats       stats_at_reset{};
    bool                has_reset_tp = false;
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
    const uint64_t key = (kernel->shader_va << 1) | (is_indirect ? 1ULL : 0ULL);
    const uint64_t rodata_va = (kernel->rodata_size_16 != 0 && kernel->code_bo)
                                   ? (kernel->code_bo->gpu_va + kernel->rodata_offset)
                                   : 0ULL;

    if (s->last_tmpl_ptr && s->last_tmpl_key == key &&
        s->last_tmpl_ptr->shader_va == kernel->shader_va &&
        s->last_tmpl_ptr->rodata_va == rodata_va) {
        return s->last_tmpl_ptr;
    }

    auto it = s->ioto_templates.find(key);
    if (it != s->ioto_templates.end() &&
        it->second.shader_va == kernel->shader_va &&
        it->second.rodata_va == rodata_va) {
        s->last_tmpl_key = key;
        s->last_tmpl_ptr = &it->second;
        return s->last_tmpl_ptr;
    }

    // Build Ioto uniform range list in optimal base-sharing order:
    //   1. `uses_root` (root_va + 0)
    //   2. `uniform_dma` spans sorted by `offset_B` ascending (all reuse root_va base -> 1 LDIMM pair!)
    //   3. `rodata` (static code_bo VA baked into template -> 0 runtime relocations!)
    constexpr uint64_t kDummyRootVa = 0x4000000000ULL;
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

    const size_t dma_start = range_count;
    for (const auto & span : kernel->uniform_dma) {
        if (span.set != -1 || range_count >= 31) {
            continue;
        }
        const uint16_t aligned_halfs = static_cast<uint16_t>((span.halfs + 1u) & ~1u);
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

    auto & slot = s->ioto_templates[key];
    slot = std::move(tmpl);
    s->last_tmpl_key = key;
    s->last_tmpl_ptr = &slot;
    return s->last_tmpl_ptr;
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

    // Upload hk_root_descriptor_table slice with push constants at push_offset_bytes.
    uint64_t root_va = 0;
    if (kernel->uses_root || !kernel->uniform_dma.empty() || push_bytes > 0) {
        const size_t root_size = std::max<size_t>(
            kernel->root_size_bytes,
            static_cast<size_t>(kernel->push_offset_bytes) + agx_align_up(push_bytes, 16u));
        const size_t alloc_root = std::max<size_t>(root_size, 64u);
        void * root_cpu = nullptr;
        root_va = alloc_args_internal(s, alloc_root, 16u, &root_cpu);
        if (root_va == 0) {
            return false;
        }
        auto * rbytes = static_cast<uint8_t *>(root_cpu);
        std::memset(rbytes, 0, alloc_root);
        // Field 0 of hk_root_descriptor_table is always its own GPU VA (reflection)
        std::memcpy(rbytes, &root_va, sizeof(root_va));
        // If shader has immutable NIR constant data, place at cs.constant_data_addr (offset 136 = 8 + 16*8)
        if (kernel->const_data_va != 0 && alloc_root >= 144u) {
            std::memcpy(rbytes + 136u, &kernel->const_data_va, sizeof(uint64_t));
        }
        // Copy push constants into root.push[]
        if (push_args && push_bytes > 0) {
            std::memcpy(rbytes + kernel->push_offset_bytes, push_args, push_bytes);
        }
    }

    const agx_ioto_template * tmpl = get_or_build_ioto_template(s, kernel, is_indirect);
    if (!tmpl) {
        return false;
    }

    // Reserve 64B-aligned slot in the stream's USC Ioto arena (bit 40 set!)
    const size_t ioto_off = agx_align_up(s->ioto_offset, AGX_IOTO_PROGRAM_ALIGN);
    const size_t written  = tmpl->bytes.size();
    if (ioto_off + written > s->ioto_bo->alloc_size) {
        AGX_LOG_ERROR("CDM USC Ioto arena overflow (%zu + %zu / %zu)",
                      ioto_off, written, s->ioto_bo->alloc_size);
        return false;
    }

    uint8_t * dst = static_cast<uint8_t *>(s->ioto_bo->cpu_ptr) + ioto_off;
    std::memcpy(dst, tmpl->bytes.data(), written);
    for (const auto & rel : tmpl->relocs) {
        const uint64_t addr = root_va + rel.root_delta;
        pack_ldimm_long(rel.dest_half, static_cast<uint32_t>(addr), dst + rel.byte_off);
        pack_ldimm_long(static_cast<uint16_t>(rel.dest_half + 2u),
                        static_cast<uint32_t>(addr >> 32), dst + rel.byte_off + 8u);
    }

    s->ioto_offset = ioto_off + written;
    *out_program_va = s->ioto_bo->gpu_va + ioto_off;
    return true;
}

static void maybe_elide_previous_barrier(agx_cdm_stream * s) {
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

bool agx_cdm_dispatch_threads(agx_cdm_stream *   s,
                              const agx_kernel * kernel,
                              uint32_t           global_x,
                              uint32_t           global_y,
                              uint32_t           global_z,
                              const void *       push_args,
                              size_t             push_bytes,
                              agx_barrier_mode   post_barrier) {
    if (!s || !kernel || s->terminated) {
        return false;
    }
    if (global_x == 0 || global_y == 0 || global_z == 0) {
        return true;
    }
    const size_t est_root_bytes = std::max<size_t>(
        kernel->root_size_bytes,
        static_cast<size_t>(kernel->push_offset_bytes) + agx_align_up(push_bytes, 16u));
    if (!maybe_auto_flush_chunk(s, AGX3_CDM_DIRECT_LAUNCH_BYTES, 4096u, std::max<size_t>(est_root_bytes, 64u))) {
        return false;
    }
    if (s->cmd_offset + AGX3_CDM_DIRECT_LAUNCH_BYTES + AGX3_CDM_BARRIER_BYTES + AGX3_CDM_TERMINATE_BYTES >
        s->cmd_bo->alloc_size) {
        AGX_LOG_ERROR("CDM command buffer overflow (%zu / %zu)", s->cmd_offset, s->cmd_bo->alloc_size);
        return false;
    }

    uint64_t program_va = 0;
    if (!prepare_launch_ioto(s, kernel, push_args, push_bytes, false, &program_va)) {
        return false;
    }

    const uint32_t global_size[3] = { global_x, global_y, global_z };
    uint8_t launch[AGX3_CDM_DIRECT_LAUNCH_BYTES];
    if (!agx_cdm_pack_direct_launch(program_va, global_size, kernel->local_size,
                                    kernel->ti_spill_bytes, kernel->ti_ipr_bytes, launch)) {
        AGX_LOG_ERROR("Failed to pack direct CDM launch for kernel '%s'", kernel->name.c_str());
        return false;
    }

    maybe_elide_previous_barrier(s);

    uint8_t * cmd = static_cast<uint8_t *>(s->cmd_bo->cpu_ptr);
    std::memcpy(cmd + s->cmd_offset, launch, sizeof(launch));
    s->cmd_offset += sizeof(launch);

    // HAL200 launches always end with a barrier record; AGX_BARRIER_NONE writes a
    // weak tail that the next launch will elide in place if still unstrengthened.
    const agx_barrier_mode emit_mode =
        (post_barrier == AGX_BARRIER_NONE) ? AGX_BARRIER_WEAK_WFI : post_barrier;
    const uint32_t bword = agx_cdm_encode_barrier_word(emit_mode);
    s->last_barrier_off     = s->cmd_offset;
    s->last_barrier_mode    = emit_mode;
    s->last_launch_elidable = (post_barrier == AGX_BARRIER_NONE);
    store_le32(cmd + s->cmd_offset, bword);
    s->cmd_offset += AGX3_CDM_BARRIER_BYTES;

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

bool agx_cdm_dispatch(agx_cdm_stream *   s,
                      const agx_kernel * kernel,
                      uint32_t           grid_x,
                      uint32_t           grid_y,
                      uint32_t           grid_z,
                      const void *       push_args,
                      size_t             push_bytes,
                      agx_barrier_mode   post_barrier) {
    if (!kernel) {
        return false;
    }
    const uint64_t gx = static_cast<uint64_t>(grid_x) * kernel->local_size[0];
    const uint64_t gy = static_cast<uint64_t>(grid_y) * kernel->local_size[1];
    const uint64_t gz = static_cast<uint64_t>(grid_z) * kernel->local_size[2];
    if (gx > UINT32_MAX || gy > UINT32_MAX || gz > UINT32_MAX) {
        return false;
    }
    return agx_cdm_dispatch_threads(s, kernel,
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

    uint8_t launch[AGX3_CDM_INDIRECT_LAUNCH_BYTES];
    if (!agx_cdm_pack_indirect_launch(program_va, geometry_va,
                                      kernel->ti_spill_bytes, kernel->ti_ipr_bytes, launch)) {
        return false;
    }

    // Indirect launches do not elide the preceding barrier (the geometry producer must settle).
    uint8_t * cmd = static_cast<uint8_t *>(s->cmd_bo->cpu_ptr);
    std::memcpy(cmd + s->cmd_offset, launch, sizeof(launch));
    s->cmd_offset += sizeof(launch);

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

int agx_cdm_submit_recorded(agx_cdm_stream * s, bool wait, uint64_t * out_gpu_ns) {
    if (!s || !s->terminated || s->cmd_offset == 0) {
        return -EINVAL;
    }

    // Reset hardware timestamp slots before submission
    if (s->ts_bo) {
        auto * ts = static_cast<uint64_t *>(s->ts_bo->cpu_ptr);
        ts[0] = 0;
        ts[1] = 0;
    }

    // Ensure all CPU writes to command stream, Ioto arena, and arg ring are visible
    // to the GPU before ringing the kernel submission doorbell.
    __sync_synchronize();

    uint64_t pt = 0;
    const int ret = agx_drm_submit_cdm(
        s->dev,
        s->cmd_bo->gpu_va,
        s->cmd_bo->gpu_va + s->cmd_offset,
        s->ts_bo ? s->ts_bo->ts_obj_handle : 0u,
        0u,
        sizeof(uint64_t),
        &pt,
        0u);
    if (ret != 0) {
        return ret;
    }

    s->in_flight_point = pt;
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
        // Ensure final stream writes are visible to host CPU and cross-alias VAs
        agx_cdm_barrier(s, AGX_BARRIER_MEMORY_HOST);

        uint8_t * cmd = static_cast<uint8_t *>(s->cmd_bo->cpu_ptr);
        store_le32(cmd + s->cmd_offset, AGX3_CDM_TERMINATE_WORD);
        s->cmd_offset += AGX3_CDM_TERMINATE_BYTES;
        s->terminated = true;
    }

    const bool emit_stats = g_agx_stats_enabled();
    const auto t_sub = emit_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    uint64_t gpu_ns_local = 0;
    const int ret = agx_cdm_submit_recorded(s, wait, &gpu_ns_local);
    if (out_gpu_ns) {
        *out_gpu_ns = gpu_ns_local;
    }

    if (emit_stats && ret == 0) {
        const auto t_done = std::chrono::steady_clock::now();
        const double encode_ms = s->has_reset_tp
            ? std::chrono::duration<double, std::milli>(t_sub - s->reset_tp).count()
            : 0.0;
        const double wait_ms = std::chrono::duration<double, std::milli>(t_done - t_sub).count();
        const double gpu_ms  = static_cast<double>(gpu_ns_local) * 1e-6;
        const uint64_t disp  = s->stats.dispatches           - s->stats_at_reset.dispatches;
        const uint64_t ind   = s->stats.indirect_dispatches  - s->stats_at_reset.indirect_dispatches;
        const uint64_t usc_b = s->stats.strong_usc_barriers  - s->stats_at_reset.strong_usc_barriers;
        const uint64_t elid  = s->stats.elided_barriers      - s->stats_at_reset.elided_barriers;
        AGX_LOG_INFO("stats submit=#%llu | cpu_encode=%.3fms gpu=%.3fms wait=%.3fms (drm_oh=%.3fms) | "
                     "disp=%llu (ind=%llu) strong_usc=%llu elided=%llu | ring(cmd=%.1fK ioto=%.1fK arg=%.1fK tmpl=%zu)",
                     (unsigned long long) s->stats.submits,
                     encode_ms, gpu_ms, wait_ms, std::max(0.0, wait_ms - gpu_ms),
                     (unsigned long long) disp, (unsigned long long) ind,
                     (unsigned long long) usc_b, (unsigned long long) elid,
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

    const int ret = agx_drm_wait_timeline(s->dev, s->in_flight_point, UINT64_MAX);
    s->in_flight_point = 0;
    if (ret != 0) {
        return ret;
    }

    uint64_t gpu_ns = 0;
    if (s->ts_bo) {
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

void agx_cdm_reset_sense_barrier(agx_cdm_stream * s, uint32_t n_workgroups) {
    if (!s || !s->sense_barrier_bo) {
        return;
    }
    auto * st = static_cast<agx_sense_barrier_state *>(s->sense_barrier_bo->cpu_ptr);
    std::memset(st, 0, sizeof(*st));
    st->n_workgroups = n_workgroups;
    __sync_synchronize();
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
