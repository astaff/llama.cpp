#include "agx-kv-hybrid.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// ============================================================================
// Host-side Reference Helpers: OCP E4M3FN & Orthonormal FWHT64
// ============================================================================

uint8_t agx_fp32_to_e4m3fn(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t sign_bit = (bits >> 24) & 0x80u;
    if (std::isnan(x)) {
        return static_cast<uint8_t>(sign_bit | 0x7Fu);
    }
    const float ax = std::min(std::fabs(x), 448.0f);
    if (ax < 0.015625f) {
        // Subnormal [0, 2^-6): step = 2^-9 = 1/512
        const float scaled = ax * 512.0f;
        const uint32_t m = static_cast<uint32_t>(std::nearbyintf(scaled));
        if (m >= 8u) {
            return static_cast<uint8_t>(sign_bit | (1u << 3));
        }
        return static_cast<uint8_t>(sign_bit | m);
    }
    uint32_t abits = 0;
    std::memcpy(&abits, &ax, sizeof(abits));
    int32_t  exp_unbiased = static_cast<int32_t>((abits >> 23) & 0xFFu) - 127;
    uint32_t mant23       = abits & 0x7FFFFFu;
    uint32_t rem          = mant23 & 0xFFFFFu;
    uint32_t m3           = mant23 >> 20;
    if (rem > 0x80000u || (rem == 0x80000u && (m3 & 1u) != 0u)) {
        m3 += 1u;
        if (m3 == 8u) {
            m3 = 0u;
            exp_unbiased += 1;
        }
    }
    int32_t e4 = exp_unbiased + 7;
    if (e4 >= 15) {
        return static_cast<uint8_t>(sign_bit | (15u << 3) | 6u);
    }
    return static_cast<uint8_t>(sign_bit | (static_cast<uint32_t>(e4) << 3) | m3);
}

float agx_e4m3fn_to_fp32(uint8_t b) {
    const uint32_t s = (static_cast<uint32_t>(b) & 0x80u) << 24;
    const uint32_t e = (static_cast<uint32_t>(b) >> 3) & 0x0Fu;
    const uint32_t m = static_cast<uint32_t>(b) & 0x07u;
    if (e == 0u) {
        const float sign = s ? -1.0f : 1.0f;
        return sign * static_cast<float>(m) * (1.0f / 512.0f);
    }
    const uint32_t fbits = s | ((e + 120u) << 23) | (m << 20);
    float out = 0.0f;
    std::memcpy(&out, &fbits, sizeof(out));
    return out;
}

void agx_fwht64_f32_ref(float * x, uint32_t n_blocks) {
    constexpr float inv_sqrt64 = 0.125f;
    for (uint32_t blk = 0; blk < n_blocks; ++blk) {
        float * v = x + blk * 64u;
        for (uint32_t stride = 1u; stride < 64u; stride <<= 1) {
            for (uint32_t base = 0u; base < 64u; base += (stride << 1)) {
                for (uint32_t i = 0u; i < stride; ++i) {
                    const float a = v[base + i];
                    const float b = v[base + i + stride];
                    v[base + i]          = a + b;
                    v[base + i + stride] = a - b;
                }
            }
        }
        for (uint32_t i = 0u; i < 64u; ++i) {
            v[i] *= inv_sqrt64;
        }
    }
}

// ============================================================================
// Inkling 2 Mini 48-Slot Hybrid Pattern
// ============================================================================
// Physical layers 0..23: 4 blocks of (5 SWA-1024 + 1 Full) -> full at 5, 11, 17, 23.
// Looped core (layers 6..17, 12 layers) executes 3 passes:
//   Pass 0: logical slots 0..23 (full: 5, 11, 17, 23)
//   Pass 1: logical slots 24..35 -> phys 6..17 (full: 29 [phys 11], 35 [phys 17])
//   Pass 2: logical slots 36..47 -> phys 6..17 (full: 41 [phys 11], 47 [phys 17])
// Total: 8 full-attention passes + 40 SWA-1024 ring passes.

void agx_kv_hybrid_fill_inkling2_pattern(bool slot_is_full[AGX_KV_MAX_SLOTS], uint32_t n_slots) {
    std::memset(slot_is_full, 0, AGX_KV_MAX_SLOTS * sizeof(bool));
    const uint32_t full_slots[8] = { 5u, 11u, 17u, 23u, 29u, 35u, 41u, 47u };
    for (uint32_t s : full_slots) {
        if (s < n_slots && s < AGX_KV_MAX_SLOTS) {
            slot_is_full[s] = true;
        }
    }
}

// ============================================================================
// Push-Constant Structs (exact layout match with kernels/fused_*_attn.comp)
// ============================================================================

struct agx_push_swa_attn {
    uint64_t q_va;
    uint64_t k_va;
    uint64_t v_va;
    uint64_t ks_va;
    uint64_t vs_va;
    uint64_t pos_ring_va;
    uint64_t rel_table_va;
    uint64_t sinks_va;
    uint64_t q_gate_va;
    uint64_t pos_q_va;
    uint64_t out_va;
    uint32_t n_tokens;
    uint32_t swa_window;
    uint32_t n_kv_scan;
    uint32_t n_head_q;
    uint32_t n_head_kv;
    uint32_t n_rel_extent;
    uint32_t pos_q_base;
    uint32_t flags;
    float    kq_scale;
    uint32_t pad0;
};
static_assert(sizeof(agx_push_swa_attn) == 128, "agx_push_swa_attn must match GLSL PushArgs (128B)");

struct agx_push_full_attn {
    uint64_t q_va;
    uint64_t k_or_pt_va;
    uint64_t v_va;
    uint64_t pos_kv_va;
    uint64_t rel_table_va;
    uint64_t sinks_va;
    uint64_t q_gate_va;
    uint64_t pos_q_va;
    uint64_t kq_mask_va;
    uint64_t partial_va;
    uint64_t out_va;

    uint32_t n_tokens;
    uint32_t n_kv;
    uint32_t n_splits;
    uint32_t n_head_q;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_rel_extent;
    uint32_t pos_q_base;
    uint32_t flags;
    uint32_t page_v_off_B;
    uint32_t page_ks_off_B;
    uint32_t page_vs_off_B;
    uint32_t ext_k_nb1;
    uint32_t ext_k_nb2;
    uint32_t ext_v_nb1;
    uint32_t ext_v_nb2;
    uint32_t ext_mask_nb1;
    uint32_t ext_mask_nb2;
    uint32_t ext_mask_ne2;
    uint32_t q_nb1;
    uint32_t q_nb2;
    uint32_t mask_n_head_log2;
    float    kq_scale;
    float    logit_softcap;
    float    m0;
    float    m1;

    uint64_t rel_pos_pack_va;
    uint32_t rel_table_len;
    uint32_t rel_cp_off;
    uint32_t rel_qp_off;
    uint32_t rel_tm_off;
    uint32_t _pad0;
    uint32_t _pad1;
};
static_assert(sizeof(agx_push_full_attn) == 224, "agx_push_full_attn must match GLSL PushAttn (224B)");

struct agx_push_kv_append {
    uint64_t k_src_va;
    uint64_t v_src_va;
    uint64_t dst_k_or_pt_va;
    uint64_t dst_v_va;
    uint64_t dst_ks_va;
    uint64_t dst_vs_va;
    uint64_t pos_dst_va;
    uint64_t pos_src_va;
    uint32_t n_tokens;
    uint32_t n_head_kv;
    uint32_t kv_write_base;
    uint32_t ring_mask;
    uint32_t page_v_off_B;
    uint32_t page_ks_off_B;
    uint32_t page_vs_off_B;
    uint32_t flags;
};
static_assert(sizeof(agx_push_kv_append) == 96, "agx_push_kv_append must match GLSL PushAppend (96B)");

// ============================================================================
// Hybrid KV Pool Internal State
// ============================================================================

static uint32_t kv_head_data_bytes(agx_kv_dtype dt, uint32_t head_dim) {
    switch (dt) {
        case AGX_KV_F8_E4M3:     return head_dim;      // 128 B
        case AGX_KV_Q8_0_PLANAR: return head_dim;      // 128 B int8 qs (scales stored separately)
        case AGX_KV_F16:         return head_dim * 2u; // 256 B
        case AGX_KV_F32:         return head_dim * 4u; // 512 B
        case AGX_KV_Q8_0_GGML:   return (head_dim / 32u) * 34u;
    }
    return head_dim;
}

static uint32_t kv_head_scale_bytes(agx_kv_dtype dt, uint32_t head_dim) {
    return (dt == AGX_KV_Q8_0_PLANAR) ? ((head_dim / 32u) * 2u) : 0u; // 8 B for D=128
}

struct agx_kv_hybrid_pool {
    agx_drm_device *     dev      = nullptr;
    agx_compiler *       compiler = nullptr;
    agx_kv_hybrid_config cfg;

    uint32_t n_swa_slots  = 0;
    uint32_t n_full_slots = 0;
    bool     slot_is_full[AGX_KV_MAX_SLOTS] = {};
    uint32_t slot_pool_idx[AGX_KV_MAX_SLOTS] = {};

    // SWA-1024 ring layout (single SLC-pinned BO!)
    size_t   swa_v_off_B      = 0;
    size_t   swa_ks_off_B     = 0;
    size_t   swa_vs_off_B     = 0;
    size_t   swa_slot_stride_B = 0;
    agx_bo * swa_ring_bo      = nullptr; // AGX_BO_SLC_PIN
    agx_bo * swa_pos_bo       = nullptr; // AGX_BO_SLC_PIN: int32[n_swa_slots][swa_window]

    // Full-attention 64-token paged pool (demand-grown SLC-streaming chunks!)
    uint32_t full_max_pages       = 0;
    uint32_t full_allocated_pages = 0;
    uint32_t page_v_off_B         = 0;
    uint32_t page_ks_off_B        = 0;
    uint32_t page_vs_off_B        = 0;
    size_t   page_bytes           = 0;

    agx_bo *              page_table_bo = nullptr; // AGX_BO_SLC_PIN: uint64[n_full_slots][full_max_pages]
    agx_bo *              full_pos_bo   = nullptr; // AGX_BO_SLC_PIN: int32[full_max_pages * 64]
    std::vector<agx_bo *> full_chunk_bos;          // AGX_BO_SLC_STREAM demand-allocated page chunks

    // Split-K Flash-Decoding partial reduction scratch BO (SLC-pinned, grows lazily with M)
    agx_bo * split_partial_bo = nullptr;
    size_t   split_partial_cap_bytes = 0;

    // Compiled kernels (head_slot: 0 = D128, 1 = D64; k_dt/v_dt: 0..4)
    const agx_kernel * k_swa_attn[3]          = {}; // F8_E4M3, Q8_0_PLANAR, F16
    const agx_kernel * k_full_phase1[2][5][5] = {}; // [head_slot][k_dt][v_dt]
    const agx_kernel * k_full_phase2[2]       = {}; // [head_slot]
    const agx_kernel * k_full_prefill[5][5]   = {}; // D128 BQ=32 coopmat prefill [k_dt][v_dt]
    const agx_kernel * k_full_prefill_m8[5][5]= {}; // D128 BQ=8 split-K coopmat (M=2..15 DSpark!) [k_dt][v_dt]
    const agx_kernel * k_kv_append[3]         = {}; // F8_E4M3, Q8_0_PLANAR, F16
};

static const agx_kernel * get_swa_kernel(agx_kv_hybrid_pool * pool, agx_kv_dtype dt) {
    const uint32_t idx = static_cast<uint32_t>(dt);
    if (idx >= 3u) {
        return nullptr;
    }
    if (!pool->k_swa_attn[idx] && pool->compiler) {
        agx_compile_opts opts;
        opts.defines.push_back("KV_DTYPE=" + std::to_string(idx));
        opts.defines.push_back("N_SG=4");
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        const std::string name = "fused_swa_ring_attn_dt" + std::to_string(idx);
        pool->k_swa_attn[idx] = agx_compiler_compile_file(
            pool->compiler, name.c_str(), "fused_swa_ring_attn.comp", opts);
    }
    return pool->k_swa_attn[idx];
}

static const agx_kernel * get_full_phase1_kernel(agx_kv_hybrid_pool * pool,
                                                 uint32_t             head_dim,
                                                 agx_kv_dtype         k_dt,
                                                 agx_kv_dtype         v_dt) {
    const uint32_t hs = (head_dim == 64u) ? 1u : (head_dim == 128u ? 0u : 2u);
    const uint32_t ik = static_cast<uint32_t>(k_dt);
    const uint32_t iv = static_cast<uint32_t>(v_dt);
    if (hs >= 2u || ik >= 5u || iv >= 5u) {
        return nullptr;
    }
    if (!pool->k_full_phase1[hs][ik][iv] && pool->compiler) {
        agx_compile_opts opts;
        opts.defines.push_back("PASS_MODE=0");
        opts.defines.push_back("HEAD_DIM=" + std::to_string(head_dim));
        opts.defines.push_back("KV_DTYPE_K=" + std::to_string(ik));
        opts.defines.push_back("KV_DTYPE_V=" + std::to_string(iv));
        opts.defines.push_back("N_SG=4");
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        const std::string name = "fused_full_flash_decode_p1w9_d" + std::to_string(head_dim)
                               + "_k" + std::to_string(ik) + "_v" + std::to_string(iv);
        pool->k_full_phase1[hs][ik][iv] = agx_compiler_compile_file(
            pool->compiler, name.c_str(), "fused_full_flash_decode.comp", opts);
    }
    return pool->k_full_phase1[hs][ik][iv];
}

static const agx_kernel * get_full_phase2_kernel(agx_kv_hybrid_pool * pool, uint32_t head_dim) {
    const uint32_t hs = (head_dim == 64u) ? 1u : (head_dim == 128u ? 0u : 2u);
    if (hs >= 2u) {
        return nullptr;
    }
    if (!pool->k_full_phase2[hs] && pool->compiler) {
        agx_compile_opts opts;
        opts.defines.push_back("PASS_MODE=1");
        opts.defines.push_back("HEAD_DIM=" + std::to_string(head_dim));
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        const std::string name = "fused_full_flash_decode_p2w9_d" + std::to_string(head_dim);
        pool->k_full_phase2[hs] = agx_compiler_compile_file(
            pool->compiler, name.c_str(), "fused_full_flash_decode.comp", opts);
    }
    return pool->k_full_phase2[hs];
}

static const agx_kernel * get_full_prefill_kernel(agx_kv_hybrid_pool * pool,
                                                  agx_kv_dtype         k_dt,
                                                  agx_kv_dtype         v_dt) {
    const uint32_t ik = static_cast<uint32_t>(k_dt);
    const uint32_t iv = static_cast<uint32_t>(v_dt);
    if (ik >= 5u || iv >= 5u) {
        return nullptr;
    }
    if (!pool->k_full_prefill[ik][iv] && pool->compiler) {
        agx_compile_opts opts;
        opts.defines.push_back("PASS_MODE=3");
        opts.defines.push_back("HEAD_DIM=128");
        opts.defines.push_back("KV_DTYPE_K=" + std::to_string(ik));
        opts.defines.push_back("KV_DTYPE_V=" + std::to_string(iv));
        opts.defines.push_back("N_SG=8");
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        const std::string name = "fused_full_flash_prefill_w9_d128_k" + std::to_string(ik)
                               + "_v" + std::to_string(iv);
        pool->k_full_prefill[ik][iv] = agx_compiler_compile_file(
            pool->compiler, name.c_str(), "fused_full_flash_decode.comp", opts);
    }
    return pool->k_full_prefill[ik][iv];
}

static const agx_kernel * get_full_prefill_m8_kernel(agx_kv_hybrid_pool * pool,
                                                     agx_kv_dtype         k_dt,
                                                     agx_kv_dtype         v_dt) {
    const uint32_t ik = static_cast<uint32_t>(k_dt);
    const uint32_t iv = static_cast<uint32_t>(v_dt);
    if (ik >= 5u || iv >= 5u) {
        return nullptr;
    }
    if (!pool->k_full_prefill_m8[ik][iv] && pool->compiler) {
        agx_compile_opts opts;
        opts.defines.push_back("PASS_MODE=4");
        opts.defines.push_back("HEAD_DIM=128");
        opts.defines.push_back("KV_DTYPE_K=" + std::to_string(ik));
        opts.defines.push_back("KV_DTYPE_V=" + std::to_string(iv));
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        const std::string name = "fused_full_flash_m8w9_d128_k" + std::to_string(ik)
                               + "_v" + std::to_string(iv);
        pool->k_full_prefill_m8[ik][iv] = agx_compiler_compile_file(
            pool->compiler, name.c_str(), "fused_full_flash_decode.comp", opts);
    }
    return pool->k_full_prefill_m8[ik][iv];
}

static const agx_kernel * get_kv_append_kernel(agx_kv_hybrid_pool * pool, agx_kv_dtype dt) {
    const uint32_t idx = static_cast<uint32_t>(dt);
    if (idx >= 3u) {
        return nullptr;
    }
    if (!pool->k_kv_append[idx] && pool->compiler) {
        agx_compile_opts opts;
        opts.defines.push_back("PASS_MODE=2");
        opts.defines.push_back("KV_DTYPE=" + std::to_string(idx));
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        const std::string name = "agx_kv_append_dt" + std::to_string(idx);
        pool->k_kv_append[idx] = agx_compiler_compile_file(
            pool->compiler, name.c_str(), "fused_full_flash_decode.comp", opts);
    }
    return pool->k_kv_append[idx];
}

static bool ensure_split_partial_bo(agx_kv_hybrid_pool * pool, uint32_t n_tokens, uint32_t n_head_q, uint32_t n_splits) {
    // Each (tok, h_q, split) stores 130 floats (128 O + m + l)
    const size_t needed = static_cast<size_t>(n_tokens) * n_head_q * n_splits * 130u * sizeof(float);
    if (needed <= pool->split_partial_cap_bytes && pool->split_partial_bo) {
        return true;
    }
    if (pool->split_partial_bo) {
        agx_bo_free(pool->dev, pool->split_partial_bo);
        pool->split_partial_bo = nullptr;
        pool->split_partial_cap_bytes = 0;
    }
    const size_t alloc_bytes = agx_align_up(std::max<size_t>(needed, 1024u * 1024u), AGX_PAGE_SIZE);
    pool->split_partial_bo = agx_bo_alloc(
        pool->dev, alloc_bytes, AGX_BO_SLC_PIN | AGX_BO_WRITEBACK, "kv-splitk-partial");
    if (!pool->split_partial_bo) {
        return false;
    }
    pool->split_partial_cap_bytes = alloc_bytes;
    return true;
}

agx_kv_hybrid_pool * agx_kv_hybrid_create(agx_drm_device *            dev,
                                          agx_compiler *              compiler,
                                          const agx_kv_hybrid_config & cfg) {
    if (!dev || cfg.n_slots == 0 || cfg.n_slots > AGX_KV_MAX_SLOTS ||
        (cfg.swa_window & (cfg.swa_window - 1u)) != 0u || cfg.swa_window == 0u) {
        AGX_LOG_ERROR("agx_kv_hybrid_create: invalid config (slots=%u, swa_window=%u)",
                      cfg.n_slots, cfg.swa_window);
        return nullptr;
    }

    auto * pool = new agx_kv_hybrid_pool();
    pool->dev      = dev;
    pool->compiler = compiler;
    pool->cfg      = cfg;

    if (cfg.use_inkling2_default_pattern) {
        agx_kv_hybrid_fill_inkling2_pattern(pool->slot_is_full, cfg.n_slots);
    } else {
        std::memcpy(pool->slot_is_full, cfg.slot_is_full, sizeof(pool->slot_is_full));
    }

    for (uint32_t s = 0; s < cfg.n_slots; ++s) {
        if (pool->slot_is_full[s]) {
            pool->slot_pool_idx[s] = pool->n_full_slots++;
        } else {
            pool->slot_pool_idx[s] = pool->n_swa_slots++;
        }
    }

    // 1. SWA-1024 Ring Pool layout & SLC-pinned allocation
    const size_t swa_data_B  = static_cast<size_t>(cfg.swa_window) * cfg.n_head_kv *
                               kv_head_data_bytes(cfg.swa_dtype, cfg.head_dim);
    const size_t swa_scale_B = static_cast<size_t>(cfg.swa_window) * cfg.n_head_kv *
                               kv_head_scale_bytes(cfg.swa_dtype, cfg.head_dim);
    pool->swa_v_off_B       = agx_align_up(swa_data_B, AGX_CACHE_LINE_BYTES);
    pool->swa_ks_off_B      = agx_align_up(pool->swa_v_off_B + swa_data_B, AGX_CACHE_LINE_BYTES);
    pool->swa_vs_off_B      = agx_align_up(pool->swa_ks_off_B + swa_scale_B, AGX_CACHE_LINE_BYTES);
    pool->swa_slot_stride_B = agx_align_up(pool->swa_vs_off_B + swa_scale_B, AGX_CACHE_LINE_BYTES);

    if (pool->n_swa_slots > 0) {
        const size_t total_swa_B = pool->swa_slot_stride_B * pool->n_swa_slots;
        pool->swa_ring_bo = agx_bo_alloc(
            dev, total_swa_B, AGX_BO_SLC_PIN | AGX_BO_WRITEBACK | AGX_BO_ZERO, "kv-swa-40ring");
        if (!pool->swa_ring_bo) {
            AGX_LOG_ERROR("agx_kv_hybrid_create: failed to allocate %.1f MiB SWA ring BO",
                          double(total_swa_B) / (1024.0 * 1024.0));
            agx_kv_hybrid_destroy(pool);
            return nullptr;
        }
        const size_t swa_pos_B = static_cast<size_t>(pool->n_swa_slots) * cfg.swa_window * sizeof(int32_t);
        pool->swa_pos_bo = agx_bo_alloc(
            dev, swa_pos_B, AGX_BO_SLC_PIN | AGX_BO_WRITEBACK, "kv-swa-pos");
        if (!pool->swa_pos_bo) {
            agx_kv_hybrid_destroy(pool);
            return nullptr;
        }
    }

    // 2. Full-Attention 64-Token Paged Pool metadata (SLC-pinned page table + pos table)
    const uint32_t max_tok = std::min<uint32_t>(
        std::max<uint32_t>(cfg.max_full_tokens, AGX_KV_PAGE_TOKENS), AGX_KV_MAX_FULL_TOKENS);
    pool->full_max_pages = (max_tok + AGX_KV_PAGE_TOKENS - 1u) / AGX_KV_PAGE_TOKENS;

    const size_t pg_data_B  = static_cast<size_t>(AGX_KV_PAGE_TOKENS) * cfg.n_head_kv *
                              kv_head_data_bytes(cfg.full_dtype, cfg.head_dim);
    const size_t pg_scale_B = static_cast<size_t>(AGX_KV_PAGE_TOKENS) * cfg.n_head_kv *
                              kv_head_scale_bytes(cfg.full_dtype, cfg.head_dim);
    pool->page_v_off_B  = static_cast<uint32_t>(agx_align_up(pg_data_B, AGX_CACHE_LINE_BYTES));
    pool->page_ks_off_B = static_cast<uint32_t>(agx_align_up(pool->page_v_off_B + pg_data_B, AGX_CACHE_LINE_BYTES));
    pool->page_vs_off_B = static_cast<uint32_t>(agx_align_up(pool->page_ks_off_B + pg_scale_B, AGX_CACHE_LINE_BYTES));
    pool->page_bytes    = agx_align_up(pool->page_vs_off_B + pg_scale_B, AGX_CACHE_LINE_BYTES);

    if (pool->n_full_slots > 0) {
        const size_t pt_bytes = static_cast<size_t>(pool->n_full_slots) * pool->full_max_pages * sizeof(uint64_t);
        pool->page_table_bo = agx_bo_alloc(
            dev, pt_bytes, AGX_BO_SLC_PIN | AGX_BO_WRITEBACK | AGX_BO_ZERO, "kv-full-pagetable");
        const size_t pos_bytes = static_cast<size_t>(pool->full_max_pages) * AGX_KV_PAGE_TOKENS * sizeof(int32_t);
        pool->full_pos_bo = agx_bo_alloc(
            dev, pos_bytes, AGX_BO_SLC_PIN | AGX_BO_WRITEBACK, "kv-full-pos");
        if (!pool->page_table_bo || !pool->full_pos_bo) {
            agx_kv_hybrid_destroy(pool);
            return nullptr;
        }
    }

    agx_kv_hybrid_reset(pool, false);

    // Pre-allocate initial split-K scratch (covers M=1..5, Hq=24, 64 splits -> ~4 MiB SLC)
    ensure_split_partial_bo(pool, 5u, 24u, AGX_KV_MAX_SPLITS);

    return pool;
}

void agx_kv_hybrid_destroy(agx_kv_hybrid_pool * pool) {
    if (!pool) {
        return;
    }
    if (pool->swa_ring_bo)      agx_bo_free(pool->dev, pool->swa_ring_bo);
    if (pool->swa_pos_bo)       agx_bo_free(pool->dev, pool->swa_pos_bo);
    if (pool->page_table_bo)    agx_bo_free(pool->dev, pool->page_table_bo);
    if (pool->full_pos_bo)      agx_bo_free(pool->dev, pool->full_pos_bo);
    if (pool->split_partial_bo) agx_bo_free(pool->dev, pool->split_partial_bo);
    for (agx_bo * chunk : pool->full_chunk_bos) {
        agx_bo_free(pool->dev, chunk);
    }
    delete pool;
}

void agx_kv_hybrid_reset(agx_kv_hybrid_pool * pool, bool release_pages) {
    if (!pool) {
        return;
    }
    if (pool->swa_pos_bo && pool->swa_pos_bo->cpu_ptr) {
        std::memset(pool->swa_pos_bo->cpu_ptr, 0xFF, pool->swa_pos_bo->size); // -1 = empty
    }
    if (pool->full_pos_bo && pool->full_pos_bo->cpu_ptr) {
        std::memset(pool->full_pos_bo->cpu_ptr, 0xFF, pool->full_pos_bo->size);
    }
    if (release_pages) {
        for (agx_bo * chunk : pool->full_chunk_bos) {
            agx_bo_free(pool->dev, chunk);
        }
        pool->full_chunk_bos.clear();
        pool->full_allocated_pages = 0;
        if (pool->page_table_bo && pool->page_table_bo->cpu_ptr) {
            std::memset(pool->page_table_bo->cpu_ptr, 0, pool->page_table_bo->size);
        }
    }
}

bool agx_kv_hybrid_ensure_full_capacity(agx_kv_hybrid_pool * pool, uint32_t required_tokens) {
    if (!pool || pool->n_full_slots == 0) {
        return true;
    }
    const uint32_t req_pages = (required_tokens + AGX_KV_PAGE_TOKENS - 1u) / AGX_KV_PAGE_TOKENS;
    if (req_pages <= pool->full_allocated_pages) {
        return true;
    }
    if (req_pages > pool->full_max_pages) {
        AGX_LOG_ERROR("agx_kv_hybrid_ensure_full_capacity: requested %u pages exceeds limit %u",
                      req_pages, pool->full_max_pages);
        return false;
    }

    const uint32_t chunk_pages = std::max<uint32_t>(pool->cfg.page_chunk_pages, 1u);
    auto * pt = static_cast<uint64_t *>(pool->page_table_bo->cpu_ptr);

    while (pool->full_allocated_pages < req_pages) {
        const uint32_t n_add = std::min<uint32_t>(chunk_pages, pool->full_max_pages - pool->full_allocated_pages);
        const size_t chunk_bytes = static_cast<size_t>(pool->n_full_slots) * n_add * pool->page_bytes;

        // Cold streaming BO: bypass SLC residency pin so 16 GiB full-attn KV doesn't evict hot working set!
        agx_bo * chunk_bo = agx_bo_alloc(
            pool->dev, chunk_bytes, AGX_BO_SLC_STREAM | AGX_BO_ZERO, "kv-full-page-chunk");
        if (!chunk_bo) {
            AGX_LOG_ERROR("agx_kv_hybrid_ensure_full_capacity: failed to allocate %.1f MiB full-KV chunk",
                          double(chunk_bytes) / (1024.0 * 1024.0));
            return false;
        }
        pool->full_chunk_bos.push_back(chunk_bo);

        const uint32_t base_pg = pool->full_allocated_pages;
        for (uint32_t s = 0; s < pool->n_full_slots; ++s) {
            for (uint32_t p = 0; p < n_add; ++p) {
                const size_t idx = (static_cast<size_t>(s) * n_add + p);
                pt[static_cast<size_t>(s) * pool->full_max_pages + (base_pg + p)] =
                    chunk_bo->gpu_va + idx * pool->page_bytes;
            }
        }
        pool->full_allocated_pages += n_add;
    }
    return true;
}

void agx_kv_hybrid_get_stats(const agx_kv_hybrid_pool * pool, agx_kv_hybrid_stats * out) {
    if (!pool || !out) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    out->n_swa_slots          = pool->n_swa_slots;
    out->n_full_slots         = pool->n_full_slots;
    out->swa_window           = pool->cfg.swa_window;
    out->full_allocated_pages = pool->full_allocated_pages;
    out->full_max_pages       = pool->full_max_pages;
    out->swa_ring_bytes       = pool->swa_ring_bo ? pool->swa_ring_bo->size : 0;
    out->full_pool_bytes      = static_cast<uint64_t>(pool->n_full_slots) *
                                pool->full_allocated_pages * pool->page_bytes;
    out->full_projected_1m_bytes = static_cast<uint64_t>(pool->n_full_slots) *
                                   AGX_KV_MAX_FULL_PAGES * pool->page_bytes;
    out->metadata_slc_bytes   = (pool->swa_pos_bo ? pool->swa_pos_bo->size : 0) +
                                (pool->page_table_bo ? pool->page_table_bo->size : 0) +
                                (pool->full_pos_bo ? pool->full_pos_bo->size : 0) +
                                (pool->split_partial_bo ? pool->split_partial_bo->size : 0);
}

bool agx_kv_hybrid_slot_is_full(const agx_kv_hybrid_pool * pool, uint32_t logical_slot) {
    return pool && logical_slot < pool->cfg.n_slots && pool->slot_is_full[logical_slot];
}

uint32_t agx_kv_hybrid_slot_pool_idx(const agx_kv_hybrid_pool * pool, uint32_t logical_slot) {
    return (pool && logical_slot < pool->cfg.n_slots) ? pool->slot_pool_idx[logical_slot] : 0u;
}

uint64_t agx_kv_hybrid_swa_slot_va(const agx_kv_hybrid_pool * pool, uint32_t logical_slot) {
    if (!pool || !pool->swa_ring_bo || logical_slot >= pool->cfg.n_slots || pool->slot_is_full[logical_slot]) {
        return 0;
    }
    return pool->swa_ring_bo->gpu_va + static_cast<uint64_t>(pool->slot_pool_idx[logical_slot]) * pool->swa_slot_stride_B;
}

uint64_t agx_kv_hybrid_swa_pos_va(const agx_kv_hybrid_pool * pool, uint32_t logical_slot) {
    if (!pool || !pool->swa_pos_bo || logical_slot >= pool->cfg.n_slots || pool->slot_is_full[logical_slot]) {
        return 0;
    }
    return pool->swa_pos_bo->gpu_va +
           static_cast<uint64_t>(pool->slot_pool_idx[logical_slot]) * pool->cfg.swa_window * sizeof(int32_t);
}

uint64_t agx_kv_hybrid_full_page_table_va(const agx_kv_hybrid_pool * pool, uint32_t logical_slot) {
    if (!pool || !pool->page_table_bo || logical_slot >= pool->cfg.n_slots || !pool->slot_is_full[logical_slot]) {
        return 0;
    }
    return pool->page_table_bo->gpu_va +
           static_cast<uint64_t>(pool->slot_pool_idx[logical_slot]) * pool->full_max_pages * sizeof(uint64_t);
}

uint64_t agx_kv_hybrid_full_pos_va(const agx_kv_hybrid_pool * pool) {
    return (pool && pool->full_pos_bo) ? pool->full_pos_bo->gpu_va : 0;
}

// ============================================================================
// KV Append Dispatch
// ============================================================================

bool agx_kv_hybrid_append(agx_kv_hybrid_pool *     pool,
                          agx_cdm_stream *         stream,
                          const agx_kv_append_args & args,
                          agx_barrier_mode         post_barrier) {
    if (!pool || !stream || args.n_tokens == 0 || args.logical_slot >= pool->cfg.n_slots) {
        return false;
    }

    const bool is_full = pool->slot_is_full[args.logical_slot];
    const agx_kv_dtype dt = is_full ? pool->cfg.full_dtype : pool->cfg.swa_dtype;
    const agx_kernel * k = get_kv_append_kernel(pool, dt);
    if (!k) {
        return false;
    }

    if (is_full) {
        if (!agx_kv_hybrid_ensure_full_capacity(pool, args.kv_write_base + args.n_tokens)) {
            return false;
        }
    }

    agx_push_kv_append pa = {};
    pa.k_src_va      = args.k_new_va;
    pa.v_src_va      = args.v_new_va;
    pa.pos_src_va    = args.pos_va;
    pa.n_tokens      = args.n_tokens;
    pa.n_head_kv     = pool->cfg.n_head_kv;
    pa.kv_write_base = args.kv_write_base;
    pa.flags         = (is_full ? 1u : 0u) | (args.src_is_f16 ? 2u : 0u);

    if (is_full) {
        pa.dst_k_or_pt_va = agx_kv_hybrid_full_page_table_va(pool, args.logical_slot);
        pa.pos_dst_va     = agx_kv_hybrid_full_pos_va(pool);
        pa.page_v_off_B   = pool->page_v_off_B;
        pa.page_ks_off_B  = pool->page_ks_off_B;
        pa.page_vs_off_B  = pool->page_vs_off_B;
    } else {
        const uint64_t slot_va = agx_kv_hybrid_swa_slot_va(pool, args.logical_slot);
        pa.dst_k_or_pt_va = slot_va;
        pa.dst_v_va       = slot_va + pool->swa_v_off_B;
        pa.dst_ks_va      = slot_va + pool->swa_ks_off_B;
        pa.dst_vs_va      = slot_va + pool->swa_vs_off_B;
        pa.pos_dst_va     = agx_kv_hybrid_swa_pos_va(pool, args.logical_slot);
        pa.ring_mask      = pool->cfg.swa_window - 1u;
    }

    return agx_cdm_dispatch(stream, k, pool->cfg.n_head_kv, args.n_tokens, 1u,
                            &pa, sizeof(pa), post_barrier);
}

// ============================================================================
// SWA-1024 Ring Attention & Split-K Flash-Decoding Dispatchers
// ============================================================================

bool agx_kv_hybrid_dispatch_swa_attn(agx_kv_hybrid_pool *        pool,
                                     agx_cdm_stream *            stream,
                                     const agx_attn_stage_args & args,
                                     agx_barrier_mode            post_barrier) {
    if (!pool || !stream || args.n_tokens == 0 || args.n_head_q == 0 ||
        args.head_dim != 128u || args.logical_slot >= pool->cfg.n_slots ||
        pool->slot_is_full[args.logical_slot]) {
        return false;
    }

    const agx_kernel * k = get_swa_kernel(pool, pool->cfg.swa_dtype);
    if (!k) {
        return false;
    }

    const uint64_t slot_va = agx_kv_hybrid_swa_slot_va(pool, args.logical_slot);

    agx_push_swa_attn pc = {};
    pc.q_va         = args.q_va;
    pc.k_va         = slot_va;
    pc.v_va         = slot_va + pool->swa_v_off_B;
    pc.ks_va        = slot_va + pool->swa_ks_off_B;
    pc.vs_va        = slot_va + pool->swa_vs_off_B;
    pc.pos_ring_va  = agx_kv_hybrid_swa_pos_va(pool, args.logical_slot);
    pc.rel_table_va = args.rel_table_va;
    pc.sinks_va     = args.sinks_va;
    pc.q_gate_va    = args.q_gate_va;
    pc.pos_q_va     = args.pos_q_va;
    pc.out_va       = args.out_va;
    pc.n_tokens     = args.n_tokens;
    pc.swa_window   = pool->cfg.swa_window;
    pc.n_kv_scan    = std::min<uint32_t>(args.n_kv > 0 ? args.n_kv : pool->cfg.swa_window, pool->cfg.swa_window);
    pc.n_head_q     = args.n_head_q;
    pc.n_head_kv    = args.n_head_kv;
    pc.n_rel_extent = args.n_rel_extent;
    pc.pos_q_base   = args.pos_q_base;
    pc.flags        = args.flags | AGX_ATTN_FLAG_SWA_RING_KV;
    pc.kq_scale     = args.kq_scale;

    return agx_cdm_dispatch(stream, k, args.n_head_q, args.n_tokens, 1u,
                            &pc, sizeof(pc), post_barrier);
}

static uint32_t choose_flash_decode_splits(uint32_t n_kv, uint32_t n_head_q, uint32_t n_tokens) {
    if (n_kv <= 256u) {
        return 1u;
    }
    // Saturate all 40 M4 Max cores (target >= 160 WGs when n_kv is large, >= 128 keys/split)
    const uint32_t max_by_kv   = std::max<uint32_t>(1u, (n_kv + 127u) / 128u);
    const uint32_t base_wgs    = std::max<uint32_t>(1u, n_head_q * n_tokens);
    const uint32_t want_splits = (base_wgs >= 160u) ? 1u : ((160u + base_wgs - 1u) / base_wgs);
    uint32_t splits = std::max<uint32_t>(want_splits, (n_kv + 511u) / 512u);
    splits = std::min<uint32_t>(splits, max_by_kv);
    splits = std::min<uint32_t>(splits, AGX_KV_MAX_SPLITS);
    return std::max<uint32_t>(1u, splits);
}

bool agx_kv_hybrid_dispatch_full_attn(agx_kv_hybrid_pool *        pool,
                                      agx_cdm_stream *            stream,
                                      const agx_attn_stage_args & args,
                                      agx_barrier_mode            post_barrier) {
    if (!pool || !stream || args.n_tokens == 0 || args.n_head_q == 0 ||
        (args.head_dim != 64u && args.head_dim != 128u)) {
        return false;
    }

    const bool is_paged = (args.flags & AGX_ATTN_FLAG_PAGED_KV) != 0u;
    if (is_paged && (args.logical_slot >= pool->cfg.n_slots || !pool->slot_is_full[args.logical_slot])) {
        return false;
    }

    const agx_kv_dtype k_dt = is_paged ? pool->cfg.full_dtype : args.ext_k_dtype;
    const agx_kv_dtype v_dt = is_paged ? pool->cfg.full_dtype : args.ext_v_dtype;
    const agx_kernel * k_p1 = get_full_phase1_kernel(pool, args.head_dim, k_dt, v_dt);
    if (!k_p1) {
        return false;
    }

    const uint32_t n_splits = choose_flash_decode_splits(args.n_kv, args.n_head_q, args.n_tokens);
    if (n_splits > 1u) {
        if (!ensure_split_partial_bo(pool, args.n_tokens, args.n_head_q, n_splits)) {
            return false;
        }
    }

    uint32_t n_head_log2 = 1u;
    while ((n_head_log2 << 1) <= args.n_head_q) {
        n_head_log2 <<= 1;
    }
    const float m0 = (args.max_bias > 0.0f) ? std::pow(2.0f, -(args.max_bias       ) / float(n_head_log2)) : 1.0f;
    const float m1 = (args.max_bias > 0.0f) ? std::pow(2.0f, -(args.max_bias * 0.5f) / float(n_head_log2)) : 1.0f;

    agx_push_full_attn pc = {};
    pc.q_va             = args.q_va;
    pc.rel_table_va     = args.rel_table_va;
    pc.sinks_va         = args.sinks_va;
    pc.q_gate_va        = args.q_gate_va;
    pc.pos_q_va         = args.pos_q_va;
    pc.kq_mask_va       = args.kq_mask_va;
    pc.partial_va       = (n_splits > 1u) ? pool->split_partial_bo->gpu_va : 0;
    pc.out_va           = args.out_va;
    pc.n_tokens         = args.n_tokens;
    pc.n_kv             = args.n_kv;
    pc.n_splits         = n_splits;
    pc.n_head_q         = args.n_head_q;
    pc.n_head_kv        = args.n_head_kv;
    pc.head_dim         = args.head_dim;
    pc.n_rel_extent     = args.n_rel_extent;
    pc.pos_q_base       = args.pos_q_base;
    pc.flags            = args.flags | (args.max_bias > 0.0f ? AGX_ATTN_FLAG_ALIBI : 0u);
    const uint32_t q_elem_B = ((args.flags & AGX_ATTN_FLAG_Q_F16) != 0u) ? 2u : 4u;
    pc.q_nb1            = args.q_nb1 ? args.q_nb1 : (args.n_head_q * args.head_dim * q_elem_B);
    pc.q_nb2            = args.q_nb2 ? args.q_nb2 : (args.head_dim * q_elem_B);
    pc.mask_n_head_log2 = n_head_log2;
    pc.kq_scale         = args.kq_scale;
    pc.logit_softcap    = args.logit_softcap;
    pc.m0               = m0;
    pc.m1               = m1;
    pc.rel_pos_pack_va  = args.rel_pos_pack_va;
    pc.rel_table_len    = args.rel_table_len;
    pc.rel_cp_off       = args.rel_cp_off;
    pc.rel_qp_off       = args.rel_qp_off;
    pc.rel_tm_off       = args.rel_tm_off;

    if (is_paged) {
        pc.k_or_pt_va    = agx_kv_hybrid_full_page_table_va(pool, args.logical_slot);
        pc.pos_kv_va     = agx_kv_hybrid_full_pos_va(pool);
        pc.page_v_off_B  = pool->page_v_off_B;
        pc.page_ks_off_B = pool->page_ks_off_B;
        pc.page_vs_off_B = pool->page_vs_off_B;
    } else {
        pc.k_or_pt_va    = args.ext_k_va;
        pc.v_va          = args.ext_v_va;
        pc.ext_k_nb1     = args.ext_k_nb1;
        pc.ext_k_nb2     = args.ext_k_nb2;
        pc.ext_v_nb1     = args.ext_v_nb1;
        pc.ext_v_nb2     = args.ext_v_nb2;
        pc.ext_mask_nb1  = args.ext_mask_nb1;
        pc.ext_mask_nb2  = args.ext_mask_nb2;
        pc.ext_mask_ne2  = args.ext_mask_ne2 ? args.ext_mask_ne2 : 1u;
    }

    static const bool disable_p3 = []() {
        const char * e = std::getenv("GGML_AGX_FA_P3_DISABLE");
        return e && std::atoi(e) != 0;
    }();
    static const uint32_t pf4_split_min_kv = []() {
        const char * e = std::getenv("GGML_AGX_PF4_SPLIT_MIN_KV");
        if (e && *e) {
            int v = std::atoi(e);
            if (v > 0) return static_cast<uint32_t>(v);
        }
        return 128u;
    }();
    auto compute_pf4_splits = [&](uint32_t n_kv_v, uint32_t n_hq_v, uint32_t n_tok_v) -> uint32_t {
        if (n_kv_v < pf4_split_min_kv) {
            return 1u;
        }
        const uint32_t gy       = (n_tok_v + 7u) / 8u;
        const uint32_t base_wgs = std::max<uint32_t>(1u, n_hq_v * gy);
        const uint32_t max_by_kv = std::max<uint32_t>(1u, (n_kv_v + 63u) / 64u);
        uint32_t sp = 1u;
        if (n_kv_v <= 512u) {
            sp = (base_wgs >= 40u) ? 1u : 2u;
        } else if (n_kv_v <= 2048u) {
            sp = (base_wgs >= 80u) ? 2u : 4u;
        } else {
            const uint32_t want = (base_wgs >= 160u) ? 1u : ((160u + base_wgs - 1u) / base_wgs);
            sp = std::max<uint32_t>(want, (n_kv_v + 255u) / 256u);
        }
        sp = std::min<uint32_t>(sp, max_by_kv);
        sp = std::min<uint32_t>(sp, AGX_KV_MAX_SPLITS);
        return std::max<uint32_t>(1u, sp);
    };

    // Wave 9 Lean 8-SG Br=16 Bc=64 Zero-K/V-SRAM Coopmat Prefill fastpath
    // (10.5 KB SRAM -> 3 WGs/core = 24 SGs/core, <= 64 GPR32, 2 barriers/64 KV, native O(n_kv) banded rel-bias!)
    if (!disable_p3 && !is_paged && args.head_dim == 128u && args.n_tokens >= 16u && args.n_kv >= 32u &&
        k_dt != AGX_KV_F32 && v_dt != AGX_KV_F32 &&
        k_dt != AGX_KV_Q8_0_PLANAR && v_dt != AGX_KV_Q8_0_PLANAR &&
        (args.flags & (AGX_ATTN_FLAG_CAUSAL | AGX_ATTN_FLAG_HAS_REL_TABLE)) == 0u) {
        const agx_kernel * k_pf = get_full_prefill_kernel(pool, k_dt, v_dt);
        if (k_pf) {
            pc.n_splits   = 1u;
            pc.partial_va = ((args.flags & AGX_ATTN_FLAG_KV_BOUNDS) != 0u) ? args.kv_bounds_va : 0ull;
            const uint32_t grid_y = (args.n_tokens + 15u) / 16u;
            return agx_cdm_dispatch(stream, k_pf, args.n_head_q, grid_y, 1u,
                                    &pc, sizeof(pc), post_barrier);
        }
    }

    // DSpark M=2..15 Split-K 8x8 FP16 Coopmat fastpath (2 SGs = 8 Q/WG, 8x KV reuse, banded rel-bias!)
    if (!is_paged && args.head_dim == 128u && args.n_tokens >= 2u && args.n_tokens < 16u &&
        args.n_kv >= 32u && k_dt != AGX_KV_F32 && v_dt != AGX_KV_F32 &&
        k_dt != AGX_KV_Q8_0_PLANAR && v_dt != AGX_KV_Q8_0_PLANAR &&
        (args.flags & (AGX_ATTN_FLAG_CAUSAL | AGX_ATTN_FLAG_HAS_REL_TABLE)) == 0u) {
        const agx_kernel * k_pf4 = get_full_prefill_m8_kernel(pool, k_dt, v_dt);
        if (k_pf4) {
            const uint32_t grid_y     = (args.n_tokens + 7u) / 8u;
            const uint32_t pf4_splits = compute_pf4_splits(args.n_kv, args.n_head_q, args.n_tokens);
            if (pf4_splits == 1u) {
                pc.n_splits   = 1u;
                pc.partial_va = 0;
                return agx_cdm_dispatch(stream, k_pf4, args.n_head_q, grid_y, 1u,
                                        &pc, sizeof(pc), post_barrier);
            }
            if (!ensure_split_partial_bo(pool, args.n_tokens, args.n_head_q, pf4_splits)) {
                return false;
            }
            const agx_kernel * k_p2_m8 = get_full_phase2_kernel(pool, args.head_dim);
            if (!k_p2_m8) {
                return false;
            }
            pc.n_splits   = pf4_splits;
            pc.partial_va = pool->split_partial_bo->gpu_va;
            if (!agx_cdm_is_empty(stream)) {
                agx_cdm_barrier_usc(stream);
            }
            if (!agx_cdm_dispatch(stream, k_pf4, args.n_head_q, grid_y, pf4_splits,
                                  &pc, sizeof(pc), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
            const agx_barrier_mode p2_bar = (post_barrier == AGX_BARRIER_NONE)
                ? AGX_BARRIER_STRONG_USC
                : post_barrier;
            return agx_cdm_dispatch(stream, k_p2_m8, args.n_head_q, args.n_tokens, 1u,
                                    &pc, sizeof(pc), p2_bar);
        }
    }

    if (n_splits == 1u) {
        return agx_cdm_dispatch(stream, k_p1, args.n_head_q, args.n_tokens, 1u,
                                &pc, sizeof(pc), post_barrier);
    }

    const agx_kernel * k_p2 = get_full_phase2_kernel(pool, args.head_dim);
    if (!k_p2) {
        return false;
    }

    // Protect shared pool->split_partial_bo across consecutive split-K calls in the same CDM stream!
    if (!agx_cdm_is_empty(stream)) {
        agx_cdm_barrier_usc(stream);
    }

    // Phase 1: Split-K across all 40 cores -> SLC-pinned partial scratch
    if (!agx_cdm_dispatch(stream, k_p1, args.n_head_q, args.n_tokens, n_splits,
                          &pc, sizeof(pc), AGX_BARRIER_STRONG_USC)) {
        return false;
    }

    // Phase 2: Reduction + single-application sink + FWHT64 + sigmoid(Qgate)
    const agx_barrier_mode p2_bar = (post_barrier == AGX_BARRIER_NONE)
        ? AGX_BARRIER_STRONG_USC
        : post_barrier;
    return agx_cdm_dispatch(stream, k_p2, args.n_head_q, args.n_tokens, 1u,
                            &pc, sizeof(pc), p2_bar);
}

bool agx_kv_hybrid_dispatch_attn(agx_kv_hybrid_pool *        pool,
                                 agx_cdm_stream *            stream,
                                 const agx_attn_stage_args & args,
                                 agx_barrier_mode            post_barrier) {
    if (!pool) {
        return false;
    }
    if ((args.flags & AGX_ATTN_FLAG_SWA_RING_KV) != 0u ||
        ((args.flags & AGX_ATTN_FLAG_PAGED_KV) == 0u &&
         args.ext_k_va == 0 &&
         !agx_kv_hybrid_slot_is_full(pool, args.logical_slot))) {
        return agx_kv_hybrid_dispatch_swa_attn(pool, stream, args, post_barrier);
    }
    agx_attn_stage_args full_args = args;
    if (full_args.ext_k_va == 0) {
        full_args.flags |= AGX_ATTN_FLAG_PAGED_KV;
    }
    return agx_kv_hybrid_dispatch_full_attn(pool, stream, full_args, post_barrier);
}

// ============================================================================
// Generic GGML_OP_FLASH_ATTN_EXT Bridge (zero-copy external ggml_tensor K/V)
// ============================================================================

static bool ggml_type_to_ext_kv(ggml_type t, agx_kv_dtype * out) {
    switch (t) {
        case GGML_TYPE_F8_E4M3: *out = AGX_KV_F8_E4M3;   return true;
        case GGML_TYPE_F16:     *out = AGX_KV_F16;       return true;
        case GGML_TYPE_F32:     *out = AGX_KV_F32;       return true;
        case GGML_TYPE_Q8_0:    *out = AGX_KV_Q8_0_GGML; return true;
        default:                return false;
    }
}

bool agx_kv_hybrid_supports_ggml_flash_attn(const struct ggml_tensor * dst) {
    if (!dst || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * q     = dst->src[0]; // [D, N, Hq, B]
    const ggml_tensor * k     = dst->src[1]; // [D, KV, Hkv, B]
    const ggml_tensor * v     = dst->src[2]; // [D, KV, Hkv, B]
    const ggml_tensor * mask  = dst->src[3]; // optional [KV, N, Hmask, B]
    const ggml_tensor * sinks = dst->src[4]; // optional [Hq] or [Hq, N*B]

    if (!q || !k || !v) {
        return false;
    }
    // Reject sparse kv_rows; accept src[6]/src[7] ONLY for banded relative-position bias (op_params[5] == 2)
    if (dst->src[5]) {
        return false;
    }
    const bool has_banded = ggml_flash_attn_ext_has_banded_rel_bias(dst);
    if (!has_banded && (dst->src[6] || dst->src[7])) {
        return false;
    }
    if (has_banded) {
        const ggml_tensor * rtab = dst->src[6];
        const ggml_tensor * rpos = dst->src[7];
        const int32_t extent     = dst->op_params[6];
        if (!rtab || !rpos || !mask || extent <= 0 || extent > 65535) {
            return false;
        }
        if (rtab->type != GGML_TYPE_F32 || !ggml_is_contiguous(rtab) ||
            rtab->ne[0] <= 0 || rtab->ne[0] > 65535 ||
            rtab->ne[1] != q->ne[2] || rtab->ne[2] != q->ne[1] * q->ne[3] || rtab->ne[3] != 1) {
            return false;
        }
        const int64_t n_tiles64 = (k->ne[1] + 63) / 64;
        const int64_t req_pos   = q->ne[3] * k->ne[1] + q->ne[1] * q->ne[3] + q->ne[3] * n_tiles64 * 2;
        if (rpos->type != GGML_TYPE_I32 || !ggml_is_contiguous(rpos) || rpos->ne[0] < req_pos) {
            return false;
        }
    }
    if (q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (q->ne[0] != k->ne[0] || k->ne[0] != v->ne[0] || dst->ne[0] != v->ne[0]) {
        return false;
    }
    if (q->ne[0] != 64 && q->ne[0] != 128) {
        return false;
    }
    if (k->ne[1] != v->ne[1] || k->ne[2] != v->ne[2] || k->ne[2] <= 0 ||
        (q->ne[2] % k->ne[2]) != 0) {
        return false;
    }
    if (k->ne[3] <= 0 || v->ne[3] <= 0 || (q->ne[3] % k->ne[3]) != 0 || (q->ne[3] % v->ne[3]) != 0) {
        return false;
    }
    if (k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type)) {
        return false;
    }
    agx_kv_dtype k_dt, v_dt;
    if (!ggml_type_to_ext_kv(k->type, &k_dt) || !ggml_type_to_ext_kv(v->type, &v_dt)) {
        return false;
    }
    if (mask) {
        if ((mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) ||
            mask->nb[0] != ggml_type_size(mask->type) || mask->ne[0] < k->ne[1] ||
            mask->ne[1] < q->ne[1] || mask->ne[2] <= 0 || mask->ne[3] <= 0) {
            return false;
        }
    }
    if (sinks) {
        if (sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks) || sinks->ne[0] != q->ne[2]) {
            return false;
        }
        if (sinks->ne[1] != 1 && sinks->ne[1] != q->ne[1] * q->ne[3]) {
            return false;
        }
        if (sinks->ne[2] != 1 || sinks->ne[3] != 1) {
            return false;
        }
    }
    return true;
}

bool agx_kv_hybrid_ggml_flash_attn_uses_scratch(const struct ggml_tensor * dst) {
    if (!agx_kv_hybrid_supports_ggml_flash_attn(dst)) {
        return true;
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    agx_kv_dtype k_dt = AGX_KV_F32;
    agx_kv_dtype v_dt = AGX_KV_F32;
    ggml_type_to_ext_kv(k->type, &k_dt);
    ggml_type_to_ext_kv(v->type, &v_dt);
    const uint32_t head_dim = static_cast<uint32_t>(q->ne[0]);
    const uint32_t n_tokens = static_cast<uint32_t>(q->ne[1]);
    const uint32_t n_kv     = static_cast<uint32_t>(k->ne[1]);
    const uint32_t n_head_q = static_cast<uint32_t>(q->ne[2]);
    static const bool disable_p3 = []() {
        const char * e = std::getenv("GGML_AGX_FA_P3_DISABLE");
        return e && std::atoi(e) != 0;
    }();
    static const uint32_t pf4_split_min_kv = []() {
        const char * e = std::getenv("GGML_AGX_PF4_SPLIT_MIN_KV");
        if (e && *e) {
            int v = std::atoi(e);
            if (v > 0) return static_cast<uint32_t>(v);
        }
        return 128u;
    }();
    if (!disable_p3 && head_dim == 128u && n_tokens >= 16u && n_kv >= 32u &&
        k_dt != AGX_KV_F32 && v_dt != AGX_KV_F32 &&
        k_dt != AGX_KV_Q8_0_PLANAR && v_dt != AGX_KV_Q8_0_PLANAR) {
        return false; // PASS_MODE=3 single-pass coopmat prefill never uses split_partial_bo
    }
    if (head_dim == 128u && n_tokens >= 2u && n_tokens < 16u && n_kv >= 32u &&
        k_dt != AGX_KV_F32 && v_dt != AGX_KV_F32 &&
        k_dt != AGX_KV_Q8_0_PLANAR && v_dt != AGX_KV_Q8_0_PLANAR) {
        if (n_kv < pf4_split_min_kv) {
            return false; // PASS_MODE=4 pf4_splits == 1
        }
        const uint32_t gy = (n_tokens + 7u) / 8u;
        if (n_kv <= 512u && n_head_q * gy >= 40u) {
            return false;
        }
        return true; // PASS_MODE=4 split-K uses split_partial_bo
    }
    return choose_flash_decode_splits(n_kv, n_head_q, n_tokens) > 1u;
}

bool agx_kv_hybrid_dispatch_ggml_flash_attn_ext(agx_kv_hybrid_pool *            pool,
                                                agx_cdm_stream *                stream,
                                                const struct ggml_tensor *      dst,
                                                const agx_ggml_flash_attn_epi * epi,
                                                agx_barrier_mode                post_barrier) {
    if (!pool || !stream || !agx_kv_hybrid_supports_ggml_flash_attn(dst)) {
        return false;
    }
    const ggml_tensor * q     = dst->src[0]; // [D, N, Hq, B]
    const ggml_tensor * k     = dst->src[1]; // [D, KV, Hkv, B]
    const ggml_tensor * v     = dst->src[2]; // [D, KV, Hkv, B]
    const ggml_tensor * mask  = dst->src[3]; // optional [KV, N, Hmask, B]
    const ggml_tensor * sinks = dst->src[4]; // optional [Hq] or [Hq, N*B]
    const bool has_banded     = ggml_flash_attn_ext_has_banded_rel_bias(dst);
    const ggml_tensor * rtab  = has_banded ? dst->src[6] : nullptr;
    const ggml_tensor * rpos  = has_banded ? dst->src[7] : nullptr;

    agx_kv_dtype k_dt = AGX_KV_F16;
    agx_kv_dtype v_dt = AGX_KV_F16;
    ggml_type_to_ext_kv(k->type, &k_dt);
    ggml_type_to_ext_kv(v->type, &v_dt);

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    std::memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const uint64_t base_q_va    = agx_tensor_gpu_va(q);
    const uint64_t base_k_va    = agx_tensor_gpu_va(k);
    const uint64_t base_v_va    = agx_tensor_gpu_va(v);
    const uint64_t base_out_va  = (epi && epi->out_va) ? epi->out_va : agx_tensor_gpu_va(dst);
    const uint64_t out_nb3      = (epi && epi->out_va && epi->out_nb3) ? epi->out_nb3 : static_cast<uint64_t>(dst->nb[3]);
    const uint64_t base_mask_va = mask  ? agx_tensor_gpu_va(mask)  : 0;
    const uint64_t sinks_va     = sinks ? agx_tensor_gpu_va(sinks) : 0;
    const uint64_t base_rtab_va = rtab  ? agx_tensor_gpu_va(rtab)  : 0;
    const uint64_t base_rpos_va = rpos  ? agx_tensor_gpu_va(rpos)  : 0;

    if (!base_q_va || !base_k_va || !base_v_va || !base_out_va || (mask && !base_mask_va) ||
        (sinks && !sinks_va) || (has_banded && (!base_rtab_va || !base_rpos_va))) {
        return false;
    }

    const int64_t  B          = q->ne[3];
    const int64_t  rk3        = q->ne[3] / k->ne[3];
    const int64_t  rv3        = q->ne[3] / v->ne[3];
    const uint32_t n_tiles64  = static_cast<uint32_t>((k->ne[1] + 63) / 64);
    const uint32_t rel_extent = has_banded ? static_cast<uint32_t>(dst->op_params[6]) : 0u;
    const uint32_t rel_t_len  = rtab ? static_cast<uint32_t>(rtab->ne[0]) : 0u;

    for (int64_t b = 0; b < B; ++b) {
        agx_attn_stage_args args = {};
        args.n_tokens      = static_cast<uint32_t>(q->ne[1]);
        args.n_kv          = static_cast<uint32_t>(k->ne[1]);
        args.n_head_q      = static_cast<uint32_t>(q->ne[2]);
        args.n_head_kv     = static_cast<uint32_t>(k->ne[2]);
        args.head_dim      = static_cast<uint32_t>(q->ne[0]);
        args.kq_scale      = (logit_softcap != 0.0f) ? (scale / logit_softcap) : scale;
        args.max_bias      = max_bias;
        args.logit_softcap = logit_softcap;
        args.flags         = (logit_softcap != 0.0f ? AGX_ATTN_FLAG_SOFTCAP : 0u)
                           | (q->type == GGML_TYPE_F16 ? AGX_ATTN_FLAG_Q_F16 : 0u)
                           | (dst->type == GGML_TYPE_F16 ? AGX_ATTN_FLAG_OUT_F16 : 0u)
                           | (epi ? (epi->extra_flags & ~AGX_ATTN_FLAG_KV_BOUNDS) : 0u);

        args.q_va          = base_q_va   + static_cast<uint64_t>(b) * q->nb[3];
        args.q_nb1         = static_cast<uint32_t>(q->nb[1]);
        args.q_nb2         = static_cast<uint32_t>(q->nb[2]);
        args.out_va        = base_out_va + static_cast<uint64_t>(b) * out_nb3;
        if (epi && epi->q_gate_va) {
            args.q_gate_va = epi->q_gate_va + static_cast<uint64_t>(b) * epi->q_gate_nb3;
        }
        if (epi && (epi->extra_flags & AGX_ATTN_FLAG_KV_BOUNDS) != 0u && epi->kv_bounds_va && B == 1) {
            args.flags       |= AGX_ATTN_FLAG_KV_BOUNDS;
            args.kv_bounds_va = epi->kv_bounds_va;
        }
        args.ext_k_dtype   = k_dt;
        args.ext_v_dtype   = v_dt;
        args.ext_k_va      = base_k_va + static_cast<uint64_t>(b / rk3) * k->nb[3];
        args.ext_v_va      = base_v_va + static_cast<uint64_t>(b / rv3) * v->nb[3];
        args.ext_k_nb1     = static_cast<uint32_t>(k->nb[1]);
        args.ext_k_nb2     = static_cast<uint32_t>(k->nb[2]);
        args.ext_v_nb1     = static_cast<uint32_t>(v->nb[1]);
        args.ext_v_nb2     = static_cast<uint32_t>(v->nb[2]);

        if (mask) {
            args.kq_mask_va   = base_mask_va + static_cast<uint64_t>(b % mask->ne[3]) * mask->nb[3];
            args.ext_mask_nb1 = static_cast<uint32_t>(mask->nb[1]);
            args.ext_mask_nb2 = (mask->ne[2] > 1) ? static_cast<uint32_t>(mask->nb[2]) : 0u;
            args.ext_mask_ne2 = static_cast<uint32_t>(mask->ne[2]);
            args.flags       |= (mask->type == GGML_TYPE_F16) ? AGX_ATTN_FLAG_HAS_MASK_F16 : AGX_ATTN_FLAG_HAS_MASK_F32;
        }
        if (sinks) {
            if (sinks->ne[1] > 1) {
                args.sinks_va = sinks_va + static_cast<uint64_t>(b * q->ne[1] * q->ne[2]) * sizeof(float);
                args.flags   |= AGX_ATTN_FLAG_SINK_PER_TOK_HEAD;
            } else {
                args.sinks_va = sinks_va;
                args.flags   |= AGX_ATTN_FLAG_SINK_PER_HEAD;
            }
        }
        if (has_banded) {
            args.flags           |= AGX_ATTN_FLAG_BANDED_REL_BIAS;
            args.n_rel_extent     = rel_extent;
            args.rel_table_len    = rel_t_len;
            args.rel_table_va     = base_rtab_va + static_cast<uint64_t>(b * q->ne[1] * q->ne[2] * rtab->ne[0]) * sizeof(float);
            args.rel_pos_pack_va  = base_rpos_va;
            args.rel_cp_off       = static_cast<uint32_t>(b * k->ne[1]);
            args.rel_qp_off       = static_cast<uint32_t>(B * k->ne[1] + b * q->ne[1]);
            args.rel_tm_off       = static_cast<uint32_t>(B * k->ne[1] + B * q->ne[1] + b * n_tiles64 * 2u);
        }

        const agx_barrier_mode b_bar = (b + 1 == B) ? post_barrier : AGX_BARRIER_NONE;
        if (!agx_kv_hybrid_dispatch_full_attn(pool, stream, args, b_bar)) {
            return false;
        }
    }
    return true;
}

bool agx_kv_hybrid_dispatch_ggml_flash_attn(agx_kv_hybrid_pool *       pool,
                                            agx_cdm_stream *           stream,
                                            const struct ggml_tensor * dst,
                                            agx_barrier_mode           post_barrier) {
    return agx_kv_hybrid_dispatch_ggml_flash_attn_ext(pool, stream, dst, nullptr, post_barrier);
}
