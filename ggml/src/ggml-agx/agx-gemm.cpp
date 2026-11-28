#include "agx-gemm.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

// ============================================================================
// Pillar 2: Kernel Cache & Dispatcher (`nvfp4_vecmat`, `mxfp8_q8_vecmat`,
//           and 8x8 `simdgroup_matrix` `agx_mma_gemm`)
// ============================================================================

struct agx_push_cvt_f16 {
    uint64_t src_va;
    uint64_t dst_va;
    uint32_t n_vec4;
    uint32_t _pad;
};
static_assert(sizeof(agx_push_cvt_f16) == 24, "agx_push_cvt_f16 size check");

struct agx_gemm_state {
    std::unordered_map<std::string, const agx_kernel *> cache;

    // Wave 9 B_IS_F16 activation pre-conversion scratch (SLC-pinned <= 24 MiB, hazard-tracked!)
    agx_bo *               x_f16_bo         = nullptr;
    size_t                 x_f16_cap_bytes  = 0;
    const agx_kernel *     k_cvt_f16        = nullptr;
    const agx_cdm_stream * last_stream      = nullptr;
    uint64_t               last_stream_gen  = 0;
    uint64_t               last_strong_bars = UINT64_MAX;
};

static agx_gemm_state * get_gemm_state(agx_backend_context * ctx) {
    if (!ctx) {
        return nullptr;
    }
    if (!ctx->gemm_state) {
        ctx->gemm_state = new agx_gemm_state();
    }
    return static_cast<agx_gemm_state *>(ctx->gemm_state);
}

bool agx_gemm_init(agx_backend_context * ctx) {
    return get_gemm_state(ctx) != nullptr;
}

void agx_gemm_cleanup(agx_backend_context * ctx) {
    if (!ctx || !ctx->gemm_state) {
        return;
    }
    auto * st = static_cast<agx_gemm_state *>(ctx->gemm_state);
    if (st->x_f16_bo && ctx->drm) {
        agx_bo_free(ctx->drm, st->x_f16_bo);
        st->x_f16_bo = nullptr;
        st->x_f16_cap_bytes = 0;
    }
    delete st;
    ctx->gemm_state = nullptr;
}

static bool gemm_ensure_x_f16_bo(agx_backend_context * ctx, agx_gemm_state * st, size_t req_bytes) {
    if (!ctx || !ctx->drm || !st || req_bytes == 0 || req_bytes > 24u * 1024u * 1024u) {
        return false;
    }
    if (st->x_f16_bo && req_bytes <= st->x_f16_cap_bytes) {
        return true;
    }
    if (st->x_f16_bo) {
        agx_bo_free(ctx->drm, st->x_f16_bo);
        st->x_f16_bo = nullptr;
        st->x_f16_cap_bytes = 0;
    }
    const size_t alloc_B = agx_align_up(std::max<size_t>(req_bytes, 256u * 1024u), AGX_PAGE_SIZE);
    st->x_f16_bo = agx_bo_alloc(ctx->drm, alloc_B, AGX_BO_SLC_PIN | AGX_BO_WRITEBACK, "gemm-x-f16-slc");
    if (!st->x_f16_bo) {
        return false;
    }
    st->x_f16_cap_bytes = alloc_B;
    return true;
}

static const agx_kernel * gemm_get_cvt_f16_kernel(agx_backend_context * ctx, agx_gemm_state * st) {
    if (!ctx || !ctx->compiler || !st) {
        return nullptr;
    }
    if (!st->k_cvt_f16) {
        static const char * k_src = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly  buffer BufV4R { vec4    v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8)  writeonly buffer BufH4W { f16vec4 v[]; };
layout(push_constant, scalar) uniform PushCvt {
    uint64_t src_va;
    uint64_t dst_va;
    uint     n_vec4;
    uint     _pad;
} pc;
void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i < pc.n_vec4) {
        BufH4W(pc.dst_va).v[i] = f16vec4(BufV4R(pc.src_va).v[i]);
    }
}
)GLSL";
        agx_compile_opts opts;
        opts.push_constant_bytes = sizeof(agx_push_cvt_f16);
        opts.is_hot_kernel       = true;
        opts.strict_zero_spill   = true;
        st->k_cvt_f16 = agx_compiler_compile_glsl(ctx->compiler, "agx_gemm_cvt_f32_to_f16_v4", k_src, opts);
    }
    return st->k_cvt_f16;
}

static const agx_kernel * get_or_compile_nvfp4_vecmat(agx_backend_context * ctx,
                                                      bool                  repacked,
                                                      uint32_t              batch_m,
                                                      uint32_t              k_sgs,
                                                      bool                  prec_f32) {
    auto * st = get_gemm_state(ctx);
    if (!st || !ctx->compiler) {
        return nullptr;
    }
    char name[96];
    std::snprintf(name, sizeof(name), "nvfp4_vecmat_r%u_m%u_k%u_p%u",
                  repacked ? 1u : 0u, batch_m, k_sgs, prec_f32 ? 1u : 0u);
    auto it = st->cache.find(name);
    if (it != st->cache.end()) {
        return it->second;
    }

    agx_compile_opts opts;
    opts.defines.push_back(std::string("REPACKED=") + (repacked ? "1" : "0"));
    opts.defines.push_back(std::string("BATCH_M=") + std::to_string(batch_m));
    opts.defines.push_back(std::string("K_SGS=") + std::to_string(k_sgs));
    opts.defines.push_back(std::string("PREC_F32=") + (prec_f32 ? "1" : "0"));
    opts.push_constant_bytes = sizeof(agx_gemm_push_args);
    opts.is_hot_kernel       = true;
    opts.strict_zero_spill   = true;

    const agx_kernel * k = agx_compiler_compile_file(ctx->compiler, name, "nvfp4_vecmat.comp", opts);
    if (k) {
        st->cache.emplace(name, k);
    }
    return k;
}

static const agx_kernel * get_or_compile_q8_mxfp8_vecmat(agx_backend_context * ctx,
                                                         uint32_t              quant_mode,
                                                         uint32_t              batch_m,
                                                         bool                  k_split_row,
                                                         uint32_t              wg_sgs = 0u,
                                                         bool                  x_aligned4 = false) {
    auto * st = get_gemm_state(ctx);
    if (!st || !ctx->compiler) {
        return nullptr;
    }
    const bool     shx_path = (quant_mode == 0u && !k_split_row && batch_m > 1u);
    const uint32_t eff_sgs  = wg_sgs ? wg_sgs : (shx_path ? 8u : 4u);
    const bool     eff_a4   = x_aligned4 && shx_path;
    char name[96];
    std::snprintf(name, sizeof(name), "mxfp8_q8_vecmat_q%u_m%u_s%u_g%u_a%u",
                  quant_mode, batch_m, k_split_row ? 1u : 0u, eff_sgs, eff_a4 ? 1u : 0u);
    auto it = st->cache.find(name);
    if (it != st->cache.end()) {
        return it->second;
    }

    agx_compile_opts opts;
    opts.defines.push_back(std::string("QUANT_MODE=") + std::to_string(quant_mode));
    opts.defines.push_back(std::string("BATCH_M=") + std::to_string(batch_m));
    opts.defines.push_back(std::string("K_SPLIT_ROW=") + (k_split_row ? "1" : "0"));
    opts.defines.push_back(std::string("WG_SGS=") + std::to_string(eff_sgs));
    opts.defines.push_back(std::string("X_ALIGNED4=") + (eff_a4 ? "1" : "0"));
    opts.push_constant_bytes = sizeof(agx_gemm_push_args);
    opts.is_hot_kernel       = true;
    opts.strict_zero_spill   = true;

    const agx_kernel * k = agx_compiler_compile_file(ctx->compiler, name, "mxfp8_q8_vecmat.comp", opts);
    if (k) {
        st->cache.emplace(name, k);
    }
    return k;
}

static const agx_kernel * get_or_compile_mma_gemm(agx_backend_context * ctx,
                                                  agx_weight_layout     layout,
                                                  uint32_t              tile_mode,
                                                  bool                  prec_f32,
                                                  bool                  x_aligned4   = false,
                                                  bool                  x_is_f16     = false,
                                                  bool                  k_aligned128 = false,
                                                  bool                  batch_z1     = true) {
    auto * st = get_gemm_state(ctx);
    if (!st || !ctx->compiler) {
        return nullptr;
    }
    const bool is_repacked = (layout == AGX_WEIGHT_NVFP4_REPACKED || layout == AGX_WEIGHT_Q8_0_REPACKED);
    const bool pf_64x64    = (tile_mode == 0u || tile_mode == 5u) && !prec_f32 && is_repacked;
    const bool m8_direct   = (tile_mode == 3u || tile_mode == 4u) && !prec_f32 && is_repacked;
    const bool eff_a4      = x_aligned4   && pf_64x64;
    const bool eff_f16     = x_is_f16     && pf_64x64;
    const bool eff_k128    = k_aligned128 && pf_64x64 && (layout == AGX_WEIGHT_NVFP4_REPACKED);
    const bool eff_bz1     = batch_z1     && (pf_64x64 || m8_direct);

    static const uint32_t pf_k_pairs = []() -> uint32_t {
        const char * v = std::getenv("GGML_AGX_GEMM_NO_K128");
        return (v && v[0] && v[0] != '0') ? 1u : 2u;
    }();
    static const int env_shx_pad = []() -> int {
        const char * v = std::getenv("GGML_AGX_GEMM_SHX_PAD");
        if (v && v[0]) {
            const int parsed = std::atoi(v);
            return parsed > 0 ? parsed : 0;
        }
        return -1;
    }();
    // TILE_MODE=5 (256T): SH_X_PAD=2 (8,704 B -> 3 WGs/core = 24 ActSGs, 0 bank conflicts).
    // TILE_MODE=0 (128T): SH_X_PAD=0 (8,192 B -> 4 WGs/core = 16 ActSGs, +33% over 8,704 B!).
    const uint32_t pf_shx_pad = (env_shx_pad >= 0) ? static_cast<uint32_t>(env_shx_pad)
                                                   : ((tile_mode == 0u) ? 0u : 2u);
    static const int env_unroll_kp = []() -> int {
        const char * v = std::getenv("GGML_AGX_GEMM_UNROLL_KP");
        if (v && v[0]) {
            return (v[0] != '0') ? 1 : 0;
        }
        return -1;
    }();
    // Wave 10 M4 Max silicon winner (task-8130: +1.03 TFLOPS Q8_0, +0.43 TFLOPS NVFP4, +8.8 t/s pp2048):
    //   - TILE_MODE=0 (128T, 64x32, PAD=0 = 8,192 B -> 4 WGs/core = 16 ActSGs): UNROLL_KP=1 overlaps
    //     s_kp=1 weight loads inside s_kp=0's 64 matmads at 0 occupancy loss!
    //   - TILE_MODE=5 (256T, 128x32, PAD=2): UNROLL_KP=0 keeps lower GPR footprint.
    const uint32_t pf_unroll_kp = (env_unroll_kp >= 0) ? static_cast<uint32_t>(env_unroll_kp)
                                                       : ((tile_mode == 0u && pf_k_pairs > 1u) ? 1u : 0u);

    char name[128];
    if (pf_64x64) {
        std::snprintf(name, sizeof(name), "agx_mma_gemm_w10_w%u_t%u_p0_a%u_kp%u_pad%u_f%u_k%u_z%u_u%u",
                      static_cast<uint32_t>(layout), tile_mode,
                      eff_a4 ? 1u : 0u, pf_k_pairs, pf_shx_pad,
                      eff_f16 ? 1u : 0u, eff_k128 ? 1u : 0u,
                      eff_bz1 ? 1u : 0u, pf_unroll_kp);
    } else if (m8_direct) {
        std::snprintf(name, sizeof(name), "agx_mma_gemm_w10_w%u_t%u_p0_z%u",
                      static_cast<uint32_t>(layout), tile_mode, eff_bz1 ? 1u : 0u);
    } else {
        std::snprintf(name, sizeof(name), "agx_mma_gemm_w%u_t%u_p%u_a0",
                      static_cast<uint32_t>(layout), tile_mode, prec_f32 ? 1u : 0u);
    }
    auto it = st->cache.find(name);
    if (it != st->cache.end()) {
        return it->second;
    }

    agx_compile_opts opts;
    opts.defines.push_back(std::string("WEIGHT_MODE=") + std::to_string(static_cast<uint32_t>(layout)));
    opts.defines.push_back(std::string("TILE_MODE=") + std::to_string(tile_mode));
    opts.defines.push_back(std::string("PREC_F32=") + (prec_f32 ? "1" : "0"));
    opts.defines.push_back(std::string("X_ALIGNED4=") + (eff_a4 ? "1" : "0"));
    if (pf_64x64) {
        opts.defines.push_back(std::string("K_PAIRS=") + std::to_string(pf_k_pairs));
        opts.defines.push_back(std::string("SH_X_PAD=") + std::to_string(pf_shx_pad));
        opts.defines.push_back(std::string("X_IS_F16=") + (eff_f16 ? "1" : "0"));
        opts.defines.push_back(std::string("K_ALIGNED128=") + (eff_k128 ? "1" : "0"));
        opts.defines.push_back(std::string("BATCH_Z1=") + (eff_bz1 ? "1" : "0"));
        opts.defines.push_back(std::string("UNROLL_KP=") + std::to_string(pf_unroll_kp));
    } else if (m8_direct) {
        opts.defines.push_back(std::string("BATCH_Z1=") + (eff_bz1 ? "1" : "0"));
    }
    opts.push_constant_bytes = sizeof(agx_gemm_push_args);
    opts.is_hot_kernel       = true;
    opts.strict_zero_spill   = true;

    const agx_kernel * k = agx_compiler_compile_file(ctx->compiler, name, "agx_mma_gemm.comp", opts);
    if (k) {
        st->cache.emplace(name, k);
    }
    return k;
}

bool agx_gemm_precompile_all(agx_backend_context * ctx) {
    if (!agx_gemm_init(ctx) || !ctx->compiler) {
        return false;
    }
    bool ok = true;
    // NVFP4 repacked vec-mat: B=1 (K_SGS=1 & K_SGS=4), B=4 (DSpark verify)
    ok &= (get_or_compile_nvfp4_vecmat(ctx, true, 1u, 1u, false) != nullptr);
    ok &= (get_or_compile_nvfp4_vecmat(ctx, true, 1u, 4u, false) != nullptr);
    ok &= (get_or_compile_nvfp4_vecmat(ctx, true, 4u, 4u, false) != nullptr);

    // Q8_0 repacked, BF16, F16, F32 B=1 vec-mat + narrow-N K-split (ctrl_down, router)
    for (uint32_t qm : {0u, 4u, 5u, 6u}) {
        ok &= (get_or_compile_q8_mxfp8_vecmat(ctx, qm, 1u, false) != nullptr);
    }
    for (uint32_t qm : {0u, 4u, 6u}) {
        ok &= (get_or_compile_q8_mxfp8_vecmat(ctx, qm, 1u, true) != nullptr);
    }
    // Q8_0 repacked B=4 (DSpark M=4 verify)
    ok &= (get_or_compile_q8_mxfp8_vecmat(ctx, 0u, 4u, false, 4u, true) != nullptr);
    ok &= (get_or_compile_q8_mxfp8_vecmat(ctx, 0u, 4u, false, 8u, true) != nullptr);

    // Wave 10 hot repacked MMA tiles:
    //   - t3/t4 (M=5..8 direct-register 32/30 ActSGs!)
    //   - t0 (default prefill 128T 64x32 PAD=0 UNROLL_KP=1: 12.58+ TFLOPS Q8_0, 11.86+ TFLOPS NVFP4!)
    //   - t1 (medium M=9..48)
    for (agx_weight_layout wl : {AGX_WEIGHT_NVFP4_REPACKED, AGX_WEIGHT_Q8_0_REPACKED}) {
        const bool is_nv = (wl == AGX_WEIGHT_NVFP4_REPACKED);
        ok &= (get_or_compile_mma_gemm(ctx, wl, 3u, false, false, false, false, true) != nullptr);
        ok &= (get_or_compile_mma_gemm(ctx, wl, 4u, false, false, false, false, true) != nullptr);
        ok &= (get_or_compile_mma_gemm(ctx, wl, 1u, false) != nullptr);
        ok &= (get_or_compile_mma_gemm(ctx, wl, 0u, false, true, false, is_nv, true) != nullptr);
        ok &= (get_or_compile_mma_gemm(ctx, wl, 0u, false, true, true,  is_nv, true) != nullptr);
    }
    ok &= (gemm_get_cvt_f16_kernel(ctx, get_gemm_state(ctx)) != nullptr);
    // FP32/BF16 full-range 8x8 MMA variants (incl. narrow-N TILE_MODE=2 for 4608x64 ctrl_down!)
    ok &= (get_or_compile_mma_gemm(ctx, AGX_WEIGHT_F32,  0u, true) != nullptr);
    ok &= (get_or_compile_mma_gemm(ctx, AGX_WEIGHT_F32,  2u, true) != nullptr);
    ok &= (get_or_compile_mma_gemm(ctx, AGX_WEIGHT_BF16, 0u, true) != nullptr);
    ok &= (get_or_compile_mma_gemm(ctx, AGX_WEIGHT_BF16, 2u, true) != nullptr);
    ok &= (get_or_compile_mma_gemm(ctx, AGX_WEIGHT_F32,  1u, true) != nullptr);
    ok &= (get_or_compile_mma_gemm(ctx, AGX_WEIGHT_BF16, 1u, true) != nullptr);
    return ok;
}

bool agx_gemm_infer_weight_layout(const struct ggml_tensor * w, agx_weight_layout * out_layout) {
    if (!w || !out_layout) {
        return false;
    }
    const bool repacked = agx_tensor_is_repacked(w) && agx_repack_supports_tensor(w);
    switch (w->type) {
        case GGML_TYPE_NVFP4:
            *out_layout = repacked ? AGX_WEIGHT_NVFP4_REPACKED : AGX_WEIGHT_NVFP4_RAW;
            return true;
        case GGML_TYPE_Q8_0:
            *out_layout = repacked ? AGX_WEIGHT_Q8_0_REPACKED : AGX_WEIGHT_Q8_0_RAW;
            return true;
        case GGML_TYPE_F8_E4M3:
            *out_layout = AGX_WEIGHT_F8_E4M3;
            return true;
        case GGML_TYPE_BF16:
            *out_layout = AGX_WEIGHT_BF16;
            return true;
        case GGML_TYPE_F16:
            *out_layout = AGX_WEIGHT_F16;
            return true;
        case GGML_TYPE_F32:
            *out_layout = AGX_WEIGHT_F32;
            return true;
        default:
            return false;
    }
}

bool agx_gemm_dispatch(agx_backend_context * ctx,
                       agx_cdm_stream *      stream,
                       const agx_gemm_desc & desc) {
    if (!ctx || !stream || desc.w_va == 0 || desc.x_va == 0 || desc.y_va == 0 ||
        desc.M == 0 || desc.N == 0 || desc.K == 0) {
        return false;
    }
    if (!agx_gemm_init(ctx)) {
        return false;
    }

    const uint32_t batch_z  = std::max(1u, desc.batch_count);
    const uint32_t x_stride = desc.x_stride_elems ? desc.x_stride_elems : desc.K;
    const uint32_t y_stride = desc.y_stride_elems ? desc.y_stride_elems : desc.N;

    agx_gemm_push_args pc{};
    pc.w_va          = desc.w_va;
    pc.x_va          = desc.x_va;
    pc.y_va          = desc.y_va;
    pc.M             = desc.M;
    pc.N             = desc.N;
    pc.K             = desc.K;
    pc.x_stride      = x_stride;
    pc.y_stride      = y_stride;
    pc.w_slice_bytes = desc.w_slice_bytes;
    pc.x_slice_elems = desc.x_slice_elems;
    pc.y_slice_elems = desc.y_slice_elems;
    pc.w_bcast_div   = std::max(1u, desc.w_bcast_div);

    // Wave 10 M=5..8 Zero-sh_w Direct-Register RT=1 (8M) 8x8 MMA fast path:
    //   - Replaces sh_w-staged 12-16 ActSG kernels with 48/52 GPR32, 2.1 KB SRAM -> 30-32 ActSGs/core!
    //   - TILE_MODE=3 (64N x 8M, 128T): N*batch_z >= 4096 (W_attn_in 8984 -> 141 WGs, LM head -> 3,141 WGs)
    //   - TILE_MODE=4 (32N x 8M,  64T): N*batch_z <  4096 (Wo 2304 -> 72 WGs saturating all 40 cores!)
    static const bool no_m8_mma = []() {
        const char * v = std::getenv("GGML_AGX_GEMM_NO_M8_MMA");
        return v && v[0] && v[0] != '0';
    }();
    static const uint32_t m8_mma_min_m = []() -> uint32_t {
        const char * v = std::getenv("GGML_AGX_GEMM_M8_MMA_MIN_M");
        if (v && v[0]) {
            const int parsed = std::atoi(v);
            return parsed > 0 ? static_cast<uint32_t>(parsed) : 9u;
        }
        return 5u;
    }();
    const bool use_m8_mma = !no_m8_mma && !desc.force_vecmat && !desc.prec_f32 &&
                            (desc.layout == AGX_WEIGHT_NVFP4_REPACKED || desc.layout == AGX_WEIGHT_Q8_0_REPACKED) &&
                            (desc.M >= m8_mma_min_m && desc.M <= 8u) &&
                            (desc.N > 256u) &&
                            (desc.K >= 256u && ((desc.K & 31u) == 0u)) &&
                            (desc.layout == AGX_WEIGHT_Q8_0_REPACKED || (desc.K & 63u) == 0u) &&
                            ((x_stride & 3u) == 0u);
    if (use_m8_mma) {
        const uint32_t tile_mode = (desc.N * batch_z >= 4096u) ? 3u : 4u;
        const uint32_t wg_n      = (tile_mode == 3u) ? 64u : 32u;
        const agx_kernel * k     = get_or_compile_mma_gemm(
            ctx, desc.layout, tile_mode, false, false, false, false, batch_z == 1u);
        if (!k) {
            return false;
        }
        const uint32_t gx = (desc.N + wg_n - 1u) / wg_n;
        return agx_cdm_dispatch(stream, k, gx, 1u, batch_z, &pc, sizeof(pc), desc.post_barrier);
    }

    // Crossover: M <= 8 -> predicated K-coalesced vec-mat; M >= 9 -> 8x8 MMA GEMM.
    const bool use_mma = desc.force_mma || (!desc.force_vecmat && desc.M >= 9u);

    if (use_mma) {
        // Select tile geometry to saturate all 40 M4 Max cores:
        //   - TILE_MODE=2 (16x16, 2x2): narrow N <= 128 (e.g. 4608x64 ctrl_down -> 128 WGs!)
        //   - TILE_MODE=1 (32x16, 4x1): medium M <= 48 (72-216 WGs on N=2304)
        //   - TILE_MODE=0 (64x32, 128T = 4 SGs, K_PAIRS=2, PAD=0 = 8,192 B, UNROLL_KP=1):
        //     Wave 10 default prefill winner (task-8130: 4 WGs/core = 16 ActSGs, 2x WG count vs t5,
        //     overlapped s_kp=1 weight prefetch -> +1.03 TFLOPS Q8_0, +8.8 t/s pp2048!).
        //   - TILE_MODE=5 (128x32, 256T = 8 SGs): opt-in A/B via GGML_AGX_GEMM_USE_T5=1.
        uint32_t tile_mode       = (desc.N <= 128u && batch_z <= 2u) ? 2u
                                 : (desc.M <= 48u)                   ? 1u
                                                                     : 0u;
        const bool is_repack_f16 = !desc.prec_f32 &&
                                   (desc.layout == AGX_WEIGHT_NVFP4_REPACKED ||
                                    desc.layout == AGX_WEIGHT_Q8_0_REPACKED);
        static const bool no_t5  = (std::getenv("GGML_AGX_GEMM_NO_T5") != nullptr);
        static const bool use_t5 = []() {
            const char * v = std::getenv("GGML_AGX_GEMM_USE_T5");
            return v && v[0] && v[0] != '0';
        }();
        static const uint32_t t5_min_wgs = []() -> uint32_t {
            const char * v = std::getenv("GGML_AGX_GEMM_T5_MIN_WGS");
            if (v && v[0]) {
                const int parsed = std::atoi(v);
                return parsed > 0 ? static_cast<uint32_t>(parsed) : 0xFFFFFFFFu;
            }
            return 256u;
        }();
        if (tile_mode == 0u && is_repack_f16 && use_t5 && !no_t5) {
            const uint32_t wgs_128 = ((desc.N + 127u) / 128u) * ((desc.M + 31u) / 32u) * batch_z;
            if (wgs_128 >= t5_min_wgs) {
                tile_mode = 5u;
            }
        }
        const bool pf_64x64      = (tile_mode == 0u || tile_mode == 5u) && is_repack_f16;
        const uint32_t wg_n      = (tile_mode == 5u) ? 128u
                                 : (tile_mode == 2u) ? 16u
                                 : (tile_mode == 1u) ? 32u
                                                     : 64u;
        const uint32_t wg_m      = (tile_mode == 0u || tile_mode == 5u) ? 32u : 16u;

        // Wave 10 B_IS_F16: pre-convert X F32->F16 into SLC-pinned scratch for N >= 4096, M >= 64
        // (covers W_attn_in 8984 & LM head 201024 while avoiding k_cvt_f16 + STRONG_USC overhead
        // on 48x Wo 2304 & single 2304x2304 matmuls where task-8130 proved N>=4096 wins!).
        static const uint32_t bf16_min_n = []() -> uint32_t {
            const char * v = std::getenv("GGML_AGX_GEMM_BF16_MIN_N");
            if (v && v[0]) {
                const int parsed = std::atoi(v);
                return parsed > 0 ? static_cast<uint32_t>(parsed) : 0xFFFFFFFFu;
            }
            return 4096u;
        }();
        const bool mma_a4   = pf_64x64 && ((desc.K & 3u) == 0u) && ((x_stride & 3u) == 0u);
        const bool mma_k128 = pf_64x64 && ((desc.K & 127u) == 0u);
        bool       use_f16  = false;
        auto *     st       = get_gemm_state(ctx);
        if (mma_a4 && batch_z == 1u && desc.M >= 64u && desc.N >= bf16_min_n && x_stride == desc.K) {
            const size_t x_elems    = static_cast<size_t>(desc.M) * desc.K;
            const size_t x_f16_B    = x_elems * sizeof(uint16_t);
            const agx_kernel * kcvt = gemm_get_cvt_f16_kernel(ctx, st);
            if (kcvt && gemm_ensure_x_f16_bo(ctx, st, x_f16_B)) {
                if (!agx_cdm_is_empty(stream)) {
                    agx_cdm_stats cst{};
                    agx_cdm_get_stats(stream, &cst);
                    const uint64_t gen  = agx_cdm_stream_record_gen(stream);
                    const uint64_t bars = cst.strong_usc_barriers + cst.memory_host_barriers;
                    if (st->last_stream == stream && st->last_stream_gen == gen && bars == st->last_strong_bars) {
                        agx_cdm_barrier_usc(stream);
                    }
                }
                agx_push_cvt_f16 pcvt{};
                pcvt.src_va = desc.x_va;
                pcvt.dst_va = st->x_f16_bo->gpu_va;
                pcvt.n_vec4 = static_cast<uint32_t>(x_elems >> 2u);
                const uint32_t cvt_gx = (pcvt.n_vec4 + 127u) / 128u;
                if (agx_cdm_dispatch(stream, kcvt, cvt_gx, 1u, 1u,
                                     &pcvt, sizeof(pcvt), AGX_BARRIER_STRONG_USC)) {
                    pc.x_va = st->x_f16_bo->gpu_va;
                    use_f16 = true;
                }
            }
        }

        const agx_kernel * k = get_or_compile_mma_gemm(
            ctx, desc.layout, tile_mode, desc.prec_f32, mma_a4, use_f16, mma_k128, batch_z == 1u);
        if (!k) {
            return false;
        }
        const uint32_t gx = (desc.N + wg_n - 1u) / wg_n;
        const uint32_t gy = (desc.M + wg_m - 1u) / wg_m;
        const bool ok = agx_cdm_dispatch(stream, k, gx, gy, batch_z, &pc, sizeof(pc), desc.post_barrier);
        if (ok && use_f16) {
            agx_cdm_stats cst{};
            agx_cdm_get_stats(stream, &cst);
            st->last_stream      = stream;
            st->last_stream_gen  = agx_cdm_stream_record_gen(stream);
            st->last_strong_bars = cst.strong_usc_barriers + cst.memory_host_barriers;
        }
        return ok;
    }

    // Predicated B=1..8 vec-mat path (quantized specializations {1, 2, 4, 5, 8} to bound cache size)
    // For arbitrary M > 8 with force_vecmat, loop in chunks of 8.
    const uint32_t m_cap  = std::min(desc.M, 8u);
    const uint32_t m_spec = (m_cap <= 1u) ? 1u
                          : (m_cap <= 2u) ? 2u
                          : (m_cap <= 4u) ? 4u
                          : (m_cap == 5u) ? 5u
                                          : 8u;

    if (desc.layout == AGX_WEIGHT_NVFP4_REPACKED || desc.layout == AGX_WEIGHT_NVFP4_RAW) {
        const bool repacked       = (desc.layout == AGX_WEIGHT_NVFP4_REPACKED);
        const bool fast_shx_batch = repacked && !desc.prec_f32 && (m_spec > 1u) && ((x_stride & 3u) == 0u);
        // fast_shx_batch (M=2..8 repacked): single-dispatch 16 rows/WG (128T, 4-way K-split, 1x W read!)
        // B=1 K_SGS=4: 8 rows/WG (144 WGs on N=1152, 288 on N=2304).
        // B=1 K_SGS=1: repacked fuses s=0&s=1 -> 128 rows/WG (512 WGs on N=65536, 482.30 GB/s!).
        const uint32_t k_sgs = fast_shx_batch
            ? 4u
            : ((repacked && !desc.prec_f32 && m_spec > 1u)
                ? 1u
                : (((desc.N * batch_z) < 10240u && desc.K >= 256u) ? 4u : 1u));

        const agx_kernel * k = get_or_compile_nvfp4_vecmat(ctx, repacked, m_spec, k_sgs, desc.prec_f32);
        if (!k) {
            return false;
        }
        const bool fused_32row = repacked && (k_sgs == 1u) && (m_spec == 1u);
        const uint32_t gx = fast_shx_batch
            ? ((desc.N + 15u) / 16u)
            : ((k_sgs == 4u)
                ? ((desc.N + 7u) / 8u)
                : (fused_32row ? ((desc.N + 127u) / 128u) : ((desc.N + 63u) / 64u)));

        if (desc.M <= 8u) {
            return agx_cdm_dispatch(stream, k, gx, 1u, batch_z, &pc, sizeof(pc), desc.post_barrier);
        }
        for (uint32_t m0 = 0; m0 < desc.M; m0 += 8u) {
            const uint32_t m_rem = std::min(8u, desc.M - m0);
            pc.x_va = desc.x_va + static_cast<uint64_t>(m0) * x_stride * sizeof(float);
            pc.y_va = desc.y_va + static_cast<uint64_t>(m0) * y_stride * sizeof(float);
            pc.M    = m_rem;
            const agx_barrier_mode bm = (m0 + m_rem >= desc.M) ? desc.post_barrier : AGX_BARRIER_NONE;
            if (!agx_cdm_dispatch(stream, k, gx, 1u, batch_z, &pc, sizeof(pc), bm)) {
                return false;
            }
        }
        return true;
    }

    // Q8_0 / MXFP8 / F8_E4M3 / BF16 / F16 / F32 vec-mat
    uint32_t quant_mode  = 6u;
    uint32_t rows_per_wg = 4u;
    bool     can_ksplit  = false;
    uint32_t ksplit_min_k = 512u;
    switch (desc.layout) {
        case AGX_WEIGHT_Q8_0_REPACKED:
            quant_mode   = 0u;
            rows_per_wg  = 8u;
            can_ksplit   = ((desc.K & 3u) == 0u && (x_stride & 3u) == 0u);
            ksplit_min_k = 1024u;
            break;
        case AGX_WEIGHT_Q8_0_RAW:       quant_mode = 1u; rows_per_wg = 4u; break;
        case AGX_WEIGHT_MXFP8_REPACKED: quant_mode = 2u; rows_per_wg = 4u; break;
        case AGX_WEIGHT_F8_E4M3:        quant_mode = 3u; rows_per_wg = 4u; break;
        case AGX_WEIGHT_BF16:           quant_mode = 4u; rows_per_wg = 4u; can_ksplit = true; break;
        case AGX_WEIGHT_F16:            quant_mode = 5u; rows_per_wg = 4u; can_ksplit = true; break;
        case AGX_WEIGHT_F32:            quant_mode = 6u; rows_per_wg = 4u; can_ksplit = true; break;
        default: return false;
    }

    // Use K_SPLIT_ROW (all 4 sgs split K over 1 row -> 4x-8x core occupancy!) on narrow-N, large-K
    const bool k_split_row = can_ksplit && (desc.N * batch_z <= 256u) && (desc.K >= ksplit_min_k);
    // For Q8_0 M=2..8 sh_x staged path:
    //   - N*batch_z <= 4096 (Wk/Wv 1024, Wo 2304, QKV 3072): WG_SGS=4 (128T, 8 rows/WG -> 128-384 WGs!)
    //   - N*batch_z >  4096 (W_attn_in 8984, LM head 201024): WG_SGS=8 (256T, 16 rows/WG -> 2x less X load!)
    const uint32_t wg_sgs = (quant_mode == 0u && !k_split_row && m_spec > 1u)
                                ? ((desc.N * batch_z > 4096u) ? 8u : 4u)
                                : 4u;
    const bool x_aligned4 = (quant_mode == 0u && !k_split_row && m_spec > 1u) &&
                            ((desc.K & 3u) == 0u) && ((x_stride & 3u) == 0u);
    if (quant_mode == 0u && !k_split_row) {
        rows_per_wg = wg_sgs * 2u;
    }
    const agx_kernel * k = get_or_compile_q8_mxfp8_vecmat(ctx, quant_mode, m_spec, k_split_row, wg_sgs, x_aligned4);
    if (!k) {
        return false;
    }
    const uint32_t gx = k_split_row ? desc.N : ((desc.N + rows_per_wg - 1u) / rows_per_wg);

    if (desc.M <= 8u) {
        return agx_cdm_dispatch(stream, k, gx, 1u, batch_z, &pc, sizeof(pc), desc.post_barrier);
    }
    for (uint32_t m0 = 0; m0 < desc.M; m0 += 8u) {
        const uint32_t m_rem = std::min(8u, desc.M - m0);
        pc.x_va = desc.x_va + static_cast<uint64_t>(m0) * x_stride * sizeof(float);
        pc.y_va = desc.y_va + static_cast<uint64_t>(m0) * y_stride * sizeof(float);
        pc.M    = m_rem;
        const agx_barrier_mode bm = (m0 + m_rem >= desc.M) ? desc.post_barrier : AGX_BARRIER_NONE;
        if (!agx_cdm_dispatch(stream, k, gx, 1u, batch_z, &pc, sizeof(pc), bm)) {
            return false;
        }
    }
    return true;
}

bool agx_gemm_dispatch_raw_q8_0(agx_backend_context * ctx,
                                agx_cdm_stream *      stream,
                                uint64_t              w_va,
                                uint64_t              x_va,
                                uint64_t              y_va,
                                uint32_t              M,
                                uint32_t              N,
                                uint32_t              K,
                                agx_barrier_mode      post_barrier,
                                uint32_t              x_stride_elems,
                                uint32_t              y_stride_elems) {
    agx_gemm_desc desc{};
    desc.w_va           = w_va;
    desc.x_va           = x_va;
    desc.y_va           = y_va;
    desc.M              = M;
    desc.N              = N;
    desc.K              = K;
    desc.x_stride_elems = x_stride_elems ? x_stride_elems : K;
    desc.y_stride_elems = y_stride_elems ? y_stride_elems : N;
    desc.batch_count    = 1u;
    desc.w_bcast_div    = 1u;
    desc.layout         = AGX_WEIGHT_Q8_0_REPACKED;
    desc.post_barrier   = post_barrier;
    return agx_gemm_dispatch(ctx, stream, desc);
}

bool agx_gemm_supports_mul_mat(const struct ggml_tensor * dst) {
    if (!dst || dst->op != GGML_OP_MUL_MAT || !dst->src[0] || !dst->src[1]) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if ((src1->nb[1] % sizeof(float)) != 0 ||
        (src1->nb[2] % sizeof(float)) != 0 ||
        (src1->nb[3] % sizeof(float)) != 0) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }
    // Support full 2D, 3D, and 4D broadcast matching ggml MUL_MAT semantics (ne2/ne3 divisible)
    if (src0->ne[2] <= 0 || src0->ne[3] <= 0 ||
        src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }

    agx_weight_layout layout;
    return agx_gemm_infer_weight_layout(src0, &layout);
}

bool agx_gemm_mul_mat(agx_backend_context *      ctx,
                      agx_cdm_stream *           stream,
                      const struct ggml_tensor * dst,
                      agx_barrier_mode           post_barrier) {
    if (!agx_gemm_supports_mul_mat(dst)) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    agx_weight_layout layout;
    if (!agx_gemm_infer_weight_layout(src0, &layout)) {
        return false;
    }

    const uint64_t w_va = agx_tensor_gpu_va(src0);
    const uint64_t x_va = agx_tensor_gpu_va(src1);
    const uint64_t y_va = agx_tensor_gpu_va(dst);
    if (w_va == 0 || x_va == 0 || y_va == 0) {
        return false;
    }

    const int64_t K    = src0->ne[0];
    const int64_t N    = src0->ne[1];
    const int64_t M    = src1->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t r2   = ne12 / ne02;
    const int64_t r3   = ne13 / ne03;

    uint64_t w_nb2 = 0;
    uint64_t w_nb3 = 0;
    if (layout == AGX_WEIGHT_NVFP4_REPACKED || layout == AGX_WEIGHT_Q8_0_REPACKED) {
        w_nb2 = static_cast<uint64_t>(agx_repack_slice_stride(src0->type, K, N));
        w_nb3 = w_nb2 * static_cast<uint64_t>(ne02);
    } else {
        w_nb2 = static_cast<uint64_t>(src0->nb[2]);
        w_nb3 = static_cast<uint64_t>(src0->nb[3]);
    }

    const enum ggml_prec p_src0 = ggml_prec_get_src(dst, 0);
    const enum ggml_prec p_src1 = ggml_prec_get_src(dst, 1);

    agx_gemm_desc desc{};
    desc.M              = static_cast<uint32_t>(M);
    desc.N              = static_cast<uint32_t>(N);
    desc.K              = static_cast<uint32_t>(K);
    desc.x_stride_elems = static_cast<uint32_t>(src1->nb[1] / sizeof(float));
    desc.y_stride_elems = static_cast<uint32_t>(dst->nb[1] / sizeof(float));
    desc.w_slice_bytes  = static_cast<uint32_t>(w_nb2);
    desc.x_slice_elems  = static_cast<uint32_t>(src1->nb[2] / sizeof(float));
    desc.y_slice_elems  = static_cast<uint32_t>(dst->nb[2] / sizeof(float));
    desc.batch_count    = static_cast<uint32_t>(ne12);
    desc.w_bcast_div    = static_cast<uint32_t>(r2);
    desc.layout         = layout;
    // Honor GGML precision policy: never downcast src0/src1 to FP16 when F32 or BF16
    // range/precision is requested (prevents FP16 >65504 overflow / NaN on BF16/F32 inputs!).
    desc.prec_f32       = (p_src1 == GGML_PREC_F32 || p_src1 == GGML_PREC_BF16 ||
                           p_src0 == GGML_PREC_F32 || p_src0 == GGML_PREC_BF16 ||
                           src0->type == GGML_TYPE_F32);

    // Fast path for 2D/3D (ne13 == 1): single CDM dispatch
    if (ne13 == 1) {
        desc.w_va         = w_va;
        desc.x_va         = x_va;
        desc.y_va         = y_va;
        desc.post_barrier = post_barrier;
        return agx_gemm_dispatch(ctx, stream, desc);
    }

    // Exact 4D broadcast & arbitrary src1 permutation (e.g. per=[0,2,1,3]) across ne13:
    // Independent i3 slices overlap with AGX_BARRIER_NONE until final slice's post_barrier.
    for (int64_t i3 = 0; i3 < ne13; ++i3) {
        const int64_t w_i3 = i3 / r3;
        desc.w_va         = w_va + static_cast<uint64_t>(w_i3) * w_nb3;
        desc.x_va         = x_va + static_cast<uint64_t>(i3)   * static_cast<uint64_t>(src1->nb[3]);
        desc.y_va         = y_va + static_cast<uint64_t>(i3)   * static_cast<uint64_t>(dst->nb[3]);
        desc.post_barrier = (i3 + 1 == ne13) ? post_barrier : AGX_BARRIER_NONE;
        if (!agx_gemm_dispatch(ctx, stream, desc)) {
            return false;
        }
    }
    return true;
}
