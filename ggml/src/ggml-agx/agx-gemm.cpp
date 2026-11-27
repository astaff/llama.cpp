#include "agx-gemm.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>

// ============================================================================
// Pillar 2: Kernel Cache & Dispatcher (`nvfp4_vecmat`, `mxfp8_q8_vecmat`,
//           and 8x8 `simdgroup_matrix` `agx_mma_gemm`)
// ============================================================================

struct agx_gemm_state {
    std::unordered_map<std::string, const agx_kernel *> cache;
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
    delete static_cast<agx_gemm_state *>(ctx->gemm_state);
    ctx->gemm_state = nullptr;
}

static const agx_kernel * get_or_compile_nvfp4_vecmat(agx_backend_context * ctx,
                                                      bool                  repacked,
                                                      uint32_t              batch_m,
                                                      uint32_t              k_sgs) {
    auto * st = get_gemm_state(ctx);
    if (!st || !ctx->compiler) {
        return nullptr;
    }
    char name[96];
    std::snprintf(name, sizeof(name), "nvfp4_vecmat_r%u_m%u_k%u",
                  repacked ? 1u : 0u, batch_m, k_sgs);
    auto it = st->cache.find(name);
    if (it != st->cache.end()) {
        return it->second;
    }

    agx_compile_opts opts;
    opts.defines.push_back(std::string("REPACKED=") + (repacked ? "1" : "0"));
    opts.defines.push_back(std::string("BATCH_M=") + std::to_string(batch_m));
    opts.defines.push_back(std::string("K_SGS=") + std::to_string(k_sgs));
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
                                                         uint32_t              batch_m) {
    auto * st = get_gemm_state(ctx);
    if (!st || !ctx->compiler) {
        return nullptr;
    }
    char name[96];
    std::snprintf(name, sizeof(name), "mxfp8_q8_vecmat_q%u_m%u", quant_mode, batch_m);
    auto it = st->cache.find(name);
    if (it != st->cache.end()) {
        return it->second;
    }

    agx_compile_opts opts;
    opts.defines.push_back(std::string("QUANT_MODE=") + std::to_string(quant_mode));
    opts.defines.push_back(std::string("BATCH_M=") + std::to_string(batch_m));
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
                                                  bool                  prec_f32) {
    auto * st = get_gemm_state(ctx);
    if (!st || !ctx->compiler) {
        return nullptr;
    }
    char name[96];
    std::snprintf(name, sizeof(name), "agx_mma_gemm_w%u_t%u_p%u",
                  static_cast<uint32_t>(layout), tile_mode, prec_f32 ? 1u : 0u);
    auto it = st->cache.find(name);
    if (it != st->cache.end()) {
        return it->second;
    }

    agx_compile_opts opts;
    opts.defines.push_back(std::string("WEIGHT_MODE=") + std::to_string(static_cast<uint32_t>(layout)));
    opts.defines.push_back(std::string("TILE_MODE=") + std::to_string(tile_mode));
    opts.defines.push_back(std::string("PREC_F32=") + (prec_f32 ? "1" : "0"));
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
    // NVFP4 repacked vec-mat: B=1 (K_SGS=1 & K_SGS=4), B=5 (DSpark verify)
    ok &= (get_or_compile_nvfp4_vecmat(ctx, true, 1u, 1u) != nullptr);
    ok &= (get_or_compile_nvfp4_vecmat(ctx, true, 1u, 4u) != nullptr);
    ok &= (get_or_compile_nvfp4_vecmat(ctx, true, 5u, 1u) != nullptr);

    // Q8_0 repacked, MXFP8 repacked, F8_E4M3, BF16, F16, F32 B=1 vec-mat
    for (uint32_t qm : {0u, 2u, 3u, 4u, 5u, 6u}) {
        ok &= (get_or_compile_q8_mxfp8_vecmat(ctx, qm, 1u) != nullptr);
    }
    // Q8_0 repacked B=5 (DSpark verify)
    ok &= (get_or_compile_q8_mxfp8_vecmat(ctx, 0u, 5u) != nullptr);

    // 8x8 simdgroup_matrix MMA GEMM (both large prefill tile=0 & medium/DSpark tile=1)
    for (agx_weight_layout wl : {
             AGX_WEIGHT_NVFP4_REPACKED,
             AGX_WEIGHT_Q8_0_REPACKED,
             AGX_WEIGHT_MXFP8_REPACKED,
             AGX_WEIGHT_BF16,
             AGX_WEIGHT_F16,
             AGX_WEIGHT_F32}) {
        ok &= (get_or_compile_mma_gemm(ctx, wl, 0u, false) != nullptr);
        ok &= (get_or_compile_mma_gemm(ctx, wl, 1u, false) != nullptr);
    }
    // FP32 full-range 8x8 MMA variants (BF16/F32/Q8_0/NVFP4)
    ok &= (get_or_compile_mma_gemm(ctx, AGX_WEIGHT_F32, 0u, true) != nullptr);
    ok &= (get_or_compile_mma_gemm(ctx, AGX_WEIGHT_BF16, 0u, true) != nullptr);
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

    // Crossover: M <= 5 -> predicated K-coalesced vec-mat; M >= 6 -> 8x8 simdgroup_matrix MMA
    const bool use_mma = desc.force_mma || (!desc.force_vecmat && desc.M >= 6u);

    if (use_mma) {
        const uint32_t tile_mode = (desc.M <= 24u) ? 1u : 0u;
        const uint32_t wg_n      = 64u;
        const uint32_t wg_m      = (tile_mode == 1u) ? 16u : (desc.prec_f32 ? 32u : 64u);

        const agx_kernel * k = get_or_compile_mma_gemm(ctx, desc.layout, tile_mode, desc.prec_f32);
        if (!k) {
            return false;
        }
        const uint32_t gx = (desc.N + wg_n - 1u) / wg_n;
        const uint32_t gy = (desc.M + wg_m - 1u) / wg_m;
        return agx_cdm_dispatch(stream, k, gx, gy, batch_z, &pc, sizeof(pc), desc.post_barrier);
    }

    // Predicated B=1..5 vec-mat path
    // For arbitrary M > 5 with force_vecmat, loop in chunks of 5
    const uint32_t m_spec = std::min(desc.M, 5u);

    if (desc.layout == AGX_WEIGHT_NVFP4_REPACKED || desc.layout == AGX_WEIGHT_NVFP4_RAW) {
        const bool repacked = (desc.layout == AGX_WEIGHT_NVFP4_REPACKED);
        // Use K_SGS=4 (4 simdgroups splitting K over 16 rows) when N*batch_z is medium
        // so all 40 GPU cores stay saturated; use K_SGS=1 (zero barrier; fused 32 rows/sg
        // = 128 rows/WG for repacked B=1!) for huge N!
        const uint32_t n_halves = (desc.N + 15u) / 16u;
        const uint32_t k_sgs    = ((n_halves * batch_z) < 640u && desc.K >= 256u) ? 4u : 1u;

        const agx_kernel * k = get_or_compile_nvfp4_vecmat(ctx, repacked, m_spec, k_sgs);
        if (!k) {
            return false;
        }
        const bool fused_32row = repacked && (k_sgs == 1u) && (m_spec == 1u);
        const uint32_t gx = fused_32row
            ? ((desc.N + 127u) / 128u)
            : ((k_sgs == 1u) ? ((n_halves + 3u) / 4u) : n_halves);

        if (desc.M <= 5u) {
            return agx_cdm_dispatch(stream, k, gx, 1u, batch_z, &pc, sizeof(pc), desc.post_barrier);
        }
        for (uint32_t m0 = 0; m0 < desc.M; m0 += 5u) {
            const uint32_t m_rem = std::min(5u, desc.M - m0);
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
    uint32_t quant_mode = 6u;
    uint32_t rows_per_wg = 4u; // 4 simdgroups x 1 row/sg
    switch (desc.layout) {
        case AGX_WEIGHT_Q8_0_REPACKED:  quant_mode = 0u; rows_per_wg = 8u; break; // 4 sg x 2 rows/sg
        case AGX_WEIGHT_Q8_0_RAW:       quant_mode = 1u; rows_per_wg = 4u; break;
        case AGX_WEIGHT_MXFP8_REPACKED: quant_mode = 2u; rows_per_wg = 4u; break;
        case AGX_WEIGHT_F8_E4M3:        quant_mode = 3u; rows_per_wg = 4u; break;
        case AGX_WEIGHT_BF16:           quant_mode = 4u; rows_per_wg = 4u; break;
        case AGX_WEIGHT_F16:            quant_mode = 5u; rows_per_wg = 4u; break;
        case AGX_WEIGHT_F32:            quant_mode = 6u; rows_per_wg = 4u; break;
        default: return false;
    }

    const agx_kernel * k = get_or_compile_q8_mxfp8_vecmat(ctx, quant_mode, m_spec);
    if (!k) {
        return false;
    }
    const uint32_t gx = (desc.N + rows_per_wg - 1u) / rows_per_wg;

    if (desc.M <= 5u) {
        return agx_cdm_dispatch(stream, k, gx, 1u, batch_z, &pc, sizeof(pc), desc.post_barrier);
    }
    for (uint32_t m0 = 0; m0 < desc.M; m0 += 5u) {
        const uint32_t m_rem = std::min(5u, desc.M - m0);
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
