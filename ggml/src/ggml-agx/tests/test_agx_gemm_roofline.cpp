#include "../agx-common.h"
#include "../agx-drm.h"
#include "../agx-cdm.h"
#include "../agx-compiler.h"
#include "../agx-repack.h"
#include "../agx-gemm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"

// Pillar 2 M4 Max Hardware Roofline & Numerical Parity Benchmark:
// Measures sustained GB/s and verifies max_abs / cosine parity vs CPU reference on
// Inkling 2 Mini exact matrix shapes:
//   1. NVFP4 repacked (AGX_NVFP4_32x16): (K=2304, N=2304, M=1 and M=5 DSpark) & (K=1152, N=2304, M=1)
//   2. Q8_0  repacked (AGX_Q8_0_8BLK):   (K=2304, N=201024, M=1 LM head) & (K=2304, N=3072, M=1 Q/gate)

namespace {

static double median_u64(std::vector<uint64_t> v) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    size_t m = v.size() / 2;
    return (v.size() & 1) ? static_cast<double>(v[m])
                          : 0.5 * (static_cast<double>(v[m - 1]) + static_cast<double>(v[m]));
}

struct Parity {
    int    n_nan   = 0;
    int    n_inf   = 0;
    double max_abs = 0.0;
    double cosine  = 0.0;
};

static Parity check_parity(const float * ref, const float * got, size_t n) {
    Parity p;
    double dot = 0.0, nr = 0.0, ng = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float g = got[i];
        float r = ref[i];
        if (std::isnan(g)) { p.n_nan++; continue; }
        if (std::isinf(g)) { p.n_inf++; continue; }
        double d = std::fabs(static_cast<double>(g) - static_cast<double>(r));
        if (d > p.max_abs) p.max_abs = d;
        dot += static_cast<double>(r) * g;
        nr  += static_cast<double>(r) * r;
        ng  += static_cast<double>(g) * g;
    }
    p.cosine = (nr > 0.0 && ng > 0.0) ? (dot / (std::sqrt(nr) * std::sqrt(ng))) : 1.0;
    return p;
}

static int bench_nvfp4(agx_backend_context * ctx, uint32_t K, uint32_t N, uint32_t M, uint32_t reps_batch) {
    agx_repack_tensor_info info;
    agx_repack_describe(GGML_TYPE_NVFP4, K, N, 1, 1, &info);

    const size_t orig_bytes = static_cast<size_t>(N) * (K / QK_NVFP4) * sizeof(block_nvfp4);
    std::vector<uint8_t> raw_w(orig_bytes);
    std::mt19937 rng(2026 + K + N + M);
    auto * blks = reinterpret_cast<block_nvfp4 *>(raw_w.data());
    const size_t n_blks = orig_bytes / sizeof(block_nvfp4);
    for (size_t i = 0; i < n_blks; ++i) {
        for (int s = 0; s < 4; ++s) {
            // Reasonable UE4M3 scales in [0x20..0x48] (~0.0625..2.0)
            blks[i].d[s] = static_cast<uint8_t>(0x28 + (rng() % 24));
        }
        for (int j = 0; j < 32; ++j) {
            blks[i].qs[j] = static_cast<uint8_t>(rng() & 0xFFu);
        }
    }

    agx_bo * bo_w = agx_bo_alloc(ctx->drm, info.total_bytes, AGX_BO_SLC_STREAM | AGX_BO_READONLY, "nvfp4_w");
    agx_bo * bo_x = agx_bo_alloc(ctx->drm, static_cast<size_t>(M) * K * sizeof(float), AGX_BO_SLC_PIN, "nvfp4_x");
    agx_bo * bo_y = agx_bo_alloc(ctx->drm, static_cast<size_t>(M) * N * sizeof(float), AGX_BO_SLC_PIN | AGX_BO_ZERO, "nvfp4_y");
    if (!bo_w || !bo_x || !bo_y) return 1;

    agx_repack_nvfp4_pack(raw_w.data(), bo_w->cpu_ptr, K, N, 1);

    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto * x_ptr = static_cast<float *>(bo_x->cpu_ptr);
    for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i) x_ptr[i] = dist(rng);

    // CPU reference on first 64 rows (and full M)
    std::vector<float> ref_y(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> row_f32(K);
    const uint32_t check_rows = std::min<uint32_t>(N, 128u);
    for (uint32_t r = 0; r < check_rows; ++r) {
        agx_repack_nvfp4_dequant_row(bo_w->cpu_ptr, K, N, r, row_f32.data());
        for (uint32_t m = 0; m < M; ++m) {
            double acc = 0.0;
            for (uint32_t k = 0; k < K; ++k) acc += static_cast<double>(row_f32[k]) * x_ptr[m * K + k];
            ref_y[m * N + r] = static_cast<float>(acc);
        }
    }

    agx_gemm_desc desc{};
    desc.w_va           = bo_w->gpu_va;
    desc.x_va           = bo_x->gpu_va;
    desc.y_va           = bo_y->gpu_va;
    desc.M              = M;
    desc.N              = N;
    desc.K              = K;
    desc.x_stride_elems = K;
    desc.y_stride_elems = N;
    desc.layout         = AGX_WEIGHT_NVFP4_REPACKED;
    desc.post_barrier   = AGX_BARRIER_MEMORY_HOST;

    std::vector<uint64_t> times_ns;
    for (int it = 0; it < 20; ++it) {
        agx_cdm_stream_reset(ctx->stream);
        for (uint32_t r = 0; r < reps_batch; ++r) {
            desc.post_barrier = (r + 1u == reps_batch) ? AGX_BARRIER_MEMORY_HOST : AGX_BARRIER_WEAK_WFI;
            if (!agx_gemm_dispatch(ctx, ctx->stream, desc)) {
                std::fprintf(stderr, "FAIL: nvfp4 agx_gemm_dispatch (%u,%u,%u)\n", K, N, M);
                return 1;
            }
        }
        uint64_t gpu_ns = 0;
        if (agx_cdm_submit(ctx->stream, true, &gpu_ns) != 0) {
            std::fprintf(stderr, "FAIL: nvfp4 submit (%u,%u,%u)\n", K, N, M);
            return 1;
        }
        if (it >= 5) times_ns.push_back(gpu_ns);
    }

    auto * y_ptr = static_cast<const float *>(bo_y->cpu_ptr);
    std::vector<float> sub_ref, sub_got;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t r = 0; r < check_rows; ++r) {
            sub_ref.push_back(ref_y[m * N + r]);
            sub_got.push_back(y_ptr[m * N + r]);
        }
    }
    Parity p = check_parity(sub_ref.data(), sub_got.data(), sub_ref.size());
    double med_ns = median_u64(times_ns);
    double total_bytes = static_cast<double>(info.total_bytes + M * K * 4u + M * N * 4u) * reps_batch;
    double gbps = med_ns > 0.0 ? (total_bytes / med_ns) : 0.0;
    double gflops = med_ns > 0.0 ? (2.0 * M * N * K * reps_batch / med_ns) : 0.0;

    std::printf("  NVFP4_32x16 (%4u x %6u, M=%u, x%2u): %.3f ms | %6.2f GB/s | %6.2f GFLOP/s | nan=%d max_abs=%.4g cos=%.7f\n",
                K, N, M, reps_batch, med_ns * 1e-6, gbps, gflops, p.n_nan + p.n_inf, p.max_abs, p.cosine);

    agx_bo_free(ctx->drm, bo_y);
    agx_bo_free(ctx->drm, bo_x);
    agx_bo_free(ctx->drm, bo_w);
    return (p.n_nan == 0 && p.n_inf == 0 && p.cosine > 0.999) ? 0 : 1;
}

static int bench_q8_0(agx_backend_context * ctx, uint32_t K, uint32_t N, uint32_t M, uint32_t reps_batch) {
    agx_repack_tensor_info info;
    agx_repack_describe(GGML_TYPE_Q8_0, K, N, 1, 1, &info);

    const size_t n_blks = static_cast<size_t>(N) * (K / QK8_0);
    std::vector<block_q8_0> raw_w(n_blks);
    std::mt19937 rng(7777 + K + N + M);
    for (size_t i = 0; i < n_blks; ++i) {
        raw_w[i].d = GGML_FP32_TO_FP16(0.01f * (1 + static_cast<int>(rng() % 10)));
        for (int j = 0; j < QK8_0; ++j) raw_w[i].qs[j] = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
    }

    agx_bo * bo_w = agx_bo_alloc(ctx->drm, info.total_bytes, AGX_BO_SLC_STREAM | AGX_BO_READONLY, "q8_0_w");
    agx_bo * bo_x = agx_bo_alloc(ctx->drm, static_cast<size_t>(M) * K * sizeof(float), AGX_BO_SLC_PIN, "q8_0_x");
    agx_bo * bo_y = agx_bo_alloc(ctx->drm, static_cast<size_t>(M) * N * sizeof(float), AGX_BO_SLC_PIN | AGX_BO_ZERO, "q8_0_y");
    if (!bo_w || !bo_x || !bo_y) return 1;

    agx_repack_q8_0_pack(raw_w.data(), bo_w->cpu_ptr, K, N, 1);

    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto * x_ptr = static_cast<float *>(bo_x->cpu_ptr);
    for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i) x_ptr[i] = dist(rng);

    std::vector<float> ref_y(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> row_f32(K);
    const uint32_t check_rows = std::min<uint32_t>(N, 128u);
    for (uint32_t r = 0; r < check_rows; ++r) {
        agx_repack_q8_0_dequant_row(bo_w->cpu_ptr, K, r, row_f32.data());
        for (uint32_t m = 0; m < M; ++m) {
            double acc = 0.0;
            for (uint32_t k = 0; k < K; ++k) acc += static_cast<double>(row_f32[k]) * x_ptr[m * K + k];
            ref_y[m * N + r] = static_cast<float>(acc);
        }
    }

    agx_gemm_desc desc{};
    desc.w_va           = bo_w->gpu_va;
    desc.x_va           = bo_x->gpu_va;
    desc.y_va           = bo_y->gpu_va;
    desc.M              = M;
    desc.N              = N;
    desc.K              = K;
    desc.x_stride_elems = K;
    desc.y_stride_elems = N;
    desc.layout         = AGX_WEIGHT_Q8_0_REPACKED;
    desc.post_barrier   = AGX_BARRIER_MEMORY_HOST;

    std::vector<uint64_t> times_ns;
    for (int it = 0; it < 15; ++it) {
        agx_cdm_stream_reset(ctx->stream);
        for (uint32_t r = 0; r < reps_batch; ++r) {
            desc.post_barrier = (r + 1u == reps_batch) ? AGX_BARRIER_MEMORY_HOST : AGX_BARRIER_WEAK_WFI;
            if (!agx_gemm_dispatch(ctx, ctx->stream, desc)) {
                std::fprintf(stderr, "FAIL: q8_0 agx_gemm_dispatch (%u,%u,%u)\n", K, N, M);
                return 1;
            }
        }
        uint64_t gpu_ns = 0;
        if (agx_cdm_submit(ctx->stream, true, &gpu_ns) != 0) {
            std::fprintf(stderr, "FAIL: q8_0 submit (%u,%u,%u)\n", K, N, M);
            return 1;
        }
        if (it >= 3) times_ns.push_back(gpu_ns);
    }

    auto * y_ptr = static_cast<const float *>(bo_y->cpu_ptr);
    std::vector<float> sub_ref, sub_got;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t r = 0; r < check_rows; ++r) {
            sub_ref.push_back(ref_y[m * N + r]);
            sub_got.push_back(y_ptr[m * N + r]);
        }
    }
    Parity p = check_parity(sub_ref.data(), sub_got.data(), sub_ref.size());
    double med_ns = median_u64(times_ns);
    double total_bytes = static_cast<double>(info.total_bytes + M * K * 4u + M * N * 4u) * reps_batch;
    double gbps = med_ns > 0.0 ? (total_bytes / med_ns) : 0.0;
    double gflops = med_ns > 0.0 ? (2.0 * M * N * K * reps_batch / med_ns) : 0.0;

    std::printf("  Q8_0_8BLK   (%4u x %6u, M=%u, x%2u): %.3f ms | %6.2f GB/s | %6.2f GFLOP/s | nan=%d max_abs=%.4g cos=%.7f\n",
                K, N, M, reps_batch, med_ns * 1e-6, gbps, gflops, p.n_nan + p.n_inf, p.max_abs, p.cosine);

    agx_bo_free(ctx->drm, bo_y);
    agx_bo_free(ctx->drm, bo_x);
    agx_bo_free(ctx->drm, bo_w);
    return (p.n_nan == 0 && p.n_inf == 0 && p.cosine > 0.999) ? 0 : 1;
}

} // namespace

int main() {
    agx_backend_context ctx{};
    ctx.drm = agx_drm_open(0);
    if (!ctx.drm) return 1;
    const char * kdir = std::getenv("GGML_AGX_KERNEL_DIR");
    if (!kdir || !*kdir) kdir = "/home/astaff/llama/llama-ggml-agx/ggml/src/ggml-agx/kernels";
    ctx.compiler = agx_compiler_create(ctx.drm, kdir, "/tmp/ggml-agx-gemm-cache");
    ctx.stream   = agx_cdm_stream_create(ctx.drm);
    if (!ctx.compiler || !ctx.stream || !agx_gemm_init(&ctx)) return 1;

    std::printf("=== Pillar 2 Repacked NVFP4 / Q8_0 M4 Max Hardware Parity & Roofline ===\n");
    int fails = 0;
    // Large DRAM-resident LM head (Q8_0 491 MB > 48 MB SLC -> true DRAM roofline!)
    fails += bench_q8_0(&ctx, 2304, 201024, 1, 1);
    // Large DRAM-resident NVFP4 matrix (K=2304, N=65536 = 84.9 MB > 48 MB SLC -> true DRAM roofline!)
    fails += bench_nvfp4(&ctx, 2304, 65536, 1, 1);
    // Inkling-2 exact MoE & dense shapes (SLC-resident / hot working set)
    fails += bench_nvfp4(&ctx, 2304, 2304, 1, 8);
    fails += bench_nvfp4(&ctx, 2304, 2304, 5, 4);
    fails += bench_nvfp4(&ctx, 2304, 2304, 8, 4);
    fails += bench_nvfp4(&ctx, 1152, 2304, 1, 8);
    fails += bench_q8_0 (&ctx, 2304, 3072, 1, 8);
    fails += bench_q8_0 (&ctx, 2304, 201024, 8, 1);
    // Prefill 8x8 simdgroup_matrix MMA TFLOPS roofline (M=512)
    fails += bench_nvfp4(&ctx, 2304, 2304, 512, 1);
    fails += bench_q8_0 (&ctx, 2304, 2304, 512, 1);

    agx_gemm_cleanup(&ctx);
    agx_cdm_stream_destroy(ctx.stream);
    agx_compiler_destroy(ctx.compiler);
    agx_drm_close(ctx.drm);

    if (fails != 0) {
        std::fprintf(stderr, "test_agx_gemm_roofline FAILED (%d)\n", fails);
        return 1;
    }
    std::printf("ALL test_agx_gemm_roofline PASSED\n");
    return 0;
}
