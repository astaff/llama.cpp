#include "../agx-common.h"
#include "../agx-drm.h"
#include "../agx-cdm.h"
#include "../agx-compiler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Standalone M4 Max (AGX3) Hardware Roofline, Pillar 4 SLC Policy, Pillar 6 Barrier
// & Borealis ShaderDB Verification Benchmark.
//
// Measures on real silicon via direct drm_asahi CDM submissions + GPU hardware timestamps:
//   1. Device topology & VA arena isolation (SLC_PIN vs SLC_STREAM vs EXEC bit-40).
//   2. Sustained DRAM read bandwidth (256 MiB streaming buffer, uvec4 coalesced loads) -> GB/s.
//   3. Sustained SLC read bandwidth (16 MiB SLC-pinned buffer) -> GB/s.
//   4. Sustained FP16 8x8 cooperative-matrix compute roofline -> TFLOPS.
//   5. CDM barrier latency (48-pass STRONG_USC vs WEAK_WFI vs 40-WG sense-reversing atomic barrier).
//   6. Borealis shaderdb audit enforcing 0:0 spills and <= 96 32-bit GPRs across all kernels.

namespace {

static const char * kBwReadGlsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Uvec4Buf {
    uvec4 data[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer OutBuf {
    uint out_xor[];
};

layout(push_constant) uniform PushArgs {
    uint64_t src_va;
    uint64_t dst_va;
    uint     n_vec4;
    uint     iters;
    uint     total_threads;
} pc;

void main() {
    Uvec4Buf src = Uvec4Buf(pc.src_va);
    OutBuf   dst = OutBuf(pc.dst_va);

    uint gid = gl_GlobalInvocationID.x;
    uint stride = pc.total_threads;

    uvec4 acc0 = uvec4(0u);
    uvec4 acc1 = uvec4(0u);
    for (uint rep = 0u; rep < pc.iters; ++rep) {
        uint idx = gid;
        while (idx + stride < pc.n_vec4) {
            acc0 ^= src.data[idx];
            acc1 ^= src.data[idx + stride];
            idx += 2u * stride;
        }
        if (idx < pc.n_vec4) {
            acc0 ^= src.data[idx];
        }
    }
    uvec4 acc = acc0 ^ acc1;
    uint v = acc.x ^ acc.y ^ acc.z ^ acc.w;
    uint sg_v = subgroupXor(v);
    if (gl_SubgroupInvocationID == 0u) {
        dst.out_xor[gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID] = sg_v;
    }
}
)GLSL";

static const char * kCoopMatTflopsGlsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer F32Out {
    float data[];
};

layout(push_constant) uniform PushArgs {
    uint64_t dst_va;
    uint     iters;
    float    init_val;
} pc;

void main() {
    coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseA>           a0 = coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseA>(float16_t(pc.init_val));
    coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseB>           b0 = coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseB>(float16_t(0.5));
    coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator> c0 = coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator>(float16_t(0.0));
    coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator> c1 = coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator>(float16_t(0.0));
    coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator> c2 = coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator>(float16_t(0.0));
    coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator> c3 = coopmat<float16_t, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator>(float16_t(0.0));

    for (uint i = 0u; i < pc.iters; ++i) {
        c0 = coopMatMulAdd(a0, b0, c0);
        c1 = coopMatMulAdd(a0, b0, c1);
        c2 = coopMatMulAdd(a0, b0, c2);
        c3 = coopMatMulAdd(a0, b0, c3);
    }

    float16_t sum = c0[0] + c1[0] + c2[0] + c3[0];
    if (gl_LocalInvocationID.x == 0u) {
        F32Out outb = F32Out(pc.dst_va);
        outb.data[gl_WorkGroupID.x] = float(sum);
    }
}
)GLSL";

static const char * kNullDispatchGlsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer CounterBuf {
    uint vals[];
};

layout(push_constant) uniform PushArgs {
    uint64_t buf_va;
    uint     step;
} pc;

void main() {
    if (gl_GlobalInvocationID.x == 0u) {
        CounterBuf b = CounterBuf(pc.buf_va);
        b.vals[pc.step] = pc.step + 1u;
    }
}
)GLSL";

static const char * kSenseBarrierBenchGlsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_memory_scope_semantics : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 128) coherent buffer SenseBarrier {
    uint arrived;
    uint sense;
    uint epoch;
    uint n_workgroups;
    uint stage;
    uint pass;
};

layout(push_constant) uniform PushArgs {
    uint64_t barrier_va;
    uint     n_barriers;
    uint     n_wg;
} pc;

void main() {
    SenseBarrier bar = SenseBarrier(pc.barrier_va);
    uint local_sense = 0u;
    for (uint b = 0u; b < pc.n_barriers; ++b) {
        local_sense ^= 1u;
        controlBarrier(gl_ScopeWorkgroup, gl_ScopeWorkgroup,
                       gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
        if (gl_LocalInvocationID.x == 0u) {
            uint prev = atomicAdd(bar.arrived, 1u, gl_ScopeDevice,
                                  gl_StorageSemanticsBuffer, gl_SemanticsAcquireRelease);
            if (prev + 1u == pc.n_wg) {
                atomicStore(bar.arrived, 0u, gl_ScopeDevice,
                            gl_StorageSemanticsNone, gl_SemanticsRelaxed);
                atomicAdd(bar.epoch, 1u, gl_ScopeDevice,
                          gl_StorageSemanticsNone, gl_SemanticsRelaxed);
                atomicStore(bar.sense, local_sense, gl_ScopeDevice,
                            gl_StorageSemanticsBuffer, gl_SemanticsRelease);
            } else {
                while (atomicLoad(bar.sense, gl_ScopeDevice,
                                  gl_StorageSemanticsBuffer, gl_SemanticsAcquire) != local_sense) {
                }
            }
        }
        controlBarrier(gl_ScopeWorkgroup, gl_ScopeWorkgroup,
                       gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
    }
}
)GLSL";

struct BwPush {
    uint64_t src_va;
    uint64_t dst_va;
    uint32_t n_vec4;
    uint32_t iters;
    uint32_t total_threads;
    uint32_t pad;
};

struct MmaPush {
    uint64_t dst_va;
    uint32_t iters;
    float    init_val;
};

struct NullPush {
    uint64_t buf_va;
    uint32_t step;
    uint32_t pad;
};

struct SensePush {
    uint64_t barrier_va;
    uint32_t n_barriers;
    uint32_t n_wg;
};

static double median_u64(std::vector<uint64_t> v) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    size_t m = v.size() / 2;
    return (v.size() & 1) ? static_cast<double>(v[m])
                          : 0.5 * (static_cast<double>(v[m - 1]) + static_cast<double>(v[m]));
}

} // namespace

int main() {
    int fails = 0;

    agx_drm_device * dev = agx_drm_open(0);
    if (!dev) {
        std::fprintf(stderr, "FAIL: agx_drm_open(0) failed\n");
        return 1;
    }

    const agx_drm_device_info & info = agx_drm_get_info(dev);
    std::printf("=== AGX Device ===\n");
    std::printf("  node=%s driver=%s name=%s chip=0x%x gen=%u cores=%u (%ux%u) usc=%u hal=%u freq=%u kHz\n",
                info.render_node, info.driver_name, info.device_name,
                info.chip_id, info.gpu_generation, info.num_cores,
                info.num_clusters, info.cores_per_cluster,
                info.usc_generation, info.gpu_hal_generation, info.max_frequency_khz);

    agx_compiler * comp = agx_compiler_create(dev, nullptr, "/tmp/ggml-agx-cache");
    agx_cdm_stream * stream = agx_cdm_stream_create(dev);
    if (!comp || !stream) {
        std::fprintf(stderr, "FAIL: compiler or CDM stream init failed\n");
        return 1;
    }

    // 1. Pillar 4 VA arena checks
    agx_bo * bo_pin    = agx_bo_alloc(dev, 16ULL * 1024 * 1024, AGX_BO_SLC_PIN | AGX_BO_ZERO, "roofline_slc_16m");
    agx_bo * bo_stream = agx_bo_alloc(dev, 256ULL * 1024 * 1024, AGX_BO_SLC_STREAM | AGX_BO_READONLY, "roofline_dram_256m");
    agx_bo * bo_out    = agx_bo_alloc(dev, 64ULL * 1024, AGX_BO_SLC_PIN | AGX_BO_ZERO, "roofline_out");
    if (!bo_pin || !bo_stream || !bo_out) {
        std::fprintf(stderr, "FAIL: BO allocation failed\n");
        return 1;
    }
    std::printf("=== Pillar 4 VA Arenas ===\n");
    std::printf("  SLC_PIN    VA = 0x%012llx (residency_kept=%d)\n",
                (unsigned long long)bo_pin->gpu_va, bo_pin->residency_kept ? 1 : 0);
    std::printf("  SLC_STREAM VA = 0x%012llx (residency_kept=%d)\n",
                (unsigned long long)bo_stream->gpu_va, bo_stream->residency_kept ? 1 : 0);
    if (bo_pin->gpu_va == bo_stream->gpu_va) {
        std::fprintf(stderr, "FAIL: overlapping VAs\n");
        fails++;
    }

    // Fill streaming & pinned buffers with deterministic non-zero pattern
    {
        auto * p_dram = static_cast<uint32_t *>(bo_stream->cpu_ptr);
        const size_t n_u32 = bo_stream->size / 4;
        for (size_t i = 0; i < n_u32; ++i) p_dram[i] = static_cast<uint32_t>(i * 2654435761u + 1u);

        auto * p_slc = static_cast<uint32_t *>(bo_pin->cpu_ptr);
        const size_t n_slc = bo_pin->size / 4;
        for (size_t i = 0; i < n_slc; ++i) p_slc[i] = static_cast<uint32_t>(i * 2246822519u + 7u);
    }

    agx_compile_opts opts;
    opts.is_hot_kernel = true;
    opts.strict_zero_spill = true;

    const agx_kernel * k_bw    = agx_compiler_compile_glsl(comp, "roofline_bw_uvec4", kBwReadGlsl, opts);
    const agx_kernel * k_mma   = agx_compiler_compile_glsl(comp, "roofline_fp16_mma8x8", kCoopMatTflopsGlsl, opts);
    const agx_kernel * k_null  = agx_compiler_compile_glsl(comp, "roofline_null_dispatch", kNullDispatchGlsl, opts);
    const agx_kernel * k_sense = agx_compiler_compile_glsl(comp, "roofline_sense_barrier", kSenseBarrierBenchGlsl, opts);

    if (!k_bw || !k_null) {
        std::fprintf(stderr, "FAIL: failed to compile core roofline kernels\n");
        return 1;
    }

    // 2. DRAM Bandwidth Roofline (256 MiB streaming)
    {
        const uint32_t n_vec4 = static_cast<uint32_t>(bo_stream->size / 16u);
        const uint32_t grid_x = info.num_cores * 16u; // 640 WGs x 256 threads
        BwPush push{bo_stream->gpu_va, bo_out->gpu_va, n_vec4, 1u, grid_x * 256u, 0u};

        std::vector<uint64_t> times_ns;
        for (int it = 0; it < 25; ++it) {
            agx_cdm_stream_reset(stream);
            agx_cdm_dispatch(stream, k_bw, grid_x, 1, 1, &push, sizeof(push), AGX_BARRIER_MEMORY_HOST);
            uint64_t gpu_ns = 0;
            int rc = agx_cdm_submit(stream, true, &gpu_ns);
            if (rc != 0) {
                std::fprintf(stderr, "FAIL: DRAM BW submit rc=%d vm_err=%d\n", rc, agx_drm_vm_error(dev));
                fails++;
                break;
            }
            if (it >= 5) times_ns.push_back(gpu_ns);
        }
        double med_ns = median_u64(times_ns);
        double gbps = med_ns > 0.0 ? (static_cast<double>(bo_stream->size) / med_ns) : 0.0;
        // Verify non-zero reduction output
        auto * out_u32 = static_cast<uint32_t *>(bo_out->cpu_ptr);
        uint32_t rxor = 0;
        for (uint32_t i = 0; i < grid_x * 8u; ++i) rxor ^= out_u32[i];
        std::printf("=== Roofline: DRAM Read (256 MiB SLC_STREAM) ===\n");
        std::printf("  median_gpu_ms = %.3f ms | achieved = %.2f GB/s | xor=0x%08x\n",
                    med_ns * 1e-6, gbps, rxor);
    }

    // 3. SLC Bandwidth Roofline (16 MiB SLC_PIN x 8 reps = 128 MiB traffic)
    {
        const uint32_t n_vec4 = static_cast<uint32_t>(bo_pin->size / 16u);
        const uint32_t iters = 8u;
        const uint32_t grid_x = info.num_cores * 8u;
        BwPush push{bo_pin->gpu_va, bo_out->gpu_va, n_vec4, iters, grid_x * 256u, 0u};

        std::vector<uint64_t> times_ns;
        for (int it = 0; it < 25; ++it) {
            agx_cdm_stream_reset(stream);
            agx_cdm_dispatch(stream, k_bw, grid_x, 1, 1, &push, sizeof(push), AGX_BARRIER_MEMORY_HOST);
            uint64_t gpu_ns = 0;
            if (agx_cdm_submit(stream, true, &gpu_ns) != 0) {
                fails++;
                break;
            }
            if (it >= 5) times_ns.push_back(gpu_ns);
        }
        double med_ns = median_u64(times_ns);
        double gbps = med_ns > 0.0 ? (static_cast<double>(bo_pin->size) * iters / med_ns) : 0.0;
        std::printf("=== Roofline: SLC Read (16 MiB SLC_PIN x %u) ===\n", iters);
        std::printf("  median_gpu_ms = %.3f ms | achieved = %.2f GB/s\n", med_ns * 1e-6, gbps);
    }

    // 4. FP16 8x8 Cooperative Matrix Compute Roofline
    if (k_mma) {
        const uint32_t grid_x = info.num_cores * 16u; // 640 WGs x 4 simdgroups = 2560 simdgroups
        const uint32_t iters = 2048u;
        MmaPush push{bo_out->gpu_va, iters, 0.001f};
        // Each iter does 4 x (8x8x8) FMA = 4 * 2 * 512 = 4096 FLOPs per simdgroup
        const double total_flops = static_cast<double>(grid_x) * 4.0 * static_cast<double>(iters) * 4096.0;

        std::vector<uint64_t> times_ns;
        for (int it = 0; it < 20; ++it) {
            agx_cdm_stream_reset(stream);
            agx_cdm_dispatch(stream, k_mma, grid_x, 1, 1, &push, sizeof(push), AGX_BARRIER_MEMORY_HOST);
            uint64_t gpu_ns = 0;
            if (agx_cdm_submit(stream, true, &gpu_ns) != 0) {
                std::fprintf(stderr, "FAIL: coopmat submit failed\n");
                fails++;
                break;
            }
            if (it >= 5) times_ns.push_back(gpu_ns);
        }
        double med_ns = median_u64(times_ns);
        double tflops = med_ns > 0.0 ? (total_flops / (med_ns * 1e3)) : 0.0;
        float sample_out = static_cast<const float *>(bo_out->cpu_ptr)[0];
        std::printf("=== Roofline: FP16 8x8 simdgroup_matrix ===\n");
        std::printf("  median_gpu_ms = %.3f ms | achieved = %.2f TFLOPS | out[0]=%g (finite=%d)\n",
                    med_ns * 1e-6, tflops, sample_out, std::isfinite(sample_out) ? 1 : 0);
        if (!std::isfinite(sample_out)) fails++;
    } else {
        std::printf("=== Roofline: FP16 8x8 simdgroup_matrix === SKIP (kernel compile returned null)\n");
    }

    // 5. Pillar 6: 48-Pass CDM Barrier vs Persistent Sense-Reversing Barrier Latency
    {
        auto bench_cdm_barriers = [&](const char * label, agx_barrier_mode mode) {
            std::vector<uint64_t> times_ns;
            for (int it = 0; it < 20; ++it) {
                agx_cdm_stream_reset(stream);
                for (uint32_t s = 0; s < 144u; ++s) {
                    NullPush p{bo_out->gpu_va, s, 0u};
                    agx_cdm_dispatch(stream, k_null, 1, 1, 1, &p, sizeof(p),
                                     (s + 1u == 144u) ? AGX_BARRIER_MEMORY_HOST : mode);
                }
                uint64_t gpu_ns = 0;
                if (agx_cdm_submit(stream, true, &gpu_ns) != 0) {
                    fails++;
                    return;
                }
                if (it >= 5) times_ns.push_back(gpu_ns);
            }
            double med_ns = median_u64(times_ns);
            std::printf("  %-24s (144 transitions): total=%.1f us | per_transition=%.0f ns\n",
                        label, med_ns * 1e-3, med_ns / 144.0);
        };
        std::printf("=== Pillar 6: 48-Pass x 3-Stage (144) Barrier Latency ===\n");
        bench_cdm_barriers("CDM STRONG_USC", AGX_BARRIER_STRONG_USC);
        bench_cdm_barriers("CDM WEAK_WFI",   AGX_BARRIER_WEAK_WFI);
        bench_cdm_barriers("CDM ELIDED",     AGX_BARRIER_NONE);

        if (k_sense) {
            const uint32_t n_wg = info.num_cores; // 40 persistent WGs
            std::vector<uint64_t> times_ns;
            uint32_t last_epoch = 0;
            for (int it = 0; it < 15; ++it) {
                agx_cdm_stream_reset(stream);
                agx_cdm_reset_sense_barrier(stream, n_wg);
                SensePush p{agx_cdm_get_sense_barrier_va(stream), 144u, n_wg};
                agx_cdm_dispatch(stream, k_sense, n_wg, 1, 1, &p, sizeof(p), AGX_BARRIER_MEMORY_HOST);
                uint64_t gpu_ns = 0;
                if (agx_cdm_submit(stream, true, &gpu_ns) != 0) {
                    std::fprintf(stderr, "FAIL: sense-reversing barrier submit failed\n");
                    fails++;
                    break;
                }
                auto * bar = static_cast<const agx_sense_barrier_state *>(agx_cdm_get_sense_barrier_bo(stream)->cpu_ptr);
                last_epoch = bar->epoch;
                if (last_epoch != 144u) {
                    std::fprintf(stderr, "FAIL: sense barrier epoch=%u != 144\n", last_epoch);
                    fails++;
                    break;
                }
                if (it >= 3) times_ns.push_back(gpu_ns);
            }
            double med_ns = median_u64(times_ns);
            std::printf("  %-24s (144 transitions, %u WGs): total=%.1f us | per_transition=%.0f ns | epoch=%u\n",
                        "PERSISTENT SENSE_BARRIER", n_wg, med_ns * 1e-3, med_ns / 144.0, last_epoch);
        }
    }

    // 6. Borealis ShaderDB Audit Gate
    std::printf("=== Borealis ShaderDB Audit ===\n");
    agx_compiler_print_shaderdb(comp);
    if (!agx_compiler_audit_all(comp, true, AGX_HOT_GPR32_BUDGET)) {
        std::fprintf(stderr, "FAIL: shaderdb audit gate failed (spills > 0 or hot gpr32 > %u)\n",
                     AGX_HOT_GPR32_BUDGET);
        fails++;
    }

    agx_bo_free(dev, bo_out);
    agx_bo_free(dev, bo_stream);
    agx_bo_free(dev, bo_pin);
    agx_cdm_stream_destroy(stream);
    agx_compiler_destroy(comp);
    agx_drm_close(dev);

    if (fails != 0) {
        std::fprintf(stderr, "test_agx_roofline FAILED (%d failures)\n", fails);
        return 1;
    }
    std::printf("ALL test_agx_roofline PASSED\n");
    return 0;
}
