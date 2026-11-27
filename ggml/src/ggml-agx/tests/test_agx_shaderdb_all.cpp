#include "../agx-common.h"
#include "../agx-drm.h"
#include "../agx-compiler.h"
#include "../agx-gemm.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Comprehensive Borealis AGX3B shaderdb auditor for ALL ggml-agx compute shaders
// across Pillars 2, 3, 4, and 5.
// Enforces:
//   - 100% of kernels compile cleanly through glslangValidator -> Borealis AGX3B
//   - 0:0 spills/fills across 100% of kernels
//   - Hot kernels <= 96 32-bit GPRs (<= 192 16-bit half-GPRs)

struct KernelSpec {
    const char *             name;
    const char *             file;
    std::vector<std::string> defines;
    bool                     hot;
};

int main() {
    agx_drm_device * dev = agx_drm_open(0);
    if (!dev) {
        std::fprintf(stderr, "FAIL: agx_drm_open(0) failed\n");
        return 1;
    }

    const char * kdir = std::getenv("GGML_AGX_KERNEL_DIR");
    if (!kdir || !*kdir) {
        kdir = "/home/astaff/llama/llama-ggml-agx/ggml/src/ggml-agx/kernels";
    }
    agx_compiler * comp = agx_compiler_create(dev, kdir, "/tmp/ggml-agx-shaderdb-cache");
    if (!comp) {
        std::fprintf(stderr, "FAIL: agx_compiler_create failed\n");
        agx_drm_close(dev);
        return 1;
    }

    const std::vector<KernelSpec> specs = {
        // Pillar 2: NVFP4 / Q8_0 / MXFP8 vec-mat & 8x8 MMA GEMM
        { "nvfp4_vecmat_repack_m1",  "nvfp4_vecmat.comp",           {"REPACKED=1", "BATCH_M=1", "K_SGS=1"}, true },
        { "nvfp4_vecmat_repack_m5",  "nvfp4_vecmat.comp",           {"REPACKED=1", "BATCH_M=5", "K_SGS=4"}, true },
        { "nvfp4_vecmat_raw_m1",     "nvfp4_vecmat.comp",           {"REPACKED=0", "BATCH_M=1", "K_SGS=1"}, true },
        { "q8_0_vecmat_repack_m1",   "mxfp8_q8_vecmat.comp",        {"QUANT_MODE=0", "BATCH_M=1"}, true },
        { "q8_0_vecmat_raw_m1",      "mxfp8_q8_vecmat.comp",        {"QUANT_MODE=1", "BATCH_M=1"}, true },
        { "mxfp8_vecmat_repack_m1",  "mxfp8_q8_vecmat.comp",        {"QUANT_MODE=2", "BATCH_M=1"}, true },
        { "bf16_vecmat_m1",          "mxfp8_q8_vecmat.comp",        {"QUANT_MODE=4", "BATCH_M=1"}, true },
        { "f32_vecmat_m1",           "mxfp8_q8_vecmat.comp",        {"QUANT_MODE=6", "BATCH_M=1"}, true },
        { "agx_mma_gemm_prefill",    "agx_mma_gemm.comp",           {"TILE_MODE=0"}, true },
        { "agx_mma_gemm_dspark",     "agx_mma_gemm.comp",           {"TILE_MODE=1"}, true },

        // Pillar 3: Fused Inkling-2 loop prep, rank-64 SRAM gates, softcap/argmax, DSpark MTP
        { "fused_ctrl_gates_sram",   "fused_ctrl_gates_sram.comp",  {}, true },
        { "fused_loop_prep_m0",      "fused_loop_prep_qkv_conv.comp", {"PREP_MODE=0"}, true },
        { "fused_loop_prep_m1",      "fused_loop_prep_qkv_conv.comp", {"PREP_MODE=1"}, true },
        { "fused_loop_prep_m2",      "fused_loop_prep_qkv_conv.comp", {"PREP_MODE=2"}, true },
        { "fused_loop_prep_m3",      "fused_loop_prep_qkv_conv.comp", {"PREP_MODE=3"}, true },
        { "fused_loop_prep_m4",      "fused_loop_prep_qkv_conv.comp", {"PREP_MODE=4"}, true },
        { "fused_loop_prep_m5",      "fused_loop_prep_qkv_conv.comp", {"PREP_MODE=5"}, true },
        { "fused_softcap_argmax",    "fused_softcap_argmax.comp",   {"SOFTCAP_MODE=0"}, true },
        { "dspark_mtp_m0",           "dspark_mtp.comp",             {"DSPARK_MODE=0"}, true },
        { "dspark_mtp_m1",           "dspark_mtp.comp",             {"DSPARK_MODE=1"}, true },
        { "dspark_mtp_m2",           "dspark_mtp.comp",             {"DSPARK_MODE=2"}, true },
        { "dspark_mtp_m3",           "dspark_mtp.comp",             {"DSPARK_MODE=3"}, true },

        // Pillar 4 & 5: Hybrid SWA/Full Attention, Bitonic MoE Bucket, Fused MoE
        { "fused_swa_ring_attn",     "fused_swa_ring_attn.comp",    {}, true },
        { "fused_full_flash_decode", "fused_full_flash_decode.comp",{}, true },
        { "moe_bitonic_bucket",      "moe_bitonic_bucket.comp",     {}, true },
        { "fused_moe_nvfp4_m1",      "fused_moe_gate_up_situ_down.comp", {"WEIGHT_LAYOUT=0", "BATCH_CAP=1"}, true },
        { "fused_moe_nvfp4_m5",      "fused_moe_gate_up_situ_down.comp", {"WEIGHT_LAYOUT=0", "BATCH_CAP=5"}, true },
        { "fused_moe_q8_0_m1",       "fused_moe_gate_up_situ_down.comp", {"WEIGHT_LAYOUT=2", "BATCH_CAP=1"}, true },
    };

    int compile_fails = 0;
    const char * skip_substr = std::getenv("AGX_SHADERDB_SKIP");
    for (const auto & sp : specs) {
        if (skip_substr && *skip_substr && std::string(sp.name).find(skip_substr) != std::string::npos) {
            std::fprintf(stderr, "[shaderdb] SKIP %s\n", sp.name);
            continue;
        }
        std::fprintf(stderr, "[shaderdb] compiling %s (%s)...\n", sp.name, sp.file);
        std::fflush(stderr);
        agx_compile_opts opts;
        opts.defines = sp.defines;
        opts.is_hot_kernel = sp.hot;
        opts.strict_zero_spill = false; // collect full table first, then enforce audit_all!
        const agx_kernel * k = agx_compiler_compile_file(comp, sp.name, sp.file, opts);
        if (!k) {
            std::fprintf(stderr, "COMPILE FAIL: %s (%s)\n", sp.name, sp.file);
            compile_fails++;
        }
    }

    std::printf("\n=== Borealis AGX3B Full-Corpus ShaderDB Audit ===\n");
    agx_compiler_print_shaderdb(comp);
    bool audit_ok = agx_compiler_audit_all(comp, true, AGX_HOT_GPR32_BUDGET);

    agx_compiler_destroy(comp);
    agx_drm_close(dev);

    std::printf("\nSummary: compile_fails=%d audit_ok=%d\n", compile_fails, audit_ok ? 1 : 0);
    return (compile_fails == 0 && audit_ok) ? 0 : 1;
}
