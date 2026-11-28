#pragma once

#include "agx-common.h"
#include "agx-drm.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ============================================================================
// Pillar 1: Native AGX3B Borealis Compiler Pipeline & ShaderDB Auditor
// ============================================================================
// Compiles GLSL 450 compute shaders (`GL_EXT_buffer_reference` + `push_constant`
// + `GL_KHR_cooperative_matrix` + `GL_KHR_shader_subgroup_*` + `GL_KHR_memory_scope_semantics`)
// -> SPIR-V (via glslangValidator/glslc) -> native AGX3B machine code (via Borealis
// compiler service) with disk caching (`.agxbin`) and strict GPR/spill auditing
// (`0:0` spills/fills, `<= 96` 32-bit GPRs on hot kernels).
//
// Also supports direct loading of raw pre-assembled AGX3B shader binaries.

struct agx_shader_stats {
    uint32_t nr_gprs_16              = 0;  // 16-bit half-registers reported by Borealis
    uint32_t nr_gprs_32              = 0;  // ceil(nr_gprs_16 / 2), hot budget <= 96
    uint32_t spills                  = 0;  // must be 0
    uint32_t fills                   = 0;  // must be 0
    uint32_t scratch_bytes           = 0;  // thread-private spill frame (bytes)
    uint32_t staging_halfs           = 0;  // AGX3 IPR staging footprint (halfwords)
    uint32_t instructions            = 0;
    uint32_t code_bytes              = 0;
    uint32_t threadgroup_bytes       = 0;
    uint32_t max_simdgroups_per_core = 0;  // dynamic caching occupancy estimate
    bool     stats_valid             = false;
};

struct agx_compile_opts {
    const char *             entry_point         = "main";
    std::vector<std::string> defines;              // e.g. "TILE_M=32", "USE_COOPMAT=1"
    std::vector<std::string> include_dirs;
    uint32_t                 push_constant_bytes = AGX_MAX_PUSH_BYTES;
    bool                     is_hot_kernel       = true;
    bool                     strict_zero_spill   = true;
    bool                     allow_preamble      = false;
};

// Uniform DMA span promoted by Borealis from constant-offset root/push loads.
struct agx_uniform_dma_span {
    uint16_t dest_half = 0;  // destination uniform half-register
    uint16_t halfs     = 0;  // span length in 16-bit halves (even)
    uint32_t offset_B  = 0;  // byte offset inside hk_root_descriptor_table
    int16_t  set       = -1; // -1 = root table / push constants
    uint16_t binding   = 0;
};

struct agx_kernel {
    std::string                 name;
    agx_bo *                    code_bo            = nullptr; // USC executable BO (>= 0x10000000000)
    agx_bo *                    const_data_bo      = nullptr; // optional NIR constant data BO
    uint64_t                    shader_va          = 0;       // 64B-aligned entry point GPU VA
    uint64_t                    const_data_va      = 0;
    uint32_t                    main_offset        = 0;
    uint32_t                    main_size          = 0;
    uint32_t                    binary_size        = 0;
    uint32_t                    local_size[3]      = { 32, 1, 1 };
    uint32_t                    threadgroup_bytes  = 0;
    uint32_t                    scratch_bytes      = 0;
    uint32_t                    staging_halfs      = 0;
    uint16_t                    root_uniform_half  = 0;       // uniform half receiving root_ptr (0 for pure compute)
    uint32_t                    push_offset_bytes  = 0;       // offsetof(hk_root_descriptor_table, push)
    uint32_t                    root_size_bytes    = 0;       // total root descriptor table upload size
    bool                        uses_root          = true;

    // Optional compiler rodata range (constant LUTs inside code_bo)
    uint16_t                    rodata_base_half   = 0;
    uint16_t                    rodata_size_16     = 0;
    uint32_t                    rodata_offset      = 0;

    // Optional promoted uniform DMA ranges from root/push buffer
    std::vector<agx_uniform_dma_span> uniform_dma;

    // Optional native constant program (preamble)
    bool                        has_preamble       = false;
    uint64_t                    preamble_va        = 0;
    uint32_t                    preamble_size      = 0;
    uint32_t                    ti_spill_bytes     = 0;
    uint32_t                    ti_ipr_bytes       = 0;

    bool                        is_hot_kernel      = true;
    agx_shader_stats            stats;
};

struct agx_compiler;

agx_compiler *     agx_compiler_create(agx_drm_device * dev,
                                       const char *     kernel_dir = nullptr,
                                       const char *     cache_dir  = nullptr);
void               agx_compiler_destroy(agx_compiler * compiler);

// Compile GLSL 450 compute source string -> SPIR-V -> native AGX3B binary on USC BO.
// Results are memoized by `name` in memory and cached on disk in `cache_dir`.
const agx_kernel * agx_compiler_compile_glsl(agx_compiler *           compiler,
                                             const char *             name,
                                             const char *             glsl_source,
                                             const agx_compile_opts & opts = {});

// Compile a `.comp` file (path relative to `kernel_dir` or absolute).
const agx_kernel * agx_compiler_compile_file(agx_compiler *           compiler,
                                             const char *             name,
                                             const char *             path,
                                             const agx_compile_opts & opts = {});

// Compile pre-assembled SPIR-V words -> native AGX3B binary.
const agx_kernel * agx_compiler_compile_spirv(agx_compiler *           compiler,
                                              const char *             name,
                                              const uint32_t *         spirv_words,
                                              size_t                   word_count,
                                              const agx_compile_opts & opts = {});

// Load a raw pre-assembled AGX3B machine-code shader directly into USC executable memory.
const agx_kernel * agx_compiler_load_raw_agx3b(agx_compiler *  compiler,
                                               const char *    name,
                                               const void *    code_bytes,
                                               size_t          code_size,
                                               const uint32_t  local_size[3],
                                               uint32_t        nr_gprs_16,
                                               uint32_t        staging_halfs,
                                               uint32_t        threadgroup_bytes,
                                               uint32_t        scratch_bytes = 0);

// Look up an already compiled kernel by name.
const agx_kernel * agx_compiler_get(const agx_compiler * compiler, const char * name);

// Audit all compiled kernels against the `0:0` spill & `<= 96` 32-bit GPR budget.
bool               agx_compiler_audit_all(const agx_compiler * compiler,
                                          bool                 fail_on_spill = true,
                                          uint32_t             max_hot_gpr32 = AGX_HOT_GPR32_BUDGET);

// Print a Borealis shaderdb summary table for every compiled kernel.
void               agx_compiler_print_shaderdb(const agx_compiler * compiler);
