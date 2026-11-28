#include "agx-moe.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

// ============================================================================
// Host-side Exact Inkling 2 SiTU Reference (matches src/llama-graph.cpp:2636)
// ============================================================================

float agx_situ_ref(float gate, float up, float beta, float linear_beta) {
    const float sig = 1.0f / (1.0f + std::exp(-gate));
    const float act = beta * std::tanh(gate / beta) * sig;
    const float u_a = (linear_beta > 0.0f) ? (linear_beta * std::tanh(up / linear_beta)) : up;
    return act * u_a;
}

// ============================================================================
// Push-Constant Structs (exact match with kernels/moe_bitonic_bucket.comp &
//                        kernels/fused_moe_gate_up_situ_down.comp)
// ============================================================================

struct agx_push_moe_bucket {
    uint64_t topk_ids_va;
    uint64_t topk_w_va;
    uint64_t header_va;
    uint64_t vecmat_buckets_va;
    uint64_t gemm_tiles_va;
    uint64_t sorted_tok_va;
    uint64_t sorted_kslot_va;
    uint64_t sorted_w_va;
    uint64_t moe_out_va;
    uint64_t skipped_slots_va;
    uint32_t n_tokens;
    uint32_t top_k;
    uint32_t n_expert;
    uint32_t n_ff;
    uint32_t n_embd;
    uint32_t wg_gate_up_vecmat;
    uint32_t wg_down_vecmat;
    uint32_t wg_gate_up_gemm;
    uint32_t wg_down_gemm;
    uint32_t max_vecmat_buckets;
    uint32_t max_gemm_tiles;
    uint32_t max_vecmat_m;
    uint32_t gemm_min_m;
    float    exp_scale0;
    uint32_t zero_elems;
    uint32_t ids_row_stride;
    uint32_t gemm_tile_m;
    uint32_t gemm_local_x;
};
static_assert(sizeof(agx_push_moe_bucket) == 152, "agx_push_moe_bucket size check");

struct agx_push_moe_exec {
    uint64_t header_va;
    uint64_t buckets_va;
    uint64_t sorted_tok_va;
    uint64_t sorted_kslot_va;
    uint64_t sorted_w_va;
    uint64_t x_va;
    uint64_t w_gate_va;
    uint64_t w_up_va;
    uint64_t y_va;
    uint64_t w_slice_B;
    uint32_t K;
    uint32_t N;
    uint32_t n_ff;
    uint32_t top_k;
    uint32_t flags;
    float    situ_beta;
    float    situ_linear_beta;
    uint32_t ids_row_stride; // PASS_STAGE=4,5,6: int32 elems between token rows in ids (0 -> top_k)
    uint64_t w_scale_va;     // optional per-expert FP32 2nd-level scale [n_expert] (0 -> 1.0f)
};
static_assert(sizeof(agx_push_moe_exec) == 120, "agx_push_moe_exec size check");

enum agx_moe_shader_flags : uint32_t {
    AGX_MOE_FLAG_SEPARATE_GU      = 1u << 0,
    AGX_MOE_FLAG_X_F16            = 1u << 1,
    AGX_MOE_FLAG_RAW_MMID         = 1u << 2,
    AGX_MOE_FLAG_ACT_SILU         = 1u << 3,
    AGX_MOE_FLAG_ACT_GELU         = 1u << 4,
    AGX_MOE_FLAG_RAW_MMID_PERSLOT = 1u << 5,
    AGX_MOE_FLAG_DOWN_NOSCATTER   = 1u << 6,
    AGX_MOE_FLAG_INTER_F16        = 1u << 7,
    AGX_MOE_FLAG_SWAP_XZ          = 1u << 8,
};

struct agx_push_moe_cvt_f16 {
    uint64_t src_va;
    uint64_t dst_va;
    uint32_t n_vec4;
    uint32_t _pad;
};
static_assert(sizeof(agx_push_moe_cvt_f16) == 24, "agx_push_moe_cvt_f16 size check");

// ============================================================================
// MoE Runner State (SLC-pinned bucket table + hidden scratch + kernel cache)
// ============================================================================

static constexpr uint32_t AGX_MOE_WEIGHT_MODES = 9u;

struct agx_moe_runner {
    agx_drm_device * dev      = nullptr;
    agx_compiler *   compiler = nullptr;

    // SLC-pinned bucket & sort workspace (grows lazily for prefill M up to 8192)
    agx_bo * meta_bo            = nullptr;
    size_t   meta_cap_assigns   = 0;
    uint32_t max_vecmat_buckets = 0;
    uint32_t max_gemm_tiles     = 0;
    size_t   off_vecmat_buckets = 0;
    size_t   off_gemm_tiles     = 0;
    size_t   off_sorted_tok     = 0;
    size_t   off_sorted_kslot   = 0;
    size_t   off_sorted_w       = 0;
    size_t   off_skipped_slots  = 0;

    // SLC-pinned intermediate hidden activations: [M * top_k, n_ff] F32 or F16
    // (B=1 -> 36 KB, DSpark M=5 -> 184 KB, pp2048 FP16 -> 36 MiB SLC-resident!)
    agx_bo * hidden_bo        = nullptr;
    size_t   hidden_cap_bytes = 0;

    // Prefill & M>=2 zero-atomic down-projection scratch: [M * top_k, n_embd] F32
    // Eliminates atomicAddF32 CAS contention in PASS_STAGE=1/3 & eliminates moe_zero_out!
    agx_bo * down_bo          = nullptr;
    size_t   down_cap_bytes   = 0;

    // Wave 10 prefill F32->F16 X pre-conversion scratch (SLC-pinned <= 24 MiB, overlaps bucket!)
    agx_bo *           x_f16_bo        = nullptr;
    size_t             x_f16_cap_bytes = 0;
    const agx_kernel * k_cvt_f16       = nullptr;

    // Stream hazard tracking: avoids redundant entry STRONG_USC barrier when a barrier
    // has already occurred since the last MoE op, while protecting back-to-back MoE calls!
    const agx_cdm_stream * last_stream      = nullptr;
    uint64_t               last_stream_gen  = 0;
    uint64_t               last_strong_bars = UINT64_MAX;

    // Compiled kernels
    const agx_kernel * k_bucket_bitonic   = nullptr;
    const agx_kernel * k_bucket_histogram = nullptr;
    const agx_kernel * k_zero_out         = nullptr;
    const agx_kernel * k_zero_skipped     = nullptr;
    const agx_kernel * k_down_reduce      = nullptr;

    // Indexed by [weight_mode][batch_slot: 0=(cap 1), 1=(cap 4), 2=(cap 8)][inter_f16: 0|1]
    const agx_kernel * k_gu_vecmat[AGX_MOE_WEIGHT_MODES][3][2]       = {};
    const agx_kernel * k_down_vecmat[AGX_MOE_WEIGHT_MODES][3][2]     = {};
    const agx_kernel * k_down_vecmat_f32[AGX_MOE_WEIGHT_MODES][3][2] = {};
    // Indexed by [weight_mode][tile_slot: 0=(32M), 1=(48M), 2=(64M), 3=(8M)][inter_f16][x_f16][merged_gu]
    const agx_kernel * k_gu_gemm[AGX_MOE_WEIGHT_MODES][4][2][2][2]   = {};
    const agx_kernel * k_down_gemm[AGX_MOE_WEIGHT_MODES][4][2]       = {};
    const agx_kernel * k_direct_m1[AGX_MOE_WEIGHT_MODES]             = {};
    const agx_kernel * k_direct_m1_f32[AGX_MOE_WEIGHT_MODES]         = {};
    const agx_kernel * k_direct_gu_situ[AGX_MOE_WEIGHT_MODES]        = {};
    const agx_kernel * k_direct_down_wsum[AGX_MOE_WEIGHT_MODES]      = {};
};

static bool ensure_meta_bo(agx_moe_runner * r, size_t req_assigns) {
    const size_t cap = std::max<size_t>(req_assigns, 512u); // covers M<=64 with K=8
    if (r->meta_bo && cap <= r->meta_cap_assigns) {
        return true;
    }
    if (r->meta_bo) {
        agx_bo_free(r->dev, r->meta_bo);
        r->meta_bo = nullptr;
    }

    const uint32_t max_vm    = static_cast<uint32_t>(cap + AGX_MOE_MAX_EXPERTS + 16u);
    const uint32_t max_tiles = static_cast<uint32_t>(cap / AGX_MOE_GEMM_TILE_M_SML + AGX_MOE_MAX_EXPERTS + 16u);
    // Offset 0..127 = agx_moe_bucket_header; offset 128..151 = grid_zero_skipped (24B)
    size_t off = sizeof(agx_moe_bucket_header) + sizeof(agx_cdm_indirect_grid);
    const size_t off_vm   = agx_align_up(off, 128u); off = off_vm   + max_vm    * sizeof(agx_moe_bucket);
    const size_t off_gt   = agx_align_up(off, 128u); off = off_gt   + max_tiles * sizeof(agx_moe_bucket);
    const size_t off_tok  = agx_align_up(off, 128u); off = off_tok  + cap       * sizeof(uint32_t);
    const size_t off_ksl  = agx_align_up(off, 128u); off = off_ksl  + cap       * sizeof(uint32_t);
    const size_t off_w    = agx_align_up(off, 128u); off = off_w    + cap       * sizeof(float);
    const size_t off_skip = agx_align_up(off, 128u); off = off_skip + cap       * sizeof(uint32_t);
    const size_t total_B  = agx_align_up(off, AGX_PAGE_SIZE);

    // Pin hot meta_bo in SLC when <= 8 MiB; stream for huge stress tests (e.g. M=65536 -> 34 MiB!)
    const uint32_t meta_flags = (total_B <= 8u * 1024u * 1024u)
        ? (AGX_BO_SLC_PIN | AGX_BO_WRITEBACK | AGX_BO_ZERO)
        : (AGX_BO_SLC_STREAM | AGX_BO_WRITEBACK | AGX_BO_ZERO);
    r->meta_bo = agx_bo_alloc(r->dev, total_B, meta_flags, "moe-bucket-slc");
    if (!r->meta_bo) {
        return false;
    }
    r->meta_cap_assigns   = cap;
    r->max_vecmat_buckets = max_vm;
    r->max_gemm_tiles     = max_tiles;
    r->off_vecmat_buckets = off_vm;
    r->off_gemm_tiles     = off_gt;
    r->off_sorted_tok     = off_tok;
    r->off_sorted_kslot   = off_ksl;
    r->off_sorted_w       = off_w;
    r->off_skipped_slots  = off_skip;
    return true;
}

static bool ensure_hidden_bo(agx_moe_runner * r, size_t req_assigns, uint32_t n_ff, bool inter_f16 = false) {
    const size_t elem_B = inter_f16 ? sizeof(uint16_t) : sizeof(float);
    const size_t needed = req_assigns * static_cast<size_t>(n_ff) * elem_B;
    if (r->hidden_bo && needed <= r->hidden_cap_bytes) {
        return true;
    }
    if (r->hidden_bo) {
        agx_bo_free(r->dev, r->hidden_bo);
        r->hidden_bo = nullptr;
        r->hidden_cap_bytes = 0;
    }
    const size_t alloc_B = agx_align_up(std::max<size_t>(needed, 256u * 1024u), AGX_PAGE_SIZE);
    // Keep MoE hidden activations SLC-pinned up to 40 MiB (covers pp512 9 MiB & pp2048 36 MiB FP16!),
    // and ALWAYS include AGX_BO_WRITEBACK so GPU writes/re-reads cache in SLC!
    const uint32_t flags = (alloc_B <= 40u * 1024u * 1024u)
        ? (AGX_BO_SLC_PIN | AGX_BO_WRITEBACK)
        : (AGX_BO_SLC_STREAM | AGX_BO_WRITEBACK);
    r->hidden_bo = agx_bo_alloc(r->dev, alloc_B, flags, "moe-hidden-slc");
    if (!r->hidden_bo) {
        return false;
    }
    r->hidden_cap_bytes = alloc_B;
    return true;
}

static bool ensure_down_bo(agx_moe_runner * r, size_t req_assigns, uint32_t n_embd) {
    const size_t needed = req_assigns * static_cast<size_t>(n_embd) * sizeof(float);
    if (needed > 256u * 1024u * 1024u) {
        return false; // cap scratch at 256 MiB for huge synthetic stress tests
    }
    if (r->down_bo && needed <= r->down_cap_bytes) {
        return true;
    }
    if (r->down_bo) {
        agx_bo_free(r->dev, r->down_bo);
        r->down_bo = nullptr;
        r->down_cap_bytes = 0;
    }
    const size_t alloc_B = agx_align_up(std::max<size_t>(needed, 256u * 1024u), AGX_PAGE_SIZE);
    // Pin in SLC when <= 40 MiB (covers M=4 288 KiB & pp512 36 MiB!), writeback-stream for pp2048 (144 MiB)
    const uint32_t flags = (alloc_B <= 40u * 1024u * 1024u)
        ? (AGX_BO_SLC_PIN | AGX_BO_WRITEBACK)
        : (AGX_BO_SLC_STREAM | AGX_BO_WRITEBACK);
    r->down_bo = agx_bo_alloc(r->dev, alloc_B, flags, "moe-down-scratch");
    if (!r->down_bo) {
        return false;
    }
    r->down_cap_bytes = alloc_B;
    return true;
}

static bool ensure_x_f16_bo(agx_moe_runner * r, size_t req_bytes) {
    if (!r || !r->dev || req_bytes == 0 || req_bytes > 24u * 1024u * 1024u) {
        return false;
    }
    if (r->x_f16_bo && req_bytes <= r->x_f16_cap_bytes) {
        return true;
    }
    if (r->x_f16_bo) {
        agx_bo_free(r->dev, r->x_f16_bo);
        r->x_f16_bo = nullptr;
        r->x_f16_cap_bytes = 0;
    }
    const size_t alloc_B = agx_align_up(std::max<size_t>(req_bytes, 256u * 1024u), AGX_PAGE_SIZE);
    r->x_f16_bo = agx_bo_alloc(r->dev, alloc_B, AGX_BO_SLC_PIN | AGX_BO_WRITEBACK, "moe-x-f16-slc");
    if (!r->x_f16_bo) {
        return false;
    }
    r->x_f16_cap_bytes = alloc_B;
    return true;
}

static const agx_kernel * get_moe_cvt_f16_kernel(agx_moe_runner * r) {
    if (!r || !r->compiler) {
        return nullptr;
    }
    if (!r->k_cvt_f16) {
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
        opts.push_constant_bytes = sizeof(agx_push_moe_cvt_f16);
        opts.is_hot_kernel       = true;
        opts.strict_zero_spill   = true;
        r->k_cvt_f16 = agx_compiler_compile_glsl(r->compiler, "agx_moe_cvt_f32_to_f16_v4", k_src, opts);
    }
    return r->k_cvt_f16;
}

static void moe_entry_barrier(agx_moe_runner * r, agx_cdm_stream * stream) {
    if (!agx_cdm_is_empty(stream)) {
        agx_cdm_stats st{};
        agx_cdm_get_stats(stream, &st);
        const uint64_t gen  = agx_cdm_stream_record_gen(stream);
        const uint64_t bars = st.strong_usc_barriers + st.memory_host_barriers;
        if (r->last_stream == stream && r->last_stream_gen == gen && bars == r->last_strong_bars) {
            agx_cdm_barrier_usc(stream);
        }
    }
}

static void moe_exit_record(agx_moe_runner * r, const agx_cdm_stream * stream) {
    agx_cdm_stats st{};
    agx_cdm_get_stats(stream, &st);
    r->last_stream      = stream;
    r->last_stream_gen  = agx_cdm_stream_record_gen(stream);
    r->last_strong_bars = st.strong_usc_barriers + st.memory_host_barriers;
}

static const agx_kernel * get_bucket_kernel(agx_moe_runner * r, uint32_t mode) {
    if (!r->compiler || mode > 4u) return nullptr;
    const agx_kernel ** slot = (mode == 0u) ? &r->k_bucket_bitonic
                             : (mode == 1u) ? &r->k_bucket_histogram
                             : (mode == 2u) ? &r->k_zero_out
                             : (mode == 3u) ? &r->k_zero_skipped
                                            : &r->k_down_reduce;
    if (!*slot) {
        agx_compile_opts opts;
        opts.defines.push_back("BUCKET_MODE=" + std::to_string(mode));
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        const char * names[5] = {
            "moe_bitonic_bucket_sg32",
            "moe_histogram_bucket",
            "moe_zero_out",
            "moe_zero_skipped",
            "moe_down_reduce"
        };
        *slot = agx_compiler_compile_file(r->compiler, names[mode], "moe_bitonic_bucket.comp", opts);
    }
    return *slot;
}

static uint32_t moe_prefill_gemm_sgs(uint32_t gemm_tile_m) {
    if (gemm_tile_m == AGX_MOE_GEMM_TILE_M_SML) {
        return 4u;
    }
    static const uint32_t env_sgs = []() -> uint32_t {
        const char * v = std::getenv("GGML_AGX_MOE_SGS");
        if (v && v[0]) {
            const int parsed = std::atoi(v);
            return (parsed == 8) ? 8u : 4u;
        }
        // Wave 10 M4 Max silicon winner (task-8148: pp512 1,049 / pp1024 1,121 / pp2048 1,145 t/s!):
        // 128T (4-SG, 64N x 32M, exact 8,192 B SRAM -> 4 WGs/core, 2x WG parallelism, u0 = 68-72 GPR32).
        // Set GGML_AGX_MOE_SGS=8 to A/B 256T (8-SG).
        return 4u;
    }();
    return (gemm_tile_m == AGX_MOE_GEMM_TILE_M) ? env_sgs : 8u;
}

static const agx_kernel * get_moe_exec_kernel(agx_moe_runner *  r,
                                              uint32_t          stage,
                                              agx_weight_layout layout,
                                              uint32_t          batch_cap,
                                              bool              prec_f32    = false,
                                              uint32_t          gemm_tile_m = AGX_MOE_GEMM_TILE_M,
                                              bool              inter_f16   = false,
                                              bool              x_f16       = false,
                                              bool              merged_gu   = false) {
    if (!r->compiler) return nullptr;
    const uint32_t wm = static_cast<uint32_t>(layout);
    if (wm >= AGX_MOE_WEIGHT_MODES) return nullptr;

    const bool is_repacked = (layout == AGX_WEIGHT_NVFP4_REPACKED || layout == AGX_WEIGHT_Q8_0_REPACKED);
    const uint32_t bslot = (batch_cap <= 1u) ? 0u
                         : (batch_cap <= AGX_MOE_VECMAT_MAX_M) ? 1u : 2u;
    const uint32_t bcap  = (bslot == 0u) ? 1u
                         : (bslot == 1u) ? AGX_MOE_VECMAT_MAX_M : AGX_MOE_VECMAT_DSPARK_M;
    const bool use_f32   = prec_f32 && is_repacked && (stage == 1u || stage == 4u);
    const uint32_t eff_tile_m = (stage == 2u || stage == 3u)
        ? ((gemm_tile_m == AGX_MOE_GEMM_TILE_M_SML && is_repacked) ? AGX_MOE_GEMM_TILE_M_SML
         : (gemm_tile_m == AGX_MOE_GEMM_TILE_M_BIG && is_repacked) ? AGX_MOE_GEMM_TILE_M_BIG
         : (gemm_tile_m == AGX_MOE_GEMM_TILE_M_MID && is_repacked) ? AGX_MOE_GEMM_TILE_M_MID
                                                                   : AGX_MOE_GEMM_TILE_M)
        : AGX_MOE_GEMM_TILE_M;
    const uint32_t tslot = (eff_tile_m == AGX_MOE_GEMM_TILE_M_SML) ? 3u
                         : (eff_tile_m == AGX_MOE_GEMM_TILE_M_BIG) ? 2u
                         : (eff_tile_m == AGX_MOE_GEMM_TILE_M_MID) ? 1u : 0u;
    const bool use_if16  = inter_f16 && is_repacked && (stage <= 3u);
    const bool use_xf16  = x_f16     && (stage == 2u);
    const bool use_mrg   = merged_gu && is_repacked && (stage == 2u);
    const uint32_t if16  = use_if16 ? 1u : 0u;
    const uint32_t xf16  = use_xf16 ? 1u : 0u;
    const uint32_t mrg   = use_mrg  ? 1u : 0u;

    const agx_kernel ** slot = nullptr;
    if      (stage == 0u) slot = &r->k_gu_vecmat[wm][bslot][if16];
    else if (stage == 1u) slot = use_f32 ? &r->k_down_vecmat_f32[wm][bslot][if16] : &r->k_down_vecmat[wm][bslot][if16];
    else if (stage == 2u) slot = &r->k_gu_gemm[wm][tslot][if16][xf16][mrg];
    else if (stage == 3u) slot = &r->k_down_gemm[wm][tslot][if16];
    else if (stage == 4u) slot = use_f32 ? &r->k_direct_m1_f32[wm] : &r->k_direct_m1[wm];
    else if (stage == 5u) slot = &r->k_direct_gu_situ[wm];
    else if (stage == 6u) slot = &r->k_direct_down_wsum[wm];
    else return nullptr;

    if (!*slot) {
        static const bool no_k128 = []() {
            const char * v = std::getenv("GGML_AGX_MOE_NO_K128");
            return v && v[0] && v[0] != '0';
        }();
        static const int env_shx_pad = []() -> int {
            const char * v = std::getenv("GGML_AGX_MOE_SHX_PAD");
            if (v && v[0]) {
                return std::atoi(v);
            }
            return -1;
        }();
        static const int env_unroll_kp = []() -> int {
            const char * v = std::getenv("GGML_AGX_MOE_UNROLL_KP");
            if (v && v[0]) {
                return (v[0] != '0') ? 1 : 0;
            }
            return -1;
        }();
        static const bool m8_no_unroll_kp = []() {
            const char * v = std::getenv("GGML_AGX_MOE_M8_NO_UNROLL_KP");
            return v && v[0] && v[0] != '0';
        }();
        agx_compile_opts opts;
        opts.defines.push_back("PASS_STAGE=" + std::to_string(stage));
        opts.defines.push_back("WEIGHT_MODE=" + std::to_string(wm));
        if (stage <= 1u) {
            opts.defines.push_back("BATCH_CAP=" + std::to_string(bcap));
        }
        bool eff_unroll_kp = false;
        uint32_t eff_sgs = 8u;
        if (stage == 2u || stage == 3u) {
            eff_sgs = moe_prefill_gemm_sgs(eff_tile_m);
            const bool is_4sg_m32 = (eff_sgs == 4u && eff_tile_m == AGX_MOE_GEMM_TILE_M);
            const int eff_pad = (env_shx_pad >= 0) ? env_shx_pad : (is_4sg_m32 ? 0 : 2);
            if (eff_tile_m != AGX_MOE_GEMM_TILE_M) {
                opts.defines.push_back("GEMM_TILE_M=" + std::to_string(eff_tile_m));
            }
            if (eff_sgs != 8u) {
                opts.defines.push_back("MOE_GEMM_SGS=" + std::to_string(eff_sgs));
            }
            if (eff_tile_m == AGX_MOE_GEMM_TILE_M_SML) {
                eff_unroll_kp = !m8_no_unroll_kp;
                if (!eff_unroll_kp) {
                    opts.defines.push_back("MOE_UNROLL_KP=0");
                }
            } else {
                eff_unroll_kp = (env_unroll_kp > 0);
                if (eff_unroll_kp) {
                    opts.defines.push_back("MOE_UNROLL_KP=1");
                }
            }
            if (no_k128) {
                opts.defines.push_back("MOE_K_PAIRS=1");
            }
            if (eff_pad != 2) {
                opts.defines.push_back("MOE_SH_X_PAD=" + std::to_string(eff_pad));
            }
        }
        if (use_f32) {
            opts.defines.push_back("PREC_F32=1");
        }
        if (use_if16) {
            opts.defines.push_back("INTER_F16=1");
        }
        if (use_xf16) {
            opts.defines.push_back("MOE_X_F16=1");
        }
        if (use_mrg) {
            opts.defines.push_back("MERGED_GU=1");
        }
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        const std::string prefix = (stage == 2u || stage == 3u) ? "fused_moe_w10_s" : "fused_moe_s";
        const std::string kname = prefix + std::to_string(stage)
                                + "_w" + std::to_string(wm)
                                + (stage <= 1u ? ("_b" + std::to_string(bcap)) : "")
                                + (tslot == 3u ? "_m8" : (tslot == 1u ? "_m48" : (tslot == 2u ? "_m64" : "")))
                                + ((stage == 2u || stage == 3u) && eff_sgs == 4u && tslot == 0u ? "_sg4" : "")
                                + (use_f32 ? "_f32" : "")
                                + (use_if16 ? "_if16" : "")
                                + (use_xf16 ? "_xf16" : "")
                                + (use_mrg ? "_mrg" : "")
                                + (eff_unroll_kp ? "_unr" : "");
        *slot = agx_compiler_compile_file(
            r->compiler, kname.c_str(), "fused_moe_gate_up_situ_down.comp", opts);
    }
    return *slot;
}

agx_moe_runner * agx_moe_create(agx_drm_device * dev, agx_compiler * compiler) {
    if (!dev) {
        return nullptr;
    }
    auto * r = new agx_moe_runner();
    r->dev      = dev;
    r->compiler = compiler;
    if (!ensure_meta_bo(r, 512u) || !ensure_hidden_bo(r, 64u, 1152u)) {
        agx_moe_destroy(r);
        return nullptr;
    }
    return r;
}

void agx_moe_destroy(agx_moe_runner * r) {
    if (!r) return;
    if (r->meta_bo)   agx_bo_free(r->dev, r->meta_bo);
    if (r->hidden_bo) agx_bo_free(r->dev, r->hidden_bo);
    if (r->down_bo)   agx_bo_free(r->dev, r->down_bo);
    if (r->x_f16_bo)  agx_bo_free(r->dev, r->x_f16_bo);
    delete r;
}

bool agx_moe_precompile_inkling2(agx_moe_runner * r) {
    if (!r || !r->compiler) return false;
    bool ok = true;
    ok &= (get_bucket_kernel(r, 0u) != nullptr);
    ok &= (get_bucket_kernel(r, 1u) != nullptr);
    ok &= (get_bucket_kernel(r, 2u) != nullptr);
    ok &= (get_bucket_kernel(r, 3u) != nullptr);
    ok &= (get_bucket_kernel(r, 4u) != nullptr);
    ok &= (get_moe_cvt_f16_kernel(r) != nullptr);

    // Lean Wave 10 Inkling 2 Mini hot kernel set (NVFP4 blk 0..19 + Q8_0 blk 20..23, repacked):
    // covers B=1/M<=4 direct (s5/s6), M=5..8 direct-reg 8M MMA (s2_m8/s3_m8), and prefill 32M MMA
    // (s0/s1 tail + s2_m32 xf16/xf32 mrg + s3_m32 if16 + s4 MUL_MAT_ID).
    const agx_weight_layout layouts[] = {
        AGX_WEIGHT_NVFP4_REPACKED,
        AGX_WEIGHT_Q8_0_REPACKED,
    };
    for (agx_weight_layout wm : layouts) {
        ok &= (get_moe_exec_kernel(r, 5u, wm, 1u) != nullptr);
        ok &= (get_moe_exec_kernel(r, 6u, wm, 1u) != nullptr);
        ok &= (get_moe_exec_kernel(r, 4u, wm, 1u) != nullptr);
        ok &= (get_moe_exec_kernel(r, 0u, wm, AGX_MOE_VECMAT_MAX_M, false, AGX_MOE_GEMM_TILE_M, true) != nullptr);
        ok &= (get_moe_exec_kernel(r, 1u, wm, AGX_MOE_VECMAT_MAX_M, false, AGX_MOE_GEMM_TILE_M, true) != nullptr);
        // M=5..8 direct-register 8M MMA (128T, 32 ActSGs/core)
        ok &= (get_moe_exec_kernel(r, 2u, wm, AGX_MOE_VECMAT_MAX_M, false, AGX_MOE_GEMM_TILE_M_SML, true, false, true) != nullptr);
        ok &= (get_moe_exec_kernel(r, 3u, wm, AGX_MOE_VECMAT_MAX_M, false, AGX_MOE_GEMM_TILE_M_SML, true) != nullptr);
        // Prefill 32M MMA (default 128T 4-SG, exact 8,192 B SRAM, 68-72 GPR32, u0): xf16 + xf32 fallback, merged gate_up
        ok &= (get_moe_exec_kernel(r, 2u, wm, AGX_MOE_VECMAT_MAX_M, false, AGX_MOE_GEMM_TILE_M, true, true,  true) != nullptr);
        ok &= (get_moe_exec_kernel(r, 2u, wm, AGX_MOE_VECMAT_MAX_M, false, AGX_MOE_GEMM_TILE_M, true, false, true) != nullptr);
        ok &= (get_moe_exec_kernel(r, 3u, wm, AGX_MOE_VECMAT_MAX_M, false, AGX_MOE_GEMM_TILE_M, true) != nullptr);
    }
    return ok;
}

// ============================================================================
// Stage 5a: Bitonic / Histogram Bucketing Dispatch
// ============================================================================

static uint32_t moe_select_gemm_tile_m(uint32_t n_tokens, agx_weight_layout lay0, agx_weight_layout lay1) {
    const bool r0 = (lay0 == AGX_WEIGHT_NVFP4_REPACKED || lay0 == AGX_WEIGHT_Q8_0_REPACKED);
    const bool r1 = (lay1 == AGX_WEIGHT_NVFP4_REPACKED || lay1 == AGX_WEIGHT_Q8_0_REPACKED);
    if (!r0 || !r1) {
        return AGX_MOE_GEMM_TILE_M;
    }
    static const int env_tile_m = []() {
        const char * v = std::getenv("GGML_AGX_MOE_TILE_M");
        return (v && v[0]) ? std::atoi(v) : 0;
    }();
    if (env_tile_m == 8)  return AGX_MOE_GEMM_TILE_M_SML;
    if (env_tile_m == 32) return AGX_MOE_GEMM_TILE_M;
    if (env_tile_m == 48) return AGX_MOE_GEMM_TILE_M_MID;
    if (env_tile_m == 64) return AGX_MOE_GEMM_TILE_M_BIG;

    static const uint32_t m64_min = []() -> uint32_t {
        const char * v = std::getenv("GGML_AGX_MOE_TILE_M64_MIN");
        if (v && v[0]) {
            const int parsed = std::atoi(v);
            return parsed > 0 ? static_cast<uint32_t>(parsed) : 0xFFFFFFFFu;
        }
        return 0xFFFFFFFFu;
    }();
    static const uint32_t m48_min = []() -> uint32_t {
        const char * v = std::getenv("GGML_AGX_MOE_TILE_M48_MIN");
        if (v && v[0]) {
            const int parsed = std::atoi(v);
            return parsed > 0 ? static_cast<uint32_t>(parsed) : 0xFFFFFFFFu;
        }
        // Wave 10 default: 32M (68 GPR32, 24 ActSGs/core -> +32/+62 t/s over 48M/64M on M=1024/2048!)
        return 0xFFFFFFFFu;
    }();
    if (n_tokens >= m64_min) {
        return AGX_MOE_GEMM_TILE_M_BIG;
    }
    if (n_tokens >= m48_min) {
        return AGX_MOE_GEMM_TILE_M_MID;
    }
    return AGX_MOE_GEMM_TILE_M;
}

static bool dispatch_bucket_internal(agx_moe_runner *           r,
                                     agx_cdm_stream *           stream,
                                     const agx_moe_stage_args & args,
                                     uint32_t                   max_vecmat_m,
                                     uint32_t                   gemm_min_m,
                                     uint32_t                   gemm_tile_m,
                                     agx_barrier_mode           post_barrier) {
    if (!r || !stream || args.n_tokens == 0 || args.top_k == 0 || args.n_expert == 0) {
        return false;
    }
    const size_t total_assigns = static_cast<size_t>(args.n_tokens) * args.top_k;
    if (!ensure_meta_bo(r, total_assigns)) {
        return false;
    }

    // Protect shared runner->meta_bo across consecutive MoE dispatches when no barrier intervened!
    moe_entry_barrier(r, stream);

    const bool use_bitonic = (total_assigns <= 64u);
    const agx_kernel * k_bkt = get_bucket_kernel(r, use_bitonic ? 0u : 1u);
    if (!k_bkt) {
        return false;
    }

    const uint64_t base_va = r->meta_bo->gpu_va;
    agx_push_moe_bucket pb = {};
    pb.topk_ids_va        = args.topk_ids_va;
    pb.topk_w_va          = args.topk_w_va;
    pb.header_va          = base_va;
    pb.vecmat_buckets_va  = base_va + r->off_vecmat_buckets;
    pb.gemm_tiles_va      = base_va + r->off_gemm_tiles;
    pb.sorted_tok_va      = base_va + r->off_sorted_tok;
    pb.sorted_kslot_va    = base_va + r->off_sorted_kslot;
    pb.sorted_w_va        = base_va + r->off_sorted_w;
    pb.moe_out_va         = args.moe_out_va;
    pb.skipped_slots_va   = base_va + r->off_skipped_slots;
    pb.n_tokens           = args.n_tokens;
    pb.top_k              = args.top_k;
    pb.n_expert           = args.n_expert;
    pb.n_ff               = args.n_ff;
    pb.n_embd             = args.n_embd;
    const uint32_t gu_vm_rows = (args.gate_up_layout == AGX_WEIGHT_Q8_0_REPACKED) ? 4u : AGX_MOE_VECMAT_WG_PAIRS;
    const uint32_t dn_vm_rows = (args.down_layout    == AGX_WEIGHT_Q8_0_REPACKED) ? 8u : AGX_MOE_VECMAT_WG_OUT;
    const uint32_t gm_sgs     = moe_prefill_gemm_sgs(gemm_tile_m);
    const bool     is_4sg     = (gm_sgs == 4u);
    const uint32_t gu_gm_rows = is_4sg ? AGX_MOE_GEMM_WG_PAIRS_M8 : AGX_MOE_GEMM_WG_PAIRS;
    const uint32_t dn_gm_rows = is_4sg ? AGX_MOE_GEMM_WG_OUT_M8   : AGX_MOE_GEMM_WG_OUT;
    pb.wg_gate_up_vecmat  = (args.n_ff   + gu_vm_rows - 1u) / gu_vm_rows;
    pb.wg_down_vecmat     = (args.n_embd + dn_vm_rows - 1u) / dn_vm_rows;
    pb.wg_gate_up_gemm    = (args.n_ff   + gu_gm_rows - 1u) / gu_gm_rows;
    pb.wg_down_gemm       = (args.n_embd + dn_gm_rows - 1u) / dn_gm_rows;
    pb.max_vecmat_buckets = r->max_vecmat_buckets;
    pb.max_gemm_tiles     = r->max_gemm_tiles;
    pb.max_vecmat_m       = max_vecmat_m;
    pb.gemm_min_m         = gemm_min_m;
    pb.exp_scale0         = (args.exp_scale0 != 0.0f) ? args.exp_scale0 : 1.0f;
    pb.zero_elems         = args.n_tokens * args.n_embd;
    pb.ids_row_stride     = args.ids_row_stride ? args.ids_row_stride : args.top_k;
    pb.gemm_tile_m        = gemm_tile_m;
    pb.gemm_local_x       = gm_sgs * 32u;

    if (!agx_cdm_dispatch(stream, k_bkt, 1u, 1u, 1u, &pb, sizeof(pb), post_barrier)) {
        return false;
    }
    moe_exit_record(r, stream);
    return true;
}

static bool moe_use_dspark_bcap8(uint32_t n_tokens, agx_weight_layout lay0, agx_weight_layout lay1) {
    const bool r0 = (lay0 == AGX_WEIGHT_NVFP4_REPACKED || lay0 == AGX_WEIGHT_Q8_0_REPACKED);
    const bool r1 = (lay1 == AGX_WEIGHT_NVFP4_REPACKED || lay1 == AGX_WEIGHT_Q8_0_REPACKED);
    return (n_tokens > AGX_MOE_VECMAT_MAX_M) && (n_tokens <= AGX_MOE_VECMAT_DSPARK_M) && r0 && r1;
}

bool agx_moe_dispatch_bucket(agx_moe_runner *           r,
                             agx_cdm_stream *           stream,
                             const agx_moe_stage_args & args,
                             agx_barrier_mode           post_barrier) {
    const bool use_b8      = moe_use_dspark_bcap8(args.n_tokens, args.gate_up_layout, args.down_layout);
    const uint32_t max_vm  = (args.n_tokens == 1u) ? 1u
                           : use_b8 ? AGX_MOE_VECMAT_DSPARK_M : AGX_MOE_VECMAT_MAX_M;
    const uint32_t gm_min  = (!use_b8 && args.n_tokens >= AGX_MOE_GEMM_MIN_M) ? AGX_MOE_GEMM_MIN_M : 0xFFFFFFFFu;
    const uint32_t tile_m  = (gm_min != 0xFFFFFFFFu)
        ? moe_select_gemm_tile_m(args.n_tokens, args.gate_up_layout, args.down_layout)
        : AGX_MOE_GEMM_TILE_M;
    return dispatch_bucket_internal(r, stream, args, max_vm, gm_min, tile_m, post_barrier);
}

// ============================================================================
// Wave 4/9 Direct 2-Stage MoE (`PASS_STAGE=5 -> PASS_STAGE=6`, M = 1..8):
//   - Strictly 2 dispatches + 1 STRONG_USC barrier (zero bucketing, zero zero-fill,
//     zero indirect grids, zero atomics, 0-ULP sequential FP32 wsum!).
//   - M == 1: exact original grid `(grid_x, top_k, 1)` -> 0.0% B=1 tg128 change!
//   - M = 2..8: SWAP_XZ grid `(M, top_k, grid_x)` / `(M, 1, grid_x)` so all M tokens'
//     slices for the same 16-row tile (40.5 KiB/expert) launch back-to-back ->
//     100% L2/SLC hit on every repeated expert (~2.9 at M=4, ~12.4 at M=8)!
// ============================================================================

static bool try_dispatch_direct_s56(agx_moe_runner *           r,
                                    agx_cdm_stream *           stream,
                                    const agx_moe_stage_args & args,
                                    agx_barrier_mode           post_barrier,
                                    bool *                     out_ok) {
    const bool nvfp4_merged_ok =
        (args.gate_up_va == 0) ||
        (args.gate_up_layout != AGX_WEIGHT_NVFP4_REPACKED) ||
        ((args.n_ff & 15u) == 0u);

    static const bool disable_direct = []() {
        const char * v = std::getenv("GGML_AGX_MOE_NO_DIRECT_M1");
        return v && v[0] && v[0] != '0';
    }();
    static const uint32_t s56_max_m = []() -> uint32_t {
        const char * v = std::getenv("GGML_AGX_MOE_S56_MAX_M");
        if (v && v[0]) {
            const int parsed = std::atoi(v);
            return parsed > 0 ? static_cast<uint32_t>(std::min(parsed, 8)) : 0u;
        }
        // Wave 10 default: M=1..4 on direct s5/s6 (B=1 tg128 100% untouched!); M=5..8 routes to
        // deduplicated 8M direct-register 128T MMA (32 ActSGs/core, -19.4% DRAM weight BW!).
        // Set GGML_AGX_MOE_S56_MAX_M=8 to A/B s5/s6 on M=5..8, or =1 to test 8M MMA on M=4!
        return 4u;
    }();
    static const bool no_swap_xz = []() {
        const char * v = std::getenv("GGML_AGX_MOE_S56_NO_SWAP_XZ");
        return v && v[0] && v[0] != '0';
    }();

    const bool accum_conflict = args.accumulate_out && args.res_va != 0 && args.res_va != args.moe_out_va;
    if (disable_direct || args.n_tokens == 0u || args.n_tokens > s56_max_m ||
        args.x_is_f16 || accum_conflict || !nvfp4_merged_ok) {
        return false;
    }

    const size_t total_assigns = static_cast<size_t>(args.n_tokens) * args.top_k;
    const size_t inter_bytes   = total_assigns * static_cast<size_t>(args.n_ff) * sizeof(float);
    uint64_t inter_va = 0;
    if (args.moe_inter_va != 0 && inter_bytes <= AGX_SLC_ARENA_MOE_INTER_BYTES) {
        inter_va = args.moe_inter_va;
    } else {
        // Protect shared runner->hidden_bo across consecutive calls when no barrier intervened!
        moe_entry_barrier(r, stream);
        if (!ensure_hidden_bo(r, total_assigns, args.n_ff)) {
            *out_ok = false;
            return true;
        }
        inter_va = r->hidden_bo->gpu_va;
    }

    const agx_kernel * k_gu = get_moe_exec_kernel(r, 5u, args.gate_up_layout, 1u);
    const agx_kernel * k_dn = get_moe_exec_kernel(r, 6u, args.down_layout,    1u);
    if (!k_gu || !k_dn) {
        *out_ok = false;
        return true;
    }

    const bool swap_xz = (args.n_tokens >= 2u) && !no_swap_xz;
    uint32_t act_flags = 0u;
    if (args.act_op == AGX_MOE_ACT_SILU) act_flags |= AGX_MOE_FLAG_ACT_SILU;
    if (args.act_op == AGX_MOE_ACT_GELU) act_flags |= AGX_MOE_FLAG_ACT_GELU;
    if (args.gate_up_va == 0)            act_flags |= AGX_MOE_FLAG_SEPARATE_GU;
    if (swap_xz)                         act_flags |= AGX_MOE_FLAG_SWAP_XZ;

    const uint32_t ids_stride = args.ids_row_stride ? args.ids_row_stride : args.top_k;

    // Stage 2 (PASS_STAGE=5): Direct Fused Gate+Up + In-Register SiTU -> SLC moe_inter
    agx_push_moe_exec p_gu = {};
    p_gu.buckets_va       = args.topk_ids_va;
    p_gu.x_va             = args.ffn_x_va;
    p_gu.w_gate_va        = (args.gate_up_va != 0) ? args.gate_up_va : args.gate_va;
    p_gu.w_up_va          = (args.gate_up_va != 0) ? args.gate_up_va : args.up_va;
    p_gu.y_va             = inter_va;
    p_gu.w_slice_B        = args.gate_up_slice_B;
    p_gu.K                = args.n_embd;
    p_gu.N                = args.n_ff;
    p_gu.n_ff             = args.n_expert;
    p_gu.top_k            = args.top_k;
    p_gu.flags            = act_flags;
    p_gu.situ_beta        = args.situ_beta;
    p_gu.situ_linear_beta = args.situ_linear_beta;
    p_gu.w_scale_va       = args.gate_up_scale_va;
    p_gu.ids_row_stride   = ids_stride;

    const uint32_t gu_pairs_wg = (args.gate_up_layout == AGX_WEIGHT_NVFP4_REPACKED) ? 16u
                               : (args.gate_up_layout == AGX_WEIGHT_Q8_0_REPACKED)  ? 8u
                                                                                    : 4u;
    const uint32_t gu_grid_x = (args.n_ff + gu_pairs_wg - 1u) / gu_pairs_wg;
    const uint32_t gu_gx     = swap_xz ? args.n_tokens : gu_grid_x;
    const uint32_t gu_gz     = swap_xz ? gu_grid_x     : args.n_tokens;
    if (!agx_cdm_dispatch(stream, k_gu, gu_gx, args.top_k, gu_gz,
                          &p_gu, sizeof(p_gu), AGX_BARRIER_STRONG_USC)) {
        *out_ok = false;
        return true;
    }

    // Stage 3 (PASS_STAGE=6): Direct Fused Down + Exact Sequential Weighted-Sum [+ res_va] -> moe_out
    agx_push_moe_exec p_dn = {};
    p_dn.buckets_va       = args.topk_ids_va;
    p_dn.sorted_w_va      = args.topk_w_va;
    p_dn.x_va             = inter_va;
    p_dn.w_gate_va        = args.down_va;
    p_dn.w_up_va          = args.accumulate_out ? args.moe_out_va : args.res_va;
    p_dn.y_va             = args.moe_out_va;
    p_dn.w_slice_B        = args.down_slice_B;
    p_dn.K                = args.n_ff;
    p_dn.N                = args.n_embd;
    p_dn.n_ff             = args.n_expert;
    p_dn.top_k            = args.top_k;
    p_dn.flags            = swap_xz ? AGX_MOE_FLAG_SWAP_XZ : 0u;
    p_dn.situ_beta        = (args.exp_scale0 != 0.0f) ? args.exp_scale0 : 1.0f;
    p_dn.ids_row_stride   = ids_stride;
    p_dn.w_scale_va       = args.down_scale_va;

    const uint32_t dn_rows_wg = (args.down_layout == AGX_WEIGHT_NVFP4_REPACKED) ? 16u
                              : (args.down_layout == AGX_WEIGHT_Q8_0_REPACKED)  ? 2u
                                                                                : 1u;
    const uint32_t dn_grid_x = (args.n_embd + dn_rows_wg - 1u) / dn_rows_wg;
    const uint32_t dn_gx     = swap_xz ? args.n_tokens : dn_grid_x;
    const uint32_t dn_gz     = swap_xz ? dn_grid_x     : args.n_tokens;
    const bool ok = agx_cdm_dispatch(stream, k_dn, dn_gx, 1u, dn_gz,
                                     &p_dn, sizeof(p_dn), post_barrier);
    if (ok && inter_va != args.moe_inter_va) {
        moe_exit_record(r, stream);
    }
    *out_ok = ok;
    return true;
}

// ============================================================================
// Full Pillar 5 Dispatch: Zero CPU Sync, Indirect Vec-Mat + 8x8 MMA Crossover
// ============================================================================

bool agx_moe_dispatch(agx_moe_runner *           r,
                      agx_cdm_stream *           stream,
                      const agx_moe_stage_args & args,
                      agx_barrier_mode           post_barrier) {
    if (!r || !stream || args.n_tokens == 0 || args.n_ff == 0 || args.n_embd == 0) {
        return false;
    }

    // Wave 9: Route M=1..8 through zero-bucket, zero-atomic s5/s6 + SWAP_XZ!
    {
        bool direct_ok = false;
        if (try_dispatch_direct_s56(r, stream, args, post_barrier, &direct_ok)) {
            return direct_ok;
        }
    }

    const size_t total_assigns = static_cast<size_t>(args.n_tokens) * args.top_k;
    if (!ensure_meta_bo(r, total_assigns)) {
        return false;
    }

    const bool is_repacked_gu = (args.gate_up_layout == AGX_WEIGHT_NVFP4_REPACKED ||
                                 args.gate_up_layout == AGX_WEIGHT_Q8_0_REPACKED);
    const bool is_repacked_dn = (args.down_layout == AGX_WEIGHT_NVFP4_REPACKED ||
                                 args.down_layout == AGX_WEIGHT_Q8_0_REPACKED);

    static const bool no_inter_f16 = (std::getenv("GGML_AGX_MOE_NO_INTER_F16") != nullptr);
    static const bool no_x_f16 = []() {
        const char * v = std::getenv("GGML_AGX_MOE_NO_X_F16");
        return v && v[0] && v[0] != '0';
    }();
    static const bool no_m8_mma = []() {
        const char * v = std::getenv("GGML_AGX_MOE_NO_M8_MMA");
        return v && v[0] && v[0] != '0';
    }();
    static const bool b8_noscat = []() {
        const char * v = std::getenv("GGML_AGX_MOE_B8_NOSCAT");
        return v && v[0] && v[0] != '0';
    }();

    // Wave 10 M=2..8 deduplicated 8M direct-register 128T MMA fastpath (used for M=5..8 by default,
    // or M=2..4 when GGML_AGX_MOE_S56_MAX_M=1): 32 ActSGs/core, -19.4% DRAM weight BW vs s5/s6,
    // skips s0/s1 vecmat, zero atomics via DOWN_NOSCATTER + moe_down_reduce!
    const bool use_m8_mma = !no_m8_mma &&
                            (args.n_tokens >= 2u && args.n_tokens <= 8u) &&
                            is_repacked_gu && is_repacked_dn &&
                            ((args.n_embd & 3u) == 0u) && ((args.n_ff & 3u) == 0u) &&
                            (args.gate_up_layout != AGX_WEIGHT_NVFP4_REPACKED ||
                             (args.n_embd >= 64u && (args.n_embd & 63u) == 0u)) &&
                            (args.down_layout != AGX_WEIGHT_NVFP4_REPACKED ||
                             (args.n_ff >= 64u && (args.n_ff & 63u) == 0u));

    const bool     use_b8         = !use_m8_mma && moe_use_dspark_bcap8(args.n_tokens, args.gate_up_layout, args.down_layout);
    const uint32_t bcap           = (args.n_tokens == 1u) ? 1u
                                  : use_b8 ? AGX_MOE_VECMAT_DSPARK_M : AGX_MOE_VECMAT_MAX_M;
    const bool     may_have_gemm  = use_m8_mma || (!use_b8 && args.n_tokens >= AGX_MOE_GEMM_MIN_M);
    const uint32_t gemm_min_m     = use_m8_mma ? 1u : (may_have_gemm ? AGX_MOE_GEMM_MIN_M : 0xFFFFFFFFu);
    const uint32_t gemm_tile_m    = use_m8_mma ? AGX_MOE_GEMM_TILE_M_SML
                                  : may_have_gemm
                                      ? moe_select_gemm_tile_m(args.n_tokens, args.gate_up_layout, args.down_layout)
                                      : AGX_MOE_GEMM_TILE_M;
    const bool     skip_vecmat    = use_m8_mma;

    // Store internal hidden_bo as SLC-pinned FP16 on prefill & 8M MMA (36 MiB <= 40 MiB on pp2048 -> 100% SLC hit!)
    const bool inter_f16 = (args.moe_inter_va == 0) && may_have_gemm &&
                           is_repacked_gu && is_repacked_dn && !no_inter_f16;

    uint64_t hidden_va = args.moe_inter_va;
    if (hidden_va == 0) {
        if (!ensure_hidden_bo(r, total_assigns, args.n_ff, inter_f16)) {
            return false;
        }
        hidden_va = r->hidden_bo->gpu_va;
    }

    // Eliminate atomicAddF32 CAS contention & moe_zero_out on prefill & 8M MMA!
    const bool use_down_noscat = (may_have_gemm || (use_b8 && b8_noscat)) &&
                                 ensure_down_bo(r, total_assigns, args.n_embd);

    // Wave 10 prefill X F32->F16 pre-conversion for PASS_STAGE=2 (overlaps with bucket histogram!):
    // halves scattered X gather BW & eliminates F32->F16 ALU inside s2 while keeping s0 tail on F32.
    const bool want_cvt_x_f16 = !no_x_f16 && !args.x_is_f16 &&
                                may_have_gemm && (gemm_tile_m >= AGX_MOE_GEMM_TILE_M) &&
                                (args.n_tokens >= 64u) && is_repacked_gu && ((args.n_embd & 3u) == 0u);
    bool cvt_x_f16 = false;
    if (want_cvt_x_f16) {
        const size_t x_f16_B = static_cast<size_t>(args.n_tokens) * args.n_embd * sizeof(uint16_t);
        if (get_moe_cvt_f16_kernel(r) && ensure_x_f16_bo(r, x_f16_B)) {
            cvt_x_f16 = true;
        }
    }
    const bool s2_x_f16 = args.x_is_f16 || cvt_x_f16;

    // 1. GPU-resident bucketing (bitonic for M<=8, histogram for M>8)
    //    Overlaps with optional X F32->F16 conversion and/or moe_out zero-fill via AGX_BARRIER_NONE!
    const bool need_init_out = !use_down_noscat && !args.accumulate_out &&
                               (args.res_va == 0 || args.res_va != args.moe_out_va);
    const agx_barrier_mode bkt_bar = (cvt_x_f16 || need_init_out) ? AGX_BARRIER_NONE : AGX_BARRIER_STRONG_USC;
    if (!dispatch_bucket_internal(r, stream, args, bcap, gemm_min_m, gemm_tile_m, bkt_bar)) {
        return false;
    }

    const uint64_t base_va = r->meta_bo->gpu_va;

    if (cvt_x_f16) {
        const agx_kernel * k_cvt = get_moe_cvt_f16_kernel(r);
        if (!k_cvt) return false;
        agx_push_moe_cvt_f16 pcvt = {};
        pcvt.src_va = args.ffn_x_va;
        pcvt.dst_va = r->x_f16_bo->gpu_va;
        pcvt.n_vec4 = (args.n_tokens * args.n_embd) >> 2u;
        const uint32_t cvt_gx = (pcvt.n_vec4 + 127u) / 128u;
        if (!agx_cdm_dispatch(stream, k_cvt, cvt_gx, 1u, 1u, &pcvt, sizeof(pcvt),
                              need_init_out ? AGX_BARRIER_NONE : AGX_BARRIER_STRONG_USC)) {
            return false;
        }
    }

    if (need_init_out) {
        const agx_kernel * k_zero = get_bucket_kernel(r, 2u);
        if (!k_zero) return false;
        agx_push_moe_bucket pz = {};
        pz.topk_w_va  = args.res_va; // 0 = zero-fill; non-zero = copy residual res_va -> moe_out_va
        pz.moe_out_va = args.moe_out_va;
        pz.zero_elems = args.n_tokens * args.n_embd;
        const uint32_t zero_work = ((pz.zero_elems & 3u) == 0u) ? (pz.zero_elems >> 2) : pz.zero_elems;
        const uint32_t wg_zero   = (zero_work + 127u) / 128u;
        if (!agx_cdm_dispatch(stream, k_zero, wg_zero, 1u, 1u,
                              &pz, sizeof(pz), AGX_BARRIER_STRONG_USC)) {
            return false;
        }
    }

    const bool merged_gu = (args.gate_up_va != 0);
    uint32_t act_flags = 0u;
    if (args.act_op == AGX_MOE_ACT_SILU) act_flags |= AGX_MOE_FLAG_ACT_SILU;
    if (args.act_op == AGX_MOE_ACT_GELU) act_flags |= AGX_MOE_FLAG_ACT_GELU;
    if (!merged_gu)                      act_flags |= AGX_MOE_FLAG_SEPARATE_GU;
    if (args.x_is_f16)                   act_flags |= AGX_MOE_FLAG_X_F16;
    if (inter_f16)                       act_flags |= AGX_MOE_FLAG_INTER_F16;

    // 2. Stage A: Fused Gate+Up + Exact SiTU (writes SLC-resident moe_inter / hidden_bo)
    //    - Vec-mat indirect dispatch for buckets with M_bucket < gemm_min_m (skipped on 8M MMA!)
    //    - 8x8 MMA GEMM indirect dispatch for buckets with M_bucket >= gemm_min_m
    {
        agx_push_moe_exec p_gu = {};
        p_gu.header_va        = base_va;
        p_gu.buckets_va       = base_va + r->off_vecmat_buckets;
        p_gu.sorted_tok_va    = base_va + r->off_sorted_tok;
        p_gu.sorted_kslot_va  = base_va + r->off_sorted_kslot;
        p_gu.sorted_w_va      = base_va + r->off_sorted_w;
        p_gu.x_va             = args.ffn_x_va;
        p_gu.w_gate_va        = merged_gu ? args.gate_up_va : args.gate_va;
        p_gu.w_up_va          = merged_gu ? args.gate_up_va : args.up_va;
        p_gu.y_va             = hidden_va;
        p_gu.w_slice_B        = args.gate_up_slice_B;
        p_gu.K                = args.n_embd;
        p_gu.N                = args.n_ff;
        p_gu.n_ff             = args.n_ff;
        p_gu.top_k            = args.top_k;
        p_gu.flags            = act_flags;
        p_gu.situ_beta        = args.situ_beta;
        p_gu.situ_linear_beta = args.situ_linear_beta;
        p_gu.w_scale_va       = args.gate_up_scale_va;

        if (!skip_vecmat) {
            const agx_kernel * k_gu_vm = get_moe_exec_kernel(r, 0u, args.gate_up_layout, bcap, false, AGX_MOE_GEMM_TILE_M, inter_f16);
            if (!k_gu_vm) return false;
            // If GEMM path will also run on disjoint expert buckets, elide barrier between vecmat & GEMM!
            const agx_barrier_mode gu_vm_bar = may_have_gemm ? AGX_BARRIER_NONE : AGX_BARRIER_STRONG_USC;
            if (!agx_cdm_dispatch_indirect(
                    stream, k_gu_vm,
                    base_va + offsetof(agx_moe_bucket_header, grid_gate_up_vecmat),
                    &p_gu, sizeof(p_gu), gu_vm_bar)) {
                return false;
            }
        }

        if (may_have_gemm) {
            const agx_kernel * k_gu_gm = get_moe_exec_kernel(
                r, 2u, args.gate_up_layout, bcap, false, gemm_tile_m, inter_f16, s2_x_f16, merged_gu);
            if (!k_gu_gm) return false;
            p_gu.buckets_va = base_va + r->off_gemm_tiles;
            if (cvt_x_f16) {
                p_gu.x_va   = r->x_f16_bo->gpu_va;
                p_gu.flags |= AGX_MOE_FLAG_X_F16;
            }
            if (!agx_cdm_dispatch_indirect(
                    stream, k_gu_gm,
                    base_va + offsetof(agx_moe_bucket_header, grid_gate_up_gemm),
                    &p_gu, sizeof(p_gu), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }
    }

    // 3. Stage B: Down Projection + Weighted Sum (zero-atomic reduce on prefill & 8M MMA)
    {
        const uint32_t dn_base_flags =
            (use_down_noscat ? AGX_MOE_FLAG_DOWN_NOSCATTER : 0u) |
            (inter_f16       ? AGX_MOE_FLAG_X_F16          : 0u);

        agx_push_moe_exec p_dn = {};
        p_dn.header_va        = base_va;
        p_dn.buckets_va       = base_va + r->off_vecmat_buckets;
        p_dn.sorted_tok_va    = base_va + r->off_sorted_tok;
        p_dn.sorted_kslot_va  = base_va + r->off_sorted_kslot;
        p_dn.sorted_w_va      = base_va + r->off_sorted_w;
        p_dn.x_va             = hidden_va;
        p_dn.w_gate_va        = args.down_va;
        p_dn.w_up_va          = 0;
        p_dn.y_va             = use_down_noscat ? r->down_bo->gpu_va : args.moe_out_va;
        p_dn.w_slice_B        = args.down_slice_B;
        p_dn.K                = args.n_ff;
        p_dn.N                = args.n_embd;
        p_dn.n_ff             = args.n_ff;
        p_dn.top_k            = args.top_k;
        p_dn.flags            = dn_base_flags;
        p_dn.w_scale_va       = args.down_scale_va;

        // Vecmat and GEMM write disjoint slots (noscat) or scatter atomically -> overlap with AGX_BARRIER_NONE!
        const agx_barrier_mode dn_end_bar = use_down_noscat ? AGX_BARRIER_STRONG_USC : post_barrier;
        if (!skip_vecmat) {
            const agx_kernel * k_dn_vm = get_moe_exec_kernel(r, 1u, args.down_layout, bcap, false, AGX_MOE_GEMM_TILE_M, inter_f16);
            if (!k_dn_vm) return false;
            const agx_barrier_mode dn_vm_bar = may_have_gemm ? AGX_BARRIER_NONE : dn_end_bar;
            if (!agx_cdm_dispatch_indirect(
                    stream, k_dn_vm,
                    base_va + offsetof(agx_moe_bucket_header, grid_down_vecmat),
                    &p_dn, sizeof(p_dn), dn_vm_bar)) {
                return false;
            }
        }

        if (may_have_gemm) {
            const agx_kernel * k_dn_gm = get_moe_exec_kernel(r, 3u, args.down_layout, bcap, false, gemm_tile_m, inter_f16);
            if (!k_dn_gm) return false;
            p_dn.buckets_va = base_va + r->off_gemm_tiles;
            if (!agx_cdm_dispatch_indirect(
                    stream, k_dn_gm,
                    base_va + offsetof(agx_moe_bucket_header, grid_down_gemm),
                    &p_dn, sizeof(p_dn), dn_end_bar)) {
                return false;
            }
        }

        if (use_down_noscat) {
            const agx_kernel * k_red = get_bucket_kernel(r, 4u);
            if (!k_red) return false;
            agx_push_moe_bucket pr = {};
            pr.topk_ids_va       = args.topk_ids_va;
            pr.topk_w_va         = args.topk_w_va;
            pr.vecmat_buckets_va = r->down_bo->gpu_va;
            pr.gemm_tiles_va     = args.accumulate_out ? args.moe_out_va : args.res_va;
            pr.moe_out_va        = args.moe_out_va;
            pr.n_tokens          = args.n_tokens;
            pr.top_k             = args.top_k;
            pr.n_expert          = args.n_expert;
            pr.n_embd            = args.n_embd;
            pr.exp_scale0        = (args.exp_scale0 != 0.0f) ? args.exp_scale0 : 1.0f;
            pr.ids_row_stride    = args.ids_row_stride ? args.ids_row_stride : args.top_k;
            const uint32_t red_work = ((args.n_embd & 3u) == 0u) ? (args.n_embd >> 2) : args.n_embd;
            const uint32_t red_gx   = (red_work + 127u) / 128u;
            if (!agx_cdm_dispatch(stream, k_red, red_gx, args.n_tokens, 1u,
                                  &pr, sizeof(pr), post_barrier)) {
                return false;
            }
        }
    }

    moe_exit_record(r, stream);
    return true;
}

// ============================================================================
// Wave 4 Theorem W4.3: Fused 2-Stage MoE (`gate_up + SiTU -> SLC moe_inter
//                      -> down + precise-float wsum [+ res_va]`)
// ============================================================================

bool agx_moe_dispatch_fused_situ_down(agx_moe_runner *           r,
                                      agx_cdm_stream *           stream,
                                      const agx_moe_stage_args & args,
                                      agx_barrier_mode           post_barrier) {
    if (!r || !stream || args.n_tokens == 0 || args.n_ff == 0 || args.n_embd == 0 ||
        args.top_k == 0 || args.n_expert == 0) {
        return false;
    }

    // Direct 2-Stage Fastpath for M=1..8 (strictly 2 dispatches + 1 STRONG_USC barrier,
    // zero bucketing, zero zero-fill, zero atomics, 0-ULP sequential wsum, SWAP_XZ SLC reuse!):
    bool direct_ok = false;
    if (try_dispatch_direct_s56(r, stream, args, post_barrier, &direct_ok)) {
        return direct_ok;
    }

    // M > s56_max_m (prefill): deduplicated histogram bucketing + lean BATCH_CAP=4 vecmat
    // + K_PAIRS=2 8x8 MMA GEMM (32M/48M/64M) with SLC-pinned FP16 hidden_bo & zero-atomic reduce!
    return agx_moe_dispatch(r, stream, args, post_barrier);
}

// ============================================================================
// Generic GGML_OP_MUL_MAT_ID Bridge (for agx-graph-exec.cpp / test-backend-ops)
// ============================================================================

bool agx_moe_supports_mul_mat_id(const struct ggml_tensor * dst) {
    if (!dst || dst->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    const ggml_tensor * w   = dst->src[0]; // [K, N, n_expert]
    const ggml_tensor * x   = dst->src[1]; // [K, n_x_slots (1 or top_k), M]
    const ggml_tensor * ids = dst->src[2]; // [top_k, M] I32
    if (!w || !x || !ids) return false;
    if (ids->type != GGML_TYPE_I32 || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // Support non-contiguous 2D ids views (e.g. ffn_moe_topk view of 128-wide ARGSORT!)
    if (ids->nb[0] != sizeof(int32_t) || (ids->nb[1] % sizeof(int32_t)) != 0 ||
        ids->ne[2] != 1 || ids->ne[3] != 1) {
        return false;
    }
    if (w->nb[0] != ggml_type_size(w->type) || w->ne[2] <= 0 || w->ne[2] > AGX_MOE_MAX_EXPERTS) {
        return false;
    }
    // Support BOTH shared input (x->ne[1] == 1: gate_up) AND per-slot input (x->ne[1] == top_k: down)!
    if (x->ne[1] != 1 && x->ne[1] != ids->ne[0]) {
        return false;
    }
    agx_weight_layout layout;
    if (!agx_gemm_infer_weight_layout(w, &layout)) {
        return false;
    }
    // Quantized formats require K multiple of 4 (and block size already verified by agx_gemm_infer_weight_layout)
    if ((w->ne[0] & 3) != 0 &&
        layout != AGX_WEIGHT_F32 && layout != AGX_WEIGHT_F16 && layout != AGX_WEIGHT_BF16) {
        return false;
    }
    return true;
}

bool agx_moe_dispatch_mul_mat_id(agx_moe_runner *           r,
                                 agx_cdm_stream *           stream,
                                 const struct ggml_tensor * dst,
                                 agx_barrier_mode           post_barrier) {
    if (!r || !stream || !agx_moe_supports_mul_mat_id(dst)) {
        return false;
    }
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];

    agx_weight_layout layout;
    if (!agx_gemm_infer_weight_layout(w, &layout)) {
        return false;
    }

    const uint32_t K        = static_cast<uint32_t>(w->ne[0]);
    const uint32_t N        = static_cast<uint32_t>(w->ne[1]);
    const uint32_t n_expert = static_cast<uint32_t>(w->ne[2]);
    const uint32_t top_k    = static_cast<uint32_t>(ids->ne[0]);
    const uint32_t M        = static_cast<uint32_t>(ids->ne[1]);
    const bool     per_slot = (x->ne[1] == top_k && top_k > 1u);

    const uint64_t slice_B = agx_tensor_is_repacked(w)
        ? static_cast<uint64_t>(agx_repack_slice_stride(w->type, w->ne[0], w->ne[1]))
        : static_cast<uint64_t>(w->nb[2]);

    // Force pure FP32 decode when F32 or BF16 src1 dynamic range is requested (avoids FP16
    // overflow on MUL_MAT_ID_PREC_SRC1_{BF16,F32}!).
    const bool req_prec_f32 =
        (ggml_prec_get_src(dst, 1) == GGML_PREC_BF16) ||
        (ggml_prec_get_src(dst, 1) == GGML_PREC_F32)  ||
        (ggml_prec_get_src(dst, 0) == GGML_PREC_F32);

    // ------------------------------------------------------------------------
    // M = 1..8 Direct Roofline Fastpath (B=1 decode + M=2..8 DSpark verify!):
    //   - ZERO bucketing, ZERO dst zero-fill, ZERO indirect grid, ZERO meta_bo barrier!
    //   - Eliminates 3 dispatches + 2 STRONG_USC barriers per MoE op across all 96 ops!
    //   - NVFP4_32x16: 64 rows/WG SWAR2 ffma16 + 2x K-unroll (<=56 GPR32, 0:0 spills)
    //   - Q8_0_8BLK:   8 rows/WG 16B uvec4 coalesced + 2x chunk-unroll (515 GB/s layout)
    //   - Exact eid < 0 / OOB -> 0.0f skip guard in-kernel; grid_z = M, grid_y = top_k.
    // ------------------------------------------------------------------------
    if (M <= AGX_MOE_DIRECT_MAX_M) {
        const agx_kernel * k_direct = get_moe_exec_kernel(r, 4u, layout, 1u, req_prec_f32);
        if (!k_direct) return false;

        agx_push_moe_exec p = {};
        p.buckets_va     = agx_tensor_gpu_va(ids);
        p.x_va           = agx_tensor_gpu_va(x);
        p.w_gate_va      = agx_tensor_gpu_va(w);
        p.y_va           = agx_tensor_gpu_va(dst);
        p.w_slice_B      = slice_B;
        p.K              = K;
        p.N              = N;
        p.n_ff           = n_expert; // n_expert bounds check for negative/OOB IDs
        p.top_k          = top_k;
        p.flags          = AGX_MOE_FLAG_RAW_MMID | (per_slot ? AGX_MOE_FLAG_RAW_MMID_PERSLOT : 0u);
        p.ids_row_stride = static_cast<uint32_t>(ids->nb[1] / sizeof(int32_t));

        const uint32_t rows_per_wg = (layout == AGX_WEIGHT_NVFP4_REPACKED) ? 64u
                                   : (layout == AGX_WEIGHT_Q8_0_REPACKED)  ? 8u
                                                                           : 4u;
        const uint32_t grid_x = (N + rows_per_wg - 1u) / rows_per_wg;
        return agx_cdm_dispatch(stream, k_direct, grid_x, top_k, M, &p, sizeof(p), post_barrier);
    }

    const bool is_repacked_quant =
        (layout == AGX_WEIGHT_NVFP4_REPACKED) || (layout == AGX_WEIGHT_Q8_0_REPACKED);
    // Wave 5 DSpark M=5..8 fastpath: route ALL cnt=1..8 buckets into single-pass sh_x BATCH_CAP=8
    // vec-mat (0:0 spills, <=56 GPR32, 1x weight DRAM read, ZERO GEMM dispatches!)
    const bool use_dspark_b8 =
        !req_prec_f32 && is_repacked_quant &&
        (M > AGX_MOE_VECMAT_MAX_M) && (M <= AGX_MOE_VECMAT_DSPARK_M);

    const uint32_t min_m_for_gemm = is_repacked_quant ? AGX_MOE_GEMM_MIN_M : 16u;

    const bool force_fp32_vecmat =
        req_prec_f32 ||
        use_dspark_b8 ||
        (M < min_m_for_gemm) ||
        (K < 32u) || ((K & 3u) != 0u);

    const uint32_t bcap          = use_dspark_b8 ? AGX_MOE_VECMAT_DSPARK_M : AGX_MOE_VECMAT_MAX_M;
    const bool     may_have_gemm = !force_fp32_vecmat;
    const uint32_t gemm_min_m    = may_have_gemm ? AGX_MOE_GEMM_MIN_M : 0xFFFFFFFFu;
    const uint32_t gemm_tile_m   = may_have_gemm
        ? moe_select_gemm_tile_m(M, layout, layout)
        : AGX_MOE_GEMM_TILE_M;

    agx_moe_stage_args bargs = {};
    bargs.n_tokens       = M;
    bargs.n_embd         = N;
    bargs.n_ff           = K;
    bargs.n_expert       = n_expert;
    bargs.top_k          = top_k;
    bargs.exp_scale0     = 1.0f;
    bargs.topk_ids_va    = agx_tensor_gpu_va(ids);
    bargs.ids_row_stride = static_cast<uint32_t>(ids->nb[1] / sizeof(int32_t));
    bargs.topk_w_va      = 0;
    bargs.down_layout    = layout;

    if (!dispatch_bucket_internal(r, stream, bargs, bcap, gemm_min_m, gemm_tile_m, AGX_BARRIER_STRONG_USC)) {
        return false;
    }

    const uint64_t base_va = r->meta_bo->gpu_va;
    const agx_kernel * k_vm = get_moe_exec_kernel(r, 1u, layout, bcap, req_prec_f32);
    const agx_kernel * k_zs = get_bucket_kernel(r, 3u);
    if (!k_vm || !k_zs) return false;

    agx_push_moe_exec p = {};
    p.header_va       = base_va;
    p.buckets_va      = base_va + r->off_vecmat_buckets;
    p.sorted_tok_va   = base_va + r->off_sorted_tok;
    p.sorted_kslot_va = base_va + r->off_sorted_kslot;
    p.sorted_w_va     = base_va + r->off_sorted_w;
    p.x_va            = agx_tensor_gpu_va(x);
    p.w_gate_va       = agx_tensor_gpu_va(w);
    p.y_va            = agx_tensor_gpu_va(dst);
    p.w_slice_B       = slice_B;
    p.K               = K;
    p.N               = N;
    p.n_ff            = K;
    p.top_k           = top_k;
    p.flags           = AGX_MOE_FLAG_RAW_MMID | (per_slot ? AGX_MOE_FLAG_RAW_MMID_PERSLOT : 0u);

    const agx_barrier_mode final_bar = (post_barrier == AGX_BARRIER_NONE)
        ? AGX_BARRIER_STRONG_USC
        : post_barrier;

    // Sparse zero-fill of ONLY skipped (eid < 0) rows — overlaps concurrently with vec-mat/GEMM!
    // During real LLM prefill (n_skipped == 0), this exits in 5 ns with ZERO DRAM writes!
    {
        agx_push_moe_bucket pz = {};
        pz.header_va        = base_va;
        pz.skipped_slots_va = base_va + r->off_skipped_slots;
        pz.moe_out_va       = agx_tensor_gpu_va(dst);
        pz.n_embd           = N;
        if (!agx_cdm_dispatch_indirect(
                stream, k_zs,
                base_va + sizeof(agx_moe_bucket_header),
                &pz, sizeof(pz), AGX_BARRIER_NONE)) {
            return false;
        }
    }

    if (!agx_cdm_dispatch_indirect(
            stream, k_vm,
            base_va + offsetof(agx_moe_bucket_header, grid_down_vecmat),
            &p, sizeof(p), may_have_gemm ? AGX_BARRIER_NONE : final_bar)) {
        return false;
    }

    if (may_have_gemm) {
        const agx_kernel * k_gm = get_moe_exec_kernel(r, 3u, layout, bcap, false, gemm_tile_m);
        if (!k_gm) return false;
        p.buckets_va = base_va + r->off_gemm_tiles;
        if (!agx_cdm_dispatch_indirect(
                stream, k_gm,
                base_va + offsetof(agx_moe_bucket_header, grid_down_gemm),
                &p, sizeof(p), final_bar)) {
            return false;
        }
    }

    moe_exit_record(r, stream);
    return true;
}

// ============================================================================
// Diagnostic CPU Accessors (SLC-pinned bucket table inspection for unit tests)
// ============================================================================

const agx_moe_bucket_header * agx_moe_get_bucket_header_cpu(const agx_moe_runner * r) {
    return (r && r->meta_bo) ? static_cast<const agx_moe_bucket_header *>(r->meta_bo->cpu_ptr) : nullptr;
}

const agx_moe_bucket * agx_moe_get_vecmat_buckets_cpu(const agx_moe_runner * r) {
    return (r && r->meta_bo)
        ? reinterpret_cast<const agx_moe_bucket *>(static_cast<const uint8_t *>(r->meta_bo->cpu_ptr) + r->off_vecmat_buckets)
        : nullptr;
}

const agx_moe_bucket * agx_moe_get_gemm_tiles_cpu(const agx_moe_runner * r) {
    return (r && r->meta_bo)
        ? reinterpret_cast<const agx_moe_bucket *>(static_cast<const uint8_t *>(r->meta_bo->cpu_ptr) + r->off_gemm_tiles)
        : nullptr;
}

const uint32_t * agx_moe_get_sorted_tokens_cpu(const agx_moe_runner * r) {
    return (r && r->meta_bo)
        ? reinterpret_cast<const uint32_t *>(static_cast<const uint8_t *>(r->meta_bo->cpu_ptr) + r->off_sorted_tok)
        : nullptr;
}

const float * agx_moe_get_sorted_weights_cpu(const agx_moe_runner * r) {
    return (r && r->meta_bo)
        ? reinterpret_cast<const float *>(static_cast<const uint8_t *>(r->meta_bo->cpu_ptr) + r->off_sorted_w)
        : nullptr;
}
