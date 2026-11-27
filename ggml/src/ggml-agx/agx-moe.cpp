#include "agx-moe.h"

#include <algorithm>
#include <cmath>
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
    uint32_t n_tokens;
    uint32_t top_k;
    uint32_t n_expert;
    uint32_t n_ff;
    uint32_t n_embd;
    uint32_t wg_gate_up_vecmat;
    uint32_t wg_down_vecmat;
    uint32_t wg_gate_up_gemm;
    uint32_t wg_down_gemm;
    uint32_t max_gemm_tiles;
    float    exp_scale0;
    uint32_t zero_elems;
};
static_assert(sizeof(agx_push_moe_bucket) == 120, "agx_push_moe_bucket size check");

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
    uint32_t pad0;
};
static_assert(sizeof(agx_push_moe_exec) == 112, "agx_push_moe_exec size check");

enum agx_moe_shader_flags : uint32_t {
    AGX_MOE_FLAG_SEPARATE_GU = 1u << 0,
    AGX_MOE_FLAG_X_F16       = 1u << 1,
    AGX_MOE_FLAG_RAW_MMID    = 1u << 2,
    AGX_MOE_FLAG_ACT_SILU    = 1u << 3,
    AGX_MOE_FLAG_ACT_GELU    = 1u << 4,
};

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
    uint32_t max_gemm_tiles     = 0;
    size_t   off_vecmat_buckets = 0;
    size_t   off_gemm_tiles     = 0;
    size_t   off_sorted_tok     = 0;
    size_t   off_sorted_kslot   = 0;
    size_t   off_sorted_w       = 0;

    // SLC-pinned intermediate hidden activations: [M * top_k, n_ff] F32
    // (B=1 -> 36 KB, DSpark M=5 -> 184 KB, 100% SLC-resident!)
    agx_bo * hidden_bo        = nullptr;
    size_t   hidden_cap_bytes = 0;

    // Compiled kernels
    const agx_kernel * k_bucket_bitonic   = nullptr;
    const agx_kernel * k_bucket_histogram = nullptr;
    const agx_kernel * k_zero_out         = nullptr;

    // Indexed by [weight_mode][batch_slot: 0=(cap 1), 1=(cap 5)]
    const agx_kernel * k_gu_vecmat[AGX_MOE_WEIGHT_MODES][2]   = {};
    const agx_kernel * k_down_vecmat[AGX_MOE_WEIGHT_MODES][2] = {};
    // Indexed by [weight_mode]
    const agx_kernel * k_gu_gemm[AGX_MOE_WEIGHT_MODES]        = {};
    const agx_kernel * k_down_gemm[AGX_MOE_WEIGHT_MODES]      = {};
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

    const uint32_t max_tiles = static_cast<uint32_t>(cap / AGX_MOE_GEMM_TILE_M + AGX_MOE_MAX_EXPERTS + 16u);
    size_t off = sizeof(agx_moe_bucket_header);
    const size_t off_vm   = agx_align_up(off, 128u); off = off_vm   + AGX_MOE_MAX_EXPERTS * sizeof(agx_moe_bucket);
    const size_t off_gt   = agx_align_up(off, 128u); off = off_gt   + max_tiles           * sizeof(agx_moe_bucket);
    const size_t off_tok  = agx_align_up(off, 128u); off = off_tok  + cap                 * sizeof(uint32_t);
    const size_t off_ksl  = agx_align_up(off, 128u); off = off_ksl  + cap                 * sizeof(uint32_t);
    const size_t off_w    = agx_align_up(off, 128u); off = off_w    + cap                 * sizeof(float);
    const size_t total_B  = agx_align_up(off, AGX_PAGE_SIZE);

    r->meta_bo = agx_bo_alloc(r->dev, total_B, AGX_BO_SLC_PIN | AGX_BO_WRITEBACK | AGX_BO_ZERO, "moe-bucket-slc");
    if (!r->meta_bo) {
        return false;
    }
    r->meta_cap_assigns   = cap;
    r->max_gemm_tiles     = max_tiles;
    r->off_vecmat_buckets = off_vm;
    r->off_gemm_tiles     = off_gt;
    r->off_sorted_tok     = off_tok;
    r->off_sorted_kslot   = off_ksl;
    r->off_sorted_w       = off_w;
    return true;
}

static bool ensure_hidden_bo(agx_moe_runner * r, size_t req_assigns, uint32_t n_ff) {
    const size_t needed = req_assigns * static_cast<size_t>(n_ff) * sizeof(float);
    if (r->hidden_bo && needed <= r->hidden_cap_bytes) {
        return true;
    }
    if (r->hidden_bo) {
        agx_bo_free(r->dev, r->hidden_bo);
        r->hidden_bo = nullptr;
        r->hidden_cap_bytes = 0;
    }
    const size_t alloc_B = agx_align_up(std::max<size_t>(needed, 256u * 1024u), AGX_PAGE_SIZE);
    // Keep hot MoE hidden activations SLC-pinned when <= 16 MiB, else stream
    const uint32_t flags = (alloc_B <= 16u * 1024u * 1024u)
        ? (AGX_BO_SLC_PIN | AGX_BO_WRITEBACK)
        : AGX_BO_SLC_STREAM;
    r->hidden_bo = agx_bo_alloc(r->dev, alloc_B, flags, "moe-hidden-slc");
    if (!r->hidden_bo) {
        return false;
    }
    r->hidden_cap_bytes = alloc_B;
    return true;
}

static const agx_kernel * get_bucket_kernel(agx_moe_runner * r, uint32_t mode) {
    if (!r->compiler) return nullptr;
    const agx_kernel ** slot = (mode == 0u) ? &r->k_bucket_bitonic
                             : (mode == 1u) ? &r->k_bucket_histogram
                                            : &r->k_zero_out;
    if (!*slot) {
        agx_compile_opts opts;
        opts.defines.push_back("BUCKET_MODE=" + std::to_string(mode));
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        const char * names[3] = { "moe_bitonic_bucket_sg32", "moe_histogram_bucket", "moe_zero_out" };
        *slot = agx_compiler_compile_file(r->compiler, names[mode], "moe_bitonic_bucket.comp", opts);
    }
    return *slot;
}

static const agx_kernel * get_moe_exec_kernel(agx_moe_runner *  r,
                                              uint32_t          stage,
                                              agx_weight_layout layout,
                                              uint32_t          batch_cap) {
    if (!r->compiler) return nullptr;
    const uint32_t wm = static_cast<uint32_t>(layout);
    if (wm >= AGX_MOE_WEIGHT_MODES) return nullptr;

    const uint32_t bslot = (batch_cap <= 1u) ? 0u : 1u;
    const uint32_t bcap  = (bslot == 0u) ? 1u : AGX_MOE_VECMAT_MAX_M;

    const agx_kernel ** slot = nullptr;
    if      (stage == 0u) slot = &r->k_gu_vecmat[wm][bslot];
    else if (stage == 1u) slot = &r->k_down_vecmat[wm][bslot];
    else if (stage == 2u) slot = &r->k_gu_gemm[wm];
    else if (stage == 3u) slot = &r->k_down_gemm[wm];
    else return nullptr;

    if (!*slot) {
        agx_compile_opts opts;
        opts.defines.push_back("PASS_STAGE=" + std::to_string(stage));
        opts.defines.push_back("WEIGHT_MODE=" + std::to_string(wm));
        if (stage <= 1u) {
            opts.defines.push_back("BATCH_CAP=" + std::to_string(bcap));
        }
        opts.is_hot_kernel     = true;
        opts.strict_zero_spill = true;
        const std::string kname = "fused_moe_s" + std::to_string(stage)
                                + "_w" + std::to_string(wm)
                                + (stage <= 1u ? ("_b" + std::to_string(bcap)) : "");
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
    delete r;
}

bool agx_moe_precompile_inkling2(agx_moe_runner * r) {
    if (!r || !r->compiler) return false;
    bool ok = true;
    ok &= (get_bucket_kernel(r, 0u) != nullptr);
    ok &= (get_bucket_kernel(r, 1u) != nullptr);
    ok &= (get_bucket_kernel(r, 2u) != nullptr);

    // Inkling 2 Mini MoE weight types: NVFP4 (blk 0..19) + Q8_0 (blk 20..23), repacked & raw
    const agx_weight_layout layouts[] = {
        AGX_WEIGHT_NVFP4_REPACKED,
        AGX_WEIGHT_Q8_0_REPACKED,
        AGX_WEIGHT_NVFP4_RAW,
        AGX_WEIGHT_Q8_0_RAW,
    };
    for (agx_weight_layout wm : layouts) {
        ok &= (get_moe_exec_kernel(r, 0u, wm, 1u) != nullptr);
        ok &= (get_moe_exec_kernel(r, 0u, wm, 5u) != nullptr);
        ok &= (get_moe_exec_kernel(r, 1u, wm, 1u) != nullptr);
        ok &= (get_moe_exec_kernel(r, 1u, wm, 5u) != nullptr);
        ok &= (get_moe_exec_kernel(r, 2u, wm, 5u) != nullptr);
        ok &= (get_moe_exec_kernel(r, 3u, wm, 5u) != nullptr);
    }
    return ok;
}

// ============================================================================
// Stage 5a: Bitonic / Histogram Bucketing Dispatch
// ============================================================================

bool agx_moe_dispatch_bucket(agx_moe_runner *           r,
                             agx_cdm_stream *           stream,
                             const agx_moe_stage_args & args,
                             agx_barrier_mode           post_barrier) {
    if (!r || !stream || args.n_tokens == 0 || args.top_k == 0 || args.n_expert == 0) {
        return false;
    }
    const size_t total_assigns = static_cast<size_t>(args.n_tokens) * args.top_k;
    if (!ensure_meta_bo(r, total_assigns)) {
        return false;
    }

    const bool use_bitonic = (total_assigns <= 64u); // M <= 8 with top_k=8 (incl. M=5 DSpark!)
    const agx_kernel * k_bkt = get_bucket_kernel(r, use_bitonic ? 0u : 1u);
    if (!k_bkt) {
        return false;
    }

    const uint64_t base_va = r->meta_bo->gpu_va;
    agx_push_moe_bucket pb = {};
    pb.topk_ids_va       = args.topk_ids_va;
    pb.topk_w_va         = args.topk_w_va;
    pb.header_va         = base_va;
    pb.vecmat_buckets_va = base_va + r->off_vecmat_buckets;
    pb.gemm_tiles_va     = base_va + r->off_gemm_tiles;
    pb.sorted_tok_va     = base_va + r->off_sorted_tok;
    pb.sorted_kslot_va   = base_va + r->off_sorted_kslot;
    pb.sorted_w_va       = base_va + r->off_sorted_w;
    pb.moe_out_va        = args.moe_out_va;
    pb.n_tokens          = args.n_tokens;
    pb.top_k             = args.top_k;
    pb.n_expert          = args.n_expert;
    pb.n_ff              = args.n_ff;
    pb.n_embd            = args.n_embd;
    pb.wg_gate_up_vecmat = (args.n_ff   + AGX_MOE_VECMAT_WG_PAIRS - 1u) / AGX_MOE_VECMAT_WG_PAIRS;
    pb.wg_down_vecmat    = (args.n_embd + AGX_MOE_VECMAT_WG_OUT   - 1u) / AGX_MOE_VECMAT_WG_OUT;
    pb.wg_gate_up_gemm   = (args.n_ff   + AGX_MOE_GEMM_WG_PAIRS   - 1u) / AGX_MOE_GEMM_WG_PAIRS;
    pb.wg_down_gemm      = (args.n_embd + AGX_MOE_GEMM_WG_OUT     - 1u) / AGX_MOE_GEMM_WG_OUT;
    pb.max_gemm_tiles    = r->max_gemm_tiles;
    pb.exp_scale0        = (args.exp_scale0 != 0.0f) ? args.exp_scale0 : 1.0f;
    pb.zero_elems        = args.n_tokens * args.n_embd;

    return agx_cdm_dispatch(stream, k_bkt, 1u, 1u, 1u, &pb, sizeof(pb), post_barrier);
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

    const size_t total_assigns = static_cast<size_t>(args.n_tokens) * args.top_k;
    if (!ensure_meta_bo(r, total_assigns) || !ensure_hidden_bo(r, total_assigns, args.n_ff)) {
        return false;
    }

    // 1. GPU-resident bucketing (bitonic for M<=8, histogram for M>8)
    //    Can overlap with moe_out zero-fill via AGX_BARRIER_NONE if !accumulate_out!
    const bool need_zero = !args.accumulate_out;
    if (!agx_moe_dispatch_bucket(r, stream, args,
                                 need_zero ? AGX_BARRIER_NONE : AGX_BARRIER_STRONG_USC)) {
        return false;
    }

    const uint64_t base_va = r->meta_bo->gpu_va;

    if (need_zero) {
        const agx_kernel * k_zero = get_bucket_kernel(r, 2u);
        if (!k_zero) return false;
        agx_push_moe_bucket pz = {};
        pz.moe_out_va = args.moe_out_va;
        pz.zero_elems = args.n_tokens * args.n_embd;
        const uint32_t wg_zero = (pz.zero_elems + 127u) / 128u;
        if (!agx_cdm_dispatch(stream, k_zero, wg_zero, 1u, 1u,
                              &pz, sizeof(pz), AGX_BARRIER_STRONG_USC)) {
            return false;
        }
    }

    uint32_t act_flags = 0u;
    if (args.act_op == AGX_MOE_ACT_SILU) act_flags |= AGX_MOE_FLAG_ACT_SILU;
    if (args.act_op == AGX_MOE_ACT_GELU) act_flags |= AGX_MOE_FLAG_ACT_GELU;
    if (args.gate_up_va == 0)            act_flags |= AGX_MOE_FLAG_SEPARATE_GU;
    if (args.x_is_f16)                   act_flags |= AGX_MOE_FLAG_X_F16;

    const uint32_t bcap = (args.n_tokens == 1u) ? 1u : AGX_MOE_VECMAT_MAX_M;
    const bool may_have_gemm = (args.n_tokens >= AGX_MOE_GEMM_MIN_M);

    // 2. Stage A: Fused Gate+Up + Exact SiTU (writes SLC-resident moe_hidden)
    //    - Vec-mat indirect dispatch for all buckets with M_bucket <= 5
    //    - 8x8 MMA GEMM indirect dispatch for all buckets with M_bucket >= 6 (when M >= 6)
    {
        const agx_kernel * k_gu_vm = get_moe_exec_kernel(r, 0u, args.gate_up_layout, bcap);
        if (!k_gu_vm) return false;

        agx_push_moe_exec p_gu = {};
        p_gu.header_va        = base_va;
        p_gu.buckets_va       = base_va + r->off_vecmat_buckets;
        p_gu.sorted_tok_va    = base_va + r->off_sorted_tok;
        p_gu.sorted_kslot_va  = base_va + r->off_sorted_kslot;
        p_gu.sorted_w_va      = base_va + r->off_sorted_w;
        p_gu.x_va             = args.ffn_x_va;
        p_gu.w_gate_va        = (args.gate_up_va != 0) ? args.gate_up_va : args.gate_va;
        p_gu.w_up_va          = (args.gate_up_va != 0) ? args.gate_up_va : args.up_va;
        p_gu.y_va             = r->hidden_bo->gpu_va;
        p_gu.w_slice_B        = args.gate_up_slice_B;
        p_gu.K                = args.n_embd;
        p_gu.N                = args.n_ff;
        p_gu.n_ff             = args.n_ff;
        p_gu.top_k            = args.top_k;
        p_gu.flags            = act_flags;
        p_gu.situ_beta        = args.situ_beta;
        p_gu.situ_linear_beta = args.situ_linear_beta;

        // If GEMM path will also run on disjoint expert buckets, elide barrier between vecmat & GEMM!
        const agx_barrier_mode gu_vm_bar = may_have_gemm ? AGX_BARRIER_NONE : AGX_BARRIER_STRONG_USC;
        if (!agx_cdm_dispatch_indirect(
                stream, k_gu_vm,
                base_va + offsetof(agx_moe_bucket_header, grid_gate_up_vecmat),
                &p_gu, sizeof(p_gu), gu_vm_bar)) {
            return false;
        }

        if (may_have_gemm) {
            const agx_kernel * k_gu_gm = get_moe_exec_kernel(r, 2u, args.gate_up_layout, bcap);
            if (!k_gu_gm) return false;
            p_gu.buckets_va = base_va + r->off_gemm_tiles;
            if (!agx_cdm_dispatch_indirect(
                    stream, k_gu_gm,
                    base_va + offsetof(agx_moe_bucket_header, grid_gate_up_gemm),
                    &p_gu, sizeof(p_gu), AGX_BARRIER_STRONG_USC)) {
                return false;
            }
        }
    }

    // 3. Stage B: Down Projection + Weighted Scatter-Add into SLC moe_out
    {
        const agx_kernel * k_dn_vm = get_moe_exec_kernel(r, 1u, args.down_layout, bcap);
        if (!k_dn_vm) return false;

        agx_push_moe_exec p_dn = {};
        p_dn.header_va        = base_va;
        p_dn.buckets_va       = base_va + r->off_vecmat_buckets;
        p_dn.sorted_tok_va    = base_va + r->off_sorted_tok;
        p_dn.sorted_kslot_va  = base_va + r->off_sorted_kslot;
        p_dn.sorted_w_va      = base_va + r->off_sorted_w;
        p_dn.x_va             = r->hidden_bo->gpu_va;
        p_dn.w_gate_va        = args.down_va;
        p_dn.w_up_va          = 0;
        p_dn.y_va             = args.moe_out_va;
        p_dn.w_slice_B        = args.down_slice_B;
        p_dn.K                = args.n_ff;
        p_dn.N                = args.n_embd;
        p_dn.n_ff             = args.n_ff;
        p_dn.top_k            = args.top_k;
        p_dn.flags            = 0u;

        // Vecmat and GEMM scatter-add atomically into moe_out -> can overlap with AGX_BARRIER_NONE!
        const agx_barrier_mode dn_vm_bar = may_have_gemm ? AGX_BARRIER_NONE : post_barrier;
        if (!agx_cdm_dispatch_indirect(
                stream, k_dn_vm,
                base_va + offsetof(agx_moe_bucket_header, grid_down_vecmat),
                &p_dn, sizeof(p_dn), dn_vm_bar)) {
            return false;
        }

        if (may_have_gemm) {
            const agx_kernel * k_dn_gm = get_moe_exec_kernel(r, 3u, args.down_layout, bcap);
            if (!k_dn_gm) return false;
            p_dn.buckets_va = base_va + r->off_gemm_tiles;
            if (!agx_cdm_dispatch_indirect(
                    stream, k_dn_gm,
                    base_va + offsetof(agx_moe_bucket_header, grid_down_gemm),
                    &p_dn, sizeof(p_dn), post_barrier)) {
                return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Generic GGML_OP_MUL_MAT_ID Bridge (for agx-graph-exec.cpp / test-backend-ops)
// ============================================================================

bool agx_moe_supports_mul_mat_id(const struct ggml_tensor * dst) {
    if (!dst || dst->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    const ggml_tensor * w   = dst->src[0]; // [K, N, n_expert]
    const ggml_tensor * x   = dst->src[1]; // [K, n_x_slots, M]
    const ggml_tensor * ids = dst->src[2]; // [top_k, M] I32
    if (!w || !x || !ids) return false;
    if (ids->type != GGML_TYPE_I32 || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (w->ne[2] > AGX_MOE_MAX_EXPERTS || x->ne[1] != 1) {
        return false; // x->ne[1] == 1 covers standard shared-input gate/up; per-slot down uses fused MoE
    }
    agx_weight_layout layout;
    return agx_gemm_infer_weight_layout(w, &layout);
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

    const uint64_t slice_B = agx_tensor_is_repacked(w)
        ? static_cast<uint64_t>(agx_repack_slice_stride(w->type, w->ne[0], w->ne[1]))
        : static_cast<uint64_t>(w->nb[2]);

    agx_moe_stage_args bargs = {};
    bargs.n_tokens    = M;
    bargs.n_embd      = N;
    bargs.n_ff        = K;
    bargs.n_expert    = n_expert;
    bargs.top_k       = top_k;
    bargs.exp_scale0  = 1.0f;
    bargs.topk_ids_va = agx_tensor_gpu_va(ids);
    bargs.topk_w_va   = 0;

    if (!agx_moe_dispatch_bucket(r, stream, bargs, AGX_BARRIER_STRONG_USC)) {
        return false;
    }

    const uint64_t base_va       = r->meta_bo->gpu_va;
    const uint32_t bcap          = (M == 1u) ? 1u : AGX_MOE_VECMAT_MAX_M;
    const bool     may_have_gemm = (M >= AGX_MOE_GEMM_MIN_M);

    const agx_kernel * k_vm = get_moe_exec_kernel(r, 1u, layout, bcap);
    if (!k_vm) return false;

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
    p.flags           = AGX_MOE_FLAG_RAW_MMID;

    if (!agx_cdm_dispatch_indirect(
            stream, k_vm,
            base_va + offsetof(agx_moe_bucket_header, grid_down_vecmat),
            &p, sizeof(p), may_have_gemm ? AGX_BARRIER_NONE : post_barrier)) {
        return false;
    }

    if (may_have_gemm) {
        const agx_kernel * k_gm = get_moe_exec_kernel(r, 3u, layout, bcap);
        if (!k_gm) return false;
        p.buckets_va = base_va + r->off_gemm_tiles;
        if (!agx_cdm_dispatch_indirect(
                stream, k_gm,
                base_va + offsetof(agx_moe_bucket_header, grid_down_gemm),
                &p, sizeof(p), post_barrier)) {
            return false;
        }
    }

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
