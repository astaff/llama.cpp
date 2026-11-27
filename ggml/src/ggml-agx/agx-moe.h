#pragma once

#include "agx-common.h"
#include "agx-drm.h"
#include "agx-cdm.h"
#include "agx-compiler.h"
#include "agx-repack.h"
#include "agx-gemm.h"

#include <cstddef>
#include <cstdint>

// ============================================================================
// Pillar 5: GPU-Resident Bitonic Token-to-Expert Bucketing & Fused MoE
//           (`moe_bitonic_bucket` + `fused_moe_gate_up_situ_down`)
// ============================================================================
//
// Zero CPU sync across the entire MoE stage:
//   1. `moe_bitonic_bucket.comp`:
//        - M <= 8 (M*K <= 64, incl. B=1 decode & M=5 DSpark verify):
//          single 32-lane simdgroup 64-element bitonic sort deduplicating
//          `(expert_id, token_idx, k_slot)` in registers/256B SRAM (~150 cycles).
//        - M > 8 (prefill M=9..8192):
//          128-bin TG-SRAM histogram + exclusive prefix sum + scatter.
//        - Applies `exp_scale0` (`hparams.expert_weights_scale`) once into
//          `sorted_weight[]` and partitions active experts into:
//            * `vecmat_buckets` (`1 <= M_bucket <= 5`) -> predicated K-coalesced vec-mat
//            * `gemm_tiles` (`M_bucket >= 6`, sliced into 16-tok M-tiles) -> 8x8 MMA GEMM
//        - Emits 4 HAL200/G17P 24-byte `agx_cdm_indirect_grid` records directly in SLC!
//   2. `fused_moe_gate_up_situ_down.comp`:
//        - Stage A (`PASS_STAGE=0`): fused gate+up projection reading each active
//          expert's weights ONCE from DRAM across all `M_bucket` tokens, applying
//          exact Inkling SiTU (or SwiGLU) activation in registers, storing compact
//          SLC-resident hidden activations `moe_hidden[assign_idx, n_ff]`.
//        - Stage B (`PASS_STAGE=1`): down projection reading `W_down` ONCE per active
//          expert across all `M_bucket` tokens, scaling by `sorted_weight`, and
//          accumulating into SLC-pinned `moe_out[tok, n_embd]` (F32).
//        - Dynamic crossover: `M_bucket <= 5` runs predicated K-coalesced vec-mat
//          (`<= 80` GPRs, `0:0` spills); `M_bucket >= 6` runs register-direct 8x8
//          `simdgroup_matrix` (`GL_KHR_cooperative_matrix`) GEMM (`<= 88` GPRs, `0:0` spills).

#define AGX_MOE_MAX_EXPERTS      256u
#define AGX_MOE_DIRECT_MAX_M     1u  // M=1 direct fastpath (M>=2 deduplicates shared experts via bitonic bucket!)
#define AGX_MOE_VECMAT_MAX_M     4u  // Lean 4-scalar BATCH_CAP=4 for M>=9 tail buckets & M<=4 (<=52 GPR32, 0:0 spills)
#define AGX_MOE_VECMAT_DSPARK_M  8u  // Wave 5: 8-way sh_x BATCH_CAP=8 for M=5..8 DSpark (1x W load, 0 GEMM dispatches!)
#define AGX_MOE_GEMM_MIN_M       5u  // M >= 9 prefill: M_bucket >= 5 -> 64N x 32M RowMajor 8x8 MMA GEMM
#define AGX_MOE_GEMM_TILE_M      32u
#define AGX_MOE_VECMAT_WG_PAIRS  32u // 32 (gate,up) pairs/WG for gate_up vec-mat (Q8_0_8BLK uses 4 pairs/WG)
#define AGX_MOE_VECMAT_WG_OUT    64u // 64 output rows/WG for down vec-mat (Q8_0_8BLK uses 8 rows/WG)
#define AGX_MOE_GEMM_WG_PAIRS    32u // 32 (gate,up) pairs/WG for gate_up 8x8 MMA GEMM
#define AGX_MOE_GEMM_WG_OUT      64u // 64 output rows/WG for down 8x8 MMA GEMM

enum agx_moe_act_op : uint32_t {
    AGX_MOE_ACT_SITU  = 0, // Inkling 2: beta*tanh(g/beta)*sigmoid(g) * (lb>0 ? lb*tanh(u/lb) : u)
    AGX_MOE_ACT_SILU  = 1, // Standard SwiGLU: silu(g) * u
    AGX_MOE_ACT_GELU  = 2, // GeGLU: gelu(g) * u
};

// One deduplicated expert bucket (or 16-token GEMM M-tile slice) in SLC
struct agx_moe_bucket {
    uint32_t expert_id;     // [0 .. n_expert-1]
    uint32_t token_offset;  // start offset in sorted_token_idx / sorted_weight / moe_hidden
    uint32_t count;         // M_bucket (1..5 for vecmat; 1..16 for gemm tile)
    uint32_t pad0;
};
static_assert(sizeof(agx_moe_bucket) == 16, "agx_moe_bucket must be 16B");

// 128-byte cache-line-aligned SLC header + 4 HAL200 indirect grids written on GPU
struct alignas(128) agx_moe_bucket_header {
    uint32_t              n_active_experts;
    uint32_t              n_vecmat_buckets;
    uint32_t              n_gemm_tiles;
    uint32_t              total_assignments;
    uint32_t              n_tokens;
    uint32_t              top_k;
    uint32_t              n_expert;
    uint32_t              n_skipped;           // count of negative/OOB expert assignments
    agx_cdm_indirect_grid grid_gate_up_vecmat; // offset 32
    agx_cdm_indirect_grid grid_down_vecmat;    // offset 56
    agx_cdm_indirect_grid grid_gate_up_gemm;   // offset 80
    agx_cdm_indirect_grid grid_down_gemm;      // offset 104
};
static_assert(sizeof(agx_moe_bucket_header) == 128,
              "agx_moe_bucket_header must occupy exactly one 128B cache line");

// Stage 3: Router -> MoE Launch Contract (called by @megakernel & agx-graph-exec)
struct agx_moe_stage_args {
    uint32_t          n_tokens;          // M (1 decode, 5 DSpark verify, up to 8192 prefill)
    uint32_t          n_embd;            // 2304
    uint32_t          n_ff;              // 1152
    uint32_t          n_expert;          // 128
    uint32_t          top_k;             // 8

    agx_moe_act_op    act_op;            // AGX_MOE_ACT_SITU for Inkling 2 Mini
    float             situ_beta;         // hparams.situ_beta
    float             situ_linear_beta;  // hparams.situ_linear_beta
    float             exp_scale0;        // hparams.expert_weights_scale (pass 1.0f if router already scaled!)

    // Activations & router outputs (SLC-pinned)
    uint64_t          ffn_x_va;          // [M, n_embd] F32 (post-ffn_norm)
    uint64_t          topk_ids_va;       // [M, top_k] int32_t (row stride = ids_row_stride elems)
    uint32_t          ids_row_stride;    // int32 elems between token rows in topk_ids (0 -> top_k)
    uint64_t          topk_w_va;         // [M, top_k] F32 (sigmoid-gated + norm eps=1e-20)
    uint64_t          moe_out_va;        // [M, n_embd] F32 output (pre-ffn_post_conv; zeroed internally)

    // Cold streaming expert weight tensors (AGX_BO_SLC_STREAM / AGX0_Repack)
    // Merged gate_up [n_embd, 2*n_ff, n_expert] (if gate_up_va != 0), OR separate gate & up
    uint64_t          gate_up_va;
    uint64_t          gate_va;           // used when gate_up_va == 0
    uint64_t          up_va;             // used when gate_up_va == 0
    uint64_t          down_va;           // [n_ff, n_embd, n_expert]
    uint64_t          gate_up_slice_B;   // byte stride between experts in gate_up (or gate/up)
    uint64_t          down_slice_B;      // byte stride between experts in down
    agx_weight_layout gate_up_layout;    // e.g. AGX_WEIGHT_NVFP4_REPACKED (blk 0..19), AGX_WEIGHT_Q8_0_REPACKED (blk 20..23)
    agx_weight_layout down_layout;

    bool              x_is_f16       = false; // optional F16 input activation
    bool              accumulate_out = false; // false = zero moe_out before scatter; true = add into existing moe_out

    // Wave 4 (Theorem W4.3) optional fused 2-stage SLC scratch & residual add:
    uint64_t          moe_inter_va     = 0;   // optional SLC-pinned [M, top_k, n_ff] F32 scratch (0 -> internal hidden_bo)
    uint64_t          res_va           = 0;   // optional [M, n_embd] F32 residual added to down+wsum -> moe_out (0 = none)

    // Optional per-expert FP32 2nd-level scale vectors [n_expert] (NVFP4 blk.<i>.ffn_*_exps.scale; 0 = 1.0f):
    uint64_t          gate_up_scale_va = 0;   // multiplied into (gate, up) before SiTU/SwiGLU/GeGLU
    uint64_t          down_scale_va    = 0;   // multiplied into down output before router weight
};

struct agx_moe_runner;

// Lifecycle (stored in ctx->moe_state)
agx_moe_runner * agx_moe_create(agx_drm_device * dev, agx_compiler * compiler);
void             agx_moe_destroy(agx_moe_runner * runner);

// Precompile Inkling 2 Mini MoE kernels (NVFP4 + Q8_0, vec-mat + 8x8 MMA GEMM)
// and verify 0:0 spills & <= 96 GPRs via Borealis shaderdb.
bool agx_moe_precompile_inkling2(agx_moe_runner * runner);

// Dispatch Stage 5a (bitonic sort / histogram bucketing) only (useful for standalone tests)
bool agx_moe_dispatch_bucket(agx_moe_runner *           runner,
                             agx_cdm_stream *           stream,
                             const agx_moe_stage_args & args,
                             agx_barrier_mode           post_barrier = AGX_BARRIER_STRONG_USC);

// Dispatch full Pillar 5 pipeline:
//   moe_bitonic_bucket -> (optional zero moe_out) -> fused_moe_gate_up_situ -> fused_moe_down_scatter
// Zero CPU sync; uses GPU-indirect HAL200 dispatches reading the SLC bucket header!
bool agx_moe_dispatch(agx_moe_runner *           runner,
                      agx_cdm_stream *           stream,
                      const agx_moe_stage_args & args,
                      agx_barrier_mode           post_barrier = AGX_BARRIER_STRONG_USC);

// Wave 4 (Theorem W4.3): Fused 2-Stage MoE (`gate_up + SiTU -> SLC moe_inter -> down + wsum [+ res_va]`)
//   - M == 1: strictly 2 dispatches + 1 STRONG_USC barrier (PASS_STAGE=5 -> PASS_STAGE=6),
//             zero bucketing, zero zero-fill, zero atomics, exact 0-ULP sequential FP32 wsum!
//   - M >= 2: deduplicated bitonic/histogram bucketing + lean BATCH_CAP=4 vecmat + gemm32 (count >= 5),
//             using args.moe_inter_va (or internal hidden_bo) and optional args.res_va.
bool agx_moe_dispatch_fused_situ_down(agx_moe_runner *           runner,
                                      agx_cdm_stream *           stream,
                                      const agx_moe_stage_args & args,
                                      agx_barrier_mode           post_barrier = AGX_BARRIER_STRONG_USC);

// Generic ggml `GGML_OP_MUL_MAT_ID` dispatch helper for `agx-graph-exec.cpp` / `test-backend-ops`.
bool agx_moe_supports_mul_mat_id(const struct ggml_tensor * dst);
bool agx_moe_dispatch_mul_mat_id(agx_moe_runner *           runner,
                                 agx_cdm_stream *           stream,
                                 const struct ggml_tensor * dst,
                                 agx_barrier_mode           post_barrier = AGX_BARRIER_STRONG_USC);

// Access the SLC-pinned bucket header (for unit tests verifying deduplication & crossover)
const agx_moe_bucket_header * agx_moe_get_bucket_header_cpu(const agx_moe_runner * runner);
const agx_moe_bucket *        agx_moe_get_vecmat_buckets_cpu(const agx_moe_runner * runner);
const agx_moe_bucket *        agx_moe_get_gemm_tiles_cpu(const agx_moe_runner * runner);
const uint32_t *              agx_moe_get_sorted_tokens_cpu(const agx_moe_runner * runner);
const float *                 agx_moe_get_sorted_weights_cpu(const agx_moe_runner * runner);

// Host-side exact Inkling 2 SiTU reference (matches src/llama-graph.cpp:2636-2650)
float agx_situ_ref(float gate, float up, float beta, float linear_beta);
