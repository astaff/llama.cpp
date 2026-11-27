#pragma once

#include "agx-common.h"
#include "agx-cdm.h"
#include "agx-compiler.h"
#include "agx-repack.h"

#include <cstdint>

// ============================================================================
// Pillar 2: Predicated B=1..5 Simdgroup Vec-Mat & 8x8 simdgroup_matrix GEMM
// ============================================================================
// Exposes both:
//   1. GGML-level MUL_MAT dispatch (`agx_gemm_supports_mul_mat`, `agx_gemm_mul_mat`)
//   2. Raw-VA CDM-stream dispatch (`agx_gemm_dispatch`) for Pillar 3 (Inkling 2
//      48-pass megakernel / LM head / DSpark MTP), Pillar 4-5 (MoE / attn), and
//      Pillar 2 roofline bandwidth / TFLOPS benchmarks.

enum agx_weight_layout : uint32_t {
    AGX_WEIGHT_NVFP4_REPACKED = 0, // AGX_NVFP4_32x16 super-tile (1152B = 9x128B)
    AGX_WEIGHT_NVFP4_RAW      = 1, // standard ggml block_nvfp4 (36B)
    AGX_WEIGHT_Q8_0_REPACKED  = 2, // AGX_Q8_0_8BLK planar (272B)
    AGX_WEIGHT_Q8_0_RAW       = 3, // standard ggml block_q8_0 (34B)
    AGX_WEIGHT_MXFP8_REPACKED = 4, // AGX_MXFP8_16BLK planar (528B)
    AGX_WEIGHT_F8_E4M3        = 5, // flat OCP FP8 E4M3 (1B/elem)
    AGX_WEIGHT_BF16           = 6, // flat BF16 (2B/elem)
    AGX_WEIGHT_F16            = 7, // flat F16 (2B/elem)
    AGX_WEIGHT_F32            = 8, // flat F32 (4B/elem)
};

// Exact push-constant payload shared by nvfp4_vecmat.comp, mxfp8_q8_vecmat.comp,
// and agx_mma_gemm.comp (64 bytes, well within 256B AGX push window).
struct agx_gemm_push_args {
    uint64_t w_va;
    uint64_t x_va;
    uint64_t y_va;
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t x_stride;
    uint32_t y_stride;
    uint32_t w_slice_bytes;
    uint32_t x_slice_elems;
    uint32_t y_slice_elems;
    uint32_t w_bcast_div;
    uint32_t _pad;
};
static_assert(sizeof(agx_gemm_push_args) == 64, "agx_gemm_push_args must be 64 bytes");

struct agx_gemm_desc {
    uint64_t          w_va           = 0;
    uint64_t          x_va           = 0;
    uint64_t          y_va           = 0;
    uint32_t          M              = 1;
    uint32_t          N              = 0;
    uint32_t          K              = 0;
    uint32_t          x_stride_elems = 0; // 0 -> K
    uint32_t          y_stride_elems = 0; // 0 -> N
    uint32_t          w_slice_bytes  = 0;
    uint32_t          x_slice_elems  = 0;
    uint32_t          y_slice_elems  = 0;
    uint32_t          batch_count    = 1; // grid_z slices (ne2 * ne3)
    uint32_t          w_bcast_div    = 1; // w_slice = slice / max(1, w_bcast_div)
    agx_weight_layout layout         = AGX_WEIGHT_F32;
    bool              force_mma      = false;
    bool              force_vecmat   = false;
    agx_barrier_mode  post_barrier   = AGX_BARRIER_STRONG_USC;
};

// Lifecycle (stored in ctx->gemm_state)
bool agx_gemm_init(agx_backend_context * ctx);
void agx_gemm_cleanup(agx_backend_context * ctx);

// Precompile all core Pillar 2 kernel specializations & verify 0:0 spills / <= 96 GPRs
bool agx_gemm_precompile_all(agx_backend_context * ctx);

// Infer weight layout from a ggml_tensor (honors agx_tensor_is_repacked(w))
bool agx_gemm_infer_weight_layout(const struct ggml_tensor * w, agx_weight_layout * out_layout);

// Raw-VA CDM-stream dispatch (selects predicated vec-mat for M<=5, 8x8 MMA GEMM for M>=6)
bool agx_gemm_dispatch(agx_backend_context * ctx,
                       agx_cdm_stream *      stream,
                       const agx_gemm_desc & desc);

// Direct ggml_tensor MUL_MAT helpers
bool agx_gemm_supports_mul_mat(const struct ggml_tensor * dst);
bool agx_gemm_mul_mat(agx_backend_context *      ctx,
                      agx_cdm_stream *           stream,
                      const struct ggml_tensor * dst,
                      agx_barrier_mode           post_barrier = AGX_BARRIER_STRONG_USC);
