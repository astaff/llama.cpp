#pragma once

#include "agx-common.h"
#include "agx-drm.h"
#include "agx-cdm.h"
#include "agx-compiler.h"

#include <cstdint>

// ============================================================================
// Pillar 3: Generic ggml_cgraph Lowering & Inkling 2 / DSpark Fusion Engine
// ============================================================================
// Implements the cross-module backend hooks declared in `agx-common.h`:
//   - agx_supports_op
//   - agx_graph_compute
//   - agx_graph_optimize
//   - agx_graph_exec_cleanup
//
// Encodes an entire `ggml_cgraph` into ONE `agx_cdm_stream` submission with
// TLA+-verified barrier elision (`AGX_BARRIER_NONE` between hazard-free ops,
// `AGX_BARRIER_STRONG_USC` on RAW hazards, `AGX_BARRIER_MEMORY_HOST` once at
// graph completion), plus automatic subgraph fusion for Inkling 2 Mini
// (depth-mix, rank-64 native-FP32 controller gates, exact L2 clip, FWHT_{64,128},
// softcap+argmax, router top-8) and DSpark MTP.

struct agx_graph_exec_stats {
    uint64_t graphs_executed;
    uint64_t nodes_executed;
    uint64_t view_nodes_elided;
    uint64_t fused_fwht_hits;
    uint64_t fused_l2_clip_hits;
    uint64_t fused_ctrl_gate_hits;
    uint64_t fused_softcap_hits;
    uint64_t fused_dspark_hits;
    uint64_t fused_rms_mul_hits;
    uint64_t fused_situ_hits;
    uint64_t fused_add_scale_hits;
    uint64_t fused_sig_mul_hits;
    uint64_t fused_mix_hits;
    uint64_t fused_moe_wsum_hits;
    uint64_t fused_ssm_res_hits;
    uint64_t fused_moe_norm_hits;
    uint64_t fused_prep_store_hits;
    uint64_t fused_full_sconv_hits;
    uint64_t fused_router_hits;
    uint64_t fused_rel_bias_hits;
    uint64_t fused_cont_absorbed;
    uint64_t wave_sched_waves;
    uint64_t inkling2_fastpath_hits;
    uint64_t barriers_elided;
    uint64_t total_gpu_ns;
    uint64_t total_cpu_encode_ns;
};

struct agx_inkling2_engine;

// Retrieve (lazily creating) the Inkling 2 + DSpark engine owned by this backend context.
agx_inkling2_engine * agx_graph_exec_get_inkling2(agx_backend_context * ctx);

// Query & reset execution telemetry
void agx_graph_exec_get_stats(const agx_backend_context * ctx, agx_graph_exec_stats * out);
void agx_graph_exec_reset_stats(agx_backend_context * ctx);
