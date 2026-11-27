#pragma once

#include "agx-common.h"
#include "agx-drm.h"
#include "agx-cdm.h"
#include "agx-compiler.h"
#include "agx-repack.h"
#include "agx-kv-hybrid.h"

#include "kernels/fused_ctrl_gates_sram.h"
#include "kernels/fused_loop_prep_qkv_conv.h"
#include "kernels/fused_softcap_argmax.h"
#include "kernels/dspark_mtp.h"

#include <cstddef>
#include <cstdint>

// ============================================================================
// Pillar 3: Inkling 2 Mini 48-Pass Single-CDM-Stream Engine & DSpark MTP
// ============================================================================
// Eliminates the ~744-dispatch bottleneck by lowering Inkling 2 Mini's 48 passes
// (Prelude 0..5 -> Looped Core 6..17 x 3 -> Coda 18..23 -> LM Head + Softcap/Argmax)
// into a single CDM submission with zero host round-trips mid-token, native-FP32
// 32 KB threadgroup-SRAM cross-pass controller gates, exact L2 clip, in-register
// FWHT_{64,128}, SLC-resident router top-8 handoff to @kv-moe, and fused DSpark MTP.

#define AGX_INKLING2_MAX_PHYS_LAYERS  32u
#define AGX_INKLING2_MAX_LOOP_LAYERS  32u
#define AGX_INKLING2_MAX_SLOTS        64u
#define AGX_INKLING2_MAX_RES_BLOCKS   16u

struct agx_inkling2_hparams {
    uint32_t n_phys                  = 24;
    uint32_t n_layer_all             = 48;
    uint32_t core_start              = 6;
    uint32_t core_end                = 18;
    uint32_t n_passes                = 3;
    uint32_t gate_rank               = 64;

    uint32_t n_embd                  = 2304;
    uint32_t n_head                  = 24;
    uint32_t n_head_kv               = 8;
    uint32_t head_dim                = 128;
    uint32_t n_ff_exp                = 1152;
    uint32_t n_expert                = 128;
    uint32_t n_expert_used           = 8;

    uint32_t n_vocab_padded          = 201024;
    uint32_t n_vocab_unpadded        = 200058;

    uint32_t n_relative              = 32;
    uint32_t n_relative_extent       = 4096;
    uint32_t n_swa                   = 1024;
    uint32_t attn_res_block_size     = 6;
    uint32_t n_shortconv_l_cache     = 4;
    uint32_t n_sconv_channels        = 2 * 1024 + 2 * 2304; // 6656

    uint32_t n_log_scale_floor       = 1024;
    float    f_log_scale_alpha       = 0.1f;
    float    f_norm_rms_eps          = 1e-6f;
    float    f_qk_clip               = 16.0f;
    float    f_relative_clip         = 16.0f;
    float    situ_beta               = 1.0f;
    float    situ_linear_beta        = 1.0f;
    float    expert_weights_scale    = 1.0f;
    float    expert_weights_norm_eps = 1e-20f;
    float    f_final_logit_softcapping = 0.0f;

    bool     is_swa[AGX_INKLING2_MAX_SLOTS] = {};
    bool     use_fwht64_attn_out     = false;
    bool     use_fwht128_kv          = false;
};

// Fill Inkling 2 Mini's default 48-pass SWA/full-attn pattern (layers 5,11,17,23 full)
void agx_inkling2_fill_default_hparams(agx_inkling2_hparams * hp);

struct agx_inkling2_layer_tensors {
    const struct ggml_tensor * wq              = nullptr;
    const struct ggml_tensor * wk              = nullptr;
    const struct ggml_tensor * wv              = nullptr;
    const struct ggml_tensor * wo              = nullptr;
    const struct ggml_tensor * wqkv_gate       = nullptr;

    const struct ggml_tensor * attn_q_clip     = nullptr;
    const struct ggml_tensor * attn_sink_proj  = nullptr;

    const struct ggml_tensor * attn_rel        = nullptr;
    const struct ggml_tensor * attn_rel_bias   = nullptr;
    const struct ggml_tensor * attn_rel_clip   = nullptr;

    const struct ggml_tensor * attn_k_conv     = nullptr;
    const struct ggml_tensor * attn_v_conv     = nullptr;
    const struct ggml_tensor * attn_post_conv  = nullptr;
    const struct ggml_tensor * ffn_post_conv   = nullptr;

    const struct ggml_tensor * attn_norm       = nullptr; // SWA only
    const struct ggml_tensor * attn_q_inp_norm = nullptr; // full-attn only
    const struct ggml_tensor * attn_k_inp_norm = nullptr; // full-attn only
    const struct ggml_tensor * attn_v_inp_norm = nullptr; // full-attn only
    const struct ggml_tensor * attn_k_conv_proj = nullptr; // full-attn only
    const struct ggml_tensor * attn_v_conv_proj = nullptr; // full-attn only

    const struct ggml_tensor * attn_res_query  = nullptr; // il > 0
    const struct ggml_tensor * ffn_res_query   = nullptr;

    const struct ggml_tensor * ffn_norm        = nullptr;
    const struct ggml_tensor * ffn_gate_inp    = nullptr;
    const struct ggml_tensor * ffn_exp_probs_b = nullptr;
    const struct ggml_tensor * ffn_gate_up_exps = nullptr;
    const struct ggml_tensor * ffn_down_exps   = nullptr;
};

struct agx_inkling2_loop_tensors {
    const struct ggml_tensor * msg_query       = nullptr;
    const struct ggml_tensor * ctrl_down[4]    = {};
    const struct ggml_tensor * ctrl_gain[4]    = {};
    const struct ggml_tensor * ctrl_forget[4]  = {};
};

struct agx_inkling2_model_tensors {
    const struct ggml_tensor * tok_embd         = nullptr;
    const struct ggml_tensor * tok_norm         = nullptr;
    const struct ggml_tensor * output           = nullptr;
    const struct ggml_tensor * output_norm      = nullptr;
    const struct ggml_tensor * output_res_query = nullptr;

    agx_inkling2_layer_tensors layers[AGX_INKLING2_MAX_PHYS_LAYERS];
    agx_inkling2_loop_tensors  loop_layers[AGX_INKLING2_MAX_LOOP_LAYERS];
};

struct agx_dspark_mtp_tensors {
    const struct ggml_tensor * markov_w1       = nullptr;
    const struct ggml_tensor * markov_w2       = nullptr;
    const struct ggml_tensor * markov_w2_s     = nullptr;
    const struct ggml_tensor * markov_w2_b     = nullptr;
    const struct ggml_tensor * joint_proj      = nullptr;
    const struct ggml_tensor * joint_proj_b    = nullptr;
    const struct ggml_tensor * markov_ctx_mix  = nullptr;
    const struct ggml_tensor * markov_ctx_norm = nullptr;

    uint32_t n_embd             = 2304;
    uint32_t rank               = 256;
    uint32_t n_vocab            = 200058;
    uint32_t n_target           = 0;
    uint32_t block_size         = 5;
    bool     sample_from_anchor = false;
    bool     has_f_rnn          = true;
    float    softcap            = 0.0f;
    float    rms_eps            = 1e-6f;
};

struct agx_inkling2_step_stats {
    uint32_t passes_executed;
    uint32_t cdm_dispatches;
    uint32_t cdm_submits;         // 1 for single-CDM-stream 48-pass decode!
    uint64_t gpu_time_ns;
    int32_t  greedy_token;
};

struct agx_inkling2_engine;

// Create / destroy the Inkling 2 Mini + DSpark MTP execution engine on an existing AGX context.
agx_inkling2_engine * agx_inkling2_engine_create(agx_backend_context *       ctx,
                                                 const agx_inkling2_hparams & hp,
                                                 uint32_t                    max_batch_tokens = 8);
void                  agx_inkling2_engine_destroy(agx_inkling2_engine * eng);

// Reset recurrent conv history & hybrid KV cache for a fresh sequence.
void agx_inkling2_engine_reset(agx_inkling2_engine * eng);

// Helper to map any ggml_tensor to its AGX_WTYPE_* enum and row byte stride
uint32_t agx_inkling2_tensor_wtype(const struct ggml_tensor * t, uint32_t * out_row_bytes);

// Execute a full 48-pass Inkling 2 Mini forward step (Prelude 0..5 -> Looped Core 6..17 x 3
// -> Coda 18..23 -> LM Head + Softcap/Argmax) in a SINGLE CDM submission (0 CPU wakeups).
bool agx_inkling2_run_forward(agx_inkling2_engine *              eng,
                              const agx_inkling2_model_tensors & weights,
                              agx_kv_hybrid_pool *               kv_pool,
                              const int32_t *                    tokens,
                              const int32_t *                    positions,
                              uint32_t                           n_tokens,
                              uint32_t                           n_kv_past,
                              float *                            out_logits, // optional (nullptr = greedy token only)
                              agx_inkling2_step_stats *          out_stats);

// ============================================================================
// Direct Pillar 3 Fused Kernel Dispatchers (used by engine, graph-exec & @verifier)
// ============================================================================

bool agx_inkling2_dispatch_depth_mix(agx_inkling2_engine * eng,
                                     agx_cdm_stream *      stream,
                                     const uint64_t *      state_vas,
                                     uint32_t              n_states,
                                     uint64_t              query_va,
                                     uint32_t              n_queries,
                                     uint64_t              norm_w_va,
                                     uint64_t              out_va,
                                     uint64_t              out_pre_va,
                                     uint32_t              n_embd,
                                     uint32_t              n_tokens,
                                     float                 eps_mix,
                                     float                 eps_norm,
                                     agx_barrier_mode      post_barrier = AGX_BARRIER_STRONG_USC);

bool agx_inkling2_dispatch_ctrl_gates(agx_inkling2_engine *      eng,
                                      agx_cdm_stream *           stream,
                                      const agx_ctrl_gate_desc * descs,
                                      uint32_t                   n_ctrls,
                                      uint32_t                   n_embd,
                                      uint32_t                   rank,
                                      uint32_t                   n_tokens,
                                      float                      eps_ctrl,
                                      float                      eps_norm,
                                      agx_barrier_mode           post_barrier = AGX_BARRIER_STRONG_USC);

bool agx_inkling2_dispatch_conv_clip_fwht(agx_inkling2_engine *          eng,
                                          agx_cdm_stream *               stream,
                                          const agx_conv_clip_fwht_push & args,
                                          agx_barrier_mode               post_barrier = AGX_BARRIER_STRONG_USC);

bool agx_inkling2_dispatch_post_conv_res(agx_inkling2_engine *         eng,
                                         agx_cdm_stream *              stream,
                                         const agx_post_conv_res_push & args,
                                         agx_barrier_mode              post_barrier = AGX_BARRIER_STRONG_USC);

bool agx_inkling2_dispatch_router_top8(agx_inkling2_engine *       eng,
                                       agx_cdm_stream *            stream,
                                       const agx_router_top8_push & args,
                                       agx_barrier_mode            post_barrier = AGX_BARRIER_STRONG_USC);

bool agx_inkling2_dispatch_rel_bias_table(agx_inkling2_engine *          eng,
                                          agx_cdm_stream *               stream,
                                          const agx_rel_bias_table_push & args,
                                          agx_barrier_mode               post_barrier = AGX_BARRIER_STRONG_USC);

bool agx_inkling2_dispatch_softcap_argmax(agx_inkling2_engine * eng,
                                          agx_cdm_stream *      stream,
                                          uint64_t              src_va,
                                          uint64_t              dst_logits_va,
                                          uint64_t              out_token_va,
                                          uint32_t              n_vocab_unpadded,
                                          uint32_t              src_stride_f32,
                                          uint32_t              n_tokens,
                                          float                 softcap,
                                          bool                  compute_argmax,
                                          agx_barrier_mode      post_barrier = AGX_BARRIER_STRONG_USC);

// DSpark MTP fused sidecar dispatchers (sharing the same AGX context & CDM stream)
bool agx_dspark_dispatch_inject_ctx_mix(agx_inkling2_engine * eng,
                                        agx_cdm_stream *      stream,
                                        uint64_t              feats_va,
                                        uint64_t              mix_w_va,
                                        uint64_t              norm_w_va,
                                        uint64_t              out_va,
                                        uint32_t              n_embd,
                                        uint32_t              n_target,
                                        uint32_t              n_tokens,
                                        float                 rms_eps,
                                        agx_barrier_mode      post_barrier = AGX_BARRIER_STRONG_USC);

bool agx_dspark_dispatch_markov_head(agx_inkling2_engine *          eng,
                                     agx_cdm_stream *               stream,
                                     const agx_dspark_mtp_tensors & w,
                                     uint64_t                       anchor_toks_va, // [n_blocks] I32
                                     uint64_t                       base_logits_va, // [n_vocab, block_drafts, n_blocks] F32
                                     uint64_t                       hidden_va,      // [n_embd, block_drafts, n_blocks] F32
                                     uint64_t                       f_rnn_va,       // optional [n_embd, n_blocks] F32
                                     uint64_t                       rnn_state_va,   // optional [n_embd, block_drafts, n_blocks] F32
                                     uint64_t                       out_logits_va,  // [n_vocab, block_drafts, n_blocks] F32
                                     uint64_t                       out_h_nextn_va, // optional [n_embd, block_drafts, n_blocks] F32
                                     uint32_t                       n_blocks,
                                     uint32_t                       block_drafts,
                                     agx_barrier_mode               post_barrier = AGX_BARRIER_STRONG_USC);

// TLA+-verified 40-workgroup sense-reversing atomic barrier hardware verification
bool agx_inkling2_run_sense_barrier_probe(agx_inkling2_engine * eng,
                                          uint32_t              n_workgroups,
                                          uint32_t              n_iters,
                                          uint64_t *            out_gpu_ns,
                                          uint32_t *            out_epoch);
