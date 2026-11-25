#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>

// S19_WAVE3: lazily materialize stacked FullAttn Q/K/V input norms for standalone inkling2
static void s19_ensure_inkling2_norms(const llama_model & model) {
    if (model.s19_inkling_ready || getenv("GGML_INKLING_NO_FUSE") || model.layers.empty()) {
        return;
    }
    static std::mutex s19_mtx;
    std::lock_guard<std::mutex> lock(s19_mtx);
    if (model.s19_inkling_ready) {
        return;
    }
    const ggml_tensor * ref = model.layers[0].ffn_norm;
    if (!ref || !ref->buffer || !ref->data || ggml_backend_buffer_get_size(ref->buffer) == 0) {
        return;
    }
    const int n_all  = (int)model.layers.size();
    const int64_t ne = model.hparams.n_embd;
    int n_full = 0;
    for (int i = 0; i < n_all; ++i) {
        if (!model.hparams.is_swa(i) && model.layers[i].attn_q_inp_norm) {
            ++n_full;
        }
    }
    if (n_full == 0) {
        model.s19_inkling_ready = true;
        return;
    }
    ggml_init_params params = { ggml_tensor_overhead() * (size_t)(n_full + 4), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    std::vector<std::pair<int, ggml_tensor *>> created;
    for (int i = 0; i < n_all; ++i) {
        if (!model.hparams.is_swa(i) && model.layers[i].attn_q_inp_norm) {
            bool dup = false;
            for (const auto & pr : created) {
                if (model.layers[pr.first].attn_q_inp_norm == model.layers[i].attn_q_inp_norm) {
                    const_cast<llama_layer &>(model.layers[i]).s19_attn_qkv_inp_norm = pr.second;
                    dup = true;
                    break;
                }
            }
            if (!dup) {
                ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne, 3);
                const_cast<llama_layer &>(model.layers[i]).s19_attn_qkv_inp_norm = t;
                created.push_back({ i, t });
            }
        }
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(ref->buffer);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    if (!buf) {
        for (int i = 0; i < n_all; ++i) {
            const_cast<llama_layer &>(model.layers[i]).s19_attn_qkv_inp_norm = nullptr;
        }
        return;
    }
    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> stage((size_t)(3 * ne * sizeof(float)));
    const size_t row_bytes = (size_t)(ne * sizeof(float));
    for (const auto & pr : created) {
        const auto & l = model.layers[pr.first];
        ggml_backend_tensor_get(l.attn_q_inp_norm, stage.data() + 0 * row_bytes, 0, row_bytes);
        ggml_backend_tensor_get(l.attn_k_inp_norm, stage.data() + 1 * row_bytes, 0, row_bytes);
        ggml_backend_tensor_get(l.attn_v_inp_norm, stage.data() + 2 * row_bytes, 0, row_bytes);
        ggml_backend_tensor_set(pr.second, stage.data(), 0, 3 * row_bytes);
    }
    model.s19_inkling_ctx   = std::move(ctx);
    model.s19_inkling_buf   = std::move(buf);
    model.s19_inkling_ready = true;
}

static ggml_tensor * s19_inkling_mix_rms(
        ggml_context * ctx0,
        const std::vector<ggml_tensor *> & blocks,
        ggml_tensor * cur,
        ggml_tensor * query,
        ggml_tensor * rms_w,
        float rms_eps) {
    GGML_ASSERT(blocks.size() + 1 <= 6);
    ggml_tensor * st[6] = {};
    const size_t nb_blk = blocks.size();
    for (size_t i = 0; i < nb_blk; ++i) {
        st[i] = blocks[nb_blk - 1 - i];
    }
    st[nb_blk] = cur;
    return ggml_inkling_depth_mix_rms(ctx0, st, (int)(nb_blk + 1), query, rms_w, 1e-6f, rms_eps);
}

// max elements of the dense relative bias per attention call, and max chunks per layer
static constexpr int64_t n_kq_bias_max  = int64_t(1) << 29;
static constexpr int64_t n_kq_chunk_max = 64;

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    prec_policy.set_acc(output, GGML_PREC_F32);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold, ggml_tensor * mul_tau) {
    if (!getenv("GGML_INKLING_NO_FUSE")) {
        return ggml_inkling_l2_clip(ctx0, cur, scale, mul_tau, threshold, 1e-6f);
    }
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    // x*min(1, t/|x|) = t*x/max(|x|, t), and l2_norm(x, eps) = x/max(|x|, eps)
    ggml_tensor * res = ggml_scale(ctx0, ggml_l2_norm(ctx0, cur, threshold), threshold);
    return mul_tau ? ggml_mul(ctx0, res, mul_tau) : res;
}

// the normalized states of list, last first: [norm(l_n-1), ..., norm(l_0)]
// the stacks are kept per graph, so each block is normalized and copied once and not again for every later mix
ggml_tensor * llama_model_inkling2::graph::mix_stack(const std::vector<ggml_tensor *> & list) {
    const auto it = mix_stacks.find(list);
    if (it != mix_stacks.end()) {
        return it->second;
    }

    ggml_tensor * res = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, list.back(), n_embd, 1, n_tokens), 1e-6f);
    if (list.size() > 1) {
        res = ggml_concat(ctx0, res, mix_stack(std::vector<ggml_tensor *>(list.begin(), list.end() - 1)), 1);
    }

    return mix_stacks[list] = res;
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
// Series 17 (D1): Outputs [n_embd, n_tokens, Q] so per-query views are 100% contiguous (zero GGML_OP_CONT!).
ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    if (!getenv("GGML_INKLING_NO_FUSE") && blocks.size() + 1 <= 6) {
        ggml_tensor * st[6] = {};
        const size_t nb_blk = blocks.size();
        // Exact reverse-prepended order (blocks.reverse ++ [cur])
        for (size_t i = 0; i < nb_blk; ++i) {
            st[i] = blocks[nb_blk - 1 - i];
        }
        st[nb_blk] = cur;
        return ggml_inkling_depth_mix(ctx0, st, (int)(nb_blk + 1), query, 1e-6f);
    }

    // [norm(b_n-1), ..., norm(b_0), norm(cur)]
    ggml_tensor * states = mix_stack({cur});
    if (!blocks.empty()) {
        states = ggml_concat(ctx0, mix_stack(blocks), states, 1);
    }

    // one token needs no copy of the queries
    ggml_tensor * queries = n_tokens == 1 ? query :
        ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    // [n_queries, n_embd, n_tokens] -> [n_embd, n_tokens, n_queries]
    ggml_tensor * out = ggml_mul_mat(ctx0, scores, ggml_cont(ctx0, ggml_transpose(ctx0, states)));
    return ggml_cont(ctx0, ggml_permute(ctx0, out, 2, 0, 1, 3));
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    if (cur->ne[2] > 1 || cur->ne[1] != n_tokens) {
        const bool fuse_conv_sum = !getenv("GGML_INKLING_NO_FUSE") && !getenv("GGML_VK_DISABLE_S19_CONV_SUM") &&
                                   n_conv == 4 && n_seq_tokens <= 16 && inp != nullptr;
        if (!fuse_conv_sum) {
            const int64_t n_exp = cur->ne[1];
            ggml_tensor * s = ggml_view_2d(ctx0, cur, n_channels, n_tokens, cur->nb[2], 0);
            for (int64_t e = 1; e < n_exp; ++e) {
                s = ggml_add(ctx0, s, ggml_view_2d(ctx0, cur, n_channels, n_tokens, cur->nb[2], e * cur->nb[1]));
            }
            cur = s;
        }
    }
    if (!getenv("GGML_INKLING_NO_FUSE") && n_conv == 4) {
        const auto * mctx_cur = inp->mctx;
        const size_t row_size = state->nb[1];
        const size_t cell_offs = mctx_cur->get_head() * row_size + offset * (n_conv - 1) * sizeof(float);
        const int64_t K  = cparams.n_rs_seq == 0 ? 1 : ((int64_t)cparams.n_rs_seq + 1);
        const int64_t Kw = std::min(n_seq_tokens, K);
        const int32_t n_heads = dynamic ? (int32_t)(dynamic->ne[0] / n_conv) : 0;
        const float scale_out = 1.0f / sqrtf((dynamic ? 2 : 1) * n_conv + 1);
        return ggml_inkling_conv(
            ctx0, cur, history, weight, dynamic, state,
            (int32_t)offset, (int32_t)n_seq_tokens, (int32_t)n_seqs, n_heads,
            (int32_t)(cell_offs / sizeof(float)),
            (int32_t)(row_size / sizeof(float)),
            (int32_t)(((size_t)mctx_cur->get_size() * row_size) / sizeof(float)),
            (int32_t)Kw, scale_out);
    }

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_3d(ctx0, dynamic, 1, n_heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_3d(ctx0, x, n_seq_tokens, n_channels, n_seqs, x->nb[1], x->nb[2], (n_conv - 1 - lag)*x->nb[0]);

            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, n_channels/n_heads, n_heads, n_tokens);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * message, const loop_layer & loop, int ctrl) {
    if (!getenv("GGML_INKLING_NO_FUSE")) {
        ggml_tensor * gate       = ggml_inkling_rms2_concat(ctx0, cur, message, 1e-6f);
        gate                     = build_lora_mm(loop.ctrl_down[ctrl], gate);
        ggml_tensor * raw_gain   = build_lora_mm(loop.ctrl_gain[ctrl],   gate);
        ggml_tensor * raw_forget = build_lora_mm(loop.ctrl_forget[ctrl], gate);
        return ggml_inkling_ctrl_apply(ctx0, cur, message, raw_gain, raw_forget);
    }
    // cur and message as two rows of one rms_norm, then side by side: [norm(cur); norm(message)]
    ggml_tensor * gate = ggml_concat(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), ggml_reshape_3d(ctx0, message, n_embd, 1, n_tokens), 1);
    gate = ggml_reshape_2d(ctx0, ggml_rms_norm(ctx0, gate, 1e-6f), 2*n_embd, n_tokens);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));

    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    attn = inp->get_attn();
    recr = inp->get_recr();

    ggml_tensor * inp_pos = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT(!messages == !loop);

    if (!model.s19_inkling_ready && !messages) {
        s19_ensure_inkling2_norms(model);
    }
    const auto & layer = model.layers[il];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // the message of controller ctrl, [n_embd, n_tokens] (contiguous slice of [n_embd, n_tokens, LOOP_CTRL_COUNT]!)
    auto message = [&](int ctrl) {
        return ggml_view_2d(ctx0, messages, n_embd, n_tokens, messages->nb[1], ctrl*messages->nb[2]);
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    ggml_tensor * inpSA = cur;

    const bool fuse_s19 = n_tokens <= 512 && !getenv("GGML_INKLING_NO_FUSE") && !getenv("S19_NO_INK_W3") && blocks.size() + 1 <= 6 && model.s19_inkling_ready;
    const bool fuse_mix_rms = fuse_s19 && !getenv("S19_NO_MIX_RMS");
    const bool has_s19_loop = fuse_s19 && !getenv("S19_NO_CTRL_RMS") && messages && n_tokens <= 8 && model.loras.empty() &&
                              ip >= 0 && (size_t)ip < model.s19_loop_layers.size() &&
                              model.s19_loop_layers[ip].ctrl_down_qkv != nullptr;

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs[3] = {cur, cur, cur};
    ggml_tensor * qkv_rms_w = is_swa ? layer.attn_norm : layer.s19_attn_qkv_inp_norm;
    if (!messages && fuse_mix_rms && layer.attn_res_query && qkv_rms_w) {
        ggml_tensor * mixed = s19_inkling_mix_rms(ctx0, blocks, cur, layer.attn_res_query, qkv_rms_w, hparams.f_norm_rms_eps);
        if (is_swa) {
            inputs[0] = ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[1], 0);
            inputs[1] = inputs[2] = inputs[0];
        } else {
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[1], j*mixed->nb[2]);
            }
        }
    } else if (has_s19_loop && layer.attn_res_query && qkv_rms_w) {
        const auto & s19_l = model.s19_loop_layers[ip];
        ggml_tensor * mixed    = mix(blocks, cur, layer.attn_res_query);
        ggml_tensor * msgs_qkv = ggml_view_3d(ctx0, messages, n_embd, n_tokens, 3, messages->nb[1], messages->nb[2], 0);
        ggml_tensor * gate_in  = ggml_inkling_rms2_concat(ctx0, mixed, msgs_qkv, 1e-6f);
        ggml_tensor * gate     = ggml_mul_mat(ctx0, s19_l.ctrl_down_qkv, gate_in);
        ggml_tensor * up_gf    = ggml_mul_mat(ctx0, s19_l.ctrl_up_qkv,   gate);
        ggml_tensor * normed   = ggml_inkling_ctrl_apply_rms_mul(ctx0, mixed, msgs_qkv, up_gf, qkv_rms_w, hparams.f_norm_rms_eps);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = ggml_view_2d(ctx0, normed, n_embd, n_tokens, normed->nb[1], j*normed->nb[2]);
        }
    } else {
        if (layer.attn_res_query) {
            ggml_tensor * mixed = mix(blocks, cur, layer.attn_res_query);
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[1], (is_swa ? 0 : j)*mixed->nb[2]);
            }
        }

        if (messages) {
            // the messages make the Q, K and V inputs of a SWA layer differ
            for (int j = 0; j < 3; ++j) {
                inputs[j] = control(inputs[j], message(LOOP_CTRL_Q + j), *loop, LOOP_CTRL_Q + j);
            }
        }

        if (is_swa && messages) {
            for (int j = 0; j < 3; ++j) {
                inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            }
        } else if (is_swa) {
            inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = inputs[2] = inputs[0];
        } else {
            inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
            inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
        }
    }

    // self-attention
    {
        ggml_tensor * Qcur      = build_lora_mm(layer.wq,             inputs[0]);
        ggml_tensor * Qgate_raw = build_lora_mm(layer.wqkv_gate,      inputs[0]);
        ggml_tensor * Qgate     = ggml_sigmoid(ctx0, Qgate_raw);
        ggml_tensor * Rcur      = nullptr;
        ggml_tensor * sinks     = nullptr;
        ggml_tensor * Kraw      = nullptr;
        ggml_tensor * Kdyn      = nullptr;
        ggml_tensor * Vraw      = nullptr;
        ggml_tensor * Vdyn      = nullptr;
        if (n_tokens <= 8 && !getenv("S19_NO_WKV_REORDER")) {
            Kraw  = build_lora_mm(layer.wk,             inputs[1]);
            Vraw  = build_lora_mm(layer.wv,             inputs[2]);
            Rcur  = build_lora_mm(layer.attn_rel,       inputs[0]);
            sinks = build_lora_mm(layer.attn_sink_proj, inputs[0]);
            Kdyn  = is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]);
            Vdyn  = is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]);
        } else {
            Rcur  = build_lora_mm(layer.attn_rel,       inputs[0]);
            sinks = build_lora_mm(layer.attn_sink_proj, inputs[0]);
            Kraw  = build_lora_mm(layer.wk,             inputs[1]);
            Kdyn  = is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]);
            Vraw  = build_lora_mm(layer.wv,             inputs[2]);
            Vdyn  = is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]);
        }

        ggml_tensor * Kcur      = convolve(Kraw, layer.attn_k_conv, Kdyn, state, history, recr, 0);
        ggml_tensor * Vcur      = convolve(Vraw, layer.attn_v_conv, Vdyn, state, history, recr, n_embd_k_gqa);

        ggml_tensor * opt_tau = is_swa ? nullptr : tau;
        if (n_tokens <= 8 && getenv("GGML_VK_ENABLE_S19_CLIP_QR") && !getenv("GGML_INKLING_NO_FUSE") && !getenv("S19_NO_INK_W3")) {
            ggml_tensor * Q3      = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            ggml_tensor * R3      = ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens);
            ggml_tensor * qr_clip = ggml_inkling_l2_clip_qr(
                    ctx0, Q3, R3, layer.attn_q_clip, layer.attn_rel_clip, opt_tau,
                    hparams.f_qk_clip, hparams.f_relative_clip, 1e-6f);
            Qcur = ggml_view_3d(ctx0, qr_clip, n_embd_head, n_head, n_tokens,
                                n_embd_head * sizeof(float), n_embd_head * n_head * sizeof(float), 0);
            Rcur = ggml_view_3d(ctx0, qr_clip, hparams.n_relative, n_head, n_tokens,
                                hparams.n_relative * sizeof(float), hparams.n_relative * n_head * sizeof(float),
                                (size_t)(n_embd_head * n_head * n_tokens) * sizeof(float));
        } else {
            Qcur = clip(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_clip,   hparams.f_qk_clip,       opt_tau);
            Rcur = clip(ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens), layer.attn_rel_clip, hparams.f_relative_clip, opt_tau);
        }
        Kcur = clip(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr,             hparams.f_qk_clip,       nullptr);
        Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps);

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

        const int64_t n_stream = (is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw)->ne[3];
        const int64_t n_kv     = rel_pos->ne[0];

        const int64_t n_extent = layer.attn_rel_bias->ne[1];
        const int64_t n_seq_tokens = n_tokens/n_stream;
        const int64_t n_kv_max = std::max<int64_t>(n_kv, (is_swa ? attn->mctx->get_swa() : attn->mctx->get_base())->get_size());
        int64_t n_rows_max = std::max({ int64_t(1), n_kq_bias_max/(n_kv_max*n_head*n_stream),
                (n_seq_tokens + n_kq_chunk_max - 1)/n_kq_chunk_max });
        if (n_rows_max < n_seq_tokens) {
            const int64_t n_chunks = (n_seq_tokens + n_rows_max - 1)/n_rows_max;
            n_rows_max = std::min(n_seq_tokens, GGML_PAD((n_seq_tokens + n_chunks - 1)/n_chunks, 64));
        }
        auto rows = [&](ggml_tensor * t, int d, int64_t r0, int64_t nr) {
            int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
            ne[d] = nr;
            return ggml_cont(ctx0, ggml_view_4d(ctx0, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], r0*t->nb[d]));
        };

        ggml_tensor * sinks_raw  = sinks;
        ggml_tensor * table      = nullptr;
        const bool fuse_rel      = !getenv("GGML_INKLING_NO_FUSE");
        int64_t eff_rows         = n_extent;
        ggml_tensor * rel_bias_v = layer.attn_rel_bias;
        bool chunk_table         = false;
        if (fuse_rel) {
            if (!is_swa && ubatch.pos != nullptr && ubatch.n_tokens > 0) {
                llama_pos max_p = 0;
                bool non_neg = true;
                for (uint32_t ti = 0; ti < ubatch.n_tokens; ++ti) {
                    if (ubatch.pos[ti] < 0) { non_neg = false; break; }
                    if (ubatch.pos[ti] > max_p) max_p = ubatch.pos[ti];
                }
                if (non_neg) {
                    eff_rows = std::min<int64_t>(n_extent, std::max<int64_t>(256, GGML_PAD((int64_t)max_p + 1, 256)));
                }
            }
            rel_bias_v = eff_rows < n_extent
                ? ggml_view_2d(ctx0, layer.attn_rel_bias, layer.attn_rel_bias->ne[0], eff_rows, layer.attn_rel_bias->nb[1], 0)
                : layer.attn_rel_bias;
            chunk_table = (n_rows_max < n_seq_tokens) &&
                          (eff_rows * n_head * n_tokens * (int64_t)sizeof(float) > (256LL << 20));
            if (!chunk_table) {
                table = ggml_mul_mat(ctx0, rel_bias_v, Rcur);
            }
        } else {
            table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);
            table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks_raw, 1, n_head, n_tokens));
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));
        }

        // one zero tensor per head count for the whole graph
        ggml_tensor *& zero = zero_sinks[n_head];
        if (!zero) {
            zero = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
        }
        sinks = zero;

        std::vector<ggml_tensor *> outs;
        for (int64_t r0 = 0; r0 < n_seq_tokens; r0 += n_rows_max) {
            const int64_t nr = std::min(n_rows_max, n_seq_tokens - r0);
            const bool all = nr == n_seq_tokens;
            ggml_tensor * q_c = Qcur;
            ggml_tensor * kq_b = nullptr;
            if (fuse_rel) {
                ggml_tensor *& rel_pos_dev = inp->self_rel_pos_dev[is_swa];
                if (!rel_pos_dev) {
                    rel_pos_dev = ggml_cont(ctx0, rel_pos);
                }
                if (!all) {
                    q_c = ggml_reshape_3d(ctx0, rows(ggml_reshape_4d(ctx0, Qcur, n_embd_head, n_head, n_seq_tokens, n_stream), 2, r0, nr),
                            n_embd_head, n_head, nr*n_stream);
                }
                if (chunk_table) {
                    ggml_tensor * Rcur_c = n_stream == 1
                        ? ggml_view_3d(ctx0, Rcur, hparams.n_relative, n_head, nr, Rcur->nb[1], Rcur->nb[2], r0 * Rcur->nb[2])
                        : ggml_reshape_3d(ctx0, rows(ggml_reshape_4d(ctx0, Rcur, hparams.n_relative, n_head, n_seq_tokens, n_stream), 2, r0, nr),
                                hparams.n_relative, n_head, nr*n_stream);
                    ggml_tensor * table_c = ggml_mul_mat(ctx0, rel_bias_v, Rcur_c);
                    ggml_tensor * sinks_c = n_stream == 1
                        ? ggml_view_2d(ctx0, sinks_raw, n_head, nr, sinks_raw->nb[1], r0 * sinks_raw->nb[1])
                        : ggml_reshape_2d(ctx0, rows(ggml_reshape_4d(ctx0, sinks_raw, n_head, 1, n_seq_tokens, n_stream), 2, r0, nr),
                                n_head, nr*n_stream);
                    ggml_tensor * rel_pos_c = n_stream == 1
                        ? ggml_view_2d(ctx0, rel_pos_dev, n_kv, nr, rel_pos_dev->nb[1], r0 * rel_pos_dev->nb[1])
                        : ggml_reshape_2d(ctx0, rows(ggml_reshape_4d(ctx0, rel_pos_dev, n_kv, 1, n_seq_tokens, n_stream), 2, r0, nr),
                                n_kv, nr*n_stream);
                    kq_b = ggml_inkling_rel_gather(ctx0, table_c, sinks_c, rel_pos_c, n_kv, nr, n_head, n_stream,
                                                   (int32_t)nr, 0, (int32_t)n_extent);
                } else {
                    kq_b = ggml_inkling_rel_gather(ctx0, table, sinks_raw, rel_pos_dev, n_kv, nr, n_head, n_stream,
                                                   (int32_t)n_seq_tokens, (int32_t)r0, (int32_t)n_extent);
                }
            } else if (all) {
                kq_b = ggml_get_rows(ctx0, table, rel_pos);
                kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, nr, n_stream);
                kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
            } else {
                q_c = ggml_reshape_3d(ctx0, rows(ggml_reshape_4d(ctx0, Qcur, n_embd_head, n_head, n_seq_tokens, n_stream), 2, r0, nr),
                        n_embd_head, n_head, nr*n_stream);
                ggml_tensor * table_c = ggml_reshape_4d(ctx0, table, table->ne[0], table->ne[1], n_seq_tokens, n_stream);
                table_c = ggml_view_4d(ctx0, table_c, table_c->ne[0], table_c->ne[1], nr, n_stream,
                        table_c->nb[1], table_c->nb[2], table_c->nb[3], r0*table_c->nb[2]);
                ggml_tensor *& rel_pos_dev = inp->self_rel_pos_dev[is_swa];
                if (!rel_pos_dev) {
                    rel_pos_dev = ggml_cont(ctx0, rel_pos);
                }
                ggml_tensor * rel_pos_c = ggml_view_3d(ctx0, rel_pos_dev, n_kv, nr, n_stream,
                        rel_pos_dev->nb[1], n_seq_tokens*rel_pos_dev->nb[1], r0*rel_pos_dev->nb[1]);
                kq_b = ggml_cont(ctx0, ggml_permute(ctx0, ggml_get_rows(ctx0, table_c, rel_pos_c), 2, 0, 1, 3));
            }
            cb(kq_b, "kq_b", il);

            // only the first chunk stores K and V
            ggml_tensor * out = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    q_c, r0 == 0 ? Kcur : nullptr, r0 == 0 ? Vcur : nullptr, kq_b, sinks, 1.0f/sqrtf(n_embd_head), il, r0);
            outs.push_back(all ? out : ggml_reshape_3d(ctx0, out, out->ne[0], nr, n_stream));
        }
        while (outs.size() > 1) {
            std::vector<ggml_tensor *> next;
            for (size_t c = 0; c + 1 < outs.size(); c += 2) {
                next.push_back(ggml_concat(ctx0, outs[c], outs[c + 1], 1));
            }
            if (outs.size() % 2) {
                next.push_back(outs.back());
            }
            outs = std::move(next);
        }
        cur = n_rows_max >= n_seq_tokens ? outs[0] : ggml_reshape_2d(ctx0, outs[0], outs[0]->ne[0], n_tokens);

        cur = ggml_mul(ctx0, cur, Qgate);
        cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa);
        cb(cur, "attn_out", il);
    }

    // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
    if (ip % hparams.attn_res_block_size == 0) {
        blocks.push_back(inpSA);
    } else if (cur->op == GGML_OP_INKLING_FUSED && cur->op_params[0] == 1 &&
               cur->src[5] == nullptr && !getenv("GGML_INKLING_NO_RES_FUSE")) {
        cur->src[5] = inpSA;
    } else {
        cur = ggml_add(ctx0, inpSA, cur);
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    if (!messages && fuse_mix_rms && layer.ffn_norm) {
        cur = ggml_reshape_2d(ctx0, s19_inkling_mix_rms(ctx0, blocks, cur, layer.ffn_res_query, layer.ffn_norm, hparams.f_norm_rms_eps), n_embd, n_tokens);
    } else if (has_s19_loop && layer.ffn_norm && model.s19_loop_layers[ip].ctrl_up_ffn) {
        const auto & s19_l = model.s19_loop_layers[ip];
        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query), n_embd, n_tokens);
        ggml_tensor * msg_ffn = message(LOOP_CTRL_FFN);
        ggml_tensor * gate_in = ggml_inkling_rms2_concat(ctx0, cur, msg_ffn, 1e-6f);
        ggml_tensor * gate    = ggml_mul_mat(ctx0, loop->ctrl_down[LOOP_CTRL_FFN], gate_in);
        ggml_tensor * up_gf   = ggml_mul_mat(ctx0, s19_l.ctrl_up_ffn, gate);
        cur = ggml_inkling_ctrl_apply_rms_mul(ctx0, cur, msg_ffn, up_gf, layer.ffn_norm, hparams.f_norm_rms_eps);
    } else {
        cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query), n_embd, n_tokens);
        if (messages) {
            cur = control(cur, message(LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN);
        }
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    }

    cur = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used(il),
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            layer.ffn_gate_up_exps);

    cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd);
    if (cur->op == GGML_OP_INKLING_FUSED && cur->op_params[0] == 1 &&
        cur->src[5] == nullptr && !getenv("GGML_INKLING_NO_RES_FUSE")) {
        cur->src[5] = ffn_inp;
    } else {
        cb(cur, "ffn_out", il);
        cur = ggml_add(ctx0, ffn_inp, cur);
    }
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    // DSpark target features: the last one is the final norm output of every token
    res->t_layer_inp[n_layer] = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // Series 19 (W1.3 / W6 / W10): Bind rs_s_copy (src[6]) onto mode=1 inkling_conv nodes when n_rs_seq == 0
    if (recr && recr->s_copy && cparams.n_rs_seq == 0 && ubatch.n_seq_tokens <= 16 && !getenv("S19_NO_SCOPY_BIND")) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx0); t != nullptr; t = ggml_get_next_tensor(ctx0, t)) {
            if (t->op == GGML_OP_INKLING_FUSED && t->op_params[0] == 1 &&
                t->src[1] && t->src[4] && t->src[6] == nullptr) {
                const ggml_tensor * vs = t->src[1]->view_src;
                if (t->src[1] == t->src[4] || vs == t->src[4] || (vs && vs->view_src == t->src[4])) {
                    t->src[6] = recr->s_copy;
                }
            }
        }
    }

    // Series 19 (W1.1): Slice model.output to n_vocab_unpadded before build_lora_mm (0-ULP, eliminates trailing CPY)
    ggml_tensor * w_out   = model.output;
    ggml_tensor * w_out_s = model.output_s;
    if (w_out->ne[1] != hparams.n_vocab_unpadded &&
        (!hdmd || hdmd->rot.find(w_out) == hdmd->rot.end()) &&
        (!loras || loras->empty())) {
        w_out = ggml_view_2d(ctx0, w_out, w_out->ne[0], hparams.n_vocab_unpadded, w_out->nb[1], 0);
        if (w_out_s) {
            w_out_s = ggml_view_1d(ctx0, w_out_s, hparams.n_vocab_unpadded, 0);
        }
    }
    cur = build_lora_mm(w_out, cur, w_out_s);
    ggml_prec_set_acc(cur, GGML_PREC_F32);

    extern bool llama_s19_is_building_greedy_argmax(void);
    const bool greedy_argmax_only = llama_s19_is_building_greedy_argmax() &&
        n_outputs > 0 && samplers.empty() && !getenv("GGML_INKLING_NO_FUSE") && !getenv("S19_NO_INK_W3");

    if (greedy_argmax_only) {
        if (cur->ne[0] != hparams.n_vocab_unpadded) {
            cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));
        }
        if (hparams.f_final_logit_softcapping != 0.0f) {
            const float cap = hparams.f_final_logit_softcapping;
            cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
        }
        ggml_tensor * tok_ids = ggml_argmax(ctx0, cur);
        ggml_set_output(tok_ids);
        res->t_sampled = { tok_ids };
        res->t_logits  = nullptr;
        ggml_build_forward_expand(gf, tok_ids);
        return;
    }

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;
        if (!getenv("GGML_INKLING_NO_FUSE") && ggml_is_contiguous(cur)) {
            cur = ggml_inkling_softcap(ctx0, cur, 1.0f/cap, cap);
        } else {
            cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
        }
    }

    if (cur->ne[0] != hparams.n_vocab_unpadded) {
        cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inputs();

    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        // DSpark target features: the partial block at the layer input
        res->t_layer_inp[il] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);
}
