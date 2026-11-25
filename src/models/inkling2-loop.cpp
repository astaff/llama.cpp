#include "models.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstring>
#include <mutex>

int llama_model_inkling2_loop::layer_phys(int il) const {
    return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);
}

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_hparams(ml);

    ml.get_key(LLM_KV_LOOP_START,     core_start);
    ml.get_key(LLM_KV_LOOP_END,       core_end);
    ml.get_key(LLM_KV_LOOP_COUNT,     n_passes);
    ml.get_key(LLM_KV_LOOP_GATE_RANK, gate_rank);
    ml.get_key(LLM_KV_LOOP_VARYING,   varying, false);

    n_phys = hparams.n_layer();
    const uint32_t block = hparams.attn_res_block_size;
    // a fixed loop needs at least 2 passes, a variable loop at least 1
    if (n_passes < (varying ? 1u : 2u) || !gate_rank || !core_start || core_start >= core_end || core_end > (uint32_t) n_phys ||
        core_start % block || core_end % block || hparams.n_layer_nextn) {
        throw std::runtime_error(format("inkling2_loop: invalid loop core [%u, %u) with block size %u, %u passes or gate rank %u for %d layers",
                core_start, core_end, block, n_passes, gate_rank, n_phys));
    }
    const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);
    if (n_slot > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("inkling2_loop: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));
    }

    // expand to the cache slots before load_tensors() allocates the layers and the memory
    hparams.n_layer_all = n_slot;
    for (int il = n_phys; il < (int) n_slot; ++il) {
        const int p = layer_phys(il);
        hparams.n_head_arr[il]        = hparams.n_head_arr[p];
        hparams.n_head_kv_arr[il]     = hparams.n_head_kv_arr[p];
        hparams.n_ff_arr[il]          = hparams.n_ff_arr[p];
        hparams.n_ff_exp_arr[il]      = hparams.n_ff_exp_arr[p];
        hparams.n_expert_used_arr[il] = hparams.n_expert_used_arr[p];
        hparams.is_swa_impl[il]       = hparams.is_swa_impl[p];
        hparams.is_recr_impl[il]      = hparams.is_recr_impl[p];
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_tensors(ml);

    const int64_t n_embd = hparams.n_embd;
    const int64_t rank   = gate_rank;
    const llm_tensor down[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_DOWN,   LLM_TENSOR_LOOP_K_CTRL_DOWN,   LLM_TENSOR_LOOP_V_CTRL_DOWN,   LLM_TENSOR_LOOP_FFN_CTRL_DOWN   };
    const llm_tensor gain[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_GAIN,   LLM_TENSOR_LOOP_K_CTRL_GAIN,   LLM_TENSOR_LOOP_V_CTRL_GAIN,   LLM_TENSOR_LOOP_FFN_CTRL_GAIN   };
    const llm_tensor forget[LOOP_CTRL_COUNT] = { LLM_TENSOR_LOOP_Q_CTRL_FORGET, LLM_TENSOR_LOOP_K_CTRL_FORGET, LLM_TENSOR_LOOP_V_CTRL_FORGET, LLM_TENSOR_LOOP_FFN_CTRL_FORGET };

    loop_layers.resize(core_end - core_start);
    for (uint32_t il = core_start; il < core_end; ++il) {
        auto & l = loop_layers[il - core_start];
        // the q, k, v and ffn message queries
        l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, 0);
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            l.ctrl_down[c]   = create_tensor(tn(down[c],   "weight", il), {2*n_embd, rank}, 0);
            l.ctrl_gain[c]   = create_tensor(tn(gain[c],   "weight", il), {rank, n_embd}, 0);
            l.ctrl_forget[c] = create_tensor(tn(forget[c], "weight", il), {rank, n_embd}, 0);
        }
    }
}

void llama_model_inkling2_loop::post_arch_tensors(llama_model_loader &) {
    // share the physical weights, each cache slot keeps its own KV and conv state
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_inkling2_loop::graph::graph(const llm_graph_params & params) : llama_model_inkling2::graph(params, no_build{}) {
    const auto & m = static_cast<const llama_model_inkling2_loop &>(model);
    const int start = m.core_start;
    const int end   = m.core_end;

    // S19_WAVE3: lazily materialize pass_msg_query [n_embd, 48], QKV/FFN coalesced loop weights, & FullAttn norms
    if (!m.s19_inkling_ready && !getenv("GGML_INKLING_NO_FUSE") && !m.layers.empty()) {
        static std::mutex s19_loop_mtx;
        std::lock_guard<std::mutex> lock(s19_loop_mtx);
        const ggml_tensor * ref = m.layers[start].ffn_norm;
        if (!m.s19_inkling_ready && ref && ref->buffer && ref->data &&
            ggml_backend_buffer_get_size(ref->buffer) > 0) {
            const int n_loop   = end - start;
            const int64_t rank = m.gate_rank;
            int n_full = 0;
            for (int p = 0; p < m.n_phys; ++p) {
                if (!m.hparams.is_swa(p) && m.layers[p].attn_q_inp_norm) {
                    ++n_full;
                }
            }
            const size_t max_tensors = (size_t)(1 + 3 * n_loop + n_full + 4);
            ggml_init_params init_p = { ggml_tensor_overhead() * max_tensors, nullptr, true };
            ggml_context_ptr ctx(ggml_init(init_p));

            ggml_tensor * pass_mq = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, (int64_t)LOOP_CTRL_COUNT * n_loop);
            std::vector<llama_model::s19_loop_coalesced> coalesced(m.n_phys);
            for (int ip = start; ip < end; ++ip) {
                const auto & ll = m.loop_layers[ip - start];
                const ggml_type down_t = ll.ctrl_down[0]->type;
                const ggml_type up_t   = ll.ctrl_gain[0]->type;
                const ggml_type ffn_t  = ll.ctrl_gain[3]->type;
                if (ll.ctrl_down[1]->type == down_t && ll.ctrl_down[2]->type == down_t &&
                    ll.ctrl_gain[1]->type == up_t   && ll.ctrl_gain[2]->type == up_t   &&
                    ll.ctrl_forget[0]->type == up_t && ll.ctrl_forget[1]->type == up_t && ll.ctrl_forget[2]->type == up_t &&
                    ll.ctrl_forget[3]->type == ffn_t) {
                    coalesced[ip].ctrl_down_qkv = ggml_new_tensor_3d(ctx.get(), down_t, 2 * n_embd, rank, 3);
                    coalesced[ip].ctrl_up_qkv   = ggml_new_tensor_3d(ctx.get(), up_t,   rank, 2 * n_embd, 3);
                    coalesced[ip].ctrl_up_ffn   = ggml_new_tensor_2d(ctx.get(), ffn_t,  rank, 2 * n_embd);
                }
            }
            std::vector<std::pair<int, ggml_tensor *>> full_norms;
            for (int p = 0; p < m.n_phys; ++p) {
                if (!m.hparams.is_swa(p) && m.layers[p].attn_q_inp_norm) {
                    ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, 3);
                    const_cast<llama_layer &>(m.layers[p]).s19_attn_qkv_inp_norm = t;
                    full_norms.push_back({ p, t });
                }
            }
            for (int il = m.n_phys; il < (int)m.layers.size(); ++il) {
                const int p = m.layer_phys(il);
                const_cast<llama_layer &>(m.layers[il]).s19_attn_qkv_inp_norm = m.layers[p].s19_attn_qkv_inp_norm;
            }

            ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(m.dev_layer(start));
            if (!buft) {
                buft = ggml_backend_buffer_get_type(ref->buffer);
            }
            ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
            if (!buf) {
                for (auto & layer : m.layers) {
                    const_cast<llama_layer &>(layer).s19_attn_qkv_inp_norm = nullptr;
                }
            } else {
                ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

                std::vector<uint8_t> stage;
                const size_t mq_bytes = (size_t)(LOOP_CTRL_COUNT * n_embd * sizeof(float));
                stage.resize(mq_bytes);
                for (int li = 0; li < n_loop; ++li) {
                    ggml_backend_tensor_get(m.loop_layers[li].msg_query, stage.data(), 0, mq_bytes);
                    ggml_backend_tensor_set(pass_mq, stage.data(), (size_t)li * mq_bytes, mq_bytes);
                }

                for (int ip = start; ip < end; ++ip) {
                    const auto & ll = m.loop_layers[ip - start];
                    auto & dst_c    = coalesced[ip];
                    if (!dst_c.ctrl_down_qkv) {
                        continue;
                    }
                    const size_t down_sl_bytes = ggml_nbytes(ll.ctrl_down[0]);
                    if (stage.size() < down_sl_bytes) stage.resize(down_sl_bytes);
                    for (int c = 0; c < 3; ++c) {
                        ggml_backend_tensor_get(ll.ctrl_down[c], stage.data(), 0, down_sl_bytes);
                        ggml_backend_tensor_set(dst_c.ctrl_down_qkv, stage.data(), (size_t)c * down_sl_bytes, down_sl_bytes);
                    }
                    const size_t up_half_bytes = ggml_nbytes(ll.ctrl_gain[0]);
                    if (stage.size() < up_half_bytes) stage.resize(up_half_bytes);
                    for (int c = 0; c < 3; ++c) {
                        const size_t base = (size_t)c * 2 * up_half_bytes;
                        ggml_backend_tensor_get(ll.ctrl_gain[c],   stage.data(), 0, up_half_bytes);
                        ggml_backend_tensor_set(dst_c.ctrl_up_qkv, stage.data(), base, up_half_bytes);
                        ggml_backend_tensor_get(ll.ctrl_forget[c], stage.data(), 0, up_half_bytes);
                        ggml_backend_tensor_set(dst_c.ctrl_up_qkv, stage.data(), base + up_half_bytes, up_half_bytes);
                    }
                    const size_t ffn_half_bytes = ggml_nbytes(ll.ctrl_gain[3]);
                    if (stage.size() < ffn_half_bytes) stage.resize(ffn_half_bytes);
                    ggml_backend_tensor_get(ll.ctrl_gain[3],   stage.data(), 0, ffn_half_bytes);
                    ggml_backend_tensor_set(dst_c.ctrl_up_ffn, stage.data(), 0, ffn_half_bytes);
                    ggml_backend_tensor_get(ll.ctrl_forget[3], stage.data(), 0, ffn_half_bytes);
                    ggml_backend_tensor_set(dst_c.ctrl_up_ffn, stage.data(), ffn_half_bytes, ffn_half_bytes);
                }

                const size_t norm_bytes = (size_t)(n_embd * sizeof(float));
                if (stage.size() < 3 * norm_bytes) stage.resize(3 * norm_bytes);
                for (const auto & pr : full_norms) {
                    const auto & l = m.layers[pr.first];
                    ggml_backend_tensor_get(l.attn_q_inp_norm, stage.data() + 0 * norm_bytes, 0, norm_bytes);
                    ggml_backend_tensor_get(l.attn_k_inp_norm, stage.data() + 1 * norm_bytes, 0, norm_bytes);
                    ggml_backend_tensor_get(l.attn_v_inp_norm, stage.data() + 2 * norm_bytes, 0, norm_bytes);
                    ggml_backend_tensor_set(pr.second, stage.data(), 0, 3 * norm_bytes);
                }

                m.s19_inkling_ctx    = std::move(ctx);
                m.s19_inkling_buf    = std::move(buf);
                m.s19_pass_msg_query = pass_mq;
                m.s19_loop_layers    = std::move(coalesced);
                m.s19_inkling_ready  = true;
            }
        }
    }

    auto * cur = build_inputs();
    std::vector<ggml_tensor *> blocks;

    // DSpark target features are indexed by layer visit, not by cache slot
    int visit = 0;

    for (int il = 0; il < start; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    // every pass starts from the prelude output
    auto * prelude = cur;
    const auto prelude_blocks = blocks;
    for (uint32_t pass = 0; pass < m.n_passes; ++pass) {
        // the bank of the previous pass: its blocks after the first core block, and its output
        std::vector<ggml_tensor *> bank;
        ggml_tensor * bank_out = nullptr;
        ggml_tensor * pass_msgs = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
            if (n_tokens <= 8 && !getenv("GGML_INKLING_NO_FUSE") && !getenv("S19_NO_INK_W3") && !getenv("S19_NO_PASS_MSG") && m.s19_inkling_ready && m.s19_pass_msg_query) {
                pass_msgs = mix(bank, bank_out, m.s19_pass_msg_query);
            }
        }
        for (int ip = start; ip < end; ++ip) {
            const int il = pass == 0 ? ip : m.n_phys + (pass - 1)*(end - start) + (ip - start);
            res->t_layer_inp[visit++] = cur;
            if (pass == 0) {
                cur = build_layer(il, ip, cur, blocks);
                continue;
            }
            const auto & loop = m.loop_layers[ip - start];
            ggml_tensor * msgs = pass_msgs
                ? ggml_view_3d(ctx0, pass_msgs, n_embd, n_tokens, LOOP_CTRL_COUNT,
                               pass_msgs->nb[1], pass_msgs->nb[2],
                               (size_t)((ip - start) * LOOP_CTRL_COUNT) * pass_msgs->nb[2])
                : mix(bank, bank_out, loop.msg_query);
            cur = build_layer(il, ip, cur, blocks, msgs, &loop);
        }
    }

    for (int il = end; il < m.n_phys; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }
    GGML_ASSERT(visit == n_layer);
    build_output(cur, blocks);
}
