#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Exact per-node differential bisector between CPU and AGX0 on Inkling 2 Mini.
// Usage:
//   test_agx_bisect <model.gguf> CPU  /tmp/bisect_cpu.bin   (dumps every node's raw bytes)
//   test_agx_bisect <model.gguf> AGX0 /tmp/bisect_cpu.bin   (compares every node live vs CPU dump)

struct DumpHeader {
    uint32_t op;
    uint32_t type;
    uint32_t ne[4];
    uint64_t nbytes;
    char     name[64];
};

struct BisectState {
    bool is_dump = false;
    FILE * fp = nullptr;
    int node_idx = 0;
    int n_diverged = 0;
    int max_print = 30;
    std::vector<uint8_t> cand_buf;
    std::vector<uint8_t> ref_buf;
};

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<BisectState *>(user_data);
    if (ask) {
        return true; // observe every node right after it completes!
    }

    const size_t nb = ggml_nbytes(t);
    const size_t cap = std::min<size_t>(nb, 4u * 1024u * 1024u); // cap at 4 MiB per node

    if (st->is_dump) {
        DumpHeader hdr{};
        hdr.op = static_cast<uint32_t>(t->op);
        hdr.type = static_cast<uint32_t>(t->type);
        for (int d = 0; d < 4; ++d) hdr.ne[d] = static_cast<uint32_t>(t->ne[d]);
        hdr.nbytes = cap;
        std::strncpy(hdr.name, t->name, sizeof(hdr.name) - 1);
        std::fwrite(&hdr, sizeof(hdr), 1, st->fp);

        st->ref_buf.resize(cap);
        if (cap > 0) {
            if (t->buffer) {
                ggml_backend_tensor_get(t, st->ref_buf.data(), 0, cap);
            } else if (t->data) {
                std::memcpy(st->ref_buf.data(), t->data, cap);
            } else {
                std::memset(st->ref_buf.data(), 0, cap);
            }
            std::fwrite(st->ref_buf.data(), 1, cap, st->fp);
        }
        st->node_idx++;
        return true;
    }

    // Compare mode
    DumpHeader hdr{};
    if (std::fread(&hdr, sizeof(hdr), 1, st->fp) != 1) {
        std::printf("[node %4d] EOF in ref dump at %s (%s)\n", st->node_idx, t->name, ggml_op_name(t->op));
        st->node_idx++;
        return true;
    }
    st->ref_buf.resize(hdr.nbytes);
    if (hdr.nbytes > 0) {
        if (std::fread(st->ref_buf.data(), 1, hdr.nbytes, st->fp) != hdr.nbytes) {
            std::printf("[node %4d] short read in ref dump\n", st->node_idx);
            st->node_idx++;
            return true;
        }
    }

    if (hdr.op != static_cast<uint32_t>(t->op) || hdr.type != static_cast<uint32_t>(t->type) ||
        hdr.nbytes != cap) {
        if (st->n_diverged < st->max_print) {
            std::printf("[node %4d] SHAPE/OP MISMATCH: ref=(%s op=%u type=%u nb=%llu) cand=(%s op=%u type=%u nb=%zu)\n",
                        st->node_idx, hdr.name, hdr.op, hdr.type, (unsigned long long) hdr.nbytes,
                        t->name, (unsigned) t->op, (unsigned) t->type, cap);
        }
        st->n_diverged++;
        st->node_idx++;
        return true;
    }

    st->cand_buf.resize(cap);
    if (cap > 0) {
        if (t->buffer) {
            ggml_backend_tensor_get(t, st->cand_buf.data(), 0, cap);
        } else if (t->data) {
            std::memcpy(st->cand_buf.data(), t->data, cap);
        } else {
            std::memset(st->cand_buf.data(), 0, cap);
        }
    }

    bool bad = false;
    double max_abs = 0.0;
    double cos_sim = 1.0;
    int bad_i32 = 0;
    int n_nan = 0;
    float r0 = 0.0f, c0 = 0.0f;
    size_t worst_idx = 0;

    if (t->type == GGML_TYPE_F32 && cap >= 4) {
        const size_t n = cap / 4;
        const float * r = reinterpret_cast<const float *>(st->ref_buf.data());
        const float * c = reinterpret_cast<const float *>(st->cand_buf.data());
        double dot = 0.0, nr = 0.0, nc = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(c[i]) || (std::isinf(c[i]) && (!std::isinf(r[i]) || (r[i] != c[i])))) {
                n_nan++;
                bad = true;
                continue;
            }
            if (!std::isfinite(r[i]) && !std::isfinite(c[i])) {
                continue; // matching +/-Inf (e.g. causal attention mask)
            }
            double d = std::fabs(static_cast<double>(r[i]) - static_cast<double>(c[i]));
            if (d > max_abs) { max_abs = d; worst_idx = i; r0 = r[i]; c0 = c[i]; }
            dot += static_cast<double>(r[i]) * static_cast<double>(c[i]);
            nr  += static_cast<double>(r[i]) * static_cast<double>(r[i]);
            nc  += static_cast<double>(c[i]) * static_cast<double>(c[i]);
        }
        if (nr > 1e-12 && nc > 1e-12) {
            cos_sim = dot / (std::sqrt(nr) * std::sqrt(nc));
        } else if (nr > 1e-12 || nc > 1e-12) {
            cos_sim = 0.0;
        }
        if (n_nan > 0 || (max_abs > 5e-2 && cos_sim < 0.999)) {
            bad = true;
        }
    } else if (t->type == GGML_TYPE_I32 && cap >= 4) {
        const size_t n = cap / 4;
        const int32_t * r = reinterpret_cast<const int32_t *>(st->ref_buf.data());
        const int32_t * c = reinterpret_cast<const int32_t *>(st->cand_buf.data());
        for (size_t i = 0; i < n; ++i) {
            if (r[i] != c[i]) {
                if (bad_i32 == 0) { worst_idx = i; r0 = (float) r[i]; c0 = (float) c[i]; }
                bad_i32++;
            }
        }
        if (bad_i32 > 0) bad = true;
    }

    if (std::strcmp(t->name, "result_output") == 0 || std::strcmp(t->name, "result_norm") == 0) {
        std::printf("[FINAL | node %4d] %-20s op=%-10s ne=[%lld,%lld,%lld,%lld] cos=%.6f max_abs=%.4g nan=%d\n",
                    st->node_idx, t->name, ggml_op_name(t->op),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                    cos_sim, max_abs, n_nan);
    }

    if (bad) {
        if (st->n_diverged < st->max_print) {
            std::printf("[DIVERGE #%2d | node %4d] %-28s op=%-14s ne=[%lld,%lld,%lld,%lld] type=%s "
                        "cos=%.6f max_abs=%.4g nan=%d i32_diff=%d @idx=%zu (ref=%.5g cand=%.5g)",
                        st->n_diverged, st->node_idx, t->name, ggml_op_name(t->op),
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                        ggml_type_name(t->type), cos_sim, max_abs, n_nan, bad_i32, worst_idx, r0, c0);
            std::printf(" | srcs:");
            for (int s = 0; s < 3; ++s) {
                if (t->src[s]) {
                    std::printf(" [%d:%s(%s,%s,contig=%d)]", s, t->src[s]->name,
                                ggml_op_name(t->src[s]->op), ggml_type_name(t->src[s]->type),
                                ggml_is_contiguous(t->src[s]) ? 1 : 0);
                }
            }
            std::printf("\n");
        }
        st->n_diverged++;
    }

    st->node_idx++;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s <model.gguf> <CPU|AGX0> <dump.bin>\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * dev_name   = argv[2];
    const char * dump_path  = argv[3];

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    const bool is_cpu = (std::strcmp(dev_name, "CPU") == 0);
    if (is_cpu) {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_dev) {
            std::fprintf(stderr, "FAIL: CPU backend device not found\n");
            return 1;
        }
        devs[0] = cpu_dev;
        mparams.devices = devs;
        mparams.n_gpu_layers = 0;
    } else {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
        if (!dev) {
            std::fprintf(stderr, "FAIL: backend device '%s' not found\n", dev_name);
            return 1;
        }
        devs[0] = dev;
        mparams.devices = devs;
        mparams.n_gpu_layers = 999;
    }

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "FAIL: could not load model\n");
        return 1;
    }

    BisectState st;
    st.is_dump = is_cpu;
    st.fp = std::fopen(dump_path, is_cpu ? "wb" : "rb");
    if (!st.fp) {
        std::fprintf(stderr, "FAIL: cannot open %s\n", dump_path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 256;
    cparams.n_batch = 32;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.cb_eval = eval_cb;
    cparams.cb_eval_user_data = &st;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: llama_init_from_model failed\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt = (argc >= 5) ? argv[4] : "The capital of France is";
    std::vector<llama_token> tokens(64);
    int n_tok = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()), true, false);
    tokens.resize(static_cast<size_t>(n_tok));

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tok);
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "FAIL: decode failed\n");
        return 1;
    }

    std::printf("DONE dev=%s total_nodes=%d diverged=%d\n", dev_name, st.node_idx, st.n_diverged);
    std::fclose(st.fp);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
