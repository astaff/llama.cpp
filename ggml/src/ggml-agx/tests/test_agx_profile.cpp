#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// Detailed op-kind, shape, byte-volume, and per-node GPU timing profiler for Inkling 2 Mini on AGX0.
// Usage:
//   test_agx_profile <model.gguf> [n_prompt=128] [n_gen=8]

using Clock = std::chrono::steady_clock;

struct OpBucket {
    uint64_t count = 0;
    double   total_us = 0.0;
    uint64_t weight_bytes = 0;
    uint64_t act_bytes = 0;
    std::string sample_shape;
};

struct ProfileState {
    bool enabled = false;
    Clock::time_point t_ask;
    std::map<std::string, OpBucket> by_op;
    std::map<std::string, OpBucket> by_key;
    uint64_t total_nodes = 0;
    uint64_t view_nodes = 0;
    double   total_cb_us = 0.0;
};

static bool is_view_op(enum ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool prof_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<ProfileState *>(user_data);
    if (!st->enabled) return false;
    if (ask) {
        st->t_ask = Clock::now();
        return true;
    }
    const double dt_us = std::chrono::duration<double, std::micro>(Clock::now() - st->t_ask).count();
    st->total_nodes++;
    st->total_cb_us += dt_us;
    if (is_view_op(t->op)) {
        st->view_nodes++;
    }

    const std::string op_name = ggml_op_name(t->op);
    std::string key = op_name;
    uint64_t w_bytes = 0;
    uint64_t a_bytes = ggml_nbytes(t);
    char sbuf[160] = {};

    if (t->op == GGML_OP_MUL_MAT && t->src[0] && t->src[1]) {
        key += ":";
        key += ggml_type_name(t->src[0]->type);
        w_bytes = ggml_nbytes(t->src[0]);
        std::snprintf(sbuf, sizeof(sbuf), "K=%lld N=%lld M=%lld B=%lld (%s)",
                      (long long) t->src[0]->ne[0], (long long) t->src[0]->ne[1],
                      (long long) t->src[1]->ne[1], (long long) (t->src[1]->ne[2] * t->src[1]->ne[3]),
                      ggml_type_name(t->src[0]->type));
    } else if (t->op == GGML_OP_MUL_MAT_ID && t->src[0] && t->src[1] && t->src[2]) {
        key += ":";
        key += ggml_type_name(t->src[0]->type);
        // Active weight bytes for M_tok * top_k experts (upper bound without expert deduplication)
        const uint64_t per_exp_bytes = (t->src[0]->ne[2] > 0)
            ? (ggml_nbytes(t->src[0]) / static_cast<uint64_t>(t->src[0]->ne[2]))
            : 0;
        const uint64_t active_exps = static_cast<uint64_t>(t->src[2]->ne[0] * t->src[2]->ne[1]);
        w_bytes = std::min<uint64_t>(ggml_nbytes(t->src[0]), per_exp_bytes * active_exps);
        std::snprintf(sbuf, sizeof(sbuf), "K=%lld N=%lld E=%lld topk=%lld M=%lld (%s)",
                      (long long) t->src[0]->ne[0], (long long) t->src[0]->ne[1],
                      (long long) t->src[0]->ne[2], (long long) t->src[2]->ne[0],
                      (long long) t->src[2]->ne[1], ggml_type_name(t->src[0]->type));
    } else if (t->op == GGML_OP_FLASH_ATTN_EXT && t->src[0] && t->src[1]) {
        key += ":";
        key += ggml_type_name(t->src[1]->type);
        w_bytes = ggml_nbytes(t->src[1]) + (t->src[2] ? ggml_nbytes(t->src[2]) : 0);
        std::snprintf(sbuf, sizeof(sbuf), "D=%lld Hq=%lld Hkv=%lld Nkv=%lld M=%lld",
                      (long long) t->src[0]->ne[0], (long long) t->src[0]->ne[2],
                      (long long) t->src[1]->ne[2], (long long) t->src[1]->ne[1],
                      (long long) t->src[0]->ne[1]);
    } else {
        std::snprintf(sbuf, sizeof(sbuf), "ne=[%lld,%lld,%lld,%lld] %s",
                      (long long) t->ne[0], (long long) t->ne[1],
                      (long long) t->ne[2], (long long) t->ne[3],
                      ggml_type_name(t->type));
    }

    auto & bo = st->by_op[op_name];
    bo.count++;
    bo.total_us += dt_us;
    bo.weight_bytes += w_bytes;
    bo.act_bytes += a_bytes;
    if (bo.sample_shape.empty()) bo.sample_shape = sbuf;

    std::string full_key = key + " " + sbuf;
    auto & bk = st->by_key[full_key];
    bk.count++;
    bk.total_us += dt_us;
    bk.weight_bytes += w_bytes;
    bk.act_bytes += a_bytes;
    return true;
}

static void print_profile(const char * label, const ProfileState & st, double uninst_ms) {
    std::vector<std::pair<std::string, OpBucket>> ops(st.by_op.begin(), st.by_op.end());
    std::sort(ops.begin(), ops.end(), [](const auto & a, const auto & b) {
        return a.second.total_us > b.second.total_us;
    });

    uint64_t total_w = 0, exec_nodes = st.total_nodes - st.view_nodes;
    for (const auto & kv : ops) total_w += kv.second.weight_bytes;

    std::printf("\n=== PROFILE: %s ===\n", label);
    std::printf("  Uninstrumented wall: %.2f ms (%.2f tok/s if B=1) | Effective weight BW: %.2f GB/s (%.2f MiB active weights)\n",
                uninst_ms, uninst_ms > 0.0 ? 1000.0 / uninst_ms : 0.0,
                uninst_ms > 0.0 ? (static_cast<double>(total_w) / 1e9) / (uninst_ms / 1e3) : 0.0,
                static_cast<double>(total_w) / (1024.0 * 1024.0));
    std::printf("  Graph nodes: %llu total (%llu compute, %llu view-elided) | Per-node sync total: %.2f ms\n",
                (unsigned long long) st.total_nodes, (unsigned long long) exec_nodes,
                (unsigned long long) st.view_nodes, st.total_cb_us / 1000.0);
    std::printf("  %-18s %7s %10s %8s %10s  %s\n", "OP", "COUNT", "SYNC_MS", "US/NODE", "W_MIB", "SAMPLE");
    int shown = 0;
    for (const auto & kv : ops) {
        if (shown++ >= 18) break;
        std::printf("  %-18s %7llu %10.2f %8.1f %10.1f  %s\n",
                    kv.first.c_str(),
                    (unsigned long long) kv.second.count,
                    kv.second.total_us / 1000.0,
                    kv.second.count ? kv.second.total_us / kv.second.count : 0.0,
                    static_cast<double>(kv.second.weight_bytes) / (1024.0 * 1024.0),
                    kv.second.sample_shape.c_str());
    }

    std::vector<std::pair<std::string, OpBucket>> keys(st.by_key.begin(), st.by_key.end());
    std::sort(keys.begin(), keys.end(), [](const auto & a, const auto & b) {
        return a.second.total_us > b.second.total_us;
    });
    std::printf("  -- Top 12 (op + shape) buckets --\n");
    shown = 0;
    for (const auto & kv : keys) {
        if (shown++ >= 12) break;
        std::printf("  %6llu x %7.2f ms (%6.1f us) | %s\n",
                    (unsigned long long) kv.second.count,
                    kv.second.total_us / 1000.0,
                    kv.second.count ? kv.second.total_us / kv.second.count : 0.0,
                    kv.first.c_str());
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.gguf> [n_prompt=128] [n_gen=6]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const int n_prompt = argc >= 3 ? std::atoi(argv[2]) : 128;
    const int n_gen    = argc >= 4 ? std::atoi(argv[3]) : 6;

    llama_backend_init();

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("AGX0");
    if (!dev) {
        std::fprintf(stderr, "FAIL: AGX0 not found\n");
        return 1;
    }
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    llama_model_params mparams = llama_model_default_params();
    mparams.devices = devs;
    mparams.n_gpu_layers = 999;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "FAIL: could not load model\n");
        return 1;
    }

    // 1. Uninstrumented steady-state benchmark (no cb_eval!)
    double pp_ms = 0.0, tg_ms = 0.0;
    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = n_prompt + n_gen + 64;
        cparams.n_batch = std::max(n_prompt, 512);
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) return 1;

        std::vector<llama_token> toks(n_prompt, 100);
        // Warmup 1 token
        llama_token wtok = 100;
        llama_decode(ctx, llama_batch_get_one(&wtok, 1));
        llama_memory_clear(llama_get_memory(ctx), true);

        auto t0 = Clock::now();
        llama_decode(ctx, llama_batch_get_one(toks.data(), n_prompt));
        pp_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

        double sum_tg = 0.0;
        for (int i = 0; i < n_gen; ++i) {
            llama_token tk = 200 + i;
            auto g0 = Clock::now();
            llama_decode(ctx, llama_batch_get_one(&tk, 1));
            double dt = std::chrono::duration<double, std::milli>(Clock::now() - g0).count();
            if (i > 0) sum_tg += dt;
        }
        tg_ms = (n_gen > 1) ? sum_tg / (n_gen - 1) : 0.0;
        std::printf("UNINSTRUMENTED AGX0: pp%d = %.2f ms (%.2f t/s) | B=1 tg = %.2f ms/tok (%.2f t/s)\n",
                    n_prompt, pp_ms, n_prompt * 1000.0 / pp_ms, tg_ms, tg_ms > 0 ? 1000.0 / tg_ms : 0.0);
        llama_free(ctx);
    }

    // 2. Instrumented per-node profile on 1 B=1 decode step and 1 pp step
    {
        ProfileState ps;
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = n_prompt + 64;
        cparams.n_batch = std::max(n_prompt, 512);
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cparams.cb_eval = prof_cb;
        cparams.cb_eval_user_data = &ps;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) return 1;

        std::vector<llama_token> toks(n_prompt, 100);
        ps.enabled = true;
        llama_decode(ctx, llama_batch_get_one(toks.data(), n_prompt));
        ps.enabled = false;
        print_profile("PREFILL (pp)", ps, pp_ms);

        ps = ProfileState{};
        llama_token tk = 201;
        ps.enabled = true;
        llama_decode(ctx, llama_batch_get_one(&tk, 1));
        ps.enabled = false;
        print_profile("DECODE (B=1 tg)", ps, tg_ms);

        llama_free(ctx);
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
