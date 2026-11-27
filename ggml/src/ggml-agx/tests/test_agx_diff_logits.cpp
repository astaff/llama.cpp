#include "llama.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Deterministic differential-logit dumper & 0-NaN/Inf verifier for Inkling 2 Mini.
// Usage:
//   test_agx_diff_logits <model.gguf> <device: CPU|AGX0> <out_logits.bin> [n_gen=4]

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "Usage: %s <model.gguf> <device: CPU|AGX0> <out_logits.bin> [n_gen=4]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * dev_name   = argv[2];
    const char * out_path   = argv[3];
    const int    n_gen      = argc >= 5 ? std::atoi(argv[4]) : 4;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    if (std::strcmp(dev_name, "CPU") == 0) {
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
        std::fprintf(stderr, "FAIL: could not load model %s\n", model_path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 512;
    cparams.n_batch = 64;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: llama_init_from_model failed\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const std::string prompt = "The quick brown fox jumps over the lazy dog.";

    std::vector<llama_token> tokens(prompt.size() + 8);
    int n_tok = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()), true, false);
    if (n_tok <= 0) {
        std::fprintf(stderr, "FAIL: tokenize failed\n");
        return 1;
    }
    tokens.resize(static_cast<size_t>(n_tok));

    FILE * fp = std::fopen(out_path, "wb");
    if (!fp) {
        std::fprintf(stderr, "FAIL: cannot open %s for writing\n", out_path);
        return 1;
    }

    auto check_and_write = [&](int step, const float * logits) -> llama_token {
        int n_nan = 0, n_inf = 0;
        int best_id = 0;
        float best_val = -1e30f;
        for (int i = 0; i < n_vocab; ++i) {
            float v = logits[i];
            if (std::isnan(v)) n_nan++;
            else if (std::isinf(v)) n_inf++;
            else if (v > best_val) { best_val = v; best_id = i; }
        }
        std::printf("[step %d dev=%s] n_vocab=%d nan=%d inf=%d top1=%d (%.4f)\n",
                    step, dev_name, n_vocab, n_nan, n_inf, best_id, best_val);
        std::fwrite(logits, sizeof(float), static_cast<size_t>(n_vocab), fp);
        if (n_nan > 0 || n_inf > 0) {
            std::fprintf(stderr, "FAIL: non-finite logits detected on %s step %d\n", dev_name, step);
            std::exit(1);
        }
        return static_cast<llama_token>(best_id);
    };

    // Prefill prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tok);
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "FAIL: prompt llama_decode failed\n");
        return 1;
    }
    llama_token next = check_and_write(0, llama_get_logits_ith(ctx, -1));

    // Greedy decode steps
    for (int s = 1; s <= n_gen; ++s) {
        llama_batch b1 = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, b1) != 0) {
            std::fprintf(stderr, "FAIL: decode step %d failed\n", s);
            return 1;
        }
        next = check_and_write(s, llama_get_logits_ith(ctx, -1));
    }

    std::fclose(fp);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::printf("PASS: wrote %d steps x %d logits to %s\n", n_gen + 1, n_vocab, out_path);
    return 0;
}
