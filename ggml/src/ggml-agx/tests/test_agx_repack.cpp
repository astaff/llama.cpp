#include "../agx-repack.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"

// Independent verifier test suite for Pillar 2 (`agx-repack.{h,cpp}`).
// Tests:
//   1. Exhaustive branchless FTZ-safe GPR decoders (E2M1, UE4M3, E4M3, Q8_0 magic-bias, UE8M0)
//      confirming 0 FP16 denormals and 0 ULP error vs ggml reference.
//   2. Lossless bit-exact pack/unpack round-trip + dequant equivalence for AGX_NVFP4_32x16,
//      AGX_Q8_0_8BLK, and AGX_MXFP8_16BLK across Inkling 2 Mini exact tensor shapes + odd edges.
//   3. Tensor-level set_tensor / get_tensor (full, slice-aligned, and arbitrary unaligned partial ranges).
//   4. 128B cache-line alignment & zero-bloat checks.

namespace {

static const int8_t ref_kvalues_fp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12,
};

static inline bool is_fp16_subnormal(uint16_t h) {
    return ((h >> 10) & 0x1Fu) == 0u && (h & 0x03FFu) != 0u;
}

static inline float f16_bits_to_f32_ftz(uint16_t h) {
    if (is_fp16_subnormal(h)) {
        return (h & 0x8000u) ? -0.0f : 0.0f;
    }
    return GGML_FP16_TO_FP32(h);
}

static int test_decoders_exhaustive() {
    int fails = 0;

    // 1. E2M1 (all 16 nibbles)
    for (uint32_t nib = 0; nib < 16; ++nib) {
        uint16_t h = agx_decode_e2m1_to_f16_bits(nib);
        if (is_fp16_subnormal(h)) {
            std::fprintf(stderr, "FAIL: e2m1 nib=%u produced subnormal 0x%04x\n", nib, h);
            fails++;
        }
        float got = f16_bits_to_f32_ftz(h);
        float exp = static_cast<float>(ref_kvalues_fp4[nib]) * 0.5f;
        if (got != exp) {
            std::fprintf(stderr, "FAIL: e2m1 nib=%u got=%g exp=%g\n", nib, got, exp);
            fails++;
        }
    }

    // 2. UE4M3 (all 256 bytes) x E2M1 (all 16 nibbles)
    for (int u = 0; u < 256; ++u) {
        uint16_t hs = agx_decode_ue4m3_to_f16_bits(static_cast<uint8_t>(u));
        if (is_fp16_subnormal(hs)) {
            std::fprintf(stderr, "FAIL: ue4m3 u=%d produced subnormal 0x%04x\n", u, hs);
            fails++;
        }
        float sc_got = f16_bits_to_f32_ftz(hs);
        float sc_ref = ggml_ue4m3_to_fp32(static_cast<uint8_t>(u)) * 2.0f;
        if (sc_got != sc_ref) {
            std::fprintf(stderr, "FAIL: ue4m3 u=0x%02x got=%g ref=%g (h=0x%04x)\n", u, sc_got, sc_ref, hs);
            fails++;
        }
        for (uint32_t nib = 0; nib < 16; ++nib) {
            float v_got = f16_bits_to_f32_ftz(agx_decode_e2m1_to_f16_bits(nib)) * sc_got;
            float v_ref = static_cast<float>(ref_kvalues_fp4[nib]) * ggml_ue4m3_to_fp32(static_cast<uint8_t>(u));
            if (v_got != v_ref) {
                std::fprintf(stderr, "FAIL: nvfp4 prod u=0x%02x nib=%u got=%g ref=%g\n", u, nib, v_got, v_ref);
                fails++;
            }
        }
    }

    // 3. Q8_0 magic-bias (all 256 int8 values)
    for (int q = -128; q <= 127; ++q) {
        uint16_t h = agx_decode_q8_magic_to_f16_bits(static_cast<int8_t>(q));
        if (is_fp16_subnormal(h)) {
            std::fprintf(stderr, "FAIL: q8 q=%d produced subnormal 0x%04x\n", q, h);
            fails++;
        }
        float got = f16_bits_to_f32_ftz(h);
        if (got != static_cast<float>(q)) {
            std::fprintf(stderr, "FAIL: q8 q=%d got=%g\n", q, got);
            fails++;
        }
    }

    // 4. Signed E4M3 (all 256 bytes)
    for (int b = 0; b < 256; ++b) {
        uint16_t h = agx_decode_e4m3_to_f16_bits(static_cast<uint8_t>(b));
        if (is_fp16_subnormal(h)) {
            std::fprintf(stderr, "FAIL: e4m3 b=0x%02x produced subnormal 0x%04x\n", b, h);
            fails++;
        }
        uint8_t ax = static_cast<uint8_t>(b & 0x7Fu);
        if (ax == 0x7Fu) {
            if (!std::isnan(f16_bits_to_f32_ftz(h))) {
                std::fprintf(stderr, "FAIL: e4m3 NaN b=0x%02x decoded non-NaN\n", b);
                fails++;
            }
        } else {
            float sign = (b & 0x80) ? -1.0f : 1.0f;
            int e = (ax >> 3) & 0xF;
            int m = ax & 0x7;
            float exp = (e == 0) ? std::ldexp(static_cast<float>(m), -9)
                                 : std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
            float got = f16_bits_to_f32_ftz(h);
            if (got != sign * exp) {
                std::fprintf(stderr, "FAIL: e4m3 b=0x%02x got=%g exp=%g\n", b, got, sign * exp);
                fails++;
            }
        }
    }

    // 5. UE8M0 (all 256 bytes)
    for (int e = 0; e < 256; ++e) {
        float got = agx_decode_ue8m0_to_f32(static_cast<uint8_t>(e));
        float exp = (e == 0) ? std::ldexp(1.0f, -127) : std::ldexp(1.0f, e - 127);
        if (e == 255) {
            // e=255 -> bits=0x7F800000 (+inf)
            if (!std::isinf(got)) {
                std::fprintf(stderr, "FAIL: ue8m0 e=255 expected inf, got=%g\n", got);
                fails++;
            }
        } else if (got != exp) {
            std::fprintf(stderr, "FAIL: ue8m0 e=%d got=%g exp=%g\n", e, got, exp);
            fails++;
        }
    }

    return fails;
}

static void ref_dequant_nvfp4_row(const block_nvfp4 * row, float * out, int64_t K) {
    const int64_t nb = K / QK_NVFP4;
    for (int64_t i = 0; i < nb; ++i) {
        for (int s = 0; s < 4; ++s) {
            float d = ggml_ue4m3_to_fp32(row[i].d[s]);
            float * yb = out + i * QK_NVFP4 + s * QK_NVFP4_SUB;
            for (int j = 0; j < 8; ++j) {
                uint8_t q = row[i].qs[s * 8 + j];
                yb[j + 0] = ref_kvalues_fp4[q & 0x0F] * d;
                yb[j + 8] = ref_kvalues_fp4[q >> 4]   * d;
            }
        }
    }
}

static void ref_dequant_q8_0_row(const block_q8_0 * row, float * out, int64_t K) {
    const int64_t nb = K / QK8_0;
    for (int64_t i = 0; i < nb; ++i) {
        float d = GGML_FP16_TO_FP32(row[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            out[i * QK8_0 + j] = static_cast<float>(row[i].qs[j]) * d;
        }
    }
}

static int test_nvfp4_roundtrip_and_dequant() {
    int fails = 0;
    std::mt19937 rng(20261127);

    struct Shape { int64_t K, N, S; bool zero_bloat; };
    std::vector<Shape> shapes = {
        {64,   1,    1, false},
        {64,   17,   3, false},
        {128,  31,   2, false},
        {128,  32,   2, true},
        {1152, 2304, 2, true},  // Inkling blk.0..19 ffn_down_exps slice geometry
        {2304, 2304, 2, true},  // Inkling blk.0..19 ffn_gate_up_exps slice geometry
    };

    for (const auto & sh : shapes) {
        agx_repack_tensor_info info;
        agx_repack_describe(GGML_TYPE_NVFP4, sh.K, sh.N, sh.S, 1, &info);
        if (info.format != AGX_REPACK_NVFP4_32x16 || (info.slice_bytes % 128u) != 0) {
            std::fprintf(stderr, "FAIL: nvfp4 describe (%lld,%lld,%lld)\n",
                         (long long)sh.K, (long long)sh.N, (long long)sh.S);
            fails++;
            continue;
        }
        const size_t orig_bytes = static_cast<size_t>(sh.S * sh.N * (sh.K / QK_NVFP4)) * sizeof(block_nvfp4);
        if (sh.zero_bloat && info.total_bytes != orig_bytes) {
            std::fprintf(stderr, "FAIL: nvfp4 bloat (%lld,%lld): got %zu != orig %zu\n",
                         (long long)sh.K, (long long)sh.N, info.total_bytes, orig_bytes);
            fails++;
        }

        std::vector<uint8_t> src(orig_bytes), dst(orig_bytes, 0xA5);
        for (auto & b : src) b = static_cast<uint8_t>(rng() & 0xFFu);

        // Allocate 128B-aligned repacked buffer
        void * repacked = nullptr;
        if (posix_memalign(&repacked, 128, info.total_bytes) != 0) {
            return 1;
        }
        std::memset(repacked, 0xCC, info.total_bytes);

        if (!agx_repack_nvfp4_pack(src.data(), repacked, sh.K, sh.N, sh.S) ||
            !agx_repack_nvfp4_unpack(repacked, dst.data(), sh.K, sh.N, sh.S)) {
            std::fprintf(stderr, "FAIL: nvfp4 pack/unpack returned false\n");
            fails++;
        } else if (std::memcmp(src.data(), dst.data(), orig_bytes) != 0) {
            std::fprintf(stderr, "FAIL: nvfp4 round-trip mismatch for (%lld,%lld,%lld)\n",
                         (long long)sh.K, (long long)sh.N, (long long)sh.S);
            fails++;
        }

        // Verify direct repacked dequant matches ggml reference bit-for-bit on sampled rows
        const int64_t nb_k = sh.K / QK_NVFP4;
        std::vector<float> ref_row(sh.K), got_row(sh.K);
        for (int64_t sl = 0; sl < sh.S; ++sl) {
            const uint8_t * sl_agx = static_cast<const uint8_t *>(repacked) + static_cast<size_t>(sl) * info.slice_bytes;
            const auto * sl_blk = reinterpret_cast<const block_nvfp4 *>(src.data()) + sl * sh.N * nb_k;
            for (int64_t r : {int64_t(0), sh.N / 2, sh.N - 1}) {
                ref_dequant_nvfp4_row(sl_blk + r * nb_k, ref_row.data(), sh.K);
                agx_repack_nvfp4_dequant_row(sl_agx, sh.K, sh.N, r, got_row.data());
                for (int64_t k = 0; k < sh.K; ++k) {
                    if (ref_row[k] != got_row[k]) {
                        std::fprintf(stderr, "FAIL: nvfp4 dequant sl=%lld r=%lld k=%lld ref=%g got=%g\n",
                                     (long long)sl, (long long)r, (long long)k, ref_row[k], got_row[k]);
                        fails++;
                        break;
                    }
                }
            }
        }
        std::free(repacked);
    }
    return fails;
}

static int test_q8_0_and_mxfp8_roundtrip() {
    int fails = 0;
    std::mt19937 rng(424242);

    // Q8_0 shapes including Inkling K=2304 (0 bloat) and K=1152 (trailing partial chunk)
    struct QShape { int64_t K, N, S; bool zero_bloat; };
    for (auto sh : std::vector<QShape>{{32, 7, 2, false}, {256, 16, 2, true}, {1152, 64, 2, false}, {2304, 64, 2, true}}) {
        agx_repack_tensor_info info;
        agx_repack_describe(GGML_TYPE_Q8_0, sh.K, sh.N, sh.S, 1, &info);
        const size_t orig_bytes = static_cast<size_t>(sh.S * sh.N * (sh.K / QK8_0)) * sizeof(block_q8_0);
        if (sh.zero_bloat && info.slice_bytes % 128 == 0 && (orig_bytes % 128 == 0) && info.total_bytes != orig_bytes) {
            std::fprintf(stderr, "FAIL: q8_0 unexpected bloat (%lld,%lld): %zu vs %zu\n",
                         (long long)sh.K, (long long)sh.N, info.total_bytes, orig_bytes);
            fails++;
        }
        std::vector<block_q8_0> src(sh.S * sh.N * (sh.K / QK8_0)), dst(src.size());
        for (auto & blk : src) {
            blk.d = GGML_FP32_TO_FP16(static_cast<float>(static_cast<int>(rng() % 200) - 100) * 0.01f);
            for (int i = 0; i < QK8_0; ++i) blk.qs[i] = static_cast<int8_t>(static_cast<int>(rng() % 256) - 128);
        }
        std::vector<uint8_t> repacked(info.total_bytes, 0xCC);
        if (!agx_repack_q8_0_pack(src.data(), repacked.data(), sh.K, sh.N, sh.S) ||
            !agx_repack_q8_0_unpack(repacked.data(), dst.data(), sh.K, sh.N, sh.S)) {
            std::fprintf(stderr, "FAIL: q8_0 pack/unpack false\n");
            fails++;
        } else if (std::memcmp(src.data(), dst.data(), orig_bytes) != 0) {
            std::fprintf(stderr, "FAIL: q8_0 round-trip mismatch\n");
            fails++;
        }
        std::vector<float> ref_row(sh.K), got_row(sh.K);
        const int64_t nb_k = sh.K / QK8_0;
        for (int64_t r : {int64_t(0), sh.N - 1}) {
            ref_dequant_q8_0_row(src.data() + r * nb_k, ref_row.data(), sh.K);
            agx_repack_q8_0_dequant_row(repacked.data(), sh.K, r, got_row.data());
            for (int64_t k = 0; k < sh.K; ++k) {
                if (ref_row[k] != got_row[k]) {
                    std::fprintf(stderr, "FAIL: q8_0 dequant r=%lld k=%lld ref=%g got=%g\n",
                                 (long long)r, (long long)k, ref_row[k], got_row[k]);
                    fails++;
                    break;
                }
            }
        }
    }

    // MXFP8 round-trip
    for (int64_t K : {int64_t(32), int64_t(512), int64_t(2304)}) {
        const int64_t N = 13, S = 2;
        const int64_t nb_k = K / AGX_MXFP8_BLOCK_K;
        const int64_t n_chunks = (nb_k + AGX_MXFP8_CHUNK_BLOCKS - 1) / AGX_MXFP8_CHUNK_BLOCKS;
        const size_t row_b = static_cast<size_t>(n_chunks) * AGX_MXFP8_CHUNK_BYTES;
        const size_t slice_b = ((static_cast<size_t>(N) * row_b) + 127u) & ~size_t(127u);
        std::vector<uint8_t> src(static_cast<size_t>(S * N * nb_k) * 33u);
        std::vector<uint8_t> dst(src.size(), 0x5A);
        for (auto & b : src) b = static_cast<uint8_t>(rng() & 0xFFu);
        std::vector<uint8_t> repacked(slice_b * static_cast<size_t>(S), 0xCC);
        if (!agx_repack_mxfp8_pack(src.data(), repacked.data(), K, N, S) ||
            !agx_repack_mxfp8_unpack(repacked.data(), dst.data(), K, N, S) ||
            std::memcmp(src.data(), dst.data(), src.size()) != 0) {
            std::fprintf(stderr, "FAIL: mxfp8 round-trip K=%lld\n", (long long)K);
            fails++;
        }
    }
    return fails;
}

static int test_partial_set_get_tensor() {
    int fails = 0;
    std::mt19937 rng(9999);

    ggml_tensor t{};
    t.type = GGML_TYPE_NVFP4;
    t.ne[0] = 128;
    t.ne[1] = 35;
    t.ne[2] = 4;
    t.ne[3] = 1;
    t.nb[0] = sizeof(block_nvfp4);
    t.nb[1] = t.nb[0] * (t.ne[0] / QK_NVFP4);
    t.nb[2] = t.nb[1] * t.ne[1];
    t.nb[3] = t.nb[2] * t.ne[2];

    const size_t orig_nbytes = ggml_nbytes(&t);
    const size_t alloc_bytes = agx_repack_get_alloc_size(&t);

    std::vector<uint8_t> orig(orig_nbytes), readback(orig_nbytes, 0);
    for (auto & b : orig) b = static_cast<uint8_t>(rng() & 0xFFu);

    void * repacked = nullptr;
    if (posix_memalign(&repacked, 128, alloc_bytes) != 0) return 1;
    std::memset(repacked, 0, alloc_bytes);
    t.data = repacked;

    if (!agx_repack_supports_tensor(&t)) {
        std::fprintf(stderr, "FAIL: agx_repack_supports_tensor returned false for NVFP4\n");
        fails++;
    }

    // Full upload, then patch an unaligned middle slice spanning slices 1..2, then partial + full readback
    agx_repack_set_tensor(&t, orig.data(), 0, orig_nbytes);

    const size_t patch_off = (orig_nbytes / 4) + 13;
    const size_t patch_sz  = (orig_nbytes / 2) - 27;
    std::vector<uint8_t> patch(patch_sz);
    for (auto & b : patch) b = static_cast<uint8_t>(rng() & 0xFFu);
    std::memcpy(orig.data() + patch_off, patch.data(), patch_sz);

    agx_repack_set_tensor(&t, patch.data(), patch_off, patch_sz);

    std::vector<uint8_t> partial_out(patch_sz, 0);
    agx_repack_get_tensor(&t, partial_out.data(), patch_off, patch_sz);
    if (std::memcmp(partial_out.data(), patch.data(), patch_sz) != 0) {
        std::fprintf(stderr, "FAIL: partial get_tensor mismatch\n");
        fails++;
    }
    agx_repack_get_tensor(&t, readback.data(), 0, orig_nbytes);
    if (std::memcmp(readback.data(), orig.data(), orig_nbytes) != 0) {
        std::fprintf(stderr, "FAIL: full get_tensor after partial set mismatch\n");
        fails++;
    }

    std::free(repacked);
    return fails;
}

} // namespace

int main() {
    int total = 0;
    auto run = [&](const char * name, int (*fn)()) {
        int f = fn();
        std::printf("[%-30s] %s (fails=%d)\n", name, f == 0 ? "PASS" : "FAIL", f);
        total += f;
    };
    run("decoders_exhaustive_ftz",   test_decoders_exhaustive);
    run("nvfp4_roundtrip_dequant",   test_nvfp4_roundtrip_and_dequant);
    run("q8_0_and_mxfp8_roundtrip",  test_q8_0_and_mxfp8_roundtrip);
    run("partial_set_get_tensor",    test_partial_set_get_tensor);

    if (total != 0) {
        std::fprintf(stderr, "test_agx_repack FAILED (%d failures)\n", total);
        return 1;
    }
    std::printf("ALL test_agx_repack PASSED\n");
    return 0;
}
