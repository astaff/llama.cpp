#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

// Standalone numerical-stability & reference math test suite for ggml-agx.
// Verifies exact bit-level / FP32 invariants required for Inkling 2 Mini (NVFP4 + Q8_0 + BF16 + F32)
// and DSpark MTP on Apple M4 Max (AGX3).

namespace {

static const int8_t kvalues_fp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12,
};

// Exact ggml_ue4m3_to_fp32 from ggml/src/ggml-impl.h (includes 0.5f factor matching 2x kvalues_fp4).
static inline float ref_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    int exp = (x >> 3) & 0xF;
    int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = std::ldexp(static_cast<float>(man), -9);
    } else {
        raw = std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}

// IEEE-754 binary16 -> float32 helper (FTZ-aware test helper).
static inline float fp16_bits_to_f32(uint16_t h, bool ftz = false) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0 || ftz) {
            f = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x03FFu;
            f = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// Branchless FTZ-safe E2M1 -> normal FP16 decode (zero denormals, zero LUT).
// Maps nibble n in [0..15] to exact E2M1 value * 2^(bias_shift) as normal FP16:
//   - n = 0,8   -> +/-0.0
//   - n = 1,9   -> +/-0.5  (handled without subnormals by emitting normal FP16 with exp bias)
// Specifically: emitting FP16 bits ((n & 8) << 12) | ((n & 7) << 9) produces:
//   mag=0 -> 0.0
//   mag=1 (0b001) -> 0x0200 (FP16 subnormal 2^-15! FLUSHED TO ZERO UNDER FTZ!)
// Therefore FTZ-safe branchless decode MUST add a normal exponent bias (e.g. +14 => +0x3800 for nonzero,
// or integer-scaled normal FP16 via float-add/mask trick, or piecewise normal exponent with zero subnormals).
// Here we verify the FTZ-safe normal-exponent shift method:
//   For mag != 0: raw = (sign << 15) | ((exp_e2m1 + 14) << 10) | (mant_e2m1 << 9) for normal E2M1,
//   and for E2M1 subnormal mag==1 (0.5): exp=14, mant=0 -> 0x3800 (which is 0.5 in IEEE FP16? No: exp=14 => 2^-1 = 0.5!).
//   WAIT: Look at the magic!
//   In IEEE FP16 (bias 15):
//     0.5 = 2^-1 -> exp=14 (0x3800), mant=0
//     1.0 = 2^0  -> exp=15 (0x3C00), mant=0
//     1.5 = 1.5  -> exp=15 (0x3E00), mant=0x200
//     2.0 = 2^1  -> exp=16 (0x4000), mant=0
//     3.0 = 1.5*2-> exp=16 (0x4200), mant=0x200
//     4.0 = 2^2  -> exp=17 (0x4400), mant=0
//     6.0 = 1.5*4-> exp=17 (0x4600), mant=0x200
//   Notice mag << 9:
//     mag=1 -> 0x0200; + 0x3600 = 0x3800 (0.5)  -- wait: mag=2 -> 0x0400 + 0x3800 = 0x3C00 (1.0)
//   Whereas if we bias by 2^13 (exponent 13 = 0x3400) or use:
//     uint16_t mag = n & 7;
//     uint16_t h_exp = (mag == 1) ? 0x3800 : ((mag == 0) ? 0 : (0x3800 + (mag << 9) - 0x0400));
//   Wait: check mag=1..7:
//     mag=1: 0x3800 (0.5)   -> 0x3400 + 0x0200 = 0x3600 (NO)
//     mag=2: 0x3C00 (1.0)   -> 0x3800 + 0x0400 = 0x3C00
//     mag=3: 0x3E00 (1.5)   -> 0x3800 + 0x0600 = 0x3E00
//     mag=4: 0x4000 (2.0)   -> 0x3800 + 0x0800 = 0x4000
//     mag=5: 0x4200 (3.0)   -> 0x3800 + 0x0A00 = 0x4200
//     mag=6: 0x4400 (4.0)   -> 0x3800 + 0x0C00 = 0x4400
//     mag=7: 0x4600 (6.0)   -> 0x3800 + 0x0E00 = 0x4600
//   LOOK AT THAT:
//     For mag in {2..7}: FP16(E2M1) = (sign << 15) | (0x3800 + (mag << 9))  (exact!)
//     For mag == 1:      FP16(0.5)  = (sign << 15) | 0x3800 (which is 0x3800 + 0, whereas formula gives 0x3A00 = 0.75!)
//     OR: scale by 2 (kvalues_fp4 = {0,1,2,3,4,6,8,12}):
//       mag=1 -> 1.0 (0x3C00), mag=2 -> 2.0 (0x4000), mag=3 -> 3.0 (0x4200), ...
//       Still mag=1 has step +0x0400 to mag=2 while mag=2..7 step +0x0200!
//     So the exact branchless FTZ-safe normal-FP16 formula is:
//       uint16_t raw = ((n & 8u) << 12) | ((n & 7u) << 9);
//       // Wait: raw as FP16 is subnormal (mag * 2^-15) — unsafe under FTZ!
//       // Safe normal formula (zero subnormals, 3 integer ops, completely branchless):
//       uint16_t mag = n & 7u;
//       uint16_t bias = (mag == 0u) ? 0u : (mag == 1u ? 0x3600u : 0x3800u); // compiles to 2 icmpsel on AGX3!
//       uint16_t h = static_cast<uint16_t>(((n & 8u) << 12) | (bias + (mag << 9)));
static inline uint16_t decode_e2m1_ftz_safe_fp16(uint8_t nibble) {
    uint16_t sign = static_cast<uint16_t>((nibble & 0x8u) << 12);
    uint16_t mag  = static_cast<uint16_t>(nibble & 0x7u);
    uint16_t bias = (mag == 0u) ? 0u : (mag == 1u ? 0x3600u : 0x3800u);
    return static_cast<uint16_t>(sign | (bias + (mag << 9)));
}

static int test_nvfp4_decode() {
    int fails = 0;
    // 1. Exhaustive check of all 16 E2M1 nibbles under BOTH FTZ=false and FTZ=true
    for (int n = 0; n < 16; ++n) {
        uint16_t h = decode_e2m1_ftz_safe_fp16(static_cast<uint8_t>(n));
        // Verify non-zero nibbles never produce FP16 subnormals (exp == 0 && mant != 0)
        uint16_t exp = (h >> 10) & 0x1Fu;
        uint16_t man = h & 0x03FFu;
        if (exp == 0 && man != 0) {
            std::fprintf(stderr, "FAIL: nibble %d produced FP16 subnormal 0x%04x\n", n, h);
            fails++;
        }
        float v_noftz = fp16_bits_to_f32(h, false);
        float v_ftz   = fp16_bits_to_f32(h, true);
        float expected = static_cast<float>(kvalues_fp4[n]) * 0.5f;
        if (v_noftz != expected || v_ftz != expected) {
            std::fprintf(stderr, "FAIL: nibble %d got (%g, ftz=%g) expected %g (h=0x%04x)\n",
                         n, v_noftz, v_ftz, expected, h);
            fails++;
        }
    }

    // 2. Exhaustive 16 x 256 (nibble x UE4M3 scale) check against ggml reference
    for (int s = 0; s < 256; ++s) {
        float d_ref = ref_ue4m3_to_fp32(static_cast<uint8_t>(s));
        // UE4M3 scale without the 0.5f factor (since decode_e2m1_ftz_safe_fp16 returns E2M1 directly)
        float scale_raw = d_ref * 2.0f;
        for (int n = 0; n < 16; ++n) {
            float ref_val = static_cast<float>(kvalues_fp4[n]) * d_ref;
            float got_val = fp16_bits_to_f32(decode_e2m1_ftz_safe_fp16(static_cast<uint8_t>(n)), true) * scale_raw;
            if (!std::isfinite(got_val) || std::fabs(ref_val - got_val) > 1e-6f * (1.0f + std::fabs(ref_val))) {
                std::fprintf(stderr, "FAIL: NVFP4 s=%d n=%d ref=%g got=%g\n", s, n, ref_val, got_val);
                fails++;
            }
        }
    }
    return fails;
}

static int test_l2_clip_invariant() {
    int fails = 0;
    // Exact Inkling 2 Mini invariant: min(t / max(||x||, t), 1.0f) — ZERO +eps bias when ||x|| == t!
    auto exact_clip = [](std::vector<float> & x, float t) {
        double ss = 0.0;
        for (float v : x) ss += static_cast<double>(v) * static_cast<double>(v);
        float norm = std::sqrt(static_cast<float>(ss));
        float scale = (norm > t) ? (t / norm) : 1.0f;
        for (float & v : x) v *= scale;
    };

    // Case 1: ||x|| == t exactly -> must be bit-identical (no +eps shrinkage!)
    {
        float t = 4.0f;
        std::vector<float> x(16, 1.0f); // ||x|| = sqrt(16) = 4.0 == t
        std::vector<float> orig = x;
        exact_clip(x, t);
        for (size_t i = 0; i < x.size(); ++i) {
            if (x[i] != orig[i]) {
                std::fprintf(stderr, "FAIL: L2 clip biased exact-threshold vector at %zu: %g != %g\n", i, x[i], orig[i]);
                fails++;
            }
        }
    }
    // Case 2: ||x|| == 0 -> must stay 0 (zero NaN!)
    {
        std::vector<float> x(128, 0.0f);
        exact_clip(x, 1.5f);
        for (float v : x) {
            if (v != 0.0f || !std::isfinite(v)) {
                std::fprintf(stderr, "FAIL: L2 clip produced non-zero/NaN on zero vector: %g\n", v);
                fails++;
                break;
            }
        }
    }
    // Case 3: ||x|| > t -> norm becomes t
    {
        float t = 3.0f;
        std::vector<float> x(128, 2.0f);
        exact_clip(x, t);
        double ss = 0.0;
        for (float v : x) ss += static_cast<double>(v) * v;
        float norm = std::sqrt(static_cast<float>(ss));
        if (std::fabs(norm - t) > 1e-5f) {
            std::fprintf(stderr, "FAIL: L2 clip post-norm %g != %g\n", norm, t);
            fails++;
        }
    }
    return fails;
}

static int test_fwht_orthogonality() {
    int fails = 0;
    auto fwht = [](std::vector<float> & a) {
        const size_t n = a.size();
        for (size_t len = 1; len < n; len <<= 1) {
            for (size_t i = 0; i < n; i += 2 * len) {
                for (size_t j = 0; j < len; ++j) {
                    float u = a[i + j];
                    float v = a[i + j + len];
                    a[i + j]       = u + v;
                    a[i + j + len] = u - v;
                }
            }
        }
        float inv = 1.0f / std::sqrt(static_cast<float>(n));
        for (float & v : a) v *= inv;
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (size_t n : {64ul, 128ul}) {
        std::vector<float> x(n);
        for (float & v : x) v = dist(rng);
        std::vector<float> y = x;
        fwht(y);
        fwht(y); // Involution check: FWHT(FWHT(x)) == x
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(y[i] - x[i]) > 1e-5f) {
                std::fprintf(stderr, "FAIL: FWHT_%zu involution mismatch at %zu: %g vs %g\n", n, i, y[i], x[i]);
                fails++;
                break;
            }
        }
    }
    return fails;
}

static int test_situ_and_router() {
    int fails = 0;
    // Inkling SiTU: silu(beta * gate) * (up * linear_beta) — verify finiteness across extreme inputs
    const float beta = 1.0f;
    const float lbeta = 1.0f;
    for (float g : {-80.0f, -10.0f, 0.0f, 10.0f, 80.0f}) {
        float sg = g / (1.0f + std::exp(-beta * g));
        float out = sg * (2.5f * lbeta);
        if (!std::isfinite(out)) {
            std::fprintf(stderr, "FAIL: SiTU non-finite for g=%g\n", g);
            fails++;
        }
    }
    // Router top-8 sigmoid renormalization with eps=1e-20 (inkling2.cpp L57)
    float logits[128];
    for (int i = 0; i < 128; ++i) logits[i] = -50.0f; // extreme underflow test
    double sum = 0.0;
    float w[8];
    for (int k = 0; k < 8; ++k) {
        w[k] = 1.0f / (1.0f + std::exp(-logits[k]));
        sum += w[k];
    }
    float denom = std::max(static_cast<float>(sum), 1e-20f);
    for (int k = 0; k < 8; ++k) {
        float wn = w[k] / denom;
        if (!std::isfinite(wn)) {
            std::fprintf(stderr, "FAIL: router renorm non-finite under extreme negative logits\n");
            fails++;
            break;
        }
    }
    return fails;
}

static int test_bitonic_moe_bucketing() {
    int fails = 0;
    // Simulate M=5 DSpark verify tokens x top-8 = 40 (expert_id, token_idx) pairs.
    // Pad to 64 for bitonic sort, sort by expert_id, verify deduplicated bucket counts sum to 40
    // and every unique expert appears in exactly one contiguous bucket.
    struct Pair { uint32_t key; }; // (expert_id << 8) | token_idx; sentinel = 0xFFFFFFFF
    std::mt19937 rng(1337);
    std::vector<Pair> arr(64, Pair{0xFFFFFFFFu});
    int orig_counts[128] = {0};
    int idx = 0;
    for (int tok = 0; tok < 5; ++tok) {
        bool used[128] = {false};
        for (int k = 0; k < 8; ++k) {
            int e;
            do { e = static_cast<int>(rng() % 32); } while (used[e]); // hot-expert overlap
            used[e] = true;
            orig_counts[e]++;
            arr[idx++].key = (static_cast<uint32_t>(e) << 8) | static_cast<uint32_t>(tok);
        }
    }
    // Bitonic sort over N=64
    for (size_t k = 2; k <= 64; k <<= 1) {
        for (size_t j = k >> 1; j > 0; j >>= 1) {
            for (size_t i = 0; i < 64; ++i) {
                size_t l = i ^ j;
                if (l > i) {
                    bool asc = ((i & k) == 0);
                    if ((asc && arr[i].key > arr[l].key) || (!asc && arr[i].key < arr[l].key)) {
                        std::swap(arr[i], arr[l]);
                    }
                }
            }
        }
    }
    int got_counts[128] = {0};
    int unique_experts = 0;
    int prev_e = -1;
    for (int i = 0; i < 40; ++i) {
        int e = static_cast<int>(arr[i].key >> 8);
        if (e < prev_e) {
            std::fprintf(stderr, "FAIL: bitonic sort out of order at %d: %d < %d\n", i, e, prev_e);
            fails++;
            break;
        }
        if (e != prev_e) {
            unique_experts++;
            prev_e = e;
        }
        got_counts[e]++;
    }
    for (int e = 0; e < 128; ++e) {
        if (got_counts[e] != orig_counts[e]) {
            std::fprintf(stderr, "FAIL: bitonic bucket count mismatch for expert %d: %d != %d\n",
                         e, got_counts[e], orig_counts[e]);
            fails++;
        }
    }
    if (unique_experts >= 40) {
        std::fprintf(stderr, "FAIL: expected expert coalescing (<40 unique), got %d\n", unique_experts);
        fails++;
    }
    return fails;
}

} // namespace

int main() {
    int total_fails = 0;
    auto run = [&](const char * name, int (*fn)()) {
        int f = fn();
        std::printf("[%-26s] %s (fails=%d)\n", name, f == 0 ? "PASS" : "FAIL", f);
        total_fails += f;
    };

    run("nvfp4_ftz_safe_decode", test_nvfp4_decode);
    run("inkling2_l2_clip",      test_l2_clip_invariant);
    run("fwht_64_128",           test_fwht_orthogonality);
    run("situ_and_router",       test_situ_and_router);
    run("bitonic_moe_bucket",    test_bitonic_moe_bucketing);

    if (total_fails != 0) {
        std::fprintf(stderr, "test_agx_numerics FAILED (%d failures)\n", total_fails);
        return 1;
    }
    std::printf("ALL test_agx_numerics PASSED\n");
    return 0;
}
