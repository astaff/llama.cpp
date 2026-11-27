#include "agx-repack.h"
#include "agx-common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"

static_assert(sizeof(agx_nvfp4_supertile) == AGX_NVFP4_ST_BYTES,
              "agx_nvfp4_supertile must be exactly 1152 bytes (9 x 128B cache lines)");
static_assert(offsetof(agx_nvfp4_supertile, qs) == AGX_NVFP4_ST_QS0_OFF,
              "agx_nvfp4_supertile::qs must start at 128B cache-line boundary");
static_assert(sizeof(agx_q8_0_8blk) == AGX_Q8_0_CHUNK_BYTES,
              "agx_q8_0_8blk must be exactly 272 bytes (17 x 16B)");
static_assert(sizeof(agx_mxfp8_16blk) == AGX_MXFP8_CHUNK_BYTES,
              "agx_mxfp8_16blk must be exactly 528 bytes (33 x 16B)");

static inline float agx_f16_bits_to_f32(uint16_t h) {
    return GGML_FP16_TO_FP32(h);
}

static inline uint16_t agx_f32_to_f16_bits(float f) {
    return GGML_FP32_TO_FP16(f);
}

// ============================================================================
// Branchless FTZ-Safe Bitwise Decoders (host bit-exact twin of GPU kernels)
// ============================================================================

uint16_t agx_decode_e2m1_to_f16_bits(uint32_t nib) {
    // E2M1 nibble: bit 3 = sign, bits [2:0] = magnitude index (0..7)
    // Values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    // Normal-exponent shift:
    //   mag == 0 -> 0x0000 (exact +0.0)
    //   mag == 1 -> 0x3800 (exact +0.5, normal FP16 exp=14, ZERO denormals!)
    //   mag >= 2 -> 0x3800 + (mag << 9) (exact normal FP16 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    const uint32_t sign = (nib & 8u) << 12;
    const uint32_t mag  = nib & 7u;
    const uint32_t norm = 0x3800u + (mag << 9);
    const uint32_t fix1 = (mag == 1u) ? 0x3800u : norm;
    const uint32_t h    = (mag == 0u) ? 0x0000u : fix1;
    return static_cast<uint16_t>(sign | h);
}

uint16_t agx_decode_ue4m3_to_f16_bits(uint8_t u) {
    // Matches 2.0f * ggml_ue4m3_to_fp32(u) bit-for-bit across all 256 byte values:
    //   u == 0 || u == 0x7F -> 0.0h
    //   exp == 0 (u7 < 8)   -> man * 2^-9 = as_half(0x2400 | (man << 7)) - 0.015625h (normal!)
    //   exp >= 1            -> (1 + man/8) * 2^(exp - 7) = as_half(0x2000 + (u7 << 7)) (normal!)
    // Note: ggml_ue4m3_to_fp32 masks bit 7 via `(u >> 3) & 0xF` except for checking u == 0 || u == 0x7F.
    if (u == 0u || u == 0x7Fu) {
        return 0x0000u;
    }
    const uint32_t u7 = static_cast<uint32_t>(u & 0x7Fu);
    if (u7 < 8u) {
        const uint16_t biased = static_cast<uint16_t>(0x2400u | (u7 << 7));
        const float val = agx_f16_bits_to_f32(biased) - 0.015625f; // 2^-6
        return agx_f32_to_f16_bits(val);
    }
    return static_cast<uint16_t>(0x2000u + (u7 << 7));
}

uint16_t agx_decode_e4m3_to_f16_bits(uint8_t b) {
    const uint32_t sign = static_cast<uint32_t>(b & 0x80u) << 8;
    const uint32_t ax   = static_cast<uint32_t>(b & 0x7Fu);
    if (ax == 0x7Fu) {
        return static_cast<uint16_t>(sign | 0x7E00u); // quiet NaN
    }
    if (ax == 0u) {
        return static_cast<uint16_t>(sign); // +-0.0h
    }
    if (ax < 8u) {
        const uint16_t biased = static_cast<uint16_t>(0x2400u | (ax << 7));
        const float val = agx_f16_bits_to_f32(biased) - 0.015625f;
        return static_cast<uint16_t>(sign | agx_f32_to_f16_bits(val));
    }
    return static_cast<uint16_t>(sign | (0x2000u + (ax << 7)));
}

uint16_t agx_decode_q8_magic_to_f16_bits(int8_t q) {
    // Magic-bias decode: 0x6400 == 1024.0h (ULP = 1.0h).
    // (uint8_t(q) ^ 0x80) maps [-128..127] -> [0..255], so
    // as_half(0x6400 | u) - 1152.0h == float16_t(q) with 0 ULP error and 0 denormals.
    const uint32_t u = static_cast<uint32_t>(static_cast<uint8_t>(q)) ^ 0x80u;
    const uint16_t biased = static_cast<uint16_t>(0x6400u | u);
    const float val = agx_f16_bits_to_f32(biased) - 1152.0f;
    return agx_f32_to_f16_bits(val);
}

float agx_decode_ue8m0_to_f32(uint8_t e) {
    const uint32_t bits = (e == 0u) ? 0x00400000u : (static_cast<uint32_t>(e) << 23);
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// ============================================================================
// Format Selection & Layout Description
// ============================================================================

enum agx_repack_format agx_repack_select_format(enum ggml_type type, int64_t ne0, int64_t ne1) {
    if (ne0 <= 0 || ne1 <= 0) {
        return AGX_REPACK_NONE;
    }
    switch (type) {
        case GGML_TYPE_NVFP4:
            if (ne0 % QK_NVFP4 == 0) {
                return AGX_REPACK_NVFP4_32x16;
            }
            return AGX_REPACK_NONE;
        case GGML_TYPE_Q8_0:
            if (ne0 % QK8_0 == 0) {
                return AGX_REPACK_Q8_0_8BLK;
            }
            return AGX_REPACK_NONE;
        default:
            return AGX_REPACK_NONE;
    }
}

void agx_repack_describe(enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                         struct agx_repack_tensor_info * out) {
    std::memset(out, 0, sizeof(*out));
    out->orig_type = type;
    out->ne0       = ne0;
    out->ne1       = ne1;
    out->ne2       = ne2 > 0 ? ne2 : 1;
    out->ne3       = ne3 > 0 ? ne3 : 1;
    out->format    = agx_repack_select_format(type, ne0, ne1);

    const int64_t n_slices = out->ne2 * out->ne3;
    if (out->format == AGX_REPACK_NVFP4_32x16) {
        out->n_tiles_n    = static_cast<uint32_t>((ne1 + AGX_NVFP4_TILE_N - 1) / AGX_NVFP4_TILE_N);
        out->n_tiles_k    = static_cast<uint32_t>(ne0 / AGX_NVFP4_TILE_K);
        out->row_stride_b = static_cast<uint32_t>(out->n_tiles_k * AGX_NVFP4_ST_BYTES);
        out->slice_bytes  = static_cast<size_t>(out->n_tiles_n) * static_cast<size_t>(out->row_stride_b);
        // AGX_NVFP4_ST_BYTES == 1152 == 9 * 128, so slice_bytes is inherently 128B aligned.
        out->total_bytes  = out->slice_bytes * static_cast<size_t>(n_slices);
    } else if (out->format == AGX_REPACK_Q8_0_8BLK) {
        out->n_tiles_n    = static_cast<uint32_t>(ne1);
        out->n_tiles_k    = static_cast<uint32_t>((ne0 + AGX_Q8_0_CHUNK_K - 1) / AGX_Q8_0_CHUNK_K);
        out->row_stride_b = static_cast<uint32_t>(out->n_tiles_k * AGX_Q8_0_CHUNK_BYTES);
        out->slice_bytes  = agx_align_up(static_cast<size_t>(ne1) * static_cast<size_t>(out->row_stride_b), 128u);
        out->total_bytes  = out->slice_bytes * static_cast<size_t>(n_slices);
    } else {
        const size_t row_bytes = ggml_row_size(type, ne0);
        out->n_tiles_n    = static_cast<uint32_t>(ne1);
        out->n_tiles_k    = 1;
        out->row_stride_b = static_cast<uint32_t>(row_bytes);
        out->slice_bytes  = row_bytes * static_cast<size_t>(ne1);
        out->total_bytes  = agx_align_up(out->slice_bytes * static_cast<size_t>(n_slices), 128u);
    }
}

size_t agx_repack_get_alloc_size(const struct ggml_tensor * tensor) {
    if (!tensor) {
        return 0;
    }
    struct agx_repack_tensor_info info;
    agx_repack_describe(tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], &info);
    if (info.format == AGX_REPACK_NONE) {
        return agx_align_up(ggml_nbytes(tensor), 128u);
    }
    return info.total_bytes;
}

size_t agx_repack_slice_stride(enum ggml_type type, int64_t ne0, int64_t ne1) {
    struct agx_repack_tensor_info info;
    agx_repack_describe(type, ne0, ne1, 1, 1, &info);
    return info.slice_bytes;
}

// ============================================================================
// NVFP4 <-> AGX_NVFP4_32x16 Lossless Pack / Unpack
// ============================================================================
// GGML block_nvfp4 layout (64 weights = 4 sub-blocks of 16):
//   d[s] (s = 0..3): UE4M3 scale for sub-block s
//   qs[s*8 + j] (j = 0..7):
//     low nibble  (& 0x0F) = weight s*16 + j
//     high nibble (>> 4)   = weight s*16 + j + 8
//
// AGX_NVFP4_32x16 lane layout (32 weights = 2 sub-blocks of 16):
//   qs[0]: sequential weights 0..7   (nibble i at bits [4*i+3 : 4*i])
//   qs[1]: sequential weights 8..15
//   qs[2]: sequential weights 16..23
//   qs[3]: sequential weights 24..31
//   d    : uint16_t(sub0_scale) | (uint16_t(sub1_scale) << 8)

static inline void pack_nvfp4_subblock16(const uint8_t qs8[8], uint32_t & out_w0, uint32_t & out_w1) {
    uint32_t w0 = 0; // weights 0..7  (low nibbles of qs8[0..7])
    uint32_t w1 = 0; // weights 8..15 (high nibbles of qs8[0..7])
    for (int j = 0; j < 8; ++j) {
        w0 |= static_cast<uint32_t>(qs8[j] & 0x0Fu) << (4 * j);
        w1 |= static_cast<uint32_t>(qs8[j] >> 4)    << (4 * j);
    }
    out_w0 = w0;
    out_w1 = w1;
}

static inline void unpack_nvfp4_subblock16(uint32_t w0, uint32_t w1, uint8_t qs8[8]) {
    for (int j = 0; j < 8; ++j) {
        const uint8_t lo = static_cast<uint8_t>((w0 >> (4 * j)) & 0x0Fu);
        const uint8_t hi = static_cast<uint8_t>((w1 >> (4 * j)) & 0x0Fu);
        qs8[j] = static_cast<uint8_t>(lo | (hi << 4));
    }
}

bool agx_repack_nvfp4_pack(const void * src_ggml, void * dst_agx, int64_t K, int64_t N, int64_t n_slices) {
    if (!src_ggml || !dst_agx || K <= 0 || N <= 0 || n_slices <= 0 || (K % QK_NVFP4) != 0) {
        return false;
    }

    const int64_t nb_k      = K / QK_NVFP4;
    const int64_t n_tiles_n = (N + AGX_NVFP4_TILE_N - 1) / AGX_NVFP4_TILE_N;

    const auto * src_blocks = static_cast<const block_nvfp4 *>(src_ggml);
    auto *       dst_tiles  = static_cast<agx_nvfp4_supertile *>(dst_agx);

    for (int64_t sl = 0; sl < n_slices; ++sl) {
        const block_nvfp4 * slice_src = src_blocks + sl * (N * nb_k);
        agx_nvfp4_supertile * slice_dst = dst_tiles + sl * (n_tiles_n * nb_k);

        for (int64_t tn = 0; tn < n_tiles_n; ++tn) {
            const int64_t row_base = tn * AGX_NVFP4_TILE_N;
            for (int64_t tk = 0; tk < nb_k; ++tk) {
                agx_nvfp4_supertile & st = slice_dst[tn * nb_k + tk];
                std::memset(&st, 0, sizeof(st));

                for (int s = 0; s < 2; ++s) {
                    for (int r = 0; r < AGX_NVFP4_SUBTILE_N; ++r) {
                        const int64_t row = row_base + s * AGX_NVFP4_SUBTILE_N + r;
                        if (row >= N) {
                            continue; // zero-padded tail rows
                        }
                        const block_nvfp4 & blk = slice_src[row * nb_k + tk];
                        // Lane 2*r: cols 0..31 (sub-blocks 0 and 1)
                        const int lane0 = r * 2;
                        st.d[s][lane0] = static_cast<uint16_t>(blk.d[0]) |
                                         (static_cast<uint16_t>(blk.d[1]) << 8);
                        pack_nvfp4_subblock16(&blk.qs[0], st.qs[s][lane0][0], st.qs[s][lane0][1]);
                        pack_nvfp4_subblock16(&blk.qs[8], st.qs[s][lane0][2], st.qs[s][lane0][3]);

                        // Lane 2*r + 1: cols 32..63 (sub-blocks 2 and 3)
                        const int lane1 = r * 2 + 1;
                        st.d[s][lane1] = static_cast<uint16_t>(blk.d[2]) |
                                         (static_cast<uint16_t>(blk.d[3]) << 8);
                        pack_nvfp4_subblock16(&blk.qs[16], st.qs[s][lane1][0], st.qs[s][lane1][1]);
                        pack_nvfp4_subblock16(&blk.qs[24], st.qs[s][lane1][2], st.qs[s][lane1][3]);
                    }
                }
            }
        }
    }
    return true;
}

bool agx_repack_nvfp4_unpack(const void * src_agx, void * dst_ggml, int64_t K, int64_t N, int64_t n_slices) {
    if (!src_agx || !dst_ggml || K <= 0 || N <= 0 || n_slices <= 0 || (K % QK_NVFP4) != 0) {
        return false;
    }

    const int64_t nb_k      = K / QK_NVFP4;
    const int64_t n_tiles_n = (N + AGX_NVFP4_TILE_N - 1) / AGX_NVFP4_TILE_N;

    const auto * src_tiles  = static_cast<const agx_nvfp4_supertile *>(src_agx);
    auto *       dst_blocks = static_cast<block_nvfp4 *>(dst_ggml);

    for (int64_t sl = 0; sl < n_slices; ++sl) {
        const agx_nvfp4_supertile * slice_src = src_tiles + sl * (n_tiles_n * nb_k);
        block_nvfp4 * slice_dst = dst_blocks + sl * (N * nb_k);

        for (int64_t tn = 0; tn < n_tiles_n; ++tn) {
            const int64_t row_base = tn * AGX_NVFP4_TILE_N;
            for (int64_t tk = 0; tk < nb_k; ++tk) {
                const agx_nvfp4_supertile & st = slice_src[tn * nb_k + tk];

                for (int s = 0; s < 2; ++s) {
                    for (int r = 0; r < AGX_NVFP4_SUBTILE_N; ++r) {
                        const int64_t row = row_base + s * AGX_NVFP4_SUBTILE_N + r;
                        if (row >= N) {
                            continue;
                        }
                        block_nvfp4 & blk = slice_dst[row * nb_k + tk];

                        const int lane0 = r * 2;
                        blk.d[0] = static_cast<uint8_t>(st.d[s][lane0] & 0xFFu);
                        blk.d[1] = static_cast<uint8_t>(st.d[s][lane0] >> 8);
                        unpack_nvfp4_subblock16(st.qs[s][lane0][0], st.qs[s][lane0][1], &blk.qs[0]);
                        unpack_nvfp4_subblock16(st.qs[s][lane0][2], st.qs[s][lane0][3], &blk.qs[8]);

                        const int lane1 = r * 2 + 1;
                        blk.d[2] = static_cast<uint8_t>(st.d[s][lane1] & 0xFFu);
                        blk.d[3] = static_cast<uint8_t>(st.d[s][lane1] >> 8);
                        unpack_nvfp4_subblock16(st.qs[s][lane1][0], st.qs[s][lane1][1], &blk.qs[16]);
                        unpack_nvfp4_subblock16(st.qs[s][lane1][2], st.qs[s][lane1][3], &blk.qs[24]);
                    }
                }
            }
        }
    }
    return true;
}

// ============================================================================
// Q8_0 <-> AGX_Q8_0_8BLK Lossless Pack / Unpack
// ============================================================================

bool agx_repack_q8_0_pack(const void * src_ggml, void * dst_agx, int64_t K, int64_t N, int64_t n_slices) {
    if (!src_ggml || !dst_agx || K <= 0 || N <= 0 || n_slices <= 0 || (K % QK8_0) != 0) {
        return false;
    }

    const int64_t nb_k     = K / QK8_0;
    const int64_t n_chunks = (nb_k + AGX_Q8_0_CHUNK_BLOCKS - 1) / AGX_Q8_0_CHUNK_BLOCKS;
    const size_t  row_b    = static_cast<size_t>(n_chunks) * AGX_Q8_0_CHUNK_BYTES;
    const size_t  slice_b  = agx_align_up(static_cast<size_t>(N) * row_b, 128u);

    const auto * src_blocks = static_cast<const block_q8_0 *>(src_ggml);
    auto *       dst_bytes  = static_cast<uint8_t *>(dst_agx);

    for (int64_t sl = 0; sl < n_slices; ++sl) {
        const block_q8_0 * slice_src = src_blocks + sl * (N * nb_k);
        uint8_t *          slice_dst = dst_bytes + sl * slice_b;

        for (int64_t row = 0; row < N; ++row) {
            const block_q8_0 * row_src = slice_src + row * nb_k;
            auto * row_dst = reinterpret_cast<agx_q8_0_8blk *>(slice_dst + row * row_b);

            for (int64_t c = 0; c < n_chunks; ++c) {
                agx_q8_0_8blk & chk = row_dst[c];
                const int64_t b0 = c * AGX_Q8_0_CHUNK_BLOCKS;
                const int64_t count = std::min<int64_t>(AGX_Q8_0_CHUNK_BLOCKS, nb_k - b0);
                if (count < AGX_Q8_0_CHUNK_BLOCKS) {
                    std::memset(&chk, 0, sizeof(chk));
                }
                for (int64_t j = 0; j < count; ++j) {
                    const block_q8_0 & blk = row_src[b0 + j];
                    std::memcpy(&chk.d[j], &blk.d, sizeof(uint16_t));
                    std::memcpy(&chk.qs[j * QK8_0], blk.qs, QK8_0);
                }
            }
        }
        // Zero any 128B slice-tail padding
        const size_t used_b = static_cast<size_t>(N) * row_b;
        if (slice_b > used_b) {
            std::memset(slice_dst + used_b, 0, slice_b - used_b);
        }
    }
    return true;
}

bool agx_repack_q8_0_unpack(const void * src_agx, void * dst_ggml, int64_t K, int64_t N, int64_t n_slices) {
    if (!src_agx || !dst_ggml || K <= 0 || N <= 0 || n_slices <= 0 || (K % QK8_0) != 0) {
        return false;
    }

    const int64_t nb_k     = K / QK8_0;
    const int64_t n_chunks = (nb_k + AGX_Q8_0_CHUNK_BLOCKS - 1) / AGX_Q8_0_CHUNK_BLOCKS;
    const size_t  row_b    = static_cast<size_t>(n_chunks) * AGX_Q8_0_CHUNK_BYTES;
    const size_t  slice_b  = agx_align_up(static_cast<size_t>(N) * row_b, 128u);

    const auto * src_bytes  = static_cast<const uint8_t *>(src_agx);
    auto *       dst_blocks = static_cast<block_q8_0 *>(dst_ggml);

    for (int64_t sl = 0; sl < n_slices; ++sl) {
        const uint8_t * slice_src = src_bytes + sl * slice_b;
        block_q8_0 *    slice_dst = dst_blocks + sl * (N * nb_k);

        for (int64_t row = 0; row < N; ++row) {
            const auto * row_src = reinterpret_cast<const agx_q8_0_8blk *>(slice_src + row * row_b);
            block_q8_0 * row_dst = slice_dst + row * nb_k;

            for (int64_t c = 0; c < n_chunks; ++c) {
                const agx_q8_0_8blk & chk = row_src[c];
                const int64_t b0 = c * AGX_Q8_0_CHUNK_BLOCKS;
                const int64_t count = std::min<int64_t>(AGX_Q8_0_CHUNK_BLOCKS, nb_k - b0);
                for (int64_t j = 0; j < count; ++j) {
                    block_q8_0 & blk = row_dst[b0 + j];
                    std::memcpy(&blk.d, &chk.d[j], sizeof(uint16_t));
                    std::memcpy(blk.qs, &chk.qs[j * QK8_0], QK8_0);
                }
            }
        }
    }
    return true;
}

// ============================================================================
// MXFP8 <-> AGX_MXFP8_16BLK Lossless Pack / Unpack
// ============================================================================

struct agx_mxfp8_raw_block {
    uint8_t e;
    uint8_t qs[AGX_MXFP8_BLOCK_K];
};
static_assert(sizeof(agx_mxfp8_raw_block) == 33, "raw MXFP8 block must be 33 bytes");

bool agx_repack_mxfp8_pack(const void * src_raw33, void * dst_agx, int64_t K, int64_t N, int64_t n_slices) {
    if (!src_raw33 || !dst_agx || K <= 0 || N <= 0 || n_slices <= 0 || (K % AGX_MXFP8_BLOCK_K) != 0) {
        return false;
    }

    const int64_t nb_k     = K / AGX_MXFP8_BLOCK_K;
    const int64_t n_chunks = (nb_k + AGX_MXFP8_CHUNK_BLOCKS - 1) / AGX_MXFP8_CHUNK_BLOCKS;
    const size_t  row_b    = static_cast<size_t>(n_chunks) * AGX_MXFP8_CHUNK_BYTES;
    const size_t  slice_b  = agx_align_up(static_cast<size_t>(N) * row_b, 128u);

    const auto * src_blocks = static_cast<const agx_mxfp8_raw_block *>(src_raw33);
    auto *       dst_bytes  = static_cast<uint8_t *>(dst_agx);

    for (int64_t sl = 0; sl < n_slices; ++sl) {
        const agx_mxfp8_raw_block * slice_src = src_blocks + sl * (N * nb_k);
        uint8_t *                   slice_dst = dst_bytes + sl * slice_b;

        for (int64_t row = 0; row < N; ++row) {
            const agx_mxfp8_raw_block * row_src = slice_src + row * nb_k;
            auto * row_dst = reinterpret_cast<agx_mxfp8_16blk *>(slice_dst + row * row_b);

            for (int64_t c = 0; c < n_chunks; ++c) {
                agx_mxfp8_16blk & chk = row_dst[c];
                const int64_t b0 = c * AGX_MXFP8_CHUNK_BLOCKS;
                const int64_t count = std::min<int64_t>(AGX_MXFP8_CHUNK_BLOCKS, nb_k - b0);
                if (count < AGX_MXFP8_CHUNK_BLOCKS) {
                    std::memset(&chk, 0, sizeof(chk));
                }
                for (int64_t j = 0; j < count; ++j) {
                    chk.e[j] = row_src[b0 + j].e;
                    std::memcpy(&chk.qs[j * AGX_MXFP8_BLOCK_K], row_src[b0 + j].qs, AGX_MXFP8_BLOCK_K);
                }
            }
        }
        const size_t used_b = static_cast<size_t>(N) * row_b;
        if (slice_b > used_b) {
            std::memset(slice_dst + used_b, 0, slice_b - used_b);
        }
    }
    return true;
}

bool agx_repack_mxfp8_unpack(const void * src_agx, void * dst_raw33, int64_t K, int64_t N, int64_t n_slices) {
    if (!src_agx || !dst_raw33 || K <= 0 || N <= 0 || n_slices <= 0 || (K % AGX_MXFP8_BLOCK_K) != 0) {
        return false;
    }

    const int64_t nb_k     = K / AGX_MXFP8_BLOCK_K;
    const int64_t n_chunks = (nb_k + AGX_MXFP8_CHUNK_BLOCKS - 1) / AGX_MXFP8_CHUNK_BLOCKS;
    const size_t  row_b    = static_cast<size_t>(n_chunks) * AGX_MXFP8_CHUNK_BYTES;
    const size_t  slice_b  = agx_align_up(static_cast<size_t>(N) * row_b, 128u);

    const auto * src_bytes  = static_cast<const uint8_t *>(src_agx);
    auto *       dst_blocks = static_cast<agx_mxfp8_raw_block *>(dst_raw33);

    for (int64_t sl = 0; sl < n_slices; ++sl) {
        const uint8_t *       slice_src = src_bytes + sl * slice_b;
        agx_mxfp8_raw_block * slice_dst = dst_blocks + sl * (N * nb_k);

        for (int64_t row = 0; row < N; ++row) {
            const auto * row_src = reinterpret_cast<const agx_mxfp8_16blk *>(slice_src + row * row_b);
            agx_mxfp8_raw_block * row_dst = slice_dst + row * nb_k;

            for (int64_t c = 0; c < n_chunks; ++c) {
                const agx_mxfp8_16blk & chk = row_src[c];
                const int64_t b0 = c * AGX_MXFP8_CHUNK_BLOCKS;
                const int64_t count = std::min<int64_t>(AGX_MXFP8_CHUNK_BLOCKS, nb_k - b0);
                for (int64_t j = 0; j < count; ++j) {
                    row_dst[b0 + j].e = chk.e[j];
                    std::memcpy(row_dst[b0 + j].qs, &chk.qs[j * AGX_MXFP8_BLOCK_K], AGX_MXFP8_BLOCK_K);
                }
            }
        }
    }
    return true;
}

// ============================================================================
// Tensor-Level set_tensor / get_tensor (Full & Partial Byte-Range Round-Trip)
// ============================================================================

bool agx_repack_supports_tensor(const struct ggml_tensor * tensor) {
    if (!tensor) {
        return false;
    }
    return agx_repack_select_format(tensor->type, tensor->ne[0], tensor->ne[1]) != AGX_REPACK_NONE;
}

bool agx_repack_set_tensor_buf(const struct ggml_tensor * tensor, void * dst_repacked,
                               const void * src_ggml, size_t offset, size_t size) {
    if (!tensor || !dst_repacked || !src_ggml) {
        return false;
    }
    const size_t orig_nbytes = ggml_nbytes(tensor);
    if (offset > orig_nbytes || size > orig_nbytes - offset) {
        return false;
    }
    if (size == 0) {
        return true;
    }

    struct agx_repack_tensor_info info;
    agx_repack_describe(tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], &info);

    if (info.format == AGX_REPACK_NONE) {
        std::memcpy(static_cast<uint8_t *>(dst_repacked) + offset, src_ggml, size);
        return true;
    }

    const int64_t n_slices = info.ne2 * info.ne3;

    // Fast path: full-tensor upload (standard model load path, zero staging copy!)
    if (offset == 0 && size == orig_nbytes) {
        if (info.format == AGX_REPACK_NVFP4_32x16) {
            return agx_repack_nvfp4_pack(src_ggml, dst_repacked, info.ne0, info.ne1, n_slices);
        }
        if (info.format == AGX_REPACK_Q8_0_8BLK) {
            return agx_repack_q8_0_pack(src_ggml, dst_repacked, info.ne0, info.ne1, n_slices);
        }
    }

    // Fast path: whole-slice aligned upload (common when llama.cpp streams MoE experts)
    const size_t orig_slice_b = orig_nbytes / static_cast<size_t>(n_slices);
    if (orig_slice_b > 0 && (offset % orig_slice_b) == 0 && (size % orig_slice_b) == 0) {
        const int64_t sl0 = static_cast<int64_t>(offset / orig_slice_b);
        const int64_t nsl = static_cast<int64_t>(size / orig_slice_b);
        uint8_t * dst_sl  = static_cast<uint8_t *>(dst_repacked) + static_cast<size_t>(sl0) * info.slice_bytes;
        if (info.format == AGX_REPACK_NVFP4_32x16) {
            return agx_repack_nvfp4_pack(src_ggml, dst_sl, info.ne0, info.ne1, nsl);
        }
        if (info.format == AGX_REPACK_Q8_0_8BLK) {
            return agx_repack_q8_0_pack(src_ggml, dst_sl, info.ne0, info.ne1, nsl);
        }
    }

    // General partial-range fallback: unpack affected slices, patch bytes, repack
    const int64_t sl_first = static_cast<int64_t>(offset / orig_slice_b);
    const int64_t sl_last  = static_cast<int64_t>((offset + size - 1u) / orig_slice_b);
    const int64_t nsl      = sl_last - sl_first + 1;

    std::vector<uint8_t> tmp(static_cast<size_t>(nsl) * orig_slice_b);
    uint8_t * dst_sl = static_cast<uint8_t *>(dst_repacked) + static_cast<size_t>(sl_first) * info.slice_bytes;

    if (info.format == AGX_REPACK_NVFP4_32x16) {
        if (!agx_repack_nvfp4_unpack(dst_sl, tmp.data(), info.ne0, info.ne1, nsl)) return false;
    } else if (info.format == AGX_REPACK_Q8_0_8BLK) {
        if (!agx_repack_q8_0_unpack(dst_sl, tmp.data(), info.ne0, info.ne1, nsl)) return false;
    }

    const size_t local_off = offset - static_cast<size_t>(sl_first) * orig_slice_b;
    std::memcpy(tmp.data() + local_off, src_ggml, size);

    if (info.format == AGX_REPACK_NVFP4_32x16) {
        return agx_repack_nvfp4_pack(tmp.data(), dst_sl, info.ne0, info.ne1, nsl);
    }
    return agx_repack_q8_0_pack(tmp.data(), dst_sl, info.ne0, info.ne1, nsl);
}

bool agx_repack_get_tensor_buf(const struct ggml_tensor * tensor, const void * src_repacked,
                               void * dst_ggml, size_t offset, size_t size) {
    if (!tensor || !src_repacked || !dst_ggml) {
        return false;
    }
    const size_t orig_nbytes = ggml_nbytes(tensor);
    if (offset > orig_nbytes || size > orig_nbytes - offset) {
        return false;
    }
    if (size == 0) {
        return true;
    }

    struct agx_repack_tensor_info info;
    agx_repack_describe(tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], &info);

    if (info.format == AGX_REPACK_NONE) {
        std::memcpy(dst_ggml, static_cast<const uint8_t *>(src_repacked) + offset, size);
        return true;
    }

    const int64_t n_slices = info.ne2 * info.ne3;

    // Fast path: full-tensor readback
    if (offset == 0 && size == orig_nbytes) {
        if (info.format == AGX_REPACK_NVFP4_32x16) {
            return agx_repack_nvfp4_unpack(src_repacked, dst_ggml, info.ne0, info.ne1, n_slices);
        }
        if (info.format == AGX_REPACK_Q8_0_8BLK) {
            return agx_repack_q8_0_unpack(src_repacked, dst_ggml, info.ne0, info.ne1, n_slices);
        }
    }

    // Whole-slice aligned readback
    const size_t orig_slice_b = orig_nbytes / static_cast<size_t>(n_slices);
    if (orig_slice_b > 0 && (offset % orig_slice_b) == 0 && (size % orig_slice_b) == 0) {
        const int64_t sl0 = static_cast<int64_t>(offset / orig_slice_b);
        const int64_t nsl = static_cast<int64_t>(size / orig_slice_b);
        const uint8_t * src_sl = static_cast<const uint8_t *>(src_repacked) + static_cast<size_t>(sl0) * info.slice_bytes;
        if (info.format == AGX_REPACK_NVFP4_32x16) {
            return agx_repack_nvfp4_unpack(src_sl, dst_ggml, info.ne0, info.ne1, nsl);
        }
        if (info.format == AGX_REPACK_Q8_0_8BLK) {
            return agx_repack_q8_0_unpack(src_sl, dst_ggml, info.ne0, info.ne1, nsl);
        }
    }

    // General partial-range readback
    const int64_t sl_first = static_cast<int64_t>(offset / orig_slice_b);
    const int64_t sl_last  = static_cast<int64_t>((offset + size - 1u) / orig_slice_b);
    const int64_t nsl      = sl_last - sl_first + 1;

    std::vector<uint8_t> tmp(static_cast<size_t>(nsl) * orig_slice_b);
    const uint8_t * src_sl = static_cast<const uint8_t *>(src_repacked) + static_cast<size_t>(sl_first) * info.slice_bytes;

    if (info.format == AGX_REPACK_NVFP4_32x16) {
        if (!agx_repack_nvfp4_unpack(src_sl, tmp.data(), info.ne0, info.ne1, nsl)) return false;
    } else if (info.format == AGX_REPACK_Q8_0_8BLK) {
        if (!agx_repack_q8_0_unpack(src_sl, tmp.data(), info.ne0, info.ne1, nsl)) return false;
    }

    const size_t local_off = offset - static_cast<size_t>(sl_first) * orig_slice_b;
    std::memcpy(dst_ggml, tmp.data() + local_off, size);
    return true;
}

void agx_repack_set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (!tensor || !tensor->data || !data) {
        return;
    }
    if (!agx_repack_set_tensor_buf(tensor, tensor->data, data, offset, size)) {
        AGX_LOG_ERROR("agx_repack_set_tensor failed for tensor '%s' (type=%d, ne=[%lld,%lld,%lld,%lld], off=%zu, sz=%zu)",
                      tensor->name, static_cast<int>(tensor->type),
                      static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
                      static_cast<long long>(tensor->ne[2]), static_cast<long long>(tensor->ne[3]),
                      offset, size);
    }
}

void agx_repack_get_tensor(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (!tensor || !tensor->data || !data) {
        return;
    }
    if (!agx_repack_get_tensor_buf(tensor, tensor->data, data, offset, size)) {
        AGX_LOG_ERROR("agx_repack_get_tensor failed for tensor '%s' (type=%d, ne=[%lld,%lld,%lld,%lld], off=%zu, sz=%zu)",
                      tensor->name, static_cast<int>(tensor->type),
                      static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
                      static_cast<long long>(tensor->ne[2]), static_cast<long long>(tensor->ne[3]),
                      offset, size);
    }
}

// ============================================================================
// Host Reference Dequantizers Operating Directly on Repacked Super-Tiles
// ============================================================================

void agx_repack_nvfp4_dequant_row(const void * slice_agx, int64_t K, int64_t N, int64_t row, float * out_f32) {
    (void) N;
    const int64_t nb_k = K / AGX_NVFP4_TILE_K;
    const int64_t tn   = row / AGX_NVFP4_TILE_N;
    const int64_t lr   = row % AGX_NVFP4_TILE_N;
    const int     s    = static_cast<int>(lr / AGX_NVFP4_SUBTILE_N);
    const int     r    = static_cast<int>(lr % AGX_NVFP4_SUBTILE_N);

    const auto * tiles = static_cast<const agx_nvfp4_supertile *>(slice_agx);

    for (int64_t tk = 0; tk < nb_k; ++tk) {
        const agx_nvfp4_supertile & st = tiles[tn * nb_k + tk];
        for (int kh = 0; kh < 2; ++kh) {
            const int lane = r * 2 + kh;
            const uint16_t d_pair = st.d[s][lane];
            const float s0 = agx_f16_bits_to_f32(agx_decode_ue4m3_to_f16_bits(static_cast<uint8_t>(d_pair & 0xFFu)));
            const float s1 = agx_f16_bits_to_f32(agx_decode_ue4m3_to_f16_bits(static_cast<uint8_t>(d_pair >> 8)));

            const int64_t col_base = tk * AGX_NVFP4_TILE_K + kh * 32;
            for (int w = 0; w < 4; ++w) {
                const uint32_t word = st.qs[s][lane][w];
                const float scale = (w < 2) ? s0 : s1;
                for (int i = 0; i < 8; ++i) {
                    const uint32_t nib = (word >> (4 * i)) & 0x0Fu;
                    const float v = agx_f16_bits_to_f32(agx_decode_e2m1_to_f16_bits(nib));
                    out_f32[col_base + w * 8 + i] = v * scale;
                }
            }
        }
    }
}

void agx_repack_q8_0_dequant_row(const void * slice_agx, int64_t K, int64_t row, float * out_f32) {
    const int64_t nb_k     = K / QK8_0;
    const int64_t n_chunks = (nb_k + AGX_Q8_0_CHUNK_BLOCKS - 1) / AGX_Q8_0_CHUNK_BLOCKS;
    const size_t  row_b    = static_cast<size_t>(n_chunks) * AGX_Q8_0_CHUNK_BYTES;

    const auto * row_src = reinterpret_cast<const agx_q8_0_8blk *>(
        static_cast<const uint8_t *>(slice_agx) + static_cast<size_t>(row) * row_b);

    for (int64_t c = 0; c < n_chunks; ++c) {
        const agx_q8_0_8blk & chk = row_src[c];
        const int64_t b0 = c * AGX_Q8_0_CHUNK_BLOCKS;
        const int64_t count = std::min<int64_t>(AGX_Q8_0_CHUNK_BLOCKS, nb_k - b0);
        for (int64_t j = 0; j < count; ++j) {
            const float d = agx_f16_bits_to_f32(chk.d[j]);
            const int64_t col_base = (b0 + j) * QK8_0;
            for (int i = 0; i < QK8_0; ++i) {
                const float q = agx_f16_bits_to_f32(agx_decode_q8_magic_to_f16_bits(chk.qs[j * QK8_0 + i]));
                out_f32[col_base + i] = q * d;
            }
        }
    }
}

void agx_repack_mxfp8_dequant_row(const void * slice_agx, int64_t K, int64_t row, float * out_f32) {
    const int64_t nb_k     = K / AGX_MXFP8_BLOCK_K;
    const int64_t n_chunks = (nb_k + AGX_MXFP8_CHUNK_BLOCKS - 1) / AGX_MXFP8_CHUNK_BLOCKS;
    const size_t  row_b    = static_cast<size_t>(n_chunks) * AGX_MXFP8_CHUNK_BYTES;

    const auto * row_src = reinterpret_cast<const agx_mxfp8_16blk *>(
        static_cast<const uint8_t *>(slice_agx) + static_cast<size_t>(row) * row_b);

    for (int64_t c = 0; c < n_chunks; ++c) {
        const agx_mxfp8_16blk & chk = row_src[c];
        const int64_t b0 = c * AGX_MXFP8_CHUNK_BLOCKS;
        const int64_t count = std::min<int64_t>(AGX_MXFP8_CHUNK_BLOCKS, nb_k - b0);
        for (int64_t j = 0; j < count; ++j) {
            const float sc = agx_decode_ue8m0_to_f32(chk.e[j]);
            const int64_t col_base = (b0 + j) * AGX_MXFP8_BLOCK_K;
            for (int i = 0; i < AGX_MXFP8_BLOCK_K; ++i) {
                const float v = agx_f16_bits_to_f32(agx_decode_e4m3_to_f16_bits(chk.qs[j * AGX_MXFP8_BLOCK_K + i]));
                out_f32[col_base + i] = v * sc;
            }
        }
    }
}
