#pragma once

#include <cstddef>
#include <cstdint>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Pillar 2: 128B Cache-Line-Aligned Super-Tile Repacking for Apple AGX3B
// ============================================================================
//
// All repacked formats are 100% lossless bit-for-bit invertible (set_tensor ->
// get_tensor reproduces the exact original GGML block bytes) and aligned for
// coalesced 32-lane AGX simdgroup uint4 (16B) vector loads.

enum agx_repack_format {
    AGX_REPACK_NONE        = 0,
    AGX_REPACK_NVFP4_32x16 = 1, // 32 rows x 64 cols = 1152 B = 9 x 128B cache lines (0 bloat)
    AGX_REPACK_Q8_0_8BLK   = 2, // 8 Q8_0 blocks (256 weights) = 272 B = 17 x 16B planar (0 bloat)
    AGX_REPACK_MXFP8_16BLK = 3, // 16 MXFP8 blocks (512 weights) = 528 B = 33 x 16B planar (0 bloat)
};

// ----------------------------------------------------------------------------
// Format 1: AGX_NVFP4_32x16 (128B cache-line aligned, 0 bytes bloat)
// ----------------------------------------------------------------------------
// Covers 32 output rows (N) x 64 input cols (K) = 2048 E2M1 weights = 32 block_nvfp4.
// Exact size: 128 B scales + 1024 B weights = 1152 B == 32 * sizeof(block_nvfp4).
//
// Divided into 2 half-tiles (s = 0 -> rows 0..15; s = 1 -> rows 16..31):
//   Within half-tile s, a 32-lane simdgroup maps lane l in [0..31] to:
//     row_in_half = l >> 1        (0..15, 2 lanes per row -> 1-step shuffle_xor(1))
//     k_half      = l & 1         (0 -> cols 0..31; 1 -> cols 32..63)
//   Each lane performs:
//     - ONE coalesced ushort load: d[s][l] = low byte UE4M3 scale for cols +0..+15,
//                                            high byte UE4M3 scale for cols +16..+31
//     - ONE coalesced uint4 load:  qs[s][l][0..3] = 16 bytes = 32 sequential E2M1 nibbles:
//         qs[s][l][0]: cols +0..+7   (nibble i at bits [4*i+3 : 4*i])
//         qs[s][l][1]: cols +8..+15
//         qs[s][l][2]: cols +16..+23
//         qs[s][l][3]: cols +24..+31
//
// Total per half-tile: 64B scales + 512B (4 x 128B cache lines) weights.
// Total per super-tile: 128B (1 cache line) scales + 1024B (8 cache lines) weights = 1152B.
#define AGX_NVFP4_TILE_N       32
#define AGX_NVFP4_TILE_K       64
#define AGX_NVFP4_SUBTILE_N    16
#define AGX_NVFP4_ST_BYTES     1152u
#define AGX_NVFP4_ST_SCALES_B  128u
#define AGX_NVFP4_ST_QS0_OFF   128u
#define AGX_NVFP4_ST_QS1_OFF   640u

struct alignas(128) agx_nvfp4_supertile {
    uint16_t d[2][32];       // offset 0..127   (128 B = 1 cache line)
    uint32_t qs[2][32][4];   // offset 128..1151 (1024 B = 8 cache lines)
};

// ----------------------------------------------------------------------------
// Format 2: AGX_Q8_0_8BLK (16B aligned planar, 0 bytes bloat for K % 256 == 0)
// ----------------------------------------------------------------------------
// Covers 8 contiguous Q8_0 blocks along K = 256 weights = 272 bytes == 8 * 34 B.
//   d[0..7]  : 16 bytes (8 FP16 scales, 1 x uint4 load)
//   qs[0..255]: 256 bytes (8 x 32 sequential int8 weights, 16 x uint4 loads = 2 x 128B lines)
// For rows where K % 256 != 0 (e.g. K = 1152 -> 36 blocks = 4*8 + 4), the final
// chunk is zero-padded in its unused trailing blocks so GPU kernels run 100%
// branchless & remainder-free.
#define AGX_Q8_0_CHUNK_K      256
#define AGX_Q8_0_CHUNK_BLOCKS 8
#define AGX_Q8_0_CHUNK_BYTES  272u

struct alignas(16) agx_q8_0_8blk {
    uint16_t d[8];     // offset 0..15   (16 B: 8 x ggml_half)
    int8_t   qs[256];  // offset 16..271 (256 B: 8 x 32 int8 weights)
};

// ----------------------------------------------------------------------------
// Format 3: AGX_MXFP8_16BLK (16B aligned planar, 0 bytes bloat for K % 512 == 0)
// ----------------------------------------------------------------------------
// Covers 16 contiguous MXFP8 blocks (32 E4M3 weights + 1 UE8M0 scale each)
// = 512 weights = 528 bytes == 16 * 33 B.
//   e[0..15]  : 16 bytes (16 UE8M0 power-of-two exponent scales, 1 x uint4 load)
//   qs[0..511]: 512 bytes (16 x 32 signed E4M3 weights, 32 x uint4 loads = 4 x 128B lines)
#define AGX_MXFP8_BLOCK_K      32
#define AGX_MXFP8_CHUNK_K      512
#define AGX_MXFP8_CHUNK_BLOCKS 16
#define AGX_MXFP8_CHUNK_BYTES  528u

struct alignas(16) agx_mxfp8_16blk {
    uint8_t e[16];     // offset 0..15   (16 B: 16 x UE8M0 scales)
    uint8_t qs[512];   // offset 16..527 (512 B: 16 x 32 E4M3 FP8 weights)
};

// ----------------------------------------------------------------------------
// Tensor layout metadata (stored in ggml_tensor->extra for AGX0_Repack tensors)
// ----------------------------------------------------------------------------
struct agx_repack_tensor_info {
    enum agx_repack_format format;
    enum ggml_type         orig_type;
    int64_t                ne0;          // K (inner dim)
    int64_t                ne1;          // N (rows per 2D slice)
    int64_t                ne2;          // experts / batch2
    int64_t                ne3;          // batch3
    size_t                 slice_bytes;  // byte stride between 2D slices (ne2/ne3)
    size_t                 total_bytes;  // total repacked allocation size (128B aligned)
    uint32_t               n_tiles_n;    // NVFP4: ceil(N/32); Q8_0/MXFP8: N
    uint32_t               n_tiles_k;    // NVFP4: K/64; Q8_0: ceil(K/256); MXFP8: ceil(K/512)
    uint32_t               row_stride_b; // Q8_0/MXFP8: bytes per row; NVFP4: bytes per 32-row tile-row
};

// Query whether a tensor shape/type is eligible for offline super-tile repacking
enum agx_repack_format agx_repack_select_format(enum ggml_type type, int64_t ne0, int64_t ne1);

// Compute exact repacked layout descriptors & allocation size (128B rounded)
void   agx_repack_describe(enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                           struct agx_repack_tensor_info * out);
size_t agx_repack_get_alloc_size(const struct ggml_tensor * tensor);
size_t agx_repack_slice_stride(enum ggml_type type, int64_t ne0, int64_t ne1);

// Direct 2D/3D pack & unpack (100% lossless bit-for-bit round-trip)
bool agx_repack_nvfp4_pack  (const void * src_ggml, void * dst_agx, int64_t K, int64_t N, int64_t n_slices);
bool agx_repack_nvfp4_unpack(const void * src_agx,  void * dst_ggml, int64_t K, int64_t N, int64_t n_slices);

bool agx_repack_q8_0_pack   (const void * src_ggml, void * dst_agx, int64_t K, int64_t N, int64_t n_slices);
bool agx_repack_q8_0_unpack (const void * src_agx,  void * dst_ggml, int64_t K, int64_t N, int64_t n_slices);

// MXFP8 raw block layout: each block is { uint8_t e; uint8_t qs[32]; } (33 bytes)
bool agx_repack_mxfp8_pack  (const void * src_raw33, void * dst_agx, int64_t K, int64_t N, int64_t n_slices);
bool agx_repack_mxfp8_unpack(const void * src_agx,   void * dst_raw33, int64_t K, int64_t N, int64_t n_slices);

// Tensor-level buffer helpers supporting explicit dst/src pointers (used by tests)
bool agx_repack_set_tensor_buf(const struct ggml_tensor * tensor, void * dst_repacked,
                               const void * src_ggml, size_t offset, size_t size);
bool agx_repack_get_tensor_buf(const struct ggml_tensor * tensor, const void * src_repacked,
                               void * dst_ggml, size_t offset, size_t size);

// Backend cross-module hooks matching agx-common.h (operate on tensor->data)
bool   agx_repack_supports_tensor(const struct ggml_tensor * tensor);
void   agx_repack_set_tensor(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void   agx_repack_get_tensor(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

// ----------------------------------------------------------------------------
// Wave 4 Theorem W4.1: Load-Time Vertical Weight Coalescing (Q8_0_8BLK)
// ----------------------------------------------------------------------------
// Concatenates up to AGX_COALESCED_MAX_MATS 2D Q8_0 matrices sharing input dim K
// (e.g. wq, wqkv_gate, attn_rel, attn_sink_proj, wk, wv -> 8,984 x 2,448 B) into
// ONE contiguous AGX_BO_SLC_STREAM | AGX_BO_READONLY GPU BO in Q8_0_8BLK layout,
// preserving 16B row alignment, 0 extra VRAM bloat, and 0-ULP GEMV equivalence.

#define AGX_COALESCED_MAX_MATS 16u

struct agx_drm_device;
struct agx_bo;

struct agx_coalesced_weight {
    struct agx_bo * bo;
    uint64_t        gpu_va;
    void *          cpu_ptr;
    uint32_t        K;
    uint32_t        total_rows;
    uint32_t        row_stride_b;
    size_t          total_bytes;
    uint32_t        n_mats;
    uint32_t        row_offsets[AGX_COALESCED_MAX_MATS];
    uint32_t        row_counts[AGX_COALESCED_MAX_MATS];
    size_t          byte_offsets[AGX_COALESCED_MAX_MATS];
};

bool agx_repack_coalesce_q8_0_matrices(struct agx_drm_device *            dev,
                                       const struct ggml_tensor * const * mats,
                                       uint32_t                           n_mats,
                                       struct agx_coalesced_weight *      out);

void agx_repack_coalesced_free(struct agx_drm_device *       dev,
                               struct agx_coalesced_weight * w);

// ----------------------------------------------------------------------------
// Host-side bit-exact reference implementations of our branchless FTZ-safe
// GPU GPR decoders (guaranteed identical to kernels/dequant_supertile.*)
// ----------------------------------------------------------------------------

// E2M1 nibble (0..15) -> exact IEEE FP16 bits (0 subnormals, FTZ-safe!)
// Maps {0..15} -> {+0, +0.5, +1, +1.5, +2, +3, +4, +6, -0, -0.5, -1, -1.5, -2, -3, -4, -6}
uint16_t agx_decode_e2m1_to_f16_bits(uint32_t nib);

// UE4M3 scale byte (0..255) -> exact IEEE FP16 bits (0 subnormals, FTZ-safe!)
// Matches 2.0 * ggml_ue4m3_to_fp32(u) so that decode_e2m1(n) * decode_ue4m3(u)
// == kvalues_fp4[n] * ggml_ue4m3_to_fp32(u) bit-for-bit across all 256 x 16 pairs.
uint16_t agx_decode_ue4m3_to_f16_bits(uint8_t u);

// Signed E4M3 FP8 byte (0..255) -> exact IEEE FP16 bits (0 subnormals, FTZ-safe!)
uint16_t agx_decode_e4m3_to_f16_bits(uint8_t b);

// Signed int8 (-128..127) -> exact IEEE FP16 bits via magic-bias (0x6400 | (b ^ 0x80)) - 1152.0h
uint16_t agx_decode_q8_magic_to_f16_bits(int8_t q);

// UE8M0 byte (0..255) -> exact FP32 power-of-two scale (2^(e - 127), e==0 -> 2^-127)
float    agx_decode_ue8m0_to_f32(uint8_t e);

// Full host reference dequantization of repacked super-tiles (for unit tests)
void agx_repack_nvfp4_dequant_row(const void * slice_agx, int64_t K, int64_t N, int64_t row, float * out_f32);
void agx_repack_q8_0_dequant_row (const void * slice_agx, int64_t K, int64_t row, float * out_f32);
void agx_repack_mxfp8_dequant_row(const void * slice_agx, int64_t K, int64_t row, float * out_f32);

#ifdef __cplusplus
}
#endif
