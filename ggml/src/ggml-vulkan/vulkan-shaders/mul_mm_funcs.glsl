// k_pair is the K coordinate measured in FLOAT_TYPEV2 elements.
uint a_shmem_index(uint m, uint k_pair) {
    if (APPLY_SLM_A_RESHAPE) {
        const uint tile_width = TK / 2;
        return (k_pair / tile_width) * BM * tile_width
             + m * tile_width
             + k_pair % tile_width;
    }
    return m * SHMEM_STRIDE + k_pair;
}

uint a_shmem_stride() {
    return APPLY_SLM_A_RESHAPE ? TK / 2 : SHMEM_STRIDE;
}

void store_a(uint m, uint k_pair, FLOAT_TYPEV2 value) {
    buf_a[a_shmem_index(m, k_pair)] = value;
}

#if !defined(MUL_MAT_ID)
uint b_shmem_index(uint n, uint k_pair) {
    return n * SHMEM_STRIDE + k_pair;
}

uint b_shmem_stride() {
    return SHMEM_STRIDE;
}

void store_b(uint n, uint k_pair, FLOAT_TYPEV2 value) {
    buf_b[b_shmem_index(n, k_pair)] = value;
}
#endif

bool mm_can_stage_a() {
#if (defined(DATA_A_F32) || defined(DATA_A_F16)) && (LOAD_VEC_A == 8 || LOAD_VEC_A == 4)
    return ALIGNED != 0;
#elif defined(DATA_A_IQ4_XS)
    return true;
#elif defined(MULMAT_QUANT)
    return MmTypeA == GGML_TYPE_Q8_0 || MmTypeA == GGML_TYPE_Q4_K || MmTypeA == GGML_TYPE_Q5_K || MmTypeA == GGML_TYPE_Q6_K;
#else
    return false;
#endif
}

void commit_stage_a(const uint col_sh, const uint row, const FLOAT_TYPEV2 stage_v[8]) {
#if (defined(DATA_A_F32) || defined(DATA_A_F16)) && LOAD_VEC_A == 8
    const uint kp = row * 4u;
    store_a(col_sh, kp + 0u, stage_v[0]);
    store_a(col_sh, kp + 1u, stage_v[1]);
    store_a(col_sh, kp + 2u, stage_v[2]);
    store_a(col_sh, kp + 3u, stage_v[3]);
#elif (defined(DATA_A_F32) || defined(DATA_A_F16)) && LOAD_VEC_A == 4
    const uint kp = row * 2u;
    store_a(col_sh, kp + 0u, stage_v[0]);
    store_a(col_sh, kp + 1u, stage_v[1]);
#elif defined(DATA_A_IQ4_XS)
    const uint kp = row * 2u;
    store_a(col_sh, kp + 0u, stage_v[0]);
    store_a(col_sh, kp + 1u, stage_v[1]);
    store_a(col_sh, kp + 8u, stage_v[2]);
    store_a(col_sh, kp + 9u, stage_v[3]);
#elif defined(MULMAT_QUANT)
    const uint kp = row * 8u;
    store_a(col_sh, kp + 0u, stage_v[0]);
    store_a(col_sh, kp + 1u, stage_v[1]);
    store_a(col_sh, kp + 2u, stage_v[2]);
    store_a(col_sh, kp + 3u, stage_v[3]);
    store_a(col_sh, kp + 4u, stage_v[4]);
    store_a(col_sh, kp + 5u, stage_v[5]);
    store_a(col_sh, kp + 6u, stage_v[6]);
    store_a(col_sh, kp + 7u, stage_v[7]);
#endif
}

#define MM_STORE_A(slot, m, kp, val) store_a(m, kp, val)

void load_a_to_shmem(const uint pos_a, const uint row, const uint col_sh, const uint col_mem, const uint idx_m, const uint block, const uint end_k) {
#define MM_DATA_A        data_a
#define MM_DATA_A_SCALAR data_a_scalar
#define MM_DATA_A_P16    data_a_packed16
#define MM_DATA_A_P32    data_a_packed32
#define MM_Q1_0          a_q1_0
#define MM_Q2_0          a_q2_0
#define MM_Q2_K          a_q2_k
#define MM_Q3_K          a_q3_k
#define MM_Q4_K          a_q4_k
#define MM_Q5_K          a_q5_k
#define MM_Q6_K          a_q6_k
#define MM_TQ1_0         a_tq1_0
#define MM_TQ2_0         a_tq2_0
#define MM_Q4_0_P16      a_q4_0_p16
#define MM_Q5_0_P16      a_q5_0_p16
#define MM_Q8_0_P16      a_q8_0_p16
#define MM_Q3_K_P16      a_q3_k_p16
#define MM_Q6_K_P16      a_q6_k_p16
#define MM_Q4_1_P32      a_q4_1_p32
#define MM_Q5_1_P32      a_q5_1_p32
#define MM_Q2_K_P32      a_q2_k_p32
#define MM_Q4_K_P32      a_q4_k_p32
#define MM_Q5_K_P32      a_q5_k_p32
#include "mul_mm_load_a_body.glsl"
#undef MM_DATA_A
#undef MM_DATA_A_SCALAR
#undef MM_DATA_A_P16
#undef MM_DATA_A_P32
#undef MM_Q1_0
#undef MM_Q2_0
#undef MM_Q2_K
#undef MM_Q3_K
#undef MM_Q4_K
#undef MM_Q5_K
#undef MM_Q6_K
#undef MM_TQ1_0
#undef MM_TQ2_0
#undef MM_Q4_0_P16
#undef MM_Q5_0_P16
#undef MM_Q8_0_P16
#undef MM_Q3_K_P16
#undef MM_Q6_K_P16
#undef MM_Q4_1_P32
#undef MM_Q5_1_P32
#undef MM_Q2_K_P32
#undef MM_Q4_K_P32
#undef MM_Q5_K_P32
}

#ifdef MM_FUSED
void load_u_to_shmem(const uint pos_a, const uint row, const uint col_sh, const uint col_mem, const uint idx_m, const uint block, const uint end_k) {
#define MM_DATA_A        data_u
#define MM_DATA_A_SCALAR data_u_scalar
#define MM_DATA_A_P16    data_u_packed16
#define MM_DATA_A_P32    data_u_packed32
#define MM_Q1_0          u_q1_0
#define MM_Q2_0          u_q2_0
#define MM_Q2_K          u_q2_k
#define MM_Q3_K          u_q3_k
#define MM_Q4_K          u_q4_k
#define MM_Q5_K          u_q5_k
#define MM_Q6_K          u_q6_k
#define MM_TQ1_0         u_tq1_0
#define MM_TQ2_0         u_tq2_0
#define MM_Q4_0_P16      u_q4_0_p16
#define MM_Q5_0_P16      u_q5_0_p16
#define MM_Q8_0_P16      u_q8_0_p16
#define MM_Q3_K_P16      u_q3_k_p16
#define MM_Q6_K_P16      u_q6_k_p16
#define MM_Q4_1_P32      u_q4_1_p32
#define MM_Q5_1_P32      u_q5_1_p32
#define MM_Q2_K_P32      u_q2_k_p32
#define MM_Q4_K_P32      u_q4_k_p32
#define MM_Q5_K_P32      u_q5_k_p32
#include "mul_mm_load_a_body.glsl"
#undef MM_DATA_A
#undef MM_DATA_A_SCALAR
#undef MM_DATA_A_P16
#undef MM_DATA_A_P32
#undef MM_Q1_0
#undef MM_Q2_0
#undef MM_Q2_K
#undef MM_Q3_K
#undef MM_Q4_K
#undef MM_Q5_K
#undef MM_Q6_K
#undef MM_TQ1_0
#undef MM_TQ2_0
#undef MM_Q4_0_P16
#undef MM_Q5_0_P16
#undef MM_Q8_0_P16
#undef MM_Q3_K_P16
#undef MM_Q6_K_P16
#undef MM_Q4_1_P32
#undef MM_Q5_1_P32
#undef MM_Q2_K_P32
#undef MM_Q4_K_P32
#undef MM_Q5_K_P32
}
#endif

#undef MM_STORE_A

#ifdef COOPMAT
#define MM_STORE_A(slot, m, kp, val) stage_v[slot] = (val)

void stage_a_to_reg(const uint pos_a, const uint row, const uint col_sh, const uint col_mem, const uint idx_m, const uint block, const uint end_k, out FLOAT_TYPEV2 stage_v[8]) {
    [[unroll]] for (uint s = 0u; s < 8u; ++s) { stage_v[s] = FLOAT_TYPEV2(0.0f); }
    if (idx_m >= p.M) { return; }
#define MM_DATA_A        data_a
#define MM_DATA_A_SCALAR data_a_scalar
#define MM_DATA_A_P16    data_a_packed16
#define MM_DATA_A_P32    data_a_packed32
#define MM_Q1_0          a_q1_0
#define MM_Q2_0          a_q2_0
#define MM_Q2_K          a_q2_k
#define MM_Q3_K          a_q3_k
#define MM_Q4_K          a_q4_k
#define MM_Q5_K          a_q5_k
#define MM_Q6_K          a_q6_k
#define MM_TQ1_0         a_tq1_0
#define MM_TQ2_0         a_tq2_0
#define MM_Q4_0_P16      a_q4_0_p16
#define MM_Q5_0_P16      a_q5_0_p16
#define MM_Q8_0_P16      a_q8_0_p16
#define MM_Q3_K_P16      a_q3_k_p16
#define MM_Q6_K_P16      a_q6_k_p16
#define MM_Q4_1_P32      a_q4_1_p32
#define MM_Q5_1_P32      a_q5_1_p32
#define MM_Q2_K_P32      a_q2_k_p32
#define MM_Q4_K_P32      a_q4_k_p32
#define MM_Q5_K_P32      a_q5_k_p32
#include "mul_mm_load_a_body.glsl"
#undef MM_DATA_A
#undef MM_DATA_A_SCALAR
#undef MM_DATA_A_P16
#undef MM_DATA_A_P32
#undef MM_Q1_0
#undef MM_Q2_0
#undef MM_Q2_K
#undef MM_Q3_K
#undef MM_Q4_K
#undef MM_Q5_K
#undef MM_Q6_K
#undef MM_TQ1_0
#undef MM_TQ2_0
#undef MM_Q4_0_P16
#undef MM_Q5_0_P16
#undef MM_Q8_0_P16
#undef MM_Q3_K_P16
#undef MM_Q6_K_P16
#undef MM_Q4_1_P32
#undef MM_Q5_1_P32
#undef MM_Q2_K_P32
#undef MM_Q4_K_P32
#undef MM_Q5_K_P32
}

#ifdef MM_FUSED
void stage_u_to_reg(const uint pos_a, const uint row, const uint col_sh, const uint col_mem, const uint idx_m, const uint block, const uint end_k, out FLOAT_TYPEV2 stage_v[8]) {
    [[unroll]] for (uint s = 0u; s < 8u; ++s) { stage_v[s] = FLOAT_TYPEV2(0.0f); }
    if (idx_m >= p.M) { return; }
#define MM_DATA_A        data_u
#define MM_DATA_A_SCALAR data_u_scalar
#define MM_DATA_A_P16    data_u_packed16
#define MM_DATA_A_P32    data_u_packed32
#define MM_Q1_0          u_q1_0
#define MM_Q2_0          u_q2_0
#define MM_Q2_K          u_q2_k
#define MM_Q3_K          u_q3_k
#define MM_Q4_K          u_q4_k
#define MM_Q5_K          u_q5_k
#define MM_Q6_K          u_q6_k
#define MM_TQ1_0         u_tq1_0
#define MM_TQ2_0         u_tq2_0
#define MM_Q4_0_P16      u_q4_0_p16
#define MM_Q5_0_P16      u_q5_0_p16
#define MM_Q8_0_P16      u_q8_0_p16
#define MM_Q3_K_P16      u_q3_k_p16
#define MM_Q6_K_P16      u_q6_k_p16
#define MM_Q4_1_P32      u_q4_1_p32
#define MM_Q5_1_P32      u_q5_1_p32
#define MM_Q2_K_P32      u_q2_k_p32
#define MM_Q4_K_P32      u_q4_k_p32
#define MM_Q5_K_P32      u_q5_k_p32
#include "mul_mm_load_a_body.glsl"
#undef MM_DATA_A
#undef MM_DATA_A_SCALAR
#undef MM_DATA_A_P16
#undef MM_DATA_A_P32
#undef MM_Q1_0
#undef MM_Q2_0
#undef MM_Q2_K
#undef MM_Q3_K
#undef MM_Q4_K
#undef MM_Q5_K
#undef MM_Q6_K
#undef MM_TQ1_0
#undef MM_TQ2_0
#undef MM_Q4_0_P16
#undef MM_Q5_0_P16
#undef MM_Q8_0_P16
#undef MM_Q3_K_P16
#undef MM_Q6_K_P16
#undef MM_Q4_1_P32
#undef MM_Q5_1_P32
#undef MM_Q2_K_P32
#undef MM_Q4_K_P32
#undef MM_Q5_K_P32
}
#endif

#undef MM_STORE_A
#endif

#if !defined(MUL_MAT_ID)
void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint idx_n, const uint block, const uint end_k) {
#if LOAD_VEC_B == 8
    if (ALIGNED != 0) {
        const uint kp = row * (LOAD_VEC_B / 2);
        if (idx_n < p.N) {
            // Not supported for b_type bf16 because bf16mat2x4 does not exist
            const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
            FLOAT_TYPEV8 bb = FLOAT_TYPEV8(data_b[idx]);
            store_b(col, kp + 0, bb[0].xy);
            store_b(col, kp + 1, bb[0].zw);
            store_b(col, kp + 2, bb[1].xy);
            store_b(col, kp + 3, bb[1].zw);
        } else {
            store_b(col, kp + 0, FLOAT_TYPEV2(0.0f));
            store_b(col, kp + 1, FLOAT_TYPEV2(0.0f));
            store_b(col, kp + 2, FLOAT_TYPEV2(0.0f));
            store_b(col, kp + 3, FLOAT_TYPEV2(0.0f));
        }
        return;
    }
#elif LOAD_VEC_B == 4
    if (ALIGNED != 0) {
        const uint kp = row * (LOAD_VEC_B / 2);
        if (idx_n < p.N) {
            const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
#if defined(DATA_B_BF16)
            FLOAT_TYPEV4 bb = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_b[idx]));
#else
            FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
            store_b(col, kp + 0, bb.xy);
            store_b(col, kp + 1, bb.zw);
        } else {
            store_b(col, kp + 0, FLOAT_TYPEV2(0.0f));
            store_b(col, kp + 1, FLOAT_TYPEV2(0.0f));
        }
        return;
    }
#endif
    const uint idx = pos_b + col * p.stride_b + row * 2;
    if (idx_n < p.N && block + row * 2 + 1 < end_k) {
        store_b(col, row, FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]),
                                       TO_FLOAT_TYPE(data_b_scalar[idx + 1])));
    } else if (idx_n < p.N && block + row * 2 < end_k) {
        store_b(col, row, FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]), 0.0f));
    } else {
        store_b(col, row, FLOAT_TYPEV2(0.0f));
    }
}
#else
void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint ic, const uint _ne1, const uint block, const uint end_k) {
#if LOAD_VEC_B == 8
    if (ALIGNED != 0) {
        // Not supported for b_type bf16 because bf16mat2x4 does not exist
        const u16vec2 row_idx = row_ids[col];
        const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
        const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
        FLOAT_TYPEV8 bb = FLOAT_TYPEV8(data_b[idx]);
        buf_b[buf_idx + 0] = bb[0].xy;
        buf_b[buf_idx + 1] = bb[0].zw;
        buf_b[buf_idx + 2] = bb[1].xy;
        buf_b[buf_idx + 3] = bb[1].zw;
        return;
    }
#elif LOAD_VEC_B == 4
    if (ALIGNED != 0) {
        const u16vec2 row_idx = row_ids[col];
        const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
        const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
#if defined(DATA_B_BF16)
        FLOAT_TYPEV4 bb = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_b[idx]));
#else
        FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
        buf_b[buf_idx + 0] = bb.xy;
        buf_b[buf_idx + 1] = bb.zw;
        return;
    }
#endif
    const uint row_i = ic * BN + col;
    const uint buf_idx = col * SHMEM_STRIDE + row;
    if (row_i < _ne1 && block + row * 2 + 1 < end_k) {
        const u16vec2 row_idx = row_ids[col];
        const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
        buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]),
                                        TO_FLOAT_TYPE(data_b_scalar[idx + 1]));
    } else if (row_i < _ne1 && block + row * 2 < end_k) {
        const u16vec2 row_idx = row_ids[col];
        const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
        buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]), 0.0f);
    } else {
        buf_b[buf_idx] = FLOAT_TYPEV2(0.0f);
    }
}
#endif

#ifdef COOPMAT
void stage_b_to_reg(const uint pos_b, const uint row, const uint col, const uint idx_n, out FLOAT_TYPEV2 stage_bv[4]) {
#if !defined(MUL_MAT_ID)
    if (idx_n >= p.N) {
        stage_bv[0] = FLOAT_TYPEV2(0.0f);
        stage_bv[1] = FLOAT_TYPEV2(0.0f);
        stage_bv[2] = FLOAT_TYPEV2(0.0f);
        stage_bv[3] = FLOAT_TYPEV2(0.0f);
        return;
    }
    const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
#else
    if (col >= _ne1) {
        stage_bv[0] = FLOAT_TYPEV2(0.0f);
        stage_bv[1] = FLOAT_TYPEV2(0.0f);
        stage_bv[2] = FLOAT_TYPEV2(0.0f);
        stage_bv[3] = FLOAT_TYPEV2(0.0f);
        return;
    }
    const u16vec2 row_idx = row_ids[col];
    const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
#endif
#if LOAD_VEC_B == 8
    FLOAT_TYPEV8 bb = FLOAT_TYPEV8(data_b[idx]);
    stage_bv[0] = bb[0].xy;
    stage_bv[1] = bb[0].zw;
    stage_bv[2] = bb[1].xy;
    stage_bv[3] = bb[1].zw;
#elif LOAD_VEC_B == 4
#if defined(DATA_B_BF16)
    FLOAT_TYPEV4 bb = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_b[idx]));
#else
    FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
    stage_bv[0] = bb.xy;
    stage_bv[1] = bb.zw;
    stage_bv[2] = FLOAT_TYPEV2(0.0f);
    stage_bv[3] = FLOAT_TYPEV2(0.0f);
#endif
}

void commit_stage_b(const uint col, const uint row, const FLOAT_TYPEV2 stage_bv[4]) {
#if LOAD_VEC_B == 8 && defined(FLOAT16) && !defined(DATA_A_BF16)
    if (AGX_DIRECT_STORE != 0) {
        if (APPLY_SLM_A_RESHAPE) {
            if (BK == 32) {
                const uint odd_c   = col & 1u;
                const bool is_odd  = odd_c != 0u;
                const uint base_sh = row * ((BN << 2u) + 8u) + ((col >> 3u) << 5u) + ((col & 7u) >> 1u);
                [[unroll]] for (uint q = 0u; q < 2u; ++q) {
                    const f16vec2 v0 = stage_bv[2u * q + 0u];
                    const f16vec2 v1 = stage_bv[2u * q + 1u];
                    const f16vec2 my_keep = is_odd ? f16vec2(v0.y, v1.y) : f16vec2(v0.x, v1.x);
                    const f16vec2 my_send = is_odd ? f16vec2(v0.x, v1.x) : f16vec2(v0.y, v1.y);
                    const f16vec2 partner = unpackFloat2x16(subgroupShuffleXor(packFloat2x16(my_send), 4u));
                    buf_b[base_sh + (((4u * q + 0u) + odd_c) << 2u)] = is_odd ? f16vec2(partner.x, my_keep.x) : f16vec2(my_keep.x, partner.x);
                    buf_b[base_sh + (((4u * q + 2u) + odd_c) << 2u)] = is_odd ? f16vec2(partner.y, my_keep.y) : f16vec2(my_keep.y, partner.y);
                }
                return;
            }
        }
    }
#endif
#if !defined(MUL_MAT_ID)
#if LOAD_VEC_B == 8
    const uint kp = row * 4u;
    store_b(col, kp + 0u, stage_bv[0]);
    store_b(col, kp + 1u, stage_bv[1]);
    store_b(col, kp + 2u, stage_bv[2]);
    store_b(col, kp + 3u, stage_bv[3]);
#elif LOAD_VEC_B == 4
    const uint kp = row * 2u;
    store_b(col, kp + 0u, stage_bv[0]);
    store_b(col, kp + 1u, stage_bv[1]);
#endif
#else
#if LOAD_VEC_B == 8
    const uint buf_idx = col * SHMEM_STRIDE + row * 4u;
    buf_b[buf_idx + 0u] = stage_bv[0];
    buf_b[buf_idx + 1u] = stage_bv[1];
    buf_b[buf_idx + 2u] = stage_bv[2];
    buf_b[buf_idx + 3u] = stage_bv[3];
#elif LOAD_VEC_B == 4
    const uint buf_idx = col * SHMEM_STRIDE + row * 2u;
    buf_b[buf_idx + 0u] = stage_bv[0];
    buf_b[buf_idx + 1u] = stage_bv[1];
#endif
#endif
}
#endif
