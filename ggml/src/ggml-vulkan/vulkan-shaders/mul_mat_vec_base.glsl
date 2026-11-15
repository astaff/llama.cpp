#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

#if USE_SUBGROUP_ADD || USE_SUBGROUP_ADD_NO_SHMEM
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif

#ifdef MUL_MAT_ID
#define EXPERT_COUNT 8
#endif

#include "mul_mat_vec_iface.glsl"

#define MAT_VEC_FUSION_FLAGS_PAIR_GLU 0x400
#define MAT_VEC_FUSION_FLAGS_PAIR_MM 0x800
#define MAT_VEC_FUSION_FLAGS_PAIR_MM_SIGMOID1 0x1000

layout (binding = 3) readonly buffer A2 {A_TYPE data_a2[];};
#if defined(A_TYPEV4)
layout (binding = 3) readonly buffer A2V4 {A_TYPEV4 data_a2_v4[];};
#endif
#if defined(A_TYPE_PACKED16)
layout (binding = 3) readonly buffer A2_PACKED16 {A_TYPE_PACKED16 data_a2_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 3) readonly buffer A2_PACKED32 {A_TYPE_PACKED32 data_a2_packed32[];};
#endif
#if defined(A_TYPE_PACKED64)
layout (binding = 3) readonly buffer A2_PACKED64 {A_TYPE_PACKED64 data_a2_packed64[];};
#endif

#ifndef MUL_MAT_ID
layout (binding = 5) writeonly buffer D2 {D_TYPE data_d2[];};
#endif

layout (push_constant) uniform parameter
{
    uint ncols;
    uint stride_a;
    uint stride_b;
    uint stride_d;

    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;

    uint fusion_flags;

#ifdef MUL_MAT_ID
    uint nei0;
    uint ne11;
    uint expert_i1;
    uint nbi1;
#else
    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
    uint pair_m0;
    uint stride_d2;
    uint batch_stride_d2;
#endif
} p;

#ifdef MUL_MAT_ID
uint expert_id;
bool expert_skip; // an id of -1 skips the slot, the result row is zeroed

// gen: the expert slot and the token of this workgroup (one dispatch per token, or all tokens in y with GEN_TOKY)
uint gen_expert_i0() { return (p.fusion_flags & MAT_VEC_FUSION_FLAGS_GEN_TOKY) != 0 ? gl_WorkGroupID.y % p.nei0 : gl_WorkGroupID.y; }
uint gen_expert_i1() { return (p.fusion_flags & MAT_VEC_FUSION_FLAGS_GEN_TOKY) != 0 ? gl_WorkGroupID.y / p.nei0 : p.expert_i1; }
#endif

void get_offsets(out uint a_offset, out uint b_offset, out uint d_offset) {
#ifdef MUL_MAT_ID
    const uint expert_i0 = gen_expert_i0();
    const uint expert_i1 = gen_expert_i1();
#else
    const uint batch_idx = gl_WorkGroupID.y + p.base_work_group_y;
#endif

#ifndef MUL_MAT_ID
    uint batch_idx_a = 0;
    if (batch_idx != 0) {
        const uint i13 = batch_idx / p.ne12;
        const uint i12 = batch_idx % p.ne12;

        const uint i03 = i13 / p.broadcast3;
        const uint i02 = i12 / p.broadcast2;

        batch_idx_a = i03 * p.ne02 + i02;
    }
#else
    const int expert = data_ids[expert_i0 + expert_i1 * p.nbi1];
    expert_skip = expert == -1;
    expert_id   = expert_skip ? 0 : uint(expert); // clamp, so that the offsets below stay in range
#endif

    a_offset =
#ifdef MUL_MAT_ID
            expert_id * (p.batch_stride_a / QUANT_K);
#else
            batch_idx_a * (p.batch_stride_a / QUANT_K);
#endif
    b_offset =
#ifdef MUL_MAT_ID
            (expert_i0 % p.ne11) * p.stride_b + expert_i1 * p.batch_stride_b;
#else
            batch_idx * p.batch_stride_b;
#endif
    d_offset =
#ifdef MUL_MAT_ID
            expert_i0 * p.stride_d + expert_i1 * p.batch_stride_d;
#else
            batch_idx * p.batch_stride_d;
#endif
}

layout (constant_id = 0) const uint BLOCK_SIZE = 32;
layout (constant_id = 1) const uint NUM_ROWS = 1;
layout (constant_id = 2) const uint NUM_COLS = 1;
// gen: 0 = plain, 1 = single-op epilogue (glu/mmgate/shexp), 2 = PAIR_MM, 3 = PAIR_GLU
layout (constant_id = 3) const uint GEN_GLU = 0;
// gen: independent subgroups per workgroup in SHADER_REDUCTION_MODE_SUBGROUP (row-split)
layout (constant_id = 4) const uint NUM_SUBGROUPS = 1;
// gen: K-quant fast reduction mode (0 = strict 0-ULP upstream layout, 1 = 16-lane vec4/packed32, 2 = 8-lane/superblock)
layout (constant_id = 5) const uint GEN_FAST_KQ = 0;

#ifndef MUL_MAT_ID
FLOAT_TYPE gen_apply_mm_epilogue(FLOAT_TYPE val, const uint j, const uint first_row, const uint n) {
    if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SIGMOID) != 0) {
        precise float gen_x = float(D_TYPE(val));
        precise float gen_y = 1.0f / (1.0f + exp(-gen_x));
        return FLOAT_TYPE(gen_y);
    }
    if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SOFTPLUS_GATE) != 0) {
        precise float gen_x = float(D_TYPE(val)) + float(data_fuse0[first_row + n]);
        precise float gen_sp = (gen_x > 20.0f) ? gen_x : log(1.0f + exp(gen_x));
        precise float gen_y = gen_sp * float(data_fuse1[first_row + n]);
        return FLOAT_TYPE(gen_y);
    }
    if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_ROW_SCALE0) != 0) {
        precise float gen_x = float(D_TYPE(val));
        precise float gen_g = float(data_fuse0[j]);
        precise float gen_y = gen_x * gen_g;
        return FLOAT_TYPE(gen_y);
    }
    return val;
}
#endif

#ifdef USE_SUBGROUP_ADD_NO_SHMEM
void reduce_result(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gen_expert_i0();
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gen_expert_i0();
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    temp[j][n] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if (GEN_GLU == 1 && (p.fusion_flags & (MAT_VEC_FUSION_FLAGS_SIGMOID | MAT_VEC_FUSION_FLAGS_SOFTPLUS_GATE | MAT_VEC_FUSION_FLAGS_ROW_SCALE0)) != 0) {
                    temp[j][n] = gen_apply_mm_epilogue(temp[j][n], j, first_row, n);
                }
#endif
                if (GEN_GLU == 1 && (p.fusion_flags & (MAT_VEC_FUSION_FLAGS_GLU_A | MAT_VEC_FUSION_FLAGS_GLU_B)) != 0) {
                    // gen: same expression as swiglu.comp op(a, b) = a / (1 + exp(-a)) * b
                    const float glu_o = float(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                    const float glu_r = float(temp[j][n]);
                    temp[j][n] = FLOAT_TYPE((p.fusion_flags & MAT_VEC_FUSION_FLAGS_GLU_A) != 0 ?
                        glu_r / (1.0f + exp(-glu_r)) * glu_o : glu_o / (1.0f + exp(-glu_o)) * glu_r);
                }
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(temp[j][n]);
            }
        }
    }
}

void reduce_pair_glu(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], inout FLOAT_TYPE temp2[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n]  = subgroupAdd(temp[j][n]);
            temp2[j][n] = subgroupAdd(temp2[j][n]);
        }
    }
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if (expert_skip) {
                    data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(0);
                    continue;
                }
#endif
                const float glu_g = float(D_TYPE(temp[j][n]));
                const float glu_u = float(D_TYPE(temp2[j][n]));
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(FLOAT_TYPE(glu_g / (1.0f + exp(-glu_g)) * glu_u));
            }
        }
    }
}

#ifndef MUL_MAT_ID
void reduce_result_d2(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }
    if (tid == 0) {
        const uint d2_offset = (gl_WorkGroupID.y + p.base_work_group_y) * p.batch_stride_d2;
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                FLOAT_TYPE val = temp[j][n];
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_PAIR_MM_SIGMOID1) != 0) {
                    precise float gen_x = float(D_TYPE(val));
                    precise float gen_y = 1.0f / (1.0f + exp(-gen_x));
                    val = FLOAT_TYPE(gen_y);
                }
                data_d2[j*p.batch_stride_d2 + d2_offset + first_row + n] = D_TYPE(val);
            }
        }
    }
}
#endif
#else
#if USE_SUBGROUP_ADD
shared FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE / 8 + 1];
#else
shared FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];
#endif

void reduce_result(FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    // subgroupAdd is probably faster on devices that support it,
    // particularly when the workgroup has more than one subgroup
#if USE_SUBGROUP_ADD
    // sum up partial sums within a subgroup
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    // Go through shared memory to sum partials across subgroups
    if (gl_SubgroupInvocationID == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                tmpsh[j][n][gl_SubgroupID] = temp[j][n];
            }
        }
    }
    barrier();
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                temp[j][n] = FLOAT_TYPE(0);
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    temp[j][n] += tmpsh[j][n][s];
                }
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gen_expert_i0();
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gen_expert_i0();
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    temp[j][n] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if (GEN_GLU == 1 && (p.fusion_flags & (MAT_VEC_FUSION_FLAGS_SIGMOID | MAT_VEC_FUSION_FLAGS_SOFTPLUS_GATE | MAT_VEC_FUSION_FLAGS_ROW_SCALE0)) != 0) {
                    temp[j][n] = gen_apply_mm_epilogue(temp[j][n], j, first_row, n);
                }
#endif
                if (GEN_GLU == 1 && (p.fusion_flags & (MAT_VEC_FUSION_FLAGS_GLU_A | MAT_VEC_FUSION_FLAGS_GLU_B)) != 0) {
                    // gen: same expression as swiglu.comp op(a, b) = a / (1 + exp(-a)) * b
                    const float glu_o = float(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                    const float glu_r = float(temp[j][n]);
                    temp[j][n] = FLOAT_TYPE((p.fusion_flags & MAT_VEC_FUSION_FLAGS_GLU_A) != 0 ?
                        glu_r / (1.0f + exp(-glu_r)) * glu_o : glu_o / (1.0f + exp(-glu_o)) * glu_r);
                }
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(temp[j][n]);
            }
        }
    }
#else
    // sum up partial sums and write back result
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            tmpsh[j][n][tid] = temp[j][n];
        }
    }
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    tmpsh[j][n][tid] += tmpsh[j][n][tid + s];
                }
            }
        }
        barrier();
    }
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gen_expert_i0();
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gen_expert_i0();
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
                if (expert_skip) {
                    tmpsh[j][n][0] = FLOAT_TYPE(0);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if (GEN_GLU == 1 && (p.fusion_flags & (MAT_VEC_FUSION_FLAGS_SIGMOID | MAT_VEC_FUSION_FLAGS_SOFTPLUS_GATE | MAT_VEC_FUSION_FLAGS_ROW_SCALE0)) != 0) {
                    tmpsh[j][n][0] = gen_apply_mm_epilogue(tmpsh[j][n][0], j, first_row, n);
                }
#endif
                if (GEN_GLU == 1 && (p.fusion_flags & (MAT_VEC_FUSION_FLAGS_GLU_A | MAT_VEC_FUSION_FLAGS_GLU_B)) != 0) {
                    // gen: same expression as swiglu.comp op(a, b) = a / (1 + exp(-a)) * b
                    const float glu_o = float(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                    const float glu_r = float(tmpsh[j][n][0]);
                    tmpsh[j][n][0] = FLOAT_TYPE((p.fusion_flags & MAT_VEC_FUSION_FLAGS_GLU_A) != 0 ?
                        glu_r / (1.0f + exp(-glu_r)) * glu_o : glu_o / (1.0f + exp(-glu_o)) * glu_r);
                }
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(tmpsh[j][n][0]);
            }
        }
    }
#endif
}

void reduce_pair_glu(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], inout FLOAT_TYPE temp2[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
#if USE_SUBGROUP_ADD
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n]  = subgroupAdd(temp[j][n]);
            temp2[j][n] = subgroupAdd(temp2[j][n]);
        }
    }
    if (gl_SubgroupInvocationID == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                tmpsh[j][n][gl_SubgroupID] = temp[j][n];
            }
        }
    }
    barrier();
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                FLOAT_TYPE acc = FLOAT_TYPE(0);
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    acc += tmpsh[j][n][s];
                }
                temp[j][n] = acc;
            }
        }
    }
    barrier();
    if (gl_SubgroupInvocationID == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                tmpsh[j][n][gl_SubgroupID] = temp2[j][n];
            }
        }
    }
    barrier();
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                FLOAT_TYPE acc2 = FLOAT_TYPE(0);
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    acc2 += tmpsh[j][n][s];
                }
#ifdef MUL_MAT_ID
                if (expert_skip) {
                    data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(0);
                    continue;
                }
#endif
                const float glu_g = float(D_TYPE(temp[j][n]));
                const float glu_u = float(D_TYPE(acc2));
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(FLOAT_TYPE(glu_g / (1.0f + exp(-glu_g)) * glu_u));
            }
        }
    }
#else
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            tmpsh[j][n][tid] = temp[j][n];
        }
    }
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    tmpsh[j][n][tid] += tmpsh[j][n][tid + s];
                }
            }
        }
        barrier();
    }
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                temp[j][n] = tmpsh[j][n][0];
            }
        }
    }
    barrier();
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            tmpsh[j][n][tid] = temp2[j][n];
        }
    }
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    tmpsh[j][n][tid] += tmpsh[j][n][tid + s];
                }
            }
        }
        barrier();
    }
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if (expert_skip) {
                    data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(0);
                    continue;
                }
#endif
                const float glu_g = float(D_TYPE(temp[j][n]));
                const float glu_u = float(D_TYPE(tmpsh[j][n][0]));
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(FLOAT_TYPE(glu_g / (1.0f + exp(-glu_g)) * glu_u));
            }
        }
    }
#endif
}

#ifndef MUL_MAT_ID
void reduce_result_d2(FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    const uint d2_offset = (gl_WorkGroupID.y + p.base_work_group_y) * p.batch_stride_d2;
#if USE_SUBGROUP_ADD
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }
    if (gl_SubgroupInvocationID == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                tmpsh[j][n][gl_SubgroupID] = temp[j][n];
            }
        }
    }
    barrier();
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                FLOAT_TYPE val = FLOAT_TYPE(0);
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    val += tmpsh[j][n][s];
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_PAIR_MM_SIGMOID1) != 0) {
                    precise float gen_x = float(D_TYPE(val));
                    precise float gen_y = 1.0f / (1.0f + exp(-gen_x));
                    val = FLOAT_TYPE(gen_y);
                }
                data_d2[j*p.batch_stride_d2 + d2_offset + first_row + n] = D_TYPE(val);
            }
        }
    }
#else
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            tmpsh[j][n][tid] = temp[j][n];
        }
    }
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    tmpsh[j][n][tid] += tmpsh[j][n][tid + s];
                }
            }
        }
        barrier();
    }
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                FLOAT_TYPE val = tmpsh[j][n][0];
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_PAIR_MM_SIGMOID1) != 0) {
                    precise float gen_x = float(D_TYPE(val));
                    precise float gen_y = 1.0f / (1.0f + exp(-gen_x));
                    val = FLOAT_TYPE(gen_y);
                }
                data_d2[j*p.batch_stride_d2 + d2_offset + first_row + n] = D_TYPE(val);
            }
        }
    }
#endif
}
#endif
#endif
