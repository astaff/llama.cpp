#include "types.glsl"

#define MAT_VEC_FUSION_FLAGS_BIAS0 0x1
#define MAT_VEC_FUSION_FLAGS_BIAS1 0x2
#define MAT_VEC_FUSION_FLAGS_SCALE0 0x4
#define MAT_VEC_FUSION_FLAGS_SCALE1 0x8
// gen: SWIGLU epilogue, this result is the gate (A) or the up operand (B); the other comes from fuse0
#define MAT_VEC_FUSION_FLAGS_GLU_A 0x10
#define MAT_VEC_FUSION_FLAGS_GLU_B 0x20
// gen: MUL_MAT_ID with all tokens in one dispatch: workgroup y = expert slot + nei0 * token
#define MAT_VEC_FUSION_FLAGS_GEN_TOKY 0x40
// gen: dense mat-vec epilogues (GGML_GEN_FUSE=mmgate,shexp), compiled into the GEN_GLU=1 twin
#define MAT_VEC_FUSION_FLAGS_SIGMOID 0x80
#define MAT_VEC_FUSION_FLAGS_SOFTPLUS_GATE 0x100
#define MAT_VEC_FUSION_FLAGS_ROW_SCALE0 0x200

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
#if defined(A_TYPEV4)
layout (binding = 0) readonly buffer AV4 {A_TYPEV4 data_a_v4[];};
#endif
#if defined(A_TYPE_PACKED16)
layout (binding = 0) readonly buffer A_PACKED16 {A_TYPE_PACKED16 data_a_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 0) readonly buffer A_PACKED32 {A_TYPE_PACKED32 data_a_packed32[];};
#endif
#if defined(A_TYPE_PACKED64)
layout (binding = 0) readonly buffer A_PACKED64 {A_TYPE_PACKED64 data_a_packed64[];};
#endif

layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
#ifdef B_TYPEV2
layout (binding = 1) readonly buffer BV2 {B_TYPEV2 data_b_v2[];};
#endif
#ifdef B_TYPEV4
layout (binding = 1) readonly buffer BV4 {B_TYPEV4 data_b_v4[];};
#endif

layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (binding = 3) readonly buffer Fuse0 {D_TYPE data_fuse0[];};
layout (binding = 4) readonly buffer Fuse1 {D_TYPE data_fuse1[];};

#ifdef MUL_MAT_ID
layout (binding = 5) readonly buffer IDS {int data_ids[];};
#endif

