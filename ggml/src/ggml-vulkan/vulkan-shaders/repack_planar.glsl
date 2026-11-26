// Shared 16B-aligned Compact 1-Superblock 0-ULP Planar layout constants &
// magic-bias decoders for Vulkan_Repack (Q4_K, Q5_K, Q6_K).
//
// Each 1-superblock (256-weight) planar block contains:
//   Q4_K (144 B = 9 uvec4 = 36 uint32, 0% bloat):
//     uvec4[0]:     exact raw { dm (u32), scales[0..3], scales[4..7], scales[8..11] }
//     uvec4[1..8]:  qs lane-interleaved: itid8=0..7 -> uvec4(qs[itid8], qs[itid8+8], qs[itid8+16], qs[itid8+24])
//   Q5_K (176 B = 11 uvec4 = 44 uint32, 0% bloat):
//     uvec4[0]:     exact raw { dm (u32), scales[0..3], scales[4..7], scales[8..11] }
//     uvec4[1..8]:  qs lane-interleaved: itid8=0..7 -> uvec4(qs[itid8], qs[itid8+8], qs[itid8+16], qs[itid8+24])
//     uvec4[9..10]: 8 qh uint32s (u32[36 + itid8] = qh[itid8])
//   Q6_K (224 B = 14 uvec4 = 56 uint32):
//     uvec4[0]:     exact raw scales[0..15] (16 int8_t)
//     uvec4[1]:     { d (u16 in .x low 16 bits), 0, 0, 0 }
//     uvec4[2..9]:  ql lane-interleaved: itid8=0..7 -> uvec4(ql[itid8], ql[itid8+8], ql[itid8+16], ql[itid8+24])
//     uvec4[10..13]: 8 qh uvec2s (u32[40 + itid8*2 + h] = qh[itid8 + 8*h])

#define REPACK_HDR_OFF_U4     0u
#define REPACK_HDR_OFF_U32    0u
#define REPACK_QS_OFF_U4      1u
#define REPACK_QS_OFF_U32     4u

#define REPACK_Q4_K_SB_U4     9u
#define REPACK_Q4_K_SB_U32    36u

#define REPACK_Q5_K_QH_OFF_U4  9u
#define REPACK_Q5_K_QH_OFF_U32 36u
#define REPACK_Q5_K_SB_U4     11u
#define REPACK_Q5_K_SB_U32    44u

#define REPACK_Q6_K_SC_OFF_U4  0u
#define REPACK_Q6_K_SC_OFF_U32 0u
#define REPACK_Q6_K_D_OFF_U4   1u
#define REPACK_Q6_K_D_OFF_U32  4u
#define REPACK_Q6_K_QL_OFF_U4  2u
#define REPACK_Q6_K_QL_OFF_U32 8u
#define REPACK_Q6_K_QH_OFF_U4  10u
#define REPACK_Q6_K_QH_OFF_U32 40u
#define REPACK_Q6_K_SB_U4     14u
#define REPACK_Q6_K_SB_U32    56u

vec4 repack_magic_u4(const uint q4) {
    const f16vec2 p02 = unpackFloat2x16((q4 & 0x000F000Fu) | 0x64006400u) - f16vec2(1024.0);
    const f16vec2 p13 = unpackFloat2x16(((q4 >> 8u) & 0x000F000Fu) | 0x64006400u) - f16vec2(1024.0);
    return vec4(float(p02.x), float(p13.x), float(p02.y), float(p13.y));
}

f16vec4 repack_magic_u4_h(const uint q4) {
    const f16vec2 p02 = unpackFloat2x16((q4 & 0x000F000Fu) | 0x64006400u) - f16vec2(1024.0);
    const f16vec2 p13 = unpackFloat2x16(((q4 >> 8u) & 0x000F000Fu) | 0x64006400u) - f16vec2(1024.0);
    return f16vec4(p02.x, p13.x, p02.y, p13.y);
}

vec4 repack_magic_u5(const uint q5_02, const uint q5_13) {
    const f16vec2 p02 = unpackFloat2x16((q5_02 & 0x001F001Fu) | 0x64006400u) - f16vec2(1024.0);
    const f16vec2 p13 = unpackFloat2x16((q5_13 & 0x001F001Fu) | 0x64006400u) - f16vec2(1024.0);
    return vec4(float(p02.x), float(p13.x), float(p02.y), float(p13.y));
}

f16vec4 repack_magic_u5_h(const uint q5_02, const uint q5_13) {
    const f16vec2 p02 = unpackFloat2x16((q5_02 & 0x001F001Fu) | 0x64006400u) - f16vec2(1024.0);
    const f16vec2 p13 = unpackFloat2x16((q5_13 & 0x001F001Fu) | 0x64006400u) - f16vec2(1024.0);
    return f16vec4(p02.x, p13.x, p02.y, p13.y);
}

vec4 repack_magic_q6(const uint q6_02, const uint q6_13) {
    const f16vec2 p02 = unpackFloat2x16((q6_02 & 0x003F003Fu) | 0x64006400u) - f16vec2(1056.0);
    const f16vec2 p13 = unpackFloat2x16((q6_13 & 0x003F003Fu) | 0x64006400u) - f16vec2(1056.0);
    return vec4(float(p02.x), float(p13.x), float(p02.y), float(p13.y));
}

f16vec4 repack_magic_q6_h(const uint q6_02, const uint q6_13) {
    const f16vec2 p02 = unpackFloat2x16((q6_02 & 0x003F003Fu) | 0x64006400u) - f16vec2(1056.0);
    const f16vec2 p13 = unpackFloat2x16((q6_13 & 0x003F003Fu) | 0x64006400u) - f16vec2(1056.0);
    return f16vec4(p02.x, p13.x, p02.y, p13.y);
}
