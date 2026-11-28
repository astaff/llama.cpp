#pragma once

#include <cstdint>

// ============================================================================
// Pillar 3: Fused Cross-Pass Rank-64 Controller Gates in Native FP32 SRAM
// ============================================================================
// Exact match to `llama_model_inkling2::graph::control` (inkling2.cpp L255-265):
//   gate_in = [rms_norm(cur, 1e-6); rms_norm(msg, 1e-6)]   // staged in 32KB TG SRAM (FP32!)
//   h       = ctrl_down   @ gate_in                         // [rank] in FP32 SRAM
//   gain    = tanh(ctrl_gain   @ h)                         // [n_embd] in FP32
//   forget  = tanh(ctrl_forget @ h)                         // [n_embd] in FP32
//   y       = (cur - forget * cur) + gain * msg             // [n_embd] in FP32 (NEVER BF16!)
// Optional fused trailing RMSNorm:
//   out     = rms_norm(y, eps_norm) * norm_w
//
// Supports single-controller (FFN) or batched multi-controller (Q, K, V in one dispatch)
// via `desc_va` pointing to `n_ctrls` `agx_ctrl_gate_desc` structs in SLC/arg-ring.

#define AGX_WTYPE_F32        0u
#define AGX_WTYPE_F16        1u
#define AGX_WTYPE_BF16       2u
#define AGX_WTYPE_Q8_0_RAW   3u
#define AGX_WTYPE_Q8_0_8BLK  4u

#define AGX_CTRL_FLAG_WRITE_PRE_NORM (1u << 0)

struct alignas(16) agx_ctrl_gate_desc {
    uint64_t cur_va;           // [n_embd, n_tokens] F32 (stride = cur_stride_f32)
    uint64_t msg_va;           // [n_embd, n_tokens] F32 (or strided by msg_stride_f32)
    uint64_t down_va;          // [2*n_embd, rank]
    uint64_t gain_va;          // [rank, n_embd]
    uint64_t forget_va;        // [rank, n_embd]
    uint64_t norm_w_va;        // optional [n_embd] F32 (0 = skip trailing RMSNorm)
    uint64_t out_va;           // [n_embd, n_tokens] F32 (post-norm if norm_w_va != 0, else y)
    uint64_t out_pre_norm_va;  // optional [n_embd, n_tokens] F32 when WRITE_PRE_NORM set
    uint32_t down_wtype;
    uint32_t gain_wtype;
    uint32_t forget_wtype;
    uint32_t down_row_bytes;   // byte stride per row of ctrl_down (2*n_embd cols)
    uint32_t gain_row_bytes;   // byte stride per row of ctrl_gain (rank cols)
    uint32_t forget_row_bytes; // byte stride per row of ctrl_forget (rank cols)
    uint32_t cur_stride_f32;   // elements between consecutive tokens in cur (default n_embd)
    uint32_t msg_stride_f32;   // elements between consecutive tokens in msg (default n_embd)
    uint32_t msg_offset_f32;   // initial float offset into msg_va for this ctrl
    uint32_t out_stride_f32;   // elements between consecutive tokens in out (default n_embd)
    uint32_t flags;
    uint32_t pad0;
};
static_assert(sizeof(agx_ctrl_gate_desc) == 112, "agx_ctrl_gate_desc layout mismatch");

struct agx_ctrl_gates_push {
    uint64_t desc_va;      // GPU VA of agx_ctrl_gate_desc[n_ctrls]
    uint32_t n_embd;       // 2304 (must be <= 3840 for 32 KB FP32 SRAM staging)
    uint32_t rank;         // 64   (must be <= 256)
    uint32_t n_tokens;     // M
    uint32_t n_ctrls;      // 1 (FFN) or 3 (Q, K, V)
    float    eps_ctrl;     // 1e-6f (Inkling 2 control RMSNorm eps)
    float    eps_norm;     // model f_norm_rms_eps (e.g. 1e-6f)
};
static_assert(sizeof(agx_ctrl_gates_push) <= 256, "agx_ctrl_gates_push exceeds 256B limit");
