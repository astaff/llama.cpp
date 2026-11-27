# `ggml-agx`: Bare-Metal Apple Silicon AGX3 Backend for `ggml` / `llama.cpp`

## Goal
Build a first-class, bare-metal Linux Apple Silicon GPU backend (`GGML_AGX`, `ggml/src/ggml-agx/`) targeting **M4 Max 40-core GPU (`J616c` / `T6041`, AGX3, 64 GB LPDDR5X-8533 @ 546 GB/s theoretical / ~505 GB/s physical ceiling)** directly over Linux `drm_asahi` (`include/drm-uapi/asahi_drm.h`) + Borealis/AGX3 ISA, bypassing generic Vulkan/SPIR-V dispatch & barrier overhead to push **Inkling 2 Mini** (26.4B-A3.4B, 3x looped middle 12 layers = 48 passes, NVFP4 + Q8_0/MXFP8 + DSpark MTP) as close as physically possible to the AGX3 hardware roofline.

Full reference specification from the shared architecture brief: `/home/astaff/m4-rig/var/pr-prep/ggml-agx-gemini-share.md`.
Reference Mesa/Borealis/`drm_asahi` source tree on this host (read-only reference): `/home/astaff/m4-rig/var/pr-prep/mesa-s24` (`include/drm-uapi/asahi_drm.h`, `src/asahi/lib/`, `src/asahi/compiler/`, `src/asahi/vulkan/`).

## Six Architectural Pillars

1. **Pillar 1 — Direct `drm_asahi` UAPI Runtime & Native AGX3 ISA / Borealis Kernel Pipeline (`ggml-agx` Backend):**
   - Direct `/dev/dri/renderD*` `DRM_IOCTL_ASAHI_*` device, VM, BO, queue, and CDM (`Command Data Master`) compute command stream submission (`ggml-agx-drm.{h,cpp}`), with optional Borealis compiler linkage or pre-assembled/JIT AGX3 binary shader emission (`<= 96` 32-bit GPRs/lane discipline for 20–32 simdgroups/core occupancy; predicated execution without warp-divergence cleanup splits; direct-register `simdgroup_matrix` 8x8 FMA).
2. **Pillar 2 — Zero-Loss Cache-Line-Aligned Repacking (`AGX_NVFP4` & `Q8_0`/`MXFP8`) + Branchless GPR Decode:**
   - 128-byte cache-line-aligned super-tiles matched to 32-lane AGX simdgroups (`uint4` 16B coalesced weight load + `ushort`/`uint` scale load) and branchless bitwise `E2M1 -> FP16` normal-exponent shift decode (FTZ-safe on AGX coopmat, zero LUT bank conflicts) + `Q8_0` / `MXFP8` fast decode.
3. **Pillar 3 — Eliminating the 700+ Dispatch CDM Bottleneck via Persistent / Single-Stream 48-Pass Execution:**
   - Collapse Inkling 2 Mini's 48 passes (`0..5`, `6..17 x 3`, `18..23`, LM head + argmax/sampler) into a single pre-recorded CDM submission / persistent 40-core execution loop with 3 fused stages per pass:
     1. `fused_loop_prep_qkv_conv_rope` (depth-mix residual attention + 4x rank-64 cross-pass gates in 32 KB threadgroup SRAM + QKV + dynamic K/V shortconv + L2 clip / Hadamard + KV write + 2304->128 router top-8)
     2. `fused_attn_outgate_proj` (40-pass SWA-1024 ring attention + 8-pass split-K Flash-Decoding + learned sink + rel-pos bias + sigmoid output gate + `W_O`)
     3. `fused_moe_gate_up_situ_down` (GPU-indirect top-8 expert workqueue reading router IDs in SLC, zero host sync)
4. **Pillar 4 — 48 MB SLC Residency & Page-Table / VM Cache Policy Partitioning:**
   - Exploit `drm_asahi` VM/BO flags and access patterns so cold 14.7 GB MoE expert pages stream without thrashing SLC while hot activations, cross-pass residuals, SWA-1024 KV ring (~84 MB / hot window), router tables, and DSpark MTP weights stay hot in SLC.
5. **Pillar 5 — GPU-Resident Bitonic/Radix Token-to-Expert Bucketing (`DSpark` `M=4..8` & Prefill `simdgroup_matrix` GEMM):**
   - Single-simdgroup bitonic sort deduplicating shared experts across `M=5` speculative verify tokens into predicated width-`M_bucket` vec-mat micro-tiles (each touched expert read once from DRAM), switching to `simdgroup_matrix` 8x8 tiled GEMM when `M_bucket >= 6..8`.
6. **Pillar 6 — TLA+-Verified Sense-Reversing Atomic / CDM `wait_for_idle` Barrier Elision:**
   - Formal weak-memory / USC-SLC coherence model + sense-reversing device-scope atomic or minimal CDM barrier transitions across all 48 passes without full dirty-cache DRAM flushes.

## Hardware & Safety Rules (`AGENTS.md` — Mandatory)
- Target hardware: `m4-nixos` via `ssh -F /home/astaff/m4-rig/config/ssh/m4.conf m4-nixos` (user-space only under `/home/astaff/llama/llama-ggml-agx` and `/tmp/ggml-agx-*`).
- **ALWAYS** wrap every GPU test/benchmark on `m4-nixos` with `su - astaff -s /run/current-system/sw/bin/bash -c "python3 /home/astaff/llama/guarded-run.py <label> -- <cmd>"` so hardware runs serialize cleanly via `~/memo/results/benchmark.lock` with concurrent jobs.
- **NEVER** touch `m4-macos`, SEP, boot policy, Linux root, ESP, NVRAM, or existing baseline directories (`/home/astaff/llama/llama-s24`, `/home/astaff/llama/build-s23`, `/home/astaff/llama/mesa-retest-20261125`).
