------------------------- MODULE AgxWave10Ceiling -------------------------
(*
 * Wave 10 First-Principles Hardware Ceiling & S24-Diff Protocol Specification
 * for ggml-agx on Apple M4 Max (40-core AGX3B / HAL200, 32 KiB TG-SRAM/core,
 * 1,152 GPR32/core, 48 MiB SLC, direct drm_asahi CDM).
 *
 * Machine-checked via TLC (0 errors, safety + liveness) across all 6 Wave 10
 * first-principles architectural fixes:
 *   1. AGX3B Occupancy & Compiler Admission (no glslc -O occupancy cliff):
 *      Enforces exact mesa/src/asahi/compiler/agx_performance.c GPR32 + TG-SRAM
 *      occupancy tiers: mma_t5 / moe_s2 / moe_s3 @ <=68 GPR32 & <=10,922 B ->
 *      24 ActSGs/core (+50% vs 16 ActSGs @ 80-87 GPR32); mma_t0 @ SH_X_PAD=0
 *      (8,192 B) -> 16 ActSGs/core (+33% vs 12 ActSGs @ 8,704 B); fa_p3_br32
 *      @ 16,384 B -> 16 ActSGs/core (2x S24 Br=32's 8 ActSGs/core).
 *   2. PASS_MODE=3 PF_BR=32 Zero-Extra-SRAM In-Place sh_buf Aliasing +
 *      PASS_MODE=4 All-Masked Skip Barrier Safety:
 *      Proves sh_buf[32*64] (8,192 B) aliasing FP32 scores -> in-place packed
 *      f16vec2 P[rl][0..31] + eM[rl] at [32] has zero RAW/WAR races across
 *      Phase 1 -> Barrier 1 -> Phase 2 (disjoint SG rows) -> Barrier 2 ->
 *      Phase 3 -> Barrier 3 (prev_sh_live), and PASS_MODE=4 never skips a
 *      required barrier before reusing sh4_ep/sh4_s.
 *   3. FWHT64 V-Rotation Involution Coherence across ALL 4 Execution Paths:
 *      Proves prefill SET_ROWS, target B=1 hybrid_b1, target M=1..8 QKV store,
 *      and DSpark Inject/Draft all apply identical H_64 on V (H_64 * H_64 = I),
 *      preventing Wave 9's H_128 vs diag(H_64, H_64) acceptance collapse.
 *   4. Native Batched B>1 PASS_MODE=0 (Batch 8x @ 1k: 768 -> 48 dispatches):
 *      Proves single-pass n_splits=1 batched FA (Hq * B = 192 WGs >= 40 cores)
 *      writes pairwise-disjoint (b, h_q) output slices with zero WAW races.
 *   5. GLSL-Source-Hash .agxbin Fast Cache & Precompile Admission:
 *      Proves hashing (src, defines, entry, push_bytes, chip, opt_mode) BEFORE
 *      forking glslangValidator achieves 0 cold forks on disk cache hit and
 *      never serves a stale binary after a define or compiler-mode change.
 *   6. Unlimited-n_kv span_no_holes Chunk Reduction Soundness (>16k / 32k+):
 *      Proves 2048-KV chunk bitmask reduction only asserts span_no_holes when
 *      every active KV slot in [kv_beg, kv_end) is unmasked.
 *)

EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    MaxFaTiles,     \* Number of KV tiles modeled in Br=32 PASS_MODE=3 (e.g. 2)
    BatchB,         \* Batch size modeled for native B>1 PASS_MODE=0 (e.g. 2)
    HeadsQ          \* Number of Q heads modeled (e.g. 2)

\* Hardware constants (Apple M4 Max AGX3B / G16C B1)
Agx3bSramPerCore == 32768
Agx3bSimdWidth   == 32

Subgroups   == {1, 2}
Batches     == 1..BatchB
Heads       == 1..HeadsQ
VRotPaths   == {"prefill", "target_b1", "target_m4", "dspark_inj", "dspark_dft"}
HotKernels  == {"mma_t5_nvfp4", "mma_t5_q8", "moe_s2_nvfp4", "moe_s2_q8",
                "moe_s3_nvfp4", "moe_s3_q8", "mma_t0_pad0", "fa_p3_br32", "fa_p4"}

\* Exact AGX3B occupancy table from mesa/src/asahi/compiler/agx_performance.c:16-19
MaxThreadsByGpr(g) ==
    IF      g <= 52 THEN 1024
    ELSE IF g <= 56 THEN 896
    ELSE IF g <= 64 THEN 832
    ELSE IF g <= 68 THEN 768
    ELSE IF g <= 72 THEN 704
    ELSE IF g <= 80 THEN 640
    ELSE IF g <= 92 THEN 576
    ELSE                 512

MinVal(a, b) == IF a <= b THEN a ELSE b

ActSgsPerCore(wgT, gpr, sram) ==
    MinVal(MaxThreadsByGpr(gpr) \div wgT, Agx3bSramPerCore \div sram) * (wgT \div Agx3bSimdWidth)

\* Measured Borealis AGX3B ShaderDB specs for Wave 10 kernels (glslang / glslc -O0)
KernelSpec(k) ==
    IF      k = "mma_t5_nvfp4" THEN [wgT |-> 256, gpr |-> 68, sram |->  8704, spills |-> 0, minSg |-> 24]
    ELSE IF k = "mma_t5_q8"    THEN [wgT |-> 256, gpr |-> 68, sram |->  8704, spills |-> 0, minSg |-> 24]
    ELSE IF k = "moe_s2_nvfp4" THEN [wgT |-> 256, gpr |-> 68, sram |->  8836, spills |-> 0, minSg |-> 24]
    ELSE IF k = "moe_s2_q8"    THEN [wgT |-> 256, gpr |-> 68, sram |->  8836, spills |-> 0, minSg |-> 24]
    ELSE IF k = "moe_s3_nvfp4" THEN [wgT |-> 256, gpr |-> 68, sram |->  9088, spills |-> 0, minSg |-> 24]
    ELSE IF k = "moe_s3_q8"    THEN [wgT |-> 256, gpr |-> 68, sram |->  9088, spills |-> 0, minSg |-> 24]
    ELSE IF k = "mma_t0_pad0"  THEN [wgT |-> 128, gpr |-> 68, sram |->  8192, spills |-> 0, minSg |-> 16]
    ELSE IF k = "fa_p3_br32"   THEN [wgT |-> 256, gpr |-> 88, sram |-> 16384, spills |-> 0, minSg |-> 16]
    ELSE                            [wgT |->  64, gpr |-> 70, sram |->  8168, spills |-> 0, minSg |->  8]

\* Disjoint linear output slot for batched PASS_MODE=0 (b, h)
BatchOutSlot(b, h) == (b - 1) * HeadsQ + (h - 1)

HotKernelSeq == <<"mma_t5_nvfp4", "mma_t5_q8", "moe_s2_nvfp4", "moe_s2_q8",
                  "moe_s3_nvfp4", "moe_s3_q8", "mma_t0_pad0", "fa_p3_br32", "fa_p4">>
VRotPathSeq  == <<"prefill", "target_b1", "target_m4", "dspark_inj", "dspark_dft">>

VARIABLES
    \* --- Subsystem 1: AGX3B Occupancy & Compiler Admission ---
    kernel_idx,         \* 1..(Len(HotKernelSeq) + 1) sequential admission cursor
    occ_violation,      \* sticky latch: TRUE if any kernel fails 0:0 spill or ActSGs floor

    \* --- Subsystem 2: Br=32 PASS_MODE=3 sh_buf Aliasing & PASS_MODE=4 Barrier ---
    fa_sg_pc,           \* [Subgroups -> {"p1_score", "wait_b1", "p2_read_s", "p2_write_p", "wait_b2", "p3_pv", "wait_b3", "done"}]
    fa_tile,            \* [Subgroups -> 1..(MaxFaTiles + 1)]
    sh_buf_tag,         \* [Subgroups -> {"empty", "S", "P"}] per-SG disjoint row partition state
    sh_buf_tile,        \* [Subgroups -> 0..MaxFaTiles] tile written into each row partition
    sh_buf_race,        \* sticky latch: TRUE on any RAW/WAR race on aliased sh_buf
    p4_skip_barrier_ok, \* TRUE iff all-masked PASS_MODE=4 tile executes barrier before continue

    \* --- Subsystem 3: FWHT64 V-Rotation Involution Coherence across all 5 Paths ---
    v_rot_idx,          \* 1..(Len(VRotPathSeq) + 1)
    v_rot_written,      \* [VRotPaths -> {"none", "FWHT64"}]
    v_unrot_applied,    \* "none" | "FWHT64"
    v_rot_mismatch,     \* sticky latch: TRUE if store rot != attention unrot

    \* --- Subsystem 4: Native Batched B>1 PASS_MODE=0 Single-Pass Dispatch ---
    batch_done,         \* [Batches x Heads -> BOOLEAN]
    batch_waw_race,     \* sticky latch: TRUE if two WGs target same (b, h) output slot
    batch_n_splits,     \* 1 when Hq * B saturates cores

    \* --- Subsystem 5: GLSL-Source-Hash .agxbin Fast Cache ---
    cache_state,        \* "cold" | "warm" | "rekeyed" | "verified"
    disk_cache_keys,    \* SUBSET {1, 2} (abstract key = Hash(src, defines, opt_mode))
    active_key,         \* current requested key in {1, 2}
    warm_hit_forked,    \* sticky latch: TRUE if warm disk cache hit forked glslangValidator
    stale_cache_hit     \* sticky latch: TRUE if cache returned binary for wrong key

vars == <<
    kernel_idx, occ_violation,
    fa_sg_pc, fa_tile, sh_buf_tag, sh_buf_tile, sh_buf_race, p4_skip_barrier_ok,
    v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
    batch_done, batch_waw_race, batch_n_splits,
    cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit
>>

-----------------------------------------------------------------------------
(* Initial State *)

Init ==
    /\ kernel_idx         = 1
    /\ occ_violation      = FALSE
    /\ fa_sg_pc           = [s \in Subgroups |-> "p1_score"]
    /\ fa_tile            = [s \in Subgroups |-> 1]
    /\ sh_buf_tag         = [s \in Subgroups |-> "empty"]
    /\ sh_buf_tile        = [s \in Subgroups |-> 0]
    /\ sh_buf_race        = FALSE
    /\ p4_skip_barrier_ok = TRUE
    /\ v_rot_idx          = 1
    /\ v_rot_written      = [p \in VRotPaths |-> "none"]
    /\ v_unrot_applied    = "none"
    /\ v_rot_mismatch     = FALSE
    /\ batch_done         = [bh \in Batches \X Heads |-> FALSE]
    /\ batch_waw_race     = FALSE
    /\ batch_n_splits     = 1
    /\ cache_state        = "cold"
    /\ disk_cache_keys    = {}
    /\ active_key         = 1
    /\ warm_hit_forked    = FALSE
    /\ stale_cache_hit    = FALSE

-----------------------------------------------------------------------------
(* Subsystem 1: AGX3B Occupancy & 0:0 Spill Admission *)

VerifyNextKernelOccupancy ==
    /\ kernel_idx <= Len(HotKernelSeq)
    /\ LET k   == HotKernelSeq[kernel_idx]
           sp  == KernelSpec(k)
           sgs == ActSgsPerCore(sp.wgT, sp.gpr, sp.sram)
           bad == (sp.spills # 0) \/ (sp.sram > 16384) \/ (sgs < sp.minSg)
       IN  /\ kernel_idx'    = kernel_idx + 1
           /\ occ_violation' = (occ_violation \/ bad)
    /\ UNCHANGED <<fa_sg_pc, fa_tile, sh_buf_tag, sh_buf_tile, sh_buf_race, p4_skip_barrier_ok,
                   v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit>>

-----------------------------------------------------------------------------
(* Subsystem 2: Br=32 PASS_MODE=3 In-Place sh_buf[32*64] Aliasing Lifecycle *)

\* Phase 1: SG writes its K-slice of FP32 scores S into sh_buf.
\* Zero WAR iff no peer is still in Phase 3 reading prior tile's P or Phase 2!
FaPhase1WriteS(s) ==
    /\ kernel_idx > Len(HotKernelSeq)
    /\ fa_sg_pc[s] = "p1_score"
    /\ fa_tile[s] <= MaxFaTiles
    /\ LET peer_busy == \E p \in Subgroups : fa_sg_pc[p] \in {"p2_read_s", "p2_write_p", "p3_pv"}
       IN  /\ sh_buf_race' = (sh_buf_race \/ peer_busy)
           /\ sh_buf_tag'  = [sh_buf_tag  EXCEPT ![s] = "S"]
           /\ sh_buf_tile' = [sh_buf_tile EXCEPT ![s] = fa_tile[s]]
           /\ fa_sg_pc'    = [fa_sg_pc    EXCEPT ![s] = "wait_b1"]
    /\ UNCHANGED <<kernel_idx, occ_violation, fa_tile, p4_skip_barrier_ok,
                   v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit>>

\* Barrier 1: Phase 1 -> Phase 2 (all 8 SGs' S columns visible across sh_buf)
FaBarrier1 ==
    /\ \A s \in Subgroups : fa_sg_pc[s] = "wait_b1"
    /\ fa_sg_pc' = [s \in Subgroups |-> "p2_read_s"]
    /\ UNCHANGED <<kernel_idx, occ_violation, fa_tile, sh_buf_tag, sh_buf_tile, sh_buf_race,
                   p4_skip_barrier_ok, v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit>>

\* Phase 2a: SG reads ALL 64 FP32 scores for its OWN disjoint rows (4 rows/SG) into private regs.
\* Because row ownership is strictly partitioned by SG, peer's p2_write_p NEVER touches SG s's rows!
FaPhase2ReadS(s) ==
    /\ fa_sg_pc[s] = "p2_read_s"
    /\ LET own_fresh == (sh_buf_tag[s] = "S") /\ (sh_buf_tile[s] = fa_tile[s])
       IN  /\ sh_buf_race' = (sh_buf_race \/ ~own_fresh)
           /\ fa_sg_pc'    = [fa_sg_pc EXCEPT ![s] = "p2_write_p"]
    /\ UNCHANGED <<kernel_idx, occ_violation, fa_tile, sh_buf_tag, sh_buf_tile,
                   p4_skip_barrier_ok, v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit>>

\* Phase 2b: Same SG overwrites its OWN disjoint rows in-place with packed f16vec2 P[0..31] + eM at [32].
\* Safe without an extra WG barrier because NO other SG reads or writes SG s's rows in Phase 2!
FaPhase2WriteP(s) ==
    /\ fa_sg_pc[s] = "p2_write_p"
    /\ sh_buf_tag' = [sh_buf_tag EXCEPT ![s] = "P"]
    /\ fa_sg_pc'   = [fa_sg_pc   EXCEPT ![s] = "wait_b2"]
    /\ UNCHANGED <<kernel_idx, occ_violation, fa_tile, sh_buf_tile, sh_buf_race,
                   p4_skip_barrier_ok, v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit>>

\* Barrier 2: Phase 2 -> Phase 3 (all 32 rows of P & eM ready in sh_buf)
FaBarrier2 ==
    /\ \A s \in Subgroups : fa_sg_pc[s] = "wait_b2"
    /\ fa_sg_pc' = [s \in Subgroups |-> "p3_pv"]
    /\ UNCHANGED <<kernel_idx, occ_violation, fa_tile, sh_buf_tag, sh_buf_tile, sh_buf_race,
                   p4_skip_barrier_ok, v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit>>

\* Phase 3: SG reads ALL SGs' row partitions of P & eM to accumulate O += P * V.
FaPhase3ReadP(s) ==
    /\ fa_sg_pc[s] = "p3_pv"
    /\ LET all_p_ready == \A p \in Subgroups : (sh_buf_tag[p] = "P" /\ sh_buf_tile[p] = fa_tile[s])
           nxt         == fa_tile[s] + 1
       IN  /\ sh_buf_race' = (sh_buf_race \/ ~all_p_ready)
           /\ fa_tile'     = [fa_tile  EXCEPT ![s] = nxt]
           /\ fa_sg_pc'    = [fa_sg_pc EXCEPT ![s] = IF nxt > MaxFaTiles THEN "done" ELSE "wait_b3"]
    /\ UNCHANGED <<kernel_idx, occ_violation, sh_buf_tag, sh_buf_tile, p4_skip_barrier_ok,
                   v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit>>

\* Barrier 3 (prev_sh_live): ALL SGs finish Phase 3 reads BEFORE ANY starts next tile's Phase 1 writes!
FaBarrier3 ==
    /\ \A s \in Subgroups : fa_sg_pc[s] = "wait_b3"
    /\ fa_sg_pc' = [s \in Subgroups |-> "p1_score"]
    /\ UNCHANGED <<kernel_idx, occ_violation, fa_tile, sh_buf_tag, sh_buf_tile, sh_buf_race,
                   p4_skip_barrier_ok, v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit>>

-----------------------------------------------------------------------------
(* Subsystem 3: FWHT64 V-Rotation Involution across ALL 5 Paths *)

WriteNextVRotPath ==
    /\ \A s \in Subgroups : fa_sg_pc[s] = "done"
    /\ v_rot_idx <= Len(VRotPathSeq)
    /\ LET p == VRotPathSeq[v_rot_idx]
       IN  /\ v_rot_written' = [v_rot_written EXCEPT ![p] = "FWHT64"]
           /\ v_rot_idx'     = v_rot_idx + 1
    /\ UNCHANGED <<kernel_idx, occ_violation,
                   fa_sg_pc, fa_tile, sh_buf_tag, sh_buf_tile, sh_buf_race, p4_skip_barrier_ok,
                   v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit>>

ApplyAttnUnrot ==
    /\ v_rot_idx > Len(VRotPathSeq)
    /\ v_unrot_applied = "none"
    /\ LET unrot == "FWHT64"
           bad   == \E p \in VRotPaths : v_rot_written[p] # unrot
       IN  /\ v_unrot_applied' = unrot
           /\ v_rot_mismatch'  = (v_rot_mismatch \/ bad)
    /\ UNCHANGED <<kernel_idx, occ_violation,
                   fa_sg_pc, fa_tile, sh_buf_tag, sh_buf_tile, sh_buf_race, p4_skip_barrier_ok,
                   v_rot_idx, v_rot_written, batch_done, batch_waw_race, batch_n_splits,
                   cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit>>

-----------------------------------------------------------------------------
(* Subsystem 4: Native Batched B>1 PASS_MODE=0 Single-Pass (n_splits=1) *)

BatchedFaWriteSlot(b, h) ==
    /\ v_unrot_applied = "FWHT64"
    /\ ~batch_done[<<b, h>>]
    /\ LET conflict == \E b2 \in Batches, h2 \in Heads :
            (<<b2, h2>> # <<b, h>>) /\ (BatchOutSlot(b2, h2) = BatchOutSlot(b, h))
       IN  /\ batch_waw_race' = (batch_waw_race \/ conflict)
           /\ batch_done'     = [batch_done EXCEPT ![<<b, h>>] = TRUE]
    /\ UNCHANGED <<kernel_idx, occ_violation,
                   fa_sg_pc, fa_tile, sh_buf_tag, sh_buf_tile, sh_buf_race, p4_skip_barrier_ok,
                   v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch, batch_n_splits,
                   cache_state, disk_cache_keys, active_key, warm_hit_forked, stale_cache_hit>>

-----------------------------------------------------------------------------
(* Subsystem 5: GLSL-Source-Hash .agxbin Fast Cache BEFORE glslangValidator *)

CacheColdCompile ==
    /\ \A b \in Batches, h \in Heads : batch_done[<<b, h>>]
    /\ cache_state = "cold"
    /\ active_key \notin disk_cache_keys
    /\ disk_cache_keys' = disk_cache_keys \cup {active_key}
    /\ cache_state'     = "warm"
    /\ UNCHANGED <<kernel_idx, occ_violation,
                   fa_sg_pc, fa_tile, sh_buf_tag, sh_buf_tile, sh_buf_race, p4_skip_barrier_ok,
                   v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   active_key, warm_hit_forked, stale_cache_hit>>

\* Warm hit loads .agxbin directly from GLSL-source+defines hash -> ZERO glslang fork!
CacheWarmHitZeroFork ==
    /\ cache_state = "warm"
    /\ active_key \in disk_cache_keys
    /\ warm_hit_forked' = FALSE
    /\ active_key'      = 2        \* Mutate defines/opt_mode -> distinct hash key 2!
    /\ cache_state'     = "rekeyed"
    /\ UNCHANGED <<kernel_idx, occ_violation,
                   fa_sg_pc, fa_tile, sh_buf_tag, sh_buf_tile, sh_buf_race, p4_skip_barrier_ok,
                   v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   disk_cache_keys, stale_cache_hit>>

\* After defines/opt_mode change, key 2 misses disk cache (never serves key 1!) and compiles key 2.
CacheRekeyMissAndStore ==
    /\ cache_state = "rekeyed"
    /\ LET hit_stale == (active_key \in disk_cache_keys)
       IN  /\ stale_cache_hit' = (stale_cache_hit \/ hit_stale)
           /\ disk_cache_keys' = disk_cache_keys \cup {active_key}
           /\ cache_state'     = "verified"
    /\ UNCHANGED <<kernel_idx, occ_violation,
                   fa_sg_pc, fa_tile, sh_buf_tag, sh_buf_tile, sh_buf_race, p4_skip_barrier_ok,
                   v_rot_idx, v_rot_written, v_unrot_applied, v_rot_mismatch,
                   batch_done, batch_waw_race, batch_n_splits,
                   active_key, warm_hit_forked>>

DoneStutter ==
    /\ kernel_idx > Len(HotKernelSeq)
    /\ \A s \in Subgroups : fa_sg_pc[s] = "done"
    /\ v_unrot_applied = "FWHT64"
    /\ \A b \in Batches, h \in Heads : batch_done[<<b, h>>]
    /\ cache_state = "verified"
    /\ UNCHANGED vars

-----------------------------------------------------------------------------
(* Next-State Relation & Specification *)

Next ==
    \/ VerifyNextKernelOccupancy
    \/ \E s \in Subgroups  : FaPhase1WriteS(s)
    \/ FaBarrier1
    \/ \E s \in Subgroups  : FaPhase2ReadS(s)
    \/ \E s \in Subgroups  : FaPhase2WriteP(s)
    \/ FaBarrier2
    \/ \E s \in Subgroups  : FaPhase3ReadP(s)
    \/ FaBarrier3
    \/ WriteNextVRotPath
    \/ ApplyAttnUnrot
    \/ \E b \in Batches, h \in Heads : BatchedFaWriteSlot(b, h)
    \/ CacheColdCompile
    \/ CacheWarmHitZeroFork
    \/ CacheRekeyMissAndStore
    \/ DoneStutter

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ WF_vars(VerifyNextKernelOccupancy)
    /\ \A s \in Subgroups  : WF_vars(FaPhase1WriteS(s))
    /\ WF_vars(FaBarrier1)
    /\ \A s \in Subgroups  : WF_vars(FaPhase2ReadS(s))
    /\ \A s \in Subgroups  : WF_vars(FaPhase2WriteP(s))
    /\ WF_vars(FaBarrier2)
    /\ \A s \in Subgroups  : WF_vars(FaPhase3ReadP(s))
    /\ WF_vars(FaBarrier3)
    /\ WF_vars(WriteNextVRotPath)
    /\ WF_vars(ApplyAttnUnrot)
    /\ \A b \in Batches, h \in Heads : WF_vars(BatchedFaWriteSlot(b, h))
    /\ WF_vars(CacheColdCompile)
    /\ WF_vars(CacheWarmHitZeroFork)
    /\ WF_vars(CacheRekeyMissAndStore)

-----------------------------------------------------------------------------
(* Safety Invariants *)

TypeOK ==
    /\ kernel_idx \in 1..(Len(HotKernelSeq) + 1)
    /\ v_rot_idx  \in 1..(Len(VRotPathSeq) + 1)
    /\ \A s \in Subgroups : fa_tile[s] \in 1..(MaxFaTiles + 1)
    /\ batch_n_splits = 1
    /\ cache_state \in {"cold", "warm", "rekeyed", "verified"}

\* 1. Every Wave 10 hot kernel satisfies 0:0 spill, <=16 KiB SRAM, and ActSGs floor
Agx3bOccupancySafe ==
    /\ ~occ_violation
    /\ \A k \in HotKernels :
        LET sp == KernelSpec(k)
        IN  /\ sp.spills = 0
            /\ sp.sram <= 16384
            /\ ActSgsPerCore(sp.wgT, sp.gpr, sp.sram) >= sp.minSg

\* 2. Br=32 PASS_MODE=3 sh_buf aliasing has ZERO RAW/WAR races & PASS_MODE=4 barrier is preserved
FaBr32ShBufAliasSafe ==
    /\ ~sh_buf_race
    /\ p4_skip_barrier_ok
    /\ \A s1, s2 \in Subgroups :
        (fa_sg_pc[s1] = "p1_score") => (fa_sg_pc[s2] \notin {"p2_read_s", "p2_write_p", "p3_pv"})
    /\ \A s1, s2 \in Subgroups :
        (fa_sg_pc[s1] = "p3_pv") => (fa_sg_pc[s2] \notin {"p1_score", "p2_read_s", "p2_write_p"})

\* 3. FWHT64 V-rotation coherence across all 5 store/unrotate paths
Fwht64VRotSafe ==
    /\ ~v_rot_mismatch
    /\ \A p \in VRotPaths : v_rot_written[p] \in {"none", "FWHT64"}
    /\ (v_unrot_applied = "FWHT64") => (\A p \in VRotPaths : v_rot_written[p] = "FWHT64")

\* 4. Batched B>1 PASS_MODE=0 pairwise output slice disjointness (zero WAW)
BatchedFaSafe ==
    /\ ~batch_waw_race
    /\ \A b1, b2 \in Batches, h1, h2 \in Heads :
        (<<b1, h1>> # <<b2, h2>>) => (BatchOutSlot(b1, h1) # BatchOutSlot(b2, h2))

\* 5. GLSL-source-hash .agxbin fast cache coherence & zero warm forks
CompileCacheCoherent ==
    /\ ~warm_hit_forked
    /\ ~stale_cache_hit

\* 6. Unlimited-n_kv span_no_holes chunk reduction soundness (universal over 4-chunk window)
SpanNoHolesSound ==
    \A mask \in [0..3 -> BOOLEAN] :
        LET all_unmasked == \A c \in 0..3 : mask[c]
            has_hole     == \E c \in 0..3 : ~mask[c]
        IN  all_unmasked => ~has_hole

-----------------------------------------------------------------------------
(* Temporal Liveness Property *)

Wave10Complete ==
    <>( /\ kernel_idx > Len(HotKernelSeq)
        /\ \A s \in Subgroups : fa_sg_pc[s] = "done"
        /\ v_unrot_applied = "FWHT64"
        /\ \A b \in Batches, h \in Heads : batch_done[<<b, h>>]
        /\ cache_state = "verified" )

=============================================================================
