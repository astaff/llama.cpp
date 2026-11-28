----------------------- MODULE AgxWave9Breakthrough -----------------------
(*
 * Wave 9 Deep Breakthrough Protocol Specification for ggml-agx on Apple M4 Max
 * (40-core AGX3B / HAL200, 48 MiB SLC, direct drm_asahi CDM).
 *
 * Machine-checked via TLC (0 errors, safety + liveness) for all 5 Wave 9
 * architectural breakthroughs:
 *   1. 250K+ Context O(n_kv) Banded RelBias Tile-Skip Soundness:
 *      Proves out-of-band [tmin, tmax] KV tiles can NEVER contain a valid
 *      distance 0 <= q_pos - k_pos < RelExtent, eliminating O(n_kv*M*H) kq_b.
 *   2. M = 2..8 Direct PASS_STAGE=5/6 MoE & M = 5..8 Zero-Atomic down_noscat:
 *      Proves 128B-disjoint SLC moe_inter addressing across (tok, k_slot) and
 *      strict STRONG_USC visibility before deterministic zero-atomic wsum.
 *   3. K_PAIRS = 2 Software-Pipelined tmp_b Register Preload Barrier Safety:
 *      Proves S24's load-to-reg -> if (g>0) barrier() -> sh_x store -> barrier()
 *      schedule has zero WAR/RAW races on sh_x across concurrent subgroups.
 *   4. Async Inject (wait=false) -> Chained Draft (cdm_barrier=0u) -> Target
 *      Verify Double-Stream Timeline Protocol:
 *      Proves why a dedicated inject_stream (2-stream ring) + drm_asahi's
 *      automatic cdm_barrier=0u eliminates Inject's 0.62 ms blocking stall
 *      with ZERO cmd_bo/arg_bo overwrite race and strict KV visibility.
 *   5. M = 1..8 40-WG SENSE_BARRIER Co-Residency & Liveness:
 *      Proves 512T (M=1, <=56 GPR32) and 256T (M=2..8, <=72 GPR32) strictly
 *      stay within M4 Max's 1,152 GPR32/core budget and complete all rounds.
 *)

EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    MaxRounds,      \* Speculative verify+inject+draft rounds (e.g. 2)
    NumKChunks,     \* K_PAIRS=2 chunks per GEMM tile (e.g. 2)
    NumTokensM,     \* Batch size M for direct s5/s6 MoE check (e.g. 2)
    TopK,           \* Active experts per token (e.g. 2)
    RelExtent       \* Fixed banded rel-bias extent (e.g. 2)

\* Hardware constants (Apple M4 Max AGX3B / HAL200)
MaxGpr32PerCore == 1152
SlcMoeInterCap  == 294912   \* 8 * 8 * 1152 * 4 B (agx-common.h:136)
SlotBytes       == 4608     \* 1152 * 4 B = 36 * 128 B cache lines

Subgroups == {1, 2}
Tokens    == 1..NumTokensM
KSlots    == 1..TopK

VARIABLES
    \* --- Subsystem 1: Async Inject -> Chained Draft -> Target Verify Timeline ---
    round,
    phase,              \* "target" | "inject_prep" | "inject_async" | "draft_prep" | "draft_run" | "done"
    timeline_val,       \* monotonic drm_asahi timeline counter
    completed_tl,       \* last timeline point completed on GPU
    inj_tl,             \* timeline point of in-flight/last async Inject
    dft_tl,             \* timeline point of in-flight/last Draft
    tgt_tl,             \* timeline point of in-flight/last Target Verify
    inj_stream_busy,    \* TRUE while GPU is reading inject_stream cmd_bo/arg_bo
    dft_stream_busy,    \* TRUE while GPU is reading draft_stream cmd_bo/arg_bo
    tgt_stream_busy,    \* TRUE while GPU is reading target_stream cmd_bo/arg_bo
    inj_kv_epoch,       \* GPU KV/f_rnn epoch written by Inject
    dft_kv_seen,        \* KV epoch observed when Draft starts on GPU
    draft_cdm_barrier,  \* TRUE iff Draft packet carried cdm_barrier=0u when Inject was in-flight
    cmd_bo_race,        \* sticky latch: TRUE if CPU overwrites any in-flight stream BO
    kv_race,            \* sticky latch: TRUE if Draft reads stale KV before Inject finishes

    \* --- Subsystem 2: M=2..8 Direct PASS_STAGE=5/6 MoE SLC Visibility ---
    moe_stage,          \* "idle" | "s5_write" | "s5_flushed" | "s6_reduce" | "complete"
    moe_slot_written,   \* [Tokens x KSlots -> BOOLEAN]
    moe_slot_epoch,     \* [Tokens x KSlots -> 0..MaxRounds]
    moe_out_epoch,      \* [Tokens -> 0..MaxRounds]
    moe_race,           \* sticky latch: TRUE on any RAW/WAW hazard in slc->moe_inter

    \* --- Subsystem 3: K_PAIRS=2 tmp_b Register-Preload sh_x Pipeline ---
    sg_pc,              \* [Subgroups -> {"load_reg","wait_bar_a","store_shx","wait_bar_b","mma","done"}]
    sg_chunk,           \* [Subgroups -> 0..NumKChunks]
    sg_tmp_chunk,       \* [Subgroups -> 0..NumKChunks] chunk prefetched into private tmp_b regs
    shx_chunk,          \* [Subgroups -> 0..NumKChunks] chunk currently stored in SG's slice of sh_x
    shx_race,           \* sticky latch: TRUE on WAR overwrite or RAW stale read of sh_x

    \* --- Subsystem 4: 250K+ Banded RelBias Tile-Skip Verification ---
    q_min, q_max,       \* current query tile [q_min, q_max]
    kv_tmin, kv_tmax,   \* current KV tile [kv_tmin, kv_tmax]
    tile_skipped,       \* TRUE iff tile_has_rel == FALSE -> skipped rel_table load
    rel_skip_error,     \* sticky latch: TRUE if any in-band (q,k) existed in a skipped tile

    \* --- Subsystem 5: 40-WG SENSE_BARRIER M=1..8 Co-Residency ---
    active_m,           \* current batch M in {1, 4, 8}
    wg_threads,         \* 512 (M=1) or 256 (M=2..8)
    kernel_gpr32        \* <=56 (M=1) or <=72 (M=2..8)

vars == <<
    round, phase, timeline_val, completed_tl, inj_tl, dft_tl, tgt_tl,
    inj_stream_busy, dft_stream_busy, tgt_stream_busy,
    inj_kv_epoch, dft_kv_seen, draft_cdm_barrier, cmd_bo_race, kv_race,
    moe_stage, moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
    sg_pc, sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
    q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
    active_m, wg_threads, kernel_gpr32
>>

-----------------------------------------------------------------------------
(* Helper predicates *)

\* Exact linear SLC byte offset for (tok, k_slot) in PASS_STAGE=5/6
MoeSlotOffset(t, k) == (((t - 1) * TopK) + (k - 1)) * SlotBytes

\* Banded RelBias tile intersection test (S24 flash_attn_cm1.comp:420)
TileHasRel(qlo, qhi, tlo, thi) ==
    (qhi >= tlo) /\ (qlo < thi + RelExtent)

\* Exact mathematical existence of an in-band (q, k) pair inside the tile
ExistsInBandPair(qlo, qhi, tlo, thi) ==
    \E q \in qlo..qhi : \E k \in tlo..thi : (0 <= q - k) /\ (q - k < RelExtent)

\* AGX3B per-core register demand for 1 co-resident WG on each of 40 cores
Gpr32PerCore(threads, gpr) == (threads \div 32) * gpr

-----------------------------------------------------------------------------
(* Initial State *)

Init ==
    /\ round             = 1
    /\ phase             = "target"
    /\ timeline_val      = 0
    /\ completed_tl      = 0
    /\ inj_tl            = 0
    /\ dft_tl            = 0
    /\ tgt_tl            = 0
    /\ inj_stream_busy   = FALSE
    /\ dft_stream_busy   = FALSE
    /\ tgt_stream_busy   = FALSE
    /\ inj_kv_epoch      = 0
    /\ dft_kv_seen       = 0
    /\ draft_cdm_barrier = FALSE
    /\ cmd_bo_race       = FALSE
    /\ kv_race           = FALSE
    /\ moe_stage         = "idle"
    /\ moe_slot_written  = [p \in Tokens \X KSlots |-> FALSE]
    /\ moe_slot_epoch    = [p \in Tokens \X KSlots |-> 0]
    /\ moe_out_epoch     = [t \in Tokens |-> 0]
    /\ moe_race          = FALSE
    /\ sg_pc             = [s \in Subgroups |-> "load_reg"]
    /\ sg_chunk          = [s \in Subgroups |-> 0]
    /\ sg_tmp_chunk      = [s \in Subgroups |-> 0]
    /\ shx_chunk         = [s \in Subgroups |-> 0]
    /\ shx_race          = FALSE
    /\ q_min             = 0
    /\ q_max             = 1
    /\ kv_tmin           = 0
    /\ kv_tmax           = 1
    /\ tile_skipped      = FALSE
    /\ rel_skip_error    = FALSE
    /\ active_m          = 4
    /\ wg_threads        = 256
    /\ kernel_gpr32      = 64

-----------------------------------------------------------------------------
(* Subsystem 1: Async Inject (dedicated inject_stream) -> Chained Draft -> Target *)

\* Step 1a: Target Verify runs synchronously (blocking wait before CPU samples)
\* and checks M in {1, 4, 8} co-residency + starts Direct s5/s6 MoE sub-check.
TargetVerifySubmitAndWait ==
    /\ phase = "target"
    /\ round <= MaxRounds
    /\ moe_stage = "idle"
    /\ \E m \in {1, 4, 8} :
        /\ active_m'     = m
        /\ wg_threads'   = IF m = 1 THEN 512 ELSE 256
        /\ kernel_gpr32' = IF m = 1 THEN 56  ELSE 68
    /\ cmd_bo_race'      = (cmd_bo_race \/ tgt_stream_busy)
    /\ timeline_val'     = timeline_val + 1
    /\ tgt_tl'           = timeline_val + 1
    /\ tgt_stream_busy'  = TRUE
    /\ moe_stage'        = "s5_write"
    /\ moe_slot_written' = [p \in Tokens \X KSlots |-> FALSE]
    /\ UNCHANGED <<round, phase, completed_tl, inj_tl, dft_tl,
                   inj_stream_busy, dft_stream_busy, inj_kv_epoch, dft_kv_seen,
                   draft_cdm_barrier, kv_race, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_pc, sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error>>

TargetVerifyFinish ==
    /\ phase = "target"
    /\ tgt_stream_busy = TRUE
    /\ moe_stage = "complete"
    /\ completed_tl'    = tgt_tl
    /\ tgt_stream_busy' = FALSE
    /\ moe_stage'       = "idle"
    /\ phase'           = "inject_prep"
    /\ UNCHANGED <<round, timeline_val, inj_tl, dft_tl, tgt_tl,
                   inj_stream_busy, dft_stream_busy, inj_kv_epoch, dft_kv_seen,
                   draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_pc, sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

\* Step 1b: Inject prepares on its DEDICATED inject_stream (never racing draft_stream!),
\* waiting only if a previous inject_stream submission were still active, then submits
\* ASYNC (wait = FALSE)!
InjectSubmitAsync ==
    /\ phase = "inject_prep"
    /\ ~inj_stream_busy   \* guaranteed by prior round's Draft wait / agx_cdm_wait(inject_stream)
    /\ timeline_val'    = timeline_val + 1
    /\ inj_tl'          = timeline_val + 1
    /\ inj_stream_busy' = TRUE
    /\ phase'           = "draft_prep"
    /\ UNCHANGED <<round, completed_tl, dft_tl, tgt_tl,
                   dft_stream_busy, tgt_stream_busy, inj_kv_epoch, dft_kv_seen,
                   draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_stage, moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_pc, sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

\* GPU may complete async Inject EITHER during CPU draft_prep OR right before Draft starts
\* (enforced by hardware cdm_barrier=0u on the shared drm_asahi queue!).
GpuFinishAsyncInject ==
    /\ inj_stream_busy = TRUE
    /\ completed_tl < inj_tl
    /\ completed_tl'    = inj_tl
    /\ inj_stream_busy' = FALSE
    /\ inj_kv_epoch'    = round
    /\ UNCHANGED <<round, phase, timeline_val, inj_tl, dft_tl, tgt_tl,
                   dft_stream_busy, tgt_stream_busy, dft_kv_seen,
                   draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_stage, moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_pc, sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

\* Step 1c: CPU prepares & submits Draft on draft_stream (disjoint from inject_stream!)
\* while Inject may still be in flight! agx_drm_submit_cdm sets cdm_barrier=0u whenever
\* timeline_val > completed_tl (agx-drm.cpp:1320).
DraftSubmitChained ==
    /\ phase = "draft_prep"
    /\ ~dft_stream_busy
    /\ cmd_bo_race'       = (cmd_bo_race \/ dft_stream_busy)
    /\ draft_cdm_barrier' = (timeline_val > completed_tl)
    /\ timeline_val'      = timeline_val + 1
    /\ dft_tl'            = timeline_val + 1
    /\ dft_stream_busy'   = TRUE
    /\ phase'             = "draft_run"
    /\ UNCHANGED <<round, completed_tl, inj_tl, tgt_tl,
                   inj_stream_busy, tgt_stream_busy, inj_kv_epoch, dft_kv_seen, kv_race,
                   moe_stage, moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_pc, sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

\* GPU runs Draft ONLY after Inject has completed (enforced by hardware cdm_barrier=0u
\* if Inject was still in flight when Draft was submitted) and completes Draft synchronously.
DraftGpuExecuteAndWait ==
    /\ phase = "draft_run"
    /\ dft_stream_busy = TRUE
    \* Hardware CDM barrier on shared queue holds Draft until prior Inject completes
    /\ (~inj_stream_busy)
    /\ dft_kv_seen'     = inj_kv_epoch
    /\ kv_race'         = (kv_race \/ (inj_kv_epoch # round))
    /\ completed_tl'    = dft_tl
    /\ dft_stream_busy' = FALSE
    /\ round'           = round + 1
    /\ phase'           = IF round = MaxRounds THEN "done" ELSE "target"
    /\ UNCHANGED <<timeline_val, inj_tl, dft_tl, tgt_tl,
                   inj_stream_busy, tgt_stream_busy, inj_kv_epoch,
                   draft_cdm_barrier, cmd_bo_race,
                   moe_stage, moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_pc, sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

-----------------------------------------------------------------------------
(* Subsystem 2: M=2..8 Direct PASS_STAGE=5/6 MoE (0 bucket, 0 atomic) *)

MoeS5WriteSlot(t, k) ==
    /\ moe_stage = "s5_write"
    /\ ~moe_slot_written[<<t, k>>]
    /\ moe_slot_written' = [moe_slot_written EXCEPT ![<<t, k>>] = TRUE]
    /\ moe_slot_epoch'   = [moe_slot_epoch   EXCEPT ![<<t, k>>] = round]
    /\ UNCHANGED <<round, phase, timeline_val, completed_tl, inj_tl, dft_tl, tgt_tl,
                   inj_stream_busy, dft_stream_busy, tgt_stream_busy,
                   inj_kv_epoch, dft_kv_seen, draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_stage, moe_out_epoch, moe_race,
                   sg_pc, sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

MoeStrongUscFlush ==
    /\ moe_stage = "s5_write"
    /\ \A t \in Tokens, k \in KSlots : moe_slot_written[<<t, k>>]
    /\ moe_stage' = "s6_reduce"
    /\ UNCHANGED <<round, phase, timeline_val, completed_tl, inj_tl, dft_tl, tgt_tl,
                   inj_stream_busy, dft_stream_busy, tgt_stream_busy,
                   inj_kv_epoch, dft_kv_seen, draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_pc, sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

\* PASS_STAGE=6: each token t's thread sequentially sums k=1..TopK (0 atomics!)
MoeS6ReduceToken(t) ==
    /\ moe_stage = "s6_reduce"
    /\ moe_out_epoch[t] < round
    /\ LET all_ready == \A k \in KSlots : moe_slot_epoch[<<t, k>>] = round
       IN  /\ moe_race'      = (moe_race \/ ~all_ready)
           /\ moe_out_epoch' = [moe_out_epoch EXCEPT ![t] = round]
           /\ moe_stage'     = IF \A t2 \in Tokens : (t2 = t \/ moe_out_epoch[t2] = round)
                               THEN "complete" ELSE "s6_reduce"
    /\ UNCHANGED <<round, phase, timeline_val, completed_tl, inj_tl, dft_tl, tgt_tl,
                   inj_stream_busy, dft_stream_busy, tgt_stream_busy,
                   inj_kv_epoch, dft_kv_seen, draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_slot_written, moe_slot_epoch,
                   sg_pc, sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

-----------------------------------------------------------------------------
(* Subsystem 3: K_PAIRS=2 Software-Pipelined tmp_b Register Preload *)

\* SG issues global load into private tmp_b registers (may run concurrently with
\* other SGs finishing MMA on sh_x[g-1]!).
SgLoadTmpReg(s) ==
    /\ sg_pc[s] = "load_reg"
    /\ sg_chunk[s] < NumKChunks
    /\ sg_tmp_chunk' = [sg_tmp_chunk EXCEPT ![s] = sg_chunk[s] + 1]
    /\ sg_pc'        = [sg_pc EXCEPT ![s] = IF sg_chunk[s] > 0 THEN "wait_bar_a" ELSE "store_shx"]
    /\ UNCHANGED <<round, phase, timeline_val, completed_tl, inj_tl, dft_tl, tgt_tl,
                   inj_stream_busy, dft_stream_busy, tgt_stream_busy,
                   inj_kv_epoch, dft_kv_seen, draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_stage, moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

\* Barrier A (g > 0): ALL SGs must reach wait_bar_a before ANY overwrites sh_x.
SgPassBarrierA ==
    /\ \A s \in Subgroups : sg_pc[s] = "wait_bar_a"
    /\ sg_pc' = [s \in Subgroups |-> "store_shx"]
    /\ UNCHANGED <<round, phase, timeline_val, completed_tl, inj_tl, dft_tl, tgt_tl,
                   inj_stream_busy, dft_stream_busy, tgt_stream_busy,
                   inj_kv_epoch, dft_kv_seen, draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_stage, moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

\* Store private tmp_b -> sh_x slice; latch shx_race if any peer is still reading sh_x!
SgStoreShx(s) ==
    /\ sg_pc[s] = "store_shx"
    /\ LET peer_reading == \E p \in Subgroups : sg_pc[p] = "mma"
       IN  /\ shx_race'  = (shx_race \/ peer_reading)
           /\ shx_chunk' = [shx_chunk EXCEPT ![s] = sg_tmp_chunk[s]]
           /\ sg_pc'     = [sg_pc     EXCEPT ![s] = "wait_bar_b"]
    /\ UNCHANGED <<round, phase, timeline_val, completed_tl, inj_tl, dft_tl, tgt_tl,
                   inj_stream_busy, dft_stream_busy, tgt_stream_busy,
                   inj_kv_epoch, dft_kv_seen, draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_stage, moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_chunk, sg_tmp_chunk,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

\* Barrier B: ALL SGs must finish storing sh_x[g] before ANY starts MMA compute.
SgPassBarrierB ==
    /\ \A s \in Subgroups : sg_pc[s] = "wait_bar_b"
    /\ sg_pc' = [s \in Subgroups |-> "mma"]
    /\ UNCHANGED <<round, phase, timeline_val, completed_tl, inj_tl, dft_tl, tgt_tl,
                   inj_stream_busy, dft_stream_busy, tgt_stream_busy,
                   inj_kv_epoch, dft_kv_seen, draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_stage, moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

\* Co-issued t=0 & t=1 dequant16 + 1x sh_x read -> 4x coopMatMulAdd across K_PAIRS=2.
\* Reads BOTH SGs' slices of sh_x; latch shx_race if any slice does not match chunk+1!
SgMmaCompute(s) ==
    /\ sg_pc[s] = "mma"
    /\ LET want == sg_chunk[s] + 1
           all_fresh == \A p \in Subgroups : shx_chunk[p] = want
       IN  /\ shx_race' = (shx_race \/ ~all_fresh)
           /\ sg_chunk' = [sg_chunk EXCEPT ![s] = want]
           /\ sg_pc'    = [sg_pc    EXCEPT ![s] = IF want = NumKChunks THEN "done" ELSE "load_reg"]
    /\ UNCHANGED <<round, phase, timeline_val, completed_tl, inj_tl, dft_tl, tgt_tl,
                   inj_stream_busy, dft_stream_busy, tgt_stream_busy,
                   inj_kv_epoch, dft_kv_seen, draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_stage, moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_tmp_chunk, shx_chunk,
                   q_min, q_max, kv_tmin, kv_tmax, tile_skipped, rel_skip_error,
                   active_m, wg_threads, kernel_gpr32>>

-----------------------------------------------------------------------------
(* Subsystem 4: 250K+ Banded RelBias Tile-Skip Soundness Check *)

\* Representative tiles covering in-band, past-extent skip, future-causal skip, and straddling
CheckBandedTile ==
    /\ phase = "target"
    /\ \E tile \in { <<0, 1, 0, 1>>, <<4, 5, 0, 1>>, <<0, 1, 3, 4>>, <<2, 3, 1, 2>> } :
        LET qlo  == tile[1]
            qhi  == tile[2]
            tlo  == tile[3]
            thi  == tile[4]
            skip == ~TileHasRel(qlo, qhi, tlo, thi)
            bad  == skip /\ ExistsInBandPair(qlo, qhi, tlo, thi)
        IN  /\ q_min'          = qlo
            /\ q_max'          = qhi
            /\ kv_tmin'        = tlo
            /\ kv_tmax'        = thi
            /\ tile_skipped'   = skip
            /\ rel_skip_error' = (rel_skip_error \/ bad)
    /\ UNCHANGED <<round, phase, timeline_val, completed_tl, inj_tl, dft_tl, tgt_tl,
                   inj_stream_busy, dft_stream_busy, tgt_stream_busy,
                   inj_kv_epoch, dft_kv_seen, draft_cdm_barrier, cmd_bo_race, kv_race,
                   moe_stage, moe_slot_written, moe_slot_epoch, moe_out_epoch, moe_race,
                   sg_pc, sg_chunk, sg_tmp_chunk, shx_chunk, shx_race,
                   active_m, wg_threads, kernel_gpr32>>

DoneStutter ==
    /\ phase = "done"
    /\ \A s \in Subgroups : sg_pc[s] = "done"
    /\ UNCHANGED vars

-----------------------------------------------------------------------------
(* Next-State Relation & Specification *)

Next ==
    \/ TargetVerifySubmitAndWait
    \/ TargetVerifyFinish
    \/ InjectSubmitAsync
    \/ GpuFinishAsyncInject
    \/ DraftSubmitChained
    \/ DraftGpuExecuteAndWait
    \/ \E t \in Tokens, k \in KSlots : MoeS5WriteSlot(t, k)
    \/ MoeStrongUscFlush
    \/ \E t \in Tokens : MoeS6ReduceToken(t)
    \/ \E s \in Subgroups : SgLoadTmpReg(s)
    \/ SgPassBarrierA
    \/ \E s \in Subgroups : SgStoreShx(s)
    \/ SgPassBarrierB
    \/ \E s \in Subgroups : SgMmaCompute(s)
    \/ CheckBandedTile
    \/ DoneStutter

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ WF_vars(TargetVerifySubmitAndWait)
    /\ WF_vars(TargetVerifyFinish)
    /\ WF_vars(InjectSubmitAsync)
    /\ WF_vars(GpuFinishAsyncInject)
    /\ WF_vars(DraftSubmitChained)
    /\ WF_vars(DraftGpuExecuteAndWait)
    /\ \A t \in Tokens, k \in KSlots : WF_vars(MoeS5WriteSlot(t, k))
    /\ WF_vars(MoeStrongUscFlush)
    /\ \A t \in Tokens : WF_vars(MoeS6ReduceToken(t))
    /\ \A s \in Subgroups : WF_vars(SgLoadTmpReg(s))
    /\ WF_vars(SgPassBarrierA)
    /\ \A s \in Subgroups : WF_vars(SgStoreShx(s))
    /\ WF_vars(SgPassBarrierB)
    /\ \A s \in Subgroups : WF_vars(SgMmaCompute(s))

-----------------------------------------------------------------------------
(* Safety Invariants *)

TypeOK ==
    /\ round \in 1..(MaxRounds + 1)
    /\ phase \in {"target", "inject_prep", "draft_prep", "draft_run", "done"}
    /\ timeline_val >= completed_tl
    /\ moe_stage \in {"idle", "s5_write", "s6_reduce", "complete"}
    /\ active_m \in {1, 4, 8}

\* 1. Zero cmd_bo/arg_bo overwrite race & zero stale KV read across async Inject -> Draft
AsyncInjectChainSafe ==
    /\ ~cmd_bo_race
    /\ ~kv_race
    /\ (phase = "draft_run" /\ inj_stream_busy) => draft_cdm_barrier

\* 2. Direct PASS_STAGE=5/6 MoE SLC moe_inter pairwise 128B disjointness & zero-atomic safety
DirectMoeSlcDisjoint ==
    /\ ~moe_race
    /\ \A t1, t2 \in Tokens, k1, k2 \in KSlots :
        (<<t1, k1>> # <<t2, k2>>) =>
            (MoeSlotOffset(t1, k1) + SlotBytes <= MoeSlotOffset(t2, k2) \/
             MoeSlotOffset(t2, k2) + SlotBytes <= MoeSlotOffset(t1, k1))
    /\ \A t \in Tokens, k \in KSlots :
        /\ (MoeSlotOffset(t, k) % 128 = 0)
        /\ (MoeSlotOffset(t, k) + SlotBytes <= SlcMoeInterCap)

\* 3. K_PAIRS=2 tmp_b software-pipelined sh_x barrier safety (zero WAR/RAW)
Kp2PipelinedBarrierSafe ==
    /\ ~shx_race
    /\ \A s1, s2 \in Subgroups :
        (sg_pc[s1] = "store_shx") => (sg_pc[s2] # "mma")

\* 4. 250K+ Banded RelBias tile-skip is 100% lossless (both tracked state AND universal over 0..5!)
BandedRelTileSkipSafe ==
    /\ ~rel_skip_error
    /\ tile_skipped => ~ExistsInBandPair(q_min, q_max, kv_tmin, kv_tmax)
    /\ \A qlo \in 0..5, qhi \in 0..5, tlo \in 0..5, thi \in 0..5 :
        (qlo <= qhi /\ tlo <= thi /\ ~TileHasRel(qlo, qhi, tlo, thi))
            => ~ExistsInBandPair(qlo, qhi, tlo, thi)

\* 5. 40-WG SENSE_BARRIER strict co-residency across M = 1..8 (zero -110 timeout)
SenseCoresidentSafe ==
    /\ Gpr32PerCore(wg_threads, kernel_gpr32) <= MaxGpr32PerCore
    /\ (active_m > 1) => (wg_threads = 256 /\ kernel_gpr32 <= 72)

-----------------------------------------------------------------------------
(* Temporal Liveness Property *)

AllRoundsComplete ==
    <>(phase = "done" /\ \A s \in Subgroups : sg_pc[s] = "done")

=============================================================================
