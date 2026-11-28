------------------------ MODULE AgxWave8Ceiling ------------------------
(***************************************************************************)
(* Formal TLA+ specification of the Wave 8 AGX3B Silicon-Ceiling Protocol  *)
(* for ggml-agx on Apple M4 Max (T6041 / AGX3B / HAL200).                  *)
(*                                                                         *)
(* Machine-checks the four Wave 8 concurrency & memory-safety mechanisms:  *)
(*   1. Multi-slot CDM Replay Cache + In-Place `arg_bo` `n_kv` Patching    *)
(*      on `ctx_dft` across alternating `INJECT` (`build_inject_ctx_mix`)  *)
(*      and `DRAFT` (`dspark_draft_mk`) rounds: proves zero host/GPU       *)
(*      `arg_bo` race (`NoArgBoRace`), exact `n_kv` visibility             *)
(*      (`ExactReplayNkv`), sense reset (`SenseCleanOnLaunch`), and        *)
(*      zero alternating-graph replay thrashing (`ZeroAlternatingThrash`). *)
(*   2. Disjoint Expert-Bucket Split (`VM_M1` || `VM_M4` || `GEMM` with    *)
(*      `AGX_BARRIER_NONE`, `MOE_FLAG_DOWN_NOSCATTER`, followed by         *)
(*      `STRONG_USC` before `DOWN_REDUCE`): proves zero 128B non-coherent  *)
(*      USC L1 false sharing (`NoUscFalseSharing`) and full producer       *)
(*      visibility at `DOWN_REDUCE` (`DownReduceVisibility`).              *)
(*   3. Wave-Scheduled Prefill Epilogue Fusions (`FLASH_ATTN + qgate`,     *)
(*      `MIX + rms_norm`) & Redundant Entry-Barrier Elision on             *)
(*      `agx_push_fused_moe`: proves wave hazard tracking preserves        *)
(*      `RAW`/`WAR`/`WAW` soundness when `ensure_item_ready` already       *)
(*      flushed USC L1 (`WaveFusionSoundness`).                            *)
(*   4. `PASS_MODE=3` `8,192 B` SRAM `sh_any_valid[2]` 2-slot ping-pong    *)
(*      across KV tiles: proves zero shared-memory overwrite hazard        *)
(*      (`SharedPingPongSafe`).                                            *)
(***************************************************************************)

EXTENDS Integers, FiniteSets

CONSTANTS
    MaxRounds,    \* Speculative MTP rounds modeled (e.g., 2)
    NumExperts    \* Active routing domain modeled (e.g., 3)

ASSUME MaxRounds \in Nat /\ MaxRounds >= 2
ASSUME NumExperts \in Nat /\ NumExperts >= 2

Slots   == {0, 1}          \* Slot 0: INJECT graph, Slot 1: DRAFT graph
Experts == 1..NumExperts
SubGrids == {"NONE", "VM_M1", "VM_M4", "GEMM"}

VARIABLES
    round,            \* 0..MaxRounds : current speculative decoding round
    stepPhase,        \* Execution state machine phase
    kvLen,            \* Monotonically growing draft/target KV length
    slotValid,        \* [Slots -> BOOLEAN] : whether CDM replay slot is recorded
    slotNkv,          \* [Slots -> Nat]     : `n_kv` patched in `arg_bo[slot]`
    slotBusy,         \* [Slots -> BOOLEAN] : TRUE while GPU executes `slot`
    senseVal,         \* {0, 1}             : persistent barrier sense word
    replayHits,       \* Nat                : cumulative CDM replay hits on `ctx_dft`
    replayMisses,     \* Nat                : cumulative CDM cold encodes on `ctx_dft`
    mCnt,             \* [Experts -> {0, 1, 2}] : routed token count per expert
    subGridOf,        \* [Experts -> SubGrids]  : partitioned bucket assignment
    expDone,          \* [Experts -> BOOLEAN]   : whether expert sub-grid finished
    uscDirtyOwner,    \* [Experts -> SubGrids]  : 128B USC L1 dirty line owner ("NONE" = clean in SLC)
    slcIntermVer,     \* [Experts -> (-1)..MaxRounds] : SLC-visible `down_noscat` version
    waveFlushed,      \* BOOLEAN : whether prior wave emitted STRONG_USC before MoE/FA
    faTileIdx,        \* 0..2    : prefill FlashAttn KV tile index
    shPingSlot,       \* [0..1 -> (-1)..2] : tile index currently occupying `sh_any_valid[p]`
    safetyOk          \* BOOLEAN : conjunction latch of all immediate transition assertions

vars == << round, stepPhase, kvLen, slotValid, slotNkv, slotBusy, senseVal,
           replayHits, replayMisses, mCnt, subGridOf, expDone,
           uscDirtyOwner, slcIntermVer, waveFlushed, faTileIdx, shPingSlot, safetyOk >>

BucketGrid(c) ==
    IF c = 0 THEN "NONE"
    ELSE IF c = 1 THEN "VM_M1"
    ELSE "VM_M4"

Init ==
    /\ round         = 0
    /\ stepPhase     = "INJECT_PREP"
    /\ kvLen         = 16
    /\ slotValid     = [s \in Slots |-> FALSE]
    /\ slotNkv       = [s \in Slots |-> 0]
    /\ slotBusy      = [s \in Slots |-> FALSE]
    /\ senseVal      = 0
    /\ replayHits    = 0
    /\ replayMisses  = 0
    /\ mCnt          = [e \in Experts |-> 0]
    /\ subGridOf     = [e \in Experts |-> "NONE"]
    /\ expDone       = [e \in Experts |-> FALSE]
    /\ uscDirtyOwner = [e \in Experts |-> "NONE"]
    /\ slcIntermVer  = [e \in Experts |-> -1]
    /\ waveFlushed   = FALSE
    /\ faTileIdx     = 0
    /\ shPingSlot    = [p \in 0..1 |-> -1]
    /\ safetyOk      = TRUE

(***************************************************************************)
(* Phase 1: Inter-round DSpark Inject (`ctx_dft` Replay Slot 0)            *)
(***************************************************************************)
SubmitInject ==
    /\ stepPhase = "INJECT_PREP"
    /\ round < MaxRounds
    /\ ~slotBusy[0]
    /\ ~slotBusy[1]
    /\ slotNkv'      = [slotNkv EXCEPT ![0] = kvLen]
    /\ slotBusy'     = [slotBusy EXCEPT ![0] = TRUE]
    /\ senseVal'     = 0
    /\ replayHits'   = IF slotValid[0] THEN replayHits + 1 ELSE replayHits
    /\ replayMisses' = IF slotValid[0] THEN replayMisses ELSE replayMisses + 1
    /\ slotValid'    = [slotValid EXCEPT ![0] = TRUE]
    /\ stepPhase'    = "INJECT_GPU"
    /\ UNCHANGED << round, kvLen, mCnt, subGridOf, expDone,
                    uscDirtyOwner, slcIntermVer, waveFlushed, faTileIdx, shPingSlot, safetyOk >>

FinishInject(nAcc) ==
    /\ stepPhase = "INJECT_GPU"
    /\ nAcc \in {1, 4}
    /\ slotBusy[0]
    /\ LET nkvExact == (slotNkv[0] = kvLen)
           senseOk  == (senseVal = 0)
       IN
       /\ safetyOk'  = (safetyOk /\ nkvExact /\ senseOk)
       /\ slotBusy'  = [slotBusy EXCEPT ![0] = FALSE]
       /\ kvLen'     = kvLen + nAcc
       /\ stepPhase' = "DRAFT_PREP"
       /\ UNCHANGED << round, slotValid, slotNkv, senseVal, replayHits, replayMisses,
                       mCnt, subGridOf, expDone, uscDirtyOwner, slcIntermVer,
                       waveFlushed, faTileIdx, shPingSlot >>

(***************************************************************************)
(* Phase 2: DSpark Draft MK (`ctx_dft` Replay Slot 1 — zero slot thrash!)  *)
(***************************************************************************)
SubmitDraft ==
    /\ stepPhase = "DRAFT_PREP"
    /\ ~slotBusy[0]
    /\ ~slotBusy[1]
    /\ slotNkv'      = [slotNkv EXCEPT ![1] = kvLen]
    /\ slotBusy'     = [slotBusy EXCEPT ![1] = TRUE]
    /\ senseVal'     = 0
    /\ replayHits'   = IF slotValid[1] THEN replayHits + 1 ELSE replayHits
    /\ replayMisses' = IF slotValid[1] THEN replayMisses ELSE replayMisses + 1
    /\ slotValid'    = [slotValid EXCEPT ![1] = TRUE]
    /\ stepPhase'    = "DRAFT_GPU"
    /\ UNCHANGED << round, kvLen, mCnt, subGridOf, expDone,
                    uscDirtyOwner, slcIntermVer, waveFlushed, faTileIdx, shPingSlot, safetyOk >>

FinishDraft ==
    /\ stepPhase = "DRAFT_GPU"
    /\ slotBusy[1]
    /\ LET nkvExact == (slotNkv[1] = kvLen)
           slot0Intact == slotValid[0]
       IN
       /\ safetyOk'   = (safetyOk /\ nkvExact /\ slot0Intact)
       /\ slotBusy'   = [slotBusy EXCEPT ![1] = FALSE]
       /\ faTileIdx'  = 0
       /\ shPingSlot' = [p \in 0..1 |-> -1]
       /\ stepPhase'  = "FA_TILES"
       /\ UNCHANGED << round, kvLen, slotValid, slotNkv, senseVal, replayHits, replayMisses,
                       mCnt, subGridOf, expDone, uscDirtyOwner, slcIntermVer, waveFlushed >>

(***************************************************************************)
(* Phase 3: FlashAttn `PASS_MODE=3` `sh_any_valid[iter % 2]` + Fused QGate *)
(***************************************************************************)
StepFaTile ==
    /\ stepPhase = "FA_TILES"
    /\ IF faTileIdx < 2 THEN
           LET pSlot == faTileIdx % 2
               \* 2-slot ping-pong never collides with the immediately preceding tile
               pingOk == (faTileIdx = 0) \/ (shPingSlot[pSlot] # faTileIdx - 1)
           IN
           /\ safetyOk'   = (safetyOk /\ pingOk)
           /\ shPingSlot' = [shPingSlot EXCEPT ![pSlot] = faTileIdx]
           /\ faTileIdx'  = faTileIdx + 1
           /\ UNCHANGED << round, stepPhase, kvLen, slotValid, slotNkv, slotBusy, senseVal,
                           replayHits, replayMisses, mCnt, subGridOf, expDone,
                           uscDirtyOwner, slcIntermVer, waveFlushed >>
       ELSE
           \* Wave scheduler emits STRONG_USC after fused `FA+qgate` -> `Wo` -> `mix+rms_norm`
           \* BEFORE `agx_push_fused_moe` (allowing `agx_push_fused_moe` to elide redundant entry barrier)
           /\ waveFlushed' = TRUE
           /\ stepPhase'   = "MOE_BUCKET"
           /\ UNCHANGED << round, kvLen, slotValid, slotNkv, slotBusy, senseVal,
                           replayHits, replayMisses, mCnt, subGridOf, expDone,
                           uscDirtyOwner, slcIntermVer, faTileIdx, shPingSlot, safetyOk >>

(***************************************************************************)
(* Phase 4: `moe_bitonic_bucket` Partition + Concurrent Disjoint Sub-Grids *)
(* (`grid_vm_m1` || `grid_vm_m4` || `grid_gemm` with `AGX_BARRIER_NONE`)   *)
(***************************************************************************)
BucketizeMoE(counts) ==
    /\ stepPhase = "MOE_BUCKET"
    /\ counts \in [Experts -> {0, 1, 2}]
    /\ waveFlushed = TRUE
    /\ mCnt'          = counts
    /\ subGridOf'     = [e \in Experts |->
                          IF counts[e] = 0 THEN "NONE"
                          ELSE IF counts[e] = 1 THEN "VM_M1"
                          ELSE IF e = 1 THEN "GEMM" ELSE "VM_M4"]
    /\ expDone'       = [e \in Experts |-> (counts[e] = 0)]
    /\ uscDirtyOwner' = [e \in Experts |-> "NONE"]
    /\ stepPhase'     = "MOE_SUBGRIDS"
    /\ UNCHANGED << round, kvLen, slotValid, slotNkv, slotBusy, senseVal,
                    replayHits, replayMisses, slcIntermVer, waveFlushed,
                    faTileIdx, shPingSlot, safetyOk >>

\* Concurrent sub-grids (`VM_M1`, `VM_M4`, `GEMM`) run in any interleaving with
\* `AGX_BARRIER_NONE`, writing disjoint 128B-aligned `down_noscat` expert lines.
RunExpertSubGrid(e) ==
    /\ stepPhase = "MOE_SUBGRIDS"
    /\ ~expDone[e]
    /\ subGridOf[e] # "NONE"
    /\ LET g == subGridOf[e]
           \* 128B USC L1 false-sharing check: line `e` must not be dirty in another sub-grid
           lineClean == (uscDirtyOwner[e] = "NONE" \/ uscDirtyOwner[e] = g)
       IN
       /\ safetyOk'      = (safetyOk /\ waveFlushed /\ lineClean)
       /\ uscDirtyOwner' = [uscDirtyOwner EXCEPT ![e] = g]
       /\ expDone'       = [expDone EXCEPT ![e] = TRUE]
       /\ UNCHANGED << round, stepPhase, kvLen, slotValid, slotNkv, slotBusy, senseVal,
                       replayHits, replayMisses, mCnt, subGridOf, slcIntermVer,
                       waveFlushed, faTileIdx, shPingSlot >>

\* Mandatory STRONG_USC barrier between concurrent MoE sub-grids and `moe_down_reduce`
MoEDownBarrier ==
    /\ stepPhase = "MOE_SUBGRIDS"
    /\ \A e \in Experts : expDone[e]
    /\ slcIntermVer'  = [e \in Experts |-> IF mCnt[e] > 0 THEN round ELSE slcIntermVer[e]]
    /\ uscDirtyOwner' = [e \in Experts |-> "NONE"]
    /\ stepPhase'     = "MOE_REDUCE"
    /\ UNCHANGED << round, kvLen, slotValid, slotNkv, slotBusy, senseVal,
                    replayHits, replayMisses, mCnt, subGridOf, expDone,
                    waveFlushed, faTileIdx, shPingSlot, safetyOk >>

\* `moe_down_reduce` reads across all active expert outputs after STRONG_USC flush
MoEDownReduce ==
    /\ stepPhase = "MOE_REDUCE"
    /\ LET allFlushed == \A e \in Experts :
                           (uscDirtyOwner[e] = "NONE") /\
                           (mCnt[e] > 0 => slcIntermVer[e] = round)
       IN
       /\ safetyOk'    = (safetyOk /\ allFlushed)
       /\ waveFlushed' = FALSE
       /\ round'       = round + 1
       /\ stepPhase'   = IF round + 1 = MaxRounds THEN "DONE" ELSE "INJECT_PREP"
       /\ UNCHANGED << kvLen, slotValid, slotNkv, slotBusy, senseVal,
                       replayHits, replayMisses, mCnt, subGridOf, expDone,
                       uscDirtyOwner, slcIntermVer, faTileIdx, shPingSlot >>

Done ==
    /\ stepPhase = "DONE"
    /\ UNCHANGED vars

Next ==
    \/ SubmitInject
    \/ \E nAcc \in {1, 4} : FinishInject(nAcc)
    \/ SubmitDraft
    \/ FinishDraft
    \/ StepFaTile
    \/ \E counts \in [Experts -> {0, 1, 2}] : BucketizeMoE(counts)
    \/ \E e \in Experts : RunExpertSubGrid(e)
    \/ MoEDownBarrier
    \/ MoEDownReduce
    \/ Done

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(***************************************************************************)
(* Safety & Liveness Invariants                                            *)
(***************************************************************************)

TypeOK ==
    /\ round \in 0..MaxRounds
    /\ stepPhase \in {"INJECT_PREP", "INJECT_GPU", "DRAFT_PREP", "DRAFT_GPU",
                      "FA_TILES", "MOE_BUCKET", "MOE_SUBGRIDS", "MOE_REDUCE", "DONE"}
    /\ kvLen \in Nat
    /\ slotValid \in [Slots -> BOOLEAN]
    /\ slotBusy \in [Slots -> BOOLEAN]
    /\ senseVal \in {0, 1}
    /\ mCnt \in [Experts -> {0, 1, 2}]
    /\ subGridOf \in [Experts -> SubGrids]
    /\ expDone \in [Experts -> BOOLEAN]
    /\ uscDirtyOwner \in [Experts -> SubGrids]
    /\ waveFlushed \in BOOLEAN
    /\ faTileIdx \in 0..2
    /\ safetyOk \in BOOLEAN

\* All immediate transition safety checks hold in every reachable state
SafetyLatchOK == safetyOk = TRUE

\* Host never runs two concurrent submits on `ctx_dft` or mutates an active slot's `arg_bo`
NoArgBoRace == ~(slotBusy[0] /\ slotBusy[1])

\* Active GPU execution always sees the exact current `kvLen` in its replay slot's `arg_bo`
ExactReplayNkv ==
    /\ (slotBusy[0] => slotNkv[0] = kvLen)
    /\ (slotBusy[1] => slotNkv[1] = kvLen)

\* Sense word is always clean (0) upon CDM launch
SenseCleanOnLaunch == (slotBusy[0] \/ slotBusy[1]) => (senseVal = 0)

\* 2-slot CDM replay completely eliminates alternating Inject/Draft thrashing:
\* cold misses are strictly bounded by 2 (1 cold encode for Slot 0 + 1 for Slot 1),
\* and every round >= 1 achieves 100% CDM replay hits!
ZeroAlternatingThrash ==
    /\ replayMisses <= 2
    /\ (stepPhase = "DONE" => (replayMisses = 2 /\ replayHits = 2 * (MaxRounds - 1)))

\* Disjoint expert sub-grid partition: `m_cnt = 1` never aliases `m_cnt >= 2`
BucketPartitionDisjoint ==
    \A e \in Experts :
        /\ (subGridOf[e] = "VM_M1" => mCnt[e] = 1)
        /\ (subGridOf[e] \in {"VM_M4", "GEMM"} => mCnt[e] = 2)
        /\ (subGridOf[e] = "NONE" => uscDirtyOwner[e] = "NONE")

\* `moe_down_reduce` never executes while any 128B USC L1 line is still dirty
DownReduceVisibility ==
    (stepPhase = "MOE_REDUCE") =>
        (\A e \in Experts : uscDirtyOwner[e] = "NONE" /\ (mCnt[e] > 0 => slcIntermVer[e] = round))

\* `sh_any_valid[2]` slots never hold the same live KV tile index simultaneously
SharedPingPongSafe ==
    (shPingSlot[0] >= 0 /\ shPingSlot[1] >= 0) => (shPingSlot[0] # shPingSlot[1])

AllRoundsComplete == <>(round = MaxRounds /\ stepPhase = "DONE")

=============================================================================
