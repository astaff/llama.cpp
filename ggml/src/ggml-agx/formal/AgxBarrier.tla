---------------------------- MODULE AgxBarrier ----------------------------
(***************************************************************************)
(* Formal TLA+ specification of Pillar 6 for the bare-metal ggml-agx       *)
(* backend on Apple Silicon AGX3 (M4 Max / HAL200 / G17P).                 *)
(*                                                                         *)
(* Models and verifies two interacting synchronization layers:             *)
(*   1. Device-scope sense-reversing atomic barrier across N persistent    *)
(*      workgroups executing the 3-stage Inkling 2 decode ring             *)
(*      (Stage 0 = prep_qkv, Stage 1 = attn_out, Stage 2 = moe_down)       *)
(*      under a weak USC L1 store-buffer -> 48 MB SLC memory model         *)
(*      (struct agx_sense_barrier_state in agx-common.h).                  *)
(*   2. HAL200/G17P CDM command-stream barrier elision & in-place          *)
(*      strengthening (agx-cdm.cpp):                                       *)
(*        - AGX_BARRIER_NONE        (elided between independent dispatches)*)
(*        - AGX_BARRIER_WEAK_WFI    (0x60000160)                           *)
(*        - AGX_BARRIER_STRONG_USC  (0x60001560, intra-VM SLC coherent)    *)
(*        - AGX_BARRIER_MEMORY_HOST (0x6000157c, host CPU readback flush)  *)
(*                                                                         *)
(* Verified invariants & liveness:                                         *)
(*   - TypeOK                 : all state variables remain well-typed      *)
(*   - StageVisibility        : no consumer reads stale/unflushed data     *)
(*   - NoBarrierOverwrite     : sense bit prevents straggler collision     *)
(*   - BoundedPhaseSkew       : workgroups are at most 1 phase apart       *)
(*   - NoAdjacentCdmBarriers  : HAL200 rule (never two barrier packets     *)
(*                              back-to-back in a CDM stream)              *)
(*   - HostReadbackCoherent   : host observes complete SLC-flushed output  *)
(*   - AllComplete (liveness) : deadlock-free & livelock-free completion   *)
(***************************************************************************)

EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    NumWGs,         \* Number of persistent workgroups (e.g., 3 in TLC; 40 on M4 Max)
    NumStages,      \* Stages per Inkling 2 pass (3: prep_qkv, attn_out, moe_down)
    NumPasses       \* Number of passes modeled (e.g., 2 in TLC; 48 in Inkling 2)

ASSUME NumWGs \in Nat /\ NumWGs >= 2
ASSUME NumStages \in Nat /\ NumStages >= 2
ASSUME NumPasses \in Nat /\ NumPasses >= 1

WGs       == 1..NumWGs
TotalPhases == NumPasses * NumStages   \* phases 0 .. TotalPhases-1
Slots     == 0..(NumStages - 1)        \* ring slots indexed by stage
NO_VER    == -1

\* CDM barrier encoding levels (matching enum agx_barrier_mode)
B_NONE        == 0
B_WEAK_WFI    == 1
B_STRONG_USC  == 2
B_MEMORY_HOST == 3

VARIABLES
    \* --- Persistent-megakernel sense-reversing barrier state (128B SLC line) ---
    arrived,        \* 0..NumWGs : atomic counter
    sense,          \* {0, 1}    : toggling barrier sense bit
    epoch,          \* 0..TotalPhases : monotonic barrier completions

    \* --- Per-workgroup execution state ---
    pc,             \* [WGs -> {"RUN", "FLUSHED", "WAIT_SENSE", "DONE"}]
    phase,          \* [WGs -> 0..TotalPhases] (pass = phase \div NumStages, stage = phase % NumStages)
    localSense,     \* [WGs -> {0, 1}]

    \* --- Weak USC store-buffer -> 48 MB SLC -> Host-coherent memory hierarchy ---
    uscBuf,         \* [WGs -> [Slots -> (-1)..(TotalPhases - 1)]] : private L1/USC write buffer
    slcMem,         \* [Slots -> [WGs -> (-1)..(TotalPhases - 1)]] : shared 48 MB SLC
    hostMem,        \* [Slots -> [WGs -> (-1)..(TotalPhases - 1)]] : host-coherent DRAM/SLC view
    readOk,         \* BOOLEAN : latched safety check for stage input visibility

    \* --- HAL200 CDM command stream state (Pillar 6 barrier elision) ---
    \* Models: 2 independent pre-dispatches (elided barrier) -> megakernel dispatch
    \*         -> in-place upgrade to MEMORY_HOST -> TERMINATE -> host readback.
    cdmStage,       \* {"INIT", "PRE1", "PRE2", "MEGA", "FLUSH_HOST", "TERMINATED", "HOST_READ"}
    cdmPackets,     \* Seq of {"DISPATCH", "B_WEAK", "B_USC", "B_HOST", "TERM"}
    trailingBarrier \* {B_NONE, B_WEAK_WFI, B_STRONG_USC, B_MEMORY_HOST}

vars == << arrived, sense, epoch, pc, phase, localSense,
           uscBuf, slcMem, hostMem, readOk,
           cdmStage, cdmPackets, trailingBarrier >>

(***************************************************************************)
(* Helper predicates                                                       *)
(***************************************************************************)

StageOf(p) == p % NumStages

IsBarrierPkt(pkt) == pkt \in {"B_WEAK", "B_USC", "B_HOST"}

BarrierPktOf(lvl) ==
    CASE lvl = B_WEAK_WFI    -> "B_WEAK"
      [] lvl = B_STRONG_USC  -> "B_USC"
      [] lvl = B_MEMORY_HOST -> "B_HOST"
      [] OTHER               -> "B_WEAK"

\* Upgrade trailing barrier in place (mirrors agx_cdm_barrier in agx-cdm.cpp)
UpgradeTrailing(pkts, newLvl) ==
    IF Len(pkts) > 0 /\ IsBarrierPkt(pkts[Len(pkts)])
    THEN [pkts EXCEPT ![Len(pkts)] = BarrierPktOf(newLvl)]
    ELSE Append(pkts, BarrierPktOf(newLvl))

\* Elide weak trailing barrier when next dispatch is independent (mirrors AGX_BARRIER_NONE)
ElideTrailingWeak(pkts) ==
    IF Len(pkts) > 0 /\ pkts[Len(pkts)] = "B_WEAK"
    THEN SubSeq(pkts, 1, Len(pkts) - 1)
    ELSE pkts

(***************************************************************************)
(* Initial state                                                           *)
(***************************************************************************)

Init ==
    /\ arrived         = 0
    /\ sense           = 0
    /\ epoch           = 0
    /\ pc              = [w \in WGs |-> "RUN"]
    /\ phase           = [w \in WGs |-> 0]
    /\ localSense      = [w \in WGs |-> 0]
    /\ uscBuf          = [w \in WGs |-> [s \in Slots |-> NO_VER]]
    /\ slcMem          = [s \in Slots |-> [w \in WGs |-> NO_VER]]
    /\ hostMem         = [s \in Slots |-> [w \in WGs |-> NO_VER]]
    /\ readOk          = TRUE
    /\ cdmStage        = "INIT"
    /\ cdmPackets      = << >>
    /\ trailingBarrier = B_NONE

(***************************************************************************)
(* CDM Stream Recording & Barrier Elision Actions                          *)
(***************************************************************************)

\* Emit first independent dispatch with trailing weak tail (may be elided).
CdmEmitPre1 ==
    /\ cdmStage = "INIT"
    /\ cdmStage' = "PRE1"
    /\ cdmPackets' = Append(Append(cdmPackets, "DISPATCH"), "B_WEAK")
    /\ trailingBarrier' = B_WEAK_WFI
    /\ UNCHANGED << arrived, sense, epoch, pc, phase, localSense,
                    uscBuf, slcMem, hostMem, readOk >>

\* Emit second independent dispatch: ELIDES the preceding weak WFI barrier so
\* both independent dispatches overlap across the 40 M4 Max GPU cores, then
\* strengthens its own tail to STRONG_USC before the dependent megakernel.
CdmEmitPre2ElideAndUpgrade ==
    /\ cdmStage = "PRE1"
    /\ cdmStage' = "PRE2"
    /\ LET trimmed == ElideTrailingWeak(cdmPackets)
           withDisp == Append(trimmed, "DISPATCH")
       IN  cdmPackets' = Append(withDisp, "B_USC")
    /\ trailingBarrier' = B_STRONG_USC
    /\ UNCHANGED << arrived, sense, epoch, pc, phase, localSense,
                    uscBuf, slcMem, hostMem, readOk >>

\* Launch the persistent Inkling 2 megakernel dispatch after the STRONG_USC flush.
CdmLaunchMegakernel ==
    /\ cdmStage = "PRE2"
    /\ cdmStage' = "MEGA"
    /\ cdmPackets' = Append(Append(cdmPackets, "DISPATCH"), "B_WEAK")
    /\ trailingBarrier' = B_WEAK_WFI
    /\ UNCHANGED << arrived, sense, epoch, pc, phase, localSense,
                    uscBuf, slcMem, hostMem, readOk >>

(***************************************************************************)
(* Persistent-Megakernel Weak-Memory + Sense-Reversing Barrier Actions     *)
(***************************************************************************)

\* Step 1: Workgroup w executes its current phase:
\*   - Reads all producers' outputs from the previous phase (phase[w] - 1)
\*     out of shared SLC (verifying release/acquire visibility!).
\*   - Writes its own output for phase[w] into its private USC store buffer.
WgComputeAndWrite(w) ==
    /\ cdmStage = "MEGA"
    /\ pc[w] = "RUN"
    /\ phase[w] < TotalPhases
    /\ LET p       == phase[w]
           curSlot == StageOf(p)
           prevOk  == IF p = 0 THEN TRUE
                      ELSE LET prevSlot == StageOf(p - 1)
                           IN  \A prod \in WGs : slcMem[prevSlot][prod] = p - 1
       IN  /\ readOk' = (readOk /\ prevOk)
           /\ uscBuf' = [uscBuf EXCEPT ![w][curSlot] = p]
           /\ pc'     = [pc EXCEPT ![w] = "FLUSHED"]
    /\ UNCHANGED << arrived, sense, epoch, phase, localSense,
                    slcMem, hostMem, cdmStage, cdmPackets, trailingBarrier >>

\* Step 2: Workgroup w executes memoryBarrierBuffer() (flushing private USC
\* store buffer into 48 MB SLC) and atomically increments `arrived`.
\*   - Last arriver (arrived + 1 = NumWGs) resets arrived := 0, flips sense,
\*     increments epoch, and advances immediately.
\*   - Non-last arriver transitions to WAIT_SENSE waiting for sense = 1 - localSense[w].
WgReleaseAndArrive(w) ==
    /\ cdmStage = "MEGA"
    /\ pc[w] = "FLUSHED"
    /\ LET p       == phase[w]
           curSlot == StageOf(p)
           nextP   == p + 1
           newSlc  == [slcMem EXCEPT ![curSlot][w] = uscBuf[w][curSlot]]
       IN  /\ slcMem' = newSlc
           /\ IF arrived + 1 = NumWGs
              THEN /\ arrived'    = 0
                   /\ sense'      = 1 - sense
                   /\ epoch'      = epoch + 1
                   /\ localSense' = [localSense EXCEPT ![w] = 1 - localSense[w]]
                   /\ phase'      = [phase EXCEPT ![w] = nextP]
                   /\ pc'         = [pc EXCEPT ![w] = IF nextP = TotalPhases THEN "DONE" ELSE "RUN"]
              ELSE /\ arrived'    = arrived + 1
                   /\ pc'         = [pc EXCEPT ![w] = "WAIT_SENSE"]
                   /\ UNCHANGED << sense, epoch, localSense, phase >>
    /\ UNCHANGED << uscBuf, hostMem, readOk, cdmStage, cdmPackets, trailingBarrier >>

\* Step 3: Waiting workgroup observes the flipped sense bit (acquire semantics),
\* flips its localSense, and advances to the next phase (or DONE).
WgObserveSense(w) ==
    /\ cdmStage = "MEGA"
    /\ pc[w] = "WAIT_SENSE"
    /\ sense = 1 - localSense[w]
    /\ LET nextP == phase[w] + 1
       IN  /\ localSense' = [localSense EXCEPT ![w] = 1 - localSense[w]]
           /\ phase'      = [phase EXCEPT ![w] = nextP]
           /\ pc'         = [pc EXCEPT ![w] = IF nextP = TotalPhases THEN "DONE" ELSE "RUN"]
    /\ UNCHANGED << arrived, sense, epoch, uscBuf, slcMem, hostMem, readOk,
                    cdmStage, cdmPackets, trailingBarrier >>

(***************************************************************************)
(* Final CDM Host-Memory Flush, Terminate & Host Readback                  *)
(***************************************************************************)

\* Once all workgroups finish, the CDM stream's trailing B_WEAK is upgraded IN PLACE
\* to B_HOST (0x6000157c) -- never emitting a second adjacent barrier -- flushing
\* 48 MB SLC to host-coherent memory.
CdmUpgradeHostFlush ==
    /\ cdmStage = "MEGA"
    /\ \A w \in WGs : pc[w] = "DONE"
    /\ cdmStage'        = "FLUSH_HOST"
    /\ cdmPackets'      = UpgradeTrailing(cdmPackets, B_MEMORY_HOST)
    /\ trailingBarrier' = B_MEMORY_HOST
    /\ hostMem'         = slcMem
    /\ UNCHANGED << arrived, sense, epoch, pc, phase, localSense,
                    uscBuf, slcMem, readOk >>

CdmTerminate ==
    /\ cdmStage = "FLUSH_HOST"
    /\ cdmStage'   = "TERMINATED"
    /\ cdmPackets' = Append(cdmPackets, "TERM")
    /\ UNCHANGED << arrived, sense, epoch, pc, phase, localSense,
                    uscBuf, slcMem, hostMem, readOk, trailingBarrier >>

HostReadback ==
    /\ cdmStage = "TERMINATED"
    /\ cdmStage' = "HOST_READ"
    /\ UNCHANGED << arrived, sense, epoch, pc, phase, localSense,
                    uscBuf, slcMem, hostMem, readOk, cdmPackets, trailingBarrier >>

DoneStutter ==
    /\ cdmStage = "HOST_READ"
    /\ UNCHANGED vars

(***************************************************************************)
(* Next-state relation & Fairness                                          *)
(***************************************************************************)

Next ==
    \/ CdmEmitPre1
    \/ CdmEmitPre2ElideAndUpgrade
    \/ CdmLaunchMegakernel
    \/ \E w \in WGs : WgComputeAndWrite(w)
    \/ \E w \in WGs : WgReleaseAndArrive(w)
    \/ \E w \in WGs : WgObserveSense(w)
    \/ CdmUpgradeHostFlush
    \/ CdmTerminate
    \/ HostReadback
    \/ DoneStutter

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(***************************************************************************)
(* Safety Invariants                                                       *)
(***************************************************************************)

TypeOK ==
    /\ arrived \in 0..(NumWGs - 1)
    /\ sense \in {0, 1}
    /\ epoch \in 0..TotalPhases
    /\ \A w \in WGs : pc[w] \in {"RUN", "FLUSHED", "WAIT_SENSE", "DONE"}
    /\ \A w \in WGs : phase[w] \in 0..TotalPhases
    /\ \A w \in WGs : localSense[w] \in {0, 1}
    /\ readOk \in BOOLEAN
    /\ cdmStage \in {"INIT", "PRE1", "PRE2", "MEGA", "FLUSH_HOST", "TERMINATED", "HOST_READ"}
    /\ trailingBarrier \in {B_NONE, B_WEAK_WFI, B_STRONG_USC, B_MEMORY_HOST}

\* 1. No data race / strict release-acquire stage visibility across SLC.
StageVisibility == readOk = TRUE

\* 2. No barrier overwrite:
\*    - Every workgroup that is still in RUN or FLUSHED has phase = epoch, so
\*      no straggler from an earlier epoch can corrupt the current barrier count.
\*    - Every workgroup in WAIT_SENSE is waiting on epoch's sense flip (phase = epoch
\*      if sense hasn't flipped yet, or phase = epoch - 1 if sense already flipped
\*      and it hasn't observed the flip yet).
NoBarrierOverwrite ==
    /\ \A w \in WGs : (pc[w] \in {"RUN", "FLUSHED"}) => (phase[w] = epoch)
    /\ \A w \in WGs : (pc[w] = "WAIT_SENSE") =>
           (  (phase[w] = epoch     /\ localSense[w] = sense)
           \/ (phase[w] = epoch - 1 /\ localSense[w] # sense) )

\* 3. Workgroups are always within at most 1 phase of each other.
BoundedPhaseSkew ==
    \A w1, w2 \in WGs :
        phase[w1] - phase[w2] \in {-1, 0, 1}

\* 4. HAL200/G17P CDM hardware rule: NEVER two adjacent barrier packets.
NoAdjacentCdmBarriers ==
    \A i \in 1..(Len(cdmPackets) - 1) :
        ~(IsBarrierPkt(cdmPackets[i]) /\ IsBarrierPkt(cdmPackets[i + 1]))

\* 5. Host readback coherence: when the host reads back after TERMINATE,
\*    every workgroup's final stage output is visible in hostMem.
HostReadbackCoherent ==
    (cdmStage = "HOST_READ") =>
        LET lastPhase == TotalPhases - 1
            lastSlot  == StageOf(lastPhase)
        IN  \A w \in WGs : hostMem[lastSlot][w] = lastPhase

(***************************************************************************)
(* Liveness Property (No Deadlock / No Livelock)                           *)
(***************************************************************************)

AllComplete ==
    <>((\A w \in WGs : pc[w] = "DONE") /\ cdmStage = "HOST_READ")

=============================================================================
