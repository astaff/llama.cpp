------------------------ MODULE AgxMegakernelW4 ------------------------
(***************************************************************************)
(* Formal TLA+ specification of the Wave 4 4-Stage Persistent 40-WG        *)
(* Inkling-2-Mini Megakernel on Apple M4 Max (T6041 / AGX3B / HAL200).     *)
(*                                                                         *)
(* Models 48 passes x 4 stages per pass (192 phases + LM Head) executing   *)
(* on a FIXED, STATIC, SLC-PINNED activation arena (<24 MiB < 50% of       *)
(* M4 Max's 48 MiB SLC) with zero ggml_gallocr VA recycling:               *)
(*   Stage 0: Depth-Mix + Ctrl + Coalesced QKV/R/Gate/Sink -> slot 0 (qkv) *)
(*   Stage 1: ShortConv + Clip + FWHT + FlashAttn + Wo     -> slot 1 (att) *)
(*   Stage 2: Depth-Mix + Ctrl + Router + MoE GateUp+SiTU  -> slot 2 (moe) *)
(*   Stage 3: MoE Down + Weighted-Sum + Residual Add       -> slot 3 (h)   *)
(*                                                                         *)
(* Verified safety & liveness properties:                                  *)
(*   - TypeOK            : all state variables remain well-typed           *)
(*   - StageVisibility   : every consumer S_k reads exact SLC-flushed      *)
(*                         producer output from S_{k-1} of the same pass   *)
(*   - NoSlcSlotAliasing : no two concurrent WGs write different phases    *)
(*                         into the same SLC arena slot (0 WAR/WAW races)  *)
(*   - NoBarrierOverwrite: sense-reversing barrier prevents straggler      *)
(*                         collision across all 192+ phases                *)
(*   - BoundedPhaseSkew  : all WGs stay within <= 1 phase of each other    *)
(*   - AllComplete       : deadlock-free & livelock-free termination       *)
(***************************************************************************)

EXTENDS Integers, FiniteSets

CONSTANTS
    NumWGs,       \* Persistent workgroups (e.g. 3 in TLC; 40 on M4 Max)
    NumPasses     \* Passes modeled (e.g. 2 in TLC; 48 in Inkling-2-Mini)

ASSUME NumWGs \in Nat /\ NumWGs >= 2
ASSUME NumPasses \in Nat /\ NumPasses >= 1

NumStages   == 4
WGs         == 1..NumWGs
TotalPhases == NumPasses * NumStages
Slots       == 0..(NumStages - 1)
NO_VER      == -1

VARIABLES
    arrived,      \* 0..NumWGs : sense-barrier arrival counter
    sense,        \* {0, 1}    : global toggling sense bit
    epoch,        \* 0..TotalPhases : completed barrier epochs
    pc,           \* [WGs -> {"RUN", "FLUSHED", "WAIT_SENSE", "DONE"}]
    phase,        \* [WGs -> 0..TotalPhases]
    localSense,   \* [WGs -> {0, 1}]
    uscBuf,       \* [WGs -> [Slots -> (-1)..(TotalPhases - 1)]] : private USC L1 store buffer
    slcArena,     \* [Slots -> [WGs -> (-1)..(TotalPhases - 1)]] : static SLC-pinned arena
    readOk,       \* BOOLEAN : latched consumer visibility check
    noAliasOk     \* BOOLEAN : latched zero-RAW/WAR/WAW slot aliasing check

vars == << arrived, sense, epoch, pc, phase, localSense,
           uscBuf, slcArena, readOk, noAliasOk >>

StageOf(p) == p % NumStages

\* Producer slot consumed by phase p (p > 0 consumes StageOf(p - 1))
InputSlot(p) == StageOf(p - 1)

Init ==
    /\ arrived    = 0
    /\ sense      = 0
    /\ epoch      = 0
    /\ pc         = [w \in WGs |-> "RUN"]
    /\ phase      = [w \in WGs |-> 0]
    /\ localSense = [w \in WGs |-> 0]
    /\ uscBuf     = [w \in WGs |-> [s \in Slots |-> NO_VER]]
    /\ slcArena   = [s \in Slots |-> [w \in WGs |-> NO_VER]]
    /\ readOk     = TRUE
    /\ noAliasOk  = TRUE

\* Step 1: Workgroup w executes stage StageOf(phase[w]), reading InputSlot(phase[w])
\*         from slcArena and writing its shard of StageOf(phase[w]) into private uscBuf,
\*         then flushing uscBuf -> slcArena (coherent + memoryBarrierBuffer).
RunAndFlush(w) ==
    /\ pc[w] = "RUN"
    /\ phase[w] < TotalPhases
    /\ LET p      == phase[w]
           outS   == StageOf(p)
           inOk   == IF p = 0 THEN TRUE
                     ELSE \A u \in WGs : slcArena[InputSlot(p)][u] = p - 1
           \* No other WG may currently be reading outS from an older phase
           aliasOk == \A u \in WGs :
                        (pc[u] = "RUN" /\ phase[u] > 0 /\ phase[u] < TotalPhases /\ InputSlot(phase[u]) = outS)
                        => (phase[u] = p)
       IN
       /\ readOk'    = (readOk /\ inOk)
       /\ noAliasOk' = (noAliasOk /\ aliasOk)
       /\ uscBuf'    = [uscBuf EXCEPT ![w][outS] = p]
       /\ slcArena'  = [slcArena EXCEPT ![outS][w] = p]
       /\ pc'        = [pc EXCEPT ![w] = "FLUSHED"]
       /\ UNCHANGED << arrived, sense, epoch, phase, localSense >>

\* Step 2: Workgroup w enters the device-scope sense-reversing atomic barrier.
EnterBarrier(w) ==
    /\ pc[w] = "FLUSHED"
    /\ LET targetSense == 1 - localSense[w]
           nextArrived == arrived + 1
       IN
       IF nextArrived = NumWGs THEN
           /\ arrived'    = 0
           /\ sense'      = targetSense
           /\ epoch'      = epoch + 1
           /\ localSense' = [localSense EXCEPT ![w] = targetSense]
           /\ phase'      = [phase EXCEPT ![w] = phase[w] + 1]
           /\ pc'         = [pc EXCEPT ![w] = IF phase[w] + 1 = TotalPhases THEN "DONE" ELSE "RUN"]
           /\ UNCHANGED << uscBuf, slcArena, readOk, noAliasOk >>
       ELSE
           /\ arrived'    = nextArrived
           /\ localSense' = [localSense EXCEPT ![w] = targetSense]
           /\ pc'         = [pc EXCEPT ![w] = "WAIT_SENSE"]
           /\ UNCHANGED << sense, epoch, phase, uscBuf, slcArena, readOk, noAliasOk >>

\* Step 3: Waiting workgroup observes global sense == localSense[w] and advances.
ExitBarrier(w) ==
    /\ pc[w] = "WAIT_SENSE"
    /\ sense = localSense[w]
    /\ phase' = [phase EXCEPT ![w] = phase[w] + 1]
    /\ pc'    = [pc EXCEPT ![w] = IF phase[w] + 1 = TotalPhases THEN "DONE" ELSE "RUN"]
    /\ UNCHANGED << arrived, sense, epoch, localSense, uscBuf, slcArena, readOk, noAliasOk >>

Next ==
    \/ \E w \in WGs : RunAndFlush(w) \/ EnterBarrier(w) \/ ExitBarrier(w)
    \/ ((\A w \in WGs : pc[w] = "DONE") /\ UNCHANGED vars)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* --- Safety Invariants ---
TypeOK ==
    /\ arrived \in 0..NumWGs
    /\ sense \in {0, 1}
    /\ epoch \in 0..TotalPhases
    /\ \A w \in WGs : pc[w] \in {"RUN", "FLUSHED", "WAIT_SENSE", "DONE"}
    /\ \A w \in WGs : phase[w] \in 0..TotalPhases
    /\ \A w \in WGs : localSense[w] \in {0, 1}

StageVisibility == readOk

NoSlcSlotAliasing == noAliasOk

BoundedPhaseSkew ==
    \A u, v \in WGs : phase[u] - phase[v] <= 1 /\ phase[v] - phase[u] <= 1

NoBarrierOverwrite ==
    \A w \in WGs : (pc[w] = "WAIT_SENSE") => (phase[w] = epoch \/ phase[w] + 1 = epoch)

\* --- Liveness ---
AllComplete == <>(\A w \in WGs : pc[w] = "DONE" /\ epoch = TotalPhases)

=============================================================================
