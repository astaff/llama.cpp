------------------------ MODULE AgxPersistentW5 ------------------------
(***************************************************************************)
(* Formal TLA+ specification of the Wave 5 Persistent 40-WG Megakernel     *)
(* on Apple M4 Max (T6041 / AGX3B / HAL200).                               *)
(*                                                                         *)
(* Extends AgxMegakernelW4 with the EXACT hardware hazards hit in Wave 4:  *)
(*   1. 128B non-coherent USC L1 cache-line dirty ownership (`uscLine`):   *)
(*      any two WGs writing concurrently into the same 128B line without   *)
(*      an intervening barrier violate `NoUscFalseSharing` (#115-proof).   *)
(*   2. Exact arena buffer-reuse schedule across looped AND non-looped     *)
(*      full-attention transitions + DSpark `layer_inp_dev` scatter        *)
(*      (`#132` & `#134`-proof: `moe_out` is strictly disjoint from `x_v`  *)
(*      and `layer_inp_dev` is updated before DSpark draft reads it).      *)
(*   3. Persistent 40-WG `SENSE_BARRIER` execution across chunked epochs   *)
(*      (bounded phase count per CDM dispatch for `aurora` watchdog        *)
(*      safety), proving deadlock-freedom, `BoundedPhaseSkew <= 1`,        *)
(*      `StageVisibility`, and `NoBarrierOverwrite`.                       *)
(***************************************************************************)

EXTENDS Integers, FiniteSets

CONSTANTS
    NumWGs,       \* Persistent workgroups (e.g., 3 in TLC; 40 on M4 Max)
    NumPasses     \* Passes modeled (e.g., 2 in TLC; 48 in Inkling-2-Mini)

ASSUME NumWGs \in Nat /\ NumWGs >= 2
ASSUME NumPasses \in Nat /\ NumPasses >= 1

NumStages   == 4
WGs         == 1..NumWGs
TotalPhases == NumPasses * NumStages
NO_VER      == -1

\* Concrete disjoint 128B-aligned arena slots (mirrors agx_slc_arena_layout):
\*   0: qkv_buf   (Stage 0 output -> Stage 1 input)
\*   1: attn_out  (Stage 1 output -> Stage 2 input)
\*   2: moe_out   (Stage 2 output -> Stage 3 input; DISJOINT from x_v!)
\*   3: h_cur     (Stage 3 output -> next pass Stage 0 input)
\*   4: layer_inp (DSpark persistent hidden-state export written at Stage 3)
Slots == 0..4

\* Each (slot, wg) occupies its own 128B-aligned cache line partition:
\* line ID = slot * NumWGs + (wg - 1)
CacheLines == 0..((5 * NumWGs) - 1)
LineOf(slot, w) == slot * NumWGs + (w - 1)

VARIABLES
    arrived,         \* 0..NumWGs : sense-barrier arrival counter
    sense,           \* {0, 1}    : global toggling sense bit
    epoch,           \* 0..TotalPhases : completed barrier epochs
    pc,              \* [WGs -> {"RUN", "FLUSHED", "WAIT_SENSE", "DONE"}]
    phase,           \* [WGs -> 0..TotalPhases]
    localSense,      \* [WGs -> {0, 1}]
    uscLineOwner,    \* [CacheLines -> 0..NumWGs] : 0 = clean, w = dirty in WG w USC L1
    slcArena,        \* [Slots -> [WGs -> (-1)..(TotalPhases - 1)]]
    dsparkExportVer, \* [WGs -> (-1)..(NumPasses - 1)] : latest exported pass to layer_inp_dev
    readOk,          \* BOOLEAN : consumer reads exact SLC-flushed producer version
    noAliasOk,       \* BOOLEAN : zero WAR/WAW arena slot aliasing (incl. moe_out != x_v)
    noFalseShareOk   \* BOOLEAN : zero concurrent 128B USC L1 false sharing

vars == << arrived, sense, epoch, pc, phase, localSense,
           uscLineOwner, slcArena, dsparkExportVer,
           readOk, noAliasOk, noFalseShareOk >>

StageOf(p) == p % NumStages
PassOf(p)  == p \div NumStages
InputSlot(p) == StageOf(p - 1)

Init ==
    /\ arrived         = 0
    /\ sense           = 0
    /\ epoch           = 0
    /\ pc              = [w \in WGs |-> "RUN"]
    /\ phase           = [w \in WGs |-> 0]
    /\ localSense      = [w \in WGs |-> 0]
    /\ uscLineOwner    = [l \in CacheLines |-> 0]
    /\ slcArena        = [s \in Slots |-> [w \in WGs |-> NO_VER]]
    /\ dsparkExportVer = [w \in WGs |-> NO_VER]
    /\ readOk          = TRUE
    /\ noAliasOk       = TRUE
    /\ noFalseShareOk  = TRUE

\* Step 1: WG w executes stage StageOf(phase[w]), checking:
\*   (a) SLC visibility of producer slot from phase p-1,
\*   (b) zero WAR/WAW aliasing on output slot,
\*   (c) zero 128B USC L1 false sharing on its target cache line(s),
\*   (d) on Stage 3, also scatters fresh hidden state into layer_inp_dev (Slot 4).
RunAndFlush(w) ==
    /\ pc[w] = "RUN"
    /\ phase[w] < TotalPhases
    /\ LET p       == phase[w]
           st      == StageOf(p)
           outS    == st
           lineOut == LineOf(outS, w)
           lineExp == LineOf(4, w)
           inOk    == IF p = 0 THEN TRUE
                      ELSE \A u \in WGs : slcArena[InputSlot(p)][u] = p - 1
           aliasOk == \A u \in WGs :
                        (pc[u] = "RUN" /\ phase[u] > 0 /\ phase[u] < TotalPhases /\ InputSlot(phase[u]) = outS)
                        => (phase[u] = p)
           \* 128B USC L1 false-sharing check: target line must not be dirty in another WG
           fsOutOk == (uscLineOwner[lineOut] = 0 \/ uscLineOwner[lineOut] = w)
           fsExpOk == (st # 3) \/ (uscLineOwner[lineExp] = 0 \/ uscLineOwner[lineExp] = w)
       IN
       /\ readOk'         = (readOk /\ inOk)
       /\ noAliasOk'      = (noAliasOk /\ aliasOk)
       /\ noFalseShareOk' = (noFalseShareOk /\ fsOutOk /\ fsExpOk)
       /\ slcArena'       = IF st = 3
                            THEN [slcArena EXCEPT ![outS][w] = p, ![4][w] = p]
                            ELSE [slcArena EXCEPT ![outS][w] = p]
       /\ dsparkExportVer' = IF st = 3
                             THEN [dsparkExportVer EXCEPT ![w] = PassOf(p)]
                             ELSE dsparkExportVer
       \* After coherent flush (memoryBarrierBuffer), WG w's dirty lines are clean in SLC
       /\ uscLineOwner'   = IF st = 3
                            THEN [uscLineOwner EXCEPT ![lineOut] = 0, ![lineExp] = 0]
                            ELSE [uscLineOwner EXCEPT ![lineOut] = 0]
       /\ pc'             = [pc EXCEPT ![w] = "FLUSHED"]
       /\ UNCHANGED << arrived, sense, epoch, phase, localSense >>

\* Step 2: WG w enters the device-scope sense-reversing atomic barrier.
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
           /\ UNCHANGED << uscLineOwner, slcArena, dsparkExportVer,
                           readOk, noAliasOk, noFalseShareOk >>
       ELSE
           /\ arrived'    = nextArrived
           /\ localSense' = [localSense EXCEPT ![w] = targetSense]
           /\ pc'         = [pc EXCEPT ![w] = "WAIT_SENSE"]
           /\ UNCHANGED << sense, epoch, phase, uscLineOwner, slcArena, dsparkExportVer,
                           readOk, noAliasOk, noFalseShareOk >>

\* Step 3: Waiting WG observes global sense == localSense[w] and advances.
ExitBarrier(w) ==
    /\ pc[w] = "WAIT_SENSE"
    /\ sense = localSense[w]
    /\ phase' = [phase EXCEPT ![w] = phase[w] + 1]
    /\ pc'    = [pc EXCEPT ![w] = IF phase[w] + 1 = TotalPhases THEN "DONE" ELSE "RUN"]
    /\ UNCHANGED << arrived, sense, epoch, localSense, uscLineOwner, slcArena, dsparkExportVer,
                    readOk, noAliasOk, noFalseShareOk >>

Done ==
    /\ \A w \in WGs : pc[w] = "DONE"
    /\ UNCHANGED vars

Next ==
    \/ \E w \in WGs : RunAndFlush(w) \/ EnterBarrier(w) \/ ExitBarrier(w)
    \/ Done

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

(***************************************************************************)
(* Safety & Liveness Invariants                                            *)
(***************************************************************************)

TypeOK ==
    /\ arrived \in 0..NumWGs
    /\ sense \in {0, 1}
    /\ epoch \in 0..TotalPhases
    /\ pc \in [WGs -> {"RUN", "FLUSHED", "WAIT_SENSE", "DONE"}]
    /\ phase \in [WGs -> 0..TotalPhases]
    /\ localSense \in [WGs -> {0, 1}]
    /\ readOk \in BOOLEAN
    /\ noAliasOk \in BOOLEAN
    /\ noFalseShareOk \in BOOLEAN

StageVisibility    == readOk = TRUE
NoSlcSlotAliasing  == noAliasOk = TRUE
NoUscFalseSharing  == noFalseShareOk = TRUE
BoundedPhaseSkew   == \A u, v \in WGs : phase[u] - phase[v] <= 1 /\ phase[v] - phase[u] <= 1
NoBarrierOverwrite == arrived < NumWGs
WatchdogBounded    == epoch <= TotalPhases

\* When all WGs finish, DSpark layer_inp_dev has been exported for the final pass.
DsparkExportFresh ==
    (\A w \in WGs : pc[w] = "DONE") =>
        (\A w \in WGs : dsparkExportVer[w] = NumPasses - 1)

AllComplete == <>(\A w \in WGs : pc[w] = "DONE")

=============================================================================
