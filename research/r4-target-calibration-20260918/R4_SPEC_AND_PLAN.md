# R4 — target calibration, safe borrowing, and information requirements

Started 2026-09-18 12:26:49 Asia/Hong_Kong. Separate authorized research;
V1/R2/R3/C2 and export archives remain read-only. No real patient input and
no opening GSE320100. No prior confirmation is reused as R4 confirmation.

## Question and contract

For independent source Cs[Ns,4,6], independent target-centered Ct[Nt,4,6],
and target Z[G,4,6], retain R3 signed 2-of-4 PC, m=2G, q=.05. Target and Ct
share kappa_t; source has kappa_s. Unknown target SPD R, Gaussian angular
structure, common positive block radial scale, and common pipeline location
remain required. Calibration blockwise nonsingular left transforms cancel.
FDR=E[FP/max(1,TP+FP)]; Power=E[TP/number of signed alternatives], undefined
for all-PC-null scenes. A whole independently generated family is one repeat.

Two states are distinct: CURRENT_INPUT_ONLY (no certified real Ct) and
CONDITIONAL_TARGET_CALIBRATION_RESEARCH (new independent simulated Ct).
The latter does not certify real molecular application feasibility.

## Fixed sequence and limits

1. Exact frozen R3 target-only adapter; preserve p/e/reference/learning
   intermediates and test exact identity. Nt>=4, Nt=0 not applicable.
2. Verify fixed bridge K=D^(1-lambda)*Ks^(1-lambda)*Kt^lambda and full chain.
   Main model: external D>=1 with 0<delta=kappa_t/kappa_s<=D. No fitted D.
   Initial candidates: lambda=0, lambda=1, lambda=Nt/(Ns+Nt) ONLY.
   Interior weight minimizes exact variance of log B among fixed bridges;
   it does not optimize Power and has a slack-dependent conservative bias.
3. Choose **target-calibration information requirements** as the single
   theoretical extension. Study reduced-pivot two-bank information with
   unknown drift and finite-sample distinguishability; do not develop an
   adaptive pretest/pooling family. No full-data/PC minimax claim.
4. Small fixed DEV (4–6 non-Cartesian cases), then one frozen confirmation
   with ordinary/heavy tails, partial nulls, dense/continuous effects, legal
   drift and an explicit underestimated-D boundary. New protocol must be
   written before results. One main bridge and existing target-only backup.
5. Freeze/replay/audit, produce results, related work and final claim map.
   Stop after delivery; no R5 or blanket algorithm search.

## Work package 1 — in progress

Evidence gap: correct source binding and target-only baseline, then a complete
bridge argument. Resources: existing Python environment, <=8 worker processes,
one main agent plus one read-only mathematical/code reviewer, 24 logical CPU,
16GB installed RAM. First soft review about 13:56:49 HKT; not a time stop.
Stage-1 checks: exact endpoints (p/e/stats/reference), delta=1 design identity,
rank support, zero/forbidden input rejection, separate random streams, complete
same-H reference tuple, frozen import SHA/path guard. No large experiment before
these pass. Any fallback retains status and zero decisions; no silent deletion.

## Preserved history / non-goals

R3 old +6.89434pp vs R2 and old strong gap23.51441pp remain their own protocols.
C2 oracle kappa gains are extra-information diagnostics, not deployable gains.
Neither fixed Power target nor full distribution-free guarantee is required.
The bridge is a classical pivot/invariance specialization unless a distinct
contribution is actually established. Validity, usefulness, novelty and real
input availability will be graded separately.

## Current state

Intake complete for actual R3/R2 implementations, original theory/protocol,
C2 mechanism audit and export information status. No prior Python research
process was active at intake. Baseline check now completed: checks/baseline-identity.json
binds exact p/e/reference/decision identity versus frozen R3 with Nt4/G32/M4095.
It is an engineering fixture, not a Power or FDR experiment.

Environment event: original .venv NumPy2.5.3 random/_common DLL was rejected by
Windows Smart App Control (CodeIntegrity3077, 2026-09-18 12:34:24 HKT). No policy
or security setting was changed. Independent .r4-venv uses the bundled allowed
NumPy2.3.5 plus official PyPI SciPy1.16.3, threadpoolctl3.6.0, pytest8.4.2.
This changes the environment, NOT historical sources/results; exact identity
checks compare old/new implementations in the SAME R4 environment. They do not
claim bytewise reproduction of the old-environment C001 experiments.
Stage1 initial tests:10 passed;1 fixture error because old global pytest temp
directory is inaccessible. Retained checks/stage1.xml. Rerun will use a new
R4-owned temp directory, not alter global permissions or delete old files.

### 12:49 HKT execution update

Stage1 now13/13 passed (checks/stage1-final.xml). Reduced primary-only kernel
exactly matches frozen endpoints including observed-scale statistics and p/e/
reference arrays. Same-H tuple checks and separate generator streams pass.
T1–T5 derivations are complete in R4_THEORY.md; independent reviewer has accepted
the mathematical starting chain with explicit scope restrictions. Deterministic
affinity integration agrees with sech²(h/4); information-bounds.json records
necessary pivot-only counts, NOT PC Power requirements.

D001 was frozen but never run (bad analysis namespace caught prelaunch).
D001b hit JSON serialization of legitimate V1 infinite-df diagnostics after
saving23 family evidence arrays. All partial artifacts retained; no results
used to select a method. D001c repeats the same predeclared seeds with tagged
nonfinite diagnostics and pre-serialized complete records. No inference formula,
scenario, tuning or sample count changed. This is record repair, not new
independent evidence. First32/384 D001c records complete, zero hard failures.
Next: finish fixed DEV, analyze paired variance/cost, preregister ONE confirmation.

### DEV complete / confirmation preparation

D001c:384/384 complete, zero hard failures,268.44s. The full raw-array scoring
audit and summary finished. Bridge-minus-target Power in six scenes:
+15.90,+29.41,+9.74,−39.19,+1.01,+54.96pp (last is outside bound, NOT a valid
efficiency success). Core mean+3.38pp has wide DEV interval[−12.10,+17.91]pp.
In the legal loose-radius case the regression is already clearly supported:
interval[−50.67,−17.93]pp. Do not discard it or seek an adaptive rescue.
DEV also retains invalid source/naive-pool high Power with inflated FDR.

Work package2: ONE frozen confirmation of unchanged count-weight bridge and
target-only backup,512 families in each of10 prespecified new parameter scenes.
Protocol-confirm.json fixes133 comparisons inside cap160, alpha.025, nominal
FDR.05;10 workers, no optional repeats, no reserve confirmation. Allocation
pair Ns32/Nt4 versusNs4/Nt32 has fixed total36 and common Z. Source data advantage,
same-information comparison and allocation changes are explicitly separated.
Necessary evidence: confirm local useful conditions, quantify the legal loose-D
loss and out-of-bound failure, do not demand or imply uniform superiority.
Expected compute based on DEV45–65min; soft review13:56:49HKT remains a progress
checkpoint, not a stop or statistical rule. No parameter/algorithm changes.

### 13:06 HKT checkpoint

C001 started12:57:50HKT, owned parentPID38700,10 worker processes; latest checked
864/5120 complete, zero hard failures. Parent plus workers used about1.185GB
working set at this snapshot (not a measured peak). No numerical outcomes used
for tuning during confirmation. All12 frozen C001 files and45 old dependency
bindings checked by the sole reviewer, who found no inference/protocol blocker.
New deterministic G256 endpoint/isolation audit passed; actual guarded CLI
invocation completed and saved under example/guarded-bridge. These are replays/
engineering examples, not additional independent confirmations.

Within the SAME information-bound route, T5 now has a worktree-only analytic
rate corollary, reviewed independently: reduced log-scale minimax risk lies
between .1402530099/Nt and2.2898681337/Nt for arbitraryNs and one-sided drift.
C001's original frozen proof remains byte-identical; no experiment changed.
No claim that raw matrices or PC Power share this lower bound.

Next: finish the fixed run, frozen analysis, full raw/schema/decision/133-interval
audit and prespecified rep0 replay, then final report and seal. No new search.

### 13:53 HKT final research checkpoint — execution complete, sealing next

C001 completed at13:42:41HKT,5120/5120,zero hard failures,2690.47s wall;
the original process/session exited normally. No re-seeding, added families,
candidate changes or protected real data access. Fixed analysis completed;
full final audit PASS at13:46:23HKT (checks/final-result-audit.json),91.73s:
5120records,46080decision/score checks,450means,133intervals,9predeclared
input/diagnostic regenerations and18exact target/bridge array replays.
All kernel DIR/PILOT convergence flags true; no numerical/strong fallback.

Scientific outcome: legal bridge FDR simultaneous upper max2.86545% passes
the fixed MC support gate. Core bridge−target+1.4463pp[−1.1428,3.9744] does
NOT establish overall benefit; legal loose-D loss−40.1348pp is confirmed.
C0/1/2 local gains confirmed. Out-of-bound X9 FDR7.2023%[6.0000,9.3444]%
is genuinely above nominal, retained rather than hidden. Bridge remains behind
both empirical strong comparators in the core. Information-bound and complete
conditional finite-control proofs are retained; priority not established.

Current deliverables: final report, results, complete theory, related-work map,
generated full tables/figure, frozen implementation/CLI and all original arrays.
Recommend target-only baseline, bridge_count only with independently justified
external D and target calibration; no truth-driven adaptive selection.
Real Ct and D are NOT certified. This is scoped research completion, not
universal success, clinical applicability or established high-impact novelty.

Work package3 (convergence/packaging only): final reviewer consistency check,
write FINAL_STATUS, run seal_delivery.py --zip to bind all artifacts/dependencies
and verify archive CRC/member hashes. Expected ordinary resource cost a few
minutes of local I/O; stop when archive receipt exists and final checks pass.
No research extension. About86minutes since12:26:49; if packaging crosses the
13:56:49 soft review, inspect actual I/O progress only, do not restart jobs.
After receipt: STOP at delivery; no next experiment command or R5 authorized.

### Final approximately90-minute review (13:55 HKT)

The sole reviewer completed final consistency review: no remaining numerical
or protocol error. Corrected C7 wording so a zero-crossing Power interval is
not misrepresented as proof of no true improvement. No algorithm or experiment
change. Current core gap+1.4463pp remains uncertain; severe C3 loss and outside
X9 error inflation remain explicit. Computation and audit are finished; no
research workers remain. Only local hash-bound packaging is now authorized.
Closure command (no inference):

    research/r4-target-calibration-20260918/.r4-venv/Scripts/python.exe -B research/r4-target-calibration-20260918/seal_delivery.py --zip

Actual final completion time/size/hash comes from delivery/ARCHIVE_RECEIPT.json,
not this pre-seal entry. If sealing fails, retain the partial package and report
the packaging failure rather than rerun research. After successful receipt,
this state is terminal for the current scope; do not resume old search jobs.
