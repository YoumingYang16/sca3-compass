# R4 results — evidence stages kept separate

## Engineering and theory (completed)

- `checks/baseline-identity.json`: frozen target-only adapter exact p/e/reference/
  decision equality, Nt4/G32/M4095, same R4 runtime; no old-environment replay claim.
- `checks/preconfirm-tests.xml`:13 tests passed, including endpoint scales,
  all primary arrays, same-H tuples, rank grids, observed-input allowlist,
  independent streams, fail-closed behavior and diagnostic serialization.
- `checks/information-bounds.json`: deterministic integration agrees with exact
  affinity sech²(h/4). Necessary Nt for95% intervals with deterministic log-width
  less than log(1.25),log(1.5),log(2) is267,81,28 respectively. Not PC Power bounds.
- Sole reviewer accepted the fixed-bridge argument and pivot-only information
  calculation, then reviewed D001c record repair and MSE/interval limitations.
  Review is not proof of global novelty or a replacement for independent data.

## Development R4-D001c (completed; NOT confirmation)

Source: `D001c/summary.json`, `D001c/index.json`, frozen `D001c/protocol.json`.
6x64 independent families, G256, M4095;384 complete, zero hard failures.
The earlier23 serialization-failed executions are SAME-seed duplicates and are
not added to the384 independent units. All their artifacts remain in D001b.

| Predeclared scene | target Power% | bridge Power% | paired difference pp | bridge empirical FDR% |
|---|---:|---:|---:|---:|
| normal, matched, Ns32/Nt4,D1 |17.19|33.09|+15.90|0.419|
| t5,delta2.2,Nt4,D2.2 |13.63|43.05|+29.41|0.702|
| t3,delta4.6,Nt12,D5 |52.79|62.53|+9.74|0.375|
| t5,matched,Nt32,D5 (legal but loose) |45.31|6.13|−39.19|0.082|
| t5,dense/continuous,Nt12,D3 |18.71|19.71|+1.01|0.025|
| t5,delta4.6,Nt4,D2 (**outside bound**) |16.24|71.20|+54.96|4.371|

Equal-weight five legal nonnull DEV scenes: bridge−target+3.375pp, simultaneous
DEVELOPMENT interval[−12.104,+17.912]pp. This is not confirmed overall gain.
The legal loose-radius loss interval is[−50.667,−17.928]pp; it is retained,
not explained away by pooling favorable scenes. The boundary's low64-repeat
FDR point is NOT proof of nominal control; its premise is false.

Strong_target and strong_pool_bound are in the full table with actual FDR;
both remain EMPIRICAL_ONLY. The main source-only comparison changes the data
information and cannot establish bridge superiority at equal Nt by itself.

## Confirmation

`protocol-confirm.json` fixed one5120-family run before new results. The
count-weight bridge is unchanged from DEV. No reserve confirmation, sample
extension or post-result candidate switch was used. **5120/5120 completed,
zero hard failures,2690.47s wall**, final raw audit PASS. Original full arrays
and receipts are retained under C001/raw, indexed and hash-bound by index.json.

All numbers below come from C001/summary.json (SHA256
10064fbe7a9684fd58d7a6f5310a944c3de2c48fad40001742af6dc857a097b3).
Complete90 method-by-scene rows and133 intervals are generated deterministically
in generated/RESULT_TABLES.md; generated/paired-power.svg shows EVERY nonnull
scene including the out-of-bound failure. All intervals are the predeclared
97.5% simultaneous family (cap160,133actually used), not gene-level intervals.

| scene | bridge−target Power pp | simultaneous interval pp | bridge FDR% / upper% |
|---|---:|---|---|
|C0 normal matched Ns32/Nt4|17.8041|[10.6774,24.6114]|0.4292 /2.5432|
|C1 t5 delta3.8,D4,Nt4|23.2039|[16.4854,29.5029]|0.3321 /2.4468|
|C2 t3 delta3,D3.3,Nt12|6.9738|[3.6984,10.3719]|0.3197 /2.4332|
|C3 legal loose delta1,D5,Nt32|−40.1348|[−43.9801,−35.9849]|0.0103 /2.1255|
|C7 dense continuous|−0.4979|[−3.4095,2.3849]|0.0238 /2.1386|
|C8 fixed36,more target|1.3289|[−1.9465,4.6653]|0.3252 /2.4389|
|X9 outside delta5.8>D2|61.0907|[55.3322,66.4962]|7.2023 /9.3444|

Core bridge37.4700% versus target36.0237%: +1.4463pp [−1.1428,3.9744].
Local gains C0/1/2 supported; legal loose-D loss supported; overall-advantage
gate FAILS. Nine legal bridge FDR upper endpoints all<=2.86545%, so the MC
support gate PASSES. C4global-null and C5singleton FDR0, upper2.1152%;
C6mixed-sign PC-null FDR.1953125%, upper2.8655%. X9 FDR interval
[6.0000,9.3444]% is wholly above5%: a confirmed OUTSIDE-premise failure.
Target-only X9 remains .4335%, upper2.5629%; it does not depend on sourceD.

Core bridge−target_ordinary+9.5491pp [6.8555,11.9080] changes BOTH calibration
and projection mixture. Core bridge−strong_target−23.0151pp [−25.1952,−20.0756];
bridge−strong_pool_bound−11.3281pp [−13.4943,−8.7475]. Strong pool's legal FDR
upper max4.6793% passes this SAME finite empirical check; it still lacks the
R4 exact finite theorem. Strong_target C5 FDR6.25% has interval[2.3410,12.7937]%,
so do not assert proven true inflation merely from that point estimate.
The protected source endpoint is already close in C1/2; no additional posthoc
bridge−source_bound interval was invented. Its full numbers are retained.

Fixed total36 allocation C1→C8 changes target-only+23.4375pp
[17.2335,29.6546], bridge+1.5625pp[−2.4042,5.5458], strong_target+1.9914pp
[−2.1295,6.7591]. These are paired512units, not1024independent units; they
measure calibration allocation, NOT identical-input algorithm improvement.

Final audit:5120records,46080decision/score checks,450means,133CIrecomputations,
9prespecified input/diagnostic regenerations and18exact inference replays.
No kernel numerical failures; no strong call fallback; each kernel method
has20480/20480converged DIR flags and20480/20480converged PILOT flags.
Convergence does not mean nonzero gamma. The only tagged nonfinite diagnostics
are infinity df entries in the frozen V1 Gaussian-limit branches; all p/e
arrays are finite. Complete counts, method p90/p99timers and worker CPU are in
checks/final-result-audit.json; audit took91.73s. Method mean times: target
.8737s,bridge.8773s,strong-target fullV1call.4153s,strong-pool fullV1call.4447s.

## Scope and interpretation

The confirmed result is a conditional utility tradeoff, not a universally
best method. External D may be too conservative to be useful, and actual delta
is unavailable to the inference API. Posthoc selection by the true slack would
be an oracle and is not offered as a deployment rule. Target-only requires no
source drift bound once Ct is genuinely target-matched. Real Ct availability
remains unverified; all results here concern newly simulated independent banks.
