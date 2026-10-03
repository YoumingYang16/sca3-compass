# C001 frozen-repair audit

2026-09-17. Sole independent reviewer. Focus: disposition of B1-B3 in
`REVIEW_CONFIRM_PREFLIGHT.md` and the associated Delta/comparator repairs.

**Disposition: B1-B3 resolved for the inspected frozen C001 execution. No
remaining blocking finding in this focused review. No interruption, re-freeze,
new confirmation, or change to the primary algorithm is requested.** Main
retains execution, integration, and final-evidence responsibility.

I read the frozen sources and performed read-only file/hash/ZIP and launch-path
checks. I did not execute project code, tests, fits, or simulations; inspect
interim scientific results; create agents; or modify the running job. Only this
review was written. This is not a completed-results/replay acceptance verdict.

## Frozen identity inspected

- Freeze timestamp: `2026-09-17T06:17:11.775705+00:00`.
- `C001/freeze.json` SHA256:
  `68068c9f1cf68eb46700b18287b0d226a8a3c2ff1c5ea592e036ee88a2538d8e`.
- Trusted release ZIP SHA256:
  `e4bd2e6061bc886796916013e30ae5c05b86c543ab3c4b129f8173afbdba8fe2`.
- All 14 frozen file hashes and all 109 extracted-source hashes match their
  freeze entries. Independently read all 109 ZIP source entries and verified
  their digests against the same dependency map: zero mismatches.
- The freeze records 37 resolved release modules. The observed live command for
  PID 85092 runs `research/r2-finite-20260917/C001/confirm.py run --out
  research/r2-finite-20260917/C001`, not the mutable parent harness.

## Prior findings: closed

### B1: actual source dependency binding

`C001/provenance.py:9-32` checks the trusted ZIP and binds every archived source
file to the extracted file, recording the resolved imported release modules.
`provenance.py:35-46` rechecks frozen files, extracted dependencies, and the paths
of recorded modules present in the process. `C001/confirm.py:91-103` checks the
executor and ZIP and installs this verification as every pool worker's
initializer. `C001/analyze_confirm.py:27-29` checks the freeze before analysis.
Together with the observed frozen launch path and matching hashes, this closes
the previous ZIP-versus-actual-import gap for this run.

### B2: task, seed, checkpoint, and metric identity

`provenance.py:49-51` now uses

    seed = 2 * ((base_seed * 32 + case) * 1024 + rep) + 1.

This is injective for fixed base seed, `0 <= case < 32`, `0 <= rep < 1024`;
the 15 x 512 plan lies strictly within those ranges. This conclusion follows
algebraically, without running a seed enumeration or experiment. There is no
longer a one-uint32 compression at this interface.

`provenance.py:54-74` checks requested task/seed/freeze identity, canonical input
and evidence paths, their hashes, the exact metric/decision method roster, and
re-scores each stored decision against stored truth. `confirm.py:40-43` invokes
that check on resumed records, and `:46-47` records the seed before computation.

`analyze_confirm.py:30-45` requires the exact planned Cartesian set, canonical
record paths, record hashes, matching record/index statuses, and the same
record/metric verification. `:48-50` blocks summaries on hard failures rather
than deleting those families. `:79-82` binds the summary to protocol, freeze,
and index. Conservative numerical no-discovery outcomes remain in the planned
metric denominators. These repair the earlier identity and trusted-metric gaps.

### B3: explicit two-sided interval contract

`C001/protocol.json:10-17` now states 338 two-sided intervals / 676 bounds, cap
512 intervals, tail budget `.05/(2*512)`, the power-only nonnegative paired
range, and the auxiliary unreported `PB_ordinary` label. It distinguishes the
worst-case MCSEs for [0,1] means and [-1,1] paired differences.

`analyze_confirm.py:34-35,55-78` reads the frozen alpha/cap and requires the
actual interval count to equal the declared count. The roster is 243 basic
method/metric intervals + 8 Delta intervals + 72 paired-power intervals + 15
calibration-domain coverage intervals = 338. The conservative simultaneous
allocation is consistent with that roster. The narrower paired range is used
only for `PB_grid - PB_main` power, not FDP.

## Associated repairs and scientific scope

- `confirm.py:65-76` saves Delta-specific p/e arrays and status/time/error
  receipts from the same evaluations. `analyze_confirm.py:73-75` reports base
  PB numerical failures, sensitivity failures, and V1 declared fallback
  families separately. The V1 counter uses the actual frozen status spelling
  `COMPUTED`; it is not a claim to count every pattern fallback or gamma-zero
  direction.
- `confirm.py:63-64` includes the already implemented `BB_BY` reference.
  `protocol.json:16-17` and frozen `THEORY.md:156-158` correctly retain
  `BB_eBH` as EMPIRICAL_ONLY. Marginal corrected-PC validity justifies fixed BY
  under M0, not the TRAIN-selected e-calibrator. Neither comparator statement
  supplies an unqualified guarantee for calibration drift.
- Frozen `predictive_bridge.py` and `grid_calibration.py` hashes are unchanged
  from the accepted preflight snapshots. The prior Delta proof disposition
  therefore stands: Delta=5 covers ratios 2.2 and 4.6 under the explicit bound
  and all other assumptions; Delta=1 or 2 does not cover these boundary cases.
  No new power, novelty, clinical, or universal floating-point validity claim
  is licensed. `PROVED_BUT_COSTLY` remains the stated diagnosis, not a promised
  outcome of C001.

## Boundary of this sign-off

The checker now verifies metric/decision rosters and core evidence finiteness;
it is not a complete semantic validator of every NPZ key or an independent
input-regeneration/e-BH replay. I do not label it as such. Final outcome,
numerical-failure, saved-array, and replay verification remain main's final
integration work. No such checks or interim outcome scan were performed here,
and their pending status is not a reason to alter this valid frozen job.

Key repaired-source SHA256 values:

```text
C001/provenance.py
8588dcccf76c01fbcd1229612c9f392a48d8ef8c88d77c302751e1b52a5a635b
C001/confirm.py
60e4862d4ef2950e097ac4b9098fb62827e7f1d2abeb8b56ac6004b6d78c8337
C001/analyze_confirm.py
a1a8420f1ab02c7b4dc6911971cf9a2ba9163ebd3a715af920f4d312412ff8d8
C001/protocol.json
34087d1d9752a905336f6b38a78be304ccd0f800bd1325a8363385629b1ebad0
```

## Addendum: bounded analytic-efficiency proof check

2026-09-17. Reviewed only `EFFICIENCY_BOUNDARY.md` and
`CALIBRATION_EFFICIENCY.md`, with the existing e-BH threshold convention checked
in the immutable source. No C001 outcomes, new experiments, fits, tests, or
algorithm changes were used. **Both analytic deductions pass; no mathematical
blocker found.** The prior frozen-audit disposition is unchanged.

### Finite-reference cap and sparse power

`EFFICIENCY_BOUNDARY.md:7-27` is correct under the stated grid calibration,
convex mixing, and model FDR guarantee. For a decreasing normalized grid table,

    t1 e(t1) <= sum_k (t_k - t_(k-1)) e(t_k) <= 1.

Hence every table value is at most 1/t1, including after PILOT selection. For
the specified n=4096, the projection cap is 4096 and ordinary cap is 4096/3;
their convex mixture cannot exceed 4096. e-BH self-consistency therefore gives
`m/(qR) <= 4096` whenever R>0. With m=512 and q=.05 this implies R>=3; pure
ordinary requires R>=8.

For a fixed number s of true signed alternatives with 0<s<k_min, every
nonempty rejection set satisfies

    FDP = 1 - TP/R >= 1 - s/R >= 1 - s/k_min.

Taking expectations yields
`FDR >= (1-s/k_min) Pr(R>0)`, while `E[TP/s] <= Pr(R>0)`.
The stated power bound follows. In particular, s=1 gives .075 for the mixed
cap and .05/(1-1/8)=.057142857... for ordinary. This uses an actual model FDR
guarantee, not an observed FDR estimate, and does not apply those sparse numbers
to scenes with roughly 51 alternatives or to unprotected drift.

`EFFICIENCY_BOUNDARY.md:30-41` correctly states necessary, not sufficient,
resolution conditions and the rho-invariance of *relative* calibration error
under the matched nonsingular model. Two minor precision notes: replace
"must exceed" at line 11 with "must be at least" (the executable uses >=);
and 10240/30720 are algebraic necessary thresholds, not allowed dyadic interface
sizes. Under the existing power-of-two constraint the first eligible sizes
would be 16384/32768. This is an interpretation of the bound, not a proposal to
change either the interface or C001.

### Even-median tails and the N=4 predictive reference

`CALIBRATION_EFFICIENCY.md:7-31` is correct. Differentiating the stated CDF
gives `f(x)=8x/(1+2x)^3`; its upper survival is asymptotic to 1/x and its lower
CDF to 4x^2. Put U=X_(m+1) for N=2m. With the averaged numerical median,

    U/(2c) <= B_N <= U/c.

For large t, U>t requires at least m exceedances, giving upper survival
asymptotic to `choose(2m,m) t^(-m)`. For small epsilon, U<=epsilon requires
at least m+1 small observations, giving lower probability of order
`epsilon^(2m+2)`. The same two median inequalities sandwich both B_N rates;
no assumption that the even median equals a single order statistic is needed.

At N=4 these rates are t^-2 above and epsilon^6 below. Tail integration gives
finite E[B_4] but infinite E[B_4^2], hence infinite variance; for even N>=6
the second moment is finite. This does not prevent bounded FDP/power
family-level intervals or invalidate the predictive construction.

`CALIBRATION_EFFICIENCY.md:35-47` also follows. Since V_H is positive finite
almost surely and Z has a positive probability of exceeding 1, some finite
positive K has `a=Pr(Z>1,V_H<=K)>0`. B is independent of that event. Thus

    Pr(W>t) >= a Pr(B < 1/(K t^2)) >= C t^-12

for a constant C>0 and all sufficiently large t. This is a **lower bound on
the positive/right-tail survival**, not an assertion that 12 is the exact tail
index. It rules out Student20's t^-20 tail order for this N=4 reference far
enough into the tail. It does not establish the size of a power loss at a
specific operating threshold, a universal calibration impossibility, or an
outcome for C001.

### Large-N median CLT

`CALIBRATION_EFFICIENCY.md:49-69` is correct. At
`c=(1+sqrt(2))/2`, the density satisfies `c f(c)=1/(2+sqrt(2))`. Consequently,

    1/(4 c^2 f(c)^2) = ((2+sqrt(2))/2)^2
                     = (1+1/sqrt(2))^2.

The standard median CLT applies because f is continuous and positive at c;
averaging the two central order statistics changes neither the sqrt(N) limit
nor its variance constant. The approximately 1.7071/sqrt(N) coefficient is an
asymptotic relative standard-error description, not finite-N coverage or an
RMSE formula usable at N=4. Rho-invariance holds for the ideal pivotal law,
not arbitrary mismatch or numerical behavior at a singular boundary.

All these arguments are classical deductions, not novelty claims. Read
"simulated exactly" in `CALIBRATION_EFFICIENCY.md:22` as sampling the specified
finite-N law in the ideal-randomness construction; preferably say that
explicitly. It must not override `NUMERICAL_SCOPE.md` or imply bit-exact
continuous randomness/linear algebra in the executable. These are wording
qualifications only, not reasons to change the frozen analysis.

Reviewed addendum SHA256 values:

```text
EFFICIENCY_BOUNDARY.md
821d81dd9a9f3c39c97da734016278d3e0b67410a1b673e7ecbb8c0ac8a307b8
CALIBRATION_EFFICIENCY.md
280fc32194f682b21c2bc0202f476ee36a2007c7fea8f93ec587328b98963b39
```
