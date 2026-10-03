# C001 narrow preflight review

Date: 2026-09-17. Sole independent reviewer. Source/proof inspection only: no
project execution, tests, fits, simulations, new agents, or edits outside this
review. Earlier V1 evidence is not re-audited or changed.

## Disposition

The discrete-support repair and the proposed externally specified Delta extension
are mathematically acceptable under the stated predictive-bridge assumptions.
No new primary-method proof blocker found. **Do not freeze/run the inspected
confirmation harness until B1-B3 below are addressed.** These are concrete
provenance/reporting-contract gaps, not evidence of contaminated observations or
an invalid primary theorem. No C001 outcome has been inspected by this reviewer.

The main's subsequent comparator qualification is correct: retain `BB_eBH` as
`EMPIRICAL_ONLY`, and add fixed `BB_BY` as the rigorous classical reference under
the matched-calibration model. This is not a change to primary PB. The inspected
source snapshot does not yet contain that additional reported label.

## Blocking preflight findings

### B1. Freeze the sources actually imported, not only the release ZIP

Locations: `confirm.py:24-32,89-94`; `experiment.py:16-24`;
`finite_calibration.py:19-26`.

The local harness files are copied/hashed, but the V1 check hashes only
`releases/K-NR-1.0.0.zip`. Execution imports the extracted release's `source/src`
and `source/scripts`, including the generator and primary PB's learning,
projection, calibrator, and e-BH helpers. Changing an extracted dependency does
not change that ZIP. Thus this is also a primary-inference/generator identity
gap, not merely a comparator-label issue.

Correction: bind the actually used extracted-source dependency closure to the
trusted release manifest/ZIP, record resolved module locations and hashes, and
verify them before workers execute and before analysis. Execute the frozen
harness copies, or otherwise verify the actual worker imports against the
freeze. Checking existing release membership/hashes is sufficient; no repeat of
the old V1 scientific audit is requested.

### B2. Carry the frozen identity through resume and analysis

Locations: `confirm.py:35-43,46-53,89-104`;
`analyze_confirm.py:26-45,68-70`.

Resume accepts a cached row after checking its freeze and optional artifact
hashes, without checking its case, repetition, expected algorithm seed, or
canonical artifact paths against the requested task. Analysis checks artifact
hashes and per-case counts/unique repetitions, but does not check the freeze,
protocol/source hashes, row-to-index identities, or that repetitions are exactly
`0..511`. It trusts stored metrics without checking them against saved truth and
decisions. For example, an altered protocol can relabel cases at analysis without
being checked against the frozen protocol digest. The duplicate-repetition check
does catch some misfiled records; it does not complete this identity chain.

Correction: require the exact planned Cartesian set of 15 x 512 case/repetition
identities; matching index/record/status/seed identities; required canonical
input/evidence paths and hashes; and frozen protocol/source identities at both
resume and analysis. Verify metric values from the already saved truth/decision
arrays, including exact expected method/array membership. Bind the summary to
the freeze/protocol as well as the index. This is deterministic evidence
verification, not another experiment. Keep hard failures analysis-blocking and
retain conservative no-discovery outcomes in all denominators.

The auxiliary seed at `confirm.py:53` compresses the tuple to one uint32. Distinct
tuples are not guaranteed distinct reference streams; no actual collision is
asserted here. Preserve the full SeedSequence entropy in an accepted integer
seed, or explicitly verify and bind the complete planned seed registry before
freeze. Sharing a seed across methods/Delta settings *within* a family is
intentional pairing; accidentally sharing auxiliary streams *between* families
is inconsistent with the independently randomized-family interval premise.

### B3. Make the interval contract agree with its valid implementation

Locations: `protocol-confirm.json:10-14`;
`analyze_confirm.py:13-23,46-68`; `confirm.py:63-74`.

The implemented Chernoff-KL calculation spends alpha/(2 x 512) per tail, i.e.
supports **512 two-sided intervals**, not 512 total one-sided endpoints. Its
counter called `endpoints` counts intervals. The current roster is:

- 216 method/metric intervals: eight methods x (15 FDP + 12 defined power).
- Eight additional Delta method/metric intervals for the two boundary cases.
- 72 paired-power intervals: six comparisons x 12 non-null scenes.
- 15 calibration-domain coverage intervals.

That is 311 intervals / 622 one-sided bounds. Adding `BB_BY` to the method table,
without adding another paired comparison, gives 338 / 676. Both fit the
implemented 512-*interval* budget. Thus this is a fixable preregistration wording
mismatch, not evidence that the implemented alpha allocation exceeds .05.

Correction: freeze the exact method/metric/comparison roster, call the cap 512
two-sided intervals, and make the analyzer read/check that cap and alpha from
the frozen protocol. Clarify that the `[0,1]` paired range applies only to
`PB_grid` minus `PB_main` **power**. The analyzer correctly uses it only there;
FDP differences are not nonnegative merely because rejection sets expand.
The stated 2.21 percentage-point worst-case MCSE is for a [0,1] mean, not a
general [-1,1] paired difference. Declare the saved-but-unreported continuous
ordinary `PB_ordinary` an auxiliary diagnostic, or explicitly add it to the
report roster and budget. No new method search is needed.

## Delta extension: accepted, with its precise scope

Locations: `predictive_bridge.py:95-126,139-153`;
`confirm.py:66-70`; `protocol-confirm.json:19,34-35`.

Write r = kappa_target/kappa_cal and B = kappa_hat/kappa_cal. With the held
denominator changed to Delta x kappa_hat, the centered directional statistic is

    sqrt(r/Delta) Z / sqrt(B r_a Q_H / D),

where r_a = b' H0 b / ||b||^2 >= lambda_min(H0). Under an intersection null,
the mean shift is nonpositive. On a positive observed statistic, the centered
numerator is positive. If r <= Delta, both inequalities therefore bound its
positive tail by the same reference

    W = Z / sqrt(B lambda_min(H0) Q_H / D).

Here the reference samples the same independent shape-estimator law, calibration
median-ratio law, Gaussian numerator, and weighted contrast-chi-square law as
in the original proof. Nonpositive statistics must still return p=1. The code
changes only the held denominator at line 115; TRAIN directions, PILOT
selection, and references do not change. TRAIN/calibration-learned directions
remain covered by the Rayleigh bound. Conditional-on-PILOT validity, ordinary
Bonferroni PC, projection PC, discrete calibration, and final e-BH follow as
before; no conditioning on the union of cyclic training sets is required.

For calibration rho=.8, target rho=.9 and .95 give ratios 2.2 and 4.6. Neither
Delta=1 nor Delta=2 covers either case; Delta=5 covers both **under the explicit
extended bound and all remaining model assumptions**. These cases remain
outside original matched M0. The bound is externally supplied, not learned from
these outcomes, and does not establish a bound for biological data. Selecting a
favorable Delta after inspection is not part of this fixed procedure.

Before freezing, retain Delta-specific p/e arrays and failure/error receipts from
the evaluations already performed in `confirm.py:68`. Currently only their
decisions plus status/time are saved; `analyze_confirm.py:65` counts only base-PB
numerical failures. Do not report that count as covering all sensitivities or
comparators. Saving existing outputs and reporting per-method statuses requires
no additional fits. These are evidence/reporting safeguards, not a new Delta
proof condition.

## Verified nonblockers and qualifications

- `grid_calibration.py:14-53` now enforces the exact dyadic rank support and
  ordinary three-rank support including p=1, checks a finite nonnegative
  decreasing table, accounts for all integer mass, sums represented heights
  with exact fractions, rounds the denominator up and each distinct quotient
  down, and uses table lookup. This addresses the previous narrow grid findings.
  It is known discrete-support calibration, not a new general theorem. It does
  not certify upstream matrix arithmetic or final gamma-mixture rounding.
- `experiment.py:49-73` uses separate whole-family seed tuples, scores the signed
  2G hypotheses, and sends no truth/true-shape oracle to PB. The three null-only
  scenes have undefined power, not zero power. The generator's t1.5 radial law
  has finite target-statistic mean and infinite variance; its scale is not a
  claim of unit variance. Independent row-radial calibration blocks remain
  covered by the matrix calibration pivot's invertible-left-factor cancellation;
  this does not relax the common-block target radial assumption.
- The fixed 15 x 512 plan is 7,680 new whole families, with no reserve or
  significance-based extension. Same-family method/sensitivity comparisons are
  paired. Old DEV observations are not part of these interval denominators.
- Main's BB qualification is correct. At `finite_calibration.py:278-282`, adding
  the bad-event probability gives a marginally valid corrected PC p under M0;
  it does not make that p conditionally valid given TRAIN for the subsequently
  TRAIN-selected calibrator/gamma. Thus `BB_eBH` is not established valid.
  Applying BY directly to the corrected PC p (`:298`) is valid under arbitrary
  dependence, under the stated M0 assumptions. Label drift outputs descriptive
  unless separately covered. `R2_main`'s restricted-good-event q adjustment is
  a different, retained proof, not a justification for `BB_eBH`.
- Keep `PROVED_BUT_COSTLY` as the primary diagnosis unless the fixed evidence
  warrants a more specific utility description. No promise of a power win,
  novelty, biological applicability, or universal float64 FDR certification
  follows from this preflight review. The ideal-randomness/exact-arithmetic
  versus executable-numerics distinction in `NUMERICAL_SCOPE.md` remains needed.

## Reviewed source snapshot

Paths above are relative to `research/r2-finite-20260917`. Source changes were
observed during review; this disposition binds the following final inspected
snapshots, not subsequent unseen repairs:

```text
predictive_bridge.py
4BDA60E9B0E0A64BFDC863476910C03CF3004D2DC01E0A58128A098EF63AF823
grid_calibration.py
D90037A8DA6E8892A34E7A55761256E949BF6019EDEEEA64F88B28611DA68383
confirm.py
074E3F651EA6F0B7E189B9C71378F23E8A1331C08AE6BD0D4D917AC2C9FEFF97
protocol-confirm.json
0FA4A8C08604E10E9A64727453A335C6D8E16DA2F9CF1583177329FACBB4EED0
analyze_confirm.py
75254CB15E941922F140F1A28BA7B5DF0196AA7CA5EF991D1E7992CFD4F6AC26
experiment.py
5FD641C51967E14FF51538E435E910B68401726B7A2033E6188F61A8EDE453DD
```
