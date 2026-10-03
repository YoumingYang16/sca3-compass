# R2-PB-1.0.0 — finite-calibration research implementation

Independent R2 workspace. This does not replace, modify or invalidate scoped
EMPIRICAL_ONLY K-NR-1.0.0. Primary output is `PB_grid`, not old `PB_main`
(the continuous-calibrator ablation). Research status is reported in
RESEARCH_FINDINGS_ZH.md: PROVED_BUT_COSTLY after completed confirmation; do not infer success
from this version string. Model/theorem and numerical qualifications differ.

## Input contract

- `z`: finite `(G,4,6)` real array, G>=16 divisible by4. Each gene is an
  independent Gaussian-angular matrix block with a single positive random
  radial factor, common unknown SPD study shape, and compound-symmetric
  pipeline shape. Each study's six pipelines target the SAME location.
- `cal`: finite `(N,4,6)`, N>=4 independent CENTERED calibration blocks,
  independent of all target genes. Same pipeline kappa as targets, unless
  an independently justified upper ratio `mismatch_bound` is supplied.
  Independent invertible left transforms per calibration block are permitted
  if independent of its Gaussian matrix. N counts blocks, NOT4N study rows.
- Mean interpretation requires finite E(sqrt(V)); variance is not required.
  Arbitrary gene dependence, study-specific target radii, nonzero calibration
  locations, unbounded migration and heterogeneous pipeline means are not
  covered. Array validation cannot establish these scientific assumptions.
- Caller fixes the randomization seed before observing outcomes, draws once,
  and records it. Replay is allowed; choosing the best of rerolled seeds is not.
- Default4095 auxiliary references, fixed2 shape iterations and q=.05 are the
  frozen design. Other permitted counts are not the confirmed configuration.

## Minimal call (project virtual environment)

From the project root, put the frozen `research/r2-finite-20260917/C001` on `sys.path`:

```python
import sys
sys.path.insert(0, 'research/r2-finite-20260917/C001')
from predictive_bridge import evaluate
result = evaluate(z, cal, seed=12345)
result['status']  # completed, or conservative_numerical_failure with cause
discoveries = result['decisions']['PB_grid']  # (G,2): positive,negative
```

Numerical failure produces no discoveries for the entire family and must be
retained in denominators. Invalid caller inputs raise. Failure receipts are
not successful analyses. Raw evidence, reference bank, fold profiles, PILOT
calibration choices, gamma and timing are returned for audit.

## Algorithm

1. Estimate kappa by the median of independent matrix calibration pivots;
   simulate its EXACT finite sampling law, not a fitted bootstrap.
2. Generate fresh iid dominating reference tuples from calibration median
   error and fixed-iteration affine shape uncertainty.
3. For each deterministic fold, independently assign TEST/PILOT/TRAIN.
   TRAIN learns study shape and nonnegative triple directions. PILOT alone
   chooses mixture weights and decreasing calibrators, without external cal.
4. Form full20-contrast standardized ordinary/projection intersection tests.
   Use +1 conservative ranks; max all four triples for signed r=2 PC.
5. Normalize each PILOT calibrator on the actual supported rank grid and mix.
6. Apply e-BH to all2G signed claims at.05. No own-contrast evidence weighting.

All proof dependencies and error accounting: THEORY.md Propositions1–7.
No universal finite-precision certificate; see NUMERICAL_SCOPE.md.

## Reproduction

Commands actually run are recorded in RESEARCH_STATE.md / experiment manifests.
Do not rerun into existing immutable result directories.

This exact CLI was executed on one frozen **simulated** input. The retained
receipt is actual-invocation.json (17 simulated rejections, not discoveries
in patient data). For a local replay use a NEW output path; the wrapper
refuses to overwrite the existing receipt.

```powershell
.venv/Scripts/python.exe research/r2-finite-20260917/run_model.py --input research/r2-finite-20260917/C001/raw/case-00/rep-00000-input.npz --output research/r2-finite-20260917/actual-invocation.json --seed 112012381454337 --acknowledge-model
```

The wrapper loads only z/calibration, never simulation truth or true nuisance
parameters. Scientific input assumptions must be justified by the caller;
acknowledgement and array checks do not establish them. COMPARATORS.md defines
the reported reference methods without mixing their guarantee levels.

```powershell
.venv/Scripts/python.exe -m pytest research/r2-finite-20260917/test_finite_calibration.py research/r2-finite-20260917/test_predictive_bridge.py research/r2-finite-20260917/test_grid_calibration.py -q
.venv/Scripts/python.exe research/r2-finite-20260917/confirm.py freeze --protocol research/r2-finite-20260917/protocol-confirm.json --out research/r2-finite-20260917/C001_REPLAY
.venv/Scripts/python.exe research/r2-finite-20260917/C001_REPLAY/confirm.py run --out research/r2-finite-20260917/C001_REPLAY
.venv/Scripts/python.exe research/r2-finite-20260917/C001_REPLAY/analyze_confirm.py --out research/r2-finite-20260917/C001_REPLAY
```

Repeating the same frozen seed is reproducibility, NOT another confirmation.
No clinical data required or supplied. The project runtime and immutable V1
source are dependencies; environment versions are in numerical-audit.json.
The whole-family generator is frozen V1's simulation interface, not an oracle
input to the deployed method. Truth is used only to score decisions.

## Cost and limitations

Dominant cost: M independent Gaussian TRAIN shape fits, O(M*K*n_train*L*S²)
plus small S³ solves; references processed in batches64, memory
O(64*n_train*S*L + M + G*S*J). This is much costlier than V1. Full actual
runtime belongs to C001, not a theoretical complexity claim. Shared reference
bank inside one family is allowed; reusing a fixed bank across all future
analyses with a conditional guarantee is not established.

Primary value is a complete finite-calibration inferential chain in a narrow
model, with explicit efficiency costs. No global originality, optimality,
real SCA3 discovery, clinical usefulness or journal acceptance is claimed.
