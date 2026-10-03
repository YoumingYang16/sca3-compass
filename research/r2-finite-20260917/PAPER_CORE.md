# Finite-calibration directional replicability with full-contrast predictive evidence

Research core, not a submission-ready or originality-certified manuscript.
Current manuscript status: methods/theory checked internally; one frozen
confirmation complete, PROVED_BUT_COSTLY under the model. Originality remains
unestablished. All source references in CLOSEST_WORK_AND_NOVELTY.md.

## Scientific question and methods

Multiple correlated analysis pipelines are not independent replication studies.
We test, for each gene and sign, whether at least two of four studies have a
positive signed location, using six common-location pipelines per study.
The input is a gene-by-study-by-pipeline tensor and independent zero-location
calibration blocks. Target blocks are independent Gaussian matrix scale
mixtures, with common unknown study shape and compound pipeline shape.
Positive gene-block radii need no specified parametric family or finite variance;
finite first moments are required to retain the directional-mean interpretation.

Orthogonal pipeline contrasts remove the common signal. A matrix calibration
ratio has an F4,2 law after dividing by the common/contrast scale ratio kappa.
Its median yields an equivariant estimator with a known finite sampling law.
A deterministic finite number of affine-equivariant shape updates on independent
TRAIN contrasts retains a pivotal whitened eigenvalue-ratio law. This avoids
treating fitted study shape as known or asserting a finite bootstrap is exact.

A positive-tail Rayleigh bound combines calibration median error, shape
uncertainty, four chi-square5 contrast energies and a standard normal into a
dominating reference distribution. Each Monte Carlo reference draws the ENTIRE
tuple independently. Conservative plus-one ranks calibrate learned nonnegative
intersection directions. Maxima over all three-study intersections yield
directional partial-conjunction p-values, including partial and mixed-sign nulls.

Four equal folds separate TEST, PILOT and two-fold TRAIN. Only PILOT chooses
the evidence calibrators and mixture weights; it cannot access the shared
calibration, reference bank, inference TRAIN or target scores. This separation
is necessary because the predictive p-values are marginal over TRAIN error.
On their finite rank support, the original decreasing calibrators are normalized
by the exact support-based expectation bound. e-BH is applied to the full2G
signed family at.05. This is a use of established calibration/PC/e-BH methods,
not a claim to invent those components.

## Theoretical results

Under M0 and ideal arithmetic/independent randomization, the procedure controls
FDR at q for unknown study shape, unknown kappa and unspecified common-block
radial laws. The proof treats inference direction learning and PILOT evidence
learning separately, integrates each claim over its own permitted field, and
does not condition on the union of folds. Full proofs are THEORY Props1–6.

If an external bound kappa_target/kappa_cal<=Delta exists, replacing kappa_hat
by Delta*kappa_hat in the squared denominator preserves positive-tail
domination and final control. The statistic's denominator increases by
sqrt(Delta); learning and the reference distribution remain unchanged.
This is a sensitivity theorem, not an estimate of migration. Calibration and
target contrasts alone do not uniformly identify such a bound: their joint
law can be held fixed as target kappa diverges. This restricted-information
example does not rule out identification from target means, controls or
additional designs (THEORY Prop7).

## Experimental design

D001 first tested a valid confidence-envelope method on768 developer families;
its utility failed. D002 used256 stored inputs to test the predictive bridge.
D003 reused exactly those rank scores to isolate discrete-calibrator recovery.
Those stages are development evidence and are not pooled with confirmation.

C001 is specified before new observations in protocol-confirm.json: one frozen
confirmation,15scenarios×512whole independent families. Thirteen are inside M0,
including normal,t3,t5,t1.5,small calibration,unseen effect,continuous/dense
alternatives,global/singleton/mixed-sign PC nulls; two are separate drift
sensitivities. All methods see identical inputs and no test truth is available
to inference. Sources/configuration and analysis rules are hashed before run.
No optional sample expansion or reserve confirmation is registered.

Signed FDP and Power are bounded family-level metrics; all-null Power is
undefined. There are338 reported two-sided finite Chernoff-KL intervals,
within a predeclared cap of512 two-sided intervals; each tail uses.05/(2*512).
Paired differences are computed within whole data
families, never across genes. Computation failure and model mismatch are
reported separately. Model guarantees are not inferred from FDR point estimates.

## Results available before confirmation

The confidence-envelope D001 method had zero Power in normal and t5N4/12/32
scenarios, and2.39% at t3N12. Its inference-only known-nuisance references
showed substantial detectable signal. This rejects its usefulness in those
tested settings, not its conditional mathematical guarantee.

In the fixed D002/D003 subset, continuous-to-grid Power increased from13.30%
to27.21% (normal),8.09% to19.12% (t5N12),31.74% to46.69% (t3N12).
Corresponding practical strong references were75.98%,51.04%,72.24%.
These are development means, not independent proof of a stable improvement
or a new methodological principle. They identify real conservatism remaining
after the exact nuisance/learning correction. Confirmed numbers belong to
the separately generated C001 result tables; none is invented here.

## Independent frozen results (C001; separate from development)

The fixed7680families completed with no numerical failures. All stored metrics
were rescored from saved truth/decisions;32768saved-evidence e-BH decisions
matched; four specified inputs regenerated and all declared methods replayed
exactly. Source: C001/index.json,summary.json; confirmation-replay.json and
saved-decision-audit.json. These checks are not additional independent data.

Across13M0scenes, main FDR estimates were at most0.390625%, with maximum
predeclared simultaneous upper bound3.101271%. These empirical intervals do
not extend the mathematical model or establish a machine-arithmetic theorem.
Power was25.69% normal,13.58% at t5/N4,22.65% at t5/N12,36.30% at t5/N128,
and45.96% at t3/N12; corresponding practical strong-reference powers were
74.00%,52.20%,51.78%,52.32%,72.43%. The normal, t5/N12 and t3/N12 paired
deficits were48.31,29.12 and26.47percentage points, with simultaneous intervals
[-64.18,-29.98],[-47.05,-9.72],[-44.61,-6.98]for main minus strong.
The complete matrix including both simple comparators is figures/RESULTS.md;
exact comparator definitions and guarantee qualifications are COMPARATORS.md.

Grid normalization versus the identical continuous-calibration ablation gained
6.69–24.12pp in the10non-null M0scenes; every prespecified simultaneous lower
bound was positive. This demonstrates an efficiency gain from an established
calibration principle, not a new general principle. Additional projection/
mixture advantages over the ordinary grid component are not established by
the conservative simultaneous intervals; continuous/dense effects nearly
reproduce ordinary power. See mechanism-summary.json for descriptive gates,
not independent fold-level samples or causal attribution.

Unprotected severe drift gave FDR16.60% [10.17,24.73], a clear outside-M0
failure. Delta5 gave0.153% FDR and25.17% Power for the true ratio4.6, but only
1.81% Power in the less severe ratio2.2 scenario. These fixed sensitivity
results illustrate the cost of externally bounded transfer, not a learned
biological drift guarantee. Delta2 is insufficient for both true ratios even
when an empirical FDR point happens to lie below5%.

The reference-count and calibration-median deductions in EFFICIENCY_BOUNDARY.md
and CALIBRATION_EFFICIENCY.md explain two concrete efficiency constraints.
They are explicitly classical consequences, not priority claims. Total batch
wall time2743.32seconds, six1-BLAS-thread workers; main per-family mean wall
times0.89–1.59seconds within M0. Runtime and intervals by scene are provided,
with all source/raw hashes. No clinical data or true biological discoveries
were introduced.

## Final limitations and contribution level

The supported tensor model is substantially narrower than arbitrary expression
data. Independent genes, matched centered calibration and common pipeline
locations are substantive assumptions, not properties certified by input-shape
checks. No actual patient analysis is included. GSE320100 remains sealed.

The spectral reference is dominating but not proved sharp for nonnegative
triple directions. PILOT splits, finite-reference resolution and p-to-e/PC
conversion retain efficiency costs. The executable uses float64 matrix algebra
and deterministic PRNGs; exact table/quantile subcalculations do not provide
a universal machine-arithmetic FDR certificate. Independent internal review
is not journal peer review. Task-specific synthesis and explicit boundaries
are documented; worldwide originality and high-impact publication prospects
are unresolved, and no clinical or journal acceptance promise is made.
