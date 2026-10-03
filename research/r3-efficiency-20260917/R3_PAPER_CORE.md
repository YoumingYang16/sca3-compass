# Paper core: finite-calibration efficiency through independent nuisance learning

Working scientific material, not a submission-ready claim of originality or
publication qualification. VersionR3-GSR-1.0.0; companion source/theory fixed in
C001. Numerical Results below come from the completed single registered
confirmation and byte-identical fresh-process re-analysis, not DEV selection.

## Introduction / scoped contribution paragraph

Replication testing with a small external calibration sample has two distinct
requirements: controlling errors for the full learned procedure and retaining
enough sensitivity to make that guarantee useful. In a matrix Gaussian-scale-
mixture model, the existing safe procedure integrates finite calibration error
but bounds unknown study-shape effects uniformly over directions. We examine
whether this protection is unnecessarily expensive. Our construction separates
direction learning from inference-shape learning and uses their independence
to derive a nuisance-free joint orientation/spectrum reference. It also replaces
the median calibration pivot summary with a log-centered geometric summary whose
finite sampling law is explicitly integrated. These substitutions preserve the
scientific partial-conjunction null and the complete pilot-selection-to-FDR
chain. A paired ablation design quantifies the gain and its cost. We do not
claim invention of invariance, sample splitting, equivariant estimation, rank
tests or e-values; whether this particular construction adds sufficient content
to the closest adaptive-detection literature remains a separate priority check.

## Methods

### Data and estimand

For g=1,...,G independent hypotheses, observe a4x6 matrix with a study-specific
location shared across six pipelines. Noise has separable study shape R and
equicorrelated pipeline shape C(rho), with an arbitrary independent positive
radial multiplier shared within a matrix. An independent sample of N centered
calibration matrices shares the pipeline common/contrast noise ratio
kappa=(1+5rho)/(6(1-rho)). Invertible study transforms and radial variation in
the calibration blocks are eliminated by the matrix pivot; target study shape
is shared. This is a restrictive research-input model, not an assumption
automatically satisfied by clinical data or correlated transcriptomic genes.

For each sign, the null asserts that fewer than two of four study effects have
that sign. We test2G signed partial-conjunction claims at nominal FDR.05.
The null includes strong singleton effects and mixed-sign configurations, not
merely all-zero means. Mean interpretation requires a finite first radial
moment; variance need not exist. The software takes only observed matrices,
fixed randomization and any externally justified mismatch bound.

### Finite calibration and independent shape orientation

Let x_l be the pipeline mean and Y_l the five orthogonal pipeline contrasts.
The dimension-specific pivot A_l=.5 x_l'(Y_lY_l')^-1 x_l satisfies
A_l/kappa~F4,2 independently. We use
kappa_g=exp[N^-1 sum log A_l]/(e/2), retaining the old median only in learning.
The ratio B_g=kappa_g/kappa has an exactly simulable parameter-free finite law.
No normal approximation to a kappa confidence interval or fitted tail law is
used. Matrix operations are rescaled; invalid numerical output is reported
explicitly and the recognized failure path returns no discoveries.
The common centering constant cancels in exact observed/reference ranks; any
efficiency change must come from the summary's sampling law and the independent
orientation construction, not from the arbitrary numerical scale convention.

Four cyclic roles allocate each hypothesis to TEST, PILOT, DIR and SHAPE.
The original signal profiles/nonnegative directions are learned on DIR using
DIR's own shape. The inference shape H comes only from SHAPE contrasts.
PILOT keeps the established component/calibrator selection procedure and sees
none of external calibration, DIR, SHAPE or TEST. No new projection family is
introduced. This reduces each direction/shape learning sample fromG/2toG/4.

Whitening and orthogonal invariance yield the null reference

    W = Z / sqrt[B_g H_*11 (sum_j U_j/lambda_j(H_*))/20],

where Z is standard normal, U_j are independent chi-square5, and H_* is the
fixed2step shape fit on G/4 independent Gaussian contrast blocks. Its diagonal
element and eigenvalues must come from the SAME matrix. The geometric error
factor is drawn from N independent F4,2 pivots. Whole reference tuples are iid
and independent of observed data. In contrast, the original safe reference
substitutes lambda_min(H_*) for a direction's Rayleigh quotient. R3 removes
that worst-direction step by an independence argument, not a plug-in estimate.

### Complete testing pipeline

For positive signed statistics use conservative plus-one ranks among4095
independent references; assign p1 to nonpositive statistics. A triple ordinary
test uses Bonferroni over its three study coordinates, while a triple projected
test uses a nonnegative learned direction. Taking the maximum over four triples
handles the signed two-of-four partial-conjunction null. The independent PILOT
selects decreasing calibration functions and mixture weights. Existing exact
rank-support normalization converts valid p-values to conditional e-values;
their PILOT-measurable convex mixture is passed to e-BH over all2G claims.
Reference sharing and cyclic roles do not require independence at that last
step. They must not be misrepresented as conditional independence given the
union of every role's data.

## Theoretical results

1. **Finite nuisance elimination.** Conditional on DIR and external calibration,
the normalized direction is fixed independently of the inference shape.
Its joint Rayleigh-quotient/spectrum law is the H_*11/spectrum law by rotation.
Integrating the actual finite calibration error gives the reference above.
Nonpositive null locations yield positive-tail domination. This integration
does not assert validity conditional on a fixed realized calibration error.

2. **Full FDR chain.** Conditional on the corresponding PILOT, each null PC
p-value is superuniform. Discrete-support calibration and PILOT-only mixing
give conditional e-expectation at most1. Integrating each claim's own PILOT
field and applying the pointwise e-BH bound yields FDR<=.05. This is an
ideal-arithmetic finite-sample model guarantee, not a statement for arbitrary
dependence, uncentered calibration, every drift distribution, or all floating
point outcomes. No new general e-BH theorem is asserted.

3. **Calibration mechanism.** For -2N<p<N,
E B_g^p=exp(-p)[Gamma(2+p/N)Gamma(1-p/N)]^N; the endpoints diverge.
Var(log B_g)=(pi²/3-1)/N exactly. AtN4 the known-shape Student20/sqrt(B_g)
reference has finite nonnegative absolute moments of order below16, compared
with12 for the median summary. This is not the full unknown-shape tail index
or a theorem of greater detection Power.

4. **A restricted necessary cost.** In the pivot-only scale experiment, the
log-location score has informationN/2. Regular unbiased log-scale estimators
therefore have variance at least2/N. The geometric estimator's exact variance
is1.144934times that bound. This classical information inequality is not an
attainable finite-N optimum, a full-data minimax result, or a lower bound on
the remaining Power gap to a plug-in strong method.

Proofs: frozenC001/R3_THEORY.md, C001/CALIBRATION_MOMENTS.md, and independently
reviewed analytic supplement SCALE_INFORMATION_BOUND.md. Numerical identity,
correctness tests, simulation evidence and novelty assessment are separate.

## Experimental design

The loss ledger reused240existing families in ten scenes, substituting true
kappa, true shape, or both while fixing learned components and evaluation.
These are DIAGNOSTIC_ONLY. The geometric/independent-shape candidate and a
single median ablation used312old families in13scenes; these two executions
are not624 independent observations. All original outputs were checked for
exact equality after the ablation was added. None of this is R3 confirmation.

We then froze one candidate, analysis code, source dependencies and protocol.
R3-C001 comprises15predeclared scenes with1024new independent G256 families
each. It includes normal,t5,t3,t1.5, calibration sizes4/12/32/128, continuous
and dense effects, global/singleton/mixed-sign PC nulls, an off-grid amplitude
and two separately reported out-of-model drifts. Scenario labels inherited
from earlier work do not make those scenarios unseen during R3 development;
the confirmation observations are new, not the parameter settings.

The unchanged generator uses radial sqrt((nu-2)/chi-square_nu) for nu>2,
so the normal/t5/t3 noises have unit marginal variance. For t1.5 it uses
sqrt(1.5/chi-square_1.5), a scatter convention with nonexistent variance,
NOT unit variance. Other scenario parameters also differ across rows; the
tables establish paired method comparisons within each scene, not a pure
causal effect of tail thickness. Source: frozen V1 scripts/robustness_screen.py
radial(), reached through frozen R2 C001/experiment.py generate().

R2, its ordinary component, frozen strong reference and V1 K-NR are evaluated
on the same inputs. R3's scale-only/median-only2x2 components and ordinary
control retain their actual input information. No true nuisance enters a
deployable comparison and no threshold is adjusted to a measured5%FDR.
The strong reference uses fitted radial working assumptions and plug-in
nuisances; it lacks R2/R3's corresponding complete finite-calibration proof.

Power and FDP are computed per entire signed family.289registered two-sided
bounded-mean betting intervals use a384comparison budget at totalalpha.025,
giving at least97.5%simultaneous coverage under independent families. Paired
differences use the same family's methods. Neither genes nor folds count as
replications. A six-scene equal-weight core gain, all13M0FDR uppers, and all
ten nonnull M0 regression guards are separately checked. The2pp regression
margin is not an improvement or publication threshold. There is one formal
confirmation, no reserve, no optional sample extension and no seed reroll.

## Results

All15360planned independent families completed, with zero R3/R2 numerical
failures. The maximum R3 M0 FDR point was.488%, and the maximum simultaneous
upper endpoint2.470%, both in the singleton null scene. All13M0FDR gates
passed. The six-scene equal-weight Power rose from28.636%to35.530%, a paired
gain6.894pp[5.461,8.201], corresponding to3.516additional correct signed
discoveries per core family. Every one of the ten nonnull M0 paired intervals
had a positive lower endpoint; hence the predeclared regression guards passed.
For t5/N12 the gain was5.946pp[3.503,8.447]; for t3/N12,6.838pp[4.644,9.134];
for infinite-variance t1.5,2.734pp[1.079,4.393]. Absolute Power in t1.5 remained
only9.773%. These intervals belong to the fixed joint97.5%comparison family.

The core2x2 means were28.636%(old scale/architecture),29.960%(geometric only),
33.963%(independent orientation with median),and35.530%(full). Main-minus-
scale-only intervals were positive in all10nonnull M0scenes, supporting the
independent architecture's contribution. Main-minus-median-orientation had a
positive lower endpoint only in t3/N12:2.246pp[.166,4.347]. Therefore the exact
geometric moment calculation does not establish a uniform extra Power gain.
The net architecture contrast also includes smaller learning sample sizes.

At t5N4/12/32/128, Power was16.242/26.840/36.573/42.046%; gains over R2 were
3.594/5.946/8.006/7.320pp. The improvement was not largest at the smallest N.
AtN4 only35.645%of families had any discovery. Scale-error quantiles and the
conditional-on-rejection final thresholds are reported in the calibration
curve; those thresholds need not be monotone because zero-discovery families
are excluded. The original projection mixture had supported benefit over the
ordinary component in8nonnull scenes, but not continuous/dense effects.

The strong reference remained materially more powerful in9nonnull M0scenes;
continuous effects was the exception, R3-minus-strong2.510pp[.772,4.226].
Descriptively, the core gap to strong decreased from30.409to23.514pp. Strong's
singleton FDR point5.762%had interval[2.773,10.201]%, failing an upper<=5%
check but not establishing true FDR>5%. No threshold was adjusted to observed
truth, and the strong method was neither weakened nor omitted.

Both mismatch scenes showed a cost: default R3 FDR was4.846%and20.008%, versus
R2's3.426%and16.341%. Paired increases1.420pp[.198,2.613]and3.667pp[2.271,5.001]
exclude zero. An externally justified Delta5 restored low FDR in the separate
sensitivity model but sharply reduced Power; it does not repair the default
algorithm's no-worsening failure. Consequently the full user-defined success
label is not met even though model-valid efficiency recovery is supported.

The eight-worker confirmation took approximately120.9minutes. R3 M0 bundle mean runtimes
were1.291–2.139seconds/family, including ablations. All archived R3 learner
convergence flags were true;15tests passed and five predetermined saved-input
replays matched109arrays exactly. A separate complete re-analysis generated a
byte-identical summary. These checks are reproducibility evidence, not extra
independent experiments or guarantees of floating-point/global-optimization
correctness. Exact numerical sources: C001/summary.json, R3_RESULTS.md,
figures/RESULT_TABLES.md, and the named audit receipts.

## Limitations / claims requiring restraint

The theory remains limited to independent target blocks and a shared separable
noise structure with matched finite calibration. It is not directly certified
for the project's available SCA3 datasets. No new clinical dataset was used.
Unknown study-specific target radii, general cross-gene dependence or pipeline
location mismatch can invalidate the reference. Severe covariance/calibration
drift remains a required failure boundary. ExternalDelta5 protection assumes
a known transfer bound and must not conceal deterioration of the default.

The fixed reference grid limits p-resolution; PC combination, data splitting,
component selection and e-BH retain power costs. Oracle contrasts are not an
additive causal attribution or attainable detection upper bound. No theorem
shows that every remaining strong-method gap is required by finite validity.
The algorithm returns component ablations; recorded cost includes them, not
an optimized primary-only implementation. Floating linear algebra is tested
but not fully interval-certified. Generic invariant adaptive detection is
established literature; closest-work priority and practical application remain
open even if simulated utility improves. No journal outcome is promised.
