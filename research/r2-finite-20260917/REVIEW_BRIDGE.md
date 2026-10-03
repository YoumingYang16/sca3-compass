# Independent review: one predictive bridge after D001

Date: 2026-09-17. This is a source/proof review of the single proposed bridge,
not another algorithm search or a new confirmation. No agents, tests, fits,
simulations or data rescoring were run by this reviewer. Only this review file
was written; D001, V1 and the earlier review were left unchanged.

## Verdict and reviewed versions

**Accept the exact-arithmetic domination, rank calibration and separate-PILOT
composition under the stated independent-block model.** Dependence of a on
TRAIN shape and calibration is handled by the uniform Rayleigh bound. The
fourfold PILOT separation addresses the adaptive-calibrator conditioning gap;
it is essential, not just a convenience. Sharing a fresh reference bank across
claims/folds is permissible under the conditions below.

No conceptual coupling or partial-null blocker was found in the inspected
implementation. This is not a finite-precision FDR certificate, a guarantee
that power will recover, or an originality claim. Concrete implementation
qualifications appear in section 6.

SHA256 snapshots:

- `PREDICTIVE_BRIDGE.md`:
  `c21a600cc0a730856837c9f41042f1eecbb8f305965e07910902549e768fd8fa`.
- `predictive_bridge.py`:
  `782ee59ce34b6bcd94b7dcc4771bac19bc321e1596583e2357be61213f33ab82`.
- Imported `finite_calibration.py`:
  `7685dff1adfe2feb6dbe67dbd2f0f7269f63e7cff418fa5cfd325f9925f049d8`.
- `test_predictive_bridge.py`:
  `9656534a23532f210daebbb4d26bc8f35d48e6fa5a10db0160ae5b36855c4c1f`.

The fixtures were read, not executed. D001's reported counts, timing and power
are background motivating this development change, not independently rescored
evidence in this review.

## 1. Domination even with TRAIN/calibration-selected directions

Write c=1-rho, v=[1+(J-1)rho]/J, kappa=v/c, L=J-1, D=S*L. For a held target,

    M = mu + sqrt(V*v) R^(1/2) xi,
    Y = sqrt(V*c) R^(1/2) G,

where xi and all entries of G are independent standard Gaussians, independent
of V and of this claim's TRAIN/calibration information. Let H be the shape
fit on TRAIN contrasts only, H0=R^(-1/2) H R^(-1/2), b=R^(1/2)a, and
B_kappa=kappa_hat/kappa. Conditional on TRAIN and calibration, a is fixed.
For a nonzero a, define

    r_a = b'H0 b / ||b||^2,
    Q_H = tr(H0^-1 GG'),
    Z_a = b'xi / ||b||.

Then the statistic in the proposal is

    T = [Z_a + a'mu/(sqrt(V*v)*||b||)]
        / sqrt(B_kappa*r_a*Q_H/D).

For a nonnegative direction supported on an all-null signed intersection,
apply this formula to the signed observations: a'mu<=0. On the event T>0,
Z_a>0, and r_a>=lambda_min(H0) implies the pointwise inequality

    T <= W_0 := Z_a / sqrt(B_kappa*lambda_min(H0)*Q_H/D).

The direction may depend on H, on TRAIN means, and on kappa_hat. This does not
invalidate the argument: conditional on all of that information, Z_a is still
standard normal and independent of G. Its conditional law does not depend on
the selected direction, so Z_a is also independent of the TRAIN/calibration
variables after integrating them. What is forbidden is selecting a using the
held block, its numerator, its contrasts, or the Monte Carlo reference bank.

Diagonalizing H0 conditionally gives

    Q_H = sum_(j=1)^S U_j/lambda_j(H0),   U_j iid chi-square_L.

The rotated Gaussian contrasts have a law independent of TRAIN, so the U_j
can be represented independently of H0, B_kappa and Z_a. This justifies the
proposed independent-factor simulation; it is not an assumption that the
observed fitted direction is independent of H0 or B_kappa.

Only H0's eigenvalue ratios need be pivotal. Trace normalization makes the
whitened H0 equal to a Gaussian-reference shape up to a possibly R-dependent
scalar, but

    lambda_min(H0) * sum_j U_j/lambda_j(H0)

is exactly invariant to that scalar. Thus W_0 has the proposed nuisance-free
reference law. This remains true for a fixed two Tyler updates: actual and
reference fits must use the identical initialization, dimensions and update
count. Convergence of Tyler iteration is not a premise.

Nonpositive T must retain p=1. Domination above is a **positive-tail** argument,
not a signed pointwise ordering for negative numerators. Partial-null mean
shifts only lower T; they do not contaminate Y because each study has the
same location across its six pipelines. Arbitrary positive target radii
cancel from the central term. No finite raw variance or fitted tail index is
needed. A common target-block radial factor/common target shape remains part
of the target argument.

## 2. Calibration B law and independence

For centered, independent calibration blocks, the matrix statistic

    A_l = ((L-S+1)/S) x_l'(Y_l Y_l')^-1 x_l

obeys A_l/kappa ~ F_(S,L-S+1), independently. At S=4,L=5 this is F_(4,2).
With a fixed positive normalizing constant c_F,

    kappa_hat = median(A_1,...,A_N)/c_F,
    B_kappa = median(F_1,...,F_N)/c_F.

The population F_(4,2) median is c_F=(1+sqrt(2))/2. This gives the reference
law specified in the proposal, independently of TRAIN shape and held target
data, provided calibration blocks really are independent of all target folds.

For even N, NumPy's median averages the two central order statistics. Simulating
N iid F draws and taking the same median is correct; replacing it by the beta
law of one order statistic is not. `reference_bank:23` correctly simulates the
full sample median. No unbiasedness, finite mean of F_(4,2), or confidence
coverage of kappa_hat is required. This is the estimator's finite sampling law,
not a Bayesian prior or a claim that kappa is random.

Calibration shape and radial-law variation that cancels in the matrix pivot
does not affect this law. In particular the earlier review's row-scale
clarification still applies: an invertible left row scaling of a centered
Gaussian block cancels from its matrix quadratic form. That does not extend
the target full-contrast argument to arbitrary per-study target radii.
Calibration centering, the shared pipeline kappa, and independence across
calibration blocks remain essential. A target/calibration kappa mismatch is
not integrated away by this construction.

The inference H must continue to use TRAIN contrasts only. If H and kappa_hat
were jointly fitted from overlapping calibration/target observations, simulating
their laws independently would not follow from the present proof.

## 3. Exact rank p-values and reference sharing

Let W_1,...,W_M be iid copies of the entire W law, independent of real target
and calibration data. Each reference needs its own mutually independent:

1. Gaussian TRAIN block sample and its fixed-iteration shape fit;
2. N calibration F draws and their sample-median ratio;
3. S chi-square_L terms;
4. Standard normal numerator.

`reference_bank:16-30` implements this whole-tuple construction, including
independent shape estimates for the individual reference draws. It must not
be replaced by a small shared bank of shapes resampled repeatedly, one shared
simulated B_kappa, or the observed fitted nuisance values without a new joint
exchangeability argument.

Define r(t)=(1+sum_b I(W_b>=t))/(M+1). The actual W_0 coupled to the held
statistic in section 1 has the same distribution as a reference draw and is
independent of that bank. Thus r(W_0) is a valid Monte Carlo rank p-value.
For T>0, T<=W_0 gives r(T)>=r(W_0); for T<=0, p=1 is at least r(W_0).
Therefore the proposed p is superuniform without asserting exchangeability
of the observed T itself with the references. This coupling step is the
important distinction between a pivotal dominating law and an exact null
law of every learned-direction statistic.

The plus-one correction and >= tie convention are essential.
`rank_tail:33-40` has the correct formula, using left-sided search in the
sorted bank. For the continuous ideal reference, the exchangeable rank has
CDF floor(alpha*(M+1))/(M+1); counting ties conservatively can only increase
the p-value. This uses the classical Monte Carlo rank result in
[Dufour, Lemma 2.1 and Proposition 2.2, with proof](https://jeanmariedufour.github.io/Dufour_1995_MCT_W.pdf),
not a new rank-testing theorem.

A fresh bank may be shared across all genes, signs, intersections and the
four equally sized folds. This creates dependence among their p/e-values,
but each marginal validity argument remains intact. No union-bound penalty
for reference reuse is needed. The bank must be independent of **all** real
data and unused in learning directions, PILOT calibrators or gamma. Each
analysis draws it once, records its seed, and replays that same bank; never
reroll or stop simulation according to discoveries. A fixed precomputed bank
does not acquire a conditional validity guarantee from the marginal rank
argument. Unequal TRAIN sizes require the corresponding reference laws.

## 4. Fourfold PILOT separation and the complete PC/e-BH argument

For test fold f, let P_f denote the sigma-field of its designated PILOT data
and any independent PILOT-only learning randomness. TRAIN is the other two
folds; TEST is f. Fixed fold membership and independent target blocks imply
that conditional on P_f, the joint law of TRAIN, TEST, calibration and the
external reference bank is unchanged. The nuisance-integration proof above
therefore gives superuniform p-values **conditional on P_f**, while still
integrating TRAIN, calibration and reference randomness.

Under a signed r=2 PC null, at least one three-study subset I0 is entirely
null. For projection, its nonnegative supported direction gives a valid
p_I0; taking max_I p_I cannot make it smaller. For the ordinary component,
min(1,3*min_(s in I) p_s) is valid on an all-null intersection by Bonferroni;
then use the same maximum. Neither step assumes independence among study
p-values or among their shared-bank ranks. Other studies' nonzero means do
not enter the all-null projection numerator or the contrast mean.

For each component c, let h_fc be a P_f-measurable decreasing nonnegative
p-to-e calibrator integrating to at most one on [0,1]. Let gamma_f be
P_f-measurable and in [0,1]. Then for each true signed null in TEST_f,

    E[h_fc(p_ic) | P_f] <= 1,
    E[(1-gamma_f)h_f0(p_i0)+gamma_f h_f1(p_i1) | P_f] <= 1.

Integration over P_f yields genuine marginal null e-values. Applying e-BH
at .05 to all 2G claims controls FDR in the ideal model. No delta_shape,
delta_kappa or approximate-e budget is left in this proof. Each claim uses
its own permissible P_f; conditioning on their union is unnecessary and
would generally expose the target data.

The constraint is the complete data path, not the name PILOT. Pilot scores
must use PILOT's own fits/directions, not inference TRAIN profiles, the
external kappa estimate, or the shared reference bank. Matching a pilot gate
to an inference TRAIN profile using both data sets would also violate the
stated measurability. This is precisely why the old TRAIN-selected calibrator
cannot simply be retained after replacing its p-values by marginal predictive
ones. Correlated target genes likewise cannot be excused by e-BH's tolerance
of dependent final evidence.

## 5. Source mapping: the inspected chain respects that boundary

In `predictive_bridge.py`:

- Lines 70-71 compute the external calibration estimate and one fresh full
  reference bank. Lines 75-80 use parity modulo four to construct disjoint
  TEST, PILOT and two-fold TRAIN sets, and learn inference H/profiles from
  TRAIN and calibration only. Inference TRAIN gamma is discarded.
- Lines 83-86 fit PILOT shape, PILOT-only kappa-like score, PILOT profiles,
  gamma and scores without inference TRAIN, calibration or reference inputs.
  `matrix_kappa_upper(z[pilot])` is applied to possibly noncentered PILOT
  targets, so its nominal coverage interpretation is **not valid there**.
  Only its median is used as a learning heuristic; that is permissible.
  Do not report the unused pilot upper bound as a certified nuisance bound.
- Lines 87-94 combine the genuine held predictive p-values with those
  PILOT-only calibrators and gamma. `calibrate`'s fixed .05 pilot rule and
  unit-integral focused calibrator are sufficient; PILOT scores need not be
  valid inferential p-values. Both `PB_main` and `PB_ordinary` are sent to
  e-BH at .05 on their full signed family at line 100.
- Lines 43-64 implement the all-study contrast denominator, positive NNLS
  directions supported on each triple, Bonferroni ordinary intersections
  and PC maxima. Rescaling a held block by its max absolute entry at 45-48
  is algebraically harmless: this common positive scalar cancels from the
  numerator/denominator ratio, even though it is computed from the target.
  It is not a learned evidence weight based on the target's Q.

The tests exercise rank/tie arithmetic, selected provenance perturbations and
an injected ArithmeticError. They do not prove distributional domination or
cover every excluded numerical failure. No fixture was rerun in this review.

## 6. Concrete remaining safeguards, wording and power limits

1. **Validate reference intermediates, not just final W.** At lines 20-29,
   checking only finite `w` can miss overflow in b, q or their product:
   a finite numerator divided by an infinite denominator becomes a finite
   zero. That can silently change the reference law. Require finite positive
   eigenvalues, b, q and denominator before forming W; record a whole-family
   conservative failure rather than dropping/redrawing individual tuples.
   This is engineering protection, not full interval certification.

2. **The no-discovery exception policy is incomplete.** Lines 117-126 catch
   ArithmeticError and LinAlgError. Imported `shape_fit` raises ValueError on
   nonfinite derived contrasts, and `positive_direction` can raise ValueError
   on a numerically non-PD subshape or RuntimeError from NNLS. Those can occur
   after valid external inputs, yet bypass the declared conservative result.
   Distinguish invalid caller inputs from recognized internal numerical
   failures and apply a documented policy to the latter. Retain every family
   and failure category. Do not fix this by indiscriminately swallowing
   programming errors or discarding unsuccessful families.

3. **Use one normalization constant on both paths.** Reference B uses the
   analytic F_(4,2) median at line 23, while observed kappa is obtained through
   `matrix_kappa_upper`, which uses `f.ppf(.5,4,2)`. They are identical in real
   arithmetic, but using one explicit shared positive constant removes an
   avoidable floating discrepancy. Its normalization need not itself be an
   exactly evaluated population median: using the same declared constant
   in actual and simulated estimators preserves the pivotal law. Also the
   bridge currently computes and discards the old .005 upper quantile for
   both calibration and PILOT. This adds work/failure exposure but no
   statistical error budget or protection to this bridge.

4. **Keep the scope of exactness clear.** Rank counts, and their division by
   4096, are exact for the represented finite numbers. Matrix solves,
   eigensolvers, projections and Gaussian/F/chi-square simulation remain
   floating-point operations. The proof is exact arithmetic with ideal
   independent random draws, not a uniform numerical certificate over
   arbitrarily ill-conditioned R. The current explicit numerical limitation
   is appropriate; fixtures do not remove it.

5. **Call W a dominating reference, without claiming sharpness.** The bound
   minimizes a Rayleigh quotient over all directions, not just allowable
   nonnegative supported directions. Its smallest-eigenvector direction
   need not be attainable by the method. Thus the stated envelope is valid,
   but no least-favourable attainment, optimality or minimal conservatism has
   been proved. Similarly, integration is over pivotal sampling errors, not
   over a prior distribution on unknown R or kappa.

6. **Rank resolution and data splitting remain genuine efficiency costs.**
   With M=4095, projection/intersection p cannot be below 1/4096; the ordinary
   three-study Bonferroni component cannot be below 3/4096. Focused thresholds
   below these floors receive no evidence from that component. G=256 leaves
   only 64 PILOT genes per fold to select thresholds/gamma, while TRAIN has
   128 and TEST 64. The dominating mixture can have heavy tails from small
   B_kappa or unfavorable shape eigenvalues. Removing coverage penalties
   does not prove useful power, and retaining 20 contrasts does not by itself
   overcome those costs.

7. **Preserve the development and comparison interpretation.** Rescoring
   stored D001 inputs after viewing D001 is development, not new independent
   confirmation. The change includes predictive integration, fourfold
   PILOT separation and two instead of twelve shape updates. A power
   difference cannot be attributed solely to removing confidence cutoffs.
   Preserve D001, bind the exact chosen inputs and auxiliary seeds, and
   retain the stipulated bounded scope; this audit requests no additional
   experiment, route or outcome-driven reference-count choice.

**Final disposition:** no new theoretical blocker to the one proposed bridge
under M0 and the inspected PILOT isolation. The stated fresh-reference rank
construction supports genuine null mean-one-or-less evidence and .05 e-BH in
the ideal model. Fix or transparently retain the concrete numerical/deployment
qualifications above; do not claim numerical certification, recovered power,
novelty, broad drift protection or biological applicability from this review.
