# Single within-line efficiency correction after D001

2026-09-17 13:27 HKT. D001768/768 completed, no failures,113.15s wall.
Guarded Power: normalreference0%,t5N4/12/32all0%,t5N1280.337%,t3N12 2.390%.
Matched known-all nuisance diagnostic:58.73%,46.02%,51.26%,53.40%,53.16%,73.22%.
Full values in D001/summary.json. This rejects utility of the confidence-envelope
prototype under the tested weak/moderate effects; it does not refute its theorem
or frozen V1. Known-shape alone recovers23.47% atN128 but only0% atN12.
Both kappa and shape protection matter; do not keep merely tuning their budgets.

## Proposed single correction: integrate the pivotal nuisance distribution

Keep the full20-coordinate target pivot and affine shape estimator. Replace
large simultaneous confidence cutoffs by an exact Monte Carlo rank tail of a
dominating predictive statistic (sharp attainability not proved). This is the same finite-calibration
line, not a new projection family. Old five-df R1 remains a negative reference;
this uses all20 contrasts and explicitly handles unknown R.

Let H=train-only shape, H0=R^-1/2 H R^-1/2 up to scalar, with known Gaussian
reference distribution by the affine proof. Let kappa_hat be the median of N
independent matrix calibration A values divided by median F(4,2). Thus
B=kappa_hat/kappa has a known distribution obtained from N iid F(4,2) draws.
It is independent of the TRAIN shape and target Gaussian block.

For any TRAIN/calibration-learned nonnegative intersection direction a,

 T = a'M / sqrt[kappa_hat (a'Ha) tr(H^-1 YY')/D].

Write b=R^1/2 a. Under the intersection null the positive tail of T is
bounded by the positive tail of

 W = Z / sqrt[B * lambda_min(H0) * sum_j ChiSquare_L,j/lambda_j(H0) / D],

where Z, the S chi-square variables, Gaussian-reference H0, and B are
independent. Proof: in whitened coordinates b'H0b/||b||²>=lambda_min(H0).
Conditional on TRAIN/calibration, the centered numerator standardized by
||b|| is an independent standard normal; the contrast quadratic has the
indicated conditional eigenvalue representation. On positive numerator the
inequality holds pointwise under a coupling. Nonpositive projected means
only lower the statistic, and nonpositive observed statistics give p=1.
The law of H0 eigenvalue ratios is nuisance-free even if a depends on H and
the calibration. The worst-direction envelope removes that selection.

Generate B_MC independent W_b references; rank p=(1+#(W_b>=T))/(B_MC+1)
for T>0, else1. A target with stochastically smaller positive tail has
superuniform rank p, integrating the reference randomness. No estimated
99.5% shape cutoff, coverage failure event or Monte Carlo approximation error
is being discarded. Rank resolution is a real efficiency cost.

## Crucial dependence change — separate PILOT from TRAIN

Marginal predictive p-values cannot feed the old TRAIN-selected calibrator.
Use four equal target folds: for test fold f, PILOT=(f+1) mod4 and TRAIN=the
remaining two folds. Calibration and auxiliary references are independent of
PILOT. TRAIN chooses the directions and inference shape; PILOT alone chooses
gamma and each p-to-e calibrator. Pilot nuisance estimates/profiles/scores
are computed ONLY from that pilot block, never from shared external calibration,
the reference bank, the inference TRAIN or HELD. Pilot scores need not be
valid p-values, because they only select an integrable calibrator.

Conditional on the entire PILOT, the target/TRAIN/calibration/reference
construction retains its distribution. Its ordinary and projection PC p
values are superuniform by the null-subset argument. PILOT-measurable gamma
and normalized decreasing calibrators therefore give genuine marginal null
mean<=1 e-values. Apply e-BH at q=.05 to all2G hypotheses, allowing the cyclic
fourfold dependence. No conditional-on-union or own-Q weight is used.

## Boundaries and novelty

The result relies on independent Gaussian angular blocks, common target
shape/radius structure, centered independent calibration with matched kappa,
and prescribed random references. Reference randomness must be drawn once
per analysis and never selected for favourable output. A frozen shared
reference table is not claimed conditionally valid. No tail-law fit is needed.

This is a problem-specific dominating pivotal construction using known
invariance/MC rank/e-BH tools. Its novelty and real usefulness remain unconfirmed.
No assertion that every nuisance parameter can be inferred without calibration.

## Minimal next test (before any new performance output)

First independent proof review and deterministic code tests. Use2 fixed Tyler
updates (not12) for inference and references: affine validity holds for any
fixed iteration count; fewer updates control the reference cost. Shape is not
claimed converged. B_MC=4095 for the initial diagnostic; no adaptive MC stopping.
Then rescore the first32 stored D001 inputs for cases0,1,2,4,5,7,9,10 only,
256 families, no new target/calibration observations. Compare this single
correction with the archived guarded output and original strong references.
This is DEVELOPMENT, not a new independent experiment. Stop after256 fixed
inputs, inspect actual information/efficiency loss before freezing confirmation.
