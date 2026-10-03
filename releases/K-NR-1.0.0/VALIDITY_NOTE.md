# K-NR-1.0.0: actual implementation and validity boundary

Pre-confirmation note, 2026-09-16. Evidence level remains EMPIRICAL_ONLY. Independent validation, when available, is linked by the release decision, not assumed here.

## Algorithm, not an invented new family

Input is a G x4xK statistic tensor and independent centered 4xn xK calibration vectors; this release validates G256,K6. Original study/pipeline definitions must describe comparable prespecified effects. Opposite signed claims are separate members of the 2G family. There is no use of biological null labels, true covariance, raw subject data or outcome-dependent method selection.

1. Fit the original centered compound Student/Gaussian calibration likelihood. Student starts5/30, original bounds/maxiter/tolerance/BIC retained. Use analytic interior Gaussian MLE only inside the original feasible box; otherwise original numerical optimization. Only successful finite Student fits can displace each other. Both likelihood models must be usable. No claim of global Student optimum.
2. Fixed odd/even gene folds. The other fold supplies pipeline contrasts, Tyler study shape (existing .001 ridge), and target contrast-energy inverse-gamma/Gaussian prior. Common pipeline effects cancel in contrasts. Both shape and prior must converge.
3. TRAIN effects determine existing nonnegative pattern profiles and gamma. Failed pattern convergence gives the existing uniform-profile/gamma0 fallback, logged. The held gene never learns its profile/prior/calibrator.
4. Resample independent calibration vectors16 times with the original seed stream. Hold target contrast-scale eta²=(1-rho)*tau², prior degrees of freedom and study shape fixed. Replace rho by the maximum of original and16 bootstrap fits; tau² rescales to preserve eta². This maximizes the conditional tail in this one-dimensional sensitivity bank. It is NOT a confidence upper limit on rho. A failed resample yields p=1 on that fold, logged; original fit failure yields an explicit family error.
5. In each of four triples, use the existing nonnegative profile projection and ordinary Bonferroni intersection. Compute analytic conditional normal/Student positive tails under the fitted parameters; nonpositive statistics return p=1. Max across all triples supplies the r2 PC value. TRAIN-only pilot c=.5 with its existing .8 concentration gives component e-like evidence; mix with TRAIN gamma. Finally eBH at.05 over512 signed claims.

The callable sidecar avoids global monkey-patching. `molecular_v1.py` returns one fixed K primary and seven fixed comparison procedures (including both pre-confirmation conditional ordinary controls). Production numerical values are tested against the repaired legacy implementation; legacy C1/C2 sources are not modified.

## Ideal conditional calculation and the unclosed gap

The finite-mean condition refers to the observed statistic/effect estimand (Student degrees of freedom >1), not the mean of the inverse-gamma variance radius V. E[V] and raw noise variance can be infinite at nu<=2; t1.5 was explicitly included before confirmation. This clarifies the frozen assumption wording, without changing cases, input or estimand.

For known compound pipeline rho, study shape R, common pipeline mean mu, and inverse-gamma radius, Y=ZH removes mu. Let d=4(K-1), Q=tr(R^-1 YY')/(1-rho), v=[1+(K-1)rho]/K. With radius V~IG(nu/2,nu*tau²/2), conditioning on Y gives V|Y~IG((nu+d)/2,(nu*tau²+Q)/2). For fixed nonnegative triple direction a, the centered projection divided by sqrt(v*(a'Ra)*(nu*tau²+Q)/(nu+d)) has Student nu+d law. The Gaussian branch is the limiting conditional normal model. Scale² is not necessarily variance; raw variance need not exist at nu<=2.

Given valid conditional p-values, max-over-triples covers partial-conjunction nulls with up to one genuine directional signal, not merely the global null. A fixed valid p-to-e calibrator and TRAIN-measurable convex mixing preserve null expectation <=1. eBH then permits dependence among hypotheses. Its known theorem does not validate an invalid fitted frontend: [Wang & Ramdas](https://arxiv.org/abs/2009.02824).

**Actual gap:** rho, study shape and prior are estimated; the finite bootstrap bank has no established simultaneous nuisance coverage; the TRAIN pilot and gamma depend on calibration fits. Correctness at the unknown true nuisance, cross-fitting, or final eBH does not prove the deployed e-like evidence has null expectation <=1. The practical output retains p/e names for traceability, but their general calibration is not certified. No MODEL_SUPPORTED label or finite-sample guarantee follows from this derivation. Independent FDR checks certify only the stated finite simulation comparisons at their report confidence level.

## Fair ordinary comparison

Pre-confirmation DEV correction: the raw plug-in marginal repair produced6/32 false-discovery families in t5/n4/singleton, so it is not a qualified blanket fair comparator. It and its guarded version remain reported calibration attempts. Formal fair gatekeepers instead use the ALREADY EXISTING K guarded ordinary Bonferroni-PC component with identical pooled conditional calibration. One retains original BY; another grants the same TRAIN pilot/eBH as K, an explicitly enhanced baseline. Both must independently pass the same empirical FDR gate, and K must beat both. This adds no method family or true parameters; it makes the Power hurdle much harder than comparing with near-powerless heavy-tail marginal Bonferroni. It is not represented as a one-line repair of the original procedure. See the dated V1_ACCEPTANCE amendment and all2688 existing-input re-scores in R0077-fair-diagnostic.

Original normal pipeline marginal tails are invalid under non-Gaussian heavy tails. B_fair_plugin uses independent calibration Student/Gaussian fitting with the same numerical correction. B_fair_guard maximizes marginal tails over exactly the primary's fold-specific original+16 calibration fits before original pipeline Bonferroni, PC and BY. B_common uses pipeline0 with that same protected frontend. No comparator receives true nuisance; no output threshold is tuned to actual FDR. The plugin comparison remains reported so the new method cannot appear superior solely by overprotecting its baseline.

Strong reference uses existing unguarded conditional pooling, supported harmonic-Simes, TRAIN pilot and eBH. It is a fixed deployable rule, not an envelope chooser. Its own empirical false-discovery control must be reported, not assumed.

## What confirmation can and cannot establish

Independent observations are entire families with fresh calibration. FDP and Power differences are bounded even when raw noise variance is infinite. Fixed-n paired intervals reuse the checked range-two form of [Maurer & Pontil, Theorem11](https://arxiv.org/abs/0907.3740), allowing independent non-identically distributed families. Equal counts inside each fixed core stratum preserve scene weights; pooled variance includes heterogeneity. Stratified MCSE is separately descriptive. Scene FDR intervals invert binary-KL Chernoff inequalities for bounded FDP, not gene-binomial likelihood. Planned union error allocation is shared across report endpoints.

New independent noise does not create previously unseen model domains. Historical severe covariance drift (R0076 case67 observed FDR74.7393%) remains unsupported and reported. No real-data/clinical validation, broad drift solution, algorithmic originality, article acceptance or admissions claim follows.
