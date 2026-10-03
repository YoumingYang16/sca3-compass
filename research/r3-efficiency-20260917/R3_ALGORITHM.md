# R3-GSR-1.0.0: geometric scale and independent shape reference

Primary output `R3_main`. This name identifies the implementation, not an
assertion of scientific priority. Theory is in R3_THEORY.md. Inputs are finite
Gx4x6 signed common-location summary matrices and Nx4x6 independent CENTERED
calibration matrices; G>=16 divisible by4,N>=4. No truth, R, kappa, distribution
label, patient outcome or oracle input enters `evaluate`.

## Model boundary

Each target is mu_g 1' + sqrt(V_g) R^(1/2) E_g C(rho)^(1/2), with independent
standard Gaussian E, common positive per-matrix radial scale independent of E,
unknown SPD R and equicorrelated pipeline C. Calibration may have different
invertible study transforms and radial laws but the SAME kappa. Gene matrices
and calibration matrices independent. Shared study shape R across target genes.
No arbitrary target-gene dependence, target-block study-specific radial scaling,
uncentered calibration, unmatched pipeline effect profiles or estimated
transport bound. Calibration left-factor cancellation is separately allowed;
it must not be confused with the stricter common target-block radius condition.
For interpreting mu as a mean require E sqrt(V)<infinity; variance may be infinite.

PC target: at least TWO of FOUR study effects have the chosen sign, separately
for positive and negative directions. Final family contains2G signed claims;
FDR is E[false signed discoveries / max(1,total signed discoveries)]. Power is
mean of true signed discovery fraction, undefined in all-PC-null scenes.

## Actual workflow

1. From each calibration matrix form mean x and five orthogonal contrasts Y.
   A=.5 x'(YY')^-1 x, using rescaled SVD. Compute geometric kappa_g and the
   original median kappa_m. Only kappa_g determines primary inference scale;
   kappa_m stays in direction learning. The receipt field
   `kappa_median_learning_only` also supplies the median ABLATION's inference.
   The constant e/2 centers log-scale error for interpretation; any common
   positive rescaling cancels between observed and reference statistics in
   exact rank comparisons. It is NOT itself a Power-producing correction.
2. Make deterministic cyclic4folds: TESTf,PILOTf+1,DIRf+2,SHAPEf+3.
   Fit existing2step block-radial/affine shape on each fold's contrasts.
   DIR learns existing profiles and nonnegative triple directions using its
   OWN shape. It never uses the inference SHAPE fit to choose a direction.
3. Independently simulate4095 complete reference tuples. Each includes the
   Gaussian shape fit on G/4 blocks, its JOINT11entry and eigenvalues,
   an N-sample geometric F4,2 summary error,4chi-square5 and one Gaussian.
   Sort Z/sqrt[B_g H11 sum(U_j/lambda_j)/20]. No fitted tail index/CLT.
4. Studentize TEST means and nonnegative projections by geometric calibration
   and SHAPE contrast energy. Positive plus-one upper-tail reference ranks;
   nonpositive statistics receive p1. Ordinary component:3min marginal p per
   triple then max over4triples. Projection component: max4triple rank p.
5. PILOT uses unchanged median/own-shape heuristic to select the existing
   calibrator and mixing gamma. Exact rank-support normalization creates two
   e-components. Mix with PILOT-only gamma. Apply e-BHq.05 over all2G claims.
   Gamma0 can be an intended learned ordinary-component choice, not necessarily
   a numerical/optimization fallback; report actual convergence flags separately.
6. Validate arrays and fail conservatively to all-zero evidence/discoveries
   on recognized numerical failures. Bad shape/seed/input requests raise an
   explicit ValueError. No claim that exceptions catch all floating-point bias.

`mismatch_bound=Delta>=1` is an externally supplied squared-denominator factor,
valid only under the justified kappa_target/kappa_cal<=Delta premise. Default1
does not solve drift; simulation Delta5 is a sensitivity, not learned correction.

## Comparisons and component attribution

| Output | Scale | Shape protection / learning |
|---|---|---|
| PB_grid (R2) | median finite law | lambda_min reference; TRAIN G/2 selects both shape/direction |
| R3_scale | geometric finite law | original R2 TRAIN and lambda_min |
| R3_median | median finite law | independent DIR/SHAPE, exact orientation law |
| R3_main | geometric finite law | independent DIR/SHAPE, exact orientation law |
| R3_ordinary | same as main | main's ordinary PC e-component only, no projection mixture |

The2x2 ablations identify the scale substitution and combined independence/
orientation architecture, not pure H11's effect holding training size fixed.
PILOT/PC/calibrator family/finalq unchanged; effects need not add. Raw grid
ranks shared across hypotheses/folds create allowed dependence, not more n.

### Exact frozen strong and V1 comparison definitions

`B_strong` is NOT an oracle or a generic unadjusted Simes test. V1's twofold
TRAIN estimates pipeline calibration, Tyler study shape and a Gaussian-versus-
inverse-gamma energy prior. It forms fitted conditional Normal/Student study
tails. Each triple retains the TRAIN-learned positive support and applies
Simes times that support's harmonic factor, then maximizes across four triples.
TRAIN-selected p-to-e conversion uses the fixed multiplier .5; final eBHq.05.
Thus empirical-Bayes variance borrowing, support restriction and nuisance
plug-in are part of its sensitivity, but no matching complete finite-calibration
M0 guarantee is proved. Source: frozen V1 molecular_v1.py evaluate, and
robustness_bootstrap_guard.py variance/components (support_simes_by).

`K_NR` is the preserved V1 mixed projection/ordinary algorithm using a fixed
16-resample calibration perturbation bank, guarded tails and the same historical
pilot conversion. A failed guard has its declared p1 fold fallback. Its finite
bank is not a Berger–Boos coverage certificate. Both controls receive the same
observed z/cal matrices as R2/R3; neither gets true kappa, shape or labels.
Their whole V1 call calculates several historical outputs, so its recorded
cost cannot be treated as an optimized strong-only runtime.

## Complexity / cost reporting

With fixed4studies/6pipelines, target shape/profile work is O(G) with fixed2
updates, calibration O(N), reference construction O(M*(G+N)); ranking costs
O(M logM + G logM) per fixed set of components. Batched reference memory
O(64*(G+N)+M); held/source/evidence O(G+N). The implementation returns all
ablations and replays the R2 reference in the same draw schedule; timings
therefore include these outputs and are NOT isolated optimized primary costs.
Fixed8worker CPU confirmation; no GPU algorithm or hidden cloud compute.

## Reproduction interface

From project root, add the frozen `research/r3-efficiency-20260917/C001`
directory to Python's sys.path, import `r3_model.evaluate`, then:

```python
result = evaluate(z, calibration, seed=22, reference_draws=4095)
discoveries = result['decisions']['R3_main']
```

The saved-input executable example and actual result receipt are packaged
after confirmation. Simulated discoveries are not biological discoveries.
Proof assumes auxiliary randomization independent of data; picking a seed
after inspecting discoveries invalidates that workflow.
