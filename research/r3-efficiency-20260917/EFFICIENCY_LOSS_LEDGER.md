# R3 efficiency loss ledger — first completed diagnostic

2026-09-17. This ledger precedes any R3 candidate implementation. Source:
sealed R2 C001 full results and R3-D001/summary.json (240 archived families,
10scenes x24,0failures,32.49s). D001 baseline p/e/reference arrays EXACTLY
replayed R2 before scoring. All oracle labels below are DIAGNOSTIC_ONLY.
They preserve actual learned directions, PILOT, gamma, support, references'
random draws and final nominal e-BH.05, replacing only inference nuisance
scales/laws. Known shape is NOT a new shape estimator. No oracle sees new
patient data. C001 is now development evidence for R3, never its confirmation.

## Paired nuisance-removal ladder

Power%, higher is better; each cell shares the SAME24 input families in its
row. These development means need not equal R2's full512-family means.

| Scene | R2 | True kappa only | True R only | Both true | Actual strong |
|---|---:|---:|---:|---:|---:|
| normal |30.80|41.18|38.64|53.10|72.71|
| t5,N4 |14.38|36.11|17.65|44.04|57.11|
| t5,N12 |21.32|37.99|27.86|48.04|52.37|
| t5,N32 |20.34|33.99|26.72|42.08|47.06|
| t5,N128 |35.13|39.71|44.77|46.08|53.35|
| t3,N12 |48.37|59.64|55.39|65.52|69.93|
| continuous |29.53|29.86|33.62|33.95|28.51|
| dense90 |15.11|20.96|17.55|24.06|21.72|
| unseen amplitude |29.74|35.13|36.11|41.75|45.42|
| t1.5 |11.11|14.62|12.50|19.44|25.98|

True-kappa gains: t5N4+21.73pp(MCSE6.28),N12+16.67(6.03),t3+11.27(3.33).
True-shape gains: normal+7.84(1.57),N12+6.54(2.66),N128+9.64(1.97).
Both at t5N12+26.72(4.86), NOT the sum of isolated16.67+6.54.
All oracle-row FDR estimates range0–1.00%;24repetitions cannot certify FDR.
They are valid ideal M0 inference substitutions with additional true nuisance
information, not attainable upper bounds, global optimal tests or deployed
algorithms. Unobserved real parameters are explicitly unavailable.

## Ten-source accounting (not an additive causal partition)

| Requested source | What can actually be established | Quantification / uncertainty |
|---|---|---|
| A finite-calibration protection | Primary integrates the full estimator error law; it does NOT spend a coverage delta | A overlaps B, not a separate loss to add. Both-nuisance removal gains4.41–29.66pp in D001. |
| B kappa uncertainty | Large small-N price of current matrix-pivot median and its predictive tail | Isolated gains above; disappears incompletely atN128(+4.58pp). Oracle does not prove all of it recoverable. |
| C unknown study shape | Shape estimation AND direction-uniform Rayleigh lower envelope | True-R replacement recovers1.39–9.64pp. It cannot separate estimation from the envelope by itself. |
| D splitting | TEST64/PILOT64/TRAIN128; all256 genes tested, none simply discarded | No isolated Power allocation identified by these substitutions. More training would change several laws. Do not report an invented percentage or zero loss. |
| E direction/gamma learning | Same learned signal extractor held fixed; gamma often reverts to ordinary under heterogeneous signals | On the D001 t5N12 subset fixed projection28.84%, versus mixed21.32 and ordinary14.46. Continuous projection17.77 vs mixed29.53, so removing the gate uniformly is not justified. This is a gate/component contrast, NOT a true-direction oracle. |
| F PC combination | Max four triple tests; ordinary triple factor3. Necessary for partial-null target, not just global null | No clean standalone percentage: removing max changes the scientific null. Ordinary support first3/4096 vs projection1/4096, giving evidence caps4096/3 vs4096. Raw .05 PC screening is NOT FDR control. |
| G p-to-e conversion | Discrete normalization already recovers6.69–24.12pp in full R2 C001 vs same continuous calibrator | Projection raw p<=.05 screening detects75.41–95.51% in normal/t5N4/t5N12/t3, but FDR12–15%; this invalid screen is not recoverable gain. PILOT focus below grid floor often(.72/.82 forN4/N12) is already renormalized, not silently discarded budget. |
| H final e-BH threshold | Uses512 signed claims and nominal.05, not a nuisance multiple-comparison error | Replacing it with individual e>=20 adds10.21pp(N4),18.95(N12),8.99(t3). This is only relaxed screening, no family guarantee; cannot adopt based on low non-null-scene FDR points. |
| I worst-case envelope | Primary uses lambda_min(H0), not confidence rectangles or numerical supremum | Component of C; algebraic bound may ignore independence/orientation. This is the second selected bottleneck. No fixed extra delta is repeatedly paid across genes. |
| J finite ranks/numerics | Reference ranks have a genuine minimum p; floating error is separately audited, not shown to explain loss | True projection ranks hit floor13.64%(N4),18.38%(N12),37.01%(t3). Evidence-cap theorem implies at least3 total rejections. These fractions do NOT equal lost Power; rank ties include already-detected signals. No numerical failures. |

The raw .05-screening figures in G come from D001 secondary metrics, not
the formal method roster. Every secondary includes its actual FDP. Small
subsets have substantial MC error; no significance or confirmation claim.
R2's 80-digit scale checks found approximately1e-15 relative errors, not an
explanation for tens of Power points. No evidence licenses threshold inflation.

## What is necessary versus possibly avoidable?

- Neither oracle gap nor strong-reference gap is an information lower bound.
- Estimating scale from only N independent matrices has a real sampling cost;
  the F4,2 median law/infinite variance atN4 is specific to that estimator.
- The spectral worst-direction bound is sufficient, not proven sharp for the
  learned nonnegative directions. Its full price is not known to be necessary.
- Grid normalizer is sharp over generic superuniform grid laws; that is not
  a minimax theorem for the attainable PC tensor model.
- Independent learning is necessary for the current proof, not a theorem
  that no other split or aggregate could work.
- Strong uses extra radial working assumptions and plug-in nuisances; its
  empirical Power is not a required lower bound for a uniform M0 method.

## Decision before new mechanism

Optimize at most TWO layers: (1) finite-kappa sampling efficiency, primary;
(2) avoidable direction-uniform spectral protection, secondary. First seek
an equivariant scale summary whose exact finite law is still simulable;
then examine whether separating direction learning from inference-shape
learning permits an exact orientation law instead of lambda_min. No new
projection family. All proposed laws require proof before performance runs.

Do NOT start conditional joint boosting, compound budget inflation, new PC,
more folds or all-grid searches now. Conditional e-vector law is not known;
there is no repeated failure budget in the frozen primary to reclaim. Leave
conversion/resolution as documented residual costs unless selected lines fail.

## Analytic cost clarification added after candidate freeze

SCALE_INFORMATION_BOUND.md supplies a narrow classical information bound:
in the experiment observing only N independent A=kappa F4,2 pivots, regular
globally-unbiased estimators of logkappa have variance>2/N at finiteN. R3's
log-geometric estimator has exact variance2.289868/N. The proof is independently
reviewed in REVIEW_THEORY.md. This supports a real calibration sampling price
under a precisely stated loss/estimator class, but does NOT quantify unavoidable
Power loss, cover all raw-data procedures, or establish a minimax detection bound.
It cannot convert the oracle/strong gaps in the table into required prices.
