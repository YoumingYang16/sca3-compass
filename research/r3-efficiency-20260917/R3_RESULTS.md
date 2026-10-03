# R3 frozen independent results — useful M0 gain, failed mismatch non-worsening

Version R3-GSR-1.0.0, primary R3_main. ONE R3-C001 confirmation completed
2026-09-17 19:49HKT: 15 scenes x1024 independent G256 families=15360,
8CPU workers,4095 references/family,7253.956s complete-invocation wall time.
Last progress was7253.799s before the index/worker finalization. No optional extension,
reserve, changed seed, candidate switch or deleted failed scene. All inputs
are SIMULATION, not patients. V1/R2 remain frozen.

Authoritative numerical source: C001/summary.json, SHA256
`85bbb0fae69f6a45b6d96c82e6c40727914b74be1b6a29131169b661fee64697`;
index `0f7423b20278321a5286d3326d548fa6adb65c9ae2c2d158a0b2d5b713339b1d`;
freeze `4ef7e448c1ee6ee6064a0e31d53e6f36e21b9825e6a7920ef82aebcee9e0b620`.
Full absolute intervals, MCSE,p90,p99,runtimes and all controls are in that
JSON. Regenerable tables/plots: figures/RESULT_TABLES.md and make_figures.py.

## Decision

All THREE predeclared M0 empirical gates pass: all13 FDR uppers<=5%, positive
six-scene core improvement, and all10 nonnull regression guards. In fact all
ten individual R3-minus-R2 Power lower endpoints are positive. Complete M0
finite-calibration theory remains gradeA in ideal arithmetic.

The default method significantly WORSENS FDR in BOTH prespecified outside-M0
drifts. User clause15 fails, and closest-literature novelty is not established.
Therefore **NOT PROVED_AND_USEFUL under the full user definition**. Scoped
status: **MODEL_VALID_WITH_CONFIRMED_EFFICIENCY_GAIN; FULL_R3_CRITERIA_NOT_MET**.
This is not POWERFUL_BUT_UNPROVED: the scoped proof is retained. Nor does the
boundary failure erase the positive M0 result. No post-confirmation rescue.

## Estimands and uncertainty

Power=TP/number of true signed PC claims; undefined in all-null scenes.
FDP=FP/max(1,TP+FP), and reported FDR is the whole-family mean FDP. Nominal
q=.05 for EVERY method; no truth-calibrated threshold. Power higher is better,
FDR lower is better. Percentage-point differences are not relative percentages.

289 two-sided intervals use the predeclared384-comparison budget at alpha.025,
giving at least97.5% simultaneous coverage within this R3 confirmation under
independent families/ideal arithmetic. The core interval concerns the fixed
equal-weight mean of cases0–5, formed into1024 independent six-scene groups.
Genes, folds, references and methods are NOT independent repetitions. Coverage
does not extend to all past research phases or all possible distributions.
Scenario labels containing 'unseen' are inherited; these parameter settings
were seen during R3 DEV. Confirmation uses NEW observations, not new scenarios.

## All nonnull M0 Power results

Power in%; paired differences/intervals in pp. Intervals use the joint rule above.

| Scene | R2 | R3 | Strong | R3-R2 [simultaneous interval] |
|---|---:|---:|---:|---|
| Normal,N32 |27.844|37.506|75.343|+9.662 [6.494,13.049]|
| t5,N4 |12.647|16.242|51.386|+3.594 [1.380,5.965]|
| t5,N12 |20.895|26.840|50.601|+5.946 [3.503,8.447]|
| t5,N32 |28.567|36.573|52.102|+8.006 [5.784,10.241]|
| t5,N128 |34.725|42.046|52.258|+7.320 [5.598,9.064]|
| t3,N12 |47.137|53.975|72.578|+6.838 [4.644,9.134]|
| Continuous effects |25.318|29.267|26.757|+3.949 [2.531,5.296]|
| Dense90 |14.780|18.113|22.657|+3.333 [1.921,4.687]|
| Effect2.625 |22.365|29.199|42.883|+6.834 [4.566,9.159]|
| t1.5,N12 |7.039|9.773|21.400|+2.734 [1.079,4.393]|

Core Power: R2 28.63594%, R3 35.53028%, strong59.04469%. Paired improvement
**6.89434pp [5.46084,8.20097]**, mean3.516 additional correct signed discoveries
per core family (51true claims). This is a material repeated gain at the same
model guarantee/calibration input, not merely 'no errors'. But absolute t1.5
Power is still9.77%, and t5/N4 Power16.24% remains costly.

Descriptive core strong gap falls from30.40875pp to23.51441pp. No new interval
for an after-the-fact aggregate strong gap is introduced. Registered per-scene
R3-minus-strong intervals remain below0 in nine nonnull M0 scenes; continuous
effects is the exception: +2.510pp [0.772,4.226]. Strong is still strong, not
weakened or removed; it lacks the matching complete finite-calibration proof.
Its null-scene FDR is also not certified by these intervals (below).

## FDR, including partial nulls

R3 maximum M0 mean FDR **0.488281%**, maximum simultaneous upper **2.469514%**,
both in singleton case9. This validates the protocol's finite-scene gate;
it is not a distribution-free theorem or realized-FDP guarantee.

| Partial-null scene | R2 FDR% [upper] | R3 FDR% [upper] | Strong FDR% [full interval] |
|---|---:|---:|---|
| Global null |0.000 [1.150]|0.098 [1.535]|3.516 [1.334,7.209]|
| Singleton |0.391 [2.266]|0.488 [2.470]|5.762 [2.773,10.201]|
| Mixed-sign null |0.293 [2.049]|0.293 [2.049]|2.246 [0.633,5.425]|

The strong singleton point exceeds5%, but its interval includes5%: do NOT
claim this experiment proves its true FDR>5%. It fails the same upper-bound
acceptance check, not a declared theorem of invalidity. All methods' per-scene
FDR including ablations/V1 are retained in figures/RESULT_TABLES.md.

Registered descriptive tail summaries also show why FDR must not be confused
with a bound on every realized FDP: primary FDPp99 is8.874%at t5N4,6.206%at
t5N12,6.352%at t3N12,and8.917%at t1.5. All-PC-null scenes havep99=0 but rare
families with any discovery haveFDP=1. These are archived empirical quantiles,
not new confidence claims. No per-family FDP<=5% guarantee is asserted.

## Contribution decomposition: architecture is the main verified increment

The2x2 full table is in figures/RESULT_TABLES.md. Core averages below are
DESCRIPTIVE component means, not additional registered aggregate tests.

| Component | Scale | Shape/direction architecture | Core Power% |
|---|---|---|---:|
| A: R2 |Median finite law|G/2 joint TRAIN,lambda_min protection|28.63594|
| B: scale only |Geometric finite law|Same R2 architecture|29.96004|
| C: orientation+median |Median finite law|Independent G/4 DIR/SHAPE,joint orientation|33.96331|
| D: full R3 |Geometric finite law|Independent DIR/SHAPE|35.53028|

Point contrasts: B-A1.32410pp; C-A5.32737pp; D-B5.57024pp; D-C1.56697pp;
interaction D-B-C+A0.24286pp. These are NOT an additive attribution of all R2
losses, nor a claim that a pure H11 substitution alone explains the gain:
training size and directions also change with this architecture.

All ten registered D-minus-B Power intervals have positive lower endpoints:
the independent-orientation architecture adds supported value beyond changing
scale alone. D-minus-C is positive in point means everywhere, but ONLY t3/N12
has a positive simultaneous lower endpoint: +2.246pp [0.166,4.347]. Thus the
geometric mechanism has valid exact-law/moment advantages but its additional
Power benefit is NOT established scene-by-scene across the whole design.
The frozen primary remains D; no post-result switch to C or per-scene selection.

Existing projection mixture versus R3 ordinary adds supported Power in eight
nonnull scenes. Continuous effects: -0.002pp [-1.153,1.149]; dense90:
+0.078pp [-1.075,1.228], unresolved extra benefit. Gamma is exactly0 in89.67%
and84.18% of signed-fold choices respectively. This is intended ordinary
selection, not evidence of optimizer failure. All archived convergence flags
are true. These projections predate R3, so their benefit is not R3 originality.

## Small-calibration curve

Same t5/rho.95/effect3 design; N is independent centered calibration matrices.

| N | Geometric kappa/true kappa q10/q50/q90 | R3 Power% | Gain vs R2,pp | R>0 families% | Median final eBH threshold conditional R>0 |
|---|---|---:|---:|---:|---:|
|4|0.369 /0.975 /2.824|16.242|3.594|35.645|445.217|
|12|0.595 /1.005 /1.821|26.840|5.946|66.113|512.000|
|32|0.702 /1.001 /1.395|36.573|8.006|86.133|465.455|
|128|0.835 /0.995 /1.194|42.046|7.320|95.508|445.217|

Scale quantiles are descriptive sampling quantiles, not confidence coverage.
The exact log-scale variance is2.289868/N. Median calibration q90 atN4 is3.833,
versus2.824 for geometric, but stochastic/Power dominance does not follow.
R>0 counts any discovery (true/false,either sign), not a true-discovery event.
Conditional eBH thresholds need not decrease monotonically: atN4 the many
zero-discovery families are excluded from that threshold statistic.

Improvement is real at smallN but NOT largest there: N32 gain8.006pp exceeds
N4gain3.594pp. The oracle suggested large small-N loss; this method recovers
only part of it. More calibration remains useful. No N was selected post hoc.

## Required failure boundary: default drift is worse

| Boundary | R2 FDR% | R3 FDR% | Paired increase,pp [simultaneous interval] | Verdict |
|---|---:|---:|---|---|
| Moderate,kappa ratio2.2 |3.426|4.846|+1.420 [0.198,2.613]|WORSENING_DETECTED|
| Severe,kappa ratio4.6 |16.341|20.008|+3.667 [2.271,5.001]|WORSENING_DETECTED|

Moderate R3 upper is6.077%, so no5% certification; the point alone below5%
does not pass. Severe R3 full interval[18.426,21.578]% is far above5%.
Power in these invalid-default scenes is not evidence of acceptable efficiency.
These results directly FAIL user clause15. Removing model-valid overprotection
also reduces incidental tolerance to mismatch in these scenes; that is an
empirical mechanism interpretation, not a universal impossibility theorem.

Externally justified Delta5 sensitivity: moderate R3 Power4.140%, FDR0.003906%
[upper1.154%]; severe Power31.687%, FDR0.249221% [upper1.403%]. R2 Delta5
Power2.507%/24.939% respectively. The guarantee requires an external ratio
bound; default inputs do not identify it. Delta sensitivity does NOT repair
the frozen default's failed criterion or justify silently recommending Delta5.
Paired Delta FDP intervals cross0; no relative nonincrease claim is made.

## Cost, failures and actual reproducibility

- M0 per-family R3 bundle means1.291–2.139s, R2 bundle0.959–1.639s. Timings
  include component outputs, not optimized primary-only implementations.
  Strong/V1 timing covers the whole historical multi-output call; no speedup
  claim or isolated-primary CPU comparison is inferred.
- Fixed confirmation wall7253.956s=120.899min,8workers/1BLAS each. Observed
  worker working sets~105–116MB and reported process peaks~119–122MB at the
  saved checkpoints; not a whole-run certified peak-memory benchmark.
- R3 and R2 numerical failures0/15360; all Delta calls completed. V1 recorded
  two top-level non-COMPUTED fallback statuses (cases8,12), retained in metrics.
  Those statuses are not a complete V1 internal optimization audit.
- IMPLEMENTATION_DIAGNOSTICS.json checks all15360 records/61440fold receipts:
  direction,PILOT,and scale-ablation learning flags have0nonconvergence. Fold
  receipts are not independent experiments and do not prove a global optimum.
- Frozen15tests pass (frozen-tests.xml). Five predetermined saved-input
  replays match109arrays exactly (REPRODUCTION_AUDIT.json). They do not rebuild
  every input from its generation seed, and do not enlarge confirmation n.
- Actual fixed example: completed in1.993s including ablations,0discoveries
  (EXAMPLE_RECEIPT.json). Retained without seed shopping; not a biological result.

Exact fresh-process re-analysis, figure provenance and final independent-review
receipts are checked before DELIVERY_MANIFEST.json is sealed. The final status
records whether those operations actually completed. No new statistical test
or confirmation is licensed by reproduction.

## Research conclusion

Retain the full M0 proof and the independently supported efficiency recovery.
The important verified component is independent orientation/shape learning;
geometric calibration is a smaller supported-in-part increment. Remaining
Power costs cannot be called inevitable: the information bound only concerns
unbiased log-scale estimation in the pivot experiment, not detection Power.
This supplies a scoped construction/comparison paper core with an explicit
validity–efficiency–mismatch tradeoff, NOT established general novelty or a
publication guarantee. Full R3 success is false. Stop at this bounded evidence
package; do not launch drift rescue,R4,more folds or conditional boosting.
