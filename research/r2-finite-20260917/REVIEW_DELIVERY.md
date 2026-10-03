# C001 delivery claim review

2026-09-17. Sole independent reviewer; bounded final claim/evidence inspection.
Only this file was written. No project code, tests, fits, replay, or experiments
were executed; no new agent or research route was opened.

**Scientific disposition: PASS for the stated model-scoped PROVED_BUT_COSTLY
conclusion. No numerical-result contradiction or unsupported superiority,
novelty, or clinical claim found in the reviewed headline results. Before
sealing, integrate the documentary corrections below.** No change to C001,
its analysis, or its observations is needed.

## Concrete corrections before sealing

1. **Interval-budget unit is still wrong in one manuscript paragraph.**
   `PAPER_CORE.md:73-75` says `512-endpoint`. Replace with: “338 reported
   two-sided intervals, within a predeclared cap of 512 two-sided intervals;
   each tail uses .05/(2*512).” There are 676 reported one-sided bounds, not
   at most 512. `C001/summary.json` and the Chinese report already use the
   correct contract. This is documentary, not an actual alpha-budget breach.

2. **Specify which scale Delta multiplies.**
   `PAPER_CORE.md:50-51` and `RESEARCH_FINDINGS_ZH.md:105` currently say
   “inference scale”/“推断尺度”. State explicitly: replace kappa_hat by
   Delta*kappa_hat in the squared denominator, so the statistic's denominator
   is multiplied by sqrt(Delta); learning and references remain unchanged.
   This describes the actual frozen implementation and the reported Delta
   outputs. Do not let readers infer that the standard-error denominator was
   multiplied by Delta itself. The existing ratio/scope qualifications are
   correct.

3. **Seal bookkeeping still says the completed run is running.**
   `EXPERIMENT_REGISTRY.json:8` retains `running_verified_pid85092`, 192
   completed families, and “primary results not yet analyzed”. Update that
   result-only registry entry to the actual completed 7,680-family state and
   its existing summary/audit references. Do not alter the frozen protocol or
   erase historical progress entries. This is a delivery consistency issue,
   not a scientific blocker.

One narrower attribution clarification: `RESEARCH_FINDINGS_ZH.md:117` calls
the four replay inputs “预设”. The receipt supports cases 0/5/10/14, rep 0,
and the helper specifies them, but the reviewed frozen C001 protocol does not
preregister that replay subset. Use “4个指定数据集” (as the English manuscript
says “four specified inputs”), or cite a dated prior selection record if
“predeclared before confirmation” is intended. Their deterministic engineering
value is unaffected; these are not additional independent confirmations.

## Claims checked against actual saved evidence

All paths below are relative to `research/r2-finite-20260917`.

- `C001/summary.json`: 7,680 families, 338 two-sided intervals, cap 512,
  alpha .05, 512 families per scene. All 15 base-PB numerical-failure counts
  and both scenes' Delta-specific failure counts are zero. Model and drift
  scenes remain separate.
- M0 maximum primary mean FDP is .00390625 = **0.390625%**, in case 10.
  Maximum simultaneous upper bound is .031012713400988375 =
  **3.1012713401%**. The reports correctly describe these as simulated
  estimates/bounds, not the source of the mathematical guarantee.
- The displayed PB_grid/ordinary/V1/strong power values match the summary.
  Representative main/strong values are normal 25.6893%/74.0005%, t5/N12
  22.6524%/51.7770%, and t3/N12 45.9597%/72.4341%. The reported signed paired
  deficits and simultaneous intervals have the correct direction and
  percentage-point units. The t5/N128 interval crosses zero; no equivalence
  claim is made.
- All ten non-null M0 scenes (0-7, 11, 12) have positive prespecified lower
  bounds for PB_grid minus PB_main power. Point gains range from
  **6.6865808824 to 24.1230085784 percentage points**, correctly rounded to
  6.69-24.12. Every corresponding grid-minus-ordinary M0 interval crosses
  zero. The text does not turn favorable projection/mixture point estimates
  into a confirmed advantage or a general causal claim.
- Severe drift case 14 has primary FDR 16.59578524%, interval
  [10.16699322%, 24.72938987%], clearly above 5% at the stated simultaneous
  level. Delta5 gives FDR .15279619% and power 25.17233456%; in case 13 its
  power is 1.80759804%. The supplied bound, ratios 4.6/2.2, and insufficiency
  of Delta2 are correctly qualified. No automatic transport guarantee is
  claimed from the favorable Delta point estimates.
- `mechanism-summary.json`, cases 6/7: gamma-zero fold/sign fractions are
  .897216796875/.834716796875 and whole-family all-zero fractions are
  .400390625/.25. These match 89.72%/83.47% and 40.04%/25.00%. All reported
  TRAIN/PILOT pattern-nonconvergence fractions are zero. The documents
  correctly treat these dependent subunits as descriptive, not repetitions.
- Batch time 2743.3235198 seconds and M0 PB mean family times approximately
  .8865-1.5930 seconds match the reported rounding. Process-memory figures
  are identified as observed process working sets, not isolated method costs;
  I did not independently remeasure them.

## Audit receipts and limits of this review

The current index, freeze, and protocol hashes match the summary and receipt
bindings. The saved-decision receipt reports 7,680 families / 32,768 exact
saved-evidence e-BH checks and primary grid rejection nesting. Reading its
helper confirms the scope: four PB outputs per family plus the two Delta
outputs in each drift family, not every comparator's unsaved internal e-values.

The four regenerated-input replay receipts report exact inputs, all declared
baseline method decisions, and primary p/e/reference arrays; their four raw
record hashes match the current files. The replay helper also checks severe-
drift Delta p/e arrays, while the full saved-eBH audit checks the corresponding
decisions. The 16-test frozen XML reports zero failures/errors/skips. These are
inspected receipts and source checks, not reruns by this reviewer.

The actual-invocation receipt's input/artifact hashes match the saved C0/rep0
input and output; it reports seed 112012381454337 and 17 simulated rejections.
The report explicitly does not call them biological discoveries.

Model assumptions, ideal arithmetic/randomization, empirical-only BB_eBH and
practical strong/V1 references, externally bounded Delta protection, unresolved
priority, and missing qualified clinical inputs remain appropriately limited.
No historical V1 claim was reopened. Final integration and sealing remain
main's responsibility; this verdict does not authorize more statistical work.

## Reviewed evidence identity

```text
C001/summary.json
863aa6f4b1126c930d19fd547a5f440e4dfff6b5ca7ce183fec204357381c97f
C001/index.json
4d1cc3d3794248eac991e18dc714c800bdf5e96627ef2ea585467f7a778fd0a7
C001/freeze.json
68068c9f1cf68eb46700b18287b0d226a8a3c2ff1c5ea592e036ee88a2538d8e
RESEARCH_FINDINGS_ZH.md
8F22BC5CEFBF03B58746CDABFA1912CB007B48BB2C13F54DB8BFBE12A1739B30
PAPER_CORE.md
6BB36559792016F611EA4ECD2ED23AA97435F6A4E9FE8A9A9C8D64E863A67179
CLAIM_EVIDENCE.md
43ce3a1c10771f82ff5e48e82155bdfe50b75f9f38b7b38f576b4498adb3d07f
mechanism-summary.json
23bf2983c8247ed9664a7c873e4a85be331b479e97bfe9c5923e9a3c850ede69
confirmation-replay.json
947cce04d7186385a7d99b620bb463e0b424816791c37ce6e31c1af13118a7f6
saved-decision-audit.json
91e8dbc6e81d784363c1b23bde26a4ab64efa59768718b48ebfdfa091818a79e
```
