# R3 bounded loss-diagnostic review

2026-09-17. Sole independent reviewer. Read the user's pasted attachment in
full, then `loss_diagnostic.py`, `protocol-loss.json`, `r3_common.py`, and the
directly relevant frozen R2 source/theory. No project code, tests, fits, or
experiments were run. No new agents, literature route, or candidate mechanism
was opened. Only this review file was written; R2/V1 remain untouched.

## Disposition

**Mathematical diagnostic design: PASS under matched M0 and ideal arithmetic /
randomization. The three oracle variants have valid ideal-model interpretations
but must remain DIAGNOSTIC_ONLY.** Their values are not deployable R3 results or
independent confirmation.

**Two engineering identity gaps remain before accepting the frozen 240-family
ledger.** They can be corrected without changing the four diagnostic laws or
adding observations. No source contamination is alleged. If a diagnostic is
already frozen/running, preserve it and use a separately recorded fail-closed
identity verifier; do not silently replace frozen files or relabel execution.

## Concrete remaining findings

### B1. Source rows are not linked to the sealed C001 index

Locations: `loss_diagnostic.py:60-70,97-103`;
`r3_common.py:12-19`.

`verify_r2()` establishes the sealed manifest, frozen dependencies, and original
index identity. But `one()` never compares the selected source record's hash
against that verified index. Nor does it check that the row's case, repetition,
algorithm seed, freeze identity, and canonical artifact paths match the
requested source task. Comparing input/evidence hashes only to values inside
that same row is not the missing index-to-record binding. A misfiled but
internally consistent source record could replay perfectly and be attributed
to the wrong case/repetition.

Correction: look up each requested pair in the verified C001 index, require its
canonical record path/hash/status, then use the existing C001 `check_record`
against its frozen protocol (including metric rescore). Record the source
index/protocol/freeze identity as well as the input/evidence hashes. Assert the
R3 protocol's declared R2 manifest equals the verified manifest. These are
checks on the already selected 240 sources, not another simulation or broad R2
audit.

### B2. Frozen executor and completed diagnostic identities are incompletely checked

Locations: `loss_diagnostic.py:110-128,130-153`.

`run()` checks the copies in `out` but not the actually executing diagnostic and
`r3_common` modules. Invoking an edited parent script with an unchanged archived
copy can therefore pass these checks. Workers have no corresponding initializer
check. `analyze()` checks only total row count, record hashes, and completed
status. It does not verify the R3 freeze/protocol, exact planned Cartesian set,
row/index identities, saved-array hashes, or re-score saved decisions. Duplicating
one index entry and omitting another can satisfy the count check and silently
bias case means; a per-case `n` is reported but not required to equal 24.

Correction: execute/verify the frozen copies and their dependency paths in each
worker. Before summary, require the exact set of ten declared cases x reps
0..23, unique canonical record paths, matching index/row/freeze identities,
array hashes and expected four-label membership, and raw-decision/metric
agreement. Bind summary to protocol, freeze, and source R2 index as well as its
own index. Keep failures blocking; no success-only subset. The existing replay
checks are valuable but do not substitute for this identity chain.

The constants 4095 references, 128 TRAIN blocks, two shape updates, 512 signed
hypotheses, and q=.05 currently match the chosen frozen C001 inputs. They are
hard-coded rather than driven by several protocol fields. Assert that exact
contract so a future metadata edit cannot pretend to change the implementation;
no additional settings or searches are requested.

## Oracle laws and coupling: checked

Locations: `loss_diagnostic.py:17-50,74-86`.

Let B=kappa_hat/kappa, H0=R^(-1/2) H R^(-1/2) up to scalar, U_j iid chi-square5,
U=sum_j U_j, and

    V_H = lambda_min(H0) * sum_j U_j/lambda_j(H0) / 20.

The inspected four-way mapping is correct:

| Diagnostic | Held inference shape / kappa | Reference statistic |
|---|---|---|
| R2 | H / kappa_hat | Z / sqrt(B V_H) |
| known_kappa | H / true kappa | Z / sqrt(V_H) |
| known_shape | true R / kappa_hat | Z / sqrt(B U/20) |
| known_both | true R / true kappa | Z / sqrt(U/20), Student20 |

The original H still chooses every projection direction via `direction_shape`;
the shape substitution changes its variance normalization/contrast energy, not
the direction. Original profiles, PILOT calibration receipts, gamma, fourfold
assignment, grids, PC aggregation, and final q remain fixed across labels.

For known_kappa, the same Rayleigh bound removes the selected direction from the
reference law, with B=1. For known_shape, whitening by true R gives U~chi-square20
and removes the shape envelope; standardized centered directional numerators
remain standard normal even though the direction was learned using H and
calibration. Their conditional law does not depend on those learners. For
known_both the remaining pivot is Student20. Nonpositive intersection shifts
only decrease the positive tail, and `rank_tail` retains p=1 on nonpositive
statistics. Hence the ordinary Bonferroni/PC and projection-PC p-values are
superuniform under their corresponding ideal oracle experiment.

Conditional on the claim's independent PILOT, these arguments still integrate
over the permitted TRAIN/calibration/reference randomness. The saved decreasing
calibrator and convex gamma mixture then retain null mean <=1, and e-BH handles
the final dependence. **Holding receipts equal across paired labels does not
license conditioning the validity proof on all those receipts simultaneously.**
Likewise, selecting a favorite oracle after seeing these diagnostics is not a
new valid data-adaptive procedure.

`banks()` consumes the same Gaussian-shape, F4,2, chi-square, and normal draws in
the same batch order as frozen R2. It consumes even the draws unused by an
oracle, preserving pairing. Each reference tuple still has the required
independence; sharing tuples across labels does not invalidate any label's
marginal rank calibration. Sorting each bank does not destroy this coupling.

Baseline reference equality at line 70 and p/e equality at lines 84-85 are
explicit exact checks. The copied operation order matches the frozen code on
inspection; I have not executed it or claimed a runtime pass. Identical evidence
with the identical frozen e-BH function entails identical decisions. Also
compare against the saved original decisions/metrics when closing B1/B2 to make
that baseline requirement explicit in the receipt.

## What these results can and cannot identify

- This is a paired two-by-two nuisance diagnostic, not an additive partition.
  If the four powers are P00, P10, P01, P11, the interaction
  `P11-P10-P01+P00` need not vanish. Single removals must not be added and
  reported as recovered total loss. Neither finite-realization gains nor a
  universal oracle power ceiling are proved monotone by this design.
- Known kappa removes its inference normalization error and its integrated
  reference factor together, but keeps any calibration effects on the original
  learned directions. Final R2 uses a sampled median-ratio law, not a kappa
  confidence interval or repeated union-bound budget. Do not describe this
  contrast as tightening a confidence set or separately count attachment
  categories A/B as two measured losses.
- Known shape jointly removes fitted-metric error and the lambda_min
  worst-direction envelope. This contrast does not identify which of those
  two costs dominates, or how much is recoverable by an implementable method
  with unknown R. Known both still has rank, PC, calibration, splitting, and
  direction-learning costs.
- Splits and learned directions are unchanged, so their individual losses are
  **not identified**. Both raw component p arrays at lines 90-91 are already
  partial-conjunction p-values; they cannot isolate the PC-combination cost.
- `e>=20` and component `p<=.05` are validly labeled as lacking a family FDR
  guarantee. They are relaxed screening counts, not recoverable valid power.
  Changing to those thresholds does not isolate a legal p-to-e or final-e-BH
  efficiency gain while holding the error criterion fixed.
- The fixed projection-only and ordinary-only e-BH diagnostics retain the M0
  proof (fixed gamma endpoints). They assess the current mixing choice versus
  its endpoints, not optimal direction learning or a new projection family.
  Do not choose an endpoint using the same results and call that choice a
  confirmed candidate.
- Rank-floor occupancy and below-floor pilot foci are descriptive compatibility
  measures, not quantified counterfactual power losses. Reference count and
  discrete calibration are unchanged in these four labels. The earlier R2
  continuous/grid comparison remains separate evidence for that specific layer.
- Twenty-four reused, already observed C001 families per case justify the
  stated descriptive paired means/MCSEs only. This is R3 development information,
  not a second independent R2/R3 confirmation, finite-sample FDR validation,
  minimax lower bound, or proof that the strong-reference gap is fully recoverable.

The bounded diagnostic is an appropriate first step toward the user's loss
ledger. Mark unmeasured layers explicitly; do not claim that four nuisance
labels already resolve all A-J categories. No additional route is recommended
by this review.

## Source attribution and reviewed identities

The executable is a transparent diagnostic specialization of frozen
`R2/C001/predictive_bridge.py:39-92,103-132`, the retained `finite_calibration`
helpers, and `grid_calibration`. The ideal oracle arguments specialize R2
THEORY Propositions 1-3, 5, and 6 and classical Gaussian/Student pivots; neither
paired nuisance substitution nor these specializations establish new priority.
The original R2 references/qualifications should remain attached to the ledger.

```text
user attachment pasted-text.txt
8368C070E1865F9F38302080051D8991A9F1098BBC624476D80543B6ED77CB9E
loss_diagnostic.py
65601B5FD3BDA5EE1BECC00D7E6C73193AE1F4C1D0B760BFC3CFD22611C9320E
protocol-loss.json
122EC3313E506247122A33162BD7A6F82C797547C349FAF078BFBC423479C764
r3_common.py
A5209F87D11C8AA3232B1A4010C3F167A7D1E6E5B12F8E99E56A6C83A3C807CD
R2/DELIVERY_MANIFEST.json
51AD4505B7D74D90ED9C0164A8D359C3BB6238DB6B2EB08BA741D9F7D1B78073
```

## Completion addendum: saved D001 result attribution

After the main reported completion, I read `D001/summary.json`, `index.json`,
and `freeze.json`; no diagnostic code or new numerical experiment was run.
The index reports 240 completed records, zero non-completed statuses, and
32.4926601 seconds. All ten summary rows have n=24. The three frozen source/
protocol hashes match the inspected versions above, and the summary's index
hash matches the actual index. Successful records necessarily passed the
frozen baseline reference and p/e equality checks. This is not a claim that I
independently reran those checks or audited all 240 raw records. B1/B2 still
need the separate saved-evidence identity verification described above, not a
rerun or retroactive edit of D001.

The main's four quoted known-kappa gains match the saved paired summary. All
entries in the following table are **percentage points**, relative to R2 on
the same 24 families; MCSE is descriptive, not a confidence bound:

| Scene | Known kappa gain (MCSE) | Known shape gain | Known both gain |
|---|---:|---:|---:|
| Normal | 10.38 (5.46) | 7.84 | 22.30 |
| t5/N4 | 21.73 (6.28) | 3.27 | 29.66 |
| t5/N12 | 16.67 (6.03) | 6.54 | 26.72 |
| t3/N12 | 11.27 (3.33) | 7.03 | 17.16 |

Specific attribution cautions for the loss ledger:

- “Known shape gains 3-10 pp” describes the six core normal/t5/t3 scenes, not
  all ten. Across all ten it ranges **1.39-9.64 pp**; dense and t1.5 gains
  are 2.45 and 1.39 pp. Keep the stated scene domain explicit.
- This 24-family subset is not the full 512-family C001 estimate. For example,
  normal R2 power here is 30.80%, known-kappa 41.18%, and strong 72.71%.
  Do not add the 10.38-pp gain to the full-confirmation R2 value 25.69% or
  subtract it from the full-confirmation strong gap and call that a paired
  recovered fraction.
- Nonadditivity is visible even in point estimates: the normal interaction
  is about +4.08 pp; t5/N128 is about -3.27 pp. Report joint improvement
  directly rather than summing single-removal improvements.
- Kappa sampling is a defensible **priority hypothesis**, especially for these
  small-calibration scenes, not a proved universal dominant bottleneck.
  Continuous-effects known-kappa gain is only .33 pp with MCSE 2.85 pp;
  at t5/N128 the shape gain 9.64 pp exceeds the kappa gain 4.58 pp.
- The second selected problem should be called “unknown-shape / spectral-
  protection efficiency,” with tightening the Rayleigh envelope a proposed
  mechanism. These oracle results alone do not isolate the envelope's share
  from fitted-metric error. A forthcoming proof must justify any tightening;
  the data do not establish it or promise a deployable gain.

Selecting these two questions stays within the attachment's bounded first
stage. Neither small observed FDPs nor the oracle gains establish R3 success,
an attainable upper envelope, or a necessary finite-guarantee loss. The
scientific attribution is acceptable with these qualifications; no expanded
analysis or additional route is requested.

```text
D001/summary.json
2FDA7EB53B372EFC6B3D74C93F4013FFBAADFFEF99958FA9BAF3D67DABEFCEAD
D001/index.json
45C4B7846283B76B85D0B7D2D074D67320160B1A667E5217D75E777F22CDF81E
D001/freeze.json
2E8D2CBB159BB80859FF6E9E58AEA4D2EB9AEBBED1FA8F45E9D23C5AB6AD5B47
```

## Identity closure received after completion

Read `audit_d001.py` and `D001-identity-audit.json`; did not execute the audit.
The receipt's D001 freeze/index/summary hashes match the files inspected above,
and its R2 index/protocol identities match the sealed sources. The helper checks
the exact planned task set, canonical source-record/index linkage, original
seed/freeze/artifact identities through C001 `check_record`, diagnostic array
membership, 960 saved e-BH decisions and raw scores, baseline score equality,
and the four-label summary means. It reports all 240 families passing.

**B1 and the saved-evidence portion of B2 are closed by this separately recorded
post-run audit. No outstanding saved-evidence blocker remains from those
findings.** Historical workers did not record contemporaneous module-path
attestation; the receipt explicitly preserves this limitation rather than
inventing a retrospective witness. I accept that qualification for this
development diagnostic, not as proof that an absent historical attestation
existed. No D001/R2 files or observations were changed by this review.

`D001-identity-audit.json` SHA256:
`54D0697F999F9D761DCCCB57686613CD316AD38ED242FFDAFAF09ED93A0C875F`.
