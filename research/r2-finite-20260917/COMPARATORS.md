# C001 comparator definitions and interpretation

All names refer to the frozen C001 implementation, not historical result
labels with similar names. Each row sees the same simulated z/calibration;
simulation truth, true R, true kappa and generator radial parameters are not
inference inputs. All methods target the same 2G signed r=2-of-4 PC claims.
Nominal family error is .05. No post-result threshold calibration was used.
V1 calibration is transposed to its original (4,N,6) interface, not supplemented
with information unavailable to R2. N still means independent calibration
blocks, NOT4N independent matrices.

| Label | Actual definition | Evidence qualification |
|---|---|---|
| PB_grid | Fourfold TEST/PILOT/TRAIN predictive bridge with fixed2-step affine shape, median matrix-pivot scale,4095 fresh dominating reference tuples, nonnegative projected/ordinary PC mixture, separately PILOT-chosen calibrators normalized on exact rank support, e-BH.05 | Primary R2-PB-1.0.0; full M0 ideal-model argument, THEORY Props5–6. |
| PB_grid_ordinary | SAME scale, reference bank and full20-contrast marginal front; triple Bonferroni min(1,3 min p), max over four triples; ordinary PILOT calibrator normalized on its support; e-BH.05 | Valid same-front ordinary component. Removes projection/mixing as a package, not a different calibration fit. |
| PB_main | SAME p-values, directions, gamma, PILOT functions and references as primary, but continuous unit-integral calibrators without discrete normalization | Valid continuous-calibration ablation, NOT the final primary output. |
| K_NR | Frozen K-NR-1.0.0: twofold fitted conditional normal/Student front, Tyler shape, radial working prior, numerical calibration repair,16 rho-only bootstrap perturbations, guarded projected/ordinary PC mixture, TRAIN-focused evidence conversion, e-BH.05 | Original scoped EMPIRICAL_ONLY remains intact. No complete finite-nuisance theorem is added to it. |
| B_fair_conditional_eBH | Frozen V1's guarded ordinary Bonferroni PC component, same conditional front and stored TRAIN evidence calibrator, e-BH.05 | Fair within-V1 ordinary comparator; EMPIRICAL_ONLY, not the rigorous R2 ordinary component. |
| B_strong | Frozen V1 plug-in conditional front; in each triple keep learned positive profile coordinates; harmonic-corrected Simes intersection, max over triples, TRAIN-focused evidence converter (multiplier.5), e-BH.05 | Actual runnable strong reference. Nuisance fitting and learning have not received the R2 full finite-calibration guarantee. Not an oracle or a per-scenario best-of-methods envelope. |
| R2_main | Earlier confidence-envelope prototype: matrix-pivot upper kappa, finite-reference shape distortion bound, guarded20-contrast tails, restricted-event e-like evidence; internal q0=.035 plus delta_total=.015 | Complete M0 bound at total.05, but almost no Power in key scenarios. Its name is historical; NOT current primary. |
| BB_BY | Same protected ordinary PC with per-test additive nuisance coverage allowance.01, capped at1; fixed BY.05 | Valid marginal classical protection under M0. The.01 p-value floor makes rejection impossible here, since.01>.05/H_512. |
| BB_eBH | Historical BB p-values converted using a TRAIN-learned calibrator | EMPIRICAL_ONLY: marginal BB validity does not justify that reused learning. Gap identified before C001. Kept visible, not used as a validity benchmark. |

For B_strong, let d be the number of positive profile coordinates within
the triple and p_(j) the ordered positive-tail conditional marginal p-values.
Its intersection value is min(1,H_d min_j d*p_(j)/j), H_d=sum(1/j).
If the learned triple profile is all zero it uses all3 coordinates. The
max over triples implements the PC union–intersection structure. This is
not the uncorrected Simes test and not BY applied directly to the final
family. The final operation is learned p-to-e conversion then e-BH.

## Source anchors

- `C001/predictive_bridge.py`: components, evaluate, reference_bank.
- `C001/grid_calibration.py`: support-specific bounded calibration.
- `C001/finite_calibration.py`: envelope/BB constructions.
- `C001/confirm.py`: fixed comparator roster and per-family scoring.
- `../../releases/K-NR-1.0.0/source/src/sca3_compass/molecular_v1.py`:
  evaluate, especially lines120–159 for ordinary/projection/strong selection.
- Same V1 source directory, `robustness_bootstrap_guard.py:69`: components
  defines the harmonic-corrected support-Simes intersection.
- `C001/freeze.json` binds the imported V1 source; provenance.py verifies
  actual source identities rather than trusting the release label alone.

No common leaderboard pools older confirmation/development results. C001
Power comparisons describe real detection costs at equal nominal error;
they do not pretend all practical comparators share the primary theorem.
Actual FDR is always reported by scenario in figures/RESULTS.md. High Power
under demonstrated FDR failure is not accepted as superior performance.
