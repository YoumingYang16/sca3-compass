# WP1 algebra / engineering / cost diagnostic, fixed before observations

2026-09-17. Not a Power confirmation. No biological data.

1. Run deterministic tests of finite-iteration affine equivariance,
   block-radial invariance, batching, kappa block counting and scale envelope.
2. Benchmark exactly one shape reference at Gtrain=128,S=4,J-1=5,
   B=199 synthetic references, 12 fixed updates, seed17091711.
   Report cutoff, reference range and elapsed time. It is an engineering
   diagnostic, not a coverage estimate and cannot prove invariance.
3. Read-only literature review checks the bad-event FDR lemma against
   approximate/compound e-values (already found Proposition5.4 equivalent
   good-event condition, so the composition itself is NOT original).

CPU<=2, expected seconds to minutes, no parallel agent simulations. Do not
launch DEV performance until full dependence proof reviewed and recorded.
Stop benchmark after one prescribed call; no tuning cutoff by its output.
