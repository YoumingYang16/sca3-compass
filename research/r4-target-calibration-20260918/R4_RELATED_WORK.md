# Targeted related-work and contribution audit

Checked 2026-09-18, directed to the two-bank mechanism only. No claim of an
exhaustive novelty search. Earlier theorem-by-theorem audit is preserved at
../r3-novelty-20260918/R3_NOVELTY_THEOREM_MATRIX.md. Absence of identical code
from an article does not establish priority.

| Source / exact version | What it supplies | R4 relationship / missing prerequisite |
|---|---|---|
| Dufour (2006), Journal of Econometrics133:443–477, Prop2.2/4.1, [author manuscript](https://jeanmariedufour.research.mcgill.ca/Dufour_2006_JE_MCT.pdf) | Finite MC rank tests; nuisance supremum version | R4 uses finite exchangeable ranks after an explicit pivot reduction. Not new MC theory; an estimated plug-in bank alone would not qualify. Reuse previous full-proof audit. |
| Wang & Ramdas, [arXiv2009.02824v4](https://arxiv.org/html/2009.02824v4), Thm5.1 | e-BH controls FDR with dependent marginally valid e-values | Final family step is adopted. Neither kappa estimation nor two-bank validity follows automatically from e-BH. |
| Vovk & Wang, [arXiv1912.06116v4](https://arxiv.org/html/1912.06116v4), Prop2.1 | Decreasing p-to-e calibration | R4 retains the frozen exact finite-support normalizer; no new calibration principle. |
| Ignatiadis, Wang & Ramdas, [arXiv2409.19812v2](https://arxiv.org/html/2409.19812v2), Def4.1/Prop5.4/Thm6.4 | Compound expectation budgets, distinguished from asymptotic budgets | Shared bank randomness is already integrated in each expectation. No mechanical per-gene failure penalty or extra unused budget appears. |
| Jin & Candès, Biometrika113(1), asaf066, [published article](https://academic.oup.com/biomet/article/113/1/asaf066/8250683), eq1, Thm1/2 | Weighted conformal selection; finite FDR given density-ratio conditions, and explicit estimated-weight degradation | Their weight is dQ/dP(x,y)=w(x). A scalar kappa ratio is not a full raw-matrix density ratio. Arbitrary calibration left factors/radii and signed-null shifts prevent simply declaring R4 raw scores exchangeable. Thm2 also requires independent weight learning and a global relative error bound. No direct replacement; not implemented as an invalid comparator. |
| Kimura & Tamano, [arXiv2602.20503v1](https://arxiv.org/html/2602.20503v1), Prop3.2, Thm3.7, Prop3.13 | Worst-case bias correction within external Wasserstein ambiguity sets and asymptotic borrowing | Closest robust-borrowing mechanism: fixed external tolerance trades efficiency for safety. Their mean/Wald framework assumes finite variance and large-sample conditions; R4 instead uses an exact finite log-F error and angular/radial invariance. Does not supply R4 finite PC guarantee. The conceptual borrowing/bias correction is established, not claimed new here. |
| Kopp-Schneider et al., [2020 borrowing limitation paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7079072/) | UMP/UMP-unbiased testing restrictions on uniform borrowing gains | Relevant caution, not a theorem that all R4 borrowing is impossible. Full text access was inconsistent (recaptcha); do not assign an unread theorem number. R4 T4/T5 are derived explicitly and do not rely on an imported general impossibility claim. |

Weighted conformal publisher version was read at eq1 and Thm1/2, including the
proof outline/conditional independence requirements. Full arXiv HTML requests
failed and were not treated as read. BOND sections3.2–3.4 were read; its
asymptotic guarantee is not re-labelled finite sample. No software was installed
from either paper, and no new broad algorithm family was added.

## Claim-to-prior classification

| R4 item | Classification | Evidence still needed |
|---|---|---|
| Direct Ct→R3 target-only | Existing method with additional valid information | Endpoint identity and availability disclosure, not novelty |
| Fixed bridge exact reference | Classical equivariance / independent-product specialization | Correct complete dependency chain and useful finite experiments |
| Count weight yields Ntotal error law | Algebraic identity / variance minimizer within fixed bridges | Not a detection-Power optimality result |
| Slack log-MSE threshold | Scoped analytic bias–variance consequence | Mechanism comparison, not a new general theorem |
| Schur-complement information Nt/2 | Classical nuisance information calculation | Narrow regular pivot-only class; not new2/N result |
| Exact Hellinger affinity and two-point target sample requirement | Explicit finite-sample boundary for this reduced experiment | Priority unestablished; raw matrices may contain more information |
| New paired target/source/bridge/allocation experiments | Potential reproducible system-comparison contribution | Frozen complete result and independent confirmation required |
| Efficient primary-only kernel and strict schema | Engineering | Exact endpoint regression, source binding, failure records |

No independently original general statistical theorem is established at this
stage. A defendable research object is the finite-calibration borrowing tradeoff
and its explicit information requirements, with a carefully scoped method and
comparison study. The final usefulness assessment must follow actual results,
not the length of this bibliography or the appearance of the word "theorem".

One additional narrowly targeted check used the queries "Hellinger affinity
beta prime scale distributions closed form Renyi divergence" and "log F
distribution Hellinger affinity hyperbolic secant Fisher information". The
returned general divergence/beta references did not establish priority of this
particular scaled-F42 identity. That limited search is **not** evidence of
originality. The derivation and minimax-rate corollary use classical two-point
arguments; the claim remains an explicit scoped consequence with unestablished
priority, not a newly discovered information-theoretic principle. No unrelated
distribution-estimation method was added to the implementation.
