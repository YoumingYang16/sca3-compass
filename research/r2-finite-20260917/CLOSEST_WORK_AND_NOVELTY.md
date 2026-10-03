# Closest-work matrix (targeted primary-source check)

Final status: TASK_SPECIFIC_SYNTHESIS_PRIORITY_UNESTABLISHED. The completed
primary is the predictive bridge with discrete calibration, not the initial
confidence envelope. Dated development entries below retain their original
stage; the final construction is described in the last section. C001 now
confirms specific discrete-calibration gains and substantial remaining
efficiency costs; see CLAIM_EVIDENCE.md. No new priority claim follows.

2026-09-17. No claim of exhaustive search or established priority. Source
versions are recorded below; articles were read at the relevant theorem and
proof, except where explicitly marked inaccessible. No external code installed.

| Source | Verified relevant result / assumptions | R2 use and difference |
|---|---|---|
| [Wasserman, Ramdas, Balakrishnan, Universal Inference](https://arxiv.org/html/1912.11436), Thm1/3 and proof | Independent likelihood fitting yields an expectation bound; a genuine null likelihood supremum gives finite-sample tests without Wilks regularity. | Valid alternative reference principle. Here arbitrary radial law is removed by invariance; no global fitted likelihood optimum is available or claimed. Not adopted as an extra algorithm family. |
| [Wang–Ramdas, e-BH](https://arxiv.org/html/2009.02824v4), main FDR argument | Nonnegative null-mean-bounded e-values allow arbitrary dependence among final evidence. | Adopt final self-consistency argument. Does not certify plug-in nuisance fits or dependence between a held gene and its TRAIN. |
| [Ignatiadis–Wang–Ramdas, compound e-values](https://arxiv.org/html/2409.19812v2), Prop5.4, Thm6.4, proof context | Good-event restricted expectation budgets imply approximate compound control; additive failure allowance enters FDR. | Our crossfold good-event proof is an application, NOT a new general theorem. Task-specific work must lie in nuisance elimination/protection and useful efficiency. |
| [Lee–Ren, conditional calibration](https://arxiv.org/html/2404.17562v1), Sec3 Thm1/2, Prop1 | Boosting requires the correct conditional joint null law given sufficient information; finite Monte Carlo approximation needs its own error control. | Not directly applicable to arbitrary-radial learned PC outputs with unknown shape. Adding this name would not close that missing conditional law. Do not implement speculative boosting. Author's [code entry](https://zhimeir.github.io/publications/) located, not executed. |
| [Dufour, finite-sample Monte Carlo tests](https://jeanmariedufour.github.io/Dufour_1995_MCT_W.pdf), Lemma2.1/Prop2.2 and proof | Exchangeable pivotal reference ranks give finite-simulation control. | Adopt fresh Gaussian shape-reference rank cutoff. This is not bootstrap, a simulated quantile theorem or an invention of rank testing. |
| [Benjamini–Heller, PC](https://www.math.tau.ac.il/~yekutiel/eBayes/BH%202008.pdf), Biometrics64 | Null allows fewer than r nonnull studies. | Existing union–intersection target, retained, not novelty. PDF intermittently unavailable in this turn; complete PC argument is independently supplied in THEORY. |
| Berger–Boos1994, DOI10.1080/01621459.1994.10476836, [university preprint](https://repository.lib.ncsu.edu/items/1e8c580a-c3db-46f8-93a3-9b963e2c9171) | Confidence-set maximization plus coverage-error penalty. Full primary text blocked by repository bot check this turn; not falsely marked read. | Classical per-test penalty implemented from an explicit independent proof. Do not claim a detailed theorem/notation from unavailable pages. |
| Tyler1987; [Lau–Ramachandran2025](https://arxiv.org/abs/2510.13751) | Tyler shape and finite-sample analysis already exist. Latest paper abstract verified; detailed optimal-bound theorem not yet read/used. | Do not claim first finite-sample shape protection. Our exact finite-iteration distribution is established by explicit affine induction rather than invoking unverified rate constants. |

## Actual research delta at prototype stage

- Classical adoption: scalar/matrix Student–Wishart identities, order coverage,
  fixed Tyler updates, NNLS direction, PC, p/e calibration, e-BH and MC ranks.
- Task-specific construction being tested: protect *unknown* R by an affine
  finite-iteration pivotal distortion bound while retaining all20 contrast
  coordinates, remove radial nuisance rather than fitting it, and close the
  entire crossfold dependency chain with a family-level failure budget.
- New-to-project knowledge, not priority: block-vs-row calibration requirements;
  per-test additive-penalty/BY floor; relative kappa width scale invariance;
  exact computational costs and efficiency vs V1/strong references.
- Not established: a globally original statistical principle, dominance,
  minimax/optimality, a clinical application, or a high-impact publication claim.

NOVELTY_STATUS: TASK_SPECIFIC_CONSTRUCTION_UNDER_REVIEW; foundations existing.
Do not upgrade this status merely because simulations or code tests pass.

### 13:32 update: closest finite-shape result checked

Read [Lau–Ramachandran full text](https://arxiv.org/html/2510.13751v1),
Definitions2.1/2.2, Theorems1.1/1.2 and the proof roadmap/invariance argument.
Their relative spectral error bound has optimal-order sample dependence with
implicit constants; it is not the explicit finite reference cutoff used here.
Our finite-iteration equivariance proof must not be represented as a first
Tyler invariance or finite-sample covariance guarantee.

D001 demonstrates the separate confidence-envelope strategy is too costly.
The single next correction is a dominating full-contrast predictive
rank test (PREDICTIVE_BRIDGE.md), plus an independent PILOT role. Monte Carlo
rank validity and splitting are classical; its task-specific joint nuisance
reduction and utility still require review and experiments. No broad literature
or alternative-algorithm search has been reopened.

### Current primary and discrete efficiency correction

The full predictive bridge and separate PILOT proof have now been reviewed
(REVIEW_BRIDGE.md), and discrete normalization reviewed(REVIEW_GRID.md).
[Vovk–Wang, E-values: Calibration, combination, and applications, v4](https://arxiv.org/html/1912.06116v4#S2),
Prop2.1 and proof, was read directly. Its decreasing integral criterion
already implies the right-endpoint step extension used by our grid correction.
Neither monotone calibration, rank tests nor convex evidence merging is new.

Task-specific research object: jointly integrate a finite matrix-scale median
and finite-iteration affine shape distortion into full-contrast learned
directional PC; explicitly separate PILOT to make that marginal construction
compatible with learned e-calibration; quantify residual costs. The chain is
more than a V1 bug fix: it changes the inferential law, dependency design and
guarantee. But whether an equivalent task-specific construction exists has
not been exhaustively settled. Closest papers do not by themselves establish
our priority. Current NOVELTY_STATUS remains TASK_SPECIFIC_SYNTHESIS_PRIORITY_UNESTABLISHED.
The calibration/contrast-only mismatch example is elementary and not a new
full-input information lower bound. No new broad literature search is planned.
