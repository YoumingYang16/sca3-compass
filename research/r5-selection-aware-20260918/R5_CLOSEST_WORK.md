# Narrow primary-source comparison; G1 OPEN

Checked2026-09-18 by main plus one internal reviewer, NOT external review.
Scope: composite MC nuisance handling, selection-aware borrowing, e-calibration.
Not an exhaustive priority search; lack of an identical algorithm name is irrelevant.

| Source/version and locations | Established content | Required difference / limitation |
|---|---|---|
| Dufour2006, J.Econometrics133:443–477, propositions4.1/4.2, equations4.20–4.26, appendixA.2; https://jeanmariedufour.research.mcgill.ca/Dufour_2006_JE_MCT.pdf | Supremum MC test with common disturbances, ties, nuisance envelope | DIRECTLY supplies A0 validity. Event sweep is an implementation-specific exact optimization lemma, not yet important independent statistical contribution. |
| Kimura–Tamano2602.20503v2, propositions3.6/3.13,theorem3.7,assumptions3.4/3.5/3.12,appendixD.9; https://arxiv.org/html/2602.20503v2 | Data-dependent bounded borrowing with asymptotic size control | Adaptive borrowing not new. Its finite-second-moment/consistent selector limit is not a finite dual-bank guarantee. Main readv1 thenv2D.9; reviewer readv2specifiedtheorems. |
| Li–Zhu–Yang–Wang2602.02703v1,HTMLtheorem5.4/PDF§5theorem1,appendixC; https://arxiv.org/html/2602.02703v1 | Redo selection within exact conditional randomization | Complete reselection not new. Randomized treatment design and sharp target null absent here, so their model-free guarantee cannot be imported. Main verified full appendixC. |
| Lee–Ren2404.17562v1,coverApril29,2024,theorems1/2,propositions1/2,appendixA.1/A.2; https://arxiv.org/pdf/2404.17562v1 | Conditional e-budget preservation and rejection inclusion; practical MC version hasalpha+alpha0 | Conditional joint law is required, not supplied by a marginal R4 pivot. Do not call a multiplier e-BH-CC. Main verified theorem1proof and MC proposition after HTML access failed; reviewer readv1. |

R4's F42 scale law, rotation pivot, fold discipline, discrete calibrator and e-BH
are inherited. The new observable selector, independent inner rank normalizers
and complete outer simulation are an implemented combination. They do not yet
pass the user's important-new-content requirement. A useful, specific adaptation
cost or capability beyond these classical ingredients is still needed.

Reviewer challenge1: branch normalization does not remove the target-only
infinity limit; prove and quantify the loss before claiming recovered efficiency.
Challenge2: coordinatewise minimal envelope among dominating reference arrays is
not an optimal test or power theorem. We explicitly do NOT claim either.
Challenge3: positive-tail truncation, simultaneous events, genuine right limits,
sameH/eigen, DIR bank dependence and PILOT isolation must survive implementation.

EXTERNAL_REVIEW=NOT_CONDUCTED. No claimed world-first result or journal readiness.

## Targeted additions following the actual information/selection bottleneck

- Romano-Wolf2005 JASA100:94-108, section2, example2/equation8:
  https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffd823d949/jasa.pdf
  Main read these parts. Standardized max and least-favorable calibration are
  established; their strongFWER construction is not directly ourPC/eBH chain.
- Lindqvist2022, Conditional Monte Carlo revisited, theorem1/equations6-7 and
  section2.3: https://onlinelibrary.wiley.com/doi/full/10.1111/sjos.12549
  Main checked theorem proof and rejection-sampling section. Generic conditional
  simulation is inherited, not a new A0.3 theorem.
- Fraser, Ancillaries and Conditional Inference, section4.4/equations4.9-4.11,
  https://www.utstat.toronto.edu/dfraser/documents/234.pdf . Main read exact
  location-residual conditional density and flat-prior identity. Our product
  density is a direct specialization; P-CAL's full-residual-space variance boundary
  is an additional calculation, not evidence of priority by itself.
- Lane2020, JRSSB82:1029-1058, DOI10.1111/rssb.12378, sections1.2/1.4 equation4:
  https://academic.oup.com/jrsssb/article/82/4/1029/7056028 . Reviewer read those
  sections. Main obtained the publisher's indexed introduction, but direct
  full-page access failed; do not represent main as having read the full proof.
  Observed information/ancillary-adaptive design is already established.
- Denisov-Zwart2007, JAP44:1031-1046, equation1.1 (reviewer targeted read),
  https://doi.org/10.1239/jap/1197908822 . Breiman-type products supply the known-
  shape conditional tail-index calculation. Actual learned-H moments remain a
  separate question; no t20 claim for the deployed reference.

Current novelty decision: exact drift boundary for the minimal switch-score
repair follows from an elementary order construction plus inherited MC validity.
This is an implementable capability, but important independent content remains
unestablished. P-CAL's arithmetic boundary is mathematically more specific than
generic conditioning; rare extreme patterns and no current Power guarantee
prevent declaring it the practical contribution that passesG1/G3.

## Selective randomization and ordered-scale priority checks (18:17HKT)

Tian-Taylor, arXiv1507.06739v5,2016-11-30; AoS2018 DOI10.1214/17-AOS1564.
https://arxiv.org/pdf/1507.06739 . Main read sections1.1 and4.2/Lemma5 proof;
reviewer also checked2.2equation13. Selected density = original density times
selection probability, and randomization can preserve leftover information.
Their Gaussian information bound is NOT our finite logF Power guarantee. A0.7
randomization, conditional simulation and integrating the coin are classical
operations; not independent originality merely because our model is different.

Misra, Choudhary, Dhariyal, Kundu2002, Metrika56:143-161, "Smooth estimators
for estimating order restricted scale parameters of two gamma distributions":
https://home.iitk.ac.in/~kundu/paper70.pdf . Main read setup and Lemma2.2 proof;
reviewer checked2.2/2.3(b),Theorems2.2/2.4 and3.1. Known gamma shapes and ordered
scales under relative squared loss; smooth improvements over best equivariant
estimators and generalized Bayes identity. Map theta1=kappa_t,theta2=D*kappa_s,
ratio=exp(-s): ordered smooth borrowing and boundary MLR arguments are NOT new.
R5conditional logF is not their observed gamma sufficient experiment (W=0 gives
2expY~BetaPrime(2N,N)); estimation-risk dominance doesn't imply selective tail
dominance, rank validity, FDR or Power. Their theorem therefore doesn't directly
prove ours, but that nonidentity DOES NOT establish important G1content.

Bagnoli-Bergstrom2004 author version, Theorem3/Corollary2, appendix8.2Lemma4:
https://escholarship.org/content/qt0xt8k6sd/qt0xt8k6sd_noSplash_d3b995fc634e40551bbd3dc90a1b0e43.pdf
Main read theorem/proof. Log-concave survival/IFR and monotone likelihood ratios
are inherited tools. FINITE_SELECTION_COST.md adds an explicit finite-threshold
cost for this reference, but its covariance argument is an elementary classical
consequence. No priority/important-originality pass claimed.

## Direct Monte Carlo e normalization priority check (19:54HKT)

Dombowsky,Engelhardt,Ramdas, arXiv2603.15845v1, header16March2026,
HTML displayed manuscript date24August2026 (date discrepancy retained):
https://arxiv.org/html/2603.15845v1 . Main read Proposition1/equation3 and proof,
Proposition5/equation19 and proof, and section3.1 computational caveat. Our F2
soft-rank normalization is directly Proposition1, not original. Infimum over
nuisance values is inherited; their validity extends to infinite parameter sets
while implementation needs further structure. F2b supplies a model-specific
continuous monotone cover, but an elementary cover is not yet important novelty.
Current G1 decision stays OPEN. No MCMC/mixing claims imported: our sampler is
conditional accept/reject with explicit cap and numerical qualifications.

Vovk-Wang, Confidence and discoveries with e-values,2022 author PDF:
https://sas.uwaterloo.ca/~wang/papers/2022Vovk-Wang-STS.pdf . Author-hosted search
extract contains MonteCarlo e normalization equation30; main opened PDF, in-tool
find failed. Do not imply the whole original proof was read. Formula is already
an earlier precedent. Current new capability cannot be based on renaming it.
