# Theorem-level audit (2026-09-18)

YES/NO/PARTIAL refer to whether the named result directly supplies the R3
step, not whether one article prints the entire R3 program. Absence of that
program from an article is not evidence of an original theorem. Numbering
below is tied to the linked version; some published versions differ.
Main agent read the stated theorem/proof sections, not just abstracts.

## Six required families and close extensions

| Source; precise result read | Scope and assumptions | R3 mapping; direct consequence? |
|---|---|---|
| Lee & Ren (2024 preprint), [Boosting e-BH via conditional calibration](https://arxiv.org/html/2404.17562v1), Thm1, eq3–5, AppendixA.1 | Finite marginal e/generalized-e validity after conditioning on S_j and controlling the actual conditional expectation of the JOINT evidence vector's rejection functional; no arbitrary plug-in conditional law. Thm2 supplies rejection-set inclusion. | PARTIAL: existing R3 e-values qualify as inputs, but R3 does not implement the conditional joint calculation. The minimal missing object is a computable uniform-null conditional bound for phi_j(c;S_j), including other means, cyclic learning, shared B and shared references. Scalar reference W does not supply that object. This blocks a DIFFERENT boosting method, not the already-proved R3 validity. |
| Wang & Ramdas (2022 JRSSB), [FDR control with e-values](https://arxiv.org/html/2009.02824v4), Thm5.1, Lemma6.1(iii), Thm6.2 and proof | Finite FDR under arbitrary dependence of nonnegative evidence; valid marginal expectations or bounds on the threshold-rounded transform suffice. No knowledge of covariance required once these bounds hold. | YES for R3 final e-BH. Distribution-aware boosting is existing theory but requires a proved expectation budget. It does not estimate kappa or build W; no claim that all R3 steps follow from e-BH alone. |
| Ignatiadis, Wang & Ramdas, [Asymptotic and compound e-values](https://arxiv.org/html/2409.19812v2), Def4.1, Prop5.4, Thm6.4 | Exact sum of null expectations<=m gives finite FDR; approximate common-event budgets give alpha(1+epsilon)+delta; asymptotic versions are separately identified. Arbitrary evidence dependence. | YES: R3 is an ordinary, hence compound, e-vector. Sharing external calibration creates dependence, not an unhandled budget. R3 already has individual unconditional bounds and pays no coverage-event penalty. No new compound theorem or unused boost budget follows. |
| Wasserman, Ramdas & Balakrishnan (2020 PNAS), [Universal inference](https://arxiv.org/html/1912.11436), Thm1/3, eq7/9 and proofs | Finite type-I/coverage via independent training density divided by a held-data null likelihood supremum. Allows nonparametric families when genuine densities and the supremum can be evaluated. | PARTIAL: can create a different valid test under suitable computable likelihoods; it does not yield R3's statistic or efficiency. Arbitrary unmodelled radial densities and composite directional nulls are not supplied by a Student plug-in. Splitting alone is insufficient; no new R3 split-likelihood result. |
| Dufour (2006 J Econometrics133:443–477), [Monte Carlo tests with nuisance parameters](https://jeanmariedufour.research.mcgill.ca/Dufour_2006_JE_MCT.pdf), Prop2.2/4.1 and Appendix proofs | Exact finite ranks for exchangeable null statistics, at any fixed MC count; full measurable null supremum protects unknown nuisance. Estimated-set/plug-in asymptotic results are separate. | YES for ranks AFTER invariant reduction; PARTIAL for the whole construction because reduction must be verified. R3 reproduces the finite sampling error law, not a bootstrap approximation. R2's spectral domination and R3's invariant equality avoid a numerical supremum. |
| Benjamini & Heller (2008 Biometrics64:1215–1222), [Screening for partial conjunction hypotheses](https://www.math.tau.ac.il/~yekutiel/eBayes/BH%202008.pdf), Sec2 Lemma1, Thm2/eq3 and adjacent union-bound proof | PC null permits r-1 arbitrarily strong alternatives; combines surviving nulls. Their Thm1 Simes version requires D1–D3 conditions; Thm2 Bonferroni is valid without them. | YES for ordinary Bonferroni-PC principle. Directional elementary nulls are permissible. Not a proof of R3's nuisance-calibrated base tests. Full PDF obtained; OCR is imperfect, use typeset PDF for symbols. |
| Wang & Owen (2019 JASA; linked2016 author version), [Admissibility in partial conjunction testing](https://arxiv.org/html/1508.00934), Def6, Prop3.1 and proof | Maximum over all n-r+1 intersection tests is PC-valid under any dependence IF every intersection test is valid uniformly over its composite null. Admissibility results have additional assumptions. | YES for max-over-triples proof. R3 projection is a raw-data intersection test; the same event inclusion applies even though it is not necessarily a function only of marginal p-values. Learning requires the upstream conditional validity already established. No new PC theorem. |
| Vovk & Wang (2021 Ann Stat49:1736–1754), [E-values: calibration, combination, applications](https://arxiv.org/html/1912.06116v4), Prop2.1 and proof | Decreasing nonnegative p-to-e function with uniform integral<=1; deterministic convex mixtures preserve validity. | YES, with elementary discrete-support integration and independent-PILOT conditioning. Exact rounding is numerical work, not a new calibration principle. |
| Dey, Banerjee, Bhuyan & Majumdar (2025 author preprint), [PC across dependent studies](https://arxiv.org/html/2511.04130v2), Lemma2/AppB.2; Prop1 and Thm1/2 hypotheses | e-PCH averages smallest n-r+1 elementary e-values; arbitrary dependence, finite FDR. The more aggressive e-Filter has extra distribution/filter conditions and different finite/asymptotic bounds. | PARTIAL: PC e-values plus e-BH already exist. Their elementary validity input does not solve R3 calibration/shape law. Do not promote the aggressive e-Filter as a same-scope comparator without checking its extra conditions. |
| He, Jiang & Sun (2026-09-07 preprint), [Structure-Adaptive E-Value Filter](https://arxiv.org/html/2609.07246v1), Property1, Thm4.1/AppC.3; Assumption1 and ThmA.1 | Learned score construction must preserve joint pair swaps (or pooled null/calibration exchangeability). Flip signs and stopped counting process yield PC e-values, then existing e-BH. | PARTIAL: finite calibration + learning + PC is not by itself a novel combination. R3 does not have their raw score exchangeability: calibration may have different radii/left factors and PC null allows nonzero negative effects. R3 uses a reduced pivot instead. This is a model distinction, not proof of a new theorem. |

## Availability limits (not silently treated as read)

Berger & Boos (1994 JASA89:1012–1016), DOI
[10.1080/01621459.1994.10476836](https://doi.org/10.1080/01621459.1994.10476836):
author repository returns a bot page with HTTP200; full paper NOT read.
The elementary protection is independently checked: for a1-beta coverage set
C and point-nuisance valid p_theta, min(1,sup_C p_theta+beta) is superuniform
by union bounding failure of C and rejection at the true theta. R3 is not
this construction. Dufour's full primary proof covers nuisance/rank audit.

He You et al. (2010), [CFAR assessment of covariance matrix estimators for
non-Gaussian clutter](https://doi.org/10.1007/s11432-010-4080-z): public PDF
requests returned418/429; publisher abstract/reference information only.
Do NOT assert exact equivalence to their adaptive normalized matched filter
or cite an unread theorem number. C decision below does not depend on this
paper being identical: the rotation corollary is proved explicitly in the
mathematical-object file. No paid access or anti-bot bypass attempted.

## Explicit six-question conditional-calibration check

### Data-structure assumptions, not keyword matches

Y=explicit structure; G=generic theorem accepts it only after its input
validity/conditional-law hypothesis is supplied; N=not supplied by that work.
These flags are not claims that G proves the upstream property automatically.

| Result | Finite external calibration | Multiple studies/direction | PC | Learned direction/shape | Heavy tails | Shared calibration uncertainty |
|---|---|---|---|---|---|---|
| CC Thm1 | G | G | G | G | G | G; joint conditional budget required |
| e-BH Thm5.1 | G | G | G | G | G | G; marginal evidence bounds sufficient |
| Compound Def4.1/Thm6.4 | G | G | G | G | G | G; actual sum budget required |
| Universal Thm3 | G via likelihood | G via composite null | G | Independent numerator training; true denominator supremum | G if density family valid | G; splitting/likelihood argument required |
| Dufour Prop2.2/4.1 | G as simulated nuisance | G | N upstream | G if statistic is pivotal or uniformly protected | G if exact simulatable null | G for each marginal rank; no fixed-bank conditional claim |
| BH Thm2 / Wang–Owen Prop3.1 | N upstream | Y / arbitrary elementary hypotheses | Y | N upstream | G through valid input tests | G; no study-independence requirement for these particular results |
| e-PCH Lemma2 | N upstream | Y / arbitrary elementary hypotheses | Y | N upstream | G through valid elementary evidence | G; dependence allowed |
| SEFT Thm4.1/A.1 | Y, with pair or pooled exchangeability | Regional elements; hypotheses can be directional | Y | Learned scores must preserve swaps, not R3 shape law | G if required symmetry holds | Y under specified joint symmetry |

R3 raw external calibration is **not exchangeable** with all target signed
null blocks: left transforms/radial laws may differ and signed null locations
can be negative. Its pivot reduction, disjoint roles and invariant shape law
are essential prerequisites; citing generic G entries alone is not a proof.

Finite N / nuisance / learned direction / learned shape are not prohibited
by Lee–Ren's abstract theorem; their effects must be included in a valid
conditional rejection-budget calculation. Directional PC can label H_j, but
its union of intersections and unrestricted other effects do not magically
give a known joint law. Same-grade FDR would follow IF that calculation were
proved. Replacing R3 e-values by a scalar pivotal rank is not that calculation.
Consequently NO direct swap; this does not justify B because R3 neither needs
nor solves the missing boosting calculation.

## Minimal proposition test and originality conclusion

The sharpest candidate for Theorem X is the JOINT direction-quadratic-form /
shape-spectrum law with finite independent B. Rotation maps b to e1 without
changing eigenvalues; affine/radial invariance removes nuisance; independent
product laws then give W. Existing exact ranks, union-intersection PC,
conditional calibration-by-independent-PILOT, and e-BH finish the argument.
There is no remaining mathematical obstruction in that chain. Hence the
currently defensible classification is C: a specialised construction and
comparative study, not an established independent new statistical theorem.
This is not a claim to have proved worldwide absence of any publishable
specialisation, nor an assertion that one prior article contains all R3 code.
