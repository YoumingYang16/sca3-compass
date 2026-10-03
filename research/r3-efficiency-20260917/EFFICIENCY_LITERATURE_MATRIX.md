# Targeted efficiency literature and applicability

2026-09-17. Read the specified primary HTML theorem/proof portions, not just
abstracts; no claim of exhaustive novelty search. This matrix precedes R3
candidate implementation. No external library or research code was installed.

| Original source / portion inspected | Loss addressed and guarantee | Mapping to this project / decision |
|---|---|---|
| [Lee–Ren, Boosting e-BH via conditional calibration, v1](https://arxiv.org/html/2404.17562v1), Sec3 equations3–5, Thm1 and conditional-expectation argument | Conditional redistribution preserves ordinary or generalized evidence budget; requires correct conditional JOINT law of all evidence under the particular null, not just a marginal p bound | R2 only provides PILOT-conditional marginal domination, not the e-vector law given a sufficient statistic for a composite directional PC null. Cyclic TRAIN/PILOT and unknown other means remain. Cannot substitute pivotal scalar references into that joint expectation. Not implemented. |
| [Ignatiadis–Wang–Ramdas, Asymptotic and compound e-values, v2](https://arxiv.org/html/2409.19812v2), Def4.1,Prop5.4,Thm6.4 | Exact sum of null expectations<=m suffices; approximate budgets imply alpha(1+epsilon)+delta; asymptotic is different | R2 already incurs no individual failure-event charge and shares random references. No certified unused global budget has been derived. Low observed FDP or a large shared calibration sample does not supply it. No arbitrary multiplicative boost. |
| [Wang–Ramdas, FDR control with e-values, v4](https://arxiv.org/html/2009.02824v4), Sec4.2,Sec5/6 and self-consistency | Final FDR from expectation bounds; distribution-aware marginal boosting needs a valid expectation of its threshold-discretized transform | Potential existing efficiency reference, not novelty. Our current primary calibrator is already support-normalized. Any further boost needs a model-bound on the transformed statistic, not the empirical alternative mixture. Not activated: selected bottlenecks are scale and shape. |
| [Vovk–Wang, E-values: calibration, combination, applications, v4](https://arxiv.org/html/1912.06116v4), Prop2.1 and proof | Decreasing calibrator integral criterion; allows mixtures and independently selected functions | R2 already exploits actual rank support. A fitted best transform on HELD is invalid. Separate PILOT meets independence; reuse unchanged in R3. Exact support normalization is established theory, not an R3 invention. |
| [Universal Inference](https://arxiv.org/html/1912.11436), Thm1/3, equations6–9 proof | Split density numerator over genuine null likelihood supremum yields finite-sample expectation control | Full unknown arbitrary radial densities are not provided by this input model. Parametric Student plug-in or a local null optimization would add assumptions/lose certification. Pivot reduction remains more directly implementable; no new likelihood family opened. |
| [Berger–Boos1994, author repository](https://repository.lib.ncsu.edu/items/1e8c580a-c3db-46f8-93a3-9b963e2c9171), DOI10.1080/01621459.1994.10476836 | Supremum over a1-beta coverage set plus beta per-test penalty | Repository still bot-blocked; full text NOT marked read. Independent elementary argument: Pr(sup_C p+beta<=a)<=Pr(p_true<=a-beta)+Pr(true notinC)<=a. R2's earlier protected BB/BY reference is already retained. The PRIMARY uses no confidence-set rectangle, so narrower BB sets cannot directly repair its current loss. |

## Current contribution position

One direct closest-mechanism check after selecting the two bottlenecks:
[Li–Li, Linear Hypothesis Testing in Linear Models With High-Dimensional Responses](https://pmc.ncbi.nlm.nih.gov/articles/PMC9996668/),
Sec2.3 and Theorem2/Conditions3–5 inspected. Independent projection learning
and testing are established, not our invention. Their general U-projection
normal limit uses moment/covariance/asymptotic assumptions not interchangeable
with this model's arbitrary common-block radial law and finite calibration.
Our current point of difference is the explicit joint H11/spectrum pivotal
reference with a separate scale-summary law and PC/PILOT chain; an existing
sample-split projection result alone neither supplies it nor proves priority.
No U-projection or new signal-extraction branch is introduced.

Equivariant pivotal summaries, separate-sample learning, rotational invariance,
rank tests, PC and e-BH are existing principles. A candidate must contribute
a verifiable task-specific sharper reference construction and quantified
efficiency tradeoff, not claim invention of those principles. Priority of
such a combination is not established by this short comparison. This matrix
will be updated with direct closest-work checks only if a concrete candidate
survives. No generic market-wide method search.

## Why not follow every promising citation?

D001 selects kappa sampling and Rayleigh protection. The frozen primary
already counts no per-gene calibration failure event. Compound/conditional
boosting cannot be justified merely by observing FDR<<.05. The joint-law
problem is separate and would violate this phase's two-bottleneck focus.
New proof work therefore concerns tighter nuisance elimination, retaining
the existing p/e and final family-level mechanism unchanged.

## Narrow closest-mechanism check during frozen confirmation (no algorithm edit)

[He et al., CFAR assessment of covariance matrix estimators for non-Gaussian
clutter, 2010](https://link.springer.com/article/10.1007/s11432-010-4080-z):
publisher abstract and references read. It describes finite-iteration adaptive
shape estimation preserving false-alarm invariance to both clutter shape and
power under spherically invariant noise. Thus nuisance-free adaptive detection
via an appropriate equivariant initializer is NOT a new R3 general principle.
Primary PDF endpoints failed/403 and publisher full text requires subscription;
no purchase/bypass, no claim to have checked its detailed statistic or proof.
This is a concrete unresolved closest-literature check, not proof of either
equivalence or priority. R3's potential distinct object remains the specified
real4x6 directional-PC construction with an independent learned direction,
JOINT shape-diagonal/spectrum reference and finite geometric calibration law.
No radar detector candidate or new branch was introduced. A high-level
invariance argument alone is insufficient to establish a novel contribution.

19:10HKT targeted access recheck (same closest paper, no new method search):
the indexed publisher English PDF and Chinese companion PDF were located,
but opening them returned non-fetchable/403 respectively. Search-index text
does not count as checking the full proof. The priority limitation remains;
no paid access, author contact or access-control bypass was attempted.
