# Selection cost / co-design notebook (not a novelty certification)

## A. Exact selection bias of current gate

For U=Es-Et at boundarys0, mean logscale error of A0 is
E[Et+aU 1{U<=c}]=-a E[U 1{U>c}]<0 for c>0. This uses EEt=EU=0 and positive
right-tail mass, so it is finite-sample, not merely a Gaussian approximation.
Under fixed Ns/Nt ratio with Ns,Nt→infinity and c=sd(U)=sigma, uniform
integrability plus CLT gives E[U1{U>sigma}]/sigma→phi(1). Consequently the
downward bias is -a*sigma*phi(1)+o(sigma). Fixed geometric endpoints have exactly
zero mean logscale error. The adaptive gate creates a FIRST-order log bias.
This does not by itself prove a Power/FDR loss, a lower bound, or new knowledge:
pretest bias is classical. It is a mechanism to check, not an acceptance result.

## B. Sharp *pathwise* drift-monotonicity threshold for a fixed score

Condition on fixed independent inner maps. For positive scorex let qt(x),qb(x)
be their positive inverse raw cutoffs. At switchu=c the borrowed raw score
corresponds to target-scaled statistic Wt*exp(-a*c/2). Borrowed exceedance at the
switch requires Wt>=qb(x)*exp(a*c/2), while target exceedance immediately after
the switch requires Wt>=qt(x).

Therefore every trajectory's exceedance indicator is nonincreasing ins iff
qb(x)*exp(a*c/2)<=qt(x). Sufficiency: borrowed trajectory falls withs; switching
cannot turn an absent exceedance into a present one; target branch is constant.
Necessity: if the inequality is reversed, choose Wt strictly between the two
cutoffs and a finite h>0. Such tuples have positive probability because the
Gaussian numerator and independent logF errors have full support. The indicator
jumps0→1 at h, disproving pathwise monotonicity. This only proves failure of a
pathwise argument, NOT that the resulting marginal test necessarily inflates.

For a set of score thresholds X the sharp permitted switch is
c<=inf_{x in X} (2/a)log(qt(x)/qb(x)). Conditional empirical maps may make this
infimum small or unbounded below in extrapolated tails; a finite grid is not a
certificate. Existing A0/A0.1 therefore correctly keep the full supremum.

## C. A co-designed rule eliminating nuisance optimization (candidate refinement)

Let qt,qb>0 be fixed conditional on independent inner references (initial
definition: empirical .99 quantile, same conventional order for both). Use
phi_j(w)=log(w/qj) for w>0 and -infinity otherwise. Set
c*=2log(qt/qb)/a; choosebridge iff u<=c*. Then the selected score is EXACTLY
max(log(Wt/qt),log(Wbridge/qb)), because the comparison cancels the shared
observed numerator, energy and shape. Thus selection uses CALIBRATION ONLY.
For s>=0, Wt is constant and Wbridge=Wbridge0*exp(-a*s/2) decreases; the maximum
is pathwise nonincreasing. The LEAST-FAVORABLE reference is s0, jointly simulating
Es,Et,H,U,Z0 and the exact same selection. No nuisance search, no uncovered tail.

This removes optimization cost, not selection cost. At s0 a selected maximum
still has a heavier upper tail than either normalized endpoint alone. Rank
calibration of that max handles it exactly. PC/finite-grid/eBH chain is unchanged.
This is a standard maximum-statistic principle instantiated in the dual-scale
model. The algebraic pathwise frontier and its statistical relevance need
closest-work review; G1 is OPEN, not granted by a new candidate label.

## D. Exact selection-cost decomposition for the co-designed rule

For fixed maps and cutofft, let A={Wt/qt>=t}, B={Wbridge0/qb>=t}. At boundary,
P(max>=t)=P(A)+P(B\A)=P(B)+P(A\B). The inflation above the larger marginal tail
is min(P(B\A),P(A\B)); it ranges from0 to min(P(A),P(B)). This identity applies
also to the empirical reference counts, so the number of additional tail draws
is observable without true drift. Dependence is preserved (sameV,Es,Et), unlike
Bonferroni. This union identity is CLASSICAL, not a novel theorem by itself.

Candidate decision: a specific mechanism refinement of the SAME joint-pivot
mainline, not an additional algorithm family. A0/A0.1 remain development history.
Only a small common-input check is warranted before any larger evidence spend.
The core research gap remains whether a useful, important new selection-cost or
efficiency result exists beyond this classical construction.
