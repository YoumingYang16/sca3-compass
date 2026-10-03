# Single next work package: calibration-contrast localized joint reference

2026-09-18, before localized diagnostic outputs. Replaces failed A0.10 gating-only effort;
keeps A0.7.1 reference. No new projection/data/algorithm search or confirmation.
All gates OPEN. No claim of originality: composite MC infimum is inherited.

## Diagnosis requiring a different protection, not more gate tuning

D016_2 exactly reproduces all20D015e/decisions. Classical max and complex soft
rule produce identical final rejection sets in all20. Exact reference union
cost gives family-level Emax=(1+Nselected)/(1+Nunion)*Eselected. C1factors.6842
and.8. Soft observed-score loss did not buy enough normalization savings.
The global denominator ignores how incompatible large candidate nuisance slack
is with the actual calibration contrast. Need examine a nuisance-indexed score,
not another g threshold. Keep scope and actual data budget fixed.

## L1: full nuisance-indexed soft ranks (under internal challenge)

Condition on W, independent meta and per-claim admissible learning sigma-field.
At true s>=0, observed(T,U) and independent reference(T_b,U_b+s) are exchangeable.
Let h(t,u)>=0 be the SAME focused selected score asA0.10; choose fixed strictly
positive proxy functions r,f using only meta/D. They are NOT asserted to equal
a true density. For each s define the entire score
 L_s(t,u)=h(t,u) r(u)/f(u-s).
The existing MC soft-rank theorem makes E_s a valid e for the null at that s.
Take the infimum over all s>=0, not an estimated best s. Algebra gives

 E=(B+1)*h_obs*r(u_obs)/(h_obs*r(u_obs)+sup_{s>=0}F(s)),
 F(s)=f(u_obs-s)*sum_b h(T_b,U_b+s)*r(U_b+s)/f(U_b).

At the true s, any pointwise upper envelope for sup F yields E<=E_s. Therefore
finite-reference validity follows WITHOUT assuming fitted f is correct. It is
essential to include r/f in BOTH observed and every reference score. No selection
outcome is frozen, no reference conditional-on-estimated-s claim is made.
Monotonicity in T holds because u fixed, h increases in T, and denominator bound
depends on u and references but not observed T. The existing PC/PILOT/eBH chain
can be used without new directional or shape algorithms if this is verified.

## L2: continuous finite-cover certificate including the infinite tail

Use Cauchy-shaped proxies f(v)=1/(1+(v/(2sigma))^2),
r(v)=1/(1+(v/(sigma+logD))^2), sigma=sqrt(meta_vs+meta_vt)>0.
Unnormalized constants cancel from the entire ratio; D>=1 known externalbound.
These widths are fixed construction choices BEFORE diagnostics, not optimized
against Power. f avoids inverse-Gaussian-proxy explosion: 1/f(U) is quadratic,
and all polynomial moments of finite-W conditional logF errors exist.

For interval[l,r], each h_b(s)=B_b(s)+A_b(s) has decreasing B and increasing A,
so h_b(s)<=B_b(l)+A_b(r). The supremum of either Cauchy proxy on an interval is
at the point nearest zero. Thus an upper bound on F throughout that cell is
 max_{s in[l,r]} f(uobs-s) *
 sum_b [B_b(l)+A_b(r)] max_{s in[l,r]}r(U_b+s)/f(U_b).
For S>=max(0,uobs,-min_b U_b), both proxy factors decrease for s>=S, and
 F(s)<=f(uobs-S)*sum_b [I_T,b+B_b(S)]r(U_b+S)/f(U_b).
Include this tail bound in the maximum. It covers the FULL legal half-line;
no finite-grid-only claim. Grid geometry may depend on observed u and references
because the pointwise bound holds for every realization and every possible s.
No conditional validity given that geometry or denominator is asserted.
For each finite reference sample F(s)->0 as s->infinity, removing the old global
target-only reference floor. This is not a proof of useful population Power.

## Novelty and cost limits

The MC normalization/nuisance infimum are direct existing results. A computable
localized envelope is a capability, not automatically important new research.
Need establish a nontrivial efficiency or adaptation-cost property and a useful
comparison beyond classical fixed mixtures/max before G1/G3 can pass.
The inverse proxy may introduce its own efficiency penalty, especially far from
the boundary. Do not hide that or treat observed-slack fitting as a free oracle.

## Fixed minimal next action

First add finite-cover and permutation normalization correctness fixtures.
Then deterministic replay ALL20D015tuples, SAMEh/thresholds/meta/learning; no new
samples, fit or nuisance-truth input. Parameter widths and cover rules frozen
before result. Quantify upper-vs-evaluated-lower gap and scalar e ratio, all old
comparators retained. This is diagnosis, not confirmation; no expansion if it
fails. Stop this work package at complete proof challenge and that attribution,
then act on the identified mechanism. Expected CPUseconds/minutes, one worker.

Implementation detail frozen before diagnostic: finite interval bounds refined
by splitting only the largest upper bound; target1% relative empirical envelope
gap, at most256cells. The unrefined tail is always included. Stopping at cap or
tail domination returns the valid upper envelope, never an unproved optimizer.
This controls numerical work only, not statistical repetitions or significance.
