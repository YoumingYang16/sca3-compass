# Focused joint-budget refinement: a necessary continuum protection

2026-09-18, focused-score work package. R5 only. All gates OPEN; no new Power batch authorized by
this note. Known existing input budgets and all wide/tight scenarios retained.

## J1 cannot be extended just by replacing the power score

Sole reviewer supplied this analytic challenge, main checked the integral and
perturbation signs. Let X approach0, Y~N(0,1), V=exp(1/2), a=1/2,
g(u)=1(u<=0), target payoff hT(t)=1(t>1), bridge payoff hB=.75 hT.
For 0<=s<1, selected expected score is
m(s)=.75 Phi(2-s)+.25 Phi(s).
Thus m0=.857937, m(.5)=.872760, minfty=Phi(1)=.841345.
The internal maximum exceeds both endpoints. Replacing X by N(0,.01^2)
changes each value by at most .01/pi; smoothing to logistic tau=.001 adds at
most2*tau*log2/sqrt(2*pi*(1+.01^2)). These two perturbations cannot remove the
strict gap. Thus even smooth nondegenerate logconcave errors do not justify
the unrestricted extension. This is NOT a counterexample in the actual logF/
learned-H model or with the actual gate parameters. It prevents a false theorem.

## F1: finite cover with a proven continuous-slack protection factor

Condition on W, independent meta, independent permissible design data. All
parameters below fixed at this conditioning. U=X-Y; T=V_+ exp(-Y/2), a>=0,
tau>0; let g(u)=expit((c-u)/tau). Let hB,hT be nonnegative increasing on[0,inf)
with finite expectations of the envelopes below and zero for nonpositive tests.
Define
 w_s=g(U+s) hB(T exp(-a(U+s)/2))+(1-g(U+s))hT(T).
For 0<=t<=s<=t+Delta, the first term decreases pointwise with s. Moreover
 d/du log(1-g(u))=g(u)/tau<=1/tau.
Therefore w_s<=exp(Delta/tau)w_t pointwise. For s>=S,
 w_s<=hT(T)+g(U+S)hB(T exp(-a(U+S)/2))=:w_tail.

Take finite intervals[t_i,t_i+Delta_i] covering[0,S], positive integrable means
mu_i=Ew_ti and mu_tail=Ew_tail. For independent-of-observations nonnegative
unbiased or conservative inverse estimators Ri with E Ri<=1/mu_i, and Rtail,
 R=min({exp(-Delta_i/tau)Ri}, Rtail)
satisfies E R<=1/max({exp(Delta_i/tau)mu_i},mu_tail). Consequently
 E[R w_s]<=1 for EVERY s>=0. The Ri can be dependent. The cover parameters and
focus thresholds must NOT be chosen using observed scale means or TEST without
an additional proof. Finite caps can only set the entire construction to0.

This is a conservative exact-model inequality over a continuum, NOT an exact
numerical nuisance maximizer, confidence based plug-in reciprocal, or uncorrected
finite-grid guarantee. No promise of useful Power or numerical certification.
Finite cover arguments and inverse-Bernoulli normalization are established tools;
important model-specific content and priority require separate examination.

## Current implementation gap

A focused inverse sampler must recompute BOTH gate and payoff for each reference
and slack, with each reciprocal's IID marginal stopping rule intact. Merely
reusing J2 power acceptance or holding the selected branch fixed is invalid.
An efficient envelope for focused tails and the actual learned shape is needed.
Before a Power batch: correct bounded fixture, proof/code review, measured cost,
then same-interface endpoint controls plus old strong valid comparators.

## F2: direct finite-reference normalization avoids reciprocal-mean estimation

Proposed simplification under challenge, before implementation. Conditional on
W/meta/independent admissible design, generate B IID reference tuples(T_b,U_b)
with the full null joint law at slack0. Recompute each tuple's full score at every
cover anchor (do not hold its branch fixed). Set
 Q=max({exp(Delta_i/tau) sum_b w_b(t_i)}, sum_b w_tail_b).
For every possible true s>=0, Q>=sum_b w_b(s) pointwise by F1. Then
 E_obs=(B+1)*w_obs/(w_obs+Q), zero when both numerator/denominator are zero.
At the true s, observed w_obs and the B true-slack reference scores are
exchangeable, conditional W/meta, so
 E E_obs <= E[(B+1)w_obs/(w_obs+sum_b w_b(s))] <=1.
This exact finite-reference argument needs no plug-in or unbiased reciprocal,
no reference independence across claims, and no moment of a bounded step score.
It is NOT a confidence bound on a population mean or a two-endpoint J1 extension.
Each ratio increases in w_obs, giving one-sided null domination; complete PC-e
chain follows J3 if learning and shape/rotation conditions are indeed the same.

Use a fixed bounded step h_q(t)=1(t>q), q fixed independently of observed means
and TEST, and hB=k*h_q with positive meta-only k. Tail envelope is bounded by1+k.
The sharpness/cost tradeoff is explicit: cover overhead exp(maxDelta/tau), B sets
the finite-e ceilingB+1. An optional underflow/overflow cap must be conservative.
The fixed cover and shared-reference normalization may be classical instances;
this solves a concrete implementation gap, NOT automatically G1 originality.

### F2b sharper monotone cover and reference-dependent geometry

Write w_b(s)=B_b(s)+A_b(s), where B decreases and A increases in s.
For each interval[l,r], sum_b w_b(s)<=sum_b B_b(l)+sum_b A_b(r).
This is a sharper direct upper sum than F1's exponential factor and holds even
for a hard gate. Tail bound is unchanged. Max over ALL covering cells and the
tail is a valid Q for every possible true s on every reference realization.
Thus geometry can be chosen from the reference sample, provided coverage and
the pointwise bound hold for every realization; this is not choosing a favorable
nuisance using TEST. Likewise upper cells may be adaptively subdivided without
losing validity, but prototype uses one deterministic width/cap rule only.
No promise that the numerical upper sum equals the exact supremum.

## A0.10 initial fixed design before any Power result

Use hT=1(t>qT), hB=1(t>qB). Meta-only independent reference draws estimate each
fixed endpoint's 1-1/1024 quantile; quantile error changes efficiency, not validity,
because it is conditioned on and the evaluation reference is independent.
Ns/Nt information-weight bridge and same count/source/target endpoints are
reported, plus endpoint fixed valid mixtures. No separate target pilot bank.
All endpoint quantiles use the same meta simulation budget. Candidate and
controls use the same independent B=8191 complete reference tuples. Tail cutoff
ensures each borrow gate<=.01/B; if floating calculations cannot verify this,
the actual tail sum is still retained, not replaced by .01. Nominal grid width
.1*tau with at most1024 cells; any coarsening remains valid via monotone upper
sums and is disclosed. Reference calculation has a finite work cap.
This may recover useful e magnitudes but might still lose to old valid methods.
No formal confirmation until important new content and utility are established.
