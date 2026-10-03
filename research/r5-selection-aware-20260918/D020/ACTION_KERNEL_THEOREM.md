# Selection of borrowing AMOUNT: a validity criterion, not a novelty pass

Development theorem draft,2026-09-18. Motivation D019: triangular calibration
helps some endpoints, but binary selective mixture still misses stronger source
and target choices. No further large experiments before clarifying this gap.

## K1: sufficient criterion and proof

Condition on jointly ancillary calibration summaries and independent meta
randomness. Let X=Es+s, Y=Et, s=log(D/delta)>=0; independent errors have positive
densities fs,ft and fs is log-concave. A finite action set has fixed weights
lambda_j in[0,1]. Given U=X-Y, select J=j with probability w_j(U). Require
sum_j w_j(u)=1 and each nonzero w_j log-concave in u. Action maps and constants
can depend on the permitted conditioning, not on the observed bank means/TEST.

Claim: L_j=lambda_j X+(1-lambda_j)Y conditional on J=j is stochastically
nondecreasing in s. Hence the positive null statistic V exp(-L_j/2) has its
largest tail at s=0, for V independent with the actual inherited learned-shape
law. This includes lambda0=0,target and lambda1=1,source-bound.

Proof: X|J=j has density proportional fs(x-s) H_j(x), where
H_j(x)=integral ft(y)w_j(x-y)dy, independent of s. Its likelihood ratio between
s2>s1 is increasing in x by fs log-concavity. Given X=x,J=j, Y has density
proportional ft(y)w_j(x-y). For x2>x1 the ratio w_j(x2-y)/w_j(x1-y) is increasing
in y because logw_j is concave. Couple the increasing X laws by quantiles and
the increasing conditional Y laws by conditional quantiles. Both coordinates,
therefore any fixed nonnegative convex combination L_j, increase. This proof
uses no independence between gate and scale, and ft need not be log-concave.

For each action sample complete independent (Es,Et) pairs at s=0 and retain
with probability w_j(Es-Et). Only then use the learned-shape reference V.
Never freeze the observed selection and reuse the unconditional reference.
The branch rank is conditionally superuniform; inherited directional PC,
grid/PILOT conversion yields E[E_j|J=j,ancillary,meta]<=1. With a conceptual
observed action coin independent of auxiliary references, integrate the coin:

 E_final=sum_j w_j(U_observed) E_j(observations,reference_j).

Total expectation yields E[E_final]<=1. Final eBH at q=.05 controls FDR over
all signedPC hypotheses with the original fold dependencies. A component e_j
alone is NOT unconditionally valid. A failed component zeros the whole family;
do not redistribute its weight. Continuous-action integrals require their own
exact/conservative computation; arbitrary numerical quadrature is not certified.

Finite affine softmax weights w_j(u)=exp(b_j+c_j*u)/sum_k exp(b_k+c_k*u)
are log-concave because d²logw_j/du²=-Var_{w(u)}(c_k)<=0. Any fixed finite
lambda set therefore permits calibrated selection of borrowing amount, not only
whether to use one bridge. This is a mechanism capability, not proof of utility.

## K2: an exact warning for two-sided conflict selection

K1 is not true for arbitrary calibration-only selectors. Take centered independent
Es~N(0,epsilon²),Et~N(0,tau²), and target-action probability
w0(u)=1/2-(1/4)exp(-u²). It is bounded in[1/4,1/2], but logw0 is convex at0.
Put d=1+2(epsilon²+tau²), C(s)=d^(-1/2)exp(-s²/d). Gaussian integration gives

 P_s(J=0)=1/2-C(s)/4,
 E_s[Y|J=0]=-[tau²*s*C(s)/(2*d)]/[1/2-C(s)/4].

This mean is zero at s0 and strictly negative for every s>0. Thus Y|target
cannot be stochastically increasing in s; s0 is not a general least-favorable
target reference. The selector read NO TEST, yet the fixed-boundary argument
fails. This is an abstract centered Gaussian-error counterexample, not a claim
that this exact law occurs in finite R5 calibration or proof of a numerical FDR
violation. It warns specifically against two-sided discrepancy rules without
their complete nuisance calibration. Gaussian integral proof: integrate
exp(-(X-Y)^2) first to obtain C(s); the corresponding tilted Gaussian has
Y mean 2*tau²*s/d. Multiplying by -1/4 gives the displayed numerator.

A general necessity claim for all smooth weights remains under review. Local
positive curvature of logw implies such counterexamples in an unrestricted
location-error class; this does not prove necessity in the smaller actual R5
conditional triangular family. Do not conflate these quantifiers.

## Closest result and honest status

Main read Saumard-Wellner arXiv1404.5886v1, Proposition2.3(a,b) and AppendixB
proof: logconcavity, PF2 and translation-family MLR equivalence are classical.
https://arxiv.org/html/1404.5886v1 . K1's two monotone-kernel couplings are a
direct use of these tools; conditional simulation/Rao-Blackwellization are also
known. Tian-Taylor's randomized selective density already gives f times w.
No new generic total-positivity theorem is claimed. A task-specific capability
alone is not yet important G1 novelty. Need exact closest-borrowing comparison,
useful performance and/or a nontrivial finite decision-cost result. G1 OPEN.

## M007 pre-output check

Deterministic Gaussian integral only: epsilon=.15,tau=.6,s=0,.25,1,2.
Compare closed form K2 to Gaussian-Hermite quadrature at64 and128 nodes each
coordinate. No random families/Power/FDR/confirmation. Save all four values and
quadrature discrepancy, never convert failed order into a claim of FDR failure.
