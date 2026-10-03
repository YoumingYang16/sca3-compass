# A0.6: selective branch least-favorable reference (under challenge)

Reason for this revision: D007's minimal score lift trades improved C1 Power
(.3015 to .3529) for degraded legal-wide C3 (.5735 to .4975), on the SAME8
families per case. Do not optimize its threshold without a new mechanism.
Instead keep the actual joint errors and calibration-only gate but condition
the reference on the observed branch. This is NOT freezing the branch in the
old unconditional reference: every reference pair must satisfy the same gate.

## Claim and complete proof

Condition on observed ancillary Ws,Wt and independent meta draws. The centered
errors Es,Et are independent with positive log-concave densities fs,ft derived
from the actual logF residual product. In particular fs is log-concave; arbitrary
bootstrap/error estimates are not substituted. Set X=Es+s,Y=Et with legal unknown
s=log(D/delta)>=0. For fixed a in(0,1],c finite, the observed branch is
B={X-Y<=c}. Borrowed logdenominator is L=aX+(1-a)Y; target branch usesY.
Both branches have positive probability for every finite s because densities
have full real support. Independent MC/meta choices fix a,c before selection.

THEOREM: (L | B,s) and (Y | notB,s) are stochastically nondecreasing in s.

Borrow branch: density of X|B,s is proportional
fs(x-s)*barFt(x-c). For s2>s1 its likelihood ratio is proportional
fs(x-s2)/fs(x-s1), nondecreasing inx by log-concavity of fs. Hence X|B,s2
stochastically dominates X|B,s1. Given X=x andB, Y has the fixed target law
truncated to[x-c,infinity), which is stochastically nondecreasing inx. Couple
X via common quantiles, then Y via common conditional quantiles; both increase.
Because a and1-a are nonnegative, L increases. No independence of L and gate
is assumed. This also coversa=1. Constant c/meta are essential.

Target branch: density of Y|notB,s is proportional
ft(y)*barFs(y+c-s). A positive log-concave density has log-concave survival,
hence increasing hazard. For s2>s1 the derivative of the log survival ratio is
h_s(y+c-s1)-h_s(y+c-s2)>=0. Therefore the density ratio is nondecreasing in y,
which establishes the second stochastic order.

For central positive signed statistics V/sqrt(exp(L)) orV/sqrt(exp(Y)), the
conditional law of V given full banks/DIR is CONSTANT in those conditioning
values by the inherited rotation law. Thus V remains independent of selected
errors given W,meta,B. Independence after fixing banks alone would be vacuous.
The positive tails given the corresponding branch are
largest ats=0. Nonpositive statistics receive p=1. Composite signed nulls
are bounded by their central case by the original common-pipeline mean coupling.

Condition on W,meta,observedB (NOT exact observed bank means). Draw independent
Es,Et pairs from their exact conditional laws and RETAIN only those satisfying
Es-Et<=c forB, or>c fornotB, at boundarys0. Only after retention generate fresh
independent complete shape/normal/chi-square tuples. Each retained selected
logdenominator then has the correct boundary law conditional on that branch.
The observed law is stochastically smaller in positive-statistic scale, so the
plus-one >= rank againstM iid retained references is superuniform. The reference
uses no oracle s or truth. A finite rejection cap maps unfinished whole families
to e0, not to an approximate conditional law or unprotected endpoint fallback.

The previous direction, PC, integer grid, PILOT and eBH proof now conditions on
W,meta,B. PILOT remains independent of these quantities. Integrating them yields
the same finite-model FDR<=q. Bank-dependentDIR still requires the inherited
invariance, not a generic cross-fitting assertion. Actual code remains to be
checked against this complete chain.

## Efficiency and novelty, not yet established

In a DIAGNOSTIC Gaussian-error model, a=Vt/(Vs+Vt) makes L atboundary independent
ofU=Es-Et; the borrowed branch then has exactly the fixed-bridge reference, with
no conditional-selection penalty on that branch. The actual logF conditional
errors are not Gaussian; zero covariance is not independence and this exact
claim must not be applied there. The target branch can still pay a truncation
penalty at wideD. No zero-regret, superiority or uniform Power claim follows.

MLR, IFR and conditional/rejection MC are classical. The question is whether
the TWO branchwise least-favorable laws in this finite dual-bank problem provide
an important useful capability beyond existing borrowing theorems. Important
novelty remains OPEN; do not declare the above ingredients newly invented.

Internal review confirmed both orders; only source log-concavity is needed.
Exact limitations: equalN, allzero residuals and a=.5 give zero Cov(L,U), but
E[(L-EL)U²]=polygamma(2,2N)-polygamma(2,N)>0, so actual logBetaPrime errors are
not independent. For centered Gaussian errors and sigma²=Vs+Vt,
E[Y|notB,s=0]=-(Vt/sigma)*phi(c/sigma)/barPhi(c/sigma). Wide-s actual target
errors approach the unconditional law, so this is a real protection penalty.
Expected pair cost M/P0(branch) has no proved uniform finite upper bound.

Main read Bagnoli-Bergstrom's March4,2004 author manuscript theorem3/corollary2
and appendix8.2 lemma4: https://escholarship.org/uc/item/0xt8k6sd . Its last
displayed proof line has an inconsistent sign; the preceding integration gives
f'barF+f²>=0, the correct implication. Reviewer read Saumard-Wellner1404.5886v1
proposition2.3(b), the classical log-concave location/MLR equivalence.
Do not call either generic result an R5 invention.

## Next bounded package

Independent reviewer challenges the order proof and its equivalence to closest
work. Implement one selective-pair sampler and selected-reference adapter,
reuse fixedfold code. Unit-test exact gate membership, covariance/normal fixture,
caps and reference streams. Then one SAME80-family fixed diagnostic only if
the proof challenge and tests allow it; no grid or extra calibration data.
Current confirmation budget used0. A0.5 history and frozen D007 remain intact.
