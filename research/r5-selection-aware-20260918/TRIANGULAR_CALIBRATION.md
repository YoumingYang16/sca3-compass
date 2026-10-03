# Full triangular calibration information — development, 2026-09-18

This is a calibration-layer refinement within R5, NOT a new TEST projection,
new data, a final candidate, or an originality/Power claim. All same-information
controls must receive the same refinement. Fixed study order; no outcome-chosen
permutation. V1/R2/R3/R4 remain frozen.

## T1: the vector pivot and its law

For one zero-location calibration block let x be the four row means and Y its
4 by 5 orthonormal pipeline contrasts. In M(D), after cancelling the common
positive radial factor, (x,Y)=(sqrt(kappa) A z, A E), with z and E independent
standard Gaussian and A arbitrary nonsingular, potentially different per block.
Here A is fixed or independent of the Gaussian angular innovations; choosing A
from those same innovations is NOT covered by the nuisance-cancellation claim.
Define W=YY', L=chol(W) LOWER with positive diagonal, T=L^{-1}x. This is not
the symmetric inverse square root of W. The distribution of T/sqrt(kappa) is
independent of A: write A=L_Sigma O; O is orthogonal, Gaussian invariance removes
O, and chol(L_Sigma E E' L_Sigma')=L_Sigma chol(E E'). This is distributional
pivotality, not pointwise invariance under arbitrary left transformations.

Let p=4, nu=5, Q_d=sum_{j<=d}T_j^2, Q_0=0. Bartlett decomposition of chol(EE')
has independent diagonal sqrt(chi2_{nu-d+1}) and lower off-diagonal normals.
Conditionally on the preceding coordinates,

 T_d = sqrt(kappa+Q_{d-1}) Z_d / sqrt(chi2_{nu-d+1})

in distribution, with independent standardized innovations. Thus
T_d^2/(kappa+Q_{d-1}) is BetaPrime(1/2,(nu-d+1)/2), NOT an unscaled F variable.
Multiplying the four conditional densities gives

 f_kappa(t) = (2/pi^2) kappa^(5/2) /
             [(kappa+Q1)(kappa+Q2)(kappa+Q3)(kappa+Q4)^(3/2)].

The old scalar A4=Q4/2 has kappa F(4,2) distribution. T is NOT the usual
spherically symmetric multivariate t. No sufficiency for all raw data claimed.

## T2: exact conditional location experiment

Set a_d=Q_d/Q4, d=1,2,3; a0=0,a4=1; y=log(Q4/kappa). Almost surely
0<a1<a2<a3<1. The signs are ancillary. Squaring and changing to cumulative
angles/radius gives Jacobian proportional to
q^(p/2-1) product_d(a_d-a_{d-1})^(-1/2) dq da. The log-radius adds dq=q dy.
Consequently, conditional on angles a,

 log g_a(y) = C(a)+2y - sum_{d=1}^3 log(1+a_d exp(y))
                       -(3/2) log(1+exp(y)).

Its second derivative is strictly negative; tail slopes are 2 and -5/2.
Angles are kappa-ancillary but NOT independent of radius. Do not condition on
the full observed W and still use this Wishart-integrated law.

For independent bank blocks with the same kappa, w_i=log Q4_i-mean(log Q4),
the angles and w jointly have parameter-free distribution. The conditional
density of Ybar=mean(log Q4)-log kappa is proportional to

 product_i g_{a_i}(Ybar+w_i).

This follows directly from the location change of variables (constant
Jacobian). It is one-dimensional log-concave, tails 2N,-(5/2)N. The arbitrary
block-specific A_i has already been eliminated. The old geometric estimate is
exp(mean(log Q4)-1), so its log error is Ybar-1. Do NOT subtract 1-log(2), which
belongs to log A4. Independent auxiliary reference draws can learn conditional
centering/variance given this ancillary, as in the existing selector. Means
from meta draws and observed errors must be centered identically.

## T3: calibration information strictly lost by the old radius compression

For eta=log kappa, the full-vector score is
2.5-sum_{d=1}^3 kappa/(kappa+Q_d)-1.5 kappa/(kappa+Q4).
Conditional transition scores are martingale differences. Writing
m_d=nu-d+1, their information is

 E[(kappa/(kappa+Q_{d-1}))^2] m_d/[2(m_d+3)].

For nu=5, Q_{d-1}/kappa is BetaPrime((d-1)/2,(7-d)/2), with Q0=0.
The beta second moment gives contributions m_d(m_d+1)/96, summing
5/16+5/24+1/8+1/16=17/24. Radius-only information is 1/2.
Thus this pivotal experiment has 41.67% more Fisher information per block.
This is NOT 41.67% more Power, not finite-sample MSE dominance, not a uniform
conditional gain, and not a proof of important originality. Keeping the old
geometric point estimate alone leaves its unconditional MSE unchanged.

## Full-chain obligations

Condition jointly on source/target angles and within-bank log residuals, plus
independent meta randomness and the permitted PILOT information. Source and
target errors remain independent conditionally and log-concave. The existing
selective-branch MLR argument must be rechecked for these densities; it does not
follow merely from using another bank. Actual DIR can depend on calibration,
but inherited rotational pivotality of the learned TEST reference is constant
conditional on calibration and DIR. Integrate that conditioning back before
using T2; do not fix all SHAPE/DIR/TEST folds simultaneously. No TEST changes.

Numerical QR with positive R diagonal implements the same lower Cholesky pivot
without forming an ill-conditioned Gram matrix. Underflow, angular ties, or
singularity fail closed, never silently drop a row or round an angle to zero.
Three-tangent accept/reject is exact in the ideal model. Floating-point guards
are not a universal machine-arithmetic proof; that qualification remains OPEN.

## Literature and review

Main and sole reviewer read Ogasawara (2023), JBDS 3(1),34-58,
doi:10.35566/jbds/v3n1/ogasawara, section2.1 Theorem1 and Bartlett derivation.
https://jbds.isdsa.org/public/journals/1/html/v3n1/ogasawara/
It explicitly identifies Bartlett decomposition as classical (Anderson,
Wijsman1957, Kshirsagar1959). These primitives are inherited, not original.
Reviewer independently checked T1-T3 including the angular Jacobian and FI.
Closest-priority review of the conditional safe-borrowing use remains OPEN.
A structural-inference PDF link failed to load; do not claim it was reviewed.

## M006 pre-output diagnostic protocol

One fixed kernel check, not whole-family FDR confirmation: 32768 independent
Gaussian (z,E) tuples, seed 605018, direct QR construction. Compare transition
Beta CDF transforms, Q4/2 versus F42, mean score and Fisher information17/24.
Use bounded-score MC standard errors; report all values, no repeated seeds to
obtain a pass. First256 tuples also test positive lower-triangular transforms
and common radial multipliers algebraically; no extra random observations.
Conditional sampler checks use deterministic angle/residual fixtures with
quadrature, 8192 reference draws each, one seed per fixture. These are sampler
checks, NOT Power gains, formal confirmation or a G2 substitute.
Stop this package on any law/centering mismatch, fix implementation first.
If coherent, next bounded old-input paired comparison shares calibration upgrade
with ALL endpoints/fixed mixes. No new family samples or parameter grid here.
