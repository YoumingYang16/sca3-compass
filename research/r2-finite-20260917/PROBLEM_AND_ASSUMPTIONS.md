# R2 finite calibration: problem and initial deduction

## Current resolved specification (R2-PB-1.0.0)

Historical intake below is retained, not the final unresolved status.
Primary uses four folds with separate PILOT; THEORY Propositions5–7 and
PREDICTIVE_BRIDGE.md supersede the intake's same-TRAIN calibration language.
Unknown kappa sampling error and unknown R are integrated against a proven
dominating pivotal law, not ignored plug-ins. Target radial law cancels.
PILOT-only mixture/calibrator choices are protected by conditional validity;
directions are TRAIN/calibration-only, never HELD or auxiliary-reference.
Fixed4095 references, fixed2 affine shape updates, q=.05, no bad-event budget.
Calibration may have independent random invertible left transforms per block
independent of its Gaussian matrix; centered means/common pipeline kappa and
block independence are still necessary. Target STUDY-specific radii remain
outside M0. See the completed input checklist in README, not a promise that actual
gene-expression datasets satisfy independent blocks or matched calibration.

Started 2026-09-17 12:53 HKT under explicit user authorization. V1 and the
postrelease archive remain immutable. This is a new research phase, not a
restart of historical R0074/R0076. Initial status UNRESOLVED.

## Model M0, estimand and information

G independent target blocks, S studies and J same-location pipelines:
Z_g = mu_g 1' + sqrt(V_g) R^(1/2) E_g C(rho)^(1/2).
E_g has iid standard Gaussian entries; R is arbitrary positive definite;
C=(1-rho)I+rho 11', -1/(J-1)<rho<1. Each V_g>0 is independent of E_g;
radial laws may differ. E sqrt(V_g)<infinity is required to interpret mu as
the mean. No finite variance or inverse-gamma assumption is required for
the proposed pivots. Calibration consists of N independent centered S-by-J
blocks of the same angular pipeline structure, independent of targets.
Studies within each calibration block are NOT independent calibration units.
Calibration and target radial laws need not match for the pivots.

The signed PC null is H_g,d: #{s:d mu_gs>0}<r, with r=2,S=4 in experiments.
There are 2G claims; FDP=FP/max(R,1), Power=TP/number of true signed claims,
undefined if no alternatives. Complete simulated families are repetitions.

For H'H=I, H'1=0, M=Z1/J, Y=ZH, and L=J-1,
kappa=[1+(J-1)rho]/[J(1-rho)] is dimensionless. It is a scale ratio when
variance does not exist, not a population variance ratio in that case.

## Verified algebraic starting points (not novelty claims)

For a fixed nonzero b independent of calibration, A_l=(b'C_l1/J)^2 /
(||b'C_lH||^2/L) satisfies A_l/kappa ~ F(1,L). Gaussian mean and contrasts
are independent before common radial multiplication; both the radial factor
and b'Rb cancel pointwise. Independent blocks give independent A_l.

At known R, the target statistic
T(a)=a'M / sqrt(kappa (a'Ra) tr(R^-1 YY')/(S L))
has t_(S L) distribution at zero projected mean, even for arbitrary radial
laws. Under a nonnegative intersection direction supported on null studies,
its location shift is nonpositive and its positive tail is dominated by the
central pivot. This is marginal over Y, NOT conditional validity given own Y.
Thus own-Q-dependent weights are not permitted.

This differs from the old inefficient five-df target pivot: it retains all
S L contrast coordinates, but REQUIRES protection of unknown R.

## Nuisance accounting at intake

| Quantity | Status / required treatment |
|---|---|
| kappa | Identified by block pivots; finite order-statistic coverage to implement |
| R up to scale | Unknown; must protect, not pretend Tyler is exact |
| common target radial law / df / scale | Eliminated by target ratio above |
| learned nonnegative directions | Must depend only on other target fold |
| pilot and mixture weights | Same allowed training information; no own-Q weights |
| numerical quantiles and matrix operations | Separate implementation gap to assess |
| target/calibration angular mismatch | Outside M0; externally bounded sensitivity only |

## Restricted-good-event composition lemma, proof draft

Suppose B_f is measurable with respect to calibration plus fold-f's TRAIN,
and for each true-null i tested by fold f,
E[1_(B_f) E_i] <= 1 (E_i may be unbounded outside B_f).
If P(union_f B_f^c)<=delta, applying e-BH at q0 yields
FDR <= q0 + delta. Indeed on B=intersection_f B_f,
FDP <= q0/m sum_(null i) E_i, and
E[1_B E_i] <= E[1_(B_f) E_i] <= 1. Off B use FDP<=1,
NOT a bound on the e-value tail. Choose q0=q-delta.

This does not condition on the union of both training folds. It does NOT
assert E(E_i)<=1 unconditionally, nor the stronger q0(1-delta)+delta bound.
The lemma is a standard bad-event decomposition; originality is unestablished.
The unresolved work is constructing useful, finite, implementable B_f and
auditing the complete learning chain.
