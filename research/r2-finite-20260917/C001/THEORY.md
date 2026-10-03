# Finite calibration, shape protection and cross-fitted directional PC

Current primary: R2-PB-1.0.0, Propositions5–7 below. Propositions1–4 retain
the valid but inefficient D001 confidence-envelope reference, not the current
inference algorithm. All guarantees are ideal-arithmetic/model guarantees;
NUMERICAL_SCOPE.md separately qualifies the executable.

Proof draft v0.2, 2026-09-17. Exact-real-arithmetic results below; floating
implementation and originality are separate questions. Model M0 is in
PROBLEM_AND_ASSUMPTIONS.md. Here S=4,J=6,L=5,D=20,r=2; general formulas
apply for L>=S. All random directions and calibration functions must be
independent of the target block to which their test is applied.

## Proposition 1: scalar and matrix calibration pivots (classical)

For a centered calibration block let x=C1/J and Y=CH. In distribution,
x=sqrt(V v) R_c^(1/2) u, Y=sqrt(V(1-rho)) R_c^(1/2) E,
where u~N_S(0,I), E has L independent N_S(0,I) columns, and u,E,V are
independent. v=[1+L rho]/J, kappa=v/(1-rho).

For fixed nonzero b, (b'x)^2/(||b'Y||^2/L)/kappa~F(1,L).
For L>=S, W=EE' is nonsingular a.s. and

 A=(L-S+1)/S * x'(YY')^-1 x,
 A/kappa ~ F(S,L-S+1).

Proof: cancel V and R_c by congruence to obtain kappa*u'W^-1u.
The normal-Wishart quadratic identity gives
(L-S+1)u'W^-1u/S~F(S,L-S+1). One direct derivation rotates u to its first
axis conditional on its norm. Its squared norm is chi-square_S; the inverse
of (W^-1)_11 is the residual sum of squares from regressing a row of E on
the other S-1 rows, chi-square_(L-S+1), independently of u. Their ratio
has the asserted law. Independence of blocks gives iid pivot ratios.

R_c may differ between calibration blocks and from target R, and positive
radial laws may differ: both cancel blockwise. Pipeline kappa must match.
The independent review identified a useful extension: any random invertible
left transform A_l independent of Gaussian u,E cancels pointwise. Thus
independent per-study positive radial factors also cancel in CALIBRATION,
provided they multiply each study's whole pipeline row and the pipeline
kappa is common. This covers the old independent calibration-row generator
after grouping the four rows at index l into one block. It does NOT extend
the full-contrast TARGET pivot or TRAIN-shape result to study-specific radii.

For fixed order k chosen from N,S,J,delta only, define
c=F_F^-1(Beta^-1(delta;k,N+1-k)), U=A_(k)/c.
Then P(kappa>U)=P(A_(k)/kappa<c)=delta. This follows from the uniform order
statistic beta law. A smaller c is conservative. Code selects k by a
deterministic median-width criterion, never from observed A values. At S4/J6,
F_(4,2)(x)=(2x/(1+2x))^2. The chosen dyadic c is checked using the exact
integer binomial upper-tail polynomial; quantile coverage does not rely on
unverified special-function rounding.

## Proposition 2: full-contrast target pivot

For a fixed a>=0 and a nonpositive projected mean a'mu, define
T_R=a'M/sqrt[kappa(a'Ra) tr(R^-1 YY')/D]. Under zero mean it is t_D.
Indeed tr(R^-1 YY')=V(1-rho)||E||_F^2 and the centered numerator is
sqrt(V v a'Ra) Z, with Z independent of E. V cancels. Under nonpositive
mean the statistic is pointwise no larger than its centered counterpart.
Consequently p_R=sf_tD(T_R) for positive T_R, and 1 otherwise, is superuniform.
This is not validity conditional on Y. No own-Y weights or selection allowed.

## Proposition 3: unknown-shape protection without solving Tyler exactly

For independent training contrast blocks Y_g (SxL), put W_g=Y_gY_g',
H_0=trace_normalize(sum_g W_g/det(W_g)^(1/S)). Perform a FIXED K iterations
H_(t+1)=trace_normalize(sum_(g,l) y_gl y_gl'/(y_gl'H_t^-1 y_gl)).
Call the output H. No ridge, unit-diagonal normalization or outcome-based
stopping is used. All W_g are SPD a.s. when L>=S; every update is SPD.

Under Y_g -> c_g A Y_g, c_g>0,A invertible, initialization transforms to a
positive scalar times A H_0 A'; the determinant removes c_g exactly.
Each update preserves this relation: its denominator transforms by the
inverse congruence, canceling column scalings. By induction H transforms
to a scalar times A H A'. Hence

 L_H=cond(R^-1/2 H R^-1/2)

has exactly the same law as cond(H) computed on independent standard
Gaussian blocks of the same dimensions. This statement holds for FINITE K,
not an asymptotic approximation or a convergence claim.

Generate B fresh independent reference fits H_b; C=max_b cond(H_b).
By exchangeability P(L_H>C)<=1/(B+1), including ties conservatively. This
is an exact Monte Carlo rank argument (Dufour), not bootstrap coverage.
The probability integrates reference randomness; a fixed reused table is
NOT covered conditionally by the result. Record and lock the analysis seed,
never rerun auxiliary seeds to choose more discoveries. Equal folds may
share C; union bound still charges each of the two actual shape fits.

For any observed target Y and any a, on L_H<=C,

 (a'Ra) tr(R^-1 YY') <= C (a'Ha) tr(H^-1 YY').

Proof: eigenvalues of R^-1/2 H R^-1/2 lie in [l,u]. Therefore a'Ha>=l a'Ra
and tr(H^-1 YY')>=tr(R^-1 YY')/u. Their product is at least l/u times the
true product. Multiply by C>=u/l. This is a GLOBAL algebraic envelope,
not a finite direction grid or a local maximization certificate.

Thus U>=kappa and L_H<=C imply conservative positive-tail probabilities
from scale U*C*(a'Ha)*tr(H^-1 YY')/D. Nuisance R, radial law, scale and
tail index have now been protected or eliminated, not simply fitted.

## Proposition 4: PC, learning, and full family error control

Partition genes deterministically into two independent folds. For a gene
in fold f, define T_f=sigma(other-fold target blocks, calibration blocks,
auxiliary reference randomness). Learn any nonnegative directions supported
on each of the four S-1 study subsets, mixture weights in [0,1], and fixed
decreasing unit-integral p-to-e calibrators using only T_f.

Ordinary intersection p=min(1,(S-1)*min_{s in I} p_s). Learned projection
intersection p is the protected positive-tail test along its nonnegative
direction. For either component p_PC=max_I p_I. If the signed PC null is
true, there exists an all-null subset I_0 of size S-1, including singleton
or mixed-sign means. p_PC>=p_(I_0), so it is superuniform on a valid nuisance
configuration. Neither study independence nor a global all-zero null is
required. The opposite sign is a separate claim in the same 2G family.

Let B_f={kappa<=U, L_(H_f)<=C}. It is T_f-measurable for a fixed distribution
P (true nuisance values are constants, though unavailable to the algorithm).
For almost every T_f on B_f the held block still follows its original M0
law, and the protected p_PC is superuniform conditional on T_f.
Each selected calibrator therefore has conditional expectation <=1;
TRAIN-measurable convex mixing preserves that bound. No own-Q weighting.
E[1_(B_f)E_i]<=P(B_f)<=1 for a true-null i in fold f.

Let B=B_0 intersect B_1. With common calibration, union bound gives
P(B^c)<=delta_kappa+2/(Brefs+1)=delta_total. Apply e-BH to the full signed
family at q0=q-delta_total>0. Pointwise on B,
FDP<=q0/(2G) sum_(null i) E_i. Thus

 E[FDP] <= q0/(2G) sum_(null i) E[1_B E_i]+P(B^c)
          <= q0+delta_total = q.

We use 1_B<=1_(B_f), NOT conditioning on both training sets jointly.
Unbounded bad-event e-like values are not integrated; FDP<=1 is used there.
We do not assert unconditional individual E_i means<=1. This is an
approximate-compound/good-event argument (Ignatiadis et al., Prop5.4 and
Thm6.4), not a new general e-BH theorem. It does not justify the stronger
q0(1-delta)+delta formula in this cyclic-training construction.

## Efficiency / applicability consequences

1. With exact R, target df=D=20 rather than L=5, eliminating the previous
   structural loss of discarding contrast coordinates. With unknown R the
   explicit price is C; smaller confidence budget also reduces q0.
2. U/kappa has a distribution independent of rho and radial law. Therefore
   relative kappa coverage width does not intrinsically diverge as rho->1
   in M0. Finite precision and departures from compound structure are separate.
   No new sample-complexity rate is claimed.
3. Classical single-test protection p_BB=min(1,p+delta_kappa+delta_shape)
   is valid by the same bad-event decomposition. For G256, delta=.01,
   BY at.05 cannot reject because all p>=.01>.05/H_512. This deterministic
   floor is not a Monte Carlo Power finding. Historical TRAIN-calibrated BB_eBH
   is retained ONLY as EMPIRICAL_ONLY: its BB p is marginally valid, which does
   not justify a TRAIN-learned calibrator. BB_BY uses fixed BY and is the valid
   classical reference. This gap does NOT affect restricted-event R2_main or
   the separately PILOT-conditioned predictive bridge proof.
4. If externally known kappa_target<=Delta*kappa_cal, replace U by Delta U.
   All proofs persist. Unknown Delta is not identified by calibration alone;
   any reported Delta curve is sensitivity analysis, not learned transport.
5. Arbitrary dependence among final evidence is allowed, but target blocks
   must be independent of their TRAIN. Correlated genes cannot be excused
   by the arbitrary-dependence e-BH statement.

## Proof status / remaining gaps

Algebra above is explicit and ready for independent review, not peer review.
Main implementation matches a restricted S4/J6 model with common block radial
factor and SPD shapes. Float64 inverse, eigensolver and t tails are audited
numerically, not fully interval-certified; this limitation must accompany
any statement about the executable. Failures must be retained, never dropped.
Random reference seeds are part of the randomized algorithm, not a new
statistical confirmation opportunity. Novelty of the combined task-specific
construction is UNESTABLISHED; classical identities, rank calibration,
Tyler updates, PC and bad-event e-BH are attributed, not renamed inventions.

## Proposition 5: nuisance-integrated dominating reference (primary)

Use four equal independent gene folds. TEST=f, PILOT=f+1 mod4, TRAIN=other2.
H is Proposition3's fixed-K shape on TRAIN contrasts only (K=2 in the release).
Let kappa_hat=median(A_cal)/c_F, c_F=(1+sqrt(2))/2, using N independent centered
calibration blocks. By Proposition1 B=kappa_hat/kappa has the exact law of
median(N iid F4,2)/c_F, including the average of middle order statistics for
even N. No finite mean of F4,2 or prior distribution on kappa is assumed.

For a TRAIN/calibration-measurable nonnegative direction a on an intersection,
put H0=R^-1/2 H R^-1/2, b=R^1/2 a, r_a=b'H0b/||b||², Q=tr(H0^-1 EE').
The observed signed statistic is

 T=[Z_a+d*a'mu/(sqrt(V*v)||b||)] / sqrt[B*r_a*Q/20],

where conditional on TRAIN/calibration, Z_a is standard normal independently
of E. Its distribution does not depend on the selected a, so it remains
independent after integrating TRAIN. Under the all-null signed intersection
d*a'mu<=0. For T>0, necessarily Z_a>0, and r_a>=lambda_min(H0) implies

 T <= W0=Z_a/sqrt[B*lambda_min(H0)*Q/20].

Diagonalization conditional on H0 yields Q=sum_j U_j/lambda_j(H0), U_j iid
chi-square5 independent of H0, B and Z_a. The product with lambda_min is
scale-invariant, so Proposition3 gives a nuisance-free law from a fresh
standard-Gaussian TRAIN fit. H0's orientation need not be pivotal; eigenvalue
ratios suffice. a may depend on H/B; the Rayleigh envelope handles selection.
It need NOT be attainable by a nonnegative triple-supported a: domination,
not a sharp least-favourable distribution or optimality, is asserted.

Draw M iid full W tuples: independent Gaussian TRAIN, independent N F draws,
independent four chi-squares and normal numerator for EACH tuple. References
are independent of ALL real data. r(t)=(1+#{W_b>=t})/(M+1). W0 is independent
of this bank and has its law; its rank is superuniform. If T>0 then
r(T)>=r(W0); otherwise set p=1. Thus p is superuniform. It is the coupled W0,
not the learned-direction T, that is exchangeable with the bank. No reference
failure probability or uncorrected estimated tiny tail is discarded.

Conditional on the entire PILOT fold, TRAIN/TEST/calibration/reference joint
law is unchanged. Therefore the preceding p validity holds conditional on
PILOT. Apply the Proposition4 null-subset PC argument separately to ordinary
Bonferroni and projected intersections. Gamma and both decreasing calibrators
must be PILOT-only: its own shape/mean-scale heuristic, profiles and scores;
NO shared calibration, inference TRAIN or reference bank may enter them.
The PILOT's own scores need not be valid p-values. They only choose functions.
Conditional normalized calibration and convex mixing yield E[e_i|PILOT]<=1.
Integrate each claim over its own PILOT field. Pointwise e-BH self-consistency
then gives FDR<=q*m0/(2G)<=q=.05, despite cyclic fold and reference sharing.
This is genuine null-mean-bounded evidence, unlike Proposition4's restricted
good-event evidence. Do not condition on all PILOT/TRAIN folds jointly.

## Proposition 6: grid normalization, an existing principle specialized here

Set n=M+1. Projection PC support is {k/n:1<=k<=n}; ordinary PC support is
{3k/n:3k<n} union{1}. Min/max of ranks and clipped Bonferroni multiplication
preserve these supports. For either ordered support 0=t0<t1<...<tK=1,
let h be the original PILOT-selected nonnegative decreasing calibrator and
C=sum_k(tk-t(k-1))*h(tk). Conditional on PILOT, Fk=P(p<=tk)<=tk. Hence

 E[h(p)|PILOT]=h(tK)+sum_(k<K)[h(tk)-h(t(k+1))]*Fk
             <=h(tK)+sum_(k<K)[h(tk)-h(t(k+1))]*tk=C.

Use h(p)/C if C>0, zero if C=0. In the latter case h is zero at every allowed
support point. C<=integral h<=1, so all evidence and e-BH rejections weakly
increase relative to the continuous calibrator with identical data/weights.
FDP need not decrease. Support must be enforced; h/C is not valid unchanged
at arbitrary off-grid p. The bound is sharp over all superuniform grid laws,
NOT necessarily over the smaller attainable PC class. This is a corollary of
standard monotone calibration (Vovk–Wang2020 Prop2.1), not a novel theorem.

The grid upper evidence caps are n for projection and n/3 for ordinary.
For n4096,m512,q.05, at least3 total discoveries (or8 pure ordinary) are
necessary for any e-BH rejection. This is one residual finite-reference cost,
not a fundamental impossibility for all tests. Changing simulation counts
or gambling on reference seeds after seeing outputs is prohibited.

## Proposition 7: externally bounded kappa mismatch and its information limit

Let calibration have kappa_c, target common kappa_t, with external Delta>=1
such that kappa_t/kappa_c<=Delta. Keep training profiles/gamma unchanged and
replace inference kappa_hat by Delta*kappa_hat. Then the denominator of
Proposition5 has factor B_c*Delta*kappa_c/kappa_t>=B_c. Positive-tail
domination follows using the same reference law. Subsequent PC, conditional
calibration and e-BH proof is unchanged. This covers pipeline noise scale
ratio mismatch, NOT arbitrary covariance drift, gene dependence or unequal
pipeline locations. Delta chosen from target test results is not covered.

No uniform finite upper confidence bound on kappa_t/kappa_c can be learned
from calibration plus TRAIN contrasts ALONE in this model. Explicitly hold
calibration distribution fixed and target contrast scatter A=(1-rho)R fixed.
For every rho in(-1/5,1), set R=A/(1-rho) and V=1. Y has the same law for
all rho, whereas kappa(rho) ranges(0,infinity). If U of these inputs is finite
a.s., then P(U>=kappa/kappa_c) tends0 as rho tends1. Thus it cannot have a
uniform1-delta coverage probability for any delta<1 over these distributions.
This elementary subexperiment result does NOT exclude using target means,
known negative controls or new calibration; it is not an impossibility theorem
for the complete observed input or fixed-effects testing. It is not claimed
original. Fixed Delta sensitivity curves are not evidence that a real dataset
has that bound. Severe historical drift remains outside unbounded M0.

## Remaining theoretical/innovation status

The complete ideal-model chain above and the classical envelope reference
have explicit proofs and internal independent review, not external peer
review. No theorem about correlated genes, arbitrary nonelliptical tails,
unrestricted transfer, clinical input feasibility, minimax efficiency or
publication priority is established. Radial invariance avoids requiring
variance; E sqrt(V)<infinity is still needed for the directional mean target.
