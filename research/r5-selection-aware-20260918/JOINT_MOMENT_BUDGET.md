# One new work package: two-boundary JOINT expectation calibration

2026-09-18 19:08HKT. UNDER INTERNAL CHALLENGE, not accepted/newness certified.
This replaces the failed A0.8 candidate, retains A0.7.1 as reference, and does
not add a method grid. G1/G2/G3/G4 remain OPEN. No new large experiment planned.

## J1: no interior maximum for a hard selected power budget

Condition on W and independent meta. Independent conditional errors X=Es,Y=Et
have positive log-concave densities fs,ft on R. Take a in(0,1), r>0, b=ra,
k>0,c real, and finite Mt=E exp(-rY), Mb=E exp(-bX-r(1-a)Y).
For legal s>=0, U_s=X+s-Y, select borrowing if U_s<=c. For the independent
actual positive pivot V_+, define alpha=2r and

 w_s=V_+^alpha * exp(-rY) *
       [1(U_s>c)+k*exp(-b U_s)*1(U_s<=c)].

Write MV=E V_+^alpha. Then

       sup_{s>=0} E w_s = max(E w_0, MV*Mt).             (J1)

This is an EXPECTATION statement, not stochastic domination, p-value rank
validity, or permission to calibrate w using a boundary rank.

Proof. Let FB be the distribution of X-Y under joint tilt exp(-bX-r(1-a)Y),
and FT that under exp(-rY). Their densities satisfy exactly

       fB(u) = (Mt/Mb)*exp(-b u)*fT(u).

Both densities are log-concave, since tilted fs/ft are log-concave and so is
their convolution. With x=c-s, m(s)=E w_s/MV equals

       k*Mb*exp(-b s)*FB(x)+Mt*(1-FT(x)).

Differentiation gives

 m'(s)=Mb*exp(-b s)*[(exp(bc)-k)*fB(x)-kb*FB(x)].

If exp(bc)-k<=0 it is negative. Otherwise fB/FB is nonincreasing in x,
hence nondecreasing in s; the derivative can change sign only from negative
to positive. There is no interior maximum. Continuity and lim m(s)=Mt giveJ1.
The k=0 case is target branch only and follows by monotonicity/limit.

For a general randomized threshold, averaging J1 does NOT automatically move
max outside the average. Reviewer supplied a stronger direct proof; main checked
its sign/cross-integral and dominated-convergence steps on19:12HKT, before any
prototype Power outputs:

For ANY nonincreasing 0<=g<=1 with g(+infinity)=0, write under the target tilt

 m(s)/Mt-1=E_T[g(U+s)*(k exp(-b(U+s))-1)] = P_s-N_s,
 A(z)=g(z)*(k exp(-bz)-1)_+, B(z)=g(z)*(1-k exp(-bz))_+.

A is nonincreasing, soP_s<=P_0. A is supported below z*=logk/b and B above.
The log-concave location density fT(z-s) has increasing LR relative fT(z).
Integrating its cross-product over z_low<=z*<=z_high gives
N_s P_0>=N_0 P_s. IfP_0>0, P_s-N_s<=P_s(1-N_0/P_0)
<=max(P_0-N_0,0); ifP_0=0 direct. Dominated convergence uses
k exp(-bU)+1 integrable under target tilt, givingm(s)→Mt. ThusJ1 HOLDS also
for the actual logistic gate and arbitrary independent-meta k. The proof does
not assert smooth derivative has only one sign crossing, or exchange max/average.
k=0 follows directly. Gate density log-concavity is unnecessary here.

If k>=Mt/Mb CERTIFIED, tilt givesFB>=FT and m0>=Mt, so boundary alone suffices.
Estimated k is not such a certificate: prototype keeps TWO endpoints. Its gate
can now be the SAME logistic gate asA0.7, isolating the joint-budget mechanism.

The conditional logF model supplies all moments for0<alpha<min(5,4Nt,4Ns/a),
with the actual learned-shape moment certificate in ADAPTATION_MOMENT_COST.md.
Use alpha4 initially, not a tuning grid. a,c,centering and positive k may be
chosen using only W and independent reference-meta draws, not observed means.

## J2: exact finite-randomness normalization, NOT inverse empirical mean

Need a nonnegative independent random factor R with E[R]<=1/max(M0,Minfty).
Two NONNEGATIVE UNBIASED inverse estimators R0,Rinfty suffice:
R=min(R0,Rinfty), regardless of their mutual dependence. Each expectation must
be proved, not approximated with1/samplemean. Then E[R*w_s]<=1 for ALLlegal s.

Proposed constructive inverse estimator (under verification):

1. Work with unnormalized conditional logF product densities fs0,ft0. Their
   normalizers Zs,Zt are not assumed known. Independent tangent-envelope
   importance averages supply nonnegative unbiased estimates Zshat,Zthat.
2. Exact log-concave tangent envelopes also majorize each exponentially tilted
   error density. Their piecewise exponential normalizers C are analytic.
3. For the actual shape pivot use the coupling V_+<=W=2(t5)_+.
   B=E W^alpha=20^(alpha/2)*Gamma((alpha+1)/2)*Gamma((5-alpha)/2)
                  /(2 sqrt(pi) Gamma(5/2)).
   Under W^alpha size bias, first row of normalY has radius^2~chi2(5-alpha),
   uniform direction in R5; remaining rows standard normal; H law unchanged.
   Z^2~chi2(alpha+1),Z>0 integrates out of the acceptance ratio. Accepting with
   (V/W)^alpha=(||Yrow1||^2/[H11 tr(Y'H^-1Y)])^(alpha/2) is a Bernoulli
   experiment whose success rate is MV/B. This is not a t20 approximation.
4. Combine this with the mixture envelope of the TWO error terms in w0.
   Rejection success rate is J0/C0 where J0=Zs*Zt*M0. If N is the exact number
   of IID proposals through L-th success, E[N/L]=C0/J0. Thus
   Zshat*Zthat*N/(L*C0) is unbiased1/M0 if numerator estimates and trials are
   independent. At infinity only target tilt is needed; Zthat*N/(L*Cinfty)
   estimates1/Minfty. Count the index ofL-thsuccess, not batch overshoot.
5. Finite cap failure returns whole-family0, a pointwise decrease from the
   ideal nonnegative construction. No accepted-case conditioning or retry until
   a favorable normalizer. All cap/roundoff/normalizer failures retained.

Unbounded negative-binomial stopping has finite expectation when acceptance
positive, but actual code must have safe cap and report expected/observed cost.
Ordinary floating operations still require explicit guards/audit; an abstract
exact sampler is not automatically a certified implementation.

## J3: complete directional PC e chain, subject to implementation audit

For any signed scalar/component or positive-direction subset, use R*w(T,u).
It is increasing in positive T and zero at nonpositive T, so central-null
domination extends to signed one-sided nulls under the same model. For2-of4PC,
ordinary e is min over3-subsets of the mean of marginal e-values; projected e
is min over3-subsets of their positive-direction e-values. Under a PC null,
one3-subset is all-null, giving an expectation bound<=1. No independence
between studies or these e-values is needed for that final argument.
DisjointPILOT may choose convex ordinary/projected mixture; retain full original
DIR/SHAPE independence/rotation requirements. eBH then controlsFDR q.05.
This replaces PC-p-to-grid-e with DIRECT PCe; comparisons MUST include target,
source-bound, fixed bridge and fixed mixtures with the SAME direct-e interface,
not attribute every gain to borrowing. Old methods remain visible too.

## Novelty and next actions

J1's exact tilted-density relation and reverse-hazard argument need targeted
comparison to ordered-parameter pretest risk/MC/e-value theory. The statement is
not simply fixed-bridge validity, but no important novelty pass is yet claimed.
J2 uses classic importance/rejection sampling and inverse Bernoulli means;
new implementation alone isn't an original theorem. No papers/SCI claims.

Current action: main write auditable sampler/prototype and small correctness
fixtures; sole reviewer challengeJ1/J2 and closest precedent. No observation
Power batch before a correct working chain and meaningful contribution check.
Next soft check20:38HKT; this is a management checkpoint, not a hard deadline.
