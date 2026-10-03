# R4 complete mathematical chain and target-information boundary

Status: derived exact-arithmetic scoped propositions; independent review of
T1–T5 received 2026-09-18. Implementation/empirical/novelty grades are separate.
No new general Monte Carlo, PC, e-value, or information inequality is claimed.
Sources are bound in r4_common.py; predecessor equations in frozen R3 C001
R3_THEORY.md and R2 C001 THEORY.md. Old C2 R4_THEORY.md remains unchanged.

## Model, inputs, and hypotheses

For g=1,...,G, Zg=mu_g 1_6' + v_g R^(1/2) E_g C(rho_t)^(1/2), where R is
unknown SPD shared across target blocks, E entries iid N(0,1), v_g>0 independent
of E and all blocks independent. Block radii may have different laws, including
infinite variance; mean interpretation requires E v_g finite. Pipeline means
must be common. C(rho)=(1-rho)I+rho11', -1/5<rho<1. All target noise ratios
equal kappa_t=(1+5rho_t)/(6(1-rho_t)). Signed PC H_gd: fewer than two studies
have d mu_gs>0, d=+1,-1. There are m=2G claims; singleton and opposite-sign
effects are null, not excluded. FDP=FP/max(1,TP+FP); FDR is its expectation.

Cs[Ns,4,6] and Ct[Nt,4,6] are mutually independent and independent of the
ENTIRE Z dataset. Each bank consists of independent, genuinely zero-location
blocks with Gaussian angular pipeline compound symmetry. Source ratio kappa_s,
target-bank ratio kappa_t. Each calibration block may have its own nonsingular
left transform independent of its Gaussian angular matrix and positive radius:
the matrix pivot eliminates these. A bank centered by subtracting its own
observed pipeline means is NOT a valid zero-location bank. Neither real Ct nor
an externally justified drift radius is presently certified for application.

## T1. Two-bank pivot and target-only validity

Let Q'Q=I5,Q'1=0; x=C1/6,Y=CQ. Then

    A=.5 x'(YY')^(-1)x = kappa X,  X~F(4,2),
    f_X(x)=8x/(1+2x)^3, x>0.

Proof: x and Y arise from orthogonal Gaussian pipeline directions. After
canceling invertible left transform and radius, x=sqrt(kappa) z and Y is a
4x5 iid Gaussian contrast matrix up to their common scale. Thus YY' is
Wishart(I4,5), independent of z~N4(0,I4); z'(YY')^-1z=(4/2)F(4,2).
Multiplication by .5 gives the expression. Independent blocks yield independent
pivots, and the two banks are independent. This reduced pivot is NOT asserted
to be sufficient for the whole matrix experiment.

Let c=e/2, Kj=exp(mean log Aji)/c, Bj=Kj/kappa_j. Then Bj has the exact law
exp(mean log Xji)/c, independent of all targets. Frozen R3 with Ct and its
own sample count Nt is therefore directly valid for target kappa_t, regardless
of kappa_s. Its API requires Nt>=4; this is NOT a mathematical minimum for
every conceivable method. Nt=0 is unavailable, and Nt=1 is not silently enabled.

## T2. Fixed safe bridge and its efficiency cost

Assume an external fixed D>=1 really satisfies delta=kappa_t/kappa_s<=D.
Fix lambda in[0,1] before observed inference data (this implementation uses
only0,1,Nt/(Ns+Nt)). Define

    K=D^(1-lambda) Ks^(1-lambda) Kt^lambda,
    B=Bs^(1-lambda) Bt^lambda,
    K/kappa_t=(D/delta)^(1-lambda) B.

No test of delta=1 is performed, and D is not selected from either bank.
Lambda1 ignores source for inference and learning; lambda0 is frozen source
bound. Interior DIR learning uses the target median exactly as target-only.
The source changes only K and its reference-law size, not the target learner.
Multiplying Ks by an estimated ratio Kt/Ks would simply reproduce Kt and is
not implemented or called borrowing.

With L=log(D/delta)>=0, v=pi²/3−1,

    E log(K/kappa_t)=(1-lambda)L,
    Var log(K/kappa_t)=v[(1-lambda)^2/Ns+lambda²/Nt].

The fixed lambda=Nt/(Ns+Nt) minimizes this variance, giving v/(Ns+Nt).
At this lambda, B has EXACTLY the pooled Ns+Nt geometric-pivot error law.
The log-MSE is v/(Ns+Nt)+(Ns/(Ns+Nt))² L². Target-only log-MSE=v/Nt.
Thus bridge has smaller log-MSE exactly when

    L² < v(Ns+Nt)/(Ns Nt).

This is a scale-estimation comparison for this fixed class, NOT a Power
dominance result and NOT unbiasedness when L>0. At fixed total sample count,
all-target calibration removes the L term. At arbitrarily small delta, any
positive source weight has arbitrarily large conservative log bias. There is
no uniform MSE dominance over the whole one-sided drift class. All statements
follow from independent centered log-F sums; no asymptotic approximation.

## T3. Joint shape law, signed tests, and the complete family theorem

Use TEST=f, PILOT=f+1, DIR=f+2, SHAPE=f+3 mod4, with G>=16 divisible by4.
SHAPE uses contrasts only, determinant-initialized trace-normalized shape with
exactly two Tyler updates. DIR fits its OWN shape and nonnegative directions;
PILOT independently fits its own shape, heuristic median, profiles, calibrators
and gamma. PILOT sees no inference bank, DIR, SHAPE, TEST or reference numbers.

The initialization mean(W/det(W)^(1/4)) is affine-equivariant up to positive
scale and invariant to separate block radii. Each fixed Tyler step preserves
these properties. Column normalization has no effect because each update
uses xx'/(x'H^-1x). There is no ridge, diagonal rescaling, or adaptive stopping.
Consequently H=c0 R^(1/2) H_* R^(1/2), where H_* is the same fit on G/4
standard Gaussian 4x5 blocks; c0 may be random and cancels from projected
shape times inverse contrast energy. H_* has an orthogonally invariant law.

Conditional on DIR and both banks, a nonnegative unit-normalized direction a
is fixed and independent of H_*. Set b=R^(1/2)a/||R^(1/2)a||. An orthogonal O
mapping e1 to b gives (b'H_*b,eigen(H_*)) =d (H_*11,eigen(H_*)), because
O'H_*O=d H_*. This joint statement applies for every b and bank realization;
it removes their dependence from the resulting central marginal law. It does
not make the entire vector of target scores jointly pivotal.

For a true signed triple intersection, the projected signed location is <=0.
The statistic is its nonpositive shift plus the centered statistic, with the
same positive denominator, and hence is no larger. After cancelling its common
radius and the nuisance study shape, the centered statistic satisfies

    T0 =d (delta/D)^((1-lambda)/2) W,
    W=Z0/sqrt[B H_*11 sum_j Uj/eigen_j(H_*) /20].

Z0~N(0,1), Uj iid chi-square5, H_* and B are independent. H11 and eigenvalues
come from the SAME H. Gaussian mean/contrast independence gives the Z0/U
independence; rotation diagonalizing H leaves the contrast Gaussian law fixed.
For the count weight, generate B from exactly Ns+Nt iid F42 variables; grouping
into the two independent banks gives the identical law, not an approximation.

Draw M=4095 independent ENTIRE reference tuples, freshly per analysis and
independently of all actual observations. Because the shrinkage factor is<=1,
each positive-tail score is stochastically dominated by its central W0. For
T>0, p=(1+#{W_b>=T})/(M+1), otherwise p=1. Central exchangeable ranks and
conservative ties prove superuniformity. The contraction argument uses only
positive tails: multiplying a negative W by a number<=1 is not itself an
upper stochastic domination on the whole real line.

For each triple I, ordinary p=min(1,3 min_s p_s) is valid by union bound;
the nonnegative projection p is also valid. Under PC null there is an all-null
triple I0. P=max_I p_I is at least the valid p_I0. This includes partial-null
configurations, arbitrary dependence within a block and both signed claims.

Condition on a claim's OWN PILOT only. Its decreasing nonnegative calibrator
h and convex mixing gamma are fixed; all preceding inference randomness keeps
the asserted distribution. On exact support0=t0<t1<...<tK=1, summation by parts:

    E h(P)=h(1)+sum_{j<K}[h(tj)-h(tj+1)] Pr(P<=tj)
          <= sum_j(tj-t(j-1))h(tj)=Z_h.

Ordinary support uses multiples of3/(M+1) plus1; projection uses1/(M+1).
The original grid normalizer and downward quotients are retained. Zero Z_h
returns zero. Each component e=h(P)/Z_h has conditional mean<=1; their
PILOT-selected convex mixture also does. Integrating over each own PILOT
gives marginal E e_i<=1. Do not condition on the union of all cyclic PILOTs.
e-BH atq=.05 with m=2G satisfies pointwise

    FDP <= (q/m) sum_{i true null} e_i,
    FDR <= q m0/m <= .05.

Shared calibration and references induce allowed dependence; no per-gene
coverage charge is inserted. The guarantee integrates banks AND reference
randomness. It is not conditional-on-fixed-bank validity, not an arbitrary
distribution-free theorem and not full floating-point certification.

If delta>D and lambda<1, the shrinkage becomes inflation. The proof no longer
gives nominal FDR; report that boundary without repairing it using truth.
Heterogeneous target R, gene dependence, noncommon pipeline location and
non-Gaussian angular structure remain different out-of-model failures.

## T4. Unknown-drift nuisance information, pivot-only scope

Observe ONLY log A_s and log A_t. Set eta=log kappa_t, zeta=log delta, so
eta_s=eta-zeta. Density of logF42 is g(y)=8e^(2y)/(1+2e^y)^3. Its location
score is3U−2 with U=2X/(1+2X)~Beta(2,1), giving Fisher information1/2.
For Ns,Nt>0 the information matrix in(eta,zeta) and its Schur complement are

    I=.5[[Ns+Nt,-Ns],[-Ns,Ns]],  I_eta.zeta=Nt/2.

At any strict interior zeta<logD, the one-sided bound does not remove a local
nuisance direction. Regular locally unbiased estimation of eta has variance
at least2/Nt. Boundary-constrained or biased estimators are not covered by this
pointwise CRB. Source samples CAN improve a biased protected procedure at some
drifts (T2), without contradicting the CRB. No full-matrix sufficiency claim.

## T5. Exact finite-sample target-information requirement (chosen extension)

Fix the same eta_s at two parameters, eta_t0 and eta_t1=eta_t0+h, h>0,
both satisfying the external upper bound. Source laws are IDENTICAL, for any
Ns. The single target-pivot Hellinger affinity is

    a(h)=integral sqrt(g(y)g(y-h)) dy = sech²(h/4).

Full derivation: set r=e^(-h), u=2exp(y). The integral is
2r integral_0^infty u/[(1+u)(1+ru)]^(3/2)du. For0<r<1, substitute
v=sqrt((1+ru)/(1+u)), from1 tosqrt(r). This becomes
4r/(1-r)² integral_sqrt(r)^1(v^-2−1)dv
=4sqrt(r)/(1+sqrt(r))²=sech²(h/4). At h0 the integral is1; symmetry covers
negative h. Independence makes the two-bank affinity a(h)^Nt: source cancels.

For any two-point test, its errors alpha0,beta1 obey

    alpha0+beta1 >= 1-TV(P0,P1)
                  >= 1-sqrt(1-a(h)^(2Nt)).

The first inequality is Neyman–Pearson/total variation. For the second, apply
Cauchy–Schwarz to .5 integral |sqrt(p)-sqrt(q)|(sqrt(p)+sqrt(q)); the two
squared integrals are2−2a and2+2a. No asymptotic or unbiasedness assumption.
Any estimator of log kappa_t induces a nearest-two-point test. Error implies
absolute estimation error>=h/2, so the sum of two squared risks is at least
h²/4 times the minimum total testing error. Taking half yields

    max_i E_i[(eta_hat-eta_ti)²] >=
      h²/8 * [1-sqrt(1-sech(h/4)^(4Nt))].

This is a TWO-POINT maximum-risk bound for any estimator based on these
pivots, including biased ones, NOT a risk bound at every parameter point.

Consequence for honest calibration intervals: suppose a pivot-only interval
has deterministic diameter<h and coverage>=1-alpha at both parameters,
0<alpha<.5. Testing whether eta_t0 is in the interval gives both errors<=alpha:
if eta_t1 is covered, eta_t0 cannot simultaneously be covered. Therefore

    Nt >= log[4alpha(1-alpha)] / [4 log sech(h/4)].

The ceiling is a necessary count for this stipulated width/coverage task;
it is not sufficient, is not the algorithm's API minimum, and is not a PC
Power guarantee. One-sided zeta<=logD allows pairs separated by any finite h
by putting both low enough. With a TWO-sided externally imposed drift range,
only pairs lying in that range may be used. Source cannot beat this bound on
its own because its law is identical at the pair. At Nt0, target eta is not
point-identified in this reduced two-bank experiment. This says nothing about
potential extra information in raw matrices, known radii or raw samples.

## Interpretation and novelty

T1/T3 extend the old pivot chain to genuinely new inputs; T2 quantifies exactly
when protected borrowing trades bias for variance; T4/T5 make the target-data
requirement explicit in a well-defined reduced experiment. These are useful
scoped analytical results, but standard invariance, information and two-point
tools. An exact affinity identity is not by itself proof of a novel high-level
theorem. No universal optimality, full PC minimax frontier or automatic unknown
drift adaptation is asserted. Review and deterministic algebra checks must be
distinguished from finite simulated FDR evidence and practical input availability.

## T5 corollary: reduced-experiment minimax RATE, not a PC frontier

Added during frozen C001 computation; changes no algorithm, protocol, data or
analysis. The frozen C001 copy contains T1–T5 already. This corollary strengthens
the interpretation of T5 within exactly the same chosen information route.

Let R*(Ns,Nt,D) be the infimum over ALL measurable estimators based only on
the two pivot banks of the supremum of squared log-kappa_t risk over
eta_s,eta_t inR with eta_t−eta_s<=logD. Ns arbitrary, Nt>=1. Then

    [2(1−sqrt(1−exp(−2)))]/Nt <= R*(Ns,Nt,D)
                             <= (pi²/3−1)/Nt.

Proof of lower bound: choose eta_s=0, eta_t1=logD and
eta_t0=logD−4/sqrt(Nt). Both points obey the drift bound. Put h=4/sqrt(Nt)
in T5. Since log cosh(x)<=x²/2 (differentiate: tanh x<=x forx>=0,
and use evenness), sech(1/sqrt(Nt))^(4Nt)>=exp(−2). The two-point risk bound
is at least (2/Nt)[1−sqrt(1−exp(−2))]. Taking the supremum over parameters
and then infimum over estimators preserves this lower bound. Randomization
independent of observations cannot improve the testing inequality.

Proof of upper bound: ignore source and estimate eta_t by mean log A_t−(1−log2).
Its exact uniform risk is(pi²/3−1)/Nt by the finite log-F variance. This
estimator exists in the reduced experiment forNt>=1; deployed API stillNt>=4.

Thus target-only geometric estimation has minimax-optimal RATE1/Nt up to the
displayed nonsharp constants for this reduced, one-sided-unbounded-drift loss,
even if Ns grows without bound. This does not prove exact minimax optimality,
exclude constant-factor improvements, or contradict local borrowing benefits.
The adversary uses different drift positions, including the boundary. A two-
sided narrow drift restriction changes allowable pairs and may change the bound.
Full matrices, known radii, raw target samples and final PC decision losses are
not covered. AtNt0, source-identical pairs can have arbitrarily large separation;
the same risk argument makes this reduced unrestricted supremum risk infinite.

The minimax-rate deduction uses the classical two-point device explicitly
proved above, not a claimed new information inequality or established priority.
