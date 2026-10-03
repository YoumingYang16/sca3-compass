# Proposed connection from P-CAL to safe information-adaptive borrowing

Not an acceptance claim. Inverse-variance weights themselves are classical.
The issue is whether the P-CAL boundary and actual conditional input-aware
finite-calibration capability constitute important, useful new knowledge.

## Procedure A0.4

Using the same observed Ws,Wt, draw an independent meta reference (4095 errors
perbank, additional COMPUTATION not additional data). Its conditional sample
variances Vs_hat,Vt_hat give source coefficient a=Vt_hat/(Vs_hat+Vt_hat).
Conditional on Ws,Wt and this meta stream, a is fixed and observed bank means
still have their exact conditional distribution. Nonpositive/nonfinite variance
is an explicit conservative failure. No true scale, drift or truth is used.

Meta means ms,mt center observed logKs/logKt and reference errors identically.
This stabilizes computation. They need not equal population conditional means
for validity: they are fixed transforms given conditioning information.
With the A0.4.1 q=1 fallback, centering can also alter the algorithm on fallback
events; validity survives, but exact scale/centering equivariance is not claimed.
Use independent inner-target and inner-bridge scores to set qt,qb, then switch
iff corrected u<=2log(qt/qb)/a. The complete outer paired maximum at boundary
gives the valid single-test rank as before. Same PC/grid/PILOT/eBH chain.

## Model/ideal-arithmetic safety

The only new adaptive layer uses ancillaryW plus independent meta randomization.
Conditioning on both makes a,ms,mt fixed; the actual bank means and ALL subsequent
inner/outer references retain their exact conditional laws. Consequently the
co-designed pathwise coupling applies to any a∈(0,1]. This proves reliability
does not require consistency or exactness of the moment estimates. Integrating
meta and W recovers the marginal guarantee. Classical conditioning plus the
already-proved coupling, not a new generic e-BH theorem.

## A specific efficiency limit — distinguish oracle moments and implementation

First define ideal conditional moments mu_s,mu_t,Vs,Vt and center errors by
the conditional means. Let a*=Vt/(Vs+Vt). For a fixed actual legal slacks,
the difference between bridge and target centered logscales is
 a*[(Es-mu_s)-(Et-mu_t)+s].
Its absolute mean is at most a*(sqrt(Vs+Vt)+s).

For a central signed statistic V=Z/sqrt(R), with R>0 independent of Z~N(0,1),
the defective positive-tail log density is bounded by
 L0=sup_{z>0}z*phi(z)=1/sqrt(2pi*e).
Conditional shape/contrast mixtures preserve this bound. Conditioning on errors
and applying this log-scale Lipschitz bound therefore yields the SINGLE-TEST
CDF distance bound

 sup_t |P(Wbridge>t)-P(Wtarget>t)|
 <= (L0/2)*[Vt/sqrt(Vs+Vt)+Vt*s/(Vs+Vt)].

This is a CENTRAL-null distribution statement, not a Power or fullFDR-regret
bound. A noncentral numerator generally needs its own density/margin assumption.
It is invalid to label this bound a theorem about all alternative Power.

For the P-CAL source plateau with Ns=3k fixed and L→infinity, Vs~.75L² while
target residuals and Vt stay fixed. Then a*=O(L^-2) and the above bound vanishes
O(L^-1). Fixed count weights do not have this property; their centered source
logerror remains O_P(L). This addresses a concrete observed-information failure,
not a claim that increasingN harms average inference.

FIXED SLACK is essential. For s_L=d/a_L, the deterministic bridge logscale
shift remains d, even when the random discrepancy disappears. With independent
inner normalizers a max-selection cost can remain. Example limiting offsets
bt=1,bb=0,d=4: observed max(T-1,T-2)=T-1 versus boundary reference T.
Hence there is NO uniform convergence over all s>=0, and no such claim is used.

For the IMPLEMENTED finite-meta rule, sample source variance/L² converges in
distribution to the positive sample variance of a fixed number of iidUniform
draws. Thus a_hat=O_P(L^-2) and a_hat*(Es-ms)=O_P(1/L). Target meta quantities
stay finite. The selected normalized pair therefore coalesces to the target
pair in probability. With a fixed finite number of inner/outer tuples and
continuous positive reference comparisons, the probability that ALL finite raw
rank comparisons agree with target tends to one (not a.s. eventual agreement
deduced merely from convergence in probability). Conditional finite-family PC/e outcomes then approach
the SAME-OUTER-DRAW target endpoint, provided the target-side DIR/PILOT/SHAPE
and score function are coupled identically. Discrete e-BH threshold equality
requires care: p ranks exactly agree with probability tending to one, not merely
e-values approaching a threshold. This supplies the relevant stronger fact.

NORMALIZER/CAP QUALIFICATION (internal review, 2026-09-18): A0.4 as executed
in D005 had an additional nonvanishing probability of failure when a fixed-size
inner sample has a nonpositive .99 quantile. Thus that IMPLEMENTATION cannot
claim unconditional convergence to the no-inner-reference target endpoint.
A0.4.1 substitutes q=1 whenever that independent inner quantile is nonpositive,
records the substitution, and preserves strictly positive independent scales.
For these scales write B_i=T_i-Delta_i, S_i=max(T_i-bt,B_i-bb),
C=max(-bt,-bb). The bound |S_i-(T_i+C)|<=|Delta_i| cancels the potentially
different normalizers. Under the above coupling max_i|Delta_i| ->P 0; positive
finite comparison gaps have no atom at zero. On nonpositive numerators both
procedures return p=1, so those ties are harmless.

The exact convergence statement is for ideal arithmetic and unlimited exact
conditional sampling, not an unconditional statement about capped float code.
For the actual capped implementation the probability of disagreement is bounded
by the ideal rank-disagreement probability PLUS the probability of any numerical
or cap failure in either coupled procedure. No vanishing bound for that extra
term has been proved. Such failures return whole-family e=0 and preserve
conservativeness but can cost Power. The padding/envelope checks are safeguards,
not a universal floating-point exactness certificate.

This last limiting statement needs independent proof review, including the
normalization cancellation and alternative target-numerator continuity. It is
not an already-checked finite-L Power guarantee or an optimality theorem.

## Diagnostic and fairness

Synthetic deterministic sampler/unit fixture: Ns12 centered logA pattern
(-10)x4,(5)x8 has much lower conditional precision than Nt4 allzero residuals.
The observable-meta rule assigns source weight<.2, whereas count weighting
assigns.75. This is not a random-family Power result or evidence of prevalence.
Include a conditional variance-weighted FIXED bridge as a necessary ablation:
any improvement over the count bridge alone cannot be credited to selection.
Use the same existing80 DEV families, all mandatory original/conditional
endpoints and simple mixtures; no new confirmation is spent at this point.
