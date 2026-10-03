# R5 A0 development derivation — NOT acceptance or novelty certification

2026-09-18. All claims below concern the unchanged R4 M(D), ideal arithmetic.
Full finite-precision certification and final code challenge are not complete.
G1/G2/G3/G4 remain OPEN. Dufour(2006) supplies composite-MC validity; this
document does not call that fact, PC, p/e conversion or e-BH new theory.

## 1. Model and exact log experiment

Cs[Ns,4,6], Ct[Nt,4,6], Z[G,4,6] are independent blocks as specified by R4
C001/R4_THEORY.md. Both banks are zero-location, with independent Gaussian
angular factors, positive within-block common radial multipliers, invertible
study factors and compound-symmetry pipeline covariance. Target tests share a
study shape; mean shifts are common across pipelines. Ns,Nt>=4. G>=16 divisible4.
For j=s,t the observed calibration A=.5*x'(YY')^-1*x has law kappa_j F(4,2),
because left factors and radius cancel. Thus K_j=exp(mean(log A)-(1-log2)) and
E_j=log K_j-log kappa_j are independent centered means of logF42. Their variance
is v/N_j, v=pi²/3-1. This is inherited R3/R4, not a new identification result.

Let a=Ns/(Ns+Nt), delta=kappa_t/kappa_s<=D and s=log(D/delta)>=0. Observed
u=logD+logKs-logKt=s+Es-Et. c=sqrt(v*(1/Ns+1/Nt)) is fixed by counts.
b(u)=1{u<=c}; log(K_ad/kappa_t)=Et+a*u*b(u). True s, delta, kappa are NOT inputs.

## 2. Complete reference and independent branch transforms

Two independent inner reference banks yield fixed nondecreasing scores
phi_j(w)=1-r_j(w)/(L+1), with r_j(w)=1+#(R_j>=w) for w>0, else L+1.
Both maps are independent of observations and the outer reference. Their finite
ranks are not claimed valid post-selection p-values. Condition on these maps.

For each independent outer tuple, draw Es,Et with their exact finite-bank law;
draw the 2-step R4 shape H on G/4 independent Gaussian contrast blocks; draw
U_i~chi²5 and Z0~N(0,1), independently. Put
V=Z0/sqrt(H11*sum_i(U_i/eigen_i(H))/20). H11 and eigenvalues use the SAME H.
Under a central signed marginal/subset null this is the inherited R4 rotation
pivot, jointly with Es,Et. It is not a pivot for the joint family of genes.

For nuisance s>=0 the full selected score is
S(s)=phi_b(V*exp(-(a*Es+(1-a)*Et+a*s)/2)) if s<=h=c-Es+Et,
and S(s)=phi_t(V*exp(-Et/2)) otherwise.
The observed score uses the actual observable u and K_ad. Hold it and ALL
reference random numbers fixed while profiling s. Do not freeze the observed
branch choice for reference tuples. The observed value is NOT reselected as s
is varied; the reference distribution of the single fixed statistic is varied.

## 3. Exact event lemma and proof

For fixed observed rank k<L+1, let q_j=max(0,R_j[L-k]), using zero-based indexing.
Then phi_j(w)>=1-k/(L+1) iff w>q_j (strict, including inner ties).
If V<=0 the tuple never exceeds. Otherwise its borrowed exceedance set is
[0,h] intersect [0,tau), tau=2*(log V-(aEs+(1-a)Et)/2-log q_b)/a,
with tau=infinity if q_b=0. Its target exceedance set is (h,infinity) intersect
[0,infinity) if V*exp(-Et/2)>q_t. The first is a prefix or empty; the second a
suffix or empty. They cannot overlap for the same tuple. At h bridge applies.

Start with exact count at0. Sort all nonnegative prefix-end and suffix-start
events. Group equal endpoints before updating. Between successive events the
count is constant (using these threshold-crossing events), and exact endpoint
counts cannot exceed the larger adjacent count: an open borrowed cutoff can
only remove a tuple, while a target suffix starts strictly AFTER the switch.
At switch0, the separately evaluated0 count covers any isolated borrowed value.
Therefore the maximum of initial count and grouped cumulative changes equals
sup_{s>=0} C_k(s). The final count is the target-only count at infinity. Complexity
O(M log M) per distinct observed rank, memoryO(M). Caching avoids repeated ranks.

This is a deterministic lemma supporting an exact continuous supremum, not a
claim of optimal statistical power or a universal floating-point certificate.
Fixed epsilon probes are used only in independent tests, never by the optimizer.
The code also checks 0<=count<=M and count>=target-limit count, failing closed.

## 4. Finite single-test validity — classical composite MC instantiation

At the true s0>=0, under a central null and conditional on the independent inner
maps, S_observed and the M complete-reference scores S_b(s0) are identically
distributed and independent. Hence (1+sum 1{S_b(s0)>=S_observed})/(M+1) is
superuniform with ties conservatively retained. Taking the supremum over the
entire nuisance set yields p_* no smaller than this rank. Therefore p_* is
superuniform, on the integer outer grid. This is Dufour propositions4.1/4.2.
It is NOT uniform after profiling and is NOT conditional on realized Cs/Ct.

For a one-sided composite null, a nonpositive mean shift decreases the signed
numerator without changing pipeline contrasts, bank selector or its denominator.
The selected transform and final tail function are monotone; couple to the
central case to get superuniformity. A subset direction is nonnegative, learned
only on DIR with its OWN shape. DIR may depend on Ct; conditional on bank+DIR,
the rotated central target/independent SHAPE law is invariant to that direction.
Thus its pivotal V law is constant across the conditioning values, giving joint
independence from Es/Et when these variables are integrated. This argument uses
R4's Gaussian angular/common-radial/shared-target-shape assumptions, not merely
the fact that another gene or a different fold was used. It does not hold for
arbitrary study-dependent radial multipliers or covariance drift outside M(D).

## 5. PC and final chain

For a signed 2-of4 null, at least one 3-subset has all signed effects<=0. For each
3-subset, ordinary intersection p=min(1,3*min three valid marginal p's) is valid
by union bound. The positive-projection intersection p is valid by section4.
Taking the maximum over all3-subsets gives two valid PC p-values without any
assumption of study independence. This covers singleton signals and wrong-sign
signals, not just the global null. Ordinary support is {min(1,3k/(M+1))}; projected
support is {k/(M+1)}. Maxima preserve these supports.

For each TEST fold, PILOT, DIR, SHAPE are three distinct other folds. PILOT
calibrator and convex mixing weight use only PILOT blocks, no banks or global
references. Conditional on PILOT, sections2–4 still apply after integrating
the other independent inputs. A nonnegative decreasing step calibrator f,
normalized by its exact grid-null expectation Z=sum f(t_j)*(t_j-t_{j-1}), has
E[f(p)/Z|PILOT]<=1 by superuniformity. The inherited grid code performs that
normalization; its degenerate fallback must be zero evidence. Conditional
convex mixing with PILOT-measurable gamma preserves the expectation bound.

Flatten the two signed claims per gene to m=2G. For e-BH at q=.05, every rejected
i satisfies e_i>=m/(q*R). Therefore FDP<=q/m*sum_true-null e_i, pointwise (R=0
gives0). Taking expectations gives FDR<=q*m0/m<=q. Dependence across genes,
folds, shared banks and shared references does not require independence here.
This elementary e-BH argument is inherited, not a novel R5 contribution.
Clarification: arbitrary dependence is allowed among the resulting e-values
once marginal e-validity is established. It does NOT waive the input-block/fold
independence used to establish each marginal guarantee. Arbitrarily dependent
raw genes cannot be inserted into the learning construction without rechecking it.

## 6. A concrete protection limitation (reviewer challenge accepted)

For every fixed reference and observed score x, p_*(x)>=p_infinity(x), because
infinity belongs to the nuisance closure and each trajectory eventually uses
target. If the observed branch is target, monotonicity of phi_t means every
outer target raw value >= the observed raw value also has score>=observed score.
Thus p_* is at least the raw target rank using THESE SAME outer draws. This is
not a comparison to an independently generated frozen R4 reference or an e-BH
rejection-set inclusion with different pilot rules. Inner normalization cannot
remove this lower bound. Its size must be measured; no efficiency claim yet.

## 7. Gaps and status

This draft closes a model/ideal-arithmetic argument conditional on the inherited
R4 pivot and implementation matching it. Main-chain code exists and18 tests pass;
these are not a general statistical verification. Internal review of full new
kernel, adversarial tie/numerical audit, same-information performance, significant
independent new content and independent confirmation are still OPEN. D000 is a
shared-reference scalar diagnostic, not evidence of whole-family FDR control.
## 8. A0.1 continuous-inner amendment (16:16HKT, before D002)

D001 identifies a rank-saturation loss. Keep its code/results frozen. Replace
ONLY phi_j by a continuous fixed map generated by the same independent inner
references. At each positive ordered reference r_i set x_i=logr_i,
y_i=-log((L+1-i)/(L+1)), zero-based original reference indexi. Duplicate x values
retain the first index. Interpolate y linearly in x; extrapolate using the secant
through the first/last16intervals (or all if fewer). Clip negative scores to0;
nonpositive statistics have score0. For score>0 this map has an explicit inverse
log cutoff q_j(x). No Pareto/tail-fit correctness is assumed: it is only a fixed
statistic definition, never itself a calibratedp.

Exceedance now uses logW>=q_j(x). Prefix tau endpoint is closed, suffix switch
endpoint open. The same grouped event proof holds: a closed prefix can only be
present before its removal, while suffix additions apply after the point; exact
point count is at most the left/right maximum. Infinite tail is covered and not
clipped. Sections4/5 hold for any such independent nondecreasing maps. Exact
outer integer grid is unchanged; no inner grid is used in e-normalization.
28 tests including inverse/extrapolation, scan/brute-cell agreement and kernel
regressions pass. This removes a score atom, not the target-limit protection
floor, and is NOT claimed as an original theory contribution. D002 will use
the SAME80 development inputs, no new independent confirmation.
## 9. A0.2 co-design amendment (before D003)

SELECTION_COST.md B/C gives a necessary-and-sufficient PATHWISE frontier, and
the specific constant log-offset construction. In current code, reference draws
are generated first from three independent streams; inner99thpercentiles define
qt,qb andc*. Observed selection uses u<=c*, not the old count-risk threshold.
Condition on independent inner references: c* is fixed. The selected reference
score is max(logtarget-logqt,logbridge-logqb), and is nonincreasing ins for every
tuple. Calibrate using s0 only, with ties conservatively counted. For the actual
s0>=0, its selected observed score is stochastically bounded by the central
boundary score; coupling proves superuniformity against the iid boundary rank.
The observed inner-reference-conditioned choice uses Cs,Ct,D andc*, notTEST,
truth or an unbudgeted pilot. All inherited nuisance and PC/e dependence checks
still apply. Numerical failure returns zero e-values; float computations remain
not interval-certified. No claim that nuisance supremum cost and selection
cost are the same, or that classical max-score MC calibration is new.

The earlier section8 draft heading says16:16HKT; actual D002 freeze and execution
timestamps are in its machine records (started before16:16). The heading was a
manual timestamp approximation, NOT protocol timing evidence. The method/code
and protocol were frozen before any D002 outputs; do not change that old freeze.
## 10. A0.3 ancillary-conditional amendment (before D004)

ANCILLARY_NOTE.md derives the exact conditional calibration law given Ws,Wt.
The new reference integrates neither over unknown eta nor over an observed
meanlogA. It samples Y independently from productg(Y+Wi)/normalizer, then
Es=Ys-(1-log2),Et=Yt-(1-log2). These conditional errors are NOT centered again.
Three independent streams generate inner-target, inner-bridge and complete
outer tuples; within each tuple stream independent substreams generate source
errors,target errors and shapes/numerators. Inner maps are conditionally
independent of the observed means and outer references, which suffices.

Conditional on Ws,Wt and inner references, co-designedc is fixed; sections9's
full choice and drift coupling hold without requiring unconditional logF-mean
error laws. DIR may use fullCt; R4 rotational law is constant given fullbank
andDIR, hence also given Ws,Wt,Ys,Yt. PILOT is independent of these inputs;
the unchanged PC/grid/eBH argument holds conditional on Ws,Wt after integrating
bank means, inner/outer references and DIR/SHAPE. Integrating Ws,Wt gives the
same marginal FDR bound. No conditioning on fixedKs/Kt is claimed.

Sampler: three log-concave tangent envelope pieces, independent rejection draws,
not MCMC. Any valid supporting tangents suffice; exact mode optimization is not
required in ideal arithmetic. Completion cap is conservative coupled failure,
not a truncated approximation substituted for the conditional law. Numeric
envelope violations fail closed; no universal floating-point certificate.

D004 adds conditional target/source-bound/bridge and fixed half conditional
e-mixture. Target uses only target error draws andCt DIR; source-bound uses only
source error draws andCs DIR, matching frozen learning choices. Extra inner
normalizers are computed only for the adaptive candidate, explicitly costed.
Compare with all original baselines too. This separates conditional information
recovery from actual adaptive borrowing benefit. No conditional variance-based
weight changes or parameter scan in this batch. Protocol freezes before results.
## 11. A0.4 information-weighted amendment (before D005)

INFORMATION_ADAPTATION.md specifies the sole new layer. A fourth independent
reference stream, conditional on observed Ws,Wt, produces finite MC conditional
variance estimates; source coefficient is Vt_hat/(Vs_hat+Vt_hat), not Ns/Ntotal.
Conditional onW plus meta draws, the subsequent observed means and inner/outer
reference remain independent with the correct conditional law. The proof works
for EVERY resulting positive finite weight, not only accurate estimates. The
estimated conditional means center observed and simulated logscales equally;
they do not serve as nuisance truths. Include variance-weighted conditional
bridge and a half target/variance-bridge mixture to attribute gains correctly.
P-CAL and the proposed limiting information-adaptation property are under
review, not automatically established by this implementation. No G1/G3PASS.
