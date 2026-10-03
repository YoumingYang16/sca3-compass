# P-CAL: a proposed finite conditional-information boundary

Status: main derivation plus independent internal proof challenge completed;
priority/importance OPEN. No external peer review. Use the continuous conditional
version given by the product density, not an arbitrarily modified null-set version.
It is NOT the old R4 unconditional1/N risk bound, NOT a PC Power bound, and
does not by itself pass G1 or G3. It may explain when count-only borrowing
misrepresents observed calibration information. Sampling such residual patterns
is not asserted common; finite-pattern prevalence must be assessed separately.

## Statement

For N>=4 iid logF(4,2) errors X_i and centered residual vector
W_i=X_i-meanX, use the regular conditional law of Y=meanX from ANCILLARY_NOTE.
Let v=pi²/3-1, so unconditional Var(Y)=v/N (old result).

(i) If N=3k, sup_{sumw=0} Var(Y|W=w)=infinity. In particular take k residuals
-2L and2k residuals+L. Then Y/L conditional on that vector converges in law
AND second moment to Uniform[-1,2]. Thus Var(Y|W=w_L)/L²→3/4.

(ii) If N is not divisible by3, the same supremum is finite. An explicit bound
is Var(Y|w)<=C_r, where r=frac(2N/3)∈{1/3,2/3}, and

 h_r(x)=exp(3r*x)/(1-r+r*exp(x))^3,
 C_r=max{integral_{x>=0}x²h_r/integral_{x>=0}h_r,
         integral_{x<=0}x²h_r/integral_{x<=0}h_r}.

These are fixed finite constants, no dependence onw orN. Symmetry gives
C_1/3=C_2/3. This is an upper bound, NOT claimed sharp exact supremum.
In particular C_r<=9/2 is a simpler nonsharp numerical upper bound.

(iii) Nevertheless for every suchN the supremum is at leastv. Construct
k=floor(2N/3) entries+L, N-k-1 entries-L, one entry0, then center all entries.
After recentering Y by the induced deterministic shift, its limit is a shifted
logBetaPrime(3r,3-3r), whose variance is v. Hence a uniform conditionalO(1/N)
precision guarantee fails also along nonmultiples of3.

## Proof of(i)

The unnormalized conditional density equals, up to a positive constant,

 f_L(y)=[1+2exp(y-2L)]^(-3k) [1+.5exp(-y-L)]^(-6k).

For y=Lt with -1<t<2 both brackets tend1; outside[-1,2] at least one tends
infinity. The functions are bounded by1 everywhere. Beyond2 and below-1,
the factors give exponential envelopes with rates3kL and6kL in the scaled
coordinate. Split the integral into[-1,2] and the two tails: dominated
convergence in the interior, and these explicit integrable envelopes in the
tails, show L^-1 integral f_L→3, the first two scaled moments tend those of
Uniform[-1,2]. This proves the variance claim and unbounded supremum. Each
finitew_L has positive-density neighborhoods and the conditional moments are
continuous there, so this is not rescued by saying the exact vector has zero
probability. This does not say such neighborhoods are frequent.

Explicit UI detail: for L>=1 the scaled kernel is bounded by1 on[-1,2], by
2^(6k)*exp(6k(t+1)) on t<-1 and by2^(-3k)*exp(-3k(t-2)) on t>2.
These bounds multiplied by1+t² are integrable. Thus the first and second scaled
moments tend1/2 and1 respectively. N=3k is fixed in this limit.

## Proof of(ii)

Let m be the unique mode; p_i=sigmoid(m+w_i+log2), so sum p_i=2N/3=k+r.
For displacementx, F_x(p)=p expx/(1-p+p expx). If x>0 it is concave on[0,1].
With a fixed sumk+r, a concave sum is minimized at the extreme vector consisting
ofk ones, one r, and remaining zeros. Thus sum F_x(p_i)>=k+F_x(r). If x<0,
F_x is convex and the same extreme vector maximizes the sum, reversing the
inequality. The derivative of the log conditional density at m+x is
2N-3sum F_x(p_i). Therefore it is <=3(r-F_x(r)) forx>0, and >= that forx<0.

The latter function is d(logh_r)/dx. Consequently the ratio
f(m+x)/h_r(x) is nonincreasing on the positive halfline and nondecreasing on
the negative halfline. Normalize separately on each side: the likelihood-ratio
order bounds the second moment of displacement conditional on each side by
the corresponding h_r halfline moment. Average over the actual side masses
to obtain E[(Y-m)²|w]<=C_r. Variance cannot exceed a second moment aboutm.
The integrals are finite since h_r has exponential slopes3r and-3(1-r).

For r=1/3, h=27exp(x)/(2+exp(x))³. Its negative/positive half-integrals are
15/8 and3/2. On x<0 use h<=27exp(x)/8, giving the conditional second-moment
bound(27/4)/(15/8)=18/5; on x>0 use h<=27exp(-2x), giving(27/4)/(3/2)=9/2.
Reflection covers r=2/3. These inequalities justify the explicit uniform bound.

This uses majorization/likelihood-ratio ordering as established tools. Whether
this particular finite logF conditional boundary is a useful independent result
requires review; do NOT claim those tools themselves are new.

## Proof of(iii)

Use the uncentered pattern above, shifting y correspondingly when centering w.
For fixed y the k positive extreme entries contribute right-tail slope-k; the
N-k-1 negative entries contribute left-tail slope2(N-k-1); the remaining entry
has densityg(y). After cancellation of constants, the limit kernel is

 exp((2N-3k)*y)/(1+2exp(y))^3 = exp(3r*y)/(1+2exp(y))^3.

Thus 2expY has BetaPrime(3r,3-3r) limit law. For r=1/3 or2/3 its log variance
is trigamma(1)+trigamma(2)=v. Moment convergence follows from part(ii)'s
envelope. In these uncentered coordinates the modes tend to the finite mode
of the limiting kernel, by convergence of strictly decreasing score functions.
Normalize each kernel by its height at the mode. Part(ii) bounds it by
h_r(y-m_L), with boundedm_L; the bound stays integrable after multiplication
byy². Dominated convergence gives normalization and second-moment convergence.
This envelope step needs independent review; numerical convergence is not proof.

Independent review supplied a shorter exact domination: with q=N-k-1 the
uncentered kernel after removal of constants equals
H_L(y)=H(y)*[1+.5exp(-y-L)]^(-3k)*[1+2exp(y-L)]^(-3q),
H(y)=exp(3ry)/(1+2exp(y))³. Thus 0<H_L<=H, and(1+y²)H is integrable.
Dominated convergence directly proves normalization and both moment limits.

## Limitations established by explicit counterexamples

For N=4,w=(3L/4,3L/4,-5L/4,-L/4), Y-L/4 has the shifted logBetaPrime(2,1)
limit. Conditional variance tends v but E[(Y-(1-log2))²|w]/L² tends1/16.
Thus 'precision' here means conditional variance, NOT conditional MSE of the
uncorrected geometric estimator or a general information measure.

For the part(i) construction the residual density in N-1 free coordinates is
N*8^k*exp(-6kL)*integral f_L(y)dy ~3N*8^k*L*exp(-6kL).
Since |(log g)'|<=2, a fixed small-radius neighborhood has probability of the
same order (constants depend on N and radius). This particular extreme pattern
is exponentially rare. This is NOT a bound on all high conditional-variance
events, but prevents claiming a practical common-case gain from this limit alone.

## Consequences and non-consequences

SameN can correspond to very different conditional precision. Observed residual
patterns legitimately inform reference distributions and potentially borrowing.
The boundary does not permit dropping unpleasant banks, altering the error
target, or labelling allN=12banks inferior toN=4. Marginal v/N remains exactly
true. No new patient information or identifiability claim follows.
An algorithm using these patterns still needs complete conditional reference,
fair conditional-endpoint ablations, G3 evidence and independent confirmation.
