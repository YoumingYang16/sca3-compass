# Why finite calibration costs power even without confidence envelopes

Analytic deductions about the F4,2 median used by the frozen method. They do
not change the estimator or C001. Classical order-statistic reasoning, not a
claim of a novel general theorem. No confirmation outcomes used in derivation.

Let X_i=A_i/kappa iid F4,2. Its CDF is F(x)=(2x/(1+2x))², x>0,
its density f(x)=8x/(1+2x)³, and its median c=(1+sqrt(2))/2.
B_N=median(X_1,...,X_N)/c is exactly the scale error integrated by PB.

## 1. Small-N scale variance can be infinite

As x tends infinity, 1-F(x)~1/x; as x tends0, F(x)~4x².
For N=2m, the numerical median is (X_(m)+X_(m+1))/2. It lies between
X_(m+1)/2 and X_(m+1). For t large, X_(m+1)>t means at least m out of2m
draws exceed t, giving probability asymptotic to choose(2m,m)*t^(-m).
The sandwich therefore gives P(B_N>t)=Theta(t^(-m)).

For N4, m2: B has finite expectation but infinite second moment and infinite
variance. Using more precise raw floating arithmetic does not remove this.
For even N>=6, its second moment is finite. The estimator remains a valid
equivariant input at N4; the specified finite-N law is sampled in the ideal-randomness construction (not bit-exact continuous simulation), and no
variance assumption about B is used in the error-control proof. Do not report
a theoretical finite scale RMSE or small-N normal confidence approximation.
The final FDP/Power remain bounded, so their family-level uncertainty analysis
does not inherit an infinite-variance problem.

Similarly, X_(m+1)<=epsilon requires at least m+1 observations in a region of
probability Theta(epsilon²). Thus P(B_N<=epsilon)=Theta(epsilon^(2m+2)).
Both median inequalities suffice to sandwich this rate. At N4 the lower-tail
exponent is6, despite the upper-tail exponent2.

## 2. A non-removable heavy positive tail of this N4 predictive reference

The PB reference is W=Z/sqrt(B*V_H), with B independent of (Z,V_H),
V_H=lambda_min(H0)*sum_j chi-square5_j/lambda_j(H0)/20 positive finite a.s.
There exists a finite K with P(Z>1,V_H<=K)>0. On that event and
B<1/(K*t²), W>t. Hence, at N4,

    P(W>t) >= constant*t^(-12), for sufficiently large t.

This is a LOWER tail bound, not a claim that12 is its exact tail index.
Known-shape/known-kappa Student20 has tail order t^(-20), so the frozen
small-calibration predictive reference cannot behave like that lighter tail
arbitrarily far out. Tail-law integration removes a confidence-budget penalty,
not the information cost of finite calibration. This concerns this reference
construction, not a universal impossibility theorem for other methods.

## 3. Large-N relative accuracy and rho

Because f is positive and smooth at c, the standard sample-quantile limit is

    sqrt(N)*(B_N-1) -> Normal(0, 1/[4*c²*f(c)²])
                      = Normal(0, (1+1/sqrt(2))²).

One route to the limit: for a median cutoff c+x/sqrt(N), the empirical count
below it is binomial with success probability 1/2+f(c)*x/sqrt(N)+o(N^-1/2).
The binomial CLT gives the quantile limit. For even N, adjacent central order
statistics are separated by O_p(1/N) under the positive continuous density;
their average has the same limit. The coefficient of relative standard error
is therefore approximately1.7071/sqrt(N) at LARGE N, not an exact finite-N
coverage formula. At N4 its variance is actually infinite, as above.

This relative sampling law and limit do not contain rho, kappa, R or the
radial-law parameters. High rho alone does not force an increasing N for
fixed relative calibration accuracy under exactly matched M0. That does not
remove near-singular floating-point sensitivity, absolute scale growth, signal
geometry or actual target/calibration drift. C001's width/power/cost curve
estimates finite-N behavior separately; it is not evidence for a new rate.
