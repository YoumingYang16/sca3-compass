# Exact moments of geometric calibration error

Analytic supplement, no algorithm/experiment change. Classical Gamma/F
identities, not a claimed new general moment theorem. This explains a
specific finite-N difference beyond asymptotic standard errors.

For X~F4,2, X=U/(2V), U~Gamma(2,1),V~Gamma(1,1) independent.
Direct integration of their densities gives

    E X^t = 2^(-t) Gamma(2+t) Gamma(1-t),  -2<t<1.

Outside these strict endpoints the relevant positive integrand diverges
at zero or infinity. With N independent pivots and B_g=exp(mean log X)/(e/2),

    E B_g^p = exp(-p) [Gamma(2+p/N) Gamma(1-p/N)]^N,
               -2N<p<N.

This is an exact finite-N identity. It proves, in particular, that at N4
the geometric scale error has a finite second moment, unlike R2's median
scale error. It does NOT imply E B_g=1: the estimator is centered on the
log scale only. The logarithmic variance identity in R3_THEORY follows by
twice differentiating this expression at0.

For s>=0, its negative moments E B_g^(-s) are finite exactly when s<2N. For N4,
this is s<8, while the previous even-sample median's
lower-tail order gave finiteness only for s<6. Calibration alone therefore
imposes a less severe negative-moment limit on a predictive denominator.
For a KNOWN shape, an independent Student20 divided by sqrt(B_g) has
finite absolute r-th moment, for r>=0, exactly when r<min(20,4N), by independence and the
Student/Gamma moment integral. AtN4 that is r<16, compared with r<12 for
the median version. This last statement is a known-shape mechanism analysis,
not a proof of the complete unknown-shape reference's exact tail index.

The error law is scale-free: none of these relative quantities depends on
rho, R or radial distribution within M0. More favorable moments do not
establish stochastic dominance, uniform Power gain or an information bound
for all finite-calibration methods. Those remain separate empirical or
theoretical questions. No Gamma approximation enters the deployed method;
its reference still samples the exact finite pivot-summary law.
