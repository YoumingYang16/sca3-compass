# A limited information bound for the calibration pivot experiment

Analytic supplement during frozen confirmation; no new algorithm, fitting or
confirmation change. Classical information inequality, not a new minimax
theory. This provides a sharply stated necessary sampling cost, NOT a lower
bound on full PC Power loss.

## Experiment and estimator class

Observe ONLY independent A_1,...,A_N with A_i=kappa X_i, X_i~F4,2,
kappa>0, eta=log(kappa). Estimators T(A) of eta must be unbiased for every
eta in R, square-integrable and satisfy the usual differentiation identity.
This is the reduced pivot experiment, not all information in raw calibration
matrices, target contrasts or patient samples. Loss is squared log-scale error.

For Y=logX, f_X(x)=8x/(1+2x)^3 and

    h(y)=8 exp(2y)/(1+2exp(y))^3.

The location score for eta in logA=eta+Y is

    s_eta(A)= -h'(logA-eta)/h(logA-eta)
            = 3 U - 2,  U=2(A/kappa)/(1+2(A/kappa)).

U has density2u on(0,1), equivalently Beta(2,1). Hence

    E U=2/3, E U²=1/2,
    E s=0, I_1=E s²=9/2-12*(2/3)+4=1/2,
    I_N=N/2.

The score is bounded(-2,1), the support is fixed and smooth, and the stated
regular estimator class allows E[(T-eta) sum_i s_i]=1. Cauchy–Schwarz yields

    Var_eta(T) >= 1/I_N = 2/N.

The inequality is strict at finite N in this global-unbiased class. Equality
at any eta0 would require T-eta0=(2/N)sum_i s_i(eta0) almost surely, hence
T in(eta0-4,eta0+2). All eta laws have strictly positive densities on the same
support, so that boundedness would hold at every eta; E_eta T=eta for all real
eta would then be impossible. This explains why the CRB must not be treated
as an achieved finite-sample improvement. It does not identify the optimal risk.

For the deployed logarithm of the geometric estimator,
T_g=mean(logA) - (1-log2), unbiasedness and

    Var(T_g) = (pi²/3-1)/N = 2.289868.../N

are exact. Its variance/bound ratio is(pi²/3-1)/2=1.144934...
This quantifies a remaining14.49% variance margin relative to a LOWER BOUND,
not an attainable finite-N estimator, and not14.49% Power. The corresponding
asymptotic efficiency ceiling comparison is2/(pi²/3-1)=.8734127396... under this
regular pivot-only log-location criterion. No MLE or alternative estimator
has been implemented or tuned in R3.

## What this establishes and what it does not

Finite N creates unavoidable log-scale sampling variance for this specified
unbiased pivot-only class; increasing N reduces that price at1/N. R2's median
moment pathology and spectral envelope are not dictated by this bound.

This is NOT a uniform/minimax bound over biased estimators, complete raw-matrix
experiments, directional-PC procedures, e-values, FDR, final detection Power,
or calibration-target drift. It cannot explain away the remaining strong-
reference Power gap as mathematically unavoidable. The usual information
inequality and F-distribution calculation are classical; the useful content
here is their exact mapping and the restriction that prevents an overclaim.
