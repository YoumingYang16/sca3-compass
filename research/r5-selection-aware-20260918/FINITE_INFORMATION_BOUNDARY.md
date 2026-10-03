# B1: finite target-calibration information boundary (under challenge)

2026-09-18. This is a model-subclass bound, not a statement that the existing
C1fixed-effect simulation is impossible, not G3success, and not a new algorithm.
Classical Neyman-Pearson reasoning is acknowledged; priority/importance OPEN.

## Actual full-input submodel, not a lossy scalar toy

Use radial multiplier1 (Gaussian case already inside M(D)). For each4x6block,
write pipeline mean x inR4 and five orthonormal contrasts Y inR4x5. Let
 x=mu+sqrt(kappa) Sigma^(1/2) z, Y=Sigma^(1/2) E,
where z and E are independent standard normals. This corresponds to a compound
symmetric pipeline covariance scaled so contrasts retain variance1 and common
mean variance kappa. Unknown positive Sigma is the same across relevant banks.
To strengthen an upper bound we may even give the procedure the COMMON CONTRAST
covariance Sigma, the same under both models; not the original R parameter if
its parametrization changes withrho, nor the unknown target kappa. This is not
deployed extra information. Source blocks have fixed kappa_s and zero mean.

Global-null P0: every target mu_g=0 and target kappa=kappa0<=D*kappa_s.
Mixture alternative P1: target kappa=kappa1=r*kappa0 with0<r<1; independently
sample fixed effects mu_g~N4(0,(kappa0-kappa1)Sigma) once, independently across
genes and from all observations/calibration. Conditional on these effects the
usual fixed-mean model applies. Integrating the prior, every target x_g underP1
has distributionN4(0,kappa0Sigma), IDENTICAL toP0; all contrasts are identical.
The source bank is identical. Both sides obey the SAME externalD bound. This is
not adding random effects as a new required inference assumption: it averages
over legitimate fixed-mean alternatives to derive an upper bound.

Only the Nt target calibration means distinguish P0/P1. Whiten and divide by
sqrt(kappa0): their stacked4Nt-vector has N_d(0,I) versus N_d(0,rI), d=4Nt.
All remaining full inputs, raw target samples, source, contrasts, DIR/PILOT and
reference randomization are ancillary for this binary comparison. Thus moreG
or moreNs does not supply information for this particular alternative mixture.

## Exact Neyman-Pearson bound

Under P0 all2GdirectionalPC nulls are true, so an algorithm withFDR<=q satisfies
P0(any rejection)<=q. At the mixture alternative, the likelihood ratio is
 r^(-d/2) exp[-(1/r-1)||x||^2/2], decreasing in the targetcal norm.
The most powerful sizeq test rejects when ||x||^2<=chi2_d_quantile(q). Therefore

 P1(any rejection) <= F_chi2_d(F_chi2_d^(-1)(q)/r).          (B1)

Any directionalPCPower, defined as correct claims/trueclaims, lies between0and
1(any rejection), including the convention for zero trueclaims. Under continuous
four-dimensional effects and2-of4 directionalPC there is at least one true
direction pergene almost surely, so the denominator is not an issue. Hence its
prior average obeys the same bound. This is NOT a bound for each particular
fixed mean realization and not a bound for the stored C1alternative.
The Neyman-Pearson variance test attains the binary-testing upper bound only;
it is NOT an FDR procedure valid over the entire scientific composite null.
No attainable-full-procedure claim follows from this upper bound.

## Interpretation and limits

A wide legal bound can permit a larger-noise global null that observationally
mimics heterogeneous signals at smaller target noise. Finite target calibration
then limits detection even with arbitrarily large source and target gene counts.
When D excludes that larger kappa0 this particular construction no longer
applies; that is an identifiable reason tight credible borrowing can help.
This is not permission to shrink D, ignore Dunderestimation, assume a sparse
prior, or substitute simulated true kappa into a selector.
It does not prove no useful same-information adaptive method exists. It separates
a concrete information limit from implementation failure and arbitrary Power
aspirations. A boundary-only result cannot pass the user's four-gate success rule.

## M005 pre-output rule

Deterministic evaluation only: Nt in[4,12,32], r in[.95,.8,.5], q=.05. ComputeB1
using SciPy chi-square cdf/ppf, verify endpoints/r-monotonicity. No random data,
no case selection, no FDR simulation or confirmation. Also report how adding
target calibration would change this bound, not claiming data actually exist.
Keep allnine values; no extension based on a favorable number.
