# Ancillary information recovery: targeted derivation, not another model family

Gap motivating this work: unconditional reference errors mix calibration banks
of different observed informativeness. Existing interfaces possess every A_i,
not merely their geometric mean. No new bank, truth or real-data access needed.
This is a refinement of the SAME finite joint-reference mainline. No full-family
performance run is authorized by this notebook alone; first validate the density
and reviewer challenge. G1 remains OPEN: ancillary conditioning is classical.

Let L_i=log A_i=eta+X_i, X_i iid logF42. Observed W_i=L_i-mean(L) have sum0 and
are ancillary for eta. Set Y=mean(L)-eta. The linear transformation from L to
(meanL,W_1,...,W_{N-1}) has constant JacobianN. Therefore

  f(Y=y | W=w) = product_i g(y+w_i) / integral product_i g(t+w_i) dt,
  g(x)=8exp(2x)/(1+2exp(x))^3.

The density does NOT depend on eta. Es=Ys-(1-log2),Et=Yt-(1-log2) remain
independent conditional on the two independent residual vectors. They need NOT
have conditional mean0; blindly re-centering them would change the reference
and be incorrect. This is a regular conditional distribution, not a Bayesian
prior assertion; the identical flat-location posterior identity is classical.

The log density has derivative sum_i[2-3sigmoid(y+w_i+log2)], second derivative
-3sum sigmoid*(1-sigmoid)<0 and limiting slopes2N and-N. It is integrable and
strictly log-concave. Tangents on opposite sides of its mode and a third interior
tangent give an integrable upper envelope, with a plateau-efficient middle piece.
Sampling from the corresponding three exponential pieces and accepting with
exp(logdensity-logenvelope) yields iid exact draws in
ideal arithmetic. This is CLASSICAL rejection sampling, not MCMC or bootstrap.
Numerical envelope failures or proposal caps must fail closed in an eventual
inference kernel. A finite cap is coupled to the unlimited valid algorithm:
replacing an unfinished whole inference by p1/e0 cannot enlarge any evidence.

If allw_i=0, 2expY has BetaPrime(2N,N) distribution. This furnishes an analytic
unit-check beyond self-comparison. For generalw, independent quadrature checks
moments; these checks are not mathematical proof of all floating-point draws.

Conditional on Ws,Wt and independent references, full bank-dependent DIR
rotation remains inherited: the target/SHAPE central pivot law is constant for
each realized bank and DIR. PILOT stays independent. Thus conditional Es/Et may
be used in full co-designed/selected reference, with s unknown and the same
drift coupling. This chain still requires review and actual implementation.
Potential gain from conditioning is NOT assumed; a conditional target-only
comparator would be mandatory to distinguish information recovery from borrowing.

Closest-work gap: location ancillary conditioning, group inference and adaptive
rejection are established. Need a useful model-specific result beyond their
direct application before G1 can pass. No new Power theorem claimed here.
