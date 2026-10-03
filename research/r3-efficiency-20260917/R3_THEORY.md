# R3 draft: geometric scale and independent orientation reference

Written after the240-family loss ledger, BEFORE any candidate implementation
or candidate Power observation. Exact-real/model argument, numerical and
novelty status separate. Reuse R2 M0, signed PC r2of4, independent G blocks,
N centered calibration matrices, J6, arbitrary common-block radial laws,
unknown SPD R and kappa. No new radial parametric assumption.

## 1. Finite-law equivariant scale, not a confidence set

Use the SAME calibration matrix pivot A_l from R2, A_l/kappa iid F4,2.
Let c_g=exp(E log F4,2)=exp(1-log2)=e/2 and

    kappa_g=exp(mean_l log A_l)/c_g.

Then B_g=kappa_g/kappa has the EXACT law exp(mean log X_l)/c_g for iid
X_l~F4,2. It is positive a.s. and independent of all target blocks; its
finite law can be sampled directly, with no delta, CLT or fitted bootstrap.
Any positive equivariant summary could replace the median in R2's dominance
argument if the same summary's law is used in all reference tuples.

The log-F density comes from X=(U/4)/(V/2), U~chi-square4,V~chi-square2
independent. Differentiating Gamma moments gives
E log X=1-log2; Var(log X)=psi1(2)+psi1(1)=pi²/3-1.
Thus Var(log B_g)=(pi²/3-1)/N EXACTLY, and the log-CLT coefficient is
sqrt(pi²/3-1)=1.5132, versus the median's large-N relative1.7071. This is
an efficiency motivation, NOT a theorem of larger Power or finite-N
stochastic dominance. Mean-square log error is not the final test loss.
Direction/PILOT learning retain the old median heuristic so its signal
extractor is not changed by this inference-scale substitution.

## 2. Eliminate worst-direction inflation by independence, not a sharper guess

Four equal deterministic target folds have roles TEST=f, PILOT=f+1,
DIR=f+2, SHAPE=f+3 modulo4. Directions are learned on DIR, using its own
shape and old median calibration heuristic. Inference H is fitted ONLY on
SHAPE contrasts with the existing fixed2-step affine estimator. The PILOT
is exactly the independent R2 heuristic; it sees neither external calibration,
DIR, SHAPE nor TEST. No target is discarded from the cyclic testing family.

Condition on DIR and external calibration, so each nonnegative intersection
direction a is fixed. H is independent of both. In whitened coordinates,
R^-1/2 H R^-1/2 is a positive scalar times H_*, where H_* is the same fit
on independent STANDARD Gaussian contrast blocks. H_* has an orthogonally
invariant distribution. Its scale-free quantities do not depend on R or
the SHAPE-block radial laws. The scalar may depend on H_*; it cancels in
all products below, so no assertion about an unnormalized nuisance-free H.

Let b=R^1/2 a/||R^1/2 a||. Conditional on DIR/calibration b is deterministic.
The joint law of b'H_*b and H_* eigenvalues equals that of (H_*)_11 and
the eigenvalues, for EVERY unit b. If b depends on the calibration error B,
this conditional invariance still gives the same joint law for each B;
it is not invalidated by a data-dependent b independent of H_*.

The held target, as in R2, has a normal angular numerator independent of
its Gaussian contrast block. After cancellation of its common radial scale,
the centered statistic has the reference law

    W=Z / sqrt[B_g * (H_*)_11 * sum_j U_j/lambda_j(H_*) /20],

with Z standard normal, U_j iid chi-square5, an independent Gaussian SHAPE
fit H_* based on G/4 blocks, and the independent N-pivot B_g. The SAME
H_* is used for its11entry and its spectrum; replacing it by an independent
Rayleigh draw would be incorrect. For a nonpositive signed projected mean,
the observed statistic is pointwise no larger than its centered version;
restricting to positive tails therefore gives stochastic domination.
This uses the actual fixed-direction quadratic form, NOT lambda_min(H_*).

The cost is G/4 rather than G/2 shape blocks and smaller DIR training.
At equal H/B/Z/U tuples, H_11>=lambda_min would reduce positive reference
tails; however the new training size and directions differ, so NO complete
algorithm pointwise dominance or no-regression theorem is asserted.

## 3. Complete learning and family guarantee (reuse, not a new e-BH theorem)

Conditional on the entire corresponding PILOT fold, DIR/SHAPE/TEST/calibration
law remains unchanged. For each valid null intersection the central W law
above is nuisance-free. Fresh iid ENTIRE reference tuples independent of real
data give the conservative plus-one rank p=(1+#{W_b>=T})/(M+1), or1 forT<=0.
The rank argument applies to a coupled central W0, not falsely to arbitrary
shifted alternatives. Ordinary Bonferroni over the3coordinates and max over
four triples retains the partial-conjunction null, including singleton and
mixed-sign cases. Nonnegative direction constraints remain unchanged.

Both PC components are superuniform conditional on PILOT. R2's same exact
support calibration yields conditional mean<=1. PILOT-only gamma makes the
convex mixture an e-value. Integrating each claim over ITS OWN PILOT field,
then the pointwise e-BH inequality gives FDR<=q*m0/(2G)<=q=.05. Shared
calibration/reference and cyclic folds are allowed at the final step. Never
condition on the union of all fold roles, and never choose auxiliary seeds.

No extra epsilon or nuisance coverage budget is paid. Error-control grade A
under the same M0/ideal arithmetic IF implementation obeys these independence
rules. Frozen-source review, tests and utility evidence remain required.

## 4. Limited mismatch statement, no new transfer program

For externally justified kappa_target/kappa_cal<=Delta, multiplying the
inference kappa_g byDelta only increases the squared denominator enough
for the same positive-tail domination. DIR/PILOT learning and reference law
remain unchanged. Unbounded drift, arbitrary target covariance departures
and real clinical input feasibility remain outside scope exactly as in R2.

## Falsifiers and unresolved aspects

- Dependence of DIR or selected directions on the actual inference H invalidates
  the exact orientation step: must reject such an implementation.
- Failure of finite-step affine/orthogonal invariance invalidates the reference.
- A shared reference bank conditioned as fixed is not the claimed guarantee.
- Log-summary improvement in an asymptotic metric does not prove Power gain.
- Two finite-sample valid components need not give a superior combination.
- Novelty of this task-specific independent-orientation construction is NOT
  established. No new projection family or global efficiency bound is claimed.
