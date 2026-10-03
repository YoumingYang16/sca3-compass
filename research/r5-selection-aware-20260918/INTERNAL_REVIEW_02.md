# Internal adversarial review, 2026-09-18 17:05 HKT

Reviewer: sole existing Maxwell agent; main independently integrated responses.
EXTERNAL_REVIEW=NOT_CONDUCTED. This is not external endorsement or acceptance.

1. Ancillary W plus independent meta draws can fix conditional weights/centers
without pretending that observed bank means are fixed. Accurate moment estimation
is not required for the ideal conditional rank proof. Inverse variance remains
classical and not an optimum for drift-biased loss or Power.
2. The proposed central scalar KS bound is a valid log-density Lipschitz bound,
not a noncentral Power or family discovery guarantee. Finite meta draws require
a separate stochastic-order argument, not the ideal population-moment rate.
3. Normalizer cancellation works for positive finite scales. Rank agreement
converges in probability; an a.s. eventual equality claim was not justified.
4. Fixed-size inner samples can all be negative. The old q>0 failure means D005
code has a nonvanishing failure event, invalidating unconditional implementation
convergence to an endpoint that has no inner samples. Main changed working code
to independent q=1 fallback and recorded it; old D005 untouched. Sampler caps and
float failures still require an explicit additional disagreement-probability term.
5. Concrete sampler bug: searchsorted(cumsum(prob),u) could return region3 when
the sum rounds below1, leaving an uninitialized proposed value. Working A0.4.1
uses explicit first/second thresholds and a remainder region. Regression added.
6. Positive log-envelope excess up to1e-10 was silently tolerated. A0.4.1 adds
a fixed vertical floating guard (proposal unchanged) and fails on ANY observed
positive excess. This is a guard, not interval-certified exact arithmetic.
7. Relevant-subset/observed-information adaptation and inverse-variance weights
are established literature. P-CAL's specific arithmetic information boundary,
its correctness and its practical importance still need challenge. G1 OPEN.

Actual receipt: checks/numerical-repair-a041.xml, 46 tests PASS. Tests do not
prove FDR or priority. D004/D005 numerical-certification claims are restricted:
known rare numerical hazards were present; no old result is silently upgraded.
No confirmation has been run, and no process was restarted. Next work package
is deterministic attribution from existing saved references, not new simulation.

Follow-up review: no counterexample to P-CAL(i)-(iii); uniform-integrability
proofs can be made explicit, and C_r<=9/2 is a convenient nonsharp bound.
Main checked the half-integrals and exponential second-moment bounds. Reviewer
provided two real limitations: bounded conditional variance does not bound the
original geometric estimator's conditional MSE; fixed-slack adaptation is NOT
uniform over legal slack sequences growing with the source plateau. The specific
plateau neighborhood is exponentially rare under the model. Main is preserving
these limitations, not upgrading a mathematically interesting boundary into G1.

The q=1 numerical fallback preserves conditional validity but breaks exact
centering/scale equivariance on that event; any claim that centering leaves the
algorithm unchanged must exclude it. No fallback was observed in old D005,
which instead would have failed on a nonpositive inner quantile.
