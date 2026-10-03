# A finite-threshold target-reference cost: exact identity, not a novelty claim

2026-09-18. Current A0.7.1; conditional on observed ancillary W and independent
meta draws. All errors below include the fixed meta centering. Let Es,Y=Et be
independent conditional errors, R independent logistic(tau),
B={Es-Y+R<=c} at the least-favorable drift boundary s=0. Let

    h(y)=P(Es+R>y+c), piT=E[h(Y)],
    k_x(y)=P(V>x exp(y/2)), x>0.

V is the ACTUAL learned-shape pivot, independent of Es,Y after the inherited
rotation argument. No assertion that V is t20; no moment condition on V.
The target-branch reference survival and unconditional target survival are

    S_T(x)=E[h(Y)k_x(Y)]/piT,  S_0(x)=E[k_x(Y)].

Consequently

    S_T(x)-S_0(x)=Cov(h(Y),k_x(Y))/piT >=0,
    1 <= S_T(x)/S_0(x) <= 1/piT  whenever S_0(x)>0.

Proof: h and k_x are nonincreasing. For independent Y', twice their covariance
equals E[(h(Y)-h(Y'))(k_x(Y)-k_x(Y'))]>=0. Upper bound uses h<=1.
In this model all conditional error densities are positive, logistic survival
is strictly decreasing, and V is a positive-scale mixture of normals with
strictly positive density. Thus covariance is strictly positive for finite x>0.
This is an elementary association/selection-tilt consequence, NOT certified G1.

## Finite-threshold near-saturation bound, with no learned-shape moment assumption

For any b in (0,min(1/tau,2Ns)) the source conditional moment
M_s(-b)=E exp(-b Es) exists: its left density tail is proportional exp(2Ns Es).
For all y, logistic g=1-h integrand satisfies

    1-h(y) <= min(1, exp(b(c+y))*M_s(-b)).

This follows pointwise from expit(z/tau)<=exp(b z) for 0<b<=1/tau:
for z>=0 the right side>=1; for z<0 use expit(z/tau)<=exp(z/tau)<=exp(bz).
Therefore for any ell>0,

    0 <= 1-piT*S_T(x)/S_0(x)
       <= min(1, exp(b(c-ell))*M_s(-b)
                    + P(V>x exp(-ell/2))/S_0(x)).

Proof: split E[(1-h(Y))k_x(Y)] at Y=-ell. The first part is bounded
by exp(b(c-ell))*M_s(-b)*S_0(x). In the second, k_x(Y)<=P(V>x exp(-ell/2));
drop P(Y>-ell)<=1. Divide by S_0(x). This is generally loose but honest,
finite-threshold and requires no Breiman lemma/high learned-H moments. It may
be vacuous near the actual rejection tail; nonvacuity must not be assumed.

This proves a cost of THIS reference conditioning. It does not prove a lower
bound for all safe borrowing procedures, final integrated e-values, number of
discoveries, or Power. It does not prove that a particular sample's p-value
is always inflated with independently simulated references.

## Deterministic saved-reference diagnostic D012 (fixed before outputs)

Use D011 ALL10cases at earliest repetition8, independent of results. No new
random observations, errors, shapes or seeds. Source and target density from
saved W; meta centering/c/tau unchanged. Evaluate h by Gauss-Legendre quadrature,
target CDFs by a deterministic fine grid, then integrate over the SAME saved
4095 actual V tuples. At thresholds x=2,4,8,16 report S0,ST,ratio,piT, and
the envelope 1/piT. Check 512source nodes/4097target grid against1024/8193.
Integration tails bounded by log-concave endpoint tangents; quadrature differences
and numerical tail bounds are diagnostics, not interval arithmetic certification.
This is Rao-Blackwellized analysis of existing references, NOT a new deployed
algorithm, new FDR experiment or independent confirmation. Main goal: determine
whether the cost actually appears in the deployed reference tail without the
unproved t20 substitution. Fixed10inputs, do not extend based on favorable ratios.

Closest work: Tian-Taylor1507.06739v5 selected density equation13; log-concavity/
IFR and covariance association are established. This identity and bound clarify
a mechanism, but independent important originality remains OPEN.
