# A0.5 proposed minimal switch closure — development, novelty OPEN

Concrete gap (D006, exact replay of D005's80 saved families): C1 selected branch
without required selection protection gives .44118 Power versus .30147 protected.
The former is INVALID, not an achievable comparator. C3 .61275 versus .57353.
Changing variance weights alone was small. A fixed .99 quantile aligns one point
of two unequal tails; this is not a guarantee at the grid's extreme rejection
levels. No quantile scan or new observational information is proposed.

## Defined procedure and minimal proposition

Condition on Ws,Wt, independent meta and two independent inner reference streams.
Let a in(0,1], c=sqrt(Vs_hat+Vt_hat), as a prespecified conditional-noise-sized
gate, NOT a Power optimizer. The source/target errors include the same meta
centering as observations. Independently fitted continuous nondecreasing maps
phi_b,phi_t on positive statistics are fixed given this conditioning. They are
statistics, not assumed exact p-value calibrators. Nonpositive inputs map to0.

Retain the CALIBRATION-ONLY selector u=s+Es-Et<=c. Set
  phi_b_plus(w)=max(phi_b(w),phi_t(w*exp(a*c/2))).
The observed score uses phi_b_plus(Wb) when borrowing and phi_t(Wt) otherwise.
For an entire reference tuple, Wb(s)=Vexp(-(aEs+(1-a)Et+a*s)/2),
Wt=Vexp(-Et/2), and the tuple switches at s=c-Es+Et. Reapply the selector in
every reference tuple. Use only s=0 after the monotonicity argument below.

PROPOSITION (ideal arithmetic): for every complete tuple the selected score is
nonincreasing for all s>=0. Moreover phi_b_plus is the pointwise smallest upward
modification of phi_b, with phi_t and c fixed, that guarantees this property
at every possible positive switch value.

Proof: before a switch, Wb decreases and both component maps are nondecreasing.
At the switch Wb=Wt*exp(-a*c/2); hence phi_b_plus(Wb)>=phi_t(Wt). After switching
the target score is constant. Negative numerators give0 throughout. If a candidate
upward bridge map psi provides the same pathwise guarantee for every positive w,
the switch requires psi(w)>=phi_t(w*exp(a*c/2)); upward modification also requires
psi(w)>=phi_b(w). Their maximum is therefore necessary and sufficient. This is
minimal among SCORE LIFTS, not optimal in Power, all tests, or arbitrary redesigns.

For the true legal slack s>=0, monotone coupling makes the observed score no
larger than an identically distributed boundary score. Independent boundary
MC ranks with ties conservative are superuniform. The inherited composite signed
null, positive-direction PC, grid e-normalization and independent PILOT/eBH chain
then applies exactly as for A0.4, conditional on W+meta+inner references.
The numerator is never used to SELECT borrowing: the same bank-dependent u is
used for every claim. No TEST-dependent tail-map fitting or free pilot is added.

CENTERING DEFINITION: here u is the CENTERED contrast
logD+logKs-logKt-ms+mt. Keeping c=sqrt(Vs_hat+Vt_hat) does change the original
uncentered gate; this is part of the new algorithm, not merely numerical scaling.
To retain an uncentered gate its threshold would have to shift by -ms+mt.
Within the fixed statistic the active max term can depend on the tested statistic;
the borrowing decision itself does not. These must not be conflated.

## Relation to established results and limits

Boundary calibration and MC validity are inherited classical facts. Upward
envelopes are established constructions. Whether this specific minimal repair
of a calibration-only switch plus tail normalization is important independent
new content remains OPEN; reviewer challenge and closest-work check are required.
Do not equate an exact finite drift reduction with a novel generic maxT theorem.

This removes a particular nuisance-envelope computation and handles different
branch tail maps without a raw fixed-quantile max. It does NOT prove no selection
cost, superior Power, or safety when delta>D. Forward maps only are used; the
retired inverse interpolation and empirical nuisance optimizer are not used.
Float guards are not universal interval certificates. Caps fail whole-family e0.

Independent internal review supports the ideal switch lemma and PC support,
not novelty or utility. If both maps are logw and c>0, the selected score reduces
exactly to max(logWt,logWb+ac/2), so closure need not improve ordinary maxT at all.
Different tail slopes can avoid this identity, but finite inner extrapolation
does not guarantee those slopes or useful behavior at1/4096. D006 identifies
protection cost, NOT proof that the .99 normalizer caused the whole cost.

## Bounded next work package

First deterministic pathwise, direct-rank, switch and negative-input tests. Then
only if these pass, D007 SAME80 existing R4 family inputs, all10 cases and all
original/conditional comparators, plus the immediately preceding D005 paired
version.4095 inner each,4095 outer,4095 meta perbank,4 workers, at most30min per
batch. No new confirmation, no extra data, no optional sample extension.
Save every failure and exact code/protocol before outputs. Goal: see if closure
reduces the established C1 protection bottleneck without restoring wideD damage;
small8percase results remain DEVELOPMENT, not an acceptance or noninferiority test.
