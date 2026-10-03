# A0.8 mechanism correction: complete soft rule instead of conditional branches

18:23HKT2026-09-18. G1/G3 OPEN. One existing mechanism refined, no algorithmgrid.
D011 shows meaningful branch-selection cost; D012_2 supports its positive-tail
mechanism. Retain A0.7.1 as reference, not as secretly modified frozen output.

Condition on W, independent meta and independent inner references. Keep a,c,tau
exactly as A0.7. Let positive independent inner .99 quantiles be qt,qb; a
nonpositive quantile uses fixed1 and is logged (no conditioning on inner success).
For target-normalized positive statistic t and observable centered log gap u,

    zT=t/qt, zB=t exp(-a*u/2)/qb, g(u)=expit((c-u)/tau),
    Q=zT+g(u)*(zB-zT),
    Qplus=zT+g(u)*(zB-zT)_+ = max(zT,Q).

At fixed target error and central V, increasing legal slack s increases u only.
Qplus is nonincreasing: until zB=zT both g and zB-zT are nonnegative and
nonincreasing; thereafter Qplus=zT. Moreover Qplus(u)=sup_{v>=u}Q(v): before
crossing Q decreases to zT; beyond crossing Q<=zT and tends to zT as v→infty.
Thus it is the least nonincreasing majorant of THIS continuous soft raw score.
This is a score-order optimality statement, NOT an optimal test/Power claim.

Unlike A0.7 no separate B-conditional reference is used. Draw whole independent
Es,Et,V tuples at s0 and apply Qplus to every tuple; plus-one upper ranks are
valid for all s>=0. The target score has a floor, but the final p-values can
still be more conservative than target-only: boundary calibration pays for the
floor/borrow advantage. No pointwise test dominance is asserted.

Nonpositive signed statistics return p1; Qplus is increasing in positive t,
so directional-null central domination holds. The inherited positiveDIR, same-H
reference eigenvalues, disjointPILOT, PC maximum, exactgrid p-to-e and eBH proof
continues with the new single-test p. DIR uses target bank and remains covered
by rotation; conditional error/shape independence must be maintained as before.
All numerical failure returns family0; no selected reference is silently reused.

Relationship: D003 used hard maximum of normalized endpoint scores; A0.8 lies
between the normalized target and that maximum pointwise. D007 used a hard
switch and raised bridge scores to the target score at its switch boundary.
A0.8 is a continuous specialization of the SAME monotone-closure/MC family;
its elementary supremum identity does NOT establish important novelty.

Next diagnostic D013 is deterministic saved-reference replay on D007ALL80inputs,
reusing existing independent inner/outer tuples and stored fold learning. No
new random draws or sample size. Required old baselines and A0.7 compared on
same source records. Expected minutes oneCPU; stop after80regardlessresults.
Before accepting output, reproduce original D007p from the same stored tuples
and directions exactly. New score must pass monotonicity and rank unit tests.
This isolates the full-rule vs branch-conditioning mechanism, not confirmation.
No larger experiments while important G1 content remains unestablished.
