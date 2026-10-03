# A0.7 proposed smoothed selective e-borrowing — not accepted

Evidence: D008 completed80sameDEV families. C1Power.37745, C3.51225, C7.26532;
conditional endpoints/mixes remain stronger in important cases. Hard conditioning
causes a specific target-side truncation penalty. This is not repaired by removing
the required protection or inventing more data. Current G1 and G3 remain OPEN.

## One mechanism revision, fixed before the next diagnostic

Use SAME conditional W/meta inverse-variance sourceweight a, centered contrastu,
and c=sqrt(Vs_hat+Vt_hat). Add conceptual independent logistic selection noiseR
with variance equal to the estimated variance of Es-Et, namely tau=c*sqrt3/pi.
Thus the noise split is fixed by a dimensionless unit-information convention,
not tuned on outcome/Power. Define B={Es+s-Et+R<=c}, g(u)=expit((c-u)/tau).

For each branch compute its OWN boundary selective reference by sampling entire
Es,Et pairs plus independent selection coin and retaining pairs in that branch.
Retained Es/Et stay paired and correlated. Fresh independent V shape tuples
give signed marginal/subset ranks, then original PC and grid/PILOT e-values.
The output is the deterministic mixture
E=g(u_observed)*E_bridge_selective+(1-g(u_observed))*E_target_selective.
Do not mix ordinary unconditional endpoint e-values using these data-dependent
weights: that would be a different, unproved algorithm.

## Finite-drift order and full e proof

Condition on W and independent meta, so all maps/constants are fixed. Write
X=Es+s,Y=Et. Borrow-conditional X density is proportional fs(x-s)*E[g(x-Y)].
As before, log-concave fs gives MLR inx as s increases. Conditional Y givenX,B
has density proportional ft(y)*g(x-y). Since logg is concave, increasingx gives
MLR increase iny. Coupling then makes aX+(1-a)Y increase withs.

For the target branch the Y density is proportional
ft(y)*barF_(Es+R)(y+c-s). Convolution of the two log-concave densities fs,fR
is log-concave, hence has increasing hazard. The earlier target MLR argument
applies. Thus both central positive-statistic laws given B are least favorable
at s0, exactly, without a finite nuisance grid or bootstrap. Entire source error
density and randomizer assumptions matter; target log-concavity is not needed
for this particular order lemma. The actual target model remains unchanged.

Each branch plus-one rank is superuniform conditional on W,meta and that branch,
after integrating its valid branch-specific MC reference. The unchanged positive
direction/shape rotation, PC/grid, independentPILOT proof yields
E[E_selected | W,meta,B]<=1 for every legal s. Generate both reference arrays
independently of the OBSERVED bank means conditional on W/meta (they may be
coupled to each other through common MC random numbers, but each marginal must
remain correct). Introduce an independent conceptual observed coin with chance
g(u_observed). Averaging that coin conditional on actual observations and the
reference arrays yields exactly the displayed weighted e-value. Total expectation
therefore gives E[E]<=1. This is a conditional-expectation/Rao-Blackwell step,
NOT ordinary deterministic-weight e-merging. eBH inherits the usual FDR guarantee.

PILOT mixing remains within each selective component before final borrowing
mixture; selection weights use no TEST or PILOT outcomes. Nonpositive test
statistics return p1. ACTUAL IMPLEMENTATION POLICY in D009 andA0.7.1: ANY component
failure returns whole-family0, with no weight redistribution. This is more
conservative than zeroing only the affected component. A0.7.1 additionally records
both branch statuses/costs and pair-cap receipt; D009 did not retain those details.
No uncounted pilot or oracle is introduced.
There is no combined p-value obtained by averaging the two component p-arrays.

## Known tools, unresolved contribution, limits

Randomized selective inference and improved leftover information are established
by Tian-Taylor, arXiv1507.06739v5 (2016), published AoS2018. Log-concave convolution,
MLR/IFR, Rao-Blackwellization and e-merging arguments are also classical. Main
must compare the precise unknown-drift selective ordering with closest results;
no claim of important originality follows automatically from this construction.

Randomization is integrated out rather than reported as a lucky choice. That
reduces variability of the randomized e-value, but does not imply higher eBH
Power. Wrong-source weight can remain positive at wideD, and both selective
reference laws can still incur cost. τ is one prespecified value, not a grid.
Cost: two computations of the SAME4095 shape tuples (shared random numbers,
not8190 independent shape samples), plus meta and selected error pairs, no new
observational data. Hypothesis to check: smaller target protection cost while
retaining useful tight borrowing, not universal dominance.

## Fixed development package

First proof/code challenge and unit tests. Then D009 only: same80families/all10
cases,8each,4095MCouterperbranch/4095meta,4workers/1800s safety timeout. All prior
required baselines, conditional endpoints and fixedmixes, D005/D007/D008 retained.
Stop after80, no significance expansion, no new confirmation. Raw paired results,
both branch p/e arrays, mixing probability, acceptance rates and failures saved.
If the mechanism fails, diagnose the identified cost; do not scan randomizer scales.
