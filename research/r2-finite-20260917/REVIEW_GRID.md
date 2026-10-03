# Independent proof review: discrete-support calibration

Date: 2026-09-17. Scope: only the proposed normalization of the existing
PILOT-selected calibrators, holding p-values, directions, PILOT receipts,
gamma and reference draws fixed. No agents, tests, fits, experiments or
rescoring were run by this reviewer. Only this review file was written.
No D003 outcomes were inspected.

## Verdict

**The support argument and normalization proof are correct.** The ordinary
component has the proposed coarser support despite dependent study tests,
Bonferroni minima and the partial-conjunction maximum. Superuniformity plus
support suffices; the PC p-value need not have an exact uniform rank law.

The inspected implementation constructs the correct endpoint masses and
upward-rounds the normalizer of its represented calibrator values. Its main
remaining contract gap is missing support validation: it currently accepts
off-grid p-values although its guarantee requires this specific grid. Final
floating-point division remains explicitly uncertified, as does the upstream
bridge's numerical inference. These are not objections to the ideal proof.

Reviewed SHA256 snapshots:

- `grid_calibration.py`:
  `7fdba737ca47b9f5775753eb83a3e5d7a9c1a808620a541bf14d01a6e4cfd8f3`.
- `grid_diagnostic.py`:
  `88e2c019fb9e5c16ddfe7cfd750d6cb51f5ab8f300496ac3230061702b96ffa5`.
- `test_grid_calibration.py`:
  `a76204dca735772034d23e9f2ece26280b46850fbbd96a4b0c601b58c2ae9bb8`.

The saved-p driver and fixtures were read, not executed. This review does not
re-audit D002 or certify the completeness/results of any rescore.

## 1. Support survives both PC constructions

Set N=M+1, where M is the fixed reference count. Each primitive bridge p-value
is either 1 or (1+integer count)/N, so it belongs to

    S_1 = {1/N, 2/N, ..., N/N}.

The projection PC value is a maximum of four such values. A finite maximum
selects an existing value, so its support is contained in S_1. The p=1 branch
for nonpositive statistics introduces no additional point.

For an ordinary three-study intersection,

    p_I = min(1, 3*min_(s in I) p_s).

The inner minimum is again k/N. Thus p_I lies in

    S_3 = {3k/N : k>=1, 3k<N} union {1}.

Taking the maximum over all four intersections preserves that same support.
No independence of studies, intersections or rank references is needed for
these algebraic support statements. The bridge's previous null-subset and
Bonferroni arguments supply conditional superuniformity; support alone does
not supply validity.

For N=4096, S_3 has 1366 points: 3/4096 through 4095/4096 in steps of 3/4096,
then 1. The last interval has length **1/4096**, not 3/4096. Including p=1 is
necessary in the general proof even though the current focused/BY calibrator
is zero there. These are support supersets; some values might not be attainable
by the particular PC construction. No attainability assumption is needed.

## 2. Conditional expectation bound and zero normalizer

Fix the entire permitted PILOT sigma-field P. Write the ordered support as

    0=t_0 < t_1 < ... < t_K=1,
    h_k=h(t_k),   F_k=Pr(p<=t_k | P).

The existing calibrator h is P-measurable, nonnegative and nonincreasing.
Conditional superuniformity gives F_k<=t_k, with F_K=1. Summation by parts gives

    E[h(p) | P]
      = h_K + sum_(k=1)^(K-1) (h_k-h_(k+1))*F_k
      <= h_K + sum_(k=1)^(K-1) (h_k-h_(k+1))*t_k
      = sum_(k=1)^K (t_k-t_(k-1))*h_k = C.

All coefficients multiplied by F_k are nonnegative; this is exactly where
decreasing h is used. The distribution with masses t_k-t_(k-1) at t_k is
itself superuniform and attains C. Thus C is the sharp bound over **all**
superuniform distributions supported on this grid, not necessarily over the
smaller class realized by these PC statistics.

For C>0, E[h(p)/C | P]<=1. If C=0, every support mass is strictly positive and
h is nonnegative, so h(t_k)=0 for every k. Consequently every allowed p has
h(p)=0; returning zero is correct. Do not redistribute its component weight
based on HELD evidence. The proposed zero convention keeps the old gamma.

Each component has its own support and normalizer. Both normalizers depend
only on that component's PILOT receipt and the fixed design, not on observed
HELD frequencies, the realized reference values, or observed power/FDP.
Therefore the unchanged PILOT-measurable convex gamma mixture still has null
expectation at most one. e-BH at .05 remains justified on the full 2G family,
with arbitrary dependence among the final evidence. The fourfold separation
from REVIEW_BRIDGE.md must be retained; normalization does not repair an
invalid conditioning boundary or an invalid upstream p-value.

## 3. Domination of the previous calibration

On each interval (t_(k-1),t_k), monotonicity gives h(u)>=h(t_k). Hence

    C <= integral_0^1 h(u) du <= 1.

For C>0 this implies h(p)/C>=h(p) at every allowed p; for C=0 both are zero.
Componentwise normalization followed by the same gamma therefore increases
or preserves every ideal evidence value. At the same level and family size,
e-BH's rejection set can only increase. This is a deterministic claim for
fixed inputs/receipts, not a population superiority claim. FDP need not
decrease, and more rejections need not all be true discoveries.

If the focused threshold lies below the first grid point, its entire focus
term is zero on the support. Normalization then removes its wasted mass from
the combined calibrator, leaving the normalized surviving BY-shaped term.
It does not move the focus threshold or create positive evidence where the
whole original h is zero. Do not replace C by an empirical average of h over
the observed targets: that is a different, unproved operation.

For represented floating-point h, the certified discrete sum is the relevant
bound. Do not clip a computed upper normalizer down to 1 to force numerical
domination; a tiny discrepancy in the continuous-calibrator implementation
does not justify underestimating its discrete expectation. Literal bitwise
domination is separate from the exact-arithmetic statement above.

## 4. Exact combinatorial accounting

Let s=1 for projection and s=3 for ordinary. Construct integers

    r_k = min(s*k,N),  k=1,...,ceil(N/s),
    r_0=0,  w_k=r_k-r_(k-1),
    t_k=r_k/N,  pi_k=w_k/N.

Require unique points and sum_k w_k=N. Using these integer interval lengths
avoids repeated floating addition, ceil/floor endpoint ambiguity and the
incorrect last ordinary mass. For 4095 references all p-grid points are
dyadic and exactly representable in float64; multiplication by three and
capping at one preserve exact membership at these magnitudes.

For an exact-rational formulation, define the extremal grid CDF

    G_s(x)=0 for x<0,
           (s/N)*floor(N*x/s) for 0<=x<1,
           1 for x>=1.

For a focused term I(p<=tau)/tau, its budget is G_s(tau)/tau. Equality at tau
must be included. For the existing capped BY step, let A be its rank cap,
H_A=sum_(j=1)^A 1/j, q its shaping level, and m the hypothesis count. Its
j-th height m/(q*j) occupies

    ((j-1)*q/(m*H_A), j*q/(m*H_A)].

Its discrete budget is therefore

    C_BY = sum_(j=1)^A [m/(q*j)] *
             [G_s(j*q/(m*H_A))-G_s((j-1)*q/(m*H_A))].

For focus fraction f, C=(1-f)*C_BY+f*G_s(tau)/tau. Counts and comparisons can
be evaluated using rational cross-products and integer floor/ceil operations.
Do not use a tolerance that includes a grid point just above tau or moves a
point across a BY step boundary.

An equally valid way to preserve the **literal existing numerical h** is to
tabulate h at every exact support point, group identical represented heights,
and sum those heights against the integer masses. This is the route taken
by the current code. It avoids redefining the archived threshold/harmonic
arithmetic. Crucially, the target evaluation and normalizer must use the
same table/function and endpoint conventions; do not combine rationalized
thresholds in one path with rounded ceil/comparisons in the other.

## 5. Current code: correct construction, narrow safeguards outstanding

`grid_calibration.py:16-26` correctly selects s=3 for component 0 and s=1 for
component 1, adds endpoint N once, and uses integer differences for mass.
It evaluates the existing focused calibrator on this support, groups identical
represented heights and uses `Fraction.from_float` to sum their exact binary
values with these masses. For N=4096, `bincount`'s floating accumulation of
integer weights is exact at this scale; it is not a general arbitrary-size
integer accumulator. Lines 29-31 round C upward when necessary. The numerator
at lines 32-33 uses the same calibrator and PILOT receipt. These choices
implement the proposed bound for supported inputs, subject to final rounding.

Concrete safeguards before treating this helper as a checked inference API:

1. **Reject off-grid input.** Lines 13-15 validate neither p's support nor
   integer/non-boolean reference count. For the fixed N=4096 path, require
   finite p in (0,1], N*p integral, and for ordinary values below 1 require
   that integer to be divisible by three. Validate component, integer N and
   the fixed count/design before constructing the grid. Do not round p down
   to a nearby supported value. `test_grid_calibration.py:19` currently
   includes .005, which is off the 1/4096 grid; use a supported value for a
   positive fixture and a rejection fixture for .005. The test's numerical
   output does not establish applicability at that off-grid point.

2. **Preserve the support-table definition.** For the current dyadic grid,
   re-evaluating the identical deterministic h at p is consistent with the
   table. For other reference counts, stored rounded p and the mathematical
   rational support need an explicit rank-integer representation; do not
   silently generalize this proof to arbitrary rounded grids. A validated
   support-index lookup would also make equality to the normalization table
   explicit. Check table values are finite, nonnegative and nonincreasing,
   and that grouped masses still sum to N.

3. **Do not overstate the numerical certificate.** Upward rounding of C
   certifies the denominator for represented heights; final float division
   at line 33 can still round individual evidence upward. Thus it alone
   does not certify a literal represented expectation <=1. If that narrow
   certificate is wanted, round each table quotient downward relative to
   its exact rational ratio, or verify its final represented weighted sum
   and conservatively correct it. Later gamma arithmetic and the upstream
   bridge retain their own numerical qualifications. The existing module
   docstring correctly discloses that final division remains float64.

4. **Keep inference inputs distinct from scoring inputs.** The inspected
   driver uses saved p, PILOT receipts, fixed rank count and gamma to form
   normalized evidence (`grid_diagnostic:35-44`); truth enters only scoring
   at 46-47. It does not refit or regenerate references. This agrees with
   the requested saved-p-only change. The evidence nondecrease assertion at
   line 45 is a diagnostic tolerance check, not a replacement for this proof
   or for exact support membership. No archived result was independently
   rescored by this reviewer.

The inspected fixtures enumerate the support for two reference counts and
check approximate budget/monotonicity, but were not run here. Their numerical
tolerance is not a proof of the conditional bound or of strict rounding safety.

## 6. Established principle, efficiency and research status

This is a direct discrete-support specialization of monotone p-to-e
calibration/stochastic ordering, not a new general theorem. For another short
proof, extend h(t_k)/C as a constant on (t_(k-1),t_k]. This is a decreasing
function with continuous integral one and agrees with the desired normalized
values at every supported p. The standard calibrator characterization then
applies; see [Vovk and Wang, Proposition 2.1 and proof](https://arxiv.org/html/1912.06116v4#S2).
That source establishes the general principle; this review does not claim
it explicitly analyzes this project's PC grid or establishes priority for a
particular implementation. No broad novelty search was performed.

The normalized original h is not automatically a valid calibrator at arbitrary
off-grid arguments: the valid generic extension is the right-endpoint step
function just described. This distinction is why support validation matters.

Normalization cannot eliminate the information limit from the finite grid.
Since C>=t_1*h(t_1), normalized evidence is at most 1/t_1: 4096 for projection
and 4096/3 for ordinary. At m=512 and q=.05 these caps require at least three
e-BH rejections for the projection-capable mixture, or eight for pure ordinary,
whenever there are any rejections. Actual calibrator/gamma choices can impose
stricter limits. These are algebraic consequences, not new performance results.

Finally, applying this correction after inspecting D002 and rescoring its
saved p-values is development. The pointwise evidence improvement is genuine,
but same-input power improvements are neither independent confirmation nor
evidence of a novel principle. Freeze this precise normalization and preserve
the old results if it is later evaluated on fresh data; this audit authorizes
no new experiment or further tuning.

**Final disposition:** accept the proposed conditional discrete-support proof,
including the ordinary PC support and C=0 branch. No additional statistical
failure allowance is needed beyond the bridge's existing assumptions. The
remaining narrow issues are enforcing the input-support contract and keeping
the finite-precision claim honest, not a gap in the summation-by-parts argument.
