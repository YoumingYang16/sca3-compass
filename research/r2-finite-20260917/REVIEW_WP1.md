# Independent WP1 theory and prototype review

Date: 2026-09-17. Scope: the explicitly authorized R2 phase only. V1 and the
postrelease archive are unchanged. This reviewer read sources and the directly
relevant primary result; ran no tests, fits, simulations or agents. This review
is the only file written.

## Disposition

**Accept the proposed exact-arithmetic argument under the assumptions below.**
The full-contrast pivot, fixed-iteration shape invariance, randomized shape
envelope, and fold-local restricted-good-event FDR composition fit together.
There is no need to condition on the union of TRAIN folds, nor to establish
conditional coverage of the shape envelope given the realized TRAIN sample.

This is not certification of a numerically exact executable. The initial
`finite_calibration.py` snapshot implemented primitives only; the subsequent
full wiring and `THEORY.md` were read in the same review and are addressed in
section 8, which supersedes the initial wiring-pending status. Numerical and
input/failure obligations in section 7 remain. The conclusion
is not a novelty claim, a power guarantee, a biological applicability claim, or
a theorem about the unchanged V1 procedure.

Reviewed snapshots (SHA256):

- `PROBLEM_AND_ASSUMPTIONS.md`:
  `6e305a650a3d6cfb3af0963c3d240a5280e2e3aa8492ff957867301b92f5a525`.
- `finite_calibration.py`:
  `ba5eccdca88f6eaedfc365b891df91bae5c05a05c105ed65f02f2a599df6ccd6`.
- `test_finite_calibration.py`:
  `f54bddfb8786fad6aa272cb4777907b419b1023fcfcb5026b34504880f5ba4b0`.

The five recorded fixture passes were inspected, not rerun. They are useful
implementation checks, not proofs of invariance or coverage. `THEORY.md`, named
in the prototype's first line, was not present at the reviewed snapshot.

## 1. Full-contrast pivot and composite intersection null

Use `D` for the J-by-L contrast matrix, L=J-1, to avoid confusing it with the
estimated study shape H. Let c=1-rho, v=[1+(J-1)rho]/J and kappa=v/c. Under M0,
orthogonal Gaussian decomposition gives, for each gene,

    Y = sqrt(V*c) R^(1/2) G,
    M = mu + sqrt(V*v) R^(1/2) xi,

where G has S*L independent standard Gaussian entries and xi is an independent
S-dimensional standard Gaussian vector; V is independent of both. Consequently

    Q_R = tr(R^-1 YY') = V*c*Q,       Q ~ chi-square_(S*L).

For a fixed nonzero direction a,

    T_R(a) = a'M / sqrt(kappa*(a'Ra)*Q_R/(S*L))
           = Z/sqrt(Q/(S*L))
             + a'mu/sqrt(V*v*(a'Ra)*Q/(S*L)),

with Z standard normal independent of Q. At a'mu=0 this is exactly t_(S*L).
At a'mu<=0 it is pointwise no larger than the central term in this coupling.
Thus its one-sided central-t p-value is superuniform. This is not a fixed
noncentral-t assertion under arbitrary V.

For sign d in {+1,-1}, apply the formula to dZ. A nonnegative a supported on an
intersection of studies satisfying d*mu_s<=0 has the required nonpositive
shift. Nonnull study means elsewhere do not contaminate Q_R: all study means
are common across pipelines and hence disappear from Y. A positive shift on
a purported null intersection would not be covered.

Important boundaries:

- The same positive radial V multiplies the **whole S-by-J target block**.
  Unrelated study-specific radii do not give this S*L pivot.
- Compound pipeline shape and common-location means across all J pipelines
  are essential. This is not validity for arbitrary pipeline covariance or
  unequal pipeline means.
- The result is marginal over the target's own Y, not conditional on it.
  Learning a, gamma, or a p-to-e calibrator from the held target's Q is not
  justified by this argument.
- E[sqrt(V)]<infinity is needed to call mu the target mean; the algebra itself
  needs neither E[V] nor an inverse-gamma law. Otherwise mu is only a location.

These points agree with the intake document. For S=4,J=6 the retained degrees
of freedom are 20, not the old five-df target pivot.

## 2. Finite-iteration shape invariance: yes, without convergence

For training blocks Y_g of size S-by-L, define W_g=Y_gY_g' and

    H_0 proportional to sum_g W_g / det(W_g)^(1/S),

with trace normalized to S. For any invertible A and arbitrary positive
block scales r_g, transforming Y_g to r_g A Y_g transforms each summand to

    |det(A)|^(-2/S) A [W_g/det(W_g)^(1/S)] A'.

The block scale cancels exactly and the determinant factor is common to all
summands. Therefore the trace-normalized initialization is affine-equivariant
up to scalar and invariant to each block radius.

For the unregularized Tyler update on columns x_l, write

    Phi(H; x) = sum_l x_l x_l' / (x_l' H^-1 x_l).

If H is transformed to t*AHA' and columns to b_l*A*x_l, then Phi is transformed
to t*A*Phi(H;x)*A'; every nonzero column scale b_l cancels. Normalizing the
trace preserves equivariance up to scalar. Induction proves the assertion for
every **fixed finite** number of updates, including zero. Neither existence of
a limiting Tyler solution nor convergence after 12 updates is being assumed.

Under M0, L>=S makes every W_g positive definite almost surely. Their sum is
positive definite, and the full-span columns keep each exact update positive
definite. Gaussian general position supplies this; numerical rank is separate.
The code's n>=2 restriction is an additional interface restriction, not a
necessary condition for the initialization algebra.

Thus, if A_K denotes this fixed algorithm,

    A_K({sqrt(V_g*c) R^(1/2) G_g})
       proportional to R^(1/2) A_K({G_g}) R^(1/2).

It follows that

    D_f = cond(R^(-1/2) H_f R^(-1/2))

has precisely the distribution of cond(A_K({G_g})), independent of R, rho,
the positive block radii and training mean vectors. Training alternatives are
allowed because their common-location means vanish in the contrasts.

This statement needs a common target study shape R and independent Gaussian
training blocks of the prescribed size. Correlated genes, selecting training
genes using observed data, missing columns, or gene-varying nonproportional R
are not automatically covered. The observed cond(H_f) is **not** D_f unless
R is proportional to the identity.

`shape_fit:33-45` does not spoil the exact algebra: max-absolute normalization
is a block scalar removed by the determinant normalization; column unit-norm
normalization occurs only after initialization and cancels inside each Tyler
update. It would not be harmless to replace the initialization by the scatter
of independently normalized columns without a new argument. Ridge terms,
diagonal standardization, eigenvalue clipping and coordinate-dependent stopping
are absent and must not silently be added to this fixed reference design.

## 3. Fresh Monte Carlo reference and the shape envelope

For a fixed training size and the identical (S,L,K) algorithm, let D_1*,...,D_B*
be independent reference condition numbers from independent Gaussian blocks,
independent of real data. D_f,D_1*,...,D_B* are iid in exact arithmetic. For
C=max_b D_b*, exchangeability yields

    P(D_f > C) <= 1/(B+1).

With ties, using the strict failure event still gives the inequality. This is
a finite rank argument, not a fitted quantile or a Monte Carlo approximation
requiring a confidence interval around its simulated maximum.

The probability includes the fresh auxiliary bank. For a realized bank c,
P(D_f>c) need not be <=1/(B+1); nor is there that coverage guarantee conditional
on TRAIN. A benchmark cutoff such as 1.5194 cannot be frozen and promoted to a
universally valid deterministic cutoff. Record the actual independent random
stream and reproduce that bank for replay; do not reroll until a favorable
cutoff or discovery count occurs. A fixed seed's output alone is not the
randomized theorem. The mathematical model idealizes independent random draws;
the pseudorandom implementation must state that convention.

One fresh common bank can be used by both equal-sized folds. Each marginal
failure bound still holds; dependence induced by sharing the bank requires no
independence assumption for the union bound. At B=199 the two-fold bound is
2/200=.01. Unequal training sizes or different update rules need appropriately
matched references and accounting; the equal-size receipt cannot be reused
without that check. No factor for genes, signs or directions is needed because
the following envelope holds simultaneously for all of them.

Let lambda_min*R <= H <= lambda_max*R, with D=lambda_max/lambda_min. Then

    (a'Ha) tr(H^-1 YY')
       >= (lambda_min/lambda_max) (a'Ra) tr(R^-1 YY').

On D<=C and kappa<=U_kappa, therefore,

    U_kappa*C*(a'Ha)*tr(H^-1 YY')/(S*L)
       >= kappa*(a'Ra)*tr(R^-1 YY')/(S*L).

The multiplier is **C on the squared scale**, or sqrt(C) on the standard
error, not C squared or an unprotected plug-in replacement. It is invariant to
the arbitrary normalization of R or H.

For a'M>0, increasing this denominator increases the one-sided p-value. For
a'M<=0, that monotonicity reverses: setting p=1, as `projection_p:126` does,
is a crucial safe choice. It ensures the protected p is pointwise at least
the ordinary oracle central-t p for every target realization on the good event.
Keep this branch; do not justify an unclipped all-sign tail by scale inflation.

## 4. Kappa coverage counts independent blocks, once

For a fixed nonzero study direction b and independent centered calibration
blocks C_l, the ratios

    A_l = (b'C_l 1/J)^2 / (||b'C_l D||^2/L)

satisfy A_l/kappa ~ F_(1,L), independently. The common block radius and b'Rb
cancel. The scalar pivot does not require matching target/calibration radial
laws. Centering and matching rho remain essential; estimating the center from
these same observations does not automatically preserve the pivot.

For a design-only order k, with u_delta the lower delta quantile of
Beta(k,N+1-k),

    U_kappa = A_(k) / F_(1,L)^(-1)(u_delta),
    P(kappa > U_kappa) = delta

in exact arithmetic. `kappa_order:74-88` chooses k from N,J,delta only; minimizing
its deterministic median-inflation proxy does not select on observed ratios.
This choice is valid but is not a proof of power optimality.

`kappa_upper:91-110` uses the first fixed study once per block, correctly giving
N ratios, not S*N independent ratios. A single such bound shared by both folds
costs delta_kappa=.005 **once**. With the proposed two shape events, the total
bad-event bound is .015 and e-BH q0=.05-.015=.035. These are algorithmic error
budgets, distinct from any later simulation-reporting confidence allocations.

The new N independent S-by-J calibration generator is the appropriate unit.
Transposing it to the old S-by-N-by-J V1 interface does not erase within-block
study dependence. V1 comparisons on those inputs must be labeled empirical
under the changed calibration design, not covered by this R2 argument or by
V1's prior frozen acceptance. The number of independent blocks and total
vectors must both be reported; this is not an unchanged-data-design comparison.

## 5. Fold-local conditional argument and cyclic crossfits

For fold f, define F_f to contain exactly:

- Its TRAIN target blocks, with fold membership fixed independently of targets.
- Independent calibration blocks and the fresh auxiliary reference bank.
- All permitted learning randomness, frozen design constants, and quantities
  computed solely from that information: H_f, directions, gamma, component
  pilots, calibrator thresholds and mixture weights.

HELD target blocks are independent of this information under M0. Define the
theoretical event

    B_f = {kappa <= U_kappa} intersect {D_f <= C}.

For a fixed model parameter R,kappa this event is F_f-measurable; it need not
be observable to the algorithm. On every good realization of F_f the envelope
holds pointwise, and the oracle pivot remains superuniform after integrating
the held target's own Gaussian contrasts and radius. Thus protected null
intersection p-values are conditionally superuniform on B_f.

For r=2, taking the maximum over all S-1-study intersections gives a valid PC
p-value: under the PC null, at least one of these intersections is wholly
null. The ordinary component's capped Bonferroni minimum is valid on that
intersection without study independence. `pc_components:129-146` implements
these two constructions. Its comment about arbitrary study dependence applies
to the **PC combination**; the supplied marginal/projection pivots still need
the matrix-normal angular M0, not arbitrary non-Gaussian dependence.

For each component c, permit a distinct F_f-measurable nonnegative decreasing
calibrator h_fc with integral over [0,1] at most one. Permit nonnegative
F_f-measurable mixture weights summing to one. Then for each null claim i in
HELD_f, the resulting evidence satisfies

    E[E_i | F_f] <= 1 on B_f,
    E[1_(B_f) E_i] <= P(B_f) <= 1.

The TRAIN pilot's input p-values need not themselves be valid tests; here
they only choose a normalized calibrator. However their entire provenance
must be TRAIN-local. Reusing another fold's cross-fitted p-values as a TRAIN
pilot can leak the present HELD data through their fitted shape/directions,
even though the final pilot array has TRAIN row indices. Audit recursively.

Shared calibration and auxiliary references may influence learning and be
included in F_f. No split of calibration solely to obtain conditional coverage
is required by this restricted-event proof. The coverage bounds are marginal;
the **target** validity conditional on F_f on B_f is the separate ingredient.

Let B=intersection_f B_f and m=2G. e-BH at q0 gives pointwise

    FDP <= (q0/m) sum_(true null i) E_i.

Since B is contained in the relevant B_f and evidence is nonnegative,

    E[1_B FDP] <= (q0/m) sum_(null i) E[1_(B_f(i)) E_i]
                  <= q0*m0/m <= q0.
    E[1_(B^c) FDP] <= P(B^c) <= delta.

Hence FDR<=q0+delta=.05 for the specified ideal procedure. This proof tolerates
the cyclic relation where TRAIN for one fold includes HELD for another. It
never conditions on the union of training folds, on the global event B when
asserting target superuniformity, or on a held gene's own Q. No mutual
independence between final e-values or between the B_f events is needed.
It does not assert unconditional E[E_i]<=1 or q0*(1-delta)+delta control.

## 6. Direct primary-source overlap and novelty boundary

I read [Ignatiadis, Wang and Ramdas, arXiv:2409.19812v2,
Proposition 5.4(E2) and Theorem 6.4](https://arxiv.org/html/2409.19812v2#S5.SS1).
The global restricted-expectation condition above is an instance of their
approximate compound e-value condition; their e-BH result gives the same
additive failure-budget FDR bound. The composition lemma therefore must not
be described as a new theorem or a new resolution of arbitrary e-value
dependence. The forward implication used here does not need the atomlessness
condition attached to their converse characterization.

The Gaussian/chi-square Student pivot, classical order-statistic inversion,
Tyler update and exchangeability rank bound likewise do not establish novelty.
The present audit verifies the proposed finite-step construction and how its
envelope composes with fold-local learning. Priority or useful methodological
contribution of that particular combination remains unestablished. Main's
targeted literature matrix is separate; no broad alternative-route search was
performed here. The old R1 five-df pivot and R0070 low-power evidence remain
historical constraints, not overturned results.

## 7. Concrete implementation obligations and limitations

1. **Complete dependency wiring remains to be inspected.** The reviewed file
   stops at PC components. Before an end-to-end claim, bind the fixed folds,
   actual training block count, K=12, B=199, shared reference design, kappa
   budget, per-component TRAIN pilots, gamma and e-BH q0=.035 in one receipt.
   All 2G claims must enter the declared family; reject/report no labels based
   on whichever observed method gives the best output. Calibrator internal
   shaping levels may differ from q0 if their integrals remain <=1, but the
   final rejection level must actually spend the declared q0.

2. **Primitive input validation is incomplete.** At `projection_p:113-126`,
   checking only the product called `variance` does not ensure finite target
   input, a finite correctly sized direction, a symmetric positive-definite
   shape, positive finite kappa and an admissible protection factor >=1.
   Negative kappa and negative factor can even have a positive product. The
   default factor=1 must not silently become the unknown-shape deployment
   path. At `pc_components:129-143`, validate finite nonnegative profiles with
   shape (2,S), complete target dimensions, and the r=2 restriction. Add these
   checks in the new phase's wrapper/primitives, not in immutable V1.

3. **Numerical exactness is a real, explicitly unresolved qualification.**
   `shape_fit:37-50` forms Gram determinants and matrix inverses; positivity of
   a determinant or final eigenvalue alone is not an accuracy certificate.
   With arbitrary PD R, finite arithmetic can lose affine equivariance long
   before an exception. Reference fits at R=I do not calibrate that error for
   ill-conditioned actual R. Cholesky/linear solves, finiteness and conditioning
   diagnostics are sensible engineering safeguards, not a proof of uniform
   floating-point coverage. Beta/F quantiles (`83-88`), kappa inversion (`108`),
   eigenvalue ratios (`68-70`) and Student survival tails (`126`) also need
   outward/certified error control for a literal finite-precision guarantee.
   No fixed epsilon or passing invariant fixture supplies it. Keep the
   exact-arithmetic theorem and numerical prototype status separate.

4. **Fail closed without selective reruns.** Require finite positive results
   at intermediate Tyler denominators q, trace normalizations, kappa upper
   bound and reference cutoff. Singularities, overflow or invalid quantiles
   must not trigger ridge repair, dropping blocks/reference draws, rerolling
   a favorable bank or reverting to an unprotected plug-in result. A declared
   whole-analysis no-discovery result is a conservative total-algorithm
   fallback for detected failure; an exception is acceptable for a prototype
   but does not by itself define a deployed procedure. Retain every failure
   in diagnostic accounting, rather than analyzing only successful families.
   This policy does not cure undetected numerical error in item 3.

5. **Coverage is not free power.** The random maximum shape factor and a
   .005-tail upper kappa bound both inflate the squared scale; q0=.035 spends
   30% of the nominal .05 rejection budget on bad events. Small N can make
   the kappa bound very large. Twenty contrast degrees of freedom do not
   guarantee recovery of those losses or superiority to the old five-df or
   V1 methods. The reported .094s benchmark and cutoff 1.5194 are one cost
   observation, not an accuracy, coverage or power result. No tuning of K,
   B, order selection or budgets to forthcoming outcomes is justified by
   this review.

**Handoff:** the algebra and proposed .015/.035 allocation are defensible under
M0, with fresh independent references and the fold-local construction above.
Proceeding with the already scoped prototype is not blocked by a union-TRAIN
conditioning objection or by lack of Tyler convergence. Full implementation
wiring, fail-closed behavior and numerical scope must remain explicit before
claiming an implemented guarantee. No new experiment or broader route is
requested by this review.

## 8. Same-turn follow-up: full draft and executable v0.2

Main completed the matrix calibration and full evaluation wiring during this
review. I read both in full, without executing them. This section updates the
initial primitive-only assessment, without silently changing its source receipt.
New reviewed SHA256:

- `THEORY.md`:
  `3bdebe84d75b44f72ffb476b31607e1eac37ad37ff4e7801c4085d6b9b918e35`.
- `finite_calibration.py`:
  `c088d790289fd2960d01339520d04718fa9e112cd9a5f1e66148f99703471e21`.

### Matrix calibration formula and integer quantile certificate: accepted

For centered calibration, x=sqrt(V*v) R_c^(1/2) u and
Y=sqrt(V*c) R_c^(1/2) E give

    x'(YY')^-1 x = kappa*u'(EE')^-1 u.
    [(L-S+1)/S] * x'(YY')^-1 x / kappa ~ F_(S,L-S+1).

The normal-Wishart derivation in `THEORY:22-28` is correct: rotate u to its
first axis; its squared norm is chi-square_S, and the relevant inverse-Wishart
Schur complement is an independent chi-square_(L-S+1). At S=4,L=5 the factor
is 1/2 and the law is F_(4,2). `matrix_kappa_upper:150-169` uses that factor and
counts one statistic per independent calibration block. Unknown R_c may vary
between blocks; no matching target R is needed for this calibration pivot.

In `matrix_order:125-147`, for a positive proposed dyadic denominator d, the
F_(4,2) CDF is exactly p=(2d/(1+2d))^2. The integer comparison checks

    sum_(i=k)^N choose(N,i) p^i (1-p)^(N-i) <= delta.

This is exactly the order-statistic failure probability. Choosing k from
design constants and then reducing d until this inequality holds is valid.
The supplied rational certificate resolves the **quantile denominator**
rounding issue for this matrix pivot. It does not certify the observed
quadratic forms, order/division arithmetic, shape fit or Student tails.
There is no basis here to call the whole executable interval-certified.
The scalar b-pivot remains a separate backup; `evaluate:245` now uses the
matrix pivot, so section 4's scalar description is not the final primary path.

### Concrete mathematical correction: separate calibration and target radii

`THEORY:32-33` says independent per-study radial factors within a calibration
block do not generally cancel in the matrix pivot. **That exclusion is too
strong for the intended row-scale Gaussian model.** If centered calibration
has x=sqrt(v)*A_l*u and Y=sqrt(c)*A_l*E for an invertible left matrix A_l,
then, pointwise,

    (A_l*u)' [(A_l*E)(A_l*E)']^-1 (A_l*u) = u'(EE')^-1 u.

In particular A_l=diag(sqrt(V_l1),...,sqrt(V_lS))*R_c^(1/2) permits independent
positive study-row scales, each shared across that row's J pipelines and
independent of the Gaussian block. Condition on these scales; the law remains
F_(S,L-S+1), with no remaining scale dependence. This includes the independent
Gaussian scale-mixture row model with R_c=I. The same blockwise affine
cancellation already underlying Proposition 1 proves it.

Narrow correction: say that common block V is sufficient but not necessary
for **matrix calibration**; independent row scales also cancel under the
specified common-column Gaussian structure. The **full-contrast target pivot
and target TRAIN shape protection as currently stated** still require their
shared target-block radius/common target R. Do not transfer the calibration
extension to those target results. Pipeline-specific scales, unequal rho
across rows, nonzero calibration means and arbitrary non-Gaussian angular
departures are not covered by this observation.

The new block-calibration generator can remain exactly as proposed; this is
an assumption/documentation correction, not a request to change it or launch
another route. Nor does the algebra alone certify every historical V1 input
generator. The changed comparison design must still be disclosed, but it
should not be described as a mathematically necessary common-radius condition
for the calibration matrix pivot.

### Full learning/evidence chain: no new dependence blocker found

The inspected `evaluate:231-294` now follows the requisite chain:

- One matrix kappa bound and one fresh 199-reference bank are shared; equal
  parity folds each contain G/2 training genes (`245-257`). Both actual shape
  events are charged, and final e-BH uses .05-delta_kappa-2/(B+1) (`247-250`,
  `286-287`). The actual and reference shape calls use the same default 12
  updates and the same dimensions.
- H, profiles and gamma use only the other fold (`257-260`). NNLS directions
  in `pc_components:199-202` use these profiles and H, not HELD means or Q.
  The imported `positive_direction` supplies nonnegative directions. Pattern
  nonconvergence gives gamma=0 (`214-221`), an ordinary-component fallback.
- TRAIN p-values use the same local H and median kappa estimate, and each
  component's pilot uses only those TRAIN probabilities (`260`, `265-267`).
  These TRAIN scores need not be calibrated. In particular the fitted pattern
  model need not be the true arbitrary-radial likelihood for this learning-only
  role. Its model fit is not an additional inference assumption.
- HELD p-values use U_kappa*C; ordinary and projection calibrators are mixed
  convexly using TRAIN gamma (`261-269`). `calibrate:224-228` uses a fixed-level
  TRAIN BH count and an integrable focused calibrator with positive threshold.
  The .05 pilot threshold is allowed even though final e-BH uses .035.

Consequently the full source wiring at this snapshot matches Proposition 4's
exact-arithmetic dependence argument. This closes section 7 item 1's
primitive-only gap at source level. A future frozen run still needs input,
source, seed and budget receipts; no generator, runner or execution outcome is
certified by this source-only finding. The scope of arbitrary final-evidence
dependence does not remove independent-gene TRAIN requirements.

### Remaining precise qualifications for this snapshot

1. `projection_p` is now at lines 172-185, `pc_components` at 188-206, and
   `shape_fit` at 32-65. Section 7's numerical/input/failure issues persist.
   The top-level `evaluate` validates target dimensions and finiteness but
   still raises on numerical failures rather than defining a no-discovery
   outcome. Prevalidate integer reference counts and finite error allocations
   before costly computations; do not let malformed budgets reach Fraction or
   quantile operations. Main can retain prototype status with these limits.

2. `THEORY:143-147` correctly identifies the deterministic BB/BY floor. With
   delta_kappa+delta_shape=.01, `bb:271` makes every corrected p at least .01;
   at m=512 every BY critical value is at most .05/H_512<.01. `BB_BY` therefore
   cannot reject in the intended G=256 design. It is not a measured low-power
   outcome or sufficient evidence of superiority to a competitive comparator.
   The further sentence saying a BB/e-BH reference "is also retained" is
   ahead of the reviewed executable: `tags:252` and decisions at `286-288`
   contain no such method. Mark that sentence planned or align it with the
   actual eventual method list; no added comparison is required by this review.

3. `evaluate:279` says learning is kept identical for oracle decomposition.
   Profiles/gamma and the TRAIN pilot arrays are indeed held fixed, but
   `pc_components(..., true_shape, ...)` recomputes the NNLS directions using
   true_shape, and the calibration/e-BH levels also change (`276-281`). Thus
   these are oracle-information references, **not an isolated intervention on
   only the inference denominator or the cost of shape protection**. Describe
   that distinction explicitly if reporting their differences.

4. An externally justified mismatch_bound>=1 multiplies U as stated in
   `THEORY:148-150`; no target-only estimate of a transport bound is supplied.
   It only covers the asserted kappa ratio, not arbitrary angular mismatch.

**Updated verdict:** Propositions 1-4 and the implemented fold-local chain are
accepted as exact-arithmetic deductions after correcting the calibration
row-radial exclusion above. The calibration F_(4,2) order-quantile certificate
is correctly scoped. No new conditional-TRAIN, Tyler-convergence or dependence
blocker was found. Numerical certification, fail-closed deployment, comparator
wording and empirical utility remain separate; no Power outcomes, extra tests,
fits, samples or agents were introduced by this reviewer.

## 9. Late BB/e-BH addition: a reference-specific proof gap

Before handoff, the file advanced again to SHA256
`b47f55ab32363fa66ca102aa3eeb85fe85eee108bd41e3e5034c7592694283a7`.
I read the changed evaluation region. `BB_eBH` now exists, closing section 8's
method-list mismatch, but introducing an inferential qualification that must
not be hidden by that bookkeeping closure.

At `finite_calibration.py:272-274`, the added BB-corrected probabilities are
fed to TRAIN-selected component calibrators and mixed using TRAIN gamma; at
`289` their final e-BH level is .05. The per-test additive correction proves
**marginal** superuniformity, not superuniformity conditional on that TRAIN
information. Therefore it does not, by itself, justify this adaptive p-to-e
conversion or adaptive mixture as ordinary mean-one e-values.

A simple counterexample to that generic reasoning: let a TRAIN-measurable
bad event have probability d. On it take uncorrected p=0; off it take an
independent uniform p. The good-event p assumption holds, and
P_BB=min(1,p+d) is marginally superuniform. Choose the unit-integral calibrator
I(P_BB<=d)/d on the bad event, and the constant-one calibrator off it. The
result has expectation d/d+(1-d)=2-d>1. The correction did not authorize
TRAIN-dependent calibration. This counterexample diagnoses the missing
implication; it is not a simulation or evidence of this prototype's actual FDR.

Narrow dispositions for this existing reference are: use genuinely fixed,
data-independent calibrators and mixing weights so marginal BB validity
suffices; or retain TRAIN learning but use the restricted-event q0 budget;
or label the learned .05 BB/e-BH reference empirical/unproved. Do not claim
its .05 guarantee without a separate argument. No extra method family or
experiment is requested. `BB_BY`, which directly uses marginal p-values with
a fixed BY rule, does not have this conditioning gap (its power floor remains).

**Final disposition at these snapshots:** the main R2 and same-information
ordinary chains retain the accepted exact-arithmetic .035+.015 proof. The
calibration row-radial exclusion needs correction; the newly added learned
BB/e-BH reference lacks the asserted standard marginal-p-to-e justification.
These are separate from the still-open numerical/deployment safeguards and
from any future power assessment. Subsequent edits require their own bounded
disposition; this review does not preapprove moving source versions.
