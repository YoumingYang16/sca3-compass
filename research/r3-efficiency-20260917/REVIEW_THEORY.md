# R3 geometric-scale / independent-orientation proof review

2026-09-17. Sole independent reviewer. Read `R3_THEORY.md` fully and checked
the inherited M0 specification and finite-step shape estimator in frozen R2.
No candidate implementation, tests, fits, simulations, new method, or literature
route was undertaken. Only this review file was written; R2/V1 were not changed.

**Disposition: PASS for the stated ideal-model proof chain. No concrete proof
objection to implementing this specified candidate.** This is not implementation
clearance, a numerical certificate, a power prediction, or a novelty finding.
The conditions below must be preserved in the executable and its interpretation.

## 1. Geometric calibration law and its moments

`R3_THEORY.md:11-30` is correct. For the inherited centered matrix pivot,
`A_l=kappa X_l`, with independent X_l~F(4,2). Hence

    kappa_g/kappa = exp(mean_l log X_l)/(e/2) = B_g.

The F representation by independent chi-squares gives

    E log X = psi(2)-psi(1)+log(2/4) = 1-log(2),
    Var(log X) = psi1(2)+psi1(1) = pi^2/3-1.

Thus E log B_g=0 and Var(log B_g)=(pi^2/3-1)/N exactly. The stated
1.5132/sqrt(N) log-scale asymptotic coefficient is correct, and the relative
scale has the same first-order coefficient by the delta method. This is
log-centering, not a statement that E[B_g]=1 or a finite-N median-unbiasedness
claim. The comparison with the median's 1.7071 coefficient does not establish
power dominance or a finite-N risk ordering.

No target radial moments beyond the inherited assumptions enter this pivot
law. Calibration independence, centering, angular structure and matching kappa
are still required. Keeping the old median for DIR/PILOT learning is legitimate:
its correlation with B_g through the shared calibration is handled below, not
assumed absent.

## 2. Finite-step shape invariance really gives the needed reference object

`R3_THEORY.md:34-53` correctly distinguishes an invariant shape representative
from the actually trace-normalized fitted H. The existing estimator
(`R2/C001/finite_calibration.py:33-66`) has, in exact arithmetic:

- block-radial invariance from determinant-normalized initialization and the
  scale cancellation in each Tyler update;
- affine equivariance up to a positive scalar, with a fixed iteration count;
- exact orthogonal equivariance for the trace-normalized representative.

The preliminary maximum-entry and column-norm scalings do not defeat these
properties: their positive scalar factors cancel in those same operations.
No convergence to a Tyler solution is needed. Gaussian full-rank blocks with
L=5>=S=4 give the required positive-definite initialization almost surely.

Under M0 the SHAPE contrasts remove their unknown locations. Write the fitted
matrix as

    H = s R^(1/2) H_* R^(1/2),  s>0,

where H_* is the same trace-normalized finite-step fit on the independent
standard-Gaussian SHAPE blocks. The scalar s may depend on R and H_*; it need
not be independent or nuisance-free. It cancels in
`(a'Ha) tr(H^-1 YY')`. Therefore neither H nor the unnormalized whitened H is
being falsely asserted to have an R-free law.

For G=256 the primary reference must fit 64 SHAPE blocks with the same fixed
two updates. The old 128-block reference would have the wrong law here.

## 3. Calibration-dependent directions do not break the joint orientation law

Let P be the entire corresponding PILOT fold and let
`C = sigma(DIR, external calibration)`. For each intersection, take its
nonzero nonnegative learned direction a, padded with zeros outside its triple,
and set `b=R^(1/2)a/||R^(1/2)a||`. Conditional on C and P, b is fixed and H_*
is independent with an orthogonally invariant law.

For any deterministic unit b, choose an orthogonal O with O e1=b. Then

    (b'H_*b, spectrum(H_*))
      has the law of ((O'H_*O)_11, spectrum(O'H_*O))
      has the law of ((H_*)_11, spectrum(H_*)).

The right-hand law is the same for every b. Consequently the conditional law
of this pair does not depend on C, even when b depends on B_g, on the old
calibration median, or on other information in the calibration matrices.
The pair is therefore independent of B_g after integrating C. Independence
of H_* from DIR/calibration is essential; independence of b from B_g is not.

This is a joint statement about the quadratic form and spectrum. H_11 and the
spectrum must come from the **same** fitted H_*. Drawing them independently,
using just E[H_11], or retaining a direction selected with the inference H
would not implement this proof.

## 4. Target central law, null shifts, and the conditioning order

For the held target, Gaussian mean and contrast coordinates are independent
before the shared radial multiplier. The common target radial factor cancels
from the centered ratio. With independent standard Gaussian mean vector g and
contrast matrix E, its centered statistic is

    T0 = (b'g) / sqrt[B_g (b'H_*b) tr(H_*^-1 EE')/20].

Conditional on the learners, b'g is standard normal. Its conditional law is
constant and it is independent of the contrasts and SHAPE fit. Diagonalizing
H_* gives the conditional contrast-energy law `sum_j U_j/lambda_j(H_*)`, with
four independent chi-square5 variables; these can be represented independently
of H_*, the learners, and the normal numerator. Combining this fact with the
joint orientation identity gives precisely

    W = Z/sqrt[B_g (H_*)_11 sum_j U_j/lambda_j(H_*)/20].

**Important conditioning clarification:** conditioning on C is an intermediate
step to prove invariance. Then integrate the calibration sampling law to obtain
the B_g mixture before claiming rank-p validity. The final validity is
conditional on P only, not conditional on the realized DIR/calibration values
or on all cached learning receipts. The draft's section 3 uses the correct
final conditioning field.

On a signed intersection null, nonnegative a gives a nonpositive projected
location. The actual statistic is therefore no greater than T0 pointwise,
including when its noncentral shift depends on the target radius. This gives
the required positive-tail domination, not a central-law assertion for every
non-null target. Keep p=1 for nonpositive observed statistics.

The inherited target model still requires a common scalar radius within each
target block; study-specific target radii or non-common pipeline means are not
covered. E[sqrt(V)]<infinity remains needed when locations are called means.
Calibration's separately permitted invertible-left-factor cancellation is not
a relaxation of the target assumption.

## 5. PC, calibration, cyclic roles, and complete FDR control

Fresh reference tuples must be independent of the real data, with independent
components B_g, Gaussian SHAPE fit, Z, and U. Integrating DIR/SHAPE/calibration
conditional on P makes a coupled central T0 identically distributed with those
references. Plus-one ranks with >= ties are superuniform; a nonpositive null
shift and clipping to p=1 preserve that property.

The same argument applies to each ordinary coordinate direction. Bonferroni
over three coordinates does not require their independence. A signed PC null
with at most one positive study contains an all-null triple; the maximum of
the four intersection p-values is at least that valid triple's p-value.
This covers partial, singleton, and mixed-sign nulls. No claim that a single
scalar reference bank reproduces the full joint distribution of every learned
direction is needed.

Projection and ordinary PC p-values retain their respective rank grids.
PILOT-only decreasing support-normalized calibration and PILOT-only convex
gamma then give `E[e_i | P_i] <= 1`. For m=2G, e-BH self-consistency yields

    FDR <= (q/m) sum_(i true null) E[e_i] <= q*m0/m <= q.

Calibration/reference sharing and the cyclic fold-role overlap are permissible
at this final step. Do not condition on the union of all PILOT/DIR/SHAPE roles.
The revised split loses learning sample size but does not omit any hypothesis
from the cyclic tested family.

## 6. Implementation boundaries and ablation interpretation

- Keep the DIR and inference-SHAPE information disjoint. NNLS must use the
  DIR-only metric/profile; no choice of its profile, fallback, tuning, or
  iteration count may consult the realized inference H or reference bank.
  Retain a nonzero nonnegative direction/fallback so its normalization exists.
- SHAPE uses only its contrasts and a fixed affine-equivariant procedure;
  no ridge, diagonal standardization, adaptive convergence choice, or different
  reference update count is licensed by this proof.
- PILOT must still learn entirely from its own fold, including its nuisance
  heuristics. External calibration or inference scores must not leak into its
  calibrator/gamma selection. The geometric inference summary does not require
  replacing those learning heuristics.
- The scale-only ablation may retain R2's 128-block overlapping direction/shape
  roles **only with the original lambda_min dominating reference**, replacing
  the median B law by the geometric B_g law. It may not use H_11 without the
  independent DIR/SHAPE split. This is a necessary ablation of the proposed
  changes, not another candidate family.
- At identical H/B/Z/U tuples, replacing lambda_min by H_11 reduces positive
  reference tails. That comparison does not order complete algorithms with
  different shape/direction sample sizes or different inference statistics.
  No full dominance, no-regression, or achievable-power claim follows.
- The limited Delta statement remains correct: replacing kappa_g by
  Delta*kappa_g multiplies the squared denominator by Delta; when
  kappa_target/kappa_cal<=Delta the same central coupling gives positive-tail
  domination. It does not identify a real-world bound or handle unrestricted
  drift.

The proof uses classical Gaussian/orthogonal invariance, finite-law
studentization, rank calibration, PC, p-to-e calibration, and e-BH. Novelty
remains unestablished. Ideal finite-sample error control, executable numerical
correctness, efficiency evidence, and source/protocol identity remain separate
claims. In particular, this proof review does not close the saved-diagnostic
identity findings in `REVIEW_LOSS.md`.

Reviewed draft SHA256:

```text
R3_THEORY.md
E95DA34CEBEA8AB74B0448D97487D53A464398348968367F453EFBDE898790A9
```

## Direct implementation follow-up, before D002 performance

Reviewed `r3_model.py`, `r3_experiment.py`, `protocol-dev.json`, the seven source
fixtures in `test_r3.py`, and the D001 post-run auditor/receipt. No code or tests
were run by this reviewer; the reported seven-test PASS is main's report.

**Disposition: no concrete implementation/proof blocker found for the fixed
312-input reused-data D002 development protocol.** This is a source-level check
of that scope, not a result verdict, universal numerical certificate, or audit
of a future fresh-input confirmation design.

### Critical role separation is implemented correctly

At `r3_model.py:82-88`, each cached `hs[j]` depends only on fold j's contrasts.
`dirs[j]` uses only that fold, its own shape, and the external calibration median.
Its gamma output is not used for primary evidence. The separate `pilots[j]`
uses the same fold's shape, its own median heuristic and its own scores; it does
not use the external calibration median or the reference bank.

At `:91-96`, the primary call is exactly

    components(z[held], hs[shape], kg*mismatch_bound,
               dirs[direction][0], refs['R3_main'], hs[direction]).

Here DIR=f+2 and SHAPE=f+3 are distinct from TEST=f and PILOT=f+1. Inside
`components`, NNLS uses `direction_shape` at `:70`, while the held contrast
energy and directional standardization use the inference `shape` at `:62,71`.
Thus **NNLS receives DIR's shape, not the inference SHAPE matrix**. Caching all
four fold-local fits together does not create a mathematical information leak.

At `:100-109`, gamma and calibration selections come from `pilots[pilot]` only.
I checked the inherited `finite_calibration.calibrate:226-230`: its receipt is
selected from the pilot scores, despite held p-values also being passed for
evaluation. It does not select the calibrator from those held p-values. Both
component grids and the fixed ordinary control therefore retain the proof's
conditioning field.

### Scale and reference match the accepted laws

`geometric_kappa:20-31` computes the inherited matrix pivot and its log-geometric
mean with log center `1-log(2)`. `references:38-49` samples the corresponding
geometric ratio law from the same F(4,2) pivot law. Its primary SHAPE fit uses
G/4 Gaussian blocks, and `half[:,0,0]` and `eh` are derived from that SAME fit.
The chi-square5 energies and normal numerator are independently sampled tuple
components in the ideal randomization model.

The half/full Gaussian sharing couples primary and ablation references but
does not invalidate either marginal reference law. Likewise, sharing F draws
for median/geometric reference factors is legitimate pairing. No inference
truth or true nuisance parameter is accepted by `r3_model.evaluate`.

The scale-only call at `:97-99` deliberately keeps R2's two-fold direction/shape
roles, and its reference at `:48` correctly retains `lambda_min(full)` rather
than H_11. Thus it does not misuse the new independent-orientation proof. The
R2 replay reference retains the old draw schedule; the reuse executor checks
its equality against the archived bank at `r3_experiment.py:66-68` when the
candidate computation completes.

The primary-versus-scale ablation still changes shape sample size, direction
sample size, and the orientation reference together. Interpret it as the net
effect of the independent-orientation design, not an isolated numerical
estimate of Rayleigh-envelope loss. No additional ablation is requested.

### Executor and evidence identity

`r3_experiment.validate:20-29` checks frozen file hashes and the actual resolved
paths/hashes of the executor, model, and common module. `:103-108` calls it in
the main process and installs it as the worker initializer. It must be invoked
from the frozen D002 copies, as this check requires.

`load_source:35-43` binds each source record to the sealed C001 index and uses
the existing full record/metric checker before loading it. `audit_rows:114-138`
requires the exact Cartesian task set, canonical paths, matching record/index/
freeze identities, original source/seed identity, evidence hashes, the declared
decision roster, and rescoring against original saved truth. This addresses the
earlier diagnostic's principal identity gaps for the new reuse workflow.

The current protocol is exactly 13 matched-model scenes x 24 reused families,
including the three null-only scenes, with seven declared outputs. Development
summaries retain undefined null-only power and do not turn paired MCSE into a
formal acceptance interval. Conservative candidate failures remain no-discovery
outcomes in the metric denominators; hard failures block analysis. The reported
`mean_seconds` excludes missing candidate times on conservative failures, so
label it completed-computation runtime if such failures occur, not an all-family
cost estimate. Final saved-evidence/decision checks remain an engineering audit,
not a new experiment or extra confirmation.

D001's post-run identity receipt closes its saved-evidence findings with the
historical-attestation qualification recorded in the appended `REVIEW_LOSS.md`.
It does not substitute for D002's new contemporaneous path checks.

Reviewed implementation/protocol SHA256 values:

```text
r3_model.py
B65CCC3181B913A4C68528F5E4B18580EAA39F9F389CC2955502DA3FEBD6C4CF
r3_experiment.py
9DDE3897437CD62506F0A6A158589CAD6B65646650555F5CCED08771398BBD89
protocol-dev.json
6E2267DB16449283C1ED8B0A575F5D136A84B35152D8D1F735E536AE0929AE45
```

## D003 median-orientation ablation and exact-moment addendum

Read `CALIBRATION_MOMENTS.md`, the new `protocol-ablation.json`, and the
prototype-0.2 diffs against frozen D002. No code/tests/experiments were run and
no additional D002 performance analysis was undertaken. Frozen D002 model and
executor hashes still match the versions cleared above. **No new D002
code/theory blocker was found.**

### Median-orientation construction: proof and code PASS

Replacing B_g by `B_m=median(X_1,...,X_N)/c_F`, with the corresponding actual
inference scale kappa_m, preserves the independent-orientation argument. The
DIR direction may depend on that same calibration median: conditional on the
full DIR/calibration field, its orientation is fixed and the independent
SHAPE law is unchanged. Integrating calibration subsequently gives the correct
B_m mixture. The median's infinite variance at N=4 does not invalidate this
distributional rank argument or p-to-e/e-BH control.

The actual 0.2 implementation matches this substitution:

- `r3_model.py:48` uses `bm*half[:,0,0]*qh/20`, with the SAME 64-block H fit
  supplying its 11-entry and eigenvalues.
- `:98` uses `km*mismatch_bound` in the held denominator, the same
  `dirs[direction][0]`, and `hs[direction]` as the NNLS metric. It does not
  use inference `hs[shape]` to choose the direction.
- `:103-111` uses the same PILOT-selected calibrators, gamma, rank grids and
  final multiplicity rule. No new random draws or changed role assignments
  are introduced by the diff.

Thus `R3_main - R3_median` isolates the inference scale-summary/reference-law
substitution within the independent-orientation architecture. It does not
isolate a particular moment as the cause of any gain. `R3_scale` retains the
old 128-block/lambda_min construction and remains a distinct component
ablation, not an additional searched family.

The declared fixed 13 x 24 reused-input comparison and global development
choice (not per-scene method selection) are consistent with this limited
question. Neither a favorable point difference nor the moment theorem is
independent confirmation of the ultimately selected primary.

### One outstanding protocol-enforcement check

`protocol-ablation.json:14` expressly requires equality of the existing main
and scale-ablation outputs to D002 on all 312 inputs. The inspected
`r3_experiment.py:66-68` checks only the older C001 R2 reference. Neither its
freeze nor its analysis binds/compares D002's outputs. The code diff makes
unchanged outputs plausible, but does not verify this declared requirement.

**Before accepting the D003 component comparison or selecting the final
primary, add a fail-closed saved-output verifier** bound to the D002 and D003
freezes/indices and exact matching task/seed/source identities. Compare the
existing primary/scale p, e, reference and decision arrays, with the unchanged
ordinary/control outputs as applicable. Do not compare version/timing fields
as if they must remain identical. Require all 312 planned tasks, not merely
successful subsets. The new median component shares the whole-family exception
handler, so a new component failure could otherwise change old outputs to zeros.
This check uses saved outputs; it requires no new observations or extra fits.
If D003 is already frozen, record the check separately without changing it.

A minor receipt-label correction for eventual packaging: the returned field
`kappa_median_learning_only` is now also the inference scale of `R3_median`.
Qualify that name's meaning or rename it in a subsequent version; it remains
learning-only for the geometric primary, not for every returned component.

### Exact geometric moments: PASS with order-domain clarification

`CALIBRATION_MOMENTS.md:7-22` is correct. For independent unit-scale Gamma
variables U~Gamma(2,1), V~Gamma(1,1), X=U/(2V), so

    E[X^t] = 2^(-t) Gamma(2+t) Gamma(1-t),  -2<t<1.

Independence of the N pivots and the normalization c_g=e/2 then give

    E[B_g^p] = exp(-p) [Gamma(2+p/N) Gamma(1-p/N)]^N,
               -2N<p<N.

The strict interval is necessary: the corresponding positive integral diverges
at and beyond either endpoint; analytic continuation of Gamma is not a moment
there. In particular, N=4 gives a finite geometric second moment, unlike the
averaged even-sample median. Log-centering is not arithmetic unbiasedness.

For clarity, state the secondary domains explicitly at `:24-30`: for **s>=0**,
`E[B_g^(-s)]` is finite exactly when s<2N; for **r>=0**, independence gives

    E[|Student20/sqrt(B_g)|^r]
      = E[|Student20|^r] E[B_g^(-r/2)],

finite exactly when r<min(20,4N). Thus the N=4 known-shape thresholds 16
(geometric) and 12 (median) are correct, with divergence at equality. Without
the nonnegative-order qualification, the blanket assertion for every r below
that upper threshold is false (sufficiently negative absolute moments diverge
at a Student variable's zero). This is a minor domain clarification, not a
defect in the intended positive-moment comparison.

These are classical finite-law moment results for the calibration factor and
the stated **known-shape** reference. They do not establish the full unknown-
shape reference's exact tail index, pointwise dominance, uniform power gain,
or novelty. Ideal-law sampling is not a universal floating-point certificate.

Reviewed snapshots:

```text
r3_model.py (prototype 0.2)
42A069DD2D386F039CF098C06CED7B07D67C4782908E50114C66DAA608223077
r3_experiment.py
11D146D782D3A65B0B53CAC40C71DAF9CDEFEFEC2DA4ACDD63B12250BAA6457C
protocol-ablation.json
1EFBDE4E136301F017149710725A7782F70F34E22BAE685091E3C01D2373EA25
CALIBRATION_MOMENTS.md
B0966F10C5DC46D17989983C7A4B3FC9DF55775C3E9C75B4BF35D8A010730924
```

## 独立解析补充审查：SCALE_INFORMATION_BOUND

2026-09-17。仅审查 `SCALE_INFORMATION_BOUND.md` 的解析论证；未读取运行中 C001 的数据、结果或进度，未运行拟合、测试、模拟或重放，未修改确认算法/协议。本节不重新开启已由 `REVIEW_CONFIRM_PREFLIGHT.md` 关闭的 D003 replay 和矩阶文字问题。

**结论：PASS。未发现信息量、CRB 或几何 log 方差公式的错误；以下范围限定应原样保留。** 这是经典信息不等式在指定 reduced experiment 下的计算，不是新 minimax 定理、完整矩阵数据的信息下界或最终 Power 下界。

### 1. 密度、score 与 Fisher 信息均正确

对 `SCALE_INFORMATION_BOUND.md:10-29`：X~F(4,2) 的密度为

    f_X(x) = 8x/(1+2x)^3, x>0.

令 Y=log X，Jacobian 给出 h(y)=8 exp(2y)/(1+2exp(y))^3。观察 log A=eta+Y；从 A 本身的密度求导也相同，因为 log 变换的 Jacobian 与 eta 无关。对 eta 求导的正负号正确：

    d/d eta log h(log A-eta) = -2 + 3U,
    U = 2(A/kappa)/(1+2(A/kappa)).

变换 X=U/[2(1-U)]、dX/dU=1/[2(1-U)^2] 得 f_U(u)=2u，0<u<1。因此 E U=2/3、E U^2=1/2；score 均值为零，二阶矩为 1/2。独立的 N 个 pivot 的交叉 score 项为零，所以 **对 eta=log kappa 而言** I_N=N/2。不要省略参数化：对 kappa 本身的信息量是 N/(2 kappa^2)，而非 N/2。

### 2. 有限 N 的 CRB 与估计量类匹配

`SCALE_INFORMATION_BOUND.md:10-14,31-34` 明确要求 T(A) 对每个 eta 都无偏、平方可积并满足微分期望交换；不是只在一个选定 eta 无偏，也不是要求 exp(T) 对 kappa 无偏。

写 S_N=sum_i s_i。固定支撑、所述 regularity 及 E_eta T=eta 给出

    E_eta[(T-eta) S_N] = 1,
    Var_eta(S_N) = N/2.

Cauchy–Schwarz 于是给出 Var_eta(T)>=2/N；在该无偏类中它也就是平方 log 误差风险的下界。没有正态近似、渐近极限或确认数据参与此推导。独立的参数无关辅助随机化也不能逃过同一不等式，但本节没有必要扩大声明的估计量类。

文档不把 CRB 当作已找到的可达有限样本方法是正确的。事实上，在这里声明的全 eta 无偏类中，有限 N 等号不能成立：若在某 eta0 等号成立，则 Cauchy–Schwarz 等号条件要求

    T-eta0 = (2/N) S_N(eta0)  a.s.

因每个 score 位于 (-2,1)，这会使 T 位于固定的 (eta0-4,eta0+2)。所有位置参数的样本密度在同一正支撑上严格为正，故该有界性对其它 eta 亦成立，不可能同时满足 E_eta T=eta 对所有实数 eta。这个简单等号检查仅解释“下界不等于有限 N 可实现改进”；它没有给出更紧的最优风险或设计新估计量。

### 3. 几何估计量的方差比较及百分比口径

`SCALE_INFORMATION_BOUND.md:36-46` 的 T_g=mean(log A)-(1-log 2) 确实对 eta 无偏。由已经审查的 log-F 矩，

    Var(T_g) = (pi^2/3-1)/N,
    Var(T_g)/(2/N) = (pi^2/3-1)/2 = 1.144934...,
    (2/N)/Var(T_g) = 2/(pi^2/3-1) = 0.87337... .

三个数值/公式相符。**14.49% 是以 CRB 为分母的超出比例**；若以当前几何方差为分母，向该下界的距离约为 12.66%。后者也只是一个不可直接兑现的上限比较，不能写成存在可将方差降低该比例的有限 N 方案。0.87337... 可称相对 Fisher 基准的渐近效率数值比较，不自动证明这里展示了一个达到效率上限的替代估计量。

这一口径提醒不构成阻塞：原文已经写明相对 LOWER BOUND，并否认相应的 Power 百分比。正文/摘要后续转述时须保留分母和限定，不能缩写为“还有 14.49% 可恢复效率”。

### 4. 允许与不允许的结论

- 允许：在仅观察 N 个独立 F(4,2) 尺度 pivot、全参数无偏/regular、平方 log 损失的实验中，有限 N 存在至少 2/N 的估计方差代价；几何估计量本身具有精确 1/N 方差。
- 不允许：将该数值推广为所有有偏估计量的逐点风险界、全 raw-matrix 实验的信息界、PC/e-value/eBH 的效率界、最小所需校准量、强参照剩余 Power 差的不可避免比例或漂移保证。
- 该界不能证明 R2 的 median 矩病态、Rayleigh 包络或 split 损失是必需的；`SCALE_INFORMATION_BOUND.md:48-59` 对此划分正确。
- 本次没有核验 CFAR2010 全文或作优先权判断。仅有出版方摘要时不能写成已核对其完整定理/假设；本信息量计算本身也不提供 R3 新颖性证据。不需要为本节另开文献路线。

审查快照：

```text
SCALE_INFORMATION_BOUND.md
063239ebd40675efc7ad61c1e41eab07376446d53f3d7aab6bef288892814f2c
```

## 信息界补充的小数更正与复核（仅追加）

2026-09-17。主任务指出上一节小数错误；本次读取修订后的 `SCALE_INFORMATION_BOUND.md`，并对常数直接作算术复核，未运行项目代码、拟合或实验，未查看或干预 C001。

正确数值为：

```text
2/(pi^2/3-1)                       = 0.8734127396111108
100*((pi^2/3-1)/2-1)               = 14.493406684822641 %
100*(1-2/(pi^2/3-1))               = 12.658726038888924 %
```

**我此前接受并重复的 `0.87337...` 有误，“三个数值/公式相符”的数值核验表述在这一项上亦有误，现以本追加更正为准。** 原公式正确；I_N=N/2、CRB=2/N、几何 log 方差、14.49%/12.66% 的四舍五入结果及百分比分母解释均不变。历史段落按用户要求原样保留。

修订文档 `SCALE_INFORMATION_BOUND.md:36-41` 新增的有限 N 等号不可能论证正确：等号迫使 T 在 eta0 下落入固定有界区间；各 eta 的样本分布在共同支撑上互相绝对连续，使该有界性对所有 eta 都成立，与全实数 eta 上无偏矛盾。这只证明所述全参数无偏 regular 类中有限 N 不取等号，未证明存在统一正间隙，也未识别最优风险。

**修订补充核验通过，无新增阻塞。** 不改变完整算法有效性、确认预审或效能结论的范围；不是全算法 Power 下界或新颖性证据。本次仅追加本审查文件。

```text
SCALE_INFORMATION_BOUND.md SHA256
f88c76e08f123135860fb527110c44fcb5415a1237a980a140ce18078790679f
```
