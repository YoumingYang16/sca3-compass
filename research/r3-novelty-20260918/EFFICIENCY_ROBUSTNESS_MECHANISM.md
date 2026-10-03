# Normalization mismatch, not an unexplained heavy-tail failure

## Historical fact retained

Frozen R3-C001 (1,024families/scene) has moderate-drift FDR3.42554% for R2
versus4.84560% for R3; severe drift16.34148% versus20.00837%. The paired
R3-minus-R2 differences have original simultaneous intervals
[0.19834,2.61295]pp and[2.27084,5.00126]pp. Both are outside M0. We retain
this deterioration; a later oracle correction does not erase it.

## Mechanism identified

Let delta=kappa_target/kappa_calibration. The actual inference denominator
uses kappa_hat=kappa_calibration B, while target common-mode noise contains
kappa_target. After the model's invariant reduction the central score is
sqrt(delta)W rather than nominal W. At positive scores, delta>1 can make
rank p-values too small and null e-values too large. PC and e-BH do not
repair invalid upstream evidence. This is a transport/normalization problem.

R3's H11/spectrum reference removes R2's spectral conservatism while its
independent DIR/SHAPE architecture changes training size and direction law.
Less conservatism plausibly removes an accidental mismatch buffer. Historical
factorials support architecture as important, but do not isolate H11 alone
as the cause of every difference. In particular, increased fragility is NOT
proved inevitable whenever efficiency improves.

The radial law and invertible calibration left factors cancel under the
stated model. No matrix-calibration-shape mismatch is needed to explain these
two scenes. Gene-specific target shape, nonspherical angles, gene dependence
and pipeline-specific means remain separate untested violations.

## Controlled one-quantity intervention: NA-D002

Prespecified128of the same saved families per drift scene. For R2 and R3
equally, supply the true delta via their existing mismatch_bound argument;
only the inference denominator changes. Retain noisy finite calibration B,
nominal reference and deterministic learning. The R3 primary reference and
listed learning receipts were checked exactly. This does not claim a full
stored replay receipt for every R2 learning field. Evaluators recompute
their deterministic fits; no new candidate was fitted or selected.

| Scene | R3 old Power / FDR % | R3 corrected Power / FDR % | R2 corrected Power / FDR % |
|---|---:|---:|---:|
| delta2.2 |73.5294 /4.2662|39.8131 /0.4121|31.9393 /0.1786|
| delta4.6 |86.6422 /20.4422|35.7384 /0.3836|28.6612 /0.1722|

The severe R3 FDR reduction is20.0586pp, simultaneous interval
[12.8445,25.1692]pp. The moderate reduction3.8541pp has interval
[-2.1374,9.4123]pp, so its direction is not certified by this small diagnostic.
Corrected R3-minus-R2 Power is7.8738pp[0.8109,14.3557] and
7.0772pp[0.3299,13.2301]. Corrected FDR differences are unresolved.

Crucially, corrected R3 FDR upper endpoints are6.1091% and6.0823% under
the12-comparison95%allocation. Point estimates below1% do NOT mean this
128-family experiment independently certifies FDR<=5%. The model-level
guarantee with a valid supplied radius comes from the proof, not these means.
This is not an automatic radius estimator or an independent confirmation.

## Analytic sensitivity and information boundary

`R4_THEORY.md` proves, under the specified scalar mismatch only,

    E e_i <= max(1,delta)^min(2N,10),
    FDR <= min(1,q max(1,delta)^min(2N,10)).

Two density bounds (Student20 mixture and N F(4,2) pivots) plus the complete
PC/PILOT/grid/eBH chain yield this expression. Sole independent reviewer
accepted the proof. It is a loose classical specialization, not a sharp or
priority-established new frontier theorem. The same conservative bound also
applies to R2; it does not prove R3 necessarily less robust.

At N>=5,q=.05, delta1.01/1.02/1.05/1.10 give upper bounds
5.5231/6.0950/8.1445/12.9687%. At2.2or4.6 the bound is100% and uninformative,
NOT a prediction of100% actual FDR. `SENSITIVITY_BOUNDS.json` records values.

An externally justified Delta>=delta restores the model guarantee using
the already existing correction; otherwise replace delta by delta/Delta.
Increasing Delta, holding all learning/references fixed, weakly reduces the
rejection set but need not strictly reduce it. No unavoidable quantitative
trade-off has been proved. Without target calibration, justified controls,
repeated measurements or a credible external bound, the correction remains
conditional sensitivity analysis. No ML gate bypasses missing information.

Evidence: old `C001/summary.json`; `D002/{protocol,index,summary}.json`;
`IDENTITY_CLOSURE.json`; source `transport.py`, frozen `r3_model.py:97` and
R2 `predictive_bridge.py:115`; independent `REVIEW_FRONTIER.md`.
