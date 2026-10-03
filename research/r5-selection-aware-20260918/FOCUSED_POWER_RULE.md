# D018: fixed score-saturation diagnostic, not candidate/grid expansion

2026-09-18 before D018 outputs. Single diagnostic changes h(t) from1(t>q) to
(t/q)^4 1(t>q). Same positive meta quantiles, same frozen gate/weights, same
20D015observational families and8191joint reference tuples. No new draws.

Reason: step-score h<=1 forces scalar e<=(B+1)/(1+Q); selected scores may be
even smaller. This can prevent a small set of strong discoveries from reaching
eBH while a smooth score would preserve strength above threshold. D014 pure
fourth powers instead spend mean budget on unfocused observations. The current
test retains the focus and tests exactly this tradeoff, not powers2/4/6grid.

Finite validity uses F2/F2b with any nonnegative increasing score, recomputing
the entire weighted score on every reference tuple. Borrow decreases withs,
target gate increases. Tail envelope=targetscore+borrow(S) retained. FiniteMC
validity does not require plugging in a moment or unproved J1extension. Under
actual model alpha4 also has an independent moment certificate. No novelty
claim for truncating powers or MonteCarlo normalization.

Same-information controls: target/source/count/variance bridge, fixed valid e
mixtures, fixed jointscore, classical max. Retain oldvalid methods andstep results.
One unfocused joint fourth-power score using SAME MC reference/quantile scaling
is an ablation, not D014 directly; D014 had different normalizer/randomization.
Allscore comparisons use fullPC/eBH and same PILOT, never raw power scores only.

Continuous monotone cover uses heap refinement to1% upper/lower empirical gap,
max256cells, and explicit unbounded-tail protection. Work cap not a statistical
stopping rule.20saved families fixed; no new observation/MC samples or formal
confirmation. A result failing tight/wide or stronger valid controls is not
promoted and will not be rescued by extra repetitions. Save all outcomes.
