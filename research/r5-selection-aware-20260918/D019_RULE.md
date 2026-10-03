# D019 fixed calibration-attribution package

Before output: existing R4 C001 cases0..9, repetitions0,1 only (20 families),
all already R5 development. No new input samples. Same seeds4095outer/4095meta,
same learned TEST/DIR/PILOT/SHAPE, same selector. Upgrade only conditional
calibration law from radius to fixed-order full triangular pivot. ALL endpoints,
count/variance bridge and fixed mixtures receive exactly the same upgrade.
Reuse D009 radial results and R4 strong comparisons, not renamed evidence.

One main (triangular selective), old A071 as reference; no hyperparameter scan.
Report per-case FDP/Power/TP/FP/counts, within-family paired differences and
failures. Two per case only diagnoses mechanism, not utility acceptance or an
independent confirmation. Include legal wideC3, smallNtC1, heavyC2, continuousC7,
smallsourceC8, partial-nullC5/C6 and OUTSIDE D case9 without pooling it.

Gate contribution = triangular selective minus triangular endpoints/fixedmix.
Calibration contribution = each triangular method minus matching D009 radial
method. Do not credit all information recovery to selective borrowing.
Exact R4 input/old evidence hashes checked. Freeze local dependencies before run.
Stop at20 whether favorable or not; no automatic sample extension. Fourworkers,
one BLAS thread, ordinary local CPU. No R5 confirmation opportunity spent.
