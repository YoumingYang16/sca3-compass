# K-NR supported scope and explicit exclusions

Current reference version K-NR-1.0.0. This document describes intended D1, not a passed confirmation. Consult RELEASE_DECISION.md when created for actual acceptance status.

| Layer | Requirement | What can be checked |
|---|---|---|
| Scientific target | Comparable prespecified effects; >=2 of4 studies share signed effect;2G family | Study metadata and investigator responsibility; arrays alone insufficient |
| Input | Finite z[256,4,6], calibration[4,n,6],n>=4,alpha.05 | Runtime dimensions, finite values, degeneracy, convergence |
| Common pipeline location | Same effect across all6 pipelines within gene/study | Declared model assumption, not certified by contrasts |
| Calibration | Independent centered null vectors, angular geometry transport to target | Provenance needed; successful fit cannot prove null/exchangeability/transport |
| Dependence | Independent genes and calibration rows; separable study/pipeline shape | Not verifiable from numeric API alone |
| Radial law | Gene-common Gaussian/inverse-gamma scale; the observed statistic mean exists | This does not require E[V] for the variance radius V; nu<=2 has no raw variance |
| Evidence | At most finite D1 simulation support | No general finite-calibration theorem |

D1 is the existing76 working scenarios (original68 plus8 small-calibration cases); definitions copied, not retuned. Original54 core equally weighted. Ordinary and t5 core both retained. Additional t1.5/t2.2, global null, singleton, mixed-sign, continuous and80% signal, negative study correlation and n4/8 calibration are present. Tested n values are4,8,16,64,256; untested intermediate settings are model extrapolation, not independently verified cases. Runtime accepting an array does not place its actual generating process inside D1.

Eight existing cases outside D1 remain visible: non-IG/lognormal radius62; modest null drift66; severe t5 drift67; pipeline loading68; gene dependence69; calibration-row dependence70; heterogeneous pipeline effects73; calibration radial mismatch75. D1 cannot be inferred or auto-certified from a caller checkbox. Case73 has genuine negative-direction alternatives despite its legacy name, so Power is not falsely reported undefined there.

No patient-level forecasts, prognosis, drug effect, ATXN3 measurement, causal mechanism or clinically validated SCA3 biomarker follows. New statistical simulations are SIMULATION_NOT_PATIENT_DATA; no external expression holdout is consumed. To use real z/calibration, the investigator must establish actual inputs and comparability; the release supplies no such biological certification.

The fixed confirmation varies data/calibration and algorithm bootstrap RNG across whole families using the declared SeedSequence rules. Its empirical FDR integrates over that experiment design. It is not conditional FDR given a realized calibration bank or an arbitrary fixed algorithm seed. A seed is for reproducibility and must not be selected after inspecting discoveries. The fixed TRAIN pattern library uses amplitudes2.5/3.5/4.5; continuous-effect stress exists, but this does not establish performance over every unseen effect distribution.

Possible statuses: COMPUTED; COMPUTED_WITH_DECLARED_FOLD_FALLBACK; UNSUPPORTED_INPUT; NUMERICAL_FAILURE; INSUFFICIENT_EVIDENCE. A computed result may still have insufficient independent evidence. Evidence/release status is separately reported, never silently changed by successful execution.
