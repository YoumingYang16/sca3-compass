# R3 finite-calibration efficiency research package

**Confirmation, analysis and reproduction completed.** Core M0 gain+6.894pp
[5.461,8.201], all10nonnull paired gains supported and all13M0 FDR uppers<=5%.
Both default mismatch scenes worsen; novelty remains unestablished. Full user
PROVED_AND_USEFUL is NOT met. Read RESEARCH_FINDINGS_ZH.md and FINAL_STATUS.json
for the bounded closure decision. The sole independent review is integrated;
REVIEW_FINAL.md preserves findings and their closure. DELIVERY_MANIFEST.json
is the final sealing receipt (absence means sealing has not completed).
K-NR-1.0.0 and all R2 releases are untouched. No further scientific job or R4.

Primary code: C001/r3_model.py, outputR3_main, versionR3-GSR-1.0.0.
Scope: independent common-location4study/6pipeline matrices, independent
centered finite calibration, separable shape/common-block radial model M0.
Research software, not a clinical device or an automatically valid real-data
pipeline. `R3_ALGORITHM.md` states assumptions and the two-of-four signed null.

The finite-calibration guarantee averages over calibration sampling and
auxiliary randomization; it is NOT an FDR bound conditional on every fixed
observed calibration set or reference bank. Input-array validation checks
dimensions/numerics, not independence, centering or matching nuisance structure.

## Read in this order

1. RESEARCH_FINDINGS_ZH.md / R3_RESULTS.md (completed actual numerical results).
2. EFFICIENCY_LOSS_LEDGER.md and VALIDITY_EFFICIENCY_FRONTIER.md.
3. R3_THEORY.md, CALIBRATION_MOMENTS.md, SCALE_INFORMATION_BOUND.md.
4. R3_ALGORITHM.md, CONTRIBUTION_MAP.md, R3_CLAIM_EVIDENCE.md.
5. R3_PAPER_CORE.md and independent REVIEW_* files.

## Environment and dependencies

Actual project root:
`C:/Users/ROG/Documents/Codex/2026-09-11/plugin-browser-openai-bundled-x20/sca3-compass`.
Commands below run from that root in PowerShell. CPython3.12.14 and the existing
`.venv` were reused. Exact package versions are in requirements-r3.txt. No
cloud service, new dependency installation, GPU or fee was used for R3.

This is a project-local reproducible package, not a dependency-free standalone
wheel: it imports frozen research/r2-finite-20260917/C001 and the V1 source in
releases/K-NR-1.0.0. Preserve those directories and their manifests when moving
the project. Startup checks actual imported module paths and frozen hashes.
Do not replace them with the live development source under src.

Portability limitation: inherited frozen R2 provenance also checks absolute
V1 import paths. These commands were reproduced at the original Windows path
above. Moving to a different root/host requires a separately audited relocation
entrypoint/manifest, not editing the old frozen receipts to suppress the checks.
Preserving only the relative folder layout is NOT sufficient for that checker.

## Actual freeze/start commands (executed)

```powershell
.venv/Scripts/python.exe research/r3-efficiency-20260917/r3_experiment.py freeze --out research/r3-efficiency-20260917/C001 --protocol research/r3-efficiency-20260917/protocol-confirm.json
.venv/Scripts/python.exe research/r3-efficiency-20260917/C001/r3_experiment.py run --out research/r3-efficiency-20260917/C001
```

Freeze executed17:48:04HKT on2026-09-17; owner receipt C001/started.json,
PID97420 at launch. Hashes in C001/freeze.json. **Do not execute these again
against the existing output directory.** They refuse to overwrite artifacts.
This is ONE formal confirmation, not a source of fresh attempts after a failure.

## Completion and reproduction commands

The following commands were ACTUALLY executed successfully after the complete
index existed. Analysis audited all15360saved records before computing results.
Do not rerun exclusive writers into existing files; use the fresh-destination
commands further below to reproduce without overwriting the evidence.

```powershell
.venv/Scripts/python.exe -B research/r3-efficiency-20260917/C001/analyze_confirm.py --out research/r3-efficiency-20260917/C001
.venv/Scripts/python.exe -B research/r3-efficiency-20260917/reproduce.py
.venv/Scripts/python.exe -B research/r3-efficiency-20260917/run_example.py --output research/r3-efficiency-20260917/EXAMPLE_RECEIPT.json
.venv/Scripts/python.exe -B research/r3-efficiency-20260917/make_figures.py
.venv/Scripts/python.exe -B research/r3-efficiency-20260917/implementation_diagnostics.py
.venv/Scripts/python.exe -B research/r3-efficiency-20260917/recompute_summary.py --report research/r3-efficiency-20260917/RECOMPUTED_SUMMARY.json
```

These completion invocations used Python's `-B` flag to suppress bytecode.
Actual15-test command (passed in10.28s; use a NEW basetemp/XML destination
for another run, do not replace the original receipt):

```powershell
.venv/Scripts/python.exe -B -m pytest research/r3-efficiency-20260917/C001/test_r3.py research/r3-efficiency-20260917/C001/test_intervals.py research/r3-efficiency-20260917/C001/test_execution.py --basetemp=research/r3-efficiency-20260917/test-tmp-frozen-final-1949 --junitxml=research/r3-efficiency-20260917/frozen-tests.xml -q
```

Actual receipts: REPRODUCTION_AUDIT.json(5savedfamilies,109exactarrays),
RECOMPUTED_SUMMARY.json(byte-identical to originalsummary),frozen-tests.xml,
IMPLEMENTATION_DIAGNOSTICS.json(61440fold receipts,all3R3roles converged),
EXAMPLE_RECEIPT.json(completed,0discoveries,1.993s including ablations).
The zero-discovery example was kept unchanged; it is not a patient finding.
Three rendered PNGs were visually checked for readable labels/captions and
complete scenario display; figures/manifest.json binds them to the summary.

Original output writers are exclusive. For a subsequent raw-record re-analysis
or another set of plots, use NEW output locations:

```powershell
.venv/Scripts/python.exe -B research/r3-efficiency-20260917/recompute_summary.py --report research/r3-efficiency-20260917/recomputed-local.json
.venv/Scripts/python.exe -B research/r3-efficiency-20260917/make_figures.py --out research/r3-efficiency-20260917/figures-local
.venv/Scripts/python.exe -B research/r3-efficiency-20260917/run_example.py
```

`recompute_summary.py` loads the exact frozen analyzer and redirects ONLY its
exclusive final write, rejecting any unexpected destination; verifies byte
equality to the original summary. No new observations/fits or metric changes.
`reproduce.py` uses the five preregistered saved inputs with original seeds;
reproduction is not more independent n. Its receipt also uses exclusive output.
The default callable example uses a saved **simulated** family and a fixed
example seed; its discovery count is not a validated molecular discovery count.

## Recovery, only after a genuine interruption

Inspect actual owner and worker processes before any restart. Existing
`running` labels alone are not proof a job is alive; conversely a missing
owner is not proof its workers have stopped. Do not kill unrelated processes.
If all task-owned processes are gone and the final index is absent:

```powershell
.venv/Scripts/python.exe research/r3-efficiency-20260917/C001/r3_experiment.py resume --out research/r3-efficiency-20260917/C001
```

Resume validates and retains completed attempts including failures; does not
reroll seeds. Orphaned partial arrays are an explicit audit blocker, not license
to delete files or re-run silently. The3h result timeout is job safety, not a
scientific stop or a guaranteed automatic process-tree kill. No scheduled
background continuation is promised outside an actually live process.

The current C001 is COMPLETE, so the recovery command is NOT applicable now.
The final local seal uses `.venv/Scripts/python.exe -B research/r3-efficiency-20260917/seal.py`;
it checks required receipts and frozen identities, then writes an exclusive
manifest. Do not rerun it over an existing manifest or edit sealed evidence.
