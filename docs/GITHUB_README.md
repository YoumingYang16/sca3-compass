# SCA3 Compass

A transparent research-evidence workbench and a family-facing evidence-literacy
portal for SCA3. Research software, **not a diagnostic, prognostic or treatment service**.

## Start with the product

- **Research results:** browse all ten R4/R5 scenarios, matched comparisons,
  observed error rates, available uncertainty intervals and failure boundaries.
- **For families:** distinguish simulations, biological studies and clinical
  evidence, with plain-language explanations and original information sources.
- **Traceability:** download the complete displayed scalar records and inspect
  source hashes. No patient uploads, sign-in or application tracking.

The static portal works without biological data or a running API.

```sh
cd apps/web
pnpm install --frozen-lockfile
pnpm build:public
```

Build on PowerShell:

```powershell
python -m http.server 4173 --bind 127.0.0.1 --directory dist-public
```

Or on a POSIX shell:

```sh
python3 -m http.server 4173 --bind 127.0.0.1 --directory dist-public
```

Visit `http://127.0.0.1:4173/#research` or `#family`.
The Python server is a **local preview**, not a hardened internet deployment.
Serve only the static build, not the repository or the local FastAPI service.

## Research status — versions are not a ranking

| Version | Supported conclusion | Important boundary |
|---|---|---|
| K-NR-1.0.0 | Scoped empirical release | Not a distribution-free finite-sample theorem |
| R2-PB-1.0.0 | Model-conditional finite calibration | Proved but statistically costly |
| R3-GSR-1.0.0 | Efficiency gains in its own M0 confirmation | Mismatch failures and unresolved novelty retained |
| R4-TCB-0.2.0 | Conditional safe fixed borrowing, confirmed local gains | Legal but loose bounds can severely reduce Power |
| R5 A0.14 | Development-only selection-aware candidate | **Not accepted; no formal independent confirmation** |

R5 has **not** been established as better than all R1–R4 methods. It is not the
product's default inference engine. R4's 512 repetitions per scenario and R5
D020's two reused development families per scenario are shown separately.
Here a *family* is an entire multiple-testing batch, not a patient household.
Simulation Power is not clinical accuracy or treatment benefit.

## Evidence and reproducibility

- [Product scope and verification](docs/PRODUCT_RELEASE_2026-10-04.md)
- [Security and deployment boundary](docs/PUBLIC_DEPLOYMENT_SECURITY.md)
- [Included files and deliberate omissions](REPOSITORY_SCOPE.md)
- [Data and claims policy](docs/DATA_POLICY.md)
- [R4 results and limitations](research/r4-target-calibration-20260918/R4_FINAL_REPORT_ZH.md)
- [R5 current acceptance gates](research/r5-selection-aware-20260918/ACCEPTANCE_GATES.json)
- [R5 paired development summary](research/r5-selection-aware-20260918/D020/summary.json)

The curated repository includes actual implementation, proofs, selected completed
freeze manifests, summaries and all 20 derived D020 scalar records. Large raw
simulation arrays, biological data, protected validation data, environments and
runtime databases remain excluded. Historical full-replay commands require the
owner's retained archives and may enforce the original Windows path. This is
**not** a portable replacement for the complete research archive.

## Sharing and rights

Initial upload is private. No open-source license has been selected; uploading
does not grant blanket redistribution rights. Dependencies retain their own
licenses. No guarantee of clinical usefulness, international novelty, journal
acceptance or outside peer-review approval is made. **External review has not
been conducted.** Source upload is separate from public website deployment.
