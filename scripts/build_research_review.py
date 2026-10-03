"""Render figures and a machine-checkable review bundle from immutable results.

This is a reporting program, not a new analysis or selection on held-out data.
No source expression, nomination, frozen method or test outcome is changed.
"""
from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from sca3_compass.molecular_data import PROJECT_ROOT, digest, write_json

ROOT = PROJECT_ROOT
OUT = ROOT / "artifacts/research-review"
FIG = OUT / "figures"
BLUE = "#235b75"
OCHRE = "#a57440"
INK = "#1a303b"
NAMES = {
    "bonferroni_PC_BY": "Bonferroni + PC + BY",
    "fixed_PC_BY": "Prespecified pipeline",
    "oracle_maxT_PC_BY": "Oracle max-T",
    "plugin_maxT_PC_BY": "Plug-in max-T",
    "envelope_maxT_PC_BY": "Confidence-envelope max-T",
    "rank_calibration_PC_BY": "Empirical rank calibration",
    "ePCH_eBH": "e-PCH + e-BH",
    "eFilter_bonferroni_split_signs": "e-Filter (empirical comparator)",
    "single_study_z": "Single-study ranking",
    "multiregion_EB": "Multiregion empirical Bayes",
    "two_study_consensus": "Two-study consensus",
    "multiverse_consensus": "Multiverse consensus",
    "zero_effect": "Zero effect",
    "unshrunk": "Unshrunk",
    "identity_mixture": "Identity mixture",
    "structured_mixture": "Structured mixture",
}


def read(name: str) -> dict:
    return json.loads((ROOT / "artifacts" / f"{name}.json").read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=200, facecolor="white", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def methods(case: dict) -> dict:
    return {r["method"]: r for r in case["rows"]}


def calibration_figures(report: dict) -> None:
    cases = {s["scenario"]: s for s in report["scenarios"]}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), layout="constrained")
    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
    for ax, effect in zip(axes, [2.5, 3.5, 4.5], strict=True):
        data = np.array([[100 * methods(cases[f"n{n}_rho{rho}_z{effect}"])["envelope_maxT_PC_BY"]
                          ["paired_power_vs_bonferroni"]["mean"] for rho in [.1, .8, .95]] for n in [16, 64, 256]])
        plot = ax.imshow(data, norm=norm, cmap="PuOr", aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{data[i,j]:+.2f}", ha="center", va="center", color=INK,
                        bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 3})
        ax.set_xticks(range(3), ["0.10", "0.80", "0.95"])
        ax.set_yticks(range(3), ["16", "64", "256"])
        ax.set_xlabel("Pipeline correlation")
        ax.set_ylabel("Calibration samples")
        ax.set_title(f"Signal strength = {effect}", loc="left", fontweight="bold")
    fig.colorbar(plot, ax=axes, shrink=.75, label="Power gain vs Bonferroni (percentage points)")
    fig.suptitle("Finite calibration: gains are regime-dependent", fontsize=17, fontweight="bold", x=.01, ha="left")
    fig.supxlabel("27 prespecified Gaussian scenarios; 600 independent-seed repetitions each. Descriptive differences, not simultaneous significance.", fontsize=9)
    save(fig, "01-calibration-regime-map")

    order = ["n256_rho0.95_z3.5", "sparse_signals", "heavy_tail_both", "dependent_calibration_rows"]
    titles = ["High correlation / n = 256", "Sparse alternatives (2%)", "Heavy tails: Gaussian assumption fails", "Dependent calibration rows"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), layout="constrained")
    for ax, scenario, title in zip(axes.flat, order, titles, strict=True):
        case = cases[scenario]
        rows = case["rows"]
        y = np.arange(len(rows))
        means = np.array([r["fdr"]["mean"]*100 for r in rows])
        bounds = np.array([r["fdr"]["ci95"] for r in rows])*100
        for j, row in enumerate(rows):
            color = BLUE if row["method"] == "envelope_maxT_PC_BY" else "#748087"
            ax.errorbar(means[j], y[j], xerr=[[means[j]-bounds[j,0]], [bounds[j,1]-means[j]]],
                        fmt="o", color=color, capsize=2, markersize=4)
        ax.axvline(5, color=OCHRE, linestyle="--", linewidth=1.2, label="Nominal 5%")
        ax.set_yticks(y, [NAMES[r["method"]] for r in rows], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(7, bounds[:,1].max()*1.05))
        ax.set_xlabel("Mean FDP (%) / pointwise 95% interval")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=.18)
    fig.suptitle("Operating characteristics include failures", x=.01, ha="left", fontsize=17, fontweight="bold")
    fig.supxlabel("600 repetitions per scenario. Bounded empirical-Bernstein intervals; not joint across methods/scenarios.\nStress panels outside assumptions do not carry the candidate's theoretical guarantee.", fontsize=9)
    save(fig, "02-fdr-stress-tests")


def transport_figure(report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True, layout="constrained")
    for ax, row in zip(axes.flat, report["strategy_results"], strict=True):
        for group, genotype in enumerate(["WT", "Q84"]):
            y = [s["score"] for s in row["sample_scores"] if s["genotype"] == genotype]
            ax.scatter(group + np.linspace(-.09, .09, len(y)), y, s=36, color="#8c9998" if group == 0 else BLUE)
            ax.hlines(np.mean(y), group-.20, group+.20, color=INK, linewidth=1.4)
        ax.axhline(0, color="#aab3b8", linestyle=":", linewidth=1)
        ax.set_xticks([0, 1], ["WT (n = 6)", "Q84 (n = 6)"])
        ax.set_xlim(-.5, 1.5)
        ax.set_title(NAMES[row["strategy"]], loc="left", fontweight="bold", fontsize=12)
        ax.set_ylabel("Frozen signed signature score")
        ax.text(.02, .96, f"{row['measured']}/50 genes measured; direction agreement {100*row['direction_concordance']:.1f}%\n"
                f"Exact p = 1/924; Holm p = {row['holm_p_four_strategies']:.5f}", transform=ax.transAxes, va="top", fontsize=8)
        ax.set_ylim(-1.6, 1.6)
    fig.suptitle("Frozen nominations in one external public cohort", fontsize=17, fontweight="bold", x=.01, ha="left")
    fig.supxlabel("GSE261670 / untreated 65-week cerebellum. Each point is a real public sample; jitter is display-only.\nAll four signatures pass; this does not establish strategy superiority or clinical validity.", fontsize=9)
    save(fig, "03-frozen-public-cohort")


def block_figure(report: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), layout="constrained")
    rows = report["summaries"]
    y = np.arange(4)
    means = [r["mean_fold_mse"] for r in rows]
    axes[0].barh(y, means, height=.55, color=["#9ba6ab", "#b59d83", "#648291", BLUE])
    axes[0].set_yticks(y, [NAMES[r["method"]] for r in rows])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, .18)
    for i, value in enumerate(means):
        axes[0].text(value+.003, i, f"{value:.5f}", va="center", fontsize=9)
    axes[0].set_xlabel("Mean squared error (noisy held-out contrast)")
    axes[0].set_title("All 48 whole-animal holdouts", loc="left", fontweight="bold")
    x = np.array([r["losses"]["identity_mixture"] for r in report["folds"]])
    z = np.array([r["losses"]["structured_mixture"] for r in report["folds"]])
    low, high = min(x.min(),z.min())*.95, max(x.max(),z.max())*1.05
    axes[1].plot([low, high], [low, high], linestyle="--", color="#b09c87", linewidth=1)
    axes[1].scatter(x, z, color=BLUE, alpha=.65, s=22)
    axes[1].set_xlabel("Identity-mixture MSE")
    axes[1].set_ylabel("Structured-mixture MSE")
    axes[1].set_title("Structure vs simple shrinkage", loc="left", fontweight="bold")
    axes[1].set_xlim(low, high)
    axes[1].set_ylim(low, high)
    fig.suptitle("Modest additional gain from covariance structure", fontsize=17, fontweight="bold", x=.01, ha="left")
    fig.supxlabel("GSE107958 / 6 disease and 8 control animals. All regions held out together. 96/96 mixture fits converged.\nOverlapping folds are not independent experiments: no iid-fold confidence intervals or significance claims.", fontsize=9)
    save(fig, "04-whole-animal-validation")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                        "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                        "svg.fonttype": "none", "savefig.transparent": False})
    release = read("molecular-release-validation")
    if not release["passed"]:
        raise SystemExit("Integrity checks have not passed; review bundle not generated")
    for name, sha in release["artifacts_sha256"].items():
        if digest(ROOT/"artifacts"/name) != sha:
            raise SystemExit(f"Artifact changed after release validation: {name}")
    for name, sha in release["sources_sha256"].items():
        if digest(ROOT/"src/sca3_compass"/name) != sha:
            raise SystemExit(f"Source changed after release validation: {name}")
    software = read("software-verification")
    if not software["passed"] or any(digest(ROOT/path) != sha for path, sha in software["file_sha256"].items()):
        raise SystemExit("Software verification receipt is stale or failing")
    envelope, transport, blocks = (read(n) for n in ["molecular-envelope-validation", "molecular-transport", "molecular-block-validation"])
    calibration_figures(envelope)
    transport_figure(transport)
    block_figure(blocks)
    flat = []
    for c in envelope["scenarios"]:
        for row in c["rows"]:
            flat.append({"scenario": c["scenario"], "assumptions_met": c["candidate_assumptions"],
                         **{k: v for k,v in row.items() if not k.endswith("_by_repetition")}})
    write_json(OUT / "all-method-comparisons.json", {"rows": flat, "source_sha256": digest(ROOT/"artifacts/molecular-envelope-validation.json")})
    distributions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    write_json(OUT / "environment.json", {"created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(), "packages": dict(sorted(distributions.items())),
        "r": read("molecular-efilter-conformance")["r_version"]})
    # This is generated metadata, not an unbounded directory scan or data export.
    selected = [ROOT / "artifacts" / name for name in release["artifacts_sha256"]]
    selected += [ROOT / "artifacts/molecular-release-validation.json", ROOT / "artifacts/software-verification.json"]
    selected += sorted((ROOT/"src/sca3_compass").glob("molecular*.py"))
    selected += sorted((ROOT/"configs").glob("molecular*.json"))
    selected += [ROOT/"scripts"/n for n in ["build_research_review.py", "research-review.ps1", "check_efilter_conformance.py",
        "efilter_conformance.R", "check_transport_reanalysis.py", "transport_crosscheck.R", "transport_gene_crosscheck.R",
        "verify.ps1", "write_verification_receipt.py"]]
    selected += [ROOT/"docs"/n for n in ["RESEARCH_REVIEW_2026-09-15.md", "RESEARCH_REPRODUCIBILITY.md",
        "CONFIDENCE_ENVELOPE_DERIVATION.md", "TRANSPORT_ACCESS_AMENDMENT.md", "RESEARCH_UI_QA.md"]]
    selected += [ROOT/"pyproject.toml"]
    selected += sorted(FIG.glob("*")) + [OUT/"environment.json", OUT/"all-method-comparisons.json"]
    manifest = {"created_at": datetime.now(UTC).isoformat(), "kind": "PRE_MANUSCRIPT_RESEARCH_REVIEW",
        "scope": "Local deliverable integrity, not an independent audit or publication certification",
        "files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p), "bytes": p.stat().st_size} for p in selected]}
    write_json(OUT/"manifest.json", manifest)
    print(json.dumps({"figures": 4, "formats": ["png", "svg"], "manifest_files": len(selected),
                      "integrity_checks": len(release["checks"]), "output": str(OUT)}))


if __name__ == "__main__":
    main()
