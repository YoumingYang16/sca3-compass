"""Gene-level, donor-pseudobulk analysis of the public GSE309548 RNA-seq data.

The workflow deliberately keeps the inferential unit at the donor level.  Regional
files are combined into donor pseudobulks, transcript estimates are mapped with a
versioned GENCODE annotation, and disease effects are estimated with a
covariate-adjusted, empirical-Bayes moderated linear model.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.special import digamma, polygamma

from .repository import PROJECT_ROOT
from .transcriptomics import (
    ARCHIVE_URL,
    benjamini_hochberg,
    download_archive,
    parse_geo_samples,
    sha256_file,
)

GENCODE_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/"
    "gencode.v50.annotation.gtf.gz"
)


def parse_gtf_attributes(value: str) -> dict[str, str]:
    """Parse the semicolon-separated GTF attribute field."""
    attributes: dict[str, str] = {}
    for item in value.rstrip(";\n").split(";"):
        item = item.strip()
        if not item or " " not in item:
            continue
        key, raw = item.split(" ", 1)
        attributes[key] = raw.strip().strip('"')
    return attributes


def load_gencode_transcript_map(gtf_path: Path) -> pd.DataFrame:
    """Return one version-stripped transcript-to-gene mapping from GENCODE."""
    rows: list[tuple[str, str, str, str]] = []
    with gzip.open(gtf_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 8)
            if len(fields) != 9 or fields[2] != "transcript":
                continue
            attributes = parse_gtf_attributes(fields[8])
            transcript_id = attributes.get("transcript_id", "").split(".", 1)[0]
            gene_id = attributes.get("gene_id", "").split(".", 1)[0]
            if not transcript_id or not gene_id:
                continue
            rows.append(
                (
                    transcript_id,
                    gene_id,
                    attributes.get("gene_name", gene_id),
                    attributes.get("gene_type", "unknown"),
                )
            )
    mapping = pd.DataFrame(
        rows,
        columns=["transcript_id", "gene_id", "gene_name", "gene_type"],
    ).drop_duplicates("transcript_id", keep="first")
    if mapping.empty:
        raise ValueError("No transcript mappings were found in the GENCODE GTF")
    return mapping.set_index("transcript_id")


def read_estimated_count_matrix(
    archive_path: Path, samples: list[dict[str, str]]
) -> pd.DataFrame:
    """Read author-provided Kallisto estimated counts from the GEO tar archive."""
    columns: dict[str, pd.Series] = {}
    accessions = {sample["accession"] for sample in samples}
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            accession = member.name.split("_", 1)[0]
            if accession not in accessions or not member.name.endswith(".csv.gz"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not extract {member.name}")
            with gzip.GzipFile(fileobj=extracted) as compressed:
                frame = pd.read_csv(
                    compressed,
                    usecols=["target_id", "est_counts"],
                    dtype={"target_id": "string", "est_counts": "float64"},
                )
            frame["target_id"] = frame["target_id"].str.replace(
                r"\..*$", "", regex=True
            )
            if frame["target_id"].duplicated().any():
                frame = frame.groupby("target_id", as_index=False)["est_counts"].sum()
            columns[accession] = frame.set_index("target_id")["est_counts"]
    missing = sorted(accessions - set(columns))
    if missing:
        raise ValueError(f"Archive is missing sample files: {missing}")
    matrix = pd.concat(columns, axis=1, join="inner").fillna(0.0)
    if matrix.empty:
        raise ValueError("No shared transcripts were found")
    return matrix


def collapse_transcripts_to_genes(
    transcript_counts: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Aggregate mapped transcript estimates to GENCODE genes."""
    joined = transcript_counts.join(mapping, how="left")
    mapped = joined[joined["gene_id"].notna()].copy()
    if mapped.empty:
        raise ValueError("None of the quantified transcripts map to the annotation")
    sample_columns = list(transcript_counts.columns)
    gene_counts = mapped.groupby("gene_id", sort=True)[sample_columns].sum()
    gene_info = (
        mapped.reset_index()
        .groupby("gene_id", sort=True)[["gene_name", "gene_type"]]
        .first()
    )
    total_abundance = float(transcript_counts.to_numpy().sum())
    mapped_abundance = float(mapped[sample_columns].to_numpy().sum())
    diagnostics: dict[str, float | int] = {
        "quantified_transcripts": int(transcript_counts.shape[0]),
        "mapped_transcripts": int(mapped.index.nunique()),
        "mapped_genes": int(gene_counts.shape[0]),
        "transcript_mapping_rate": round(
            float(mapped.index.nunique() / transcript_counts.shape[0]), 6
        ),
        "abundance_mapping_rate": round(
            mapped_abundance / total_abundance if total_abundance else 0.0, 6
        ),
    }
    return gene_counts, gene_info, diagnostics


def _estimate_variance_prior(
    residual_variance: np.ndarray, residual_df: int
) -> tuple[float, float]:
    """Method-of-moments scaled inverse-chi-square variance prior."""
    values = np.clip(np.asarray(residual_variance, dtype=float), 1e-12, None)
    log_values = np.log(values)
    low, high = np.quantile(log_values, [0.05, 0.95])
    robust_log_values = np.clip(log_values, low, high)
    target = float(
        np.var(robust_log_values, ddof=1) - polygamma(1, residual_df / 2)
    )
    if target <= 1e-8:
        prior_df = 1_000_000.0
    else:
        prior_df = float(
            brentq(
                lambda value: float(polygamma(1, value / 2) - target),
                0.1,
                1_000_000.0,
            )
        )
    expected_log_f = (
        digamma(residual_df / 2)
        - np.log(residual_df / 2)
        - digamma(prior_df / 2)
        + np.log(prior_df / 2)
    )
    prior_variance = float(np.exp(np.mean(robust_log_values) - expected_log_f))
    return prior_df, prior_variance


def fit_moderated_model(
    expression: np.ndarray,
    design: np.ndarray,
    coefficient_index: int,
) -> dict[str, np.ndarray | float | int]:
    """Fit all genes jointly and moderate residual variances across genes."""
    design = np.asarray(design, dtype=float)
    expression = np.asarray(expression, dtype=float)
    rank = int(np.linalg.matrix_rank(design))
    residual_df = int(design.shape[0] - rank)
    if residual_df < 2:
        raise ValueError("At least two residual degrees of freedom are required")
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = inverse @ design.T @ expression
    residuals = expression - design @ coefficients
    residual_variance = np.sum(residuals**2, axis=0) / residual_df
    prior_df, prior_variance = _estimate_variance_prior(
        residual_variance, residual_df
    )
    posterior_variance = (
        prior_df * prior_variance + residual_df * residual_variance
    ) / (prior_df + residual_df)
    coefficient_variance = float(inverse[coefficient_index, coefficient_index])
    standard_error = np.sqrt(posterior_variance * coefficient_variance)
    statistic = coefficients[coefficient_index] / standard_error
    total_df = prior_df + residual_df
    p_value = 2 * stats.t.sf(np.abs(statistic), df=total_df)
    return {
        "coefficient": coefficients[coefficient_index],
        "standard_error": standard_error,
        "statistic": statistic,
        "p_value": p_value,
        "q_value": benjamini_hochberg(p_value),
        "residual_df": residual_df,
        "prior_df": prior_df,
        "prior_variance": prior_variance,
        "design_rank": rank,
        "design_condition_number": float(np.linalg.cond(design)),
    }


def _design_matrix(
    metadata: pd.DataFrame, include_age: bool = False
) -> tuple[np.ndarray, list[str]]:
    condition = (metadata["genotype"].str.upper() == "SCA3").astype(float).to_numpy()
    male = (metadata["sex"].str.upper() == "M").astype(float).to_numpy()
    columns = [np.ones(len(metadata)), condition, male]
    names = ["intercept", "SCA3", "male"]
    if include_age:
        age = pd.to_numeric(metadata["time"], errors="raise").to_numpy(dtype=float)
        age = (age - age.mean()) / age.std(ddof=0)
        columns.append(age)
        names.append("age_z")
    return np.column_stack(columns), names


def _pca(expression: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    variable = expression.var(axis=1).nlargest(min(1000, len(expression))).index
    values = expression.loc[variable].T.to_numpy()
    values -= values.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(values, full_matrices=False)
    coordinates = u[:, :2] * singular[:2]
    variance = singular**2
    ratio = variance / variance.sum() if variance.sum() else variance
    frame = pd.DataFrame(
        {
            "donor_id": expression.columns,
            "pc1": coordinates[:, 0],
            "pc2": coordinates[:, 1],
        }
    )
    return frame, [round(float(value * 100), 2) for value in ratio[:2]]


def _leave_one_out_stability(
    expression: np.ndarray,
    design: np.ndarray,
    full_effect: np.ndarray,
    gene_indices: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for gene_index in gene_indices:
        estimates: list[float] = []
        for omitted in range(design.shape[0]):
            keep = np.arange(design.shape[0]) != omitted
            reduced_design = design[keep]
            if np.linalg.matrix_rank(reduced_design) < design.shape[1]:
                continue
            beta = np.linalg.pinv(reduced_design) @ expression[keep, gene_index]
            estimates.append(float(beta[1]))
        direction = np.sign(full_effect[gene_index])
        agreement = (
            np.mean(np.sign(estimates) == direction) if estimates else float("nan")
        )
        rows.append(
            {
                "gene_position": int(gene_index),
                "refits": len(estimates),
                "direction_agreement": round(float(agreement), 6),
                "minimum_effect": round(float(min(estimates)), 6),
                "maximum_effect": round(float(max(estimates)), 6),
            }
        )
    return rows


def _assignment_calibration(
    expression: np.ndarray,
    metadata: pd.DataFrame,
    observed_max_statistic: float,
) -> dict[str, Any]:
    """Enumerate all 4-vs-4 label assignments as a small-sample calibration."""
    sex = (metadata["sex"].str.upper() == "M").astype(float).to_numpy()
    discovery_counts: list[int] = []
    maximum_statistics: list[float] = []
    valid_assignments = 0
    group_size = int((metadata["genotype"].str.upper() == "SCA3").sum())
    for selected in itertools.combinations(range(len(metadata)), group_size):
        label = np.zeros(len(metadata))
        label[list(selected)] = 1.0
        design = np.column_stack([np.ones(len(metadata)), label, sex])
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        fit = fit_moderated_model(expression, design, 1)
        discovery_counts.append(int(np.sum(np.asarray(fit["q_value"]) <= 0.05)))
        maximum_statistics.append(float(np.max(np.abs(fit["statistic"]))))
        valid_assignments += 1
    maxima = np.asarray(maximum_statistics)
    return {
        "enumerated_assignments": valid_assignments,
        "median_fdr_discoveries": float(np.median(discovery_counts)),
        "maximum_fdr_discoveries": int(max(discovery_counts)),
        "max_statistic_tail_probability": round(
            float(np.mean(maxima >= observed_max_statistic)), 6
        ),
        "null_max_statistic_p95": round(float(np.quantile(maxima, 0.95)), 6),
        "interpretation": (
            "All fixed-size disease-label assignments were enumerated as a diagnostic. "
            "Because this is observational tissue data, the result is calibration rather "
            "than a randomized-experiment permutation p-value."
        ),
    }


def analyze_gene_expression(
    gene_counts: pd.DataFrame,
    gene_info: pd.DataFrame,
    samples: list[dict[str, str]],
    minimum_cpm: float = 1.0,
    minimum_donors: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run donor pseudobulk normalization, moderated models and diagnostics."""
    metadata = pd.DataFrame(samples).set_index("accession").loc[gene_counts.columns]
    donor_counts = gene_counts.T.groupby(metadata["donor_id"], sort=True).sum().T
    donor_metadata = metadata.groupby("donor_id", sort=True).first()
    library_sizes = donor_counts.sum(axis=0)
    cpm = donor_counts.divide(library_sizes, axis=1) * 1_000_000
    keep = (cpm >= minimum_cpm).sum(axis=1) >= minimum_donors
    filtered_cpm = cpm.loc[keep]
    expression = np.log2(
        donor_counts.loc[keep].add(0.5).divide(library_sizes + 1.0, axis=1)
        * 1_000_000
    )

    design, design_names = _design_matrix(donor_metadata, include_age=False)
    primary = fit_moderated_model(expression.T.to_numpy(), design, 1)
    age_design, age_design_names = _design_matrix(donor_metadata, include_age=True)
    age_adjusted = fit_moderated_model(expression.T.to_numpy(), age_design, 1)

    sca3 = donor_metadata.index[donor_metadata["genotype"].str.upper() == "SCA3"]
    control = donor_metadata.index[donor_metadata["genotype"].str.upper() == "CTRL"]
    results = gene_info.loc[expression.index].copy()
    results.index.name = "gene_id"
    results["mean_sca3_cpm"] = filtered_cpm.loc[:, sca3].mean(axis=1)
    results["mean_control_cpm"] = filtered_cpm.loc[:, control].mean(axis=1)
    results["average_log2_cpm"] = expression.mean(axis=1)
    results["log2_fold_change"] = primary["coefficient"]
    results["moderated_t"] = primary["statistic"]
    results["p_value"] = primary["p_value"]
    results["q_value"] = primary["q_value"]
    results["age_adjusted_log2_fold_change"] = age_adjusted["coefficient"]
    results["age_adjusted_p_value"] = age_adjusted["p_value"]
    results["age_adjusted_q_value"] = age_adjusted["q_value"]
    results["direction_concordant"] = (
        np.sign(results["log2_fold_change"])
        == np.sign(results["age_adjusted_log2_fold_change"])
    )
    results = results.reset_index().sort_values(
        ["q_value", "p_value", "log2_fold_change"],
        ascending=[True, True, False],
    )

    pca, pca_variance = _pca(expression)
    pca = pca.set_index("donor_id").join(
        donor_metadata[["genotype", "sex", "time"]]
    ).reset_index()

    top_positions = results.head(20).index.to_numpy()
    # Results were sorted, so map their gene ids back to expression column positions.
    gene_position = {gene: index for index, gene in enumerate(expression.index)}
    selected_positions = np.array(
        [gene_position[gene] for gene in results.head(20)["gene_id"]], dtype=int
    )
    stability = _leave_one_out_stability(
        expression.T.to_numpy(),
        design,
        np.asarray(primary["coefficient"]),
        selected_positions,
    )
    stability_by_position = {
        int(item["gene_position"]): item for item in stability
    }
    results["leave_one_out_direction_agreement"] = np.nan
    results["leave_one_out_minimum_effect"] = np.nan
    results["leave_one_out_maximum_effect"] = np.nan
    for row_index, gene in zip(top_positions, results.head(20)["gene_id"], strict=True):
        item = stability_by_position[gene_position[gene]]
        results.loc[row_index, "leave_one_out_direction_agreement"] = item[
            "direction_agreement"
        ]
        results.loc[row_index, "leave_one_out_minimum_effect"] = item[
            "minimum_effect"
        ]
        results.loc[row_index, "leave_one_out_maximum_effect"] = item[
            "maximum_effect"
        ]

    observed_max = float(np.max(np.abs(np.asarray(primary["statistic"]))))
    metrics: dict[str, Any] = {
        "sample_files": int(gene_counts.shape[1]),
        "donors": int(donor_counts.shape[1]),
        "sca3_donors": len(sca3),
        "control_donors": len(control),
        "genes_before_filter": int(gene_counts.shape[0]),
        "genes_after_filter": int(expression.shape[0]),
        "fdr_005": int((results["q_value"] <= 0.05).sum()),
        "age_adjusted_fdr_005": int(
            (results["age_adjusted_q_value"] <= 0.05).sum()
        ),
        "pca_variance_percent": pca_variance,
        "primary_model": {
            "design": design_names,
            "rank": int(primary["design_rank"]),
            "residual_df": int(primary["residual_df"]),
            "condition_number": round(float(primary["design_condition_number"]), 6),
            "empirical_bayes_prior_df": round(float(primary["prior_df"]), 6),
            "empirical_bayes_prior_variance": round(
                float(primary["prior_variance"]), 6
            ),
        },
        "sensitivity_model": {
            "design": age_design_names,
            "rank": int(age_adjusted["design_rank"]),
            "residual_df": int(age_adjusted["residual_df"]),
            "condition_number": round(
                float(age_adjusted["design_condition_number"]), 6
            ),
        },
        "assignment_calibration": _assignment_calibration(
            expression.T.to_numpy(), donor_metadata, observed_max
        ),
    }
    return results, pca, metrics


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.replace({np.nan: None})
    records: list[dict[str, Any]] = []
    for row in cleaned.to_dict(orient="records"):
        records.append(
            {
                key: round(float(value), 8) if isinstance(value, float) else value
                for key, value in row.items()
            }
        )
    return records


def run(
    archive_path: Path,
    soft_path: Path,
    gtf_path: Path,
    output_json: Path,
    output_csv: Path,
) -> dict[str, Any]:
    """Execute the complete public-data workflow and write auditable artifacts."""
    download_archive(ARCHIVE_URL, archive_path)
    download_archive(GENCODE_GTF_URL, gtf_path)
    samples = parse_geo_samples(soft_path)
    transcript_counts = read_estimated_count_matrix(archive_path, samples)
    mapping = load_gencode_transcript_map(gtf_path)
    gene_counts, gene_info, mapping_diagnostics = collapse_transcripts_to_genes(
        transcript_counts, mapping
    )
    results, pca, metrics = analyze_gene_expression(gene_counts, gene_info, samples)

    payload = {
        "schema_version": "2.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "accession": "GSE309548",
        "provenance_tier": "PUBLIC_REAL_DERIVED",
        "analysis_status": "complete",
        "research_question": (
            "Which gene-level expression differences between SCA3 and control spinal cord "
            "donors remain after sex adjustment, and how stable are they to age "
            "adjustment, donor omission and exact label-assignment calibration?"
        ),
        "method": {
            "quantification": "Author-provided Kallisto estimated counts, GRCh38",
            "annotation": "GENCODE release 50 comprehensive gene annotation",
            "unit_of_analysis": "donor pseudobulk",
            "regional_handling": (
                "Regional files belonging to one donor are summed with their library "
                "sizes before normalization; punch samples remain one donor library."
            ),
            "normalization": "Library-size normalized log2 counts per million",
            "expression_filter": "CPM >= 1 in at least 3 of 8 donor profiles",
            "primary_model": "gene expression ~ SCA3 + sex",
            "variance_model": (
                "Across-gene scaled inverse-chi-square empirical-Bayes shrinkage "
                "of residual variances"
            ),
            "multiplicity": "Benjamini-Hochberg false-discovery rate",
            "sensitivity": [
                "Age-adjusted model: gene expression ~ SCA3 + sex + standardized age",
                "Leave-one-donor-out effect-direction stability for the top 20 genes",
                "Enumeration of every fixed-size 4-vs-4 disease-label assignment",
            ],
        },
        "metrics": {**mapping_diagnostics, **metrics},
        "pca": _records(pca),
        "top_signals": _records(results.head(30)),
        "interpretation_boundary": (
            "Exploratory post-mortem spinal cord analysis with four SCA3 and four control "
            "donors. Estimated counts are transcript-to-gene aggregates, the cohort "
            "is observational, and neither nominal nor FDR signals establish a causal "
            "mechanism, clinical utility or an individual prediction."
        ),
        "limitations": [
            "Eight independent donors sharply limit precision and covariate adjustment.",
            "Disease status and age are imbalanced; the age-adjusted model is a sensitivity analysis.",
            "Some donors have separate dorsal and ventral files while two SCA3 donors have punch libraries.",
            "The empirical-Bayes model is limma-inspired but is an independently implemented educational analysis, not a validated clinical pipeline.",
            "External cohort replication is required before interpreting biological mechanisms.",
        ],
        "sources": {
            "geo_archive": {
                "url": ARCHIVE_URL,
                "sha256": sha256_file(archive_path),
                "bytes": archive_path.stat().st_size,
                "path": str(archive_path),
            },
            "gencode_annotation": {
                "url": GENCODE_GTF_URL,
                "release": "GENCODE 50 (GRCh38.p14)",
                "sha256": sha256_file(gtf_path),
                "bytes": gtf_path.stat().st_size,
                "path": str(gtf_path),
            },
            "sample_metadata": str(soft_path),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False, compression="gzip")
    return payload


def main(argv: list[str] | None = None) -> int:
    metadata_candidates = sorted(
        (PROJECT_ROOT / "data" / "raw" / "geo_gse309548_metadata").glob(
            "*.soft.txt"
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "GSE309548_RAW.tar",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=metadata_candidates[-1]
        if metadata_candidates
        else Path("missing.soft.txt"),
    )
    parser.add_argument(
        "--gtf",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "gencode.v50.annotation.gtf.gz",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "gse309548_gene_analysis.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "gse309548_differential_genes.csv.gz",
    )
    args = parser.parse_args(argv)
    payload = run(
        args.archive,
        args.metadata,
        args.gtf,
        args.output_json,
        args.output_csv,
    )
    print(json.dumps(payload["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
