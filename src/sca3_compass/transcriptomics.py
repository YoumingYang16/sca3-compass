"""Reproducible donor-aware exploratory analysis of public GSE309548 RNA-seq TPMs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from scipy import stats

from .repository import PROJECT_ROOT

ARCHIVE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE309nnn/GSE309548/"
    "suppl/GSE309548_RAW.tar"
)
GENCODE_HGNC_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/"
    "gencode.v50.metadata.HGNC.gz"
)
USER_AGENT = "SCA3-Compass-Transcriptomics/0.1 (public-data research)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, destination: Path) -> None:
    """Stream the public GEO archive to disk if it is not already present."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(destination)


def parse_geo_samples(soft_path: Path) -> list[dict[str, str]]:
    """Parse the public GEO SOFT sample blocks and derive donor grouping keys."""
    samples: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in soft_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw_line.startswith("^SAMPLE = "):
            if current:
                samples.append(current)
            current = {"accession": raw_line.split("=", 1)[1].strip()}
        elif current is not None and raw_line.startswith("!Sample_title = "):
            current["title"] = raw_line.split("=", 1)[1].strip()
        elif current is not None and raw_line.startswith("!Sample_characteristics_ch1 = "):
            value = raw_line.split("=", 1)[1].strip()
            if ":" in value:
                key, item = value.split(":", 1)
                current[key.strip().lower()] = item.strip()
    if current:
        samples.append(current)

    for sample in samples:
        required = ("accession", "genotype", "sex", "time", "pmi", "tissue type")
        missing = [field for field in required if not sample.get(field)]
        if missing:
            raise ValueError(f"{sample.get('accession', 'unknown')}: missing {missing}")
        donor_key = "|".join(
            sample[field].strip().lower() for field in ("genotype", "sex", "time", "pmi")
        )
        sample["donor_id"] = hashlib.sha256(donor_key.encode()).hexdigest()[:10]
    return samples


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    values = np.nan_to_num(np.asarray(p_values, dtype=float), nan=1.0, posinf=1.0)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def read_tpm_matrix(archive_path: Path, samples: list[dict[str, str]]) -> pd.DataFrame:
    """Read the 14 compressed Kallisto tables directly from the GEO tar archive."""
    columns: dict[str, pd.Series] = {}
    sample_by_accession = {item["accession"]: item for item in samples}
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            accession = member.name.split("_", 1)[0]
            if accession not in sample_by_accession or not member.name.endswith(".csv.gz"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not extract {member.name}")
            with gzip.GzipFile(fileobj=extracted) as compressed:
                frame = pd.read_csv(compressed, usecols=["target_id", "tpm"])
            if frame["target_id"].duplicated().any():
                raise ValueError(f"Duplicate transcript ids in {member.name}")
            columns[accession] = frame.set_index("target_id")["tpm"]
    missing = sorted(set(sample_by_accession) - set(columns))
    if missing:
        raise ValueError(f"Archive is missing sample files: {missing}")
    matrix = pd.concat(columns, axis=1, join="inner")
    if matrix.empty:
        raise ValueError("No shared transcripts were found")
    return matrix


def analyze_donor_profiles(
    sample_matrix: pd.DataFrame,
    samples: list[dict[str, str]],
    minimum_tpm: float = 1.0,
    minimum_donors: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Aggregate regions by donor, then compare log2-TPM donor profiles."""
    metadata = pd.DataFrame(samples).set_index("accession").loc[sample_matrix.columns]
    donor_matrix = sample_matrix.T.groupby(metadata["donor_id"], sort=True).mean().T
    donor_metadata = metadata.groupby("donor_id", sort=True).first()
    if set(donor_metadata["genotype"].str.upper()) != {"SCA3", "CTRL"}:
        raise ValueError("Expected SCA3 and CTRL donor groups")

    keep = (donor_matrix >= minimum_tpm).sum(axis=1) >= minimum_donors
    filtered = donor_matrix.loc[keep]
    logged = np.log2(filtered + 0.5)
    sca_columns = donor_metadata.index[donor_metadata["genotype"].str.upper() == "SCA3"]
    ctrl_columns = donor_metadata.index[donor_metadata["genotype"].str.upper() == "CTRL"]
    if len(sca_columns) < 2 or len(ctrl_columns) < 2:
        raise ValueError("At least two donors per group are required")

    sca = logged.loc[:, sca_columns].to_numpy()
    ctrl = logged.loc[:, ctrl_columns].to_numpy()
    sca_variance = sca.var(axis=1, ddof=1)
    ctrl_variance = ctrl.var(axis=1, ddof=1)
    sca_component = sca_variance / sca.shape[1]
    ctrl_component = ctrl_variance / ctrl.shape[1]
    standard_error_squared = sca_component + ctrl_component
    mean_difference = sca.mean(axis=1) - ctrl.mean(axis=1)
    denominator = (
        sca_component**2 / (sca.shape[1] - 1)
        + ctrl_component**2 / (ctrl.shape[1] - 1)
    )
    valid = (standard_error_squared > np.finfo(float).eps) & (denominator > 0)
    p_values = np.ones(len(filtered))
    degrees_freedom = standard_error_squared[valid] ** 2 / denominator[valid]
    t_statistic = mean_difference[valid] / np.sqrt(standard_error_squared[valid])
    p_values[valid] = 2 * stats.t.sf(np.abs(t_statistic), degrees_freedom)
    results = pd.DataFrame(
        {
            "transcript_version": filtered.index,
            "transcript_id": filtered.index.str.replace(r"\..*$", "", regex=True),
            "mean_sca3_tpm": filtered.loc[:, sca_columns].mean(axis=1).to_numpy(),
            "mean_control_tpm": filtered.loc[:, ctrl_columns].mean(axis=1).to_numpy(),
            "log2_fold_change": mean_difference,
            "p_value": p_values,
            "q_value": benjamini_hochberg(p_values),
        }
    ).sort_values(["q_value", "p_value", "log2_fold_change"], ascending=[True, True, False])

    variable = logged.var(axis=1).nlargest(min(1000, len(logged))).index
    pca_input = logged.loc[variable].T.to_numpy()
    pca_input -= pca_input.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(pca_input, full_matrices=False)
    coordinates = u[:, :2] * singular[:2]
    variance = singular**2
    variance_ratio = variance / variance.sum() if variance.sum() else variance
    pca_rows = pd.DataFrame(
        {
            "donor_id": donor_metadata.index,
            "genotype": donor_metadata["genotype"].str.upper().to_numpy(),
            "pc1": coordinates[:, 0],
            "pc2": coordinates[:, 1],
        }
    )
    metrics = {
        "sample_files": int(sample_matrix.shape[1]),
        "transcripts_in_archive": int(sample_matrix.shape[0]),
        "retained_transcripts": int(filtered.shape[0]),
        "sca3_donors": len(sca_columns),
        "control_donors": len(ctrl_columns),
        "fdr_005": int((results["q_value"] <= 0.05).sum()),
        "fdr_005_abs_log2fc_1": int(
            ((results["q_value"] <= 0.05) & (results["log2_fold_change"].abs() >= 1)).sum()
        ),
        "pca_variance_percent": [round(float(value * 100), 2) for value in variance_ratio[:2]],
    }
    return results, pca_rows, metrics


def load_gencode_annotations(
    transcript_ids: list[str], raw_root: Path
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Map Ensembl transcripts using the public, versioned GENCODE HGNC table."""
    annotation_path = raw_root / "gencode.v50.metadata.HGNC.gz"
    download_archive(GENCODE_HGNC_URL, annotation_path)
    wanted = set(transcript_ids)
    annotations: dict[str, dict[str, str]] = {}
    with gzip.open(annotation_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) < 3:
                continue
            transcript_id = fields[0].split(".", 1)[0]
            if transcript_id in wanted:
                annotations[transcript_id] = {
                    "gene_name": fields[1],
                    "hgnc_id": fields[2],
                }
    return annotations, {
        "url": GENCODE_HGNC_URL,
        "release": "GENCODE 50 (GRCh38.p14)",
        "sha256": sha256_file(annotation_path),
        "bytes": annotation_path.stat().st_size,
        "path": str(annotation_path),
        "retrieved_at": datetime.now(UTC).isoformat(),
        "mapped_top_transcripts": len(annotations),
    }


def run(
    archive_path: Path,
    soft_path: Path,
    output_json: Path,
    output_csv: Path,
    raw_root: Path,
) -> dict[str, Any]:
    download_archive(ARCHIVE_URL, archive_path)
    samples = parse_geo_samples(soft_path)
    sample_matrix = read_tpm_matrix(archive_path, samples)
    results, pca, metrics = analyze_donor_profiles(sample_matrix, samples)
    annotation, annotation_snapshot = load_gencode_annotations(
        results.head(50)["transcript_id"].tolist(), raw_root
    )
    top_signals = []
    for row in results.head(25).to_dict(orient="records"):
        item = annotation.get(row["transcript_id"], {})
        top_signals.append(
            {
                **{key: (round(float(value), 8) if isinstance(value, float) else value) for key, value in row.items()},
                "gene_name": item.get("gene_name") or row["transcript_id"],
                "gene_id": item.get("hgnc_id"),
            }
        )

    archive_digest = sha256_file(archive_path)
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "accession": "GSE309548",
        "provenance_tier": "PUBLIC_REAL_DERIVED",
        "analysis_status": "complete",
        "method": {
            "quantification": "Author-provided Kallisto TPM, GRCh38",
            "unit_of_analysis": "donor",
            "regional_handling": "Dorsal and ventral files from one donor are averaged before inference",
            "expression_filter": "TPM >= 1 in at least 2 donor profiles",
            "transform": "log2(TPM + 0.5)",
            "contrast": "Welch two-sample t-test, SCA3 minus CTRL",
            "multiplicity": "Benjamini-Hochberg FDR",
            "pca": "SVD of centered top-1000 variable retained transcripts",
        },
        "metrics": metrics,
        "pca": pca.round(6).to_dict(orient="records"),
        "top_signals": top_signals,
        "interpretation_boundary": (
            "Exploratory post-mortem tissue analysis with four SCA3 and four control donors. "
            "Small-n results require replication and cannot predict an individual or establish causality."
        ),
        "limitations": [
            "The two punch samples combine dorsal and ventral tissue while other donors have region-specific files.",
            "Age, sex and post-mortem interval cannot all be adjusted reliably with only eight donors.",
            "Transcript-level TPM tests are exploratory and do not replace a count-based gene-level model.",
        ],
        "source": {
            "url": ARCHIVE_URL,
            "sha256": archive_digest,
            "bytes": archive_path.stat().st_size,
            "path": str(archive_path),
            "sample_metadata": str(soft_path),
        },
        "annotation_snapshot": annotation_snapshot,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False, compression="gzip")
    return payload


def main(argv: list[str] | None = None) -> int:
    metadata_candidates = sorted(
        (PROJECT_ROOT / "data" / "raw" / "geo_gse309548_metadata").glob("*.soft.txt")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=PROJECT_ROOT / "data" / "raw" / "GSE309548_RAW.tar"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=metadata_candidates[-1] if metadata_candidates else Path("missing.soft.txt"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "gse309548_analysis.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "gse309548_differential_transcripts.csv.gz",
    )
    args = parser.parse_args(argv)
    payload = run(
        args.archive,
        args.metadata,
        args.output_json,
        args.output_csv,
        PROJECT_ROOT / "data" / "raw",
    )
    print(json.dumps(payload["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
