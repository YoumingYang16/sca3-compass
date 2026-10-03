"""Non-destructive expression QC and sample provenance; no holdout unsealing."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .molecular_data import (
    PROJECT_ROOT,
    REPORT,
    assert_expression_access,
    digest,
    write_json,
)

OUTPUT = PROJECT_ROOT / "artifacts/molecular-qc.json"
PROCESSED = PROJECT_ROOT / "data/processed/molecular"


def attributes(sample: dict) -> dict[str, str]:
    output = {}
    for field in sample.get("characteristics_ch1", []):
        if ":" in field:
            key, value = field.split(":", 1)
            output[key.strip().lower()] = value.strip()
    return output


def match_columns(columns: list[str], samples: list[dict]) -> list[dict]:
    """Require exactly one metadata match per column; never use row position."""
    matched = []
    for column in columns:
        hits = []
        for sample in samples:
            aliases = [sample["sample_id"], *sample.get("description", []), *sample.get("title", [])]
            # LUM identifiers are recorded inside the actual sample title.
            aliases += re.findall(r"\bLUM\d+\b", " ".join(sample.get("title", [])))
            if column in aliases:
                hits.append(sample)
        if len(hits) != 1:
            raise ValueError(f"Ambiguous or missing sample mapping: {column} ({len(hits)} matches)")
        matched.append(hits[0])
    if len({s["sample_id"] for s in matched}) != len(matched):
        raise ValueError("Two expression columns map to the same GEO sample")
    if {s["sample_id"] for s in matched} != {s["sample_id"] for s in samples}:
        raise ValueError("Expression and metadata sample sets differ")
    return matched


def load_expression(dataset: dict) -> pd.DataFrame:
    assert_expression_access(dataset)
    file = next(f for f in dataset["files"] if f["kind"] == "expression")
    path = PROJECT_ROOT / file["path"]
    if digest(path) != file["sha256"]:
        raise ValueError("Raw input checksum mismatch")
    if dataset["format"] in {"counts_csv", "filtered_counts_tsv"}:
        sep = "," if dataset["format"] == "counts_csv" else "\t"
        frame = pd.read_csv(path, sep=sep, index_col=0)
        frame.index = frame.index.astype(str)
        return frame
    if dataset["format"] == "kallisto_tar":
        columns = {}
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".csv.gz"):
                    continue
                accession = member.name.split("_", 1)[0]
                if accession in columns:
                    raise ValueError("Duplicate archive sample")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Missing archive stream")
                with gzip.GzipFile(fileobj=stream) as handle:
                    table = pd.read_csv(handle, usecols=["target_id", "est_counts"])
                if table["target_id"].duplicated().any():
                    raise ValueError("Duplicate transcript identifiers")
                columns[accession] = table.set_index("target_id")["est_counts"]
        if not columns:
            raise ValueError("No quantification files found")
        first_index = next(iter(columns.values())).index
        if any(not x.index.equals(first_index) for x in columns.values()):
            raise ValueError("Different transcript universes; explicit reconciliation required")
        return pd.DataFrame(columns)
    raise ValueError("A platform-specific parser is required; not a count matrix")


def profile_dataset(dataset: dict) -> dict:
    result = {"accession": dataset["accession"], "role": dataset["role"], "family": dataset["family"],
              "organism": dataset["organism"], "issues": [], "samples": [],
              "independent_validation": False, "provenance": "PUBLIC_REAL_DERIVED"}
    if dataset["role"] == "sealed_candidate":
        return {**result, "status": "sealed_not_parsed", "features": None,
                "issues": ["Eligibility and study-family independence pending; expression remains sealed"]}
    if dataset["format"] in {"cel_tar", "fpkm_xlsx"}:
        return {**result, "status": "platform_specific_preprocessing_pending", "features": None,
                "issues": dataset["notes"]}
    matrix = load_expression(dataset)
    if matrix.empty or matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError("Empty matrix or duplicate identifiers")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Missing, infinite or negative counts; no zero imputation permitted")
    matched = match_columns(list(matrix.columns), dataset["metadata"]["samples"])
    totals = values.sum(axis=0)
    if (totals <= 0).any():
        raise ValueError("Empty library")
    for index, sample in enumerate(matched):
        attrs = attributes(sample)
        title = " ".join(sample["title"])
        animal = re.match(r"^(\d+)\s*\[LUM\d+\]", title)
        if animal is None:
            animal = re.search(r"\bmouse\s+(\d+)\s+LUM\d+", title)
        result["samples"].append({"sample_id": sample["sample_id"], "matrix_column": str(matrix.columns[index]),
            "source_title": title, "source_characteristics": sample["characteristics_ch1"],
            "tissue": attrs.get("tissue", attrs.get("tissue type", "unknown")),
            "genotype": attrs.get("genotype", attrs.get("genotype/variation", "unknown")),
            "age": attrs.get("age", attrs.get("age in weeks", attrs.get("time", "unknown"))),
            "unit_id": animal.group(1) if animal else None,
            "unit_basis": "animal identifier in source title" if animal else "unresolved; sample is not certified as independent",
            "library_total": float(totals[index])})
    units = {s["unit_id"] for s in result["samples"] if s["unit_id"] is not None}
    result["resolved_units"] = len(units) if all(s["unit_id"] for s in result["samples"]) else None
    if result["resolved_units"] is None:
        result["issues"].append("Independent biological units not fully resolved")
    if dataset["accession"] == "GSE145613":
        result["issues"].append("Source conflict: overall design says ATXN3; WT characteristics say ataxin-2. Preserved verbatim; confirmatory inference excluded pending resolution.")
    if dataset["format"] == "filtered_counts_tsv":
        result["issues"].append("Upstream expression filtering already applied; missing genes are not zero counts")
        result["issues"].append("Source unit conflict: supplementary-file metadata says CPM, while supplied values are integer-valued with count-like column totals. No count likelihood or confirmatory effect interpretation is certified; sum rescaling is exploratory.")
        result["value_unit_status"] = "provided_expression_count_like_but_metadata_says_CPM"
    if dataset["accession"] == "GSE309548":
        result["issues"].append("Legacy eight-donor aggregation uses demographic grouping; do not treat it as independently verified donor identity without source confirmation")
    # Label-free QC PCA only; normalization is not a fitted disease model.
    cpm = values / totals * 1e6
    keep = (cpm >= 1).sum(axis=1) >= max(2, int(np.ceil(values.shape[1] * 0.2)))
    logged = np.log2(cpm[keep] + 0.5)
    centered = logged - logged.mean(axis=1, keepdims=True)
    gram = centered.T @ centered
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    variance_sum = float(np.maximum(eigenvalues, 0).sum())
    if variance_sum <= 0:
        raise ValueError("No variance for QC PCA")
    scores = vectors[:, order[:2]] * np.sqrt(np.maximum(eigenvalues[order[:2]], 0))
    coordinates = [{"sample_id": sample["sample_id"], "x": float(scores[i, 0]), "y": float(scores[i, 1]),
                    "tissue": result["samples"][i]["tissue"], "genotype": result["samples"][i]["genotype"]}
                   for i, sample in enumerate(matched)]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    target = PROCESSED / f"{dataset['accession']}-counts.npz"
    np.savez_compressed(target, counts=values, features=matrix.index.to_numpy(dtype=str),
                        samples=np.asarray([s["sample_id"] for s in matched], dtype=str))
    result.update({"status": "matrix_qc_complete_inference_not_certified", "features": int(matrix.shape[0]),
                   "sample_count": int(matrix.shape[1]), "qc_features": int(keep.sum()),
                   "zero_fraction": float((values == 0).mean()),
                   "noninteger_fraction": float((np.abs(values - np.round(values)) > 1e-6).mean()),
                   "normalization_label": "log2(1e6 * provided expression / column sum + 0.5); QC transform, not certified original CPM",
                   "matrix_input_sha256": next(f["sha256"] for f in dataset["files"] if f["kind"] == "expression"),
                   "feature_id_digest": hashlib.sha256("\n".join(matrix.index.astype(str)).encode()).hexdigest(),
                   "derived_path": str(target.relative_to(PROJECT_ROOT)), "derived_sha256": digest(target),
                   "pca": coordinates, "pca_variance": (eigenvalues[order[:2]] / variance_sum).tolist(),
                   "tissue_counts": dict(Counter(s["tissue"] for s in result["samples"]))})
    return result


def main() -> None:
    acquisition = json.loads(REPORT.read_text(encoding="utf-8"))
    results = []
    for dataset in acquisition["datasets"]:
        try:
            result = profile_dataset(dataset)
        except (OSError, ValueError, KeyError, EOFError, tarfile.TarError) as exc:
            result = {"accession": dataset["accession"], "status": "qc_failed", "issues": [str(exc)]}
        results.append(result)
        print(dataset["accession"], result["status"], result.get("features"), flush=True)
    write_json(OUTPUT, {"created_at": datetime.now(UTC).isoformat(), "acquisition_sha256": digest(REPORT),
        "source_sha256": digest(Path(__file__)),
        "datasets": results, "summary": {"matrix_qc_complete": sum(r["status"].startswith("matrix_qc_complete") for r in results),
        "qc_failed": sum(r["status"] == "qc_failed" for r in results), "external_validation_certified": 0}})
    if any(r["status"] == "qc_failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
