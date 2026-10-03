from __future__ import annotations

import numpy as np
import pandas as pd

from sca3_compass.gene_expression import (
    analyze_gene_expression,
    collapse_transcripts_to_genes,
    fit_moderated_model,
    parse_gtf_attributes,
)


def test_parse_gtf_attributes() -> None:
    value = 'gene_id "ENSG1.2"; transcript_id "ENST1.4"; gene_name "ATXN3";'
    parsed = parse_gtf_attributes(value)
    assert parsed["gene_id"] == "ENSG1.2"
    assert parsed["gene_name"] == "ATXN3"


def test_collapse_transcripts_to_genes() -> None:
    counts = pd.DataFrame(
        {"S1": [2.0, 3.0, 7.0], "S2": [4.0, 1.0, 8.0]},
        index=["T1", "T2", "T3"],
    )
    mapping = pd.DataFrame(
        {
            "gene_id": ["G1", "G1", "G2"],
            "gene_name": ["ONE", "ONE", "TWO"],
            "gene_type": ["protein_coding", "protein_coding", "lncRNA"],
        },
        index=["T1", "T2", "T3"],
    )
    gene_counts, gene_info, diagnostics = collapse_transcripts_to_genes(
        counts, mapping
    )
    assert gene_counts.loc["G1", "S1"] == 5.0
    assert gene_info.loc["G2", "gene_name"] == "TWO"
    assert diagnostics["mapped_genes"] == 2


def test_moderated_model_finds_large_signal() -> None:
    rng = np.random.default_rng(42)
    design = np.column_stack(
        [np.ones(10), np.array([0] * 5 + [1] * 5), np.tile([0, 1], 5)]
    )
    expression = rng.normal(0, 0.2, size=(10, 80))
    expression[5:, 0] += 3.0
    fit = fit_moderated_model(expression, design, 1)
    assert np.asarray(fit["coefficient"])[0] > 2.5
    assert np.asarray(fit["p_value"])[0] < 1e-6
    assert fit["residual_df"] == 7


def test_gene_analysis_keeps_donor_as_unit() -> None:
    rng = np.random.default_rng(7)
    accessions = [f"S{index}" for index in range(8)]
    counts = pd.DataFrame(
        rng.poisson(100, size=(40, 8)).astype(float),
        index=[f"G{index}" for index in range(40)],
        columns=accessions,
    )
    counts.iloc[0, 4:] += 300
    gene_info = pd.DataFrame(
        {
            "gene_name": [f"GENE{index}" for index in range(40)],
            "gene_type": ["protein_coding"] * 40,
        },
        index=counts.index,
    )
    samples = []
    for index, accession in enumerate(accessions):
        samples.append(
            {
                "accession": accession,
                "donor_id": f"D{index}",
                "genotype": "CTRL" if index < 4 else "SCA3",
                "sex": "M" if index in {1, 3, 4} else "F",
                "time": str(45 + index),
            }
        )
    results, pca, metrics = analyze_gene_expression(
        counts, gene_info, samples, minimum_cpm=0, minimum_donors=1
    )
    assert metrics["donors"] == 8
    assert metrics["primary_model"]["residual_df"] == 5
    assert len(pca) == 8
    assert results.iloc[0]["gene_id"] == "G0"
