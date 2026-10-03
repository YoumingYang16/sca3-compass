"""Math and provenance regression tests; toy inputs never enter public-data artifacts."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from sca3_compass.literature_corpus import (
    digest_json,
    exact_spans,
    extract_article,
    load_corpus,
    verify_spans,
)
from sca3_compass.retrieval_benchmark import (
    BENCHMARK_PATH,
    paired_summary,
    ranking_metrics,
)
from sca3_compass.semantic_retrieval import (
    METHODS,
    LiteratureIndex,
    diversify,
    reciprocal_rank_fusion,
)


def test_chunking_is_lossless_and_bounded():
    text = ("A paragraph with Unicode Δ SCA3. 中文句子。 " * 200) + "end"
    spans = exact_spans(text)
    assert "".join(text[start:end] for start, end in spans) == text
    assert max(end - start for start, end in spans) <= 1400
    assert len(spans) > 1


def test_corpus_bytes_and_all_exact_spans():
    corpus = load_corpus()
    assert verify_spans(corpus)["exact_spans_checked"] == len(corpus["chunks"])
    assert len({p["id"] for p in corpus["chunks"]}) == len(corpus["chunks"])
    assert all(a["license"] for a in corpus["articles"])


def test_frozen_qrels_are_real_unique_passage_ids():
    spec = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    ids = {p["id"] for p in load_corpus()["chunks"]}
    assert len({i["id"] for i in spec["intents"]}) == len(spec["intents"])
    assert all(
        set(i["relevant"]).issubset(ids)
        and len(i["relevant"]) == len(set(i["relevant"]))
        for i in spec["intents"]
    )


def test_license_and_identifier_gate():
    spec = {"pmcid": "PMC123"}
    front = {
        "text": "Title",
        "infons": {
            "section_type": "TITLE",
            "article-id_pmc": "PMC123",
            "license": "Creative Commons Attribution-NonCommercial-NoDerivatives",
        },
    }
    with pytest.raises(ValueError, match="no-derivatives"):
        extract_article([{"documents": [{"passages": [front]}]}], spec, {})
    with pytest.raises(ValueError, match="identifier mismatch"):
        extract_article(
            [{"documents": [{"passages": [front]}]}], {"pmcid": "PMC456"}, {}
        )


def test_rrf_handles_zero_lexical_matches_without_arbitrary_votes():
    score = reciprocal_rank_fusion(np.zeros(3), np.array([0.2, 0.8, 0.4]))
    assert np.argmax(score) == 1
    np.testing.assert_allclose(score, [1 / 63, 1 / 61, 1 / 62])


def test_rrf_hand_computable_and_diversification_no_duplicates():
    fused = reciprocal_rank_fusion(np.array([3.0, 1.0, 2.0]), np.array([0.1, 0.9, 0.5]))
    np.testing.assert_allclose(fused, [1 / 61 + 1 / 63, 1 / 63 + 1 / 61, 2 / 62])
    selected = diversify(np.argsort(-fused), fused, np.eye(3), limit=10)
    assert len(selected) == len(set(selected)) == 3


def test_ranking_metrics_on_known_cases():
    perfect = ranking_metrics(["a", "b", "c"], ["a", "b"])
    assert perfect == {"mrr_at_10": 1, "ndcg_at_10": 1, "recall_at_5": 1}
    assert ranking_metrics(["x", "a"], ["a"])["mrr_at_10"] == 0.5
    assert ranking_metrics([], ["a"])["ndcg_at_10"] == 0
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], ["a"])


def test_bootstrap_pairs_languages_and_methods():
    rows = [
        {
            "intent": intent,
            "language": language,
            "metrics": {m: {"ndcg_at_10": value} for m in METHODS},
        }
        for intent, value in [("one", 0.2), ("two", 0.8)]
        for language in ["en", "zh"]
    ]
    result = paired_summary(rows, "ndcg_at_10", seed=3, draws=100)
    assert result == paired_summary(rows, "ndcg_at_10", seed=3, draws=100)
    for value in result.values():
        assert value["estimate"] == 0.5
        assert (
            value["difference_vs_bm25"]
            == value["difference_low"]
            == value["difference_high"]
            == 0
        )


def test_lexical_query_does_not_load_model_or_generate_answer():
    index = LiteratureIndex(vectors=None)
    with patch.object(
        index, "encode_query", side_effect=AssertionError("must remain offline")
    ):
        result = index.search("mainland China SARA progression", method="bm25")
        assert result["results"]
        assert result["generation"] == "none"
        assert result["action"] == "passages_found"
        assert "answer" not in result


def test_personal_query_abstains_before_neural_inference():
    index = LiteratureIndex()
    with patch.object(
        index, "encode_query", side_effect=AssertionError("safety must run first")
    ):
        result = index.search("Which medicine should I take for my SCA3?", method="rrf")
    assert result["action"] == "safety_abstain"
    assert not result["results"]


def test_api_contract_and_source_reader():
    from fastapi.testclient import TestClient

    from sca3_compass.app import app

    client = TestClient(app)
    response = client.get("/api/literature")
    assert response.status_code == 200
    result = client.post(
        "/api/literature/search", json={"query": "SARA progression", "method": "bm25"}
    )
    assert result.status_code == 200
    passage = result.json()["results"][0]
    assert (
        client.get("/api/literature/passage/" + passage["id"]).json()["text"]
        == passage["text"]
    )
    assert client.get("/api/literature/passage/does-not-exist").status_code == 404
    assert (
        client.post(
            "/api/literature/search", json={"query": "a", "method": "rrf"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/literature/search", json={"query": "SCA3", "method": "invented"}
        ).status_code
        == 422
    )


def test_json_hash_ignores_key_order():
    assert digest_json({"a": 1, "b": [2]}) == digest_json({"b": [2], "a": 1})
