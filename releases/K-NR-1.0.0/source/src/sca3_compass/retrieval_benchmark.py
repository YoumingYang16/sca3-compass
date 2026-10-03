"""Frozen, paired bilingual pilot with intent-cluster bootstrap uncertainty."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import numpy as np

from .literature_corpus import digest_json
from .repository import PROJECT_ROOT
from .semantic_retrieval import METHODS, REPORT_PATH, LiteratureIndex, file_sha256

BENCHMARK_PATH = PROJECT_ROOT / "configs/literature_benchmark.json"


def ranking_metrics(ranked: list[str], relevant: list[str]) -> dict[str, float]:
    gold = set(relevant)
    if not gold or len(ranked) != len(set(ranked)):
        raise ValueError(
            "Nonempty relevance labels and unique ranked passages are required"
        )
    hits = [i + 1 for i, value in enumerate(ranked[:10]) if value in gold]
    dcg = sum(1 / np.log2(rank + 1) for rank in hits)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(gold))))
    return {
        "mrr_at_10": 1 / min(hits) if hits else 0.0,
        "ndcg_at_10": float(dcg / ideal),
        "recall_at_5": len(gold.intersection(ranked[:5])) / len(gold),
    }


def paired_summary(
    rows: list[dict], metric: str, seed: int = 20270915, draws: int = 5000
) -> dict:
    intents = sorted({r["intent"] for r in rows})
    values = np.array(
        [
            [
                np.mean(
                    [
                        r["metrics"][method][metric]
                        for r in rows
                        if r["intent"] == intent
                    ]
                )
                for method in METHODS
            ]
            for intent in intents
        ]
    )
    samples = np.random.default_rng(seed).integers(
        0, len(intents), size=(draws, len(intents))
    )
    bootstraps = values[samples].mean(axis=1)
    results = {}
    for column, method in enumerate(METHODS):
        low, high = np.quantile(bootstraps[:, column], [0.025, 0.975])
        delta = bootstraps[:, column] - bootstraps[:, 0]
        delta_low, delta_high = np.quantile(delta, [0.025, 0.975])
        results[method] = {
            "estimate": float(values[:, column].mean()),
            "low": float(low),
            "high": float(high),
            "difference_vs_bm25": float((values[:, column] - values[:, 0]).mean()),
            "difference_low": float(delta_low),
            "difference_high": float(delta_high),
        }
    return results


def run_benchmark() -> dict:
    spec = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    frozen_sha = digest_json(spec)
    index = LiteratureIndex()
    ids = {c["id"] for c in index.chunks}
    for intent in spec["intents"]:
        if not set(intent["relevant"]).issubset(ids):
            raise ValueError(f"Invalid frozen relevance labels: {intent['id']}")
    cases = [
        {
            "intent": item["id"],
            "language": language,
            "query": item[language],
            "relevant": item["relevant"],
        }
        for item in spec["intents"]
        for language in ("en", "zh")
    ]
    index.encode_query(cases[0]["query"])
    encoder = index.encoder
    vectors = encoder.encode([c["query"] for c in cases], query=True)
    print(
        f"Evaluating {len(cases)} queries / {len(spec['intents'])} intent clusters",
        flush=True,
    )
    rows = []
    for case, vector in zip(cases, vectors, strict=True):
        rank = index.rank(case["query"], vector)
        ranked = {
            name: [index.chunks[i]["id"] for i in rank["orders"][name][:10]]
            for name in METHODS
        }
        rows.append(
            {
                **case,
                "ranked": ranked,
                "metrics": {
                    name: ranking_metrics(ranked[name], case["relevant"])
                    for name in METHODS
                },
            }
        )
    # Safety is a finite regression suite, not a measurement of general clinical safety.
    safety = [
        {"query": query, "action": index.search(query, method="bm25")["action"]}
        for query in spec["safety_cases"]
    ]
    report = {
        "version": spec["version"],
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark_sha256": frozen_sha,
        "corpus_sha256": index.corpus["sha256"],
        "index_sha256": index.manifest["sha256"],
        "model_revision": index.manifest["revision"],
        "purpose": spec["purpose"],
        "limitations": [
            "Relevance annotations are incomplete and authored by the system builder, not blinded domain experts.",
            "No patient outcomes, treatment effects or human learning gains are measured by this benchmark.",
            "Pretraining exposure to these articles is unknown; pretrained embeddings do not imply an independently unseen corpus.",
            "Choosing a default method from this pilot is a development decision, not held-out validation.",
        ],
        "reproducibility": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": encoder.torch.__version__,
            "device": encoder.device,
            "model_dtype": str(next(encoder.model.parameters()).dtype),
            "source_sha256": {
                name: file_sha256(PROJECT_ROOT / "src/sca3_compass" / name)
                for name in (
                    "literature_corpus.py",
                    "semantic_retrieval.py",
                    "retrieval_benchmark.py",
                )
            },
        },
        "query_count": len(rows),
        "intent_clusters": len(spec["intents"]),
        "bootstrap": {
            "unit": "intent (English and Chinese paired)",
            "draws": 5000,
            "seed": spec["seed"],
            "boundary": "Percentile intervals conditional on this small authored query set; no external generalization guarantee.",
        },
        "aggregate": {
            metric: paired_summary(rows, metric, spec["seed"])
            for metric in ("mrr_at_10", "ndcg_at_10", "recall_at_5")
        },
        "by_language": {
            language: {
                name: {
                    metric: float(
                        np.mean(
                            [
                                r["metrics"][name][metric]
                                for r in rows
                                if r["language"] == language
                            ]
                        )
                    )
                    for metric in ("mrr_at_10", "ndcg_at_10", "recall_at_5")
                }
                for name in METHODS
            }
            for language in ("en", "zh")
        },
        "safety_cases": safety,
        "case_results": rows,
    }
    if (
        digest_json(json.loads(BENCHMARK_PATH.read_text(encoding="utf-8")))
        != frozen_sha
    ):
        raise ValueError("Benchmark labels changed during evaluation")
    report["sha256"] = digest_json(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["aggregate"], indent=2))
    return report


if __name__ == "__main__":
    run_benchmark()
