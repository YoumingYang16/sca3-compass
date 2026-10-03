"""Citation-bound evidence retrieval, abstention, and benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .repository import PROJECT_ROOT, load_claims

WORD_PATTERN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", re.IGNORECASE)
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]+")
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "that",
    "the",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}
SAFETY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(diagnose|diagnosis)\b.*\b(me|my|relative|family)\b",
        r"\b(which|what)\s+(medicine|drug|treatment)\s+should\s+i\b",
        r"\bpredict\s+my\b",
        r"\bmy\s+exact\b",
        r"诊断.*(我|家人|亲属)",
        r"预测我",
        r"我.*(吃什么药|用什么药|应该吃)",
    )
]
UNSUPPORTED_PROOF_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(prove|proves|proven)\b.*\b(treatment|target|effective|efficacy)\b",
        r"证明.*(治疗|靶点|有效)",
    )
]


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    tokens = [token for token in WORD_PATTERN.findall(normalized) if token not in STOPWORDS]
    for sequence in CJK_PATTERN.findall(normalized):
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


class BM25Index:
    """Small deterministic BM25 implementation used as an auditable baseline."""

    def __init__(self, documents: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("At least one document is required")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(str(item["search_text"])) for item in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths)
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        self.idf = {
            token: math.log(1.0 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        rows = []
        for document, tokens, length in zip(
            self.documents, self.tokens, self.lengths, strict=True
        ):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self.average_length, 1.0)
                )
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1.0) / denominator
            rows.append({**document, "score": score})
        rows.sort(key=lambda item: (-item["score"], item["id"]))
        return rows[:limit]


def build_claim_index(claim_payload: dict[str, Any] | None = None) -> BM25Index:
    claims = (claim_payload or load_claims())["claims"]
    documents = []
    for claim in claims:
        documents.append(
            {
                **claim,
                "search_text": " ".join(
                    [
                        claim["topic"],
                        claim["title"],
                        claim["plain_language"],
                        claim["boundary"],
                        claim["source_title"],
                    ]
                ),
            }
        )
    return BM25Index(documents)


def query_evidence(
    query: str,
    *,
    index: BM25Index | None = None,
    top_k: int = 3,
    minimum_score: float = 1.15,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be empty")
    trace = [
        {"stage": "policy", "status": "started"},
        {"stage": "retrieval", "status": "pending"},
        {"stage": "claim_source_linkage", "status": "pending"},
    ]
    if any(pattern.search(query) for pattern in SAFETY_PATTERNS):
        trace[0]["status"] = "safety_abstain"
        return {
            "query": query,
            "action": "safety_abstain",
            "answer": (
                "This research system cannot diagnose, predict an individual's course, "
                "or recommend treatment. A qualified clinician or genetic counsellor "
                "should address personal medical questions."
            ),
            "citations": [],
            "trace": trace,
            "medical_device": False,
        }
    if any(pattern.search(query) for pattern in UNSUPPORTED_PROOF_PATTERNS):
        trace[0]["status"] = "evidence_abstain"
        return {
            "query": query,
            "action": "evidence_abstain",
            "answer": (
                "The indexed evidence does not support that causal or clinical-effectiveness claim."
            ),
            "citations": [],
            "trace": trace,
            "medical_device": False,
        }

    trace[0]["status"] = "passed"
    retrieval = (index or build_claim_index()).search(query, limit=top_k)
    trace[1]["status"] = "complete"
    trace[1]["retrieved"] = len(retrieval)
    if not retrieval or retrieval[0]["score"] < minimum_score:
        trace[2]["status"] = "evidence_abstain"
        return {
            "query": query,
            "action": "evidence_abstain",
            "answer": "The indexed evidence is insufficient to answer this question reliably.",
            "citations": [],
            "trace": trace,
            "medical_device": False,
        }

    top_score = retrieval[0]["score"]
    retained = [row for row in retrieval if row["score"] >= max(minimum_score, top_score * 0.45)]
    citations = [
        {
            "claim_id": row["id"],
            "title": row["title"],
            "source_title": row["source_title"],
            "source_url": row["source_url"],
            "evidence_type": row["evidence_type"],
            "score": row["score"],
            "supported_statement": row["plain_language"],
            "boundary": row["boundary"],
        }
        for row in retained
    ]
    trace[2]["status"] = "curated_linkage_only_not_entailment"
    trace[2]["citation_coverage"] = 1.0
    return {
        "query": query,
        "action": "answer",
        "answer": " ".join(item["supported_statement"] for item in citations),
        "boundary": " ".join(item["boundary"] for item in citations),
        "citations": citations,
        "trace": trace,
        "retrieval_model": "BM25 lexical baseline v1",
        "generation": "extractive only",
        "provenance_tier": "PUBLIC_REAL_RETRIEVAL",
        "medical_device": False,
    }


def evaluate_benchmark(
    benchmark: dict[str, Any], index: BM25Index | None = None
) -> dict[str, Any]:
    engine = index or build_claim_index()
    reciprocal_ranks: list[float] = []
    recall_at_three: list[float] = []
    action_correct: list[float] = []
    cases = []
    for case in benchmark["cases"]:
        result = query_evidence(case["query"], index=engine)
        ranked = engine.search(case["query"], limit=3)
        ranked_ids = [row["id"] for row in ranked]
        expected = set(case["expected_claim_ids"])
        if expected:
            ranks = [ranked_ids.index(value) + 1 for value in expected if value in ranked_ids]
            reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
            recall_at_three.append(len(expected.intersection(ranked_ids)) / len(expected))
        action_match = result["action"] == case["expected_action"]
        action_correct.append(float(action_match))
        cases.append(
            {
                "id": case["id"],
                "expected_action": case["expected_action"],
                "actual_action": result["action"],
                "action_correct": action_match,
                "expected_claim_ids": sorted(expected),
                "ranked_claim_ids": ranked_ids,
            }
        )
    return {
        "benchmark_name": benchmark["benchmark_name"],
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "BM25 lexical baseline v1",
        "cases": len(cases),
        "answerable_cases": len(reciprocal_ranks),
        "metrics": {
            "mrr": sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1),
            "recall_at_3": sum(recall_at_three) / max(len(recall_at_three), 1),
            "action_accuracy": sum(action_correct) / len(action_correct),
        },
        "case_results": cases,
        "provenance_tier": "BENCHMARK_RESULT",
        "interpretation_boundary": (
            "This small curated benchmark measures technical retrieval and abstention "
            "behavior; it is not clinical validation."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT_ROOT / "configs" / "evidence_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evidence-retrieval-evaluation.json",
    )
    args = parser.parse_args(argv)
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    report = evaluate_benchmark(benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
