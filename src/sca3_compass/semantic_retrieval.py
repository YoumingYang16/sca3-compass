"""Local, frozen-model retrieval: lexical, neural, rank fusion and diversification.

No generated medical answers. Similarity is not a probability of correctness.
Model downloads happen only through the explicit CLI build, never an API request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .evidence_engine import SAFETY_PATTERNS, UNSUPPORTED_PROOF_PATTERNS, BM25Index
from .literature_corpus import digest_json, load_corpus, verify_spans
from .repository import PROJECT_ROOT

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MODEL_PATH = PROJECT_ROOT / "data/models/qwen3-embedding-0.6b"
INDEX_PATH = PROJECT_ROOT / "data/processed/literature-vectors.npz"
MANIFEST_PATH = PROJECT_ROOT / "data/processed/literature-index.json"
REPORT_PATH = PROJECT_ROOT / "artifacts/literature-retrieval-benchmark.json"
METHODS = ("bm25", "dense", "rrf", "diverse")
QUERY_INSTRUCTION = (
    "Retrieve scientific passages relevant to this SCA3 or ataxia research question."
)
MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
    "model.safetensors",
]


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class LocalEncoder:
    def __init__(self, model_path: Path = MODEL_PATH):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(6)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            padding_side="left",
        )
        self.model = (
            AutoModel.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )
        self.lock = threading.Lock()

    def encode(
        self, texts: list[str], *, query: bool = False, batch_size: int = 6
    ) -> np.ndarray:
        torch = self.torch
        values = (
            [f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}" for text in texts]
            if query
            else texts
        )
        rows = []
        with self.lock, torch.inference_mode():
            for start in range(0, len(values), batch_size):
                encoded = self.tokenizer(
                    values[start : start + batch_size],
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                if encoded["input_ids"].shape[1] > 2048:
                    raise ValueError(
                        "Passage exceeds the audited 2048-token limit; no silent truncation allowed"
                    )
                encoded = encoded.to(self.device)
                hidden = self.model(**encoded).last_hidden_state[:, -1].float()
                hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
                rows.append(hidden.cpu().numpy())
        return np.concatenate(rows, axis=0)


def stable_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="stable")


def reciprocal_rank_fusion(
    lexical: np.ndarray, dense: np.ndarray, depth: int = 50, k: int = 60
) -> np.ndarray:
    fused = np.zeros(len(dense))
    for scores, exclude_zero in ((lexical, True), (dense, False)):
        order = stable_order(scores)[:depth]
        if exclude_zero:
            order = order[scores[order] > 0]
        fused[order] += 1.0 / (k + np.arange(1, len(order) + 1))
    return fused


def diversify(
    order: np.ndarray,
    fused: np.ndarray,
    vectors: np.ndarray,
    limit: int = 10,
    weight: float = 0.8,
) -> list[int]:
    candidates = [int(i) for i in order[:30] if fused[i] > 0]
    selected: list[int] = []
    if not candidates:
        return selected
    maximum = max(float(fused[i]) for i in candidates)
    while candidates and len(selected) < limit:
        scores = []
        for index in candidates:
            redundancy = (
                max(0.0, float(np.max(vectors[selected] @ vectors[index])))
                if selected
                else 0.0
            )
            scores.append(
                weight * float(fused[index]) / maximum - (1 - weight) * redundancy
            )
        selected.append(candidates.pop(int(np.argmax(scores))))
    return selected


class LiteratureIndex:
    def __init__(
        self,
        corpus: dict | None = None,
        *,
        vectors: np.ndarray | None = None,
        manifest: dict | None = None,
    ):
        self.corpus = corpus or load_corpus()
        self.chunks = self.corpus["chunks"]
        self.articles = {a["pmcid"]: a for a in self.corpus["articles"]}
        docs = [
            {
                **c,
                "search_text": self.articles[c["pmcid"]]["title"]
                + " "
                + c["heading"]
                + " "
                + c["text"],
            }
            for c in self.chunks
        ]
        self.lexical = BM25Index(docs)
        self.id_to_index = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.manifest = manifest
        self.vectors = vectors
        if vectors is None and MANIFEST_PATH.exists() and INDEX_PATH.exists():
            self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            recorded_digest = self.manifest.get("sha256")
            if (
                digest_json({k: v for k, v in self.manifest.items() if k != "sha256"})
                != recorded_digest
            ):
                raise ValueError("Index manifest integrity mismatch")
            if (
                self.manifest["model_id"] != MODEL_ID
                or self.manifest["revision"] != MODEL_REVISION
                or self.manifest["query_instruction"] != QUERY_INSTRUCTION
            ):
                raise ValueError(
                    "Model configuration changed; rebuild the semantic index"
                )
            if self.manifest["corpus_sha256"] != self.corpus["sha256"]:
                raise ValueError("Vector index belongs to a different corpus")
            if file_sha256(INDEX_PATH) != self.manifest["vectors_sha256"]:
                raise ValueError("Vector index integrity mismatch")
            with np.load(INDEX_PATH, allow_pickle=False) as archive:
                if archive["ids"].tolist() != [c["id"] for c in self.chunks]:
                    raise ValueError("Vector row order does not match passages")
                self.vectors = archive["vectors"]
        if self.vectors is not None:
            if (
                len(self.vectors) != len(self.chunks)
                or not np.isfinite(self.vectors).all()
            ):
                raise ValueError("Invalid embedding matrix")
            if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-4):
                raise ValueError("Embeddings must be L2-normalized")
        self.encoder: LocalEncoder | None = None
        self.encoder_lock = threading.Lock()

    def encode_query(self, query: str) -> np.ndarray:
        if self.vectors is None:
            raise FileNotFoundError("Semantic index has not been built")
        with self.encoder_lock:
            if self.encoder is None:
                # Runtime tokenization and weights must exactly match the indexed model.
                for name, expected in self.manifest["model_files_sha256"].items():
                    if file_sha256(MODEL_PATH / name) != expected:
                        raise ValueError(
                            "Local model does not match the index manifest"
                        )
                self.encoder = LocalEncoder()
        return self.encoder.encode([query], query=True)[0]

    def rank(self, query: str, embedding: np.ndarray | None = None) -> dict:
        lexical = np.zeros(len(self.chunks))
        for row in self.lexical.search(query, limit=len(self.chunks)):
            lexical[self.id_to_index[row["id"]]] = row["score"]
        orders = {"bm25": [int(i) for i in stable_order(lexical) if lexical[i] > 0]}
        scores: dict[str, np.ndarray] = {"bm25": lexical}
        if embedding is not None and self.vectors is not None:
            dense = self.vectors @ embedding
            fused = reciprocal_rank_fusion(lexical, dense)
            orders["dense"] = stable_order(dense).tolist()
            orders["rrf"] = [int(i) for i in stable_order(fused) if fused[i] > 0]
            orders["diverse"] = diversify(stable_order(fused), fused, self.vectors)
            scores.update(dense=dense, rrf=fused)
        return {"orders": orders, "scores": scores}

    def search(self, query: str, method: str = "rrf", top_k: int = 6) -> dict[str, Any]:
        if method not in METHODS or not query.strip() or not 1 <= top_k <= 10:
            raise ValueError("Invalid search request")
        response = {
            "query": query,
            "method": method,
            "corpus_sha256": self.corpus["sha256"],
            "generation": "none",
            "medical_device": False,
            "boundary": "Related passages only; no entailment verification, diagnosis or treatment recommendation. Scores are not confidence probabilities.",
        }
        if any(p.search(query) for p in SAFETY_PATTERNS + UNSUPPORTED_PROOF_PATTERNS):
            return {
                **response,
                "action": "safety_abstain",
                "results": [],
                "comparison": {},
            }
        embedding = self.encode_query(query) if method != "bm25" else None
        ranking = self.rank(query, embedding)
        if method not in ranking["orders"]:
            raise FileNotFoundError(
                "Requested semantic method is unavailable; build the index or select BM25"
            )
        rank_lookup = {
            name: {i: rank + 1 for rank, i in enumerate(order)}
            for name, order in ranking["orders"].items()
        }
        results = []
        for i in ranking["orders"][method][:top_k]:
            chunk = self.chunks[i]
            results.append(
                {
                    **chunk,
                    "article": self.articles[chunk["pmcid"]],
                    "ranks": {
                        name: values.get(i) for name, values in rank_lookup.items()
                    },
                    "scores": {
                        name: float(values[i])
                        for name, values in ranking["scores"].items()
                    },
                }
            )
        comparison = {
            name: [
                {
                    "id": self.chunks[i]["id"],
                    "pmcid": self.chunks[i]["pmcid"],
                    "section": self.chunks[i]["section"],
                }
                for i in order[:top_k]
            ]
            for name, order in ranking["orders"].items()
        }
        return {
            **response,
            "action": "passages_found" if results else "no_lexical_match",
            "results": results,
            "comparison": comparison,
        }

    def summary(self) -> dict:
        manifest = self.manifest or {}
        return {
            "status": "available",
            "sha256": self.corpus["sha256"],
            "generated_at": self.corpus["generated_at"],
            "selection": self.corpus["selection"],
            "articles": self.corpus["articles"],
            "chunk_count": len(self.chunks),
            "verification": self.corpus["verification"],
            "not_indexed": self.corpus["not_indexed"],
            "semantic_ready": self.vectors is not None,
            "model": {
                key: manifest.get(key)
                for key in (
                    "model_id",
                    "revision",
                    "device",
                    "dimensions",
                    "method_version",
                )
            },
            "projection": manifest.get("projection"),
            "methods": {
                "bm25": "Lexical baseline · k1=1.5, b=0.75",
                "dense": "Frozen Qwen3-Embedding-0.6B · normalized last-token embeddings",
                "rrf": "Equal-weight reciprocal rank fusion · k=60, depth=50",
                "diverse": "RRF + MMR · relevance weight 0.8, candidate pool 30",
            },
        }


def build_index() -> dict:
    import os

    # Download weights only from the pinned publisher repository; never execute remote code.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    corpus = load_corpus()
    verify_spans(corpus)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(MODEL_PATH),
        allow_patterns=MODEL_FILES,
        max_workers=2,
    )
    encoder = LocalEncoder()
    article_by_id = {a["pmcid"]: a for a in corpus["articles"]}
    texts = [
        article_by_id[c["pmcid"]]["title"] + " " + c["heading"] + " " + c["text"]
        for c in corpus["chunks"]
    ]
    print(f"Encoding {len(texts)} passages on {encoder.device}", flush=True)
    vectors = encoder.encode(texts)
    np.savez_compressed(
        INDEX_PATH, vectors=vectors, ids=np.array([c["id"] for c in corpus["chunks"]])
    )
    centered = vectors - vectors.mean(axis=0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * singular[:2]
    projection = {
        "method": "PCA of centered neural embeddings; exploratory, not a disease manifold",
        "explained_variance": (singular[:2] ** 2 / (singular**2).sum()).tolist(),
        "points": [
            {
                "id": c["id"],
                "pmcid": c["pmcid"],
                "section": c["section"],
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
            }
            for i, c in enumerate(corpus["chunks"])
        ],
    }
    manifest = {
        "method_version": "hybrid-literature-1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_sha256": corpus["sha256"],
        "vectors_sha256": file_sha256(INDEX_PATH),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_files_sha256": {
            name: file_sha256(MODEL_PATH / name)
            for name in MODEL_FILES
            if (MODEL_PATH / name).exists()
        },
        "device": encoder.device,
        "dimensions": int(vectors.shape[1]),
        "query_instruction": QUERY_INSTRUCTION,
        "projection": projection,
    }
    manifest["sha256"] = digest_json(manifest)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in manifest.items()
                if k not in {"projection", "model_files_sha256"}
            },
            indent=2,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        build_index()


if __name__ == "__main__":
    main()
