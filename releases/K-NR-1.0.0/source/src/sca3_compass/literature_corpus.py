"""Content-addressed, exact-span PMC corpus for local noncommercial research."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ingest_public_data import fetch, store_snapshot
from .repository import PROJECT_ROOT

CORPUS_PATH = PROJECT_ROOT / "data/processed/literature-corpus.json"
SOURCE_PATH = PROJECT_ROOT / "configs/literature_sources.json"
ALLOWED_SECTIONS = {
    "ABSTRACT",
    "INTRO",
    "METHODS",
    "RESULTS",
    "DISCUSS",
    "CONCL",
    "SUPPL",
}


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def exact_spans(text: str, maximum: int = 1400) -> list[tuple[int, int]]:
    """Nonoverlapping contiguous slices; offsets index the unmodified BioC text."""
    spans, start = [], 0
    while start < len(text):
        end = min(start + maximum, len(text))
        if end < len(text):
            boundaries = list(re.finditer(r"[.!?]\s+", text[start:end]))
            if boundaries and boundaries[-1].end() > maximum // 2:
                end = start + boundaries[-1].end()
            else:
                space = text.rfind(" ", start + maximum // 2, end)
                if space > start:
                    end = space + 1
        spans.append((start, end))
        start = end
    return spans


def extract_article(
    payload: Any, spec: dict, snapshot: dict
) -> tuple[dict, list[dict]]:
    document = payload[0]["documents"][0]
    passages = document["passages"]
    front = next(
        p for p in passages if p.get("infons", {}).get("section_type") == "TITLE"
    )
    info = front["infons"]
    if info.get("article-id_pmc") != spec["pmcid"]:
        raise ValueError("PMC identifier mismatch")
    license_text = info.get("license", "")
    normalized = license_text.lower()
    if (
        "creativecommons.org/licenses/by" not in normalized
        and "creative commons attribution" not in normalized
    ):
        raise ValueError(f"{spec['pmcid']}: license requires manual review")
    if any(term in normalized for term in ("noderiv", "no-deriv", "by-nc-nd", "by-nd")):
        raise ValueError(f"{spec['pmcid']}: no-derivatives source not admitted")
    authors = [
        value.replace("surname:", "").replace(";given-names:", ", ")
        for key, value in sorted(info.items())
        if key.startswith("name_")
    ]
    article = {
        **spec,
        "title": front["text"],
        "year": info.get("year"),
        "doi": info.get("article-id_doi"),
        "authors": authors,
        "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{spec['pmcid']}/",
        "license": license_text,
        "snapshot": snapshot,
    }
    chunks = []
    heading = ""
    for ordinal, passage in enumerate(passages):
        infons = passage.get("infons", {})
        section = infons.get("section_type", "")
        kind = infons.get("type", "")
        text = passage.get("text", "")
        if "title" in kind:
            heading = text
            continue
        # References, captions, author disclosures and table fragments are deliberately excluded.
        if (
            section not in ALLOWED_SECTIONS
            or len(text) < 100
            or kind not in {"paragraph", "abstract"}
        ):
            continue
        for start, end in exact_spans(text):
            excerpt = text[start:end]
            chunks.append(
                {
                    "id": f"{spec['pmcid']}-p{ordinal:03d}-{start}",
                    "pmcid": spec["pmcid"],
                    "section": section,
                    "heading": heading,
                    "passage_ordinal": ordinal,
                    "start": start,
                    "end": end,
                    "bioc_offset": int(passage.get("offset", 0)) + start,
                    "text": excerpt,
                    "text_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
                }
            )
    if not chunks:
        raise ValueError(f"{spec['pmcid']}: no usable passages")
    article["chunks"] = len(chunks)
    return article, chunks


def load_corpus(path: Path = CORPUS_PATH) -> dict:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if (
        digest_json({"articles": corpus["articles"], "chunks": corpus["chunks"]})
        != corpus["sha256"]
    ):
        raise ValueError("Literature corpus integrity mismatch")
    return corpus


def verify_spans(corpus: dict) -> dict:
    """Re-read each source snapshot and check every exported span, not just its hash."""
    checked = 0
    for article in corpus["articles"]:
        snapshot = article["snapshot"]
        path = (PROJECT_ROOT / snapshot["path"]).resolve()
        if not path.is_relative_to((PROJECT_ROOT / "data/raw").resolve()):
            raise ValueError("Snapshot must remain within project data/raw")
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != snapshot["sha256"]:
            raise ValueError("Source snapshot integrity mismatch")
        original = json.loads(body)[0]["documents"][0]["passages"]
        for chunk in (c for c in corpus["chunks"] if c["pmcid"] == article["pmcid"]):
            passage = original[chunk["passage_ordinal"]]
            if passage["text"][chunk["start"] : chunk["end"]] != chunk["text"]:
                raise ValueError("Excerpt does not match its claimed source span")
            if int(passage["offset"]) + chunk["start"] != chunk["bioc_offset"]:
                raise ValueError("BioC offset mismatch")
            if (
                hashlib.sha256(chunk["text"].encode()).hexdigest()
                != chunk["text_sha256"]
            ):
                raise ValueError("Excerpt hash mismatch")
            checked += 1
    return {
        "status": "passed",
        "exact_spans_checked": checked,
        "meaning": "Byte hashes and text spans checked; relevance and clinical validity not certified.",
    }


def build_corpus() -> dict[str, Any]:
    specs = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    previous = (
        {a["pmcid"]: a["snapshot"] for a in load_corpus()["articles"]}
        if CORPUS_PATH.exists()
        else {}
    )
    articles, chunks = [], []
    raw_root = PROJECT_ROOT / "data/raw/literature"
    for spec in specs["sources"]:
        pmcid = spec["pmcid"]
        url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
        existing = sorted((raw_root / pmcid).glob("*.json"))
        if existing:
            body = existing[-1].read_bytes()
            snapshot = {
                "path": str(existing[-1].relative_to(PROJECT_ROOT)),
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
                "url": url,
            }
            if previous.get(pmcid, {}).get("sha256") == snapshot["sha256"]:
                snapshot = previous[pmcid]
        else:
            body, meta = fetch(url, attempts=3)
            snapshot = store_snapshot(raw_root, pmcid, url, body, meta, ".json")
            snapshot["path"] = str(Path(snapshot["path"]).relative_to(PROJECT_ROOT))
        article, excerpts = extract_article(json.loads(body), spec, snapshot)
        articles.append(article)
        chunks.extend(excerpts)
        print(f"{pmcid}: {len(excerpts)} exact-span chunks", flush=True)
    corpus = {
        "version": specs["version"],
        "generated_at": datetime.now(UTC).isoformat(),
        "selection": specs["selection"],
        "not_indexed": specs["not_indexed"],
        "articles": articles,
        "chunks": chunks,
        "sha256": digest_json({"articles": articles, "chunks": chunks}),
    }
    corpus["verification"] = verify_spans(corpus)
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = (
        verify_spans(load_corpus()) if args.verify else build_corpus()["verification"]
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
