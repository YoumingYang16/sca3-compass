"""Release-matched public Ensembl annotations with immutable acquisition receipts."""
from __future__ import annotations

import gzip
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .molecular_data import PROJECT_ROOT, digest, write_json


def annotation_path(release: int) -> Path:
    if release not in (91,108):
        raise ValueError("Only explicitly registered releases allowed")
    assembly="GRCm38" if release==91 else "GRCm39"
    filename=f"Mus_musculus.{assembly}.{release}.gtf.gz"
    return PROJECT_ROOT/"data/raw/molecular/annotation"/filename


def acquire_annotation(release: int) -> Path:
    path=annotation_path(release)
    url=f"https://ftp.ensembl.org/pub/release-{release}/gtf/mus_musculus/{path.name}"
    receipt=path.with_suffix(path.suffix+".receipt.json")
    if path.exists():
        if not receipt.exists():
            raise ValueError("Unreceipted annotation; refusing overwrite")
        record=json.loads(receipt.read_text(encoding="utf-8"))
        if record["sha256"]!=digest(path) or record["url"]!=url:
            raise ValueError("Annotation cache integrity failure")
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".partial")
    with httpx.stream("GET",url,timeout=60,follow_redirects=True) as response:
        response.raise_for_status()
        total=0
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes(1024*1024):
                total+=len(chunk)
                if total>100_000_000:
                    raise ValueError("Annotation exceeds 100MB safety bound")
                handle.write(chunk)
        if total==0 or (response.headers.get("content-length") and total!=int(response.headers["content-length"])):
            raise ValueError("Incomplete annotation")
    os.replace(temporary,path)
    write_json(receipt,{"url":url,"sha256":digest(path),"bytes":total,"retrieved_at":datetime.now(UTC).isoformat(),"release":release,"provenance":"PUBLIC_REFERENCE_ANNOTATION"})
    return path


def parse_annotation(path: Path) -> tuple[dict[str,str],dict[str,str]]:
    """Return transcript->gene and unambiguous gene->symbol, preserve exclusions."""
    tx_to_gene={}
    names: dict[str,set[str]]={}
    with gzip.open(path,"rt",encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields=line.rstrip("\n").split("\t")
            if len(fields)!=9 or fields[2] not in {"gene","transcript"}:
                continue
            attrs=dict(re.findall(r'(\w+) "([^"]+)";',fields[8]))
            gene=attrs.get("gene_id","").split(".")[0]
            symbol=attrs.get("gene_name")
            if symbol:
                names.setdefault(gene,set()).add(symbol)
            if "transcript_id" in attrs:
                tx=attrs["transcript_id"].split(".")[0]
                if tx in tx_to_gene and tx_to_gene[tx]!=gene:
                    raise ValueError("Transcript maps to multiple genes")
                tx_to_gene[tx]=gene
    unique={g:next(iter(s)) for g,s in names.items() if len(s)==1}
    reverse: dict[str,list[str]]={}
    for g,s in unique.items():
        reverse.setdefault(s,[]).append(g)
    unique={g:s for g,s in unique.items() if len(reverse[s])==1}
    if not tx_to_gene or not unique:
        raise ValueError("Empty annotation")
    return tx_to_gene,unique
