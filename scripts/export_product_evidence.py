"""Export only allowlisted, existing simulation metrics for a read-only portal.

No simulator/model imports, fitting, new observations or protected-data reads.
The D020 export is derived from all 20 indexed records, not a selected subset.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R4 = Path("research/r4-target-calibration-20260918/C001")
R5 = Path("research/r5-selection-aware-20260918/D020")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_member(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Unsafe evidence member")
    path = root / rel
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes source")
    return path


def verify_freeze(root: Path):
    for name, expected in read(root / "freeze.json")["files"].items():
        if digest(safe_member(root, name)) != expected:
            raise ValueError("Frozen source mismatch: " + name)


def scalar_metrics(methods):
    """Only named finite scalar statistics; no arbitrary fields copied through."""
    result = {}
    for method, metrics in methods.items():
        result[method] = {}
        for key in ["power", "fdp", "tp", "fp", "discoveries"]:
            value = metrics[key]
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("Invalid scalar metric")
            result[method][key] = value
    return result


def extract(root: Path = ROOT) -> dict:
    old, new = root / R4, root / R5
    verify_freeze(new)
    r4 = read(old / "summary.json")
    delivery = read(old.parent / "DELIVERY_MANIFEST.json")
    if digest(old / "summary.json") != delivery["files"]["C001/summary.json"]["sha256"]:
        raise ValueError("R4 summary differs from historical delivery receipt")
    protocol = read(old / "protocol.json")
    index = read(new / "index.json")
    if r4["freeze_sha256"] != digest(old / "freeze.json"):
        raise ValueError("R4 summary/freeze mismatch")
    if r4["index_sha256"] != digest(old / "index.json"):
        raise ValueError("R4 summary/index mismatch")
    records = []
    for item in index["rows"]:
        p = safe_member(new, item["path"])
        if digest(p) != item["sha"]:
            raise ValueError("D020 record hash mismatch")
        with gzip.open(p, "rt", encoding="utf-8") as f:
            row = json.load(f)
        if row["status"] != "completed":
            raise ValueError("Incomplete D020: no silent success-only export")
        if row["freeze_sha"] != digest(new / "freeze.json"):
            raise ValueError("D020 source version mismatch")
        if digest(safe_member(new, row["evidence_path"])) != row["evidence_sha"]:
            raise ValueError("D020 saved numeric output mismatch")
        previous_path = safe_member(new.parent, row["previous_record"])
        if digest(previous_path) != row["previous_record_sha"]:
            raise ValueError("D019 comparison record mismatch")
        with gzip.open(previous_path, "rt", encoding="utf-8") as f:
            previous = json.load(f)
        if previous["metrics"] != row["prior_metrics"]:
            raise ValueError("D020 comparators differ from indexed D019 metrics")
        if digest(safe_member(new.parent, row["previous_evidence"])) != row["previous_evidence_sha"]:
            raise ValueError("Saved D019 comparison arrays mismatch")
        # Whitelisted scalar metrics only; never export observed/diagnostic arrays.
        records.append({
            "case": row["case"], "rep": row["rep"], "status": row["status"],
            "metrics": scalar_metrics(row["metrics"]), "prior_metrics": scalar_metrics(row["prior_metrics"]),
            "candidate_status": row["candidate"]["status"],
            "seconds": row["seconds"], "source_record_sha256": item["sha"],
            "evidence_sha256": row["evidence_sha"],
        })
    records.sort(key=lambda x: (x["case"], x["rep"]))
    if [(r["case"], r["rep"]) for r in records] != [(c, r) for c in range(10) for r in range(2)]:
        raise ValueError("Expected all 20 unique paired development families")
    summary = read(new / "summary.json")
    cases = []
    methods = [
        ("R5_A014", "metrics", "A014", "R5 三动作候选"),
        ("R4_target", "prior_metrics", "R4_target", "R4 target-only"),
        ("R4_source_bound", "prior_metrics", "R4_source_bound", "R4 source-bound"),
        ("R4_bridge", "prior_metrics", "R4_bridge", "R4 固定桥"),
        ("matched_mix", "metrics", "matched_TS_mix", "同校准升级固定 e 混合"),
        ("triangular_target", "prior_metrics", "target", "同校准升级 target-only"),
        ("matched_source", "metrics", "matched_source", "同方向学习 source-bound"),
        ("matched_three_mix", "metrics", "matched_three_mix", "三端点固定 e 混合"),
        ("strong_target", "prior_metrics", "R4_strong_target", "强参照 target · 仅实证"),
        ("strong_pool", "prior_metrics", "R4_strong_pool_bound", "强参照 pool · 仅实证"),
    ]
    for c in range(10):
        rr = [r for r in records if r["case"] == c]
        scores = []
        for key, field, source, label in methods:
            metrics = {}
            for metric in ["power", "fdp", "tp", "fp", "discoveries"]:
                vals = [r[field][source][metric] for r in rr]
                metrics[metric] = None if vals[0] is None else sum(vals) / len(vals)
            if source == "A014":
                expected = summary["rows"][c]["means"]["A014"]
                for k, v in metrics.items():
                    if v is None:
                        if expected[k] is not None: raise ValueError("Null Power mismatch")
                    elif not math.isclose(v, expected[k], abs_tol=1e-12):
                        raise ValueError("D020 summary does not match raw metrics")
            scores.append({"id": key, "label": label, **metrics})
        cases.append({"case": c, "scene": protocol["cases"][c], "n": 2,
                      "outside": c == 9, "methods": scores})
    sources = [R4 / "summary.json", R4 / "protocol.json", R4 / "freeze.json", R4.parent / "DELIVERY_MANIFEST.json",
               Path("research/r3-efficiency-20260917/README.md"),
               R5 / "summary.json", R5 / "protocol.json", R5 / "freeze.json", R5 / "index.json"]
    return {
        "schema_version": 1, "snapshot_date": "2026-10-03", "kind": "SIMULATION_NOT_PATIENT_DATA",
        "decision": {"R5_promoted": False, "R5_formal_confirmations": 0,
                     "all_predecessors_superiority": "NOT_ESTABLISHED",
                     "default_mode": "READ_ONLY_EVIDENCE_NOT_PATIENT_INFERENCE"},
        "r4": {"stage": r4["stage"], "rows": r4["rows"], "core_paired_power": r4["core_paired_power"],
               "alpha": r4["alpha"], "intervals_count": r4["intervals_count"], "interval_cap": r4["interval_cap"]},
        "r5": {"stage": "DEVELOPMENT_ONLY_NOT_ACCEPTANCE", "cases": cases, "records": records,
               "confirmation_count": 0, "n": len(records), "new_observed_families": 0},
        "sources": [{"path": p.as_posix(), "sha256": digest(root / p)} for p in sources],
        "limitations": [
            "R4 的 512 次/场景正式确认与 R5 的 2 次/场景开发诊断不得合并。",
            "R5 复用了已见模拟输入，没有完成正式独立确认；不展示虚假的精度或胜出声明。",
            "R1/V1、R2、R3 与 R5 尚无统一范围的全版本确认比较。版本号不表示性能排名。",
            "所有 PC 零假设场景的 Power 未定义；不得将它当成 0 或完美表现。",
            "D 被低估的 X9 保留；高 Power 不代表错误控制成立。",
            "尚无认证真实 SCA3 数据同时满足独立零位置校准与外部漂移上界。",
            "导出哈希证明本次文件对应关系，不证明数学正确或患者受益。",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "apps/web/public/research-evidence.json")
    args = parser.parse_args()
    value = extract()
    write(args.out, value)
    print(json.dumps({"output": args.out.as_posix(), "paired_records": len(value["r5"]["records"]),
                      "R5_promoted": False, "sha256": digest(args.out)}))


if __name__ == "__main__":
    main()
