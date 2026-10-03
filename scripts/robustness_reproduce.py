"""Exact historical replay, never overwrites any old scientific artifact."""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime

import numpy as np

from sca3_compass.molecular_data import PROJECT_ROOT, digest, write_json
from sca3_compass.molecular_envelope_benchmark import SOURCE_NAMES, one_scenario, scenarios


def main():
    root = PROJECT_ROOT
    old_path = root / "artifacts/molecular-envelope-validation.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    protocol = old["effective"]["protocol"]
    for name in SOURCE_NAMES:
        assert digest(root / "src/sca3_compass" / name) == old["effective"]["fingerprint"]["sources"][name]
    chosen = [(i,c) for i,c in enumerate(scenarios(protocol)) if c["name"] in ["heavy_tail_both", "n256_rho0.95_z3.5"]]
    work = root / "artifacts/robustness"
    started = datetime.now(UTC).isoformat()
    registry = {"experiments": [{"id": "R0001", "status": "running", "purpose": "Exact historical replay; NOT independent confirmation",
        "started_at": started, "source_artifact_sha256": digest(old_path), "settings": protocol,
        "fingerprint": old["effective"]["fingerprint"], "scenarios": [c["name"] for _,c in chosen]}]}
    write_json(work / "EXPERIMENT_REGISTRY.json", registry)
    results = []
    start = time.perf_counter()
    with ProcessPoolExecutor(2) as pool:
        futures = [pool.submit(one_scenario, protocol, c, i, "validation") for i,c in chosen]
        for f in as_completed(futures):
            got = f.result()
            expected = next(c for c in old["scenarios"] if c["scenario"] == got["scenario"])
            for a,b in zip(got["rows"], expected["rows"], strict=True):
                assert a["method"] == b["method"]
                for key in ["fdp_by_repetition", "power_by_repetition"]:
                    np.testing.assert_array_equal(a[key], b[key])
            got["exact_historical_arrays_match"] = True
            results.append(got)
            print(got["scenario"], "EXACT MATCH", flush=True)
    elapsed = time.perf_counter()-start
    write_json(work / "R0001-historical-replay.json", {"created_at": datetime.now(UTC).isoformat(),
        "source_sha256": digest(old_path), "provenance": "SIMULATION_NOT_PATIENT_DATA",
        "independent_confirmation": False, "elapsed_seconds": elapsed, "scenarios": results})
    registry["experiments"][0].update(status="completed", elapsed_seconds=elapsed,
        completed_at=datetime.now(UTC).isoformat(), all_eight_methods_exact_match=True)
    write_json(work / "EXPERIMENT_REGISTRY.json", registry)
    print("Reproduction completed", elapsed, flush=True)


if __name__ == "__main__":
    main()
