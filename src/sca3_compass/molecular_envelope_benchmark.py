"""Prospective factorial benchmark of finite-calibration uncertainty."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from .molecular_benchmark import summarize
from .molecular_data import digest, write_json
from .molecular_envelope import (
    correlation_envelope,
    efilter,
    envelope_pvalues,
    gaussian_bank,
    rank_tail,
)
from .molecular_methods import (
    covariance_root,
    ebh,
    equicorrelation,
    fdr_adjust,
    p_to_e_mixture,
    partial_conjunction,
)
from .repository import PROJECT_ROOT

PROTOCOL = PROJECT_ROOT / "configs/molecular_envelope_protocol.json"
SOURCE_NAMES = ["molecular_envelope_benchmark.py", "molecular_envelope.py", "molecular_methods.py", "molecular_benchmark.py"]


def scenarios(protocol: dict) -> list[dict]:
    cases = []
    defaults = protocol["stress_defaults"]
    for n in protocol["calibration_sizes"]:
        for rho in protocol["pipeline_correlations"]:
            for effect in protocol["effect_sizes"]:
                cases.append({**defaults, "name": f"n{n}_rho{rho}_z{effect}", "calibration_n": n,
                              "pipeline": "equicorrelation", "rho": rho, "effect": effect})
    cases += [{**defaults, **case} for case in protocol["stress_cases"]]
    return cases


def pipeline_covariance(k: int, case: dict, calibration: bool = False) -> np.ndarray:
    rho = case.get("calibration_rho", case["rho"]) if calibration else case["rho"]
    if case["pipeline"] == "equicorrelation":
        return equicorrelation(k, rho)
    return rho ** np.abs(np.arange(k)[:, None] - np.arange(k)[None, :])


def fixed_truth(g: int, s: int, case: dict, fraction: float) -> np.ndarray:
    mu = np.zeros((g, s))
    effect = case["effect"]
    if case.get("truth") == "global_null":
        return mu
    if case.get("truth") == "single_study_only":
        mu[:, 0] = effect
        return mu
    nrep = int(g * case.get("replicated_fraction", fraction))
    mu[:nrep//2, :2] = effect
    mu[nrep//2:nrep, :] = -effect
    end = min(g, nrep + int(g*.2))
    mu[nrep:end, 0] = effect
    end2 = min(g, end + int(g*.1))
    mu[end:end2, 0], mu[end:end2, 1] = effect, -effect
    return mu


def generate(rng: np.random.Generator, mu: np.ndarray, k: int, case: dict,
             root_p: np.ndarray, root_s: np.ndarray) -> np.ndarray:
    g, s = mu.shape
    gr = case["gene_rho"]
    noise = np.sqrt(1-gr)*rng.normal(size=(g,s,k)) + np.sqrt(gr)*rng.normal(size=(1,s,k))
    noise = np.einsum("ab,gbk,lk->gal", root_s, noise, root_p, optimize=True)
    if case["distribution"] == "t5":
        noise *= np.sqrt(3/rng.chisquare(5, size=(g,1,1)))
    return noise + mu[..., None]


def one_scenario(protocol: dict, case: dict, index: int, phase: str) -> dict:
    with threadpool_limits(limits=1):
        return _one_scenario(protocol, case, index, phase)


def _one_scenario(protocol: dict, case: dict, index: int, phase: str) -> dict:
    start = time.perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([protocol[f"seed_{phase}"], index]))
    g,s,k = (protocol[key] for key in ("genes","studies","pipelines"))
    alpha, delta, r = (protocol[key] for key in ("alpha","calibration_failure_budget","replication_r"))
    b, repetitions, n = protocol["null_draws"], protocol[f"repetitions_{phase}"], case["calibration_n"]
    cov = pipeline_covariance(k, case)
    root_p = covariance_root(cov)
    root_c = covariance_root(pipeline_covariance(k, case, True))
    root_s = covariance_root(equicorrelation(s, case["study_rho"]))
    mu = fixed_truth(g, s, case, protocol["replicated_fraction"])
    truth = np.stack(((mu > 0).sum(axis=1)>=r, (mu < 0).sum(axis=1)>=r), axis=1)
    records = {method: {"fdp":[], "power":[], "discoveries":[]} for method in protocol["methods"]}
    coverage, fallback = [], []
    for _rep in range(repetitions):
        z = generate(rng, mu, k, case, root_p, root_s)
        signed = np.stack((z,-z), axis=1)
        base_p = ndtr(-signed)
        bonf = np.minimum(1, k*base_p.min(axis=-1))
        adjusted = lambda p, a=alpha: fdr_adjust(partial_conjunction(p, r)) <= a
        decisions = {"bonferroni_PC_BY": adjusted(bonf), "fixed_PC_BY": adjusted(base_p[...,0])}
        # A fresh independent bank in EVERY repetition integrates bank uncertainty.
        oracle_bank = gaussian_bank(cov,b,rng)
        decisions["oracle_maxT_PC_BY"] = adjusted(rank_tail(oracle_bank,signed.max(axis=-1)))
        envelope_p, plugin_p, empirical_p = (np.empty((g,2,s)) for _ in range(3))
        good, fall = True, 0
        for study in range(s):
            x = rng.normal(size=(n,k)) @ root_c.T
            row_rho = case.get("calibration_row_rho",0)
            if row_rho:
                for row in range(1,n):
                    x[row] = row_rho*x[row-1]+np.sqrt(1-row_rho**2)*x[row]
            if case["distribution"] == "t5":
                x *= np.sqrt(3/rng.chisquare(5,size=(n,1)))
            envelope = correlation_envelope(x, delta/s)
            good &= bool(np.all(envelope.lower_matrix <= cov+1e-12))
            fall += int(envelope.fallback)
            envelope_p[:,:,study] = envelope_pvalues(signed[:,:,study,:],envelope,b,rng)
            estimated = np.corrcoef(x,rowvar=False)
            plugin_p[:,:,study] = rank_tail(gaussian_bank(estimated,b,rng),signed[:,:,study,:].max(axis=-1))
            empirical_p[:,:,study] = rank_tail(x.max(axis=1),signed[:,:,study,:].max(axis=-1))
        decisions["envelope_maxT_PC_BY"] = adjusted(envelope_p,alpha-delta)
        decisions["plugin_maxT_PC_BY"] = adjusted(plugin_p)
        decisions["rank_calibration_PC_BY"] = adjusted(empirical_p)
        study_e = p_to_e_mixture(base_p).mean(axis=-1)
        decisions["ePCH_eBH"] = ebh(np.sort(study_e,axis=-1)[...,:s-r+1].mean(axis=-1),alpha)
        # Split signs: do not pretend the two directions of one gene are independent.
        decisions["eFilter_bonferroni_split_signs"] = np.stack([
            efilter(bonf[:,sign,:],r,alpha/2,protocol["efilter_kappa"]) for sign in range(2)],axis=1)
        coverage.append(float(good))
        fallback.append(fall/s)
        for name, reject in decisions.items():
            count = int(reject.sum())
            records[name]["fdp"].append(float((reject & ~truth).sum()/max(1,count)))
            records[name]["power"].append(float((reject & truth).sum()/max(1,truth.sum())))
            records[name]["discoveries"].append(count)
    assumption_ok = case["distribution"] == "normal" and not case.get("calibration_rho") and not case.get("calibration_row_rho")
    rows = []
    for name, rec in records.items():
        difference = np.asarray(rec["power"])-np.asarray(records["bonferroni_PC_BY"]["power"])
        # Map paired differences from [-1,1] to [0,1] for a bounded CI.
        ci = summarize(((difference+1)/2).tolist())
        rows.append({"method":name,"fdr":summarize(rec["fdp"]),"power":summarize(rec["power"]),
                     "power_defined":bool(truth.sum()), "mean_discoveries":float(np.mean(rec["discoveries"])),
                     "paired_power_vs_bonferroni":{"mean":float(difference.mean()),"ci95":[2*x-1 for x in ci["ci95"]]},
                     "fdp_by_repetition":rec["fdp"],"power_by_repetition":rec["power"]})
    return {"scenario":case["name"],"settings":case,"candidate_assumptions":assumption_ok,
            "replicated_signed_truths":int(truth.sum()),"rows":rows,
            "envelope_simultaneous_coverage":summarize(coverage),"fallback_fraction":summarize(fallback),
            "elapsed_seconds":time.perf_counter()-start}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=["development","validation"],default="development")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sources={name:digest(Path(__file__).with_name(name)) for name in SOURCE_NAMES}
    fingerprint={"protocol_sha256":digest(PROTOCOL),"sources":sources}
    freeze=PROJECT_ROOT/"artifacts/molecular-envelope-freeze.json"
    if args.phase=="validation" and (not freeze.exists() or json.loads(freeze.read_text(encoding="utf-8"))["fingerprint"] != fingerprint):
        raise SystemExit("Validation requires matching completed development freeze; no silent retuning")
    cases=scenarios(protocol)
    if args.smoke:
        protocol[f"repetitions_{args.phase}"]=2
        protocol["null_draws"]=1023
        cases=[cases[0],cases[12],cases[-1]]
    effective={"protocol":protocol,"phase":args.phase,"fingerprint":fingerprint,"smoke":args.smoke}
    run_digest=hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()
    started=time.perf_counter()
    output=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one_scenario,protocol,case,i,args.phase):i for i,case in enumerate(cases)}
        for future in as_completed(futures):
            result=future.result()
            output.append((futures[future],result))
            print(f"{args.phase}: {result['scenario']} ({len(output)}/{len(cases)}) {result['elapsed_seconds']:.1f}s",flush=True)
    result={"schema_version":"1.0","created_at":datetime.now(UTC).isoformat(),"run_digest":run_digest,
            "provenance":"SIMULATION_NOT_PATIENT_DATA","effective":effective,"phase":args.phase,
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
            "scenarios":[record for _,record in sorted(output)],"elapsed_seconds":time.perf_counter()-started,
            "uncertainty":"Pointwise bounded empirical-Bernstein 95% intervals; fresh calibration and MC banks in each repetition. No simultaneous coverage across scenario panels.",
            "limits":["Gaussian unit-variance calibration assumptions are not certified on real expression data", "e-Filter CDF/calibrator conditions not certified", "Sparse discrete MC p-values may have zero power", "No novelty/publication/clinical guarantee"]}
    archive=PROJECT_ROOT/"artifacts/molecular-runs"/f"envelope-{args.phase}-{run_digest[:16]}.json"
    if not archive.exists():
        write_json(archive,result)
    if not args.smoke:
        write_json(PROJECT_ROOT/f"artifacts/molecular-envelope-{args.phase}.json",result)
        if args.phase=="development":
            write_json(freeze,{"fingerprint":fingerprint,"created_at":result["created_at"],"development_run":run_digest})
    print(json.dumps({"archive":str(archive),"seconds":result["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    main()
