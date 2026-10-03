"""Persist mechanism diagnostic; oracle parameters are never deployable input.

Fresh development replay of early inline reasoning, not an exact archive of
the original unsaved exploratory run and not independent confirmation.
"""
import json
import time
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import t
from threadpoolctl import threadpool_limits

from sca3_compass.molecular_data import PROJECT_ROOT, digest, write_json
from sca3_compass.molecular_envelope_benchmark import fixed_truth
from sca3_compass.molecular_methods import covariance_root, equicorrelation, partial_conjunction, fdr_adjust


def main():
    path=PROJECT_ROOT/"artifacts/robustness/R0002-oracle-mechanism-replay.json"
    if path.exists():
        raise SystemExit("Preserve existing diagnostic")
    started=time.perf_counter()
    cases=[]
    with threadpool_limits(limits=1):
        for i,(kind,rho) in enumerate((d,r) for d in ["normal","t5"] for r in [.1,.8,.95]):
            rng=np.random.default_rng(np.random.SeedSequence([491,i]))
            rs=equicorrelation(4,.65)
            mu=fixed_truth(256,4,{"effect":3.5},.2)
            truth=np.stack(((mu>0).sum(1)>=2,(mu<0).sum(1)>=2),1)
            records={}
            for rep in range(300):
                raw=rng.normal(size=(256,4,6))
                z=np.einsum("ab,gbk,lk->gal",covariance_root(rs),raw,covariance_root(equicorrelation(6,rho)))
                if kind=="t5":
                    z*=np.sqrt(3/rng.chisquare(5,size=(256,1,1)))
                z+=mu[:,:,None]
                means=np.stack((z,-z),1).mean(-1)
                residual=z-z.mean(-1,keepdims=True)
                q=np.einsum("gsk,st,gtk->g",residual,np.linalg.inv(rs),residual)/(1-rho)
                local=(residual**2).sum(-1)/(1-rho)
                v=(1+5*rho)/6
                if kind=="normal":
                    marginal=ndtr(-means/np.sqrt(v))
                    all_conditional=local_conditional=marginal
                else:
                    marginal=t.sf(means/np.sqrt(v*.6),5)
                    all_conditional=t.sf(means/np.sqrt(v*(3+q[:,None,None])/25),25)
                    local_conditional=t.sf(means/np.sqrt(v*(3+local[:,None,:])/10),10)
                angular=t.sf(means/np.sqrt(v*q[:,None,None]/20),20)
                for name,p in [("oracle_marginal",marginal),("oracle_conditional_all",all_conditional),
                               ("oracle_conditional_local",local_conditional),("oracle_radial_studentized",angular)]:
                    reject=fdr_adjust(partial_conjunction(p,2))<=.05
                    record=records.setdefault(name,{"fdp":[],"power":[]})
                    record["fdp"].append(float((reject&~truth).sum()/max(1,reject.sum())))
                    record["power"].append(float((reject&truth).sum()/truth.sum()))
            cases.append({"distribution":kind,"rho":rho,"records":records})
    result={"phase":"DEVELOPMENT_ORACLE_MECHANISM_ONLY","seed":491,"repetitions":300,
        "source_sha256":digest(Path(__file__)),"cases":cases,"elapsed_seconds":time.perf_counter()-started,
        "provenance":"SIMULATION_NOT_PATIENT_DATA","limitation":"Known parameters, ineligible as deployable method; fresh replay after unsaved preliminary exploration"}
    write_json(path,result)
    registry_path=path.parent/"EXPERIMENT_REGISTRY.json"
    registry=json.loads(registry_path.read_text(encoding="utf-8"))
    registry["experiments"].append({"id":"R0002","status":"completed","completed_at":datetime.now(UTC).isoformat(),
        "purpose":result["phase"],"result_sha256":digest(path),"source_sha256":result["source_sha256"],
        "elapsed_seconds":result["elapsed_seconds"],"historical_inline_results_preserved":False,
        "note":"Fresh reproducible replay, not reconstruction of every earlier unsaved inline random stream"})
    write_json(registry_path,registry)
    for case in cases:
        print(case["distribution"],case["rho"],{n:[np.mean(v["fdp"]),np.mean(v["power"])] for n,v in case["records"].items()})


if __name__=="__main__":
    main()
