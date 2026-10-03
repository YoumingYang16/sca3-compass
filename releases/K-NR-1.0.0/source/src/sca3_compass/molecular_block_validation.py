"""Retrospective leave-pair-out validation respecting all repeated tissues."""
from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .molecular_data import PROJECT_ROOT, digest, write_json
from .molecular_shrinkage import covariance_dictionary, fit_mixture, posterior

CONFIG=PROJECT_ROOT/"configs/molecular_block_validation.json"
REGIONS=["brainstem","cerebellum","cortex","striatum"]


def unit_partition(samples: list[dict], held: tuple[str,str]) -> tuple[list[int],list[int]]:
    if len(set(held))!=2 or any(not s.get("unit_id") for s in samples):
        raise ValueError("Two distinct resolved held-out units required")
    train=[i for i,s in enumerate(samples) if s["unit_id"] not in held]
    test=[i for i,s in enumerate(samples) if s["unit_id"] in held]
    if not test or {samples[i]["unit_id"] for i in train}&{samples[i]["unit_id"] for i in test}:
        raise ValueError("Unit leakage or absent test units")
    return train,test


def training_effects(x: np.ndarray, samples: list[dict], train: list[int]) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    effects,errors,residuals=[],[],[]
    for region in REGIONS:
        case=[i for i in train if samples[i]["tissue"]==region and samples[i]["genotype"]=="SCA3 transgenic"]
        control=[i for i in train if samples[i]["tissue"]==region and samples[i]["genotype"]=="wildtype"]
        if min(len(case),len(control))<3:
            raise ValueError("Insufficient training animals")
        effects.append(x[:,case].mean(axis=1)-x[:,control].mean(axis=1))
        errors.append(np.sqrt(x[:,case].var(axis=1,ddof=1)/len(case)+x[:,control].var(axis=1,ddof=1)/len(control)))
        residual={}
        for group in (case,control):
            centered=x[:,group]-x[:,group].mean(axis=1,keepdims=True)
            for j,i in enumerate(group):
                residual[samples[i]["unit_id"]]=centered[:,j]
        residuals.append(residual)
    b,se=np.stack(effects,axis=1),np.stack(errors,axis=1)
    keep=np.isfinite(b).all(axis=1)&np.isfinite(se).all(axis=1)&(se>1e-6).all(axis=1)
    corr=np.eye(4)
    for i,j in itertools.combinations(range(4),2):
        common=sorted(set(residuals[i])&set(residuals[j]))
        a=np.stack([residuals[i][u][keep] for u in common]).ravel()
        c=np.stack([residuals[j][u][keep] for u in common]).ravel()
        corr[i,j]=corr[j,i]=np.corrcoef(a,c)[0,1]
    eig,vec=np.linalg.eigh(corr)
    corr=vec@np.diag(np.maximum(eig,1e-6))@vec.T
    corr/=np.sqrt(np.outer(np.diag(corr),np.diag(corr)))
    corr=.5*corr+.5*np.eye(4)
    v=se[keep,:,None]*corr*se[keep,None,:]
    return b[keep],v,keep


def one_fold(held: tuple[str,str]) -> dict:
    with threadpool_limits(limits=1):
        config=json.loads(CONFIG.read_text(encoding="utf-8"))
        qc=json.loads((PROJECT_ROOT/"artifacts/molecular-qc.json").read_text(encoding="utf-8"))
        profile=next(d for d in qc["datasets"] if d["accession"]==config["accession"])
        path=PROJECT_ROOT/profile["derived_path"]
        if digest(path)!=profile["derived_sha256"]:
            raise ValueError("QC input hash mismatch")
        with np.load(path,allow_pickle=False) as a:
            counts,ids=a["counts"],a["samples"]
        metadata={s["sample_id"]:s for s in profile["samples"]}
        samples=[metadata[s] for s in ids]
        x=np.log2(counts/counts.sum(axis=0)*1e6+.5)
        train,test=unit_partition(samples,held)
        b,v,keep=training_effects(x,samples,train)
        predictions={"zero_effect":np.zeros_like(b),"unshrunk":b}
        convergence={}
        for name,identity in [("identity_mixture",True),("structured_mixture",False)]:
            priors,_=covariance_dictionary(b,v,config["prior_variance_scales"],identity_only=identity)
            fit=fit_mixture(b,v,priors,max_iter=200)
            predictions[name]=posterior(b,v,priors,fit["weights"])["mean"]
            convergence[name]={"converged":fit["converged"],"kkt":fit["simplex_kkt_residual"]}
        losses={name:[] for name in predictions}
        regions=[]
        for j,region in enumerate(REGIONS):
            pair=[[i for i in test if samples[i]["unit_id"]==unit and samples[i]["tissue"]==region] for unit in held]
            if any(len(indices)>1 for indices in pair):
                raise ValueError("Duplicate same-animal region")
            if not all(pair):
                continue
            target=x[keep,pair[0][0]]-x[keep,pair[1][0]]
            for name,prediction in predictions.items():
                losses[name].append(float(np.mean((prediction[:,j]-target)**2)))
            regions.append(region)
        return {"held_disease":held[0],"held_control":held[1],"training_animals":len({samples[i]["unit_id"] for i in train}),
                "training_samples":len(train),"held_samples":len(test),"genes":int(keep.sum()),"evaluated_regions":regions,
                "losses":{name:float(np.mean(values)) for name,values in losses.items()},"convergence":convergence}


def main() -> None:
    config=json.loads(CONFIG.read_text(encoding="utf-8"))
    qc=json.loads((PROJECT_ROOT/"artifacts/molecular-qc.json").read_text(encoding="utf-8"))
    profile=next(d for d in qc["datasets"] if d["accession"]==config["accession"])
    cases=sorted({s["unit_id"] for s in profile["samples"] if s["genotype"]=="SCA3 transgenic"})
    controls=sorted({s["unit_id"] for s in profile["samples"] if s["genotype"]=="wildtype"})
    folds=list(itertools.product(cases,controls))
    rows=[]
    with ProcessPoolExecutor(4) as pool:
        tasks=[pool.submit(one_fold,pair) for pair in folds]
        for future in as_completed(tasks):
            rows.append(future.result())
            print(f"Animal-blocked fit {len(rows)}/{len(folds)}",flush=True)
    rows.sort(key=lambda r:(r["held_disease"],r["held_control"]))
    summaries=[]
    for name in config["models"]:
        mse=np.array([row["losses"][name] for row in rows])
        raw=np.array([row["losses"]["unshrunk"] for row in rows])
        summaries.append({"method":name,"mean_fold_mse":float(mse.mean()),"median_fold_mse":float(np.median(mse)),
                          "paired_mean_difference_vs_unshrunk":float(np.mean(mse-raw)),"folds_better_than_unshrunk":int((mse<raw).sum())})
    result={"created_at":datetime.now(UTC).isoformat(),"provenance":"PUBLIC_REAL_DERIVED", "protocol":config,
            "protocol_sha256":digest(CONFIG),"source_sha256":digest(Path(__file__)),
            "shrinkage_source_sha256":digest(Path(__file__).with_name("molecular_shrinkage.py")),
            "qc_sha256":digest(PROJECT_ROOT/"artifacts/molecular-qc.json"),"input_sha256":profile["derived_sha256"],
            "folds":rows,"summaries":summaries,"all_fits_converged":all(c["converged"] for row in rows for c in row["convergence"].values())}
    write_json(PROJECT_ROOT/"artifacts/molecular-block-validation.json",result)
    print(json.dumps({"folds":len(rows),"all_converged":result["all_fits_converged"],"summary":summaries}))


if __name__=="__main__":
    main()
