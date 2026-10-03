"""Frozen nominations -> untouched, untreated public mouse-cohort evaluation.

The two stages are enforced by digests. This is transport evidence, not a
clinical or causal validation, and does not certify CE-maxT's assumptions.
"""
from __future__ import annotations

import argparse
import itertools
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata

from .molecular_annotation import annotation_path, parse_annotation
from .molecular_data import (
    PROJECT_ROOT,
    acquire,
    digest,
    parse_matrix_metadata,
    write_json,
)
from .molecular_methods import fdr_adjust
from .molecular_qc import attributes

CONFIG=PROJECT_ROOT/"configs/molecular_transport_protocol.json"
FREEZE=PROJECT_ROOT/"artifacts/molecular-transport-freeze.json"
REPORT=PROJECT_ROOT/"artifacts/molecular-transport.json"


def normalized_views(values: np.ndarray) -> list[np.ndarray]:
    x=np.asarray(values,float)
    if x.ndim!=2 or not np.isfinite(x).all() or (x<0).any() or (x.sum(axis=0)<=0).any():
        raise ValueError("Invalid provided expression matrix; no imputation")
    scaled=x/x.sum(axis=0)*1e6
    # Label-invariant rank transformation; effects have different units.
    rank=rankdata(scaled,axis=1,method="average")
    return [np.log2(scaled+.5),np.log2(scaled+1),ndtri((rank-.5)/x.shape[1])]


def contrast_statistics(x: np.ndarray, disease: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    disease=np.asarray(disease,bool)
    if x.ndim!=2 or x.shape[1]!=len(disease) or min(disease.sum(),(~disease).sum())<2:
        raise ValueError("Need >=2 samples in each group")
    a,b=x[:,disease],x[:,~disease]
    effect=a.mean(axis=1)-b.mean(axis=1)
    se=np.sqrt(a.var(axis=1,ddof=1)/a.shape[1]+b.var(axis=1,ddof=1)/b.shape[1])
    z=np.divide(effect,se,out=np.zeros_like(effect),where=se>1e-10)
    return effect,z


def exact_assignments(n: int, cases: int) -> np.ndarray:
    if not 1<=cases<n or n>18:
        raise ValueError("Unsupported exact enumeration size")
    combinations=list(itertools.combinations(range(n),cases))
    labels=np.zeros((len(combinations),n),dtype=bool)
    for row,index in enumerate(combinations):
        labels[row,list(index)]=True
    return labels


def permutation_contrasts(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float)
    labels=np.asarray(labels,bool)
    if x.ndim!=2 or labels.ndim!=2 or x.shape[1]!=labels.shape[1] or not np.isfinite(x).all():
        raise ValueError("Permutation matrix mismatch")
    n1=labels.sum(axis=1,keepdims=True)
    n0=labels.shape[1]-n1
    if (n1==0).any() or (n0==0).any():
        raise ValueError("Empty permutation group")
    weights=labels/n1-(~labels)/n0
    return weights@x.T


def holm(p: np.ndarray) -> np.ndarray:
    p=np.asarray(p,float)
    if p.ndim!=1 or not p.size or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid p-values")
    order=np.argsort(p,kind="stable")
    adjusted=np.empty_like(p)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return adjusted


def fingerprint() -> dict:
    paths=[CONFIG,Path(__file__),Path(__file__).with_name("molecular_annotation.py"),
           Path(__file__).with_name("molecular_data.py"),Path(__file__).with_name("molecular_qc.py"),
           Path(__file__).with_name("molecular_methods.py"),
           PROJECT_ROOT/"artifacts/molecular-qc.json",PROJECT_ROOT/"artifacts/molecular-shrinkage.json",
           PROJECT_ROOT/"data/processed/molecular/GSE107958-counts.npz",
           PROJECT_ROOT/"data/processed/molecular/GSE145613-counts.npz",
           PROJECT_ROOT/"data/processed/molecular/GSE107958-shrinkage.npz",
           PROJECT_ROOT/"data/raw/molecular/GSE261670/GSE261670_series_matrix.txt.gz",
           annotation_path(91),annotation_path(108)]
    return {str(path.relative_to(PROJECT_ROOT)):digest(path) for path in paths}


def metadata_selection(config: dict) -> list[dict]:
    metadata=parse_matrix_metadata(PROJECT_ROOT/"data/raw/molecular/GSE261670/GSE261670_series_matrix.txt.gz")
    selected=[s for s in metadata["samples"] if attributes(s).get("tissue")=="Cerebellum"
              and attributes(s).get("time")=="65 weeks" and not attributes(s).get("treatment")]
    if [s["sample_id"] for s in selected]!=config["external_sample_ids"]:
        raise ValueError("Metadata no longer matches frozen external eligibility")
    genotypes=[attributes(s).get("genotype") for s in selected]
    if genotypes.count("WT")!=6 or genotypes.count("Q84")!=6:
        raise ValueError("External groups must be exactly 6 WT and 6 Q84")
    return selected


def freeze_nominations() -> dict:
    if FREEZE.exists():
        frozen=json.loads(FREEZE.read_text(encoding="utf-8"))
        if frozen["fingerprint"]!=fingerprint():
            raise ValueError("Frozen source/input changed; refuse silent overwrite")
        return frozen
    config=json.loads(CONFIG.read_text(encoding="utf-8"))
    metadata_selection(config)
    qc=json.loads((PROJECT_ROOT/"artifacts/molecular-qc.json").read_text(encoding="utf-8"))
    _,symbols=parse_annotation(annotation_path(91))
    views,feature_sets=[],[]
    for accession in ("GSE107958","GSE145613"):
        profile=next(d for d in qc["datasets"] if d["accession"]==accession)
        path=PROJECT_ROOT/profile["derived_path"]
        if digest(path)!=profile["derived_sha256"]:
            raise ValueError("Development input is stale")
        with np.load(path,allow_pickle=False) as a:
            x,features,ids=a["counts"],a["features"],a["samples"]
        samples={s["sample_id"]:s for s in profile["samples"]}
        if accession=="GSE107958":
            subset=[i for i,s in enumerate(ids) if samples[s]["tissue"]=="cerebellum"]
            labels=np.array([samples[ids[i]]["genotype"]=="SCA3 transgenic" for i in subset])
            names=features.tolist()
        else:
            subset=[i for i,s in enumerate(ids) if samples[s]["age"]=="12 months"]
            labels=np.array([samples[ids[i]]["source_title"].startswith("KI_") for i in subset])
            names=[symbols.get(f.split(".")[0],"") for f in features]
        if len({name for name in names if name})!=sum(bool(name) for name in names):
            raise ValueError("Ambiguous development gene-symbol mapping")
        view=normalized_views(x[:,subset])
        stats=np.stack([contrast_statistics(v,labels)[1] for v in view],axis=1)
        views.append({name:stats[i] for i,name in enumerate(names) if name})
        feature_sets.append(set(views[-1]))
    with np.load(PROJECT_ROOT/"data/processed/molecular/GSE107958-shrinkage.npz",allow_pickle=False) as a:
        eb={str(f):float(a["posterior_mean"][i,1]/max(a["posterior_sd"][i,1],1e-10)) for i,f in enumerate(a["features"])}
    universe=sorted(feature_sets[0]&feature_sets[1]&set(eb))
    nominations={}
    for strategy in config["strategies"]:
        candidates=[]
        for gene in universe:
            stats=np.stack([v[gene] for v in views])
            if strategy=="single_study_z":
                signed=stats[0,0]
            elif strategy=="multiregion_EB":
                signed=eb[gene]
            else:
                selected=stats[:,0] if strategy=="two_study_consensus" else stats.ravel()
                if not (np.all(selected>0) or np.all(selected<0)):
                    continue
                signed=np.sign(selected[0])*np.min(np.abs(selected))
            if np.isfinite(signed) and signed!=0:
                candidates.append({"gene":gene,"direction":int(np.sign(signed)),"score":float(abs(signed)),"development_t":stats.tolist()})
        candidates.sort(key=lambda x:(-x["score"],x["gene"]))
        if len(candidates)<config["nomination_count"]:
            raise ValueError("Insufficient eligible nominations")
        nominations[strategy]=candidates[:config["nomination_count"]]
    frozen={"created_at":datetime.now(UTC).isoformat(),"fingerprint":fingerprint(),"protocol_id":config["protocol_id"],
            "expression_holdout_read":False,"development_shared_universe":len(universe),"nominations":nominations,
            "adjudication":{"GSE145613":"ATXN3 design corroborated by primary paper PMC8786755, methods: 5 WT and 5 304Q/304Q mice per age. Source ataxin-2 text preserved. No individual animal IDs beyond reported independent samples.",
                            "GSE107958":"Source counts/CPM unit conflict retained; transformed-expression ranking only; no count likelihood."}}
    write_json(FREEZE,frozen)
    return frozen


def acquire_external(config: dict) -> list[dict]:
    base="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE261nnn/GSE261670/suppl"
    listing=PROJECT_ROOT/"data/raw/molecular/GSE261670/filelist.txt"
    acquire(f"{base}/filelist.txt",listing)
    names={}
    for line in listing.read_text(encoding="utf-8").splitlines():
        fields=line.split("\t")
        if len(fields)>=2 and fields[0]=="File":
            names[fields[1].split("_")[0]]=fields[1]
    def download(sample_id):
        filename=names[sample_id]
        # The series filelist lists members of RAW.tar, not direct series URLs.
        # GEO exposes individual source files in each sample's suppl directory.
        sample_base=f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{sample_id[:-3]}nnn/{sample_id}/suppl"
        return {"sample_id":sample_id,**acquire(f"{sample_base}/{filename}",listing.parent/filename)}
    with ThreadPoolExecutor(3) as pool:
        receipts=list(pool.map(download,config["external_sample_ids"]))
    write_json(PROJECT_ROOT/"artifacts/molecular-transport-acquisition.json",{"files":receipts,"metadata_sha256":digest(listing.parent/"GSE261670_series_matrix.txt.gz")})
    return receipts


def read_external(receipts: list[dict]) -> tuple[pd.DataFrame,dict]:
    tx_map,symbols=parse_annotation(annotation_path(108))
    columns={}
    mapping=[]
    expected_index=None
    for file in receipts:
        path=PROJECT_ROOT/file["path"]
        if digest(path)!=file["sha256"]:
            raise ValueError("External source hash mismatch")
        with tarfile.open(path,"r:gz") as tar:
            candidates=[m for m in tar if m.isfile() and Path(m.name).name=="abundance.tsv"]
            if len(candidates)!=1:
                raise ValueError("Need exactly one abundance.tsv per biological sample")
            stream=tar.extractfile(candidates[0])
            if stream is None:
                raise ValueError("Missing abundance stream")
            frame=pd.read_csv(stream,sep="\t",usecols=["target_id","tpm"])
        tx=frame["target_id"].str.split(".").str[0]
        if tx.duplicated().any() or not np.isfinite(frame["tpm"]).all() or (frame["tpm"]<0).any():
            raise ValueError("Invalid transcript measurements")
        genes=tx.map(tx_map)
        fraction=float(genes.notna().mean())
        if fraction<.99:
            raise ValueError(f"Annotation mismatch ({fraction:.3%}); held analysis stopped without substituting annotation")
        frame=frame.assign(gene=genes)
        aggregated=frame.dropna(subset=["gene"]).groupby("gene")["tpm"].sum()
        aggregated=aggregated[aggregated.index.isin(symbols)]
        aggregated.index=[symbols[g] for g in aggregated.index]
        aggregated=aggregated.sort_index()
        if expected_index is None:
            expected_index=aggregated.index
        elif not aggregated.index.equals(expected_index):
            raise ValueError("External gene universes differ; no zero fill")
        columns[file["sample_id"]]=aggregated
        mapping.append({"sample_id":file["sample_id"],"transcripts":len(tx),"mapped_fraction":fraction,"total_tpm":float(frame["tpm"].sum())})
    return pd.DataFrame(columns),{"mapping":mapping,"annotation_sha256":digest(annotation_path(108))}


def evaluate_holdout() -> dict:
    if not FREEZE.exists():
        raise ValueError("Freeze nominations before accessing external expression")
    frozen=json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen["fingerprint"]!=fingerprint():
        raise ValueError("Holdout analysis refused: frozen code/input changed")
    if REPORT.exists():
        prior=json.loads(REPORT.read_text(encoding="utf-8"))
        if prior["freeze_sha256"]!=digest(FREEZE):
            raise ValueError("Existing holdout result belongs to another freeze")
        return prior
    config=json.loads(CONFIG.read_text(encoding="utf-8"))
    metadata=metadata_selection(config)
    receipts=acquire_external(config)
    # Record access BEFORE expression parsing, including any failed attempt.
    access=PROJECT_ROOT/"artifacts/molecular-transport-access.json"
    if not access.exists():
        write_json(access,{"first_expression_access_at":datetime.now(UTC).isoformat(),"freeze_sha256":digest(FREEZE),"protocol_sha256":digest(CONFIG)})
    expression,qc=read_external(receipts)
    disease=np.array([attributes(s)["genotype"]=="Q84" for s in metadata])
    labels=exact_assignments(12,6)
    observed_index=int(np.flatnonzero(np.all(labels==disease,axis=1))[0])
    all_signed=sorted({(n["gene"],n["direction"]) for ns in frozen["nominations"].values() for n in ns})
    rng=np.random.default_rng(config["bootstrap_seed"])
    results=[]
    gene_results=[]
    for gene,direction in all_signed:
        if gene not in expression.index:
            gene_results.append({"gene":gene,"direction":direction,"external_effect":None,"signed_p":1.0,"missing":True})
            continue
        x=np.log2(expression.loc[[gene]].to_numpy()+.5)
        permutations=permutation_contrasts(x,labels)[:,0]*direction
        observed=permutations[observed_index]
        p=float(np.mean(permutations>=observed-1e-12))
        gene_results.append({"gene":gene,"direction":direction,"external_effect":float(observed*direction),"signed_p":p})
    adjusted=fdr_adjust(np.array([g["signed_p"] for g in gene_results]))
    for record,q in zip(gene_results,adjusted,strict=True):
        record["BY_q_union_signed_family"]=float(q)
    for strategy,signature in frozen["nominations"].items():
        available=[s for s in signature if s["gene"] in expression.index]
        missing=[s["gene"] for s in signature if s["gene"] not in expression.index]
        if len(missing)/len(signature)>config["maximum_missing_fraction"]:
            raise ValueError("Too many missing held-out genes; no replacement nominations allowed")
        x=np.log2(expression.loc[[s["gene"] for s in available]].to_numpy()+.5)
        signs=np.array([s["direction"] for s in available])
        sd=x.std(axis=1,ddof=1)
        if (sd<=1e-12).any():
            raise ValueError("Constant nominated gene; holdout requires explicit numerical audit")
        standardized=(x-x.mean(axis=1,keepdims=True))/sd[:,None]
        score=(signs[:,None]*standardized).mean(axis=0)
        perms=permutation_contrasts(score[None,:],labels)[:,0]
        observed=float(perms[observed_index])
        p=float(np.mean(perms>=observed-1e-12))
        effects=x[:,disease].mean(axis=1)-x[:,~disease].mean(axis=1)
        case_score,control_score=score[disease],score[~disease]
        boot=case_score[rng.integers(0,6,size=(2000,6))].mean(axis=1)-control_score[rng.integers(0,6,size=(2000,6))].mean(axis=1)
        influence=[]
        for i,s in enumerate(metadata):
            keep=np.arange(12)!=i
            effect=float(score[keep & disease].mean()-score[keep & ~disease].mean())
            influence.append({"omitted_sample":s["sample_id"],"effect":effect})
        results.append({"strategy":strategy,"nominated":len(signature),"measured":len(available),"missing":missing,
                        "score_effect":observed,"exact_permutation_p":p,"bootstrap_percentile_ci95":np.quantile(boot,[.025,.975]).tolist(),
                        "direction_concordance":float(np.mean(signs*effects>0)),"leave_one_sample":influence,
                        "sample_scores":[{"sample_id":s["sample_id"],"genotype":attributes(s)["genotype"],"score":float(score[i])} for i,s in enumerate(metadata)]})
    primary_holm=holm(np.array([r["exact_permutation_p"] for r in results]))
    for record,p in zip(results,primary_holm,strict=True):
        record["holm_p_four_strategies"]=float(p)
    target=PROJECT_ROOT/"data/processed/molecular/GSE261670-external-gene-tpm.npz"
    np.savez_compressed(target,expression=expression.to_numpy(),features=np.array(expression.index,dtype=str),samples=np.array(expression.columns,dtype=str))
    report={"created_at":datetime.now(UTC).isoformat(),"provenance":"PUBLIC_REAL_DERIVED",
            "freeze_sha256":digest(FREEZE),"source_sha256":digest(Path(__file__)),"protocol_sha256":digest(CONFIG),
            "accession":"GSE261670","sample_count":12,"groups":{"WT":6,"Q84":6},"tissue":"cerebellum","age_weeks":65,
            "external_stage":"untouched expression after frozen nominations; published biology not blinded",
            "permutation_assignments":len(labels),"minimum_attainable_p":1/len(labels),"qc":qc,
            "expression_path":str(target.relative_to(PROJECT_ROOT)),"expression_sha256":digest(target),
            "strategy_results":results,"gene_results":gene_results,"signed_gene_family_size":len(gene_results),
            "signed_genes_BY_005":sum(g["BY_q_union_signed_family"]<=.05 for g in gene_results),
            "limits":["Exchangeability within the untreated age/tissue stratum is assumed, not established by random genotype assignment", "Sex/litter identifiers unavailable in public sample annotations", "Cross-model and age transport, not identical biological effects", "Single external cohort, not clinical validation", "Primary signatures are tested jointly with Holm; gene dependence preserved by whole-sample permutations", "Bootstrap intervals are descriptive with only 6 samples per group", "Existing oligodendrocyte biology known before this evaluation", "This does not validate CE-maxT's Gaussian calibration assumptions"]}
    write_json(REPORT,report)
    return report


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["freeze","evaluate"])
    args=parser.parse_args()
    result=freeze_nominations() if args.stage=="freeze" else evaluate_holdout()
    print(json.dumps({"stage":args.stage,"file":str(FREEZE if args.stage=="freeze" else REPORT),"strategies":list(result.get("nominations",{})),"results":result.get("strategy_results",[])}),flush=True)


if __name__=="__main__":
    main()
