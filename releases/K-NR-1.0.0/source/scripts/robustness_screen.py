"""Versioned development screen; NEVER a confirmatory acceptance run."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import t
from threadpoolctl import threadpool_limits

from sca3_compass.molecular_benchmark import summarize
from sca3_compass.molecular_data import PROJECT_ROOT, digest
from sca3_compass.robustness_io import write_json,content_digest,load_case_checkpoints,write_json_gzip,compact_case_result
from sca3_compass.molecular_envelope_benchmark import fixed_truth
from sca3_compass.molecular_methods import covariance_root, equicorrelation, fdr_adjust, partial_conjunction, ebh
from sca3_compass.robustness_methods import evaluate_candidates
from sca3_compass.robustness_mc import additional_comparators
from sca3_compass.robustness_registry import update_registry
from sca3_compass.robustness_softscore import softscore_pc
from sca3_compass.robustness_methods import contrasts
from sca3_compass.robustness_weighting import power_radial_weight
from sca3_compass.robustness_calibrators import focused_calibrator
from sca3_compass.robustness_loading import loading_candidates
from sca3_compass.robustness_prior import energy_prior_candidates
from sca3_compass.robustness_pattern_test import pattern_test_candidates
from sca3_compass.robustness_geometry import fit_fold_geometries
from sca3_compass.robustness_dispatch import bounded_results


def radial(rng, shape, kind):
    if kind=="normal":
        return np.ones(shape)
    if kind.startswith("t"):
        df=float(kind[1:])
        # df<=2 uses scatter=1, NOT nonexistent unit variance.
        return np.sqrt((df-2 if df>2 else df)/rng.chisquare(df,size=shape))
    if kind=="lognormal":
        return np.exp(rng.normal(size=shape)-1)
    if kind=="mixture":
        return np.where(rng.random(shape)<.9,.5,2.)/np.sqrt(.9*.25+.1*4)
    raise ValueError("Unknown radial distribution")


def simulation_equicorrelation(dimension, rho):
    """Full positive-definite compound range, preserving old positive draws.

    The frozen historical helper only accepts nonnegative correlations.
    Extending the SIMULATOR must not silently clip a negative stress to zero
    or modify any historical method. Singular boundaries are not used here.
    """
    if (isinstance(rho,(bool,np.bool_)) or not np.isscalar(rho)
            or not np.isfinite(rho) or not -1/(dimension-1)<rho<1):
        raise ValueError('Simulation compound correlation must be strictly positive definite')
    if rho>=0:
        return equicorrelation(dimension,rho)
    matrix=np.full((dimension,dimension),float(rho))
    np.fill_diagonal(matrix,1.)
    return matrix


def data(rng, case, g=256,s=4,k=6):
    if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in [g,s,k])
            or g<4 or s!=4 or k<2):
        raise ValueError('Generator requires integer G>=4, S=4 and K>=2')
    mu=fixed_truth(g,s,{"effect":case["effect"],"truth":case.get("truth"),
        "replicated_fraction":case.get("fraction",.2)},.2)
    layout=case.get('effect_layout')
    if layout=='permuted':
        mu=mu[:,[2,0,3,1]]
    elif layout in ['random_gene','two_patterns']:
        order=(np.argsort(rng.random((g,s)),axis=1) if layout=='random_gene' else
            np.where((rng.random(g)<.5)[:,None],np.arange(s),np.array([2,3,0,1])))
        mu=np.take_along_axis(mu,order,axis=1)
    elif layout is not None:
        raise ValueError('Unknown effect layout')
    if case.get('effect_jitter'):
        mu*=np.exp(case['effect_jitter']*rng.normal(size=mu.shape)-case['effect_jitter']**2/2)
    rs=simulation_equicorrelation(s,case.get("study_rho",.65))
    if case.get("pipeline_kind")=="toeplitz":
        rp=case["rho"]**np.abs(np.arange(k)[:,None]-np.arange(k)[None,:])
    else:
        rp=simulation_equicorrelation(k,case["rho"])
    gr=case.get("gene_rho",0)
    raw=np.sqrt(1-gr)*rng.normal(size=(g,s,k))+np.sqrt(gr)*rng.normal(size=(1,s,k))
    noise=np.einsum("ab,gbk,lk->gal",covariance_root(rs),raw,covariance_root(rp),optimize=True)
    radial_shape=(g,s,1) if case.get("study_specific_radial") else (g,1,1)
    noise*=radial(rng,radial_shape,case["distribution"])
    if case.get("pipeline_heterogeneity")=='gene_random':
        loadings=np.exp(.6*rng.normal(size=(g,1,k)))
        loadings/=loadings.mean(axis=-1,keepdims=True)
    elif case.get("pipeline_heterogeneity")=='study_random':
        loadings=np.exp(.6*rng.normal(size=(g,s,k)))
        loadings/=loadings.mean(axis=-1,keepdims=True)
    elif case.get("pipeline_heterogeneity"):
        reference=np.array([.2,.5,.8,1.2,1.5,1.8])
        loadings=reference if k==6 else np.interp(np.linspace(0,1,k),np.linspace(0,1,6),reference)
        if k!=6:
            loadings/=loadings.mean()
    else:
        loadings=np.ones(k)
    effects=np.broadcast_to(mu[...,None]*loadings,noise.shape).copy()
    if case.get('null_pipeline_shift'):
        if case.get('truth') not in ['global_null','single_study_only']:
            raise ValueError('Composite-null shift diagnostic requires a null-only truth case')
        reference=np.array([0,0,.2,.8,2.,4.])
        shift=reference if k==6 else np.interp(np.linspace(0,1,k),np.linspace(0,1,6),reference)
        effects-=case['null_pipeline_shift']*(mu<=0)[...,None]*shift
    z=effects+noise
    rc=simulation_equicorrelation(k,case["calibration_rho"]) if "calibration_rho" in case else rp
    calibration=rng.normal(size=(s,case["n"],k))@covariance_root(rc).T
    calibration*=radial(rng,(s,case["n"],1),case.get("calibration_distribution",case["distribution"]))
    ar=case.get("calibration_row_rho",0)
    if ar:
        for row in range(1,case["n"]):
            calibration[:,row,:]=ar*calibration[:,row-1,:]+np.sqrt(1-ar*ar)*calibration[:,row,:]
    if case.get("skew"):
        # Centered, finite-variance additive lognormal common component. This
        # is a deliberate NON-elliptical stress, not a radial t model.
        skew=lambda shape:(np.exp(rng.normal(size=shape)) - np.exp(.5))/np.sqrt(np.e*(np.e-1))
        z+=case["skew"]*skew((g,1,1))
        calibration+=case["skew"]*skew((s,case["n"],1))
    # Composite component null: all pipeline means are nonpositive (or
    # nonnegative for the opposite claim). Derive truth from ACTUAL means,
    # including deliberate shifts, not the pre-shift latent generator label.
    truth=np.stack((np.any(effects>0,axis=-1).sum(1)>=2,
                    np.any(effects<0,axis=-1).sum(1)>=2),axis=1)
    return z,calibration,truth


def one(case,index,repetitions,seed,mc=False,draws=131071,profile="full",soft=False,loading=False,prior=False,domain=False,patterns=False,transport_patterns=False,pilot_patterns=False,joint_patterns=False,block_patterns=False,loading_patterns=False,loading_audit_patterns=False,predictive_patterns=False,directional_patterns=False,pilot_selection_patterns=False,rank_budget_patterns=False,continuous_patterns=False):
    with threadpool_limits(limits=1):
        started=time.perf_counter()
        rng=np.random.default_rng(np.random.SeedSequence([seed,index]))
        records={}
        diagnostics=[]
        for rep in range(repetitions):
            z,x,truth=data(rng,case,g=case.get('genes',256),k=case.get('pipelines',6))
            p,diag=evaluate_candidates(z,x,profile)
            if prior or patterns or pilot_patterns or block_patterns:
                extra=energy_prior_candidates(z,x,diag)
                p.update(extra)
                for name,value in extra.items():
                    for fraction in [0.,.2]:
                        p[f'capped_{name[:-3]}_f{fraction}_eBH']=focused_calibrator(value,2*len(z),.001,fraction,cap=max(1,len(z)//4))
            if patterns or pilot_patterns or block_patterns:
                p.update(pattern_test_candidates(z,diag,p['target_only_weighted_cone_PC'],pilot=pilot_patterns,block=block_patterns,predictive=predictive_patterns,directional=directional_patterns,pilot_selection=pilot_selection_patterns,rank_budget=rank_budget_patterns))
            if transport_patterns or joint_patterns or continuous_patterns:
                for scale_mode in (['shrink_fcentral','transport_fcentral','transport_pooled_fcentral','anchored_fcentral'] if transport_patterns else [])+(['joint_pattern','anchored_joint_pattern','central_capped_joint_pattern','free_central_capped_joint_pattern'] if joint_patterns else [])+(['continuous_pattern','continuous_floor_pattern'] if continuous_patterns else []):
                    extra_diag={'fit':diag['fit'],'shape':diag['shape']}
                    extra=energy_prior_candidates(z,x,extra_diag,scale_mode=scale_mode)
                    extra.update(pattern_test_candidates(z,extra_diag,extra['target_only_weighted_cone_PC'],pilot=pilot_patterns,block=block_patterns,predictive=predictive_patterns,directional=directional_patterns,pilot_selection=pilot_selection_patterns,rank_budget=rank_budget_patterns))
                    for name,value in extra.items():
                        p[f'{scale_mode}_{name}']=value
                        if name.startswith('target_only'):
                            p[f'capped_{scale_mode}_{name[:-3]}_eBH']=focused_calibrator(value,2*len(z),.001,0.,cap=max(1,len(z)//4))
                    diag[f'{scale_mode}_combined']=extra_diag
            for audit_mode in (['legacy'] if loading_patterns else [])+(['intersection','veto','focused_veto'] if loading_audit_patterns else []):
                geometries,geometry_receipt=fit_fold_geometries(z,diag,audit_mode=audit_mode)
                for scale_mode in [None,'joint_pattern']+(['transport_pooled_fcentral'] if loading_audit_patterns else []):
                    extra_diag={'fit':diag['fit'],'shape':diag['shape'],'loading_geometry':geometry_receipt}
                    extra=energy_prior_candidates(z,x,extra_diag,scale_mode=scale_mode,geometries=geometries)
                    extra.update(pattern_test_candidates(z,extra_diag,extra['target_only_weighted_cone_PC'],
                        pilot=pilot_patterns,block=block_patterns,geometries=geometries,predictive=predictive_patterns,directional=directional_patterns,pilot_selection=pilot_selection_patterns,rank_budget=rank_budget_patterns))
                    prefix='geometry'+('' if audit_mode=='legacy' else '_'+audit_mode)
                    prefix+='_'+('fixed' if scale_mode is None else 'joint' if scale_mode=='joint_pattern' else 'pooled')
                    for name,value in extra.items():
                        p[f'{prefix}_{name}']=value
                        if name.startswith('target_only'):
                            p[f'capped_{prefix}_{name[:-3]}_eBH']=focused_calibrator(value,2*len(z),.001,0.,cap=max(1,len(z)//4))
                    diag[f'{prefix}_combined']=extra_diag
            if domain:
                for mode in ['gated_point','gated_upper','always_point','gated_fcentral','always_fcentral','shrink_fcentral']:
                    domain_diag={'fit':diag['fit'],'shape':diag['shape']}
                    extra=energy_prior_candidates(z,x,domain_diag,scale_mode=mode)
                    for name,value in extra.items():
                        if name.startswith('target_only'):
                            p[f'{mode}_{name}']=value
                            p[f'capped_{mode}_{name[:-3]}_eBH']=focused_calibrator(value,2*len(z),.001,0.,cap=max(1,len(z)//4))
                    diag[mode]=domain_diag['energy_prior']
            if loading:
                for forced in [False,True]:
                    extra,loading_diag=loading_candidates(z,diag,p['conditional_cone_PC'],p['power_weighted_cone_PC'],force=forced,base_baselines=p)
                    prefix='forced_' if forced else ''
                    for name,value in extra.items():
                        p[prefix+name]=value
                        for fraction in [0.,.2]:
                            p[f'capped_{prefix}{name[:-3]}_f{fraction}_eBH']=focused_calibrator(value,2*len(z),.001,fraction,cap=max(1,len(z)//4))
                    diag[prefix+'loading']=loading_diag
            if soft:
                fit=diag['fit']
                shapes=[np.asarray(info['matrix']) for info in diag['shape']]
                y=z@contrasts(z.shape[-1])
                q=np.empty(len(z))
                for fold,shape in enumerate(shapes):
                    held=np.arange(len(z))%2==fold
                    q[held]=np.einsum('gsk,st,gtk->g',y[held],np.linalg.inv(shape),y[held])/(1-fit['rho'])
                d=z.shape[1]*(z.shape[2]-1)
                v=(1+(z.shape[2]-1)*fit['rho'])/z.shape[2]
                df=np.inf if fit['gaussian_bic_selected'] else fit['df']+d
                variance=np.full(len(z),v*fit['scatter']) if np.isinf(df) else v*(fit['df']*fit['scatter']+q)/df
                mean=np.stack((z.mean(-1),-z.mean(-1)),1)
                extra,soft_diag=softscore_pc(mean,variance,shapes,df,
                    np.random.default_rng(np.random.SeedSequence([seed,index,rep,2025])),draws)
                weight=np.ones(len(z)) if np.isinf(df) else power_radial_weight(q,d,fit['df'],fit['scatter'],v)[0]
                for name,value in extra.items():
                    p[f'soft_{name}_PC']=value
                    weighted=np.minimum(1,value/weight[:,None])
                    p[f'soft_weighted_{name}_PC']=weighted
                    for fraction in [0.,.2,.5]:
                        p[f'capped_soft_weighted_{name}_f{fraction}_eBH']=focused_calibrator(weighted,2*len(z),.001,fraction,cap=max(1,len(z)//4))
                    p[f'focused_soft_weighted_{name}_eBH']=focused_calibrator(weighted,2*len(z),.001,.8)
                diag['softscore']=soft_diag
            if mc:
                mc_rng=np.random.default_rng(np.random.SeedSequence([seed,index,rep,8901]))
                extra,mc_diagnostics=additional_comparators(z,x,diag,mc_rng,draws)
                p.update(extra)
                diag["monte_carlo"]=mc_diagnostics
            # Correct model-parameter references are excluded from deployment.
            k=z.shape[-1]
            v=(1+(k-1)*case["rho"])/k
            signed=np.stack((z,-z),1)
            oracle_eligible=not any(case.get(key) for key in ["pipeline_kind","pipeline_heterogeneity","skew"])
            if oracle_eligible and case["distribution"]=="normal":
                p["oracle_marginal_mean"]=ndtr(-signed.mean(-1)/np.sqrt(v))
            elif oracle_eligible and case["distribution"].startswith("t") and float(case["distribution"][1:])>2:
                df=float(case["distribution"][1:])
                p["oracle_marginal_mean"]=t.sf(signed.mean(-1)/np.sqrt(v*(df-2)/df),df)
            diagnostics.append(diag)
            for name,values in p.items():
                alpha=.045 if name in ["angular_envelope","gaussian_CE_original"] else .05
                reject=(values if name.endswith("_decision") else ebh(values,alpha) if name.endswith("_eBH") else
                        fdr_adjust(values if name.endswith("_PC") else partial_conjunction(values,2))<=alpha)
                entry=records.setdefault(name,{"fdp":[],"power":[],"discoveries":[]})
                entry["fdp"].append(float(np.sum(reject&~truth)/max(1,reject.sum())))
                entry["power"].append(float(np.sum(reject&truth)/max(1,truth.sum())))
                entry["discoveries"].append(int(reject.sum()))
        return {"case":case,"repetitions":repetitions,"elapsed_seconds":time.perf_counter()-started,
            "dimensions":{"genes":len(z),"studies":z.shape[1],"pipelines":z.shape[2],"signed_family":2*len(z)},
            "power_defined":bool(truth.sum()),"replicated_signed_truths":int(truth.sum()),
            "rows":[{"method":name,"fdr":summarize(r["fdp"]),"power":summarize(r["power"]),
                "fdp_q95":float(np.quantile(r["fdp"],.95)),"fdp_q99":float(np.quantile(r["fdp"],.99)),
                "fdp_by_repetition":r["fdp"],"power_by_repetition":r["power"],
                "mean_discoveries":float(np.mean(r["discoveries"]))} for name,r in records.items()],
            "diagnostics":diagnostics}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",default="R0003")
    parser.add_argument("--repetitions",type=int,default=60)
    parser.add_argument("--seed",type=int,default=732601)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--config")
    parser.add_argument("--mc",action="store_true")
    parser.add_argument("--soft",action="store_true")
    parser.add_argument("--loading",action="store_true")
    parser.add_argument("--prior",action="store_true")
    parser.add_argument("--domain",action="store_true")
    parser.add_argument("--patterns",action="store_true")
    parser.add_argument("--transport-patterns",action="store_true")
    parser.add_argument("--pilot-patterns",action="store_true")
    parser.add_argument("--joint-patterns",action="store_true")
    parser.add_argument("--block-patterns",action="store_true")
    parser.add_argument("--loading-patterns",action="store_true")
    parser.add_argument("--loading-audit-patterns",action="store_true")
    parser.add_argument("--predictive-patterns",action="store_true")
    parser.add_argument("--directional-patterns",action="store_true")
    parser.add_argument("--pilot-selection-patterns",action="store_true")
    parser.add_argument("--rank-budget-patterns",action="store_true")
    parser.add_argument("--continuous-patterns",action="store_true")
    parser.add_argument("--compact",action="store_true",help="Lossless compressed raw checkpoints and compact long diagnostic arrays")
    parser.add_argument("--resume",action="store_true",help="Resume only source/protocol-identical per-case checkpoints")
    parser.add_argument("--draws",type=int,default=131071)
    parser.add_argument("--profile",choices=["full","core","frontier"],default="full")
    args=parser.parse_args()
    if args.predictive_patterns and not args.pilot_patterns:
        parser.error('--predictive-patterns requires --pilot-patterns')
    if args.directional_patterns and not args.pilot_patterns:
        parser.error('--directional-patterns requires --pilot-patterns')
    if args.pilot_selection_patterns and not args.pilot_patterns:
        parser.error('--pilot-selection-patterns requires --pilot-patterns')
    if args.rank_budget_patterns and not args.pilot_patterns:
        parser.error('--rank-budget-patterns requires --pilot-patterns')
    cases=([{"name":f"{d}_rho{rho}","distribution":d,"rho":rho,"n":64,"effect":3.5}
            for d in ["normal","t5","t3","lognormal"] for rho in [.1,.8,.95]] if not args.config
           else json.loads(Path(args.config).read_text(encoding="utf-8"))["cases"])
    sources=[Path(__file__),PROJECT_ROOT/"src/sca3_compass/robustness_methods.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_conjunction.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_mc.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_limma.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_registry.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_io.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_evidence.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_universal.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_cone.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_weighting.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_calibrators.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_adaptive_cone.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_softscore.py",
        PROJECT_ROOT/"src/sca3_compass/robustness_loading.py",
        PROJECT_ROOT/"src/sca3_compass/molecular_envelope.py",
        PROJECT_ROOT/"src/sca3_compass/molecular_methods.py",PROJECT_ROOT/"src/sca3_compass/molecular_envelope_benchmark.py"]
    # Include transitive local sources; earlier archives pinned the leading
    # methods only. This closes that provenance gap for future experiments.
    sources=list(dict.fromkeys([*sources,*sorted((PROJECT_ROOT/'src/sca3_compass').glob('*.py'))]))
    settings={"phase":"DEVELOPMENT_ONLY","run_id":args.run_id,"seed":args.seed,"repetitions":args.repetitions,
              "cases":cases,"source_sha256":{str(p.relative_to(PROJECT_ROOT)):digest(p) for p in sources},
              "provenance":"SIMULATION_NOT_PATIENT_DATA","independent_confirmation":False,
              "mc":args.mc,"soft":args.soft,"loading":args.loading,"prior":args.prior,"domain":args.domain,"patterns":args.patterns,"transport_patterns":args.transport_patterns,"pilot_patterns":args.pilot_patterns,"joint_patterns":args.joint_patterns,"block_patterns":args.block_patterns,"loading_patterns":args.loading_patterns,"loading_audit_patterns":args.loading_audit_patterns,"predictive_patterns":args.predictive_patterns,"directional_patterns":args.directional_patterns,"compact":args.compact,"draws":args.draws,"profile":args.profile,
              "pilot_selection_patterns":args.pilot_selection_patterns,
              "rank_budget_patterns":args.rank_budget_patterns,
              "continuous_patterns":args.continuous_patterns,
              "execution_dispatch":"bounded_worker_count_backlog_v1",
              "calibration_information":"All deployable candidates may pool S*n centered null vectors under common pipeline/radial law; no oracle tail labels"}
    target=PROJECT_ROOT/f"artifacts/robustness/{args.run_id}-screen.json"
    registry_path=PROJECT_ROOT/"artifacts/robustness/EXPERIMENT_REGISTRY.json"
    registry=json.loads(registry_path.read_text(encoding="utf-8"))
    prior=[r for r in registry['experiments'] if r['id']==args.run_id]
    if target.exists():
        raise SystemExit("Completed output exists; it must not be overwritten")
    checkpoints=target.parent/f'{args.run_id}-cases'
    output=[]
    if args.resume:
        if len(prior)!=1 or json.loads(target.with_suffix('.protocol.json').read_text(encoding='utf-8'))!=settings:
            raise SystemExit('Resume requires identical existing protocol, source hashes and run ID')
        transform=(lambda result,path:compact_case_result(result,path.relative_to(target.parent),audit_only=True)) if args.compact else None
        output=load_case_checkpoints(checkpoints,settings,transform=transform)
        entry={**prior[0],'status':'running','resume_count':prior[0].get('resume_count',0)+1,
            'previous_attempts':prior[0].get('previous_attempts',[])+[
                {k:prior[0].get(k) for k in ['status','failure','failure_type','elapsed_seconds']}],
            'resumed_at':datetime.now(UTC).isoformat(),'resumed_completed_cases':len(output)}
    else:
        if prior or checkpoints.exists():
            raise SystemExit("Run ID already used; preserve it and choose a new explicit run ID")
        entry={"id":args.run_id,"status":"running","started_at":datetime.now(UTC).isoformat(),"settings":settings}
        write_json(target.with_suffix(".protocol.json"),settings)
    update_registry(registry_path,entry,create=not args.resume)
    snapshot=target.parent/f"{args.run_id}-source"
    for source in ([] if args.resume else sources):
        destination=snapshot/source.relative_to(PROJECT_ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    started=time.perf_counter()
    try:
        completed={i for i,_ in output}
        with ProcessPoolExecutor(args.workers) as pool:
            jobs=((i,one,(c,i,args.repetitions,args.seed,args.mc,args.draws,args.profile,args.soft,args.loading,args.prior,args.domain,args.patterns,args.transport_patterns,args.pilot_patterns,args.joint_patterns,args.block_patterns,args.loading_patterns,args.loading_audit_patterns,args.predictive_patterns,args.directional_patterns,args.pilot_selection_patterns,args.rank_budget_patterns,args.continuous_patterns),{}) for i,c in enumerate(cases) if i not in completed)
            for index,result in bounded_results(pool,jobs,max_pending=args.workers):
                checkpoint=checkpoints/f'case-{index:04}.json{(".gz" if args.compact else "")}'
                writer=write_json_gzip if args.compact else write_json
                writer(checkpoint,{'settings_digest':content_digest(settings),
                    'case_index':index,'result_digest':content_digest(result),'result':result})
                if args.compact:
                    result=compact_case_result(result,checkpoint.relative_to(target.parent),audit_only=True)
                output.append((index,result))
                # Small manifest avoids repeatedly replacing a huge open file.
                write_json(target.with_suffix(".partial.json"),{"schema_version":2,"settings":settings,
                    "completed_case_indices":[i for i,_ in sorted(output)],"case_checkpoint_directory":str(checkpoints)})
                brief={r["method"]:[round(r["fdr"]["mean"],4),round(r["power"]["mean"],4)] for r in result["rows"]}
                print(result["case"]["name"],f"{result['elapsed_seconds']:.1f}s",brief,flush=True)
        entry.update(status="completed",elapsed_seconds=time.perf_counter()-started,completed_at=datetime.now(UTC).isoformat())
        write_json(target,{"settings":settings,"elapsed_seconds":entry["elapsed_seconds"],"scenarios":[r for _,r in sorted(output)]})
        entry["result_sha256"]=digest(target)
    except BaseException as exc:
        entry.update(status="failed",failure_type=type(exc).__name__,failure=str(exc),elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        update_registry(registry_path,entry)


if __name__=="__main__":
    main()
