"""Verify final failed-C2 artifact chain without new simulations or inference.

Checks an independently regenerated report, 84 per-case summaries, hard-failure
denominators, whole-family paired strata and mutually exclusive contribution
partition. It does not re-open the failed C2 confirmation gate.
"""
from research_window import wait_start_gate
wait_start_gate()
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_report(folder):
    manifest=read(folder/'manifest.json')
    for name,digest in manifest['generated_files'].items():
        if sha(folder/name)!=digest:raise ValueError('Generated artifact hash mismatch')
    for v in manifest['inputs'].values():
        if sha(v['path'])!=v['sha256']:raise ValueError('Report source hash mismatch')
    if not manifest['no_new_samples'] or not manifest['no_new_formal_intervals']:raise ValueError('Evidence role changed')
    return manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve();base=ROOT/'artifacts/robustness';begun=time.perf_counter()
    if out.exists() or not out.is_relative_to(base.resolve()):raise ValueError('New project-local receipt required')
    original=base/'R0076-closeout-report-v2';rerender=base/'R0076-closeout-rerender-v2'
    first,second=verify_report(original),verify_report(rerender)
    if first['inputs']!=second['inputs'] or first['script_sha256']!=second['script_sha256']:raise ValueError('Not the same reproduction inputs')
    parity={name:sha(original/name)==sha(rerender/name) for name in first['generated_files']}
    if not all(parity.values()):raise ValueError('Report byte reproduction differs; disclose before delivery')
    audit=read(base/'R0076-failed-audit/summary.json');plan=audit['plan'];methods=plan['reported_methods']
    ik,ir=methods.index(plan['candidate']),methods.index(plan['reference']);diffs={};groups={};counts=[0,0]
    max_completion_roundoff=0.;distribution_rows=[]
    for row in audit['rows']:
        i=row['case_index'];path=Path(row['array_path'])
        if sha(path)!=row['array_sha256']:raise ValueError('Audit array hash')
        with np.load(path,allow_pickle=False) as data:
            observed=data['observed'];fdp=data['fdp'];power=data['power'];failure_group=data['failure_group']
            np.testing.assert_array_equal(data['methods'],methods)
            assert len(observed)==row['n_planned']
            assert int(observed.sum())==row['n_successful']
            assert int((~observed).sum())==row['n_failed']
            counts[0]+=int(observed.sum());counts[1]+=int((~observed).sum())
            if not np.isnan(fdp[~observed]).all() or not np.isnan(power[~observed]).all():raise ValueError('Missing output imputed as observation')
            for j,m in enumerate(methods):
                value=row['methods'][m];x=fdp[observed,j]
                bounds=[float(x.sum()/len(observed)),float((x.sum()+sum(~observed))/len(observed))]
                max_completion_roundoff=max(max_completion_roundoff,float(np.max(np.abs(np.array(bounds)-value['fdp_sample_completion_bounds_not_CI']))))
                np.testing.assert_allclose(bounds,value['fdp_sample_completion_bounds_not_CI'],rtol=0,atol=1e-15)
                if not observed.all():
                    if value['fdp'] is not None or value['power'] is not None:raise ValueError('Incomplete scene reported as complete mean')
                else:
                    np.testing.assert_allclose(x.mean(),value['fdp'],rtol=0,atol=1e-15)
                if not row['power_defined'] and value['power'] is not None:raise ValueError('Undefined Power substituted')
            diffs[i]=power[:,ik]-power[:,ir];groups[i]=failure_group.copy()
            distribution_rows.append({'case':i,'observed_families':int(observed.sum()),
                'role':'COMPLETE_SCENE_DESCRIPTIVE' if observed.all() else 'SUCCESS_ONLY_NOT_FULL_SCENE_ESTIMATE',
                'K_FDP_quantiles_05_50_95_99':np.quantile(fdp[observed,ik],[.05,.5,.95,.99]).tolist(),
                'K_FDP_sd':float(fdp[observed,ik].std(ddof=1)),
                'K_Power_quantiles_05_50_95_99':np.quantile(power[observed,ik],[.05,.5,.95,.99]).tolist() if row['power_defined'] else None,
                'K_Power_sd':float(power[observed,ik].std(ddof=1)) if row['power_defined'] else None})
    if counts!=[82798,2]:raise ValueError('Original counts changed')
    strata_check={}
    for label,indices in plan['strata'].items():
        arrays=[diffs[i] for i in indices]
        if any(not np.isfinite(x).all() for x in arrays):raise ValueError('Stratum missingness masked')
        mean=float(np.mean([x.mean() for x in arrays]));mcse=float(np.sqrt(sum(x.var(ddof=1)/len(x) for x in arrays))/len(arrays))
        np.testing.assert_allclose([mean,mcse],[audit['strata'][label]['I']['mean_difference'],audit['strata'][label]['I']['paired_mcse']],atol=1e-15,rtol=0)
        components=audit['component_partition_descriptive'][label]
        np.testing.assert_allclose(sum(v['weighted_fraction'] for v in components.values()),1,atol=1e-15,rtol=0)
        np.testing.assert_allclose(sum(v['contribution_to_I'] for v in components.values()),mean,atol=1e-15,rtol=0)
        for group,v in components.items():
            contribution=np.mean([np.sum(diffs[i][groups[i]==group])/len(diffs[i]) for i in indices])
            np.testing.assert_allclose(contribution,v['contribution_to_I'],atol=1e-15,rtol=0)
        # Recompute sample envelopes from the per-case source arrays, not rounded tables.
        for family,names in plan['families'].items():
            local=[]
            for i in indices:
                with np.load(audit['rows'][i]['array_path'],allow_pickle=False) as data:
                    pw=data['power'];local.append(pw[:,ik].mean()-max(pw[:,methods.index(m)].mean() for m in names))
            np.testing.assert_allclose(np.mean(local),audit['comparisons'][label+'/'+family]['sample_envelope_difference'],atol=1e-15,rtol=0)
        strata_check[label]={'paired_mean':mean,'paired_mcse':mcse,'partition_exact':True,'envelopes_recomputed':True}
    repair=read(base/'R0076-post-C2-numeric-regression-v2/summary.json')
    for row in repair['rows']:
        if sha(row['evidence_path'])!=row['evidence_sha256']:raise ValueError('Repair evidence changed')
    protocol=read(base/'R0076-screen.protocol.json')
    for rel,digest in protocol['source_sha256'].items():
        if sha(base/'R0076-source'/rel)!=digest:raise ValueError('Frozen source modified')
    # The original complete-only outputs must remain absent.
    absent=['R0076-confirmation-analysis.json','R0076-replay.json','R0076-evidence-report']
    if any((base/name).exists() for name in absent):raise ValueError('Unexpected successful-C2 artifact')
    junit=base/'R1R2-20260916/continuation-20260916-0153/closeout-release-software.xml'
    suites=list(ET.parse(junit).getroot().iter('testsuite'))
    software={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    if software['failures'] or software['errors']:raise ValueError('Software test failures')
    payload={'kind':'CLOSEOUT_REPRODUCTION_AND_SECOND_PASS_AUDIT_NOT_CONFIRMATION','complete':True,
        'new_samples':0,'confirmation_promoted':False,'byte_parity':parity,'strata_recomputed':strata_check,
        'families_successful_failed':counts,'cases_verified':84,'methods_resummarized':len(methods),
        'per_case_K_distributions_descriptive':distribution_rows,
        'maximum_redundant_completion_range_float_roundoff':max_completion_roundoff,
        'roundoff_note':'Original complete-scene redundant upper bounds used sum+n-n; sub-ulp cancellation is preserved in original audit. Stable grouping corrected in rerun CLI, not a change to metrics or missing-scene results.',
        'all_original_frozen_source_unchanged':True,'original_complete_only_outputs_absent':absent,
        'software':software,'software_junit_sha256':sha(junit),'audit_sha256':sha(base/'R0076-failed-audit/summary.json'),
        'report_manifest_sha256':sha(original/'manifest.json'),'rerender_manifest_sha256':sha(rerender/'manifest.json'),
        'script_sha256':sha(__file__),'finished_utc':datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds':time.perf_counter()-begun,
        'limitations':['Reproduction is not a new independent experiment','Visual review saved separately','No FDR theorem or publication certification']}
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__':main()
