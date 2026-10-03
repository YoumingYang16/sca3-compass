"""Result-only verification, tables and scoped package. No new experiments.

Not a scientific analyzer replacement: consumes the actual archived analyzer
and replay receipts, verifies raw case arrays, and never changes their gates.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sca3_compass.robustness_io import write_json,content_digest
from sca3_compass.robustness_registry import update_registry
from sca3_compass.robustness_confirmation_bounds import kl_interval
from sca3_compass.robustness_paired_bounds import paired_interval


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def percent(value):return 'undefined' if value is None else f'{100*value:.4f}%'
def pp(value):return f'{100*value:+.4f}'


def verify(root,summary_path,replay_path):
    summary=read(summary_path);replay=read(replay_path);base=root/'artifacts/robustness'
    protocol_path=base/'R0078-v1.protocol.json';p=read(protocol_path)
    if summary['protocol_sha256']!=sha(protocol_path) or replay['protocol_sha256']!=sha(protocol_path):
        raise ValueError('Evidence protocol mismatch')
    if summary['run_id']!='R0078' or replay['run_id']!='R0078' or not summary['complete'] or not replay['complete']:
        raise ValueError('Incomplete evidence cannot be released')
    if not summary['formal_confirmation'] or summary['attempts']!=p['plan']['fixed_total']:
        raise ValueError('Not the fixed formal confirmation')
    if set(summary['methods'])!=set(p['methods']):raise ValueError('Method family changed')
    source=base/'R0078-v1-source'
    for rel,digest in p['source_sha256'].items():
        if sha(source/rel)!=digest:raise ValueError('Frozen source changed')
    if sha(summary_path.parent/'analyzer_source.py')!=p['source_sha256']['scripts/molecular_v1_validation.py']:
        raise ValueError('Result analyzer is not the frozen analyzer')
    methods=p['methods'];arrays={};maximum_difference=0.
    for row in summary['scene_rows']:
        i=row['index']
        with np.load(summary_path.parent/f'case-{i:04}.npz',allow_pickle=False) as data:
            arrays[i]={key:data[key].copy() for key in data.files}
        if any(len(x)!=p['counts'][i] for x in arrays[i].values()):raise ValueError('Case count changed')
        for m in methods:
            for metric in ['fdp','power','tp','fp','discoveries']:
                x=arrays[i][m+'__'+metric];value=float(np.nanmean(x)) if np.isfinite(x).any() else None
                old=row['methods'][m][metric]
                if value is None:
                    if old is not None:raise ValueError('Null Power misreported')
                elif abs(value-old)>1e-14:raise ValueError('Reported mean not reproducible')
            fdp=arrays[i][m+'__fdp']
            ci=kl_interval(float(fdp.mean()),len(fdp),p['plan']['error_allocation']['fdr']/(2*84*len(methods)))
            if not np.allclose(ci,row['methods'][m]['fdr_interval'],atol=1e-13,rtol=0):
                raise ValueError('FDP interval mismatch')
    for stratum,ids in [('core',range(54)),('normal',range(27)),('t5',range(27,54))]:
        for m in methods[1:]:
            diffs=[arrays[i]['K_NR__power']-arrays[i][m+'__power'] for i in ids]
            result=paired_interval(diffs,p['plan']['error_allocation']['paired']/(2*3*(len(methods)-1)))
            reported=summary['paired'][stratum]['comparisons'][m]
            for key in ['mean_difference','lower','upper','paired_mcse']:
                maximum_difference=max(maximum_difference,abs(result[key]-reported[key]))
                if abs(result[key]-reported[key])>1e-13:raise ValueError('Paired bound mismatch')
    for i,rows in summary['local'].items():
        for m,reported in rows.items():
            difference=arrays[int(i)]['K_NR__power']-arrays[int(i)][m+'__power']
            result=paired_interval([difference],p['plan']['error_allocation']['local']/(2*84*(len(methods)-1)))
            if any(abs(result[k]-reported[k])>1e-13 for k in ['mean_difference','lower','upper']):
                raise ValueError('Local bound mismatch')
    outside=set(p['outside_indices']);fair=p['plan']['fair_gatekeepers']
    valid=all(row['methods']['K_NR']['fdr_interval'][1]<=.05 for row in summary['scene_rows'] if row['index'] not in outside)
    fair_valid=all(row['methods'][m]['fdr_interval'][1]<=.05 for row in summary['scene_rows'] if row['index'] not in outside for m in fair)
    superior=all(summary['paired'][s]['comparisons'][m]['lower']>0 for s in ['core','normal','t5'] for m in fair)
    local_ok=not any(values[m]['upper']<0 for i,values in summary['local'].items() if int(i) not in outside for m in fair)
    derived=valid and fair_valid and superior and local_ok
    if derived!=summary['gates']['scientific_pass_pending_replay_review']:raise ValueError('Gate calculation changed')
    return summary,p,{'passed':True,'all_84_case_arrays_recomputed':True,
            'paired_and_local_bounds_recomputed':True,'maximum_summary_difference':maximum_difference,
            'scientific_pass':derived,'report_budget':p['plan']['report_budget']}


def tables(summary):
    text=['# Frozen independent comparison — R0078','',
          'SIMULATION_NOT_PATIENT_DATA. Power higher is better; FDR lower is safer. Same nominal .05. Percentages below are not percentage-point differences.',
          '','|Method|Core Power|Normal Power|t5 Power|Worst D1 FDR|Largest D1 FDR upper|',
          '|---|---:|---:|---:|---:|---:|']
    scope=[r for r in summary['scene_rows'] if r['scope']=='D1']
    for m in summary['methods']:
        text.append('|'+m+'|'+'|'.join(percent(summary['paired'][s]['power'][m]) for s in ['core','normal','t5'])+
                    '|'+percent(max(r['methods'][m]['fdp'] for r in scope))+
                    '|'+percent(max(r['methods'][m]['fdr_interval'][1] for r in scope))+'|')
    text+=['','The maximum point estimate and maximum upper bound can occur in different scenes. All scene-specific values follow below.',
           '','|Stratum|Comparator|K minus comparator, pp|Joint lower, pp|Joint upper, pp|Extra correct claims/family|',
           '|---|---|---:|---:|---:|---:|']
    for s in ['core','normal','t5']:
        for m,v in summary['paired'][s]['comparisons'].items():
            text.append(f"|{s}|{m}|{pp(v['mean_difference'])}|{pp(v['lower'])}|{pp(v['upper'])}|{v['additional_tp_per_family']:.4f}|")
    text+=['','Intervals belong to this phase-wide .05 union report budget; NOT 95% coverage over all historical selection. No gene pseudo-replication. Raw marginal fair attempts may be invalid; strong-reference advantage is not a release requirement.',
           '','## All scenes, including every unsupported historical boundary','',
           '|Case|Scope|Name|Method|Power|FDR|FDR lower|FDR upper|',
           '|---:|---|---|---|---:|---:|---:|---:|']
    for row in summary['scene_rows']:
        for m,v in row['methods'].items():
            text.append(f"|{row['index']}|{row['scope']}|{row['case']['name']}|{m}|{percent(v['power'])}|{percent(v['fdp'])}|{percent(v['fdr_interval'][0])}|{percent(v['fdr_interval'][1])}|")
    text+=['','## Local Power differences against the mandatory fair controls','',
           'The five smallest D1 point differences per control are displayed for inspection; this is descriptive ranking, not a new selected hypothesis. Intervals are the already frozen simultaneous local intervals. All local comparisons remain in evidence/confirmation.json. Null-only Power is undefined and is not ranked.',
           '','|Control|Case|K minus control, pp|Joint lower, pp|Joint upper, pp|',
           '|---|---:|---:|---:|---:|']
    inside={r['index'] for r in scope}
    for m in ['B_fair_conditional_eBH','B_fair_conditional_BY']:
        values=[(int(i),v[m]) for i,v in summary.get('local',{}).items() if int(i) in inside and m in v]
        for i,v in sorted(values,key=lambda x:(x[1]['mean_difference'],x[0]))[:5]:
            text.append(f"|{m}|{i}|{pp(v['mean_difference'])}|{pp(v['lower'])}|{pp(v['upper'])}|")
    text+=['','## Preserved costs and limitations','',
           f"Whole-family joint inference+checkpoint CPU {summary['cpu_seconds']:.3f}s; batch wall {summary['wall_seconds']:.3f}s. These include all8 outputs, not optimized standalone method timing.",
           f"Hard failures {len(summary['failed'])}; declared conservative fold-fallback families {len(summary['soft_fallback_families'])}.",
           'Raw per-family records/input/evidence are in artifacts/robustness/R0078-v1-repetitions; no observations removed. Exact retry/replay uses the same data, not a new independent attempt.',
           'No claim of per-scene noninferiority from absence of a negative significant difference. No finite-estimation theorem, clinical validation, new algorithmic priority or SCI guarantee.']
    return '\n'.join(text)+'\n'


def make_package(root,out,summary_path,replay_path):
    if out.exists():raise FileExistsError('New release directory required')
    summary,p,audit=verify(root,summary_path,replay_path)
    diagnostic_path=root/'artifacts/robustness/R0078-v1-diagnostics/summary.json'
    diagnostic=read(diagnostic_path)
    if not diagnostic['complete'] or diagnostic['totals']['families']!=summary['attempts']:
        raise ValueError('Complete fallback disclosure required')
    if diagnostic['index_sha256']!=sha(root/'artifacts/robustness/R0078-v1-index.json'):
        raise ValueError('Diagnostic index changed')
    if sorted(diagnostic['calibration_guard_fallback_families'])!=sorted(summary['soft_fallback_families']):
        raise ValueError('Calibration fallback counts disagree')
    replayed={(r['case'],r['rep']) for r in read(replay_path)['results']}
    if any(tuple(r) not in replayed for r in diagnostic['pattern_fallback_families']):
        raise ValueError('Pattern-failure replay needs an explicit engineering addendum')
    junit=root/'artifacts/robustness/V1-reference/release-full-tests.xml'
    suites=ET.parse(junit).getroot().findall('testsuite')
    if not suites or any(int(s.attrib.get(k,0)) for s in suites for k in ['errors','failures']):
        raise ValueError('Final intended software suite did not pass')
    audit['software_tests']={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    if audit['software_tests']['tests']<2000:raise ValueError('Not the full intended suite')
    out.mkdir(parents=True,exist_ok=False)
    evidence=out/'evidence';evidence.mkdir()
    shutil.copy2(__file__,out/'package_source.py')
    shutil.copy2(summary_path,evidence/'confirmation.json');shutil.copy2(replay_path,evidence/'replay.json')
    shutil.copy2(diagnostic_path,evidence/'diagnostics.json')
    shutil.copy2(diagnostic_path.parent/'source.py',out/'diagnostic_audit_source.py')
    shutil.copy2(root/'artifacts/robustness/R0078-v1.protocol.json',evidence/'protocol.json')
    for name in ['freeze-decision.json','precision-plan.json','pre-freeze-intended-tests.xml','final-targeted-tests-v2.xml','release-full-tests.xml','package-helper-tests.xml','test-collection-note.md','running-resource-snapshot.json']:
        shutil.copy2(root/'artifacts/robustness/V1-reference'/name,evidence/name)
    for src,dst in [('V1_ACCEPTANCE.md','ACCEPTANCE.md'),('docs/V1_VALIDITY_NOTE.md','VALIDITY_NOTE.md'),
                    ('docs/V1_SUPPORTED_SCOPE.md','SUPPORTED_SCOPE.md'),('docs/V1_PREFLIGHT_REVIEW.md','PREFLIGHT_REVIEW.md')]:
        shutil.copy2(root/src,out/dst)
    shutil.copytree(root/'artifacts/robustness/R0078-v1-source',out/'source',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for file in summary_path.parent.glob('case-*.npz'):shutil.copy2(file,evidence/file.name)
    (out/'COMPARISON.md').write_text(tables(summary),encoding='utf-8')
    counts=diagnostic['totals']
    (out/'DIAGNOSTICS.md').write_text('\n'.join([
        '# Existing-record failure and gating disclosure','',
        f"Audited {counts['families']} whole families / {counts['folds']} held folds; every original record hash checked.",
        f"Calibration guard failure folds: {counts['calibration_guard_failure_folds']}; affected families: {len(diagnostic['calibration_guard_fallback_families'])}. These use the frozen conservative fold rule, not dropped observations.",
        f"Pattern-convergence fallback folds: {counts['pattern_failure_folds']}; affected families: {len(diagnostic['pattern_fallback_families'])}.",
        f"Across {2*counts['folds']} fold-directions: gamma0={counts['gamma_zero_direction_folds']}, gamma1={counts['gamma_one_direction_folds']}, interior={counts['gamma_interior_direction_folds']}.",
        'Gamma0 selects the ordinary component. It is not necessarily an optimization failure. These are dependent diagnostic units, NOT independent statistical repetitions. No causal component contribution is inferred from these counts.',
        'All per-case counts and affected family identities are in evidence/diagnostics.json. This is a post-computation descriptive audit, not an additional confirmation or change to the frozen analysis.','']),encoding='utf-8')
    write_json(out/'verification.json',audit)
    if not audit['scientific_pass']:
        write_json(out/'release.json',{'decision':'EVIDENCE_BLOCKED','method_version':p['version'],
                                     'gates':summary['gates'],'no_new_confirmation':True})
        return
    # This receipt is provisional until the packaged CLI and final review pass;
    # main must not announce completion merely because the file exists.
    receipt={'decision':'RELEASED_SCOPED_EMPIRICAL','method_version':p['version'],
             'evidence_level':'EMPIRICAL_ONLY','confirmation_run':'R0078',
             'source_sha256':p['source_sha256'],'confirmation_path':'evidence/confirmation.json',
             'confirmation_sha256':sha(evidence/'confirmation.json'),'replay_path':'evidence/replay.json',
             'replay_sha256':sha(evidence/'replay.json'),'packaging_status':'PENDING_ACTUAL_CLI_AND_FINAL_REVIEW',
             'history':'Original C2 remains failed; this new phase has its own explicitly authorized fixed report budget.'}
    write_json(out/'release.json',receipt)
    example=out/'example';example.mkdir()
    source_input=root/'artifacts/robustness/R0078-v1-repetitions/case-0055/rep-000000-input.npz'
    with np.load(source_input,allow_pickle=False) as data:
        with (example/'input.npz').open('xb') as stream:
            np.savez_compressed(stream,z=data['z'],calibration=data['calibration'])
    write_json(example/'provenance.json',{'kind':'SIMULATION_NOT_PATIENT_DATA',
               'source_input_sha256':sha(source_input),'case':p['cases'][55],
               'note':'Truth removed from public calling example; this is a selected demonstration, not extra independent evidence.'})
    command=[str(root/'.venv/Scripts/python.exe'),str(out/'source/scripts/molecular_v1.py'),
             '--input',str(example/'input.npz'),'--output',str(example/'output.json'),
             '--acknowledge-research-scope','--validation-receipt',str(out/'release.json'),
             '--seed',str(int(np.random.SeedSequence([p['seed'],55,0,913]).generate_state(1)[0]))]
    executed=subprocess.run(command,cwd=root,text=True,capture_output=True)
    write_json(out/'actual-command.json',{'command':command,'returncode':executed.returncode,
                                        'stdout':executed.stdout,'stderr':executed.stderr})
    if executed.returncode:raise RuntimeError('Packaged actual CLI failed')
    answer=read(example/'output.json')
    if answer['release_status']!='V1_RESEARCH_EMPIRICAL':raise ValueError('CLI did not validate receipt')
    with np.load(root/'artifacts/robustness/R0078-v1-repetitions/case-0055/rep-000000-evidence.npz',allow_pickle=False) as old:
        for group,prefix in [('p_values','p_'),('e_values','e_')]:
            if any(not np.array_equal(np.asarray(v),old[prefix+k]) for k,v in answer[group].items()):
                raise ValueError('Actual packaged CLI differs from confirmation evidence')
    write_json(out/'actual-command-verification.json',{'passed':True,'all_p_e_arrays_exact':True,
        'version':answer['method_version'],'release_status':answer['release_status'],'example_input_sha256':sha(example/'input.npz'),
        'example_output_sha256':sha(example/'output.json'),'caller_truth_input':False})
    print('Package prepared; final main/reviewer review required',out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--summary',type=Path,default=ROOT/'artifacts/robustness/R0078-v1-analysis/summary.json')
    parser.add_argument('--replay',type=Path,default=ROOT/'artifacts/robustness/R0078-v1-replay.json')
    args=parser.parse_args();make_package(ROOT,args.out.resolve(),args.summary.resolve(),args.replay.resolve())


if __name__=='__main__':main()
