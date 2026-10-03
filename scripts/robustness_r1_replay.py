"""Frozen-source, no-new-data replay: exact arrays and scoring, not new evidence."""
from research_window import wait_start_gate
wait_start_gate()
from pathlib import Path
import argparse,gzip,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def read(p):
    with (gzip.open(p,'rt',encoding='utf-8') if str(p).endswith('.gz') else Path(p).open(encoding='utf-8')) as s:return json.load(s)
def sha(p):
    with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()

def artifact_path(stored,root):
    path=Path(stored)
    if path.exists():return path
    normalized=str(stored).replace('\\','/')
    marker='/artifacts/robustness/'
    if marker not in normalized:raise FileNotFoundError(stored)
    # Relocation changes no archived record/protocol bytes or array hashes.
    candidate=root/'artifacts/robustness'/normalized.split(marker,1)[1]
    if not candidate.is_file():raise FileNotFoundError(candidate)
    return candidate
def main():
    a=argparse.ArgumentParser();a.add_argument('--run-id',required=True);a.add_argument('--execute',action='store_true');a.add_argument('--cases',default='0,1,5,9,13');a.add_argument('--project-root',type=Path,default=ROOT);a.add_argument('--out',type=Path);args=a.parse_args()
    root=args.project_root.resolve();out=args.out.resolve() if args.out else root/'artifacts/robustness'/f'{args.run_id}-replay.json'
    if out.exists():raise FileExistsError('Preserve previous replay; choose a fresh --out path')
    protocol=read(root/'artifacts/robustness'/f'{args.run_id}-screen.protocol.json');source=root/'artifacts/robustness'/f'{args.run_id}-source'
    if protocol['phase'] in ['C1','C2']:
        index_path=root/'artifacts/robustness'/f'{args.run_id}-results-index.json'
        if not index_path.exists() or not read(index_path)['complete']:raise ValueError('Do not replay incomplete confirmation data')
    if not args.execute:
        env=os.environ.copy();env['PYTHONPATH']=str(source/'src')+os.pathsep+str(source/'scripts');env['PYTHONDONTWRITEBYTECODE']='1'
        raise SystemExit(subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run-id',args.run_id,'--cases',args.cases,'--project-root',str(root),'--out',str(out),'--execute'],cwd=source,env=env).returncode)
    for rel,digest in protocol['source_sha256'].items():
        if sha(source/rel)!=digest:raise ValueError('Source mismatch:'+rel)
    sys.path.insert(0,str(source/'scripts'));sys.path.insert(0,str(source/'src'))
    import numpy as np
    from threadpoolctl import threadpool_limits
    from sca3_compass.robustness_bootstrap_guard import evaluate
    from sca3_compass.robustness_pivotal import evaluate as pivotal
    from sca3_compass.molecular_methods import ebh
    from robustness_r1_window import generate,_r0
    import robustness_r1_window as runner
    if sha(runner.__file__)!=protocol['source_sha256']['scripts/robustness_r1_window.py']:raise ValueError('Imported generator source differs from frozen version')
    start=time.perf_counter();receipts=[]
    with threadpool_limits(1):
        for i in map(int,args.cases.split(',')):
            for rep in [0,1]:
                path=root/'artifacts/robustness'/f'{args.run_id}-repetitions'/f'case-{i:04}'/f'rep-{rep:06}.json.gz';record=read(path)
                input_path=artifact_path(record['input_path'],root);evidence_path=artifact_path(record['evidence_path'],root)
                if sha(input_path)!=record['input_sha256'] or sha(evidence_path)!=record['evidence_sha256']:raise ValueError('Archived array checksum mismatch')
                z,cal,truth=generate(protocol['seed'],i,rep,protocol['cases'][i])
                with np.load(input_path) as old:
                    for name,new in [('z',z),('calibration',cal),('truth',truth)]:np.testing.assert_array_equal(new,old[name])
                r0,d0=_r0(z,cal)
                r1,d1=(evaluate(z,cal,draws=protocol['bootstrap_draws'],seed=int(np.random.SeedSequence([protocol['seed'],i,rep,913]).generate_state(1)[0])) if protocol.get('backend')=='bootstrap_guard' else pivotal(z,cal))
                with np.load(evidence_path) as old:
                    for name,value in {**r0,**r1}.items():
                        np.testing.assert_array_equal(value,old[name]);reject=ebh(value,.05);total=int(reject.sum());fp=int((reject&~truth).sum());tp=int((reject&truth).sum())
                        metrics={'fdp':fp/max(1,total),'power':tp/max(1,int(truth.sum())),'tp':tp,'fp':fp,'discoveries':total,'power_defined':bool(truth.sum())}
                        if metrics!=record['metrics'][name]:raise ValueError('Scoring mismatch:'+name)
                    for name,value in d1['held_pvalues'].items():np.testing.assert_array_equal(value,old['p_'+name])
                receipts.append({'case_index':i,'rep':rep,'record_sha256':sha(path),'input_sha256':sha(input_path),'evidence_sha256':sha(evidence_path),'methods':len(r0)+len(r1),'status':'EXACT_ARRAYS_AND_METRICS'})
    with out.open('x',encoding='utf-8') as s:json.dump({'run_id':args.run_id,'kind':'REPRODUCTION_NOT_INDEPENDENT_CONFIRMATION','protocol_sha256':sha(root/'artifacts/robustness'/f'{args.run_id}-screen.protocol.json'),'replay_script_sha256':sha(__file__),'elapsed_seconds':time.perf_counter()-start,'receipts':receipts},s,indent=2)
    print('Exact replay passed',len(receipts),'whole families',out)
if __name__=='__main__':main()
