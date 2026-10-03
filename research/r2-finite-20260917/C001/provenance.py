"""Narrow confirmation identity checks, not a repeat of V1 scientific audit."""
from pathlib import Path
import hashlib
import sys
import zipfile
from experiment import ROOT,sha,read,score
import numpy as np

TRUSTED_V1_ZIP='e4bd2e6061bc886796916013e30ae5c05b86c543ab3c4b129f8173afbdba8fe2'


def bind_v1():
    archive=ROOT/'releases/K-NR-1.0.0.zip'
    if sha(archive)!=TRUSTED_V1_ZIP: raise ValueError('release ZIP differs from trusted V1')
    files={}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.startswith('K-NR-1.0.0/source/') or name.endswith('/'): continue
            path=ROOT/'releases'/name
            digest=hashlib.sha256(z.read(name)).hexdigest()
            if sha(path)!=digest: raise ValueError('extracted V1 source differs: '+str(path))
            files[path.relative_to(ROOT).as_posix()]=digest
    modules={}
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if not path: continue
        resolved=Path(path).resolve()
        if resolved.is_relative_to(ROOT/'releases/K-NR-1.0.0/source'):
            rel=resolved.relative_to(ROOT).as_posix()
            if rel not in files or sha(resolved)!=files[rel]: raise ValueError('unbound imported module')
            modules[name]={'resolved':str(resolved),'sha256':files[rel]}
    return {'files':files,'resolved_modules':modules,'zip_sha256':TRUSTED_V1_ZIP}


def verify_freeze(out):
    fr=read(out/'freeze.json')
    for file,digest in fr['files'].items():
        if sha(out/file)!=digest: raise ValueError('frozen source/protocol changed: '+file)
    for rel,digest in fr['v1_dependencies']['files'].items():
        if sha(ROOT/rel)!=digest: raise ValueError('V1 imported source changed')
    # Imported runtime paths must resolve to the dependency identity recorded.
    for name,receipt in fr['v1_dependencies']['resolved_modules'].items():
        module=sys.modules.get(name)
        if module is not None and str(Path(module.__file__).resolve())!=receipt['resolved']:
            raise ValueError('runtime import path changed')
    return fr


def algorithm_seed(p,case,rep):
    if not (0<=case<32 and 0<=rep<1024): raise ValueError('seed identity range')
    return ((int(p['seed'])*32+case)*1024+rep)*2+1


def check_record(row,out,p,case,rep,metrics=False):
    if (row['case'],row['rep'])!=(case,rep) or row['algorithm_seed']!=algorithm_seed(p,case,rep):
        raise ValueError('record task/seed identity differs')
    if row['freeze_sha256']!=sha(out/'freeze.json'): raise ValueError('record freeze differs')
    if row['status']=='hard_failure': return
    for kind in ['input','evidence']:
        rel=f'raw/case-{case:02}/rep-{rep:05}-{kind}.npz'
        if row[kind+'_path']!=rel or sha(out/rel)!=row[kind+'_sha256']:
            raise ValueError('canonical artifact identity differs')
    if metrics:
        methods=set(p['methods'])|{'PB_ordinary'}
        if p['cases'][case].get('outside_M0'): methods|={'PB_Delta2','PB_Delta5'}
        if set(row['metrics'])!=methods: raise ValueError('unexpected metric roster')
        with np.load(out/row['input_path'],allow_pickle=False) as a: truth=a['truth']
        with np.load(out/row['evidence_path'],allow_pickle=False) as a:
            actual={name[len('decision_'):] for name in a.files if name.startswith('decision_')}
            if actual!=methods: raise ValueError('unexpected saved decision roster')
            for name in methods:
                if score(a['decision_'+name],truth)!=row['metrics'][name]: raise ValueError('stored metric differs from raw score')
            for name in ['PB_grid','PB_main','PB_grid_ordinary','PB_ordinary']:
                if a['e_'+name].shape!=truth.shape or not np.isfinite(a['e_'+name]).all(): raise ValueError('invalid saved evidence')
