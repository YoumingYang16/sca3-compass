"""Explicitly bound read-only frozen dependencies; no name/path guessing."""
from pathlib import Path
import sys, hashlib, importlib, json
sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
PHASE = next(p for p in [CODE,*CODE.parents] if p.name=='r4-target-calibration-20260918')
ROOT = PHASE.parent.parent
R3 = ROOT/'research/r3-efficiency-20260917/C001'
R2 = ROOT/'research/r2-finite-20260917/C001'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

ANCHORS = {
 'research/r3-efficiency-20260917/C001/freeze.json':'4ef7e448c1ee6ee6064a0e31d53e6f36e21b9825e6a7920ef82aebcee9e0b620',
 'research/r2-finite-20260917/C001/freeze.json':'68068c9f1cf68eb46700b18287b0d226a8a3c2ff1c5ea592e036ee88a2538d8e',
 'research/r3-novelty-20260918/DELIVERY_MANIFEST.json':'10bf73416b58512ecb6e8a0e5a6c554ac3c928875d57b01ae4520692b410660c',
}
for path, digest in ANCHORS.items():
    if sha(ROOT/path) != digest:
        raise ValueError('frozen anchor differs: '+path)
EXPECTED = {}
for base in [R2,R3]:
    for name, digest in read(base/'freeze.json')['files'].items():
        EXPECTED[(base/name).resolve()] = digest
for name, digest in read(R2/'freeze.json')['v1_dependencies']['files'].items():
    if name.endswith('.py'):
        EXPECTED[(ROOT/name).resolve()] = digest
for path, digest in EXPECTED.items():
    if sha(path) != digest:
        raise ValueError('frozen dependency changed: '+str(path))

def bound_import(name, base):
    expected = (base/(name+'.py')).resolve()
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() != expected:
        raise ValueError('wrong existing module: '+name)
    sys.path.insert(0,str(base))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != expected or sha(expected) != EXPECTED[expected]:
        raise ValueError('wrong imported source: '+name)
    return module

model = bound_import('r3_model', R3)
fc = bound_import('finite_calibration', R2)
pb = bound_import('predictive_bridge', R2)
grid = bound_import('grid_calibration', R2)
exp = bound_import('experiment', R2)
ci = bound_import('bounded_intervals', R3)

def verify_runtime():
    receipts={}
    for name, module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not file: continue
        p=Path(file).resolve()
        if any(p.is_relative_to(b) for b in [R2,R3,ROOT/'releases/K-NR-1.0.0/source']):
            if p.suffix=='.py' and (p not in EXPECTED or sha(p)!=EXPECTED[p]):
                raise ValueError('unbound local import: '+name)
            if p in EXPECTED:
                receipts[name]={'path':str(p),'sha256':EXPECTED[p]}
        elif p.is_relative_to(ROOT) and name.startswith('sca3_compass'):
            raise ValueError('nonfrozen application import: '+name)
    return receipts

DEPENDENCIES=verify_runtime()

def js(x):
    import numpy as np
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:
        json.dump(value,f,default=js,ensure_ascii=False,indent=2,allow_nan=False)
