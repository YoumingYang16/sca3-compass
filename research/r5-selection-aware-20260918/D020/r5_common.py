"""Read-only R4 dependency binding for isolated R5 development."""
from pathlib import Path
import sys,hashlib,json
sys.dont_write_bytecode=True
PHASE=next(p for p in [Path(__file__).resolve().parent,*Path(__file__).resolve().parents]
           if p.name=='r5-selection-aware-20260918')
ROOT=PHASE.parent.parent;R4=ROOT/'research/r4-target-calibration-20260918'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
for p,d in [('DELIVERY_MANIFEST.json','8fb2ff2a20e3be8620bccd725e1d0f69ef9b6d641e02c94430f4dc0d34ff0452'),
            ('C001/freeze.json','0fa6e578932db51caba13a92ba2e474e234392f6e8771af5ecf1e3b4b517a6a5')]:
    if sha(R4/p)!=d:raise ValueError('R4 frozen identity changed: '+p)
for f,d in read(R4/'C001/freeze.json')['files'].items():
    if sha(R4/'C001'/f)!=d:raise ValueError('R4 frozen file changed:'+f)
sys.path.insert(0,str(R4/'C001'))
import r4_common as inherited
from r4_kernel import evaluate_kernel as fixed_kernel
from two_bank_reference import reference as fixed_reference
fc=inherited.fc;pb=inherited.pb;model=inherited.model;grid=inherited.grid;exp=inherited.exp;ci=inherited.ci
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False,default=inherited.js)
