"""Read-only verified R2 dependencies; no writes or bytecode into frozen areas."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
PHASE = next(p for p in Path(__file__).resolve().parents if p.name == 'r3-efficiency-20260917')
ROOT = PHASE.parent.parent
R2 = ROOT/'research/r2-finite-20260917'
sys.path.insert(0, str(R2/'C001'))
from experiment import read, write, sha, js, score, generate
from provenance import verify_freeze

def verify_r2():
    if sha(R2/'DELIVERY_MANIFEST.json') != '51ad4505b7d74d90ed9c0164a8d359c3bb6238db6b2eb08ba741d9f7d1b78073':
        raise ValueError('R2 delivery identity differs')
    verify_freeze(R2/'C001')
    manifest = read(R2/'DELIVERY_MANIFEST.json')
    for name in ['C001/summary.json','C001/index.json','C001/protocol.json','FINAL_STATUS.json']:
        if sha(R2/name) != manifest['files'][name]['sha256']:
            raise ValueError('R2 result identity differs: '+name)
    return manifest
