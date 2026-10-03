"""Guarded CLI entry for frozen R4-C001. No research execution on import."""
from pathlib import Path
import hashlib,json,subprocess,sys

PHASE=Path(__file__).resolve().parent
TRUSTED_FREEZE='0fa6e578932db51caba13a92ba2e474e234392f6e8771af5ecf1e3b4b517a6a5'

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    base=PHASE/'C001';freeze=base/'freeze.json'
    if sha(freeze)!=TRUSTED_FREEZE: raise ValueError('R4 confirmation freeze identity changed')
    fr=json.loads(freeze.read_text(encoding='utf8'))
    for name,digest in fr['files'].items():
        if sha(base/name)!=digest: raise ValueError('frozen source changed: '+name)
    import numpy,scipy
    for name,module in [('numpy',numpy),('scipy',scipy)]:
        if module.__version__!=fr['environment'][name]:
            raise ValueError('environment differs from confirmed R4: '+name)
    raise SystemExit(subprocess.call([sys.executable,'-B',str(base/'r4_cli.py'),*sys.argv[1:]]))

if __name__=='__main__': main()
