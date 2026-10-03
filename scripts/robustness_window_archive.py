"""Archive exact adopted instructions and R0 receipts without moving originals."""
from pathlib import Path
import json,shutil,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def main():
    dest=ROOT/'artifacts/robustness/R1R2-20260916/archive';dest.mkdir(exist_ok=False)
    instruction=Path('C:/Users/ROG/Downloads/Codex_Research_R1_Closure_R2_Targeted_7h (1).md')
    files=[instruction,*[ROOT/'docs'/n for n in ['ROBUSTNESS_CONVERGENCE_DECISION_2026-09-16.md','ROBUSTNESS_CONVERGENCE_REVIEW_2026-09-15.md']]]
    receipts=[]
    for p in files:
        target=dest/p.name;shutil.copy2(p,target);receipts.append({'original':str(p),'archived':str(target),'sha256':sha(target)})
    for run in ['R0062','R0069']:
        for p in sorted((ROOT/'artifacts/robustness').glob(run+'*')):
            if p.is_file():receipts.append({'historical_readonly_reference':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
        source=ROOT/'artifacts/robustness'/f'{run}-source'
        for p in sorted(source.rglob('*.py')):receipts.append({'historical_readonly_source':str(p),'sha256':sha(p)})
    (dest/'R0-manifest.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'method':'pilotc0.5_pattern_projection_support_gate_eBH','no_git_HEAD':'source checksums are version authority','historical_h_pp':[2.3535,2.3951],'phase':'DEVELOPMENT_NOT_CONFIRMATION','receipts':receipts},indent=2),encoding='utf-8')
    print('Archived',len(receipts),'receipts at',dest)
if __name__=='__main__':main()
