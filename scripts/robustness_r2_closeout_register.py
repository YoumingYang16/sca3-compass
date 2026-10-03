"""Append verified closeout metadata without changing historical experiment status."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sca3_compass.robustness_registry import update_registry
from sca3_compass.robustness_io import write_json,content_digest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    base=ROOT/'artifacts/robustness';registry=base/'EXPERIMENT_REGISTRY.json'
    verify=read(base/'R0076-closeout-verification-v2.json')
    review=read(base/'R0076-closeout-main-review.json')
    if not verify['complete'] or not review['visual_review_complete'] or not review['scientific_review_complete']:
        raise ValueError('Complete actual main review and reproduction required')
    if review['report_manifest_sha256']!=sha(base/'R0076-closeout-report-v2/manifest.json'):
        raise ValueError('Review/report mismatch')
    out=base/'R0076-closeout-registration';out.mkdir(exist_ok=False)
    shutil.copy2(registry,out/'registry-before.json')
    before=read(registry);now=datetime.now(timezone.utc).isoformat()
    important={e['id']:{'status':e['status'],'settings':content_digest(e.get('settings',{}))}
        for e in before['experiments']}
    for entry in before['experiments']:
        if entry['id'] not in ['R0073','R0074','R0075','R0076']:continue
        item=dict(entry)
        if entry['id']=='R0076':
            if entry['status']!='failed':raise ValueError('Preserve original FAILED status')
            item.update(independent_confirmation_complete=False,
                closeout={'status':'DESCRIPTIVE_FAILED_C2_CLOSED_NOT_CONFIRMED','updated_utc':now,
                    'successful_families':82798,'failed_families':2,'planned_families':82800,
                    'original_failures_preserved':True,'new_confirmation_batches':0,
                    'descriptive_audit':'artifacts/robustness/R0076-failed-audit/summary.json',
                    'descriptive_audit_sha256':sha(base/'R0076-failed-audit/summary.json'),
                    'frozen_engineering_replay':'artifacts/robustness/R0076-failed-batch-frozen-replay.json',
                    'numerical_repair_role':'POST_C2_DEVELOPMENT_ONLY_NOT_PROMOTED',
                    'report':'artifacts/robustness/R0076-closeout-report-v2/README_ZH.md',
                    'verification':'artifacts/robustness/R0076-closeout-verification-v2.json',
                    'main_review':'artifacts/robustness/R0076-closeout-main-review.json',
                    'broad_research_goals_complete':False})
        elif entry['id']=='R0073':
            item['later_disclosure_R0076']={'updated_utc':now,
                'finding':'Original calibration numerical failure on two newly expanded small-calibration inputs',
                'C1_original_68_scope_preserved':True,'general_numerical_reliability_not_claimed':True,
                'separate_repair_does_not_inherit_C1_confirmation':True}
        else:
            ap=base/(entry['id']+'-analysis')/'summary.json'
            item['closeout_development_analysis']={'path':str(ap),'sha256':sha(ap),
                'independent_confirmation':False,'updated_utc':now}
        update_registry(registry,item)
    after=read(registry)
    if len(before['experiments'])!=len(after['experiments']):raise ValueError('No added experiment IDs allowed')
    for e in after['experiments']:
        if important[e['id']]!={'status':e['status'],'settings':content_digest(e.get('settings',{}))}:
            raise ValueError('Historical scientific settings/status changed')
    write_json(out/'receipt.json',{'complete':True,'updated_utc':now,'new_experiment_ids':0,
        'original_status_and_settings_preserved':True,'registry_before_sha256':sha(out/'registry-before.json'),
        'registry_after_sha256':sha(registry),'script_sha256':sha(__file__),
        'decision':'RETAIN_SCOPED_R1_REFERENCE_K_UNCONFIRMED_REPAIR_DEVELOPMENT_STOP',
        'original_research_goals_complete':False})
    print(out/'receipt.json')


if __name__=='__main__':main()
