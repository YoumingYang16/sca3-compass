"""One deterministic successful-input replay after metadata-only A0.7.1 repair."""
import gzip,json,numpy as np
from r5_common import PHASE,R4,read,write,sha
from smooth_borrowing import evaluate

def main():
    root=PHASE/'D009';path=root/'raw/case-01/rep-00000.json.gz'
    r=json.load(gzip.open(path,'rt',encoding='utf8'));a=r['source_artifacts']['observed']
    if sha(R4/'C001'/a['path'])!=a['sha']:raise ValueError('observed input hash')
    with np.load(R4/'C001'/a['path'],allow_pickle=False) as f:obs=dict(f)
    if sha(root/r['evidence_path'])!=r['evidence_sha']:raise ValueError('old evidence hash')
    with np.load(root/r['evidence_path'],allow_pickle=False) as f:old=dict(f)
    out=evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],bound=4,
        seed=r['algorithm_seed'],reference_draws=4095,inner_draws=4095)
    if out['status']!='completed':raise ValueError('repair replay failure')
    for field in ['p','e','component_e','decision']:
        if not np.array_equal(out[field],old[field]):raise ValueError('metadata repair changed successful statistics:'+field)
    write(PHASE/'checks/a071-successful-replay.json',{'status':'BITWISE_SAME_SUCCESSFUL_STATISTICS',
        'script_sha':sha(__file__),'source_record_sha':sha(path),'source_freeze_sha':sha(root/'freeze.json'),
        'fields':['p','e','component_e','decision'],'one_family_only':True,'version':out['version'],
        'working_code_sha':{n:sha(PHASE/n) for n in ['smooth_borrowing.py','selective_reference.py','r5_kernel.py']}})
    print('Bitwise p/e/component_e/decisions match D009 C1rep0; not a new independent experiment.')

if __name__=='__main__':main()
