"""Actual NPZ input/output entry; no truth read, no fictional sample input."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from sca3_compass.molecular_v1 import evaluate,VERSION,ASSUMPTIONS,SCOPE
from sca3_compass.robustness_pattern_test import _json_value
from sca3_compass.robustness_io import write_json


class EvidenceError(RuntimeError):
    pass


def verify_release(path):
    receipt=json.loads(path.read_text(encoding='utf-8'))
    if receipt.get('decision')!='RELEASED_SCOPED_EMPIRICAL' or receipt.get('method_version')!=VERSION:
        raise EvidenceError('Receipt does not clear this exact version')
    for rel,digest in receipt['source_sha256'].items():
        target=(ROOT/rel).resolve()
        if not target.is_relative_to(ROOT.resolve()) or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise EvidenceError('Source differs from independently checked frozen release')
    for kind in ['confirmation','replay']:
        target=(path.parent/receipt[kind+'_path']).resolve()
        if not target.is_relative_to(path.parent.resolve()) or hashlib.sha256(target.read_bytes()).hexdigest()!=receipt[kind+'_sha256']:
            raise EvidenceError('Release evidence changed')
        value=json.loads(target.read_text(encoding='utf-8'))
        if not value.get('complete') or (kind=='confirmation' and not value['gates']['scientific_pass_pending_replay_review']):
            raise EvidenceError('Evidence does not pass its predefined gates')
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--acknowledge-research-scope',action='store_true')
    parser.add_argument('--seed',type=int,default=17091601)
    parser.add_argument('--validation-receipt',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Output already exists; preserve prior results')
    try:
        receipt=verify_release(args.validation_receipt.resolve()) if args.validation_receipt else None
        with np.load(args.input,allow_pickle=False) as data:
            z,cal=data['z'],data['calibration']  # NEVER read truth
        with threadpool_limits(1):
            result=evaluate(z,cal,seed=args.seed,acknowledge_scope=args.acknowledge_research_scope)
        result['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
        result['input_path']=str(args.input.resolve())
        if receipt:
            result['release_status']='V1_RESEARCH_EMPIRICAL'
            result['validation_receipt_sha256']=hashlib.sha256(args.validation_receipt.read_bytes()).hexdigest()
            result['validation_run']=receipt['confirmation_run']
            result['scope_warning']='Passing receipt certifies only the frozen finite-scenario evidence; input distribution assumptions remain unverified.'
        write_json(args.output,_json_value(result))
        print(result['method_version'],result['status'],'K discoveries',int(result['discoveries']['K_NR'].sum()))
    except (EvidenceError,ValueError,KeyError,ArithmeticError,np.linalg.LinAlgError) as error:
        result={'method_version':VERSION,'status':'INSUFFICIENT_EVIDENCE' if isinstance(error,EvidenceError) else 'UNSUPPORTED_INPUT' if isinstance(error,(ValueError,KeyError)) else 'NUMERICAL_FAILURE',
                'evidence_level':'UNRESOLVED','supported_scope':SCOPE,'assumptions':ASSUMPTIONS,
                'diagnostics':{'error':repr(error)},'discoveries':None,'p_values':None,'e_values':None}
        write_json(args.output,result)
        raise SystemExit(2)


if __name__=='__main__':main()
