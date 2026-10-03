"""Run the EXACT frozen analysis, redirecting ONLY its final write to a new file.

Does not rerun models, alter inputs/analysis or overwrite the original summary.
Output redirection exists because frozen analysis uses exclusive one-time writes.
"""
from pathlib import Path
import argparse,importlib.util,sys
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent;OUT=BASE/'C001'
sys.path.insert(0,str(OUT))
from r3_common import sha,read,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True,type=Path);a=parser.parse_args()
    target=a.report.resolve()
    if target.exists() or target==OUT/'summary.json':raise ValueError('choose a NEW output file')
    fr=read(OUT/'freeze.json')
    for name in ['analyze_confirm.py','bounded_intervals.py']:
        if sha(OUT/name)!=fr['files'][name]:raise ValueError('frozen analysis changed')
    spec=importlib.util.spec_from_file_location('r3_frozen_analysis',OUT/'analyze_confirm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def output_only(path,value):
        if Path(path).resolve()!=OUT/'summary.json':raise ValueError('unexpected write by frozen analyzer')
        write(target,value)
    module.write=output_only
    module.analyze(OUT)
    print({'original_sha256':sha(OUT/'summary.json'),'recomputed_sha256':sha(target),
           'byte_identical':sha(target)==sha(OUT/'summary.json'),'new_file':str(target)})
    if sha(target)!=sha(OUT/'summary.json'):raise ValueError('recomputed summary differs; retain both and investigate')

if __name__=='__main__':main()
