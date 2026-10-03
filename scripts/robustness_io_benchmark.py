"""Synthetic serialization benchmark; no statistical experiment or patient data."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from sca3_compass import robustness_io


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('historical_io',args.reference)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    values=np.sin(np.arange(20000)/100).tolist()
    value={'purpose':'SYNTHETIC SERIALIZATION ONLY','nested':[{'unicode':'重尾',
        'weights':values,'converged':i%3!=0} for i in range(50)]}
    result={'reference_source_sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'new_source_sha256':hashlib.sha256(Path(robustness_io.__file__).read_bytes()).hexdigest(),
        'measurements':[]}
    reference_bytes=None
    for label,module in [('unbuffered',old),('buffered',robustness_io)]:
        path=args.output/(label+'.json.gz')
        start=time.perf_counter();module.write_json_gzip(path,value);elapsed=time.perf_counter()-start
        decoded=gzip.decompress(path.read_bytes())
        if reference_bytes is None:reference_bytes=decoded
        if decoded!=reference_bytes:raise ArithmeticError('Serialization bytes changed')
        result['measurements'].append({'method':label,'seconds':elapsed,'compressed_bytes':path.stat().st_size,
            'uncompressed_bytes':len(decoded),'payload_sha256':hashlib.sha256(decoded).hexdigest()})
    result['uncompressed_bytes_exact']=True
    result['gzip_bytes_exact']=(args.output/'unbuffered.json.gz').read_bytes()==(args.output/'buffered.json.gz').read_bytes()
    result['speed_ratio']=result['measurements'][0]['seconds']/result['measurements'][1]['seconds']
    robustness_io.write_json(args.output/'result.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
