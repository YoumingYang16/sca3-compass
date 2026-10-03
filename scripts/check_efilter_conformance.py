"""Generated input/output fixtures checked against unchanged author R function."""
import json
import subprocess
from pathlib import Path

import numpy as np

from sca3_compass.molecular_data import digest, write_json
from sca3_compass.molecular_envelope import efilter_adjusted
from sca3_compass.repository import PROJECT_ROOT

root=PROJECT_ROOT
rscript=root/".tools/R-4.6.1/bin/Rscript.exe"
work=root/"artifacts/efilter-conformance"
work.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(151501)
records=[]
for case in range(20):
    s=np.exp(rng.uniform(-5,15,size=100+case*17))
    if case%3==0:
        s=np.round(s,2)+.01
    f=s+np.exp(rng.uniform(-5,15,size=len(s)))
    f[::7]=s[::7]
    x=work/f"input-{case}.txt"
    y=work/f"output-{case}.txt"
    np.savetxt(x,np.column_stack((s,f)),fmt="%.17g")
    subprocess.run([str(rscript),"--vanilla","scripts/efilter_conformance.R",str(x),str(y)],
                   cwd=root,check=True,capture_output=True,text=True)
    author=np.loadtxt(y)
    port=efilter_adjusted(s,f)
    np.testing.assert_allclose(port,author[:,0],rtol=1e-12)
    np.testing.assert_array_equal(port>40,author[:,1].astype(bool))
    records.append({"case":case,"hypotheses":len(s),"maximum_relative_error":float(np.max(np.abs(port-author[:,0])/np.maximum(1,port))),
                    "input_sha256":digest(x),"output_sha256":digest(y)})
report={"passed":True,"author_commit":"918ad5c7d31c42e82af33c0aa36fe147b04e0969", "license":"external/efilter/LICENSE",
        "author_function_sha256":digest(root/"external/efilter/funcs.R"),"port_sha256":digest(root/"src/sca3_compass/molecular_envelope.py"),
        "runner_sha256":digest(Path(__file__)),"r_version":subprocess.check_output([str(rscript),"--version"],text=True).strip(),"cases":records,
        "scope":"Only author's e_filter FDR function, unchanged AST; not author's complete simulation or all paper claims"}
write_json(root/"artifacts/molecular-efilter-conformance.json",report)
print(json.dumps({"passed":True,"cases":len(records),"hypotheses":sum(r["hypotheses"] for r in records)}))
