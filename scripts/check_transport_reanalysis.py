"""Independent R recomputation from saved public TPM, not rounded UI scores."""
import json
import subprocess
from pathlib import Path

import numpy as np

from sca3_compass.molecular_data import PROJECT_ROOT, digest, write_json

root=PROJECT_ROOT
frozen=json.loads((root/"artifacts/molecular-transport-freeze.json").read_text(encoding="utf-8"))
report=json.loads((root/"artifacts/molecular-transport.json").read_text(encoding="utf-8"))
path=root/report["expression_path"]
assert digest(path)==report["expression_sha256"]
with np.load(path,allow_pickle=False) as archive:
    expression,features,ids=archive["expression"],archive["features"],archive["samples"]
assert ids.tolist()==[f"GSM{n}" for n in range(8148389,8148401)]
lookup={g:i for i,g in enumerate(features)}
work=root/"artifacts/transport-r-crosscheck"
work.mkdir(parents=True,exist_ok=True)
rscript=root/".tools/R-4.6.1/bin/Rscript.exe"
checks=[]
for strategy,nominations in frozen["nominations"].items():
    available=[n for n in nominations if n["gene"] in lookup]
    input_path=work/f"{strategy}-expression.txt"
    direction_path=work/f"{strategy}-directions.txt"
    output_path=work/f"{strategy}-result.txt"
    np.savetxt(input_path,expression[[lookup[n["gene"]] for n in available]],fmt="%.17g")
    np.savetxt(direction_path,np.array([n["direction"] for n in available]),fmt="%d")
    subprocess.run([str(rscript),"--vanilla","scripts/transport_crosscheck.R",str(input_path),str(direction_path),str(output_path)],cwd=root,check=True,capture_output=True)
    independent=np.loadtxt(output_path)
    expected=next(r for r in report["strategy_results"] if r["strategy"]==strategy)
    np.testing.assert_allclose(independent,[expected["score_effect"],expected["exact_permutation_p"]],atol=1e-12)
    checks.append({"strategy":strategy,"passed":True,"score_difference":float(independent[0]-expected["score_effect"]),"p_difference":float(independent[1]-expected["exact_permutation_p"]),"input_sha256":digest(input_path),"result_sha256":digest(output_path)})
genes=report["gene_results"]
gene_input=work/"union-gene-expression.txt"
gene_direction=work/"union-gene-directions.txt"
gene_output=work/"union-gene-results.txt"
matrix=np.stack([expression[lookup[g["gene"]]] if g["gene"] in lookup else np.full(12,np.nan) for g in genes])
np.savetxt(gene_input,matrix,fmt="%.17g")
np.savetxt(gene_direction,np.array([g["direction"] for g in genes]),fmt="%d")
subprocess.run([str(rscript),"--vanilla","scripts/transport_gene_crosscheck.R",str(gene_input),str(gene_direction),str(gene_output)],cwd=root,check=True,capture_output=True)
independent_genes=np.loadtxt(gene_output)
expected_genes=np.array([[g["external_effect"] if g["external_effect"] is not None else np.nan,g["signed_p"],g["BY_q_union_signed_family"],int(g.get("missing",False))] for g in genes])
np.testing.assert_allclose(independent_genes,expected_genes,atol=1e-12,equal_nan=True)
checks.append({"test":"all_signed_genes_and_BY","hypotheses":len(genes),"passed":True,
               "max_finite_absolute_error":float(np.nanmax(np.abs(independent_genes-expected_genes))),
               "input_sha256":digest(gene_input),"result_sha256":digest(gene_output)})
result={"passed":True,"checks":checks,"report_sha256":digest(root/"artifacts/molecular-transport.json"),
        "python_runner_sha256":digest(Path(__file__)),"r_runner_sha256":digest(root/"scripts/transport_crosscheck.R"),
        "r_gene_runner_sha256":digest(root/"scripts/transport_gene_crosscheck.R"),
        "scope":"Independent numerical recomputation in base R; not independent data collection or external statistical peer review"}
write_json(root/"artifacts/molecular-transport-reanalysis.json",result)
print(json.dumps(result))
