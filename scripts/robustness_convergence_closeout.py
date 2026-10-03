"""Fail-closed receipts for this bounded diagnosis; no research execution."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import platform
import numpy as np
import scipy
import robustness_replay as rp

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'artifacts/robustness';audit=base/'convergence-20260915'
    start=rp.read_json(audit/'audit-start.json');done=rp.read_json(audit/'completed.json')
    assert done['new_simulation_draws']==0 and not done['independent_confirmation']
    assert rp.sha(ROOT/'scripts/robustness_convergence_audit.py')==start['source_sha256']
    protocol=(ROOT/'docs/ROBUSTNESS_CONVERGENCE_PROTOCOL_2026-09-15.md').read_bytes()
    initial=protocol.split(b'\n## Reviewer-triggered comparator-scope addendum')[0]
    assert hashlib.sha256(initial).hexdigest()==start['protocol_sha256']
    hybrid=rp.read_json(base/'convergence-20260915-hybrid-scope.json')
    # The addendum's rounded time label was corrected to date-only. Preserve
    # executed bytes and exact UTC execution receipt, without changing scope.
    executed_hybrid_protocol=protocol.replace(b'addendum (2026-09-16 HKT)',b'addendum (2026-09-16 00:14 HKT)')
    assert hashlib.sha256(executed_hybrid_protocol).hexdigest()==hybrid['protocol_sha256']
    original=rp.read_json(audit/'original-interval-exact-reproduction.json')
    assert original['exact_macro_equal'] and original['exact_uncertainty_equal']
    mainline=rp.read_json(audit/'mainline-comparison.json')
    for run,n in [('R0062',200),('R0069',100)]:
        assert len(mainline[run]['per_case'])==54
        assert all(r['paired_observed_best']['n']==n for r in mainline[run]['per_case'])
    for run in ['R0064','R0067']:
        for row in rp.read_json(audit/f'{run}-contribution-decomposition.json'):
            if row['effects'] is None:continue
            e=row['effects'];m={k:v['mean'] for k,v in e.items()}
            assert np.isclose(m['interaction_D-C-B+A'],m['package_complex_D-C']-m['package_simple_B-A'])
            for k,v in row['cells'].items():
                assert np.isclose(np.mean(v['power_array']),v['power'])
                assert np.isclose(np.mean(v['fdp_array']),v['fdr'])
    cf=rp.read_json(base/'convergence-20260915-auxiliary/cf-truth-identifiability.json')
    assert [r['global_zero_genes'] for r in cf['truth']]==[129,0]
    paths=[ROOT/'RESEARCH_STATE.md',ROOT/'RESEARCH_PLAN.md',
        ROOT/'docs/ROBUSTNESS_CONVERGENCE_DECISION_2026-09-16.md',
        ROOT/'docs/ROBUSTNESS_CONVERGENCE_REVIEW_2026-09-15.md',
        ROOT/'docs/ROBUSTNESS_CONVERGENCE_PROTOCOL_2026-09-15.md',
        ROOT/'tests/test_robustness_convergence_audit.py']
    paths+=list(audit.glob('*.json'))
    paths+=list((base/'convergence-20260915-auxiliary').glob('*.json'))
    paths+=list((base/'convergence-20260915-tables').glob('*'))
    paths+=[base/'convergence-20260915-hybrid-scope.json']
    paths+=list((ROOT/'scripts').glob('robustness_convergence_*.py'))
    rp.write_new(base/'convergence-20260915-closeout.json',{
        'utc':datetime.now(timezone.utc).isoformat(),'phase':'CHECKPOINT_STOPPED',
        'objective_A':'UNACCEPTED','objective_B':'UNACCEPTED','new_simulations':0,'new_fits':0,'independent_confirmation':False,
        'audit_integrity_checks':'passed','source_version_policy':'No git HEAD; archived per-run SHA256 manifests',
        'initial_protocol_utf8':initial.decode(),'initial_protocol_sha256':start['protocol_sha256'],
        'hybrid_executed_protocol_utf8':executed_hybrid_protocol.decode(),
        'hybrid_execution_utc':hybrid['utc'],
        'protocol_revision_notice':'Added existing-comparator audit after initial reanalysis; corrected rounded addendum heading time to date-only. Scientific scope and original interval family unchanged.',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'files':{str(p.relative_to(ROOT)):rp.sha(p) for p in paths}})
    print('CLOSEOUT INTEGRITY PASSED; no experiment started')


if __name__=='__main__':main()
