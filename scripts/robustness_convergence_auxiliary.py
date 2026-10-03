"""Read-only branch receipts and deterministic identifiability diagnostics; no draws."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import robustness_replay as rp
from robustness_convergence_audit import load_run
from sca3_compass.molecular_envelope_benchmark import fixed_truth


def compound_scale_reference(target_rho, calibration_rho, k=6):
    """Known-model diagnostic ratio, not supplied to any deployable method.

    Q uses 1-rho_cal while the target has 1-rho_target. The target-only
    radial prior absorbs that ratio. Correcting the mean variance then needs
    v_target/v_cal * (1-rho_cal)/(1-rho_target), under true study shape,
    common-location compound noise and an exactly fitted rescaled radial law.
    """
    if k<2 or not all(-1/(k-1)<r<1 for r in [target_rho,calibration_rho]):
        raise ValueError('Positive definite compound correlations required')
    return ((1+(k-1)*target_rho)/(1+(k-1)*calibration_rho)
            *(1-calibration_rho)/(1-target_rho))


def covariance_equivalence():
    """Two distinct random-effect models with identical full observed Gaussian law.

    An example in an unrestricted target/calibration-drift model, not an assertion
    of exact nonidentifiability of R67's finite deterministic/jittered alternatives.
    Both supply the same independent calibration law. Pipeline-common random
    effects add no energy to pipeline contrasts.
    """
    rs = .65*np.ones((4,4))+.35*np.eye(4)
    j = np.ones((6,6)); ident=np.eye(6)
    null_noise = np.kron(rs, ident+4*j)  # 5 * R_pipeline(.8)
    alternative_noise = np.kron(rs, ident+j)  # 2 * R_pipeline(.5)
    alternative_effect = np.kron(3*rs,j)
    centering=ident-j/6
    assert np.array_equal(null_noise,alternative_noise+alternative_effect)
    assert np.max(np.abs(centering@j@centering))<1e-14
    return {'full_observed_covariance_max_error':float(np.max(np.abs(null_noise-alternative_noise-alternative_effect))),
        'effect_covariance_after_pipeline_centering_max':float(np.max(np.abs(centering@j@centering))),
        'positive_definite_noise_min_eigenvalues':[float(np.linalg.eigvalsh(x).min()) for x in [null_noise,alternative_noise]],
        'scope':'unrestricted random effects plus unknown target covariance drift; not proof about the exact R67 alternative law',
        'model0':'mu=0; noise covariance Rs x [I6+4J6]=Rs x 5Rp(.8)',
        'model1':'mu_g iid N4(0,3Rs), common across pipelines; independent noise covariance Rs x [I6+J6]=Rs x 2Rp(.5)',
        'calibration':'same independent external centered Gaussian calibration distribution in both models',
        'conclusion':'Full Z and its derived mean/contrast-energy distributions coincide; unrestricted noise/effect split cannot be uniformly identified from these inputs.'}


def main():
    out=ROOT/'artifacts/robustness/convergence-20260915-auxiliary'
    out.mkdir(exist_ok=False)
    cf=ROOT/'artifacts/robustness/cf-repeated-development/full32'
    manifest=rp.read_json(cf/'manifest.json')
    for relative,expected in manifest.items():
        path=rp.contained(cf,relative.replace('\\','/'))
        if rp.sha(path)!=expected:raise ValueError('CF manifest mismatch: '+relative)
    protocol=rp.read_json(cf/'protocol.json')
    sources=protocol['source_sha256']
    for relative,expected in sources.items():
        if rp.sha(rp.contained(cf/'source',relative.replace('\\','/')))!=expected:
            raise ValueError('CF source mismatch')
    truth=[]
    for fraction in [.2,.8]:
        mu=fixed_truth(256,4,{'effect':3.5,'replicated_fraction':fraction},fraction)
        truth.append({'fraction':fraction,'global_zero_genes':int((mu==0).all(1).sum()),
            'replicated_signed_truths':int(((mu>0).sum(1)>=2).sum()+((mu<0).sum(1)>=2).sum()),
            'nonnull_genes':int((mu!=0).any(1).sum()),
            'zero_train_folds':[int(((mu==0).all(1)&(np.arange(256)%2==fold)).sum()) for fold in [0,1]]})
    rp.write_new(out/'cf-truth-identifiability.json',{'cf_manifest_files_verified':len(manifest),
        'cf_source_files_verified':len(sources),'truth':truth,'covariance_equivalence':covariance_equivalence(),
        'new_simulation_draws':0})
    artifact=load_run(ROOT/'artifacts/robustness','R0068',out)
    names=['geometry_focused_veto_joint_pilot80_pattern_projection_gatedmix_eBH',
        'geometry_focused_veto_joint_pilot80_pattern_support_simes_eBH']
    scenes=[]
    for scene in artifact['scenarios']:
        rows={r['method']:r for r in scene['rows']}
        if any(n not in rows for n in names):
            raise ValueError('Unknown R68 names: '+str([n for n in rows if 'focused_veto' in n]))
        scenes.append({'case':scene['case'],'repetitions':scene['repetitions'],
            'methods':{n:{'power':rows[n]['power'],'fdr':rows[n]['fdr'],
                'power_by_repetition':rows[n]['power_by_repetition'],'fdp_by_repetition':rows[n]['fdp_by_repetition']} for n in names}})
    rp.write_new(out/'R0068-focused-veto.json',scenes)
    print('COMPLETE auxiliary read-only audit')


if __name__=='__main__':main()
