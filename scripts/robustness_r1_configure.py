"""Materialize predeclared development cases, preserving core54 definitions."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
    core=json.loads((ROOT/'configs/robustness_core_development.json').read_text())['cases']
    assert len(core)==54
    stress=json.loads((ROOT/'configs/robustness_r1_development.json').read_text())['cases']
    cases=[{**c,'stratum':'core','assumption_class':'matched_working_model'} for c in core]
    cases += [{**c,'stratum':'stress','assumption_class':('radial_misspecification' if c['distribution']=='lognormal' else c['assumption_class'])} for c in stress]
    extra=[
        {'name':'t1.5_singleton_small','distribution':'t1.5','rho':.95,'n':16,'effect':8.,'truth':'single_study_only','assumption_class':'matched_working_model'},
        {'name':'normal_singleton_small','distribution':'normal','rho':.95,'n':16,'effect':8.,'truth':'single_study_only','assumption_class':'matched_working_model'},
        {'name':'t5_nonpositive_heterogeneous_null','distribution':'t5','rho':.8,'n':64,'effect':8.,'truth':'single_study_only','null_pipeline_shift':2.,'assumption_class':'noncentral_target_extension_unconfirmed'},
        {'name':'t5_negative_study_correlation','distribution':'t5','rho':.8,'n':64,'effect':3.5,'study_rho':-.2,'assumption_class':'matched_working_model'},
        {'name':'t5_calibration_lognormal','distribution':'t5','calibration_distribution':'lognormal','rho':.8,'n':64,'effect':3.5,'assumption_class':'calibration_radial_misspecification'},
    ]
    cases += [{**c,'stratum':'stress'} for c in extra]
    dest=ROOT/'configs/robustness_r1_expanded.json'
    with dest.open('x',encoding='utf-8') as f:json.dump({'phase':'DEVELOPMENT','core54_source':'configs/robustness_core_development.json','core_weight':'equal1/54; stress not included inH','cases':cases},f,indent=2)
    print(len(cases),'cases at',dest)
if __name__=='__main__':main()
