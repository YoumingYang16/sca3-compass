"""Fail-closed membership checks for the prespecified C1 design."""
import math
def validate_plan(plan,cases):
    names=plan['reported_methods'];candidate=plan['candidate'];count=len(cases)
    if len(names)!=len(set(names)) or candidate not in names or len(names)<2:raise ValueError('Distinct reported methods/candidate required')
    core=[i for i,c in enumerate(cases) if c.get('stratum')=='core']
    expected={'core54':core,'normal27':[i for i in core if cases[i]['distribution']=='normal'],'t5_27':[i for i in core if cases[i]['distribution']=='t5']}
    if len(core)!=54 or len(expected['normal27'])!=27 or len(expected['t5_27'])!=27 or plan['strata']!=expected:raise ValueError('Exact unique historical core strata required')
    if set(plan['families'])!={'H','F_safe','F_empirical'}:raise ValueError('Frozen H/F families required')
    for family,labels in plan['families'].items():
        if not labels or len(set(labels))!=len(labels) or not set(labels)<=set(names) or candidate in labels:raise ValueError('Invalid comparator family')
        mapping=plan['development_selected_baselines'][family]
        if set(mapping)!=set(map(str,core)) or not set(mapping.values())<=set(labels):raise ValueError('DEV baseline mapping mismatch')
    expected_scope=[i for i,c in enumerate(cases) if c.get('assumption_class') in {'in_scope','matched_working_model'}]
    if not expected_scope or plan['validity_scope_indices']!=expected_scope:raise ValueError('Nonempty prespecified validity scope required')
    counts=plan['repetition_counts']
    if len(counts)!=count or any(type(n) is not int or not 2<=n<=(1000 if i in core else 2000) for i,n in enumerate(counts)):raise ValueError('Invalid fixed sample counts')
    alloc=plan['error_allocation']
    if set(alloc)!={'envelopes','FDR_candidate','FDR_other','contributions'} or any(not math.isfinite(x) or x<=0 for x in alloc.values()) or abs(sum(alloc.values())-.025)>1e-14 or plan['reporting_error_budget']!=.025:raise ValueError('Wrong reporting-error allocation')
    pairs=plan['contribution_pairs']
    if len(pairs)!=4 or any(len(pair)!=2 or not set(pair)<=set(names) or pair[0]==pair[1] for pair in pairs.values()):raise ValueError('Four valid fixed contribution pairs required')
