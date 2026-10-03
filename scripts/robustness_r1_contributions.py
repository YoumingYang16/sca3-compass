"""Evidence-only factorial table. No new fits, samples, thresholds or selection.

For C1 the interaction interval is obtained by subtracting two already jointly
covered contrast intervals. It does not spend a new error budget or claim an
independent test. A/B/C/D are raw projection vs ordinary Bonferroni, NOT the
strong F envelope and NOT the gamma-gated primary candidate.
"""
from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = 'R1B_guard_pilotc0.5_projection_gate_eBH'
CELLS = {
    'A': 'R1B_plugin_pilotc0.5_ordinary_bonf_eBH',
    'B': 'R1B_guard_pilotc0.5_ordinary_bonf_eBH',
    'C': 'R1B_plugin_pilotc0.5_projection_eBH',
    'D': 'R1B_guard_pilotc0.5_projection_eBH',
}
PAIRS = {
    'guard_cost_simple': ('B', 'A'),
    'guard_cost_complex': ('D', 'C'),
    'projection_extra_plugin': ('C', 'A'),
    'projection_extra_guard': ('D', 'B'),
}

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    folder = args.project_root.resolve() / 'artifacts/robustness'
    source = folder / f"{args.run_id}-{'summary' if args.development else 'confirmation-analysis'}.json"
    data = json.loads(source.read_text(encoding='utf-8'))
    if not data['complete'] or (not args.development and data['phase'] not in ['C1', 'C2']):
        raise ValueError('Complete, appropriately labelled evidence required')
    core = [row for row in data['rows'] if row['case'].get('stratum') == 'core']
    if len(core) != 54:
        raise ValueError('Do not silently change the historical core')
    strata = {'core54': core,
              'normal27': [row for row in core if row['case']['distribution'] == 'normal'],
              't5_27': [row for row in core if row['case']['distribution'] == 't5']}
    result = {'run_id': args.run_id, 'source': str(source), 'source_sha256': digest(source),
              'script_sha256': digest(__file__), 'stage': data['phase'],
              'not_patient_data': True, 'cells': CELLS, 'strata': {},
              'interpretation': [
                  'All cells omit the old fitted own-Q weights; this is not an isolated old-weight ablation.',
                  'Guard versus plugin also recomputes TRAIN pilot: a frontend-bundle contrast, not variance alone.',
                  'A/B use ordinary Bonferroni only; C/D use raw projection, not the gamma-gated primary.',
                  'Strong F envelopes are reported separately and cannot be replaced by the weaker A/B comparator.',
                  'Interaction interval, if present, is a conservative algebraic consequence of jointly covered intervals.',
                  'Scope/FDR and hypothesis-violation failures remain in the full source evidence.']}
    lines = ['# R1 minimal contribution decomposition', '',
             f"{args.run_id} — {data['phase']}; simulation, not patient data.", '',
             'Power is higher-is-better. FDR is the mean of whole-family FDP; scenario maxima are not pooled error ratios.', '',
             '|Stratum|Cell|Mean Power (%)|Maximum scenario FDR (%)|',
             '|---|---|---:|---:|']
    for name, rows in strata.items():
        cells = {key: {'power': float(np.mean([r['methods'][method]['power'] for r in rows])),
                       'max_scene_fdr': float(max(r['methods'][method]['fdp'] for r in rows))}
                 for key, method in CELLS.items()}
        for key, cell in cells.items():
            lines.append(f"|{name}|{key}|{100*cell['power']:.4f}|{100*cell['max_scene_fdr']:.4f}|")
        contrasts = {}
        for label, (first, second) in PAIRS.items():
            item = {'difference': cells[first]['power'] - cells[second]['power']}
            if not args.development:
                bound = data['contributions'][name + '/' + label]
                if abs(item['difference'] - bound['sample_envelope_difference']) > 1e-12:
                    raise ValueError('Source contrasts and cell means disagree')
                item.update(lower=bound['lower'], upper=bound['upper'])
            contrasts[label] = item
        interaction = {'difference': contrasts['guard_cost_complex']['difference'] - contrasts['guard_cost_simple']['difference']}
        if not args.development:
            interaction.update(lower=contrasts['guard_cost_complex']['lower'] - contrasts['guard_cost_simple']['upper'],
                               upper=contrasts['guard_cost_complex']['upper'] - contrasts['guard_cost_simple']['lower'],
                               interpretation='Derived from jointly covered frozen intervals, not a new unallocated test')
        result['strata'][name] = {'cells': cells, 'contrasts': contrasts, 'interaction': interaction,
                                'primary_minus_raw_projection_descriptive': float(np.mean([
                                    r['methods'][MAIN]['power'] - r['methods'][CELLS['D']]['power'] for r in rows]))}
    lines += ['', 'A=plugin Bonferroni; B=guard Bonferroni; C=plugin projection; D=guard projection.', '',
              '|Stratum|Contrast|Power difference (pp)|Jointly covered interval (pp)|', '|---|---|---:|---|']
    for name, value in result['strata'].items():
        for label, c in {**value['contrasts'], 'interaction_D-C-B+A': value['interaction']}.items():
            interval = 'DEVELOPMENT: no confirmatory interval' if args.development else f"[{100*c['lower']:.4f}, {100*c['upper']:.4f}]"
            lines.append(f"|{name}|{label}|{100*c['difference']:+.4f}|{interval}|")
    lines += ['', '## Interpretation limits', ''] + ['- ' + s for s in result['interpretation']]
    out = folder / f'{args.run_id}-contributions'
    out.mkdir(exist_ok=False)
    (out / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    shutil.copy2(__file__, out / 'generate_source.py')
    print('\n'.join(lines))
    print(out)

if __name__ == '__main__':
    main()
