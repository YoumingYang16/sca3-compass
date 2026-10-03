"""Existing R0072 DEV loss decomposition; no new data, fits or C1 access."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/robustness/R0072-summary.json'
GUARD = 'R1B_guard_pilotc0.5_projection_gate_eBH'
PLUGIN = 'R1B_plugin_pilotc0.5_projection_gate_eBH'


def main():
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    if not data['complete'] or data['phase'] != 'development' or data['run_id'] != 'R0072':
        raise ValueError('Existing complete DEV only')
    core = [r for r in data['rows'] if r['case'].get('stratum') == 'core']
    if len(core) != 54:
        raise ValueError('Entire original core required')
    output = []
    lines = ['# R0072 DEV: calibration size / pipeline correlation loss map', '',
             'Descriptive post-development decomposition; no new experiment or confirmation interval.',
             'Every cell contains both normal/t5 laws and all three original effect sizes (six equally weighted scenes).',
             'Guard versus plugin changes tail protection AND the training pilot, not rho alone.', '',
             '|rho|Calibration n per study|Guard Power %|Guard minus plugin pp|Guard minus F-empirical pp|Worst cell FDR mean %|',
             '|---:|---:|---:|---:|---:|---:|']
    for rho in [.1, .8, .95]:
        for n in [16, 64, 256]:
            rows = [r for r in core if r['case']['rho'] == rho and r['case']['n'] == n]
            if len(rows) != 6:
                raise ValueError('Exact six-scene cell required')
            mean = lambda f: sum(f(r) for r in rows) / len(rows)
            row = {'rho': rho, 'calibration_n_per_study': n,
                   'case_indices': [r['case_index'] for r in rows],
                   'guard_power': mean(lambda r: r['methods'][GUARD]['power']),
                   'guard_minus_plugin': mean(lambda r: r['methods'][GUARD]['power']-r['methods'][PLUGIN]['power']),
                   'guard_minus_F_empirical': mean(lambda r: r['comparisons']['F_empirical']['delta']),
                   'worst_mean_fdp': max(r['methods'][GUARD]['fdp'] for r in rows)}
            output.append(row)
            lines.append(f"|{rho}|{n}|{100*row['guard_power']:.4f}|{100*row['guard_minus_plugin']:+.4f}|{100*row['guard_minus_F_empirical']:+.4f}|{100*row['worst_mean_fdp']:.4f}|")
    result = {'run_id': 'R0072', 'status': 'DEVELOPMENT_DESCRIPTIVE_ONLY',
              'provenance': 'SIMULATION_NOT_PATIENT_DATA', 'source': str(SOURCE),
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'rows': output,
              'limits': ['Not isolated rho causality', 'Not confirmatory bounds', 'Not R2 candidate selection',
                         'Observed FDR means are not guarantees', 'No C1 data accessed']}
    target = ROOT / 'artifacts/robustness/R0072-dev-loss-map'
    target.mkdir(exist_ok=False)
    (target/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (target/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    shutil.copy2(__file__, target/'generate_source.py')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
