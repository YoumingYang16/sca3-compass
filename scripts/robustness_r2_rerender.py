"""Re-render completed C2 evidence without overwriting it or rerunning research.

The report's copied generate_source.py is a provenance snapshot. This CLI is
the supported new-output entry point, using the exact pinned live renderer.
"""
from research_window import wait_start_gate
wait_start_gate()

from datetime import datetime, timezone
from pathlib import Path
import argparse
import shutil

from robustness_r2_development import read, sha
from robustness_r2_evidence_report import render, figures
from sca3_compass.robustness_io import write_json

ROOT = Path(__file__).resolve().parents[1]
RENDERER_SHA256 = '341d782cbc6392cf03f2c97ea395e03f3563da68373a5f103e3fd61ca4c6a9aa'
PREFIX = 'R0076-rerender-'


def fresh_output(base, requested):
    base = Path(base).resolve()
    out = Path(requested).resolve()
    if out.parent != base or not out.name.startswith(PREFIX) or out.name == PREFIX:
        raise ValueError('Use a new artifacts/robustness/R0076-rerender-<name> directory')
    if out.exists():
        raise FileExistsError('Preserve existing output: ' + str(out))
    return out


def verified_inputs(base, renderer):
    base = Path(base)
    report = base / 'R0076-evidence-report'
    manifest_path = report / 'manifest.json'
    manifest = read(manifest_path)
    if manifest.get('run_id') != 'R0076':
        raise ValueError('Only the completed R0076 report is supported')
    if manifest['script_sha256'] != RENDERER_SHA256 or sha(renderer) != RENDERER_SHA256:
        raise ValueError('Pinned renderer changed')
    receipts = {}
    for name, digest in manifest['generated_files'].items():
        if Path(name).name != name:
            raise ValueError('Invalid original report member')
        path = report / name
        if sha(path) != digest:
            raise ValueError('Original report file changed: ' + name)
        receipts[str(path.resolve())] = digest
    files = {
        'analysis': (base / 'R0076-confirmation-analysis.json', manifest['analysis_sha256']),
        'replay': (base / 'R0076-replay.json', manifest['replay_sha256']),
        'C1': (base / 'R0073-confirmation-analysis.json', manifest['C1_analysis_sha256']),
    }
    values = {}
    for name, (path, digest) in files.items():
        if sha(path) != digest:
            raise ValueError('Evidence changed: ' + name)
        values[name] = read(path)
        if not values[name].get('complete'):
            raise ValueError('Completed evidence required: ' + name)
        receipts[str(path.resolve())] = digest
    if values['replay']['analysis_sha256'] != manifest['analysis_sha256']:
        raise ValueError('Replay is not bound to this analysis')
    receipts[str(manifest_path.resolve())] = sha(manifest_path)
    return values, receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base = ROOT / 'artifacts/robustness'
    out = fresh_output(base, args.out)
    renderer = ROOT / 'scripts/robustness_r2_evidence_report.py'
    data, receipts = verified_inputs(base, renderer)
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out / 'rerender_source.py')
    shutil.copy2(renderer, out / 'renderer_source.py')
    with (out / 'README_ZH.md').open('x', encoding='utf-8') as stream:
        stream.write(render(data['analysis'], data['replay'], data['C1']))
    figures(data['analysis'], out)
    original = base / 'R0076-evidence-report'
    png_exact = {p.name: sha(p) == sha(original / p.name) for p in out.glob('*.png')}
    write_json(out / 'manifest.json', {
        'run_id': 'R0076', 'complete': True,
        'kind': 'RERENDER_ONLY_NOT_NEW_SCIENTIFIC_REPEAT',
        'provenance': 'SIMULATION_NOT_PATIENT_DATA',
        'source_evidence_sha256': receipts,
        'renderer_sha256': sha(renderer), 'entrypoint_sha256': sha(__file__),
        'figure_png_byte_identical_to_original': png_exact,
        'generated_files': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'limits': ['No new fits, samples, intervals, thresholds or promotion decision',
                   'Report timestamps/SVG metadata may differ; numerical evidence is unchanged',
                   'Copied source files are provenance; use the checked project CLI to rerender'],
    })
    print(out, flush=True)


if __name__ == '__main__':
    main()
