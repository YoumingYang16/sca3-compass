"""Fail-closed, source-isolated replay of robustness_screen archives (stdlib only).

This is an operational reproducibility check, NEVER independent confirmation.
No current sca3_compass module is imported by this launcher.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import gzip
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

RUNNER = "scripts/robustness_screen.py"
PACKAGE = "sca3_compass"
BOOTSTRAP = "_replay_bootstrap.py"
FEATURES = ("mc", "soft", "loading", "prior", "domain", "patterns", "compact", "transport_patterns", "pilot_patterns", "joint_patterns", "block_patterns", "loading_patterns", "loading_audit_patterns", "predictive_patterns", "directional_patterns", "pilot_selection_patterns", "rank_budget_patterns", "continuous_patterns")
METRICS = ("fdp_by_repetition", "power_by_repetition")


class ReplayError(ValueError):
    """Unsafe, incomplete, incompatible, or nonmatching replay evidence."""


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReplayError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ReplayError(f"Non-finite JSON number: {value}")

    payload=(gzip.open(path,'rt',encoding='utf-8').read() if Path(path).suffix=='.gz' else Path(path).read_text(encoding='utf-8'))
    return json.loads(payload,
                      object_pairs_hook=unique, parse_constant=invalid)


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    """Exclusive creation; neither historical nor existing replay files replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", value):
        raise ReplayError("Run IDs must be safe single path components, starting with a letter")
    return value


def relative_source(value):
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (path.is_absolute() or ":" in normalized or ".." in path.parts
            or normalized != path.as_posix() or not normalized.endswith(".py")):
        raise ReplayError(f"Unsafe source path: {value}")
    if normalized != RUNNER and not normalized.startswith("src/sca3_compass/"):
        raise ReplayError(f"Unsupported source outside the frozen package/runner: {value}")
    return normalized


def contained(root, relative):
    root = Path(root).resolve()
    target = root.joinpath(*PurePosixPath(relative).parts)
    if not target.resolve().is_relative_to(root):
        raise ReplayError(f"Path escapes archive: {relative}")
    for item in (target, *target.parents):
        if item == root:
            break
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ReplayError(f"Linked archive paths are not accepted: {relative}")
    return target


def module_map(sources):
    result = {"robustness_screen": RUNNER}
    for path in sources:
        if path.startswith("src/"):
            name = path[4:-3].replace("/", ".")
            name = name.removesuffix(".__init__")
            if name in result:
                raise ReplayError(f"Ambiguous frozen module: {name}")
            result[name] = path
    return result


def audit_source(snapshot, protocol, *, staged=False):
    """Independently hash ALL listed files and walk local transitive imports.

    Runtime guarding also rejects missing dynamically requested package modules.
    Static audit is deliberately conservative, including imports inside functions.
    """
    hashes = protocol.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ReplayError("Protocol has no source SHA-256 manifest")
    sources, normalized_hashes = {}, {}
    for raw, expected in hashes.items():
        path = relative_source(raw)
        if path.casefold() in {p.casefold() for p in sources}:
            raise ReplayError(f"Duplicate normalized source path: {raw}")
        if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
            raise ReplayError(f"Invalid source SHA-256: {raw}")
        source = contained(snapshot, path)
        if not source.is_file():
            raise ReplayError(f"Missing frozen source: {path}; current source is NEVER substituted")
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ReplayError(f"Frozen source hash mismatch: {path}")
        sources[path], normalized_hashes[path] = payload, expected
    python_files = (list((Path(snapshot) / "src").rglob("*.py"))
                    + list((Path(snapshot) / "scripts").rglob("*.py"))) if staged else Path(snapshot).rglob("*.py")
    extras = {p.relative_to(snapshot).as_posix() for p in python_files} - set(sources)
    if extras:
        raise ReplayError(f"Unhashed Python sources in archive: {sorted(extras)}")
    modules = module_map(sources)
    todo, visited, external = ["robustness_screen", PACKAGE], set(), set()
    while todo:
        module = todo.pop()
        if module in visited:
            continue
        if module not in modules or modules[module] not in sources:
            raise ReplayError(f"Missing transitive module {module}; archive is NOT a full replay freeze")
        visited.add(module)
        filename = modules[module]
        tree = ast.parse(sources[filename], filename=filename)
        package = module if filename.endswith("/__init__.py") else module.rpartition(".")[0]
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    try:
                        base = importlib.util.resolve_name("." * node.level + base, package)
                    except (ImportError, ValueError) as exc:
                        raise ReplayError(f"Unresolvable relative import in {filename}") from exc
                imports = [base]
                if base in modules and modules[base].endswith("/__init__.py"):
                    for alias in node.names:
                        if alias.name == "*":
                            raise ReplayError(f"Package star import requires manual audit: {filename}")
                        child = base + "." + alias.name
                        if child in modules:
                            imports.append(child)
                        else:
                            # A package may export constants/functions, but a missing
                            # module must not be silently treated as such.
                            init = ast.parse(sources[modules[base]])
                            exports = {n.name for n in init.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                            exports |= {n.id for n in ast.walk(init) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                            if alias.name not in exports:
                                raise ReplayError(f"Unresolved package import {child}; manual audit needed")
            elif isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                if name in {"__import__", "import_module", "spec_from_file_location", "run_path"}:
                    raise ReplayError(f"Dynamic source loading needs explicit audit: {filename}:{node.lineno}")
            for dependency in imports:
                if dependency == PACKAGE or dependency.startswith(PACKAGE + ".") or dependency == "robustness_screen":
                    pieces = dependency.split(".")
                    todo.extend(".".join(pieces[:i]) for i in range(1, len(pieces) + 1))
                else:
                    external.add(dependency.split(".")[0])
    return {"source_sha256": normalized_hashes, "reachable_local_modules": sorted(visited),
            "external_import_roots": sorted(external), "verified_files": len(sources)}, sources


def load_archive(project, old_id):
    old_id = run_id(old_id)
    folder = Path(project).resolve() / "artifacts/robustness"
    protocol_path = contained(folder, old_id + "-screen.protocol.json")
    protocol = read_json(protocol_path)
    if protocol.get("run_id") != old_id or protocol.get("provenance") != "SIMULATION_NOT_PATIENT_DATA":
        raise ReplayError("This launcher supports registered simulation screen protocols only")
    audit, sources = audit_source(contained(folder, old_id + "-source"), protocol)
    return folder, protocol_path, protocol, audit, sources


def raw_case(result):
    reps = result.get("repetitions")
    if type(reps) is not int or reps < 1:
        raise ReplayError("Invalid repetition count in result")
    rows = {}
    for row in result["rows"]:
        name = row["method"]
        if not isinstance(name, str) or name in rows:
            raise ReplayError("Invalid or duplicate method name")
        values = {}
        for metric in METRICS:
            array = row[metric]
            if not isinstance(array, list) or len(array) != reps or any(
                    type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in array):
                raise ReplayError(f"Invalid raw {metric} for {name}")
            values[metric] = array
        rows[name] = values
    if not rows:
        raise ReplayError("No raw method arrays")
    return {"case": result["case"], "repetitions": reps, "rows": rows,
            "power_defined": result.get("power_defined"),
            "replicated_signed_truths": result.get("replicated_signed_truths")}


def reference_results(folder, protocol):
    """Capture comparison evidence, accepting both historical partial formats."""
    old_id = protocol["run_id"]
    completed = folder / f"{old_id}-screen.json"
    partial = folder / f"{old_id}-screen.partial.json"
    evidence, indexed = [], {}
    path = completed if completed.exists() else partial
    if not path.exists():
        raise ReplayError("No completed or partial output to replay")
    # Modern complete screens can have multi-gigabyte diagnostic summaries.
    # Their registered, independently checksummed per-case records are the
    # reference, not an in-memory parse of the duplicated aggregate. Do not
    # silently fall back to a less strict path if a declared checkpoint fails.
    directory=contained(folder,f"{old_id}-cases")
    if completed.exists() and directory.is_dir():
        registry=read_json(contained(folder,"EXPERIMENT_REGISTRY.json"))
        entries=[entry for entry in registry['experiments'] if entry['id']==old_id]
        if (len(entries)!=1 or entries[0].get('status')!='completed'
                or entries[0].get('settings')!=protocol):
            raise ReplayError('Completed checkpoints require matching completed registry settings')
        cases=protocol['cases']
        if len({content_digest(case) for case in cases})!=len(cases):
            raise ReplayError('Duplicate case definitions in checkpoint reference')
        expected_digest=content_digest(protocol)
        expected_paths={f'case-{index:04}.json'+('.gz' if protocol.get('compact') else '')
            for index in range(len(cases))}
        actual_paths={item.name for item in directory.glob('case-*.json*')}
        if actual_paths!=expected_paths:
            raise ReplayError('Complete checkpoint paths differ from exact expected case set')
        for index,case in enumerate(cases):
            checkpoint=contained(directory,f'case-{index:04}.json'+('.gz' if protocol.get('compact') else ''))
            item=read_json(checkpoint)
            if (item.get('case_index')!=index or item.get('settings_digest')!=expected_digest
                    or item.get('result_digest')!=content_digest(item['result'])
                    or item['result']['case']!=case):
                raise ReplayError('Checkpoint digest/index/case mismatch')
            indexed[index]=raw_case(item['result'])
            if indexed[index]['repetitions']!=protocol['repetitions']:
                raise ReplayError('Checkpoint repetition count differs from protocol')
            evidence.append({'path':str(checkpoint),'sha256':sha(checkpoint),
                'result_digest':item['result_digest']})
            del item
        if not indexed:
            raise ReplayError('Reference has no usable cases')
        return {'source_status':'completed','files':evidence,
            'cases':{str(i):r for i,r in sorted(indexed.items())},
            'reference_storage':'registered_complete_case_checkpoints; aggregate_not_parsed',
            'registry_entry_digest':content_digest(entries[0])}
    document = read_json(path)
    evidence.append({"path": str(path), "sha256": sha(path)})
    if document.get("settings") != protocol:
        raise ReplayError("Result/partial settings disagree with frozen protocol")
    cases = protocol["cases"]
    identities = [content_digest(case) for case in cases]
    if len(set(identities)) != len(identities):
        raise ReplayError("Duplicate case definitions make historical index matching ambiguous")
    if completed.exists() or "completed" in document:
        results = document["scenarios"] if completed.exists() else document["completed"]
        for result in results:
            identity = content_digest(result["case"])
            if identity not in identities:
                raise ReplayError("Unknown reference case")
            index = identities.index(identity)
            if index in indexed:
                raise ReplayError("Duplicate reference case")
            indexed[index] = raw_case(result)
        if completed.exists() and set(indexed) != set(range(len(cases))):
            raise ReplayError("Completed output is missing cases")
    elif document.get("schema_version") == 2:
        # Do not trust a machine-specific absolute checkpoint path in JSON.
        directory = contained(folder, f"{old_id}-cases")
        indices = document["completed_case_indices"]
        if len(set(indices)) != len(indices):
            raise ReplayError("Duplicate checkpoint indices")
        for index in indices:
            if type(index) is not int or not 0 <= index < len(cases):
                raise ReplayError("Invalid checkpoint index")
            checkpoint = contained(directory, f"case-{index:04}.json{('.gz' if protocol.get('compact') else '')}")
            item = read_json(checkpoint)
            evidence.append({"path": str(checkpoint), "sha256": sha(checkpoint)})
            if (item.get("case_index") != index or item.get("settings_digest") != content_digest(protocol)
                    or item.get("result_digest") != content_digest(item["result"])
                    or item["result"]["case"] != cases[index]):
                raise ReplayError("Checkpoint digest/index/case mismatch")
            indexed[index] = raw_case(item["result"])
    else:
        raise ReplayError("Unsupported partial output schema")
    if not indexed or any(r["repetitions"] != protocol["repetitions"] for r in indexed.values()):
        raise ReplayError("Reference has no usable cases or inconsistent repetitions")
    return {"source_status": "completed" if completed.exists() else "partial",
            "files": evidence, "cases": {str(i): r for i, r in sorted(indexed.items())}}


ENV_PROBE = r'''
import sys, sysconfig, pathlib, json, platform, importlib.metadata, hashlib
exe=pathlib.Path(sys.executable)
venv=exe.parent.parent if exe.parent.name.lower() in ('scripts','bin') else exe.parent
if (venv/'pyvenv.cfg').is_file():
    paths=[venv/'Lib/site-packages'] if sys.platform=='win32' else [venv/('lib/python%d.%d/site-packages'%sys.version_info[:2])]
else:
    paths=[pathlib.Path(sysconfig.get_path(k)) for k in ('purelib','platlib')]
paths=list(dict.fromkeys(str(p.resolve()) for p in paths if p.is_dir()))
packages=sorted([{'name':d.metadata['Name'],'version':d.version} for d in importlib.metadata.distributions(path=paths)],key=lambda x:(x['name'] or '',x['version']))
print(json.dumps({'python':sys.version,'executable':str(exe.resolve()),'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'platform':platform.platform(),'machine':platform.machine(),'dependency_paths':paths,'packages':packages,'site_hooks_disabled':True}))
'''


def environment(python):
    process = subprocess.run([str(python), "-I", "-S", "-c", ENV_PROBE],
                             check=True, text=True, encoding="utf-8", capture_output=True, timeout=30)
    return json.loads(process.stdout)


def runner_arguments(protocol, source, new_id, workers, repetitions, cases_path):
    allowed = {"phase", "run_id", "seed", "repetitions", "cases", "source_sha256", "provenance",
               "independent_confirmation", "draws", "profile", "calibration_information", *FEATURES}
    unknown = set(protocol) - allowed
    if unknown:
        raise ReplayError(f"Unrecognized protocol settings require explicit CLI mapping: {sorted(unknown)}")
    run_id(new_id)
    if type(workers) is not int or workers < 1:
        raise ReplayError("Workers must be positive")
    tree = ast.parse(source)
    supported = {arg.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                 for arg in n.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
    arguments = ["--run-id", new_id, "--repetitions", str(repetitions), "--seed", str(protocol["seed"]),
                 "--workers", str(workers), "--config", str(cases_path)]
    for key in ("draws", "profile"):
        if key in protocol:
            arguments.extend(["--" + key, str(protocol[key])])
    for key in FEATURES:
        if type(protocol.get(key, False)) is not bool:
            raise ReplayError(f"Nonboolean feature setting: {key}")
        if protocol.get(key, False):
            arguments.append("--" + key.replace('_','-'))
    if any(arg.startswith("--") and arg not in supported for arg in arguments):
        raise ReplayError("Frozen runner CLI does not support the recorded protocol options")
    return arguments


def stage_replay(project, old_id, new_id, *, python=sys.executable, workers=1,
                 smoke_repetitions=None, smoke_cases=None):
    old_id, new_id = run_id(old_id), run_id(new_id)
    if old_id.casefold() == new_id.casefold():
        raise ReplayError("Replay MUST have a new run ID")
    if type(workers) is not int or workers < 1:
        raise ReplayError("Workers must be positive")
    folder, protocol_path, protocol, audit, sources = load_archive(project, old_id)
    if any(folder.glob(new_id + "-*")):
        raise ReplayError("New run ID already has historical artifacts")
    registry = folder / "EXPERIMENT_REGISTRY.json"
    if registry.exists() and any(e["id"].casefold() == new_id.casefold() for e in read_json(registry)["experiments"]):
        raise ReplayError("New run ID is already registered")
    reference = reference_results(folder, protocol)
    repetitions = protocol["repetitions"] if smoke_repetitions is None else smoke_repetitions
    case_count = len(protocol["cases"]) if smoke_cases is None else smoke_cases
    if type(repetitions) is not int or not 1 <= repetitions <= protocol["repetitions"]:
        raise ReplayError("Replay repetition count must be a nonempty prefix of the original")
    if type(case_count) is not int or not 1 <= case_count <= len(protocol["cases"]):
        raise ReplayError("Smoke cases must be a nonempty PREFIX, preserving RNG case indices")
    env = environment(python)
    root = contained(folder, "replays/" + new_id)
    config = root / "configs/replay-cases.json"
    args = runner_arguments(protocol, sources[RUNNER], new_id, workers, repetitions, config)
    root.mkdir(parents=True, exist_ok=False)
    for relative, payload in sources.items():
        target = contained(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
    # Byte copies plus independent post-copy verification close hash/copy races.
    copied_audit, _ = audit_source(root, protocol)
    if copied_audit != audit:
        raise ReplayError("Post-copy source audit changed")
    bootstrap = Path(__file__).read_bytes()
    with (root / BOOTSTRAP).open("xb") as stream:
        stream.write(bootstrap)
    write_new(root / "original-protocol.json", protocol)
    write_new(root / "reference-arrays.json", reference)
    write_new(config, {"cases": protocol["cases"][:case_count]})
    write_new(root / "artifacts/robustness/EXPERIMENT_REGISTRY.json", {"experiments": []})
    manifest = {"schema_version": 1, "old_run_id": old_id, "new_run_id": new_id,
                "source_protocol": str(protocol_path), "source_protocol_sha256": sha(protocol_path),
                "source_archive": str(folder / (old_id + "-source")), "audit": audit,
                "python_environment": env, "runner_arguments": args,
                "repetitions": repetitions, "case_count": case_count,
                "mode": "smoke_prefix" if repetitions != protocol["repetitions"] or case_count != len(protocol["cases"]) else "full_replay",
                "independent_confirmation": False,
                "created_at": datetime.now(UTC).isoformat(),
                "integrity": {BOOTSTRAP: hashlib.sha256(bootstrap).hexdigest(),
                              "original-protocol.json": sha(root / "original-protocol.json"),
                              "reference-arrays.json": sha(root / "reference-arrays.json"),
                              "configs/replay-cases.json": sha(config)},
                "limitations": ["Dependencies and OS/native libraries are inventoried, not vendored or historically frozen.",
                                "Source isolation is not an OS security sandbox; execute only trusted frozen code.",
                                "Exact raw numeric replay does not validate scientific truth labels or establish independence.",
                                "No historical checkpoints are reused; partial archives are rerun from the original seeds."]}
    write_new(root / "replay-manifest.json", manifest)
    return root


class FrozenFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Hash every project import and compile source bytes, never stale bytecode."""

    def __init__(self, root, hashes):
        self.root, self.hashes = Path(root), hashes
        self.modules = module_map(hashes)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "robustness_screen" and fullname != PACKAGE and not fullname.startswith(PACKAGE + "."):
            return None
        if fullname not in self.modules:
            raise ImportError(f"Unfrozen project import rejected: {fullname}")
        relative = self.modules[fullname]
        return importlib.util.spec_from_file_location(
            fullname, contained(self.root, relative), loader=self,
            submodule_search_locations=[] if relative.endswith("/__init__.py") else None)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative = self.modules[module.__name__]
        source = contained(self.root, relative)
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.hashes[relative]:
            raise ImportError(f"Frozen import hash mismatch: {relative}")
        # Equivalent to a Python source loader, after explicit content validation.
        exec(compile(payload, str(source), "exec"), module.__dict__)  # noqa: S102


def verify_stage(root):
    root = Path(root).resolve()
    manifest = read_json(root / "replay-manifest.json")
    required_inputs = {BOOTSTRAP, "original-protocol.json", "reference-arrays.json", "configs/replay-cases.json"}
    if set(manifest["integrity"]) != required_inputs:
        raise ReplayError("Replay manifest is missing required input integrity checks")
    new_id = run_id(manifest["new_run_id"])
    old_id = run_id(manifest["old_run_id"])
    if (old_id.casefold() == new_id.casefold() or root.name != new_id or root.parent.name != "replays"):
        raise ReplayError("Replay root/IDs are inconsistent")
    for path, expected in manifest["integrity"].items():
        if sha(contained(root, path)) != expected:
            raise ReplayError(f"Replay input integrity mismatch: {path}")
    protocol = read_json(root / "original-protocol.json")
    # Independently rebuild the import/hash audit from the original protocol.
    # Exclude only the bootstrap and runner-generated artifacts, not src/scripts.
    audit, _ = audit_source(root, protocol, staged=True)
    if audit != manifest["audit"]:
        raise ReplayError("Staged source audit differs from original frozen protocol")
    if (protocol["run_id"] != old_id or type(manifest["case_count"]) is not int
            or not 1 <= manifest["case_count"] <= len(protocol["cases"])
            or type(manifest["repetitions"]) is not int
            or not 1 <= manifest["repetitions"] <= protocol["repetitions"]):
        raise ReplayError("Staged replay prefix/identity is invalid")
    config = read_json(root / "configs/replay-cases.json")
    if config["cases"] != protocol["cases"][:manifest["case_count"]]:
        raise ReplayError("Staged case prefix differs from original protocol")
    expected_args = runner_arguments(protocol, (root / RUNNER).read_bytes(), manifest["new_run_id"],
                                     int(manifest["runner_arguments"][manifest["runner_arguments"].index("--workers") + 1]),
                                     manifest["repetitions"], root / "configs/replay-cases.json")
    if expected_args != manifest["runner_arguments"]:
        raise ReplayError("Staged runner arguments differ from frozen protocol")
    return manifest, protocol


def bootstrap(invoke):
    """Executed in the parent AND Windows spawn children, with -I -S inherited."""
    root = Path(__file__).resolve().parent
    manifest, _ = verify_stage(root)
    if not sys.flags.isolated or not sys.flags.no_site:
        raise ReplayError("Frozen execution requires Python -I -S")
    # -S means .pth/editable hooks and user site have not executed. Add only
    # dependency directories, not their .pth contents or the live repository.
    sys.path[:] = [str(root / "src"), str(root / "scripts"), *sys.path,
                   *manifest["python_environment"]["dependency_paths"]]
    if any(name == PACKAGE or name.startswith(PACKAGE + ".") for name in sys.modules):
        raise ReplayError("Project package was imported before the frozen guard")
    sys.meta_path.insert(0, FrozenFinder(root, manifest["audit"]["source_sha256"]))
    import robustness_screen as runner

    from sca3_compass.molecular_data import PROJECT_ROOT
    if Path(PROJECT_ROOT).resolve() != root:
        raise ReplayError("Frozen PROJECT_ROOT points outside replay workspace")
    origins = {name: getattr(module, "__file__", None) for name, module in sys.modules.items()
               if name == "robustness_screen" or name == PACKAGE or name.startswith(PACKAGE + ".")}
    write_new(root / f"import-audits/process-{os.getpid()}.json",
              {"pid": os.getpid(), "role": "parent" if invoke else "spawn_worker",
               "project_root": str(PROJECT_ROOT), "origins": origins,
               "isolated": bool(sys.flags.isolated), "no_site": bool(sys.flags.no_site)})
    if invoke:
        sys.argv = [str(root / RUNNER), *manifest["runner_arguments"]]
        runner.main()


def compare_replay(root):
    root = Path(root).resolve()
    manifest, protocol = verify_stage(root)
    new_path = root / f"artifacts/robustness/{manifest['new_run_id']}-screen.json"
    result = read_json(new_path)
    settings = result["settings"]
    for key in ("seed", "draws", "profile", *FEATURES):
        if settings.get(key, False if key in FEATURES else None) != protocol.get(key, False if key in FEATURES else None):
            raise ReplayError(f"Regenerated protocol changed {key}")
    if (settings["run_id"] != manifest["new_run_id"] or settings["repetitions"] != manifest["repetitions"]
            or settings["cases"] != protocol["cases"][:manifest["case_count"]]
            or {relative_source(k): v for k, v in settings["source_sha256"].items()} != manifest["audit"]["source_sha256"]):
        raise ReplayError("Regenerated protocol/cases/source hashes disagree")
    reference = read_json(root / "reference-arrays.json")
    scenarios = result["scenarios"]
    if len(scenarios) != manifest["case_count"]:
        raise ReplayError("Regenerated output missing cases")
    count, mismatches, compared_cases = 0, [], []
    for index, scenario in enumerate(scenarios):
        new = raw_case(scenario)
        if new["case"] != settings["cases"][index] or new["repetitions"] != manifest["repetitions"]:
            raise ReplayError("Regenerated case ordering/repetition mismatch")
        old = reference["cases"].get(str(index))
        if old is None:
            continue
        compared_cases.append(index)
        if (set(new["rows"]) != set(old["rows"]) or new["power_defined"] != old["power_defined"]
                or new["replicated_signed_truths"] != old["replicated_signed_truths"]):
            mismatches.append({"case_index": index, "reason": "method set/truth metadata mismatch"})
        for method in sorted(set(new["rows"]) & set(old["rows"])):
            for metric in METRICS:
                count += 1
                a, b = new["rows"][method][metric], old["rows"][method][metric][:manifest["repetitions"]]
                if a != b:
                    mismatches.append({"case_index": index, "method": method, "metric": metric,
                                       "first_difference": next(i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                                       "max_absolute_difference": max(abs(x - y) for x, y in zip(a, b))})
    if not compared_cases:
        raise ReplayError("No reference cases overlap the requested replay prefix")
    return {"exact_match": not mismatches, "arrays_compared": count,
            "case_indices_compared": compared_cases, "mismatches": mismatches,
            "reference_status": reference["source_status"], "mode": manifest["mode"],
            "all_original_cases_compared": len(compared_cases) == len(protocol["cases"]),
            "full_repetition_arrays": manifest["repetitions"] == protocol["repetitions"],
            "regenerated_sha256": sha(new_path), "independent_confirmation": False}


def execute_replay(root, *, compare=False):
    root = Path(root).resolve()
    manifest, _ = verify_stage(root)
    if (root / "execution-start.json").exists():
        raise ReplayError("This replay stage has already been executed; choose another new run ID")
    env = environment(manifest["python_environment"]["executable"])
    if env != manifest["python_environment"]:
        raise ReplayError("Python/dependency inventory changed since staging; create a fresh stage")
    command = [env["executable"], "-I", "-S", str(root / BOOTSTRAP)]
    write_new(root / "execution-start.json", {"command": command, "environment": env,
                                            "started_at": datetime.now(UTC).isoformat()})
    # No shell, no inherited PYTHONPATH/.pth. Python isolation handles those;
    # explicit BLAS bounds avoid oversubscription without touching global settings.
    child_env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        child_env[name] = "1"
    with (root / "console.log").open("xb") as log:
        process = subprocess.run(command, cwd=root, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=False)
    finish = {"returncode": process.returncode, "finished_at": datetime.now(UTC).isoformat()}
    write_new(root / "execution-finish.json", finish)
    if process.returncode:
        raise ReplayError(f"Frozen replay failed (exit {process.returncode}); retained {root / 'console.log'}")
    verify_stage(root)
    if compare:
        comparison = compare_replay(root)
        write_new(root / "comparison.json", comparison)
        if not comparison["exact_match"]:
            raise ReplayError("Raw replay mismatch; see comparison.json (no tolerances silently applied)")
        return comparison
    return finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "stage"):
        item = commands.add_parser(name)
        item.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
        item.add_argument("--source-run", required=True)
        if name == "stage":
            item.add_argument("--new-run-id", required=True)
            item.add_argument("--python", default=sys.executable)
            item.add_argument("--workers", type=int, default=1)
            item.add_argument("--smoke-repetitions", type=int)
            item.add_argument("--smoke-cases", type=int)
    for name in ("execute", "compare"):
        item = commands.add_parser(name)
        item.add_argument("--stage", type=Path, required=True)
        if name == "execute":
            item.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "audit":
            _, _, _, result, _ = load_archive(args.project_root, args.source_run)
        elif args.command == "stage":
            root = stage_replay(args.project_root, args.source_run, args.new_run_id,
                                python=args.python, workers=args.workers,
                                smoke_repetitions=args.smoke_repetitions, smoke_cases=args.smoke_cases)
            result = {"stage": str(root), "status": "staged_not_executed"}
        elif args.command == "execute":
            result = execute_replay(args.stage, compare=args.compare)
        else:
            result = compare_replay(args.stage)
            if not result["exact_match"]:
                print(json.dumps(result, indent=2))
                return 2
        print(json.dumps(result, indent=2))
        return 0
    except (ReplayError, OSError, KeyError, SyntaxError, subprocess.SubprocessError) as exc:
        print(f"Replay rejected: {exc}", file=sys.stderr)
        return 2


if Path(__file__).name == BOOTSTRAP:
    # Spawn re-executes this file as __mp_main__; install the same guard and
    # import the same function-defining runner module, without launching twice.
    bootstrap(invoke=__name__ == "__main__")
elif __name__ == "__main__":
    raise SystemExit(main())
