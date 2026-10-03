"""Serialized experiment registry updates; do not lose concurrent run records."""
import json
import os
from contextlib import contextmanager

from .robustness_io import write_json


@contextmanager
def registry_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as handle:
        handle.seek(0,2)
        if handle.tell()==0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name=="nt":
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def update_registry(path,entry,create=False):
    with registry_lock(path):
        registry=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"experiments":[]}
        positions=[i for i,r in enumerate(registry["experiments"]) if r["id"]==entry["id"]]
        if create and positions:
            raise ValueError("Experiment ID already exists; preserve history")
        if len(positions)>1:
            raise ValueError("Duplicate experiment IDs in registry")
        if positions:
            registry["experiments"][positions[0]]=entry
        else:
            registry["experiments"].append(entry)
        write_json(path,registry)
