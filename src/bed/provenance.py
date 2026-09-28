"""Run manifests and atomic result writes for the investigation experiments.

Every run directory gets a ``manifest.json`` with the source revision (from git if the tree is a
checkout, else from a ``SOURCE_REVISION`` file written by the Pelle staging script), a hash of the
configuration, package versions, host and scheduler identifiers, and timestamps. Results are
written atomically (temporary file + rename) so a killed job never leaves a truncated JSON behind,
and an existing complete output makes a restarted job a no-op (idempotent shards).
"""

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path


def source_revision(root=None):
    """``(sha, dirty)`` of the source tree containing this file, from git or ``SOURCE_REVISION``."""
    root = Path(root or Path(__file__).resolve().parents[2])
    marker = root / "SOURCE_REVISION"
    if marker.exists():
        parts = marker.read_text().split()
        return parts[0], (len(parts) > 1 and parts[1] == "dirty")
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL).strip())
        return sha, dirty
    except Exception:
        return "unknown", True


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def versions():
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "jax", "jaxlib", "flax"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def manifest(config, extra=None):
    sha, dirty = source_revision()
    m = {"source_sha": sha, "source_dirty": dirty, "config": config, "config_hash": config_hash(config),
         "versions": versions(), "host": socket.gethostname(), "platform": platform.platform(),
         "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
         "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_") and k in
                   ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NODELIST", "SLURM_MEM_PER_CPU")},
         "package_path": str(Path(__file__).resolve().parent)}
    if extra:
        m.update(extra)
    return m


def write_json_atomic(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_default)
    os.replace(tmp, path)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except Exception:
        pass
    return str(o)


def shard_done(path):
    """True if a COMPLETE JSON result already exists at ``path`` (idempotent restarts).

    Parseable is not the same as complete. Runners that write incrementally leave a valid JSON file
    after every row, so a job killed partway through leaves a file this function used to accept --
    and the resubmitted task then skipped it, silently losing the rows that had never been written.
    That happened: a smoke test killed after four of eight subjects left a shard the production
    array declined to redo, and the run finished 3996 subjects while reporting success.

    Runners therefore record `partial` in the manifest while rows remain, and this reads it. A file
    written before that flag existed has no `partial` key and is treated as complete, so older
    results are unaffected.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return False
    try:
        return not bool(d["manifest"]["config"]["partial"])
    except (KeyError, TypeError):
        return True


class Timer:
    """Wall-clock and process-CPU time of a block, for the cost accounting in every result."""

    def __enter__(self):
        self.t0 = time.time(); self.c0 = time.process_time(); return self

    def __exit__(self, *a):
        self.wall = time.time() - self.t0; self.cpu = time.process_time() - self.c0

    def as_dict(self):
        return {"wall_s": self.wall, "cpu_s": self.cpu}
