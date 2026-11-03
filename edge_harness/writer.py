"""JSONL writer that puts the run identity on the first line.

Same format as ``mlsys_harness.writer``: the first record has ``kind: run`` with
the commit, dirty flag, clock, interpreter, platform and argv. Readers skip it
with ``read_rows``.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _git(args: list[str], cwd: Path) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def provenance(cwd: Path | None = None) -> dict:
    root = cwd or Path.cwd()
    commit = _git(["rev-parse", "HEAD"], root)
    dirty = _git(["status", "--porcelain"], root)
    return {
        "kind": "run",
        "git_commit": commit or None,
        "git_dirty": bool(dirty) if commit else None,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
    }


class JsonlWriter:
    """Append-only JSONL. The first record is the provenance line."""

    def __init__(self, path: Path, cwd: Path | None = None, fresh: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and self.path.exists():
            self.path.unlink()
        self._f = open(self.path, "a", encoding="utf-8")
        self.write(provenance(cwd))

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_rows(path: Path) -> list[dict]:
    """Rows of a JSONL file without the provenance line."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "run":
            continue
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.name} ({len(rows)} rows)")


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.name}")
