"""Check numbers quoted in write-ups against the raw files they came from.

Same spec format as ``mlsys_harness.campaign.check_spec``: ``file``, optional
``where`` filter, ``field``, then ``expect`` with ``tol_abs`` / ``tol_rel``, or
``min`` / ``max``. ``type: ratio`` divides two such values. JSON files may be a
dict (one row), a list of rows, or a nested dict addressed with a dotted
``field`` such as ``output_rel_rmse.per_tensor``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


class ClaimError(RuntimeError):
    pass


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise ClaimError(f"missing result file {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ClaimError(f"{path} is not an object or a list")
    if path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    return [r for r in (json.loads(line) for line in text.splitlines() if line.strip()) if r.get("kind") != "run"]


def _match(row: dict, where: dict) -> bool:
    for key, want in where.items():
        got = row.get(key)
        if isinstance(got, str) and not isinstance(want, str):
            try:
                got = type(want)(got)
            except (TypeError, ValueError):
                return False
        if got != want:
            return False
    return True


def _dig(row: dict, field: str):
    if field in row:  # flat keys may contain dots, e.g. "mobilenet_v2 w0.35@160:lower_bound"
        return row[field]
    value = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ClaimError(f"field {field} not found")
        value = value[part]
    return value


def _raw(root: Path, spec: dict):
    rows = [r for r in _rows(root / spec["file"]) if _match(r, spec.get("where", {}))]
    if len(rows) != 1:
        raise ClaimError(f"{spec['file']} where {spec.get('where')} matched {len(rows)} rows, want 1")
    return _dig(rows[0], spec["field"])


def _value(root: Path, spec: dict) -> float:
    value = _raw(root, spec)
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as exc:
            raise ClaimError(f"{spec['field']} is not numeric: {value!r}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClaimError(f"{spec['field']} is not numeric: {value!r}")
    return float(value) * float(spec.get("scale", 1))


def check_spec(root: Path, spec: dict) -> float:
    """Raise ClaimError if a claim does not hold. Returns the value checked."""
    kind = spec.get("type", "field")
    if "equals" in spec:  # exact match for non-numeric values such as booleans
        got = _raw(root, spec)
        if got != spec["equals"]:
            raise ClaimError(f"{spec.get('id', spec['field'])}: got {got!r}, want {spec['equals']!r}")
        return got
    if kind == "field":
        got = _value(root, spec)
    elif kind == "ratio":
        den = _value(root, spec["den"])
        if den == 0:
            raise ClaimError(f"ratio denominator is 0 ({spec.get('id', spec)})")
        got = _value(root, spec["num"]) / den
    else:
        raise ClaimError(f"unknown check type {kind}")
    label = spec.get("id", spec.get("field", "check"))
    if "expect" in spec:
        tol = spec.get("tol_abs")
        if tol is None:
            tol = abs(spec["expect"]) * spec.get("tol_rel", 0.0)
        if abs(got - spec["expect"]) > tol:
            raise ClaimError(f"{label}: got {got:.6g}, expect {spec['expect']} ± {tol}")
    if "min" in spec and got < spec["min"]:
        raise ClaimError(f"{label}: got {got:.6g} < min {spec['min']}")
    if "max" in spec and got > spec["max"]:
        raise ClaimError(f"{label}: got {got:.6g} > max {spec['max']}")
    return got
