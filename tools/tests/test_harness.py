import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from edge_harness.claims import ClaimError, check_spec
from edge_harness.cpp import BuildError, build, compiler, run_json
from edge_harness.timing import percentile, summarize
from edge_harness.writer import JsonlWriter, read_rows, write_csv


def test_percentile_interpolates_and_handles_one_sample():
    assert percentile([3.0], 0.9) == 3.0
    assert percentile([0.0, 10.0], 0.25) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        percentile([], 0.5)
    r = summarize([float(i) for i in range(101)], "per_call")
    assert (r.p10_s, r.p50_s, r.p99_s) == (10.0, 50.0, 99.0)


def test_jsonl_provenance_is_skipped_by_readers(tmp_path):
    path = tmp_path / "x.jsonl"
    with JsonlWriter(path, cwd=tmp_path) as w:
        w.write({"kind": "gemm", "gops": 1.5})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["kind"] == "run"
    assert read_rows(path) == [{"kind": "gemm", "gops": 1.5}]


def test_claims_field_ratio_and_failures(tmp_path):
    with JsonlWriter(tmp_path / "r.jsonl", cwd=tmp_path) as w:
        w.write({"method": "a", "us": 10.0})
        w.write({"method": "b", "us": 4.0})
    write_csv(tmp_path / "t.csv", [{"k": 1, "v": 2.5}, {"k": 2, "v": 7.0}])
    (tmp_path / "s.json").write_text(json.dumps({"acc": {"w8": 0.88}, "net w0.35@96:peak": 7, "exact": True}),
                                     encoding="utf-8")
    assert check_spec(tmp_path, {"file": "s.json", "field": "exact", "equals": True}) is True
    with pytest.raises(ClaimError, match="want False"):
        check_spec(tmp_path, {"file": "s.json", "field": "exact", "equals": False})
    with pytest.raises(ClaimError, match="not numeric"):
        check_spec(tmp_path, {"file": "s.json", "field": "exact", "expect": 1, "tol_abs": 0})
    assert check_spec(tmp_path, {"file": "s.json", "field": "net w0.35@96:peak", "expect": 7, "tol_abs": 0}) == 7
    assert check_spec(tmp_path, {"file": "r.jsonl", "where": {"method": "a"}, "field": "us", "expect": 10, "tol_abs": 0}) == 10
    assert check_spec(tmp_path, {"file": "t.csv", "where": {"k": 2}, "field": "v", "min": 7, "max": 7}) == 7
    assert check_spec(tmp_path, {"file": "s.json", "field": "acc.w8", "expect": 0.88, "tol_rel": 1e-9}) == 0.88
    ratio = {"type": "ratio", "num": {"file": "r.jsonl", "where": {"method": "a"}, "field": "us"},
             "den": {"file": "r.jsonl", "where": {"method": "b"}, "field": "us"}, "expect": 2.5, "tol_abs": 1e-9}
    assert check_spec(tmp_path, ratio) == 2.5
    with pytest.raises(ClaimError, match="matched 2 rows"):
        check_spec(tmp_path, {"file": "r.jsonl", "field": "us", "min": 0})
    with pytest.raises(ClaimError, match="expect"):
        check_spec(tmp_path, {"file": "r.jsonl", "where": {"method": "b"}, "field": "us", "expect": 5, "tol_abs": 0.1})
    with pytest.raises(ClaimError, match="missing"):
        check_spec(tmp_path, {"file": "nope.jsonl", "field": "us", "min": 0})


@pytest.mark.skipif(compiler() is None, reason="no C++ compiler")
def test_build_caches_and_reports_json(tmp_path):
    src = tmp_path / "hello.cpp"
    src.write_text('#include "edgebench.h"\nint main(){ eb::Json().s("kind","hi").i("n",3).emit(); }\n', encoding="utf-8")
    inc = (ROOT / "edge_harness/include",)
    exe = build(src, tmp_path / "bin", includes=inc)
    assert build(src, tmp_path / "bin", includes=inc) == exe
    assert run_json(exe) == [{"kind": "hi", "n": 3}]
    src.write_text("int main(){ return }\n", encoding="utf-8")
    with pytest.raises(BuildError):
        build(src, tmp_path / "bin")


POOL_PROGRAM = r"""
#include "edgebench.h"
#include <atomic>
int main() {
    std::vector<int64_t> hits(1000, 0);
    std::atomic<int64_t> sum{0};
    for (int call = 0; call < 2000; ++call)
        eb::parallel_for(4, 1000, [&](int64_t lo, int64_t hi, int) {
            int64_t s = 0;
            for (int64_t i = lo; i < hi; ++i) { ++hits[i]; s += i; }
            sum += s;
        });
    bool ok = true;
    for (int64_t h : hits) ok = ok && h == 2000;
    eb::Stats st = eb::time_fn([] { eb::parallel_for(4, 4, [](int64_t, int64_t, int) {}); }, 10, 15, 1e-3);
    eb::Json().b("every_index_once", ok).i("sum", sum.load()).f("empty_call_us", st.p50 * 1e6).emit();
}
"""


@pytest.mark.skipif(compiler() is None, reason="no C++ compiler")
def test_parallel_for_pool_covers_each_index_once_and_is_cheap(tmp_path):
    src = tmp_path / "pool.cpp"
    src.write_text(POOL_PROGRAM, encoding="utf-8")
    (row,) = run_json(build(src, tmp_path / "bin", includes=(ROOT / "edge_harness/include",)))
    assert row["every_index_once"] and row["sum"] == 2000 * 999 * 1000 // 2
    assert row["empty_call_us"] < 100, "a pool dispatch should cost microseconds, not a thread launch"
