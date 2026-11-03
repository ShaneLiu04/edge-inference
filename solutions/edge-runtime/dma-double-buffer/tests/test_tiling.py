from pathlib import Path

import pytest
from edge_harness.cpp import build, compiler, run_json
from tiling.dory import TARGETS, Conv, ideal_cycles, search
from tiling.pipeline import double_buffer_time, pipelined_time, serial_time, simulate

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]


@pytest.mark.parametrize("c,m", [(3.0, 1.0), (1.0, 3.0), (2.0, 2.0), (0.1, 5.0)])
def test_two_stage_double_buffer_matches_closed_form(c, m):
    for n in (1, 2, 7, 50):
        assert simulate(n, [m, c], buffers=2) == pytest.approx(double_buffer_time(n, c, m))
        assert simulate(n, [m, c], buffers=1) == pytest.approx(serial_time(n, [m, c]))


def test_three_stage_pipeline_with_separate_engines():
    st = [1.0, 2.5, 0.7]
    for n in (1, 4, 20):
        assert simulate(n, st, buffers=3) == pytest.approx(pipelined_time(n, st))


def test_shared_dma_engine_adds_in_and_out():
    # One DMA channel: in + out share it, so the DMA side costs 1.0 + 0.7 per tile against compute 1.2.
    t = simulate(40, [1.0, 1.2, 0.7], buffers=2, shared_dma=True)
    assert t > 40 * 1.2 + 1.0
    assert t == pytest.approx(40 * 1.7, rel=0.06)


def test_more_buffers_never_hurt():
    st = [1.0, 2.0, 1.5]
    times = [simulate(30, st, b) for b in (1, 2, 3, 4)]
    assert times == sorted(times, reverse=True)


def test_dory_tiles_fit_and_double_buffer_wins():
    layer = Conv(h=32, c=64, k=64)
    for t in TARGETS.values():
        best2 = search(layer, t, buffers=2)[0]
        best1 = search(layer, t, buffers=1)[0]
        assert best2.l1_bytes <= t.l1_bytes and best1.l1_bytes <= t.l1_bytes
        assert best2.total_cycles < best1.total_cycles
        assert best2.total_cycles >= ideal_cycles(layer, t)


def test_smaller_l1_forces_smaller_tiles_and_more_time():
    layer = Conv(h=32, c=64, k=64)
    big = search(layer, TARGETS["pulp-cluster"])[0]
    from dataclasses import replace

    small_t = replace(TARGETS["pulp-cluster"], l1_bytes=16 * 1024)
    small = search(layer, small_t)[0]
    assert small.l1_bytes <= 16 * 1024 < big.l1_bytes
    assert small.total_cycles > big.total_cycles


def test_depthwise_from_offchip_is_dma_bound_and_overlap_is_capped():
    layer = Conv(32, 96, 96, depthwise=True)
    t = TARGETS["pulp-l3-hyperram"]
    single, double = search(layer, t, 1)[0], search(layer, t, 2)[0]
    assert double.dma_cycles > double.compute_cycles
    per_tile_cap = (double.dma_cycles + double.compute_cycles) / double.dma_cycles
    assert 1.0 < single.total_cycles / double.total_cycles <= per_tile_cap + 0.05
    assert search(layer, TARGETS["pulp-cluster"], 2)[0].total_cycles < double.total_cycles


def test_cpp_rig_smoke(tmp_path):
    if compiler() is None:
        pytest.skip("no C++ compiler")
    exe = build(LAB / "cpp/dma.cpp", tmp_path, includes=(ROOT / "edge_harness/include",))
    rows = run_json(exe, ["--mode=tiles", "--work=1", "--repeats=1"])
    assert len(rows) == 5
    for r in rows:
        assert r["serial_s"] > 0 and r["double_s"] > 0
        assert r["model_double_s"] <= r["model_serial_s"]
