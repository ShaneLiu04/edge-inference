from pathlib import Path

import numpy as np
import pytest
from edge_harness.cpp import build, compiler, run_json
from hostroof.analysis import attainable, conv_intensity, efficiency, plateaus, ridge

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parents[2]


def test_plateaus_find_three_levels():
    sizes = [2 ** k for k in range(14, 30)]
    bw = [150, 152, 149, 151, 80, 82, 79, 81, 40, 41, 39, 12, 11, 12, 11, 12]
    levels = plateaus(sizes, bw)
    assert [lv.name for lv in levels] == ["L1", "L2", "L3", "DRAM"]
    assert levels[0].gbps == pytest.approx(150.5)
    assert levels[-1].gbps == pytest.approx(12)
    assert levels[-1].lo_bytes == 2 ** 25


def test_plateaus_ignore_noise():
    sizes = [2 ** k for k in range(10)]
    bw = [100, 90, 110, 95, 105, 100, 92, 108, 99, 101]
    assert len(plateaus(sizes, bw)) == 1


def test_plateaus_reject_bad_input():
    with pytest.raises(ValueError):
        plateaus([], [])


def test_ridge_and_attainable():
    assert ridge(140, 14) == pytest.approx(10)
    got = attainable(np.array([1, 10, 100]), 140, 14)
    assert got.tolist() == [14, 140, 140]


def test_conv_intensity_and_efficiency():
    # 1x1 conv 56x56, 24 -> 144 channels, fp32: low intensity, bandwidth bound on any CPU.
    macs = 56 * 56 * 24 * 144
    act = 4 * 56 * 56 * (24 + 144)
    wts = 4 * 24 * 144
    i = conv_intensity(macs, act, wts)
    assert 10 < i < 21
    assert efficiency(70, i, 140, 14) == pytest.approx(70 / min(140, 14 * i))


def test_cpp_smoke(tmp_path):
    if compiler() is None:
        pytest.skip("no C++ compiler")
    exe = build(LAB / "cpp/roofline.cpp", tmp_path, includes=(ROOT / "edge_harness/include",))
    rows = run_json(exe, ["--mode=bw", "--threads=1", "--max-mib=1"])
    assert rows and all(r["GBps"] > 0 for r in rows)
    fma = run_json(exe, ["--mode=fma", "--threads=1"])
    assert fma[0]["Gops"] > 1
