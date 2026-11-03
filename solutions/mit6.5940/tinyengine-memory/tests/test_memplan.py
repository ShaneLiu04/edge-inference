import pytest
import torch
from edgequant.model import MicroMobileNetV2
from memplan.graph import Tensor, inplace_depthwise, live_profile, trace
from memplan.patch import input_extent, patch_plan, spatial_convs
from memplan.planners import PLANNERS, arena_size, first_fit, greedy_by_size, lower_bound, validate
from torch import nn


def T(name, size, b, d):
    return Tensor(name, size, b, d, (size,))


def test_lower_bound_and_planners_on_a_chain():
    ts = [T("a", 100, 0, 1), T("b", 50, 1, 2), T("c", 100, 2, 3)]
    assert lower_bound(ts) == 150
    for fn in PLANNERS.values():
        off = fn(ts)
        validate(ts, off)
    assert arena_size(ts, PLANNERS["naive"](ts)) == 250
    assert arena_size(ts, greedy_by_size(ts)) == 150


def test_first_fit_fragments_where_greedy_by_size_does_not():
    # Small long-lived buffer placed first at offset 0 pushes the big late buffers up.
    ts = [T("small", 10, 0, 4), T("big1", 100, 1, 2), T("mid", 60, 0, 0), T("big2", 100, 3, 4)]
    ff = arena_size(ts, first_fit(ts))
    gs = arena_size(ts, greedy_by_size(ts))
    assert gs <= ff
    assert lower_bound(ts) <= gs


def test_validate_catches_overlap():
    ts = [T("a", 10, 0, 2), T("b", 10, 1, 3)]
    with pytest.raises(AssertionError):
        validate(ts, {"a": 0, "b": 5})


def test_trace_fuses_bn_relu_and_keeps_residual_input_alive():
    m = MicroMobileNetV2().eval()
    ts = trace(m, (1, 3, 32, 32))
    producers = [t.producer for t in ts]
    assert not any("bn" in p.lower() or "relu" in p.lower() for p in producers)
    adds = [t for t in ts if "add" in t.producer]
    assert adds, "residual adds must be buffers"
    names = {t.name: t for t in ts}
    for a in adds:
        for src in a.inputs:
            assert names[src].death >= a.birth
    assert ts[0].birth == 0 and ts[0].size == 3 * 32 * 32 // 16 * 16


def test_planners_valid_and_ordered_on_mobilenet():
    ts = trace(MicroMobileNetV2(), (1, 3, 32, 32))
    lb = lower_bound(ts)
    sizes = {}
    for name, fn in PLANNERS.items():
        off = fn(ts)
        validate(ts, off)
        sizes[name] = arena_size(ts, off)
    assert lb <= sizes["greedy_by_size"] <= sizes["naive"]
    assert lb <= sizes["first_fit"] <= sizes["naive"]
    assert max(live_profile(ts)) == lb


def test_inplace_depthwise_lowers_peak():
    ts = trace(MicroMobileNetV2(), (1, 3, 64, 64))
    ip = inplace_depthwise(ts)
    off = greedy_by_size(ip)
    validate(ip, off)
    assert lower_bound(ip) < lower_bound(ts)
    assert any(t.producer == "scratch" for t in ip)


def test_receptive_field_extent():
    assert input_extent([(3, 1)], 4) == 6
    assert input_extent([(3, 2), (3, 1)], 4) == 13
    assert input_extent([(1, 1), (3, 2), (1, 1)], 5) == 11
    seq = nn.Sequential(nn.Conv2d(3, 4, 3, 2, 1), nn.Conv2d(4, 4, 1), nn.Conv2d(4, 4, 3, 1, 1, groups=4))
    assert spatial_convs(list(seq)) == [(3, 2), (1, 1), (3, 1)]


def test_patch_plan_trades_memory_for_compute():
    m = MicroMobileNetV2().eval()
    head = nn.Sequential(m.pool, nn.Flatten(), m.fc)
    full = patch_plan(m.features, head, 128, cut=4, patches=1)
    tiled = patch_plan(m.features, head, 128, cut=4, patches=4)
    assert tiled.patch_stage_peak < full.peak
    assert tiled.mac_overhead > 1.0
    assert tiled.peak == max(tiled.patch_stage_peak, tiled.rest_peak)


def test_trace_handles_torchvision_mobilenet():
    tv = pytest.importorskip("torchvision")
    m = tv.models.mobilenet_v2(weights=None, width_mult=0.35)
    ts = trace(m, (1, 3, 96, 96))
    off = greedy_by_size(ts)
    validate(ts, off)
    assert lower_bound(ts) <= arena_size(ts, off) < sum(t.size for t in ts)
    with torch.no_grad():
        assert m.eval()(torch.zeros(1, 3, 96, 96)).shape == (1, 1000)
