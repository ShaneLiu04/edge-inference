from collections import OrderedDict

import pytest
import torch
from mcue2e import costs
from mcue2e.fold import fold_bn, remaining_bn
from torch import nn


def _layer(cin, cout, k=3, s=1, groups=1, act=True):
    mods = OrderedDict(conv=nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), bn=nn.BatchNorm2d(cout))
    if act:
        mods["act"] = nn.ReLU6()
    return nn.Sequential(mods)


def _net():
    torch.manual_seed(0)
    net = nn.Sequential(_layer(3, 16, s=2), _layer(16, 16, groups=16), _layer(16, 8, k=1, act=False),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2))
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-1, 1)
            m.running_var.uniform_(0.5, 2)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.5, 0.5)
    return net.eval()


def test_fold_preserves_outputs():
    net = _net()
    f = fold_bn(net)
    assert f.folded_pairs == 3 and remaining_bn(f) == 0 and remaining_bn(net) == 3
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(net(x), f(x), atol=1e-5)


def test_fold_skips_bn_before_conv():
    m = nn.Sequential(OrderedDict(bn=nn.BatchNorm2d(3), conv=nn.Conv2d(3, 4, 1))).eval()
    assert fold_bn(m).folded_pairs == 0


def test_flash_bytes_counts():
    f = fold_bn(_net())
    fb = costs.flash_bytes(f)
    w = 3 * 16 * 9 + 16 * 9 + 16 * 8 + 8 * 2
    assert fb["weights"] == w
    assert fb["bias"] == 4 * (16 + 16 + 8 + 2)
    assert fb["total"] == fb["weights"] + fb["bias"] + fb["scales"]
    assert costs.flash_bytes(f, per_channel=False)["scales"] < fb["scales"]


def test_conv_macs():
    f = fold_bn(_net())
    assert costs.conv_macs(f, (1, 3, 32, 32)) == 16 * 16 * 16 * 27 + 16 * 16 * 16 * 9 + 16 * 16 * 8 * 16 + 16


def test_latency_fit_reproduces_tinyengine_points():
    fit = costs.fit_latency()
    for mm, ms in costs.TINYENGINE_VWW.values():
        assert fit(mm) == pytest.approx(ms, rel=fit.max_rel_residual + 1e-9)
    assert 3.5 < fit.ms_per_mmac < 5
    assert fit.max_rel_residual < 0.1
    assert costs.fit_latency(through_origin=True).intercept_ms == 0
