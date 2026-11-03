import numpy as np
import pytest
import torch
from edgequant.model import MicroMobileNetV2, count_macs, count_params
from edgequant.qsim import QuantConfig, QuantSim, act_points, fold_model, is_depthwise, weight_layers
from edgequant.quant import (
    act_qparams,
    calib_kl,
    calib_kl_nonzero,
    calib_minmax,
    calib_mse,
    calib_percentile,
    fake_quant_act,
    fake_quant_weight,
    quant_mse,
    sqnr_db,
    ste_act,
)


def _randomize_bn(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5, generator=g)
            m.running_var.uniform_(0.2, 2.0, generator=g)
            m.weight.data.uniform_(0.1, 3.0, generator=g)
            m.bias.data.uniform_(-0.2, 0.2, generator=g)
    return model.eval()


def test_model_size_is_stable():
    m = MicroMobileNetV2()
    assert count_params(m) == 209354
    assert count_macs(m) == 8571392


def test_fold_model_preserves_outputs():
    torch.manual_seed(0)
    model = _randomize_bn(MicroMobileNetV2())
    folded = fold_model(model)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in folded.modules())
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), folded(x), atol=1e-4, rtol=1e-4)


def test_act_qparams_cover_zero_and_relu6():
    scale, zp = act_qparams(0.0, 6.0, 8)
    assert zp == 0
    assert scale == pytest.approx(6.0 / 255)
    scale, zp = act_qparams(-1.0, 3.0, 8)
    assert 0 < zp < 255
    x = torch.tensor([0.0])
    assert fake_quant_act(x, scale, zp, 8).item() == pytest.approx(0.0, abs=1e-9)


def test_weight_fake_quant_error_bound():
    w = torch.randn(8, 4, 3, 3)
    for per_channel in (True, False):
        wq = fake_quant_weight(w, 8, per_channel)
        amax = w.abs().reshape(8, -1).amax(1).reshape(-1, 1, 1, 1) if per_channel else w.abs().amax()
        assert torch.all((wq - w).abs() <= amax / 127 / 2 + 1e-7)


def test_calibrators_on_heavy_tail():
    rng = np.random.default_rng(0)
    x = np.abs(rng.standard_normal(200_000)).astype(np.float32)
    x[:20] = 40.0
    lo, hi = calib_minmax(x)
    assert hi == pytest.approx(40.0)
    _, hi_p = calib_percentile(x, pct=99.99)
    assert hi_p < 10.0
    lo_m, hi_m = calib_mse(x)
    assert quant_mse(x, lo_m, hi_m, 8) <= quant_mse(x, lo, hi, 8)
    _, hi_k = calib_kl(x, bits=8)
    assert 2.0 < hi_k < 40.0


def test_kl_signed_returns_symmetric_range():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(100_000)
    lo, hi = calib_kl(x, bits=8)
    assert lo == pytest.approx(-hi)


def test_zero_spike_drags_kl_threshold_down():
    rng = np.random.default_rng(2)
    n = 200_000
    x = np.where(rng.random(n) < 0.6, 0.0, np.minimum(rng.exponential(0.7, n), 6.0)).astype(np.float32)
    _, hi_mse = calib_mse(x)
    _, hi_kl = calib_kl(x)
    _, hi_nz = calib_kl_nonzero(x)
    assert hi_kl < 0.4 * hi_mse
    assert hi_nz > 0.7 * hi_mse


def test_ste_gradient_masks_clipped_values():
    x = torch.tensor([-1.0, 0.5, 2.0, 10.0], requires_grad=True)
    y = ste_act(x, 0.01, 0, 8)
    y.sum().backward()
    assert x.grad.tolist() == [0.0, 1.0, 1.0, 0.0]


def test_sqnr():
    a = np.ones(100)
    assert sqnr_db(a, a * 1.1) == pytest.approx(20.0)


def test_quantsim_16bit_is_close_to_float():
    torch.manual_seed(1)
    folded = fold_model(_randomize_bn(MicroMobileNetV2(), seed=1))
    x = torch.randn(32, 3, 32, 32)
    sim = QuantSim(folded, QuantConfig(w_bits=16, a_bits=16))
    sim.calibrate(x)
    with torch.no_grad():
        ref = folded(x)
        out = sim(x)
    assert sqnr_db(ref.numpy(), out.numpy()) > 40


def test_quantsim_points_and_overrides():
    folded = fold_model(_randomize_bn(MicroMobileNetV2(), seed=2))
    pts = act_points(folded)
    assert "pool" in pts
    layers = weight_layers(folded)
    dws = [n for n, m in layers.items() if is_depthwise(m)]
    assert len(dws) == 8
    sim = QuantSim(folded, QuantConfig(w_bits=4, w_overrides={dws[0]: None}, a_bits=None))
    sim.calibrate(torch.randn(8, 3, 32, 32))
    assert sim.ranges == {}
    original = dict(folded.named_modules())[dws[0]].weight
    kept = dict(sim.model.named_modules())[dws[0]].weight
    assert torch.equal(original, kept)
