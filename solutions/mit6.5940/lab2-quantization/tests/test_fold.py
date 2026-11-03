import numpy as np
import pytest
from edgequant.fold import (
    bn_inference,
    conv2d_valid,
    fold_bn,
    per_channel_rel_rmse,
    quantize_symmetric,
    rel_rmse,
)


def test_fold_matches_conv_then_bn():
    rng = np.random.default_rng(1)
    weight = rng.normal(size=(4, 3, 3, 3))
    bias = rng.normal(size=(4,))
    mean = rng.normal(size=(4,))
    var = rng.uniform(0.3, 2.0, size=(4,))
    gamma = rng.normal(size=(4,))
    beta = rng.normal(size=(4,))
    x = rng.normal(size=(2, 3, 8, 8))
    folded_w, folded_b = fold_bn(weight, bias, mean, var, gamma, beta)
    left = conv2d_valid(x, folded_w, folded_b)
    right = bn_inference(conv2d_valid(x, weight, bias), mean, var, gamma, beta)
    assert rel_rmse(left, right) < 1e-9


def test_per_tensor_erases_the_small_channel():
    weight = np.zeros((2, 1, 1, 1), dtype=np.float64)
    weight[0, 0, 0, 0] = 0.4
    weight[1, 0, 0, 0] = 127.0
    bias = np.zeros(2)
    x = np.ones((1, 1, 1, 1), dtype=np.float64)

    _, scale_t, per_tensor = quantize_symmetric(weight, channel_axis=None)
    _, scale_c, per_channel = quantize_symmetric(weight, channel_axis=0)
    assert scale_t.shape == (1, 1, 1, 1)
    assert float(per_tensor[0, 0, 0, 0]) == pytest.approx(0.0)
    assert float(per_channel[0, 0, 0, 0]) == pytest.approx(0.4)

    reference = conv2d_valid(x, weight, bias)
    tensor_out = conv2d_valid(x, per_tensor, bias)
    channel_out = conv2d_valid(x, per_channel, bias)
    assert abs(float(tensor_out[0, 0, 0, 0] - reference[0, 0, 0, 0])) == pytest.approx(0.4)
    assert abs(float(channel_out[0, 0, 0, 0] - reference[0, 0, 0, 0])) < 1e-9
    assert float(scale_c[0, 0, 0, 0]) == pytest.approx(0.4 / 127)


def test_per_channel_quant_commutes_with_positive_channel_scale():
    rng = np.random.default_rng(2)
    weight = rng.normal(size=(6, 3, 3, 3))
    inv = rng.uniform(0.01, 30.0, size=6)
    _, _, q_then_scale = quantize_symmetric(weight, channel_axis=0)
    q_then_scale = q_then_scale * inv[:, None, None, None]
    _, _, scale_then_q = quantize_symmetric(weight * inv[:, None, None, None], channel_axis=0)
    assert np.max(np.abs(q_then_scale - scale_then_q)) < 1e-12


def test_per_channel_rel_rmse_shape():
    ref = np.ones((2, 3, 4, 4))
    cand = ref.copy()
    cand[:, 1] = 0.0
    err = per_channel_rel_rmse(ref, cand, axis=1)
    assert err.shape == (3,)
    assert err[1] == pytest.approx(1.0)
    assert err[0] == pytest.approx(0.0)


def test_points_follow_execution_order():
    torch = pytest.importorskip("torch")
    from edgequant.model import MicroMobileNetV2
    from edgequant.qsim import act_points, execution_order, fold_model

    model = fold_model(MicroMobileNetV2())
    names = list(act_points(model))
    order = execution_order(model, torch.zeros(1, 3, 32, 32), names)
    assert sorted(order) == sorted(names)
    for block in (n for n in names if f"{n}.body.0" in names):
        assert order.index(block) > order.index(f"{block}.body.0")
    assert order[-1] == "pool"
