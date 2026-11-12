import numpy as np
import torch
from mlce2e.reference import NAME_MAP, conv2d_nchw, forward_numpy, pytorch_model


def _weights(seed=0):
    r = np.random.default_rng(seed)
    shapes = {"conv2d_weight": (32, 1, 3, 3), "conv2d_bias": (32,), "linear0_weight": (100, 5408),
              "linear0_bias": (100,), "linear1_weight": (10, 100), "linear1_bias": (10,)}
    return {k: (r.standard_normal(s) * 0.05).astype(np.float32) for k, s in shapes.items()}


def test_name_map_covers_every_parameter():
    assert set(NAME_MAP.values()) == set(_weights())


def test_numpy_conv_matches_torch():
    r = np.random.default_rng(1)
    x = r.standard_normal((4, 3, 10, 9)).astype(np.float32)
    w = r.standard_normal((5, 3, 3, 3)).astype(np.float32)
    ref = torch.nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    assert np.allclose(conv2d_nchw(x, w), ref, atol=1e-5)


def test_numpy_forward_matches_the_pytorch_model():
    wm = _weights()
    x = np.random.default_rng(2).random((4, 1, 28, 28), dtype=np.float32)
    with torch.no_grad():
        ref = pytorch_model(wm)(torch.from_numpy(x)).numpy()
    assert np.allclose(forward_numpy(x, wm), ref, atol=1e-6)
