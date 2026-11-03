"""Compare per-channel and per-tensor int8 after a real BN fold.

Writes artifacts/raw/fold_toy.json. One random conv layer with a wide BN gamma
spread; the trained-network version of the same question is in run_experiments.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "code"))

from edgequant.fold import (  # noqa: E402
    bn_inference,
    conv2d_valid,
    fold_bn,
    per_channel_rel_rmse,
    quantize_symmetric,
    rel_rmse,
)


def main() -> None:
    rng = np.random.default_rng(0)
    n, ic, oc, spatial, kh = 2, 16, 32, 12, 3
    weight = rng.normal(0.0, 0.05, size=(oc, ic, kh, kh))
    bias = rng.normal(0.0, 0.01, size=(oc,))
    # Channel scales span about three orders of magnitude after the fold.
    gamma = np.exp(rng.uniform(np.log(0.05), np.log(20.0), size=(oc,)))
    var = rng.uniform(0.25, 1.5, size=(oc,))
    mean = rng.normal(0.0, 0.5, size=(oc,))
    beta = rng.normal(0.0, 0.05, size=(oc,))
    x = rng.normal(0.0, 1.0, size=(n, ic, spatial, spatial))

    folded_w, folded_b = fold_bn(weight, bias, mean, var, gamma, beta)
    reference = conv2d_valid(x, folded_w, folded_b)
    unfolded = bn_inference(conv2d_valid(x, weight, bias), mean, var, gamma, beta)
    if rel_rmse(reference, unfolded) > 1e-9:
        raise SystemExit("BN fold is not algebraically identical to conv-then-BN")

    _, _, per_channel = quantize_symmetric(folded_w, channel_axis=0)
    _, _, per_tensor = quantize_symmetric(folded_w, channel_axis=None)
    _, _, raw_channel = quantize_symmetric(weight, channel_axis=0)

    out_channel = conv2d_valid(x, per_channel, folded_b)
    out_tensor = conv2d_valid(x, per_tensor, folded_b)
    # Per-channel symmetric quant commutes with a positive per-channel BN scale,
    # so quant-then-BN matches fold-then-quant. The fold still matters once an
    # activation quant sits between the conv and the BN: that quant cannot see
    # the per-channel BN scale.
    out_then_bn = bn_inference(conv2d_valid(x, raw_channel, bias), mean, var, gamma, beta)
    conv_fp = conv2d_valid(x, weight, bias)
    _, _, conv_q = quantize_symmetric(conv_fp, channel_axis=None)
    out_mid_quant = bn_inference(conv_q, mean, var, gamma, beta)

    channel_amax = np.max(np.abs(folded_w), axis=(1, 2, 3))
    weight_channel = per_channel_rel_rmse(folded_w, per_channel, axis=0)
    weight_tensor = per_channel_rel_rmse(folded_w, per_tensor, axis=0)
    out_channel_err = per_channel_rel_rmse(reference, out_channel, axis=1)
    out_tensor_err = per_channel_rel_rmse(reference, out_tensor, axis=1)
    out_mid_err = per_channel_rel_rmse(reference, out_mid_quant, axis=1)

    def _stats(err: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(err)),
            "worst": float(err.max()),
            "worst_channel": int(err.argmax()),
            "n_above_10pct": int(np.sum(err > 0.10)),
        }

    summary = {
        "seed": 0,
        "shape": {"N": n, "IC": ic, "OC": oc, "H": spatial, "W": spatial, "K": kh},
        "folded_channel_amax_ratio": float(channel_amax.max() / channel_amax.min()),
        "output_rel_rmse": {
            "per_channel_folded": rel_rmse(reference, out_channel),
            "per_tensor_folded": rel_rmse(reference, out_tensor),
            "per_channel_then_bn": rel_rmse(reference, out_then_bn),
            "act_quant_between_conv_and_bn": rel_rmse(reference, out_mid_quant),
        },
        "weight_rel_rmse_by_channel": {
            "per_channel": _stats(weight_channel),
            "per_tensor": _stats(weight_tensor),
        },
        "output_rel_rmse_by_channel": {
            "per_channel": _stats(out_channel_err),
            "per_tensor": _stats(out_tensor_err),
            "act_quant_between_conv_and_bn": _stats(out_mid_err),
        },
    }
    out = LAB / "artifacts" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fold_toy.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
