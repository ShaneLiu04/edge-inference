import torch
from awqlab.quant import pseudo_quantize, quantize_keep_salient, search_clip, search_scale


def _wx(seed=0, out=64, inp=256, tokens=512):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inp, generator=g)
    x = torch.randn(tokens, inp, generator=g)
    x[:, :4] *= 30  # a few outlier channels, as in LLM activations
    return w, x


def test_pseudo_quantize_uses_at_most_2_pow_n_levels_per_group_and_bounds_error():
    w, _ = _wx()
    q = pseudo_quantize(w, 3, 128)
    for grp_w, grp_q in zip(w.reshape(-1, 128), q.reshape(-1, 128)):
        assert grp_q.unique().numel() <= 8
        step = (grp_w.max() - grp_w.min()) / 7
        assert (grp_w - grp_q).abs().max() <= step / 2 + 1e-5


def test_keep_salient_restores_exactly_the_top_columns():
    w, x = _wx()
    imp = x.abs().mean(0)
    q = quantize_keep_salient(w, imp, 4 / 256)
    assert torch.equal(q[:, :4], w[:, :4])
    assert not torch.equal(q[:, 4:], w[:, 4:])


def test_scale_search_alpha_zero_is_rtn_and_best_is_no_worse():
    w, x = _wx()
    s, alpha, losses = search_scale([w], x, 3, 128, grid=10)
    rtn = (x @ pseudo_quantize(w, 3, 128).t() - x @ w.t()).pow(2).mean().item()
    assert abs(losses[0] - rtn) / rtn < 1e-4
    assert min(losses) <= rtn and alpha > 0
    assert torch.allclose((x / s) @ (w * s).t(), x @ w.t(), rtol=1e-3, atol=1e-2)


def test_clip_search_never_widens_and_does_not_increase_error():
    w, x = _wx()
    c = search_clip(w, x, 3, 128, n_tokens=256, oc_batch=16)
    amax = w.reshape(64, 2, 128).abs().amax(-1)
    assert (c <= amax + 1e-6).all()
    e0 = (x @ pseudo_quantize(w, 3, 128).t() - x @ w.t()).pow(2).mean()
    e1 = (x @ pseudo_quantize(w, 3, 128, clip=c).t() - x @ w.t()).pow(2).mean()
    assert e1 <= e0 * 1.0001
