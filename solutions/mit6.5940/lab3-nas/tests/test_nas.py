"""CPU checks of the design space, the closed-form efficiency model, the predictor and the searchers.

The supernet itself is not needed here: the efficiency model is checked against a plain torch net
built from the same config and counted with forward hooks. The server run checks it against the
course library on 1,004 subnets.
"""

import random

import numpy as np
import pytest
import torch
from nas import efficiency as eff
from nas import predictor, search, space
from torch import nn


def _cfgs(n, seed=0, widths=(0, 1, 2)):
    rng = random.Random(seed)
    return [space.sample(rng, widths) for _ in range(n)]


# ---------------------------------------------------------------- space


def test_sample_valid_and_encoding_shape():
    for c in _cfgs(200):
        space.validate(c)
        f = space.encode(c)
        assert f.shape == (space.N_FEATURES,) == (128,)
        n_active = sum(space.active(c))
        assert f.sum() == 2 + 2 * n_active


def test_encoding_ignores_skipped_blocks():
    c = space.uniform(0, 3, 3, 0, 96)
    d = dict(c, ks=list(c["ks"]), e=list(c["e"]))
    skipped = [i for i, on in enumerate(space.active(c)) if not on]
    assert skipped
    for i in skipped:
        d["ks"][i], d["e"][i] = 7, 6
    assert space.key(c) == space.key(d)
    assert eff.efficiency(c) == eff.efficiency(d)


def test_active_counts():
    assert sum(space.active(space.smallest())) == sum(space.BASE_DEPTH)
    assert sum(space.active(space.largest())) == space.N_BLOCKS


def test_mutate_crossover_stay_in_space():
    rng = random.Random(1)
    a, b = _cfgs(2, 3, widths=(0, 1))
    for _ in range(200):
        m = space.mutate(a, rng, 0.3, 0.5)
        x = space.crossover(a, b, rng)
        space.validate(m)
        space.validate(x)
        assert m["wid"] in (0, 1)
        assert all(v in (p, q) for v, p, q in zip(x["ks"], a["ks"], b["ks"]))
    assert space.mutate(a, rng, 0.0, 0.0) == a


# ------------------------------------------------- efficiency vs a torch net


def _cbr(cin, cout, k, s=1, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.ReLU6())


class _SE(nn.Module):
    def __init__(self, c, mid):
        super().__init__()
        self.reduce = nn.Conv2d(c, mid, 1)
        self.expand = nn.Conv2d(mid, c, 1)

    def forward(self, x):
        y = x.mean((2, 3), keepdim=True)
        return x * torch.sigmoid(self.expand(torch.relu(self.reduce(y))))


class _Block(nn.Module):
    def __init__(self, cin, cout, k, s, e, se):
        super().__init__()
        mid = eff.make_divisible(round(cin * e), 8)
        layers = [_cbr(cin, mid, 1), _cbr(mid, mid, k, s, groups=mid)]
        if se:
            layers.append(_SE(mid, cin // 4))
        layers.append(nn.Conv2d(mid, cout, 1, bias=False))
        self.body = nn.Sequential(*layers)
        self.shortcut = False

    def forward(self, x):
        y = self.body(x)
        return x + y if self.shortcut else y


def _reference(cfg):
    w = space.WIDTHS[cfg["wid"]]
    c0 = eff.make_divisible(32 * w, 8)
    c1 = eff.make_divisible(16 * w, 8)
    mods = [_cbr(3, c0, 3, 2), nn.Conv2d(c0, c1, 1, bias=False)]
    cin, i = c1, 0
    for s, n in enumerate(space.STAGE_BLOCKS):
        cout = eff.make_divisible(eff.STAGE_WIDTHS[s] * w, 8)
        for j in range(n):
            idx, i = i, i + 1
            if j >= space.stage_depth(cfg, s):
                continue
            b = _Block(cin, cout, cfg["ks"][idx], eff.STAGE_STRIDES[s] if j == 0 else 1, cfg["e"][idx],
                       eff.STAGE_SE[s][j])
            b.shortcut = j > 0
            mods.append(b)
            cin = cout
    mods += [_cbr(cin, eff.MIX_WIDTH, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(eff.MIX_WIDTH, 2)]
    return nn.Sequential(*mods)


def _count(net, r):
    total = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            total[0] += out.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total[0] += m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        net(torch.zeros(1, 3, r, r))
    for h in hs:
        h.remove()
    return total[0], sum(p.numel() for p in net.parameters())


@pytest.mark.parametrize("cfg", _cfgs(12, seed=7) + [space.smallest(), space.largest()])
def test_macs_params_match_torch_reference(cfg):
    m, p = _count(_reference(cfg), cfg["image_size"])
    assert eff.conv_macs(cfg) == m
    assert eff.params(cfg) == p
    e = eff.efficiency(cfg)
    assert e["mmacs_conv"] * 1e6 == pytest.approx(m)
    assert 1.05 < e["mmacs"] / e["mmacs_conv"] < 1.2  # the lab counter's BatchNorm charge


def test_peak_rules():
    for c in _cfgs(100, seed=4):
        lab, _ = eff.peak(c, "lab")
        res, _ = eff.peak(c, "residual")
        assert res >= lab > 0
        assert max(eff.unit_peak(u) for u in eff.units(c)) == lab


def test_smallest_config_hand_numbers():
    c = space.smallest()
    us = {u.name: u for u in eff.units(c)}
    # stem: 3x96x96 in, 16x48x48 out
    assert eff.unit_peak(us["stem"]) == 3 * 96 * 96 + 16 * 48 * 48
    assert eff.unit_peak(us["block0"]) == 16 * 48 * 48 + 8 * 48 * 48
    e = eff.efficiency(c)
    assert e["kb"] > 64  # 64 KB is out of reach without patch-based inference
    assert e["mmacs"] < 15


def test_efficiency_monotone_in_resolution():
    rng = random.Random(5)
    for _ in range(20):
        c = space.sample(rng)
        vals = [eff.efficiency(dict(c, image_size=r)) for r in space.RESOLUTIONS]
        assert all(a["mmacs"] < b["mmacs"] and a["kb"] < b["kb"] for a, b in zip(vals, vals[1:]))


# ------------------------------------------------------------- predictor


def test_rank_metrics():
    a = np.arange(50, dtype=float)
    assert predictor.spearman(a, a) == pytest.approx(1)
    assert predictor.kendall_tau(a, -a) == pytest.approx(-1)
    assert predictor.pearson(a, 2 * a + 1) == pytest.approx(1)
    assert predictor.top_overlap(a, a, 0.1) == pytest.approx(1)
    t = np.array([1, 1, 2, 3], dtype=float)
    assert predictor.spearman(t, t) == pytest.approx(1)


def test_predictor_learns_synthetic_target():
    rng = random.Random(0)
    cfgs = [space.sample(rng) for _ in range(3000)]
    x = torch.tensor(np.stack([space.encode(c) for c in cfgs]))
    wts = torch.linspace(-1, 1, x.shape[1])
    y = 90 + x @ wts * 0.3
    tr, va = predictor.split(len(cfgs), 2500, seed=0)
    m = predictor.AccuracyMLP(base_acc=float(y[tr].mean()))
    before = predictor.evaluate_l1(m, x[va], y[va])
    hist = predictor.fit(m, x[tr], y[tr], x[va], y[va], epochs=15, seed=0)
    after = predictor.evaluate_l1(m, x[va], y[va])
    assert after < 0.5 * before
    assert hist[-1]["val_l1"] == pytest.approx(after, rel=1e-4)


# --------------------------------------------------------------- search


def _toy_score(cfgs):
    return [sum(c["e"]) + sum(c["ks"]) / 7 + c["image_size"] / 16 for c in cfgs]


def test_constraint_label():
    assert search.Constraint(250, 60).label() == "250KB/60M"
    assert search.Constraint(mmacs=50).label() == "50M"
    assert search.Constraint().label() == "none"


@pytest.mark.parametrize("con", [search.Constraint(mmacs=50), search.Constraint(kb=256),
                                 search.Constraint(250, 60)])
def test_searchers_respect_constraint_and_count_queries(con):
    r = search.random_search(50, con, eff.efficiency, _toy_score, random.Random(0))
    assert con.ok(eff.efficiency(r.best)) and r.queries == 50
    p = search.EvoParams(population=20, generations=5)
    ev = search.evolution(con, eff.efficiency, _toy_score, random.Random(0), p)
    assert con.ok(eff.efficiency(ev.best))
    n_parents = round(20 * p.parent_ratio)
    assert ev.queries == 20 + 5 * (20 - n_parents)
    best = [b for _, b in ev.history]
    assert best == sorted(best)


def test_evolution_beats_random_at_equal_budget_on_toy():
    con = search.Constraint(mmacs=60)
    p = search.EvoParams(population=20, generations=10)
    wins = 0
    for seed in range(5):
        ev = search.evolution(con, eff.efficiency, _toy_score, random.Random(seed), p)
        rs = search.random_search(ev.queries, con, eff.efficiency, _toy_score, random.Random(seed))
        wins += ev.best_score >= rs.best_score
    assert wins >= 4


def test_infeasible_raises():
    with pytest.raises(search.Infeasible):
        search.random_search(1, search.Constraint(kb=10), eff.efficiency, _toy_score, random.Random(0),
                             max_tries=200)
