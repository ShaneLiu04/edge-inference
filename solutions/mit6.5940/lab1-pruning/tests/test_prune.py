import torch
from prune.channel import (
    apply_channel_sorting,
    channel_prune,
    count_macs,
    get_input_channel_importance,
    get_num_channels_to_keep,
)
from prune.fine import (
    FineGrainedPruner,
    choose_sparsities,
    fine_grained_prune,
    get_model_size,
    kept_fraction,
    pick_sparsity,
)
from prune.train import augment
from prune.vgg import VGG


def test_fine_grained_prune_keeps_the_largest_magnitudes():
    sign = torch.where(torch.arange(25) % 2 == 0, 1.0, -1.0)
    t = (torch.arange(1, 26, dtype=torch.float32) * sign).reshape(5, 5)
    mask = fine_grained_prune(t, 0.6)
    assert int(t.count_nonzero()) == 10
    assert set(t[mask].abs().tolist()) == set(range(16, 26))
    assert torch.equal(mask, t != 0)


def test_fine_grained_prune_edges():
    t = torch.randn(4, 4)
    assert fine_grained_prune(t.clone(), 0.0).all()
    z = t.clone()
    assert not fine_grained_prune(z, 1.0).any() and not z.any()


def test_pruner_reapplies_masks_after_an_update():
    m = VGG()
    pruner = FineGrainedPruner(m, {"backbone.conv1.weight": 0.5})
    w = m.backbone.conv1.weight
    with torch.no_grad():
        w.add_(1.0)
    pruner.apply(m)
    assert abs(1 - float(w.count_nonzero()) / w.numel() - 0.5) < 1e-3


def test_model_size_counts_bits():
    m = VGG()
    assert get_model_size(m) == 32 * sum(p.numel() for p in m.parameters())
    assert get_model_size(m, count_nonzero_only=True) <= get_model_size(m)


def test_pick_sparsity_stops_at_the_first_drop():
    assert pick_sparsity([0.4, 0.5, 0.6], [90, 85, 89], 88) == 0.4
    assert pick_sparsity([0.4, 0.5], [80, 80], 88) == 0.0


def test_choose_sparsities_meets_the_budget():
    sp = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    accs = {"a": [92, 92, 91, 90, 85, 70], "b": [92, 91, 90, 80, 60, 30]}
    numels = {"a": 900, "b": 100}
    choice, tol = choose_sparsities(sp, accs, numels, dense_numel=1000, dense_acc=92, budget=0.3)
    assert kept_fraction(choice, numels, 1000) <= 0.3
    assert choice["a"] >= choice["b"]
    assert tol > 0


def test_num_channels_to_keep_rounds():
    assert get_num_channels_to_keep(64, 0.3) == 45
    assert get_num_channels_to_keep(512, 0.0) == 512


def test_channel_prune_shapes_and_macs():
    torch.manual_seed(0)
    m = VGG().eval()
    p = channel_prune(m, 0.3)
    x = torch.randn(2, 3, 32, 32)
    assert p(x).shape == (2, 10)
    convs = [c for c in p.backbone if isinstance(c, torch.nn.Conv2d)]
    assert [c.out_channels for c in convs] == [45, 90, 179, 179, 358, 358, 358, 512]
    ratio = count_macs(p, x) / count_macs(m, x)
    assert 0.45 < ratio < 0.55


def test_channel_sorting_preserves_the_function():
    torch.manual_seed(0)
    m = VGG()
    for bn in [b for b in m.backbone if isinstance(b, torch.nn.BatchNorm2d)]:
        bn.running_mean.uniform_(-0.5, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.data.uniform_(0.5, 1.5)
    m.eval()
    s = apply_channel_sorting(m)
    x = torch.randn(3, 3, 32, 32)
    assert torch.allclose(m(x), s(x), atol=1e-5)
    nxt = [c for c in s.backbone if isinstance(c, torch.nn.Conv2d)][1]
    imp = get_input_channel_importance(nxt.weight)
    assert torch.all(imp[:-1] >= imp[1:])


def test_augment_is_a_shifted_crop():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(8, 3, 32, 32)
    y = augment(x, g)
    assert y.shape == x.shape
    assert float(y.max()) <= float(x.max())
