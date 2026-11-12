import torch
import torch.nn.functional as F
from prune.vgg import VGG
from vggquant.kmeans import KMeansQuantizer, kmeans_1d, kmeans_model_size_bits
from vggquant.linear import (
    INPUT_SCALE,
    INPUT_ZP,
    QuantizedVGG,
    calibrate,
    fuse_vgg,
    get_scale_and_zero_point,
    linear_quantize,
    per_channel_symmetric,
    quantize_input,
    quantized_linear,
)


def _vgg(seed=0):
    torch.manual_seed(seed)
    m = VGG().eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.1, 0.1)
    return m


def test_kmeans_1d_recovers_well_separated_clusters():
    x = torch.cat([torch.full((50,), -1.0), torch.full((30,), 0.5), torch.full((20,), 3.0)])
    x = x + 0.01 * torch.randn(100, generator=torch.Generator().manual_seed(0))
    cb = kmeans_1d(x, 3)
    assert torch.allclose(cb.centroids, torch.tensor([-1.0, 0.5, 3.0]), atol=0.01)
    assert cb.labels[:50].eq(0).all() and cb.labels[80:].eq(2).all()


def test_kmeans_quantizer_leaves_at_most_2_pow_bits_values_and_qat_step_keeps_them():
    m = _vgg()
    q = KMeansQuantizer(m, bits=2)
    w = dict(m.named_parameters())["backbone.conv3.weight"]
    assert w.unique().numel() <= 4
    with torch.no_grad():
        w.add_(0.01)
    q.apply(m, update=True)
    assert w.unique().numel() <= 4
    assert kmeans_model_size_bits(m, 2) < sum(p.numel() for p in m.parameters()) * 32 / 10


def test_scale_zero_point_maps_range_endpoints():
    s, z = get_scale_and_zero_point(-1.0, 3.0, 8)
    assert linear_quantize(torch.tensor([-1.0, 3.0]), 8, s, z).tolist() == [-128, 127]
    s, z = get_scale_and_zero_point(0.0, 6.0, 8)
    assert z == -128


def test_input_quantization_matches_the_course_convention():
    x = torch.tensor([0.0, 1.0, 128 / 255])
    assert quantize_input(x).tolist() == [-128, 127, 0]
    assert torch.allclose((quantize_input(x).float() - INPUT_ZP) * INPUT_SCALE, x)


def test_quantized_linear_tracks_float():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(8, 64, generator=g)
    w = torch.randn(16, 64, generator=g) * 0.1
    b = torch.randn(16, generator=g) * 0.1
    ref = F.linear(x, w, b)
    sin, zin = get_scale_and_zero_point(0, 1, 8)
    sout, zout = get_scale_and_zero_point(ref.min().item(), ref.max().item(), 8)
    qw, ws = per_channel_symmetric(w, 8)
    qb = torch.round(b / (ws * sin)).to(torch.int32)
    out = quantized_linear(linear_quantize(x, 8, sin, zin), qw, qb, zin, sin, ws, sout, zout)
    assert ((out.float() - zout) * sout - ref).abs().max() < 4 * sout


def test_fused_vgg_matches_and_integer_vgg_agrees_on_argmax():
    m = _vgg()
    x = torch.rand(16, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    fused = fuse_vgg(m)
    with torch.no_grad():
        ref = m(x)
        assert torch.allclose(fused(x), ref, atol=1e-4)
        qm = QuantizedVGG(fused, calibrate(fused, x))
        out = qm(quantize_input(x))
    assert out.dtype == torch.int8
    deq = (out.float() - qm.logit_zp) * qm.logit_scale
    assert (deq - ref).abs().max() < 0.1 * ref.abs().max() + 8 * qm.logit_scale
    assert qm.size_bits() < sum(p.numel() for p in m.parameters()) * 32 / 3.5
