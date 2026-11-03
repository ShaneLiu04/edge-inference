import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edge_calc as ec  # noqa: E402


def test_bus_bandwidth_lpddr4x_example():
    # 4266 MT/s x 64 bit = 34.128 GB/s. An example bus, not a named SoC.
    assert ec.bus_bandwidth(4266, 64) == pytest.approx(34.128e9)


def test_sram_upper_bound_64mhz():
    assert ec.sram_bandwidth_upper(64e6, 4) == pytest.approx(256e6)


def test_gemv_intensity_limits():
    # Large fp32 GEMV approaches 0.5 FLOP/byte; fp16 approaches 1; int8 approaches 2.
    assert ec.gemv_intensity(4096, 4096, "fp32") == pytest.approx(0.5, rel=1e-3)
    assert ec.gemv_intensity(4096, 4096, "fp16") == pytest.approx(1.0, rel=1e-3)
    assert ec.gemv_intensity(4096, 4096, "int8") == pytest.approx(2.0, rel=1e-3)


def test_requant_and_bias_share_the_product_of_scales():
    assert ec.requant_mul(0.02, 0.05, 0.01) == pytest.approx(0.1)
    assert ec.bias_int(0.02 * 0.05 * 40, 0.02, 0.05) == pytest.approx(40.0)
    assert ec.affine(3, 0.25, 1) == pytest.approx(0.5)


def test_bn_fold_scale():
    assert ec.bn_fold_scale(4.0, 1.0, eps=0.0) == pytest.approx(4.0)


def test_depthwise_and_im2col():
    assert ec.depthwise_intensity(3, 3, 4.0) == pytest.approx(4.5)
    # 3x3, 8x8 output, 16 channels, fp32: 9x the spatial input patch dimension
    assert ec.im2col_workspace(8, 8, 3, 3, 16, 4) == 8 * 8 * 9 * 16 * 4


def test_qblock_layout():
    assert ec.qblock_bytes_per_param("q8_0") == pytest.approx(34 / 32)
    assert ec.qblock_bytes_per_param("q4_0") == pytest.approx(18 / 32)
    # 500M params, Q4_0, 34 GB/s: one weight read is about 8.3 ms
    nbytes = 5e8 * ec.qblock_bytes_per_param("q4_0")
    assert ec.time_lower_bound(nbytes, 34.128e9) * 1e3 == pytest.approx(8.24, abs=0.02)


def test_double_buffer_does_not_invent_bandwidth():
    # Equal sides: almost 2x for many tiles.
    single = ec.single_buffer_time(8, 2.0, 2.0)
    double = ec.pipeline_time(8, 2.0, 2.0)
    assert single == pytest.approx(32.0)
    assert double == pytest.approx(18.0)
    # DMA already longer: double buffer cannot beat n * dma + compute.
    assert ec.pipeline_time(4, 5.0, 1.0) == pytest.approx(21.0)
    assert ec.pipeline_time(4, 5.0, 1.0) < ec.single_buffer_time(4, 5.0, 1.0)


def test_int32_accumulator_holds_a_small_conv_and_not_a_wide_one():
    qmax = 127
    assert qmax * qmax * 3 * 3 * 256 == 37_161_216
    assert qmax * qmax * 3 * 3 * 256 < 2**31
    assert qmax * qmax * 7 * 7 * 4096 == 3_237_154_816
    assert qmax * qmax * 7 * 7 * 4096 > 2**31 - 1


def test_first_fit_can_fragment():
    buffers = [
        ec.Buffer(0, 0, 3),
        ec.Buffer(0, 1, 3),
        ec.Buffer(1, 1, 2),
        ec.Buffer(1, 1, 2),
    ]
    assert ec.peak_live(buffers) == 7
    assert ec.first_fit_arena(buffers) == 8
