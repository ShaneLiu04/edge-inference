import json

import pytest
from lcpp.parse import bench_rows, final_ppl, kld_stats, quantize_sizes, wikitext_raw

# Format strings copied from tools/perplexity/perplexity.cpp (llama.cpp 2026-11).
KLD_LOG = """\
llama_perplexity: something unrelated: 12.5
====== Perplexity statistics ======
Mean PPL(Q)                   :   6.912345 ±   0.043210
Mean PPL(base)                :   6.850000 ±   0.042800
Cor(ln(PPL(Q)), ln(PPL(base))):  99.71%
Mean ln(PPL(Q)/PPL(base))     :   0.009060 ±   0.000500
Mean PPL(Q)/PPL(base)         :   1.009101 ±   0.000505
Mean PPL(Q)-PPL(base)         :   0.062345 ±   0.003400

====== KL divergence statistics ======
Mean    KLD:   0.021000 ±   0.000210
Maximum KLD:   5.123000
99.9%   KLD:   1.200000
99.0%   KLD:   0.250000
95.0%   KLD:   0.080000
90.0%   KLD:   0.045000
Median  KLD:   0.008000
Minimum KLD:  -0.000010

====== Token probability statistics ======
Mean    Δp: -0.123 ± 0.012 %
Maximum Δp: 45.000%
Median  Δp: -0.001%
RMS Δp    :  3.456 ± 0.050 %
Same top p: 95.432 ± 0.090 %
"""


def test_kld_stats_reads_every_section():
    s = kld_stats(KLD_LOG)
    assert s["mean kld"] == pytest.approx(0.021)
    assert s["mean kld err"] == pytest.approx(0.00021)
    assert s["99.9% kld"] == pytest.approx(1.2)
    assert s["median kld"] == pytest.approx(0.008)
    assert s["mean ppl(q)"] == pytest.approx(6.912345)
    assert s["cor(ln(ppl(q)), ln(ppl(base)))"] == pytest.approx(99.71)
    assert s["same top p"] == pytest.approx(95.432)
    assert s["same top p err"] == pytest.approx(0.09)
    assert s["mean δp"] == pytest.approx(-0.123)
    assert s["rms δp"] == pytest.approx(3.456)
    assert "something unrelated" not in s


def test_final_ppl():
    assert final_ppl("[1]5.1,[2]6.0\nFinal estimate: PPL = 6.8501 +/- 0.04193\n") == (6.8501, 0.04193)
    assert final_ppl("no estimate") is None


def test_bench_rows_names_tests():
    lines = [json.dumps({"n_prompt": 512, "n_gen": 0, "avg_ts": 9000.0}),
             json.dumps({"n_prompt": 0, "n_gen": 128, "avg_ts": 150.0}),
             "main: some log line",
             json.dumps({"n_prompt": 512, "n_gen": 128, "avg_ts": 600.0})]
    rows = bench_rows("\n".join(lines))
    assert [r["test"] for r in rows] == ["pp512", "tg128", "pp512+tg128"]


def test_quantize_sizes():
    log = ("llama_model_quantize_impl: model size  = 14526.27 MiB (16.00 BPW)\n"
           "llama_model_quantize_impl: quant size  =  4165.37 MiB (4.59 BPW)\n")
    assert quantize_sizes(log) == {"src_MiB": 14526.27, "src_bpw": 16.0, "quant_MiB": 4165.37, "quant_bpw": 4.59}


def test_wikitext_raw_restores_lines():
    assert wikitext_raw(["", " = Title = \n", "", " body text \n"]) == " \n = Title = \n \n body text \n"
