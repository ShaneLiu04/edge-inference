import pytest
from rtcmp.measure import agreement, combine, cpu_quota_cores, time_call


def test_quota_half_core_and_uncapped():
    assert cpu_quota_cores("50000 100000") == pytest.approx(0.5)
    assert cpu_quota_cores("max 100000") is None


def test_quota_rejects_garbage():
    with pytest.raises(ValueError):
        cpu_quota_cores("50000 0")


def test_time_call_reports_both_clocks():
    def burn():
        x = 0
        for _ in range(200_000):
            x += 1
        return x

    out = time_call(burn, repeats=2, warmup=1)
    assert out["repeats"] == 2
    assert out["wall_us"] > 0
    assert out["cpu_us"] > 0


def test_agreement_on_a_tie_and_a_flip():
    same = agreement([[0.0, 1.0], [2.0, 0.0]], [[0.1, 0.9], [1.5, 0.2]])
    assert same["argmax_agree"] == 1.0
    assert same["logit_max_abs"] == pytest.approx(0.5)
    flipped = agreement([[0.0, 1.0]], [[1.0, 0.0]])
    assert flipped["argmax_agree"] == 0.0


def test_combine_takes_median_and_keeps_the_spread():
    trials = [
        {"runtime": "ort", "status": "ok", "trial": 0, "acc": 0.88, "cpu_us_eval": 300.0},
        {"runtime": "ort", "status": "ok", "trial": 1, "acc": 0.88, "cpu_us_eval": 200.0},
        {"runtime": "ort", "status": "error", "trial": 2, "error": "killed"},
        {"runtime": "ort", "status": "ok", "trial": 3, "acc": 0.88, "cpu_us_eval": 210.0},
    ]
    row = combine(trials, metrics=("cpu_us_eval",))
    assert row["trials"] == 3
    assert row["cpu_us_eval"] == 210.0
    assert (row["cpu_us_eval_min"], row["cpu_us_eval_max"]) == (200.0, 300.0)
    assert "trial" not in row
    assert combine([trials[2]])["status"] == "error"
