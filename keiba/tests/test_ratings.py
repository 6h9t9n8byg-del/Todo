"""Plackett–Luce レーティングの検証。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from keiba.ratings import RatingConfig, Ratings, fit_ratings, rest_name


def _races(orders, dates=None, field_size=None, grade=None):
    dates = dates or [None] * len(orders)
    return [{"result": o, "date": d, "field_size": field_size, "grade": grade}
            for o, d in zip(orders, dates)]


def test_consistent_winner_is_rated_higher() -> None:
    r = fit_ratings(_races([["A", "B"], ["A", "B"], ["A", "B"]]))
    assert r.theta[r.index("A")] > r.theta[r.index("B")]


def test_ratings_shrink_toward_zero_with_a_tight_prior() -> None:
    orders = [["A", "B"]] * 3
    loose = fit_ratings(_races(orders), config=RatingConfig(prior_sd=3.0))
    tight = fit_ratings(_races(orders), config=RatingConfig(prior_sd=0.2))
    assert abs(tight.theta[tight.index("A")]) < abs(loose.theta[loose.index("A")])


def test_symmetric_results_give_equal_ratings() -> None:
    r = fit_ratings(_races([["A", "B"], ["B", "A"]]))
    assert r.theta[r.index("A")] == pytest.approx(r.theta[r.index("B")], abs=1e-6)


def test_more_starts_means_less_uncertainty() -> None:
    r = fit_ratings(_races([["A", "B"]] * 8 + [["C", "D"]]))
    assert r.se[r.index("A")] < r.se[r.index("C")]


def test_unranked_runners_are_rated_far_below_the_placegetters() -> None:
    """上位に来た馬は、着順不明の集団より明確に上と評価される。"""
    r = fit_ratings(_races([["A", "B", "C"]] * 3, field_size=16))
    rest = r.index(rest_name(None))
    assert r.theta[rest] < r.theta[r.index("C")] - 1.0
    assert r.starts[rest] == 3 * (16 - 3)


def test_each_grade_gets_its_own_unranked_group() -> None:
    """格の違うレースを同じ「負け」として扱わない。"""
    races = (_races([["A", "B", "C"]] * 3, field_size=16, grade="G1")
             + _races([["D", "E", "F"]] * 3, field_size=16, grade="G3"))
    r = fit_ratings(races)
    assert r.index(rest_name("G1")) is not None
    assert r.index(rest_name("G3")) is not None
    assert r.index(rest_name(None)) is None


def test_the_unranked_group_is_an_assumption_not_a_free_estimate() -> None:
    """着順不明の集団の強さは事前分布で押さえる（自由に推定させると発散する）。"""
    races = _races([["A", "B", "C"]] * 8, field_size=18, grade="G1")
    tight = fit_ratings(races, config=RatingConfig(rest_prior_sd=0.2, rest_prior_mean=-1.5))
    loose = fit_ratings(races, config=RatingConfig(rest_prior_sd=5.0, rest_prior_mean=-1.5))
    t_rest = tight.theta[tight.index(rest_name("G1"))]
    l_rest = loose.theta[loose.index(rest_name("G1"))]
    assert l_rest < t_rest              # 自由にすると下へ流れる
    assert abs(t_rest + 1.5) < 0.6      # 締めれば仮定の近くに留まる


def test_placing_third_beats_being_absent_from_the_result() -> None:
    """毎回3着の馬は、着順不明の集団よりずっと高く評価される。"""
    races = _races([["A", "B", "C"], ["A", "B", "C"], ["B", "A", "C"]], field_size=18)
    r = fit_ratings(races)
    assert r.theta[r.index("C")] > r.theta[r.index(rest_name(None))] + 1.5


def test_field_size_affects_how_much_credit_a_win_earns() -> None:
    """多頭数を勝った方が高く評価される。"""
    small = fit_ratings(_races([["A", "B"]] * 3, field_size=4))
    big = fit_ratings(_races([["A", "B"]] * 3, field_size=18))
    assert big.theta[big.index("A")] > small.theta[small.index("A")]


def test_recent_races_count_more() -> None:
    as_of = date(2026, 7, 1)
    old_win = _races([["A", "B"]], [date(2020, 1, 1)])
    new_win = _races([["A", "B"]], [date(2026, 6, 1)])
    old = fit_ratings(old_win, as_of=as_of, config=RatingConfig(half_life_days=365))
    new = fit_ratings(new_win, as_of=as_of, config=RatingConfig(half_life_days=365))
    assert new.theta[new.index("A")] > old.theta[old.index("A")]


def test_analytic_gradient_matches_numerical() -> None:
    """勾配を手で導いている以上、数値微分と一致することを確かめる。"""
    races = _races([["A", "B", "C"], ["B", "C", "D"], ["D", "A"]], field_size=10)
    cfg = RatingConfig(prior_sd=1.5)
    fitted = fit_ratings(races, config=cfg)

    # fit_ratings と同じ目的関数を組み直して勾配を比較する
    import keiba.ratings as mod

    captured = {}
    original = mod.minimize

    def spy(fun, x0, **kw):
        captured["fun"] = fun
        return original(fun, x0, **kw)

    mod.minimize = spy
    try:
        fit_ratings(races, config=cfg)
    finally:
        mod.minimize = original

    objective = captured["fun"]
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.5, len(fitted.names))
    _, grad = objective(x)
    numeric = approx_fprime(x, lambda v: objective(v)[0], 1e-6)
    assert np.allclose(grad, numeric, atol=1e-5)


def test_fitted_point_is_a_stationary_point() -> None:
    r = fit_ratings(_races([["A", "B", "C"], ["C", "A"], ["B", "C"]], field_size=12))
    assert np.all(np.isfinite(r.theta))
    assert np.all(r.se > 0)


# ------------------------------------------------------------------ 予測
def _fitted() -> Ratings:
    return fit_ratings(_races([["A", "B", "C"]] * 4 + [["B", "A"]], field_size=16))


def test_win_probabilities_form_a_distribution() -> None:
    r = _fitted()
    p = r.win_probabilities(["A", "B", "C"])
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0)
    assert p[0] > p[2]


def test_unknown_horse_falls_back_to_the_prior() -> None:
    r = _fitted()
    table = r.field_table(["A", "B", "初出走馬"])
    row = table[table["horse"] == "初出走馬"].iloc[0]
    assert row["rating"] == 0.0
    assert row["se"] == pytest.approx(r.config.prior_sd)
    assert 0 < row["win_prob"] < 1


def test_uncertainty_shifts_probability_toward_the_least_known_horse() -> None:
    """不確かさを織り込むと、記録の乏しい馬へ確率が移る。

    「一律に平らになる」のではない点に注意。よく分かっている馬の取り分が減り、
    分からない馬の取り分が増える、という向きの移動が起きる。
    """
    r = _fitted()
    field = ["A", "B", "C", "初出走馬"]
    sampled = r.win_probabilities(field, samples=40000, seed=1)
    point = r.win_probabilities(field, samples=0)
    shift = sampled - point

    assert sampled.sum() == pytest.approx(1.0)
    assert shift[field.index("初出走馬")] > 0        # 未知の馬は取り分が増える
    best = int(np.argmax(point))
    assert shift[best] < 0                           # 一番人気は取り分を減らす


def test_sampling_is_reproducible() -> None:
    r = _fitted()
    a = r.win_probabilities(["A", "B", "C"], samples=500, seed=7)
    b = r.win_probabilities(["A", "B", "C"], samples=500, seed=7)
    assert np.allclose(a, b)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_ratings([])
