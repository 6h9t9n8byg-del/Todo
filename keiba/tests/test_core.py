"""確率の正規化・オッズ変換・評価指標の単体テスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.grouping import group_normalize, group_softmax, log_linear_pool, race_codes
from keiba.market import blend, fit_blend_weight, implied_probabilities, overround
from keiba.metrics import baseline_nll, race_nll, top1_accuracy, topk_hit_rate
from keiba.schema import SchemaError, coerce, validate


def _races(sizes: list[int]) -> np.ndarray:
    return np.concatenate([[f"R{i}"] * n for i, n in enumerate(sizes)])


def test_group_softmax_sums_to_one_per_race() -> None:
    races = _races([3, 5, 2])
    codes, n = race_codes(races)
    p = group_softmax(np.random.default_rng(0).normal(size=len(races)), codes, n)
    sums = pd.Series(p).groupby(races).sum()
    assert np.allclose(sums, 1.0)


def test_group_softmax_is_shift_invariant() -> None:
    races = _races([4, 4])
    codes, n = race_codes(races)
    s = np.random.default_rng(1).normal(size=8)
    assert np.allclose(group_softmax(s, codes, n), group_softmax(s + 10.0, codes, n))


def test_log_linear_pool_with_identical_inputs_is_identity() -> None:
    races = _races([5, 5])
    codes, n = race_codes(races)
    p = group_normalize(np.random.default_rng(2).random(10), codes, n)
    pooled = log_linear_pool([p, p], np.array([0.5, 0.5]), codes, n)
    assert np.allclose(pooled, p)


def test_implied_probabilities_remove_overround() -> None:
    odds = np.array([2.0, 3.0, 6.0, 10.0, 20.0])
    races = _races([5])
    assert overround(odds, races)[0] > 1.0
    for method in ("proportional", "power"):
        p = implied_probabilities(odds, races, method)
        assert p.sum() == pytest.approx(1.0)
        assert np.all(np.diff(p) < 0)  # オッズが高いほど確率は低い


def test_power_method_discounts_longshots_relative_to_proportional() -> None:
    """本命-大穴バイアスの補正方向が正しいこと（人気薄は割り引く）。"""
    odds = np.array([1.5, 4.0, 12.0, 40.0, 100.0])
    races = _races([5])
    prop = implied_probabilities(odds, races, "proportional")
    power = implied_probabilities(odds, races, "power")
    assert power[-1] < prop[-1]   # 人気薄は引き下げ
    assert power[0] > prop[0]     # 本命は引き上げ


def test_implied_probabilities_handle_missing_odds() -> None:
    odds = np.array([2.0, 4.0, np.nan, 8.0])
    p = implied_probabilities(odds, _races([4]), "power")
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0)


def test_blend_endpoints() -> None:
    races = _races([4])
    codes, n = race_codes(races)
    pm = group_normalize(np.array([0.4, 0.3, 0.2, 0.1]), codes, n)
    pk = group_normalize(np.array([0.1, 0.2, 0.3, 0.4]), codes, n)
    assert np.allclose(blend(pm, pk, 1.0, races), pm)
    assert np.allclose(blend(pm, pk, 0.0, races), pk)


def test_fit_blend_weight_prefers_the_informative_source() -> None:
    rng = np.random.default_rng(5)
    sizes = [8] * 400
    races = _races(sizes)
    codes, n = race_codes(races)
    truth = group_softmax(rng.normal(size=len(races)), codes, n)
    winner = np.zeros(len(races))
    for g in range(n):
        idx = np.flatnonzero(codes == g)
        winner[rng.choice(idx, p=truth[idx] / truth[idx].sum())] = 1.0
    noise = group_normalize(rng.random(len(races)), codes, n)
    w, table = fit_blend_weight(truth, noise, winner, races)
    assert w > 0.7
    assert table["nll"].idxmin() == table["nll"].argmin()


def test_metrics_on_a_perfect_and_a_useless_model() -> None:
    races = _races([10] * 50)
    codes, n = race_codes(races)
    y = np.zeros(len(races))
    # 勝ち馬の位置はレースごとにランダム（先頭固定だと同着処理の検証にならない）
    winners = np.arange(n) * 10 + np.random.default_rng(9).integers(0, 10, n)
    y[winners] = 1.0
    perfect = group_normalize(y + 1e-9, codes, n)
    uniform = np.full(len(races), 0.1)
    assert top1_accuracy(perfect, y, races) == pytest.approx(1.0)
    assert top1_accuracy(uniform, y, races) == pytest.approx(0.1)
    assert topk_hit_rate(perfect, y, races, 3) == pytest.approx(1.0)
    assert race_nll(uniform, y, races) == pytest.approx(baseline_nll(races))
    assert race_nll(perfect, y, races) < race_nll(uniform, y, races)


def test_schema_rejects_broken_races() -> None:
    df = pd.DataFrame({
        "race_id": ["R1"] * 3,
        "date": ["2024-01-01"] * 3,
        "venue": ["Tokyo"] * 3,
        "surface": ["turf"] * 3,
        "distance": [1600] * 3,
        "going": ["firm"] * 3,
        "race_class": [5] * 3,
        "horse_id": ["A", "B", "C"],
        "draw": [1, 2, 3],
        "finish_pos": [1, 1, 3],
    })
    with pytest.raises(SchemaError):
        validate(coerce(df), require_results=True)


def test_schema_requires_single_date_per_race() -> None:
    df = pd.DataFrame({
        "race_id": ["R1", "R1"],
        "date": ["2024-01-01", "2024-01-02"],
        "venue": ["Tokyo"] * 2,
        "surface": ["turf"] * 2,
        "distance": [1600] * 2,
        "going": ["firm"] * 2,
        "race_class": [5] * 2,
        "horse_id": ["A", "B"],
        "draw": [1, 2],
    })
    with pytest.raises(SchemaError):
        validate(coerce(df))
