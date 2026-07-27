"""特徴量が未来の情報を参照していないことを検証する。

競馬モデルで最も起きやすく、最も致命的なバグがリークなので、ここは重点的に守る。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.data import GeneratorConfig, generate_dataset
from keiba.features import FeatureBuilder
from keiba.schema import coerce


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return coerce(generate_dataset(GeneratorConfig(seed=3, n_race_days=40, n_horses=400)))


def test_future_results_do_not_change_past_features(raw: pd.DataFrame) -> None:
    """カットオフ以降の結果を壊しても、それ以前の行の特徴量は1つも変わらない。"""
    fb = FeatureBuilder()
    base = fb.build(raw)

    cutoff = raw["date"].quantile(0.6)
    corrupted = raw.copy()
    future = corrupted["date"] > cutoff
    rng = np.random.default_rng(0)
    corrupted.loc[future, "finish_pos"] = rng.permutation(corrupted.loc[future, "finish_pos"].values)
    corrupted.loc[future, "finish_time"] += rng.normal(0, 10, int(future.sum()))
    corrupted.loc[future, "corner_pos"] = rng.permutation(corrupted.loc[future, "corner_pos"].values)
    corrupted.loc[future, "last_3f"] += rng.normal(0, 5, int(future.sum()))
    after = fb.build(corrupted)

    cols = fb.numeric_features
    a = base.loc[base["date"] <= cutoff, ["race_id", "horse_id"] + cols]
    b = after.loc[after["date"] <= cutoff, ["race_id", "horse_id"] + cols]
    assert len(a) == len(b) and len(a) > 0
    merged = a.merge(b, on=["race_id", "horse_id"], suffixes=("_a", "_b"))
    assert len(merged) == len(a)
    for c in cols:
        left, right = merged[f"{c}_a"], merged[f"{c}_b"]
        assert np.allclose(left.fillna(-999), right.fillna(-999), equal_nan=True), f"リーク: {c}"


def test_horse_history_features_use_only_previous_starts(raw: pd.DataFrame) -> None:
    """h_si_avg5 が「直前5走までの平均」と厳密に一致する。"""
    feat = FeatureBuilder().build(raw)
    horse = feat["horse_id"].value_counts().idxmax()
    h = feat[feat["horse_id"] == horse].sort_values(["date", "race_id"])
    si = h["speed_index"].tolist()
    for i, expected_rows in enumerate(range(len(h))):
        prev = [v for v in si[max(0, i - 5):i] if pd.notna(v)]
        got = h["h_si_avg5"].iloc[i]
        if not prev:
            assert pd.isna(got)
        else:
            assert got == pytest.approx(float(np.mean(prev)), rel=1e-9)


def test_first_start_has_no_history(raw: pd.DataFrame) -> None:
    feat = FeatureBuilder().build(raw)
    debuts = feat[feat["h_starts"] == 0]
    assert len(debuts) > 0
    assert debuts["h_si_avg5"].isna().all()
    assert debuts["h_rest_days"].isna().all()


def test_target_encoding_excludes_current_row(raw: pd.DataFrame) -> None:
    """騎手成績のエンコーディングに自分自身の結果が入っていない。"""
    feat = FeatureBuilder().build(raw)
    jockey = feat["jockey_id"].value_counts().idxmax()
    rows = feat[feat["jockey_id"] == jockey].sort_values(["date", "race_id"])
    first = rows.iloc[0]
    # 初騎乗は事前分布そのもの（データ0件）。相対着順の事前分布は定義上 0.5
    assert first["j_rel"] == pytest.approx(0.5, rel=1e-9)
