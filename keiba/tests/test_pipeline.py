"""学習〜予測〜シミュレーションの結合テスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.backtest import BacktestConfig, BettingConfig, simulate_bets, walk_forward
from keiba.data import GeneratorConfig, generate_dataset
from keiba.features import FeatureBuilder
from keiba.metrics import baseline_nll, race_nll, top1_accuracy
from keiba.models import ConditionalLogit
from keiba.pipeline import Predictor, PredictorConfig
from keiba.report import render_report
from keiba.schema import coerce


@pytest.fixture(scope="module")
def feat() -> pd.DataFrame:
    raw = coerce(generate_dataset(GeneratorConfig(seed=11, n_race_days=150, n_horses=700)))
    return FeatureBuilder().build(raw)


@pytest.fixture(scope="module")
def fitted(feat: pd.DataFrame) -> tuple[Predictor, pd.DataFrame]:
    cutoff = feat["date"].quantile(0.8)
    train = feat[feat["date"] <= cutoff]
    test = feat[feat["date"] > cutoff]
    predictor = Predictor(config=PredictorConfig(valid_days=60, min_valid_races=100))
    predictor.fit(train)
    return predictor, test


def test_predictions_are_valid_probabilities(fitted) -> None:
    predictor, test = fitted
    preds = predictor.predict(test)
    for col in ("p_model", "p_final"):
        assert preds[col].between(0, 1).all()
        sums = preds.groupby("race_id")[col].sum()
        assert np.allclose(sums, 1.0, atol=1e-8), f"{col} がレース内で合計1になっていない"


def test_model_beats_the_uniform_baseline(fitted) -> None:
    predictor, test = fitted
    preds = predictor.predict(test)
    y = preds["y_win"].to_numpy()
    assert race_nll(preds["p_model"].to_numpy(), y, preds["race_id"]) < baseline_nll(preds["race_id"])
    assert top1_accuracy(preds["p_model"].to_numpy(), y, preds["race_id"]) > 1.0 / 12.0


def test_blending_with_the_market_does_not_hurt(fitted) -> None:
    predictor, test = fitted
    preds = predictor.predict(test)
    y = preds["y_win"].to_numpy()
    nll_market = race_nll(preds["p_market"].to_numpy(), y, preds["race_id"])
    nll_final = race_nll(preds["p_final"].to_numpy(), y, preds["race_id"])
    # 検証データで重みを選んでいるので、市場単独より極端に悪化してはいけない
    assert nll_final < nll_market + 0.05


def test_prediction_on_a_race_without_results(fitted, feat: pd.DataFrame) -> None:
    """結果が未確定（NaN）の出馬表でも予測できる。"""
    predictor, test = fitted
    target_race = test["race_id"].iloc[-1]
    entries = test[test["race_id"] == target_race].copy()
    entries[["finish_pos", "finish_time", "corner_pos", "last_3f"]] = np.nan
    preds = predictor.predict(entries)
    assert len(preds) == len(entries)
    assert preds["p_final"].sum() == pytest.approx(1.0)
    assert render_report(preds).startswith("<!doctype html>")


def test_conditional_logit_recovers_a_known_signal() -> None:
    """既知の線形構造を条件付きロジットが復元できる。"""
    rng = np.random.default_rng(4)
    n_races, size = 600, 10
    x = rng.normal(size=n_races * size)
    noise = rng.gumbel(size=n_races * size)
    races = np.repeat([f"R{i}" for i in range(n_races)], size)
    utility = 1.5 * x + noise
    df = pd.DataFrame({"x": x, "race_id": races, "u": utility})
    y = (df.groupby("race_id")["u"].transform("max") == df["u"]).astype(float).to_numpy()

    model = ConditionalLogit(l2=0.01).fit(df[["x"]], y, df["race_id"])
    assert model.coefficients()["x"] > 0.5
    p = model.predict_proba(df[["x"]], df["race_id"])
    assert np.allclose(pd.Series(p).groupby(races).sum(), 1.0)


def test_walk_forward_produces_out_of_sample_predictions(feat: pd.DataFrame) -> None:
    cfg = BacktestConfig(
        initial_train_days=300, step_days=90, verbose=False,
        predictor=PredictorConfig(valid_days=60, min_valid_races=80),
    )
    preds = walk_forward(feat, cfg)
    assert preds["fold"].nunique() >= 1
    # 各フォールドの予測期間は学習期間より後
    assert preds["date"].min() > feat["date"].min() + pd.Timedelta(days=299)
    assert preds.duplicated(["race_id", "horse_id"]).sum() == 0


def test_betting_simulation_accounting_is_consistent(feat: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    n = 2000
    preds = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="h"),
        "race_id": np.repeat([f"R{i}" for i in range(n // 10)], 10),
        "horse_id": [f"H{i}" for i in range(n)],
        "odds": rng.uniform(2, 30, n),
        "p_final": rng.uniform(0.02, 0.4, n),
        "p_market": rng.uniform(0.02, 0.4, n),
        "y_win": 0.0,
    })
    preds.loc[::10, "y_win"] = 1.0
    preds["ev"] = preds["p_final"] * preds["odds"] - 1.0
    preds["kelly"] = (preds["ev"] / (preds["odds"] - 1.0)).clip(lower=0)

    res = simulate_bets(preds, BettingConfig(ev_threshold=0.1, stake_mode="flat", flat_stake=100))
    bets, summary = res["bets"], res["summary"]
    assert summary["n_bets"] == len(bets)
    assert summary["total_stake"] == pytest.approx(bets["stake"].sum())
    assert summary["roi"] == pytest.approx(bets["payout"].sum() / bets["stake"].sum() - 1.0)
    # 払戻はオッズ×掛け金、外れは0
    assert np.allclose(bets.loc[bets["won"], "payout"],
                       bets.loc[bets["won"], "stake"] * bets.loc[bets["won"], "odds"])
    assert (bets.loc[~bets["won"], "payout"] == 0).all()
    assert summary["final_bankroll"] == pytest.approx(
        1_000_000 + bets["payout"].sum() - bets["stake"].sum()
    )


def test_save_and_load_roundtrip(fitted, tmp_path) -> None:
    predictor, test = fitted
    path = tmp_path / "model.joblib"
    predictor.save(str(path))
    loaded = Predictor.load(str(path))
    a = predictor.predict(test)["p_final"].to_numpy()
    b = loaded.predict(test)["p_final"].to_numpy()
    assert np.allclose(a, b)
