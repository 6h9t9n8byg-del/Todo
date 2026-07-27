"""学習〜予測の統合パイプライン。

``Predictor.fit`` の中でやっていること（精度に効く順）:

1. 時系列で学習/検証を分ける（未来のデータで検証しない）
2. 3種のモデルを学習し、検証データで温度スケーリング（確率の校正）
3. 検証データで対数線形プーリングの重みを最適化（アンサンブル）
4. アイソトニック回帰で最終的な確率のズレを補正
5. 市場（オッズ）確率とのブレンド重みを検証データで最適化
6. 早期終了で決まった本数を使い、検証期間も含めた全データで再学習
   （直近データは最も価値が高いので捨てない）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import market as market_mod
from .features import CATEGORICAL_FEATURES, FeatureBuilder
from .grouping import group_normalize, race_codes
from .models import (
    ConditionalLogit,
    LgbBinary,
    LgbRanker,
    RaceEnsemble,
    make_default_models,
)


@dataclass
class PredictorConfig:
    valid_days: int = 150          # 検証に使う直近日数
    min_valid_races: int = 200
    use_market: bool = True        # オッズとのブレンドを行うか
    market_method: str = "power"
    isotonic: bool = True
    min_isotonic_rows: int = 1500
    refit_full: bool = True
    verbose: bool = False


@dataclass
class Predictor:
    config: PredictorConfig = field(default_factory=PredictorConfig)
    builder: FeatureBuilder = field(default_factory=FeatureBuilder)

    models: list = field(default_factory=list, init=False)
    ensemble: RaceEnsemble = field(default_factory=RaceEnsemble, init=False)
    isotonic_: IsotonicRegression | None = field(default=None, init=False)
    blend_weight_: float = field(default=1.0, init=False)
    blend_table_: pd.DataFrame | None = field(default=None, init=False)
    valid_report_: dict = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ 学習
    def fit(self, feat: pd.DataFrame) -> "Predictor":
        cfg = self.config
        labeled = feat[feat["finish_pos"].notna()].sort_values(["date", "race_id"])
        if labeled.empty:
            raise ValueError("学習に使える着順付きデータがありません")

        train, valid = _time_split(labeled, cfg.valid_days, cfg.min_valid_races)
        self.models = make_default_models(CATEGORICAL_FEATURES)

        prob_valid = []
        for m in self.models:
            cols = _columns_for(m, self.builder)
            m.fit(
                train[cols], _target(train, m.target_kind), train["race_id"],
                valid[cols], _target(valid, m.target_kind), valid["race_id"],
            )
            prob_valid.append(m.predict_proba(valid[cols], valid["race_id"]))
            if cfg.verbose:
                print(f"  [fit] {m.name} done")

        y_valid = _target(valid, "win")
        self.ensemble = RaceEnsemble(self.models)
        weights = self.ensemble.fit_weights(prob_valid, y_valid, valid["race_id"])
        p_ens = self.ensemble.combine(prob_valid, valid["race_id"])

        # --- 市場ブレンド ---
        self.blend_weight_, self.blend_table_ = 1.0, None
        p_blend = p_ens
        if cfg.use_market and valid["odds"].notna().any():
            p_mkt = market_mod.implied_probabilities(
                valid["odds"], valid["race_id"], cfg.market_method
            )
            self.blend_weight_, self.blend_table_ = market_mod.fit_blend_weight(
                p_ens, p_mkt, y_valid, valid["race_id"]
            )
            p_blend = market_mod.blend(p_ens, p_mkt, self.blend_weight_, valid["race_id"])

        # --- アイソトニック校正（最終確率に対して行う） ---
        # ブレンド後に残る系統的なズレ（人気馬を過小・人気薄を過大に見積もる傾向）を
        # 単調変換で補正する。悪化する場合は採用しない。
        self.isotonic_ = None
        if cfg.isotonic and len(valid) >= cfg.min_isotonic_rows:
            if _isotonic_helps(p_blend, y_valid, valid["race_id"]):
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                self.isotonic_ = iso.fit(p_blend, y_valid)

        self.valid_report_ = {
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "train_period": (str(train["date"].min().date()), str(train["date"].max().date())),
            "valid_period": (str(valid["date"].min().date()), str(valid["date"].max().date())),
            "ensemble_weights": dict(zip(self.ensemble.names, np.round(weights, 3))),
            "blend_weight_model": self.blend_weight_,
            "isotonic": self.isotonic_ is not None,
        }

        # --- 検証期間も含めて再学習（直近データを捨てない） ---
        if cfg.refit_full:
            for m in self.models:
                _refit_on_full(m, labeled, _columns_for(m, self.builder))

        return self

    # ------------------------------------------------------------------ 予測
    def predict(self, feat: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("fit されていません")
        races = feat["race_id"]
        probs = [m.predict_proba(feat[_columns_for(m, self.builder)], races) for m in self.models]
        p_model = self.ensemble.combine(probs, races)

        out = feat[["race_id", "date", "venue", "surface", "distance", "going", "race_class",
                    "draw", "horse_id", "horse_name", "jockey_id", "odds", "n_runners"]].copy()
        out["p_model"] = p_model

        if self.config.use_market and feat["odds"].notna().any():
            p_mkt = market_mod.implied_probabilities(
                feat["odds"], races, self.config.market_method
            )
            out["p_market"] = p_mkt
            p_final = market_mod.blend(p_model, p_mkt, self.blend_weight_, races)
        else:
            out["p_market"] = np.nan
            p_final = p_model

        if self.isotonic_ is not None:
            p_final = _apply_isotonic(self.isotonic_, p_final, races)
        out["p_final"] = p_final

        odds = pd.to_numeric(out["odds"], errors="coerce")
        out["ev"] = out["p_final"] * odds - 1.0
        out["kelly"] = np.where(odds > 1.0, out["ev"] / (odds - 1.0), np.nan).clip(min=0.0)
        if "finish_pos" in feat.columns:
            out["finish_pos"] = feat["finish_pos"].values
            out["y_win"] = (feat["finish_pos"].values == 1).astype(float)
        return out.sort_values(["date", "race_id", "p_final"], ascending=[True, True, False])

    # ------------------------------------------------------------------ 保存
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Predictor":
        return joblib.load(path)

    # --------------------------------------------------------------- 説明用
    def feature_importance(self, top: int = 30) -> pd.DataFrame:
        frames = {}
        for m in self.models:
            if isinstance(m, (LgbRanker, LgbBinary)):
                imp = m.feature_importance()
                frames[m.name] = imp / imp.sum()
            elif isinstance(m, ConditionalLogit):
                coef = m.coefficients().abs()
                frames[m.name] = coef / coef.sum()
        df = pd.DataFrame(frames).fillna(0.0)
        df["mean"] = df.mean(axis=1)
        return df.sort_values("mean", ascending=False).head(top)


# ------------------------------------------------------------------ ヘルパー
def _columns_for(model, builder: FeatureBuilder) -> list[str]:
    """線形モデルには数値列のみ、GBM にはカテゴリ列も渡す。"""
    return builder.feature_columns if getattr(model, "cat_cols", None) else builder.numeric_features


def _target(df: pd.DataFrame, kind: str) -> np.ndarray:
    pos = df["finish_pos"].to_numpy(dtype=float)
    return pos if kind == "pos" else (pos == 1).astype(float)


def _time_split(df: pd.DataFrame, valid_days: int, min_valid_races: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """直近 ``valid_days`` を検証に回す。少なすぎる場合はレース数で確保する。"""
    cutoff = df["date"].max() - pd.Timedelta(days=valid_days)
    valid = df[df["date"] > cutoff]
    if valid["race_id"].nunique() < min_valid_races:
        race_order = df[["race_id", "date"]].drop_duplicates().sort_values("date")
        take = race_order.tail(max(min_valid_races, 1))["race_id"]
        valid = df[df["race_id"].isin(set(take))]
    train = df[~df.index.isin(valid.index)]
    if train.empty:
        raise ValueError("学習データが不足しています（期間を延ばしてください）")
    return train, valid


def _apply_isotonic(iso: IsotonicRegression, probs: np.ndarray, race_ids) -> np.ndarray:
    codes, n = race_codes(race_ids)
    return group_normalize(iso.predict(probs) + 1e-6, codes, n)


def _isotonic_helps(p: np.ndarray, y: np.ndarray, race_ids, seed: int = 0) -> bool:
    """アイソトニック校正を採用すべきか、検証データ内の2分割交差検証で判定する。

    同じデータで当てはめて同じデータで評価すると必ず「改善」してしまい、
    アウトオブサンプルではむしろ悪化することがあるため。
    """
    races = pd.Series(np.asarray(race_ids))
    uniq = races.unique()
    rng = np.random.default_rng(seed)
    half = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    mask_a = races.isin(half).to_numpy()

    gains = []
    for train_mask in (mask_a, ~mask_a):
        test_mask = ~train_mask
        if train_mask.sum() < 500 or test_mask.sum() < 500 or y[train_mask].sum() < 50:
            return False
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p[train_mask], y[train_mask])
        p_cal = _apply_isotonic(iso, p[test_mask], races[test_mask])
        gains.append(_nll(p[test_mask], y[test_mask]) - _nll(p_cal, y[test_mask]))
    return float(np.mean(gains)) > 0.0


def _nll(probs: np.ndarray, y_win: np.ndarray) -> float:
    mask = y_win > 0.5
    return float(-np.log(np.clip(probs[mask], 1e-12, None)).mean())


def _refit_on_full(model, labeled: pd.DataFrame, cols: list[str]) -> None:
    """早期終了で得た本数のまま、検証期間も含めた全データで学習し直す。

    温度スケーリング（校正）は検証データで得た値を保持する。学習データで
    校正し直すと必ず過信気味になるため。
    """
    y = _target(labeled, model.target_kind)
    if isinstance(model, (LgbRanker, LgbBinary)):
        best = model.booster_.best_iteration or model.num_boost_round
        temp = model.temperature_
        model.num_boost_round = max(int(best * 1.15), 50)
        model.fit(labeled[cols], y, labeled["race_id"])
        model.temperature_ = temp
    else:
        model.fit(labeled[cols], y, labeled["race_id"])
