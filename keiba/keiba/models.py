"""予測モデル群。

すべてのモデルは **レース内で合計1になる勝率ベクトル** を出力する。
単勝的中率・期待値計算・馬券シミュレーションのすべてがこの前提に依存する。

構成
----
* :class:`ConditionalLogit`  … 条件付きロジット（多項ロジット）。統計的に素直で
  少データでも安定。線形なので外挿に強く、GBM の過学習を打ち消す役割。
* :class:`LgbRanker`         … LightGBM の lambdarank。レース内順位を直接最適化する。
* :class:`LgbBinary`         … LightGBM の二値分類（勝ち/負け）+ レース内正規化。
* :class:`RaceEnsemble`      … 上記を対数線形プーリングで統合。重みは検証データで学習。

温度スケーリング（temperature scaling）を全モデルに入れているのは、順位学習の
スコアがそのままでは確率として過信/過小になるため。校正済み確率でないと
期待値ベースの馬券選択が成立しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax as _softmax_1d

from .grouping import group_softmax, log_linear_pool, race_codes

EPS = 1e-12


# --------------------------------------------------------------- 前処理ユーティリティ
@dataclass
class _NumericPrep:
    """欠損補完 + 標準化（線形モデル用）。統計量は学習データからのみ推定。"""

    medians: pd.Series | None = None
    means: pd.Series | None = None
    stds: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "_NumericPrep":
        self.medians = X.median(numeric_only=True)
        filled = X.fillna(self.medians)
        self.means = filled.mean()
        self.stds = filled.std().replace(0.0, 1.0).fillna(1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        filled = X.fillna(self.medians).fillna(0.0)
        return ((filled - self.means) / self.stds).to_numpy(dtype=float)


def _as_categorical(X: pd.DataFrame, cat_cols: list[str], categories: dict) -> pd.DataFrame:
    X = X.copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c].astype("string"), categories=categories[c])
    return X


# ------------------------------------------------------------------ 条件付きロジット
@dataclass
class ConditionalLogit:
    """レース内 softmax を直接最尤推定する多項ロジットモデル。

    尤度: L = Π_races p(winner)、p_i = exp(x_i·w) / Σ_j exp(x_j·w)
    """

    l2: float = 1.0
    max_iter: int = 300
    name: str = "conditional_logit"
    target_kind: str = "win"
    cat_cols: list[str] = field(default_factory=list)

    prep: _NumericPrep = field(default_factory=_NumericPrep, init=False)
    coef_: np.ndarray | None = field(default=None, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def fit(self, X: pd.DataFrame, y_win: np.ndarray, race_ids, *_ignored) -> "ConditionalLogit":
        self.feature_names_ = list(X.columns)
        self.prep.fit(X)
        Z = self.prep.transform(X)
        codes, n_races = race_codes(race_ids)
        y = np.asarray(y_win, dtype=float)

        def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
            scores = Z @ w
            p = group_softmax(scores, codes, n_races)
            nll = -np.log(np.clip(p[y > 0.5], EPS, None)).sum()
            grad = Z.T @ (p - y)
            nll += self.l2 * float(w @ w)
            grad += 2.0 * self.l2 * w
            return nll, grad

        w0 = np.zeros(Z.shape[1])
        res = minimize(objective, w0, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.coef_ = res.x
        return self

    def predict_proba(self, X: pd.DataFrame, race_ids) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit されていません")
        Z = self.prep.transform(X[self.feature_names_])
        codes, n_races = race_codes(race_ids)
        return group_softmax(Z @ self.coef_, codes, n_races)

    def coefficients(self) -> pd.Series:
        return pd.Series(self.coef_, index=self.feature_names_).sort_values(key=np.abs, ascending=False)


# ------------------------------------------------------------------ 温度スケーリング
def _fit_temperature(scores: np.ndarray, y_win: np.ndarray, codes: np.ndarray, n_races: int) -> float:
    """レース内 softmax の温度を NLL 最小化で決める。"""

    def obj(log_t: float) -> float:
        p = group_softmax(scores / float(np.exp(log_t)), codes, n_races)
        return -np.log(np.clip(p[y_win > 0.5], EPS, None)).sum()

    scale = float(np.std(scores)) or 1.0
    lo, hi = np.log(scale) - 4.0, np.log(scale) + 4.0
    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-4, "maxiter": 200})
    return float(np.exp(res.x))


# --------------------------------------------------------------------- LightGBM 系
_DEFAULT_LGB = {
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "num_threads": 0,
}


@dataclass
class _LgbBase:
    params: dict = field(default_factory=dict)
    num_boost_round: int = 600
    early_stopping_rounds: int = 60
    cat_cols: list[str] = field(default_factory=list)
    name: str = "lgb"

    booster_: lgb.Booster | None = field(default=None, init=False)
    temperature_: float = field(default=1.0, init=False)
    categories_: dict = field(default_factory=dict, init=False)
    feature_names_: list[str] = field(default_factory=list, init=False)

    def _prepare(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit:
            self.feature_names_ = list(X.columns)
            self.categories_ = {
                c: pd.Index(X[c].astype("string").dropna().unique()) for c in self.cat_cols
            }
        X = X[self.feature_names_]
        return _as_categorical(X, self.cat_cols, self.categories_)

    def feature_importance(self) -> pd.Series:
        if self.booster_ is None:
            raise RuntimeError("fit されていません")
        return pd.Series(
            self.booster_.feature_importance("gain"), index=self.booster_.feature_name()
        ).sort_values(ascending=False)


@dataclass
class LgbRanker(_LgbBase):
    """lambdarank。レース内の順位関係そのものを損失にする。"""

    name: str = "lgb_ranker"
    target_kind: str = "pos"

    def fit(self, X, y_pos, race_ids, X_valid=None, y_pos_valid=None, race_ids_valid=None):
        Xf = self._prepare(X, fit=True)
        rel = _relevance(y_pos)
        codes, n_races = race_codes(race_ids)
        params = {**_DEFAULT_LGB, "objective": "lambdarank", "metric": "ndcg",
                  "ndcg_eval_at": [1, 3], "label_gain": list(range(0, 32)), **self.params}
        train_set = lgb.Dataset(Xf, label=rel, group=_group_sizes_in_order(codes, n_races))

        valid_sets, callbacks = [], []
        if X_valid is not None:
            Xv = self._prepare(X_valid, fit=False)
            vcodes, vn = race_codes(race_ids_valid)
            valid_sets = [lgb.Dataset(Xv, label=_relevance(y_pos_valid),
                                      group=_group_sizes_in_order(vcodes, vn), reference=train_set)]
            callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]

        self.booster_ = lgb.train(params, train_set, num_boost_round=self.num_boost_round,
                                  valid_sets=valid_sets, callbacks=callbacks)
        self._calibrate(X_valid if X_valid is not None else X,
                        y_pos_valid if y_pos_valid is not None else y_pos,
                        race_ids_valid if race_ids_valid is not None else race_ids)
        return self

    def _calibrate(self, X, y_pos, race_ids) -> None:
        scores = self.booster_.predict(self._prepare(X, fit=False))
        codes, n = race_codes(race_ids)
        y_win = (np.asarray(y_pos, dtype=float) == 1).astype(float)
        self.temperature_ = _fit_temperature(np.asarray(scores), y_win, codes, n)

    def predict_proba(self, X: pd.DataFrame, race_ids) -> np.ndarray:
        scores = self.booster_.predict(self._prepare(X, fit=False))
        codes, n = race_codes(race_ids)
        return group_softmax(np.asarray(scores) / self.temperature_, codes, n)


@dataclass
class LgbBinary(_LgbBase):
    """勝ち馬を1とする二値分類。出力はレース内で正規化して確率にする。"""

    name: str = "lgb_binary"
    target_kind: str = "win"

    def fit(self, X, y_win, race_ids, X_valid=None, y_win_valid=None, race_ids_valid=None):
        Xf = self._prepare(X, fit=True)
        params = {**_DEFAULT_LGB, "objective": "binary", "metric": "binary_logloss", **self.params}
        train_set = lgb.Dataset(Xf, label=np.asarray(y_win, dtype=float))

        valid_sets, callbacks = [], []
        if X_valid is not None:
            Xv = self._prepare(X_valid, fit=False)
            valid_sets = [lgb.Dataset(Xv, label=np.asarray(y_win_valid, dtype=float),
                                      reference=train_set)]
            callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]

        self.booster_ = lgb.train(params, train_set, num_boost_round=self.num_boost_round,
                                  valid_sets=valid_sets, callbacks=callbacks)
        self._calibrate(X_valid if X_valid is not None else X,
                        y_win_valid if y_win_valid is not None else y_win,
                        race_ids_valid if race_ids_valid is not None else race_ids)
        return self

    def _calibrate(self, X, y_win, race_ids) -> None:
        p = np.clip(self.booster_.predict(self._prepare(X, fit=False)), EPS, 1 - EPS)
        codes, n = race_codes(race_ids)
        logit = np.log(p / (1 - p))
        self.temperature_ = _fit_temperature(logit, np.asarray(y_win, dtype=float), codes, n)

    def predict_proba(self, X: pd.DataFrame, race_ids) -> np.ndarray:
        p = np.clip(self.booster_.predict(self._prepare(X, fit=False)), EPS, 1 - EPS)
        codes, n = race_codes(race_ids)
        return group_softmax(np.log(p / (1 - p)) / self.temperature_, codes, n)


def _relevance(finish_pos) -> np.ndarray:
    """着順を lambdarank 用の関連度（0〜4）に変換する。"""
    pos = np.asarray(finish_pos, dtype=float)
    rel = np.clip(5.0 - pos, 0.0, 4.0)
    return np.nan_to_num(rel, nan=0.0).astype(int)


def _group_sizes_in_order(codes: np.ndarray, n_groups: int) -> np.ndarray:
    """LightGBM の group は「連続した行がひとつのクエリ」を要求する。"""
    if len(codes) and np.any(np.diff(codes) < 0):
        raise ValueError("race_id が連続していません。レース単位でソートしてください。")
    return np.bincount(codes, minlength=n_groups).astype(int)


# --------------------------------------------------------------------- アンサンブル
@dataclass
class RaceEnsemble:
    """複数モデルの確率を対数線形プーリングで統合する。

    重みは検証データの NLL 最小化で決める（softmax パラメータ化により非負・合計1）。
    """

    members: list = field(default_factory=list)
    weights_: np.ndarray | None = field(default=None, init=False)

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.members]

    def fit_weights(self, prob_list: list[np.ndarray], y_win: np.ndarray, race_ids) -> np.ndarray:
        """検証データの NLL を最小化する重みを求める（解析勾配つき L-BFGS）。"""
        codes, n = race_codes(race_ids)
        y = np.asarray(y_win, dtype=float)
        logs = np.stack([np.log(np.clip(p, EPS, None)) for p in prob_list])  # (K, N)

        def obj(theta: np.ndarray) -> tuple[float, np.ndarray]:
            w = _softmax_1d(theta)
            p = group_softmax(w @ logs, codes, n)
            nll = -np.log(np.clip(p[y > 0.5], EPS, None)).sum()
            # dNLL/dw_k = Σ_i (p_i - y_i) log p_ki, さらに softmax の連鎖律
            g_w = logs @ (p - y)
            g_theta = w * (g_w - float(w @ g_w))
            return nll, g_theta

        k = len(prob_list)
        res = minimize(obj, np.zeros(k), jac=True, method="L-BFGS-B",
                       options={"maxiter": 300})
        self.weights_ = _softmax_1d(res.x)
        return self.weights_

    def combine(self, prob_list: list[np.ndarray], race_ids) -> np.ndarray:
        codes, n = race_codes(race_ids)
        w = self.weights_ if self.weights_ is not None else np.full(len(prob_list), 1 / len(prob_list))
        return log_linear_pool(prob_list, w, codes, n)


def make_default_models(cat_cols: Iterable[str]) -> list:
    cat_cols = list(cat_cols)
    return [
        LgbRanker(cat_cols=cat_cols),
        LgbBinary(cat_cols=cat_cols),
        ConditionalLogit(l2=2.0),
    ]
