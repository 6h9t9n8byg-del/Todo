"""オッズ（市場確率）の扱い。

単勝オッズは「多数の予想の集約」であり、単体の予測子として極めて強い。
一方で控除率（overround）と本命-大穴バイアス（favourite–longshot bias）が
乗っているため、そのまま 1/オッズ を確率として使うと必ず過大評価になる。

ここでは
1. 控除率を除去して確率に戻す（proportional / power 法）
2. モデル確率と対数線形プーリングでブレンドする
3. ブレンド重みは検証データで最適化する
という手順を提供する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .grouping import EPS, group_normalize, group_sum, log_linear_pool, race_codes

BLEND_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)


def overround(odds: np.ndarray, race_ids) -> np.ndarray:
    """レースごとの控除率込み合計（1.0 が公正、日本の単勝は概ね 1.25 前後）。"""
    codes, n = race_codes(race_ids)
    inv = 1.0 / np.clip(np.asarray(odds, dtype=float), 1.01, None)
    return group_sum(inv, codes, n)


def implied_probabilities(odds, race_ids, method: str = "power") -> np.ndarray:
    """オッズを確率に変換する（レース内で合計1）。

    method
    ------
    ``proportional``
        単純に 1/odds を正規化。実装は簡単だが人気馬を過大評価しやすい。
    ``power``
        Σ (1/o_i)^k = 1 となる k を解く。控除率のぶん Σ1/o_i > 1 なので必ず k>1 に
        なり、確率比が (1/o_i)^k の形で押し広げられる。結果として人気薄の確率が
        相対的に切り下げられ、「大穴が買われすぎてオッズが実力より低い」という
        本命-大穴バイアスの補正になる。
    """
    odds = np.asarray(odds, dtype=float)
    codes, n = race_codes(race_ids)
    valid = np.isfinite(odds) & (odds > 1.0)
    inv = np.where(valid, 1.0 / np.clip(odds, 1.01, None), np.nan)

    if method == "proportional":
        filled = np.where(valid, inv, 0.0)
        return group_normalize(filled, codes, n)

    if method != "power":
        raise ValueError(f"未知の method: {method}")

    out = np.full(len(odds), np.nan)
    for g in range(n):
        idx = np.flatnonzero(codes == g)
        v = inv[idx]
        ok = np.isfinite(v)
        if ok.sum() < 2:
            out[idx] = 1.0 / max(len(idx), 1)
            continue
        p = _solve_power(v[ok])
        vals = np.full(len(idx), np.nan)
        vals[ok] = p
        # オッズ欠損馬は残余確率を等分
        if (~ok).any():
            vals[ok] *= 0.95
            vals[~ok] = 0.05 / (~ok).sum()
        out[idx] = vals
    return group_normalize(out, codes, n)


def _solve_power(inv: np.ndarray, tol: float = 1e-10, max_iter: int = 100) -> np.ndarray:
    """Σ inv_i^k = 1 を満たす k を二分探索し、確率ベクトルを返す。"""
    lo, hi = 0.5, 5.0
    for _ in range(max_iter):
        k = 0.5 * (lo + hi)
        s = np.sum(inv**k)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = k
        else:
            hi = k
    p = inv**k
    return p / p.sum()


def blend(p_model: np.ndarray, p_market: np.ndarray, weight: float, race_ids) -> np.ndarray:
    """対数線形ブレンド: p ∝ p_model^w * p_market^(1-w)。

    weight=1.0 でモデル単独、0.0 で市場単独。
    """
    codes, n = race_codes(race_ids)
    return log_linear_pool([p_model, p_market], np.array([weight, 1.0 - weight]), codes, n)


def fit_blend_weight(
    p_model: np.ndarray, p_market: np.ndarray, y_win: np.ndarray, race_ids,
    grid: np.ndarray = BLEND_GRID,
) -> tuple[float, pd.DataFrame]:
    """検証データの NLL を最小化するブレンド重みを選ぶ。"""
    y = np.asarray(y_win, dtype=float) > 0.5
    rows = []
    for w in grid:
        p = blend(p_model, p_market, float(w), race_ids)
        rows.append({"weight": float(w), "nll": float(-np.log(np.clip(p[y], EPS, None)).mean())})
    table = pd.DataFrame(rows)
    best = float(table.loc[table["nll"].idxmin(), "weight"])
    return best, table
