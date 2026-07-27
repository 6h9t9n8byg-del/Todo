"""評価指標。

「当たったかどうか」だけでなく、確率としての質（NLL・Brier・キャリブレーション）
を必ず見る。馬券の期待値計算は確率の絶対値に依存するため、順位が合っていても
確率がズレていれば損をするため。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .grouping import group_max, group_sum, race_codes

EPS = 1e-12


def _codes(race_ids):
    return race_codes(race_ids)


def top1_accuracy(probs: np.ndarray, y_win: np.ndarray, race_ids) -> float:
    """本命（最高確率の馬）の的中率。"""
    codes, n = _codes(race_ids)
    probs = np.asarray(probs, dtype=float)
    top = group_max(probs, codes, n)[codes]
    is_top = probs >= top - 1e-15
    # 同確率が複数なら等分カウント
    n_top = group_sum(is_top.astype(float), codes, n)[codes]
    hit = np.asarray(y_win, dtype=float) * is_top / np.maximum(n_top, 1)
    return float(group_sum(hit, codes, n).sum() / n)


def topk_hit_rate(probs: np.ndarray, y_win: np.ndarray, race_ids, k: int = 3) -> float:
    """上位k頭の中に1着馬が含まれる割合。"""
    df = pd.DataFrame({"race": np.asarray(race_ids), "p": np.asarray(probs, dtype=float),
                       "y": np.asarray(y_win, dtype=float)})
    rank = df.groupby("race")["p"].rank(ascending=False, method="first")
    hit = df.loc[rank <= k].groupby("race")["y"].max()
    return float(hit.mean())


def race_nll(probs: np.ndarray, y_win: np.ndarray, race_ids) -> float:
    """1着馬に付けた確率の負の対数尤度（レース平均）。低いほど良い。"""
    p = np.clip(np.asarray(probs, dtype=float), EPS, None)
    mask = np.asarray(y_win, dtype=float) > 0.5
    return float(-np.log(p[mask]).mean())


def brier_score(probs: np.ndarray, y_win: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_win, dtype=float)
    return float(np.mean((p - y) ** 2))


def baseline_nll(race_ids) -> float:
    """全馬等確率（1/頭数）のときの NLL。モデルの下限比較用。"""
    codes, n = _codes(race_ids)
    sizes = np.bincount(codes, minlength=n)
    return float(np.log(sizes).mean())


def calibration_table(probs: np.ndarray, y_win: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """予測確率帯ごとの実際の勝率（キャリブレーション曲線）。"""
    df = pd.DataFrame({"p": np.asarray(probs, dtype=float), "y": np.asarray(y_win, dtype=float)})
    edges = np.unique(np.quantile(df["p"], np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(0, 1, bins + 1)
    df["bin"] = pd.cut(df["p"], bins=edges, include_lowest=True)
    out = df.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")
    )
    out["diff"] = out["actual"] - out["predicted"]
    return out.reset_index()


def summarize(probs: np.ndarray, y_win: np.ndarray, race_ids) -> dict:
    return {
        "races": int(pd.Series(np.asarray(race_ids)).nunique()),
        "runners": int(len(probs)),
        "top1_accuracy": top1_accuracy(probs, y_win, race_ids),
        "top3_hit_rate": topk_hit_rate(probs, y_win, race_ids, 3),
        "nll": race_nll(probs, y_win, race_ids),
        "baseline_nll": baseline_nll(race_ids),
        "brier": brier_score(probs, y_win),
    }


def format_summary(name: str, s: dict) -> str:
    return (
        f"{name:<22} races={s['races']:>5} "
        f"top1={s['top1_accuracy']:.3f} top3={s['top3_hit_rate']:.3f} "
        f"NLL={s['nll']:.4f} (baseline {s['baseline_nll']:.4f}) Brier={s['brier']:.4f}"
    )
