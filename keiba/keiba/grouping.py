"""レース（グループ）単位のベクトル演算ユーティリティ。

競馬の確率は「1レースの中で合計1」でなければならない。各馬を独立に予測すると
この制約が壊れるため、softmax / 正規化は必ずレース単位で行う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def race_codes(race_ids) -> tuple[np.ndarray, int]:
    """レースIDを 0..R-1 の整数コードに変換する（出現順を保持）。"""
    codes, uniques = pd.factorize(pd.Series(np.asarray(race_ids)), sort=False)
    return codes.astype(np.int64), len(uniques)


def group_max(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.full(n_groups, -np.inf)
    np.maximum.at(out, codes, values)
    return out


def group_sum(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(codes, weights=values, minlength=n_groups)


def group_softmax(scores: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """レース内 softmax。返り値はレースごとに合計1。"""
    scores = np.asarray(scores, dtype=float)
    shifted = scores - group_max(scores, codes, n_groups)[codes]
    exp = np.exp(shifted)
    denom = group_sum(exp, codes, n_groups)[codes]
    return exp / np.maximum(denom, EPS)


def group_normalize(probs: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """レース内で合計1になるように正規化する。"""
    p = np.clip(np.asarray(probs, dtype=float), EPS, None)
    denom = group_sum(p, codes, n_groups)[codes]
    return p / np.maximum(denom, EPS)


def group_sizes(codes: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(codes, minlength=n_groups).astype(int)


def log_linear_pool(
    prob_list: list[np.ndarray], weights: np.ndarray, codes: np.ndarray, n_groups: int
) -> np.ndarray:
    """対数線形プーリング: p ∝ Π p_k^{w_k}（レース内で再正規化）。

    確率の「意見」を幾何平均で統合する方法。線形平均と違い、どのモデルも低確率と
    見なした馬を確実に低く保てるため、期待値ベースの馬券選択と相性がよい。
    """
    logs = np.zeros_like(prob_list[0], dtype=float)
    for w, p in zip(weights, prob_list):
        logs += w * np.log(np.clip(p, EPS, None))
    return group_softmax(logs, codes, n_groups)
