"""単勝勝率から各券種の的中確率を導き、買い目を組み立てる。

モデルが出すのは「1着になる確率」だけなので、馬連や三連単のような
着順の組み合わせを買うには、そこから着順分布を作る必要がある。

Harville モデル
--------------
1着が決まったら、残りの馬で同じ理屈を繰り返す、という考え方::

    P(i→j)   = p_i · p_j / (1 - p_i)
    P(i→j→k) = p_i · p_j/(1-p_i) · p_k/(1-p_i-p_j)

ただしこの素朴な形は **人気馬の2着・3着確率を過大評価する** ことが知られている
（強い馬が勝てなかったレースは、何か起きているレースなので、2着にすら来ない）。
そこで Lo–Bacon-Shone の補正を入れ、2着以降は勝率を累乗して弱めた強さ
``p^λ`` を使う（λ2 < 1, λ3 < λ2）。λ は実データで推定するのが本筋だが、
既定値は文献で広く使われている値を置いている。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

# Lo & Bacon-Shone (1994) 系の研究でよく使われる値
LAMBDA_2ND = 0.81
LAMBDA_3RD = 0.65


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    return p / p.sum()


def exacta_matrix(p: np.ndarray, lam2: float = LAMBDA_2ND) -> np.ndarray:
    """``M[i, j]`` = i が1着・j が2着になる確率（馬単の確率表）。"""
    p = _normalize(p)
    q = p**lam2
    denom = q.sum() - q                      # i を除いた強さの合計
    m = p[:, None] * q[None, :] / denom[:, None]
    np.fill_diagonal(m, 0.0)
    return m


def quinella_matrix(p: np.ndarray, lam2: float = LAMBDA_2ND) -> np.ndarray:
    """``M[i, j]`` = i と j が1・2着を占める確率（馬連。順不同）。"""
    m = exacta_matrix(p, lam2)
    return m + m.T


def trifecta_prob(p: np.ndarray, i: int, j: int, k: int,
                  lam2: float = LAMBDA_2ND, lam3: float = LAMBDA_3RD) -> float:
    """i→j→k の順で1〜3着になる確率（三連単1点の的中率）。"""
    p = _normalize(p)
    q, r = p**lam2, p**lam3
    second = q[j] / (q.sum() - q[i])
    third = r[k] / (r.sum() - r[i] - r[j])
    return float(p[i] * second * third)


def trio_prob(p: np.ndarray, combo, **kw) -> float:
    """3頭が順不同で1〜3着を占める確率（三連複1点の的中率）。"""
    return float(sum(trifecta_prob(p, *order, **kw) for order in itertools.permutations(combo)))


def top_n_probs(p: np.ndarray, places: int = 3, **kw) -> np.ndarray:
    """各馬が ``places`` 着以内に入る確率（複勝の的中率）。"""
    p = _normalize(p)
    n = len(p)
    out = np.array(p, dtype=float)
    if places >= 2:
        out = exacta_matrix(p, kw.get("lam2", LAMBDA_2ND)).sum(axis=0) + p
    if places >= 3:
        out = np.zeros(n)
        for order in itertools.permutations(range(n), 3):
            prob = trifecta_prob(p, *order, **kw)
            for idx in order:
                out[idx] += prob
    return out


def wide_matrix(p: np.ndarray, places: int = 3, **kw) -> np.ndarray:
    """``M[i, j]`` = i と j が揃って ``places`` 着以内に入る確率（ワイド）。"""
    p = _normalize(p)
    n = len(p)
    m = np.zeros((n, n))
    if places <= 2:
        return quinella_matrix(p, kw.get("lam2", LAMBDA_2ND))
    for order in itertools.permutations(range(n), 3):
        prob = trifecta_prob(p, *order, **kw)
        for a, b in itertools.combinations(order, 2):
            m[a, b] += prob
            m[b, a] += prob
    return m


def bracket_pair_prob(quinella: np.ndarray, brackets, b1: int, b2: int) -> float:
    """2つの枠が1・2着を占める確率（枠連1点の的中率）。

    同じ枠に複数頭いる場合、そのどれが来ても的中するので、枠に属する馬の
    組み合わせをすべて足し上げる。
    """
    left = [i for i, b in enumerate(brackets) if b == b1]
    right = [i for i, b in enumerate(brackets) if b == b2]
    if b1 == b2:
        return float(sum(quinella[i, j] for i, j in itertools.combinations(left, 2)))
    return float(sum(quinella[i, j] for i in left for j in right))


# --------------------------------------------------------------------- 買い目
@dataclass
class Proposal:
    """1つの券種に対する買い目のまとまり。"""

    code: str                     # "tansho" など
    name: str                     # 表示名
    tickets: list = field(default_factory=list)   # [{"legs": [5, 8], "prob": 0.07}, ...]
    note: str = ""                # 買い方の一言説明
    overlap: float = 0.0          # 2点以上が同時に当たる確率（ワイドのみ非ゼロ）

    @property
    def points(self) -> int:
        return len(self.tickets)

    @property
    def expected_hits(self) -> float:
        """的中する枚数の期待値。払戻の期待値はこれに比例する。"""
        return float(sum(t["prob"] for t in self.tickets))

    @property
    def hit_prob(self) -> float:
        """どれか1点でも的中する確率。

        ほとんどの券種は結果が1通りなので同時に当たることはないが、
        ワイドだけは「軸-A」と「軸-B」が同時に当たりうる。その分を引く。
        """
        return self.expected_hits - self.overlap

    @property
    def breakeven_odds(self) -> float:
        """損益が釣り合う最低オッズ = 点数 ÷ 的中枚数の期待値。

        m点を各s円で買い、的中時のオッズをOとすると、期待値が0になるのは
        ``E[的中枚数]·s·O = m·s`` すなわち ``O = m / E[的中枚数]`` のとき。
        """
        e = self.expected_hits
        return float("inf") if e <= 0 else self.points / e


def build_proposals(
    p: np.ndarray,
    draws,
    brackets,
    *,
    n_runners: int | None = None,
    partners_pair: int = 3,
    partners_trio: int = 4,
    **kw,
) -> list[Proposal]:
    """勝率から券種ごとの買い目を組み立てる。

    軸は最も勝率が高い馬（◎）。相手は勝率上位から順に取る「軸1頭流し」で、
    初心者が理解しやすく、点数も抑えられる買い方に統一している。
    """
    p = _normalize(p)
    draws = list(draws)
    brackets = list(brackets)
    n = len(p) if n_runners is None else n_runners
    places = 3 if n >= 8 else 2

    order = list(np.argsort(-p))
    axis = order[0]
    pair_mates = order[1:1 + partners_pair]
    trio_mates = order[1:1 + partners_trio]

    ex = exacta_matrix(p, kw.get("lam2", LAMBDA_2ND))
    qn = ex + ex.T
    place = top_n_probs(p, places, **kw)
    wide = wide_matrix(p, places, **kw)

    def horse(i: int) -> int:
        return draws[i]

    out: list[Proposal] = []

    out.append(Proposal(
        "tansho", "単勝",
        [{"legs": [horse(axis)], "prob": float(p[axis])}],
        "勝率がいちばん高い1頭",
    ))
    out.append(Proposal(
        "fukusho", "複勝",
        [{"legs": [horse(axis)], "prob": float(place[axis])}],
        f"同じ1頭を{places}着以内で",
    ))
    wide_mates = pair_mates[:2]
    # 軸と相手2頭が3頭とも3着以内に入ると、ワイドは2点とも的中する
    wide_overlap = (
        trio_prob(p, (axis, *wide_mates), **kw)
        if places >= 3 and len(wide_mates) == 2 else 0.0
    )
    out.append(Proposal(
        "wide", "ワイド",
        [{"legs": [horse(axis), horse(j)], "prob": float(wide[axis, j])} for j in wide_mates],
        "軸1頭から相手2頭",
        overlap=wide_overlap,
    ))
    out.append(Proposal(
        "umaren", "馬連",
        [{"legs": [horse(axis), horse(j)], "prob": float(qn[axis, j])} for j in pair_mates],
        f"軸1頭から相手{len(pair_mates)}頭",
    ))
    out.append(Proposal(
        "umatan", "馬単",
        [{"legs": [horse(axis), horse(j)], "prob": float(ex[axis, j])} for j in pair_mates],
        "軸を1着に固定して相手へ",
    ))

    if n >= 9:
        # 枠連は「枠」単位なので、同じ枠にいる別の馬が来ても当たる。
        # 軸と相手の組み合わせだけを数えると過小評価になるため、枠全体で足し上げる。
        keys: list[tuple[int, int]] = []
        for j in pair_mates[:2]:
            key = tuple(sorted((brackets[axis], brackets[j])))
            if key not in keys:
                keys.append(key)
        out.append(Proposal(
            "wakuren", "枠連",
            [{"legs": list(k), "prob": bracket_pair_prob(qn, brackets, *k)} for k in keys],
            "馬連と同じ狙いを枠で",
        ))

    out.append(Proposal(
        "sanrenpuku", "三連複",
        [{"legs": sorted([horse(axis), horse(a), horse(b)]),
          "prob": trio_prob(p, (axis, a, b), **kw)}
         for a, b in itertools.combinations(trio_mates, 2)],
        f"軸1頭 + 相手{len(trio_mates)}頭から2頭",
    ))
    out.append(Proposal(
        "sanrentan", "三連単",
        [{"legs": [horse(axis), horse(a), horse(b)],
          "prob": trifecta_prob(p, axis, a, b, **kw)}
         for a, b in itertools.permutations(trio_mates, 2)],
        f"軸を1着に固定、2・3着に相手{len(trio_mates)}頭",
    ))
    return out
