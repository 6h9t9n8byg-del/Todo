"""実在レースの入線順から馬の強さを推定する（Plackett–Luce レーティング）。

なぜこのモデルか
----------------
合成データ用の GBM は数千レース分の特徴量があって初めて働く。実在馬について
手作業で集められるのは「大レースの上位入線順」程度なので、そこだけを使って
強さを推定できる方法が要る。Plackett–Luce（＝多項ロジットを順位に拡張したもの）は

    P(この着順) = Π_i  exp(θ_i) / Σ_{まだ決まっていない馬} exp(θ_j)

という形で、着順そのものを尤度にできる。チェスの Elo と同じ発想で、
「誰が誰に勝ったか」だけから強さの目盛りを作る。

上位3着しか分からないレースの扱い
--------------------------------
判明分だけで Plackett–Luce を組むと、「G1で3着を3回」のような馬が不当に低く出る。
その3頭が **残り15頭に先着した** という一番大事な情報が抜けるからだ。

そこで「その他の出走馬」を1つの仮想的な相手として置き、出走頭数から判明分を引いた
数だけ分母に加える。着順が分からない馬たちは互いに交換可能とみなす、という仮定で、
これにより「上位に来た」こと自体がきちんと評価される。仮想馬の強さも一緒に推定するので、
G1の平均的な出走馬がどのあたりかという目盛りも同時に得られる。

出走頭数が分からないレースは ``RatingConfig.default_field_size`` を使う。JRAのGIは
14〜18頭立てが大半なので既定を16としているが、これは仮定なので設定で変えられる。

不確かさを確率に反映する
------------------------
1〜2走しか記録のない馬の θ は当然あてにならない。点推定の softmax をそのまま勝率に
すると過信した予想になるため、事後分布 N(θ̂, Σ) から標本を取って softmax の平均を
とる。データが薄い馬ほど確率が平らに寄り、実態に近い謙虚さが出る。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


REST_NAME = "（着順不明の出走馬）"


@dataclass
class RatingConfig:
    prior_sd: float = 1.2          # θ の事前分布の標準偏差（小さいほど強く0へ縮める）
    half_life_days: float = 540.0  # 古いレースの重みが半分になるまでの日数
    default_field_size: int = 16   # 出走頭数が不明なレースで仮定する頭数
    max_iter: int = 500


@dataclass
class Ratings:
    """推定された強さ θ とその不確かさ。"""

    names: list[str]
    theta: np.ndarray
    cov: np.ndarray
    starts: np.ndarray             # 尤度に使えたレース数（判明分）
    config: RatingConfig

    def index(self, name: str) -> int | None:
        try:
            return self.names.index(name)
        except ValueError:
            return None

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0.0, None))

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "horse": self.names,
            "rating": self.theta,
            "se": self.se,
            "starts": self.starts,
        })
        return df.sort_values("rating", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------ 予測
    def _field_moments(self, field: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """出走馬の θ の平均と共分散。未知の馬は事前分布そのもの（θ=0）で扱う。"""
        n = len(field)
        mean = np.zeros(n)
        cov = np.zeros((n, n))
        known = {}
        for i, name in enumerate(field):
            idx = self.index(name)
            if idx is None:
                cov[i, i] = self.config.prior_sd**2
            else:
                mean[i] = self.theta[idx]
                known[i] = idx
        rows = list(known)
        if rows:
            src = np.array([known[i] for i in rows])
            block = self.cov[np.ix_(src, src)]
            cov[np.ix_(rows, rows)] = block
        return mean, cov

    def win_probabilities(self, field: list[str], *, samples: int = 4000,
                          seed: int = 0) -> np.ndarray:
        """出走馬の勝率（レース内で合計1）。

        ``samples`` を 0 にすると点推定の softmax。既定では θ の事後分布から
        標本を取り、推定の不確かさを確率に織り込む。
        """
        mean, cov = self._field_moments(field)
        if samples <= 0:
            return _softmax(mean)

        rng = np.random.default_rng(seed)
        jitter = 1e-9 * np.eye(len(field))
        draws = rng.multivariate_normal(mean, cov + jitter, size=samples, method="eigh")
        shifted = draws - draws.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs.mean(axis=0)

    def field_table(self, field: list[str], **kw) -> pd.DataFrame:
        p = self.win_probabilities(field, **kw)
        point = self.win_probabilities(field, samples=0)
        rows = []
        for i, name in enumerate(field):
            idx = self.index(name)
            rows.append({
                "horse": name,
                "win_prob": p[i],
                "win_prob_point": point[i],
                "rating": self.theta[idx] if idx is not None else 0.0,
                "se": self.se[idx] if idx is not None else self.config.prior_sd,
                "starts": int(self.starts[idx]) if idx is not None else 0,
            })
        return pd.DataFrame(rows).sort_values("win_prob", ascending=False).reset_index(drop=True)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _race_weight(race_date, as_of: date, half_life_days: float) -> float:
    if race_date is None or half_life_days <= 0:
        return 1.0
    age = (as_of - race_date).days
    if age <= 0:
        return 1.0
    return float(0.5 ** (age / half_life_days))


def fit_ratings(races: list[dict], *, config: RatingConfig | None = None,
                as_of: date | None = None) -> Ratings:
    """入線順のリストから Plackett–Luce レーティングを推定する。

    Parameters
    ----------
    races
        ``{"result": [1着, 2着, ...], "date": date | None}`` の並び。
        ``result`` は判明している範囲でよい（2頭でも可）。
    """
    cfg = config or RatingConfig()
    as_of = as_of or date.today()

    names: list[str] = []
    lookup: dict[str, int] = {}
    for race in races:
        for name in race.get("result", []):
            if name not in lookup:
                lookup[name] = len(names)
                names.append(name)
    if not names:
        raise ValueError("入線順のデータがありません")

    # 「着順不明の出走馬」をまとめて表す仮想的な1頭を最後に置く
    rest = len(names)
    names = names + [REST_NAME]
    n = len(names)

    orders: list[np.ndarray] = []
    weights: list[float] = []
    others: list[int] = []            # そのレースで着順不明だった頭数
    starts = np.zeros(n)
    for race in races:
        result = [lookup[name] for name in race.get("result", [])]
        for i in result:
            starts[i] += 1
        field_size = race.get("field_size") or cfg.default_field_size
        unknown = max(int(field_size) - len(result), 0)
        if len(result) < 2 and unknown == 0:
            continue                  # 順序の情報が無い
        orders.append(np.array(result, dtype=int))
        others.append(unknown)
        weights.append(_race_weight(race.get("date"), as_of, cfg.half_life_days))
        starts[rest] += unknown

    precision = 1.0 / cfg.prior_sd**2

    def _steps(order: np.ndarray, unknown: int):
        """各ステップの (勝者, 残りの馬, 残りの不明馬の頭数) を返す。"""
        for step in range(len(order)):
            remaining = order[step:]
            if len(remaining) == 1 and unknown == 0:
                return            # 相手がいないので情報にならない
            yield order[step], remaining, unknown

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nll = 0.5 * precision * float(theta @ theta)
        grad = precision * theta.copy()
        for order, unknown, w in zip(orders, others, weights):
            for winner, remaining, m in _steps(order, unknown):
                vals = theta[remaining]
                if m > 0:
                    denom = logsumexp(np.append(vals, theta[rest] + np.log(m)))
                else:
                    denom = logsumexp(vals)
                nll -= w * (theta[winner] - denom)
                grad[remaining] += w * np.exp(vals - denom)
                if m > 0:
                    grad[rest] += w * np.exp(theta[rest] + np.log(m) - denom)
                grad[winner] -= w
        return nll, grad

    res = minimize(objective, np.zeros(n), jac=True, method="L-BFGS-B",
                   options={"maxiter": cfg.max_iter})
    theta = res.x

    hessian = precision * np.eye(n)
    for order, unknown, w in zip(orders, others, weights):
        for _, remaining, m in _steps(order, unknown):
            idx = np.append(remaining, rest) if m > 0 else remaining
            vals = theta[remaining]
            logits = np.append(vals, theta[rest] + np.log(m)) if m > 0 else vals
            p = _softmax(logits)
            block = np.diag(p) - np.outer(p, p)
            hessian[np.ix_(idx, idx)] += w * block
    cov = np.linalg.inv(hessian)

    return Ratings(names=names, theta=theta, cov=cov, starts=starts, config=cfg)
