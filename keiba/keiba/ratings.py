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
これにより「上位に来た」こと自体がきちんと評価される。

さらにこの仮想馬は **レースの格ごとに別々** に持つ。GIで5着だった馬とG3で5着だった馬を
同じ「負け」として扱うと、GIで善戦した馬が記録の無い馬より低く評価されてしまうためだ
（実際、格を分ける前はヴィクトリアマイル4着・5着の馬が、戦績ゼロの馬より下に来ていた）。
格ごとの仮想馬の強さも一緒に推定されるので、「GIの平均的な出走馬」と「G3の平均的な
出走馬」がどれだけ違うか、という目盛りも同時に得られる。

ただし「着順不明の馬がどれだけ強いか」はデータからはほとんど決まらない。着順不明の
馬は毎回別の馬で、同じ馬として追跡できないからだ。実際、自由に推定させると −4 まで
下がってしまい、「その集団に先着しても何の情報にもならない」という極端な解に落ちる
（GIで4着・5着だった馬が、戦績ゼロの馬より低く評価されるのはこのため）。

そこでこの量は **推定するものではなく仮定** として扱い、``rest_prior_mean`` /
``rest_prior_sd`` で狭い事前分布を置く。既定の −1.5 は「上位に来る馬は、着順不明の
集団の1頭に対して8割方は先着する」という程度の想定にあたる。格ごとに別々の
パラメータを持たせてあるので、データが動かせる範囲で class の差も出る。

出走頭数が分からないレースは ``RatingConfig.default_field_size`` を使う。JRAのGIは
14〜18頭立てが大半なので既定を16としているが、これも仮定なので設定で変えられる。

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


REST_PREFIX = "（着順不明の出走馬"
REST_NAME = REST_PREFIX + "）"        # 格が不明なレースぶん


def rest_name(grade: str | None) -> str:
    """格ごとの仮想的な「着順不明の出走馬」の名前。"""
    return REST_NAME if not grade else f"{REST_PREFIX}・{grade}）"


def is_rest(name: str) -> bool:
    return name.startswith(REST_PREFIX)


@dataclass
class RatingConfig:
    prior_sd: float = 1.2          # θ の事前分布の標準偏差（小さいほど強く0へ縮める）
    half_life_days: float = 540.0  # 古いレースの重みが半分になるまでの日数
    default_field_size: int = 16   # 出走頭数が不明なレースで仮定する頭数
    rest_prior_mean: float = -1.5  # 着順不明の集団の強さ（データでは決まらないので仮定）
    rest_prior_sd: float = 0.6     # その仮定をどれだけ動かしてよいか
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

    # 「着順不明の出走馬」をレースの格ごとに1頭ずつ、最後に置く
    grades = []
    for race in races:
        grade = race.get("grade")
        if grade not in grades:
            grades.append(grade)
    rest_of: dict = {}
    for grade in grades:
        rest_of[grade] = len(names)
        names = names + [rest_name(grade)]
    n = len(names)

    orders: list[np.ndarray] = []
    weights: list[float] = []
    others: list[int] = []            # そのレースで着順不明だった頭数
    rests: list[int] = []             # そのレースの仮想馬（格ごと）
    starts = np.zeros(n)
    for race in races:
        result = [lookup[name] for name in race.get("result", [])]
        for i in result:
            starts[i] += 1
        field_size = race.get("field_size") or cfg.default_field_size
        unknown = max(int(field_size) - len(result), 0)
        if len(result) < 2 and unknown == 0:
            continue                  # 順序の情報が無い
        rest = rest_of[race.get("grade")]
        orders.append(np.array(result, dtype=int))
        others.append(unknown)
        rests.append(rest)
        weights.append(_race_weight(race.get("date"), as_of, cfg.half_life_days))
        starts[rest] += unknown

    # 事前分布: 実在馬は 0 中心、仮想馬は rest_prior_mean 中心（狭め）
    precision = np.full(n, 1.0 / cfg.prior_sd**2)
    prior_mean = np.zeros(n)
    for idx in rest_of.values():
        precision[idx] = 1.0 / cfg.rest_prior_sd**2
        prior_mean[idx] = cfg.rest_prior_mean

    def _steps(order: np.ndarray, unknown: int):
        """各ステップの (勝者, 残りの馬, 残りの不明馬の頭数) を返す。"""
        for step in range(len(order)):
            remaining = order[step:]
            if len(remaining) == 1 and unknown == 0:
                return            # 相手がいないので情報にならない
            yield order[step], remaining, unknown

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        dev = theta - prior_mean
        nll = 0.5 * float(precision @ (dev**2))
        grad = precision * dev
        for order, unknown, rest, w in zip(orders, others, rests, weights):
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

    hessian = np.diag(precision)
    for order, unknown, rest, w in zip(orders, others, rests, weights):
        for _, remaining, m in _steps(order, unknown):
            idx = np.append(remaining, rest) if m > 0 else remaining
            vals = theta[remaining]
            logits = np.append(vals, theta[rest] + np.log(m)) if m > 0 else vals
            p = _softmax(logits)
            block = np.diag(p) - np.outer(p, p)
            hessian[np.ix_(idx, idx)] += w * block
    cov = np.linalg.inv(hessian)

    return Ratings(names=names, theta=theta, cov=cov, starts=starts, config=cfg)
