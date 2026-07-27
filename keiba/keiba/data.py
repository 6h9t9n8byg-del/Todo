"""合成レースデータ生成器。

実データ（JRA-VAN / netkeiba 等）を用意しなくてもパイプライン全体を動かし、
バックテストや回帰テストを再現可能にするためのもの。

生成器は「潜在能力 → 着順 → オッズ」という因果順序を持つ:

1. 各馬に潜在能力・適性（芝ダート・距離・馬場・脚質）と調子（AR(1)）を持たせる
2. レースごとに出走馬を選び、展開（ペース）を含めて着順とタイムを決める
3. 市場（オッズ）は「真の勝率をノイズ付きで観測したもの」として生成する

市場は強いが完全ではなく、意図的に展開適性（pace fit）を過小評価する。
そのため理論上は EV>0 の馬券が存在する。**これは合成データ上の性質であり、
実馬券が儲かることの証拠ではない**（README の注意書きを参照）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VENUES = ["Tokyo", "Nakayama", "Hanshin", "Kyoto", "Chukyo", "Kokura", "Niigata", "Sapporo"]
SURFACES = ["turf", "dirt"]
GOINGS = ["firm", "good", "yielding", "soft"]
GOING_PENALTY = {"firm": 0.0, "good": 0.6, "yielding": 1.6, "soft": 2.6}  # 秒/1000m
DISTANCES = [1200, 1400, 1600, 1800, 2000, 2200, 2400]


@dataclass
class GeneratorConfig:
    n_horses: int = 1200
    n_jockeys: int = 70
    n_trainers: int = 45
    n_sires: int = 40
    n_race_days: int = 220          # 開催日数（週2日 ≒ 2年強）
    races_per_day: int = 8
    start_date: str = "2021-01-02"
    min_field: int = 8
    max_field: int = 16
    class_band: int = 2             # 同一レースに集まるクラスの幅（大きいほど能力差が開く）
    seed: int = 42
    # 以下3つは「1番人気の勝率 ≒ 32%・単勝オッズ中央値 ≒ 2.5」という JRA の実績に
    # 合うよう調整してある。実データに近い難易度でないとバックテストの意味がない。
    outcome_noise: float = 1.7      # レース当日の偶然のばらつき（大きいほど荒れる）
    market_noise: float = 0.9       # 市場の観測ノイズ（大きいほど市場が弱い）
    market_pace_blindness: float = 1.0  # 市場が展開適性を無視する割合 (0=完全に織り込む)
    overround: float = 1.20         # 控除率込みの合計オッズ率


@dataclass
class _HorseState:
    form: float = 0.0
    last_date: pd.Timestamp | None = None
    starts: int = 0
    rating: float = 0.0


@dataclass
class SyntheticRaceGenerator:
    config: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng

        self.sire_ability = rng.normal(0, 0.55, cfg.n_sires)
        self.jockey_skill = rng.normal(0, 0.35, cfg.n_jockeys)
        self.trainer_skill = rng.normal(0, 0.22, cfg.n_trainers)

        h = cfg.n_horses
        self.horse_sire = rng.integers(0, cfg.n_sires, h)
        # 能力は「種牡馬の影響 + 個体差」
        self.ability = 0.6 * self.sire_ability[self.horse_sire] + rng.normal(0, 0.85, h)
        self.turf_pref = rng.normal(0, 0.45, h)          # 正=芝向き, 負=ダート向き
        self.opt_distance = rng.uniform(1150, 2450, h)
        self.going_pref = rng.normal(0, 0.35, h)          # 正=道悪巧者
        self.front_tendency = rng.beta(2, 2, h)           # 1に近いほど逃げ・先行
        self.consistency = rng.uniform(0.45, 0.95, h)     # 走りのブレ（小さいほど堅実）
        self.base_body_weight = rng.normal(478, 28, h)
        self.jockey_of_horse = rng.integers(0, cfg.n_jockeys, h)   # 主戦騎手
        self.trainer_of_horse = rng.integers(0, cfg.n_trainers, h)
        self.sex = rng.choice(["M", "F", "G"], h, p=[0.45, 0.4, 0.15])
        self.birth_year = rng.integers(2017, 2021, h)
        # クラス（能力に応じた出走区分）
        self.horse_class = np.clip(np.round(4.5 + 1.6 * self.ability), 1, 9).astype(int)

    # ------------------------------------------------------------------ 生成
    def generate(self) -> pd.DataFrame:
        cfg = self.config
        rng = self.rng
        states = {i: _HorseState() for i in range(cfg.n_horses)}

        dates = _race_dates(cfg.start_date, cfg.n_race_days)
        rows: list[dict] = []
        race_seq = 0

        for date in dates:
            venue = VENUES[rng.integers(0, len(VENUES))]
            day_going = str(rng.choice(GOINGS, p=[0.45, 0.3, 0.15, 0.10]))
            for _ in range(cfg.races_per_day):
                race_seq += 1
                race_id = f"R{race_seq:06d}"
                surface = str(rng.choice(SURFACES, p=[0.62, 0.38]))
                distance = int(rng.choice(DISTANCES))
                race_class = int(np.clip(rng.normal(4.5, 1.8), 1, 9).round())
                field_size = int(rng.integers(cfg.min_field, cfg.max_field + 1))

                runners = self._pick_field(race_class, field_size, date, states)
                if len(runners) < cfg.min_field:
                    continue

                rows.extend(
                    self._run_race(
                        race_id=race_id,
                        date=date,
                        venue=venue,
                        surface=surface,
                        distance=distance,
                        going=day_going,
                        race_class=race_class,
                        runners=runners,
                        states=states,
                    )
                )

        df = pd.DataFrame(rows)
        return df.sort_values(["date", "race_id", "draw"]).reset_index(drop=True)

    # ---------------------------------------------------------------- 出走馬
    def _pick_field(
        self, race_class: int, field_size: int, date: pd.Timestamp, states: dict[int, _HorseState]
    ) -> np.ndarray:
        rng = self.rng
        eligible = np.flatnonzero(np.abs(self.horse_class - race_class) <= self.config.class_band)
        # 中2週未満は出走させない（ローテーションの再現）
        ready = [
            h
            for h in eligible
            if states[h].last_date is None or (date - states[h].last_date).days >= 14
        ]
        if len(ready) <= field_size:
            return np.array(ready, dtype=int)
        return rng.choice(np.array(ready, dtype=int), size=field_size, replace=False)

    def _run_race(
        self,
        *,
        race_id: str,
        date: pd.Timestamp,
        venue: str,
        surface: str,
        distance: int,
        going: str,
        race_class: int,
        runners: np.ndarray,
        states: dict[int, _HorseState],
    ) -> list[dict]:
        cfg = self.config
        rng = self.rng
        n = len(runners)

        # --- 調子(AR(1)) と休養 ---
        rest_days = np.empty(n)
        form = np.empty(n)
        for i, h in enumerate(runners):
            st = states[h]
            st.form = 0.6 * st.form + rng.normal(0, 0.45)
            form[i] = st.form
            rest_days[i] = 28.0 if st.last_date is None else (date - st.last_date).days

        # 適度な間隔（3〜8週）が最良、長期休養明けは割引
        rest_effect = -0.25 * np.abs(np.log(np.clip(rest_days, 7, 400) / 35.0))

        # --- 適性 ---
        ability = self.ability[runners]
        surf_fit = np.where(surface == "turf", self.turf_pref[runners], -self.turf_pref[runners])
        dist_fit = -0.9 * ((distance - self.opt_distance[runners]) / 700.0) ** 2
        going_fit = self.going_pref[runners] * (GOING_PENALTY[going] / 2.6)

        age = date.year - self.birth_year[runners] + (date.month >= 6)
        age_effect = -0.12 * (age - 5.0) ** 2 / 4.0

        jk = self.jockey_of_horse[runners]
        tr = self.trainer_of_horse[runners]
        # たまに乗り替わり
        switch = rng.random(n) < 0.25
        jk = np.where(switch, rng.integers(0, cfg.n_jockeys, n), jk)

        # --- 斤量・馬体重・枠 ---
        weight_carried = (
            52.0
            + 0.4 * race_class
            + np.where(self.sex[runners] == "F", -2.0, 0.0)
            + np.clip(np.round(ability * 1.5 * 2) / 2, -2.0, 3.0)
            + np.round(rng.normal(0, 0.5, n) * 2) / 2
        )
        body_weight = np.round(self.base_body_weight[runners] + rng.normal(0, 8, n))
        draw = rng.permutation(n) + 1

        # 内枠有利（短距離・小回りほど強い）
        draw_effect = -0.35 * (draw - 1) / max(n - 1, 1) * (2400.0 / distance) ** 0.5

        # --- 展開（ペース） ---
        front = self.front_tendency[runners]
        pace_pressure = float(np.clip(front.sum() / n * 1.6 - 0.5, -0.5, 1.2))
        # ハイペースなら差し・追込有利、スローなら逃げ・先行有利
        pace_fit = (0.5 - front) * pace_pressure * 2.2

        weight_effect = -0.09 * (weight_carried - 55.0)
        day_noise = rng.normal(0, 1.0, n) * self.consistency[runners] * cfg.outcome_noise

        perf = (
            ability
            + surf_fit
            + dist_fit
            + going_fit
            + age_effect
            + self.jockey_skill[jk]
            + self.trainer_skill[tr]
            + weight_effect
            + draw_effect
            + pace_fit
            + form
            + rest_effect
            + day_noise
            + rng.normal(0, 1.0, n) * self.consistency[runners]
        )

        # --- 着順・タイム ---
        order = np.argsort(-perf)
        finish_pos = np.empty(n, dtype=int)
        finish_pos[order] = np.arange(1, n + 1)

        base_time = distance / 1000.0 * (59.0 if surface == "turf" else 61.5) + 2.5
        base_time += GOING_PENALTY[going] * distance / 1000.0
        base_time += 0.35 * (9 - race_class) * distance / 1000.0  # 下級条件は時計が掛かる
        finish_time = np.round(base_time - 0.22 * perf + rng.normal(0, 0.08, n), 1)

        # 4角通過順: 脚質 + ノイズ
        corner_score = -front + rng.normal(0, 0.35, n)
        corner_pos = np.empty(n, dtype=int)
        corner_pos[np.argsort(corner_score)] = np.arange(1, n + 1)
        # 上がり3ハロン: 差し馬ほど速い（着順が良いほど速い）
        last_3f = np.round(
            35.5 - 0.30 * perf + 1.2 * (front - 0.5) + rng.normal(0, 0.25, n), 1
        )

        # --- 市場（オッズ） ---
        # 市場は真の期待値をノイズ付きで観測するが、展開適性は一部しか織り込まない
        market_signal = (
            perf
            - day_noise  # 当日の偶然は市場にも知り得ない
            - cfg.market_pace_blindness * pace_fit
            + rng.normal(0, cfg.market_noise, n)
        )
        # 市場の温度は「当日の偶然のばらつき + 市場自身の推定誤差」で決まる。
        # これを無視すると市場が過信気味になり、人気馬のオッズが不自然に低くなる。
        outcome_sd = 0.7 * cfg.outcome_noise
        market_temp = float(np.sqrt(outcome_sd**2 + cfg.market_noise**2))
        p_market = _softmax(market_signal / market_temp)
        # 本命-大穴バイアス: 大穴が買われすぎて、実力よりオッズが低くなる
        # （= 市場のインプライド確率が実際の勝率より高くなる）
        p_market = p_market**0.94
        p_market /= p_market.sum()
        odds = np.round(np.clip(1.0 / (p_market * cfg.overround), 1.1, 999.0), 1)
        popularity = np.empty(n, dtype=int)
        popularity[np.argsort(odds)] = np.arange(1, n + 1)

        for i, h in enumerate(runners):
            st = states[h]
            st.last_date = date
            st.starts += 1

        return [
            {
                "race_id": race_id,
                "date": date.strftime("%Y-%m-%d"),
                "venue": venue,
                "surface": surface,
                "distance": distance,
                "going": going,
                "race_class": race_class,
                "horse_id": f"H{runners[i]:05d}",
                "horse_name": f"Horse{runners[i]:05d}",
                "sex": self.sex[runners[i]],
                "age": int(age[i]),
                "jockey_id": f"J{jk[i]:03d}",
                "trainer_id": f"T{tr[i]:03d}",
                "sire_id": f"S{self.horse_sire[runners[i]]:03d}",
                "draw": int(draw[i]),
                "weight_carried": float(weight_carried[i]),
                "body_weight": float(body_weight[i]),
                "odds": float(odds[i]),
                "popularity": int(popularity[i]),
                "finish_pos": int(finish_pos[i]),
                "finish_time": float(finish_time[i]),
                "corner_pos": int(corner_pos[i]),
                "last_3f": float(last_3f[i]),
            }
            for i in range(n)
        ]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _race_dates(start: str, n_days: int) -> list[pd.Timestamp]:
    """土日開催を n_days 日分。"""
    cur = pd.Timestamp(start)
    out: list[pd.Timestamp] = []
    while len(out) < n_days:
        if cur.dayofweek in (5, 6):
            out.append(cur)
        cur += pd.Timedelta(days=1)
    return out


def generate_dataset(config: GeneratorConfig | None = None) -> pd.DataFrame:
    return SyntheticRaceGenerator(config or GeneratorConfig()).generate()
