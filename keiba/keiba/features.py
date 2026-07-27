"""特徴量エンジニアリング（リーク防止を最優先に設計）。

原則
----
* **過去のみ参照**: 馬・騎手・調教師・種牡馬などの集計は、必ず「その行より前の行」
  だけを使う（exclusive cumulative）。行はレース日→レースID順に並んでいるため、
  同日の先行レースは参照可能（実運用でも既知の情報）だが、同一レース内の他行や
  未来の行は決して参照しない。
* **ターゲットエンコーディングも時系列**: 騎手成績などは K-fold ではなく
  「その時点までの成績」を事前分布にスムージングして使う。学習時と推論時で
  同じ手続きになるため、分布のズレが起きない。
* **相対化**: 競馬は同一レース内の相対比較なので、主要特徴量はレース内 z-score も
  併せて与える（絶対値だけではメンバー構成を表現できない）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import GOING_ORDER

# 事前分布へのスムージング強度（件数がこの値と同程度になるまで事前分布寄り）
SMOOTH_M = 25.0

CATEGORICAL_FEATURES = ["venue", "surface", "going", "sex", "dist_bucket"]


# --------------------------------------------------------------------------- 汎用
def _exclusive_past_mean(
    df: pd.DataFrame, keys: list[str], value: pd.Series, prior, m: float = SMOOTH_M
) -> tuple[pd.Series, pd.Series]:
    """キー単位で「現在行より前の行」の平均をスムージングして返す。

    ``prior`` はスカラーでも行ごとのベクトルでもよい。データ全体の平均を事前分布に
    使うと未来の情報が混ざるため、ここでは理論値（相対着順なら 0.5、勝率なら
    1/出走頭数）を使う。

    Returns
    -------
    (smoothed_mean, prior_count)
    """
    prior = np.asarray(prior, dtype=float)
    v = value.astype(float)
    tmp = pd.DataFrame({"_v": v.fillna(0.0), "_m": v.notna().astype(float)})
    for k in keys:
        tmp[k] = df[k].values
    grp = tmp.groupby(keys, dropna=False, sort=False)
    csum = grp["_v"].cumsum() - tmp["_v"]
    ccnt = grp["_m"].cumsum() - tmp["_m"]
    smoothed = (csum + prior * m) / (ccnt + m)
    return smoothed, ccnt


def _horse_shift_roll(
    df: pd.DataFrame, col: str, window: int | None, *, how: str = "mean"
) -> pd.Series:
    """馬ごとに「前走まで」の移動集計。window=None で全走（expanding）。"""
    def _f(s: pd.Series) -> pd.Series:
        s = s.shift(1)
        r = s.expanding(min_periods=1) if window is None else s.rolling(window, min_periods=1)
        return getattr(r, how)()

    return df.groupby("horse_id", sort=False, observed=True)[col].transform(_f)


def _race_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("race_id", sort=False, observed=True)[col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, np.nan)
    return (df[col] - mean) / std


def _dist_bucket(distance: pd.Series) -> pd.Series:
    return pd.cut(
        distance,
        bins=[0, 1400, 1800, 2200, 10_000],
        labels=["sprint", "mile", "middle", "long"],
    ).astype("string")


# ------------------------------------------------------------------ スピード指数
def _speed_index(df: pd.DataFrame) -> pd.Series:
    """タイムを「同条件の平均との差」に変換した指数（50 が平均、10 が 1SD）。

    基準タイムは *過去のレースのみ* から推定する。条件の細かいキーから順に
    フォールバックし、データが薄いキーは上位キーで代替する。
    """
    race = (
        df.groupby("race_id", sort=False)
        .agg(
            date=("date", "first"),
            venue=("venue", "first"),
            surface=("surface", "first"),
            distance=("distance", "first"),
            going=("going", "first"),
            race_class=("race_class", "first"),
            mean_time=("finish_time", "mean"),
        )
        .reset_index()
        .sort_values(["date", "race_id"])
    )
    race["class_bucket"] = pd.cut(
        race["race_class"], bins=[0, 3, 6, 9], labels=["low", "mid", "high"]
    ).astype("string")

    key_sets = [
        ["venue", "surface", "distance", "going", "class_bucket"],
        ["surface", "distance", "going", "class_bucket"],
        ["surface", "distance", "class_bucket"],
        ["surface", "distance"],
    ]
    base = pd.Series(np.nan, index=race.index)
    std = pd.Series(np.nan, index=race.index)
    for keys in key_sets:
        grp = race.groupby(keys, dropna=False, sort=False)["mean_time"]
        cnt = grp.cumcount()
        mean_k = grp.transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
        std_k = grp.transform(lambda s: s.shift(1).expanding(min_periods=8).std())
        ok = cnt >= 3
        base = base.where(base.notna(), mean_k.where(ok))
        std = std.where(std.notna(), std_k.where(ok))

    # 最終フォールバック: 距離に比例した経験的ばらつき
    fallback_std = race["distance"] / 1000.0 * 0.9
    std = std.fillna(fallback_std).clip(lower=0.15)
    race["base_time"] = base
    race["base_std"] = std

    merged = df[["race_id", "finish_time"]].merge(
        race[["race_id", "base_time", "base_std"]], on="race_id", how="left"
    )
    si = 50.0 + 10.0 * (merged["base_time"] - merged["finish_time"]) / merged["base_std"]
    si.index = df.index
    return si.clip(lower=0.0, upper=120.0)


# --------------------------------------------------------------------------- 本体
@dataclass
class FeatureBuilder:
    """履歴 DataFrame から学習・推論用の特徴量行列を作る。

    ``build`` は履歴と予測対象を **同じ DataFrame** に含めて呼ぶ。予測対象行は
    結果列が NaN でよく、その行の特徴量は過去行だけから作られる。
    """

    smooth_m: float = SMOOTH_M

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["date", "race_id", "draw"]).reset_index(drop=True).copy()

        n = df["n_runners"].astype(float)
        pos = df["finish_pos"].astype(float)
        df["rel_finish"] = np.where(n > 1, 1.0 - (pos - 1.0) / (n - 1.0), np.nan)
        df["is_win"] = (pos == 1).astype(float).where(pos.notna())
        df["is_top3"] = (pos <= 3).astype(float).where(pos.notna())
        df["speed_index"] = _speed_index(df)
        df["corner_ratio"] = df["corner_pos"].astype(float) / n
        df["front_score"] = 1.0 - df["corner_ratio"]
        df["last_3f_rel"] = df.groupby("race_id", sort=False)["last_3f"].transform(
            lambda s: s.mean() - s
        )
        df["dist_bucket"] = _dist_bucket(df["distance"])
        df["going_ord"] = df["going"].map(GOING_ORDER).astype(float)
        df["draw_ratio"] = (df["draw"].astype(float) - 1.0) / (n - 1.0).replace(0, np.nan)
        df["is_turf"] = (df["surface"] == "turf").astype(float)
        df["month"] = df["date"].dt.month
        df["log_distance"] = np.log(df["distance"].astype(float))

        # 事前分布は理論値を使う（データ全体の平均を使うと未来の情報が漏れる）。
        # 相対着順は定義上どのレースでも平均 0.5、勝率は 1/出走頭数、
        # スピード指数は 50 を中心にするよう構成してある。
        prior_rel = 0.5
        prior_win = (1.0 / n).to_numpy()
        prior_top3 = np.minimum(3.0 / n, 1.0)
        prior_si = 50.0

        out = df

        # ---------------- 馬の実績 ----------------
        out["h_starts"] = out.groupby("horse_id", sort=False).cumcount()
        out["h_rel_avg"], _ = _exclusive_past_mean(out, ["horse_id"], out["rel_finish"], prior_rel, 4)
        out["h_win_rate"], _ = _exclusive_past_mean(out, ["horse_id"], out["is_win"], prior_win, 8)
        out["h_top3_rate"], _ = _exclusive_past_mean(out, ["horse_id"], out["is_top3"], prior_top3, 8)
        out["h_rel_last3"] = _horse_shift_roll(out, "rel_finish", 3)
        out["h_si_avg5"] = _horse_shift_roll(out, "speed_index", 5)
        out["h_si_avg3"] = _horse_shift_roll(out, "speed_index", 3)
        out["h_si_best"] = _horse_shift_roll(out, "speed_index", None, how="max")
        out["h_si_last"] = out.groupby("horse_id", sort=False)["speed_index"].shift(1)
        out["h_si_std"] = _horse_shift_roll(out, "speed_index", 5, how="std")
        out["h_si_trend"] = out["h_si_last"] - out["h_si_avg5"]
        out["h_agari_avg"] = _horse_shift_roll(out, "last_3f_rel", 5)
        out["h_front_score"] = _horse_shift_roll(out, "front_score", 5)
        out["h_front_std"] = _horse_shift_roll(out, "front_score", 5, how="std")

        # ---------------- ローテーション・当日条件 ----------------
        last_date = out.groupby("horse_id", sort=False)["date"].shift(1)
        out["h_rest_days"] = (out["date"] - last_date).dt.days
        out["h_log_rest"] = np.log(out["h_rest_days"].clip(lower=1))
        out["h_layoff"] = (out["h_rest_days"] > 120).astype(float)
        out["h_bw_diff"] = out["body_weight"] - out.groupby("horse_id", sort=False)["body_weight"].shift(1)
        out["h_wc_diff"] = out["weight_carried"] - out.groupby("horse_id", sort=False)["weight_carried"].shift(1)
        out["h_class_avg"] = _horse_shift_roll(out, "race_class", 5)
        out["h_class_change"] = out["race_class"] - out["h_class_avg"]
        out["h_class_best"] = _horse_shift_roll(out, "race_class", None, how="max")

        # ---------------- 条件適性（コース・距離・馬場） ----------------
        out["h_course_rel"], out["h_course_n"] = _exclusive_past_mean(
            out, ["horse_id", "venue", "surface", "dist_bucket"], out["rel_finish"], prior_rel, 3
        )
        out["h_surface_si"], _ = _exclusive_past_mean(
            out, ["horse_id", "surface"], out["speed_index"], prior_si, 3
        )
        out["h_dist_si"], _ = _exclusive_past_mean(
            out, ["horse_id", "dist_bucket"], out["speed_index"], prior_si, 3
        )
        out["h_going_rel"], _ = _exclusive_past_mean(
            out, ["horse_id", "going"], out["rel_finish"], prior_rel, 3
        )

        # ---------------- 騎手・調教師・種牡馬 ----------------
        out["j_win_rate"], out["j_n"] = _exclusive_past_mean(
            out, ["jockey_id"], out["is_win"], prior_win, 60
        )
        out["j_rel"], _ = _exclusive_past_mean(out, ["jockey_id"], out["rel_finish"], prior_rel, 60)
        out["j_venue_rel"], _ = _exclusive_past_mean(
            out, ["jockey_id", "venue"], out["rel_finish"], prior_rel, 30
        )
        out["t_win_rate"], _ = _exclusive_past_mean(out, ["trainer_id"], out["is_win"], prior_win, 60)
        out["t_rel"], _ = _exclusive_past_mean(out, ["trainer_id"], out["rel_finish"], prior_rel, 60)
        out["s_rel"], _ = _exclusive_past_mean(out, ["sire_id"], out["rel_finish"], prior_rel, 80)
        out["s_surface_si"], _ = _exclusive_past_mean(
            out, ["sire_id", "surface"], out["speed_index"], prior_si, 60
        )
        out["s_dist_si"], _ = _exclusive_past_mean(
            out, ["sire_id", "dist_bucket"], out["speed_index"], prior_si, 60
        )
        out["combo_rel"], out["combo_n"] = _exclusive_past_mean(
            out, ["horse_id", "jockey_id"], out["rel_finish"], prior_rel, 3
        )

        # ---------------- コースバイアス（枠・脚質） ----------------
        out["_draw_bucket"] = pd.cut(
            out["draw_ratio"], bins=[-0.01, 0.33, 0.66, 1.01], labels=["in", "mid", "out"]
        ).astype("string")
        out["draw_bias"], _ = _exclusive_past_mean(
            out, ["venue", "surface", "dist_bucket", "_draw_bucket"], out["rel_finish"], prior_rel, 100
        )
        out["_style_bucket"] = pd.cut(
            out["h_front_score"], bins=[-0.01, 0.4, 0.6, 1.01], labels=["closer", "mid", "front"]
        ).astype("string")
        out["style_bias"], _ = _exclusive_past_mean(
            out, ["venue", "surface", "dist_bucket", "_style_bucket"], out["rel_finish"], prior_rel, 100
        )

        # ---------------- 展開（ペース）予測 ----------------
        race_grp = out.groupby("race_id", sort=False)
        out["pace_pressure"] = race_grp["h_front_score"].transform("mean")
        out["pace_top3"] = race_grp["h_front_score"].transform(
            lambda s: s.nlargest(min(3, len(s))).mean()
        )
        out["pace_fit"] = (0.5 - out["h_front_score"]) * (out["pace_pressure"] - 0.5) * 4.0
        out["front_rank"] = race_grp["h_front_score"].rank(ascending=False, method="average")

        # ---------------- メンバー相対（レース内の力関係） ----------------
        out["field_si_mean"] = race_grp["h_si_avg5"].transform("mean")
        out["field_si_max"] = race_grp["h_si_avg5"].transform("max")
        out["si_vs_field"] = out["h_si_avg5"] - out["field_si_mean"]
        out["si_vs_best"] = out["h_si_avg5"] - out["field_si_max"]

        for col in [
            "h_si_avg5",
            "h_si_best",
            "h_si_last",
            "h_rel_avg",
            "h_rel_last3",
            "h_win_rate",
            "j_rel",
            "t_rel",
            "s_rel",
            "weight_carried",
            "h_rest_days",
            "age",
            "body_weight",
            "h_class_avg",
            "h_agari_avg",
        ]:
            out[f"z_{col}"] = _race_zscore(out, col)

        out = out.drop(columns=["_draw_bucket", "_style_bucket"])
        return out

    # ------------------------------------------------------------------ 列一覧
    @property
    def numeric_features(self) -> list[str]:
        base = [
            "n_runners", "distance", "log_distance", "race_class", "going_ord", "is_turf",
            "month", "age", "draw", "draw_ratio", "weight_carried", "body_weight",
            "h_starts", "h_rel_avg", "h_win_rate", "h_top3_rate", "h_rel_last3",
            "h_si_avg5", "h_si_avg3", "h_si_best", "h_si_last", "h_si_std", "h_si_trend",
            "h_agari_avg", "h_front_score", "h_front_std",
            "h_rest_days", "h_log_rest", "h_layoff", "h_bw_diff", "h_wc_diff",
            "h_class_avg", "h_class_change", "h_class_best",
            "h_course_rel", "h_course_n", "h_surface_si", "h_dist_si", "h_going_rel",
            "j_win_rate", "j_n", "j_rel", "j_venue_rel", "t_win_rate", "t_rel",
            "s_rel", "s_surface_si", "s_dist_si", "combo_rel", "combo_n",
            "draw_bias", "style_bias",
            "pace_pressure", "pace_top3", "pace_fit", "front_rank",
            "field_si_mean", "field_si_max", "si_vs_field", "si_vs_best",
        ]
        z = [
            "z_h_si_avg5", "z_h_si_best", "z_h_si_last", "z_h_rel_avg", "z_h_rel_last3",
            "z_h_win_rate", "z_j_rel", "z_t_rel", "z_s_rel", "z_weight_carried",
            "z_h_rest_days", "z_age", "z_body_weight", "z_h_class_avg", "z_h_agari_avg",
        ]
        return base + z

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureBuilder]:
    fb = FeatureBuilder()
    return fb.build(df), fb
