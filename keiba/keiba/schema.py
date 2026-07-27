"""レースデータのスキーマ定義と検証。

1行 = 1頭の出走レコード（horse-race row）。同じ ``race_id`` を持つ行が1レースを構成する。
学習には結果列（着順・タイムなど）が必要だが、予測対象の未来レースでは欠損でよい。
"""

from __future__ import annotations

import pandas as pd

# --- レース単位の列（同一 race_id 内で同じ値） ---
RACE_COLS = [
    "race_id",       # レースID
    "date",          # 開催日 (YYYY-MM-DD)
    "venue",         # 競馬場
    "surface",       # 芝 / ダート  ("turf" / "dirt")
    "distance",      # 距離 (m)
    "going",         # 馬場状態 ("firm"/"good"/"yielding"/"soft")
    "race_class",    # クラス水準 (数値が大きいほど上位。例: 1=未勝利 ... 9=G1)
]

# --- 出走馬単位の列 ---
ENTRY_COLS = [
    "horse_id",
    "horse_name",
    "sex",             # "M"/"F"/"G"
    "age",
    "jockey_id",
    "trainer_id",
    "sire_id",
    "draw",            # 馬番 (1..n_runners)
    "weight_carried",  # 斤量 (kg)
    "body_weight",     # 馬体重 (kg)
    "odds",            # 単勝オッズ（確定前は前日オッズ等。無い場合は NaN 可）
]

# --- 結果列（学習に必要。予測時は NaN で可） ---
RESULT_COLS = [
    "finish_pos",     # 着順 (1が1着)
    "finish_time",    # 走破タイム (秒)
    "corner_pos",     # 4コーナー通過順位
    "last_3f",        # 上がり3ハロン (秒)
]

ALL_COLS = RACE_COLS + ENTRY_COLS + RESULT_COLS

NUMERIC_COLS = [
    "distance",
    "race_class",
    "age",
    "draw",
    "weight_carried",
    "body_weight",
    "odds",
    "finish_pos",
    "finish_time",
    "corner_pos",
    "last_3f",
]

CATEGORICAL_COLS = ["venue", "surface", "going", "sex"]

ID_COLS = ["race_id", "horse_id", "jockey_id", "trainer_id", "sire_id"]

GOING_ORDER = {"firm": 0, "good": 1, "yielding": 2, "soft": 3}


class SchemaError(ValueError):
    """必須列の欠落やレース構造の不整合。"""


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """dtype を揃え、欠損している任意列を補完した DataFrame を返す。"""
    df = df.copy()

    missing = [c for c in RACE_COLS + ["horse_id", "draw"] if c not in df.columns]
    if missing:
        raise SchemaError(f"必須列が存在しません: {missing}")

    for col in ALL_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df["date"] = pd.to_datetime(df["date"])
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ID_COLS + CATEGORICAL_COLS + ["horse_name"]:
        df[col] = df[col].astype("string")

    df["surface"] = df["surface"].str.lower()
    df["going"] = df["going"].str.lower()

    # 出走頭数はデータから導出する（入力を信用しない）
    df["n_runners"] = df.groupby("race_id")["horse_id"].transform("size")

    return df.sort_values(["date", "race_id", "draw"]).reset_index(drop=True)


def validate(df: pd.DataFrame, *, require_results: bool = False) -> None:
    """整合性チェック。問題があれば :class:`SchemaError` を送出する。"""
    if df.empty:
        raise SchemaError("データが空です")

    dup = df.duplicated(["race_id", "horse_id"]).sum()
    if dup:
        raise SchemaError(f"同一レースに同じ馬が {dup} 件重複しています")

    # 1レースの日付は1つでなければならない
    per_race_dates = df.groupby("race_id")["date"].nunique()
    if (per_race_dates > 1).any():
        bad = per_race_dates[per_race_dates > 1].index.tolist()[:5]
        raise SchemaError(f"1レースに複数の日付が存在します: {bad}")

    if require_results:
        finished = df.dropna(subset=["finish_pos"])
        if finished.empty:
            raise SchemaError("学習には finish_pos が必要ですが、全て欠損しています")
        winners = finished[finished["finish_pos"] == 1].groupby("race_id").size()
        if (winners > 1).any():
            bad = winners[winners > 1].index.tolist()[:5]
            raise SchemaError(f"1着が複数存在するレースがあります: {bad}")


def load_csv(path: str, *, require_results: bool = False) -> pd.DataFrame:
    """CSV を読み込み、整形・検証して返す。"""
    df = coerce(pd.read_csv(path))
    validate(df, require_results=require_results)
    return df
