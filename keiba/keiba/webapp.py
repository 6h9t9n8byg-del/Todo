"""予測結果を単一HTMLのWebアプリとして書き出す。

スマートフォンで見ることを想定し、外部リソースを一切参照しない1ファイルにまとめる。

ブラウザ側では ``p_model``（モデル単独の勝率）とオッズから
「控除率除去 → 市場とのブレンド → 期待値・ケリー」を再計算する。そのため
利用者がオッズを入れ替えると予測がその場で更新される。計算式は
:mod:`keiba.market` と同一で、``_check_reproducible`` で一致を検証している。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import market as market_mod
from .metrics import calibration_table, summarize

TEMPLATE = Path(__file__).parent / "assets" / "app.html"
PLACEHOLDER = "/*__KEIBA_DATA__*/"

VENUE_JA = {
    "Tokyo": "東京", "Nakayama": "中山", "Hanshin": "阪神", "Kyoto": "京都",
    "Chukyo": "中京", "Kokura": "小倉", "Niigata": "新潟", "Sapporo": "札幌",
}
SURFACE_JA = {"turf": "芝", "dirt": "ダ"}
GOING_JA = {"firm": "良", "good": "稍重", "yielding": "重", "soft": "不良"}


def bracket_of(draw: int, n_runners: int) -> int:
    """馬番から枠番を求める（JRAの割り当て順に準拠）。

    8頭以下は馬番＝枠番。9頭以上は各枠1頭ずつ配ったうえで、余りを外枠から
    順に足していく（例: 12頭なら5〜8枠が2頭ずつ）。
    """
    if n_runners <= 8:
        return int(draw)
    sizes = [n_runners // 8] * 8
    for i in range(n_runners % 8):
        sizes[7 - i] += 1
    cursor = 0
    for bracket, size in enumerate(sizes, start=1):
        cursor += size
        if draw <= cursor:
            return bracket
    return 8


def _check_reproducible(preds: pd.DataFrame) -> float:
    """ブラウザ側の計算式で ``p_final`` を再現できるか確認し、最大誤差を返す。

    ブレンド重みは検証期間（フォールド）ごとに異なるため、重み単位で確認する。
    """
    worst = 0.0
    for _, grp in preds.groupby("blend_weight", sort=False):
        p_market = market_mod.implied_probabilities(grp["odds"], grp["race_id"], "power")
        p_final = market_mod.blend(
            grp["p_model"].to_numpy(), p_market, float(grp["blend_weight"].iloc[0]), grp["race_id"]
        )
        worst = max(worst, float(np.abs(p_final - grp["p_final"].to_numpy()).max()))
    return worst


def build_payload(
    preds: pd.DataFrame,
    *,
    n_races: int | None = 120,
    metrics_source: pd.DataFrame | None = None,
) -> dict:
    """予測結果からアプリに埋め込む JSON を組み立てる。

    Parameters
    ----------
    preds
        アプリに表示するレース（``Predictor.predict`` の出力）。
    n_races
        表示するレース数の上限。日付の新しい順に選ぶ。
    metrics_source
        検証タブに出す集計の元データ。省略時は ``preds`` 自身。
        バックテスト全期間を渡すと、より信頼できる集計になる。
    """
    preds = preds.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    if "blend_weight" not in preds:
        preds["blend_weight"] = 1.0

    if n_races is not None:
        latest = (
            preds[["race_id", "date"]].drop_duplicates()
            .sort_values("date").tail(n_races)["race_id"]
        )
        preds = preds[preds["race_id"].isin(set(latest))]

    mismatch = _check_reproducible(preds)
    if mismatch > 5e-3:
        raise ValueError(
            f"ブラウザ側の計算式で p_final を再現できません（最大誤差 {mismatch:.4f}）。"
            "アイソトニック校正が使われたレースが含まれている可能性があります。"
        )

    races = []
    race_no: dict[tuple, int] = {}
    for race_id, grp in preds.sort_values(["date", "race_id", "draw"]).groupby(
        "race_id", sort=False
    ):
        head = grp.iloc[0]
        n = int(head["n_runners"])
        # 同じ日・同じ競馬場の中での通し番号（第Nレース）
        day_key = (head["date"], head["venue"])
        race_no[day_key] = race_no.get(day_key, 0) + 1
        races.append({
            "id": str(race_id),
            "no": race_no[day_key],
            "date": head["date"].strftime("%Y-%m-%d"),
            "venue": VENUE_JA.get(str(head["venue"]), str(head["venue"])),
            "surface": SURFACE_JA.get(str(head["surface"]), str(head["surface"])),
            "distance": int(head["distance"]),
            "going": GOING_JA.get(str(head["going"]), str(head["going"])),
            "cls": int(head["race_class"]),
            "n": n,
            "w": round(float(head["blend_weight"]), 4),
            "runners": [
                {
                    "draw": int(r["draw"]),
                    "waku": bracket_of(int(r["draw"]), n),
                    "name": str(r["horse_name"]),
                    "jockey": str(r["jockey_id"]),
                    "odds": round(float(r["odds"]), 1),
                    "pm": round(float(r["p_model"]), 6),
                    "pos": None if pd.isna(r["finish_pos"]) else int(r["finish_pos"]),
                }
                for _, r in grp.sort_values("draw").iterrows()
            ],
        })

    src = metrics_source if metrics_source is not None else preds
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "blend_weight": round(float(preds["blend_weight"].iloc[-1]), 4),
        "races": races,
        "validation": _validation_block(src),
        "real": build_real_block(),
    }


def build_real_block() -> dict | None:
    """実在馬のレーティングをアプリに埋め込む形にする。

    共分散のコレスキー分解も一緒に渡す。ブラウザ側で θ の事後分布から標本を取り、
    「記録が1走しかない馬の勝率を過信しない」計算を再現するため。
    """
    from .ratings import fit_ratings, is_rest
    from .realdata import (
        check_consistency, load_horses, load_races, load_upcoming, races_for_rating,
    )

    try:
        races, race_meta = load_races()
        horses, horse_meta = load_horses()
        upcoming, up_meta = load_upcoming()
    except FileNotFoundError:
        return None

    problems = check_consistency(races, horses, upcoming)
    if problems:
        raise ValueError("実在データに矛盾があります: " + "; ".join(problems[:3]))

    ratings = fit_ratings(races_for_rating(races), as_of=datetime.now().date())

    # 戦績のある馬を先に、記録の無い馬をそのあとに置く。記録の無い馬は事前分布
    # （θ=0、分散 prior_sd²、他馬とは独立）なので、共分散は対角ブロックで足りる。
    rated = [n for n in ratings.names if not is_rest(n)]
    unrated = [h.name for h in horses if h.name not in set(rated)]
    names = rated + unrated

    idx = np.array([ratings.index(n) for n in rated], dtype=int)
    cov = np.zeros((len(names), len(names)))
    if len(rated):
        cov[np.ix_(range(len(rated)), range(len(rated)))] = ratings.cov[np.ix_(idx, idx)]
    prior_var = ratings.config.prior_sd**2
    for i in range(len(rated), len(names)):
        cov[i, i] = prior_var
    chol = np.linalg.cholesky(cov + 1e-9 * np.eye(len(names)))

    theta = np.zeros(len(names))
    theta[:len(rated)] = ratings.theta[idx]
    se = np.zeros(len(names))
    se[:len(rated)] = ratings.se[idx]
    se[len(rated):] = ratings.config.prior_sd
    starts = np.zeros(len(names), dtype=int)
    starts[:len(rated)] = ratings.starts[idx].astype(int)

    by_name = {h.name: h for h in horses}
    return {
        "collected": str(race_meta.get("collected") or horse_meta.get("collected") or ""),
        "names": names,
        "theta": [round(float(v), 6) for v in theta],
        "chol": [[round(float(v), 6) for v in row] for row in chol],
        "rest": {n: round(float(ratings.theta[i]), 6)
                 for i, n in enumerate(ratings.names) if is_rest(n)},
        "horses": [
            {
                "name": name,
                "sex": by_name[name].sex,
                "foaled": by_name[name].foaled,
                "status": by_name[name].status,
                "surface": by_name[name].surface,
                "note": by_name[name].note,
                "source": (by_name[name].sources or [None])[0],
                "se": round(float(se[i]), 4),
                "starts": int(starts[i]),
            }
            for i, name in enumerate(names) if name in by_name
        ],
        "races": [
            {
                "name": r.name,
                "date": str(r.date) if r.date else None,
                "venue": r.venue,
                "surface": SURFACE_JA.get(str(r.surface), str(r.surface)),
                "distance": r.distance,
                "grade": r.grade,
                "result": r.result,
                "source": (r.sources or [None])[0],
            }
            for r in races
        ],
        "upcoming": [
            {
                "id": u.id,
                "name": u.name,
                "grade": u.grade,
                "date": str(u.date) if u.date else None,
                "venue": u.venue,
                "race_no": u.race_no,
                "post_time": u.post_time,
                "surface": SURFACE_JA.get(str(u.surface), str(u.surface)),
                "distance": u.distance,
                "conditions": u.conditions,
                "full_gate": u.full_gate,
                "partial": u.entries_partial,
                "entries": u.entries,
                "known": u.coverage(set(rated))[0],
                "note": u.note,
                "source": (u.sources or [None])[0],
            }
            for u in upcoming
        ],
        "upcoming_note": up_meta.get("note", ""),
    }


def _validation_block(src: pd.DataFrame) -> dict:
    """検証タブに出す集計（精度・キャリブレーション・戦略グリッド）。"""
    from .backtest import strategy_grid

    y = src["y_win"].to_numpy()
    races = src["race_id"]
    accuracy = {
        key: summarize(src[col].to_numpy(), y, races)
        for key, col in [("model", "p_model"), ("market", "p_market"), ("final", "p_final")]
    }
    cal = calibration_table(src["p_final"].to_numpy(), y)
    grid = strategy_grid(src)
    return {
        "period": [str(src["date"].min())[:10], str(src["date"].max())[:10]],
        "accuracy": accuracy,
        "calibration": [
            {"predicted": round(r.predicted, 4), "actual": round(r.actual, 4), "n": int(r.n)}
            for r in cal.itertuples()
        ],
        "strategy": [
            {
                "threshold": float(r.ev_threshold),
                "bets": int(r.n_bets),
                "hit": None if pd.isna(getattr(r, "hit_rate", np.nan)) else round(r.hit_rate, 4),
                "roi": None if pd.isna(getattr(r, "roi", np.nan)) else round(r.roi, 4),
                "lo": None if pd.isna(getattr(r, "roi_lo", np.nan)) else round(r.roi_lo, 4),
                "hi": None if pd.isna(getattr(r, "roi_hi", np.nan)) else round(r.roi_hi, 4),
            }
            for r in grid.itertuples()
        ],
    }


DOCUMENT = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>勝率手帖</title>
</head>
<body>
{fragment}
</body>
</html>
"""


def render_fragment(payload: dict) -> str:
    """テンプレートに JSON を差し込む（head/body を含まない断片）。"""
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"テンプレートに {PLACEHOLDER} がありません")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # データ中に </script> が現れてもHTMLが壊れないようにする
    data = data.replace("</", "<\\/")
    return template.replace(PLACEHOLDER, data)


def write_app(payload: dict, path: str, *, fragment_only: bool = False) -> int:
    """単一HTMLとして書き出し、バイト数を返す。

    ``fragment_only`` を立てると head/body を含まない断片を出力する
    （外側の雛形を用意するホスティングに貼る場合用）。
    """
    body = render_fragment(payload)
    html = body if fragment_only else DOCUMENT.format(fragment=body)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return len(html.encode("utf-8"))
