"""予測結果を HTML の予想紙として出力する。"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 20px 64px; font-family: "Hiragino Sans", "Noto Sans JP", system-ui, sans-serif;
       background: #f5f5f2; color: #1d1d1b; }
h1 { font-size: 20px; letter-spacing: .08em; margin: 0 0 4px; }
.sub { color: #6f6f68; font-size: 13px; margin-bottom: 28px; }
.race { background: #fff; border: 1px solid #e3e3dd; border-radius: 10px; padding: 18px 20px;
        margin-bottom: 22px; max-width: 1080px; }
.race h2 { font-size: 16px; margin: 0 0 2px; }
.race .meta { color: #6f6f68; font-size: 12px; margin-bottom: 14px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; min-width: 720px; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid #eeeee8; white-space: nowrap; }
th { font-weight: 600; font-size: 11px; color: #6f6f68; text-align: right; border-bottom: 1px solid #ddd; }
th:nth-child(-n+3), td:nth-child(-n+3) { text-align: left; }
tr.pick { background: #fdf6e3; }
tr.bet td { font-weight: 600; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 11px; }
.badge.buy { background: #1d7a4c; color: #fff; }
.badge.hold { background: #eaeae4; color: #6f6f68; }
.num { font-variant-numeric: tabular-nums; }
.note { max-width: 1080px; color: #6f6f68; font-size: 12px; line-height: 1.7; }
@media (prefers-color-scheme: dark) {
  body { background: #14140f; color: #ececdf; }
  .race { background: #1d1d18; border-color: #33332b; }
  th { color: #9a9a8e; border-bottom-color: #33332b; }
  td { border-bottom-color: #26261f; }
  tr.pick { background: #2a2618; }
  .badge.hold { background: #2a2a22; color: #9a9a8e; }
  .note, .sub, .race .meta { color: #9a9a8e; }
}
"""

_SURFACE_JA = {"turf": "芝", "dirt": "ダート"}
_GOING_JA = {"firm": "良", "good": "稍重", "yielding": "重", "soft": "不良"}


def render_report(preds: pd.DataFrame, *, ev_threshold: float = 0.15) -> str:
    """予測 DataFrame から HTML 文字列を作る。"""
    parts = [
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>予想シート</title><style>", _CSS, "</style></head><body>",
        "<h1>予想シート</h1>",
        f"<p class='sub'>{len(preds):,} 頭 / {preds['race_id'].nunique():,} レース"
        f"　期待値しきい値 +{ev_threshold:.0%}</p>",
    ]

    for race_id, grp in preds.groupby("race_id", sort=False):
        head = grp.iloc[0]
        surface = _SURFACE_JA.get(str(head.get("surface")), str(head.get("surface")))
        going = _GOING_JA.get(str(head.get("going")), str(head.get("going")))
        date = pd.to_datetime(head.get("date")).strftime("%Y-%m-%d")
        parts.append("<section class='race'>")
        parts.append(f"<h2>{html.escape(str(race_id))}</h2>")
        parts.append(
            f"<p class='meta'>{date}　{html.escape(str(head.get('venue')))}　"
            f"{surface}{int(head.get('distance'))}m　馬場:{going}　"
            f"クラス:{head.get('race_class')}　{int(head.get('n_runners'))}頭</p>"
        )
        parts.append("<div class='scroll'><table><thead><tr>"
                     "<th>馬番</th><th>馬名</th><th>騎手</th><th>オッズ</th>"
                     "<th>予測勝率</th><th>市場勝率</th><th>期待値</th><th>推奨</th>"
                     "</tr></thead><tbody>")
        ordered = grp.sort_values("p_final", ascending=False)
        for i, r in enumerate(ordered.itertuples()):
            ev = float(r.ev) if pd.notna(r.ev) else float("nan")
            buy = pd.notna(ev) and ev > ev_threshold
            classes = " ".join(filter(None, ["pick" if i == 0 else "", "bet" if buy else ""]))
            badge = ("<span class='badge buy'>買い</span>" if buy
                     else "<span class='badge hold'>見送り</span>")
            market = f"{r.p_market:.1%}" if pd.notna(r.p_market) else "—"
            odds = f"{r.odds:.1f}" if pd.notna(r.odds) else "—"
            parts.append(
                f"<tr class='{classes}'>"
                f"<td class='num'>{int(r.draw)}</td>"
                f"<td>{html.escape(str(r.horse_name))}</td>"
                f"<td>{html.escape(str(r.jockey_id))}</td>"
                f"<td class='num'>{odds}</td>"
                f"<td class='num'>{r.p_final:.1%}</td>"
                f"<td class='num'>{market}</td>"
                f"<td class='num'>{ev:+.1%}</td>"
                f"<td>{badge}</td></tr>"
            )
        parts.append("</tbody></table></div></section>")

    parts.append(
        "<p class='note'>予測勝率はモデルと市場オッズをブレンドした推定値です。"
        "期待値は「予測勝率 × 単勝オッズ − 1」で、プラスであっても的中を保証するものでは"
        "ありません。控除率のある公営競技では長期的な収支がマイナスになるのが通常です。"
        "余裕資金の範囲で自己責任のもとご利用ください。</p>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def write_report(preds: pd.DataFrame, path: str, *, ev_threshold: float = 0.15) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(preds, ev_threshold=ev_threshold), encoding="utf-8")
