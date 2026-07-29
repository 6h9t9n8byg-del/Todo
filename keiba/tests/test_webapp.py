"""Webアプリ書き出しのテスト。"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from keiba.webapp import PLACEHOLDER, bracket_of, build_payload, write_app


def test_bracket_matches_jra_allocation() -> None:
    # 8頭以下は馬番＝枠番
    assert [bracket_of(d, 8) for d in range(1, 9)] == list(range(1, 9))
    # 12頭は1〜4枠が1頭、5〜8枠が2頭
    assert [bracket_of(d, 12) for d in range(1, 13)] == [1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    # 16頭は全枠2頭ずつ
    assert [bracket_of(d, 16) for d in range(1, 17)] == [i for i in range(1, 9) for _ in range(2)]
    # どの頭数でも枠は1〜8に収まり、単調非減少
    for n in range(2, 19):
        brackets = [bracket_of(d, n) for d in range(1, n + 1)]
        assert min(brackets) >= 1 and max(brackets) <= 8
        assert brackets == sorted(brackets)


def _fake_preds(n_races: int = 3, size: int = 8, weight: float = 0.2) -> pd.DataFrame:
    from keiba.market import blend, implied_probabilities

    rng = np.random.default_rng(0)
    rows = []
    for r in range(n_races):
        odds = np.round(np.sort(rng.uniform(1.5, 40, size)), 1)
        p_model = rng.dirichlet(np.ones(size) * 3)
        for i in range(size):
            rows.append({
                "race_id": f"R{r}", "date": f"2024-05-0{r + 1}", "venue": "Tokyo",
                "surface": "turf", "distance": 1600, "going": "firm", "race_class": 5,
                "draw": i + 1, "horse_id": f"H{r}{i}", "horse_name": f"Horse{r}{i}",
                "jockey_id": "J01", "odds": odds[i], "n_runners": size,
                "p_model": p_model[i], "finish_pos": i + 1,
                "y_win": 1.0 if i == 0 else 0.0, "blend_weight": weight,
            })
    df = pd.DataFrame(rows)
    df["p_market"] = implied_probabilities(df["odds"], df["race_id"], "power")
    df["p_final"] = blend(df["p_model"].to_numpy(), df["p_market"].to_numpy(),
                          weight, df["race_id"])
    df["ev"] = df["p_final"] * df["odds"] - 1.0
    df["kelly"] = (df["ev"] / (df["odds"] - 1.0)).clip(lower=0)
    return df


def test_payload_structure() -> None:
    payload = build_payload(_fake_preds(), n_races=None)
    assert len(payload["races"]) == 3
    race = payload["races"][0]
    assert set(race) >= {"id", "date", "venue", "surface", "distance", "n", "w", "runners"}
    assert race["venue"] == "東京" and race["surface"] == "芝"
    assert len(race["runners"]) == race["n"]
    assert all(0 <= r["pm"] <= 1 for r in race["runners"])
    assert payload["validation"]["accuracy"]["final"]["races"] == 3


def test_payload_keeps_only_the_latest_races() -> None:
    payload = build_payload(_fake_preds(n_races=3), n_races=2)
    assert [r["date"] for r in payload["races"]] == ["2024-05-02", "2024-05-03"]


def test_payload_rejects_predictions_the_browser_cannot_reproduce() -> None:
    """ブラウザ側の式で再現できない p_final は、黙って表示せず弾く。"""
    preds = _fake_preds()
    preds["p_final"] = preds["p_final"] * 0.5 + 0.05   # 校正が掛かった状態を模す
    with pytest.raises(ValueError, match="再現できません"):
        build_payload(preds, n_races=None)


def test_written_app_is_self_contained(tmp_path) -> None:
    payload = build_payload(_fake_preds(), n_races=None)
    path = tmp_path / "app.html"
    size = write_app(payload, str(path))
    html = path.read_text(encoding="utf-8")

    assert size == len(html.encode("utf-8"))
    assert PLACEHOLDER not in html
    assert html.startswith("<!doctype html>")
    # 外部リソースを一切参照しない
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', html)
    # データが実際に埋め込まれている
    assert "Horse00" in html


def test_fragment_output_has_no_document_wrapper(tmp_path) -> None:
    payload = build_payload(_fake_preds(), n_races=None)
    path = tmp_path / "fragment.html"
    write_app(payload, str(path), fragment_only=True)
    html = path.read_text(encoding="utf-8")
    assert "<!doctype" not in html.lower()
    assert "<body" not in html.lower()
    assert html.lstrip().startswith("<style>")


def test_embedded_json_is_valid() -> None:
    payload = build_payload(_fake_preds(), n_races=None)
    from keiba.webapp import render_fragment

    fragment = render_fragment(payload)
    raw = fragment.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    parsed = json.loads(raw.replace("<\\/", "</"))
    assert len(parsed["races"]) == len(payload["races"])


# ------------------------------------------------------------ 実在馬ブロック
def test_real_block_includes_every_horse_and_a_usable_covariance() -> None:
    from keiba.webapp import build_real_block

    block = build_real_block()
    n = len(block["names"])
    assert n == len(block["horses"]) == len(block["theta"]) == len(block["chol"])

    chol = np.array(block["chol"])
    assert chol.shape == (n, n)
    assert np.allclose(np.triu(chol, 1), 0.0), "下三角のはず"
    assert np.all(np.diag(chol) > 0)

    # 記録の無い馬は θ=0 で、他馬と相関しない（事前分布そのもの）
    blank = [i for i, h in enumerate(block["horses"]) if h["starts"] == 0]
    assert blank, "記録の無い馬が1頭も無いのは不自然"
    for i in blank:
        assert block["theta"][i] == 0.0
        assert np.allclose(chol[i, :i], 0.0)


def test_real_block_reports_how_many_entries_have_records() -> None:
    from keiba.webapp import build_real_block

    block = build_real_block()
    rated = {h["name"] for h in block["horses"] if h["starts"] > 0}
    for race in block["upcoming"]:
        expected = sum(1 for name in race["entries"] if name in rated)
        assert race["known"] == expected, race["name"]
        assert race["known"] <= len(race["entries"])
