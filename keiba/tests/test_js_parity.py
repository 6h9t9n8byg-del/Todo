"""ブラウザ側(JS)とPython側の計算が一致することを確かめる。

同じ式を2言語で持つ以上、片方を直してもう片方を忘れる事故が起きうる。
テンプレートから該当部分を切り出して node で実行し、Python の結果と突き合わせる。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap

import numpy as np
import pytest

from keiba.tickets import build_proposals, exacta_matrix, top_n_probs, wide_matrix
from keiba.webapp import TEMPLATE

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node が無い環境")

START, END = "// <<TICKET_MATH", "// TICKET_MATH>>"
RATING_START, RATING_END = "// <<RATING_MATH", "// RATING_MATH>>"


def _js_math() -> str:
    src = TEMPLATE.read_text(encoding="utf-8")
    assert START in src and END in src, "テンプレートの目印が見つかりません"
    return src.split(START, 1)[1].split(END, 1)[0]


def _extract(start: str, end: str) -> str:
    src = TEMPLATE.read_text(encoding="utf-8")
    assert start in src and end in src, f"テンプレートに {start} が無い"
    return src.split(start, 1)[1].split(end, 1)[0]


def _node(script: str, payload: dict) -> dict:
    out = subprocess.run(
        ["node", "-e", script],
        env={**os.environ, "KEIBA_INPUT": json.dumps(payload)},
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def _run_node(p, draws, brackets, n) -> dict:
    harness = textwrap.dedent("""
        const input = JSON.parse(process.env.KEIBA_INPUT);
        const stats = orderStats(input.p, input.places);
        const proposals = buildProposals(input.p, input.draws, input.brackets, input.n);
        console.log(JSON.stringify({
          exacta: exactaMatrix(input.p),
          top: stats.top,
          wide: stats.wide,
          proposals: proposals.map(x => ({
            code: x.code, points: x.points, hitProb: x.hitProb,
            expectedHits: x.expectedHits, breakeven: x.breakeven,
            tickets: x.tickets.map(t => ({ legs: t.legs, prob: t.prob }))
          }))
        }));
    """)
    payload = json.dumps({
        "p": list(map(float, p)), "draws": list(map(int, draws)),
        "brackets": list(map(int, brackets)), "n": int(n),
        "places": 3 if n >= 8 else 2,
    })
    out = subprocess.run(
        ["node", "-e", _js_math() + harness],
        env={**os.environ, "KEIBA_INPUT": payload},
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(7)
    n = 12
    p = rng.dirichlet(np.ones(n) * 2.0)
    draws = list(range(1, n + 1))
    brackets = [min(1 + (d - 1) * 8 // n, 8) for d in draws]
    return p, draws, brackets, n


@pytest.fixture(scope="module")
def js(case):
    return _run_node(*case)


def test_exacta_matrix_matches(case, js) -> None:
    p = case[0]
    assert np.allclose(np.array(js["exacta"]), exacta_matrix(p), atol=1e-12)


def test_place_and_wide_probabilities_match(case, js) -> None:
    p = case[0]
    assert np.allclose(np.array(js["top"]), top_n_probs(p, 3), atol=1e-12)
    assert np.allclose(np.array(js["wide"]), wide_matrix(p, 3), atol=1e-12)


def test_proposals_match(case, js) -> None:
    p, draws, brackets, n = case
    expected = build_proposals(p, draws, brackets, n_runners=n)
    assert [x["code"] for x in js["proposals"]] == [x.code for x in expected]

    for got, want in zip(js["proposals"], expected):
        assert got["points"] == want.points, want.code
        assert got["hitProb"] == pytest.approx(want.hit_prob, rel=1e-9), want.code
        assert got["expectedHits"] == pytest.approx(want.expected_hits, rel=1e-9), want.code
        assert got["breakeven"] == pytest.approx(want.breakeven_odds, rel=1e-9), want.code
        assert [t["legs"] for t in got["tickets"]] == [t["legs"] for t in want.tickets], want.code
        for a, b in zip(got["tickets"], want.tickets):
            assert a["prob"] == pytest.approx(b["prob"], rel=1e-9), want.code


def test_small_field_matches(case) -> None:
    """7頭立て（2着まで・枠連なし）でも一致する。"""
    rng = np.random.default_rng(11)
    n = 7
    p = rng.dirichlet(np.ones(n) * 2.0)
    draws = list(range(1, n + 1))
    brackets = draws[:]
    js = _run_node(p, draws, brackets, n)
    expected = build_proposals(p, draws, brackets, n_runners=n)

    assert "wakuren" not in [x["code"] for x in js["proposals"]]
    assert np.allclose(np.array(js["top"]), top_n_probs(p, 2), atol=1e-12)
    for got, want in zip(js["proposals"], expected):
        assert got["hitProb"] == pytest.approx(want.hit_prob, rel=1e-9), want.code


# ------------------------------------------------------- 実在馬のレーティング
def _real_ratings():
    from datetime import date

    from keiba.ratings import fit_ratings, is_rest
    from keiba.realdata import load_races, races_for_rating

    races, _ = load_races()
    ratings = fit_ratings(races_for_rating(races), as_of=date(2026, 7, 28))
    names = [n for n in ratings.names if not is_rest(n)]
    idx = np.array([ratings.index(n) for n in names])
    cov = ratings.cov[np.ix_(idx, idx)]
    chol = np.linalg.cholesky(cov + 1e-9 * np.eye(len(names)))
    theta = ratings.theta[idx]
    return names, theta, chol, ratings


def _run_rating_node(theta, chol, indices, samples) -> list:
    harness = """
        const i = JSON.parse(process.env.KEIBA_INPUT);
        console.log(JSON.stringify(realWinProbs(i.theta, i.chol, i.indices, i.samples)));
    """
    return _node(_extract(RATING_START, RATING_END) + harness, {
        "theta": [float(v) for v in theta],
        "chol": [[float(v) for v in row] for row in chol],
        "indices": [int(v) for v in indices],
        "samples": int(samples),
    })


def test_rating_point_estimate_matches_python() -> None:
    """標本を取らない場合は、Python と桁まで一致するはず。"""
    names, theta, chol, ratings = _real_ratings()
    indices = [0, 3, 5, 7]
    js = _run_rating_node(theta, chol, indices, 0)
    field = [names[i] for i in indices]
    py = ratings.win_probabilities(field, samples=0)
    assert np.allclose(js, py, atol=1e-12)


def test_rating_sampled_probabilities_agree_within_monte_carlo_error() -> None:
    """乱数は言語ごとに違うので、一致は標本誤差の範囲で確かめる。"""
    names, theta, chol, ratings = _real_ratings()
    indices = list(range(6))
    js = np.array(_run_rating_node(theta, chol, indices, 20000))
    py = ratings.win_probabilities([names[i] for i in indices], samples=60000, seed=5)
    assert js.sum() == pytest.approx(1.0)
    assert np.max(np.abs(js - py)) < 0.01


def test_rating_sampling_shifts_probability_the_same_way_in_both_languages() -> None:
    """不確かさを織り込むと確率がどちらへ動くか、JS と Python で向きが一致する。

    「一様に平らになる」わけではなく、推定が曖昧な馬へ確率が移る。その向きが
    両実装で揃っていることを確かめる（大きさは標本誤差があるので符号で見る）。
    """
    names, theta, chol, ratings = _real_ratings()
    # 記録の多さがばらつく程度に広い顔ぶれを取る（狭いと動きが標本誤差に埋もれる）
    indices = list(range(10))
    field = [names[i] for i in indices]

    js_point = np.array(_run_rating_node(theta, chol, indices, 0))
    js_sampled = np.array(_run_rating_node(theta, chol, indices, 40000))
    py_point = ratings.win_probabilities(field, samples=0)
    py_sampled = ratings.win_probabilities(field, samples=200000, seed=1)

    js_shift, py_shift = js_sampled - js_point, py_sampled - py_point
    moved = np.abs(py_shift) > 0.003          # 標本誤差に埋もれない分だけ見る
    assert moved.sum() >= 3, f"動いた馬が少なすぎる: {py_shift.round(4)}"
    assert np.all(np.sign(js_shift[moved]) == np.sign(py_shift[moved]))
