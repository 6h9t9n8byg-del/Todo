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


def _js_math() -> str:
    src = TEMPLATE.read_text(encoding="utf-8")
    assert START in src and END in src, "テンプレートの目印が見つかりません"
    return src.split(START, 1)[1].split(END, 1)[0]


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
