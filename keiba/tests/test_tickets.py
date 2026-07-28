"""着順の組み合わせ確率と買い目の組み立てを検証する。"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from keiba.tickets import (
    LAMBDA_2ND,
    LAMBDA_3RD,
    bracket_pair_prob,
    build_proposals,
    exacta_matrix,
    quinella_matrix,
    top_n_probs,
    trifecta_prob,
    trio_prob,
    wide_matrix,
)


@pytest.fixture
def p() -> np.ndarray:
    rng = np.random.default_rng(1)
    v = rng.dirichlet(np.ones(9) * 2.0)
    return v / v.sum()


def test_exacta_rows_and_total_sum_to_one(p) -> None:
    m = exacta_matrix(p)
    # i が1着のときの2着の分布は、i の勝率に一致する
    assert np.allclose(m.sum(axis=1), p)
    assert m.sum() == pytest.approx(1.0)
    assert np.allclose(np.diag(m), 0.0)


def test_trifecta_over_all_orders_sums_to_one(p) -> None:
    total = sum(trifecta_prob(p, *o) for o in itertools.permutations(range(len(p)), 3))
    assert total == pytest.approx(1.0)


def test_place_probabilities_sum_to_the_number_of_places(p) -> None:
    # 3着以内には必ず3頭入るので、合計は3
    assert top_n_probs(p, 3).sum() == pytest.approx(3.0)
    assert top_n_probs(p, 2).sum() == pytest.approx(2.0)
    assert np.all(top_n_probs(p, 3) >= p - 1e-12)


def test_wide_pairs_sum_to_three_combinations(p) -> None:
    # 3着以内の2頭の組み合わせは C(3,2)=3 通りなので、全ペアの合計は3
    m = wide_matrix(p, 3)
    assert np.allclose(np.diag(m), 0.0)
    assert m.sum() / 2 == pytest.approx(3.0)


def test_quinella_is_the_sum_of_both_orders(p) -> None:
    ex, qn = exacta_matrix(p), quinella_matrix(p)
    assert np.allclose(qn, ex + ex.T)
    assert qn.sum() / 2 == pytest.approx(1.0)


def test_trio_is_the_sum_of_its_six_orders(p) -> None:
    combo = (0, 3, 5)
    direct = trio_prob(p, combo)
    manual = sum(trifecta_prob(p, *o) for o in itertools.permutations(combo))
    assert direct == pytest.approx(manual)


def test_stronger_horses_get_higher_place_probability(p) -> None:
    place = top_n_probs(p, 3)
    order_by_win = np.argsort(-p)
    assert list(np.argsort(-place)) == list(order_by_win)


def test_correction_lowers_the_favourite_place_probability(p) -> None:
    """Lo–Bacon-Shone の補正が、人気馬の2・3着を素朴なHarvilleより下げる。"""
    plain = top_n_probs(p, 3, lam2=1.0, lam3=1.0)
    corrected = top_n_probs(p, 3, lam2=LAMBDA_2ND, lam3=LAMBDA_3RD)
    best = int(np.argmax(p))
    worst = int(np.argmin(p))
    assert corrected[best] < plain[best]
    assert corrected[worst] > plain[worst]


def test_uniform_probabilities_give_uniform_combinations() -> None:
    n = 6
    p = np.full(n, 1 / n)
    ex = exacta_matrix(p)
    off = ex[~np.eye(n, dtype=bool)]
    assert np.allclose(off, 1 / (n * (n - 1)))
    assert top_n_probs(p, 3) == pytest.approx(np.full(n, 3 / n))


# ------------------------------------------------------------------ 買い目
def _proposals(n: int = 12):
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(n) * 2.0)
    draws = list(range(1, n + 1))
    brackets = [min(1 + (d - 1) * 8 // n, 8) for d in draws]
    return p, build_proposals(p, draws, brackets, n_runners=n)


def test_every_bet_type_is_offered_with_consistent_points() -> None:
    p, proposals = _proposals()
    codes = [x.code for x in proposals]
    assert codes == ["tansho", "fukusho", "wide", "umaren", "umatan",
                     "wakuren", "sanrenpuku", "sanrentan"]
    by_code = {x.code: x for x in proposals}
    assert by_code["umaren"].points == 3
    assert by_code["wide"].points == 2
    assert by_code["sanrenpuku"].points == 6      # C(4,2)
    assert by_code["sanrentan"].points == 12      # 4P2


def test_hit_probability_falls_as_the_bet_gets_harder() -> None:
    _, proposals = _proposals()
    by_code = {x.code: x.hit_prob for x in proposals}
    assert by_code["fukusho"] > by_code["tansho"]
    assert by_code["wide"] > by_code["umaren"]
    assert by_code["umaren"] > by_code["umatan"]
    assert by_code["sanrenpuku"] > by_code["sanrentan"]
    assert all(0 < v < 1 for v in by_code.values())


def test_breakeven_odds_uses_expected_hits_not_hit_rate() -> None:
    """必要オッズは払戻の期待値と釣り合う倍率なので、分母は期待的中枚数。"""
    _, proposals = _proposals()
    for x in proposals:
        assert x.breakeven_odds == pytest.approx(x.points / x.expected_hits)
        assert x.breakeven_odds > 1.0


def test_only_wide_can_hit_more_than_one_ticket() -> None:
    """ワイド以外は結果が1通りなので、的中確率と期待的中枚数は一致する。"""
    p, proposals = _proposals()
    for x in proposals:
        if x.code == "wide":
            continue
        assert x.overlap == 0.0, x.code
        assert x.hit_prob == pytest.approx(x.expected_hits), x.code


def test_wide_hit_probability_subtracts_the_double_hit() -> None:
    """軸と相手2頭が揃って3着以内なら、ワイドは2点とも当たる。"""
    p, proposals = _proposals()
    wide = next(x for x in proposals if x.code == "wide")
    order = list(np.argsort(-p))
    expected_overlap = trio_prob(p, (order[0], order[1], order[2]))

    assert wide.overlap == pytest.approx(expected_overlap)
    assert wide.overlap > 0
    assert wide.hit_prob < wide.expected_hits
    assert wide.hit_prob == pytest.approx(wide.expected_hits - expected_overlap)
    # 的中確率は「どれか当たる」なので、個々の的中率の最大値以上・合計以下
    assert max(t["prob"] for t in wide.tickets) <= wide.hit_prob <= wide.expected_hits


def test_bracket_quinella_counts_every_horse_in_the_bracket() -> None:
    """枠連は同じ枠の別の馬が来ても当たるので、馬連より的中率が高い。"""
    p, proposals = _proposals()
    by_code = {x.code: x for x in proposals}
    waku, umaren = by_code["wakuren"], by_code["umaren"]
    qn = quinella_matrix(p)
    brackets = [min(1 + (d - 1) * 8 // 12, 8) for d in range(1, 13)]

    axis = int(np.argmax(p))
    mate = int(np.argsort(-p)[1])
    key = tuple(sorted((brackets[axis], brackets[mate])))
    direct = bracket_pair_prob(qn, brackets, *key)
    ticket = next(t for t in waku.tickets if tuple(t["legs"]) == key)

    assert ticket["prob"] == pytest.approx(direct)
    # 軸と相手のペアだけを数えた値より必ず大きい（同枠の別の馬の分が乗る）
    assert ticket["prob"] > qn[axis, mate]
    umaren_top2 = sum(t["prob"] for t in umaren.tickets[:2])
    assert waku.hit_prob > umaren_top2


def test_same_bracket_pair_counts_only_pairs_inside_it() -> None:
    p = np.full(10, 0.1)
    brackets = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    qn = quinella_matrix(p)
    same = bracket_pair_prob(qn, brackets, 1, 1)
    cross = bracket_pair_prob(qn, brackets, 1, 2)
    # 同枠は1組、別枠は2×2=4組ぶん
    assert cross == pytest.approx(same * 4)


def test_all_tickets_share_the_same_axis_horse() -> None:
    p, proposals = _proposals()
    axis = int(np.argmax(p)) + 1     # draws は 1..n
    for x in proposals:
        if x.code == "wakuren":
            continue
        assert all(axis in t["legs"] for t in x.tickets)


def test_tickets_are_unique_within_a_bet_type() -> None:
    _, proposals = _proposals()
    for x in proposals:
        keys = [tuple(t["legs"]) for t in x.tickets]
        assert len(keys) == len(set(keys)), x.code


def test_small_fields_use_two_places_and_skip_bracket_quinella() -> None:
    _, proposals = _proposals(n=7)
    codes = [x.code for x in proposals]
    assert "wakuren" not in codes          # 8頭以下は枠連の発売がない
    by_code = {x.code: x for x in proposals}
    assert "2着以内" in by_code["fukusho"].note
