"""実在データ（data/real/*.yaml）の読み込みと整合性。

同梱データそのものを検査対象にしている。手で集めたデータなので、
書き間違いは静かに予想を歪める。テストで気づけるようにしておく。
"""

from __future__ import annotations

from datetime import date

import pytest

from keiba.ratings import REST_NAME, fit_ratings
from keiba.realdata import (
    Horse,
    check_consistency,
    eligible_horses,
    load_horses,
    load_races,
    races_for_rating,
)


@pytest.fixture(scope="module")
def data():
    races, race_meta = load_races()
    horses, horse_meta = load_horses()
    return races, horses, race_meta, horse_meta


def test_shipped_data_is_internally_consistent(data) -> None:
    races, horses, _, _ = data
    assert check_consistency(races, horses) == []


def test_every_race_has_a_source_and_a_result(data) -> None:
    races, _, _, _ = data
    assert len(races) >= 5
    for race in races:
        assert race.sources, race.id
        assert race.result, race.id
        assert race.grade
        assert race.surface in {"turf", "dirt"}


def test_every_horse_named_in_a_result_has_a_profile(data) -> None:
    races, horses, _, _ = data
    known = {h.name for h in horses}
    for race in races:
        for name in race.result:
            assert name in known, f"{race.id} の {name} に馬情報が無い"


def test_horse_profiles_do_not_invent_unknown_fields(data) -> None:
    """調べがつかなかった項目は None のまま。推測で埋めない。"""
    _, horses, _, _ = data
    assert any(h.foaled is None for h in horses), "全頭に生年が入っているのは不自然"
    for horse in horses:
        assert horse.sources, horse.name
        assert horse.status in {"現役", "引退", "未確認"}


def test_races_are_sorted_newest_first(data) -> None:
    races, _, _, _ = data
    dates = [r.date for r in races if r.date]
    assert dates == sorted(dates, reverse=True)


def test_metadata_records_how_the_data_was_collected(data) -> None:
    _, _, race_meta, horse_meta = data
    assert race_meta.get("method") == "web_search"
    assert isinstance(race_meta.get("collected"), date)
    assert horse_meta.get("collected")


def test_consistency_check_catches_a_missing_profile() -> None:
    from keiba.realdata import Race

    races = [Race(id="x", name="test", grade="G1", date=None, venue=None,
                  surface="turf", distance=2000, result=["幽霊馬"], sources=["http://x"])]
    problems = check_consistency(races, [])
    assert any("幽霊馬" in p for p in problems)


def test_consistency_check_catches_duplicate_finishers() -> None:
    from keiba.realdata import Race

    horse = Horse(name="A", sources=["http://x"])
    races = [Race(id="x", name="test", grade="G1", date=None, venue=None,
                  surface="turf", distance=2000, result=["A", "A"], sources=["http://x"])]
    assert any("同じ馬" in p for p in check_consistency(races, [horse]))


def test_eligibility_filters_by_surface_and_activity() -> None:
    horses = [
        Horse(name="芝馬", status="現役", surface="turf", best_distance=[2000, 2400]),
        Horse(name="ダート馬", status="現役", surface="dirt", best_distance=[1800, 2000]),
        Horse(name="引退馬", status="引退", surface="turf", best_distance=[2000, 2400]),
        Horse(name="不明馬", status="現役"),
    ]
    names = [h.name for h in eligible_horses(horses, surface="turf", distance=2400)]
    assert "芝馬" in names
    assert "ダート馬" not in names
    assert "引退馬" not in names
    assert "不明馬" in names          # 分からない項目で足切りはしない


def test_distance_filter_allows_reasonable_stretch() -> None:
    miler = Horse(name="マイラー", status="現役", surface="turf", best_distance=[1600, 1800])
    assert miler.suits("turf", 2000)      # 200m の延長は許容
    assert not miler.suits("turf", 3200)  # 3200m は無理


def test_shipped_data_produces_usable_ratings(data) -> None:
    races, horses, _, _ = data
    ratings = fit_ratings(races_for_rating(races), as_of=date(2026, 7, 28))

    named = [n for n in ratings.names if n != REST_NAME]
    assert len(named) >= 15
    # 実在の馬はどれも「着順不明の集団」より上に来るはず
    rest = ratings.theta[ratings.index(REST_NAME)]
    for name in named:
        assert ratings.theta[ratings.index(name)] > rest, name

    field = ["クロワデュノール", "メイショウタバル", "ダノンデサイル"]
    probs = ratings.win_probabilities(field)
    assert probs.sum() == pytest.approx(1.0)


def test_ratings_reflect_the_collected_head_to_head(data) -> None:
    """大阪杯・宝塚記念で先着している馬が上に来る。"""
    races, _, _, _ = data
    r = fit_ratings(races_for_rating(races), as_of=date(2026, 7, 28))
    assert r.theta[r.index("クロワデュノール")] > r.theta[r.index("ダノンデサイル")]
    assert r.theta[r.index("メイショウタバル")] > r.theta[r.index("ダノンデサイル")]
    assert r.theta[r.index("ロブチェン")] > r.theta[r.index("リアライズシリウス")]
