"""実在馬・実在レースのデータ（``data/real/*.yaml``）の読み込み。

このデータは手作業で集めたもので、出典URLと収集日を各レコードに持たせてある。
機械で取ってきた完全なデータベースではないので、**必ず出典で裏を取ってから使う**
という前提を型と関数名で表している（``Race.confirmed`` など）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real"
RACES_FILE = DATA_DIR / "races.yaml"
HORSES_FILE = DATA_DIR / "horses.yaml"


@dataclass
class Race:
    id: str
    name: str
    grade: str
    date: date | None
    venue: str | None
    surface: str | None
    distance: int | None
    result: list[str] = field(default_factory=list)
    note: str | None = None
    field_size: int | None = None
    date_confirmed: bool = True
    sources: list[str] = field(default_factory=list)

    @property
    def known_places(self) -> int:
        """入線順が判明している頭数。"""
        return len(self.result)


@dataclass
class Horse:
    name: str
    sex: str | None = None
    foaled: int | None = None
    sire: str | None = None
    status: str = "未確認"
    surface: str | None = None
    best_distance: list = field(default_factory=list)
    country: str = "JPN"
    note: str | None = None
    sources: list[str] = field(default_factory=list)
    record: str | None = None

    @property
    def active(self) -> bool:
        return self.status == "現役"

    def suits(self, surface: str | None, distance: int | None) -> bool:
        """その条件に出しても不自然でないか（分からない項目は「可」とみなす）。"""
        if surface and self.surface and self.surface != surface:
            return False
        if distance and self.best_distance:
            lo, hi = (list(self.best_distance) + [None, None])[:2]
            if lo and distance < lo - 400:
                return False
            if hi and distance > hi + 400:
                return False
        return True


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"データが見つかりません: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_races(path: Path | str = RACES_FILE) -> tuple[list[Race], dict]:
    raw = _load(Path(path))
    races = []
    for item in raw.get("races", []):
        item = dict(item)
        raw_date = item.pop("date", None)
        if isinstance(raw_date, str):
            raw_date = date.fromisoformat(raw_date)
        races.append(Race(date=raw_date, **{k: v for k, v in item.items()
                                            if k in Race.__dataclass_fields__}))
    races.sort(key=lambda r: (r.date or date.min), reverse=True)
    return races, raw.get("meta", {})


def load_horses(path: Path | str = HORSES_FILE) -> tuple[list[Horse], dict]:
    raw = _load(Path(path))
    horses = [
        Horse(**{k: v for k, v in item.items() if k in Horse.__dataclass_fields__})
        for item in raw.get("horses", [])
    ]
    return horses, raw.get("meta", {})


def check_consistency(races: list[Race], horses: list[Horse]) -> list[str]:
    """データの矛盾を洗い出す（黙って壊れたデータで予想しないため）。"""
    problems: list[str] = []
    known = {h.name for h in horses}

    for race in races:
        if len(set(race.result)) != len(race.result):
            problems.append(f"{race.id}: 入線順に同じ馬が複数回登場している")
        for name in race.result:
            if name not in known:
                problems.append(f"{race.id}: 馬の情報が無い -> {name}")
        if not race.sources:
            problems.append(f"{race.id}: 出典が無い")
        if race.field_size and race.known_places > race.field_size:
            problems.append(f"{race.id}: 出走頭数より入線頭数が多い")

    seen: set[str] = set()
    for horse in horses:
        if horse.name in seen:
            problems.append(f"馬名が重複している -> {horse.name}")
        seen.add(horse.name)
        if not horse.sources:
            problems.append(f"{horse.name}: 出典が無い")
    return problems


def races_for_rating(races: list[Race]) -> list[dict]:
    """:func:`keiba.ratings.fit_ratings` に渡す形へ変換する。"""
    return [{"result": r.result, "date": r.date, "id": r.id} for r in races]


def eligible_horses(horses: list[Horse], *, surface: str | None = None,
                    distance: int | None = None, active_only: bool = True) -> list[Horse]:
    """その条件の大レースに出られそうな馬を絞り込む。"""
    out = []
    for horse in horses:
        if active_only and not horse.active:
            continue
        if not horse.suits(surface, distance):
            continue
        out.append(horse)
    return out
