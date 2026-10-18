"""Lab session discovery — always via Report 2 session_select (import, never copy)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from analysis.real_quadrant.session_select import (
    _data_root,
    _passes_room_a_exclusions,
    animal_id_from_session,
    list_2room_sessions,
)


def all_2room_sessions(data_root: Path | None = None) -> list[str]:
    """Same discovery as M3/Report 2: ``\"2rooms\" in name``."""
    return list_2room_sessions(data_root)


def eligible_room_a_sessions(data_root: Path | None = None) -> list[str]:
    """Sessions that pass ``_passes_room_a_exclusions`` (M3 gate; all eligible)."""
    r = _data_root(data_root)
    out: list[str] = []
    for name in list_2room_sessions(r):
        ok, _n_u, _reason = _passes_room_a_exclusions(r / name)
        if ok:
            out.append(name)
    return out


def eligibility_by_animal(data_root: Path | None = None) -> dict[str, Any]:
    """Per-animal totals / eligible / excluded-by-reason (full set, not M3 cohort)."""
    r = _data_root(data_root)
    names = list_2room_sessions(r)
    by: dict[str, dict[str, Any]] = {}
    for name in names:
        animal = animal_id_from_session(name)
        rec = by.setdefault(
            animal,
            {
                "animal": animal,
                "total": 0,
                "eligible": 0,
                "excluded": 0,
                "exclude_reasons": Counter(),
                "eligible_sessions": [],
                "excluded_sessions": [],
            },
        )
        rec["total"] += 1
        ok, n_u, reason = _passes_room_a_exclusions(r / name)
        if ok:
            rec["eligible"] += 1
            rec["eligible_sessions"].append({"session": name, "n_units": n_u})
        else:
            rec["excluded"] += 1
            rec["exclude_reasons"][reason or "unknown"] += 1
            rec["excluded_sessions"].append(
                {"session": name, "reason": reason, "n_units": n_u}
            )
    animals = []
    for animal in sorted(by.keys()):
        rec = by[animal]
        rec["exclude_reasons"] = dict(rec["exclude_reasons"])
        animals.append(rec)
    return {
        "rule": "list_2room_sessions + _passes_room_a_exclusions (import)",
        "n_2rooms": len(names),
        "n_animals": len(animals),
        "n_eligible": sum(a["eligible"] for a in animals),
        "animals": animals,
    }
