"""Unit-order validation between training bundle and live spike sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class UnitMappingReport:
    expected_unit_ids: list[int]
    live_unit_ids: list[int]
    mapped_unit_ids: list[int]
    missing_unit_ids: list[int]
    unexpected_unit_ids: list[int]
    permutation: list[int | None] = field(default_factory=list)
    exact_match: bool = False

    @property
    def n_expected(self) -> int:
        return len(self.expected_unit_ids)

    @property
    def n_mapped(self) -> int:
        return len(self.mapped_unit_ids)

    @property
    def n_missing(self) -> int:
        return len(self.missing_unit_ids)

    @property
    def n_unexpected(self) -> int:
        return len(self.unexpected_unit_ids)

    def to_dict(self) -> dict:
        return {
            "n_expected": self.n_expected,
            "n_mapped": self.n_mapped,
            "n_missing": self.n_missing,
            "n_unexpected": self.n_unexpected,
            "expected_unit_ids": self.expected_unit_ids,
            "live_unit_ids": self.live_unit_ids,
            "mapped_unit_ids": self.mapped_unit_ids,
            "missing_unit_ids": self.missing_unit_ids,
            "unexpected_unit_ids": self.unexpected_unit_ids,
            "exact_match": self.exact_match,
        }


def map_units(
    expected_unit_ids: Sequence[int],
    live_unit_ids: Sequence[int],
) -> UnitMappingReport:
    """Compare training unit order against live stream unit ids.

    Never assumes live index ``i`` equals training index ``i``. Mapping is by
    identity (unit_id). Missing training units are tracked; unexpected live
    units are reported but ignored for the feature vector.
    """
    expected = [int(u) for u in expected_unit_ids]
    live = [int(u) for u in live_unit_ids]
    live_set = set(live)
    expected_set = set(expected)
    missing = [u for u in expected if u not in live_set]
    unexpected = [u for u in live if u not in expected_set]
    mapped = [u for u in expected if u in live_set]
    # permutation[i] = live index of expected[i], or None if missing
    live_index = {u: i for i, u in enumerate(live)}
    perm = [live_index.get(u) for u in expected]
    return UnitMappingReport(
        expected_unit_ids=expected,
        live_unit_ids=live,
        mapped_unit_ids=mapped,
        missing_unit_ids=missing,
        unexpected_unit_ids=unexpected,
        permutation=perm,
        exact_match=(not missing and not unexpected and perm == list(range(len(expected)))),
    )
