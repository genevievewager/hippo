from __future__ import annotations

from typing import Any


def tidy_row(
    *,
    dataset: str,
    animal_or_seed: str,
    session: str,
    chunk: int | str,
    region_set: str,
    metric: str,
    W: float | None,
    N: int | None,
    value: float,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "dataset": dataset,
        "animal|seed": animal_or_seed,
        "session": session,
        "chunk": chunk,
        "region_set": region_set,
        "metric": metric,
        "W": W,
        "N": N,
        "value": float(value),
    }
    row.update(extra)
    return row
