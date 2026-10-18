from __future__ import annotations

import numpy as np


def seconds_guard(spike_times: list[np.ndarray], *, name: str = "data") -> None:
    """Fail loudly if times look like sample indices (30 kHz) rather than seconds."""
    med_isis = []
    n_hot = 0
    n_u = 0
    for st in spike_times:
        st = np.asarray(st, dtype=float)
        st = st[np.isfinite(st)]
        if st.size < 5:
            continue
        st = np.sort(st)
        d = np.diff(st)
        d = d[d > 0]
        if d.size == 0:
            continue
        med_isis.append(float(np.median(d)))
        dur = max(st[-1] - st[0], 1e-9)
        rate = st.size / dur
        n_u += 1
        if rate > 500:
            n_hot += 1
    if med_isis and float(np.median(med_isis)) < 1e-4:
        raise RuntimeError(
            f"seconds_guard FAIL ({name}): median ISI {np.median(med_isis):.3e} s "
            "(looks like sample indices)"
        )
    if n_u and (n_hot / n_u) > 0.01:
        raise RuntimeError(
            f"seconds_guard FAIL ({name}): {n_hot}/{n_u} units fire above 500 Hz"
        )
