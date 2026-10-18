"""Sorting realism: refractory contamination + presence; GT vs sorted (sim)."""
from __future__ import annotations

from typing import Any

import numpy as np

from agents.data_quality.metrics.schema import tidy_row
from agents.data_quality.metrics.units import refractory_violation_rate


def presence_ratio(st, t0, t1, bin_s=60.0):
    st = np.asarray(st, float)
    st = st[(st >= t0) & (st < t1)]
    n_bins = max(int(np.floor((t1 - t0) / bin_s)), 1)
    edges = np.linspace(t0, t0 + n_bins * bin_s, n_bins + 1)
    counts, _ = np.histogram(st, bins=edges)
    return float(np.mean(counts > 0))


def sorting_metric_rows(
    spike_times,
    *,
    dataset,
    animal_or_seed,
    session,
    chunk,
    region_set,
    t0,
    t1,
    spike_times_gt=None,
) -> list[dict[str, Any]]:
    refr = []
    pres = []
    for st in spike_times:
        refr.append(refractory_violation_rate(st))
        # contamination estimate ~ refractory rate (proxy; Phy amplitude cutoff dropped)
        pres.append(presence_ratio(st, t0, t1))
    base = dict(
        dataset=dataset,
        animal_or_seed=animal_or_seed,
        session=session,
        chunk=chunk,
        region_set=region_set,
    )
    rows = [
        tidy_row(**base, metric="refractory_contam_proxy_median", W=None, N=None, value=float(np.nanmedian(refr))),
        tidy_row(**base, metric="presence_ratio_median", W=None, N=None, value=float(np.nanmedian(pres))),
    ]
    # Sim-only: pre-degradation (GT) vs post-degradation (sorted) rate retention
    if spike_times_gt is not None and len(spike_times_gt) == len(spike_times):
        ret = []
        for st, gt in zip(spike_times, spike_times_gt):
            st = np.asarray(st, float)
            gt = np.asarray(gt, float)
            st = st[(st >= t0) & (st < t1)]
            gt = gt[(gt >= t0) & (gt < t1)]
            if gt.size == 0:
                ret.append(np.nan)
            else:
                ret.append(float(st.size / gt.size))
        rows.append(
            tidy_row(**base, metric="sorted_over_gt_spike_count_median", W=None, N=None, value=float(np.nanmedian(ret)))
        )
        refr_gt = [refractory_violation_rate(st) for st in spike_times_gt]
        rows.append(
            tidy_row(**base, metric="refractory_contam_gt_median", W=None, N=None, value=float(np.nanmedian(refr_gt)))
        )
    return rows
