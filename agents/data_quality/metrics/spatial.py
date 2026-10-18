"""Spatial coding metrics from train-only rate maps (Report 4a revised).

Primary scheme: 20×20 grid over the bounding box (~5 cm). Secondary: 3 cm bins.
Smoothing: Gaussian σ = 1.5 bins. Min occupancy: 0.15 s/bin. Speed filter: > 2 cm/s.
Eligible units: ≥100 train spikes and mean rate ≥ 0.2 Hz.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from agents.data_quality.metrics.schema import tidy_row

# Defaults (lab and sim identical)
SPEED_MIN_CM_S = 2.0
MIN_OCC_S = 0.15
SMOOTH_SIGMA_BINS = 1.5
MIN_TRAIN_SPIKES = 100
MIN_MEAN_RATE_HZ = 0.2
N_SHUFFLES = 100
FIELD_FRAC_OF_PEAK = 0.2
TRAIN_FRAC = 0.8


def _spike_frame_indices(
    st: np.ndarray,
    t_frames: np.ndarray,
    *,
    keep: np.ndarray | None = None,
) -> np.ndarray:
    """Frame index for each spike on a contiguous parent timeline.

    Frame ``i`` owns ``[t[i], t[i+1])``; the last frame owns
    ``[t[-1], t[-1] + dt)`` with ``dt = median(diff(t))``.
    Spikes whose assigned frame has ``keep[i] == False`` are dropped
    (speed filter, valid mask, split arm, chunk edge, etc.).
    ``t_frames`` must be the contiguous parent sampling times — not a
    pre-subsetted arm — so excluded frames exist in the timeline.
    """
    st = np.asarray(st, float)
    t_frames = np.asarray(t_frames, float)
    n = len(t_frames)
    if st.size == 0 or n == 0:
        return np.zeros(0, dtype=int)
    if keep is not None:
        keep = np.asarray(keep, bool)
        if keep.shape != (n,):
            raise ValueError(f"keep shape {keep.shape} != ({n},)")
    dt = float(np.median(np.diff(t_frames))) if n > 1 else 0.05
    dt = max(dt, 1e-9)
    t_end = float(t_frames[-1]) + dt
    st = st[(st >= float(t_frames[0])) & (st < t_end)]
    if st.size == 0:
        return np.zeros(0, dtype=int)
    idx = np.searchsorted(t_frames, st, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    next_t = np.empty(n, float)
    next_t[:-1] = t_frames[1:]
    next_t[-1] = t_end
    in_interval = (st >= t_frames[idx]) & (st < next_t[idx])
    if keep is not None:
        in_interval = in_interval & keep[idx]
    return idx[in_interval].astype(int)


def _spike_positions(
    st: np.ndarray,
    t_frames: np.ndarray,
    p_frames: np.ndarray,
    keep: np.ndarray | None = None,
) -> np.ndarray:
    """Position of every spike on kept frames (multiple spikes/frame counted)."""
    p_frames = np.asarray(p_frames, float)
    idx = _spike_frame_indices(st, t_frames, keep=keep)
    if idx.size == 0:
        return np.zeros((0, 2), float)
    return p_frames[idx]


def _rate_map(pos, spike_pos, edges_x, edges_y, *, dt: float, occ_thresh_frames: float = 1.0):
    """Rate map in Hz: spikes / (occupancy_frames × dt).

    ``occ`` returned in seconds. Occupancy threshold is applied in frames
    (``occ_frames >= occ_thresh_frames``), matching min-occupancy in seconds
    when ``occ_thresh_frames = MIN_OCC_S / dt``.
    """
    pos = np.asarray(pos, float)
    occ_fr, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[edges_x, edges_y])
    spike_pos = np.asarray(spike_pos, float).reshape(-1, 2) if len(np.asarray(spike_pos)) else np.zeros((0, 2))
    if spike_pos.size:
        spk, _, _ = np.histogram2d(spike_pos[:, 0], spike_pos[:, 1], bins=[edges_x, edges_y])
    else:
        spk = np.zeros_like(occ_fr)
    dt = float(max(dt, 1e-9))
    occ_s = occ_fr * dt
    ok = occ_fr >= float(occ_thresh_frames)
    rate = np.zeros_like(occ_fr, dtype=float)
    rate[ok] = spk[ok] / occ_s[ok]
    return rate, occ_s, ok


def rate_map_mean_hz(rate: np.ndarray, occ_s: np.ndarray, ok: np.ndarray) -> float:
    """Occupancy-weighted mean of the rate map (Hz)."""
    if not ok.any():
        return float("nan")
    w = occ_s.astype(float)
    return float((rate[ok] * w[ok]).sum() / max(w[ok].sum(), 1e-12))


def smooth_rate_map(rate: np.ndarray, ok: np.ndarray, sigma: float = SMOOTH_SIGMA_BINS) -> np.ndarray:
    """Occupancy-aware Gaussian smooth; unvisited stay NaN/0 outside ok."""
    r = np.array(rate, float)
    w = ok.astype(float)
    r2 = r.copy()
    r2[~ok] = 0.0
    num = gaussian_filter(r2, sigma=sigma, mode="constant", cval=0.0)
    den = gaussian_filter(w, sigma=sigma, mode="constant", cval=0.0)
    out = np.zeros_like(r)
    m = den > 1e-8
    out[m] = num[m] / den[m]
    out[~ok] = 0.0
    return out


def spatial_info_bits_per_spike(rate, occ, ok):
    """Skaggs spatial information (bits/spike)."""
    if not ok.any():
        return float("nan")
    p = occ.astype(float)
    p = p / max(p.sum(), 1e-12)
    lam = rate
    mean = float((p[ok] * lam[ok]).sum())
    if mean <= 0:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.zeros_like(lam)
        r[ok] = lam[ok] / mean
        term = np.zeros_like(lam)
        m = ok & (lam > 0)
        term[m] = p[m] * r[m] * np.log2(r[m])
    return float(term.sum())


def sparsity(rate, occ, ok):
    """Skaggs sparsity: (Σ p_i λ_i)² / Σ p_i λ_i² on occupancy-normalised maps.

    Near 1 for broadly firing cells; near 0 for sparse place fields.
    """
    if not ok.any():
        return float("nan")
    p = occ.astype(float)
    p = p / max(p.sum(), 1e-12)
    lam = rate
    mean = float((p[ok] * lam[ok]).sum())
    den = float((p[ok] * (lam[ok] ** 2)).sum())
    if den <= 0 or mean <= 0:
        return float("nan")
    return float((mean ** 2) / den)


FIELD_MIN_HZ = 1.0
FIELD_MIN_BINS = 4


def place_field_mask(
    rate: np.ndarray,
    ok: np.ndarray,
    frac: float = FIELD_FRAC_OF_PEAK,
    *,
    min_hz: float = FIELD_MIN_HZ,
    min_bins: int = FIELD_MIN_BINS,
) -> np.ndarray:
    """Contiguous components of bins with rate ≥ max(frac×peak, min_hz); keep components ≥ min_bins."""
    from scipy import ndimage

    if not ok.any():
        return np.zeros_like(ok, dtype=bool)
    mx = float(np.nanmax(rate[ok]))
    if mx <= 0:
        return np.zeros_like(ok, dtype=bool)
    thr = max(frac * mx, float(min_hz))
    cand = ok & (rate >= thr)
    if not cand.any():
        return np.zeros_like(ok, dtype=bool)
    labeled, nlab = ndimage.label(cand)
    keep = np.zeros_like(ok, dtype=bool)
    for lab in range(1, int(nlab) + 1):
        comp = labeled == lab
        if int(comp.sum()) >= int(min_bins):
            keep |= comp
    return keep


def coverage_frac(rate: np.ndarray, ok: np.ndarray) -> float:
    """Fraction of visited bins inside place fields (legacy / secondary)."""
    if not ok.any():
        return float("nan")
    field = place_field_mask(rate, ok)
    if not field.any():
        return 0.0
    return float(field.sum() / ok.sum())


def place_field_size_among_tuned(rate: np.ndarray, ok: np.ndarray) -> float:
    """Field size = field_bins / visited, only meaningful for place-tuned units."""
    return coverage_frac(rate, ok)


def _grid_score(rate: np.ndarray, ok: np.ndarray) -> float:
    """Sargolini-style grid score via Pearson corr of 60° vs 30° rotations."""
    if rate.shape[0] < 5 or rate.shape[1] < 5 or not ok.any():
        return float("nan")
    r = np.array(rate, float)
    r[~ok] = np.nan
    cy, cx = (np.array(r.shape) - 1) / 2.0

    def rot(deg: float) -> np.ndarray:
        th = np.deg2rad(deg)
        c, s = np.cos(th), np.sin(th)
        out = np.full_like(r, np.nan)
        yy, xx = np.indices(r.shape)
        x0, y0 = xx - cx, yy - cy
        xr = c * x0 - s * y0 + cx
        yr = s * x0 + c * y0 + cy
        xi = np.rint(xr).astype(int)
        yi = np.rint(yr).astype(int)
        m = (xi >= 0) & (xi < r.shape[1]) & (yi >= 0) & (yi < r.shape[0])
        out[yy[m], xx[m]] = r[yi[m], xi[m]]
        return out

    def pearson(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10:
            return float("nan")
        aa, bb = a[m], b[m]
        aa = aa - aa.mean()
        bb = bb - bb.mean()
        den = np.sqrt((aa ** 2).sum() * (bb ** 2).sum())
        if den <= 0:
            return float("nan")
        return float((aa * bb).sum() / den)

    c60 = [pearson(r, rot(d)) for d in (60, 120)]
    c30 = [pearson(r, rot(d)) for d in (30, 90, 150)]
    if not any(np.isfinite(c60)) or not any(np.isfinite(c30)):
        return float("nan")
    return float(np.nanmin(c60) - np.nanmax(c30))


def _hd_mean_vector_length(spike_hd: np.ndarray) -> float:
    if spike_hd.size < 5:
        return float("nan")
    return float(np.abs(np.mean(np.exp(1j * spike_hd))))


def _border_score(rate: np.ndarray, ok: np.ndarray, n_border: int = 2) -> float:
    """Solstad border score ≈ (CM - DM) / (CM + DM)."""
    if not ok.any():
        return float("nan")
    r = np.array(rate, float)
    mx = np.nanmax(r[ok]) if ok.any() else 0.0
    if mx <= 0:
        return float("nan")
    field = ok & (r >= 0.3 * mx)
    if not field.any():
        return float("nan")
    h, w = r.shape
    border = np.zeros_like(field)
    border[:n_border, :] = True
    border[-n_border:, :] = True
    border[:, :n_border] = True
    border[:, -n_border:] = True
    cm = float((field & border).sum() / max(field.sum(), 1))
    yy, xx = np.indices(r.shape)
    dist = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(float)
    dist = dist / max(min(h, w) / 2.0, 1.0)
    dm = float(dist[field].mean()) if field.any() else 1.0
    den = cm + dm
    if den <= 0:
        return float("nan")
    return float((cm - dm) / den)


def _speed_tuning_r(spike_speeds: np.ndarray, all_speeds: np.ndarray, n_bins: int = 10) -> float:
    """|corr| of mean rate vs speed-bin center (occupancy-normalized)."""
    if spike_speeds.size < 5 or all_speeds.size < 20:
        return float("nan")
    lo, hi = np.nanpercentile(all_speeds, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, n_bins + 1)
    occ, _ = np.histogram(all_speeds, bins=edges)
    spk, _ = np.histogram(spike_speeds, bins=edges)
    ok = occ > 0
    if ok.sum() < 3:
        return float("nan")
    rate = np.zeros(n_bins)
    rate[ok] = spk[ok] / occ[ok]
    centers = 0.5 * (edges[:-1] + edges[1:])
    if np.std(rate[ok]) < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(centers[ok], rate[ok])[0, 1]))


def unit_eligible(st: np.ndarray, t0: float, t1: float) -> bool:
    st = np.asarray(st, float)
    st = st[(st >= t0) & (st < t1)]
    dur = max(t1 - t0, 1e-9)
    return st.size >= MIN_TRAIN_SPIKES and (st.size / dur) >= MIN_MEAN_RATE_HZ


CHANCE_PASS = 0.05  # 95th-percentile shuffle → ~5% false-positive rate


def _build_map(
    st,
    t_frames,
    p_frames,
    edges_x,
    edges_y,
    occ_thresh_frames,
    dt,
    reachable=None,
    shared_ok=None,
    keep=None,
):
    """Build smoothed rate map from parent frames + optional keep mask.

    Occupancy uses ``p_frames[keep]``; spikes are assigned on the full
    ``t_frames`` timeline and dropped when their frame is not kept.
    """
    t_frames = np.asarray(t_frames, float)
    p_frames = np.asarray(p_frames, float)
    if keep is None:
        keep = np.ones(len(t_frames), dtype=bool)
    else:
        keep = np.asarray(keep, bool)
    pos_occ = p_frames[keep]
    if pos_occ.size == 0:
        n_x, n_y = len(edges_x) - 1, len(edges_y) - 1
        z = np.zeros((n_x, n_y), float)
        return z, z.copy(), np.zeros((n_x, n_y), dtype=bool)
    spike_pos = _spike_positions(st, t_frames, p_frames, keep=keep)
    rate, occ, ok = _rate_map(
        pos_occ, spike_pos, edges_x, edges_y, dt=dt, occ_thresh_frames=occ_thresh_frames,
    )
    if reachable is not None:
        if reachable.shape != ok.shape:
            raise ValueError(f"reachable shape {reachable.shape} != rate-map {ok.shape}")
        ok = ok & reachable
    if shared_ok is not None:
        if shared_ok.shape != ok.shape:
            raise ValueError(f"shared_ok shape {shared_ok.shape} != rate-map {ok.shape}")
        ok = ok & shared_ok
    rate = smooth_rate_map(rate, ok, sigma=SMOOTH_SIGMA_BINS)
    rate = rate.copy()
    rate[~ok] = 0.0
    occ = occ.copy()
    occ[~ok] = 0.0
    return rate, occ, ok


def _unit_spatial_scores(rate, occ, ok) -> dict[str, float]:
    info = spatial_info_bits_per_spike(rate, occ, ok)
    spars = sparsity(rate, occ, ok)
    peak = float(np.nanmax(rate[ok])) if ok.any() else float("nan")
    field_mask = place_field_mask(rate, ok)
    field = int(field_mask.any())
    # Field size = field bins / visited (report among place-tuned units only).
    pfs = float(field_mask.sum() / ok.sum()) if ok.any() else float("nan")
    return {
        "info": info,
        "spars": spars,
        "peak": peak,
        "grid": _grid_score(rate, ok),
        "border": _border_score(rate, ok),
        "field": field,
        "place_field_size": pfs,
        "map_mean_hz": rate_map_mean_hz(rate, occ, ok),
    }


def _circular_shift_spikes(st: np.ndarray, t0: float, t1: float, shift: float) -> np.ndarray:
    st = np.asarray(st, float)
    st = st[(st >= t0) & (st < t1)]
    if st.size == 0:
        return st
    dur = t1 - t0
    return t0 + np.mod(st - t0 + shift, dur)


def _shuffle_threshold(
    st,
    t_frames,
    p_frames,
    edges_x,
    edges_y,
    occ_thresh_frames,
    dt,
    score_fn,
    *,
    keep=None,
    reachable=None,
    shared_ok=None,
    n_shuffles: int = N_SHUFFLES,
    rng: np.random.Generator,
) -> float:
    """95th percentile of score under circular spike-time shifts."""
    t_frames = np.asarray(t_frames, float)
    if keep is None:
        keep = np.ones(len(t_frames), dtype=bool)
    kept_t = t_frames[keep]
    if kept_t.size < 2:
        return float("nan")
    t0, t1 = float(kept_t[0]), float(kept_t[-1])
    dur = t1 - t0
    if dur <= 1.0 or np.asarray(st).size < MIN_TRAIN_SPIKES:
        return float("nan")
    min_shift = max(1.0, 0.05 * dur)
    scores = []
    for _ in range(int(n_shuffles)):
        shift = float(rng.uniform(min_shift, dur - min_shift))
        st_s = _circular_shift_spikes(st, t0, t1, shift)
        rate, occ, ok = _build_map(
            st_s, t_frames, p_frames, edges_x, edges_y, occ_thresh_frames, dt,
            reachable, shared_ok, keep=keep,
        )
        scores.append(score_fn(rate, occ, ok))
    scores = np.asarray(scores, float)
    scores = scores[np.isfinite(scores)]
    if scores.size < max(20, n_shuffles // 5):
        return float("nan")
    return float(np.percentile(scores, 95))


def split_half_stability(
    st, t_frames, p_frames, edges_x, edges_y, occ_thresh_frames, dt,
    reachable=None, keep=None,
) -> float:
    """Pearson corr of rate maps on first vs second half of kept training frames."""
    t_frames = np.asarray(t_frames, float)
    if keep is None:
        keep = np.ones(len(t_frames), dtype=bool)
    kept_idx = np.flatnonzero(keep)
    if kept_idx.size < 20:
        return float("nan")
    mid = kept_idx.size // 2
    keep_a = np.zeros(len(t_frames), dtype=bool)
    keep_b = np.zeros(len(t_frames), dtype=bool)
    keep_a[kept_idx[:mid]] = True
    keep_b[kept_idx[mid:]] = True
    r1, _, ok1 = _build_map(
        st, t_frames, p_frames, edges_x, edges_y, occ_thresh_frames, dt, reachable, keep=keep_a,
    )
    r2, _, ok2 = _build_map(
        st, t_frames, p_frames, edges_x, edges_y, occ_thresh_frames, dt, reachable, keep=keep_b,
    )
    m = ok1 & ok2
    if m.sum() < 10:
        return float("nan")
    a, b = r1[m], r2[m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _aggregate_unit_list(vals: list[float], idx: np.ndarray | None = None) -> float:
    if idx is None:
        arr = np.asarray(vals, float)
    else:
        arr = np.asarray([vals[int(k)] for k in idx], float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def spatial_metric_rows(
    spike_times,
    positions,
    times,
    *,
    dataset,
    animal_or_seed,
    session,
    chunk,
    region_set,
    t0,
    t1,
    bin_cm=3.0,
    grid_n=20,
    speed=None,
    head_direction=None,
    speed_min=SPEED_MIN_CM_S,
    train_frac=TRAIN_FRAC,
    boundary_poly=None,
    matched_Ns=None,
    n_subsamples: int = 20,
    rng=None,
    shared_ok_by_scheme: dict | None = None,
    edges_by_scheme: dict | None = None,
    n_shuffles: int = N_SHUFFLES,
    do_shuffles: bool = True,
    unit_regions: list[str] | None = None,
    target_region_counts: dict[str, int] | None = None,
    path_reachable: np.ndarray | None = None,
    use_path_visited_mask: bool = False,
) -> list[dict[str, Any]]:
    """Spatial metrics. Same code path for lab and sim.

    Rate maps are in Hz (spikes / occupancy_seconds). When
    ``target_region_counts`` is set (matched arm), unit-level medians and
    frac-pass are the median over ``n_subsamples`` composition-matched draws.

    ``use_path_visited_mask``: when True (path-containment flag, session kept),
    ignore the boundary polygon and restrict bins to those visited by the
    path. ``path_reachable`` may supply an explicit mask instead.
    Occupancy-matched joint bins (``shared_ok_by_scheme``) are ignored: a
    grid-index intersection across lab polygon and sim square is not meaningful.
    """
    from agents.data_quality.metrics.boundary import bin_centers_reachable, path_visited_bins
    from agents.data_quality.matching.regions import composition_matched_subsample

    times = np.asarray(times, float)
    pos = np.asarray(positions, float)
    # Contiguous parent window; keep = first train_frac of *moving* frames
    # (preserves prior train-split semantics on the speed-filtered sequence).
    m_win = (times >= t0) & (times < t1)
    t_win = times[m_win]
    p_win = pos[m_win]
    spd_win = np.asarray(speed, float)[m_win] if speed is not None else None
    hd_win = np.asarray(head_direction, float)[m_win] if head_direction is not None else None
    if t_win.size < 10:
        return []
    if spd_win is not None:
        moving = np.isfinite(spd_win) & (spd_win > float(speed_min))
    else:
        moving = np.ones(len(t_win), dtype=bool)
    moving_idx = np.flatnonzero(moving)
    if moving_idx.size < 10:
        return []
    n_train_move = int(train_frac * moving_idx.size)
    if n_train_move < 10:
        return []
    keep = np.zeros(len(t_win), dtype=bool)
    keep[moving_idx[:n_train_move]] = True
    p_tr, t_tr = p_win[keep], t_win[keep]
    spd_tr = spd_win[keep] if spd_win is not None else None
    hd_tr = hd_win[keep] if hd_win is not None else None
    dt = float(np.median(np.diff(t_tr))) if t_tr.size > 1 else 0.05
    occ_thresh = max(1.0, MIN_OCC_S / max(dt, 1e-6))  # frames
    t_lo, t_hi = float(t_tr[0]), float(t_tr[-1])
    xmin, ymin = p_tr.min(0)
    xmax, ymax = p_tr.max(0)
    rng = rng or np.random.default_rng(0)
    train_dur = max(t_hi - t_lo, 1e-9)

    # eligibility on training window
    eligible_idx = [i for i, st in enumerate(spike_times) if unit_eligible(st, t_lo, t_hi)]
    n_elig = len(eligible_idx)
    n_tot = len(spike_times)
    # region labels for eligible units (for matched composition subsample)
    elig_regions = None
    if unit_regions is not None and len(unit_regions) == len(spike_times):
        elig_regions = np.asarray([str(unit_regions[i]) for i in eligible_idx], dtype=str)

    rows: list[dict[str, Any]] = []
    base0 = dict(
        dataset=dataset,
        animal_or_seed=animal_or_seed,
        session=session,
        chunk=chunk,
        region_set=region_set,
    )
    rows.append(tidy_row(**base0, metric="n_units_total", W=None, N=None, value=float(n_tot)))
    rows.append(tidy_row(**base0, metric="n_units_eligible_spatial", W=None, N=None, value=float(n_elig)))
    rows.append(
        tidy_row(
            **base0,
            metric="frac_units_eligible_spatial",
            W=None,
            N=None,
            value=float(n_elig / max(n_tot, 1)),
        )
    )
    if n_elig == 0:
        return rows

    schemes = [
        ("grid20", np.linspace(xmin, xmax, grid_n + 1), np.linspace(ymin, ymax, grid_n + 1)),
        ("bin3cm", np.arange(xmin, xmax + bin_cm, bin_cm), np.arange(ymin, ymax + bin_cm, bin_cm)),
    ]

    for scheme, edges_x0, edges_y0 in schemes:
        if edges_x0.size < 3 or edges_y0.size < 3:
            continue

        # Occupancy-matched joint-bin intersection across lab polygon vs sim square
        # is not meaningful (different coordinate frames / shapes). Dropped; behaviour
        # confound is noted in the report instead. shared_ok_by_scheme retained in
        # signature for call-site compatibility but ignored.
        _ = (shared_ok_by_scheme, edges_by_scheme)
        edges_x, edges_y = edges_x0, edges_y0
        reachable = None
        if use_path_visited_mask:
            reachable = path_visited_bins(p_tr, edges_x, edges_y, min_frames=1)
        elif boundary_poly is not None:
            reachable = bin_centers_reachable(edges_x, edges_y, boundary_poly)
        if path_reachable is not None:
            if path_reachable.shape != (len(edges_x) - 1, len(edges_y) - 1):
                raise ValueError(
                    f"path_reachable shape {path_reachable.shape} != "
                    f"({len(edges_x) - 1}, {len(edges_y) - 1})"
                )
            reachable = path_reachable if reachable is None else (reachable & path_reachable)
        per_unit = []
        tot_occ_s = float(int(keep.sum()) * dt)
        # Headings / speeds on parent window (indexed by keep-assigned spikes)
        if hd_win is not None:
            hd_parent = hd_win
        elif t_win.size >= 2:
            vv = np.diff(p_win, axis=0, prepend=p_win[:1])
            hd_parent = np.arctan2(vv[:, 1], vv[:, 0])
        else:
            hd_parent = None
        for i in eligible_idx:
            # Unsmoothed map (1-frame thresh) for Hz consistency vs empirical rate
            spike_pos = _spike_positions(spike_times[i], t_win, p_win, keep=keep)
            rate_raw, occ_raw, ok_raw = _rate_map(
                p_tr, spike_pos, edges_x, edges_y, dt=dt, occ_thresh_frames=1.0,
            )
            rate, occ, ok = _build_map(
                spike_times[i], t_win, p_win, edges_x, edges_y, occ_thresh, dt,
                reachable, None, keep=keep,
            )
            sc = _unit_spatial_scores(rate, occ, ok)
            idx = _spike_frame_indices(spike_times[i], t_win, keep=keep)
            sc["emp_rate_hz"] = float(idx.size / max(tot_occ_s, 1e-12))
            sc["map_mean_hz"] = rate_map_mean_hz(rate_raw, occ_raw, ok_raw)
            if hd_parent is not None and idx.size:
                sc["hd"] = _hd_mean_vector_length(hd_parent[idx])
            else:
                sc["hd"] = float("nan")
            if spd_win is not None and idx.size:
                sc["speed_r"] = _speed_tuning_r(spd_win[idx], spd_tr)
            else:
                sc["speed_r"] = float("nan")
            per_unit.append(sc)

        infos = [u["info"] for u in per_unit]
        spars = [u["spars"] for u in per_unit]
        peaks = [u["peak"] for u in per_unit]
        grids = [u["grid"] for u in per_unit]
        borders = [u["border"] for u in per_unit]
        pfs = [u["place_field_size"] for u in per_unit]
        hds = [u["hd"] for u in per_unit]
        spr = [u["speed_r"] for u in per_unit]
        map_means = [u["map_mean_hz"] for u in per_unit]
        emp_rates = [u["emp_rate_hz"] for u in per_unit]

        # Rate-map Hz consistency: |map_mean - emp_rate| / emp_rate within 5%
        rel_err = []
        for a, b in zip(map_means, emp_rates):
            if np.isfinite(a) and np.isfinite(b) and b > 1e-9:
                rel_err.append(abs(a - b) / b)
        rate_consist = float(np.nanmedian(rel_err)) if rel_err else float("nan")

        # Per-unit shuffle pass flags (grid20 only), including place-tuning (SI)
        g_flag = b_flag = h_flag = s_flag = pt_flag = [float("nan")] * n_elig
        run_shuf = bool(do_shuffles and scheme == "grid20")
        if run_shuf and n_elig:
            g_flag, b_flag, h_flag, s_flag, pt_flag = [], [], [], [], []
            dur = t_hi - t_lo
            min_shift = max(1.0, 0.05 * dur)
            hd_src = hd_parent
            for j, i in enumerate(eligible_idx):
                st = spike_times[i]
                thr_g = _shuffle_threshold(
                    st, t_win, p_win, edges_x, edges_y, occ_thresh, dt,
                    lambda r, o, ok: _grid_score(r, ok),
                    keep=keep, reachable=reachable, shared_ok=None,
                    n_shuffles=n_shuffles, rng=rng,
                )
                thr_b = _shuffle_threshold(
                    st, t_win, p_win, edges_x, edges_y, occ_thresh, dt,
                    lambda r, o, ok: _border_score(r, ok),
                    keep=keep, reachable=reachable, shared_ok=None,
                    n_shuffles=n_shuffles, rng=rng,
                )
                thr_si = _shuffle_threshold(
                    st, t_win, p_win, edges_x, edges_y, occ_thresh, dt,
                    spatial_info_bits_per_spike,
                    keep=keep, reachable=reachable, shared_ok=None,
                    n_shuffles=n_shuffles, rng=rng,
                )
                hd_null, sp_null = [], []
                for _ in range(int(n_shuffles)):
                    shift = float(rng.uniform(min_shift, dur - min_shift))
                    st_s = _circular_shift_spikes(st, t_lo, t_hi, shift)
                    if st_s.size == 0:
                        continue
                    idx = _spike_frame_indices(st_s, t_win, keep=keep)
                    if idx.size == 0:
                        continue
                    if hd_src is not None:
                        hd_null.append(_hd_mean_vector_length(hd_src[idx]))
                    if spd_win is not None and spd_tr is not None:
                        sp_null.append(_speed_tuning_r(spd_win[idx], spd_tr))
                thr_h = float(np.nanpercentile(hd_null, 95)) if len(hd_null) >= 20 else float("nan")
                thr_s = float(np.nanpercentile(sp_null, 95)) if len(sp_null) >= 20 else float("nan")
                g_flag.append(float(grids[j] > thr_g) if np.isfinite(grids[j]) and np.isfinite(thr_g) else float("nan"))
                b_flag.append(float(borders[j] > thr_b) if np.isfinite(borders[j]) and np.isfinite(thr_b) else float("nan"))
                h_flag.append(float(hds[j] > thr_h) if np.isfinite(hds[j]) and np.isfinite(thr_h) else float("nan"))
                s_flag.append(float(spr[j] > thr_s) if np.isfinite(spr[j]) and np.isfinite(thr_s) else float("nan"))
                pt_flag.append(
                    float(infos[j] > thr_si)
                    if np.isfinite(infos[j]) and np.isfinite(thr_si)
                    else float("nan")
                )

        # Composition-matched subsample indices into eligible list (matched arm)
        draw_idxs: list[np.ndarray | None]
        if (
            target_region_counts
            and elig_regions is not None
            and any(int(v) > 0 for v in target_region_counts.values())
        ):
            try:
                caps = {}
                for reg, need in target_region_counts.items():
                    avail = int((elig_regions == reg).sum())
                    caps[reg] = min(int(need), avail)
                if sum(caps.values()) <= 0:
                    draw_idxs = [None]
                else:
                    draw_idxs = composition_matched_subsample(
                        elig_regions, caps, rng=rng, n_subsamples=n_subsamples,
                    )
            except Exception:
                draw_idxs = [None]
        else:
            draw_idxs = [None]

        def _med_over_draws(vals, mean_not_median=False):
            parts = []
            for di in draw_idxs:
                if di is None:
                    arr = np.asarray(vals, float)
                else:
                    arr = np.asarray([vals[int(k)] for k in di], float)
                arr = arr[np.isfinite(arr)]
                if not arr.size:
                    continue
                parts.append(float(np.mean(arr) if mean_not_median else np.nanmedian(arr)))
            return float(np.nanmedian(parts)) if parts else float("nan")

        def _med_field_among_tuned(di=None):
            """Median field size among place-tuned units (SI > shuffle 95%)."""
            if not run_shuf:
                return float("nan")
            idxs = range(n_elig) if di is None else [int(k) for k in di]
            vals = []
            for k in idxs:
                if k >= len(pt_flag):
                    continue
                if pt_flag[k] == 1.0 and np.isfinite(pfs[k]):
                    vals.append(pfs[k])
            return float(np.nanmedian(vals)) if vals else float("nan")

        def _med_field_over_draws():
            parts = [_med_field_among_tuned(di) for di in draw_idxs]
            parts = [p for p in parts if np.isfinite(p)]
            return float(np.nanmedian(parts)) if parts else float("nan")

        base = dict(
            dataset=dataset,
            animal_or_seed=animal_or_seed,
            session=session,
            chunk=chunk,
            region_set=f"{region_set}:{scheme}",
        )
        tag = ""
        if scheme != "grid20":
            tag += f"_{scheme}"

        def mname(stem: str) -> str:
            return f"{stem}{tag}"

        rows.append(tidy_row(**base, metric=mname("spatial_info_median"), W=None, N=None, value=_med_over_draws(infos)))
        rows.append(tidy_row(**base, metric=mname("encoding_bits_per_spike_median"), W=None, N=None, value=_med_over_draws(infos)))
        rows.append(tidy_row(**base, metric=mname("peak_rate_median"), W=None, N=None, value=_med_over_draws(peaks)))
        rows.append(tidy_row(**base, metric=mname("grid_score_median"), W=None, N=None, value=_med_over_draws(grids)))
        rows.append(tidy_row(**base, metric=mname("border_score_median"), W=None, N=None, value=_med_over_draws(borders)))
        rows.append(tidy_row(**base, metric=mname("sparsity_median"), W=None, N=None, value=_med_over_draws(spars)))
        # Place-field size only among place-tuned units (SI > shuffle 95%).
        # coverage_bins_mean dropped (was identical to the old all-unit field size).
        rows.append(tidy_row(**base, metric=mname("place_field_size_median"), W=None, N=None, value=_med_field_over_draws()))

        if run_shuf:
            pass_grid = _med_over_draws(g_flag, mean_not_median=True)
            pass_border = _med_over_draws(b_flag, mean_not_median=True)
            pass_hd = _med_over_draws(h_flag, mean_not_median=True)
            pass_speed = _med_over_draws(s_flag, mean_not_median=True)
            pass_place = _med_over_draws(pt_flag, mean_not_median=True)
            rows.append(tidy_row(**base, metric="frac_grid_pass", W=None, N=None, value=pass_grid))
            rows.append(tidy_row(**base, metric="frac_border_pass", W=None, N=None, value=pass_border))
            rows.append(tidy_row(**base, metric="frac_hd_pass", W=None, N=None, value=pass_hd))
            rows.append(tidy_row(**base, metric="frac_speed_pass", W=None, N=None, value=pass_speed))
            rows.append(tidy_row(**base, metric="frac_place_tuned", W=None, N=None, value=pass_place))
            for stem, raw in (
                ("frac_grid_excess", pass_grid),
                ("frac_border_excess", pass_border),
                ("frac_hd_excess", pass_hd),
                ("frac_speed_excess", pass_speed),
                ("frac_place_tuned_excess", pass_place),
            ):
                rows.append(tidy_row(
                    **base, metric=stem, W=None, N=None,
                    value=float(raw - CHANCE_PASS) if np.isfinite(raw) else float("nan"),
                ))

        Ns = list(matched_Ns) if matched_Ns is not None else []
        for N in Ns:
            if n_elig < int(N):
                continue
            spars_ss, pfs_ss = [], []
            for _ in range(int(n_subsamples)):
                pick = rng.choice(n_elig, size=int(N), replace=False)
                spars_ss.append(float(np.nanmedian([spars[k] for k in pick])))
                if run_shuf:
                    tuned_pfs = [
                        pfs[k] for k in pick
                        if k < len(pt_flag) and pt_flag[k] == 1.0 and np.isfinite(pfs[k])
                    ]
                    pfs_ss.append(float(np.nanmedian(tuned_pfs)) if tuned_pfs else float("nan"))
            rows.append(tidy_row(**base, metric=mname("sparsity_median"), W=None, N=int(N), value=float(np.nanmedian(spars_ss))))
            if pfs_ss:
                rows.append(tidy_row(
                    **base, metric=mname("place_field_size_median"), W=None, N=int(N),
                    value=float(np.nanmedian([v for v in pfs_ss if np.isfinite(v)]))
                    if any(np.isfinite(v) for v in pfs_ss) else float("nan"),
                ))

        if scheme == "grid20":
            rows.append(tidy_row(**base0, metric="hd_mean_vector_length_median", W=None, N=None, value=_med_over_draws(hds)))
            rows.append(tidy_row(**base0, metric="speed_tuning_abs_r_median", W=None, N=None, value=_med_over_draws(spr)))
            rows.append(tidy_row(**base0, metric="rate_map_consistency_rel_err", W=None, N=None, value=rate_consist))

    return rows
