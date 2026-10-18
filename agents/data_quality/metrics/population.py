"""Population metrics: noise corr, synchrony, PR, PCA spectrum (matched-N)."""
from __future__ import annotations

from typing import Any

import numpy as np

from agents.data_quality.metrics.schema import tidy_row

MATCHED_NS_DEFAULT = (10, 20, 40)
N_SUBSAMPLES_DEFAULT = 20


def _counts(spike_times, t0, t1, W):
    edges = np.arange(t0, t1 + 1e-9, W)
    if edges.size < 3:
        return np.zeros((0, len(spike_times)))
    X = np.zeros((edges.size - 1, len(spike_times)), dtype=float)
    for j, st in enumerate(spike_times):
        st = np.asarray(st, float)
        X[:, j] = np.histogram(st, bins=edges)[0]
    return X


def participation_ratio(X):
    if X.size == 0 or X.shape[0] < 2 or X.shape[1] < 2:
        return float("nan")
    Xc = X - X.mean(0, keepdims=True)
    c = np.cov(Xc, rowvar=False)
    if np.ndim(c) == 0:
        return float("nan")
    ev = np.linalg.eigvalsh(c)
    ev = ev[ev > 0]
    if ev.size == 0:
        return float("nan")
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def _noise_corr_mean(X: np.ndarray, positions: np.ndarray | None) -> float:
    if X.shape[1] < 2 or X.shape[0] < 5:
        return float("nan")
    R = X.astype(float)
    if positions is not None and len(positions) == len(X):
        pos = np.asarray(positions, float)
        if pos.ndim == 1:
            pos = pos[:, None]
        A = np.column_stack([np.ones(len(pos)), pos[:, 0], pos[:, 1], pos[:, 0] ** 2, pos[:, 1] ** 2])
        for j in range(R.shape[1]):
            y = R[:, j]
            try:
                coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                R[:, j] = y - A @ coef
            except np.linalg.LinAlgError:
                pass
    C = np.corrcoef(R.T)
    iu = np.triu_indices(C.shape[0], 1)
    return float(np.nanmean(C[iu]))


def _pca_fracs(X: np.ndarray, n: int = 5) -> list[float]:
    Xc = X - X.mean(0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        var = s ** 2
        var = var / max(var.sum(), 1e-12)
        out = [float(v) for v in var[:n]]
        while len(out) < n:
            out.append(float("nan"))
        return out
    except np.linalg.LinAlgError:
        return [float("nan")] * n


def population_metric_rows(
    spike_times,
    *,
    dataset,
    animal_or_seed,
    session,
    chunk,
    region_set,
    t0,
    t1,
    W_grid=(0.05, 0.1, 0.25, 0.5, 1.0),
    k_frac=0.2,
    positions=None,
    position_times=None,
    matched_Ns=MATCHED_NS_DEFAULT,
    n_subsamples: int = N_SUBSAMPLES_DEFAULT,
    rng=None,
) -> list[dict[str, Any]]:
    """PR / PCA / synchrony / noise-corr on matched-N subsamples (SPEC §3.3)."""
    rng = rng or np.random.default_rng(0)
    n_units = len(spike_times)
    # max common: use all units as an extra N when larger than 40
    Ns = [int(n) for n in matched_Ns if int(n) <= n_units]
    if n_units >= 5 and (not Ns or n_units not in Ns):
        Ns = list(Ns) + [n_units]
    rows = []
    for W in W_grid:
        X_full = _counts(spike_times, t0, t1, W)
        base = dict(
            dataset=dataset,
            animal_or_seed=animal_or_seed,
            session=session,
            chunk=chunk,
            region_set=region_set,
        )
        if X_full.shape[0] < 5:
            continue
        pos_aligned = None
        if positions is not None and position_times is not None:
            pt = np.asarray(position_times, float)
            edges = np.arange(t0, t1 + 1e-9, W)
            centers = 0.5 * (edges[:-1] + edges[1:])
            if len(centers) == X_full.shape[0]:
                idx = np.clip(np.searchsorted(pt, centers) - 1, 0, len(pt) - 1)
                pos_aligned = np.asarray(positions, float)[idx]
        for N in Ns:
            prs, syncs, ncs, pca_lists = [], [], [], []
            n_rep = int(n_subsamples) if N < n_units else 1
            for _ in range(n_rep):
                if N < n_units:
                    idx = rng.choice(n_units, size=N, replace=False)
                else:
                    idx = np.arange(n_units)
                X = X_full[:, idx]
                prs.append(participation_ratio(X))
                active = (X > 0).sum(1) / max(X.shape[1], 1)
                syncs.append(float(np.mean(active >= k_frac)))
                ncs.append(_noise_corr_mean(X, pos_aligned))
                pca_lists.append(_pca_fracs(X))
            rows.append(
                tidy_row(**base, metric="participation_ratio", W=float(W), N=int(N), value=float(np.nanmedian(prs)))
            )
            rows.append(
                tidy_row(**base, metric="synchrony_frac", W=float(W), N=int(N), value=float(np.nanmedian(syncs)))
            )
            rows.append(
                tidy_row(**base, metric="noise_corr_mean", W=float(W), N=int(N), value=float(np.nanmedian(ncs)))
            )
            # also mean pairwise corr on matched N
            mp = []
            for _ in range(n_rep):
                if N < n_units:
                    idx = rng.choice(n_units, size=N, replace=False)
                else:
                    idx = np.arange(n_units)
                X = X_full[:, idx]
                if X.shape[1] >= 2:
                    C = np.corrcoef(X.T)
                    iu = np.triu_indices(C.shape[0], 1)
                    mp.append(float(np.nanmean(C[iu])))
            if mp:
                rows.append(
                    tidy_row(**base, metric="mean_pairwise_corr", W=float(W), N=int(N), value=float(np.nanmedian(mp)))
                )
            pca_arr = np.asarray(pca_lists, float)
            for i in range(5):
                rows.append(
                    tidy_row(
                        **base,
                        metric=f"pca_var_frac_{i+1}",
                        W=float(W),
                        N=int(N),
                        value=float(np.nanmedian(pca_arr[:, i])),
                    )
                )
    return rows
