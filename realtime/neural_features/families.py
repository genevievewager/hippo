"""Causal feature-family computations (compact population-state statistics).

Mathematical definitions
------------------------
Counts (baseline)::
    n_i(t) = #spikes from unit i in [t - W, t)

Count dynamics::
    Δn_i(t) = n_i(t) - n_i(t_prev)
    optionally n'_i(t) = Δn_i(t) / Δt_update

Population statistics (global)::
    total_spikes(t) = Σ_i n_i(t)
    mean_unit_count(t) = total_spikes / N
    population_rate(t) = total_spikes / W
    fraction_active_units(t) = (# units with n_i ≥ 1) / N
    count_variance_across_units(t) = Var_i(n_i)

Regional statistics (per anatomical region R with |R| units)::
    same quantities restricted to units in R, plus
    regional_fraction_of_population_activity =
        regional_total / max(global_total, ε)

Within-region coactivity (internal bins of width ``coactivity_bin_dt``)::
    Build unit×bin counts A_{u,b} inside [t-W, t).
    For units in region R, form binary activity Ã = 1[A > 0].
    mean_pairwise_coactivity =
        mean_{i<j} PearsonCorr(A_i, A_j)  (safe value 0 if undefined)
    mean_pairwise_covariance =
        mean_{i<j} Cov(A_i, A_j)
    fraction_active_pairs =
        fraction of pairs that are co-active in ≥1 shared bin
    population_synchrony_index =
        Var_b(Σ_u A_{u,b}) / max(Σ_u Var_b(A_{u,b}), ε)
        (Var ratio; 1 ≈ independent Poisson sum under equal rates)

Cross-region coactivity::
    Using regional population traces ρ_R(b) = Σ_{u∈R} A_{u,b},
    zero-lag Cov / Corr between regions; undefined corr → 0 with validity flag.

Lagged coupling (causal only)::
    Corr(ρ_A[b − k], ρ_B[b]) for lag k ≥ 1 using only past bins.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from realtime.neural_features.types import FeatureSpec

EPS = 1e-12


def _safe_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    """Pearson correlation; returns (0.0, False) when undefined (zero variance)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or b.size < 2:
        return 0.0, False
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < EPS or sb < EPS:
        return 0.0, False
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return 0.0, False
    return c, True


def _safe_cov(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2:
        return 0.0
    c = float(np.cov(a, b, ddof=0)[0, 1])
    return c if np.isfinite(c) else 0.0


def build_count_specs(unit_ids: np.ndarray) -> list[FeatureSpec]:
    return [
        FeatureSpec(
            name=f"count_u{int(u)}",
            family="counts",
            unit_id=int(u),
            realtime_safe=True,
            computational_complexity="O(N)",
            requires_history=False,
            required_history_seconds=0.0,
        )
        for u in unit_ids
    ]


def append_counts(
    parts: list[np.ndarray],
    specs: list[FeatureSpec],
    counts: np.ndarray,
    unit_ids: np.ndarray,
) -> None:
    parts.append(np.asarray(counts, dtype=float))
    specs.extend(build_count_specs(unit_ids))


def append_count_dynamics(
    parts: list[np.ndarray],
    specs: list[FeatureSpec],
    counts: np.ndarray,
    prev_counts: np.ndarray | None,
    unit_ids: np.ndarray,
    *,
    update_dt: float,
    include_derivative: bool,
) -> None:
    """Δcount and optional derivative vs previous decoder update."""
    n_times, n_units = counts.shape
    if prev_counts is None:
        prev = np.zeros_like(counts)
        # First sample has no history — still emit zeros (deterministic).
        delta = counts.copy()
        delta[0] = 0.0
        if n_times > 1:
            delta[1:] = counts[1:] - counts[:-1]
    else:
        prev = np.asarray(prev_counts, dtype=float)
        if prev.shape != counts.shape:
            raise ValueError("prev_counts must match counts shape")
        delta = counts - prev

    parts.append(delta)
    for u in unit_ids:
        specs.append(
            FeatureSpec(
                name=f"delta_count_u{int(u)}",
                family="count_dynamics",
                unit_id=int(u),
                realtime_safe=True,
                computational_complexity="O(N)",
                requires_history=True,
                required_history_seconds=float(update_dt),
            )
        )
    if include_derivative:
        dt = max(float(update_dt), EPS)
        deriv = delta / dt
        parts.append(deriv)
        for u in unit_ids:
            specs.append(
                FeatureSpec(
                    name=f"count_deriv_u{int(u)}",
                    family="count_dynamics",
                    unit_id=int(u),
                    realtime_safe=True,
                    computational_complexity="O(N)",
                    requires_history=True,
                    required_history_seconds=float(update_dt),
                )
            )


def append_population_statistics(
    parts: list[np.ndarray],
    specs: list[FeatureSpec],
    counts: np.ndarray,
    decode_window: float,
) -> None:
    """
    Global population features.

    total_spikes              Σ_i n_i
    mean_unit_count           total / N
    population_rate           total / W   (spikes/s across the population)
    fraction_active_units     (# n_i ≥ 1) / N
    count_variance_across_units  sample variance of n_i across units
    """
    X = np.asarray(counts, dtype=float)
    n_units = max(X.shape[1], 1)
    w = max(float(decode_window), EPS)
    total = X.sum(axis=1)
    mean_c = total / n_units
    pop_rate = total / w
    frac_active = (X >= 1.0).sum(axis=1) / n_units
    var_c = X.var(axis=1) if X.shape[1] else np.zeros(len(X))
    block = np.column_stack([total, mean_c, pop_rate, frac_active, var_c])
    parts.append(block)
    names = (
        "total_spikes",
        "mean_unit_count",
        "population_rate",
        "fraction_active_units",
        "count_variance_across_units",
    )
    for name in names:
        specs.append(
            FeatureSpec(
                name=name,
                family="population_statistics",
                realtime_safe=True,
                computational_complexity="O(N)",
            )
        )


def append_regional_statistics(
    parts: list[np.ndarray],
    specs: list[FeatureSpec],
    counts: np.ndarray,
    region_labels: np.ndarray,
    region_order: list[str],
    decode_window: float,
) -> None:
    X = np.asarray(counts, dtype=float)
    labels = np.asarray(region_labels, dtype=object)
    w = max(float(decode_window), EPS)
    global_total = X.sum(axis=1)
    cols: list[np.ndarray] = []
    for region in region_order:
        idx = np.where(labels == region)[0]
        n_r = max(len(idx), 1)
        if len(idx) == 0:
            zeros = np.zeros(X.shape[0], dtype=float)
            block = np.column_stack([zeros, zeros, zeros, zeros, zeros])
        else:
            sub = X[:, idx]
            total = sub.sum(axis=1)
            mean_c = total / n_r
            rate = total / w
            frac = (sub >= 1.0).sum(axis=1) / n_r
            frac_pop = total / np.maximum(global_total, EPS)
            block = np.column_stack([total, mean_c, rate, frac, frac_pop])
        cols.append(block)
        prefix = f"region_{region}"
        for suffix in (
            "total_spikes",
            "mean_count",
            "population_rate",
            "fraction_active",
            "fraction_of_population_activity",
        ):
            specs.append(
                FeatureSpec(
                    name=f"{prefix}_{suffix}",
                    family="regional_statistics",
                    region=str(region),
                    realtime_safe=True,
                    computational_complexity="O(N)",
                )
            )
    if cols:
        parts.append(np.concatenate(cols, axis=1))


def _within_region_coactivity_vector(
    A: np.ndarray,
    *,
    allow_full_pairwise: bool,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """
    Compact within-region coactivity from unit×bin matrix A [n_units, n_bins].

    Returns values for:
      mean_pairwise_coactivity
      mean_pairwise_covariance
      fraction_active_pairs
      population_synchrony_index
    """
    validity: dict[str, Any] = {}
    n_units, n_bins = A.shape
    if n_units < 2 or n_bins < 2:
        return (
            np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
            [
                "mean_pairwise_coactivity",
                "mean_pairwise_covariance",
                "fraction_active_pairs",
                "population_synchrony_index",
            ],
            {"valid": False, "reason": "insufficient_units_or_bins"},
        )

    corrs: list[float] = []
    covs: list[float] = []
    active_pair = 0
    n_pairs = 0
    binary = (A > 0).astype(float)
    for i, j in combinations(range(n_units), 2):
        n_pairs += 1
        c, ok = _safe_corr(A[i], A[j])
        corrs.append(c if ok else 0.0)
        covs.append(_safe_cov(A[i], A[j]))
        if np.any(binary[i] * binary[j] > 0):
            active_pair += 1

    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    mean_cov = float(np.mean(covs)) if covs else 0.0
    frac_pairs = active_pair / max(n_pairs, 1)

    pop = A.sum(axis=0)
    var_pop = float(np.var(pop))
    sum_var = float(np.sum(np.var(A, axis=1)))
    sync = var_pop / max(sum_var, EPS)
    if not np.isfinite(sync):
        sync = 0.0

    values = [mean_corr, mean_cov, frac_pairs, sync]
    names = [
        "mean_pairwise_coactivity",
        "mean_pairwise_covariance",
        "fraction_active_pairs",
        "population_synchrony_index",
    ]
    validity["n_pairs"] = n_pairs
    validity["valid"] = True

    if allow_full_pairwise:
        # Optional O(N^2) emissions — disabled by default.
        for i, j in combinations(range(n_units), 2):
            c, _ = _safe_corr(A[i], A[j])
            values.append(c)
            names.append(f"pair_corr_{i}_{j}")

    return np.asarray(values, dtype=float), names, validity


def append_within_region_coactivity(
    parts: list[np.ndarray],
    specs: list[FeatureSpec],
    X_bins: np.ndarray,
    region_labels: np.ndarray,
    region_order: list[str],
    *,
    allow_full_pairwise: bool = False,
) -> list[dict[str, Any]]:
    """Append compact per-region coactivity features; return validity records."""
    labels = np.asarray(region_labels, dtype=object)
    n_times = X_bins.shape[0]
    region_mats: list[np.ndarray] = []
    validity_rows: list[dict[str, Any]] = []

    for region in region_order:
        idx = np.where(labels == region)[0]
        rows: list[np.ndarray] = []
        names: list[str] = []
        for t in range(n_times):
            if idx.size == 0:
                vec = np.zeros(4, dtype=float)
                names = [
                    "mean_pairwise_coactivity",
                    "mean_pairwise_covariance",
                    "fraction_active_pairs",
                    "population_synchrony_index",
                ]
                validity = {"valid": False, "reason": "empty_region"}
            else:
                vec, names, validity = _within_region_coactivity_vector(
                    X_bins[t, idx, :],
                    allow_full_pairwise=allow_full_pairwise,
                )
            rows.append(vec)
            if t == 0:
                validity_rows.append({"region": region, **validity})
        region_mats.append(np.vstack(rows) if rows else np.zeros((n_times, 4)))
        for name in names:
            specs.append(
                FeatureSpec(
                    name=f"within_{region}_{name}",
                    family="within_region_coactivity",
                    region=str(region),
                    realtime_safe=True,
                    computational_complexity="O(N_r^2 * B)",
                    requires_history=False,
                )
            )

    if region_mats:
        parts.append(np.concatenate(region_mats, axis=1))
    return validity_rows


def append_cross_region_coactivity(
    parts: list[np.ndarray],
    specs: list[FeatureSpec],
    region_traces: np.ndarray,
    region_order: list[str],
) -> list[dict[str, Any]]:
    """
    Zero-lag regional population coupling from traces [T, R, B].

    For each unordered pair (A, B):
      cross_cov, cross_corr, normalized_coactivity
    where normalized_coactivity = cov / (σ_A σ_B + ε) ≡ corr when defined.
    """
    T, R, B = region_traces.shape
    validity: list[dict[str, Any]] = []
    if R < 2 or B < 2:
        return validity

    pairs = list(combinations(range(R), 2))
    n_feats = 3 * len(pairs)
    out = np.zeros((T, n_feats), dtype=float)

    for pi, (i, j) in enumerate(pairs):
        ra, rb = region_order[i], region_order[j]
        for t in range(T):
            a = region_traces[t, i]
            b = region_traces[t, j]
            cov = _safe_cov(a, b)
            corr, ok = _safe_corr(a, b)
            sa = float(np.std(a))
            sb = float(np.std(b))
            norm = cov / max(sa * sb, EPS) if (sa >= EPS and sb >= EPS) else 0.0
            if not np.isfinite(norm):
                norm = 0.0
            base = 3 * pi
            out[t, base] = cov
            out[t, base + 1] = corr
            out[t, base + 2] = norm
            if t == 0:
                validity.append({
                    "region_a": ra,
                    "region_b": rb,
                    "corr_valid": ok,
                })
        for suffix in ("cov", "corr", "normalized_coactivity"):
            specs.append(
                FeatureSpec(
                    name=f"cross_{ra}__{rb}_{suffix}",
                    family="cross_region_coactivity",
                    region=str(ra),
                    region_b=str(rb),
                    realtime_safe=True,
                    computational_complexity="O(R^2 * B)",
                )
            )
    parts.append(out)
    return validity


def append_lagged_coupling(
    parts: list[np.ndarray],
    specs: list[FeatureSpec],
    region_traces: np.ndarray,
    region_order: list[str],
    lags: tuple[int, ...],
    *,
    coactivity_bin_dt: float,
) -> list[dict[str, Any]]:
    """
    Causal lagged coupling: Corr(ρ_A[b-k], ρ_B[b]) for lag k ≥ 1.

    Only past bins of region A are used to predict the current bin of B.
    Directed pairs (A → B) with A ≠ B are emitted.
    """
    T, R, B = region_traces.shape
    lags = tuple(int(k) for k in lags if int(k) >= 1)
    validity: list[dict[str, Any]] = []
    if R < 2 or not lags or B <= max(lags):
        # Still emit deterministic zeros with specs so dimensions stay stable
        # when B is too short for requested lags.
        n_feats = R * (R - 1) * len(lags) if lags else 0
        if n_feats:
            parts.append(np.zeros((T, n_feats), dtype=float))
            for i, ra in enumerate(region_order):
                for j, rb in enumerate(region_order):
                    if i == j:
                        continue
                    for k in lags:
                        specs.append(
                            FeatureSpec(
                                name=f"lagged_{ra}_to_{rb}_lag{k}",
                                family="lagged_coupling",
                                region=str(ra),
                                region_b=str(rb),
                                lag_bins=int(k),
                                realtime_safe=True,
                                computational_complexity="O(R^2 * B * L)",
                                requires_history=True,
                                required_history_seconds=float(k) * float(coactivity_bin_dt),
                            )
                        )
                        if T > 0:
                            validity.append({
                                "region_a": ra, "region_b": rb, "lag": k,
                                "valid": False, "reason": "insufficient_bins",
                            })
        return validity

    cols: list[np.ndarray] = []
    for i, ra in enumerate(region_order):
        for j, rb in enumerate(region_order):
            if i == j:
                continue
            for k in lags:
                series = np.zeros(T, dtype=float)
                for t in range(T):
                    a = region_traces[t, i, : B - k]
                    b = region_traces[t, j, k:]
                    corr, ok = _safe_corr(a, b)
                    series[t] = corr
                    if t == 0:
                        validity.append({
                            "region_a": ra, "region_b": rb, "lag": k,
                            "valid": ok,
                        })
                cols.append(series)
                specs.append(
                    FeatureSpec(
                        name=f"lagged_{ra}_to_{rb}_lag{k}",
                        family="lagged_coupling",
                        region=str(ra),
                        region_b=str(rb),
                        lag_bins=int(k),
                        realtime_safe=True,
                        computational_complexity="O(R^2 * B * L)",
                        requires_history=True,
                        required_history_seconds=float(k) * float(coactivity_bin_dt),
                    )
                )
    if cols:
        parts.append(np.column_stack(cols))
    return validity
