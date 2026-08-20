"""Landmark-count tradeoff benchmark for diffusion_nystrom.

Compares decoding performance, embedding fidelity, latency, memory, and
realtime qualification across landmark counts. Decode window length is
independent of computation latency: a 500 ms spike window can still emit
a prediction every 25 ms if the operation finishes in a few milliseconds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from realtime.latency_profiler import DEFAULT_OPERATION_DEADLINE_MS, qualify_latency_values
from realtime.manifolds.diffusion_nystrom import (
    DEFAULT_LANDMARK_METHOD,
    DEFAULT_N_COMPONENTS,
    LANDMARK_COUNT_BENCHMARK,
    DiffusionNystrom,
)
from realtime.manifolds.isomap_metrics import trustworthiness_at_ks


def run_diffusion_landmark_benchmark_from_experiment(
    experiment_dir: Path,
    *,
    output_dir: Path | None = None,
    spike_source: str = "sorted",
    decode_window_s: float = 0.250,
    train_frac: float = 0.70,
    landmark_counts: tuple[int, ...] | list[int] | None = None,
    n_components: int = DEFAULT_N_COMPONENTS,
    landmark_method: str = DEFAULT_LANDMARK_METHOD,
    local_scale_k: int = 10,
    alpha: float = 1.0,
    diffusion_time: int | float = 1,
    random_state: int = 42,
    operation_deadline_ms: float = DEFAULT_OPERATION_DEADLINE_MS,
) -> pd.DataFrame:
    """Load causal train/test counts from a simulation directory and benchmark."""
    from realtime.data_loading import load_simulation_data, make_decode_times
    from realtime.decoding_targets import align_extended_behavior_to_decoder_times
    from realtime.spike_features import build_causal_spike_matrix
    from realtime.timing import extract_behavior_times, resolve_update_dt_s
    from realtime.train_decoder import causal_train_test_split

    experiment_dir = Path(experiment_dir)
    data = load_simulation_data(experiment_dir, spike_source)
    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = resolve_update_dt_s(
        data["summary"], derive_from_behavior=True, behavior_times=behavior_times,
    )
    decode_times = make_decode_times(
        data["session_duration"], decode_window_s, update_dt, behavior_times=behavior_times,
    )
    X = build_causal_spike_matrix(
        data["spikes_df"], data["unit_ids"], decode_times, decode_window_s,
    )
    train_mask, test_mask = causal_train_test_split(decode_times, train_frac)
    aligned = align_extended_behavior_to_decoder_times(
        data["behavior_df"], decode_times, data["summary"],
    )
    y = aligned[["x", "y"]].to_numpy(dtype=float)
    out = Path(output_dir) if output_dir else (
        experiment_dir / "decoder_comparison" / spike_source / "diffusion_landmark_benchmark"
    )
    return run_diffusion_landmark_benchmark(
        X[train_mask],
        X[test_mask],
        y_train=y[train_mask],
        y_test=y[test_mask],
        landmark_counts=landmark_counts or LANDMARK_COUNT_BENCHMARK,
        n_components=n_components,
        landmark_method=landmark_method,
        local_scale_k=local_scale_k,
        alpha=alpha,
        diffusion_time=diffusion_time,
        random_state=random_state,
        operation_deadline_ms=operation_deadline_ms,
        output_dir=out,
    )


def _position_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        return float(np.mean(np.abs(y_pred.ravel() - y_true.ravel())))
    return float(np.mean(np.hypot(y_pred[:, 0] - y_true[:, 0], y_pred[:, 1] - y_true[:, 1])))


def run_diffusion_landmark_benchmark(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    y_train: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    landmark_counts: tuple[int, ...] | list[int] = LANDMARK_COUNT_BENCHMARK,
    n_components: int = DEFAULT_N_COMPONENTS,
    landmark_method: str = DEFAULT_LANDMARK_METHOD,
    local_scale_k: int = 10,
    alpha: float = 1.0,
    diffusion_time: int | float = 1,
    random_state: int = 42,
    n_latency_repeats: int = 80,
    operation_deadline_ms: float = DEFAULT_OPERATION_DEADLINE_MS,
    output_dir: Path | None = None,
    transform: str = "sqrt_counts",
) -> pd.DataFrame:
    """Fit diffusion_nystrom at each landmark count and record the tradeoff.

    Returns a table with decoding, fidelity, latency, memory, and
    ``realtime_qualified`` columns. Writes CSV + JSON when ``output_dir`` is set.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    rows: list[dict[str, Any]] = []
    n_probe = min(n_latency_repeats, max(1, len(X_test)))
    probe = X_test[:n_probe]

    for n_landmarks in landmark_counts:
        n_landmarks = int(n_landmarks)
        if n_landmarks < 2:
            continue
        enc = DiffusionNystrom(
            n_landmarks=n_landmarks,
            landmark_method=landmark_method,
            n_components=n_components,
            local_scale_k=local_scale_k,
            alpha=alpha,
            diffusion_time=diffusion_time,
            random_state=random_state,
            transform=transform,
        )
        t_fit0 = time.perf_counter()
        enc.fit(X_train)
        fit_s = float(time.perf_counter() - t_fit0)
        Z_train = enc.transform(X_train)
        Z_test = enc.transform(X_test)

        fidelity: dict[str, Any] = {}
        try:
            n_tw = min(800, len(X_train))
            idx = np.linspace(0, len(X_train) - 1, n_tw).astype(int)
            Xp = enc._transform_preprocess(X_train[idx])
            fidelity.update(trustworthiness_at_ks(Xp, Z_train[idx], neighbor_ks=(5, 10)))
        except Exception as exc:  # noqa: BLE001
            fidelity["trustworthiness_error"] = str(exc)
        try:
            fidelity.update(enc.nystrom_landmark_consistency())
        except Exception as exc:  # noqa: BLE001
            fidelity["nystrom_consistency_error"] = str(exc)

        decode: dict[str, Any] = {}
        if y_train is not None and y_test is not None:
            y_tr = np.asarray(y_train)
            y_te = np.asarray(y_test)
            model = Ridge(alpha=1.0)
            model.fit(Z_train, y_tr)
            pred = model.predict(Z_test)
            decode["decode_r2"] = float(r2_score(y_te, pred, multioutput="uniform_average"))
            if y_te.ndim == 2 and y_te.shape[1] >= 2:
                decode["mean_position_error_cm"] = _position_error(y_te, pred)

        # Per-update embedding latency (transform_one).
        for i in range(min(5, len(probe))):
            enc.transform_one(probe[i])
        lat = []
        for i in range(len(probe)):
            t0 = time.perf_counter_ns()
            enc.transform_one(probe[i])
            lat.append((time.perf_counter_ns() - t0) / 1e6)
        qual = qualify_latency_values(lat, deadline_ms=operation_deadline_ms)
        ood = enc.query_diagnostics_batch(X_test)
        ood_frac = float(np.mean(ood["ood_flag"])) if len(ood["ood_flag"]) else 0.0

        rows.append({
            "embedding_type": "diffusion_nystrom",
            "n_landmarks": int(enc.n_landmarks_fitted_ or n_landmarks),
            "n_landmarks_requested": n_landmarks,
            "landmark_method": landmark_method,
            "n_components": int(enc.actual_n_components_ or n_components),
            "local_scale_k": local_scale_k,
            "alpha": alpha,
            "diffusion_time": diffusion_time,
            "fit_seconds": fit_s,
            "memory_bytes": enc.memory_bytes_,
            "memory_mb": (
                None if enc.memory_bytes_ is None else float(enc.memory_bytes_) / 1e6
            ),
            "mean_embedding_latency_ms": qual["mean_ms"],
            "median_embedding_latency_ms": qual["median_ms"],
            "p95_embedding_latency_ms": qual["p95_ms"],
            "p99_embedding_latency_ms": qual["p99_ms"],
            "max_embedding_latency_ms": qual["max_ms"],
            "deadline_ms": operation_deadline_ms,
            "deadline_miss_count": qual["deadline_miss_count"],
            "deadline_miss_pct": qual["deadline_miss_pct"],
            "realtime_qualified": qual["realtime_qualified"],
            "headroom_ms": qual["headroom_ms"],
            "ood_fraction_test": ood_frac,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            **fidelity,
            **decode,
        })

    df = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "diffusion_landmark_benchmark.csv", index=False)
        with open(output_dir / "diffusion_landmark_benchmark.json", "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2, default=str)
        _write_tradeoff_plot(df, output_dir / "diffusion_landmark_tradeoff.png")
    return df


def _write_tradeoff_plot(df: pd.DataFrame, path: Path) -> None:
    if df.empty or "n_landmarks" not in df.columns:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    x = df["n_landmarks"].to_numpy()
    ax = axes[0]
    if "mean_position_error_cm" in df.columns and df["mean_position_error_cm"].notna().any():
        ax.plot(x, df["mean_position_error_cm"], "o-", label="position error (cm)")
        ax.set_ylabel("mean position error (cm)")
    elif "decode_r2" in df.columns and df["decode_r2"].notna().any():
        ax.plot(x, df["decode_r2"], "o-", label="decode R²")
        ax.set_ylabel("decode R²")
    elif "trustworthiness" in df.columns:
        ax.plot(x, df["trustworthiness"], "o-", label="trustworthiness")
        ax.set_ylabel("trustworthiness")
    ax.set_xlabel("landmarks M")
    ax.set_title("Decoding / fidelity")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(x, df["mean_embedding_latency_ms"], "o-", label="mean")
    ax.plot(x, df["p95_embedding_latency_ms"], "s--", label="P95")
    ax.plot(x, df["p99_embedding_latency_ms"], "^:", label="P99")
    deadline = float(df["deadline_ms"].iloc[0]) if "deadline_ms" in df.columns else 25.0
    ax.axhline(deadline, color="k", ls="--", lw=1, label=f"{deadline:.0f} ms deadline")
    ax.set_xlabel("landmarks M")
    ax.set_ylabel("embedding latency (ms)")
    ax.set_title("Latency vs landmark count")
    ax.legend(frameon=False)
    fig.suptitle(
        "Decode window ≠ compute latency  (predictions can still emit every 25 ms)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
