"""Draft diagnostic PNGs for M2 (ignored outputs; rough styling OK)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analysis.real_quadrant.adapter import _room_boundary

REPO_ROOT = Path(__file__).resolve().parents[2]
M2_DIR = REPO_ROOT / "outputs" / "real_quadrant" / "m2" / "room_A" / "all_methods"
PLOT_DIR = M2_DIR / "diagnostics"
WINDOW_S = 120.0


def _room_outline_local(session_name: str, room: str = "A") -> np.ndarray:
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    with open(root / session_name / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cx, cy, _w, _h, _xmin, _ymin = _room_boundary(cfg, room)
    mr = cfg["preprocessing"]["map_rooms"]
    idx_map = {int(k): str(v) for k, v in (mr.get("index") or {}).items()}
    room_idx = next(i for i, lab in idx_map.items() if lab == room)
    bdf = pd.DataFrame(cfg["preprocessing"]["boundary"])
    sub = bdf[bdf["Room"].astype(int) == int(room_idx)]
    xy = np.column_stack([
        sub["X"].to_numpy(dtype=float) - cx,
        sub["Y"].to_numpy(dtype=float) - cy,
    ])
    if len(xy) and not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    return xy


def main() -> int:
    report = json.loads((M2_DIR / "m2_report.json").read_text())
    sel = json.loads((M2_DIR / "session_selection.json").read_text())
    session = sel["selected_session"]
    contract = Path(report["figure_contract_dir"])
    arrays = np.load(contract / "arrays.npz")
    preds = np.load(contract / "predictions.npz")
    a13 = json.loads((contract / "a13.json").read_text())

    decode_times = arrays["decode_times"]
    y = arrays["y"]
    eval_mask = arrays["eval_mask"]
    y_floor = arrays["y_floor"].reshape(-1)
    floor_med = float(report["floor_median_cm"])

    methods = [r["method"] for r in report["methods"]]
    best = min(report["methods"], key=lambda r: float(r["ridge_median_cm"]))
    best_m = best["method"]

    t_eval = decode_times[eval_mask]
    y_true = y[eval_mask]
    outline = _room_outline_local(session, "A")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Predicted vs true path on room A outline (raw + best)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, method, title in (
        (axes[0], "raw", "raw"),
        (axes[1], best_m, f"best ({best_m})"),
    ):
        pred = preds[f"pred_{method}_ridge"]
        ax.plot(outline[:, 0], outline[:, 1], "k-", lw=1.0, alpha=0.7)
        ax.plot(y_true[:, 0], y_true[:, 1], "-", color="0.55", lw=0.6, label="true")
        ax.plot(pred[:, 0], pred[:, 1], "-", color="C1", lw=0.6, alpha=0.85, label="pred")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x (cm, room-local)")
        ax.set_ylabel("y (cm, room-local)")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("M2 room A: Ridge predicted vs true path")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "path_raw_and_best.png", dpi=120)
    plt.close(fig)

    # 2) x(t), y(t) over a 2-minute test window
    t0 = float(t_eval[0])
    win = (t_eval >= t0) & (t_eval < t0 + WINDOW_S)
    if win.sum() < 10:
        # Fall back to middle of eval
        mid = t_eval[len(t_eval) // 2]
        win = (t_eval >= mid - WINDOW_S / 2) & (t_eval < mid + WINDOW_S / 2)
    tw = t_eval[win] - t_eval[win][0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, dim, lab in ((axes[0], 0, "x"), (axes[1], 1, "y")):
        ax.plot(tw, y_true[win, dim], "k-", lw=1.0, label="true")
        ax.plot(
            tw, preds["pred_raw_ridge"][win, dim],
            color="C0", lw=0.9, label="raw",
        )
        ax.plot(
            tw, preds[f"pred_{best_m}_ridge"][win, dim],
            color="C1", lw=0.9, label=best_m,
        )
        ax.set_ylabel(f"{lab} (cm)")
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time in window (s)")
    fig.suptitle(f"M2: true vs predicted over {WINDOW_S:.0f}s test window")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "xy_traces_2min.png", dpi=120)
    plt.close(fig)

    # 3) Error distribution per method with chance floor
    fig, ax = plt.subplots(figsize=(9, 5))
    data = []
    labels = []
    for m in methods:
        err = np.linalg.norm(preds[f"pred_{m}_ridge"] - y_true, axis=1)
        data.append(err)
        labels.append(m)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(floor_med, color="0.3", ls="--", lw=1.2, label=f"chance floor med={floor_med:.1f} cm")
    ax.set_ylabel("Euclidean error (cm)")
    ax.set_title("M2 Ridge error distribution (eval)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "error_distributions.png", dpi=120)
    plt.close(fig)

    # 4) A13 null vs observed per method
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(methods))
    obs = [float(next(r for r in report["methods"] if r["method"] == m)["ridge_median_cm"]) for m in methods]
    # A13 stores median(null_error - floor); reconstruct null median ≈ delta + floor
    null_med = []
    for m in methods:
        rec = a13.get(m) or {}
        delta = rec.get("ridge_median_minus_floor")
        if delta is None or (isinstance(delta, float) and not np.isfinite(delta)):
            null_med.append(np.nan)
        else:
            null_med.append(float(delta) + floor_med)
    width = 0.35
    ax.bar(xs - width / 2, obs, width, label="observed Ridge median", color="C0")
    ax.bar(xs + width / 2, null_med, width, label="A13 null Ridge median (approx)", color="C3")
    ax.axhline(floor_med, color="0.3", ls="--", lw=1.0, label="chance floor")
    ax.set_xticks(xs)
    ax.set_xticklabels(methods)
    ax.set_ylabel("median error (cm)")
    ax.set_title("M2 A13 null vs observed (Ridge)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "a13_null_vs_observed.png", dpi=120)
    plt.close(fig)

    print(str(PLOT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
