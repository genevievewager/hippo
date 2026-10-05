"""M2 diagnostic PNGs + summary table (ignored outputs)."""

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
from analysis.real_quadrant.posthoc_smooth import (
    CAUSAL_METHODS,
    GPFA_LABEL,
    OFFLINE_METHODS,
    fraction_outside_boundary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
M2_DIR = REPO_ROOT / "outputs" / "real_quadrant" / "m2" / "room_A" / "all_methods"
PLOT_DIR = M2_DIR / "diagnostics"
WINDOW_S = 120.0
GAP_BREAK_S = 0.25


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


def _segment_slices(times: np.ndarray, gap_s: float = GAP_BREAK_S) -> list[slice]:
    """Contiguous slices broken where ``diff(times) > gap_s``."""
    t = np.asarray(times, dtype=float)
    if len(t) == 0:
        return []
    dt = np.diff(t)
    breaks = np.where(~np.isfinite(dt) | (dt > float(gap_s)))[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(t)]
    return [slice(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def _plot_xy_gapless(ax, x, y, *, color, lw, label=None, alpha=1.0) -> None:
    """Plot (x,y) polyline, breaking at NaNs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return
    # Break runs of validity
    idx = np.where(valid)[0]
    gaps = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps + 1, len(idx)]
    first = True
    for s, e in zip(starts, ends):
        ii = idx[s:e]
        ax.plot(
            x[ii], y[ii], "-", color=color, lw=lw, alpha=alpha,
            label=label if first else None,
        )
        first = False


def _plot_series_gapless(ax, t, y, *, color, lw, label=None) -> None:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    for sl in _segment_slices(t):
        ax.plot(t[sl], y[sl], "-", color=color, lw=lw, label=label)
        label = None


def _best_causal_method(methods: list[dict]) -> dict:
    """Best causal method for path/trace panels — excludes offline GPFA only."""
    causal = [
        m for m in methods
        if m["method"] not in OFFLINE_METHODS
        and m.get("causal", m["method"] != "gpfa")
        and m.get("ridge_median_cm") is not None
    ]
    if not causal:
        raise RuntimeError("no causal methods in report")
    return min(causal, key=lambda r: float(r["ridge_median_cm"]))


def _median_error_window(
    t_eval: np.ndarray,
    err: np.ndarray,
    window_s: float = WINDOW_S,
) -> tuple[np.ndarray, str]:
    """Pick the 2-min eval window whose median error is the median across windows."""
    t_eval = np.asarray(t_eval, dtype=float)
    err = np.asarray(err, dtype=float)
    if len(t_eval) < 10:
        return np.ones(len(t_eval), dtype=bool), "full_eval_fallback"
    t0 = float(t_eval[0])
    t1 = float(t_eval[-1])
    if t1 - t0 < window_s:
        return np.ones(len(t_eval), dtype=bool), "full_eval_short_session"
    # Candidate starts every update step present in eval.
    starts = t_eval[t_eval <= t1 - window_s]
    if len(starts) == 0:
        return np.ones(len(t_eval), dtype=bool), "full_eval_fallback"
    # Subsample starts to keep this cheap (~every 1s).
    step = max(1, int(round(1.0 / max(np.median(np.diff(t_eval)), 1e-3))))
    starts = starts[::step]
    window_meds = []
    window_masks = []
    for ts in starts:
        m = (t_eval >= ts) & (t_eval < ts + window_s)
        if int(m.sum()) < 20:
            continue
        window_meds.append(float(np.median(err[m])))
        window_masks.append(m)
    if not window_meds:
        mid = t_eval[len(t_eval) // 2]
        m = (t_eval >= mid - window_s / 2) & (t_eval < mid + window_s / 2)
        return m, "midpoint_fallback"
    order = np.argsort(window_meds)
    pick = order[len(order) // 2]
    rule = (
        f"2-min window whose median error is the median across "
        f"{len(window_meds)} candidate windows"
    )
    return window_masks[int(pick)], rule


def _method_label(m: dict) -> str:
    name = m["method"]
    if name == "gpfa":
        return f"gpfa ({GPFA_LABEL})"
    if name == "gpfa_causal":
        return "gpfa_causal (filter)"
    return name


def build_table(report: dict, outline: np.ndarray, preds, y_true: np.ndarray) -> list[dict]:
    rows = []
    for m in report["methods"]:
        name = m["method"]
        key = f"pred_{name}_ridge"
        if key not in preds:
            continue
        pred = np.asarray(preds[key], dtype=float)
        frac_out = fraction_outside_boundary(pred, outline)
        rows.append({
            "method": name,
            "display_name": _method_label(m),
            "causal": bool(m.get("causal", name != "gpfa")),
            "normalized_error": m.get("normalized_error"),
            "ridge_median_cm": m.get("ridge_median_cm"),
            "effective_lag_s": m.get("effective_lag_s"),
            "frac_outside_boundary": frac_out,
            "ema_tau_s": m.get("ema_tau_s"),
            "a13_status": m.get("a13_status"),
        })
    return rows


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
    floor_med = float(report["floor_median_cm"])

    best = _best_causal_method(report["methods"])
    best_m = best["method"]
    assert best_m != "gpfa"

    t_eval = decode_times[eval_mask]
    y_true = y[eval_mask]
    outline = _room_outline_local(session, "A")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    table = build_table(report, outline, preds, y_true)
    (M2_DIR / "m2_table.json").write_text(json.dumps({"rows": table}, indent=2) + "\n")
    # Human-readable markdown table
    lines = [
        "| method | causal | normalized_error | effective_lag_s | frac_outside_boundary | ema_tau_s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in table:
        tau = r["ema_tau_s"]
        tau_s = "" if tau is None else f"{tau:g}"
        ne = r["normalized_error"]
        ne_s = "" if ne is None else f"{ne:.3f}"
        lag = r.get("effective_lag_s")
        lag_s = "" if lag is None or not np.isfinite(lag) else f"{float(lag):.2f}"
        fo = r["frac_outside_boundary"]
        fo_s = "" if fo is None or not np.isfinite(fo) else f"{fo:.3f}"
        lines.append(
            f"| {r['display_name']} | {str(r['causal']).lower()} | {ne_s} | {lag_s} | {fo_s} | {tau_s} |"
        )
    (PLOT_DIR / "m2_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # 1) Path: raw + best causal, with outline and gap breaks
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, method, title in (
        (axes[0], "raw", "raw"),
        (axes[1], best_m, f"best causal ({best_m})"),
    ):
        pred = preds[f"pred_{method}_ridge"]
        ax.plot(outline[:, 0], outline[:, 1], "k-", lw=1.0, alpha=0.8, label="arena")
        # Break true/pred by time gaps on eval
        for sl in _segment_slices(t_eval):
            _plot_xy_gapless(
                ax, y_true[sl, 0], y_true[sl, 1],
                color="0.55", lw=0.6, label="true" if sl.start == 0 else None,
            )
            _plot_xy_gapless(
                ax, pred[sl, 0], pred[sl, 1],
                color="C1", lw=0.6, alpha=0.85,
                label="pred" if sl.start == 0 else None,
            )
        frac = next(r for r in table if r["method"] == method)["frac_outside_boundary"]
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}\nfrac outside boundary={frac:.3f}")
        ax.set_xlabel("x (cm, room-local)")
        ax.set_ylabel("y (cm, room-local)")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("M2 room A: Ridge predicted vs true path (gaps broken)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "path_raw_and_best.png", dpi=120)
    plt.close(fig)

    # 2) x(t), y(t) over median-error 2-min window
    err_best = np.linalg.norm(preds[f"pred_{best_m}_ridge"] - y_true, axis=1)
    win, win_rule = _median_error_window(t_eval, err_best, WINDOW_S)
    tw = t_eval[win] - float(t_eval[win][0])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, dim, lab in ((axes[0], 0, "x"), (axes[1], 1, "y")):
        _plot_series_gapless(ax, tw, y_true[win, dim], color="k", lw=1.0, label="true")
        _plot_series_gapless(
            ax, tw, preds["pred_raw_ridge"][win, dim], color="C0", lw=0.9, label="raw",
        )
        _plot_series_gapless(
            ax, tw, preds[f"pred_{best_m}_ridge"][win, dim],
            color="C1", lw=0.9, label=best_m,
        )
        ax.set_ylabel(f"{lab} (cm)")
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time in window (s)")
    fig.suptitle(f"M2: {WINDOW_S:.0f}s test window — {win_rule}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "xy_traces_2min.png", dpi=120)
    plt.close(fig)

    # 3) Error distributions (causal methods + gpfa labeled)
    plot_methods = [r["method"] for r in report["methods"] if f"pred_{r['method']}_ridge" in preds]
    fig, ax = plt.subplots(figsize=(11, 5))
    data, labels = [], []
    for m in plot_methods:
        err = np.linalg.norm(preds[f"pred_{m}_ridge"] - y_true, axis=1)
        data.append(err)
        row = next(r for r in report["methods"] if r["method"] == m)
        labels.append(_method_label(row))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(floor_med, color="0.3", ls="--", lw=1.2, label=f"chance floor med={floor_med:.1f} cm")
    ax.set_ylabel("Euclidean error (cm)")
    ax.set_title("M2 Ridge error distribution (eval)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "error_distributions.png", dpi=120)
    plt.close(fig)

    # 4) A13 null vs observed — use stored null ridge medians from shifts when present
    a13_methods = [m for m in plot_methods if m in a13]
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(a13_methods))
    obs = []
    null_med = []
    for m in a13_methods:
        row = next(r for r in report["methods"] if r["method"] == m)
        obs.append(float(row["ridge_median_cm"]))
        rec = a13.get(m) or {}
        shifts = rec.get("shifts") or []
        if shifts:
            null_med.append(float(np.median([s["ridge_median"] for s in shifts])))
        else:
            delta = rec.get("ridge_median_minus_floor")
            null_med.append(
                float("nan") if delta is None else float(delta) + floor_med
            )
    width = 0.35
    ax.bar(xs - width / 2, obs, width, label="observed Ridge median", color="C0")
    ax.bar(
        xs + width / 2, null_med, width,
        label="A13 null Ridge median (median over circular shifts)",
        color="C3",
    )
    ax.axhline(floor_med, color="0.3", ls="--", lw=1.0, label="chance floor")
    ax.set_xticks(xs)
    ax.set_xticklabels([_method_label(next(r for r in report["methods"] if r["method"] == m)) for m in a13_methods], rotation=25)
    ax.set_ylabel("median error (cm)")
    ax.set_title("M2 A13 null vs observed (Ridge)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "a13_null_vs_observed.png", dpi=120)
    plt.close(fig)

    print(str(PLOT_DIR), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
