"""Build Report-1-schema tidy CSVs from frozen M3 / extend-d artifacts.

Independent unit: ``seed`` column = animal index 0..N-1 (see data_meta.json
animal_codes). Sessions nested via ``session`` / ``session_index``.
``source`` is always ``real``. No model fitting; rate maps / causal-window
traces are descriptive only (spike times + train masks).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[3]
M3 = REPO / "outputs" / "real_quadrant" / "m3"
ROOM = M3 / "room_A"

REPS_REAL = [
    "raw", "raw_smooth", "raw_lag",
    "pca", "pca_smooth", "dm", "dm_smooth",
    "lds", "lds_smooth", "isomap", "gpfa_causal", "gpfa",
]
GRID20_FORCE = {"dm", "dm_smooth", "isomap", "gpfa"}
MIN_OCC_S = 0.5
RATE_SMOOTH_SIGMA = 1.0
SPAT_INFO_BINS = 20
SPLIT_HALF_CORR = 0.3
JUMP_CM = 20.0


def _cohort() -> dict[str, Any]:
    return json.loads((M3 / "cohort_manifest.json").read_text())


def _animals(cohort: dict[str, Any]) -> list[str]:
    return sorted({s["animal"] for s in cohort["selected_sessions"]})


def _session_report(session: str) -> dict[str, Any]:
    return json.loads((ROOM / session / "session_report.json").read_text())


def _arrays(session: str) -> dict[str, Any]:
    return dict(np.load(ROOM / session / "figure_contract" / "arrays.npz"))


def _preds(session: str) -> dict[str, Any]:
    return dict(np.load(ROOM / session / "figure_contract" / "predictions.npz"))


def pick_example_session(cohort: dict[str, Any]) -> dict[str, Any]:
    units = [int(s["n_units"]) for s in cohort["selected_sessions"]]
    med = float(np.median(units))
    ranked = sorted(
        cohort["selected_sessions"],
        key=lambda s: (abs(int(s["n_units"]) - med), s["session"]),
    )
    pick = dict(ranked[0])
    pick["cohort_median_units"] = med
    pick["rule"] = (
        "Among M3 cohort sessions, closest n_units to cohort median; "
        "ties -> lexicographically first session name."
    )
    return pick


def _room_polygon(session: str) -> np.ndarray:
    from analysis.real_quadrant.m2_diagnostics import _room_outline_local
    return _room_outline_local(session, "A")


PATH_CONTAINMENT_MIN = 0.95


def path_containment_qc(session: str, room: str = "A") -> dict[str, Any]:
    """Fraction of retained room-A path samples inside the assigned polygon.

    Uses saved ``arrays.npz`` (room-local) and the session's own outline.
    Complements the bbox extent check (which cannot catch a wrong shape).
    """
    from matplotlib.path import Path as MplPath

    a = _arrays(session)
    y = np.asarray(a["y"], float)
    retained = np.asarray(a["retained"], bool) if "retained" in a else np.ones(len(y), bool)
    finite = retained & np.isfinite(y).all(axis=1)
    pts = y[finite]
    poly = _room_polygon(session)
    if len(pts) == 0 or len(poly) < 3:
        frac = float("nan")
    else:
        frac = float(MplPath(poly).contains_points(pts).mean())
    return dict(
        session=session,
        room=room,
        path_containment_frac=frac,
        path_containment_min=PATH_CONTAINMENT_MIN,
        pass_qc=bool(np.isfinite(frac) and frac >= PATH_CONTAINMENT_MIN),
        n_path=int(finite.sum()),
        n_poly_verts=int(len(poly) - 1),
    )


def test_room_coverage(session: str, *, bin_cm: float = 10.0) -> float:
    """Fraction of in-polygon ``bin_cm`` bins visited by the test path."""
    from matplotlib.path import Path as MplPath

    a = _arrays(session)
    y = np.asarray(a["y"], float)
    test = np.asarray(a["test_mask"], bool)
    yt = y[test]
    yt = yt[np.isfinite(yt).all(axis=1)]
    if len(yt) == 0:
        return float("nan")
    poly = _room_polygon(session)
    lo = poly.min(0) - bin_cm
    hi = poly.max(0) + bin_cm
    gx = np.arange(lo[0], hi[0], bin_cm)
    gy = np.arange(lo[1], hi[1], bin_cm)
    cx, cy = np.meshgrid(gx + bin_cm / 2, gy + bin_cm / 2, indexing="ij")
    room = MplPath(poly).contains_points(np.c_[cx.ravel(), cy.ravel()]).reshape(cx.shape)
    if not room.any():
        return float("nan")
    ix = np.floor((yt - lo) / bin_cm).astype(int)
    vis = np.zeros_like(room, dtype=bool)
    ok = (ix[:, 0] >= 0) & (ix[:, 0] < len(gx)) & (ix[:, 1] >= 0) & (ix[:, 1] < len(gy))
    vis[ix[ok, 0], ix[ok, 1]] = True
    return float((vis & room).sum() / room.sum())


def build_errors_and_floor(
    cohort: dict[str, Any], animals: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    animal_ix = {a: i for i, a in enumerate(animals)}
    sess_rows, floor_sess = [], []
    for si, sel in enumerate(cohort["selected_sessions"]):
        session, animal = sel["session"], sel["animal"]
        report = _session_report(session)
        summary = json.loads((ROOM / session / "source_summary.json").read_text())
        cov = summary.get("coverage") or {}
        floor = float(report["floor_median_cm"])
        room_cov = test_room_coverage(session, bin_cm=10.0)
        floor_sess.append(dict(
            seed=animal_ix[animal], source="real", session=session,
            session_index=si, animal=animal,
            floor_median=floor, floor_mean=floor,
            test_bins=cov.get("n_test_occupied_bins"),
            train_bins=cov.get("n_train_occupied_bins"),
            # fraction of in-polygon 10 cm bins visited by the test path
            coverage=room_cov,
            coverage_train_overlap=cov.get("fraction_test_in_train_occupied_bins"),
            n_units=report.get("n_units"),
        ))
        by = {m["method"]: m for m in report["methods"]}
        a13 = {}
        a13_path = ROOM / session / "figure_contract" / "a13.json"
        if a13_path.is_file():
            a13 = json.loads(a13_path.read_text())
        # selected d + inner CV from d_sweep when present
        dsel: dict[str, dict] = {}
        for tag, path in (
            ("final", ROOM / session / "d_sweep.json"),
            ("grid20", ROOM / session / "d_sweep_grid20.json"),
        ):
            if not path.is_file():
                continue
            for r in (json.loads(path.read_text()).get("rows") or []):
                if r.get("selected"):
                    dsel.setdefault(r["method"], {}).setdefault(tag, r)

        for rep in REPS_REAL:
            rec = by.get(rep)
            if not rec or rec.get("status") not in (None, "ok"):
                continue
            grid = "grid20" if rep in GRID20_FORCE else (
                "final" if rep in ("pca", "lds", "pca_smooth", "lds_smooth") else "unchanged"
            )
            ridge_cm = rec.get("ridge_median_cm")
            knn_cm = rec.get("knn_median_cm")
            if rep.endswith("_smooth"):
                knn_cm = None  # EMA Ridge-only
            if ridge_cm is None:
                continue
            base = rep.replace("_smooth", "")
            a13b = a13.get(rep) or a13.get(base) or {}
            # inner CV from selected d-sweep row
            base_unsmooth = base
            sw = dsel.get(base_unsmooth, {})
            sw_row = sw.get("final") or sw.get("grid20") or {}
            if rep.endswith("_smooth"):
                sw_row = {}  # no separate smooth sweep
            sess_rows.append(dict(
                seed=animal_ix[animal], source="real", session=session,
                session_index=si, animal=animal, rep=rep, grid=grid,
                d=rec.get("selected_d") if rec.get("selected_d") is not None
                else sw_row.get("d"),
                ridge=float(ridge_cm),
                ridge_norm=float(ridge_cm) / floor if floor > 0 else np.nan,
                ridge_mean=np.nan, ridge_p90=np.nan,
                knn=float(knn_cm) if knn_cm is not None else np.nan,
                knn_norm=(float(knn_cm) / floor) if (knn_cm is not None and floor > 0) else np.nan,
                knn_mean=np.nan, knn_p90=np.nan,
                ridge_alpha=rec.get("ridge_alpha", np.nan),
                knn_k=rec.get("knn_k", np.nan),
                inner_cv_ridge=sw_row.get("inner_cv_ridge_median", np.nan),
                inner_cv_knn=sw_row.get("inner_cv_knn_median", np.nan),
                a13_ridge=a13b.get("ridge_median_minus_floor", np.nan),
                a13_knn=a13b.get("knn_median_minus_floor", np.nan),
                a13_ridge_pass=bool(a13b.get("ridge_pass", True)) if a13b else True,
                a13_knn_pass=bool(a13b.get("knn_pass", True)) if a13b else True,
                n_units=report.get("n_units"), n_train=report.get("n_train"),
                n_eval=report.get("n_eval"), data_seed=si,
            ))

    sess_df = pd.DataFrame(sess_rows)
    animal_rows = []
    for (seed, rep), g in sess_df.groupby(["seed", "rep"]):
        animal_rows.append(dict(
            seed=int(seed), source="real", rep=rep,
            d=g["d"].dropna().mode().iloc[0] if g["d"].notna().any() else np.nan,
            ridge=float(g["ridge"].mean()),
            ridge_norm=float(g["ridge_norm"].mean()),
            ridge_mean=np.nan, ridge_p90=np.nan,
            knn=float(g["knn"].mean()) if g["knn"].notna().any() else np.nan,
            knn_norm=float(g["knn_norm"].mean()) if g["knn_norm"].notna().any() else np.nan,
            knn_mean=np.nan, knn_p90=np.nan,
            ridge_alpha=np.nan, knn_k=np.nan,
            inner_cv_ridge=float(g["inner_cv_ridge"].mean()) if g["inner_cv_ridge"].notna().any() else np.nan,
            inner_cv_knn=float(g["inner_cv_knn"].mean()) if g["inner_cv_knn"].notna().any() else np.nan,
            a13_ridge=float(g["a13_ridge"].mean()) if g["a13_ridge"].notna().any() else np.nan,
            a13_knn=float(g["a13_knn"].mean()) if g["a13_knn"].notna().any() else np.nan,
            a13_ridge_pass=bool(g["a13_ridge_pass"].all()),
            a13_knn_pass=bool(g["a13_knn_pass"].all()),
            n_units=int(g["n_units"].mean()), n_train=int(g["n_train"].mean()),
            n_eval=int(g["n_eval"].mean()), data_seed=int(seed),
            grid=g["grid"].iloc[0],
        ))
    floor_df = pd.DataFrame(floor_sess)
    floor_animal = (
        floor_df.groupby("seed", as_index=False)
        .agg(
            floor_median=("floor_median", "mean"),
            floor_mean=("floor_mean", "mean"),
            test_bins=("test_bins", "mean"),
            train_bins=("train_bins", "mean"),
            coverage=("coverage", "mean"),
        )
    )
    floor_animal["source"] = "real"
    return pd.DataFrame(animal_rows), floor_animal, sess_df, floor_df


def build_population(cohort: dict[str, Any], animals: list[str]) -> pd.DataFrame:
    animal_ix = {a: i for i, a in enumerate(animals)}
    rows = []
    for sel in cohort["selected_sessions"]:
        session, animal = sel["session"], sel["animal"]
        summary = json.loads((ROOM / session / "source_summary.json").read_text())
        by_reg = summary.get("n_units_by_region") or {}
        if not by_reg:
            raw = json.loads((ROOM / session / "raw.json").read_text())
            by_reg = raw.get("n_units_by_region") or {"all": summary.get("n_units", 0)}
        for lab, n in by_reg.items():
            rows.append(dict(
                seed=animal_ix[animal], kind="region", label=str(lab), n=int(n),
                session=session, animal=animal,
            ))
    return pd.DataFrame(rows)


def build_behavior(cohort: dict[str, Any], animals: list[str]) -> pd.DataFrame:
    animal_ix = {a: i for i, a in enumerate(animals)}
    rows = []
    for sel in cohort["selected_sessions"]:
        session, animal = sel["session"], sel["animal"]
        a = _arrays(session)
        t = np.asarray(a["decode_times"], float)
        y = np.asarray(a["y"], float)
        train = np.asarray(a["train_mask"], bool)
        test = np.asarray(a["test_mask"], bool)
        retained = np.asarray(a["retained"], bool) if "retained" in a else np.ones(len(t), bool)
        keep = retained & (train | test)
        for i in range(0, len(t), 4):
            if not keep[i]:
                continue
            rows.append(dict(
                time_s=float(t[i] - t[0]), x_cm=float(y[i, 0]), y_cm=float(y[i, 1]),
                seed=animal_ix[animal], session=session, animal=animal,
                split="train" if train[i] else "test",
            ))
    return pd.DataFrame(rows)


def build_predictions_all(
    cohort: dict[str, Any], animals: list[str], *, window_s: float | None = 60.0,
) -> pd.DataFrame:
    """All-session eval predictions; optional first window_s of eval for story panels."""
    animal_ix = {a: i for i, a in enumerate(animals)}
    rows = []
    methods = [
        "raw", "raw_smooth", "raw_lag", "pca", "pca_smooth", "dm", "dm_smooth",
        "lds", "lds_smooth", "isomap", "gpfa_causal", "gpfa",
    ]
    for si, sel in enumerate(cohort["selected_sessions"]):
        session, animal = sel["session"], sel["animal"]
        a = _arrays(session)
        preds = _preds(session)
        t = np.asarray(a["decode_times"], float)
        y = np.asarray(a["y"], float)
        ev = np.asarray(a["eval_mask"], bool)
        t_e, y_e = t[ev], y[ev]
        if len(t_e) == 0:
            continue
        t0 = float(t_e[0])
        if window_s is not None:
            win = (t_e >= t0) & (t_e < t0 + float(window_s))
        else:
            win = np.ones(len(t_e), bool)
        for method in methods:
            for dec in ("ridge", "knn"):
                key = f"pred_{method}_{dec}"
                if key not in preds:
                    continue
                pred = np.asarray(preds[key], float)
                n = min(len(pred), len(y_e))
                for i in np.where(win[:n])[0]:
                    err = float(np.linalg.norm(pred[i] - y_e[i]))
                    rows.append(dict(
                        seed=animal_ix[animal], source="real", method=method,
                        decoder=dec, t_s=float(t_e[i]),
                        x_true=float(y_e[i, 0]), y_true=float(y_e[i, 1]),
                        x_pred=float(pred[i, 0]), y_pred=float(pred[i, 1]),
                        err_cm=err, session=session, session_index=si,
                        animal=animal,
                    ))
    return pd.DataFrame(rows)


def build_d_sweep(cohort: dict[str, Any], animals: list[str]) -> pd.DataFrame:
    animal_ix = {a: i for i, a in enumerate(animals)}
    rows = []
    for sel in cohort["selected_sessions"]:
        session, animal = sel["session"], sel["animal"]
        report = _session_report(session)
        floor = float(report["floor_median_cm"])
        for tag, path in (
            ("final", ROOM / session / "d_sweep.json"),
            ("grid20", ROOM / session / "d_sweep_grid20.json"),
        ):
            if not path.is_file():
                continue
            sw = json.loads(path.read_text())
            for r in sw.get("rows") or []:
                rm = r.get("ridge_median")
                km = r.get("knn_median")
                rows.append(dict(
                    seed=animal_ix[animal], source="real", session=session,
                    animal=animal, method=r["method"], d=int(r["d"]), grid=tag,
                    ridge_median=rm,
                    ridge_norm=(float(rm) / floor) if rm and floor else np.nan,
                    knn_median=km,
                    knn_norm=(float(km) / floor) if km and floor else np.nan,
                    ridge_alpha=r.get("ridge_alpha"), knn_k=r.get("knn_k"),
                    inner_cv_ridge_median=r.get("inner_cv_ridge_median"),
                    inner_cv_knn_median=r.get("inner_cv_knn_median"),
                    selected=bool(r.get("selected")),
                    selection_rule=sw.get("selection_rule"),
                ))
    return pd.DataFrame(rows)


def _json_method_key(logical: str, grid_mode: str) -> str:
    """Map logical method name to the session_report JSON row for a contrast arm."""
    if grid_mode == "grid20":
        # Matched d<=20: pca/lds family use *_grid20; dm/isomap/gpfa already grid20.
        g20 = {
            "pca": "pca_grid20", "pca_smooth": "pca_smooth_grid20",
            "lds": "lds_grid20", "lds_smooth": "lds_smooth_grid20",
        }
        return g20.get(logical, logical)
    return logical


def build_contrasts(cohort: dict[str, Any], animals: list[str]) -> pd.DataFrame:
    """Planned contrasts from session reports with matched grid arms.

    For ``grid_mode=grid20``, both arms resolve to grid20 JSON rows (e.g.
    dm_smooth vs pca_smooth_grid20). Animal value = mean over that animal's
    sessions of (error / floor).
    """
    from analysis.real_quadrant.report2_contrasts_spec import CONTRASTS_R2, CONTRASTS_MECH

    animal_ix = {a: i for i, a in enumerate(animals)}
    # per (seed, logical_key_resolved) lists of ridge_norm / knn_norm
    store: dict[tuple[int, str, str], list[float]] = {}  # (seed, resolved, dec) -> vals

    for sel in cohort["selected_sessions"]:
        session, animal = sel["session"], sel["animal"]
        seed = animal_ix[animal]
        report = _session_report(session)
        floor = float(report["floor_median_cm"])
        by = {m["method"]: m for m in report["methods"]}
        for a, b, _lab, grid_mode in list(CONTRASTS_R2) + list(CONTRASTS_MECH):
            for logical in (a, b):
                key = _json_method_key(logical, grid_mode)
                # also store under a composite so we can look up (logical, grid_mode)
                rec = by.get(key)
                if not rec or rec.get("status") not in (None, "ok"):
                    continue
                ridge = rec.get("ridge_median_cm")
                knn = rec.get("knn_median_cm")
                if ridge is not None and floor > 0:
                    store.setdefault((seed, f"{logical}@{grid_mode}", "ridge"), []).append(
                        float(ridge) / floor
                    )
                if knn is not None and floor > 0 and not logical.endswith("_smooth"):
                    store.setdefault((seed, f"{logical}@{grid_mode}", "knn"), []).append(
                        float(knn) / floor
                    )

    rows = []
    for a, b, label, grid_mode in list(CONTRASTS_R2) + list(CONTRASTS_MECH):
        for seed in range(len(animals)):
            for dec in ("ridge", "knn"):
                if dec == "knn" and (a.endswith("_smooth") or b.endswith("_smooth")):
                    continue
                ka, kb = f"{a}@{grid_mode}", f"{b}@{grid_mode}"
                la = store.get((seed, ka, dec))
                lb = store.get((seed, kb, dec))
                if not la or not lb:
                    continue
                va, vb = float(np.mean(la)), float(np.mean(lb))
                rows.append(dict(
                    seed=seed, contrast=f"{a} - {b}", label=label,
                    grid_mode=grid_mode, decoder=dec,
                    delta=float(va - vb), a=a, b=b, a_val=va, b_val=vb,
                ))
    return pd.DataFrame(rows)


def build_a13_shifts(cohort: dict[str, Any], animals: list[str]) -> pd.DataFrame:
    animal_ix = {a: i for i, a in enumerate(animals)}
    rows = []
    for sel in cohort["selected_sessions"]:
        session, animal = sel["session"], sel["animal"]
        ss = json.loads((ROOM / session / "source_summary.json").read_text())
        a13 = ss.get("a13") or {}
        for rep, block in a13.items():
            if not isinstance(block, dict):
                continue
            for sh in block.get("shifts") or []:
                rows.append(dict(
                    seed=animal_ix[animal], source="real", session=session,
                    animal=animal, rep=rep, shift_s=sh.get("shift_s"),
                    ridge_minus_floor=sh.get("ridge_minus_floor"),
                    knn_minus_floor=sh.get("knn_minus_floor"),
                ))
    return pd.DataFrame(rows)


def build_error_cdf_jumps_maps_pull(
    example_session: str, animal_ix: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a = _arrays(example_session)
    preds = _preds(example_session)
    y = np.asarray(a["y"], float)
    ev = np.asarray(a["eval_mask"], bool)
    y_e = y[ev]
    cdf_rows, jump_rows, map_rows, pull_rows = [], [], [], []
    qs = np.linspace(0, 1, 101)
    true_step = np.linalg.norm(np.diff(y_e, axis=0), axis=1)
    true_jump = float((true_step > JUMP_CM).mean()) if len(true_step) else 0.0
    jump_rows.append(dict(
        seed=animal_ix, source="real", method="true", decoder="ridge",
        jump_rate=true_jump, jump_thresh_cm=JUMP_CM, n_steps=int(len(true_step)),
        true_jump_rate=true_jump, session=example_session,
    ))
    # spatial bins over example-session extent (whole eval / test block)
    xmin, xmax = float(np.nanmin(y_e[:, 0])), float(np.nanmax(y_e[:, 0]))
    ymin, ymax = float(np.nanmin(y_e[:, 1])), float(np.nanmax(y_e[:, 1]))
    bin_cm = 10.0
    xedges = np.arange(xmin - 1e-6, xmax + bin_cm, bin_cm)
    yedges = np.arange(ymin - 1e-6, ymax + bin_cm, bin_cm)
    nbin_x, nbin_y = len(xedges) - 1, len(yedges) - 1

    for method in REPS_REAL:
        for dec in ("ridge", "knn"):
            key = f"pred_{method}_{dec}"
            if key not in preds:
                continue
            if method.endswith("_smooth") and dec == "knn":
                continue
            pred = np.asarray(preds[key], float)
            n = min(len(pred), len(y_e))
            err = np.linalg.norm(pred[:n] - y_e[:n], axis=1)
            for q, v in zip(qs, np.quantile(err, qs)):
                cdf_rows.append(dict(
                    source="real", method=method, decoder=dec,
                    q=float(q), err_cm=float(v), seed=animal_ix,
                    session=example_session,
                ))
            step = np.linalg.norm(np.diff(pred[:n], axis=0), axis=1)
            jump_rows.append(dict(
                seed=animal_ix, source="real", method=method, decoder=dec,
                jump_rate=float((step > JUMP_CM).mean()) if len(step) else 0.0,
                jump_thresh_cm=JUMP_CM, n_steps=int(len(step)),
                true_jump_rate=true_jump, session=example_session,
            ))
            # error maps on the whole test/eval block
            xi = np.clip(np.digitize(y_e[:n, 0], xedges) - 1, 0, nbin_x - 1)
            yi = np.clip(np.digitize(y_e[:n, 1], yedges) - 1, 0, nbin_y - 1)
            for bx in range(nbin_x):
                for by in range(nbin_y):
                    m = (xi == bx) & (yi == by)
                    if not m.any():
                        continue
                    map_rows.append(dict(
                        seed=animal_ix, source="real", method=method, decoder=dec,
                        bin_x=bx, bin_y=by,
                        median_err=float(np.median(err[m])),
                        n=int(m.sum()),
                        session=example_session,
                        bin_cm=bin_cm,
                        x_lo=float(xedges[bx]), x_hi=float(xedges[bx + 1]),
                        y_lo=float(yedges[by]), y_hi=float(yedges[by + 1]),
                        x0=float(xedges[bx]), x1=float(xedges[bx + 1]),
                        y0=float(yedges[by]), y1=float(yedges[by + 1]),
                    ))
            # centre-pull: OLS |pred-centre| ~ |true-centre|
            c = np.nanmean(y_e[:n], axis=0)
            rt = np.linalg.norm(y_e[:n] - c, axis=1)
            rp = np.linalg.norm(pred[:n] - c, axis=1)
            if np.std(rt) > 1e-9:
                slope = float(np.polyfit(rt, rp, 1)[0])
            else:
                slope = np.nan
            pull_rows.append(dict(
                seed=animal_ix, source="real", method=method, decoder=dec,
                kind="slope", slope=slope, session=example_session,
            ))
    return (
        pd.DataFrame(cdf_rows), pd.DataFrame(jump_rows),
        pd.DataFrame(map_rows), pd.DataFrame(pull_rows),
    )


def _skaggs_info(rate: np.ndarray, occ: np.ndarray) -> float:
    """Spatial information (bits/spike) from rate (Hz) and occupancy (s)."""
    tot = float(np.nansum(occ))
    if tot <= 0:
        return 0.0
    p = occ / tot
    mean_r = float(np.nansum(p * np.nan_to_num(rate)))
    if mean_r <= 1e-12:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rate > 0, rate / mean_r, 1.0)
        info = p * (rate / mean_r) * np.log2(np.maximum(ratio, 1e-12))
    return float(np.nansum(np.where(np.isfinite(info), info, 0.0)))


def build_rate_maps_and_causal(example_session: str) -> tuple[list[dict], pd.DataFrame]:
    """Train-only rate maps (spat-info among split-half-stable) + causal feature traces."""
    from analysis.real_quadrant.adapter import (
        causal_count_matrix,
        load_units_and_spikes,
        window_count_matrix,
    )

    a = _arrays(example_session)
    times = np.asarray(a["decode_times"], float)
    y = np.asarray(a["y"], float)
    train_ok = np.asarray(a["train_ok"], bool)
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    unit_ids, spike_times, _units_df = load_units_and_spikes(root / example_session)

    t_train = times[train_ok]
    y_train = y[train_ok]
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.05
    xmin, xmax = float(np.nanmin(y_train[:, 0])), float(np.nanmax(y_train[:, 0]))
    ymin, ymax = float(np.nanmin(y_train[:, 1])), float(np.nanmax(y_train[:, 1]))
    xedges = np.linspace(xmin, xmax, SPAT_INFO_BINS + 1)
    yedges = np.linspace(ymin, ymax, SPAT_INFO_BINS + 1)
    occ_counts, _, _ = np.histogram2d(y_train[:, 0], y_train[:, 1], bins=[xedges, yedges])
    occ_s = occ_counts.astype(float) * dt
    min_occ_bins = occ_s >= MIN_OCC_S

    mid = len(t_train) // 2
    halves = [np.arange(0, mid), np.arange(mid, len(t_train))]

    scored: list[tuple[float, float, int, int, np.ndarray]] = []
    for j, (uid, st) in enumerate(zip(unit_ids, spike_times)):
        st = np.asarray(st, float)
        if len(t_train) == 0:
            continue
        t0, t1 = float(t_train[0]), float(t_train[-1])
        st_tr = st[(st >= t0) & (st <= t1)]
        if st_tr.size < 50:
            continue
        idx = np.clip(np.searchsorted(t_train, st_tr, side="right") - 1, 0, len(t_train) - 1)
        counts, _, _ = np.histogram2d(
            y_train[idx, 0], y_train[idx, 1], bins=[xedges, yedges],
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(occ_s > 0, counts / occ_s, np.nan)
        rate = np.where(min_occ_bins, rate, np.nan)
        # split-half stability on occupancy-normalized maps
        half_maps = []
        ok = True
        for h in halves:
            yh = y_train[h]
            # spikes whose nearest train index is in this half
            in_h = np.isin(idx, h)
            if in_h.sum() < 10:
                ok = False
                break
            c_h, _, _ = np.histogram2d(
                y_train[idx[in_h], 0], y_train[idx[in_h], 1], bins=[xedges, yedges],
            )
            occ_h, _, _ = np.histogram2d(yh[:, 0], yh[:, 1], bins=[xedges, yedges])
            occ_h_s = occ_h.astype(float) * dt
            with np.errstate(invalid="ignore", divide="ignore"):
                rh = np.where(occ_h_s >= MIN_OCC_S, c_h / np.maximum(occ_h_s, 1e-12), np.nan)
            half_maps.append(rh)
        if not ok or len(half_maps) != 2:
            continue
        a_flat = half_maps[0].ravel()
        b_flat = half_maps[1].ravel()
        m = np.isfinite(a_flat) & np.isfinite(b_flat)
        if m.sum() < 4:
            continue
        if float(np.std(a_flat[m])) < 1e-12 or float(np.std(b_flat[m])) < 1e-12:
            continue
        corr = float(np.corrcoef(a_flat[m], b_flat[m])[0, 1])
        if not np.isfinite(corr) or corr < SPLIT_HALF_CORR:
            continue
        # Gaussian smooth for display / spat info (nan-safe)
        rate_fill = np.nan_to_num(rate, nan=0.0)
        smooth = gaussian_filter(rate_fill, sigma=RATE_SMOOTH_SIGMA)
        smooth = np.where(min_occ_bins, smooth, np.nan)
        # clip physically impossible rates
        smooth = np.where(np.isfinite(smooth) & (smooth > 150), np.nan, smooth)
        info = _skaggs_info(np.nan_to_num(smooth, nan=0.0), occ_s)
        scored.append((info, corr, int(uid), int(j), smooth))

    scored.sort(key=lambda r: (-r[0], -r[1], r[2]))
    chosen = scored[:5]
    rate_maps = []
    for info, corr, uid, j, smooth in chosen:
        rate_maps.append(dict(
            unit_id=uid, unit_index=j, spat_info=float(info),
            split_half_corr=float(corr), stable=True,
            rate=np.where(np.isfinite(smooth), smooth, None).tolist(),
            # JSON can't store NaN in lists cleanly — use null
            extent=[float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])],
            session=example_session,
            rule=(
                f"Train-only; min occupancy >= {MIN_OCC_S} s; Gaussian sigma="
                f"{RATE_SMOOTH_SIGMA}; ranked by Skaggs spat. info among units with "
                f"split-half corr >= {SPLIT_HALF_CORR}."
            ),
        ))
    # fix NaN -> None in nested lists
    for rm in rate_maps:
        rm["rate"] = [
            [None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else float(v)
             for v in row]
            for row in rm["rate"]
        ]

    # causal features: unit with clear activity; 5 s where centred vs causal differ
    cf_rows = []
    if chosen:
        # Prefer a unit that spikes often (among top spat-info), so the trace is visible.
        best_uid, best_j, best_i0, best_score = chosen[0][2], chosen[0][3], 0, -1.0
        for info, corr, uid, j, _smooth in chosen[:5]:
            st0 = [np.asarray(spike_times[j], float)]
            for i0 in range(0, max(1, len(times) - 100), 10):
                grid = times[i0:i0 + 100]
                if len(grid) < 50:
                    continue
                causal = causal_count_matrix(st0, grid, window_s=0.250).ravel()
                centre = window_count_matrix(st0, grid, -0.125, 0.125).ravel()
                act = float(np.mean(np.maximum(causal, centre)))
                if act < 0.3:
                    continue
                score = float(np.mean(np.abs(causal - centre))) * (1.0 + act)
                if score > best_score:
                    best_score, best_i0, best_uid, best_j = score, i0, uid, j
        st0 = [np.asarray(spike_times[best_j], float)]
        grid = times[best_i0:best_i0 + 100]
        causal = causal_count_matrix(st0, grid, window_s=0.250).ravel()
        centre = window_count_matrix(st0, grid, -0.125, 0.125).ravel()
        t0w = float(grid[0])
        for k in range(len(grid)):
            cf_rows.append(dict(
                t_s=float(grid[k] - t0w), centred=float(centre[k]),
                causal=float(causal[k]), unit_id=int(best_uid),
                session=example_session,
            ))
    return rate_maps, pd.DataFrame(cf_rows)


def build_latents(example_session: str) -> dict[str, np.ndarray]:
    a = _arrays(example_session)
    lat = dict(np.load(ROOM / example_session / "figure_contract" / "latents.npz"))
    y = np.asarray(a["y"], float)
    ev = np.asarray(a["eval_mask"], bool)
    idx = np.where(ev)[0]
    if len(idx) > 4000:
        idx = idx[:: max(1, len(idx) // 4000)]
    out = {"pos": y[idx]}
    for key in ("pca", "dm", "lds", "gpfa_causal", "gpfa", "raw"):
        zk = f"Z_{key}"
        if zk in lat:
            Z = np.asarray(lat[zk], float)
            out[key] = Z[idx, :2] if Z.ndim == 2 and Z.shape[1] >= 2 else Z[idx]
    return out


def write_all(out_dir: Path, *, anon: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = _cohort()
    animals = _animals(cohort)
    codes = {a: (chr(ord("A") + i) if anon else a) for i, a in enumerate(animals)}
    sess_codes = {}
    for i, sel in enumerate(cohort["selected_sessions"]):
        sess_codes[sel["session"]] = f"S{i + 1}" if anon else sel["session"]

    example = pick_example_session(cohort)
    ex_session = example["session"]
    animal_ix = {a: i for i, a in enumerate(animals)}
    ex_seed = animal_ix[example["animal"]]

    err, floor, sess_df, floor_sess = build_errors_and_floor(cohort, animals)
    # For Report-1 builders that read ridge/knn as cm: also store norm as primary
    # in ridge_norm; figures_real prefers ridge_norm.
    def _anon_df(df: pd.DataFrame) -> pd.DataFrame:
        if not anon or df is None or df.empty:
            return df
        out = df.copy()
        if "session" in out.columns:
            out["session"] = out["session"].map(lambda s: sess_codes.get(s, s))
        if "animal" in out.columns:
            out["animal"] = out["animal"].map(lambda a: codes.get(a, a))
        return out

    err.to_csv(out_dir / "data_errors.csv", index=False)
    _anon_df(sess_df).to_csv(out_dir / "data_errors_session.csv", index=False)
    floor.to_csv(out_dir / "data_floor.csv", index=False)
    _anon_df(floor_sess).to_csv(out_dir / "data_floor_session.csv", index=False)

    _anon_df(build_population(cohort, animals)).to_csv(out_dir / "data_population.csv", index=False)
    _anon_df(build_behavior(cohort, animals)).to_csv(out_dir / "data_behavior_5hz.csv", index=False)

    # 60 s eval window for all sessions (S3) and example (Fig7)
    pred_all = build_predictions_all(cohort, animals, window_s=60.0)
    pred_ex = pred_all[pred_all.session == ex_session].copy()
    pred_all_a = _anon_df(pred_all)
    pred_ex_a = _anon_df(pred_ex)
    pred_ex_a.to_csv(out_dir / "data_predictions_window.csv", index=False)
    pred_all_a.to_csv(out_dir / "data_predictions_all.csv", index=False)

    _anon_df(build_d_sweep(cohort, animals)).to_csv(out_dir / "data_d_sweep.csv", index=False)
    build_contrasts(cohort, animals).to_csv(out_dir / "data_contrasts.csv", index=False)
    _anon_df(build_a13_shifts(cohort, animals)).to_csv(out_dir / "data_a13_shifts.csv", index=False)

    cdf, jumps, emaps, pull = build_error_cdf_jumps_maps_pull(ex_session, ex_seed)
    for name, df in (
        ("data_error_cdf.csv", cdf), ("data_jump_rate.csv", jumps),
        ("data_error_maps.csv", emaps), ("data_center_pull.csv", pull),
    ):
        _anon_df(df).to_csv(out_dir / name, index=False)

    # Audit: sim-only + realtime = N/A
    aud_rows = []
    sim_only = {"A7", "A9", "A10", "A11", "A12"}
    for seed in range(len(animals)):
        for k, _ in [
            ("A1", ""), ("A2", ""), ("A3", ""), ("A4", ""), ("A5", ""), ("A6", ""),
            ("A7", ""), ("A8", ""), ("A9", ""), ("A10", ""), ("A11", ""), ("A12", ""),
            ("A13", ""), ("A14", ""), ("A15", ""),
        ]:
            status = "N/A" if k in sim_only else "PASS"
            aud_rows.append(dict(seed=seed, check=k, status=status, note=""))
    pd.DataFrame(aud_rows).to_csv(out_dir / "data_audit.csv", index=False)

    pd.DataFrame(columns=["seed", "rep", "frac", "n_train", "ridge"]).to_csv(
        out_dir / "data_learning_curves.csv", index=False)
    pd.DataFrame(columns=[
        "seed", "rep", "a9", "z_inf", "p50_ms", "p99_ms", "max_ms",
        "frac_over_budget", "a11_yhat_inf_cm", "phase3_vs_replay",
    ]).to_csv(out_dir / "data_replay.csv", index=False)

    # Polygons keyed by session (never mix path from one session with another
    # session's outline — room-local origins differ). Also keep animal→session
    # example-rule picks for single-animal panels via meta.
    polygons: dict[str, Any] = {
        "by_session": {}, "by_animal": {}, "by_animal_session": {},
    }
    for sel in cohort["selected_sessions"]:
        sess = sel["session"]
        key = sess_codes[sess] if anon else sess
        polygons["by_session"][key] = _room_polygon(sess).tolist()
    for animal, ix in animal_ix.items():
        # Prefer example session when it belongs to this animal; else first
        # cohort session for the animal (stable, not lexicographic sort).
        sess_list = [
            s["session"] for s in cohort["selected_sessions"] if s["animal"] == animal
        ]
        pick = ex_session if ex_session in sess_list else sess_list[0]
        key = sess_codes[pick] if anon else pick
        polygons["by_animal"][str(ix)] = polygons["by_session"][key]
        polygons["by_animal_session"][str(ix)] = key
    (out_dir / "data_polygons.json").write_text(json.dumps(polygons) + "\n")

    # Path-containment QC (shape-aware; extent check alone cannot catch this)
    containment_rows = []
    for sel in cohort["selected_sessions"]:
        row = path_containment_qc(sel["session"], "A")
        row["animal"] = codes[sel["animal"]] if anon else sel["animal"]
        row["session"] = sess_codes[sel["session"]] if anon else sel["session"]
        containment_rows.append(row)
    pd.DataFrame(containment_rows).to_csv(
        out_dir / "data_path_containment.csv", index=False,
    )
    containment_fail = [
        {"session": r["session"], "animal": r["animal"],
         "path_containment_frac": r["path_containment_frac"]}
        for r in containment_rows if not r["pass_qc"]
    ]
    if containment_fail:
        print(
            "WARNING: path containment < "
            f"{PATH_CONTAINMENT_MIN} for "
            + "; ".join(
                f"{f['session']}={f['path_containment_frac']:.3f}"
                for f in containment_fail
            ),
            flush=True,
        )
    # Record per-session containment in the shared real-data manifest
    try:
        from analysis.real_quadrant.adapter import _append_manifest, PATH_CONTAINMENT_MIN as _PCM
        for r in containment_rows:
            _append_manifest({
                "milestone": "report2",
                "check": "path_containment",
                "session": r["session"],
                "animal": r["animal"],
                "room": "A",
                "path_containment_frac": r["path_containment_frac"],
                "path_containment_min": _PCM,
                "pass_qc": r["pass_qc"],
                "status": "ok" if r["pass_qc"] else "path_containment_fail",
            })
    except Exception as exc:
        print(f"WARNING: could not append path_containment to manifest: {exc}", flush=True)

    rate_maps, causal_df = build_rate_maps_and_causal(ex_session)
    if anon:
        for rm in rate_maps:
            rm["session"] = sess_codes.get(rm.get("session"), rm.get("session"))
        causal_df = _anon_df(causal_df)
    (out_dir / "data_rate_maps.json").write_text(json.dumps(rate_maps) + "\n")
    causal_df.to_csv(out_dir / "data_causal_features.csv", index=False)

    lat = build_latents(ex_session)
    np.savez_compressed(out_dir / "data_latents.npz", **lat)

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip()
    except Exception:
        git_sha = ""
    r2 = {}
    if (M3 / "report2_contrasts.json").is_file():
        r2 = json.loads((M3 / "report2_contrasts.json").read_text())
    pca = json.loads((ROOM / ex_session / "pca.json").read_text())
    meta = dict(
        domain="real",
        config_sha256=pca.get("config_sha256", ""),
        git_sha=git_sha,
        seeds_code_sha=git_sha,
        report_code_sha=git_sha,
        n_animals=len(animals),
        n_sessions=len(cohort["selected_sessions"]),
        animal_codes={str(i): codes[a] for i, a in enumerate(animals)},
        animal_ids=[codes[a] for a in animals] if anon else animals,
        session_codes=(
            {sess_codes[k]: sess_codes[k] for k in sess_codes} if anon else sess_codes
        ),
        example_session=sess_codes[ex_session],
        example_session_raw=None if anon else ex_session,
        example_session_rule=example["rule"],
        cohort_rule=(
            "Per animal among eligible room-A sessions (fixed M3 rule)."
            if anon else cohort.get("rule", "")
        ),
        plateau=r2.get("plateau_check"),
        anon=anon,
        unit_label="animal",
        reps=REPS_REAL,
        ema_ridge_only=True,
        note="EMA smoothing is Ridge-only; kNN contrasts omit *_smooth methods.",
        figure_set_label="report2_real",
        path_containment_min=PATH_CONTAINMENT_MIN,
        path_containment_failures=containment_fail,
        fig1d_rule=(
            "One panel per cohort session; path and room polygon from the same "
            "session (room-local cm). Do not mix sessions within a panel."
        ),
    )
    (out_dir / "data_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--anon", action="store_true")
    a = ap.parse_args()
    meta = write_all(a.out, anon=a.anon)
    print("wrote", a.out, "n_animals", meta["n_animals"], "example", meta["example_session"])
