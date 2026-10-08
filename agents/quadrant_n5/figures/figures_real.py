"""
Report 2 (real data) -- publication figure set: Fig 1-7 and Fig S1-S5.

Real-data counterpart of ``figures.py`` (same style helpers, same file names).
Renders from tidy tables only (``analysis/real_quadrant/report/extract_tidy.py``);
nothing is recomputed from raw recordings except descriptive geometry
(coverage, centre-pull, spatial error bins) from tables already in ``D``.

    python figures_real.py --data <tidy_dir> --out <fig_dir>

Conventions
-----------
* Independent unit = animal. The ``seed`` column holds the animal index 0..N-1;
  ``meta["animal_codes"]`` maps it to a label. N = 6 animals.
* Primary y-axis = normalized error (median Euclidean error / chance floor,
  chance = predict training-mean position, so chance = 1.0). Median cm values
  are available from ``cm_summary(D)`` and are printed in footnotes.
* EMA (``*_smooth``) = causal exponential moving average on Ridge predictions.
  It is Ridge-only: kNN columns for ``*_smooth`` are NaN by design.
* DM / Isomap / GPFA-c / GPFA-off were run on the d <= 20 grid ("grid20");
  PCA and LDS on the extended grid d <= 80.
* ASCII only in labels (Liberation Sans has no fancy unicode).

``D`` keys (all optional unless marked *)
-----------------------------------------
E*, F*, A*, B*, P*   as in ``figures.load`` (E: animal means, one row per rep).
ES, FS               session-level errors / floors (``data_errors_session``,
                     ``data_floor_session``).
predictions_window   example-session test window, columns as Report 1.
d_sweep, error_cdf, jump_rate, center_pull, error_maps, contrasts, SH
meta                 dict (or path to ``data_meta.json``).
polygons             {animal_index(int): (N, 2) array}, room-local cm.
rate_maps            list of dicts: rate (2-D, [iy, ix], Hz, NaN where
                     unvisited), extent (x0, x1, y0, y1), unit_id, spat_info,
                     optional stable (bool) and seed.
causal_features      DataFrame: t_s, centred, causal (+ optional unit_id).
latents              {method: (n, k) array, "pos": (n, 2) array}
                     (or pass positions as ``D["latents_pos"]``).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agents.quadrant_n5.figures import figures as _F  # noqa: E402  (registers font + rcParams)
from agents.quadrant_n5.figures.figures import (  # noqa: E402
    head, head_fig, save as _save_sim, neg, INK, INK2, MUTED, GRID, FAILC, W_FULL, MM,
    FancyBboxPatch, Rectangle,
    SHORT as _SHORT, COL as _COL, EDGE as _EDGE, MRK as _MRK, TINT as _TINT,
    QUAD as _QUAD, AUD,
)


def save(fig, out, name):
    """Save PDF+PNG; fail if any artist or Text bbox exceeds the figure area."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Fail on clear overflows past the figure; small pad for tick labels / AA.
    pad = 0.08
    offenders = []
    for i, ax in enumerate(fig.axes):
        try:
            bb = ax.get_tightbbox(renderer)
        except Exception:
            continue
        if bb is None:
            continue
        bb_fig = bb.transformed(fig.transFigure.inverted())
        if (bb_fig.x0 < -pad or bb_fig.y0 < -pad
                or bb_fig.x1 > 1 + pad or bb_fig.y1 > 1 + pad):
            offenders.append(
                f"ax{i} bbox=({bb_fig.x0:.3f},{bb_fig.y0:.3f})-({bb_fig.x1:.3f},{bb_fig.y1:.3f})"
            )
    # Explicit Text artists (annotations can overflow even when axes bbox is OK)
    for j, txt in enumerate(fig.findobj(match=lambda a: isinstance(a, mpl.text.Text))):
        s = (txt.get_text() or "").strip()
        if not s or not txt.get_visible():
            continue
        try:
            bb = txt.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb.width == 0 and bb.height == 0:
            continue
        bb_fig = bb.transformed(fig.transFigure.inverted())
        if (bb_fig.x0 < -pad or bb_fig.y0 < -pad
                or bb_fig.x1 > 1 + pad or bb_fig.y1 > 1 + pad):
            preview = s.replace("\n", " ")[:40]
            offenders.append(
                f"text[{j}] '{preview}' "
                f"bbox=({bb_fig.x0:.3f},{bb_fig.y0:.3f})-({bb_fig.x1:.3f},{bb_fig.y1:.3f})"
            )
    if offenders:
        raise RuntimeError(
            f"{name}: artist bbox exceeds figure area (clipped content): "
            + "; ".join(offenders[:8])
        )
    _save_sim(fig, out, name)

# ------------------------------------------------------------------ methods
REPS_REAL = ["raw", "raw_smooth", "raw_lag", "pca", "pca_smooth", "dm", "dm_smooth",
             "lds", "lds_smooth", "isomap", "gpfa_causal", "gpfa"]

SHORT = {**_SHORT, "raw_smooth": "Raw+EMA", "pca_smooth": "PCA+EMA", "dm_smooth": "DM+EMA",
         "lds_smooth": "LDS+EMA", "gpfa_causal": "GPFA-c", "gpfa": "GPFA-off"}
COL = {**_COL, "raw_smooth": "#d3d1cb", "pca_smooth": "#9cc3f0", "dm_smooth": "#f5b79a",
       "lds_smooth": "#8fdcbf", "gpfa_causal": "#b7ecd7"}
EDGE = {**_EDGE, "raw_smooth": "#3a3936", "pca_smooth": "#1f5fae", "dm_smooth": "#c24e1e",
        "lds_smooth": "#0e7a54", "gpfa_causal": "#0e7a54"}
MRK = {**_MRK, "raw_smooth": "o", "pca_smooth": "o", "dm_smooth": "D", "lds_smooth": "^",
       "gpfa_causal": "X"}
TINT = dict(_TINT)
QUAD = tuple(_QUAD) + ("gpfa_causal",)
QUAD_SET = set(QUAD) | {"pca_smooth", "dm_smooth", "lds_smooth"}
HOLLOW = ("gpfa", "isomap")                     # offline / baseline: open markers

NORM_LAB = "Normalized error (error / chance)"
GREY_PT = "#9a9893"
SIM_ONLY = ("A7", "A10", "A11", "A12")
CAUSAL_CAND = ["raw", "raw_smooth", "raw_lag", "pca", "pca_smooth", "dm", "dm_smooth",
               "lds", "lds_smooth", "gpfa_causal"]
GRID20_NOTE = "DM, Isomap, GPFA-c, GPFA-off: d <= 20 grid; PCA, LDS: d up to 80"
REGION_COLS = ["#7b5ea7", "#b08a3e", "#3c8d93", "#a65a5a", "#6e6c66", "#9aa05a", "#c4c2bb"]

# planned contrasts (a - b; negative = a lower error). Primary list is pre-registered.
PRIMARY = [("lds", "raw_smooth", "Dynamics vs EMA"),
           ("gpfa_causal", "raw_smooth", "Causal GPFA vs EMA"),
           ("lds_smooth", "raw_smooth", "LDS+EMA vs EMA"),
           ("pca_smooth", "raw_smooth", "PCA+EMA vs EMA"),
           ("dm_smooth", "pca_smooth", "DM+EMA vs PCA+EMA"),
           ("raw_smooth", "raw", "EMA alone")]
SECONDARY = [("lds", "pca", "Dynamic vs static"),
             ("dm", "pca", "Nonlinear vs linear"),
             ("raw_lag", "raw", "History alone"),
             ("lds", "raw_lag", "Dynamics beyond history")]


# ===================================================================== data
def load_real(here):
    """figures.load() plus the real-data extras found next to it."""
    try:
        D = _F.load(here)
    except FileNotFoundError:
        rd = lambda n: pd.read_csv(os.path.join(here, f"data_{n}.csv"))
        D = dict(E=rd("errors"), F=rd("floor"),
                 A=pd.read_csv(os.path.join(here, "data_audit.csv"), keep_default_na=False),
                 B=rd("behavior_5hz"), P=rd("population"))
        for k in ("a13_shifts", "predictions_window", "error_maps", "center_pull", "error_cdf",
                  "d_sweep", "jump_rate"):
            p = os.path.join(here, f"data_{k}.csv")
            if os.path.isfile(p):
                D["SH" if k == "a13_shifts" else k] = pd.read_csv(p)
    for key, name in (("ES", "errors_session"), ("FS", "floor_session"), ("contrasts", "contrasts"),
                      ("causal_features", "causal_features"),
                      ("predictions_all", "predictions_all")):
        p = os.path.join(here, f"data_{name}.csv")
        if os.path.isfile(p) and key not in D:
            D[key] = pd.read_csv(p)
    p = os.path.join(here, "data_meta.json")
    if os.path.isfile(p):
        D["meta"] = json.load(open(p))
    p = os.path.join(here, "data_polygons.json")
    if os.path.isfile(p):
        raw = json.load(open(p))
        # New schema: {by_session, by_animal, by_animal_session}; legacy: animal→poly
        if isinstance(raw, dict) and "by_session" in raw:
            D["polygons_by_session"] = {
                str(k): np.asarray(v, float) for k, v in raw["by_session"].items()
            }
            D["polygons"] = {
                int(k): np.asarray(v, float) for k, v in (raw.get("by_animal") or {}).items()
            }
            D["polygon_session_by_animal"] = {
                int(k): str(v) for k, v in (raw.get("by_animal_session") or {}).items()
            }
        else:
            D["polygons"] = {int(k): np.asarray(v, float) for k, v in raw.items()}
            D["polygons_by_session"] = {}
            D["polygon_session_by_animal"] = {}
    p = os.path.join(here, "data_rate_maps.json")
    if os.path.isfile(p):
        rm = json.load(open(p))
        for r in rm:
            r["rate"] = np.array(r["rate"], dtype=float)
        D["rate_maps"] = rm
    p = os.path.join(here, "data_latents.npz")
    if os.path.isfile(p):
        z = np.load(p)
        D["latents"] = {k: z[k] for k in z.files}
    return D


def _meta(D, kw):
    m = kw.get("meta")
    if m is None:
        m = D.get("meta")
    if isinstance(m, (str, os.PathLike)) and os.path.isfile(m):
        m = json.load(open(m))
    return m if isinstance(m, dict) else {}


def _seeds(D):
    return sorted(int(s) for s in D["E"].seed.unique())


def _code(meta, s):
    return str((meta.get("animal_codes") or {}).get(str(s), s + 1))


def _is_norm(D):
    E = D["E"]
    return "ridge_norm" in E.columns and E["ridge_norm"].notna().any()


def _wide(D, dec, reps=REPS_REAL):
    """animal x rep table of the primary-axis value (normalized if available)."""
    E = D["E"]
    col = f"{dec}_norm" if (f"{dec}_norm" in E.columns and E[f"{dec}_norm"].notna().any()) else dec
    X = E.pivot_table(index="seed", columns="rep", values=col, aggfunc="mean")
    return X.reindex(columns=reps)


def _wide_cm(D, dec, reps=REPS_REAL):
    X = D["E"].pivot_table(index="seed", columns="rep", values=dec, aggfunc="mean")
    return X.reindex(columns=reps)


# Methods with a saved latent-d sweep (kNN-best d is defined only for these).
KNN_BEST_METHODS = ("pca", "dm", "lds", "isomap", "gpfa")


def _knn_best_wide(D, reps=REPS_REAL, value="norm"):
    """Animal x method: kNN error at each method's own kNN-best d (test-chosen).

    From the saved d_sweep only (no refit). Per session: argmin knn_median,
    tie-break lower d (Report 1). Per animal: mean of session values. Cohort
    aggregation matches elsewhere. Empty columns for methods without a sweep.
    """
    DS = D.get("d_sweep")
    seeds = _seeds(D)
    out = pd.DataFrame(index=seeds, columns=list(reps), dtype=float)
    if DS is None or DS.empty:
        return out
    ds = DS.copy()
    if "grid" in ds.columns:
        ds = ds[ds.grid == "final"]
    col = "knn_norm" if value == "norm" and "knn_norm" in ds.columns else "knn_median"
    if col not in ds.columns:
        return out
    for method in KNN_BEST_METHODS:
        if method not in reps:
            continue
        sub = ds[ds.method == method]
        if sub.empty:
            continue
        animal_vals = {}
        for seed, gseed in sub.groupby("seed"):
            sess_vals = []
            groups = gseed.groupby("session") if "session" in gseed.columns else [(None, gseed)]
            for _, g in groups:
                g = g.dropna(subset=[col, "knn_median"] if col != "knn_median" else [col])
                if g.empty:
                    continue
                best = g.sort_values(["knn_median", "d"]).iloc[0]
                sess_vals.append(float(best[col]))
            if sess_vals:
                animal_vals[int(seed)] = float(np.mean(sess_vals))
        for s, v in animal_vals.items():
            if s in out.index:
                out.loc[s, method] = v
    return out


def _ylab(D):
    return NORM_LAB if _is_norm(D) else "Median position error (cm)"


def cm_summary(D):
    """Per-method numbers: mean over animals of per-animal session-mean medians.

    Same aggregation everywhere (story, Fig3 footer, Fig6): animal = mean of its
    session medians; cohort = mean over animals. Reported as ``ridge_cm_mean``.
    """
    out = {}
    Xr, Xk = _wide(D, "ridge"), _wide(D, "knn")
    Cr, Ck = _wide_cm(D, "ridge"), _wide_cm(D, "knn")
    for r in REPS_REAL:
        out[r] = dict(
            ridge_mean=float(Xr[r].mean()) if r in Xr and Xr[r].notna().any() else float("nan"),
            ridge_sd=float(Xr[r].std(ddof=1)) if r in Xr and Xr[r].notna().sum() > 1 else 0.0,
            knn_mean=float(Xk[r].mean()) if r in Xk and Xk[r].notna().any() else float("nan"),
            knn_sd=float(Xk[r].std(ddof=1)) if r in Xk and Xk[r].notna().sum() > 1 else 0.0,
            ridge_cm_mean=float(Cr[r].mean()) if r in Cr and Cr[r].notna().any() else float("nan"),
            knn_cm_mean=float(Ck[r].mean()) if r in Ck and Ck[r].notna().any() else float("nan"),
            # aliases kept for older call sites
            ridge_cm_median=float(Cr[r].mean()) if r in Cr and Cr[r].notna().any() else float("nan"),
            knn_cm_median=float(Ck[r].mean()) if r in Ck and Ck[r].notna().any() else float("nan"),
        )
    out["chance_cm_median"] = float(D["F"].floor_median.mean())
    out["chance_cm_mean"] = float(D["F"].floor_median.mean())
    return out


def _contrast(D, a, b, dec):
    """Per-animal paired difference a - b (normalized units), Series indexed by animal."""
    C = D.get("contrasts")
    if C is not None and len(C) and _is_norm(D) and {"a", "b", "decoder", "delta"} <= set(C.columns):
        sub = C[(C.a == a) & (C.b == b) & (C.decoder == dec)]
        if len(sub):
            return sub.groupby("seed").delta.mean()
    X = _wide(D, dec)
    return (X[a] - X[b]).dropna()


def _verdict(v, tol=0.03):
    """Plain-language reading of a paired difference (first - second)."""
    v = np.asarray(v, float)
    n, k = len(v), neg(v)
    m = float(np.mean(v))
    if n == 0:
        return "n/a"
    if k == n:
        return "lower in all"
    if k == 0:
        return "higher in all"
    if abs(m) < tol:
        return "tie"
    return "lower in most" if k > n / 2 else "higher in most"


# ---------------------------------------------------------------- geometry
def _poly_session(D, session):
    """Room outline for one session (preferred; matches that session's path)."""
    if session is None:
        return None
    P = D.get("polygons_by_session") or {}
    p = P.get(str(session))
    return None if p is None else np.asarray(p, float)


def _poly(D, s, session=None):
    """Room outline. Prefer ``session``; else animal-keyed fallback."""
    if session is not None:
        p = _poly_session(D, session)
        if p is not None:
            return p
    # animal→matched session from tidy extract
    ses_map = D.get("polygon_session_by_animal") or {}
    if int(s) in ses_map:
        p = _poly_session(D, ses_map[int(s)])
        if p is not None:
            return p
    P = D.get("polygons") or {}
    p = P.get(int(s))
    if p is None:
        p = P.get(str(s))
    return None if p is None else np.asarray(p, float)


def _draw_poly(ax, poly, lw=0.7, color="#6e6c66"):
    if poly is not None:
        ax.add_patch(MplPolygon(poly, closed=True, fill=False, ec=color, lw=lw, zorder=1))


def _lims(D, seeds, pad=0.06, extra=None, sessions=None):
    """Shared symmetric room-local limits from polygons (+ optional extra xy)."""
    ext = []
    if sessions:
        for ses in sessions:
            p = _poly_session(D, ses)
            if p is not None:
                ext.append(np.abs(p).max())
    for s in seeds:
        p = _poly(D, s)
        if p is not None:
            ext.append(np.abs(p).max())
    if extra is not None and len(extra):
        ext.append(float(np.nanmax(np.abs(extra))))
    L = max(ext) if ext else 70.0
    return L * (1 + pad)


def _style_box(ax, lw=0.5):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("#6e6c66")
        sp.set_linewidth(lw)


def _split_masks(b, trim_end_s=10.0):
    """Train / test masks for one session's downsampled path.

    ``behavior_5hz.split`` labels everything that is not train as "test", which
    also covers the trimmed head and tail of the segment; those are dropped here.
    """
    t = b.time_s.values
    tr = (b.split == "train").values
    if not tr.any():
        return tr, ~tr
    t_tr0, t_tr1 = t[tr].min(), t[tr].max()
    te = (b.split == "test").values & (t > t_tr1) & (t <= t.max() - trim_end_s)
    return tr, te


def _pick_session(D, s, meta):
    """Session whose path+polygon are drawn for animal ``s``.

    Prefer the tidy ``polygon_session_by_animal`` (example session when it
    belongs to the animal; else first cohort session). Never use a bare
    lexicographic sort that can disagree with the polygon source.
    """
    ses_map = D.get("polygon_session_by_animal") or {}
    if int(s) in ses_map:
        return ses_map[int(s)]
    B = D["B"]
    ses = list(B[B.seed == s].session.unique()) if "session" in B.columns else [None]
    ex = meta.get("example_session_raw") or meta.get("example_session")
    return ex if ex in ses else (ses[0] if ses else None)


def _session_label(meta, session, seed=None):
    """Compact panel label: animal letter/code · session code."""
    codes = meta.get("session_codes") or {}
    # anon: session already S#; named: map raw→S# if present as values
    if session in (codes.values() if isinstance(codes, dict) else []):
        sess_lab = str(session)
    elif isinstance(codes, dict) and session in codes:
        sess_lab = str(codes[session])
    else:
        # try reverse map raw→code
        rev = {v: k for k, v in codes.items()} if isinstance(codes, dict) else {}
        # named meta stores session_codes as raw→raw or raw→S#
        sess_lab = str(codes.get(session, session)) if isinstance(codes, dict) else str(session)
    if seed is not None:
        return f"{_code(meta, seed)}\u00b7{sess_lab}"
    return sess_lab


def _coverage(D, s, meta, bin_cm=10.0):
    """Fraction of in-polygon ``bin_cm`` bins visited by the test path.

    Prefers tidy ``F.coverage`` (extract_tidy writes room coverage). Falls back
    to a polygon computation from behavior paths.
    """
    F = D["F"]
    if "coverage" in F.columns:
        v = F[F.seed == s].coverage.values
        if len(v) and np.isfinite(v).any():
            return float(np.nanmean(v))
    B = D["B"][D["B"].seed == s]
    vals = []
    sessions = list(B.session.unique()) if "session" in B.columns else [None]
    te_end = float(meta.get("trim_end_s", 10.0))
    for ses in sessions:
        b = B if ses is None else B[B.session == ses]
        poly = _poly(D, s, session=ses)
        tr, te = _split_masks(b, te_end)
        if not te.any():
            continue
        xy_all = b[["x_cm", "y_cm"]].values
        xy_te = xy_all[te]
        xy_all, xy_te = xy_all[np.isfinite(xy_all).all(1)], xy_te[np.isfinite(xy_te).all(1)]
        if not len(xy_te):
            continue
        lo = (poly.min(0) if poly is not None else xy_all.min(0)) - bin_cm
        hi = (poly.max(0) if poly is not None else xy_all.max(0)) + bin_cm
        gx = np.arange(lo[0], hi[0], bin_cm)
        gy = np.arange(lo[1], hi[1], bin_cm)
        ix_all = np.floor((xy_all - lo) / bin_cm).astype(int)
        ix_te = np.floor((xy_te - lo) / bin_cm).astype(int)
        vis_all = np.zeros((len(gx), len(gy)), bool)
        vis_te = np.zeros_like(vis_all)
        ok = (ix_all[:, 0] >= 0) & (ix_all[:, 0] < len(gx)) & (ix_all[:, 1] >= 0) & (ix_all[:, 1] < len(gy))
        vis_all[ix_all[ok, 0], ix_all[ok, 1]] = True
        ok = (ix_te[:, 0] >= 0) & (ix_te[:, 0] < len(gx)) & (ix_te[:, 1] >= 0) & (ix_te[:, 1] < len(gy))
        vis_te[ix_te[ok, 0], ix_te[ok, 1]] = True
        if poly is not None:
            cx, cy = np.meshgrid(gx + bin_cm / 2, gy + bin_cm / 2, indexing="ij")
            room = MplPath(poly).contains_points(np.c_[cx.ravel(), cy.ravel()]).reshape(cx.shape)
        else:
            room = vis_all
        if room.sum():
            vals.append(float((vis_te & room).sum() / room.sum()))
    return float(np.mean(vals)) if vals else float("nan")


def _floor_for(D, session=None, seed=None):
    FS = D.get("FS")
    if session is not None and FS is not None and "session" in FS.columns:
        v = FS[FS.session == session].floor_median.values
        if len(v):
            return float(v[0])
    F = D["F"]
    if seed is not None and (F.seed == seed).any():
        return float(F[F.seed == seed].floor_median.iloc[0])
    return float(F.floor_median.median())


def _best_causal(D, dec="ridge", k=3, avail=None):
    """k causal methods with the lowest median (across animals) normalized error."""
    X = _wide(D, dec)
    cand = [m for m in CAUSAL_CAND if (avail is None or m in avail) and m in X and X[m].notna().any()]
    med = X[cand].median().sort_values(kind="stable")
    return list(med.index[:k])


def _xt(ax, reps, rot=35, fs=None):
    kw = dict(fontsize=fs) if fs else {}
    ax.set_xticks(range(len(reps)), [SHORT[r] for r in reps], rotation=rot,
                  ha="right" if rot else "center", rotation_mode="anchor", **kw)
    for t, r in zip(ax.get_xticklabels(), reps):
        if r in QUAD_SET:
            t.set_color(EDGE[r])
            t.set_fontweight("bold")


def _mfc(rep):
    return "white" if rep in HOLLOW else COL[rep]


def _note(fig, x, y, text, fs=5.6, wrap_to=0.98, **kw):
    """Figure footnote; wrap so text stays inside the figure width."""
    import textwrap
    # ~0.01 figure-width per character at fs≈5.5 (Liberation Sans)
    char_w = 0.0105 * (fs / 5.5)
    max_chars = max(20, int((wrap_to - x) / char_w))
    # Preserve intentional newlines; wrap each paragraph
    parts = []
    for para in str(text).split("\n"):
        parts.append(textwrap.fill(para, width=max_chars) if para else "")
    wrapped = "\n".join(parts)
    fig.text(x, y, wrapped, fontsize=fs, color=INK2, va="top", **kw)


def _empty_panel(ax, text):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=6,
            color=MUTED, fontstyle="italic", linespacing=1.4)


BANDS = [("pca", ("pca", "pca_smooth"), "linear static"),
         ("dm", ("dm", "dm_smooth"), "nonlinear static"),
         ("lds", ("lds", "lds_smooth"), "linear dynamic"),
         ("lds", ("gpfa_causal",), "lin. dyn.")]


def _bands(ax, reps, label=True):
    for key, members, name in BANDS:
        idx = [reps.index(m) for m in members if m in reps]
        if not idx:
            continue
        lo, hi = min(idx) - 0.5, max(idx) + 0.5
        ax.axvspan(lo, hi, color=TINT[key], lw=0, zorder=0)
        if label:
            ax.text((lo + hi) / 2, 1.012, name, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=5.3, color=EDGE[key])


# ================================================================= FIG 1
def fig1(D, out, **kw):
    E, P, B = D["E"], D["P"], D["B"]
    meta = _meta(D, kw)
    seeds = _seeds(D)
    nA = len(seeds)
    ES = D.get("ES")
    nS = int(meta.get("n_sessions") or (ES.session.nunique() if ES is not None else len(seeds)))
    FW, FH = W_FULL, 150 * MM
    fig = plt.figure(figsize=(FW, FH))

    # a -- quadrant
    ax = fig.add_axes([0.0, 0.585, 0.27, 0.41]); ax.set_xlim(0, 48); ax.set_ylim(-8, 52)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0, 52, "a", fontsize=9, fontweight="bold", va="top")
    cw = 19.5
    cells = [((7.5, 25.5), "pca", "PCA", ""), ((28.0, 25.5), "lds", "LDS", "+ GPFA-c"),
             ((7.5, 4.0), "dm", "Diffusion\nmap", ""), ((28.0, 4.0), None, "", "")]
    for (x, y), key, name, sub in cells:
        if key:
            ax.add_patch(FancyBboxPatch((x, y), cw, cw, boxstyle="round,pad=0,rounding_size=1.2",
                                        fc=TINT[key], ec=EDGE[key], lw=0.9))
            ax.text(x + cw / 2, y + cw / 2 + (1.8 if sub else 0), name, ha="center", va="center",
                    fontsize=7.5, fontweight="bold", linespacing=1.1)
            if sub:
                ax.text(x + cw / 2, y + 5.0, sub, ha="center", va="center", fontsize=5.8, color=INK2)
        else:
            ax.add_patch(FancyBboxPatch((x, y), cw, cw, boxstyle="round,pad=0,rounding_size=1.2",
                                        fc="white", ec=MUTED, lw=0.8, ls=(0, (2, 1.5)), hatch="/////"))
            ax.text(x + cw / 2, y + cw / 2, "empty", ha="center", va="center", fontsize=6.5,
                    color=INK2, fontstyle="italic", bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(17.25, 47.0, "Static", ha="center", fontsize=7, color=INK2)
    ax.text(37.75, 47.0, "Dynamic", ha="center", fontsize=7, color=INK2)
    ax.text(4.5, 35.25, "Linear", ha="center", va="center", rotation=90, fontsize=7, color=INK2)
    ax.text(4.5, 13.75, "Nonlinear", ha="center", va="center", rotation=90, fontsize=7, color=INK2)
    ax.text(7.5, 1.2, "Controls  raw counts | raw + 250 ms history | raw + EMA\n"
            "Each method also with causal EMA (Ridge only)\n"
            "Baselines  Isomap | GPFA-off (offline smoother)",
            fontsize=5.5, color=INK2, va="top", linespacing=1.4)

    # b -- pipeline + split timeline (no simulator, no ground truth)
    ax = fig.add_axes([0.29, 0.585, 0.71, 0.41]); ax.set_xlim(0, 128); ax.set_ylim(0, 61); ax.axis("off")
    ax.text(0, 61, "b", fontsize=9, fontweight="bold", va="top")
    boxes = ["Recorded\nspike times\n(sorted units,\none room segment)",
             "Causal counts\n[t - W, t)\nW = 250 ms, step 50 ms\nsqrt, z-score",
             "Representation\nE\n(fit on training\nblock only)",
             "Decoder\nRidge (primary)\nkNN (readout)\n+ optional causal\nEMA (Ridge only)"]
    bw, gap, y0, bh = 25.0, 6.0, 27, 22
    xs = [3 + i * (bw + gap) for i in range(4)]
    for i, (t, x) in enumerate(zip(boxes, xs)):
        hl = i == 2
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc="#fff6d6" if hl else "#f3f2ef", ec="#b98200" if hl else "#6e6c66", lw=0.7))
        ax.text(x + bw / 2, y0 + bh / 2, t, ha="center", va="center", fontsize=6, linespacing=1.3)
        x_to = xs[i + 1] - 0.4 if i < 3 else x + bw + 3.4
        ax.annotate("", (x_to, y0 + bh / 2), (x + bw + 0.4, y0 + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2, mutation_scale=6))
    ax.text(xs[3] + bw + 3.6, y0 + bh / 2, "x,y", va="center", fontsize=6)
    ax.text(xs[1] + bw / 2, y0 + bh + 6.5,
            "source-file Cell_* is a centred window (+125 ms look-ahead): never a feature (Fig. 2d)",
            ha="center", fontsize=5.6, color=FAILC)
    if ES is not None and "n_units" in ES.columns and ES.n_units.notna().any():
        ax.text(xs[0] + bw / 2, y0 - 2.5, f"{int(ES.n_units.min())}-{int(ES.n_units.max())} units / session",
                ha="center", fontsize=5.8, color=INK2)
    ax.text(xs[2] + bw / 2, y0 - 2.5, "latent d = 2 ... 80 (PCA, LDS)\nd <= 20 (DM, Isomap, GPFA)",
            ha="center", va="top", fontsize=5.6, color=INK2, linespacing=1.3)
    ax.text(xs[3] + bw / 2, y0 - 2.5, "identical for every E", ha="center", fontsize=5.8, color=INK2)
    ty, th = 9.5, 4.0
    L0, L1 = 3, 125
    seg = [("trim", 0, 7), ("train", 7, 72), ("purge", 72, 73), ("test", 73, 96), ("trim", 96, 100)]
    tx = lambda p: L0 + (L1 - L0) * p / 100
    for kind, a_, b_ in seg:
        if kind == "trim":
            ax.add_patch(Rectangle((tx(a_), ty), tx(b_) - tx(a_), th, fc="white", ec=MUTED, lw=0.5, hatch="////"))
        elif kind == "train":
            nf = 5
            for k in range(nf):
                a2 = a_ + (b_ - a_) * k / nf
                b2 = a_ + (b_ - a_) * (k + 1) / nf
                ax.add_patch(Rectangle((tx(a2), ty), tx(b2) - tx(a2) - 0.3, th,
                                       fc="#b3b1aa" if k == 2 else "#dcdad4", ec="none"))
        elif kind == "test":
            ax.add_patch(Rectangle((tx(a_), ty), tx(b_) - tx(a_), th, fc="#3a3936", ec="none"))
    ax.text(tx(39.5), ty + th + 1.6, "training block (80%) | 5 contiguous inner-CV folds with purge gaps (one held out)",
            ha="center", fontsize=5.6, color=INK2)
    ax.text(tx(84.5), ty + th + 1.6, "test (20%)", ha="center", fontsize=5.6, color=INK2)
    ax.text(tx(3.5), ty - 2.0, "first 60 s\ntrimmed", ha="center", va="top", fontsize=5.2, color=INK2, linespacing=1.2)
    ax.text(tx(98), ty - 2.0, "last 10 s\ntrimmed", ha="center", va="top", fontsize=5.2, color=INK2, linespacing=1.2)
    ax.annotate("1 s purge gap", (tx(72.5), ty), (tx(64), ty - 5.5), fontsize=5.4, color=INK2,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2))
    ax.text(L0, -2.8, "One room segment (room A, schematic, not to scale). Every split and fit stays inside one session; "
            f"{nA} animals, {nS} sessions.\nEvery method is scored on the same test frames (paired design); "
            "animal = independent unit (1-2 sessions averaged).", fontsize=5.6, color=INK2)

    # c -- population by region (pooled + per animal)
    head_fig(fig, 0.0, 0.535, "c", "Recorded population by region")
    Pr = P[P.kind == "region"] if "kind" in P.columns else P
    pooled = Pr.groupby("label").n.sum().sort_values(ascending=False)
    order = list(pooled.index)
    cmap_r = {r: REGION_COLS[i % len(REGION_COLS)] for i, r in enumerate(order)}
    ax = fig.add_axes([0.075, 0.385, 0.115, 0.125])
    yy = np.arange(len(order))[::-1]
    ax.barh(yy, pooled.values, height=0.62, color=[cmap_r[r] for r in order])
    for y, v in zip(yy, pooled.values):
        ax.text(v + pooled.max() * 0.02, y, str(int(v)), va="center", fontsize=5.8)
    ax.set_yticks(yy, order); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, pooled.max() * 1.28); ax.set_xlabel(f"Units, pooled ({nS} sessions)", fontsize=6)
    ax = fig.add_axes([0.255, 0.385, 0.14, 0.125])
    for j, s in enumerate(seeds):
        sub = Pr[Pr.seed == s]
        nses = sub.session.nunique() if "session" in sub.columns else 1
        bot = 0.0
        for r in order:
            v = sub[sub.label == r].n.sum() / max(nses, 1)
            ax.bar(j, v, bottom=bot, width=0.7, color=cmap_r[r], lw=0)
            bot += v
    ax.set_xticks(range(nA), [_code(meta, s) for s in seeds], fontsize=5.5, rotation=35, ha="right",
                  rotation_mode="anchor")
    ax.set_ylabel("Units / session"); ax.set_xlabel("Animal")

    # d -- one panel per cohort session (path + polygon from the SAME session)
    head_fig(fig, 0.435, 0.535, "d", "Test path (black) vs training path (grey); one panel per session")
    if "session" in B.columns:
        units = []
        for s in seeds:
            for ses in B[B.seed == s].session.unique():
                units.append((int(s), str(ses)))
    else:
        units = [(int(s), _pick_session(D, s, meta)) for s in seeds]
    n_pan = max(len(units), 1)
    gapx = 0.006
    x0 = 0.445
    span = 0.545
    w = (span - (n_pan - 1) * gapx) / n_pan
    allxy = []
    panels = []
    for s, ses in units:
        b = B[(B.seed == s) & ((B.session == ses) if "session" in B.columns else True)]
        if len(b):
            allxy.append(b[["x_cm", "y_cm"]].values)
        panels.append((s, ses, b))
    L = _lims(D, seeds, extra=np.vstack(allxy) if allxy else None,
              sessions=[ses for _, ses, _ in panels])
    FS = D.get("FS")
    for j, (s, ses, b) in enumerate(panels):
        ax = fig.add_axes([x0 + j * (w + gapx), 0.385, w, w * FW / FH])
        tr, te = _split_masks(b, float(meta.get("trim_end_s", 10.0)))
        _draw_poly(ax, _poly(D, s, session=ses))
        if len(b):
            ax.plot(b.x_cm[tr], b.y_cm[tr], color="#d9d7d1", lw=0.3, zorder=2)
            ax.plot(b.x_cm[te], b.y_cm[te], color=INK, lw=0.45, zorder=3)
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
        _style_box(ax)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("y (cm)", labelpad=-2, fontsize=5.5)
        cov = float("nan")
        if FS is not None and "session" in FS.columns and "coverage" in FS.columns:
            v = FS[FS.session == ses].coverage.values
            if len(v) and np.isfinite(v).any():
                cov = float(np.nanmean(v))
        if not np.isfinite(cov):
            cov = _coverage(D, s, meta)
        title = _session_label(meta, ses, seed=s)
        if np.isfinite(cov):
            title += f"\n{cov:.2f}"
        ax.set_title(title, fontsize=4.6, pad=1.0, linespacing=0.95)
    _note(fig, 0.445, 0.340,
          "Path and polygon from the same session (room-local cm; do not mix). "
          "Title = animal·session; cov = in-polygon 10 cm bins visited by test path.",
          fs=4.7, wrap_to=0.99)

    # e -- example train-only rate maps
    head_fig(fig, 0.0, 0.245, "e", "Example units: train-only rate maps")
    RM = D.get("rate_maps")
    if RM:
        cand = [r for r in RM if r.get("stable", True)]
        cand = sorted(cand, key=lambda r: (-float(r.get("spat_info", 0.0)), int(r.get("unit_id", 0))))[:8]
        wm = 0.098
        for j, r in enumerate(cand):
            ax = fig.add_axes([0.02 + j * (wm + 0.0085), 0.065, wm, wm * FW / FH])
            rate = np.ma.masked_invalid(np.asarray(r["rate"], float))
            ext = list(r["extent"])
            im = ax.imshow(rate, origin="lower", extent=ext, cmap="viridis", vmin=0,
                           vmax=float(rate.max()) if rate.count() else 1.0, interpolation="nearest")
            if r.get("seed") is not None:
                _draw_poly(ax, _poly(D, r["seed"], session=r.get("session")), lw=0.5, color="white")
            ax.set_xticks([]); ax.set_yticks([]); _style_box(ax, 0.4)
            ax.set_title(f"u{int(r.get('unit_id', j))} | SI {float(r.get('spat_info', np.nan)):.2f}\n"
                         f"peak {float(rate.max()) if rate.count() else 0:.1f} Hz",
                         fontsize=5.2, pad=2, linespacing=1.15)
        _note(fig, 0.02, 0.040, "Example session (fixed rule in meta). Train frames only; bins with occupancy < 0.5 s masked; "
              "Gaussian-smoothed; Hz.\nUnits ranked by train spatial information (SI, bits/spike) among split-half-stable units; "
              "each map scaled to its own peak.", fs=5.2)
    else:
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.16])
        _empty_panel(ax, "rate maps not supplied (D['rate_maps'])")
    save(fig, out, "Fig1_design")


# ================================================================= FIG 2
def _audit_table(D):
    A, E = D["A"], D["E"]
    seeds = _seeds(D)
    st = A.pivot_table(index="check", columns="seed", values="status", aggfunc="first")
    st = st.reindex(index=[k for k, _ in AUD], columns=seeds).fillna("N/A")
    st = st.astype(object)
    for k in SIM_ONLY:
        st.loc[k, :] = "N/A"
    if {"a13_ridge_pass", "a13_knn_pass"} <= set(E.columns):
        for s in seeds:
            e = E[E.seed == s]
            if len(e):
                ok = bool(e[["a13_ridge_pass", "a13_knn_pass"]].astype(bool).all().all())
                st.loc["A13", s] = "PASS" if ok else "FAIL"
    return st


def _tick_path(ax, x, y, kind, color):
    if kind == "PASS":
        ax.plot([x + 0.22, x + 0.38, x + 0.68], [y + 0.45, y + 0.62, y + 0.24], color=color, lw=0.8,
                solid_capstyle="round", zorder=3)
    elif kind == "FAIL":
        ax.plot([x + 0.22, x + 0.68], [y + 0.20, y + 0.68], color=color, lw=0.9, zorder=3)
        ax.plot([x + 0.22, x + 0.68], [y + 0.68, y + 0.20], color=color, lw=0.9, zorder=3)
    else:
        ax.plot([x + 0.28, x + 0.62], [y + 0.44, y + 0.44], color=color, lw=0.7, zorder=3)


def _feat_cols(cf):
    pick = lambda names: next((c for c in names if c in cf.columns), None)
    return (pick(("t_s", "time_s", "t")), pick(("centred", "centered", "cell_star", "cell")),
            pick(("causal", "causal_count")))


def fig2(D, out, **kw):
    E = D["E"]
    meta = _meta(D, kw)
    seeds = _seeds(D)
    nA = len(seeds)
    fig = plt.figure(figsize=(W_FULL, 112 * MM))

    # a -- audit grid
    ax = fig.add_axes([0.205, 0.07, 0.14, 0.77])
    st = _audit_table(D)
    n = len(AUD)
    for i, (k, _) in enumerate(AUD):
        for j, s in enumerate(seeds):
            v = st.loc[k, s]
            fc = {"PASS": "#eeede9", "FAIL": FAILC, "N/A": "white"}.get(v, "white")
            ax.add_patch(Rectangle((j, i), 0.9, 0.86, fc=fc, ec="#d0cec8" if v != "FAIL" else FAILC, lw=0.4))
            _tick_path(ax, j, i, v, "white" if v == "FAIL" else (INK2 if v == "PASS" else MUTED))
    ax.set_xlim(0, nA); ax.set_ylim(n, 0)
    ax.set_yticks(np.arange(n) + 0.45, [f"{k}  {t}" for k, t in AUD], fontsize=5.9)
    ax.set_xticks(np.arange(nA) + 0.45, [_code(meta, s) for s in seeds], fontsize=5.5, rotation=90)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.25, -0.9, "animal", ha="right", fontsize=6.3, color=INK2)
    head_fig(fig, 0.0, 0.935, "a", f"Fairness audit, 15 checks x {nA} animals")
    fl = E[~E.a13_ridge_pass.astype(bool) | ~E.a13_knn_pass.astype(bool)] if "a13_ridge_pass" in E else E.iloc[0:0]
    ftxt = "; ".join(f"{_code(meta, s)}: " + ", ".join(SHORT[r] for r in g.rep) for s, g in fl.groupby("seed"))
    msg = f"A13 fails - {ftxt} (see b)" if len(fl) else "no failures"
    ax.text(nA / 2, n + 0.35, msg, ha="center", va="top", fontsize=5.6, color=FAILC if len(fl) else INK2)
    ax.text(nA / 2, n + 1.5, "-  n/a: simulation-only check", ha="center", va="top", fontsize=5.4, color=MUTED)

    # b -- A13 strips (Ridge + kNN; no ground-truth branch)
    ax = fig.add_axes([0.44, 0.60, 0.55, 0.29])
    head_fig(fig, 0.395, 0.935, "b", "Time-shift null control (A13): shifted-label decoding sits at chance")
    reps = [r for r in REPS_REAL if not r.endswith("_smooth")]
    offs = {"a13_ridge": -0.17, "a13_knn": 0.17}
    ax.axhspan(-8, -2, color="#fbe9e9", lw=0, zorder=0)
    ax.axhline(-2, color=FAILC, lw=0.7, ls=(0, (3, 2)))
    ax.axhline(0, color=MUTED, lw=0.5)
    ymax = 4.0
    for j, rep in enumerate(reps):
        d = E[E.rep == rep].sort_values("seed")
        got = False
        for colname, mk, filled in (("a13_ridge", "o", True), ("a13_knn", "s", False)):
            v = d[colname].values if colname in d else np.array([])
            if len(v) and np.isfinite(v).any():
                got = True
                x = j + offs[colname] + np.linspace(-0.05, 0.05, len(v))
                ax.scatter(x, v, s=8, marker=mk, facecolor=COL[rep] if filled else "white",
                           edgecolor=EDGE[rep], lw=0.55, zorder=3)
                ymax = max(ymax, float(np.nanmax(v)))
                bad = v < -2
                if bad.any():
                    ax.scatter(x[bad], v[bad], s=40, facecolor="none", edgecolor=FAILC, lw=0.8, zorder=4)
        if not got:
            ax.text(j, 0.6, "n/a", ha="center", fontsize=5.5, color=MUTED, fontstyle="italic")
    _xt(ax, reps, rot=0, fs=6)
    ax.text(-0.55, -2.25, "fail < -2 cm", color=FAILC, fontsize=5.6, va="top")
    ax.set_ylabel("Shifted - chance floor\n(cm, median over shifts)")
    ax.set_ylim(-5.0, ymax * 1.25 + 1); ax.set_xlim(-0.6, len(reps) - 0.4)
    hs = [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=3.3, label="Ridge"),
          Line2D([], [], marker="s", ls="", mfc="white", mec=INK2, ms=3.3, label="kNN")]
    ax.legend(handles=hs, loc="upper right", ncol=2, handletextpad=0.1, columnspacing=0.9)
    ax.text(0.01, 0.97, "EMA variants inherit the base-method null (Ridge-only post-processing)",
            transform=ax.transAxes, fontsize=5.2, color=MUTED, va="top")

    # c -- held-out test vs inner-CV (selected d, session level)
    ax = fig.add_axes([0.44, 0.10, 0.18, 0.33])
    head_fig(fig, 0.395, 0.475, "c", "Held-out error tracks inner-CV error")
    DS = D.get("d_sweep")
    ok = DS is not None and "inner_cv_ridge_median" in DS and DS.inner_cv_ridge_median.notna().any()
    if ok:
        sel = DS[DS.selected.astype(bool)]
        if "grid" in sel:
            sel = sel[sel.grid == "final"]
        sel = sel.dropna(subset=["inner_cv_ridge_median", "ridge_median"])
        lo = float(min(sel.inner_cv_ridge_median.min(), sel.ridge_median.min())) - 2
        hi = float(max(sel.inner_cv_ridge_median.max(), sel.ridge_median.max())) + 2
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, ls=(0, (3, 2)))
        for rep in ("pca", "dm", "lds", "isomap", "gpfa"):
            d = sel[sel.method == rep]
            ax.scatter(d.inner_cv_ridge_median, d.ridge_median, s=9, marker=MRK[rep], facecolor=_mfc(rep),
                       edgecolor=EDGE[rep], lw=0.55, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.legend(handles=[Line2D([], [], marker=MRK[r], ls="", mfc=_mfc(r), mec=EDGE[r], ms=3.2, label=SHORT[r])
                           for r in ("pca", "dm", "lds", "isomap", "gpfa")], loc="lower right", fontsize=5.2,
                  handletextpad=0.1, labelspacing=0.3)
        ax.set_xlabel("Inner-CV error (cm)"); ax.set_ylabel("Held-out test error (cm)")
        gap = sel.ridge_median - sel.inner_cv_ridge_median
        ax.text(0.04, 0.96, f"test - CV: {gap.median():+.1f} cm\n(median, {len(sel)} session x method)",
                transform=ax.transAxes, fontsize=5.6, color=INK2, va="top")
    else:
        _empty_panel(ax, "inner-CV error not exported\nfor this cohort")

    # d -- features are causal
    head_fig(fig, 0.665, 0.475, "d", "Features are causal (not centred)")
    cf = D.get("causal_features")
    axd = fig.add_axes([0.70, 0.355, 0.29, 0.10] if cf is not None else [0.70, 0.17, 0.29, 0.20])
    axd.set_xlim(-0.30, 0.22); axd.set_ylim(-0.2, 2.15)
    axd.axvspan(0, 0.125, color="#fbe9e9", lw=0, zorder=0)
    axd.add_patch(Rectangle((-0.125, 1.15), 0.25, 0.55, fc="#f3c9c8", ec=FAILC, lw=0.7))
    axd.add_patch(Rectangle((-0.25, 0.25), 0.25, 0.55, fc="#dcdad4", ec=INK2, lw=0.7))
    axd.axvline(0, color=INK, lw=0.9)
    axd.text(0.0, 2.0, "label time t", ha="center", va="bottom", fontsize=5.6)
    axd.text(0.0, 1.425, "centred (source file)", ha="center", va="center", fontsize=5.4)
    axd.text(-0.125, 0.525, "causal [t - W, t)", ha="center", va="center", fontsize=5.4)
    axd.text(0.0625, 0.525, "125 ms\nlook-ahead\nnever used", ha="center", va="center", fontsize=5.0,
             color=FAILC, linespacing=1.15)
    axd.set_yticks([]); axd.spines["left"].set_visible(False)
    axd.set_xticks([-0.25, -0.125, 0, 0.125]); axd.set_xticklabels(["-250", "-125", "0", "+125"], fontsize=5.5)
    axd.set_xlabel("time relative to t (ms)", fontsize=5.8, labelpad=1)
    if cf is not None and len(cf):
        tc, cc, kc = _feat_cols(cf)
        axt = fig.add_axes([0.70, 0.10, 0.29, 0.15])
        sub = cf.sort_values(tc)
        t0 = float(sub[tc].iloc[0])
        t = sub[tc].values - t0
        centred = sub[cc].values.astype(float)
        causal = sub[kc].values.astype(float)
        axt.plot(t, centred, color=FAILC, lw=1.2, label="centred Cell_*", zorder=3)
        axt.plot(t, causal, color=INK, lw=1.2, label="causal [t-250,t)", zorder=2)
        ymax = max(float(np.nanmax(np.r_[centred, causal])), 1.0)
        axt.set_xlim(0, max(float(t[-1]) if len(t) else 5.0, 1.0))
        axt.set_ylim(0, ymax * 1.35)
        axt.set_xlabel("Time in window (s)", fontsize=5.8)
        axt.set_ylabel("Spike count\n(250 ms)", fontsize=6)
        uid = int(sub["unit_id"].iloc[0]) if "unit_id" in sub.columns else "?"
        axt.set_title(f"unit {uid}: centred leads causal by 125 ms", fontsize=5.2, color=INK2, pad=1.5)
        axt.legend(loc="upper right", ncol=2, fontsize=5.4, handlelength=1.2, columnspacing=0.8)
        axt.tick_params(labelsize=5.5)
    else:
        axd.text(0.5, -0.62, "feature series not supplied (D['causal_features'])",
                 transform=axd.transAxes, ha="center", fontsize=5.2, color=MUTED, fontstyle="italic")
    save(fig, out, "Fig2_validity")


# ================================================================= FIG 3
def _slope_real(ax, X, reps, norm, ylim, ylab=None, note_missing=True):
    xs = np.arange(len(reps))
    _bands(ax, reps)
    if norm:
        ax.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.text(len(reps) - 0.45, 1.0 + 0.012, "chance (predict training mean)", fontsize=5.5, color=INK2,
                ha="right", va="bottom")
    for s in X.index:
        y = X.loc[s, reps].values.astype(float)
        ok = np.isfinite(y)
        ax.plot(xs[ok], y[ok], color="#d3d1cb", lw=0.6, zorder=1)
    for j, rep in enumerate(reps):
        v = X[rep].dropna().values
        if len(v) == 0:
            if note_missing:
                ax.text(j, ylim[0] + 0.04 * (ylim[1] - ylim[0]), "n/a", ha="center", fontsize=5.2,
                        color=MUTED, fontstyle="italic")
            continue
        ax.scatter(np.full(len(v), j), v, s=7, facecolor=GREY_PT, edgecolor="none", zorder=2)
        ax.plot([j - 0.3, j + 0.3], [v.mean()] * 2, color=EDGE[rep], lw=2.0, solid_capstyle="round", zorder=3)
    _xt(ax, reps)
    ax.set_ylim(*ylim); ax.set_xlim(-0.6, len(reps) - 0.4)
    if ylab:
        ax.set_ylabel(ylab)


def _ylim_norm(D):
    vals = np.concatenate([_wide(D, d).values.ravel() for d in ("ridge", "knn")])
    vals = vals[np.isfinite(vals)]
    if _is_norm(D):
        return (max(0.0, np.floor(vals.min() * 10 - 1) / 10), max(1.08, np.ceil(vals.max() * 10 + 0.5) / 10))
    return (0, vals.max() * 1.15)


def _contrast_rows(ax, D, rows, y0):
    """Draw contrast rows (Ridge filled above, kNN open below); returns nothing."""
    for i, (a, b, lab) in enumerate(rows):
        y = y0 - i
        for dec, dy, filled in (("ridge", 0.17, True), ("knn", -0.17, False)):
            v = _contrast(D, a, b, dec)
            if dec == "knn" and (a.endswith("_smooth") or b.endswith("_smooth")):
                v = v.iloc[0:0]
            if len(v) == 0:
                continue
            vv = v.values
            ax.scatter(vv, np.full(len(vv), y + dy), s=8, marker=MRK[a], facecolor=COL[a] if filled else "white",
                       edgecolor=EDGE[a], lw=0.55, zorder=3)
            ax.plot([vv.mean()] * 2, [y + dy - 0.13, y + dy + 0.13], color=INK, lw=1.2, zorder=4)
            k = neg(vv)
            ax.text(1.03, y + dy, f"{k}/{len(vv)}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=6, color=INK if k in (0, len(vv)) else MUTED,
                    fontweight="bold" if k in (0, len(vv)) else None)
        ax.axhline(y - 0.5, color=GRID, lw=0.5)


def fig3(D, out, **kw):
    meta = _meta(D, kw)
    seeds = _seeds(D)
    nA = len(seeds)
    norm = _is_norm(D)
    Xr, Xk = _wide(D, "ridge"), _wide(D, "knn")
    Xkb = _knn_best_wide(D, value="norm" if norm else "cm")
    ylim = _ylim_norm(D)
    # include knn-best in ylim so second markers are not clipped
    kb_vals = Xkb.values.ravel().astype(float)
    kb_vals = kb_vals[np.isfinite(kb_vals)]
    if len(kb_vals):
        ylim = (min(ylim[0], max(0.0, np.floor(kb_vals.min() * 10 - 1) / 10)),
                max(ylim[1], np.ceil(kb_vals.max() * 10 + 0.5) / 10))
    fig = plt.figure(figsize=(W_FULL, 124 * MM))
    a = fig.add_axes([0.07, 0.585, 0.52, 0.31])
    _slope_real(a, Xr, REPS_REAL, norm, ylim, _ylab(D))
    a.set_xticklabels([])
    head_fig(fig, 0.0, 0.955, "a", "Ridge readout (primary)")
    b = fig.add_axes([0.07, 0.14, 0.52, 0.31], sharey=a)
    _slope_real(b, Xk, REPS_REAL, norm, ylim, _ylab(D))
    # second marker: kNN at kNN-best d (descriptive, chosen on test)
    for j, rep in enumerate(REPS_REAL):
        if rep not in KNN_BEST_METHODS or rep not in Xkb.columns:
            continue
        v = Xkb[rep].dropna().values
        if len(v) == 0:
            continue
        b.scatter(np.full(len(v), j), v, s=9, marker="D", facecolor="white",
                  edgecolor=EDGE[rep], lw=0.7, zorder=4)
        b.plot([j - 0.3, j + 0.3], [v.mean()] * 2, color=EDGE[rep], lw=1.0,
               ls=(0, (2, 1.5)), solid_capstyle="round", zorder=4)
    head_fig(fig, 0.0, 0.505, "b", "kNN readout (sensitivity; EMA is Ridge-only)")
    a.text(0.995, 0.03, "lines join animals; bar = mean over animals", transform=a.transAxes, ha="right",
           fontsize=5.2, color=MUTED)
    hs_b = [
        Line2D([], [], marker="o", ls="", mfc=GREY_PT, mec="none", ms=3.2,
               label="kNN @ Ridge-selected d (primary)"),
        Line2D([], [], marker="D", ls="", mfc="white", mec=INK2, ms=3.2,
               label="kNN-best d (descriptive, chosen on test)"),
    ]
    # lower-left: avoid overlap with chance label at y=1
    b.legend(handles=hs_b, loc="lower left", fontsize=4.8, frameon=True, fancybox=False,
             framealpha=0.92, edgecolor="#e8e6e1", handletextpad=0.25, borderaxespad=0.2,
             borderpad=0.3)

    # c -- planned contrasts (leave room on the right for k/N)
    rows = PRIMARY + SECONDARY
    n = len(rows)
    ax = fig.add_axes([0.78, 0.16, 0.14, 0.735])
    head_fig(fig, 0.615, 0.955, "c", "Planned paired contrasts (animal level)")
    allv = []
    for a_, b_, _ in rows:
        for dec in ("ridge", "knn"):
            v = _contrast(D, a_, b_, dec)
            if dec == "knn" and (a_.endswith("_smooth") or b_.endswith("_smooth")):
                continue
            allv.extend(v.values.tolist())
    lo, hi = (min(allv + [0]) - 0.03, max(allv + [0]) + 0.03) if allv else (-0.2, 0.2)
    _contrast_rows(ax, D, rows[:len(PRIMARY)], n - 1)
    _contrast_rows(ax, D, rows[len(PRIMARY):], n - 1 - len(PRIMARY) - 0.6)
    ypos = [n - 1 - i for i in range(len(PRIMARY))] + [n - 1 - len(PRIMARY) - 0.6 - i for i in range(len(SECONDARY))]
    ax.set_yticks(ypos, [f"{lab}\n{SHORT[x]} - {SHORT[y]}" for x, y, lab in rows], fontsize=5.8)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(ypos[-1] - 0.6, n - 0.4)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Paired difference ({'normalized error' if norm else 'cm'})\n<- first term lower error")
    ax.text(1.02, 1.0, f"k/{nA} < 0", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5, color=INK2)
    ysep = n - 1 - len(PRIMARY) + 0.1
    ax.axhline(ysep, color=INK2, lw=0.5, ls=(0, (2, 2)))
    ax.text(lo, ysep + 0.12, "primary (pre-registered)", fontsize=5.0, color=MUTED, va="bottom")
    ax.text(lo, ysep - 0.12, "secondary (no EMA)", fontsize=5.0, color=MUTED, va="top")
    hs = [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=3.3, label="Ridge"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=3.3, label="kNN")]
    ax.legend(handles=hs, loc="upper left", ncol=1, handletextpad=0.1, borderaxespad=0.2)
    cs = cm_summary(D)
    # two short lines so footer stays inside figure width
    _note(fig, 0.02, 0.055,
          f"N = {nA} animals (mean over animals of per-animal session means); sign counts, no p-values. "
          f"EMA = causal moving average on Ridge outputs (Ridge-only). {GRID20_NOTE}.",
          fs=5.0)
    _note(fig, 0.02, 0.028,
          f"Mean cm: chance {cs['chance_cm_mean']:.1f}; Raw {cs['raw']['ridge_cm_mean']:.1f}; "
          f"Raw+EMA {cs['raw_smooth']['ridge_cm_mean']:.1f}; LDS {cs['lds']['ridge_cm_mean']:.1f}; "
          f"LDS+EMA {cs['lds_smooth']['ridge_cm_mean']:.1f}; GPFA-c {cs['gpfa_causal']['ridge_cm_mean']:.1f} (Ridge).",
          fs=5.0)
    save(fig, out, "Fig3_quadrant_answer")


# ================================================================= FIG 4
def fig4(D, out, **kw):
    meta = _meta(D, kw)
    nA = len(_seeds(D))
    norm = _is_norm(D)
    Xr, Xk = _wide(D, "ridge"), _wide(D, "knn")
    Xkb = _knn_best_wide(D, value="norm" if norm else "cm")
    reps = [r for r in REPS_REAL if not r.endswith("_smooth") and r != "gpfa_causal"]
    fig = plt.figure(figsize=(W_FULL, 80 * MM))

    # a -- Ridge vs kNN (filled = Ridge-selected d; open diamond = kNN-best d)
    ax = fig.add_axes([0.08, 0.18, 0.28, 0.66])
    head_fig(fig, 0.0, 0.93, "a", "Readout: linear (Ridge) vs nonlinear (kNN)")
    vals = np.concatenate([
        Xr[reps].values.ravel(), Xk[reps].values.ravel(), Xkb[reps].values.ravel(),
    ])
    vals = vals[np.isfinite(vals)]
    lo, hi = (np.floor(vals.min() * 20) / 20 - 0.03, np.ceil(vals.max() * 20) / 20 + 0.03) if len(vals) else (0, 1)
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
    for rep in reps:
        ax.scatter(Xr[rep], Xk[rep], s=10, marker=MRK[rep], facecolor=_mfc(rep),
                   edgecolor=EDGE[rep], lw=0.55, zorder=3)
        if rep in KNN_BEST_METHODS and rep in Xkb.columns and Xkb[rep].notna().any():
            ax.scatter(Xr[rep], Xkb[rep], s=12, marker="D", facecolor="white",
                       edgecolor=EDGE[rep], lw=0.7, zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    unit = "normalized error" if norm else "cm"
    ax.set_xlabel(f"Ridge ({unit})"); ax.set_ylabel(f"kNN ({unit})")
    ax.text(hi - 0.01 * (hi - lo), lo + 0.03 * (hi - lo), "below diagonal: kNN lower", fontsize=5.4, color=INK2,
            ha="right")
    g = (Xr - Xk)
    gb = (Xr - Xkb)
    show = [r for r in ("raw", "pca", "dm", "lds") if r in g]
    gtxt = "; ".join(f"{SHORT[r]} {g[r].mean():+.3f}" for r in show)
    gbtxt = "; ".join(
        f"{SHORT[r]} {gb[r].mean():+.3f}" for r in show if r in KNN_BEST_METHODS and gb[r].notna().any()
    )
    ax.text(0.03, 0.97,
            f"Ridge - kNN @ Ridge-sel d:\n{gtxt}\n"
            f"Ridge - kNN-best d (descriptive):\n{gbtxt}",
            transform=ax.transAxes, fontsize=4.8, color=INK2, va="top", linespacing=1.25)
    hs = [Line2D([], [], marker=MRK[r], ls="", mfc=_mfc(r), mec=EDGE[r], ms=3.6, label=SHORT[r]) for r in reps]
    hs += [
        Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=3.2,
               label="kNN @ Ridge-selected d"),
        Line2D([], [], marker="D", ls="", mfc="white", mec=INK2, ms=3.2,
               label="kNN-best d (descriptive, chosen on test)"),
    ]
    # gap between panels: keep clear of axes
    fig.legend(handles=hs, loc="center left", bbox_to_anchor=(0.40, 0.52), fontsize=5.2,
               handletextpad=0.2, labelspacing=0.35, frameon=False)

    # b -- LDS - Raw+hist by readout
    ax = fig.add_axes([0.68, 0.18, 0.28, 0.66])
    head_fig(fig, 0.60, 0.93, "b", "Dynamics beyond history: LDS - Raw+hist")
    allv = []
    for xpos, (dec, X) in enumerate((("Ridge", Xr), ("kNN", Xk))):
        v = (X["lds"] - X["raw_lag"]).dropna().values
        allv.extend(v.tolist())
        ax.scatter(np.full(len(v), xpos) + np.linspace(-0.12, 0.12, len(v)), v, s=10, marker="^",
                   facecolor=COL["lds"], edgecolor=EDGE["lds"], lw=0.55, zorder=3)
        ax.plot([xpos - 0.25, xpos + 0.25], [v.mean()] * 2, color=INK, lw=1.3, zorder=4)
        k = neg(v)
        ax.text(xpos, 1.02, f"{k}/{len(v)} < 0\nmean {v.mean():+.3f}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5.8, color=INK if k in (0, len(v)) else MUTED)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks([0, 1], ["Ridge", "kNN"]); ax.set_xlim(-0.6, 1.6)
    pad = 0.04
    ax.set_ylim(min(allv + [0]) - pad, max(allv + [0]) + pad)
    ax.set_ylabel(f"LDS - Raw+hist ({unit})")
    ax.text(-0.22, 0.5, "<- LDS lower", transform=ax.transAxes, rotation=90,
            va="center", ha="center", fontsize=5.5, color=INK2)
    _note(fig, 0.02, 0.045, f"N = {nA} animals; DM, Isomap, GPFA-off on d <= 20 grid. The simulated ground-truth vs sorted "
          "comparison has no real-data counterpart (single spike source).", fs=5.2)
    save(fig, out, "Fig4_mechanism")


# ================================================================= FIG 5
def fig5(D, out, **kw):
    meta = _meta(D, kw)
    seeds = _seeds(D)
    nA = len(seeds)
    norm = _is_norm(D)
    Xr, Xk = _wide(D, "ridge"), _wide(D, "knn")
    fig = plt.figure(figsize=(W_FULL, 76 * MM))

    # a -- cost of causality
    ax = fig.add_axes([0.07, 0.20, 0.27, 0.64])
    head_fig(fig, 0.0, 0.92, "a", "Cost of causality: LDS / GPFA-c vs offline GPFA")
    Xkb = _knn_best_wide(D, value="norm" if norm else "cm")
    layouts = {"Ridge": ("gpfa", "gpfa_causal", "lds"), "kNN": ("gpfa", "lds")}
    xoff = {"Ridge": 0.0, "kNN": 3.6}
    xt_pos, xt_lab = [], []
    allv = []
    knn_lds_pos = None
    for dec, X in (("Ridge", Xr), ("kNN", Xk)):
        seq = [r for r in layouts[dec] if X[r].notna().any()]
        pos = {r: xoff[dec] + 0.9 * i for i, r in enumerate(seq)}
        for s in X.index:
            pts = [(pos[r], X.loc[s, r]) for r in seq if np.isfinite(X.loc[s, r])]
            if len(pts) > 1:
                ax.plot(*zip(*pts), color="#d3d1cb", lw=0.6, zorder=1)
        for r in seq:
            v = X[r].dropna().values
            allv.extend(v.tolist())
            ax.scatter(np.full(len(v), pos[r]), v, s=9, marker=MRK[r],
                       facecolor=_mfc(r) if r != "gpfa_causal" else COL[r],
                       edgecolor=EDGE[r], lw=0.55, zorder=3)
            ax.plot([pos[r] - 0.25, pos[r] + 0.25], [v.mean()] * 2, color=EDGE[r], lw=1.5, zorder=4)
            xt_pos.append(pos[r]); xt_lab.append(SHORT[r])
        if dec == "kNN" and "lds" in pos:
            knn_lds_pos = pos["lds"]
            # open diamonds: LDS kNN at kNN-best d (descriptive, chosen on test)
            if "lds" in Xkb.columns and Xkb["lds"].notna().any():
                vkb = Xkb["lds"].dropna().values
                allv.extend(vkb.tolist())
                ax.scatter(np.full(len(vkb), pos["lds"]), vkb, s=12, marker="D",
                           facecolor="white", edgecolor=EDGE["lds"], lw=0.7, zorder=5)
                ax.plot([pos["lds"] - 0.3, pos["lds"] + 0.3], [vkb.mean()] * 2,
                        color=EDGE["lds"], lw=1.0, ls=(0, (2, 1.5)), zorder=5)
        if "lds" in seq and "gpfa" in seq:
            dv = (X["lds"] - X["gpfa"]).dropna().values
            tag = "Ridge" if dec == "Ridge" else "kNN @ Ridge-sel d"
            ax.text(np.mean([pos[r] for r in seq]), 1.0,
                    f"LDS - GPFA-off ({tag})\n{dv.mean():+.3f}  ({int((dv > 0).sum())}/{len(dv)} higher)",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=5.0, color=INK2, linespacing=1.15)
        ax.text(np.mean(list(pos.values())), 0.015, dec, transform=ax.get_xaxis_transform(), ha="center",
                fontsize=6, color=INK2, fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    ax.set_xticks(xt_pos, xt_lab, rotation=35, ha="right", rotation_mode="anchor", fontsize=6)
    ax.set_xlim(-0.6, 5.5)
    ax.set_ylabel(_ylab(D))
    if norm:
        ax.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    top = max(allv + [1.0 if norm else 0]) * 1.12
    ax.set_ylim(max(min(allv) - 0.1, 0), top)
    # Notes inside axes (legend + corner text) — never past the figure edge
    leg_h = []
    if knn_lds_pos is not None:
        leg_h.append(Line2D([], [], marker="D", ls="", mfc="white", mec=EDGE["lds"],
                            ms=4.0, label="LDS kNN-best d (descriptive)"))
    dv_r = (Xr["lds"] - Xr["gpfa"]).dropna().values
    if len(dv_r):
        leg_h.append(Line2D([], [], color="none", label=(
            f"Clean: Ridge LDS - GPFA-off = {dv_r.mean():+.3f} "
            f"({int((dv_r > 0).sum())}/{len(dv_r)}); kNN gap mixes d"
        )))
    if not Xk["gpfa_causal"].notna().any():
        leg_h.append(Line2D([], [], color="none", label="GPFA-c: Ridge only (no kNN run)"))
    if leg_h:
        ax.legend(handles=leg_h, loc="upper left", fontsize=4.6, frameon=True,
                  fancybox=False, framealpha=0.92, edgecolor="#e8e6e1",
                  handletextpad=0.3, borderaxespad=0.25, borderpad=0.3,
                  labelspacing=0.25)

    # b -- selected d (session level)
    ax = fig.add_axes([0.46, 0.20, 0.24, 0.64])
    head_fig(fig, 0.395, 0.92, "b", "Selected latent d")
    ES = D.get("ES")
    src = ES if ES is not None and "d" in ES.columns else D["E"]
    ds = [2, 3, 5, 10, 20, 40, 80]
    reps = [r for r in ("pca", "dm", "lds", "isomap", "gpfa", "gpfa_causal") if src[src.rep == r].d.notna().any()]
    for j, rep in enumerate(reps):
        d = src[src.rep == rep].d.dropna().values
        for i, dv in enumerate(ds):
            nn = int((np.isclose(d, dv)).sum())
            if nn:
                ax.scatter(j, i, s=5 + 7 * nn, facecolor=_mfc(rep) if rep != "gpfa_causal" else COL[rep],
                           edgecolor=EDGE[rep], lw=0.55, zorder=3)
                ax.text(j, i - 0.42, str(nn), ha="center", va="top", fontsize=5.2, color=INK2)
    ax.axhspan(4.5, 6.6, color="#f3f2ef", lw=0, zorder=0)
    ax.text(len(reps) - 0.5, 6.78, "extended grid (PCA, LDS only)", ha="right", fontsize=5.3, color=INK2)
    ax.set_yticks(range(len(ds)), [str(v) for v in ds]); ax.set_ylim(-0.8, 7.2)
    _xt(ax, reps)
    ax.set_xlim(-0.5, len(reps) - 0.5)
    unit_lab = "session" if ES is not None else "animal"
    ax.set_ylabel(f"d (count of {unit_lab}s)")

    # c -- spread vs test coverage (drop if uninformative)
    ax = fig.add_axes([0.80, 0.22, 0.17, 0.62])
    head_fig(fig, 0.725, 0.92, "c", "Spread vs test coverage")
    cov = pd.Series({s: _coverage(D, s, meta) for s in seeds})
    ok = cov.notna()
    informative = ok.sum() >= 3 and float(cov[ok].max() - cov[ok].min()) >= 0.05
    if informative:
        xx = np.array([cov[ok].min() - 0.02, cov[ok].max() + 0.02])
        ys_all = []
        for rep in ("raw", "pca", "lds"):
            y = Xr[rep].reindex(cov.index)[ok]
            ys_all.extend(y.dropna().values.tolist())
            ax.scatter(cov[ok], y, s=10, marker=MRK[rep], facecolor=COL[rep], edgecolor=EDGE[rep], lw=0.55, zorder=3)
            b1, b0 = np.polyfit(cov[ok].values, y.values, 1)
            ax.plot(xx, b0 + b1 * xx, color=EDGE[rep], lw=0.7)
            r = np.corrcoef(cov[ok].values, y.values)[0, 1]
            ax.text(0.03, {"raw": 0.22, "pca": 0.14, "lds": 0.06}[rep], f"{SHORT[rep]} r = {r:.3f}",
                    transform=ax.transAxes, fontsize=5.3, color=EDGE[rep], va="top")
        ylo, yhi = min(ys_all), max(ys_all)
        yspan = yhi - ylo + 1e-9
        pad_y = 0.04 * yspan
        # Extra bottom room for staggered animal letters
        ax.set_ylim(ylo - pad_y - 0.14 * yspan, yhi + pad_y)
        ax.set_xlim(xx[0], xx[1])
        # animal letters (A..): stagger in y when coverage values cluster
        y0 = ax.get_ylim()[0]
        y1 = ax.get_ylim()[1]
        base = y0 + 0.02 * (y1 - y0)
        step = 0.045 * (y1 - y0)
        xspan = float(xx[1] - xx[0]) + 1e-9
        # sort by coverage; bump row when within 8% of x-span of previous
        ordered = sorted(cov[ok].index, key=lambda s: float(cov[s]))
        rows = {}
        prev_x, row = None, 0
        for s in ordered:
            x = float(cov[s])
            if prev_x is not None and abs(x - prev_x) < 0.08 * xspan:
                row = (row + 1) % 3
            else:
                row = 0
            rows[s] = row
            prev_x = x
        for s in ordered:
            letter = chr(ord("A") + int(s))
            ax.text(cov[s], base + rows[s] * step, letter, ha="center", va="bottom",
                    fontsize=5.5, color=INK2, fontweight="bold", zorder=5)
        ax.set_xlabel("Test coverage\n(fraction of room bins)")
        ax.set_ylabel(f"Ridge, {'normalized error' if norm else 'cm'}")
    else:
        _empty_panel(ax, "panel dropped: test-room coverage\nvaries too little across animals\nto be informative")
    save(fig, out, "Fig5_deployability")


# ================================================================= FIG 6
def _sgn(v):
    return f"{v:+.3f}"


def fig6(D, out, **kw):
    meta = _meta(D, kw)
    nA = len(_seeds(D))
    Xr, Xk = _wide(D, "ridge"), _wide(D, "knn")
    cs = cm_summary(D)
    norm = _is_norm(D)
    fig = plt.figure(figsize=(W_FULL, 96 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 180); ax.set_ylim(0, 96); ax.axis("off")

    def ms(X, r):
        v = X[r].dropna()
        return f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}" if len(v) > 1 else "n/a"

    cw, chh = 46, 33

    def cell(x, y, key, name, base, ema, vs_raw_ema=True):
        ax.add_patch(FancyBboxPatch((x, y), cw, chh, boxstyle="round,pad=0,rounding_size=2", fc=TINT[key], ec=EDGE[key], lw=1.0))
        ax.text(x + cw / 2, y + chh - 5, name, ha="center", fontsize=8.5, fontweight="bold")
        for i, (lab, X, r, col) in enumerate((("Ridge", Xr, base, INK), ("+ EMA", Xr, ema, INK), ("kNN", Xk, base, INK2))):
            ax.text(x + 3, y + chh - 12 - 5.5 * i, lab, fontsize=6.5, color=col)
            ax.text(x + 17, y + chh - 12 - 5.5 * i, ms(X, r), fontsize=6.5, color=col)
        if vs_raw_ema:
            v = _contrast(D, ema, "raw_smooth", "ridge").values
            ax.text(x + 3, y + 2.6, f"{SHORT[ema]} - Raw+EMA {_sgn(v.mean())}  ({neg(v)}/{len(v)} lower)", fontsize=5.5, color=INK2)

    X0, Y0 = 14, 18
    cell(X0, Y0 + 38, "pca", "PCA", "pca", "pca_smooth")
    cell(X0 + 54, Y0 + 38, "lds", "LDS", "lds", "lds_smooth")
    cell(X0, Y0, "dm", "Diffusion map (d <= 20)", "dm", "dm_smooth", vs_raw_ema=False)
    ax.add_patch(FancyBboxPatch((X0 + 54, Y0), cw, chh, boxstyle="round,pad=0,rounding_size=2", fc="white", ec=MUTED, lw=0.9, ls=(0, (3, 2))))
    ax.text(X0 + 54 + cw / 2, Y0 + 23, "Nonlinear dynamic", ha="center", fontsize=8.5, fontweight="bold", color=INK2)
    ax.text(X0 + 54 + cw / 2, Y0 + 12, "not tested", ha="center", fontsize=7, color=INK2, fontstyle="italic")
    ax.text(X0 + 20, 92.5, "STATIC", ha="center", fontsize=7, color=INK2, fontweight="bold")
    ax.text(X0 + 54 + 23, 92.5, "DYNAMIC", ha="center", fontsize=7, color=INK2, fontweight="bold")
    ax.text(X0 - 5, Y0 + 54, "LINEAR", rotation=90, va="center", ha="center", fontsize=7, color=INK2, fontweight="bold")
    ax.text(X0 - 5, Y0 + 16, "NONLINEAR", rotation=90, va="center", ha="center", fontsize=7, color=INK2, fontweight="bold")
    dv = _contrast(D, "dm_smooth", "pca_smooth", "ridge").values
    ax.annotate("", (X0 + 20, Y0 + chh + 0.5), (X0 + 20, Y0 + chh + 4.2),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=MUTED, mutation_scale=8))
    ax.text(X0 + 23, Y0 + chh + 2.4, f"DM+EMA - PCA+EMA {_sgn(dv.mean())}, {neg(dv)}/{len(dv)} lower",
            fontsize=5.8, va="center", color=INK2)
    # controls strip
    ax.text(X0, 11.0, "Controls (Ridge)", fontsize=6.3, fontweight="bold", color=INK2)
    ax.text(X0, 6.8, f"Raw {ms(Xr, 'raw')}    Raw+hist {ms(Xr, 'raw_lag')}    Raw+EMA {ms(Xr, 'raw_smooth')}    "
            f"GPFA-c {ms(Xr, 'gpfa_causal')}    chance 1.00", fontsize=6, color=INK2)

    # right-hand story, adapted from the planned contrasts (numbers computed here)
    c1 = _contrast(D, "raw_smooth", "raw", "ridge").values
    c2 = _contrast(D, "pca_smooth", "raw_smooth", "ridge").values
    c3 = _contrast(D, "dm_smooth", "pca_smooth", "ridge").values
    c4 = _contrast(D, "lds_smooth", "raw_smooth", "ridge").values
    c5 = _contrast(D, "gpfa_causal", "raw_smooth", "ridge").values
    c6 = _contrast(D, "lds", "raw_smooth", "ridge").values
    n1 = len(c1)
    heads = [
        ("1  Temporal integration helps." if c1.mean() < 0 else "1  Temporal integration does not help.",
         f"Raw+EMA - Raw {_sgn(c1.mean())} ({neg(c1)}/{n1} lower)."),
        ("2  Static compression does not beat the full population." if c2.mean() > 0 else
         "2  Static compression beats the full population.",
         f"PCA+EMA - Raw+EMA {_sgn(c2.mean())} ({neg(c2)}/{len(c2)} lower)."),
        ("3  No static nonlinear advantage (d <= 20)." if c3.mean() >= 0 else "3  Nonlinear static advantage (d <= 20).",
         f"DM+EMA - PCA+EMA {_sgn(c3.mean())} ({neg(c3)}/{len(c3)} lower)."),
        ("4  LDS+EMA " + ("ties" if _verdict(c4) == "tie" else ("beats" if c4.mean() < 0 else "loses to")) +
         " Raw+EMA; causal GPFA " + ("loses." if c5.mean() > 0.03 else ("ties." if abs(c5.mean()) <= 0.03 else "wins.")),
         f"LDS+EMA - Raw+EMA {_sgn(c4.mean())} ({neg(c4)}/{len(c4)} lower); GPFA-c - Raw+EMA {_sgn(c5.mean())} "
         f"({neg(c5)}/{len(c5)} lower); LDS (no EMA) - Raw+EMA {_sgn(c6.mean())} ({neg(c6)}/{len(c6)} lower)."),
    ]
    import textwrap
    tx, y = 117, 91
    ax.text(tx, y, "What does decoding need?", fontsize=9, fontweight="bold", va="top"); y -= 8
    for hd, body in heads:
        for i, line in enumerate(textwrap.wrap(hd, 44)):
            ax.text(tx, y, line, fontsize=6.8, fontweight="bold", va="top"); y -= 3.5
        for line in textwrap.wrap(body, 52):
            ax.text(tx + 2.5, y, line, fontsize=5.9, va="top", color=INK2); y -= 3.0
        y -= 2.2
    cmtxt = (f"Mean cm (mean over animals): chance {cs['chance_cm_mean']:.1f}; Raw {cs['raw']['ridge_cm_mean']:.1f}; "
             f"Raw+EMA {cs['raw_smooth']['ridge_cm_mean']:.1f}; PCA+EMA {cs['pca_smooth']['ridge_cm_mean']:.1f}; "
             f"DM+EMA {cs['dm_smooth']['ridge_cm_mean']:.1f}; LDS+EMA {cs['lds_smooth']['ridge_cm_mean']:.1f}; "
             f"GPFA-c {cs['gpfa_causal']['ridge_cm_mean']:.1f}.")
    ax.text(tx, 21, "\n".join(textwrap.wrap(cmtxt, 62)), fontsize=5.3, color=INK2, va="top", linespacing=1.3)
    ax.text(X0 - 8, 1.8, f"Normalized error = error / chance (chance = 1.0); mean +/- SD over {nA} animals (each the mean of 1-2 "
            "sessions); k/N = animals with the stated sign; no p-values. Room A.\n"
            "EMA = causal moving average on Ridge predictions (Ridge only). PCA, LDS: d up to 80; DM, GPFA-c: d <= 20.",
            fontsize=5.0, color=MUTED, va="center", linespacing=1.3)
    save(fig, out, "Fig6_answer")


# ================================================================= FIG 7
def _example_seed(D, meta):
    ES = D.get("ES")
    ex = meta.get("example_session_raw") or meta.get("example_session")
    if ES is not None and ex is not None and "session" in ES.columns:
        v = ES[ES.session == ex].seed.values
        if len(v):
            return int(v[0]), ex
    W = D.get("predictions_window")
    if W is not None and len(W):
        s = int(W.seed.iloc[0])
        return s, (W.session.iloc[0] if "session" in W.columns else ex)
    return int(_seeds(D)[0]), ex


def _pw(W, method, dec, session=None, seed=None):
    m = (W.method == method) & (W.decoder == dec)
    if session is not None and "session" in W.columns:
        m &= (W.session == session)
    elif seed is not None:
        m &= (W.seed == seed)
    return W[m].sort_values("t_s")


def _mm_axes(fig, H, x, ytop, w, h):
    """Axes from top-left in mm (y measured down from the top of the figure)."""
    return fig.add_axes([x, 1 - (ytop + h) / H, w, h / H])


def _wrap(text, n=150):
    import textwrap
    return "\n".join(textwrap.wrap(text, n))


def fig7(D, out, **kw):
    W = D.get("predictions_window")
    if W is None or not len(W):
        print("Fig7 skipped (no predictions_window)")
        return
    meta = _meta(D, kw)
    seed, ses = _example_seed(D, meta)
    floor = _floor_for(D, ses, seed)
    # Same three methods in a/b/c/d: best causal by Ridge; kNN row uses unsmoothed aliases.
    methods_ridge = _best_causal(D, "ridge", 3, set(W[W.decoder == "ridge"].method))
    if not methods_ridge:
        print("Fig7 skipped (no causal methods)")
        return
    methods_knn = [m.replace("_smooth", "") for m in methods_ridge]
    methods = methods_ridge
    poly = _poly(D, seed, session=ses)
    L = _lims(D, [seed], extra=W[["x_true", "y_true"]].values.ravel(), sessions=[ses] if ses else None)
    msk = (W.session == ses) if "session" in W.columns else (W.seed == seed)
    t0 = float(W[msk].t_s.min())
    wmap = 0.27 * 180.0
    nd = 2
    H = 10 + 56 * nd + 58 + 46 + 12
    fig = plt.figure(figsize=(W_FULL, H * MM))
    hf = lambda y_mm: 1 - (y_mm - 1.5) / H

    head_fig(fig, 0.0, hf(5), "a", "Arena view (first 60 s of test)")
    for ri, (dec, mlist) in enumerate((("ridge", methods_ridge), ("knn", methods_knn))):
        for ci, m in enumerate(mlist):
            ax = _mm_axes(fig, H, 0.085 + ci * 0.30, 10 + ri * 56, 0.27, wmap)
            sub = _pw(W, m, dec, ses, seed)
            if not len(sub):
                _empty_panel(ax, f"{SHORT.get(m, m)}: no {dec} predictions")
                continue
            _draw_poly(ax, poly)
            ax.plot(sub.x_true, sub.y_true, color=INK, lw=0.8, zorder=2)
            ax.scatter(sub.x_pred, sub.y_pred, s=1.5, c=COL.get(m, MUTED), alpha=0.4,
                       linewidths=0, zorder=3, rasterized=True)
            ax.plot(sub.x_true.iloc[0], sub.y_true.iloc[0], "o", mfc="none", mec=INK, ms=4, mew=0.7, zorder=4)
            med = float(sub.err_cm.median())
            ax.text(0.04, 0.96, f"{SHORT.get(m, m)}  {med / floor:.3f}  ({med:.1f} cm)",
                    transform=ax.transAxes, fontsize=5.6, va="top", color=EDGE.get(m, INK),
                    fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
            ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); _style_box(ax)
            ax.set_xticks([-round(L, -1), 0, round(L, -1)]); ax.set_yticks(ax.get_xticks())
            ax.tick_params(labelsize=5.5)
            if ri == nd - 1:
                ax.set_xlabel("x (cm)")
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(f"{'Ridge' if dec == 'ridge' else 'kNN'}\ny (cm)")
            else:
                ax.set_yticklabels([])

    yb = 10 + 56 * nd + 4
    head_fig(fig, 0.0, hf(yb), "b", "Coordinates over the same window (Ridge)")
    for yi, coord in enumerate(("x", "y")):
        ax = _mm_axes(fig, H, 0.08, yb + 5 + yi * 24, 0.60, 17)
        true = _pw(W, methods[0], "ridge", ses, seed)
        ax.plot(true.t_s - t0, true[f"{coord}_true"], color=INK, lw=1.0, zorder=2)
        for m in methods:
            sub = _pw(W, m, "ridge", ses, seed)
            ax.plot(sub.t_s - t0, sub[f"{coord}_pred"], color=EDGE[m], lw=0.8, zorder=3)
        ax.set_xlim(0, 60); ax.set_ylim(-L, L); ax.set_ylabel(f"{coord} (cm)")
        ax.tick_params(labelsize=5.5)
        if yi:
            ax.set_xlabel("Time in window (s)")
        else:
            ax.set_xticklabels([])
    for i, (lab, col) in enumerate([("true", INK)] + [(SHORT[m], EDGE[m]) for m in methods]):
        fig.text(0.72, hf(yb + 7 + i * 3.6), lab, fontsize=5.8, color=col, va="top", fontweight="bold")

    yc = yb + 58
    head_fig(fig, 0.0, hf(yc), "c", "Normalized error (2 s rolling median, Ridge)")
    ax = _mm_axes(fig, H, 0.08, yc + 5, 0.50, 26)
    for m in methods:
        sub = _pw(W, m, "ridge", ses, seed)
        dt = float(np.median(np.diff(sub.t_s))) if len(sub) > 1 else 0.05
        roll = sub.err_cm.rolling(max(int(round(2.0 / dt)), 1), center=True, min_periods=1).median() / floor
        ax.plot(sub.t_s - t0, roll, color=EDGE[m], lw=0.9, label=SHORT[m])
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.5, 1.03, "chance", fontsize=5.5, color=INK2, va="bottom")
    ax.set_xlim(0, 60); ax.set_xlabel("Time in window (s)"); ax.set_ylabel("Error / chance")
    ax.legend(loc="upper right", fontsize=5.4, ncol=3, frameon=True, fancybox=False,
              edgecolor="none", framealpha=0.9)

    head_fig(fig, 0.64, hf(yc), "d", "Whole test block, all sessions (Ridge)")
    ax = _mm_axes(fig, H, 0.68, yc + 5, 0.30, 26)
    ES = D.get("ES")
    for j, m in enumerate(methods):
        if ES is None:
            break
        for _, r in ES[ES.rep == m].iterrows():
            is_ex = r.get("session") == ses
            jit = 0.0 if is_ex else np.random.RandomState(int(r.get("session_index", 0))).uniform(-0.18, 0.18)
            ax.scatter(j + jit, r.ridge_norm, s=20 if is_ex else 9,
                       facecolor=COL[m] if is_ex else "white",
                       edgecolor=EDGE[m], lw=0.7, zorder=4 if is_ex else 3)
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(len(methods)), [SHORT[m] for m in methods], fontsize=5.8)
    ax.set_ylabel("Error / chance"); ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.text(0.5, 1.07, "filled = example session", transform=ax.transAxes, fontsize=5.2,
            color=INK2, ha="center")
    rule = meta.get("example_session_rule", "fixed rule in meta")
    _note(fig, 0.0, 1 - (H - 9) / H, _wrap(
        f"Example session: {rule} Same three methods in a-d: best causal by median Ridge "
        f"normalized error ({', '.join(SHORT[m] for m in methods)}); kNN row uses unsmoothed "
        f"aliases (EMA is Ridge-only). GPFA-off excluded. Panel a: window median / chance (cm).",
        165), fs=5.0)
    save(fig, out, "Fig7_trajectories")


# ================================================================= FIG S1
def _ds_norm(D):
    """Session-level d-sweep with ridge/knn normalized by the session floor."""
    DS = D["d_sweep"].copy()
    if "grid" in DS.columns:
        DS = DS[DS.grid == "final"]
    FS = D.get("FS")
    if "knn_norm" not in DS.columns:
        if FS is not None and "session" in DS.columns:
            fl = FS.set_index("session").floor_median
            DS["knn_norm"] = DS.knn_median / DS.session.map(fl)
            if "ridge_norm" not in DS.columns or DS.ridge_norm.isna().all():
                DS["ridge_norm"] = DS.ridge_median / DS.session.map(fl)
        else:
            DS["knn_norm"] = DS.knn_median
    return DS


def figS1(D, out, **kw):
    if "d_sweep" not in D:
        print("FigS1 skipped (no d_sweep)")
        return
    DS = _ds_norm(D)
    norm = "ridge_norm" in DS and DS.ridge_norm.notna().any()
    methods = [m for m in ("pca", "dm", "isomap", "lds", "gpfa") if m in set(DS.method)]
    dims = sorted(DS.d.unique())
    fig = plt.figure(figsize=(W_FULL, 80 * MM))
    head_fig(fig, 0.0, 0.955, "", "Latent dimensionality: extended grid (final), per-fold refit")
    for ci, (dec, col) in enumerate((("Ridge", "ridge_norm"), ("kNN", "knn_norm"))):
        ax = fig.add_axes([0.07 + ci * 0.31, 0.25, 0.25, 0.57])
        ax.axvspan(40, 80, color="#f3f2ef", lw=0, zorder=0)
        for m in methods:
            sub = DS[DS.method == m]
            g = sub.groupby(["seed", "d"])[col].mean().unstack("seed")   # d x animal
            if g.empty:
                continue
            mu, se = g.mean(axis=1), g.std(axis=1, ddof=1) / np.sqrt(g.shape[1])
            ax.errorbar(mu.index, mu.values, yerr=se.values, color=EDGE[m], ecolor=EDGE[m], elinewidth=0.6, capsize=0, lw=1.0,
                        marker=MRK[m], ms=3.5, mfc=_mfc(m), mec=EDGE[m], mew=0.5, label=SHORT[m])
            sel = sub[sub.selected.astype(bool)]
            ax.scatter(sel.d, sel[col], s=24, facecolor="none", edgecolor=EDGE[m], lw=0.8, zorder=4)
        ax.set_xscale("log"); ax.set_xticks(dims); ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.minorticks_off(); ax.tick_params(axis="x", labelsize=5.8)
        if norm:
            ax.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlabel("Latent d"); ax.set_ylabel(f"{dec} {'normalized error' if norm else 'median (cm)'}\nmean +/- s.e.m. over animals")
        head(ax, "ab"[ci], f"{dec} readout", y=1.05)
        ax.text(57, ax.get_ylim()[1], "40 -> 80", ha="center", va="top", fontsize=5.4, color=INK2)
        if ci == 0:
            hs = [Line2D([], [], marker=MRK[m], color=EDGE[m], lw=1.0, mfc=_mfc(m), mec=EDGE[m], ms=3.5, label=SHORT[m])
                  for m in methods]
            fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.06, 0.035), ncol=len(methods), fontsize=5.8,
                       handletextpad=0.3, columnspacing=1.2)
    # c -- plateau
    ax = fig.add_axes([0.80, 0.25, 0.17, 0.57])
    head(ax, "c", "Plateau check (Ridge)", y=1.05, dx=-0.0)
    thr = 0.01
    rows = []
    for m in ("pca", "lds"):
        s = DS[(DS.method == m) & DS.d.isin([40, 80])].pivot_table(index="session", columns="d", values="ridge_norm", aggfunc="mean") \
            if "session" in DS.columns else DS[(DS.method == m) & DS.d.isin([40, 80])].pivot_table(index="seed", columns="d", values="ridge_norm")
        if {40, 80} <= set(s.columns):
            rows.append((m, float(s[40].median() - s[80].median())))
    for i, (m, v) in enumerate(rows):
        ax.bar(i, v, width=0.55, color=COL[m], edgecolor=EDGE[m], lw=0.7)
        ax.text(i, v + (0.0006 if v >= 0 else -0.0006), f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=5.8)
    ax.axhline(thr, color=FAILC, lw=0.7, ls=(0, (3, 2)))
    ax.text(len(rows) - 0.4, thr, "stop rule 0.01", color=FAILC, fontsize=5.3, ha="right", va="bottom")
    ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xticks(range(len(rows)), [SHORT[m] for m, _ in rows]); ax.set_xlim(-0.6, max(len(rows) - 0.4, 0.6))
    ax.set_ylabel("median error(d=40) - error(d=80)\n(positive = d=80 better)", fontsize=6)
    ymax = max([abs(v) for _, v in rows] + [thr]) * 1.5
    ax.set_ylim(-ymax, ymax)
    _note(fig, 0.0, 0.035, f"Rings = selected d per session. Shaded: extended-grid plateau region 40 -> 80 (PCA, LDS only). "
          f"{GRID20_NOTE}.", fs=5.3)
    save(fig, out, "FigS1_latent_d")


# ================================================================= FIG S2
def figS2(D, out, **kw):
    meta = _meta(D, kw)
    SH = D.get("SH")
    ES = D.get("ES")
    reps = [r for r in REPS_REAL if not r.endswith("_smooth")]
    have_sh = SH is not None and len(SH) > 0
    fig = plt.figure(figsize=(W_FULL, 82 * MM))
    head_fig(fig, 0.0, 0.955, "", "Time-shift null (A13): real data, no ground-truth branch")
    for ci, (dec, lab) in enumerate((("ridge", "Ridge"), ("knn", "kNN"))):
        ax = fig.add_axes([0.07 + ci * 0.49, 0.22, 0.42, 0.62])
        ymax = 4.0
        for j, rep in enumerate(reps):
            if have_sh:
                sub = SH[SH.rep == rep]
                vals = sub[f"{dec}_minus_floor"].values
            elif ES is not None:
                vals = ES[ES.rep == rep][f"a13_{dec}"].values
            else:
                vals = D["E"][D["E"].rep == rep][f"a13_{dec}"].values
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                ax.text(j, 0.6, "n/a", ha="center", fontsize=5.4, color=MUTED, fontstyle="italic")
                continue
            jit = (np.random.RandomState(j).uniform(-0.2, 0.2, len(vals)))
            ax.scatter(j + jit, vals, s=5 if have_sh else 9, marker="o" if dec == "ridge" else "s",
                       facecolor=COL[rep] if dec == "ridge" else "white", edgecolor=EDGE[rep], lw=0.4,
                       alpha=0.75 if have_sh else 1.0, zorder=3)
            if not have_sh:
                ax.plot([j - 0.28, j + 0.28], [np.median(vals)] * 2, color=EDGE[rep], lw=1.3, zorder=4)
            ymax = max(ymax, float(vals.max()))
        ax.axhspan(-8, -2, color="#fbe9e9", lw=0, zorder=0)
        ax.axhline(-2.0, color=FAILC, lw=0.8)
        ax.axhline(0.0, color=MUTED, lw=0.5, ls=(0, (2, 2)))
        _xt(ax, reps)
        ax.set_ylim(-5, ymax * 1.15 + 1); ax.set_xlim(-0.6, len(reps) - 0.4)
        ax.set_ylabel("Shifted - floor (cm)" if ci == 0 else "")
        head(ax, "ab"[ci], f"{lab}", y=1.05)
        ax.text(0.01, 0.03, "fail < -2 cm", transform=ax.transAxes, fontsize=5.3, color=FAILC, va="bottom")
    src = "All shifts, all sessions" if have_sh else "One point per session (median over shifts); bar = median"
    _note(fig, 0.0, 0.065, f"{src}. EMA variants inherit the base-method null (post-processing only); "
          "GPFA-c has no A13 null exported (verified by the leakage test).", fs=5.3)
    save(fig, out, "FigS2_a13_full")


# ================================================================= FIG S3
def figS3(D, out, **kw):
    # Prefer all-session predictions; fall back to example window.
    W = D.get("predictions_all")
    if W is None or not len(W):
        W = D.get("predictions_window")
    if W is None or not len(W):
        print("FigS3 skipped (no predictions)")
        return
    meta = _meta(D, kw)
    by_session = "session" in W.columns
    units = sorted(W.session.unique()) if by_session else sorted(W.seed.unique())
    for dec, tag in (("ridge", "FigS3_trajectories_ridge"), ("knn", "FigS3_trajectories_knn")):
        Wd = W[W.decoder == dec]
        if not len(Wd):
            print(f"{tag} skipped (no {dec} predictions)")
            continue
        methods = _best_causal(D, dec, 3, set(Wd.method.unique()))
        if not methods:
            print(f"{tag} skipped (no scored methods)")
            continue
        n = len(units)
        rowh = 20.0
        fig = plt.figure(figsize=(W_FULL, (16 + rowh * n) * MM))
        H = fig.get_figheight() / MM
        head_fig(fig, 0.0, 1 - 6 / H, "", f"Decoded trajectories, all sessions ({'kNN' if dec == 'knn' else 'Ridge'}; "
                 "first 60 s of test)")
        for ri, u in enumerate(units):
            seed = int(Wd[Wd.session == u].seed.iloc[0]) if by_session and (Wd.session == u).any() else (None if by_session else int(u))
            if seed is None:
                continue
            floor = _floor_for(D, u if by_session else None, seed)
            L = _lims(D, [seed], extra=Wd[["x_true", "y_true"]].values.ravel())
            # animal+session label (e.g. F·S9)
            animal_lab = _code(meta, seed)
            sess_lab = str(u) if by_session else ""
            # prefer S# codes from meta.session_codes values / animal map
            row_lab = f"{animal_lab}\u00b7{sess_lab}" if by_session else animal_lab
            for ci, m in enumerate(methods):
                ax = fig.add_axes([0.06 + ci * 0.32, 1 - (12 + rowh * (ri + 1) - 3) / H, 0.28, (rowh - 4) / H])
                sub = Wd[(Wd.session == u) if by_session else (Wd.seed == u)]
                sub = sub[sub.method == m].sort_values("t_s")
                if len(sub):
                    t0 = float(sub.t_s.min())
                    sub = sub[sub.t_s < t0 + 60.0]
                _draw_poly(ax, _poly(D, seed, session=u if by_session else None), lw=0.5)
                ax.plot(sub.x_true, sub.y_true, color=INK, lw=0.7, zorder=2)
                ax.scatter(sub.x_pred, sub.y_pred, s=1.3, c=COL[m] if m != "raw_smooth" else "#7a7872", alpha=0.4,
                           linewidths=0, zorder=3, rasterized=True)
                if len(sub):
                    med = float(sub.err_cm.median())
                    ax.text(0.04, 0.95, f"{row_lab} {SHORT[m]}  {med / floor:.3f} ({med:.1f} cm)",
                            transform=ax.transAxes, fontsize=5.0, va="top", color=EDGE[m],
                            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
                ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); _style_box(ax, 0.4)
                ax.set_xticks([-round(L, -1), round(L, -1)]); ax.set_yticks(ax.get_xticks()); ax.tick_params(labelsize=4.8)
                if ri == n - 1:
                    ax.set_xlabel("x (cm)")
                if ci == 0:
                    ax.set_ylabel("y (cm)")
        save(fig, out, tag)


# ================================================================= FIG S4
def _center_slopes(D, W, methods, dec="ridge"):
    """OLS slope of predicted vs true distance from the room centre, per method."""
    rows = []
    for m in methods:
        sub = W[(W.method == m) & (W.decoder == dec)]
        if len(sub) < 10:
            continue
        rt = np.hypot(sub.x_true, sub.y_true)
        rp = np.hypot(sub.x_pred, sub.y_pred)
        rows.append((m, float(np.cov(rt, rp)[0, 1] / np.var(rt, ddof=1)), int(sub.seed.iloc[0])))
    return rows


def figS4(D, out, **kw):
    W = D.get("predictions_window")
    CDF, JR = D.get("error_cdf"), D.get("jump_rate")
    if W is None and CDF is None and JR is None:
        print("FigS4 skipped (missing failure-mode tables)")
        return
    meta = _meta(D, kw)
    seed, ses = _example_seed(D, meta)
    floor = _floor_for(D, ses, seed)
    has_w = W is not None and len(W) > 0
    decs = [d for d in ("ridge", "knn") if has_w and (W.decoder == d).any()] or ["ridge"]
    rowh = 46.0
    ya_h = 8 + rowh * len(decs) + 6
    H = ya_h + 112
    fig = plt.figure(figsize=(W_FULL, H * MM))
    hf = lambda y_mm: 1 - (y_mm - 1.5) / H

    # a -- spatial error maps, example session only
    head_fig(fig, 0.0, hf(5), "a", "Spatial error maps (example session only)")
    if has_w:
        methods_r = _best_causal(D, "ridge", 3, set(W.method.unique()))
        # kNN row: unsmoothed aliases only (EMA is Ridge-only)
        methods_k = [m.replace("_smooth", "") for m in methods_r]
        EM = D.get("error_maps")
        use_em = EM is not None and {"x_lo", "y_lo", "bin_cm", "median_err", "method", "decoder"} <= set(EM.columns)
        L = _lims(D, [seed], extra=W[["x_true", "y_true"]].values.ravel())
        bin_cm = float(EM.bin_cm.iloc[0]) if use_em else 10.0
        edges = np.arange(-L, L + bin_cm, bin_cm)
        nb = len(edges) - 1
        cmap = mpl.colors.LinearSegmentedColormap.from_list("err", ["#ffffff", "#3a3936"])
        grids = {}
        for dec, methods in (("ridge", methods_r), ("knn", methods_k)):
            if dec not in decs:
                continue
            for m in methods:
                g = np.full((nb, nb), np.nan)
                if use_em:
                    sub_em = EM[(EM.method == m) & (EM.decoder == dec)]
                    for r in sub_em.itertuples():
                        ix = int((float(r.x_lo) + L) // bin_cm)
                        iy = int((float(r.y_lo) + L) // bin_cm)
                        if 0 <= ix < nb and 0 <= iy < nb and r.n >= 5:
                            g[iy, ix] = float(r.median_err) / floor
                grids[(dec, m)] = g
        allv = np.concatenate([g[np.isfinite(g)] for g in grids.values()]) if grids else np.array([1.0])
        vmax = float(np.nanpercentile(allv, 95)) if len(allv) else 1.0
        im = None
        for ri, (dec, methods) in enumerate((("ridge", methods_r), ("knn", methods_k))):
            if dec not in decs:
                continue
            for ci, m in enumerate(methods):
                ax = _mm_axes(fig, H, 0.06 + ci * 0.22, 12 + ri * rowh, 0.18, 0.18 * 180)
                im = ax.imshow(np.ma.masked_invalid(grids[(dec, m)]), origin="lower", extent=[-L, L, -L, L], cmap=cmap,
                               vmin=0, vmax=max(vmax, 1e-6), interpolation="nearest")
                _draw_poly(ax, _poly(D, seed, session=ses), lw=0.6, color="#c83a39")
                ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{'Ridge' if dec == 'ridge' else 'kNN'} | {SHORT.get(m, m)}", fontsize=6,
                             color=EDGE.get(m, INK), pad=2)
                _style_box(ax, 0.4)
        if im is not None:
            cax = _mm_axes(fig, H, 0.72, 14, 0.012, 40)
            cb = fig.colorbar(im, cax=cax); cb.set_label("Median error / chance", fontsize=6); cb.ax.tick_params(labelsize=5.5)
        _note(fig, 0.06, 1 - (ya_h - 6) / H,
              f"Whole test block ({bin_cm:.0f} cm bins, >= 5 frames/bin; blank = not visited). "
              "kNN row uses unsmoothed methods (EMA is Ridge-only). Red outline = room polygon.", fs=5.1)
    else:
        ax = _mm_axes(fig, H, 0.06, 12, 0.6, 30)
        _empty_panel(ax, "no predictions for the example session")

    # b -- centre-pull
    yb = ya_h + 2
    head_fig(fig, 0.0, hf(yb), "b", "Centre-pull slope (Ridge)")
    ax = _mm_axes(fig, H, 0.08, yb + 6, 0.40, 28)
    CP = D.get("center_pull")
    rows, sub_txt = [], "no data"
    if CP is not None and len(CP) and "kind" in CP.columns:
        sl = CP[(CP.kind == "slope") & (CP.decoder == "ridge")]
        rows = [(m, float(v), int(sd)) for m, sd, v in zip(sl.method, sl.seed, sl.slope)]
        sub_txt = "one point per animal"
    elif has_w:
        rows = _center_slopes(D, W, [m for m in REPS_REAL if m in set(W.method)])
        sub_txt = "example session, first 60 s of test"
    reps_cp = [m for m in REPS_REAL if any(r[0] == m for r in rows)]
    for j, m in enumerate(reps_cp):
        v = np.array([r[1] for r in rows if r[0] == m])
        ax.scatter(np.full(len(v), j), v, s=12, facecolor=COL[m], edgecolor=EDGE[m], lw=0.6, zorder=3)
        ax.plot([j - 0.25, j + 0.25], [v.mean()] * 2, color=EDGE[m], lw=1.2)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(max(len(reps_cp) - 0.5, 0.5), 1.02, "no shrinkage", fontsize=5.5, color=INK2, va="bottom", ha="right")
    if reps_cp:
        _xt(ax, reps_cp, fs=5.6)
    ax.set_ylabel("OLS slope"); ax.set_ylim(0.0, 1.35); ax.set_xlim(-0.6, max(len(reps_cp) - 0.4, 0.6))
    ax.text(0.01, 0.04, sub_txt, transform=ax.transAxes, fontsize=5.0, color=MUTED)

    # c -- CDFs
    head_fig(fig, 0.52, hf(yb), "c", "Error CDF (Ridge, pooled over animals present)")
    ax = _mm_axes(fig, H, 0.58, yb + 6, 0.38, 28)
    if CDF is not None and len(CDF):
        show = [m for m in dict.fromkeys(["raw", "raw_smooth"] + _best_causal(D, "ridge", 3)) if m in set(CDF.method)]
        for m in show:
            sub = CDF[(CDF.method == m) & (CDF.decoder == "ridge")].groupby("q").err_cm.mean() / floor
            ax.plot(sub.values, sub.index.values, color=EDGE[m], lw=0.9, label=SHORT[m])
        ax.axvline(1.0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.set_xlabel("Error / chance"); ax.set_ylabel("CDF"); ax.set_xlim(0, 2.2); ax.set_ylim(0, 1)
        ax.legend(fontsize=5.4, loc="lower right", ncol=1)
    else:
        _empty_panel(ax, "error_cdf not supplied")

    # d -- jump rate
    yd = yb + 50
    head_fig(fig, 0.0, hf(yd), "d", "Jump rate (> 20 cm / 50 ms step, Ridge)")
    ax = _mm_axes(fig, H, 0.08, yd + 6, 0.88, 36)
    if JR is None or JR.empty:
        _empty_panel(ax, "jump-rate table absent")
    else:
        reps_j = [m for m in REPS_REAL if m in set(JR.method)]
        for j, m in enumerate(reps_j):
            v = JR[(JR.method == m) & (JR.decoder == "ridge")].jump_rate.values
            if not len(v):
                continue
            ax.scatter(np.full(len(v), j), v, s=14, facecolor=COL[m], edgecolor=EDGE[m], lw=0.7, zorder=3)
            ax.plot([j - 0.28, j + 0.28], [v.mean(), v.mean()], color=EDGE[m], lw=1.3)
        if "true_jump_rate" in JR.columns:
            tj = float(JR.true_jump_rate.mean())
            ax.axhline(tj, color=MUTED, lw=0.7, ls=(0, (2, 2)))
            ax.text(-0.5, tj + 0.001, "true path", fontsize=5.3, color=INK2, ha="left", va="bottom")
        _xt(ax, reps_j, fs=6)
        ax.set_ylabel("Jump fraction"); ax.set_xlim(-0.6, len(reps_j) - 0.4); ax.set_ylim(-0.005, None)
    save(fig, out, "FigS4_failure_modes")


# ================================================================= FIG S5
def figS5(D, out, **kw):
    lat = D.get("latents")
    pos = None
    if lat:
        pos = lat.get("pos") if "pos" in lat else D.get("latents_pos")
    if not lat or pos is None:
        print("FigS5 skipped (no latents / positions supplied)")
        return
    # Prefer gpfa_causal when present; else offline GPFA (figure_contract has Z_gpfa only).
    methods = []
    for m in ("pca", "dm", "lds"):
        if m in lat:
            methods.append(m)
    if "gpfa_causal" in lat:
        methods.append("gpfa_causal")
    elif "gpfa" in lat:
        methods.append("gpfa")
    if len(methods) < 2:
        print("FigS5 skipped (need at least two latent methods)")
        return
    pos = np.asarray(pos, float)
    n_m = len(methods)
    fig = plt.figure(figsize=(W_FULL, 105 * MM))
    head_fig(fig, 0.0, 0.965, "", "Latent trajectories coloured by position (first two dimensions)")
    cmaps = ["viridis", "magma"]
    for ri, (lab, k) in enumerate((("x (cm)", 0), ("y (cm)", 1))):
        sc = None
        for ci, m in enumerate(methods):
            ax = fig.add_axes([0.05 + ci * (0.88 / n_m), 0.52 - ri * 0.43, 0.88 / n_m - 0.03, 0.36])
            Z = np.asarray(lat[m], float)
            ok = np.isfinite(Z[:, :2]).all(1) & np.isfinite(pos).all(1)
            sc = ax.scatter(Z[ok, 0], Z[ok, 1], c=pos[ok, k], s=0.6, cmap=cmaps[ri],
                            linewidths=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); _style_box(ax, 0.4)
            ax.set_title(SHORT[m], fontsize=7, color=EDGE.get(m, INK), fontweight="bold", pad=3)
            ax.set_xlabel("latent dim 1", fontsize=5.8, labelpad=1)
            if ci == 0:
                ax.set_ylabel("latent dim 2", fontsize=5.8)
        if sc is not None:
            cax = fig.add_axes([0.94, 0.52 - ri * 0.43, 0.012, 0.36])
            cb = fig.colorbar(sc, cax=cax)
            cb.set_label(f"position {lab}", fontsize=6)
            cb.ax.tick_params(labelsize=5)
    gpfa_note = ("GPFA-c" if "gpfa_causal" in methods else "GPFA-off (causal GPFA latents not in figure_contract)")
    _note(fig, 0.05, 0.04,
          f"Example session (fixed rule). Eval samples; colour = room-local position. "
          f"Columns: {', '.join(SHORT[m] for m in methods)} ({gpfa_note}).",
          fs=5.3)
    save(fig, out, "FigS5_phase8")


ALL_FIGS = (fig1, fig2, fig3, fig4, fig5, fig6, fig7, figS1, figS2, figS3, figS4, figS5)


def build_all(D, out, **kw):
    os.makedirs(out, exist_ok=True)
    for f in ALL_FIGS:
        f(D, out, **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="tidy data dir (data_*.csv, data_meta.json)")
    ap.add_argument("--out", default="figures_real")
    a = ap.parse_args()
    D = load_real(a.data)
    build_all(D, a.out)
    print("wrote", sorted(x for x in os.listdir(a.out) if not x.startswith("prev")))
