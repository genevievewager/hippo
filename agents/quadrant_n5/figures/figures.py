"""
Quadrant N=5 — publication figure set (Fig 1–6).

Source data: data_*.csv, built by extract_json.py from the quadrant_n5 result
JSONs (config 4da17a5d…, seeds_0_4_code 775fa1c3, report_code_sha eb442e9f).
Writes vector PDF + 600-dpi PNG per figure.

    python extract_json.py outputs/quadrant_n5 <data_dir>
    python figures.py --data <data_dir> --out <fig_dir>
"""
import argparse, os, textwrap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

MM = 1 / 25.4
W_FULL = 180 * MM
_HERE = os.path.dirname(os.path.abspath(__file__))
_FONT_DIR = os.path.join(_HERE, "fonts")


def _register_publication_font():
    """Prefer Liberation Sans / Arial; fail if only DejaVu would be used."""
    candidates = [
        (os.path.join(_FONT_DIR, "LiberationSans-Regular.ttf"),
         os.path.join(_FONT_DIR, "LiberationSans-Bold.ttf")),
        ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]
    for reg, bold in candidates:
        if os.path.isfile(reg) and os.path.isfile(bold):
            font_manager.fontManager.addfont(reg)
            font_manager.fontManager.addfont(bold)
            prop = font_manager.FontProperties(fname=reg)
            name = prop.get_name()
            if name not in ("Liberation Sans", "Arial"):
                raise RuntimeError(
                    f"Publication font file {reg} resolved to {name!r}; "
                    "need Liberation Sans or Arial."
                )
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = [name, "Liberation Sans", "Arial"]
            resolved = font_manager.findfont(
                font_manager.FontProperties(family=name), fallback_to_default=False,
            )
            resolved_name = font_manager.FontProperties(fname=resolved).get_name()
            if resolved_name not in ("Liberation Sans", "Arial"):
                raise RuntimeError(
                    f"Publication font resolved to {resolved_name!r} ({resolved}); "
                    "need Liberation Sans or Arial. Vendor TTFs under "
                    f"{_FONT_DIR} or install liberation-fonts."
                )
            return name
    raise RuntimeError(
        "Liberation Sans / Arial not found. Install liberation-fonts, or vendor "
        f"LiberationSans-{{Regular,Bold}}.ttf (SIL OFL) into {_FONT_DIR}/."
    )


_PUB_FONT = _register_publication_font()
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_PUB_FONT, "Liberation Sans", "Arial"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3a3936", "xtick.color": "#3a3936", "ytick.color": "#3a3936",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 600, "figure.dpi": 150, "lines.linewidth": 1.0, "hatch.linewidth": 0.5,
})
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8f8d87", "#e4e3df"
FAILC = "#c83a39"

REPS = ["raw", "raw_lag", "pca", "dm", "lds", "isomap", "gpfa"]
SHORT = {"raw": "Raw", "raw_lag": "Raw+hist", "pca": "PCA", "dm": "DM",
         "lds": "LDS", "isomap": "Isomap", "gpfa": "GPFA†"}
COL = {"raw": "#52514e", "raw_lag": "#a3a19b", "pca": "#2a78d6", "dm": "#eb6834",
       "lds": "#1baf7a", "isomap": "#c4c2bb", "gpfa": "#c4c2bb"}
EDGE = {"raw": "#3a3936", "raw_lag": "#6e6c66", "pca": "#1f5fae", "dm": "#c24e1e",
        "lds": "#0e7a54", "isomap": "#6e6c66", "gpfa": "#6e6c66"}
MRK = {"raw": "o", "raw_lag": "s", "pca": "o", "dm": "D", "lds": "^", "isomap": "v", "gpfa": "P"}
TINT = {"pca": "#e3eefb", "dm": "#fde6dc", "lds": "#dcf4eb"}
QUAD = ("pca", "dm", "lds")


def head(ax, letter, title, x=0.0, y=1.06, dx=0.0):
    """Panel letter + short bold title on one line, anchored to the axes."""
    ax.text(x + dx, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")
    if title:
        ax.text(x + dx + 0.075 * (1 if dx == 0 else 1), y + 0.004, title, transform=ax.transAxes,
                fontsize=7, fontweight="bold", va="bottom", ha="left")


def head_fig(fig, x, y, letter, title=""):
    fig.text(x, y, letter, fontsize=9, fontweight="bold", va="bottom", ha="left")
    if title:
        fig.text(x + 0.018, y + 0.002, title, fontsize=7, fontweight="bold", va="bottom", ha="left")


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), pad_inches=0.02)
    plt.close(fig)


def neg(v):
    return int((np.asarray(v) < 0).sum())


def xticks_methods(ax, reps=REPS, rot=35):
    ax.set_xticks(range(len(reps)), [SHORT[r] for r in reps], rotation=rot, ha="right" if rot else "center",
                  rotation_mode="anchor")
    for t, r in zip(ax.get_xticklabels(), reps):
        if r in QUAD:
            t.set_color(EDGE[r]); t.set_fontweight("bold")


# ----------------------------------------------------------------- data
def load(here):
    rd = lambda n: pd.read_csv(os.path.join(here, f"data_{n}.csv"))
    D = dict(E=rd("errors"), F=rd("floor"), SH=rd("a13_shifts"), LC=rd("learning_curves"),
             R=rd("replay"), A=pd.read_csv(os.path.join(here, "data_audit.csv"), keep_default_na=False), B=rd("behavior_5hz"),
             P=rd("population"))
    for optional in ("predictions_window", "error_maps", "center_pull", "error_cdf", "d_sweep", "jump_rate"):
        path = os.path.join(here, f"data_{optional}.csv")
        if os.path.isfile(path):
            D[optional] = pd.read_csv(path)
    return D


def wide(E, src, dec):
    return E[E.source == src].pivot(index="seed", columns="rep", values=dec)[REPS]


# ================================================================= FIG 1
def fig1(D, out):
    E, F, P, B = D["E"], D["F"], D["P"], D["B"]
    FW, FH = W_FULL, 118 * MM
    fig = plt.figure(figsize=(FW, FH))

    # a — quadrant
    ax = fig.add_axes([0.0, 0.46, 0.27, 0.52]); ax.set_xlim(0, 48); ax.set_ylim(-8, 52)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0, 52, "a", fontsize=9, fontweight="bold", va="top")
    cw = 19.5
    cells = [((7.5, 25.5), "pca", "PCA"), ((28.0, 25.5), "lds", "LDS"),
             ((7.5, 4.0), "dm", "Diffusion\nmap"), ((28.0, 4.0), None, "")]
    for (x, y), key, name in cells:
        if key:
            ax.add_patch(FancyBboxPatch((x, y), cw, cw, boxstyle="round,pad=0,rounding_size=1.2",
                                        fc=TINT[key], ec=EDGE[key], lw=0.9))
            ax.text(x + cw / 2, y + cw / 2, name, ha="center", va="center", fontsize=7.5,
                    fontweight="bold", linespacing=1.1)
        else:
            ax.add_patch(FancyBboxPatch((x, y), cw, cw, boxstyle="round,pad=0,rounding_size=1.2",
                                        fc="white", ec=MUTED, lw=0.8, ls=(0, (2, 1.5)), hatch="/////"))
            ax.text(x + cw / 2, y + cw / 2, "empty", ha="center", va="center", fontsize=6.5,
                    color=INK2, fontstyle="italic", bbox=dict(fc="white", ec="none", pad=1.5))
    ax.text(17.25, 47.0, "Static", ha="center", fontsize=7, color=INK2)
    ax.text(37.75, 47.0, "Dynamic", ha="center", fontsize=7, color=INK2)
    ax.text(4.5, 35.25, "Linear", ha="center", va="center", rotation=90, fontsize=7, color=INK2)
    ax.text(4.5, 13.75, "Nonlinear", ha="center", va="center", rotation=90, fontsize=7, color=INK2)
    ax.text(7.5, 1.2, "Controls  raw counts · raw + 250 ms history\nBaselines  Isomap · GPFA (offline smoother)",
            fontsize=5.8, color=INK2, va="top", linespacing=1.4)

    # b — pipeline + split timeline
    ax = fig.add_axes([0.29, 0.46, 0.71, 0.52]); ax.set_xlim(0, 128); ax.set_ylim(0, 61); ax.axis("off")
    ax.text(0, 61, "b", fontsize=9, fontweight="bold", va="top")
    boxes = ["Simulated\nhippocampal\nformation\n(lab probe track)",
             "Neuropixels\ndegradation\n+ simulated\nsorting",
             "Causal counts\nW = 250 ms\nstep = 50 ms\nsqrt, z-score",
             "Representation\nE\n(fit on training\nblock only)",
             "Decoder\nRidge (primary)\nkNN\n(sensitivity)"]
    bw, gap, y0, bh = 20.5, 4.0, 27, 22
    xs = [3 + i * (bw + gap) for i in range(5)]
    for i, (t, x) in enumerate(zip(boxes, xs)):
        hl = i == 3
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc="#fff6d6" if hl else "#f3f2ef", ec="#b98200" if hl else "#6e6c66", lw=0.7))
        ax.text(x + bw / 2, y0 + bh / 2, t, ha="center", va="center", fontsize=6, linespacing=1.3)
        x_to = xs[i + 1] - 0.4 if i < 4 else x + bw + 3.4
        ax.annotate("", (x_to, y0 + bh / 2), (x + bw + 0.4, y0 + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2, mutation_scale=6))
    ax.text(xs[4] + bw + 3.6, y0 + bh / 2, "x,y", va="center", fontsize=6)
    ax.annotate("", (xs[2] + bw / 2, y0 + bh + 0.4), (xs[0] + bw / 2, y0 + bh + 0.4),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MUTED, ls=(0, (3, 2)),
                                connectionstyle="arc3,rad=-0.22", mutation_scale=6))
    ax.text(xs[1] + bw / 2, y0 + bh + 6.8, "ground-truth spikes (reference only, non-deployable)",
            ha="center", fontsize=5.8, color=INK2)
    ax.text(xs[0] + bw / 2, y0 - 2.5, "86 units", ha="center", fontsize=5.8, color=INK2)
    ax.text(xs[3] + bw / 2, y0 - 2.5, "latent d = 2, 3, 5, 10, 20", ha="center", fontsize=5.8, color=INK2)
    ax.text(xs[4] + bw / 2, y0 - 2.5, "identical for every E", ha="center", fontsize=5.8, color=INK2)
    ty, th = 9.5, 4.0
    L0, L1 = 3, 125
    tx = lambda s: L0 + (L1 - L0) * s / 600
    for k in range(5):
        a_, b_ = 480 * k / 5, 480 * (k + 1) / 5
        ax.add_patch(Rectangle((tx(a_), ty), tx(b_) - tx(a_) - 0.3, th, fc="#b3b1aa" if k == 2 else "#dcdad4", ec="none"))
    ax.add_patch(Rectangle((tx(486), ty), tx(600) - tx(486), th, fc="#3a3936", ec="none"))
    ax.text(tx(240), ty + th + 1.6, "training block (80%) · 5 contiguous inner-CV folds with purge gaps (one held-out fold shaded)",
            ha="center", fontsize=5.8, color=INK2)
    ax.text(tx(543), ty + th + 1.6, "test (20%)", ha="center", fontsize=5.8, color=INK2)
    for s in (0, 120, 240, 360, 480, 600):
        ax.plot([tx(s)] * 2, [ty - 0.8, ty], color=INK2, lw=0.5)
        ax.text(tx(s), ty - 3.6, f"{s} s", ha="center", fontsize=5.5, color=INK2)
    ax.annotate("1 s purge gap", (tx(483), ty + th / 2), (tx(405), ty - 7.2), fontsize=5.5, color=INK2,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2))
    ax.text(L0, -2.5, "× 5 seeds: trajectory, neural activity, recording noise and sorting errors drawn independently;\n"
            "every method is scored on the same seeds (paired design)", fontsize=5.8, color=INK2)

    # c — population
    ax = fig.add_axes([0.125, 0.075, 0.19, 0.27])
    ct = P[(P.kind == "cell_type") & (P.seed == 0)].set_index("label").n
    same = all((P[(P.kind == "cell_type") & (P.seed == s)].set_index("label").n == ct).all() for s in range(5))
    names = {"MEC_grid": "MEC grid", "MEC_hd": "Head direction", "MEC_speed": "Speed",
             "Sub_bvc": "Subicular BVC", "INT_CA1": "CA1 interneuron"}
    order = ["MEC_grid", "MEC_hd", "MEC_speed", "Sub_bvc", "INT_CA1"]
    yy = np.arange(len(order))[::-1]
    ax.barh(yy, [ct[o] for o in order], height=0.62, color="#6e6c66")
    for y, o in zip(yy, order):
        ax.text(ct[o] + 0.6, y, str(ct[o]), va="center", fontsize=6)
    ax.set_yticks(yy, [names[o] for o in order]); ax.set_xlabel("Units")
    ax.set_xlim(0, 31); ax.tick_params(axis="y", length=0)
    head_fig(fig, 0.0, 0.385, "c", f"Recorded population (n = {int(ct.sum())}{', every seed' if same else ''})")
    ax.text(1.0, 0.02, "no CA1/CA3\nplace cells", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.8, color=INK2, fontstyle="italic")

    # d — trajectories: full session grey, test segment black
    head_fig(fig, 0.395, 0.385, "d", "Test segment (black) covers a different part of the arena in each seed")
    w, gapx, x0 = 0.108, 0.012, 0.405
    for s in range(5):
        ax = fig.add_axes([x0 + s * (w + gapx), 0.10, w, w * FW / FH])
        b = B[B.seed == s]
        tr, te = b[b.time_s < 480], b[b.time_s >= 481]
        ax.plot(tr.x_cm, tr.y_cm, color="#d9d7d1", lw=0.35, zorder=1)
        ax.plot(te.x_cm, te.y_cm, color=INK, lw=0.5, zorder=2)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
        for sp in ("top", "right"): ax.spines[sp].set_visible(True)
        for sp in ax.spines.values(): sp.set_color("#6e6c66"); sp.set_linewidth(0.5)
        ax.set_xticks([0, 100]); ax.set_yticks([0, 100]); ax.tick_params(labelsize=5.5, pad=1.5)
        if s: ax.set_yticklabels([]); ax.set_xticklabels([])
        else: ax.set_ylabel("y (cm)", labelpad=-5)
        if s == 0: ax.set_xlabel("x (cm)", labelpad=-1)
        f = F[(F.seed == s) & (F.source == "sorted")].iloc[0]
        ax.set_title(f"Seed {s} · {int(f.test_bins)} bins", fontsize=6.3, pad=2.5)
    fig.text(x0, 0.005, "Grey: training segment; black: test segment (last 20%). 100 cm arena. "
             "Training covered ≥ 98/100 bins in every seed.", fontsize=5.6, color=INK2)
    save(fig, out, "Fig1_design")


# ================================================================= FIG 2
AUD = [("A1", "Split identity"), ("A2", "Same preprocessing"), ("A3", "Same sample counts"),
       ("A4", "Same window / label time"), ("A5", "No future access"), ("A6", "Selection on train only"),
       ("A7", "GT marked non-deployable"), ("A8", "Save / reload"), ("A9", "Realtime check (empirical)"),
       ("A10", "Seed isolation"), ("A11", "Offline = replay"), ("A12", "UI = frozen config"),
       ("A13", "Time-shift null control"), ("A14", "LDS readout on filtered latents"),
       ("A15", "Predictions + d-sweep")]


def fig2(D, out):
    E, A, R = D["E"], D["A"], D["R"]
    fig = plt.figure(figsize=(W_FULL, 112 * MM))

    # a — audit matrix
    ax = fig.add_axes([0.205, 0.07, 0.14, 0.82])
    st = A.pivot(index="check", columns="seed", values="status")
    n = len(AUD)
    for i, (k, _) in enumerate(AUD):
        for s in range(5):
            v = st.loc[k, s]
            fc = {"PASS": "#eeede9", "FAIL": FAILC, "N/A": "white"}[v]
            ax.add_patch(Rectangle((s, i), 0.9, 0.86, fc=fc, ec="#d0cec8" if v != "FAIL" else FAILC, lw=0.4))
            ax.text(s + 0.45, i + 0.45, {"PASS": "✓", "FAIL": "✗", "N/A": "–"}[v], ha="center", va="center",
                    fontsize=6, color="white" if v == "FAIL" else INK2,
                    fontweight="bold" if v == "FAIL" else None,
                    fontfamily="DejaVu Sans")
    ax.set_xlim(0, 5); ax.set_ylim(n, 0)
    ax.set_yticks(np.arange(n) + 0.45, [f"{k}  {t}" for k, t in AUD], fontsize=5.9)
    ax.set_xticks(np.arange(5) + 0.45, [str(s) for s in range(5)])
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(-0.25, -0.62, "seed", ha="right", fontsize=6.3, color=INK2)
    head_fig(fig, 0.0, 0.935, "a", "Fairness audit, 15 checks × 5 seeds")
    fl = E[~E.a13_ridge_pass.astype(bool) | ~E.a13_knn_pass.astype(bool)]
    ftxt = "; ".join(f"seed {s} {'GT' if src == 'ground_truth' else 'sorted'}: " + ", ".join(SHORT[r] for r in g.rep)
                     for (s, src), g in fl.groupby(["seed", "source"])) or "none"
    ax.text(2.5, n + 0.35, f"✗ A13 fails — {ftxt} (see b)" if len(fl) else "no failures", ha="center", va="top",
            fontsize=5.6, color=FAILC, fontfamily="DejaVu Sans")

    # b — A13 per condition
    ax = fig.add_axes([0.44, 0.60, 0.55, 0.29])
    head_fig(fig, 0.395, 0.935, "b", "Time-shift null control (A13): decoding with shuffled-in-time labels sits at chance")
    conds = [("sorted", "a13_ridge", True, "o"), ("sorted", "a13_knn", True, "s"),
             ("ground_truth", "a13_ridge", False, "o"), ("ground_truth", "a13_knn", False, "s")]
    offs = np.linspace(-0.3, 0.3, 4)
    ax.axhspan(-6, -2, color="#fbe9e9", lw=0, zorder=0)
    ax.axhline(-2, color=FAILC, lw=0.7, ls=(0, (3, 2)))
    ax.axhline(0, color=MUTED, lw=0.5)
    for (src, col, filled, mk), off in zip(conds, offs):
        for j, rep in enumerate(REPS):
            d = E[(E.source == src) & (E.rep == rep)].sort_values("seed")
            x = j + off + np.linspace(-0.035, 0.035, 5)
            ax.scatter(x, d[col], s=8, marker=mk, facecolor=COL[rep] if filled else "white",
                       edgecolor=EDGE[rep], lw=0.55, zorder=3)
    bad = E[(E[["a13_ridge"]].values.ravel() < -2)]
    for r in bad.itertuples():
        j = REPS.index(r.rep)
        ax.scatter([j + offs[2]], [r.a13_ridge], s=40, facecolor="none", edgecolor=FAILC, lw=0.8, zorder=4)
    if len(bad):
        ax.text(REPS.index("isomap") - 0.3, -3.4, "circled: failing cells", fontsize=5.6, color=FAILC, va="center")
    ax.text(-0.55, -2.35, "fail < −2 cm", color=FAILC, fontsize=5.6, va="top")
    xticks_methods(ax, rot=0)
    ax.set_ylabel("Shifted − chance floor\n(cm, median of 20 shifts)")
    ax.set_ylim(-4.2, 14.8); ax.set_xlim(-0.6, len(REPS) - 0.4)
    hs = [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=3.3, label="Sorted · Ridge"),
          Line2D([], [], marker="s", ls="", mfc=INK2, mec=INK2, ms=3.3, label="Sorted · kNN"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=3.3, label="GT · Ridge"),
          Line2D([], [], marker="s", ls="", mfc="white", mec=INK2, ms=3.3, label="GT · kNN")]
    ax.legend(handles=hs, loc="lower right", ncol=4, handletextpad=0.05, columnspacing=0.9,
              bbox_to_anchor=(1.0, 0.98))

    # c — test vs inner-CV
    ax = fig.add_axes([0.44, 0.10, 0.22, 0.33])
    head_fig(fig, 0.395, 0.475, "c", "Held-out error tracks inner-CV error")
    Es = E[E.source == "sorted"]
    ax.plot([0, 50], [0, 50], color=MUTED, lw=0.6, ls=(0, (3, 2)))
    for rep in REPS:
        d = Es[Es.rep == rep]
        ax.scatter(d.inner_cv_ridge, d.ridge, s=9, marker=MRK[rep], facecolor=COL[rep] if rep != "gpfa" else "white",
                   edgecolor=EDGE[rep], lw=0.55, zorder=3)
    ax.set_xlim(8, 48); ax.set_ylim(8, 48); ax.set_aspect("equal")
    ax.set_xlabel("Inner-CV error (cm)"); ax.set_ylabel("Held-out test error (cm)")
    gap = (Es.ridge - Es.inner_cv_ridge)
    ax.text(0.04, 0.96, f"test − CV: {gap.median():+.1f} cm\n(median, 35 cells)", transform=ax.transAxes,
            fontsize=5.8, color=INK2, va="top")

    # d — replay agreement
    ax = fig.add_axes([0.77, 0.10, 0.22, 0.33])
    head_fig(fig, 0.715, 0.475, "d", "Causal replay = offline")
    FLOOR = 1e-16
    for j, rep in enumerate(REPS):
        v = R[R.rep == rep].sort_values("seed").a11_yhat_inf_cm.values
        ax.scatter(np.full(5, j) + np.linspace(-0.16, 0.16, 5), np.where(v <= 0, FLOOR, v), s=8, marker=MRK[rep],
                   facecolor=COL[rep] if rep != "gpfa" else "white", edgecolor=EDGE[rep], lw=0.55, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(3e-17, 800)
    ax.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1, 100], ["exact", "1e−12", "1e−8", "1e−4", "1", "100"])
    ax.minorticks_off()
    xticks_methods(ax)
    ax.set_ylabel("Max |offline − per-step| (cm)")
    ax.text(5.9, 1.2, "smoother\n(future)", fontsize=5.6, color=INK2, ha="right", va="center")
    save(fig, out, "Fig2_validity")


# ================================================================= FIG 3
CONTRASTS = [("dm", "pca", "Nonlinear vs linear (static)", "DM − PCA"),
             ("lds", "pca", "Dynamic vs static (linear)", "LDS − PCA"),
             ("raw_lag", "raw", "History alone", "Raw+hist − Raw"),
             ("lds", "raw_lag", "Dynamics beyond history", "LDS − Raw+hist"),
             ("pca", "raw", "Linear reduction", "PCA − Raw"),
             ("dm", "raw", "Nonlinear reduction", "DM − Raw")]


def slope(ax, X, F, ylab=True, ylim=(0, 58)):
    xs = np.arange(len(REPS))
    fl = F[F.source == "sorted"].floor_median
    ax.axhspan(fl.min(), fl.max(), color="#efeeea", lw=0, zorder=0)
    ax.text(len(REPS) - 0.45, fl.max() + 0.8, "chance (predict training mean)", fontsize=5.6, color=INK2,
            ha="right", va="bottom")
    ax.axvspan(1.5, 4.5, ymax=0.78, color="#f7f6f3", zorder=0, lw=0)
    for s in X.index:
        ax.plot(xs, X.loc[s].values, color="#d3d1cb", lw=0.6, zorder=1)
    for j, rep in enumerate(REPS):
        v = X[rep].values
        ax.scatter(np.full(5, j), v, s=7, facecolor="#9a9893", edgecolor="none", zorder=2)
        ax.plot([j - 0.3, j + 0.3], [v.mean()] * 2, color=EDGE[rep], lw=2.0, solid_capstyle="round", zorder=3)
    xticks_methods(ax)
    ax.set_ylim(*ylim); ax.set_xlim(-0.6, len(REPS) - 0.4)
    if ylab: ax.set_ylabel("Median position error (cm)")


def fig3(D, out):
    E, F = D["E"], D["F"]
    Xr, Xk = wide(E, "sorted", "ridge"), wide(E, "sorted", "knn")
    fig = plt.figure(figsize=(W_FULL, 76 * MM))
    a = fig.add_axes([0.065, 0.2, 0.235, 0.66]); slope(a, Xr, F)
    head_fig(fig, 0.0, 0.915, "a", "Ridge readout (primary)")
    b = fig.add_axes([0.335, 0.2, 0.235, 0.66], sharey=a); slope(b, Xk, F, ylab=False)
    plt.setp(b.get_yticklabels(), visible=False)
    head_fig(fig, 0.315, 0.915, "b", "kNN readout (sensitivity)")
    a.text(3, 0.775 * 58, "quadrant", ha="center", fontsize=5.6, color=INK2)
    b.text(3, 0.775 * 58, "quadrant", ha="center", fontsize=5.6, color=INK2)

    ax = fig.add_axes([0.75, 0.2, 0.2, 0.66])
    head_fig(fig, 0.595, 0.915, "c", "Planned paired contrasts (sorted spikes)")
    n = len(CONTRASTS)
    for i, (x1, x2, lab, sh) in enumerate(CONTRASTS):
        y = n - 1 - i
        for X, dy, filled in ((Xr, 0.17, True), (Xk, -0.17, False)):
            v = (X[x1] - X[x2]).values
            ax.scatter(v, np.full(5, y + dy), s=8, marker=MRK[x1],
                       facecolor=COL[x1] if filled else "white", edgecolor=EDGE[x1], lw=0.55, zorder=3)
            ax.plot([v.mean()] * 2, [y + dy - 0.13, y + dy + 0.13], color=INK, lw=1.2, zorder=4)
            k = neg(v)
            ax.text(1.03, (y + dy + 0.6) / n, f"{k}/5", transform=ax.transAxes, va="center", fontsize=6,
                    color=INK if k in (0, 5) else MUTED, fontweight="bold" if k in (0, 5) else None)
        if i < n - 1: ax.axhline(y - 0.5, color=GRID, lw=0.5)
    ax.set_yticks(np.arange(n)[::-1], [f"{lab}\n{sh}" for _, _, lab, sh in CONTRASTS], fontsize=6)
    for t, (x1, _, _, _) in zip(ax.get_yticklabels(), CONTRASTS): pass
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlim(-32, 8); ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([-30, -20, -10, 0])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Paired difference (cm)\n← first term lower error")
    ax.text(1.03, 1.035, "k/5 < 0", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5, color=INK2)
    hs = [Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=3.3, label="Ridge"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=3.3, label="kNN")]
    ax.legend(handles=hs, loc="lower left", ncol=1, handletextpad=0.1, borderaxespad=0.2)
    save(fig, out, "Fig3_quadrant_answer")


# ================================================================= FIG 4
def fig4(D, out):
    E = D["E"]
    Xr, Xk = wide(E, "sorted", "ridge"), wide(E, "sorted", "knn")
    Gr, Gk = wide(E, "ground_truth", "ridge"), wide(E, "ground_truth", "knn")
    fig = plt.figure(figsize=(W_FULL, 72 * MM))

    # a — ridge vs knn
    ax = fig.add_axes([0.06, 0.2, 0.24, 0.66])
    head_fig(fig, 0.0, 0.915, "a", "Readout: linear vs nonlinear")
    ax.plot([0, 48], [0, 48], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
    for rep in REPS:
        ax.scatter(Xr[rep], Xk[rep], s=10, marker=MRK[rep], facecolor=COL[rep] if rep != "gpfa" else "white",
                   edgecolor=EDGE[rep], lw=0.55, zorder=3)
    ax.set_xlim(0, 48); ax.set_ylim(0, 48); ax.set_aspect("equal")
    ax.set_xlabel("Ridge error, sorted (cm)"); ax.set_ylabel("kNN error, sorted (cm)")
    ax.text(47, 45.5, "y = x", fontsize=5.5, color=INK2, ha="right")
    ax.text(26, 5, "LDS", fontsize=6.3, color=EDGE["lds"], fontweight="bold")
    ax.text(3, 40, "static & raw:\nnear the diagonal", fontsize=5.8, color=INK2, va="top")
    gl, gp = (Xr.lds - Xk.lds).values, (Xr.pca - Xk.pca).values
    ax.text(26, 1.2, f"gain {gl.mean():.1f} cm (PCA {gp.mean():.1f})", fontsize=5.6, color=INK2)

    # b — GT vs sorted (kNN)
    ax = fig.add_axes([0.385, 0.2, 0.33, 0.66])
    head_fig(fig, 0.33, 0.915, "b", "Information survives recording only with dynamics (kNN)")
    for j, rep in enumerate(REPS):
        g, s = Gk[rep].values, Xk[rep].values
        xo = np.linspace(-0.2, 0.2, 5)
        for k in range(5):
            ax.plot([j + xo[k]] * 2, [g[k], s[k]], color="#d6d4ce", lw=0.6, zorder=1)
        ax.scatter(j + xo, g, s=8, marker=MRK[rep], facecolor="white", edgecolor=EDGE[rep], lw=0.55, zorder=3)
        ax.scatter(j + xo, s, s=8, marker=MRK[rep], facecolor=COL[rep], edgecolor=EDGE[rep], lw=0.55, zorder=3)
        pen = (s - g).mean()
        ax.text(j, 46.5, f"+{pen:.0f}", ha="center", fontsize=6.2, color=EDGE[rep] if rep in QUAD else INK,
                fontweight="bold" if rep == "lds" else None)
    xticks_methods(ax)
    ax.set_ylim(0, 49); ax.set_xlim(-0.8, len(REPS) - 0.4)
    ax.set_ylabel("kNN median error (cm)")
    hs = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=3.3, label="ground-truth spikes"),
          Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=3.3, label="sorted spikes")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.2), handletextpad=0.1, ncol=2)

    # c — dynamics beyond history vs noise
    ax = fig.add_axes([0.80, 0.2, 0.18, 0.66])
    head_fig(fig, 0.745, 0.915, "c", "Dynamics model vs noise")
    xpos, ticks, tl = 0, [], []
    for dec, G, X in (("Ridge", Gr, Xr), ("kNN", Gk, Xk)):
        for lab, M, filled in (("GT", G, False), ("sorted", X, True)):
            v = (M.lds - M.raw_lag).values
            ax.scatter(np.full(5, xpos) + np.linspace(-0.12, 0.12, 5), v, s=9, marker="^",
                       facecolor=COL["lds"] if filled else "white", edgecolor=EDGE["lds"], lw=0.55, zorder=3)
            ax.plot([xpos - 0.22, xpos + 0.22], [v.mean()] * 2, color=INK, lw=1.2)
            k = neg(v)
            ax.text(xpos, 2.6, f"{k}/5", ha="center", fontsize=6, color=INK if k in (0, 5) else MUTED)
            ticks.append(xpos); tl.append(lab); xpos += 1
        xpos += 0.5
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(ticks, tl, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(-25, 4.5); ax.set_xlim(-0.6, xpos - 0.9)
    ax.text(0.5, -24.3, "Ridge", ha="center", fontsize=6, color=INK2)
    ax.text(3.0, -24.3, "kNN", ha="center", fontsize=6, color=INK2)
    ax.set_ylabel("LDS − Raw+hist (cm)")
    save(fig, out, "Fig4_mechanism")


# ================================================================= FIG 5
def fig5(D, out):
    E, R, F, LC = D["E"], D["R"], D["F"], D["LC"]
    fig = plt.figure(figsize=(W_FULL, 118 * MM))

    # a — latency
    ax = fig.add_axes([0.075, 0.585, 0.37, 0.34])
    head_fig(fig, 0.0, 0.955, "a", "Realtime cost per 50 ms update (replay, sorted)")
    for j, rep in enumerate(REPS):
        d = R[R.rep == rep].sort_values("seed")
        yy = len(REPS) - 1 - j + np.linspace(-0.22, 0.22, 5)
        for y, p50, p99 in zip(yy, d.p50_ms, d.p99_ms):
            ax.plot([p50, p99], [y, y], color=EDGE[rep], lw=0.7)
            ax.scatter([p50], [y], s=6, marker=MRK[rep], facecolor=COL[rep] if rep != "gpfa" else "white",
                       edgecolor=EDGE[rep], lw=0.5, zorder=3)
            ax.plot([p99], [y], marker="|", color=EDGE[rep], ms=3.3, mew=0.8)
        ob = d.frac_over_budget.values
        if ob.max() > 0:
            ax.text(170, len(REPS) - 1 - j, f"{100*ob.min():.1f}–{100*ob.max():.1f}% of\nsteps over", fontsize=5.4,
                    color=FAILC, va="center", ha="left")
    ax.axvline(50, color=FAILC, lw=0.8)
    ax.text(45, len(REPS) - 0.35, "50 ms budget", color=FAILC, fontsize=5.8, ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_xlim(5e-4, 3000); ax.minorticks_off()
    ax.set_yticks(range(len(REPS)), [SHORT[r] for r in REPS][::-1]); ax.tick_params(axis="y", length=0)
    for t, r in zip(ax.get_yticklabels(), REPS[::-1]):
        if r in QUAD: t.set_color(EDGE[r]); t.set_fontweight("bold")
    ax.set_ylim(-0.6, len(REPS) - 0.1)
    ax.set_xlabel("Transform time (ms; dot = p50, tick = p99, one row per seed)")

    # b — cost of causality
    ax = fig.add_axes([0.54, 0.585, 0.17, 0.34])
    head_fig(fig, 0.475, 0.955, "b", "Cost of causality")
    for dec, xo in (("ridge", 0), ("knn", 1.5)):
        X = wide(E, "sorted", dec)
        for s in range(5):
            ax.plot([xo, xo + 0.8], [X.gpfa[s], X.lds[s]], color="#d3d1cb", lw=0.6)
        ax.scatter(np.full(5, xo), X.gpfa, s=9, marker="P", facecolor="white", edgecolor=EDGE["gpfa"], lw=0.55, zorder=3)
        ax.scatter(np.full(5, xo + 0.8), X.lds, s=9, marker="^", facecolor=COL["lds"], edgecolor=EDGE["lds"], lw=0.55, zorder=3)
        dv = (X.lds - X.gpfa).values
        ax.text(xo + 0.4, 37.5, f"+{dv.mean():.1f} cm\n{int((dv > 0).sum())}/5", ha="center", fontsize=5.8)
    ax.set_xticks([0, 0.8, 1.5, 2.3], ["GPFA†", "LDS", "GPFA†", "LDS"], rotation=35, ha="right", rotation_mode="anchor")
    ax.text(0.4, -7.5, "Ridge", ha="center", fontsize=6, color=INK2)
    ax.text(1.9, -7.5, "kNN", ha="center", fontsize=6, color=INK2)
    ax.set_ylim(0, 43); ax.set_xlim(-0.4, 2.7)
    ax.set_ylabel("Median error, sorted (cm)")

    # c — chosen d
    ax = fig.add_axes([0.80, 0.585, 0.18, 0.34])
    head_fig(fig, 0.745, 0.955, "c", "Selected latent d")
    ds = [2, 3, 5, 10, 20]
    reps = ["pca", "dm", "lds", "isomap", "gpfa"]
    for j, rep in enumerate(reps):
        for src, xo in (("sorted", -0.16), ("ground_truth", 0.16)):
            d = E[(E.source == src) & (E.rep == rep)].d.values
            for i, dv in enumerate(ds):
                nn = int((d == dv).sum())
                if nn:
                    ax.scatter(j + xo, i, s=5 + 7 * nn, facecolor=COL[rep] if src == "sorted" else "white",
                               edgecolor=EDGE[rep], lw=0.55, zorder=3)
                    ax.text(j + xo, i - 0.38, str(nn), ha="center", va="top", fontsize=5.2, color=INK2)
    ax.axhspan(3.55, 4.45, color="#f3f2ef", lw=0, zorder=0)
    ax.text(2, 4.62, "upper edge of sweep", ha="center", fontsize=5.5, color=INK2)
    ax.set_yticks(range(5), [str(v) for v in ds]); ax.set_ylim(-0.7, 5.0)
    xticks_methods(ax, reps)
    ax.set_ylabel("d (count of seeds)")

    # d — learning curves
    ax = fig.add_axes([0.075, 0.10, 0.37, 0.33])
    head_fig(fig, 0.0, 0.475, "d", "More training data helps every method; mean ranking unchanged")
    for rep in REPS:
        d = LC[LC.rep == rep].pivot(index="seed", columns="frac", values="ridge")
        fr = d.columns.values; m = d.mean().values; se = d.std(ddof=1).values / np.sqrt(len(d))
        x = fr * 480
        ax.errorbar(x, m, yerr=se, fmt="none", ecolor=EDGE[rep], elinewidth=0.6, capsize=0)
        ax.plot(x, m, color=EDGE[rep], lw=1.0, marker=MRK[rep], ms=3,
                mfc=COL[rep] if rep != "gpfa" else "white", mec=EDGE[rep], mew=0.5)
        ax.text(x[-1] + 12, m[-1], SHORT[rep], fontsize=5.8, va="center", color=EDGE[rep])
    ax.set_xlim(100, 560); ax.set_xticks([120, 240, 480])
    ax.set_xlabel("Training data (s, most recent part of training block)")
    ax.set_ylabel("Ridge error, sorted (cm)\nmean ± s.e.m., 5 seeds")
    # label spacing: nudge overlapping end labels
    ys = {}
    for t in ax.texts:
        ys[t] = t.get_position()[1]
    order_ = sorted(ys, key=lambda t: ys[t])
    last = -1e9
    for t in order_:
        y = max(ys[t], last + 1.6); t.set_y(y); last = y

    # e — seed difficulty
    ax = fig.add_axes([0.55, 0.10, 0.28, 0.33])
    head_fig(fig, 0.475, 0.475, "e", "Seed-to-seed spread tracks test difficulty")
    X = wide(E, "sorted", "ridge")
    tb = F[F.source == "sorted"].set_index("seed").test_bins
    for rep in ("raw", "pca", "lds"):
        ax.scatter(tb, X[rep], s=10, marker=MRK[rep], facecolor=COL[rep], edgecolor=EDGE[rep], lw=0.55, zorder=3)
        b1, b0 = np.polyfit(tb, X[rep], 1)
        xx = np.array([tb.min() - 2, tb.max() + 2])
        ax.plot(xx, b0 + b1 * xx, color=EDGE[rep], lw=0.7)
        r = np.corrcoef(tb, X[rep])[0, 1]
        ax.text(xx[1] + 1, b0 + b1 * xx[1], f"{SHORT[rep]} r = {r:.2f}", fontsize=5.8, va="center", color=EDGE[rep])
    for s in range(5):
        ax.text(tb[s], 11.3, str(s), ha="center", fontsize=5.5, color=INK2)
    ax.set_xlim(35, 88); ax.set_ylim(9.5, 44)
    ax.set_xlabel("Test-segment extent (bins visited, of 100; numbers = seed)")
    ax.set_ylabel("Ridge error, sorted (cm)")
    save(fig, out, "Fig5_deployability")


# ================================================================= FIG 6
def fig6(D, out):
    E = D["E"]
    Xr, Xk = wide(E, "sorted", "ridge"), wide(E, "sorted", "knn")
    Gk = wide(E, "ground_truth", "knn")
    fig = plt.figure(figsize=(W_FULL, 90 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 180); ax.set_ylim(0, 90); ax.axis("off")

    def cell(x, y, key, name):
        if key:
            ax.add_patch(FancyBboxPatch((x, y), 40, 27, boxstyle="round,pad=0,rounding_size=2",
                                        fc=TINT[key], ec=EDGE[key], lw=1.0))
            ax.text(x + 20, y + 20.5, name, ha="center", fontsize=8.5, fontweight="bold")
            ax.text(x + 20, y + 12.5, f"Ridge  {Xr[key].mean():.1f} ± {Xr[key].std(ddof=1):.1f} cm", ha="center", fontsize=7)
            ax.text(x + 20, y + 6.5, f"kNN  {Xk[key].mean():.1f} ± {Xk[key].std(ddof=1):.1f} cm", ha="center",
                    fontsize=7, color=INK2)
        else:
            ax.add_patch(FancyBboxPatch((x, y), 40, 27, boxstyle="round,pad=0,rounding_size=2",
                                        fc="white", ec=MUTED, lw=0.9, ls=(0, (3, 2))))
            ax.text(x + 20, y + 18.5, "Nonlinear dynamic", ha="center", fontsize=8.5, fontweight="bold", color=INK2)
            ax.text(x + 20, y + 9.5, "not tested\nthe next experiment", ha="center", fontsize=7, color=INK2,
                    fontstyle="italic", linespacing=1.3)
    X0, Y0 = 16, 9
    cell(X0, Y0 + 36, "pca", "PCA"); cell(X0 + 56, Y0 + 36, "lds", "LDS")
    cell(X0, Y0, "dm", "Diffusion map"); cell(X0 + 56, Y0, None, "")
    ax.text(X0 + 20, 84.5, "STATIC", ha="center", fontsize=7, color=INK2, fontweight="bold")
    ax.text(X0 + 76, 84.5, "DYNAMIC", ha="center", fontsize=7, color=INK2, fontweight="bold")
    ax.text(X0 - 5, Y0 + 49.5, "LINEAR", rotation=90, va="center", ha="center", fontsize=7, color=INK2, fontweight="bold")
    ax.text(X0 - 5, Y0 + 13.5, "NONLINEAR", rotation=90, va="center", ha="center", fontsize=7, color=INK2, fontweight="bold")
    dv = (Xr.lds - Xr.pca).values
    ax.annotate("", (X0 + 55, Y0 + 49.5), (X0 + 41, Y0 + 49.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color=EDGE["lds"], mutation_scale=10))
    ax.text(X0 + 48, Y0 + 54, f"{dv.mean():+.1f} cm", ha="center", fontsize=7.5, fontweight="bold", color=EDGE["lds"])
    ax.text(X0 + 48, Y0 + 43.5, f"{neg(dv)}/5 seeds", ha="center", fontsize=6.3, color=INK2)
    dv2 = (Xr.dm - Xr.pca).values
    ax.annotate("", (X0 + 20, Y0 + 28), (X0 + 20, Y0 + 35),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=MUTED, mutation_scale=8))
    ax.text(X0 + 22.5, Y0 + 31.5, f"{dv2.mean():+.1f} cm, {neg(dv2)}/5 lower — no advantage", fontsize=6.3,
            va="center", color=INK2)
    ax.text(X0 + 76, Y0 + 31.5, "?", ha="center", va="center", fontsize=10, color=MUTED, fontweight="bold")

    rl = (Xr.raw_lag - Xr.raw).values; lr = (Xr.lds - Xr.raw_lag).values
    gl = (Xr.lds - Xk.lds).values; gp = (Xr.pca - Xk.pca).values
    ps = (Xk.pca - Gk.pca).values; ls_ = (Xk.lds - Gk.lds).values
    tx, y = 120, 84.5
    ax.text(tx, y, "What has to be preserved?", fontsize=9, fontweight="bold", va="top"); y -= 8
    items = [
        ("1  Temporal continuity of the population state.",
         f"Dynamic beat static by {-dv.mean():.1f} cm in 5/5 seeds. Seeing the past helps "
         f"({-rl.mean():.1f} cm), and the Kalman state model adds {-lr.mean():.1f} cm beyond it (5/5)."),
        ("2  Not static nonlinear geometry.",
         f"The diffusion map matched PCA ({dv2.mean():+.1f} cm, {neg(dv2)}/5) with either readout."),
        ("3  A nonlinear readout of that state.",
         f"kNN lowered LDS error by {gl.mean():.1f} cm vs {gp.mean():.1f} cm for PCA "
         f"(larger in {int((gl > gp).sum())}/5 seeds)."),
        ("4  Why: robustness to recording.",
         f"Sorting raised kNN error by {ps.mean():.0f} cm for PCA but {ls_.mean():.0f} cm for LDS."),
    ]
    for hd, body in items:
        ax.text(tx, y, hd, fontsize=7, fontweight="bold", va="top"); y -= 3.7
        for line in textwrap.wrap(body, 50):
            ax.text(tx + 3.2, y, line, fontsize=6.3, va="top", color=INK2); y -= 3.1
        y -= 2.4
    ax.text(tx, 16, "Sorted spikes; N = 5 paired seeds; mean ± SD of per-seed\n"
            "medians; k/5 = seeds with the stated sign; no p-values\n"
            "(smallest attainable p = 0.0625). Simulated, entorhinal-\n"
            "dominated population; 600 s sessions.", fontsize=5.6, color=MUTED, va="top", linespacing=1.35)
    save(fig, out, "Fig6_answer")


def _fig7_seed(E):
    """Seed whose LDS − PCA Ridge (sorted) is closest to the across-seed median."""
    X = wide(E, "sorted", "ridge")
    d = (X.lds - X.pca).values
    target = float(np.median(d))
    return int(np.argmin(np.abs(d - target)))


# ================================================================= FIG 7
def fig7(D, out):
    if "predictions_window" not in D:
        print("Fig7 skipped (no predictions_window)")
        return
    E, F, W = D["E"], D["F"], D["predictions_window"]
    seed = _fig7_seed(E)
    floor = float(F[(F.source == "sorted") & (F.seed == seed)].floor_median.iloc[0])
    methods = ("pca", "dm", "lds")
    fig = plt.figure(figsize=(W_FULL, 205 * MM))

    # a — arena 2x3
    head_fig(fig, 0.0, 0.975, "a", "Arena view (first 60 s of test; sorted)")
    for ri, dec in enumerate(("ridge", "knn")):
        for ci, method in enumerate(methods):
            ax = fig.add_axes([0.055 + ci * 0.31, 0.755 - ri * 0.245, 0.27, 0.175])
            sub = W[(W.seed == seed) & (W.source == "sorted") & (W.method == method) & (W.decoder == dec)]
            ax.plot(sub.x_true, sub.y_true, color=INK, lw=0.8, zorder=2)
            ax.scatter(sub.x_pred, sub.y_pred, s=1.5, c=COL[method], alpha=0.35,
                       linewidths=0, zorder=3, rasterized=True)
            ax.plot(sub.x_true.iloc[0], sub.y_true.iloc[0], "o", mfc="none", mec=INK, ms=4, mew=0.7, zorder=4)
            med = float(sub.err_cm.median())
            ax.text(0.04, 0.96, f"{SHORT[method]}  {med:.1f} cm", transform=ax.transAxes,
                    fontsize=6, va="top", color=EDGE[method], fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
            ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color("#6e6c66"); sp.set_linewidth(0.5)
            if ri == 1:
                ax.set_xlabel("x (cm)")
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(f"{'Ridge' if dec == 'ridge' else 'kNN'}\ny (cm)")
            else:
                ax.set_yticklabels([])
            ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
            ax.tick_params(labelsize=5.5)

    # b — x(t), y(t)
    head_fig(fig, 0.0, 0.465, "b", "Coordinates over the same window (Ridge; kNN inset)")
    t0 = float(W[(W.seed == seed) & (W.source == "sorted")].t_s.min())
    for yi, coord in enumerate(("x", "y")):
        ax = fig.add_axes([0.08, 0.325 - yi * 0.090, 0.52, 0.075])
        true = W[(W.seed == seed) & (W.source == "sorted") & (W.method == "pca") & (W.decoder == "ridge")]
        ax.plot(true.t_s - t0, true[f"{coord}_true"], color=INK, lw=1.0, zorder=2, label="true")
        for method in methods:
            sub = W[(W.seed == seed) & (W.source == "sorted") & (W.method == method) & (W.decoder == "ridge")]
            ax.plot(sub.t_s - t0, sub[f"{coord}_pred"], color=COL[method], lw=0.8, zorder=3)
        ax.set_xlim(0, 60); ax.set_ylabel(f"{coord} (cm)"); ax.set_ylim(0, 100)
        if yi == 1:
            ax.set_xlabel("Time in window (s)")
        else:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=5.5)
    # kNN inset
    ax = fig.add_axes([0.68, 0.285, 0.28, 0.140])
    true = W[(W.seed == seed) & (W.source == "sorted") & (W.method == "pca") & (W.decoder == "knn")]
    ax.plot(true.t_s - t0, true.x_true, color=INK, lw=0.9, zorder=2, label="true")
    for method in methods:
        sub = W[(W.seed == seed) & (W.source == "sorted") & (W.method == method) & (W.decoder == "knn")]
        ax.plot(sub.t_s - t0, sub.x_pred, color=COL[method], lw=0.7, zorder=3, label=SHORT[method])
    ax.set_title("kNN  x(t)", fontsize=6, pad=2)
    ax.set_xlim(0, 60); ax.set_ylim(0, 100)
    ax.set_xlabel("Time (s)", fontsize=5.5)
    ax.set_ylabel("x (cm)", fontsize=5.5)
    ax.tick_params(labelsize=5)
    ax.legend(loc="upper right", fontsize=5, ncol=2, frameon=True, fancybox=False,
              edgecolor="none", framealpha=0.9, handlelength=1.2, columnspacing=0.8)

    # c — rolling error
    head_fig(fig, 0.0, 0.200, "c", "Euclidean error (2 s rolling median)")
    ax = fig.add_axes([0.08, 0.050, 0.52, 0.120])
    win = int(round(2.0 / 0.05))  # 2 s at 50 ms
    for method in methods:
        sub = W[(W.seed == seed) & (W.source == "sorted") & (W.method == method) & (W.decoder == "ridge")].sort_values("t_s")
        roll = sub.err_cm.rolling(win, center=True, min_periods=1).median()
        ax.plot(sub.t_s - t0, roll, color=COL[method], lw=0.9, label=SHORT[method])
    ax.axhline(floor, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(1.0, floor + 1.5, "chance", fontsize=5.5, color=INK2, va="bottom", ha="left")
    ax.set_xlim(0, 60); ax.set_xlabel("Time in window (s)"); ax.set_ylabel("Error (cm)")
    ax.legend(loc="upper right", fontsize=5.5, ncol=3, frameon=True, fancybox=False,
              edgecolor="none", framealpha=0.9)

    # d — per-seed strip
    head_fig(fig, 0.66, 0.215, "d", "Window median, all seeds")
    ax = fig.add_axes([0.70, 0.055, 0.26, 0.125])
    for j, method in enumerate(methods):
        for s in range(5):
            sub = W[(W.seed == s) & (W.source == "sorted") & (W.method == method) & (W.decoder == "ridge")]
            med = float(sub.err_cm.median())
            ax.scatter([j], [med], s=18 if s == seed else 10,
                       facecolor=COL[method] if s == seed else "white",
                       edgecolor=EDGE[method], lw=0.7, zorder=3)
    ax.set_xticks(range(3), [SHORT[m] for m in methods])
    for t, m in zip(ax.get_xticklabels(), methods):
        t.set_color(EDGE[m]); t.set_fontweight("bold")
    ax.set_ylabel("Median error (cm)"); ax.set_xlim(-0.5, 2.5)
    ax.text(0.5, 1.08, f"seed {seed} filled (closest LDS−PCA to median)",
            transform=ax.transAxes, fontsize=5.2, color=INK2, ha="center")
    save(fig, out, "Fig7_trajectories")


# ================================================================= FIG S1
def figS1(D, out):
    if "d_sweep" not in D:
        print("FigS1 skipped (no d_sweep)")
        return
    DS = D["d_sweep"]
    dims = [2, 3, 5, 10, 20]
    methods = [m for m in ("pca", "dm", "isomap", "lds", "gpfa")
               if m in set(DS.method)]
    fig = plt.figure(figsize=(W_FULL, 130 * MM))
    head_fig(fig, 0.0, 0.955, "", "Latent dimensionality (Phase 3 per-fold refit)")
    panels = [("sorted", "ridge", 0, 0), ("sorted", "knn", 0, 1),
              ("ground_truth", "ridge", 1, 0), ("ground_truth", "knn", 1, 1)]
    for src, dec, r, c in panels:
        ax = fig.add_axes([0.08 + c * 0.48, 0.55 - r * 0.45, 0.40, 0.35])
        col = f"{dec}_median"
        for method in methods:
            sub = DS[(DS.source == src) & (DS.method == method)]
            ys, ses, xs = [], [], []
            for d in dims:
                v = sub[sub.d == d][col].values
                if len(v) == 0:
                    continue
                xs.append(d); ys.append(v.mean()); ses.append(v.std(ddof=1) / np.sqrt(len(v)))
            ax.errorbar(xs, ys, yerr=ses, color=EDGE[method], ecolor=EDGE[method],
                        elinewidth=0.6, capsize=0, lw=1.0, marker=MRK[method], ms=3.5,
                        mfc=COL[method] if method not in ("gpfa", "isomap") else "white",
                        mec=EDGE[method], mew=0.5, label=SHORT[method])
            # selected d open rings
            for s in range(5):
                sel = sub[(sub.seed == s) & (sub.selected)]
                if len(sel):
                    d = int(sel.d.iloc[0])
                    y = float(sel[col].iloc[0])
                    ax.scatter([d], [y], s=28, facecolor="none", edgecolor=EDGE[method],
                               lw=0.9, zorder=4)
        ax.set_xscale("log"); ax.set_xticks(dims); ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("Latent d")
        ax.set_ylabel(f"{'Ridge' if dec == 'ridge' else 'kNN'} test median (cm)\nmean ± s.e.m.")
        ttl = f"{'Sorted' if src == 'sorted' else 'Ground truth'} · {'Ridge' if dec == 'ridge' else 'kNN'}"
        ax.set_title(ttl, fontsize=7, loc="left", fontweight="bold")
        if r == 0 and c == 1:
            ax.legend(fontsize=5.5, loc="lower left", ncol=1, frameon=True,
                      fancybox=False, edgecolor="none", framealpha=0.92)
    save(fig, out, "FigS1_latent_d")


# ================================================================= FIG S2
def figS2(D, out):
    SH = D["SH"]
    fig = plt.figure(figsize=(W_FULL, 110 * MM))
    head_fig(fig, 0.0, 0.955, "", "Full time-shift null (20 shifts × 5 seeds)")
    methods = REPS
    for ci, src in enumerate(("sorted", "ground_truth")):
        ax = fig.add_axes([0.08 + ci * 0.48, 0.12, 0.40, 0.75])
        for j, method in enumerate(methods):
            sub = SH[(SH.source == src) & (SH.rep == method)]
            for dec, xo, mk in (("ridge", -0.15, "o"), ("knn", 0.15, "s")):
                col = f"{dec}_minus_floor"
                vals = sub[col].values
                jitter = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.008
                ax.scatter(np.full(len(vals), j + xo) + jitter, vals, s=4, marker=mk,
                           facecolor=COL[method] if dec == "ridge" else "white",
                           edgecolor=EDGE[method], lw=0.35, alpha=0.7, zorder=3)
        ax.axhline(-2.0, color=FAILC, lw=0.8)
        ax.axhline(0.0, color=MUTED, lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks(range(len(methods)), [SHORT[m] for m in methods],
                      rotation=35, ha="right", rotation_mode="anchor")
        for t, m in zip(ax.get_xticklabels(), methods):
            if m in QUAD:
                t.set_color(EDGE[m]); t.set_fontweight("bold")
        ax.set_ylabel("Shifted − floor (cm)" if ci == 0 else "")
        ax.set_title("Sorted" if src == "sorted" else "Ground truth", fontsize=7, loc="left", fontweight="bold")
        ax.set_ylim(-15, 20)
        if ci == 1:
            ax.legend(handles=[
                Line2D([0], [0], marker="o", color="none", markerfacecolor=INK,
                       markeredgecolor=INK, markersize=4, label="Ridge"),
                Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
                       markeredgecolor=INK, markersize=4, label="kNN"),
                Line2D([0], [0], color=FAILC, lw=0.8, label="−2 cm line"),
            ], fontsize=5.5, loc="upper right")
    save(fig, out, "FigS2_a13_full")


# ================================================================= FIG S3
def figS3(D, out):
    if "predictions_window" not in D:
        print("FigS3 skipped (no predictions_window)")
        return
    W = D["predictions_window"]
    methods = ("pca", "dm", "lds")
    for dec, tag in (("knn", "FigS3_trajectories_knn"), ("ridge", "FigS3_trajectories_ridge")):
        fig = plt.figure(figsize=(W_FULL, 220 * MM))
        head_fig(fig, 0.0, 0.975, "", f"Decoded trajectories, all seeds ({'kNN' if dec == 'knn' else 'Ridge'})")
        for s in range(5):
            for ci, method in enumerate(methods):
                ax = fig.add_axes([0.06 + ci * 0.32, 0.78 - s * 0.155, 0.28, 0.13])
                sub = W[(W.seed == s) & (W.source == "sorted") & (W.method == method) & (W.decoder == dec)]
                ax.plot(sub.x_true, sub.y_true, color=INK, lw=0.8, zorder=2)
                ax.scatter(sub.x_pred, sub.y_pred, s=1.5, c=COL[method], alpha=0.35,
                           linewidths=0, zorder=3, rasterized=True)
                ax.plot(sub.x_true.iloc[0], sub.y_true.iloc[0], "o", mfc="none", mec=INK, ms=3, mew=0.6)
                med = float(sub.err_cm.median())
                ax.text(0.04, 0.95, f"s{s} {SHORT[method]}  {med:.1f}", transform=ax.transAxes,
                        fontsize=5.5, va="top", color=EDGE[method],
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
                ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color("#6e6c66"); sp.set_linewidth(0.4)
                ax.set_xticks([0, 100]); ax.set_yticks([0, 100])
                ax.tick_params(labelsize=5)
                if s == 4:
                    ax.set_xlabel("x (cm)")
                if ci == 0:
                    ax.set_ylabel("y (cm)")
        save(fig, out, tag)


# ================================================================= FIG S4
def figS4(D, out):
    if not all(k in D for k in ("error_maps", "center_pull", "error_cdf")):
        print("FigS4 skipped (missing failure-mode tables)")
        return
    EM, CP, CDF = D["error_maps"], D["center_pull"], D["error_cdf"]
    JR = D.get("jump_rate")
    methods = ("pca", "dm", "lds")
    fig = plt.figure(figsize=(W_FULL, 230 * MM))

    # a — spatial error maps
    head_fig(fig, 0.0, 0.975, "a", "Spatial error maps (median across seeds; sorted)")
    vals = []
    grids = {}
    for dec in ("ridge", "knn"):
        for method in methods:
            sub = EM[(EM.source == "sorted") & (EM.method == method) & (EM.decoder == dec)]
            g = np.full((10, 10), np.nan)
            nmat = np.full((10, 10), 0.0)
            for (ix, iy), grp in sub.groupby(["bin_x", "bin_y"]):
                per_seed = grp.groupby("seed").agg(med=("median_err", "first"), n=("n", "first"))
                meds = per_seed.med.values
                meds = meds[~np.isnan(meds)]
                g[int(iy), int(ix)] = float(np.median(meds)) if len(meds) else np.nan
                nmat[int(iy), int(ix)] = float(per_seed.n.mean())
            grids[(dec, method)] = (g, nmat)
            vals.extend(g[~np.isnan(g)].tolist())
    vmax = float(np.nanpercentile(vals, 95)) if vals else 40
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("err", ["#ffffff", "#3a3936"])
    im = None
    for ri, dec in enumerate(("ridge", "knn")):
        for ci, method in enumerate(methods):
            ax = fig.add_axes([0.06 + ci * 0.22, 0.74 - ri * 0.20, 0.18, 0.145])
            g, nmat = grids[(dec, method)]
            show = np.ma.array(g, mask=(nmat < 10) | np.isnan(g))
            im = ax.imshow(show, origin="lower", extent=[0, 100, 0, 100],
                           cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            for ix in range(10):
                for iy in range(10):
                    if nmat[iy, ix] < 10:
                        ax.add_patch(Rectangle((ix * 10, iy * 10), 10, 10,
                                               fill=False, hatch="////", edgecolor="#b3b1aa", lw=0.3))
            ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{'Ridge' if dec == 'ridge' else 'kNN'} · {SHORT[method]}",
                         fontsize=6, color=EDGE[method])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.4)
    cax = fig.add_axes([0.72, 0.60, 0.015, 0.26])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Median error (cm)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)

    # b — centre-pull slopes
    head_fig(fig, 0.0, 0.52, "b", "Centre-pull slope (sorted)")
    ax = fig.add_axes([0.08, 0.365, 0.40, 0.12])
    slopes = CP[(CP.kind == "slope") & (CP.source == "sorted")]
    for di, dec in enumerate(("ridge", "knn")):
        for j, method in enumerate(methods):
            v = slopes[(slopes.decoder == dec) & (slopes.method == method)].slope.values
            x = di * 4 + j
            ax.scatter(np.full(5, x), v, s=12, facecolor=COL[method] if dec == "ridge" else "white",
                       edgecolor=EDGE[method], lw=0.6, zorder=3)
            ax.plot([x - 0.25, x + 0.25], [v.mean(), v.mean()], color=EDGE[method], lw=1.2)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(7.2, 1.02, "no shrinkage", fontsize=5.5, color=INK2, va="bottom")
    ax.set_xticks([0, 1, 2, 4, 5, 6],
                  [SHORT[m] for m in methods] + [SHORT[m] for m in methods])
    for i, t in enumerate(ax.get_xticklabels()):
        m = methods[i % 3]; t.set_color(EDGE[m])
    # Group labels inside the axes (avoid bleeding into panel d)
    ax.text(1.0, 0.02, "Ridge", ha="center", va="bottom", fontsize=6.5, color=INK2,
            fontweight="bold", transform=ax.get_xaxis_transform(),
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    ax.text(5.0, 0.02, "kNN", ha="center", va="bottom", fontsize=6.5, color=INK2,
            fontweight="bold", transform=ax.get_xaxis_transform(),
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    ax.set_ylabel("OLS slope"); ax.set_ylim(0.15, 1.2)
    ax.set_xlim(-0.6, 6.8)

    # c — pooled CDFs
    head_fig(fig, 0.52, 0.52, "c", "Pooled error CDF (sorted)")
    ax = fig.add_axes([0.58, 0.365, 0.38, 0.12])
    for method in methods:
        for dec, ls in (("ridge", "-"), ("knn", "--")):
            sub = CDF[(CDF.source == "sorted") & (CDF.method == method) & (CDF.decoder == dec)]
            ax.plot(sub.err_cm, sub.q, color=COL[method], lw=0.9, ls=ls)
    ax.set_xlabel("Error (cm)"); ax.set_ylabel("CDF")
    ax.set_xlim(0, 80); ax.set_ylim(0, 1)
    ax.legend(handles=[
        Line2D([0], [0], color=COL["pca"], lw=0.9, label="PCA"),
        Line2D([0], [0], color=COL["dm"], lw=0.9, label="DM"),
        Line2D([0], [0], color=COL["lds"], lw=0.9, label="LDS"),
        Line2D([0], [0], color=INK, lw=0.9, ls="-", label="Ridge"),
        Line2D([0], [0], color=INK, lw=0.9, ls="--", label="kNN"),
    ], fontsize=5.5, loc="lower right", ncol=2)

    # d — jump rate
    head_fig(fig, 0.0, 0.30, "d", "Jump rate (> 20 cm / 50 ms step)")
    ax = fig.add_axes([0.08, 0.07, 0.88, 0.18])
    if JR is None or JR.empty:
        ax.text(0.5, 0.5, "jump-rate table absent", transform=ax.transAxes,
                ha="center", va="center", color=MUTED)
        ax.axis("off")
    else:
        groups = [("sorted", "ridge"), ("sorted", "knn"),
                  ("ground_truth", "ridge"), ("ground_truth", "knn")]
        xticks, xlabels = [], []
        for gi, (src, dec) in enumerate(groups):
            for j, method in enumerate(methods):
                v = JR[(JR.source == src) & (JR.decoder == dec) & (JR.method == method)].jump_rate.values
                x = gi * 4 + j
                xticks.append(x)
                xlabels.append(SHORT[method])
                if len(v) == 0:
                    continue
                ax.scatter(np.full(len(v), x), v, s=14,
                           facecolor=COL[method] if dec == "ridge" else "white",
                           edgecolor=EDGE[method], lw=0.7, zorder=3)
                ax.plot([x - 0.28, x + 0.28], [v.mean(), v.mean()], color=EDGE[method], lw=1.3)
            true_v = JR[(JR.source == src) & (JR.method == "true")].jump_rate.values
            if len(true_v):
                cx = gi * 4 + 1
                ax.plot([cx - 1.4, cx + 1.4], [true_v.mean(), true_v.mean()],
                        color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.set_xticks(xticks, xlabels)
        for i, t in enumerate(ax.get_xticklabels()):
            t.set_color(EDGE[methods[i % 3]])
        for gi, (src, dec) in enumerate(groups):
            lab = f"{'Sorted' if src == 'sorted' else 'GT'} · {'Ridge' if dec == 'ridge' else 'kNN'}"
            ax.text(gi * 4 + 1, -0.12, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.2, color=INK2, fontweight="bold")
        ax.axhline(0.0, color=MUTED, lw=0.5)
        ax.set_ylabel("Jump fraction")
        ax.set_ylim(-0.02, max(0.55, float(JR[JR.method != "true"].jump_rate.max()) * 1.15))
        ax.set_xlim(-0.7, 15.5)
        ax.text(0.99, 0.95, "dashed = true path (0)", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.5, color=INK2)
    save(fig, out, "FigS4_failure_modes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figures")
    ap.add_argument("--data", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = load(a.data)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, figS1, figS2, figS3, figS4):
        f(D, a.out)
    print("wrote", sorted(x for x in os.listdir(a.out) if not x.startswith("prev")))
