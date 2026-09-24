"""Phase 5 PDFs for the frozen n=5 quadrant experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from realtime.quadrant_n5 import SEEDS_0_4_PROVENANCE_SHA, load_quadrant_n5_yaml, report_code_sha
from realtime.quadrant_n5_replay import display_a9_label
from realtime.quadrant_n5_run import METHOD_KEYS, OUTPUT_ROOT

POP_BLURB = (
    "Population: hippocampal formation sampled by the lab NP2.0 probe track "
    "(mostly entorhinal grid / head-direction / speed / BVC plus "
    "subicular–CA1 transition units). No CA1/CA3 place cells."
)

QUADRANT = {
    "linear_static": "pca",
    "nonlinear_static": "dm",
    "linear_dynamic": "lds",
    "nonlinear_dynamic": None,
}

SEED_4_RIDGE_WEAK_CONTROL = (
    "Seed 4 is a weak-control session for Ridge across all methods "
    "(sorted Ridge Δ range [−1.984, −0.837]; every method negative). "
    "The kNN control is clean on seed 4 (all completed kNN Δ ≥ +1.16). "
    "Ground-truth DM Ridge is A13 FAIL / unreliable "
    "(Δ = −2.544; non-null control / session residual, not leakage). "
    "Ground-truth LDS Ridge also A13 FAIL (Δ = −2.561); kNN remains clean."
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, pad=12)
    ax.text(0.02, 0.96, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace",
            transform=ax.transAxes, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def _composition_page(pdf: PdfPages, seed: int, summary: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.set_title(f"Seed {seed} — population composition", loc="left", fontsize=12)
    by_ct = summary.get("n_units_by_cell_type") or {}
    by_reg = summary.get("n_units_by_region") or {}
    lines = [
        POP_BLURB,
        "",
        f"n_units = {summary.get('n_units')}",
        "",
        "By cell type:",
    ]
    for k, v in sorted(by_ct.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k:12s}  {v}")
    lines.append("")
    lines.append("By region label:")
    for k, v in sorted(by_reg.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {k:36s}  {v}")
    cov = summary.get("coverage") or {}
    lines += [
        "",
        f"Test coverage (bins occupied in train): "
        f"{cov.get('fraction_test_in_train_occupied_bins')}",
        f"train occupied bins {cov.get('n_train_occupied_bins')} / "
        f"test occupied bins {cov.get('n_test_occupied_bins')}",
    ]
    ax.text(0.04, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9,
            transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)
    png = OUTPUT_ROOT / f"seed_{seed}" / "sorted" / "occupancy_maps.png"
    if png.is_file():
        img = plt.imread(png)
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Seed {seed} train / test occupancy")
        pdf.savefig(fig)
        plt.close(fig)


def write_seed_pdf(seed_index: int, cfg: dict[str, Any]) -> Path:
    root = OUTPUT_ROOT / f"seed_{seed_index}"
    sorted_sum = _load(root / "sorted" / "source_summary.json")
    gt_sum = _load(root / "ground_truth" / "source_summary.json")
    audit = _load(root / "audit.json") if (root / "audit.json").is_file() else {}
    dest = OUTPUT_ROOT / "reports" / f"seed_{seed_index}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(dest) as pdf:
        _text_page(pdf, f"quadrant_n5 seed {seed_index} provenance", [
            f"config_sha256  {cfg['config_sha256']}",
            f"probe_track    {cfg.get('probe_track', {}).get('file')}",
            f"probe_sha256   {cfg.get('probe_track_sha256')}",
            f"git_sha        {sorted_sum.get('git_sha')}",
            f"dirty_tree     {sorted_sum.get('dirty_tree')}",
            f"seeds_0_4_code {SEEDS_0_4_PROVENANCE_SHA}",
            f"report_code_sha {report_code_sha()}",
            f"numpy          {(sorted_sum.get('versions') or {}).get('numpy')}",
            f"sklearn        {(sorted_sum.get('versions') or {}).get('sklearn')}",
            f"seed_streams   {sorted_sum.get('seed_streams')}",
            "",
            *( [SEED_4_RIDGE_WEAK_CONTROL, ""] if seed_index == 4 else [] ),
            POP_BLURB,
        ])
        rows = audit.get("rows") or []
        _text_page(pdf, f"Seed {seed_index} audit A1–A14", [
            f"{r['check']:4s} {r['status']:4s}  {r['note']}" for r in rows
        ] or ["audit.json missing"])
        _composition_page(pdf, seed_index, sorted_sum)
        lines = ["sorted  (deployable)", ""]
        for m in sorted_sum.get("methods") or []:
            lines.append(
                f"{m['method']:8s} d={m.get('primary_d')}  "
                f"ridge={m['ridge']['median']:.2f}  knn={m['knn']['median']:.2f}  "
                f"A13 Δr={((m.get('a13') or {}).get('ridge_median_minus_floor'))}"
            )
        lines += ["", "ground_truth  (non-deployable)", ""]
        for m in gt_sum.get("methods") or []:
            a13 = m.get("a13") or {}
            flag = ""
            if a13.get("status") == "FAIL" or a13.get("unreliable"):
                flag = "  A13 FAIL / unreliable"
            lines.append(
                f"{m['method']:8s} d={m.get('primary_d')}  "
                f"ridge={m['ridge']['median']:.2f}  knn={m['knn']['median']:.2f}  "
                f"A13 Δr={a13.get('ridge_median_minus_floor')} "
                f"Δk={a13.get('knn_median_minus_floor')}{flag}"
            )
        if seed_index == 4:
            lines += ["", SEED_4_RIDGE_WEAK_CONTROL]
        _text_page(pdf, f"Seed {seed_index} primary-d errors (cm, median)", lines)
        replay = _load(root / "replay" / "sorted_summary.json") if (
            root / "replay" / "sorted_summary.json"
        ).is_file() else {}
        rlines = [
            "A9 is empirical step-vs-batch on sorted. "
            "No per-step path → 'untested: no per-step path', never offline_only.",
            "A11: Phase-3 ridge vs replay batch ridge, and per-sample pred |offline−step|.",
            f"latency budget {cfg['latency_budget_ms']} ms",
            "",
        ]
        for m in replay.get("methods") or []:
            a9 = display_a9_label(m.get("a9_label"))
            rlines.append(
                f"{m['method']:8s} A9={a9}  "
                f"|Z|_∞={m.get('max_abs_step_vs_batch')}  "
                f"p50={m.get('step_ms_p50')} p99={m.get('step_ms_p99')} ms  "
                f"A11 |ŷ|_∞={m.get('a11_max_abs_pred')}  "
                f"phase3−offline={m.get('phase3_vs_replay_offline_ridge')}"
            )
        if not replay.get("methods"):
            rlines.append("replay/sorted_summary.json missing")
        _text_page(pdf, f"Seed {seed_index} A9 / A11 replay", rlines)
    return dest


def write_aggregate_pdf(cfg: dict[str, Any]) -> Path:
    n = int(cfg["seeds"]["n_seeds"])
    seeds = []
    for i in range(n):
        p = OUTPUT_ROOT / f"seed_{i}" / "sorted" / "source_summary.json"
        if p.is_file():
            seeds.append(_load(p))
    dest = OUTPUT_ROOT / "reports" / "quadrant_n5_summary.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(dest) as pdf:
        _text_page(pdf, "quadrant_n5 aggregate", [
            f"config_sha256 {cfg['config_sha256']}",
            f"seeds_0_4_code {SEEDS_0_4_PROVENANCE_SHA}",
            f"report_code_sha {report_code_sha()}",
            f"n_seeds with sorted results: {len(seeds)}",
            "",
            POP_BLURB,
            "",
            "Nonlinear-dynamic cell is empty (not implemented; GPFA is not that cell).",
            "No p-values. N=5 sign counts only.",
            "",
            SEED_4_RIDGE_WEAK_CONTROL,
        ])
        # Quadrant medians
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5))
        labels = [
            ("linear static", "pca"),
            ("nonlinear static", "dm"),
            ("linear dynamic", "lds"),
            ("nonlinear dynamic", None),
        ]
        for ax, (title, key) in zip(axes.ravel(), labels):
            ax.set_title(title)
            if key is None:
                ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            vals = []
            for s in seeds:
                rec = next(m for m in s["methods"] if m["method"] == key)
                vals.append(rec["ridge"]["median"])
            ax.plot(range(len(vals)), vals, "o-")
            ax.set_xlabel("seed")
            ax.set_ylabel("ridge median cm")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        contrasts = [
            ("dm − pca", "dm", "pca"),
            ("lds − pca", "lds", "pca"),
            ("raw_lag − raw", "raw_lag", "raw"),
            ("lds − raw_lag", "lds", "raw_lag"),
        ]
        lines = ["Paired contrasts on sorted Ridge median (cm)", ""]
        for name, a, b in contrasts:
            diffs = []
            for s in seeds:
                ma = next(m for m in s["methods"] if m["method"] == a)["ridge"]["median"]
                mb = next(m for m in s["methods"] if m["method"] == b)["ridge"]["median"]
                diffs.append(ma - mb)
            sign = sum(d < 0 for d in diffs)
            lines.append(
                f"{name:16s}  mean={np.mean(diffs):+.2f}  "
                f"sd={np.std(diffs, ddof=1) if len(diffs)>1 else float('nan'):.2f}  "
                f"sign(a<b) {sign}/{len(diffs)}  values={[round(d,2) for d in diffs]}"
            )
        _text_page(pdf, "Planned contrasts (sorted, Ridge)", lines)
        a9_lines = [
            "A9 labels and A11 agreement across seeds (sorted).",
            "No per-step path is 'untested: no per-step path', never offline_only.",
            "",
        ]
        for i in range(n):
            rp = OUTPUT_ROOT / f"seed_{i}" / "replay" / "sorted_summary.json"
            if not rp.is_file():
                a9_lines.append(f"seed {i}: replay missing")
                continue
            rec = _load(rp)
            a9_lines.append(f"seed {i}")
            for m in rec.get("methods") or []:
                a9_lines.append(
                    f"  {m['method']:8s} {display_a9_label(m.get('a9_label'))}  "
                    f"|Z|_∞={m.get('max_abs_step_vs_batch')}  "
                    f"p50={m.get('step_ms_p50')} p99={m.get('step_ms_p99')}  "
                    f"A11 |ŷ|_∞={m.get('a11_max_abs_pred')}"
                )
        _text_page(pdf, "A9 / A11 across seeds", a9_lines)
    return dest


def write_all_reports() -> dict[str, str]:
    cfg = load_quadrant_n5_yaml()
    paths = {}
    for i in range(int(cfg["seeds"]["n_seeds"])):
        if (OUTPUT_ROOT / f"seed_{i}" / "sorted" / "source_summary.json").is_file():
            paths[f"seed_{i}"] = str(write_seed_pdf(i, cfg))
    if paths:
        paths["summary"] = str(write_aggregate_pdf(cfg))
    return paths


if __name__ == "__main__":
    print(json.dumps(write_all_reports(), indent=2))
