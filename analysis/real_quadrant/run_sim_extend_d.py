"""Extend sim pca/lds latent-d grid to {2,3,5,10,20,40}; archive grid20.

Sim n_units ≈ 86 → ⌊n_units/2⌋ = 43, so stop at d=40 (no d=80).
Existing results kept as *_grid20.json; new rows written alongside.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from analysis.real_quadrant.posthoc_smooth import apply_smoothed_method
from analysis.real_quadrant.run_m3 import PRIMARY_CONTRASTS, SECONDARY_CONTRASTS, _contrast_pair
from realtime.quadrant_n5 import derive_seed_streams, load_quadrant_n5_yaml
from realtime.quadrant_n5_run import OUTPUT_ROOT, _git_sha, analyze_source

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = OUTPUT_ROOT
GRID20 = (2, 3, 5, 10, 20)
GRID_EXT = (2, 3, 5, 10, 20, 40)  # n_units/2 ≈ 43
METHODS = ("pca", "lds")
SOURCES = ("sorted", "ground_truth")
N_SEEDS = 5
BLAS_THREADS = 8


def _overlay_cfg() -> dict[str, Any]:
    cfg = dict(load_quadrant_n5_yaml())
    cfg["latent_dims"] = list(GRID_EXT)
    cfg["nested_fit_d"] = int(max(GRID_EXT))
    cfg["phase3"] = dict(cfg["phase3"], learning_curve_source="__skip__")
    cfg["grid_extension"] = {
        "rule": (
            "if selected d equals grid max on most sessions, double until "
            "mode selected d < max or d reaches n_units/2"
        ),
        "domain": "sim",
        "grid20": list(GRID20),
        "grid_ext": list(GRID_EXT),
        "nested_fit_d": int(max(GRID_EXT)),
        "n_units_approx": 86,
        "n_units_half_cap": 43,
    }
    stamp = hashlib.sha256(
        json.dumps(
            {
                "base_config_sha256": cfg["config_sha256"],
                "latent_dims": cfg["latent_dims"],
                "nested_fit_d": cfg["nested_fit_d"],
                "grid_extension": "sim",
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cfg["config_sha256"] = stamp
    return cfg


def _set_blas(n: int = BLAS_THREADS) -> None:
    for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(n)


def _archive_grid20(out_dir: Path) -> list[str]:
    touched = []
    for m in METHODS:
        src = out_dir / f"{m}.json"
        dst = out_dir / f"{m}_grid20.json"
        if not src.is_file():
            continue
        if not dst.is_file():
            shutil.copy2(src, dst)
            src.unlink()
            touched.append(f"{m}->grid20")
        else:
            touched.append(f"{m}_grid20 exists")
    ds = out_dir / "d_sweep.json"
    ds_bak = out_dir / "d_sweep_grid20.json"
    if ds.is_file() and not ds_bak.is_file():
        shutil.copy2(ds, ds_bak)
        touched.append("d_sweep_grid20")
    return touched


def _archive_smooth_grid20(out_dir: Path) -> None:
    for m in METHODS:
        src = out_dir / f"{m}_smooth_posthoc.json"
        dst = out_dir / f"{m}_smooth_posthoc_grid20.json"
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)


def _write_ext_smooth(out_dir: Path, cfg: dict[str, Any], streams: dict[str, int]) -> None:
    """Refit pca/lds once at selected d and write extended-grid smooth posthoc rows."""
    from realtime.quadrant_n5_run import (
        _load_arrays_for_analyze_source,
        _make_rep,
        _sqrt_zscore_train,
        _valid_mask,
        fit_transform_representation,
    )
    from realtime.train_decoder import causal_train_test_split

    sim_dir = out_dir.parent / "sim"
    spike_source = out_dir.name
    seed_index = int(out_dir.parent.name.split("_")[1])
    X_counts, y, decode_times, _units_df, _unit_ids, load_meta = (
        _load_arrays_for_analyze_source(cfg, sim_dir, spike_source, bundle=None)
    )
    split = cfg["split"]
    train_mask, test_mask = causal_train_test_split(
        decode_times, float(split["train_frac"]), gap_s=float(split["gap_s"]),
    )
    target_valid = np.asarray(load_meta["target_valid"], dtype=bool)
    train_mask = train_mask & target_valid
    test_mask = test_mask & target_valid
    X, _ = _sqrt_zscore_train(X_counts, train_mask)
    n = len(decode_times)
    summary = json.loads((out_dir / "source_summary.json").read_text())
    floor = float(summary["floor"]["median"])
    for method in METHODS:
        rec_path = out_dir / f"{method}.json"
        if not rec_path.is_file():
            continue
        rec = json.loads(rec_path.read_text())
        if "ridge" not in rec:
            continue
        keys = ("raw_lag", method)
        valid = {m: _valid_mask(m, n, cfg) for m in keys}
        train_ok = train_mask.copy()
        eval_mask = test_mask.copy()
        for m in keys:
            train_ok &= valid[m]
            eval_mask &= valid[m]
        primary_d = int(rec["primary_d"])
        nested = bool((cfg["representations"].get(method) or {}).get("nested"))
        d_fit = int(cfg["nested_fit_d"]) if nested else primary_d
        model = _make_rep(
            method, d_fit, cfg, int(streams["methods"]), n_fit=int(train_ok.sum()),
        )
        Z = fit_transform_representation(method, model, X, train_ok)
        if nested:
            Z = Z[:, :primary_d]
        sm = apply_smoothed_method(
            Z, y, decode_times, train_ok, eval_mask, float(rec["ridge_alpha"]),
            n_blocks=int(split["inner_cv_blocks"]), gap_s=float(split["gap_s"]),
        )
        ridge_med = float(sm["ridge"]["median"])
        payload = {
            "method": f"{method}_smooth",
            "base_method": method,
            "grid_label": "grid_ext",
            "seed_index": seed_index,
            "spike_source": spike_source,
            "selected_d": primary_d,
            "ema_tau_s": sm["tau_s"],
            "inner_cv_median_cm": sm["inner_cv_median_cm"],
            "ridge": sm["ridge"],
            "ridge_median_cm": ridge_med,
            "floor_median_cm": floor,
            "normalized_error": ridge_med / floor if floor > 0 else None,
            "note": "Extended-grid smooth; grid20 copy in *_smooth_posthoc_grid20.json",
        }
        (out_dir / f"{method}_smooth_posthoc.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )


def _streams(seed_index: int, out_dir: Path, cfg: dict[str, Any]) -> dict[str, int]:
    for name in ("pca_grid20.json", "pca.json", "raw.json", "source_summary.json"):
        p = out_dir / name
        if p.is_file():
            rec = json.loads(p.read_text())
            streams = rec.get("seed_streams")
            if streams:
                return dict(streams)
    seeds = cfg["seeds"]
    return derive_seed_streams(
        int(seeds["master_seed"]),
        int(seeds["n_seeds"]),
        seed_index,
        tuple(seeds["components"]),
    )


def _run_one(seed_index: int, spike_source: str, cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = SIM_ROOT / f"seed_{seed_index}" / spike_source
    sim_dir = SIM_ROOT / f"seed_{seed_index}" / "sim"
    streams = _streams(seed_index, out_dir, cfg)
    t0 = time.perf_counter()
    _archive_smooth_grid20(out_dir)
    archived = _archive_grid20(out_dir)
    print(
        f"[seed {seed_index}/{spike_source}] archive={archived} grid={cfg['latent_dims']}",
        flush=True,
    )
    summary = analyze_source(
        cfg, sim_dir, spike_source, streams, seed_index,
        output_dir=out_dir,
        method_keys=("raw_lag", "pca", "lds"),
        save_figure_contract=False,
    )
    by_m = {}
    for m in ("raw", "raw_lag", "pca", "dm", "lds", "isomap", "gpfa"):
        p = out_dir / (f"{m}_grid20.json" if m in METHODS else f"{m}.json")
        if m in METHODS and not p.is_file():
            p = out_dir / f"{m}.json"
        if p.is_file():
            rec = json.loads(p.read_text())
            if "ridge" in rec:
                by_m[m] = rec
    for m in summary.get("methods", []):
        by_m[m["method"]] = m
    merged = dict(summary)
    merged["methods"] = [
        by_m[k] for k in ("raw", "raw_lag", "pca", "dm", "lds", "isomap", "gpfa")
        if k in by_m
    ]
    (out_dir / "source_summary.json").write_text(
        json.dumps(merged, indent=2, default=str) + "\n"
    )
    _write_ext_smooth(out_dir, cfg, streams)

    wall = time.perf_counter() - t0
    sel = {
        m: json.loads((out_dir / f"{m}.json").read_text()).get("primary_d")
        for m in METHODS if (out_dir / f"{m}.json").is_file()
    }
    sel_g20 = {
        f"{m}_grid20": json.loads((out_dir / f"{m}_grid20.json").read_text()).get("primary_d")
        for m in METHODS if (out_dir / f"{m}_grid20.json").is_file()
    }
    print(
        f"[seed {seed_index}/{spike_source}] done {wall:.1f}s selected={sel | sel_g20}",
        flush=True,
    )
    return {
        "ok": True, "seed_index": seed_index, "spike_source": spike_source,
        "wall_s": wall, "selected_d": {**sel, **sel_g20},
    }


def _norm_from_rec(rec: dict[str, Any], floor: float) -> float | None:
    if "ridge" not in rec or floor <= 0:
        return None
    return float(rec["ridge"]["median"]) / floor


def write_sim_extend_report(cfg: dict[str, Any]) -> dict[str, Any]:
    """Selected d + contrasts for grid_ext and grid20 (normalized error)."""
    selected_rows = []
    err_vs_d: dict[str, dict[int, list[float]]] = {
        "pca": {}, "lds": {}, "pca_grid20": {}, "lds_grid20": {},
    }
    entries_ext: list[dict[str, float]] = []
    entries_20: list[dict[str, float]] = []

    for seed in range(N_SEEDS):
        for src in SOURCES:
            out_dir = SIM_ROOT / f"seed_{seed}" / src
            summary = json.loads((out_dir / "source_summary.json").read_text())
            floor = float(summary["floor"]["median"])
            row: dict[str, Any] = {
                "seed": seed, "source": src, "n_units": summary.get("n_units"),
            }
            norms_ext: dict[str, float] = {}
            norms_20: dict[str, float] = {}
            for m in METHODS:
                live = out_dir / f"{m}.json"
                g20 = out_dir / f"{m}_grid20.json"
                if live.is_file():
                    rec = json.loads(live.read_text())
                    row[m] = rec.get("primary_d")
                    ne = _norm_from_rec(rec, floor)
                    if ne is not None:
                        norms_ext[m] = ne
                        row[f"{m}_norm"] = ne
                if g20.is_file():
                    rec = json.loads(g20.read_text())
                    row[f"{m}_grid20"] = rec.get("primary_d")
                    ne = _norm_from_rec(rec, floor)
                    if ne is not None:
                        norms_20[m] = ne
                        row[f"{m}_grid20_norm"] = ne
            for m in ("raw", "raw_lag", "dm", "isomap", "gpfa"):
                p = out_dir / f"{m}.json"
                if not p.is_file():
                    continue
                ne = _norm_from_rec(json.loads(p.read_text()), floor)
                if ne is not None:
                    norms_ext[m] = ne
                    norms_20[m] = ne
            for m, path in (
                ("raw_smooth", "raw_smooth_posthoc.json"),
                ("dm_smooth", "dm_smooth_posthoc.json"),
            ):
                p = out_dir / path
                if p.is_file():
                    ne = float(json.loads(p.read_text())["normalized_error"])
                    norms_ext[m] = ne
                    norms_20[m] = ne
            for m in METHODS:
                ext_p = out_dir / f"{m}_smooth_posthoc.json"
                g20_p = out_dir / f"{m}_smooth_posthoc_grid20.json"
                if ext_p.is_file():
                    norms_ext[f"{m}_smooth"] = float(
                        json.loads(ext_p.read_text())["normalized_error"]
                    )
                if g20_p.is_file():
                    norms_20[f"{m}_smooth"] = float(
                        json.loads(g20_p.read_text())["normalized_error"]
                    )
                elif (out_dir / f"{m}_smooth_posthoc.json").is_file() and m not in (
                    # if only one file and no archive, treat as grid20-era
                ):
                    pass
            selected_rows.append(row)
            if norms_ext:
                entries_ext.append({"seed": seed, "source": src, **norms_ext})
            if norms_20:
                entries_20.append({"seed": seed, "source": src, **norms_20})

            ds = out_dir / "d_sweep.json"
            if ds.is_file():
                for r in json.loads(ds.read_text()).get("rows") or []:
                    if r.get("method") not in METHODS:
                        continue
                    d = int(r["d"])
                    err = r.get("ridge_median") or r.get("inner_cv_ridge_median")
                    if err is None:
                        continue
                    err_vs_d[r["method"]].setdefault(d, []).append(float(err) / floor)
            ds20 = out_dir / "d_sweep_grid20.json"
            if ds20.is_file():
                for r in json.loads(ds20.read_text()).get("rows") or []:
                    if r.get("method") not in METHODS:
                        continue
                    d = int(r["d"])
                    err = r.get("ridge_median") or r.get("inner_cv_ridge_median")
                    if err is None:
                        continue
                    err_vs_d[f"{r['method']}_grid20"].setdefault(d, []).append(
                        float(err) / floor
                    )

    def _paired_contrasts(entries: list[dict[str, float]]) -> dict[str, Any]:
        animal_means = {
            f"s{e['seed']}_{e['source']}": {
                k: float(v) for k, v in e.items()
                if k not in ("seed", "source") and isinstance(v, (int, float))
            }
            for e in entries
        }
        return {
            "primary": [
                _contrast_pair(a, b, animal_means) for a, b in PRIMARY_CONTRASTS
                if "gpfa_causal" not in (a, b)
            ],
            "secondary": [
                _contrast_pair(a, b, animal_means) for a, b in SECONDARY_CONTRASTS
                if "gpfa_causal" not in (a, b)
            ],
            "n_units": len(animal_means),
        }

    out = {
        "domain": "sim",
        "grid_ext": list(GRID_EXT),
        "grid20": list(GRID20),
        "n_seeds": N_SEEDS,
        "sources": list(SOURCES),
        "selected_d_per_seed_source": selected_rows,
        "selected_d_mode": {
            "pca": Counter(r.get("pca") for r in selected_rows).most_common(3),
            "lds": Counter(r.get("lds") for r in selected_rows).most_common(3),
            "pca_grid20": Counter(r.get("pca_grid20") for r in selected_rows).most_common(3),
            "lds_grid20": Counter(r.get("lds_grid20") for r in selected_rows).most_common(3),
        },
        "n_at_grid_max": {
            "pca": sum(1 for r in selected_rows if r.get("pca") == max(GRID_EXT)),
            "lds": sum(1 for r in selected_rows if r.get("lds") == max(GRID_EXT)),
        },
        "error_vs_d_median_normalized": {
            m: {str(d): float(np.median(v)) for d, v in sorted(by_d.items())}
            for m, by_d in err_vs_d.items()
        },
        "contrasts_grid_ext": _paired_contrasts(entries_ext),
        "contrasts_grid20": _paired_contrasts(entries_20),
        "git_sha": _git_sha(),
    }
    (SIM_ROOT / "extend_d_report.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--source", type=str, default=None, choices=["sorted", "ground_truth"])
    args = parser.parse_args(argv)
    _set_blas()
    cfg = _overlay_cfg()

    if args.report_only:
        report = write_sim_extend_report(cfg)
        print(f"wrote {SIM_ROOT / 'extend_d_report.json'} n={len(report['selected_d_per_seed_source'])}")
        return 0

    seeds = [args.seed] if args.seed is not None else list(range(N_SEEDS))
    sources = [args.source] if args.source else list(SOURCES)
    results = []
    t0 = time.perf_counter()
    for seed in seeds:
        for src in sources:
            try:
                results.append(_run_one(seed, src, cfg))
            except Exception as exc:
                print(f"FAILED seed={seed} src={src}: {exc}", flush=True)
                results.append({
                    "ok": False, "seed_index": seed, "spike_source": src,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    wall = time.perf_counter() - t0
    (SIM_ROOT / "extend_d_results.json").write_text(
        json.dumps({"wall_s": wall, "grid": list(GRID_EXT), "results": results},
                   indent=2, default=str) + "\n"
    )
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"sim extend finished {n_ok}/{len(results)} in {wall:.1f}s", flush=True)
    if n_ok == len(results):
        write_sim_extend_report(cfg)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
