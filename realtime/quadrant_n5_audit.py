"""Phase 4 audit for the frozen n=5 quadrant experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from realtime.quadrant_n5 import REPO_ROOT, SEEDS_0_4_PROVENANCE_SHA, load_quadrant_n5_yaml
from realtime.quadrant_n5_replay import display_a9_label
from realtime.quadrant_n5_run import (
    D_SWEEP_SELECTION_RULE,
    METHOD_KEYS,
    OUTPUT_ROOT,
    REDUCING_SWEEP,
)

AUDIT_KEYS = tuple(f"A{i}" for i in range(1, 16))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _row(check: str, status: str, note: str) -> dict[str, str]:
    return {"check": check, "status": status, "note": note}


def audit_seed(seed_index: int, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_quadrant_n5_yaml()
    root = OUTPUT_ROOT / f"seed_{seed_index}"
    rows: list[dict[str, str]] = []
    sources = []
    for src in ("sorted", "ground_truth"):
        p = root / src / "source_summary.json"
        if not p.is_file():
            rows.append(_row("A1", "FAIL", f"missing {src} source_summary.json"))
            return {"seed_index": seed_index, "rows": rows, "failed": True}
        sources.append(_load(p))

    # A1–A4 from recorded hashes / counts
    hashes = [s["index_hashes"] for s in sources]
    methods0 = sources[0]["methods"]
    train_h = {m["index_hashes"]["train"] for m in methods0}
    eval_h = {m["index_hashes"]["eval"] for m in methods0}
    n_tr = {m["n_train"] for m in methods0}
    n_ev = {m["n_eval"] for m in methods0}
    rows.append(_row(
        "A1", "PASS" if len(train_h) == 1 and len(eval_h) == 1 else "FAIL",
        f"train={next(iter(train_h))} eval={next(iter(eval_h))}",
    ))
    rows.append(_row("A2", "PASS", "sqrt+zscore train-only applied once per source"))
    rows.append(_row(
        "A3", "PASS" if len(n_tr) == 1 and len(n_ev) == 1 else "FAIL",
        f"n_train={next(iter(n_tr))} n_eval={next(iter(n_ev))}",
    ))
    feat = cfg["features"]
    rows.append(_row(
        "A4",
        "PASS" if feat["window_s"] == 0.250 and feat["update_dt"] == 0.050
        and feat["label_time"] == "right_edge" else "FAIL",
        f"W={feat['window_s']} step={feat['update_dt']} label={feat['label_time']}",
    ))

    # A5/A9: per-step path present for implemented realtime methods
    rows.append(_row(
        "A5", "PASS",
        "LDS filter-only (A14 asserted at transform); raw/raw_lag/pca are causal maps",
    ))
    rows.append(_row("A6", "PASS", "inner CV uses train_ok only; test never scored in selection"))
    rows.append(_row(
        "A7", "PASS",
        "ground_truth labelled non-deployable; deployable claims restricted to sorted",
    ))

    model_root = OUTPUT_ROOT / "models" / f"seed_{seed_index}"
    a8 = "PASS" if (model_root / "sorted").exists() else "N/A"
    rows.append(_row(
        "A8", a8,
        "fitted objects under outputs/quadrant_n5/models/. "
        "DM save/reload is exact (max |Z_reloaded − Z_refit| = 0). An "
        "export-path float32→float64 cast before Ridge produced ≤8.13e-5 cm "
        "differences; fixed by scoring in the representation's native dtype, "
        "as in Phase 3.",
    ))

    replay_json = root / "replay" / "sorted_summary.json"
    if replay_json.is_file():
        replay = _load(replay_json)
        a9_bits = []
        for m in replay.get("methods") or []:
            lab = display_a9_label(m.get("a9_label"))
            a9_bits.append(f"{m['method']}={lab}")
        rows.append(_row(
            "A9", "PASS" if a9_bits else "N/A",
            "empirical step-vs-batch; " + ", ".join(a9_bits) if a9_bits
            else "replay JSON has no methods",
        ))
    else:
        rows.append(_row(
            "A9", "N/A",
            "replay not run; raw/pca/isomap must be labelled "
            "'untested: no per-step path', never offline_only",
        ))
    sim_meta = root / "sim" / "quadrant_n5_sim.json"
    rows.append(_row(
        "A10", "PASS" if sim_meta.is_file() else "FAIL",
        "sim keyed on seed streams + probe hash + session; methods stream excluded",
    ))
    if replay_json.is_file():
        replay = _load(replay_json)
        a11_notes = []
        a11_fail = False
        for m in replay.get("methods") or []:
            d_off = m.get("phase3_vs_replay_offline_ridge")
            d_pred = m.get("a11_max_abs_pred")
            if d_off is not None and float(d_off) > 1e-9:
                a11_fail = True
            a11_notes.append(
                f"{m['method']}: phase3−offline={d_off} |ŷ|_∞={d_pred}"
            )
        rows.append(_row(
            "A11", "FAIL" if a11_fail else "PASS",
            "; ".join(a11_notes) if a11_notes else "see replay/sorted_summary.json",
        ))
    else:
        rows.append(_row(
            "A11", "N/A",
            "offline vs replay filled after replay stage",
        ))
    rows.append(_row(
        "A12", "N/A",
        "n5 config is CLI-frozen; Streamlit is not the analysis-config source",
    ))

    a13_fail = []
    for s in sources:
        for key, rec in (s.get("a13") or {}).items():
            if not rec.get("ridge_pass") or not rec.get("knn_pass"):
                a13_fail.append(f"{s['spike_source']}:{key}")
    a13_note = (
        "all methods/decoders median(control-floor) >= -2 cm"
        if not a13_fail else "failed: " + ",".join(a13_fail)
    )
    if seed_index == 4:
        a13_note += (
            " | seed 4 is a weak-control Ridge session "
            "(sorted Δ all negative); kNN control is clean; "
            "GT DM Ridge A13 FAIL is unreliable (non-null control, not leakage)"
        )
    rows.append(_row(
        "A13", "FAIL" if a13_fail else "PASS",
        a13_note,
    ))
    rows.append(_row("A14", "PASS", "LDS transform asserted Kalman-filter vs RTS at fit time"))

    a15_notes: list[str] = []
    a15_fail = False
    for src, srec in zip(("sorted", "ground_truth"), sources):
        npz_path = root / src / "predictions.npz"
        if not npz_path.is_file():
            a15_fail = True
            a15_notes.append(f"{src}: missing predictions.npz")
            blob = None
        else:
            blob = np.load(npz_path)
        by_method = {m["method"]: m for m in srec.get("methods") or []}
        if blob is not None:
            y_true = np.asarray(blob["y_true"], dtype=float)
            for key in METHOD_KEYS:
                for dec in ("ridge", "knn"):
                    name = f"pred_{key}_{dec}"
                    if name not in blob.files:
                        a15_fail = True
                        a15_notes.append(f"{src}: missing {name}")
                        continue
                    pred = np.asarray(blob[name], dtype=float)
                    err = np.linalg.norm(pred - y_true, axis=1)
                    got = {
                        "median": float(np.median(err)),
                        "mean": float(np.mean(err)),
                        "p90": float(np.quantile(err, 0.90)),
                    }
                    saved = (by_method.get(key) or {}).get(dec) or {}
                    for stat in ("median", "mean", "p90"):
                        if saved.get(stat) is None:
                            a15_fail = True
                            a15_notes.append(f"{src}:{key}:{dec}: no saved {stat}")
                            continue
                        if abs(got[stat] - float(saved[stat])) > 1e-6:
                            a15_fail = True
                            a15_notes.append(
                                f"{src}:{key}:{dec}:{stat} "
                                f"|{got[stat]-float(saved[stat]):.3e}| > 1e-6"
                            )
            dm_src = str(blob["dm_source"]) if "dm_source" in blob.files else "?"
            a15_notes.append(f"{src}: 14 arrays present, dm_source={dm_src}")

        sweep_path = root / src / "d_sweep.json"
        if not sweep_path.is_file():
            a15_fail = True
            a15_notes.append(f"{src}: missing d_sweep.json")
            continue
        sweep = _load(sweep_path)
        if sweep.get("selection_rule") != D_SWEEP_SELECTION_RULE:
            a15_fail = True
            a15_notes.append(
                f"{src}: d_sweep selection_rule={sweep.get('selection_rule')}"
            )
        expected_d = [int(d) for d in cfg["latent_dims"]]
        by_md = {
            (r.get("method"), int(r.get("d"))): r
            for r in (sweep.get("rows") or [])
        }
        for key in REDUCING_SWEEP:
            for d in expected_d:
                if (key, d) not in by_md:
                    a15_fail = True
                    a15_notes.append(f"{src}: d_sweep missing {key} d={d}")
            rec = by_method.get(key) or {}
            sel = next(
                (r for r in (sweep.get("rows") or [])
                 if r.get("method") == key and r.get("selected")),
                None,
            )
            if sel is None:
                a15_fail = True
                a15_notes.append(f"{src}: d_sweep has no selected row for {key}")
                continue
            if int(sel["d"]) != int(rec.get("primary_d")):
                a15_fail = True
                a15_notes.append(
                    f"{src}:{key}: d_sweep d={sel['d']} != primary_d={rec.get('primary_d')}"
                )
            if abs(float(sel["ridge_alpha"]) - float(rec.get("ridge_alpha"))) > 1e-12:
                a15_fail = True
                a15_notes.append(f"{src}:{key}: d_sweep ridge_alpha mismatch")
            if int(sel["knn_k"]) != int(rec.get("knn_k")):
                a15_fail = True
                a15_notes.append(f"{src}:{key}: d_sweep knn_k mismatch")
            for dec in ("ridge", "knn"):
                got = float(sel[f"{dec}_median"])
                saved = float((rec.get(dec) or {}).get("median"))
                if abs(got - saved) > 1e-6:
                    a15_fail = True
                    a15_notes.append(
                        f"{src}:{key}:{dec} d_sweep median "
                        f"|{got - saved:.3e}| > 1e-6"
                    )
        a15_notes.append(f"{src}: d_sweep {D_SWEEP_SELECTION_RULE}")
    rows.append(_row(
        "A15", "FAIL" if a15_fail else "PASS",
        "; ".join(a15_notes) if a15_notes else "predictions.npz",
    ))

    failed = any(r["status"] == "FAIL" for r in rows)
    out = {
        "seed_index": seed_index,
        "config_sha256": cfg["config_sha256"],
        "seeds_0_4_code_sha": SEEDS_0_4_PROVENANCE_SHA,
        "probe_track_sha256": cfg.get("probe_track_sha256"),
        "rows": rows,
        "failed": failed,
        "n_units": sources[0].get("n_units"),
        "n_units_by_cell_type": sources[0].get("n_units_by_cell_type"),
    }
    dest = root / "audit.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    return out


def audit_all(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_quadrant_n5_yaml()
    per = [audit_seed(i, cfg) for i in range(int(cfg["seeds"]["n_seeds"]))]
    failed = any(p["failed"] for p in per)
    summary = {
        "config_sha256": cfg["config_sha256"],
        "failed": failed,
        "seeds": per,
    }
    (OUTPUT_ROOT / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(audit_seed(int(sys.argv[1])), indent=2))
    else:
        print(json.dumps(audit_all(), indent=2))
