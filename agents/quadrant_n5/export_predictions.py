"""Regenerate primary-d test predictions for the frozen n=5 quadrant run.

Stage 1 (--check-only) writes nothing under outputs/. Stage 2 writes
predictions.npz after every cell passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.dynamic_latents.lds import assert_readout_latents_are_filtered
from realtime.manifold_features import load_feature_transformer
from realtime.pipeline_artifacts import hash_train_indices
from realtime.quadrant_n5 import SEEDS_0_4_PROVENANCE_SHA, load_quadrant_n5_yaml
from realtime.quadrant_n5_run import (
    METHOD_KEYS,
    OUTPUT_ROOT,
    _euclid,
    _fit_predict_knn,
    _fit_predict_ridge,
    _make_rep,
    _position,
    _sqrt_zscore_train,
    _valid_mask,
    fit_transform_representation,
)
from realtime.spike_binner import build_causal_spike_matrix
from realtime.timing import extract_behavior_times
from realtime.train_decoder import align_behavior_to_decoder_times, causal_train_test_split

EVAL_HASH = "aecdcfd66f49eebe"
TOL_CM = 1e-6


def export_code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_observation(
    cfg: dict[str, Any],
    sim_dir: Path,
    spike_source: str,
) -> dict[str, Any]:
    feat = cfg["features"]
    split = cfg["split"]
    data = load_simulation_data(
        sim_dir, spike_source,
        include_regions=list(cfg["unit_inclusion"]["regions"]),
    )
    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = float(feat["update_dt"])
    W = float(feat["window_s"])
    decode_times = make_decode_times(
        data["session_duration"], W, update_dt, behavior_times=behavior_times,
    )
    X_counts = build_causal_spike_matrix(
        data["spikes_df"], data["unit_ids"], decode_times, W,
    )
    beh = align_behavior_to_decoder_times(data["behavior_df"], decode_times)
    y = _position(beh)
    train_mask, test_mask = causal_train_test_split(
        decode_times, float(split["train_frac"]), gap_s=float(split["gap_s"]),
    )
    X, _ = _sqrt_zscore_train(X_counts, train_mask)
    n = len(decode_times)
    valid = {m: _valid_mask(m, n, cfg) for m in METHOD_KEYS}
    eval_mask = test_mask.copy()
    train_ok = train_mask.copy()
    for m in METHOD_KEYS:
        eval_mask &= valid[m]
        train_ok &= valid[m]
    hashes = {
        "train": hash_train_indices(train_ok),
        "test": hash_train_indices(test_mask),
        "eval": hash_train_indices(eval_mask),
    }
    return {
        "X": X,
        "y": y,
        "decode_times": decode_times,
        "train_ok": train_ok,
        "eval_mask": eval_mask,
        "index_hashes": hashes,
    }


def transform_loaded(name: str, model: Any, X: np.ndarray) -> np.ndarray:
    """Match Phase 3 transform after a saved model is loaded (no refit)."""
    # Phase 3 also passes the z-scored session as float64 into transform.
    X = np.asarray(X, dtype=float)
    if name == "lds":
        Z = model.transform(X, causal=True, reset=True)
        inner = getattr(model, "model_", model)
        assert_readout_latents_are_filtered(inner, Z, X)
        return Z
    if name == "gpfa":
        return model.transform(X)
    return model.transform(X)


def _refit_latents(
    name: str,
    cfg: dict[str, Any],
    obs: dict[str, Any],
    rec: dict[str, Any],
    methods_seed: int,
) -> np.ndarray:
    reducing = name not in ("raw", "raw_lag")
    primary_d = rec.get("primary_d")
    nested = bool((cfg["representations"].get(name) or {}).get("nested"))
    d_final = int(primary_d) if reducing else 3
    d_fit = int(cfg["nested_fit_d"]) if (nested and reducing) else d_final
    model = _make_rep(name, d_fit, cfg, methods_seed, n_fit=int(obs["train_ok"].sum()))
    return fit_transform_representation(name, model, obs["X"], obs["train_ok"])


def latents_for_method(
    name: str,
    cfg: dict[str, Any],
    obs: dict[str, Any],
    rec: dict[str, Any],
    model_dir: Path,
    methods_seed: int,
    *,
    force_refit: bool = False,
) -> tuple[np.ndarray, str]:
    reducing = name not in ("raw", "raw_lag")
    primary_d = rec.get("primary_d")
    nested = bool((cfg["representations"].get(name) or {}).get("nested"))
    if force_refit:
        Z = _refit_latents(name, cfg, obs, rec, methods_seed)
        route = "refit"
    else:
        route = "reloaded"
        try:
            if not (model_dir / "meta.json").is_file():
                raise FileNotFoundError(f"missing {model_dir / 'meta.json'}")
            model = load_feature_transformer(model_dir)
            Z = transform_loaded(name, model, obs["X"])
        except Exception as exc:
            route = f"fallback:{type(exc).__name__}:{exc}"
            Z = _refit_latents(name, cfg, obs, rec, methods_seed)
    if nested and reducing and primary_d is not None:
        Z = Z[:, : int(primary_d)]
    # Native dtype into the decoders (Phase 3). Do not cast float32 → float64.
    return Z, route


def predict_both(
    Z: np.ndarray,
    y: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
    ridge_alpha: float,
    knn_k: int,
) -> dict[str, np.ndarray]:
    Ztr, Zte = Z[train_ok], Z[eval_mask]
    ytr = y[train_ok]
    return {
        "ridge": _fit_predict_ridge(Ztr, ytr, Zte, float(ridge_alpha)),
        "knn": _fit_predict_knn(Ztr, ytr, Zte, int(knn_k)),
    }


def _stat_diffs(saved: dict[str, float], recomputed: dict[str, float]) -> dict[str, float]:
    return {
        key: abs(float(recomputed[key]) - float(saved[key]))
        for key in ("median", "mean", "p90")
    }


def reconstruct_source(
    cfg: dict[str, Any],
    seed_index: int,
    spike_source: str,
    results_root: Path,
    *,
    methods: tuple[str, ...] = METHOD_KEYS,
    force_refit: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]], dict[str, str]]:
    src_dir = results_root / f"seed_{seed_index}" / spike_source
    summary = json.loads((src_dir / "source_summary.json").read_text())
    streams = summary["seed_streams"]
    obs = build_observation(cfg, results_root / f"seed_{seed_index}" / "sim", spike_source)
    saved_eval = (summary.get("index_hashes") or {}).get("eval")
    if obs["index_hashes"]["eval"] != saved_eval:
        raise RuntimeError(
            f"seed {seed_index} {spike_source} rebuilt eval hash "
            f"{obs['index_hashes']['eval']} != saved {saved_eval}"
        )
    if results_root.resolve() == OUTPUT_ROOT.resolve() and saved_eval != EVAL_HASH:
        raise RuntimeError(
            f"frozen eval hash {saved_eval} != {EVAL_HASH}"
        )
    pred_arrays: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    rows = []
    yte = obs["y"][obs["eval_mask"]]
    for key in methods:
        rec = json.loads((src_dir / f"{key}.json").read_text())
        model_dir = results_root / "models" / f"seed_{seed_index}" / spike_source / key
        Z, route = latents_for_method(
            key, cfg, obs, rec, model_dir, int(streams["methods"]),
            force_refit=force_refit,
        )
        sources[key] = "refit" if route == "refit" or route.startswith("fallback") else "reloaded"
        preds = predict_both(
            Z, obs["y"], obs["train_ok"], obs["eval_mask"],
            rec["ridge_alpha"], rec["knn_k"],
        )
        pred_arrays[f"pred_{key}_ridge"] = preds["ridge"]
        pred_arrays[f"pred_{key}_knn"] = preds["knn"]
        for dec in ("ridge", "knn"):
            got = _euclid(preds[dec], yte)
            diffs = _stat_diffs(rec[dec], got)
            ok = all(d <= TOL_CM for d in diffs.values())
            rows.append({
                "seed": seed_index,
                "source": spike_source,
                "method": key,
                "decoder": dec,
                "saved_median": rec[dec]["median"],
                "recomputed_median": got["median"],
                "diff_median": diffs["median"],
                "saved_mean": rec[dec]["mean"],
                "recomputed_mean": got["mean"],
                "diff_mean": diffs["mean"],
                "saved_p90": rec[dec]["p90"],
                "recomputed_p90": got["p90"],
                "diff_p90": diffs["p90"],
                "pass": ok,
                "route": route,
            })
    return obs, pred_arrays, rows, sources


def check_source(
    cfg: dict[str, Any],
    seed_index: int,
    spike_source: str,
    results_root: Path,
    *,
    methods: tuple[str, ...] = METHOD_KEYS,
    force_refit: bool = False,
) -> list[dict[str, Any]]:
    _obs, _preds, rows, _src = reconstruct_source(
        cfg, seed_index, spike_source, results_root, methods=methods,
        force_refit=force_refit,
    )
    return rows


def write_predictions_npz(
    path: Path,
    cfg: dict[str, Any],
    obs: dict[str, Any],
    preds: dict[str, np.ndarray],
    *,
    method_source: dict[str, str] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "t_s": np.asarray(obs["decode_times"][obs["eval_mask"]], dtype=float),
        "y_true": np.asarray(obs["y"][obs["eval_mask"]], dtype=float),
        "eval_index_hash": np.asarray(obs["index_hashes"]["eval"]),
        "config_sha256": np.asarray(cfg["config_sha256"]),
        "seeds_code_sha": np.asarray(SEEDS_0_4_PROVENANCE_SHA),
        "export_code_sha": np.asarray(export_code_sha()),
    }
    src = dict(method_source or {})
    payload["dm_source"] = np.asarray(src.get("dm", "reloaded"))
    for key in METHOD_KEYS:
        payload[f"source_{key}"] = np.asarray(src.get(key, "reloaded"))
    if src.get("dm_reload_matches_refit") is not None:
        payload["dm_reload_matches_refit"] = np.asarray(
            bool(src["dm_reload_matches_refit"])
        )
        payload["dm_reload_note"] = np.asarray(str(src.get("dm_reload_note") or ""))
    payload.update(preds)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def annotate_existing_npz(
    path: Path,
    *,
    dm_reload_matches_refit: bool,
    note: str,
) -> None:
    """Add metadata only; prediction arrays stay byte-for-byte the same values."""
    blob = np.load(path)
    payload = {k: blob[k] for k in blob.files}
    payload["dm_reload_matches_refit"] = np.asarray(bool(dm_reload_matches_refit))
    payload["dm_reload_note"] = np.asarray(note)
    np.savez_compressed(path, **payload)


def parse_seeds(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument(
        "--methods",
        default=",".join(METHOD_KEYS),
        help="Comma-separated method keys (default: all 7)",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Stage 1: print the table and write nothing under --results",
    )
    args = ap.parse_args()
    cfg = load_quadrant_n5_yaml()
    if cfg["config_sha256"] != (
        "4da17a5dad704866a1078231fb0795a0a1505211328946967c9c58ebaa9b1ee3"
    ):
        raise SystemExit(f"config hash moved: {cfg['config_sha256']}")
    methods = tuple(m.strip() for m in str(args.methods).split(",") if m.strip())
    rows: list[dict[str, Any]] = []
    for seed in parse_seeds(args.seeds):
        for src in ("sorted", "ground_truth"):
            print(f"# seed {seed} {src}", flush=True)
            obs, pred_arrays, src_rows, method_source = reconstruct_source(
                cfg, seed, src, args.results, methods=methods,
            )
            rows.extend(src_rows)
            n_fail = sum(1 for r in src_rows if not r["pass"])
            print(f"# seed {seed} {src} fail_this_source={n_fail}", flush=True)
            if (not args.check_only) and methods == METHOD_KEYS and n_fail == 0:
                dest = args.results / f"seed_{seed}" / src / "predictions.npz"
                write_predictions_npz(
                    dest, cfg, obs, pred_arrays, method_source=method_source,
                )
                print(f"# wrote {dest}", flush=True)
    print(
        "seed\tsource\tmethod\tdecoder\tsaved_med\trecomp_med\t|d_med|\t"
        "saved_mean\trecomp_mean\t|d_mean|\tsaved_p90\trecomp_p90\t|d_p90|\tpass\troute"
    )
    for r in rows:
        print(
            f"{r['seed']}\t{r['source']}\t{r['method']}\t{r['decoder']}\t"
            f"{r['saved_median']:.9f}\t{r['recomputed_median']:.9f}\t{r['diff_median']:.3e}\t"
            f"{r['saved_mean']:.9f}\t{r['recomputed_mean']:.9f}\t{r['diff_mean']:.3e}\t"
            f"{r['saved_p90']:.9f}\t{r['recomputed_p90']:.9f}\t{r['diff_p90']:.3e}\t"
            f"{'PASS' if r['pass'] else 'FAIL'}\t{r['route']}"
        )
    n_pass = sum(1 for r in rows if r["pass"])
    print(f"SUMMARY {n_pass}/{len(rows)}", flush=True)
    fallbacks = [r for r in rows if not str(r["route"]).startswith("load")]
    if fallbacks:
        print(f"FALLBACK_CELLS {len(fallbacks)}", flush=True)
    if args.check_only:
        return 0 if n_pass == len(rows) else 1
    if n_pass != len(rows):
        print("Stage 1 failed; not writing predictions.npz", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
