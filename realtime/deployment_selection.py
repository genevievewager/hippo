"""Deployment decoder selection: sorted spikes only, target-specific windows.

Ground-truth spikes may be used as oracle diagnostics elsewhere, but deployable
models are selected exclusively from Neuropixels/Open Ephys/Kilosort-like
sorted spike comparisons.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from realtime.decoder_comparison import PRIMARY_METRIC
from realtime.timing import DEFAULT_UPDATE_DT_S

DEPLOYMENT_SPIKE_SOURCE = "sorted"
ORACLE_TAG = "oracle_non_deployable"
ORACLE_TAG_LABEL = "oracle / non-deployable"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if pd.isna(obj):
        return None
    return obj


def higher_is_better_for_target(target: str) -> bool:
    _, direction = PRIMARY_METRIC[target]
    return direction == "higher"


def build_all_window_scores_table(
    metrics_df: pd.DataFrame,
    *,
    best_df: pd.DataFrame | None = None,
    spike_source: str = DEPLOYMENT_SPIKE_SOURCE,
) -> pd.DataFrame:
    """Flatten comparison metrics into target × decoder × window scores."""
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()

    df = metrics_df.copy()
    if "spike_source" in df.columns:
        df = df[df["spike_source"].astype(str) == spike_source]
    if "target_name" in df.columns:
        df = df[df["target_name"].notna()]
    if df.empty:
        return pd.DataFrame()

    # Identify selected (target, decoder, window, feature) from best table
    selected_keys: set[tuple] = set()
    if best_df is not None and not best_df.empty:
        for _, row in best_df.iterrows():
            if str(row.get("spike_source", spike_source)) != spike_source:
                continue
            selected_keys.add((
                str(row.get("target_name")),
                str(row.get("recommended_realtime_decoder_name") or row.get("best_decoder_name")),
                round(float(row.get("recommended_realtime_window_s") or row.get("best_decode_window_s")), 6),
                str(row.get("best_feature_type") or ""),
            ))
            # Also mark absolute best-accuracy row
            selected_keys.add((
                str(row.get("target_name")),
                str(row.get("best_decoder_name")),
                round(float(row.get("best_decode_window_s")), 6),
                str(row.get("best_feature_type") or ""),
            ))

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        target = str(r.get("target_name"))
        if target not in PRIMARY_METRIC:
            continue
        metric_name, direction = PRIMARY_METRIC[target]
        if metric_name not in r or pd.isna(r.get(metric_name)):
            continue
        decoder = str(r.get("decoder_name"))
        feature = str(r.get("feature_type"))
        window = float(r.get("decode_window_s"))
        key = (target, decoder, round(window, 6), feature)
        # Prefer realtime-selected key match loosely: target+decoder+window
        selected = False
        for sk in selected_keys:
            if sk[0] == target and sk[1] == decoder and sk[2] == round(window, 6):
                # feature may differ when offline Isomap best vs realtime PCA
                if sk[3] == feature or sk[3] == "":
                    selected = True
                    break
        rows.append({
            "spike_source": spike_source,
            "target": target,
            "decoder": decoder,
            "feature_mode": feature,
            "causal_window_s": window,
            "metric_name": metric_name,
            "metric_value": float(r[metric_name]),
            "higher_is_better": direction == "higher",
            "train_samples": r.get("n_train_samples"),
            "test_samples": r.get("n_test_samples"),
            "selected_best": bool(selected),
            "realtime_compatible": r.get("realtime_compatible", True),
            "manifold_n_components": r.get("manifold_n_components"),
            "n_neighbors": r.get("n_neighbors"),
        })
    return pd.DataFrame(rows)


def warn_if_uniform_window(
    best_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    *,
    spike_source: str = DEPLOYMENT_SPIKE_SOURCE,
) -> str | None:
    """Warn when every target selected the same causal window."""
    if best_df is None or best_df.empty:
        return None
    df = best_df.copy()
    if "spike_source" in df.columns:
        df = df[df["spike_source"].astype(str) == spike_source]
    if df.empty:
        return None

    window_col = (
        "recommended_realtime_window_s"
        if "recommended_realtime_window_s" in df.columns
        else "best_decode_window_s"
    )
    windows = sorted({round(float(w), 6) for w in df[window_col].dropna().tolist()})
    tested = []
    if scores_df is not None and not scores_df.empty:
        tested = sorted({round(float(w), 6) for w in scores_df["causal_window_s"].dropna().tolist()})

    msg = None
    if len(windows) == 1:
        w = windows[0]
        msg = (
            f"All {spike_source}-spike targets selected the same causal window "
            f"({w:.3f} s). This may be valid, but check whether a "
            f"fallback/default window was applied."
        )
        if tested and len(tested) == 1:
            msg += (
                f" Only one window was tested in the comparison metrics "
                f"({tested[0]:.3f} s) — re-run with the full grid "
                f"[0.050, 0.100, 0.250, 0.500, 1.000] before trusting deployment."
            )
        else:
            msg += (
                " Inspect all_sorted_window_scores.csv to confirm this window "
                "actually won for each target."
            )
        print(f"  WARNING: {msg}")
        warnings.warn(msg, UserWarning, stacklevel=2)
    return msg


def build_best_realtime_decoders_payload(
    best_df: pd.DataFrame,
    *,
    comparison_dir: Path,
    input_dir: Path,
    spike_source: str = DEPLOYMENT_SPIKE_SOURCE,
    update_dt_s: float = DEFAULT_UPDATE_DT_S,
    seed: int = 42,
    selection_policy: str = "shortest_near_optimal",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build deployable-only JSON payload from sorted best_decoder_by_target."""
    comparison_dir = Path(comparison_dir)
    input_dir = Path(input_dir)
    df = best_df.copy()
    if "spike_source" in df.columns:
        df = df[df["spike_source"].astype(str) == spike_source]
    if df.empty:
        raise ValueError(
            f"No {spike_source} rows in best_decoder_by_target — cannot build "
            "deployable realtime decoder registry."
        )

    sorted_spikes = input_dir / "sorting" / "sorted_spikes.csv"
    if not sorted_spikes.exists():
        # Common alternate layout
        for alt in (
            input_dir / "sorted_spikes.csv",
            input_dir / "sorting" / "spikes.csv",
        ):
            if alt.exists():
                sorted_spikes = alt
                break

    targets: dict[str, Any] = {}
    for _, row in df.iterrows():
        target = str(row["target_name"])
        if selection_policy == "best_accuracy":
            decoder = str(row["best_decoder_name"])
            window = float(row["best_decode_window_s"])
            feature = str(row.get("best_feature_type", "counts"))
            model_path = row.get("best_window_model_path") or row.get("model_path")
            cfg_raw = row.get("decoder_config_json", "{}")
        else:
            decoder = str(
                row.get("recommended_realtime_decoder_name", row["best_decoder_name"])
            )
            window = float(row["recommended_realtime_window_s"])
            cfg_raw = row.get(
                "realtime_decoder_config_json", row.get("decoder_config_json", "{}")
            )
            model_path = row.get("realtime_model_path") or row.get("model_path")
            feature = str(row.get("best_feature_type", "counts"))
            try:
                cfg_probe = json.loads(cfg_raw) if isinstance(cfg_raw, str) else (cfg_raw or {})
            except json.JSONDecodeError:
                cfg_probe = {}
            if cfg_probe.get("feature_type"):
                feature = str(cfg_probe["feature_type"])

        try:
            cfg = json.loads(cfg_raw) if isinstance(cfg_raw, str) else (cfg_raw or {})
        except json.JSONDecodeError:
            cfg = {}

        transform = (
            cfg.get("manifold_transform_path")
            or row.get("manifold_transform_path")
        )
        # Never deploy offline-only classic Isomap; distilled Isomap is allowed
        # when comparison marked it realtime_compatible.
        if feature == "global_isomap" or str(cfg.get("manifold_type")) == "isomap":
            feature = "counts"
            transform = None
        if feature == "global_isomap_distilled":
            rt_flag = cfg.get("realtime_compatible")
            if rt_flag is False:
                feature = "counts"
                transform = None

        n_comp = cfg.get("manifold_n_components", row.get("best_manifold_n_components"))
        targets[target] = {
            "target": target,
            "selected_decoder": decoder,
            "selected_causal_window_s": window,
            "selected_feature_mode": feature,
            "selected_metric": row.get("primary_metric"),
            "metric_value": float(row["best_metric_value"])
            if selection_policy == "best_accuracy"
            else float(row.get("best_metric_value")),
            "selection_policy": selection_policy,
            "model_artifact_path": str(model_path) if model_path else None,
            "scaler_path": None,  # scalers live inside joblib pipelines
            "manifold_transform_path": str(transform) if transform else None,
            "manifold_n_components": (
                None if n_comp is None or (isinstance(n_comp, float) and pd.isna(n_comp))
                else int(n_comp)
            ),
            "spike_source": spike_source,
            "sorted_spike_file": str(sorted_spikes),
            "deployable": True,
            "oracle_non_deployable": False,
            "decoder_config": cfg,
        }

    payload = {
        "schema_version": 1,
        "deployable": True,
        "spike_source": spike_source,
        "deployment_rule": (
            "Models selected only from sorted (Neuropixels/Open Ephys/Kilosort-like) "
            "spike comparisons. Ground-truth spikes are never deployable."
        ),
        "run_id": run_id or Path(comparison_dir).parent.name,
        "training_run_id": run_id or Path(comparison_dir).parent.name,
        "comparison_dir": str(comparison_dir / spike_source),
        "input_dir": str(input_dir),
        "sorted_spike_file": str(sorted_spikes),
        "random_seed": int(seed),
        "update_rate_hz": float(1.0 / update_dt_s),
        "update_interval_s": float(update_dt_s),
        "selection_policy": selection_policy,
        "targets": targets,
    }
    return payload


def write_deployment_selection_artifacts(
    *,
    experiment_dir: Path,
    comparison_dir: Path,
    input_dir: Path,
    metrics_df: pd.DataFrame | None = None,
    best_df: pd.DataFrame | None = None,
    update_dt_s: float = DEFAULT_UPDATE_DT_S,
    seed: int = 42,
    selection_policy: str = "shortest_near_optimal",
) -> dict[str, Path]:
    """Write deployment_decoder_selection/ + models/best_realtime_decoders.json."""
    experiment_dir = Path(experiment_dir)
    comparison_dir = Path(comparison_dir)
    sorted_dir = comparison_dir / DEPLOYMENT_SPIKE_SOURCE

    if best_df is None:
        best_path = sorted_dir / "best_decoder_by_target.csv"
        if not best_path.exists():
            best_path = comparison_dir / "best_decoder_by_target.csv"
        if not best_path.exists():
            raise FileNotFoundError(
                f"Missing best_decoder_by_target.csv under {comparison_dir}"
            )
        best_df = pd.read_csv(best_path)

    if metrics_df is None:
        metrics_path = sorted_dir / "decoder_comparison_metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
        else:
            metrics_df = pd.DataFrame()

    # Filter to sorted only for deployable artifacts
    best_sorted = best_df
    if "spike_source" in best_df.columns:
        best_sorted = best_df[best_df["spike_source"].astype(str) == DEPLOYMENT_SPIKE_SOURCE].copy()
    if best_sorted.empty:
        # Single-source runs may omit spike_source filter match if column missing values
        best_sorted = best_df.copy()
        best_sorted["spike_source"] = DEPLOYMENT_SPIKE_SOURCE

    scores = build_all_window_scores_table(
        metrics_df, best_df=best_sorted, spike_source=DEPLOYMENT_SPIKE_SOURCE,
    )
    warning = warn_if_uniform_window(best_sorted, scores)

    out_dir = experiment_dir / "deployment_decoder_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = experiment_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    scores_path = out_dir / "all_sorted_window_scores.csv"
    if not scores.empty:
        scores.to_csv(scores_path, index=False)
        # Also under comparison/sorted for convenience
        scores.to_csv(sorted_dir / "all_window_scores_sorted.csv", index=False)

    payload = build_best_realtime_decoders_payload(
        best_sorted,
        comparison_dir=comparison_dir,
        input_dir=input_dir,
        spike_source=DEPLOYMENT_SPIKE_SOURCE,
        update_dt_s=update_dt_s,
        seed=seed,
        selection_policy=selection_policy,
        run_id=experiment_dir.name,
    )
    if warning:
        payload["uniform_window_warning"] = warning
        payload["n_unique_windows_tested"] = (
            int(scores["causal_window_s"].nunique()) if not scores.empty else 0
        )

    deploy_json = out_dir / "best_realtime_decoders.json"
    models_json = models_dir / "best_realtime_decoders.json"
    with open(deploy_json, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)
    with open(models_json, "w") as f:
        json.dump(_json_safe(payload), f, indent=2)

    # Mirror sorted best table into deployment dir (deployable only)
    best_sorted.to_csv(out_dir / "best_decoder_by_target_sorted.csv", index=False)

    # Mark any ground_truth comparison outputs as oracle if present
    gt_dir = comparison_dir / "ground_truth"
    if gt_dir.exists():
        meta = {
            "deployable": False,
            "tag": ORACLE_TAG,
            "label": ORACLE_TAG_LABEL,
            "note": (
                "Ground-truth spike decoder results are oracle / non-deployable "
                "simulation diagnostics only. Do not load these models for realtime."
            ),
        }
        with open(gt_dir / "ORACLE_NON_DEPLOYABLE.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"  Wrote deployment selection → {out_dir}")
    print(f"  Wrote deployable models registry → {models_json}")
    return {
        "deployment_dir": out_dir,
        "scores_csv": scores_path,
        "best_realtime_json": deploy_json,
        "models_best_realtime_json": models_json,
    }


def load_best_realtime_decoders(
    experiment_dir: Path | str,
) -> dict[str, Any]:
    """Load deployable sorted-spike decoder registry."""
    experiment_dir = Path(experiment_dir)
    candidates = [
        experiment_dir / "models" / "best_realtime_decoders.json",
        experiment_dir / "deployment_decoder_selection" / "best_realtime_decoders.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                payload = json.load(f)
            if not payload.get("deployable", True):
                raise ValueError(f"{path} is marked non-deployable")
            if payload.get("spike_source") != DEPLOYMENT_SPIKE_SOURCE:
                raise ValueError(
                    f"{path} spike_source={payload.get('spike_source')!r}; "
                    f"expected {DEPLOYMENT_SPIKE_SOURCE!r}"
                )
            return payload
    raise FileNotFoundError(
        "Could not find models/best_realtime_decoders.json. "
        "Run deployment decoder selection on sorted spikes first."
    )


def tag_ground_truth_outputs_as_oracle(comparison_dir: Path) -> None:
    """Write oracle markers under ground_truth comparison outputs."""
    gt_dir = Path(comparison_dir) / "ground_truth"
    if not gt_dir.exists():
        return
    meta = {
        "deployable": False,
        "tag": ORACLE_TAG,
        "label": ORACLE_TAG_LABEL,
    }
    with open(gt_dir / "ORACLE_NON_DEPLOYABLE.json", "w") as f:
        json.dump(meta, f, indent=2)
    best = gt_dir / "best_decoder_by_target.csv"
    if best.exists():
        df = pd.read_csv(best)
        df["deployable"] = False
        df["selection_role"] = ORACLE_TAG_LABEL
        df.to_csv(best, index=False)
