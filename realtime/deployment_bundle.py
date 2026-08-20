"""Deployment bundle: serialize / load a single-target deployable pipeline.

Bundle layout::

    deployment_bundle/
        config.json          # F × E × D × W × C + scores
        metadata.json        # schema, timestamps, training origins
        unit_order.json      # expected unit_ids
        feature_config.json  # feature_set / embedding notes
        decoder.joblib       # fitted decoder pipeline (includes scaler)
        embedding/           # optional fitted manifold / E transform dir
        neural_extractor/    # optional neural feature extractor dir

Omit components that are not needed (identity embedding → no embedding/).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from realtime.decoder_comparison import PRIMARY_METRIC
from realtime.deployment_selection import (
    DEPLOYMENT_SPIKE_SOURCE,
    _composite_feature_mode,
    load_best_realtime_decoders,
)
from realtime.live.config import DeployableConfiguration
from realtime.manifold_features import (
    is_realtime_compatible_feature_mode,
    load_feature_transformer,
)
from realtime.search_space import resolve_manifold_alias

SCHEMA_VERSION = 1
BUNDLE_DIRNAME = "deployment_bundles"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(p: str | Path | None, *, base: Path | None = None) -> Path | None:
    if p is None or str(p) in ("", "None", "nan"):
        return None
    path = Path(p)
    if path.is_absolute() and path.exists():
        return path
    candidates: list[Path] = []
    if base is not None:
        candidates.append(base / path)
        candidates.append(base.parent / path)
    candidates.append(Path.cwd() / path)
    # Repo-root-relative (outputs/...)
    here = Path(__file__).resolve().parent.parent
    candidates.append(here / path)
    for c in candidates:
        if c.exists():
            return c
    return path if path.exists() else (candidates[0] if candidates else path)


def _resolve_embedding_type(feature_mode: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or {}
    emb = cfg.get("embedding_type") or cfg.get("manifold_type")
    if emb and str(emb) not in ("", "none", "null"):
        if str(emb) == "none":
            return "identity"
        return resolve_manifold_alias(str(emb))
    mode = str(feature_mode or "counts")
    # Composite modes like global_pca / region_pca / counts
    if mode in ("counts", "rates", "sqrt_counts", "identity"):
        return "identity"
    return resolve_manifold_alias(mode)


def best_deployable(
    experiment_dir: Path | str,
    target: str,
    *,
    spike_source: str = DEPLOYMENT_SPIKE_SOURCE,
    selection_policy: str = "shortest_near_optimal",
    deployable_only: bool = True,
) -> DeployableConfiguration:
    """Select the best **deployable** F×E×D×W(+C) configuration for one target.

    Preference order:
    1. ``models/best_realtime_decoders.json`` registry entry for ``target``
    2. Fallback: ``best_decoder_by_target.csv`` with deployability filters

    Metric orientation comes from ``PRIMARY_METRIC`` (never arbitrary maximize).
    """
    experiment_dir = Path(experiment_dir)
    if spike_source != DEPLOYMENT_SPIKE_SOURCE and deployable_only:
        raise ValueError(
            f"deployable_only requires spike_source={DEPLOYMENT_SPIKE_SOURCE!r}, "
            f"got {spike_source!r}"
        )

    # Prefer public registry when present.
    try:
        payload = load_best_realtime_decoders(experiment_dir)
        targets = payload.get("targets") or {}
        if target not in targets:
            available = sorted(targets)
            raise KeyError(
                f"Target {target!r} not in best_realtime_decoders.json. "
                f"Available: {available}"
            )
        entry = targets[target]
        feature_mode = str(entry.get("selected_feature_mode") or "counts")
        if deployable_only and not is_realtime_compatible_feature_mode(feature_mode):
            raise ValueError(
                f"Registry entry for {target!r} has non-deployable mode {feature_mode!r}"
            )
        cfg = entry.get("decoder_config") or {}
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except json.JSONDecodeError:
                cfg = {}
        emb = _resolve_embedding_type(feature_mode, cfg if isinstance(cfg, dict) else {})
        metric_name, direction = PRIMARY_METRIC.get(target, (entry.get("selected_metric"), "higher"))
        comparison_dir = payload.get("comparison_dir")
        return DeployableConfiguration(
            target=target,
            feature_set=str(cfg.get("feature_set") or feature_mode),
            embedding_type=emb,
            decoder_name=str(entry.get("selected_decoder")),
            decode_window_s=float(entry.get("selected_causal_window_s")),
            extras={
                "decoder_config": cfg,
                "registry_path": "models/best_realtime_decoders.json",
                "feature_mode": feature_mode,
                "sorted_spike_file": entry.get("sorted_spike_file") or payload.get("sorted_spike_file"),
            },
            model_path=entry.get("model_artifact_path"),
            manifold_transform_path=entry.get("manifold_transform_path"),
            comparison_dir=str(comparison_dir) if comparison_dir else None,
            experiment_dir=str(experiment_dir),
            metric_name=str(metric_name) if metric_name else None,
            metric_value=(
                float(entry["metric_value"])
                if entry.get("metric_value") is not None
                else None
            ),
            metric_direction=direction,
            selection_policy=str(entry.get("selection_policy") or selection_policy),
            spike_source=str(payload.get("spike_source") or spike_source),
            realtime_compatible=bool(entry.get("realtime_compatible", True)),
            deployable=bool(entry.get("deployable", True)),
            remapped_from_offline=bool(entry.get("remapped_from_offline_isomap", False)),
            update_dt_s=float(payload.get("update_interval_s") or 0.025),
            manifold_n_components=(
                int(entry["manifold_n_components"])
                if entry.get("manifold_n_components") is not None
                else None
            ),
            training_run_id=str(payload.get("training_run_id") or payload.get("run_id") or experiment_dir.name),
        )
    except FileNotFoundError:
        pass

    # Fallback: best_decoder_by_target from a comparison tree.
    from realtime.best_decoder_selection import select_best_decoder_row

    comparison_dir = experiment_dir / "decoder_comparison"
    if not comparison_dir.exists():
        # Some runs write under decoder_comparison/sorted at top-level experiment.
        raise FileNotFoundError(
            f"No best_realtime_decoders.json or decoder_comparison under {experiment_dir}. "
            "Run Decoder Benchmark (sorted spikes) first."
        )
    row, _ = select_best_decoder_row(
        comparison_dir, spike_source, target, selection_policy=selection_policy,
    )
    feature_mode = _composite_feature_mode(row)
    if deployable_only and not is_realtime_compatible_feature_mode(str(feature_mode)):
        # Fall back already happened inside registry writer; here refuse silently substitute.
        raise ValueError(
            f"Best row for {target!r} uses non-deployable mode {feature_mode!r}. "
            "Re-run comparison / write_deployment_selection_artifacts so a deployable "
            "recommendation exists."
        )
    cfg = row.get("decoder_config") or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except json.JSONDecodeError:
            cfg = {}
    emb = _resolve_embedding_type(str(feature_mode), cfg if isinstance(cfg, dict) else {})
    metric_name, direction = PRIMARY_METRIC.get(target, (row.get("primary_metric"), "higher"))
    return DeployableConfiguration(
        target=target,
        feature_set=str((cfg or {}).get("feature_set") or feature_mode),
        embedding_type=emb,
        decoder_name=str(row.get("selected_decoder_name") or row.get("best_decoder_name")),
        decode_window_s=float(row.get("selected_decode_window_s") or row.get("best_decode_window_s")),
        extras={
            "decoder_config": cfg,
            "feature_mode": feature_mode,
            "from_file": row.get("from_file"),
        },
        model_path=str(row.get("selected_model_path") or "") or None,
        manifold_transform_path=(
            str(row["selected_manifold_transform_path"])
            if row.get("selected_manifold_transform_path")
            else None
        ),
        comparison_dir=str(row.get("comparison_models_dir") or comparison_dir),
        experiment_dir=str(experiment_dir),
        metric_name=str(metric_name) if metric_name else None,
        metric_value=(
            float(row["best_metric_value"])
            if row.get("best_metric_value") is not None
            else None
        ),
        metric_direction=direction,
        selection_policy=selection_policy,
        spike_source=spike_source,
        realtime_compatible=True,
        deployable=True,
        update_dt_s=0.025,
        manifold_n_components=(
            int(row["selected_manifold_n_components"])
            if row.get("selected_manifold_n_components") is not None
            and str(row.get("selected_manifold_n_components")) != "nan"
            else None
        ),
        training_run_id=experiment_dir.name,
    )


@dataclass
class DeploymentBundle:
    """In-memory handle to a loaded deployment bundle."""

    path: Path
    config: DeployableConfiguration
    metadata: dict[str, Any]
    unit_ids: list[int]
    decoder: Any
    embedding: Any | None = None
    neural_extractor: Any | None = None
    feature_config: dict[str, Any] | None = None

    @property
    def target(self) -> str:
        return self.config.target

    @property
    def decode_window_s(self) -> float:
        return float(self.config.decode_window_s)

    @property
    def update_dt_s(self) -> float:
        return float(self.config.update_dt_s)

    def is_simulation_trained(self) -> bool:
        return bool(self.metadata.get("simulation_trained", True))


def pack_deployment_bundle(
    experiment_dir: Path | str,
    target: str,
    *,
    output_dir: Path | str | None = None,
    selection_policy: str = "shortest_near_optimal",
) -> Path:
    """Build a deployment bundle for ``target`` under the experiment tree."""
    experiment_dir = Path(experiment_dir)
    cfg = best_deployable(
        experiment_dir, target, selection_policy=selection_policy, deployable_only=True,
    )
    if output_dir is None:
        output_dir = (
            experiment_dir / BUNDLE_DIRNAME / f"{target}__{cfg.D}__w{int(round(cfg.W * 1000)):04d}ms"
        )
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    from realtime.data_loading import load_simulation_data

    data = load_simulation_data(experiment_dir, cfg.spike_source)
    unit_ids = [int(u) for u in data["unit_ids"]]

    # Decoder artifact
    model_src = _resolve_path(cfg.model_path, base=experiment_dir)
    if model_src is None or not model_src.exists():
        # Try resolve from comparison models dir.
        from realtime.best_decoder_selection import (
            load_windowed_model,
            resolve_comparison_models_dir,
        )

        comparison_dir = Path(cfg.comparison_dir) if cfg.comparison_dir else experiment_dir / "decoder_comparison"
        try:
            models_dir = resolve_comparison_models_dir(
                comparison_dir.parent if comparison_dir.name == "models" else comparison_dir,
                cfg.spike_source,
            )
        except FileNotFoundError:
            models_dir = Path(cfg.comparison_dir) if cfg.comparison_dir else None
        if models_dir is None or not models_dir.exists():
            raise FileNotFoundError(
                f"Missing decoder artifact for target={target!r}. "
                "Run Decoder Benchmark so windowed models exist."
            )
        decoder = load_windowed_model(
            models_dir,
            target,
            cfg.decode_window_s,
            feature_type=str(cfg.extras.get("feature_mode") or cfg.feature_set),
            n_components=cfg.manifold_n_components,
        )
    else:
        decoder = joblib.load(model_src)
    joblib.dump(decoder, out / "decoder.joblib")

    # Embedding / E
    emb = None
    if cfg.manifold_transform_path:
        tpath = _resolve_path(cfg.manifold_transform_path, base=experiment_dir)
        if tpath is not None and tpath.exists():
            dest = out / "embedding"
            if tpath.is_dir():
                shutil.copytree(tpath, dest)
            else:
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tpath, dest / tpath.name)
            try:
                emb = load_feature_transformer(dest if dest.is_dir() else tpath)
            except Exception:
                emb = load_feature_transformer(tpath)

    # Metadata
    meta = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_iso(),
        "target": target,
        "simulation_trained": True,
        "pipeline_test_only_on_real_data": True,
        "training_spike_source": cfg.spike_source,
        "training_run_id": cfg.training_run_id,
        "experiment_dir": str(experiment_dir),
        "n_units": len(unit_ids),
        "expected_feature_dim": None,
        "software": {"deployment_bundle_schema": SCHEMA_VERSION},
    }
    # Probe feature dim via a tiny causal count → transform when possible.
    try:
        from realtime.spike_binner import count_spikes_in_window

        t_probe = float(data["spikes_df"]["time"].iloc[min(100, len(data["spikes_df"]) - 1)])
        counts = count_spikes_in_window(
            data["spikes_df"], unit_ids, t_probe - cfg.W, t_probe,
        ).reshape(1, -1)
        if emb is not None:
            feats = np.asarray(emb.transform(counts))
        else:
            feats = counts
        meta["expected_feature_dim"] = int(feats.shape[1])
    except Exception as exc:  # noqa: BLE001
        meta["feature_dim_probe_error"] = str(exc)

    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str) + "\n")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (out / "unit_order.json").write_text(
        json.dumps({"unit_ids": unit_ids, "n_units": len(unit_ids)}, indent=2) + "\n"
    )
    (out / "feature_config.json").write_text(
        json.dumps(
            {
                "feature_set": cfg.feature_set,
                "feature_mode": cfg.extras.get("feature_mode"),
                "embedding_type": cfg.embedding_type,
                "decode_window_s": cfg.decode_window_s,
                "update_dt_s": cfg.update_dt_s,
                "manifold_n_components": cfg.manifold_n_components,
            },
            indent=2,
        )
        + "\n"
    )
    return out


def load_deployment_bundle(bundle_dir: Path | str) -> DeploymentBundle:
    """Load a previously packed deployment bundle."""
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Deployment bundle not found: {bundle_dir}")
    config = DeployableConfiguration.from_dict(
        json.loads((bundle_dir / "config.json").read_text())
    )
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    unit_payload = json.loads((bundle_dir / "unit_order.json").read_text())
    unit_ids = [int(u) for u in unit_payload["unit_ids"]]
    feature_config = None
    fc = bundle_dir / "feature_config.json"
    if fc.exists():
        feature_config = json.loads(fc.read_text())
    decoder = joblib.load(bundle_dir / "decoder.joblib")
    embedding = None
    emb_dir = bundle_dir / "embedding"
    if emb_dir.exists():
        embedding = load_feature_transformer(emb_dir)
    neural = None
    ne_dir = bundle_dir / "neural_extractor"
    if ne_dir.exists():
        try:
            from realtime.neural_features import load_neural_feature_extractor

            neural = load_neural_feature_extractor(ne_dir)
        except Exception:
            neural = None
    return DeploymentBundle(
        path=bundle_dir,
        config=config,
        metadata=metadata,
        unit_ids=unit_ids,
        decoder=decoder,
        embedding=embedding,
        neural_extractor=neural,
        feature_config=feature_config,
    )
