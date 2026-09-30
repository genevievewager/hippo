"""Controlled four-quadrant representation comparison.

Consumes one shared neural observation (FeatureDataset / ObservationConfig)
and trains the same decoder family on each selected representation. This is
not four end-to-end pipelines.

Window W is inherited from the observation. A labeled window sweep runs a
*separate* fair comparison per W rather than mixing windows in one table.

Computation is delegated to ``run_decoder_comparison`` with
``reuse_transforms=True`` so existing F/E caches are reused. Changing only
the decoder does not change the observation hash or representation fit key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from realtime.decoder_comparison import (
    PRIMARY_METRIC,
    ComparisonRunConfig,
    run_decoder_comparison,
)
from realtime.decoder_models import TARGET_FAMILY, resolve_model_names_for_target
from realtime.pipeline_artifacts import ObservationConfig, windows_close
from realtime.pipeline_invariants import PipelineInvariantError
from realtime.representation_registry import (
    CANONICAL_QUADRANTS,
    DEFAULT_QUADRANT_METHODS,
    QUADRANT_DISPLAY_LABELS,
    QUADRANT_NONLINEAR_DYNAMIC,
    QUADRANT_UNKNOWN,
    annotate_metrics_rows,
    get_spec,
    infer_quadrant,
    representation_fit_dependencies,
    resolve_representation_name,
)
from realtime.search_space import resolve_manifold_alias

ProgressCallback = Callable[..., None]


class QuadrantComparisonError(PipelineInvariantError):
    """Raised when a controlled quadrant comparison would be unfair or invalid."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_decoder_for_target(target: str) -> str:
    family = TARGET_FAMILY.get(str(target), "continuous")
    if family == "categorical":
        return "logistic_regression"
    return "ridge"


def default_quadrant_methods() -> dict[str, str | None]:
    return dict(DEFAULT_QUADRANT_METHODS)


def quadrant_output_dir(experiment_dir: Path, spike_source: str = "sorted") -> Path:
    return Path(experiment_dir) / "quadrant_comparison" / str(spike_source)


def summary_path(output_dir: Path) -> Path:
    return Path(output_dir) / "quadrant_summary.json"


def metrics_path(output_dir: Path) -> Path:
    return Path(output_dir) / "decoder_comparison_metrics.csv"


@dataclass
class QuadrantExperimentConfig:
    """One controlled comparison: shared observation → N representation branches."""

    input_dir: Path
    output_dir: Path
    observation: ObservationConfig
    target: str = "position"
    decoder_name: str = "ridge"
    methods: dict[str, str | None] = field(default_factory=default_quadrant_methods)
    extra_methods: tuple[str, ...] = ()
    n_components: int = 3
    train_frac: float = 0.70
    split_gap_s: float = 0.0
    seed: int = 42
    reuse_transforms: bool = True
    reuse_search_roots: tuple[str, ...] = ()
    enable_trigger_search: bool = False
    # Extra windows for a *labeled* timescale benchmark. Each W is its own run.
    window_sweep_s: tuple[float, ...] = ()

    def selected_embeddings(self) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for qid in CANONICAL_QUADRANTS:
            raw = self.methods.get(qid)
            if not raw:
                continue
            emb = resolve_manifold_alias(str(raw))
            if emb in seen:
                continue
            seen.add(emb)
            names.append(emb)
        for raw in self.extra_methods:
            if not raw:
                continue
            spec = get_spec(str(raw))
            if not spec.implemented:
                continue
            emb = resolve_manifold_alias(str(raw))
            if emb in seen:
                continue
            seen.add(emb)
            names.append(emb)
        return tuple(names)

    def observation_hash(self) -> str:
        return self.observation.hash()

    def representation_keys(self) -> list[dict[str, Any]]:
        obs_hash = self.observation_hash()
        return [
            representation_fit_dependencies(
                observation_hash=obs_hash,
                embedding_type=emb,
                n_components=self.n_components,
                seed=self.seed,
                train_frac=self.train_frac,
            )
            for emb in self.selected_embeddings()
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "observation": self.observation.to_dict(),
            "observation_hash": self.observation.hash(),
            "target": self.target,
            "decoder_name": self.decoder_name,
            "methods": dict(self.methods),
            "extra_methods": list(self.extra_methods),
            "n_components": int(self.n_components),
            "train_frac": float(self.train_frac),
            "split_gap_s": float(self.split_gap_s),
            "seed": int(self.seed),
            "reuse_transforms": bool(self.reuse_transforms),
            "reuse_search_roots": list(self.reuse_search_roots),
            "enable_trigger_search": bool(self.enable_trigger_search),
            "window_sweep_s": list(self.window_sweep_s),
            "selected_embeddings": list(self.selected_embeddings()),
            "scientific_hash": self.scientific_hash(),
        }

    def scientific_hash(self) -> str:
        """Identity of this controlled comparison (shared observation + E set + D)."""
        from realtime.pipeline_artifacts import config_hash

        return config_hash({
            "observation_hash": self.observation.hash(),
            "target": self.target,
            "decoder_name": self.decoder_name,
            "methods": {qid: self.methods.get(qid) for qid in CANONICAL_QUADRANTS},
            "extra_methods": list(self.extra_methods),
            "n_components": int(self.n_components),
            "train_frac": float(self.train_frac),
            "split_gap_s": float(self.split_gap_s),
            "seed": int(self.seed),
            "window_sweep_s": [float(w) for w in self.window_sweep_s],
            "causal": True,
        })


def _window_mismatch_message(result_window_s: float, active_window_s: float) -> str:
    return (
        "Quadrant comparison requires a shared neural observation and "
        "train/test partition. The selected result was generated from a "
        f"different {float(result_window_s) * 1000:.0f} ms observation while "
        f"the active experiment uses {float(active_window_s) * 1000:.0f} ms."
    )


def validate_controlled_quadrant_comparison(
    config: QuadrantExperimentConfig,
    *,
    allow_unimplemented_empty: bool = True,
) -> None:
    """Fail loudly if the comparison would mix incompatible observations."""
    obs = config.observation
    if float(obs.window_s) <= 0 or float(obs.update_dt) <= 0:
        raise QuadrantComparisonError(
            f"Invalid observation timing W={obs.window_s}s update_dt={obs.update_dt}s."
        )
    if not str(obs.source_spikes or "").strip():
        raise QuadrantComparisonError("Spike source must be explicit on the observation.")
    if not str(obs.feature_set or "").strip():
        raise QuadrantComparisonError("Feature set must be explicit on the observation.")

    if config.window_sweep_s:
        mixed = {float(obs.window_s), *[float(w) for w in config.window_sweep_s]}
        if len(mixed) > 1:
            # Allowed as a labeled sweep of *separate* experiments, not one mixed table.
            pass

    selected: dict[str, str] = {}
    for qid in CANONICAL_QUADRANTS:
        raw = config.methods.get(qid)
        if raw is None or str(raw).strip() == "":
            if qid == QUADRANT_NONLINEAR_DYNAMIC and allow_unimplemented_empty:
                continue
            if qid == QUADRANT_NONLINEAR_DYNAMIC:
                continue
            raise QuadrantComparisonError(
                f"{QUADRANT_DISPLAY_LABELS.get(qid, qid)} has no selected method."
            )
        spec = get_spec(str(raw))
        if not spec.implemented:
            raise QuadrantComparisonError(
                f"{spec.display_name} is not implemented. "
                "The quadrant UI should show it as unavailable rather than running it."
            )
        emb = resolve_representation_name(str(raw))
        inferred = infer_quadrant(emb)
        if inferred == QUADRANT_UNKNOWN:
            raise QuadrantComparisonError(
                f"{emb!r} cannot be assigned to a quadrant safely "
                f"(ambiguous or unknown representation)."
            )
        if inferred != qid:
            raise QuadrantComparisonError(
                f"{emb!r} belongs to {inferred}, not {qid}. "
                "A method cannot be moved into another quadrant for a controlled comparison."
            )
        selected[qid] = emb

    for raw in config.extra_methods:
        if not raw:
            continue
        spec = get_spec(str(raw))
        if not spec.implemented:
            raise QuadrantComparisonError(
                f"Advanced method {raw!r} is not implemented."
            )
        if infer_quadrant(str(raw)) == QUADRANT_UNKNOWN:
            raise QuadrantComparisonError(
                f"Advanced method {raw!r} cannot be assigned to a quadrant safely."
            )

    if not selected and not config.selected_embeddings():
        raise QuadrantComparisonError(
            "Select at least one implemented representation for the quadrant comparison."
        )

    decoder = str(config.decoder_name or "").strip()
    if not decoder:
        raise QuadrantComparisonError("A single decoder family is required.")
    allowed = resolve_model_names_for_target(
        config.target, max_models="full", selected=(decoder,),
    )
    if decoder not in allowed:
        raise QuadrantComparisonError(
            f"Decoder {decoder!r} is not valid for target {config.target!r}."
        )


def assert_result_matches_observation(
    result: Mapping[str, Any],
    observation: ObservationConfig,
) -> None:
    """Reject a saved result that does not share the active observation."""
    result_w = result.get("window_s", result.get("decode_window_s"))
    if result_w is not None and not windows_close(float(result_w), observation.window_s):
        raise QuadrantComparisonError(
            _window_mismatch_message(float(result_w), observation.window_s)
        )
    result_dt = result.get("update_dt_s", result.get("update_dt"))
    if result_dt is not None and not windows_close(float(result_dt), observation.update_dt):
        raise QuadrantComparisonError(
            "Quadrant comparison requires a shared update interval. "
            f"The selected result uses {float(result_dt) * 1000:.0f} ms updates; "
            f"the active observation uses {observation.update_dt * 1000:.0f} ms."
        )
    result_fs = result.get("feature_set")
    if result_fs is not None and str(result_fs) != str(observation.feature_set):
        raise QuadrantComparisonError(
            "Quadrant comparison requires identical feature construction. "
            f"The selected result used feature_set={result_fs!r}; "
            f"the active observation uses {observation.feature_set!r}."
        )
    result_src = result.get("spike_source") or result.get("source") or result.get("source_spikes")
    if result_src is not None and str(result_src) != str(observation.source_spikes):
        raise QuadrantComparisonError(
            "Quadrant comparison requires the same spike source. "
            f"The selected result used {result_src!r}; "
            f"the active observation uses {observation.source_spikes!r}."
        )
    result_hash = result.get("observation_config_hash")
    if result_hash and str(result_hash) != observation.hash():
        # Hash mismatch can also come from simulation_run_id; window/feature already checked.
        if result_w is not None and not windows_close(float(result_w), observation.window_s):
            raise QuadrantComparisonError(
                _window_mismatch_message(float(result_w), observation.window_s)
            )


def assert_metrics_share_observation(
    metrics: pd.DataFrame,
    observation: ObservationConfig,
    *,
    target: str | None = None,
    decoder_name: str | None = None,
) -> None:
    """Every method-level row in a controlled comparison must share O(W,F) and split identity."""
    if metrics is None or metrics.empty:
        raise QuadrantComparisonError("Quadrant comparison produced no metric rows.")
    for _, row in metrics.iterrows():
        assert_result_matches_observation(row.to_dict(), observation)
        if target is not None:
            row_target = row.get("target_name")
            if pd.notna(row_target) and str(row_target) != str(target):
                raise QuadrantComparisonError(
                    "Quadrant comparison requires an identical behavioral target. "
                    f"Found {row_target!r} while the experiment uses {target!r}."
                )
        if decoder_name is not None:
            row_dec = row.get("decoder_name") or row.get("model")
            if pd.notna(row_dec) and str(row_dec) != str(decoder_name):
                raise QuadrantComparisonError(
                    "This comparison is intended to isolate representation effects, "
                    f"so all branches must use decoder {decoder_name!r}; found {row_dec!r}."
                )
    windows = metrics["decode_window_s"] if "decode_window_s" in metrics.columns else None
    if windows is not None and windows.nunique(dropna=True) > 1:
        raise QuadrantComparisonError(
            "Do not mix incompatible windows within a single controlled "
            "quadrant comparison. Use Window Sweep for separate experiments per W."
        )


def annotate_metrics_dataframe(df: pd.DataFrame, *, overwrite: bool = False) -> pd.DataFrame:
    """Add quadrant columns to a metrics table without dropping method-level rows."""
    if df is None or df.empty:
        return df
    rows = annotate_metrics_rows(df.to_dict(orient="records"), overwrite=overwrite)
    return pd.DataFrame(rows)


def to_comparison_config(config: QuadrantExperimentConfig) -> ComparisonRunConfig:
    """Adapter: one shared observation, selected E branches, one decoder, one target."""
    validate_controlled_quadrant_comparison(config)
    embeddings = config.selected_embeddings()
    obs = config.observation
    roots = tuple(str(p) for p in config.reuse_search_roots if p)
    return ComparisonRunConfig(
        input_dir=Path(config.input_dir),
        output_dir=Path(config.output_dir),
        spike_source=str(obs.source_spikes),
        decode_windows=(float(obs.window_s),),
        update_dt=float(obs.update_dt),
        train_frac=float(config.train_frac),
        split_gap_s=float(config.split_gap_s),
        feature_types=("counts",),
        embedding_types=embeddings,
        use_fe_grid=True,
        feature_sets=(str(obs.feature_set),),
        manifold_n_components=(int(config.n_components),),
        max_models="quick",
        seed=int(config.seed),
        enable_trigger_search=bool(config.enable_trigger_search),
        include_controls=False,
        region_ablation=False,
        layer_ablation=False,
        population_ablation=False,
        compute_latent_stability=False,
        write_dynamic_latent_figures=False,
        compute_dynamic_latent_extras=False,
        n_jobs=1,
        reuse_transforms=bool(config.reuse_transforms),
        reuse_search_roots=roots,
        decoder_names=(str(config.decoder_name),),
        targets=(str(config.target),),
        run_id=str(obs.simulation_run_id or Path(config.input_dir).name),
    )


def _method_level_summary(
    metrics: pd.DataFrame,
    config: QuadrantExperimentConfig,
) -> list[dict[str, Any]]:
    metric_key, direction = PRIMARY_METRIC.get(
        config.target, ("r2", "higher"),
    )
    rows: list[dict[str, Any]] = []
    methods_by_emb = {
        resolve_manifold_alias(str(m)): q
        for q, m in config.methods.items()
        if m
    }
    for _, row in metrics.iterrows():
        if str(row.get("target_name") or "") != str(config.target):
            continue
        emb = str(row.get("embedding_type") or row.get("manifold") or "")
        qid = str(row.get("quadrant") or methods_by_emb.get(emb) or infer_quadrant(emb))
        value = row.get(metric_key)
        try:
            value_f = float(value) if value is not None and pd.notna(value) else None
        except (TypeError, ValueError):
            value_f = None
        spec = get_spec(emb)
        rows.append({
            "quadrant": qid,
            "quadrant_label": QUADRANT_DISPLAY_LABELS.get(qid, qid),
            "representation_name": emb,
            "display_name": spec.display_name,
            "decoder_name": str(row.get("decoder_name") or config.decoder_name),
            "target_name": config.target,
            "primary_metric": metric_key,
            "primary_metric_direction": direction,
            "primary_metric_value": value_f,
            "realtime_capable": spec.realtime_capable,
            "offline_only": spec.offline_only,
            "window_s": float(row.get("decode_window_s") or config.observation.window_s),
            "update_dt_s": float(row.get("update_dt_s") or config.observation.update_dt),
            "feature_set": str(row.get("feature_set") or config.observation.feature_set),
            "observation_config_hash": str(
                row.get("observation_config_hash") or config.observation.hash()
            ),
            "config_id": row.get("config_id"),
        })
    return rows


def _write_summary(
    output_dir: Path,
    config: QuadrantExperimentConfig,
    metrics: pd.DataFrame,
) -> dict[str, Any]:
    method_rows = _method_level_summary(metrics, config)
    payload = {
        "schema": "quadrant_comparison_v1",
        "created_at": _now_iso(),
        "experiment": config.to_dict(),
        "shared_observation": True,
        "shared_train_test": True,
        "reused_feature_dataset": bool(config.reuse_transforms),
        "reused_observation_hash": config.observation.hash(),
        "scientific_hash": config.scientific_hash(),
        "unimplemented_quadrants": [
            qid for qid in CANONICAL_QUADRANTS if not config.methods.get(qid)
        ],
        "primary_metric": list(PRIMARY_METRIC.get(config.target, ("r2", "higher"))),
        "method_results": method_rows,
        # Intentionally no quadrant averages — method-level rows are the scientific unit.
        "metrics_csv": str(metrics_path(output_dir)),
    }
    dest = summary_path(output_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, default=str))
    latest = Path(config.input_dir) / "quadrant_comparison" / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps({
        "output_dir": str(output_dir),
        "summary": str(dest),
        "created_at": payload["created_at"],
        "target": config.target,
        "decoder_name": config.decoder_name,
        "observation_hash": config.observation.hash(),
        "window_s": config.observation.window_s,
        "scientific_hash": config.scientific_hash(),
    }, indent=2))
    annotated_path = output_dir / "quadrant_method_metrics.csv"
    if method_rows:
        pd.DataFrame(method_rows).to_csv(annotated_path, index=False)
    return payload


def _run_one(
    config: QuadrantExperimentConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    validate_controlled_quadrant_comparison(config)
    cmp_cfg = to_comparison_config(config)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    metrics = run_decoder_comparison(cmp_cfg, progress_callback=progress_callback)
    if not isinstance(metrics, pd.DataFrame):
        metrics = pd.DataFrame(metrics or [])
    metrics = annotate_metrics_dataframe(metrics)
    csv = metrics_path(config.output_dir)
    if csv.exists():
        # Stamp quadrant columns onto the written table without dropping methods.
        written = pd.read_csv(csv)
        annotated = annotate_metrics_dataframe(written)
        annotated.to_csv(csv, index=False)
        metrics = annotated
    else:
        metrics.to_csv(csv, index=False)
    scored = metrics
    if "target_name" in metrics.columns:
        scored = metrics[metrics["target_name"].notna()].copy()
    assert_metrics_share_observation(
        scored,
        config.observation,
        target=config.target,
        decoder_name=config.decoder_name,
    )
    return _write_summary(Path(config.output_dir), config, scored)


def run_controlled_quadrant_experiment(
    config: QuadrantExperimentConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the fair four-quadrant experiment (or a per-W sweep of them)."""
    validate_controlled_quadrant_comparison(config)
    extra = tuple(float(w) for w in config.window_sweep_s if w)
    windows = (float(config.observation.window_s),)
    if extra:
        ordered: list[float] = []
        for w in (*windows, *extra):
            if not any(windows_close(w, x) for x in ordered):
                ordered.append(float(w))
        runs: list[dict[str, Any]] = []
        n = len(ordered)
        for i, w in enumerate(ordered, start=1):
            if progress_callback is not None:
                try:
                    progress_callback(
                        f"Window sweep {i}/{n}: W={w:.3f}s",
                        i - 1,
                        n,
                        stage="window_sweep",
                        task=f"W={w:.3f}s",
                    )
                except TypeError:
                    progress_callback(f"Window sweep {i}/{n}: W={w:.3f}s", i - 1, n)
            sub_obs = replace(config.observation, window_s=float(w))
            sub_out = Path(config.output_dir) / f"w{int(round(w * 1000)):04d}ms"
            sub = replace(
                config,
                observation=sub_obs,
                output_dir=sub_out,
                window_sweep_s=(),
            )
            runs.append(_run_one(sub, progress_callback=progress_callback))
        bundle = {
            "schema": "quadrant_window_sweep_v1",
            "created_at": _now_iso(),
            "windows_s": ordered,
            "runs": runs,
        }
        sweep_path = Path(config.output_dir) / "window_sweep_summary.json"
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_path.write_text(json.dumps(bundle, indent=2, default=str))
        return bundle
    return _run_one(config, progress_callback=progress_callback)


def load_quadrant_summary(experiment_dir: Path) -> dict[str, Any] | None:
    latest = Path(experiment_dir) / "quadrant_comparison" / "latest.json"
    if latest.exists():
        try:
            pointer = json.loads(latest.read_text())
            summary = Path(pointer.get("summary") or "")
            if summary.exists():
                return json.loads(summary.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    matches = sorted(Path(experiment_dir).glob("quadrant_comparison/**/quadrant_summary.json"))
    if not matches:
        return None
    try:
        return json.loads(matches[-1].read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_legacy_metrics_with_quadrants(csv_path: Path) -> pd.DataFrame:
    """Load an older decoder-comparison CSV and infer quadrant without rewriting the file."""
    df = pd.read_csv(csv_path)
    return annotate_metrics_dataframe(df, overwrite=False)
