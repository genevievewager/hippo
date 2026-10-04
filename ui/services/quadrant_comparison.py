"""Streamlit-free helpers for the Quadrant Comparison page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from realtime.pipeline_artifacts import ObservationConfig
from realtime.quadrant_experiment import (
    QuadrantExperimentConfig,
    load_quadrant_summary,
    quadrant_output_dir,
    run_controlled_quadrant_experiment,
)


def default_comparison_reuse_roots(experiment_dir: Path, spike_source: str) -> tuple[str, ...]:
    """Reuse F/E caches written by Latent Representations / Decoder Benchmark."""
    roots: list[str] = []
    sorted_cmp = Path(experiment_dir) / "decoder_comparison" / str(spike_source)
    if sorted_cmp.exists():
        roots.append(str(sorted_cmp))
    generic = Path(experiment_dir) / "decoder_comparison"
    if generic.exists() and str(generic) not in roots:
        roots.append(str(generic))
    return tuple(roots)


def build_experiment_config(
    *,
    experiment_dir: Path,
    observation: ObservationConfig,
    target: str,
    decoder_name: str,
    methods: dict[str, str | None],
    extra_methods: tuple[str, ...] = (),
    n_components: int = 3,
    window_sweep_s: tuple[float, ...] = (),
    train_frac: float = 0.70,
    seed: int = 42,
) -> QuadrantExperimentConfig:
    out = quadrant_output_dir(experiment_dir, observation.source_spikes)
    return QuadrantExperimentConfig(
        input_dir=Path(experiment_dir),
        output_dir=out,
        observation=observation,
        target=str(target),
        decoder_name=str(decoder_name),
        methods=dict(methods),
        extra_methods=tuple(extra_methods),
        n_components=int(n_components),
        train_frac=float(train_frac),
        seed=int(seed),
        reuse_transforms=True,
        reuse_search_roots=default_comparison_reuse_roots(
            experiment_dir, observation.source_spikes,
        ),
        enable_trigger_search=False,
        window_sweep_s=tuple(float(w) for w in window_sweep_s),
    )


def run_quadrant_comparison_job(
    *,
    experiment_dir: Path,
    observation: dict[str, Any],
    target: str,
    decoder_name: str,
    methods: dict[str, str | None],
    extra_methods: tuple[str, ...] = (),
    n_components: int = 3,
    window_sweep_s: tuple[float, ...] = (),
    train_frac: float = 0.70,
    seed: int = 42,
    progress_callback=None,
) -> dict[str, Any]:
    cfg = build_experiment_config(
        experiment_dir=Path(experiment_dir),
        observation=ObservationConfig.from_dict(observation),
        target=target,
        decoder_name=decoder_name,
        methods=methods,
        extra_methods=extra_methods,
        n_components=n_components,
        window_sweep_s=window_sweep_s,
        train_frac=train_frac,
        seed=seed,
    )
    _stamp_pipeline_analysis(Path(experiment_dir), cfg)
    return run_controlled_quadrant_experiment(cfg, progress_callback=progress_callback)


def _stamp_pipeline_analysis(experiment_dir: Path, cfg) -> None:
    """Persist the shared observation + selected D/target on pipeline_run.json."""
    from realtime.pipeline_artifacts import AnalysisConfig
    from realtime.pipeline_graph import load_or_infer_pipeline, save_pipeline_run

    embs = cfg.selected_embeddings()
    analysis = AnalysisConfig.from_observation(
        cfg.observation,
        representation=str(embs[0] if embs else "global_pca"),
        decoder=str(cfg.decoder_name),
        target=str(cfg.target),
        train_frac=float(cfg.train_frac),
        seed=int(cfg.seed),
        n_components=int(cfg.n_components),
        causal=True,
    )
    run = load_or_infer_pipeline(Path(experiment_dir))
    run.set_analysis(analysis)
    save_pipeline_run(run)


def existing_summary(experiment_dir: Path) -> dict[str, Any] | None:
    return load_quadrant_summary(Path(experiment_dir))
