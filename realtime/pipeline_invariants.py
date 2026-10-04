"""Pipeline compatibility assertions.

Fail with informative errors rather than silently recovering incompatible
states. These checks never mutate scientific artifacts.
"""

from __future__ import annotations

import numpy as np

from realtime.pipeline_artifacts import (
    AnalysisConfig,
    DecoderResult,
    FeatureDataset,
    ObservationConfig,
    RepresentationResult,
    windows_close,
)


class PipelineInvariantError(ValueError):
    """Raised when a pipeline compatibility rule is violated."""


# Sample-index clocks (30 kHz × a few seconds) sit well above this; real
# sessions in this project are hundreds of seconds, written as floats.
_SAMPLE_INDEX_INTEGER_MAX_S = 10_000.0
_MAX_PLAUSIBLE_SESSION_S = 24 * 3600.0


def assert_times_in_seconds(
    times,
    *,
    session_length_s: float | None = None,
    context: str = "timestamps",
    plausible_cap: bool = True,
) -> None:
    """Reject spike/position times that are sample indices, not seconds.

    Heuristic (spec Phase 1): ``max > session_length × 2`` when a session
    length is known; integer dtype with a large max; otherwise a 24 h cap
    at file/stream loaders. Live buffers omit the 24 h cap so a single
    clock-glitch sample is handled by retention, not this gate.
    """
    arr = np.asarray(times)
    if arr.size == 0:
        return
    integer_in = np.issubdtype(arr.dtype, np.integer)
    t = np.asarray(arr, dtype=float).ravel()
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return
    tmax = float(np.max(finite))
    if session_length_s is not None and float(session_length_s) > 0:
        limit = float(session_length_s) * 2.0
        if tmax > limit:
            raise PipelineInvariantError(
                f"{context} are not in seconds: max={tmax:.6g} exceeds "
                f"session_length×2={limit:.6g}. This pattern matches sample "
                "indices (e.g. 30 kHz Phy spike_times.npy) read as seconds."
            )
    elif plausible_cap and tmax > _MAX_PLAUSIBLE_SESSION_S:
        raise PipelineInvariantError(
            f"{context} are not in seconds: max={tmax:.6g} exceeds the "
            f"{_MAX_PLAUSIBLE_SESSION_S:.0f} s plausibility cap. "
            "Spike and position times must be seconds, not sample indices."
        )
    if integer_in and tmax > _SAMPLE_INDEX_INTEGER_MAX_S:
        raise PipelineInvariantError(
            f"{context} have integer dtype and max={tmax:.6g}, consistent "
            "with sample indices rather than seconds."
        )


def assert_valid_timing(window_s: float, update_dt: float) -> None:
    if float(window_s) <= 0:
        raise PipelineInvariantError(f"window_s must be > 0, got {window_s}")
    if float(update_dt) <= 0:
        raise PipelineInvariantError(f"update_dt must be > 0, got {update_dt}")


def assert_spike_source_explicit(source: str) -> str:
    src = str(source or "").strip()
    if src not in {"sorted", "ground_truth"}:
        raise PipelineInvariantError(
            f"Spike source must be explicit 'sorted' or 'ground_truth', got {source!r}."
        )
    return src


def assert_timestamps_align(
    left: np.ndarray | None,
    right: np.ndarray | None,
    *,
    context: str = "timestamps",
    atol: float = 1e-9,
) -> None:
    if left is None or right is None:
        return
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise PipelineInvariantError(
            f"{context} length mismatch: {a.shape[0]} vs {b.shape[0]}"
        )
    if a.size and not np.allclose(a, b, atol=atol, rtol=0.0):
        raise PipelineInvariantError(
            f"{context} do not align (max abs diff "
            f"{float(np.max(np.abs(a - b))):.6g} s)"
        )


def assert_representation_matches_features(
    representation: RepresentationResult,
    features: FeatureDataset,
) -> None:
    if representation.source_feature_hash and features.observation.hash() not in {
        representation.source_feature_hash,
        features.observation.fit_hash(),
    }:
        # Allow either observation hash or fit hash as the stored source.
        if representation.source_feature_hash not in {
            features.observation.hash(),
            features.observation.fit_hash(),
        }:
            raise PipelineInvariantError(
                "Representation was fit on a different feature dataset "
                f"(source_feature_hash={representation.source_feature_hash}, "
                f"feature_hash={features.observation.hash()})."
            )
    if not windows_close(representation.window_s, features.window_s):
        raise PipelineInvariantError(
            "Representation window "
            f"{representation.window_s:.3f}s does not match feature window "
            f"{features.window_s:.3f}s. W is owned by observation construction."
        )
    assert_timestamps_align(
        representation.timestamps,
        features.timestamps,
        context="representation vs feature timestamps",
    )


def assert_decoder_matches_observation(
    decoder: DecoderResult,
    observation: ObservationConfig,
) -> None:
    if not windows_close(decoder.observation.window_s, observation.window_s):
        raise PipelineInvariantError(
            f"Decoder {decoder.decoder_name!r} is a "
            f"{decoder.observation.window_s * 1000:.0f} ms model; "
            f"requested observation is {observation.window_s * 1000:.0f} ms. "
            "Retrain rather than changing W at replay."
        )
    if decoder.observation.hash() != observation.hash():
        # Window match is required; feature-set mismatch is also incompatible.
        if decoder.observation.feature_set != observation.feature_set:
            raise PipelineInvariantError(
                f"Decoder was trained on feature_set={decoder.observation.feature_set!r} "
                f"but the observation is {observation.feature_set!r}."
            )
        if decoder.observation.source_spikes != observation.source_spikes:
            raise PipelineInvariantError(
                f"Decoder spike source {decoder.observation.source_spikes!r} "
                f"does not match observation {observation.source_spikes!r}."
            )


def assert_replay_window_matches(
    model_window_s: float,
    requested_window_s: float,
    *,
    decoder_name: str | None = None,
) -> None:
    if windows_close(model_window_s, requested_window_s):
        return
    who = f"Decoder {decoder_name!r}" if decoder_name else "Saved model"
    raise PipelineInvariantError(
        f"{who} was trained with W={float(model_window_s)*1000:.0f} ms; "
        f"replay requested W={float(requested_window_s)*1000:.0f} ms. "
        "A saved model permanently owns its observation window."
    )


def assert_feature_dimension(X: np.ndarray, expected: int | None, *, context: str = "X") -> None:
    if expected is None:
        return
    got = int(np.asarray(X).shape[-1])
    if got != int(expected):
        raise PipelineInvariantError(
            f"{context} feature dimension {got} does not match fitted dimension {expected}."
        )


def assert_target_present(behavior, timestamps: np.ndarray | None, target: str) -> None:
    if behavior is None:
        raise PipelineInvariantError(f"Behavior table missing for target {target!r}.")
    columns = getattr(behavior, "columns", None)
    if columns is None:
        return
    aliases = {
        "position": ("x", "y", "position_x", "position_y"),
        "speed": ("speed", "speed_cm_s"),
        "acceleration": ("acceleration", "acceleration_cm_s2"),
        "head_direction": ("head_direction", "head_direction_rad", "hd_sin", "hd_cos"),
        "distance_to_wall": ("distance_to_wall", "distance_to_wall_cm"),
        "spatial_context": ("spatial_context",),
        "movement_state": ("movement_state",),
        "wall_distance_bin": ("wall_distance_bin",),
    }
    names = set(str(c) for c in columns)
    if target in names:
        return
    for alias in aliases.get(target, ()):
        if alias in names:
            return
    raise PipelineInvariantError(
        f"Target {target!r} is not present in the behavior table "
        f"(columns={sorted(names)[:20]})."
    )


def assert_realtime_compatible(
    embedding_type: str,
    *,
    deployment: bool = False,
    replay: bool = False,
) -> None:
    """Reject offline-only representations for deployment and/or realtime replay."""
    if not deployment and not replay:
        return
    from realtime.representation_registry import get_spec
    from realtime.search_space import resolve_manifold_alias

    name = resolve_manifold_alias(str(embedding_type or "").strip())
    if name in {"identity", "counts", "rates"}:
        return
    spec = get_spec(name)
    if spec.offline_only or not spec.realtime_capable:
        kind = "deployment" if deployment else "realtime replay"
        raise PipelineInvariantError(
            f"Representation {embedding_type!r} is offline-only / not "
            f"realtime-capable and cannot be used for {kind}."
        )


def validate_analysis_config(
    cfg: AnalysisConfig,
    *,
    require_realtime: bool = False,
) -> None:
    """Lightweight checks at analysis boundaries. Does not mutate ``cfg``."""
    assert_valid_timing(cfg.window_s, cfg.observation.update_dt)
    if not 0.0 < float(cfg.train_frac) < 1.0:
        raise PipelineInvariantError(
            f"train_frac must be in (0, 1), got {cfg.train_frac}."
        )
    from realtime.representation_registry import get_spec

    spec = get_spec(cfg.representation)
    if spec.temporal_type == "dynamic" and float(cfg.window_s) + 1e-12 < float(
        cfg.observation.update_dt
    ):
        raise PipelineInvariantError(
            "Dynamic representation requires a causal history window "
            f"W={cfg.window_s}s >= update_dt={cfg.observation.update_dt}s."
        )
    want_rt = bool(require_realtime) or (cfg.realtime is True)
    if want_rt:
        assert_realtime_compatible(cfg.representation, replay=True)


def assert_cache_matches_analysis(
    stored_hash: str | None,
    cfg: AnalysisConfig,
    *,
    context: str = "cached result",
) -> None:
    """Refuse to treat a result from a different analysis config as current."""
    if not stored_hash:
        return
    allowed = {cfg.hash(), cfg.observation.hash(), cfg.observation.fit_hash()}
    if str(stored_hash) not in allowed:
        raise PipelineInvariantError(
            f"{context} was generated under a different configuration "
            f"(stored={stored_hash}, active={cfg.hash()})."
        )


def assert_feature_matches_analysis(
    features: FeatureDataset,
    cfg: AnalysisConfig,
) -> None:
    if features.observation.hash() != cfg.observation.hash():
        raise PipelineInvariantError(
            "Feature dataset observation hash does not match the active "
            f"analysis (features={features.observation.hash()}, "
            f"active={cfg.observation.hash()}). W and F are owned by "
            "the committed observation."
        )
    if not windows_close(features.window_s, cfg.window_s):
        raise PipelineInvariantError(
            f"Feature window {features.window_s:.3f}s does not match active "
            f"analysis window {cfg.window_s:.3f}s."
        )


def assert_decoder_matches_analysis(
    decoder: DecoderResult,
    cfg: AnalysisConfig,
) -> None:
    from realtime.search_space import resolve_manifold_alias

    assert_decoder_matches_observation(decoder, cfg.observation)
    if decoder.target != cfg.target:
        raise PipelineInvariantError(
            f"Decoder target {decoder.target!r} does not match analysis "
            f"target {cfg.target!r}."
        )
    left = resolve_manifold_alias(decoder.representation_name)
    right = resolve_manifold_alias(cfg.representation)
    if left != right:
        raise PipelineInvariantError(
            f"Decoder representation {decoder.representation_name!r} does not "
            f"match analysis representation {cfg.representation!r}."
        )


def assert_train_only_fit(train_mask: np.ndarray | None, *, context: str = "transform") -> None:
    if train_mask is None:
        return
    mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        raise PipelineInvariantError(f"{context}: empty train mask.")
    if not np.any(mask):
        raise PipelineInvariantError(f"{context}: train mask has no True samples.")


def validate_observation_against_features(
    observation: ObservationConfig,
    features: FeatureDataset,
) -> None:
    if observation.hash() != features.observation.hash() and not windows_close(
        observation.window_s, features.window_s
    ):
        raise PipelineInvariantError(
            "Active observation does not match the feature dataset "
            f"(obs W={observation.window_s:.3f}s vs features W={features.window_s:.3f}s)."
        )
    if observation.feature_set != features.observation.feature_set:
        raise PipelineInvariantError(
            f"Active feature_set {observation.feature_set!r} does not match "
            f"dataset {features.observation.feature_set!r}."
        )
