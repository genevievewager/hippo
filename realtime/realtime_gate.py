"""Realtime latency / history-window gating for deployable decoders."""

from __future__ import annotations

from typing import Any


DEFAULT_MAX_COMPUTE_MS = 25.0
DEFAULT_MAX_EFFECTIVE_HISTORY_S = 0.500


def realtime_gate_reason(
    *,
    total_compute_ms: float,
    decode_window_s: float,
    max_compute_ms: float = DEFAULT_MAX_COMPUTE_MS,
    max_effective_history_s: float = DEFAULT_MAX_EFFECTIVE_HISTORY_S,
) -> str:
    compute_ok = float(total_compute_ms) <= float(max_compute_ms)
    history_ok = float(decode_window_s) <= float(max_effective_history_s)
    if compute_ok and history_ok:
        return "passes"
    if (not compute_ok) and (not history_ok):
        return "compute_and_history_too_large"
    if not compute_ok:
        return "compute_too_slow"
    return "history_window_too_long"


def apply_realtime_gate(
    *,
    feature_compute_ms: float,
    embedding_transform_ms: float,
    decoder_predict_ms: float,
    decode_window_s: float,
    update_dt_s: float,
    max_compute_ms: float = DEFAULT_MAX_COMPUTE_MS,
    max_effective_history_s: float = DEFAULT_MAX_EFFECTIVE_HISTORY_S,
) -> dict[str, Any]:
    total = float(feature_compute_ms) + float(embedding_transform_ms) + float(decoder_predict_ms)
    reason = realtime_gate_reason(
        total_compute_ms=total,
        decode_window_s=decode_window_s,
        max_compute_ms=max_compute_ms,
        max_effective_history_s=max_effective_history_s,
    )
    return {
        "feature_compute_ms": float(feature_compute_ms),
        "embedding_transform_ms": float(embedding_transform_ms),
        "decoder_predict_ms": float(decoder_predict_ms),
        "total_compute_ms": total,
        "update_dt_s": float(update_dt_s),
        "decode_window_s": float(decode_window_s),
        "effective_history_s": float(decode_window_s),
        "passes_realtime_gate": reason == "passes",
        "realtime_gate_reason": reason,
    }
