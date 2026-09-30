"""Load the frozen n=5 quadrant experiment config and its file hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from realtime.pipeline_artifacts import config_hash
from realtime.train_decoder import purge_gap_s

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "quadrant_n5.yaml"

# Commit (a) on quadrant-n5: runner as of 01:56 plus the n5 numerical tree.
# Provenance SHA for seeds 0–4. Later runner fixes (A13 continue, skip, EXIT)
# are not this SHA.
# pre-rewrite: 775fa1c38063a1ecea07df28c64174beea8f122e
SEEDS_0_4_PROVENANCE_SHA = "dcc78ed74194aa1d3e8fa7e1e25e90e02b26265e"


def report_code_sha() -> str | None:
    """Git HEAD of the tree that writes reports (commit (c) when PDFs are regenerated)."""
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:
        return None


def config_sha256(path: Path | None = None) -> str:
    loc = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    digest = hashlib.sha256(loc.read_bytes()).hexdigest()
    return digest


def load_quadrant_n5_yaml(path: Path | None = None) -> dict[str, Any]:
    loc = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(loc.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{loc} must be a mapping")
    raw = dict(raw)
    raw["config_path"] = str(loc.resolve())
    raw["config_sha256"] = config_sha256(loc)
    split = dict(raw.get("split") or {})
    features = dict(raw.get("features") or {})
    window_s = float(features.get("window_s", 0.250))
    update_dt = float(features.get("update_dt", 0.050))
    max_history_s = float(split.get("max_history_s", 0.250))
    computed_gap = purge_gap_s(
        window_s, max_history_s=max_history_s, update_dt=update_dt,
    )
    recorded = float(split.get("gap_s", computed_gap))
    if abs(recorded - computed_gap) > 1e-12:
        raise ValueError(
            f"split.gap_s={recorded} does not match "
            f"max(W+max_history, 1.0) rounded to steps ({computed_gap})"
        )
    raw["split"] = split
    if float(raw.get("latency_budget_ms", 50)) != 50.0:
        raise ValueError("latency_budget_ms must be 50")
    gpfa = (raw.get("representations") or {}).get("gpfa") or {}
    if gpfa.get("offline_only") is True:
        raise ValueError("GPFA label is expected_label from A9, not offline_only: true")
    probe = dict(raw.get("probe_track") or {})
    probe_rel = str(probe.get("file") or "").strip()
    if not probe_rel:
        raise ValueError("probe_track.file must name the lab insertion YAML")
    probe_path = (REPO_ROOT / probe_rel).resolve()
    if not probe_path.is_file():
        raise FileNotFoundError(f"probe_track.file not found: {probe_path}")
    raw["probe_track"] = probe
    raw["probe_track_path"] = str(probe_path)
    raw["probe_track_sha256"] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    inclusion = dict(raw.get("unit_inclusion") or {})
    regions = list(inclusion.get("regions") or [])
    if "hippocampal_formation_transition" not in regions:
        raise ValueError("unit_inclusion.regions must include hippocampal_formation_transition")
    a13 = dict((raw.get("phase3") or {}).get("a13") or {})
    if int(a13.get("n_shifts", 0)) != 20:
        raise ValueError("phase3.a13.n_shifts must be 20")
    if float(a13.get("pass_min_cm", 0)) != -2.0:
        raise ValueError("phase3.a13.pass_min_cm must be -2.0")
    return raw


def runtime_versions() -> dict[str, str]:
    import numpy
    import sklearn

    return {"numpy": numpy.__version__, "sklearn": sklearn.__version__}


def derive_seed_streams(
    master_seed: int,
    n_seeds: int,
    seed_index: int,
    components: list[str] | tuple[str, ...],
) -> dict[str, int]:
    """Independent streams for one data seed (SPEC §4)."""
    import numpy as np

    if not 0 <= int(seed_index) < int(n_seeds):
        raise ValueError(f"seed_index {seed_index} not in [0, {n_seeds})")
    root = np.random.SeedSequence(int(master_seed))
    per_seed = root.spawn(int(n_seeds))[int(seed_index)]
    spawned = per_seed.spawn(len(components))
    out: dict[str, int] = {}
    for name, seq in zip(components, spawned):
        out[str(name)] = int(seq.generate_state(1, dtype=np.uint32)[0])
    out["data_seed"] = int(per_seed.generate_state(1, dtype=np.uint32)[0])
    out["master_seed"] = int(master_seed)
    out["seed_index"] = int(seed_index)
    return out


SIM_HASH_STREAMS = (
    "data_seed",
    "seed_index",
    "master_seed",
    "trajectory",
    "neural",
    "recording_noise",
    "sorting_errors",
)


def sim_data_identity(cfg: dict[str, Any], streams: dict[str, Any]) -> str:
    """Simulator inputs only (seed streams + session + probe). Not the analysis overlay."""
    session = cfg["session"]
    payload = {
        "session_s": float(session["session_s"]),
        "arena_shape": str(session["arena_shape"]),
        "arena_size_cm": float(session["arena_size_cm"]),
        "thigmotaxis": float(session["thigmotaxis"]),
        "behavior_dt": float(session["behavior_dt"]),
        "probe_track_sha256": str(cfg.get("probe_track_sha256") or ""),
    }
    for key in SIM_HASH_STREAMS:
        payload[key] = int(streams[key])
    return config_hash(payload, n=16)


def sim_fit_hash(cfg: dict[str, Any], streams: dict[str, Any]) -> str:
    """Identity of a generated simulation (seed + frozen config).

    ``methods`` is excluded: it does not enter the simulator. Missing hash
    or mismatch is a cache miss; unkeyed sim dirs are never reused.
    """
    session = cfg["session"]
    payload = {
        "config_sha256": str(cfg["config_sha256"]),
        "session_s": float(session["session_s"]),
        "arena_shape": str(session["arena_shape"]),
        "arena_size_cm": float(session["arena_size_cm"]),
        "thigmotaxis": float(session["thigmotaxis"]),
        "behavior_dt": float(session["behavior_dt"]),
        "probe_track_sha256": str(cfg.get("probe_track_sha256") or ""),
    }
    for key in SIM_HASH_STREAMS:
        payload[key] = int(streams[key])
    return config_hash(payload, n=16)


def read_sim_provenance(sim_dir: Path) -> dict[str, Any] | None:
    loc = Path(sim_dir) / "quadrant_n5_sim.json"
    if not loc.is_file():
        return None
    raw = loc.read_text()
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def sim_provenance_matches(sim_dir: Path, expected_hash: str) -> bool:
    meta = read_sim_provenance(sim_dir)
    if not meta:
        return False
    stored = meta.get("sim_fit_hash")
    return stored is not None and str(stored) == str(expected_hash)


def sim_dir_is_complete(sim_dir: Path) -> bool:
    dest = Path(sim_dir)
    return (dest / "behavior.csv").is_file() and (dest / "spikes_sorted.csv").is_file()


def promote_unkeyed_sim(
    sim_dir: Path,
    cfg: dict[str, Any],
    streams: dict[str, Any],
    expected_hash: str,
) -> bool:
    """Stamp an unkeyed seed-0-style dir only when seed + config already match.

    Returns True if the directory is now keyed and reusable.
    """
    dest = Path(sim_dir)
    if not sim_dir_is_complete(dest):
        return False
    if sim_provenance_matches(dest, expected_hash):
        return True
    meta = read_sim_provenance(dest) or {}
    summary_path = dest / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}
    recorded_cfg = meta.get("config_sha256")
    recorded_seed = summary.get("seed", meta.get("seed_streams", {}).get("data_seed"))
    if recorded_cfg != cfg["config_sha256"]:
        return False
    if recorded_seed is None or int(recorded_seed) != int(streams["data_seed"]):
        return False
    meta = dict(meta)
    meta["sim_fit_hash"] = str(expected_hash)
    meta["config_sha256"] = cfg["config_sha256"]
    meta["seed_streams"] = dict(streams)
    dest.joinpath("quadrant_n5_sim.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n"
    )
    return True


def inner_cv_block_masks(
    train_times,
    *,
    n_blocks: int = 5,
    gap_s: float = 1.0,
) -> list[tuple[Any, Any]]:
    """Contiguous inner-CV folds on the training segment.

    The held-out block is purged by ``gap_s`` on each side. Representation
    fits must use only the train mask of each fold so validation sees the
    same out-of-sample path as test (Nyström / filter / lag stack).
    """
    import numpy as np

    t = np.asarray(train_times, dtype=float)
    if t.ndim != 1 or t.size < n_blocks * 3:
        raise ValueError("Not enough training times for inner CV blocks")
    n = int(t.size)
    edges = np.linspace(0, n, int(n_blocks) + 1).astype(int)
    gap = float(gap_s)
    folds: list[tuple[Any, Any]] = []
    for i in range(int(n_blocks)):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            raise ValueError(f"Empty inner-CV block {i}")
        val_t0 = float(t[lo])
        val_t1 = float(t[hi - 1])
        val_mask = (t >= val_t0 + gap) & (t <= val_t1 - gap)
        train_mask = (t < val_t0 - gap) | (t > val_t1 + gap)
        if not val_mask.any() or not train_mask.any():
            raise ValueError(
                f"Inner-CV block {i} empty after purge "
                f"(train={int(train_mask.sum())}, val={int(val_mask.sum())})"
            )
        folds.append((train_mask, val_mask))
    return folds

