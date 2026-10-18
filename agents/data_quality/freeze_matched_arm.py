"""Freeze matched-arm provenance and MANIFEST (no regenerates)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.session_discovery import eligible_room_a_sessions
from agents.data_quality.matching.yield_boost import MEC_KEYS, SUB_KEYS


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _recording_day(session: str) -> str:
    parts = session.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else session


def _escalate_boost(before: dict[str, dict[str, int]], required: dict[str, int], n_regen: int) -> dict[str, dict[str, int]]:
    """Recompute FINAL after-keys with safety = 1.3 * 1.5**n_regen (pilot regen rule)."""
    safety = 1.3 * (1.5 ** int(n_regen))
    after: dict[str, dict[str, int]] = {}
    for keys, lab in ((MEC_KEYS, "MEC"), (SUB_KEYS, "SUB")):
        need = int(math.ceil(max(0, int(required.get(lab, 0))) * safety))
        cur = [int(before.get(lab, {}).get(k, 0)) for k in keys]
        total = sum(cur)
        if need <= 0 or total <= 0:
            after[lab] = {k: int(v) for k, v in zip(keys, cur)}
            continue
        if total < need:
            scale = need / total
            new_vals = [max(1, int(math.ceil(v * scale))) for v in cur]
            while sum(new_vals) < need:
                new_vals[int(max(range(len(new_vals)), key=lambda i: new_vals[i]))] += 1
            after[lab] = {k: int(v) for k, v in zip(keys, new_vals)}
        else:
            after[lab] = {k: int(v) for k, v in zip(keys, cur)}
    return after


def _final_ratinabox_params(rep: dict[str, Any]) -> dict[str, int]:
    boost = rep.get("ratinabox_boost") or {}
    before = boost.get("before") or {}
    n_regen = int(rep.get("n_regenerations") or 0)
    required = rep.get("required_counts") or {}
    after = _escalate_boost(before, required, n_regen)
    # flatten + legacy aliases
    flat: dict[str, int] = {}
    for lab in ("MEC", "SUB"):
        flat.update({k: int(v) for k, v in (after.get(lab) or {}).items()})
    aliases = {
        "n_mec_hd_cells": "n_head_direction_cells",
        "n_sub_bvc_cells": "n_boundary_vector_cells",
        "n_mec_speed_cells": "n_speed_cells",
        "n_mec_grid_cells": "n_grid_cells",
        "n_int_ca1": "n_interneurons",
    }
    for src, alias in aliases.items():
        if src in flat:
            flat[alias] = int(flat[src])
    return flat


def freeze_matched_arm() -> dict[str, Any]:
    matched = DATA_QUALITY_ROOT / "matched"
    eligible = set(eligible_room_a_sessions())
    entries: list[dict[str, Any]] = []
    for d in sorted(matched.glob("*_A")):
        session = d.name[:-2]
        if session not in eligible:
            continue
        pr = d / "pilot_report.json"
        if not pr.is_file():
            continue
        rep = json.loads(pr.read_text())
        yo = rep.get("yield_ok") or {}
        if not (yo.get("MEC") and yo.get("SUB")):
            continue

        final_rp = _final_ratinabox_params(rep)
        match_spec_path = d / "match_spec.json"
        capture_path = d / "cell_capture_matched.yaml"
        # derived-config SHA over match_spec + capture yaml
        h = hashlib.sha256()
        for p in (match_spec_path, capture_path):
            if p.is_file():
                h.update(p.read_bytes())
        derived_sha = h.hexdigest()

        sim = d / "sim"
        file_hashes = {
            "units.csv": _sha256_file(sim / "units.csv"),
            "spikes_sorted.csv": _sha256_file(sim / "spikes_sorted.csv"),
            "spikes_ground_truth.csv": _sha256_file(sim / "spikes_ground_truth.csv"),
            "behavior.csv": _sha256_file(sim / "behavior.csv"),
            "summary.json": _sha256_file(sim / "summary.json"),
            "match_spec.json": _sha256_file(match_spec_path),
            "cell_capture_matched.yaml": _sha256_file(capture_path),
        }
        seed = None
        if (sim / "summary.json").is_file():
            seed = json.loads((sim / "summary.json").read_text()).get("seed")

        provenance = {
            "session": session,
            "recording_day": _recording_day(session),
            "seed": seed,
            "n_regenerations": int(rep.get("n_regenerations") or 0),
            "final_ratinabox_params": final_rp,
            "ratinabox_boost_recorded": rep.get("ratinabox_boost"),
            "derived_config_sha256": derived_sha,
            "required_counts": rep.get("required_counts"),
            "sim_yield": rep.get("sim_yield"),
            "session_duration_s": rep.get("session_duration_s"),
            "arena_size_cm": rep.get("arena_size_cm"),
            "file_hashes": file_hashes,
            "frozen": True,
            "do_not_regenerate_without_asking": True,
        }
        prov_path = assert_dq_path(d / "provenance.json")
        prov_path.write_text(json.dumps(provenance, indent=2) + "\n")

        # also patch pilot_report with final params (additive keys only)
        if "final_ratinabox_params" not in rep or "derived_config_sha256" not in rep:
            rep["final_ratinabox_params"] = final_rp
            rep["derived_config_sha256"] = derived_sha
            rep["seed"] = seed
            pr.write_text(json.dumps(rep, indent=2) + "\n")

        entries.append(
            {
                "session": session,
                "recording_day": provenance["recording_day"],
                "final_ratinabox_params": final_rp,
                "derived_config_sha256": derived_sha,
                "seed": seed,
                "sim_yield": rep.get("sim_yield"),
                "required_counts": rep.get("required_counts"),
                "n_regenerations": provenance["n_regenerations"],
                "file_hashes": file_hashes,
            }
        )

    manifest = {
        "version": 1,
        "n_sessions": len(entries),
        "rule": "eligible room-A + yield_ok; never regenerate manifested session without asking",
        "sessions": entries,
    }
    out = assert_dq_path(matched / "MANIFEST.json")
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"n_sessions": len(entries), "manifest": str(out)}


if __name__ == "__main__":
    print(json.dumps(freeze_matched_arm(), indent=2))
