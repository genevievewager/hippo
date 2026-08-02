"""Map region/layer identity to simulated Neuropixels unit capture.

Supports two config styles:
1. Legacy ``region_layer_probabilities`` (schematic CA1–MEC bands)
2. Lab trajectory ``cell_capture:`` blocks keyed by region name with
   ``unit_density_per_mm`` (configs/trajectories/*_cell_capture.yaml)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Default capture model: probabilities by (region, layer) → cell_type weights.
DEFAULT_CAPTURE_PROBS: dict[str, dict[str, dict[str, float]]] = {
    "CA1": {
        "oriens": {"INT_CA1": 0.70, "CA1_pyr": 0.30},
        "pyramidal": {"CA1_pyr": 0.85, "INT_CA1": 0.15},
        "radiatum": {"CA1_pyr": 0.55, "INT_CA1": 0.45},
        "*": {"CA1_pyr": 0.75, "INT_CA1": 0.25},
    },
    "CA2": {
        "pyramidal": {"CA2_pyr": 0.90, "INT_CA2": 0.10},
        "*": {"CA2_pyr": 1.0},
    },
    "CA3": {
        "pyramidal": {"CA3_pyr": 0.90, "INT_CA3": 0.10},
        "*": {"CA3_pyr": 1.0},
    },
    "DG": {
        "granule": {"DG_granule": 0.92, "INT_DG": 0.08},
        "hilus": {"DG_granule": 0.40, "INT_DG": 0.60},
        "*": {"DG_granule": 1.0},
    },
    "Subiculum": {
        "pyramidal": {"Sub_bvc": 0.90, "INT_SUB": 0.10},
        "*": {"Sub_bvc": 1.0},
    },
    "MEC": {
        "layer2": {"MEC_grid": 0.55, "MEC_speed": 0.25, "MEC_hd": 0.20},
        "layer3": {"MEC_hd": 0.70, "MEC_grid": 0.20, "MEC_speed": 0.10},
        "*": {"MEC_grid": 0.40, "MEC_hd": 0.40, "MEC_speed": 0.20},
    },
}

CELL_TYPE_TO_N_KEY: dict[str, str] = {
    "CA1_pyr": "n_ca1_place_cells",
    "INT_CA1": "n_int_ca1",
    "INT_CA2": "n_int_ca2",
    "INT_CA3": "n_int_ca3",
    "INT_DG": "n_int_dg",
    "INT_SUB": "n_int_sub",
    "CA2_pyr": "n_ca2_place_cells",
    "CA3_pyr": "n_ca3_place_cells",
    "DG_granule": "n_dg_place_cells",
    "MEC_grid": "n_mec_grid_cells",
    "MEC_hd": "n_mec_hd_cells",
    "MEC_speed": "n_mec_speed_cells",
    "Sub_bvc": "n_sub_bvc_cells",
}

# Known RiaB / decoder cell types (ignore optional placeholders like visual_units).
SIMULATED_CELL_TYPES = set(CELL_TYPE_TO_N_KEY)

# Legacy cell-type names remapped on load / sampling (region-aware for bare interneuron).
_CELL_TYPE_ALIASES: dict[str, str] = {
    "CA1_int": "INT_CA1",
    "interneuron": "INT_CA1",  # default when region context is absent
}

# Region / lab-band → local INT cell type for legacy "interneuron" mass.
_REGION_TO_LOCAL_INT: dict[str, str] = {
    "CA1": "INT_CA1",
    "CA2": "INT_CA2",
    "CA3": "INT_CA3",
    "DG": "INT_DG",
    "dentate_gyrus": "INT_DG",
    "Subiculum": "INT_SUB",
    "subiculum": "INT_SUB",
    "HPF_ProS_transition": "INT_SUB",
}


def canonicalize_cell_type(cell_type: str, region: str | None = None) -> str:
    """Map legacy cell-type labels onto the current vocabulary.

    Bare ``interneuron`` / ``CA1_int`` become the local INT for ``region`` when
    provided; otherwise they default to ``INT_CA1``.
    """
    raw = str(cell_type)
    if raw in ("interneuron", "CA1_int") and region is not None:
        return _REGION_TO_LOCAL_INT.get(str(region), "INT_CA1")
    return _CELL_TYPE_ALIASES.get(raw, raw)


def _rename_prob_weights(
    weights: dict[str, Any],
    region: str | None = None,
) -> dict[str, float]:
    """Merge legacy INT probability mass into the region-local INT type."""
    out: dict[str, float] = {}
    for ct, p in (weights or {}).items():
        key = canonicalize_cell_type(ct, region=region)
        out[key] = float(out.get(key, 0.0)) + float(p)
    return out


DEFAULT_UNIT_DENSITY_PER_CHANNEL = 0.8
DEFAULT_DETECTION_RADIUS_UM = 50.0


def load_cell_capture_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load a cell-capture YAML (lab or legacy) or return schematic defaults."""
    defaults: dict[str, Any] = {
        "unit_density_per_channel": DEFAULT_UNIT_DENSITY_PER_CHANNEL,
        "detection_radius_um": DEFAULT_DETECTION_RADIUS_UM,
        "amplitude_decay_length_um": 40.0,
        "oversample": {},
        "undersample": {},
        "capture_bias": {},
        "region_layer_probabilities": DEFAULT_CAPTURE_PROBS,
        "lab_region_capture": {},
        "recording_capture": {},
        "zero_uncrossed_populations": True,
        "format": "legacy",
        "_config_path": None,
    }
    if path is None:
        return defaults
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cell capture config not found: {path}")
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}

    cfg = dict(defaults)
    cfg["_config_path"] = str(path.resolve())

    # Lab trajectory format.
    if "cell_capture" in loaded and isinstance(loaded["cell_capture"], dict):
        cfg["format"] = "lab_regions"
        lab: dict[str, Any] = {}
        for region, block in loaded["cell_capture"].items():
            block = dict(block or {})
            probs = block.get("cell_type_probabilities") or {}
            block["cell_type_probabilities"] = _rename_prob_weights(probs, region=str(region))
            lab[str(region)] = block
        cfg["lab_region_capture"] = lab
        rec = loaded.get("recording_capture") or {}
        cfg["recording_capture"] = dict(rec)
        if "detection_radius_um" in rec:
            cfg["detection_radius_um"] = float(rec["detection_radius_um"])
        if "amplitude_decay_um" in rec:
            cfg["amplitude_decay_length_um"] = float(rec["amplitude_decay_um"])
        # Also expose as region_layer * entries for assign_probe_channels.
        probs: dict[str, dict[str, dict[str, float]]] = {}
        for region, block in cfg["lab_region_capture"].items():
            weights = {
                ct: float(p)
                for ct, p in (block.get("cell_type_probabilities") or {}).items()
                if ct in SIMULATED_CELL_TYPES
            }
            probs[region] = {"*": weights}
            # Also alias layer_or_area if present later via anatomy rows.
        cfg["region_layer_probabilities"] = probs
        return cfg

    # Legacy format.
    cfg.update({k: v for k, v in loaded.items() if k != "region_layer_probabilities"})
    probs = {k: dict(v) for k, v in DEFAULT_CAPTURE_PROBS.items()}
    for region, layers in (loaded.get("region_layer_probabilities") or {}).items():
        probs.setdefault(region, {})
        for layer, weights in (layers or {}).items():
            probs[region][layer] = _rename_prob_weights(weights, region=str(region))
    cfg["region_layer_probabilities"] = probs
    cfg["format"] = "legacy"
    return cfg


def crossed_regions(anatomy_df: pd.DataFrame) -> set[str]:
    """Regions with at least one assigned channel."""
    regions: set[str] = set()
    for _, row in anatomy_df.iterrows():
        if "include_in_hippocampal_simulation" in anatomy_df.columns:
            flag = row.get("include_in_hippocampal_simulation")
            if flag is False or (isinstance(flag, str) and flag.lower() in {"false", "0", "no"}):
                continue
        n_ch = row.get("n_channels")
        ch0 = row.get("channel_start")
        if pd.isna(ch0) or int(ch0 or 0) <= 0:
            continue
        if n_ch is not None and not pd.isna(n_ch) and int(n_ch) <= 0:
            continue
        regions.add(str(row["region"]))
    return regions


def _probs_for_band(cfg: dict[str, Any], region: str, layer: str) -> dict[str, float]:
    # Lab format: look up region block first.
    lab = cfg.get("lab_region_capture") or {}
    if region in lab:
        block = lab[region]
        if block.get("include") is False:
            return {}
        weights = _rename_prob_weights(
            block.get("cell_type_probabilities") or {}, region=region,
        )
        weights = {ct: p for ct, p in weights.items() if ct in SIMULATED_CELL_TYPES}
    else:
        table = cfg.get("region_layer_probabilities") or DEFAULT_CAPTURE_PROBS
        region_table = table.get(region) or table.get(str(region).upper()) or {}
        weights = region_table.get(layer) or region_table.get("*") or {}
        weights = _rename_prob_weights(weights, region=region)
        weights = {ct: p for ct, p in weights.items() if ct in SIMULATED_CELL_TYPES}

    if not weights:
        return {}
    bias = cfg.get("capture_bias") or {}
    out = {}
    for ct, p in weights.items():
        scale = float(bias.get(ct, 1.0))
        # Legacy bias keys for the CA1-local INT pool.
        if ct == "INT_CA1" and ct not in bias:
            for legacy in ("interneuron", "CA1_int"):
                if legacy in bias:
                    scale = float(bias[legacy])
                    break
        out[ct] = float(p) * scale
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def _band_depth_mm(row: pd.Series) -> float:
    if "depth_start_mm" in row.index and not pd.isna(row.get("depth_start_mm")):
        return float(row["depth_end_mm"]) - float(row["depth_start_mm"])
    z0, z1 = row.get("depth_start_um"), row.get("depth_end_um")
    if pd.isna(z0) or pd.isna(z1):
        return 0.0
    return (float(z1) - float(z0)) / 1000.0


def expected_cell_type_counts(
    anatomy_df: pd.DataFrame,
    capture_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Expected captured unit counts by cell type from region bands."""
    cfg = capture_cfg or load_cell_capture_config(None)
    counts: dict[str, float] = {ct: 0.0 for ct in CELL_TYPE_TO_N_KEY}
    lab = cfg.get("lab_region_capture") or {}
    use_lab = cfg.get("format") == "lab_regions" and bool(lab)

    for _, row in anatomy_df.iterrows():
        region = str(row["region"])
        layer = str(row.get("layer") or row.get("layer_or_area") or "*")
        if "include_in_hippocampal_simulation" in row.index:
            flag = row.get("include_in_hippocampal_simulation")
            if flag is False or (isinstance(flag, str) and flag.lower() in {"false", "0", "no"}):
                continue

        probs = _probs_for_band(cfg, region, layer)
        if not probs:
            continue

        if use_lab and region in lab:
            density = float(lab[region].get("unit_density_per_mm", 0.0))
            expected_n = density * _band_depth_mm(row)
        else:
            density = float(cfg.get("unit_density_per_channel", DEFAULT_UNIT_DENSITY_PER_CHANNEL))
            n_ch = row.get("n_channels")
            if pd.isna(n_ch) or int(n_ch) <= 0:
                ch0, ch1 = row.get("channel_start"), row.get("channel_end")
                if pd.isna(ch0) or pd.isna(ch1) or int(ch0) <= 0:
                    continue
                n_ch = int(ch1) - int(ch0) + 1
            else:
                n_ch = int(n_ch)
            expected_n = density * n_ch

        for ct, p in probs.items():
            counts[ct] = counts.get(ct, 0.0) + expected_n * p

    for ct, factor in (cfg.get("oversample") or {}).items():
        counts[ct] = counts.get(ct, 0.0) * float(factor)
    for ct, factor in (cfg.get("undersample") or {}).items():
        counts[ct] = counts.get(ct, 0.0) * float(factor)
    return counts


def apply_cell_capture_to_ratinabox_params(
    anatomy_df: pd.DataFrame,
    ratinabox_params: dict[str, Any],
    capture_cfg: dict[str, Any] | None = None,
    *,
    baseline_params: dict[str, Any] | None = None,
    scale_populations: bool = True,
) -> dict[str, Any]:
    """Scale / zero RatInABox population sizes from trajectory capture."""
    cfg = capture_cfg or load_cell_capture_config(None)
    rp = dict(ratinabox_params)
    expected = expected_cell_type_counts(anatomy_df, cfg)
    regions = crossed_regions(anatomy_df)
    use_lab = cfg.get("format") == "lab_regions"

    cell_type_regions = {
        "CA1_pyr": {"CA1", "HPF_ProS_transition"},
        "INT_CA1": {"CA1", "HPF_ProS_transition"},
        "INT_CA2": {"CA2"},
        "INT_CA3": {"CA3"},
        "INT_DG": {"DG", "dentate_gyrus"},
        "INT_SUB": {
            "Subiculum", "subiculum", "HPF_ProS_transition",
        },
        "CA2_pyr": {"CA2"},
        "CA3_pyr": {"CA3"},
        "DG_granule": {"DG", "dentate_gyrus"},
        "MEC_grid": {"MEC", "entorhinal_cortex", "deep_entorhinal_HATA"},
        "MEC_hd": {
            "MEC", "entorhinal_cortex", "deep_entorhinal_HATA",
            "HPF_ProS_transition", "subiculum", "dentate_gyrus",
        },
        "MEC_speed": {
            "MEC", "entorhinal_cortex", "deep_entorhinal_HATA",
            "subiculum", "dentate_gyrus",
        },
        "Sub_bvc": {
            "Subiculum", "subiculum", "HPF_ProS_transition",
            "entorhinal_cortex", "deep_entorhinal_HATA", "dentate_gyrus",
        },
    }

    baseline = baseline_params or {}
    for ct, n_key in CELL_TYPE_TO_N_KEY.items():
        base_n = int(rp.get(n_key, baseline.get(n_key, 0)))
        exp_n = float(expected.get(ct, 0.0))

        if use_lab:
            # Lab: presence is determined by expected capture mass.
            if exp_n <= 0:
                rp[n_key] = 0
                _sync_legacy_alias(rp, n_key, 0)
                continue
            if not scale_populations:
                continue
            new_n = int(max(1, round(min(base_n, exp_n)))) if base_n > 0 else int(max(1, round(exp_n)))
            new_n = min(base_n if base_n > 0 else new_n, new_n)
            if base_n > 0:
                new_n = min(base_n, max(1, int(round(min(base_n, exp_n)))))
            rp[n_key] = int(new_n)
            _sync_legacy_alias(rp, n_key, int(new_n))
            continue

        if cfg.get("zero_uncrossed_populations", True):
            if not (cell_type_regions.get(ct, set()) & regions):
                rp[n_key] = 0
                _sync_legacy_alias(rp, n_key, 0)
                continue
        if not scale_populations:
            continue
        if exp_n <= 0:
            rp[n_key] = 0
            _sync_legacy_alias(rp, n_key, 0)
            continue
        if base_n <= 0:
            new_n = int(max(0, round(exp_n)))
        else:
            scale = min(1.0, exp_n / max(base_n, 1.0))
            new_n = int(max(0, round(base_n * scale)))
            if exp_n >= 0.5 and new_n == 0:
                new_n = 1
            new_n = min(base_n, max(new_n, 1 if exp_n >= 0.5 else 0))
        rp[n_key] = int(new_n)
        _sync_legacy_alias(rp, n_key, int(new_n))
    return rp


def _sync_legacy_alias(rp: dict[str, Any], n_key: str, value: int) -> None:
    aliases = {
        "n_ca1_place_cells": "n_place_cells",
        "n_mec_hd_cells": "n_head_direction_cells",
        "n_sub_bvc_cells": "n_boundary_vector_cells",
        "n_mec_speed_cells": "n_speed_cells",
        "n_mec_grid_cells": "n_grid_cells",
        "n_int_ca1": "n_interneurons",
    }
    alias = aliases.get(n_key)
    if alias is not None and alias in rp:
        rp[alias] = value
    # Keep the older CA1-int count key in sync with INT_CA1.
    if n_key == "n_int_ca1" and "n_ca1_interneurons" in rp:
        rp["n_ca1_interneurons"] = value


def sample_units_from_regions(
    anatomy_df: pd.DataFrame,
    cell_capture_config: dict[str, Any] | None = None,
    *,
    rng: np.random.Generator | None = None,
    probe_config: dict[str, Any] | None = None,
    trajectory_config_name: str | None = None,
    include_non_hippocampal: bool = False,
) -> pd.DataFrame:
    """Sample a captured population table from trajectory-informed regions."""
    cfg = cell_capture_config or load_cell_capture_config(None)
    rng = rng or np.random.default_rng(0)
    probe_config = probe_config or {}
    pitch = float(probe_config.get("site_pitch_um", 15.0))
    rec = cfg.get("recording_capture") or {}
    decay = float(cfg.get("amplitude_decay_length_um", rec.get("amplitude_decay_um", 35.0)))
    radius = float(cfg.get("detection_radius_um", rec.get("detection_radius_um", 60.0)))
    amp_min = float(rec.get("min_template_amplitude_uv", 20.0))
    amp_max = float(rec.get("max_template_amplitude_uv", 200.0))

    rows: list[dict[str, Any]] = []
    unit_id = 0
    for _, band in anatomy_df.iterrows():
        include_flag = band.get("include_in_hippocampal_simulation", True)
        if isinstance(include_flag, str):
            include_flag = include_flag.lower() not in {"false", "0", "no"}
        if not include_flag and not include_non_hippocampal:
            continue

        region = str(band["region"])
        layer = str(band.get("layer") or band.get("layer_or_area") or "")
        acronym = str(band.get("acronym") or "")
        probs = _probs_for_band(cfg, region, layer)
        if not probs:
            continue

        ch0, ch1 = band.get("channel_start"), band.get("channel_end")
        if pd.isna(ch0) or pd.isna(ch1) or int(ch0) <= 0:
            continue
        channels = list(range(int(ch0), int(ch1) + 1))

        lab = (cfg.get("lab_region_capture") or {}).get(region)
        if lab is not None:
            density = float(lab.get("unit_density_per_mm", 0.0))
            n_units = int(rng.poisson(max(0.0, density * _band_depth_mm(band))))
        else:
            density = float(cfg.get("unit_density_per_channel", DEFAULT_UNIT_DENSITY_PER_CHANNEL))
            n_units = int(rng.poisson(max(0.0, density * len(channels))))

        types = list(probs.keys())
        weights = np.array([probs[t] for t in types], dtype=float)
        weights = weights / weights.sum()
        for _ in range(n_units):
            ct = str(rng.choice(types, p=weights))
            ch = int(rng.choice(channels))
            depth_um = float(ch - 1) * pitch
            offset_um = float(rng.uniform(0.0, radius))
            amp_scale = float(np.exp(-offset_um / max(decay, 1e-6)))
            amp = float(np.clip(amp_min + (amp_max - amp_min) * amp_scale, amp_min, amp_max))
            rows.append({
                "unit_id": unit_id,
                "cell_type": ct,
                "region": region,
                "layer": layer,
                "layer_or_area": layer,
                "acronym": acronym,
                "depth_um": depth_um,
                "channel": ch,
                "template_amplitude_uv": amp,
                "capture_source": "trajectory_config",
                "trajectory_config_name": trajectory_config_name or "",
                "include_in_decoder": bool(include_flag) or include_non_hippocampal,
                "detection_offset_um": offset_um,
            })
            unit_id += 1
    return pd.DataFrame(rows)


# Back-compat alias.
sample_captured_units = sample_units_from_regions


def annotate_anatomy_cell_types(
    anatomy_df: pd.DataFrame,
    capture_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fill ``cell_types`` column with dominant / listed capture classes."""
    cfg = capture_cfg or load_cell_capture_config(None)
    out = anatomy_df.copy()
    labels = []
    for _, row in out.iterrows():
        layer = str(row.get("layer") or row.get("layer_or_area") or "*")
        probs = _probs_for_band(cfg, str(row["region"]), layer)
        if not probs:
            # Fall back to candidate_cell_classes column if present.
            cand = row.get("candidate_cell_classes")
            labels.append(str(cand).replace(";", ",") if cand is not None and not pd.isna(cand) else "")
            continue
        ordered = sorted(probs.items(), key=lambda kv: -kv[1])
        labels.append(",".join(ct for ct, _ in ordered))
    out["cell_types"] = labels
    return out
