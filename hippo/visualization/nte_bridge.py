"""Discover / import Neuropixels Trajectory Explorer assets when present.

The NTE repo is optional. Typical locations::

    external/neuropixels_trajectory_explorer
    neuropixels_trajectory_explorer
    ../neuropixels_trajectory_explorer

Native saves are MATLAB ``.mat`` files with ``probe_areas`` (tip_distance,
structure-tree fields, probe_shank) and ``probe_positions_ccf``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NTE_CANDIDATE_ROOTS = [
    Path("external/neuropixels_trajectory_explorer"),
    Path("neuropixels_trajectory_explorer"),
    Path("../neuropixels_trajectory_explorer"),
]

NTE_REPO_URL = "https://github.com/petersaj/neuropixels_trajectory_explorer"


def find_nte_repository(search_roots: list[Path] | None = None) -> Path | None:
    """Return the first existing NTE repo root, or None."""
    roots = search_roots or NTE_CANDIDATE_ROOTS
    for root in roots:
        p = Path(root)
        if p.is_dir() and any(p.rglob("*.m")):
            return p.resolve()
    return None


def matlab_available() -> bool:
    return shutil.which("matlab") is not None


def _as_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [obj.flat[i] for i in range(obj.size)]
        return list(np.atleast_1d(obj).ravel())
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def _field(struct: Any, *names: str, default: Any = None) -> Any:
    if struct is None:
        return default
    if isinstance(struct, dict):
        for name in names:
            if name in struct and struct[name] is not None:
                return struct[name]
        return default
    for name in names:
        if hasattr(struct, name):
            val = getattr(struct, name)
            if val is not None:
                return val
    return default


def _name_from_structure(entry: Any) -> tuple[str, str, str]:
    """Return (acronym, full_name, parent) from a CCF structure-tree-like entry."""
    if entry is None:
        return "UNK", "unknown", ""
    if isinstance(entry, (str, bytes)):
        s = entry.decode() if isinstance(entry, bytes) else entry
        return s, s, ""
    if isinstance(entry, dict):
        acronym = str(entry.get("acronym") or entry.get("safe_name") or entry.get("name") or "UNK")
        full = str(entry.get("name") or entry.get("safe_name") or acronym)
        parent = str(entry.get("parent") or entry.get("parent_structure_id") or "")
        return acronym, full, parent
    acronym = str(_field(entry, "acronym", "safe_name", "name", default="UNK"))
    full = str(_field(entry, "name", "safe_name", "acronym", default=acronym))
    parent = str(_field(entry, "parent", "parent_structure_id", default="") or "")
    return acronym, full, parent


def import_nte_probe_areas_mat(
    path: Path | str,
    *,
    site_pitch_um: float = 15.0,
    n_channels: int = 384,
    active_length_um: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert an NTE ``.mat`` save into a standard anatomy DataFrame.

    tip_distance is distance from shank tip (0 at tip). We convert to
    dorsal-origin depth along the active site span.
    """
    from scipy.io import loadmat

    path = Path(path)
    mat = loadmat(str(path), simplify_cells=True)
    if "probe_areas" not in mat:
        raise ValueError(f"{path} has no probe_areas (NTE save expected)")

    active_len = float(active_length_um) if active_length_um is not None else float(n_channels) * float(site_pitch_um)
    rows: list[dict[str, Any]] = []
    probes = _as_list(mat.get("probe_areas"))
    for probe_idx, probe in enumerate(probes):
        if probe is None:
            continue
        tip = _field(probe, "tip_distance")
        shank = _field(probe, "probe_shank", default=1)
        # Structure arrays may be parallel fields or nested list of structs.
        structs = _field(probe, "st", "structure", "structures", "area", default=None)
        names = _field(probe, "name", "acronym", "safe_name", default=None)

        tip_arr = np.asarray(_as_list(tip), dtype=float).ravel() if tip is not None else np.array([])
        if tip_arr.size < 2:
            continue

        # Build per-interval labels.
        n_intervals = tip_arr.size - 1
        labels: list[tuple[str, str, str]] = []
        if structs is not None:
            for entry in _as_list(structs)[:n_intervals]:
                labels.append(_name_from_structure(entry))
        elif names is not None:
            for entry in _as_list(names)[:n_intervals]:
                labels.append(_name_from_structure(entry))
        else:
            labels = [(f"area_{i}", f"area_{i}", "") for i in range(n_intervals)]

        while len(labels) < n_intervals:
            labels.append((f"area_{len(labels)}", f"area_{len(labels)}", ""))

        shank_list = _as_list(shank) if shank is not None else [1]
        for i in range(n_intervals):
            tip0, tip1 = float(tip_arr[i]), float(tip_arr[i + 1])
            # Convert tip-distance (0=tip) → dorsal-origin depth along active sites.
            # Assume tip_distance units are µm if values look large, else mm.
            scale = 1000.0 if max(abs(tip0), abs(tip1)) < 20 else 1.0
            tip0_um, tip1_um = tip0 * scale, tip1 * scale
            d0 = active_len - max(tip0_um, tip1_um)
            d1 = active_len - min(tip0_um, tip1_um)
            depth_start_um = float(min(d0, d1))
            depth_end_um = float(max(d0, d1))
            acronym, full, parent = labels[i]
            shank_id = shank_list[min(i, len(shank_list) - 1)] if shank_list else 1
            rows.append({
                "depth_start_um": depth_start_um,
                "depth_end_um": depth_end_um,
                "depth_start_mm": depth_start_um / 1000.0,
                "depth_end_mm": depth_end_um / 1000.0,
                "region": full,
                "acronym": acronym,
                "layer_or_area": acronym,
                "parent_structure": parent,
                "probe_shank": int(shank_id) if str(shank_id).isdigit() else shank_id,
                "include_in_hippocampal_simulation": _is_hippocampal_acronym(acronym, full),
                "candidate_cell_classes": _candidate_classes(acronym, full),
                "source": "neuropixels_trajectory_explorer_mat",
                "notes": f"Imported from {path.name} probe={probe_idx}",
            })

    positions_meta: dict[str, Any] = {}
    positions = mat.get("probe_positions_ccf")
    if positions is not None:
        try:
            first = _as_list(positions)[0]
            arr = np.asarray(first, dtype=float)
            if arr.ndim >= 2 and arr.shape[0] >= 3:
                positions_meta = {
                    "ccf_ap_start_um": float(arr[0, 0]),
                    "ccf_ap_end_um": float(arr[0, 1]) if arr.shape[1] > 1 else float(arr[0, 0]),
                    "ccf_dv_start_um": float(arr[1, 0]),
                    "ccf_dv_end_um": float(arr[1, 1]) if arr.shape[1] > 1 else float(arr[1, 0]),
                    "ccf_ml_start_um": float(arr[2, 0]),
                    "ccf_ml_end_um": float(arr[2, 1]) if arr.shape[1] > 1 else float(arr[2, 0]),
                }
        except Exception:
            pass

    if not rows:
        raise ValueError(f"Could not parse probe_areas intervals from {path}")

    df = pd.DataFrame(rows).sort_values("depth_start_um").reset_index(drop=True)
    return df, {"nte_mat": str(path.resolve()), "probe_positions_ccf": positions_meta}


def _is_hippocampal_acronym(acronym: str, full: str) -> bool:
    text = f"{acronym} {full}".lower()
    keys = (
        "ca1", "ca2", "ca3", "dg", "dentate", "subiculum", "sub", "pros",
        "hpf", "ent", "mec", "lec", "hata", "parasubiculum", "postsubiculum",
    )
    vis = ("vis", "visual", "ctx", "cortex")
    if any(k in text for k in keys):
        # Exclude pure visual cortex even if "ctx" matched via hippocampal context.
        if any(v in text for v in ("visp", "visl", "visual cortex")) and not any(
            k in text for k in ("ca1", "ca2", "ca3", "dg", "sub", "ent", "hpf")
        ):
            return False
        return True
    if any(v in text for v in vis) and "hippocamp" not in text:
        return False
    return False


def _candidate_classes(acronym: str, full: str) -> str:
    text = f"{acronym} {full}".lower()
    if "dg" in text or "dentate" in text:
        return "DG_granule;INT_DG"
    if "ca1" in text:
        return "CA1_pyr;INT_CA1"
    if "ca2" in text:
        return "CA2_pyr;INT_CA2"
    if "ca3" in text:
        return "CA3_pyr;INT_CA3"
    if "sub" in text or "pros" in text:
        return "Sub_bvc;INT_SUB;MEC_hd"
    if "ent" in text or "mec" in text or "hata" in text:
        return "MEC_grid;MEC_hd;MEC_speed;Sub_bvc"
    return ""


def try_matlab_nte_probe_areas_export(
    nte_repo: Path,
    mat_export: Path,
    output_png: Path,
) -> bool:
    """Best-effort MATLAB export of an NTE-style probe-areas figure.

    Returns True if a PNG was produced. Skips gracefully when MATLAB is absent
    or the call fails — NTE does not ship a documented headless plot API.
    """
    if not matlab_available():
        return False
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    # Documented save format is loadable; plotting GUI internals vary by version.
    # We emit a minimal MATLAB script that loads tip_distance and draws bands.
    script = output_png.with_suffix(".m")
    script.write_text(
        f"""
addpath(genpath('{nte_repo.as_posix()}'));
S = load('{Path(mat_export).as_posix()}');
if ~isfield(S, 'probe_areas'); error('no probe_areas'); end
pa = S.probe_areas;
if iscell(pa); pa = pa{{1}}; end
td = pa.tip_distance(:);
figure('Visible','off','Color','w');
hold on;
cmap = lines(max(1, numel(td)-1));
for i = 1:(numel(td)-1)
  y0 = td(i); y1 = td(i+1);
  fill([0 1 1 0], [y0 y0 y1 y1], cmap(i,:), 'EdgeColor','k');
end
set(gca, 'YDir', 'reverse');
xlabel('Probe areas (NTE-style)'); ylabel('tip\\_distance');
title('NTE probe areas');
exportgraphics(gcf, '{output_png.as_posix()}', 'Resolution', 300);
close(gcf);
"""
    )
    try:
        subprocess.run(
            [
                "matlab", "-batch",
                f"run('{script.as_posix()}')",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return output_png.exists()
    except Exception:
        return False


def nte_clone_instructions() -> str:
    return (
        "Neuropixels Trajectory Explorer not found locally.\n"
        f"  git clone {NTE_REPO_URL} external/neuropixels_trajectory_explorer\n"
        "Then save a trajectory from the GUI (File → Save) and pass --nte-export."
    )
