# Anatomy and trajectory modeling

Trajectories are **config-driven** (not hard-coded in simulation logic). Each trial snapshots the active insertion under `outputs/<run>/trajectory/`.

## Default lab NP2.0 trajectory

| Field | Value |
|-------|--------|
| Config | `configs/trajectories/lab_npx2_default.yaml` |
| Strain | C57/WT |
| Probe | NP2.0 (`site_pitch_um: 15`, confirm from channel map) |
| Bregma→lambda | 3.8 mm |
| AP | −3.967 mm from bregma |
| ML | 3.758 mm from bregma (right hemisphere inferred) |
| DV | ≈ 3.0 mm (**uncertain**) |
| Angle | 330h / 80V (**convention uncertain**) |
| Status | Approximate, screenshot-/trajectory-informed, **not histology-confirmed** |

Related files:

- Region–depth table: `configs/trajectories/lab_npx2_default_regions.csv` (VIS → HPF/ProS → SUB → DG_mo → ENT → deep ENT/HATA)
- Cell capture: `configs/trajectories/lab_npx2_default_cell_capture.yaml`
- Future template: `configs/trajectories/example_new_insertion.yaml`
- Optimal dorsal stack: `configs/trajectories/hpc_optimal.yaml` (+ `_regions.csv`, `_cell_capture.yaml`)

This default trajectory emphasizes **subiculum / entorhinal** capture more than a canonical CA1–CA3–DG stack. Visual cortex is crossed superficially but **excluded by default** from hippocampal decoder units.

## `hpc_optimal`

For decoder / figure runs that need every allowlisted cell type (including CA2 and CA3):

```bash
python run_simulation.py \
  --output outputs/hpc_optimal_001 \
  --trajectory hpc_optimal \
  --seed 1
```

`hpc_optimal` follows CA1 → CA2 → CA3 → DG → Subiculum → MEC along depth with plausible dorsal targeting (AP −2.0, ML 1.75, DV 2.6). Coordinates are anatomically motivated but **not histology-confirmed**.

## Analysis allowlist

Decoding and manifold features use **only** RatInABox-modeled hippocampal / MEC-afferent cell types:

`CA1_pyr`, `INT_CA1`, `INT_CA2`, `INT_CA3`, `INT_DG`, `INT_SUB`, `CA2_pyr`, `CA3_pyr`, `DG_granule`, `Sub_bvc`, `MEC_grid`, `MEC_hd`, `MEC_speed`.

Units outside that system may appear on the trajectory figure if the probe crosses those bands, but they are marked `include_in_decoder=false` and are **dropped** when loading data for decoder comparison / PCA / Isomap / realtime replay. Opt in during simulation with `--include-non-hippocampal-regions`. Analysis loaders still default to the allowlist unless overridden in code.

Lab region labels (`subiculum`, `entorhinal_cortex`, …) are canonicalized to `Subiculum` / `MEC` / … for `region_pca` and anatomical partitions.

## Selecting a trajectory

```bash
python run_simulation.py --list-trajectories

python run_simulation.py \
  --output outputs/ratinabox_006 \
  --trajectory lab_npx2_default \
  --seed 1
```

`--trajectory` accepts a **name** under `configs/trajectories/` or a full YAML path. Default is `lab_npx2_default`.

## Per-trial coordinate bundle

| Path | Contents |
|------|----------|
| `trajectory/active.json` | Which insertion is active (name, AP/ML/DV, paths) |
| `trajectory/active_trajectory.yaml` | Snapshot of the config used for this trial |
| `trajectory/anatomy_regions_used.csv` | Region–depth table driving capture |
| `trajectory/cell_capture.yaml` | Copy of cell-capture rules (if any) |
| `trajectory_metadata.json` | Same coords + uncertainty flags at trial root |
| `figures/trajectory/probe_trajectory_*.png` | Probe figures for this trial |
| `figures/trajectory/unit_count_*.png`, `channel_region_map.png` | Capture / channel diagnostics |

Do **not** keep a separate `outputs/lab_npx2_default/` tree for coords — attach them to the trial with `--output`.

Also writes:

- `anatomy_regions.csv` — depth (µm) + channel ranges from site pitch
- `units.csv` — trajectory-informed regions / cell types
- `figures/trajectory/probe_trajectory_regions.png` (+ `.pdf`), `unit_count_by_region.png`, `unit_count_by_cell_type.png`, `channel_region_map.png`

## Future insertions (no code edits)

```bash
cp configs/trajectories/example_new_insertion.yaml \
   configs/trajectories/my_new_insertion.yaml
# Edit AP/ML/DV/angles/probe; point anatomy_regions_file or trajectory_export_file
# at a new CSV or Neuropixels Trajectory Explorer export.

python run_simulation.py \
  --output outputs/new_insertion_001 \
  --trajectory my_new_insertion
```

## Simulation trajectory CLI flags

| Flag | Effect |
|------|--------|
| `--trajectory NAME\|PATH` | Select active lab coordinates (`--trajectory-config` / `--trajectory-name` aliases) |
| `--list-trajectories` | Print available insertion configs |
| `--trajectory-export PATH` | Prefer NTE export over the approximate region CSV |
| `--anatomy-regions-file PATH` | Override region–depth table |
| `--cell-capture-config PATH` | Override capture probabilities |
| `--include-non-hippocampal-regions` | Keep visual-cortex (etc.) in the capture model |
| `--fallback-schematic-anatomy` | Allow old CA1–MEC schematic if anatomy missing |
| `--no-trajectory` | Force schematic hippocampal geometry |

## Workflow

```text
trajectory config
  → anatomy region-depth table (or NTE export)
  → cell-type capture config
  → simulated units + channel assignments
  → Neuropixels-like recording degradation
  → Open Ephys / Kilosort-like sorted spikes
  → sorted-spike-only decoder selection
  → real-time deployment models
  → manifold visualization
```

## Neuropixels Trajectory Explorer (optional GUI)

The simulation does **not** open the GUI. Plan coordinates separately, then point a config / export at this project:

```bash
scripts/launch_trajectory_explorer.sh
# or: git clone https://github.com/petersaj/neuropixels_trajectory_explorer \
#          external/neuropixels_trajectory_explorer
```

## Visualizing probe trajectory only

Publication figures use the **same** region-depth table that drives simulated cell capture.

```bash
python -m hippo.visualization.probe_trajectory \
  --trajectory lab_npx2_default \
  --output outputs/ratinabox_006 \
  --make-3d
```

Optional: `--nte-export path/to/save.mat`, `--include-non-hippocampal-regions`, `--no-use-nte-style`, `--list-trajectories`.

| File | Description |
|------|-------------|
| `figures/trajectory/probe_trajectory_regions.png` / `.pdf` | Region-depth strip with probe shank + channel markers |
| `figures/trajectory/probe_areas_nte_style.png` / `.pdf` | NTE-style probe-areas strip (Python; MATLAB if available) |
| `figures/trajectory/probe_trajectory_3d.png` / `.pdf` | Optional coordinate-space 3D path (`--make-3d`) |
| `trajectory/anatomy_regions_used.csv` | Region-depth table shared with the simulation |
| `trajectory/active.json` | Active insertion name + AP/ML/DV for this trial |

If MATLAB / the NTE repo is missing, the CLI still writes the Python figures.

## What remains approximate

- DV and angle convention are flagged uncertain in the default config
- Screenshot-derived region depths until a real NTE export / histology registration replaces them
- Cell-type probabilities are configurable priors
- NP2.0 site pitch should be confirmed from the channel map
- Multi-shank selection (`shank_count` / `selected_shank`) is reserved in YAML for future use
- 3D plot is coordinate-space only (not a full Allen CCF mesh) unless NTE CCF endpoints are imported
- NTE exports **override** the screenshot-derived CSV when provided
