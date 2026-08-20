# CLI reference

Public happy path remains in the root [README](../README.md). This page documents profiles, advanced flags, and developer grid commands.

## Profiles (`run_decoder.py --profile`)

| Profile | `W` grid | Feature modes (summary) | `max-models` | Isomap distillation | Notes |
|---------|----------|-------------------------|--------------|---------------------|-------|
| `manifolds` (**default**) | `[0.050, 0.250, 0.500, 1.000]` | counts + global/region PCA + classic/distilled Isomap + `diffusion_nystrom` | quick | on | Day-to-day RT embedding search |
| `quick` | coarse grid above | counts, global_pca, region_pca | quick | off | Smoke test |
| `standard` | `[0.050, 0.100, 0.250, 0.500, 1.000]` | counts, global_pca, region_pca | quick | off | Faster counts+PCA, denser W |
| `full` | `[0.025, …, 1.000]` | counts + PCA family (profile modes; override for more) | full | off | Dense research; temporal still opt-in |
| `feature_robustness` | full default windows | F×E grid + ablation | quick | on | Modular full grid on existing sim outputs |

Explicit CLI flags always override profile defaults. Source: `realtime/workflow_profiles.py`.

## Public scripts

### `run_simulation.py`

```bash
python run_simulation.py \
    --output outputs/ratinabox_001 \
    --trajectory hpc_optimal \
    --seed 1
```

| Flag | Purpose |
|------|---------|
| `--output` | Trial directory |
| `--seed` | Random seed |
| `--duration` | Session duration seconds (default 600) |
| `--trajectory` / `--trajectory-config` / `--trajectory-name` | Insertion config name or YAML path |
| `--list-trajectories` | List configs and exit |
| `--trajectory-export` | NTE export override |
| `--anatomy-regions-file` | Override region CSV |
| `--cell-capture-config` | Override capture YAML |
| `--include-non-hippocampal-regions` | Keep non-HPC capture |
| `--fallback-schematic-anatomy` | Schematic if anatomy missing |
| `--no-trajectory` | Force schematic geometry |

Default trajectory: `lab_npx2_default`.

### `run_decoder.py`

```bash
python run_decoder.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001 \
    --skip-visualization
```

| Flag | Purpose |
|------|---------|
| `--profile` | `manifolds` / `standard` / `quick` / `full` / `feature_robustness` |
| `--deployment-only` | Sorted-only deployable selection (default) |
| `--include-ground-truth-diagnostics` | Also run GT oracle comparisons (non-deployable) |
| `--compare-sources` | Deprecated alias for GT diagnostics |
| `--closed-loop-target` | Primary realtime / trigger target (default `position`) |
| `--selection-policy` | `shortest_near_optimal` or `best_accuracy` |
| `--decode-windows` | Override `W` grid |
| `--update-dt` | Decoder update interval (default **0.050** s) |
| `--behavior-rate` | Derive update_dt from Hz when needed |
| `--feature-modes` | Override observation / manifold / dynamic modes |
| `--max-models` | `quick` or `full` |
| `--enable-isomap-distillation` | Add classic + distilled Isomap |
| `--enable-temporal-manifold` | W×L temporal comparison |
| `--manifold-components-list` | Latent-dim grid |
| `--isomap-neighbors` | Neighbor grid for `global_isomap` |
| `--isomap-latent-dim` | Latent dim for Isomap / distilled |
| `--n-landmarks` | Landmark-count grid for `diffusion_nystrom` (default 512) |
| `--landmark-method` | `random` / `kmeans` / `minibatch_kmeans` |
| `--diffusion-components` | Diffusion latent dim if the shared k-grid is omitted |
| `--diffusion-local-scale-k` | Self-tuning kernel neighbor k (default 10) |
| `--diffusion-alpha` | Density-normalization α (default 1.0) |
| `--diffusion-time` | Diffusion time τ (default 1) |
| `--benchmark-diffusion-landmarks` | Write landmark-count vs accuracy/latency CSV |
| `--skip-comparison` | Reuse existing `decoder_comparison/` |
| `--skip-visualization` | Skip in-decode figures (preferred) |
| `--compile-pdf` | Optional in-decode PDF |
| `--adaptive-windows` | After coarse W pass, densify near optima |

### `run_visualizations.py`

```bash
python run_visualizations.py \
    --experiment outputs/ratinabox_001 \
    --all \
    --compile-pdf
```

See [visualizations.md](visualizations.md).

## Advanced examples

### Detached / background simulation

```bash
mkdir -p outputs/run_001
nohup .hippo/bin/python run_simulation.py --output outputs/run_001 \
  > outputs/run_001/simulation.log 2>&1 &
echo $! > outputs/run_001/simulation.pid
tail -f outputs/run_001/simulation.log
```

### Profile variants

```bash
# Faster counts+PCA smoke
python run_decoder.py --input OUTPUTS --output OUTPUTS --profile standard --skip-visualization

# Dense temporal
python run_decoder.py --input OUTPUTS --output OUTPUTS --profile full \
  --enable-temporal-manifold --skip-visualization

# Oracle GT diagnostics (deployable still from sorted)
python run_decoder.py --input OUTPUTS --output OUTPUTS \
  --include-ground-truth-diagnostics --skip-visualization
```

### Comparison-only grid

```bash
python run_decoder_comparison.py \
    --input outputs/ratinabox_002 \
    --output outputs/ratinabox_002/decoder_comparison \
    --compare-sources \
    --decode-windows 0.025 0.050 0.100 0.250 0.500 1.000
```

Static + dynamic grids: see [dynamic_latents.md](dynamic_latents.md).

Isomap-specific grids: see [manifolds.md](manifolds.md).

Diffusion + Nyström grids: see [diffusion_nystrom.md](diffusion_nystrom.md).

Trajectory flags: see [anatomy_and_trajectory.md](anatomy_and_trajectory.md).

Live bundle packing, Replay, and Open Ephys are **not** CLI flags. Use Streamlit **Live Deployment** (`realtime/live_decoder.py`). See [realtime_deployment.md](realtime_deployment.md#live-runtime).
