# Dynamic latent representations

Dynamic latent-state models carry temporal state `z_(t−1)` when forming `z_t`. They are an additive pathway for scientific comparison against static manifolds under the same decoder zoo.

**Numeric head-to-head results (static vs dynamic) are TBD** until a local comparison is run and recorded. LDS / GPFA are implemented and runnable; that is not a claim that dynamic latents outperform static manifolds.

## Definitions

### Static manifold

```text
x_t → z_t
```

Examples: `counts`, `global_pca`, `region_pca`, `global_isomap`, `global_isomap_distilled`, `diffusion_nystrom`.

### Dynamic latent state

```text
z_(t−1), x_t → z_t
```

The latent representation is dynamic; the downstream behavioral decoder may remain a fixed static map `z_t → ŷ_t`.

| Method | Realtime? | Causal transform? | Notes |
|--------|-----------|-------------------|-------|
| `global_lds` | Yes (`REALTIME / CAUSAL`) | Filtered Kalman | Linear Gaussian LDS; deployable via `step()` |
| `gpfa` | No (`OFFLINE / ACAUSAL`) | Smoothed (default) | Offline GPFA-style benchmark; not for realtime |

Registry: `realtime/dynamic_latents/` (`DYNAMIC_LATENT_REGISTRY`). Future slots reserved: `plds`, `switching_lds`, `recurrent_slds`, `lfads`, `adaptive_lds`.

## Dynamic latent ≠ neural plasticity

A standard LDS models a changing neural state `z_t` while assuming a fixed observation mapping `C`. Future plasticity modeling may allow a slowly varying `C_t` so neuron-to-latent mapping drift can be separated from latent-state changes. Serialization already reserves a `C_t` field.

## Offline-only modes

`gpfa` and classic `global_isomap` are offline-only (`OFFLINE_ONLY_FEATURE_MODES`). They can appear in comparison metrics but are rejected for realtime closed-loop replay / auto-deployment.

## Train LDS + decode comparison

```bash
python run_decoder_comparison.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001/decoder_comparison/dynamic \
    --feature-sets counts \
    --manifolds counts global_pca \
    --dynamic-latents global_lds \
    --dynamic-latent-dims 3 5 10 \
    --decode-windows 0.100 0.250 \
    --update-dt 0.025 \
    --max-models quick
```

Examples below use `--update-dt 0.025` (40 Hz) for denser dynamic-state stepping. The public default cadence remains **0.050 s (20 Hz)**; both are supported via CLI / UI.

## GPFA offline benchmark

```bash
python run_decoder_comparison.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001/decoder_comparison/dynamic \
    --feature-sets counts \
    --dynamic-latents gpfa \
    --dynamic-latent-dims 3 5 \
    --decode-windows 0.250 \
    --update-dt 0.025 \
    --max-models quick
```

## Static vs dynamic comparison (same run)

```bash
python run_decoder_comparison.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001/decoder_comparison \
    --feature-sets counts \
    --manifolds counts global_pca region_pca global_isomap global_isomap_distilled diffusion_nystrom \
    --dynamic-latents global_lds gpfa \
    --dynamic-latent-dims 3 5 10 \
    --decode-windows 0.025 0.050 0.100 0.250 \
    --update-dt 0.025 \
    --max-models quick
```

Group results by `representation_family`, `embedding_type`, `feature_set`, `decoder_name`, `target_name`, `decode_window_s`, `latent_dimension` / `manifold_n_components`, `spike_source`, and `causal_status`.

## Realtime LDS replay

```bash
python run_realtime_decoding.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001/realtime_decoding/dynamic \
    --spike-source sorted \
    --update-dt 0.025 \
    --decode-window 0.250 \
    --representation global_lds \
    --dynamic-latent-dims 5
```

Offline-only methods (`gpfa`, `global_isomap`) are rejected for realtime replay.

If using the developer helper:

```bash
python run_realtime_decoding.py \
    --input outputs/ratinabox_001 \
    --output outputs/ratinabox_001/realtime_decoding/dynamic \
    --spike-source sorted \
    --update-dt 0.025 \
    --decode-window 0.250 \
    --representation global_lds \
    --dynamic-latent-dims 5
```

(An archived copy also exists under `archive/`. Prefer Streamlit **Realtime Replay** or the public `run_decoder.py` registry path for deployable sorted-spike winners.)

## UI

```bash
streamlit run ui/app.py
```

- **Latent Representations**: four class tabs (static/dynamic × linear/nonlinear). GPFA is under Dynamic linear (offline). Dynamic nonlinear is a placeholder.
- **Decoder Benchmark**: Continuous vs Discrete target tabs; representation pickers grouped by the same four classes.
- **Realtime Replay**: one behavior × three realtime-capable quadrants (`global_pca`, `diffusion_nystrom`, `global_lds`) with stability and behavior/latency figures. GPFA cannot be launched as a realtime model.

## Outputs

```text
decoder_comparison/
  dynamic/<method>/behavioral_association_*.csv
  models/manifold_transforms/...   # includes DynamicLatentEmbedding meta + params
figures/dynamic/<method>/
  figA_latent_trajectory_time__*.png
  figB_latent_trajectory_position__*.png
  figC_latent_trajectory_reward__*.png   # if reward labels exist
  figD_latent_trajectory_wall__*.png
  figE_latent_timeseries__*.png
  figF_static_vs_dynamic_prediction__*.png  # when predictions provided
realtime_decoding/dynamic/
  decoded_realtime.csv   # includes z1,z2,… when using LDS
```

Latent metrics (when valid): reconstruction error, one-step latent prediction error, log-likelihood, smoothness / velocity / acceleration, state-transition magnitude. Behavioral association tables report R² / accuracy per latent dimension × behavioral variable — labeled as association, not latent biological meaning.
