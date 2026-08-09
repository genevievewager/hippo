# Neuropixels recording and spike-sorting model

Ground-truth Poisson spikes are passed through a multi-channel Neuropixels observation model and a Kilosort-like re-extraction step before deployment-relevant decoding.

```text
ground-truth spikes
→ multi-channel Neuropixels templates
→ recording noise / amplitude drift / collisions
→ misses / jitter / contamination / merges
→ sorted spikes
```

## Deployment rule

**Deployable model selection uses sorted spikes only.**

Ground-truth spikes are oracle / diagnostic / non-deployable. Optional oracle comparisons are available via `--include-ground-truth-diagnostics` on `run_decoder.py`. Realtime never loads ground-truth-selected models.

Config flags in trajectory YAMLs:

```yaml
decoder:
  deployment_spike_source: sorted
  allow_ground_truth_diagnostics: false
  use_ground_truth_for_model_selection: false
```

## Probe defaults

| Context | Probe | Channels | Site pitch |
|---------|-------|----------|------------|
| Schematic anatomy (`--no-trajectory`) | Neuropixels 1.0 (`hippo_sim/config.py`) | 384 | 20 µm |
| Default lab trajectory (`lab_npx2_default`) | NP2.0 | 384 | 15 µm (confirm from channel map) |

Sample rate: 30 kHz.

## Recording parameters

From `hippo_sim/config.py` → `RECORDING_PARAMS`:

| Parameter | Value |
|-----------|-------|
| Template span | 3–10 channels |
| Amplitude range | 20–200 µV |
| Noise σ | 15 µV |
| Noise correlation | 0.3 |
| Motion amplitude drift | 0.15 / min |
| Overlap collision probability | 0.08 |
| Burst noise probability | 0.002 |
| Burst noise amplitude | 80 µV |

## Sorting parameters

From `hippo_sim/config.py` → `SORTING_PARAMS`:

| Parameter | Value |
|-----------|-------|
| Detection threshold | 25 µV |
| Match correlation threshold | 0.85 |
| `miss_rate` | 0.12 (12%) |
| `false_positive_rate` | 0.005 |
| `jitter_ms` | 0.3 |
| `contamination_rate` | 0.08 |
| `merge_prob` | 0.04 |

## Modeled degradation sources (summary)

- Multi-channel template spread and amplitude variation
- Additive / correlated recording noise and burst noise
- Motion-related amplitude drift
- Spike collisions / overlaps
- Detection misses and false positives
- Spike-time jitter
- Unit contamination and merges

Deployable models must survive these degradations on sorted spikes. Sorted-vs-GT information-loss summaries are written when GT diagnostics are enabled.
