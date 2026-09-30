# Pipeline contracts

This document records **stage boundaries**. It does not require a new Python
class for every heading. Equivalent types already live in
`realtime/pipeline_artifacts.py`.

Scientific analysis configuration is **not** Streamlit session state.
Visualization flags, tab selection, and job chrome are **not** analysis
configuration.

---

## AnalysisConfig

Authoritative experimental overlay on a committed neural observation.

**Python:** `realtime.pipeline_artifacts.AnalysisConfig`

| Field | Owner | Notes |
| ----- | ----- | ----- |
| `observation` | Feature Construction / `ObservationConfig` | Owns `W`, `update_dt`, feature set, spike source |
| `feature_mode` | derived from `observation.feature_mode` | Not a second copy of `F` |
| `representation` | Latent Representations / Quadrant Comparison | Method name (`global_pca`, …) |
| `decoder` | Decoder Benchmark / Quadrant Comparison | Family name (`ridge`, …) |
| `target` | same | Behavioral target |
| `causal` | analysis config | Default `True`. Do not silently flip. |
| `realtime` | optional | If `True`, representation must be realtime-capable |
| `train_frac` / `seed` | analysis config | Train/test identity; also on `ObservationConfig.fit_hash()` |
| `window_s` | **property of `observation`** | Downstream pages display or sweep; they do not own it |

**Hash:** `AnalysisConfig.hash()` covers observation identity plus E / D /
target / causal / train split. `ObservationConfig.hash()` is the observation
only (changing decoder must not invalidate feature caches).

**Active vs grid:** one `AnalysisConfig` is the current pipeline. A window
sweep or F×E×D×W benchmark is a **list of experiments**, each with its own
observation / config hash. Do not mix them in one results table.

---

## FeatureResult (`FeatureDataset`)

Causal observation matrix produced from spikes in `[t−W, t)`.

**Python:** `realtime.pipeline_artifacts.FeatureDataset`

Conceptually includes:

* timestamps
* feature matrix `X`
* feature names / channel information
* feature mode (`observation.feature_set` / `feature_mode`)
* window size (`observation.window_s`)
* causal metadata (causal bins; `W` is history, not a centered window)
* provenance (`observation.hash()`, `fit_hash()`, optional `ArtifactProvenance`)

Downstream stages **inherit** this object. They do not rebuild `W` from UI
widgets.

---

## RepresentationResult

Fitted transform `E` applied to a `FeatureDataset`.

**Python:** `realtime.pipeline_artifacts.RepresentationResult`

Conceptually includes:

* timestamps (must align with the source features)
* latent matrix `Z`
* fitted transform/model
* representation name
* linear vs nonlinear and static vs dynamic (from
  `realtime.representation_registry.RepresentationSpec`, not ad-hoc strings)
* realtime capability / offline-only
* provenance: `source_feature_hash`, inherited `window_s` / `update_dt`

A representation must not be displayed as matching the active pipeline if
`source_feature_hash` disagrees with the active observation hash.

---

## DecoderResult

Fitted decoder `D` plus enough provenance to replay the same observation.

**Python:** `realtime.pipeline_artifacts.DecoderResult`

Conceptually includes:

* target
* observation (includes `W`)
* `y_true` / `y_pred` (metrics and optional prediction artifacts)
* metrics
* fitted decoder
* latency information when a realtime/deployment path recorded it
* representation name and `source_feature_hash` /
  `source_representation_hash`
* `analysis_config_hash` in `to_meta()` so UI cannot treat a different
  (F, E, D, W, target, split) as the same run

---

## ObservationConfig

Neural observation `O(W, F)`. This is **not** replaced by `AnalysisConfig`.
`AnalysisConfig` wraps it.

`W` is committed on Feature Construction (`PipelineRun.set_observation` /
`commit_observation`) and stored in `pipeline_run.json`.

---

## ComparisonRunConfig

CLI/UI **benchmark grid** (`F × E × D × W`). Not the active pipeline
configuration. Window lists here are a deliberate sweep, not a silent
override of `PipelineRun.observation`.

---

## QuadrantExperimentConfig

Controlled four-quadrant comparison: **one** shared observation and train/test
split, several representation branches, one decoder family and target.
Computation delegates to `run_decoder_comparison`. A labeled window sweep is
separate experiments per `W`.

---

## Runtime / deployment configuration

Live bundles, closed-loop policies, and Replay session settings are
deployment (`C`). They consume a fitted `DecoderResult` whose observation
must match replay. They are not visualization config and not a second
analysis config.

---

## UI state

Allowed in `st.session_state`: selected tab, expanders, plot toggles,
pagination, job request flags, active dataset path.

Not allowed as a second source of truth: a session-state `window_s` that can
diverge from `PipelineRun.observation.window_s`. Overlay dicts may cache
representation / decoder / target; `AnalysisConfig.resolve()` always takes
`W` from the committed observation.
