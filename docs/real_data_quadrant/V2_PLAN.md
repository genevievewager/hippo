# Report 2 v2 — HF-only plan (Phase 0)

Worktree: `/home/gmwager/projects/hippo-r2v2`, branch `report2-v2`,
base tag `report2-v1`. Tag `report2-v1` / `report2-v1.1` and main-tree v1
outputs stay frozen. **No decoding until this plan is approved. No commits
until explicitly approved.**

This file is **data-free** (safe for a public repo). Cohort lists, per-session
counts, d caps, v1 diffs, and MANIFEST audits live only in gitignored
`V2_COHORT.local.md`.

---

## 1. Why v2

Report 1 checks found that v1 decoded populations include many non-HF units
in several animals. The project goal is an HF BCI, so v2 decodes **HF units
only**. Report 1 will use the v2 cohort as its matched-sim cohort; its launch
is on hold until the v2 cohort is fixed (Phase 0 steps 2–3 unblock it).

---

## 2. Locked decisions (before any v2 result)

| Decision | Rule |
| --- | --- |
| HF Region allow-list | `MEC`, `SUB`, `PARA`, `LEC` (exact `Region` strings) |
| Excluded Region codes | `VI`, `TEa`, `NC` |
| `RegionConfidence` | **Ignored**; use `Region` as given. Record as a limitation. |
| NON-SOMA | Still dropped (`BClabel == NON-SOMA`) |
| Unit-based quantities | **HF units only**: ≥30 threshold, split-half stability fraction, median-count cohort rule |
| d grid | Cap at `⌊n_HF_units / 2⌋` per session; plateau check re-run on v2 |
| Bridge | Run v2 decode on **all 10 v1-cohort sessions**, including any with &lt;30 HF units; **flag** those sessions in every table and figure |
| Primary cohort | Same M3 rule as v1, but eligibility and median use HF-only counts |
| Matched sims | `hippo-dq/outputs/data_quality/matched/` is **READ-ONLY**. Never write, regenerate, or add files there. If a v2 cohort (or HF-eligible) session is missing from `MANIFEST.json`, list it and **STOP**; Report 4 generates it |
| Matched-arm coverage | Check **every** HF-eligible v2 session (not only newly admitted ones): HF-only eligibility can admit sessions that were not v1-eligible |
| Tests | Committed test = synthetic fixture only. Local uncommitted check: default (all-region) parameter must reproduce the v1 cohort exactly; report pass/fail |
| Docs | `V2_PLAN.md` (this file) + gitignored `V2_COHORT.local.md` |

Nothing in the claude.ai decision/handoff docs overrides the table above.

---

## 3. Region codes (Phase 0.1)

**Sources checked (dataset + local docs only; no outreach):**

- Per-session `clusters_dataset.csv` / `Clusters.mat` fields `Region` and
  `RegionConfidence`
- Session `config.yaml` (`project_info` has no region glossary)
- Local survey notes under `docs/real_data_quadrant/` (code list only)
- No README / codebook / abbreviation table under `HIPPO_DATA_ROOT`

**Observed `Region` string codes:** `LEC`, `MEC`, `NC`, `PARA`, `SUB`, `TEa`, `VI`.

**Working meanings** (standard neuroanatomical abbreviations; the package does
**not** ship expanded prose definitions):

| Code | Working meaning | Role in v2 |
| --- | --- | --- |
| MEC | Medial entorhinal cortex | HF (keep) |
| LEC | Lateral entorhinal cortex | HF (keep) |
| SUB | Subiculum | HF (keep) |
| PARA | Parasubiculum | HF (keep) |
| VI | Visual cortex | non-HF (drop) |
| TEa | Temporal association areas | non-HF (drop) |
| NC | Neocortex (generic) | non-HF (drop) |

**STOP check:** VI / TEa / NC match non-HF cortex as their names suggest;
LEC / PARA are hippocampal-formation / parahippocampal regions. **No STOP.**

**Limitation (record in report):** meanings are not defined in dataset-native
docs; `RegionConfidence` is ignored by decision.

---

## 4. Phase 0 deliverables (this run)

1. Region confirmation (above) — done; no STOP.
2. HF-only eligibility + M3 cohort + v1 diff — done (`V2_COHORT.local.md`).
3. Matched `MANIFEST.json` audit for every HF-eligible session — done (local);
   no missing sessions; no STOP.
4. v1 wording inventory — §5 below (proposed notes only; v1 untouched).
5. This plan + gitignored `V2_COHORT.local.md`; stop for approval.

**Local uncommitted v1 reproduction check:** **PASS** (default all-region
loader + `select_m3_cohort` matched the frozen v1 selected session set and
unit counts). Details in the local cohort file.

---

## 5. Phase 0 step 4 — v1 wording inventory (do not edit v1)

Places in Report 2 v1 **report text / captions / figure labels** that say
“hippocampal …” or otherwise describe the **decoded population**. Backend /
PLAN unit-rule lines included only where they feed report methods copy.
Proposed notes only — **no edits until approval**.

Exact phrase “hippocampal formation” does **not** appear in the real-data
Report 2 story/figures (it appears in Report 1 sim copy:
`build_story.py` “simulated hippocampal formation”). Real Report 2 uses
“hippocampal recordings” and “full population” instead.

| # | File:line (approx) | Current wording | Proposed note for v2 |
| --- | --- | --- | --- |
| 1 | `agents/quadrant_n5/figures/build_story_real.py` ~211–213 (H1 title) | “full population … on real **hippocampal recordings**” | Title: “full **HF** population … on real **HF-only** recordings” (or “hippocampal-formation units only”) |
| 2 | same ~227–234 (The question) | “population structure” / spike times / cohort — no region filter stated | Add one clause: decoded units = HF Region allow-list after NON-SOMA drop |
| 3 | same ~257–258 (answer §1) | “Causal EMA on the **full population**” | “full **HF** population” |
| 4 | same ~265 (answer §2) | “does not beat the **full population**” | “full **HF** population” |
| 5 | same ~311–314 (one sentence) | “compressing below the **full population**” | “full **HF** population” |
| 6 | same ~364–375 (Fig1 legend) | “Units by region (pooled and per animal)” — implies all recorded regions | State panel is HF-only (or show dropped non-HF as excluded counts); caption must not imply all regions enter decoding |
| 7 | same ~366–367 (Fig1b) | “spike times to causal counts …” — no unit filter | Mention HF filter in pipeline sentence |
| 8 | `agents/quadrant_n5/figures/figures_real.py` ~590 | Pipeline box: “Recorded spike times (sorted units, one room segment)” | “sorted **HF** units …” |
| 9 | same ~642 | Panel head: “**Recorded population** by region” | “**Decoded (HF) population** by region” or keep recorded bars but mark non-HF as excluded |
| 10 | same ~1339–1340 | Takeaway strip: “full population” | “full HF population” |
| 11 | `analysis/real_quadrant/report/pages.py` ~41, ~44 (story page) | “full population” (answers 1–2) | Same HF qualifier |
| 12 | same ~109–111 (methods page) | “≥30 units after NON-SOMA filter” / “median units” | “≥30 **HF** units after NON-SOMA”; median on HF counts |
| 13 | `docs/real_data_quadrant/PLAN.md` ~207, ~333, ~349, ~402, ~416 | “all regions; exclude NON-SOMA” / min 30 after NON-SOMA | Point at v2 HF rule (or add v2 overlay section); leave v1 historical text marked superseded |
| 14 | same ~540 (frozen answer blurb) | “full population” in contrast (2) | When PLAN answer is rewritten for v2, use HF framing |
| 15 | `analysis/real_quadrant/session_select.py` `M2_RULE` / `M3_RULE` | “min 30 units after NON-SOMA” | Report methods quote these strings — update when loader is parameterized |
| 16 | `analysis/real_quadrant/adapter.py` `load_units_and_spikes` docstring | “drop BClabel NON-SOMA” only | Document optional HF allow-list; default preserves v1 |
| 17 | Fig1c data path (`data_population` / region bars) | Bars include VI/TEa/NC when present | Under v2, bars = HF only (or split included vs excluded) |

Out of scope for Report 2 real PDF (sim Report 1 only): `build_story.py`
“simulated hippocampal formation” / entorhinal-dominated 86-unit population.

Do **not** edit any of the above until the cohort plan is approved.

---

## 6. Implementation outline (post-approval; not started)

1. Parameterize unit loader: `regions=None` (v1 / all non-NON-SOMA) vs
   `regions=HF_ALLOWLIST` (v2). Default path must keep the local v1
   reproduction check green.
2. Stability gate, ≥30, and median cohort rule consume the filtered unit list.
3. Per-session d grid: `min(grid, ⌊n_HF/2⌋)`; re-run plateau / extend-d logic.
4. Bridge decode set = union(v2 HF cohort, 10 v1 sessions); flag `&lt;30 HF`.
5. Synthetic fixture test only in CI; keep local v1 reproduction script
   uncommitted or under `outputs/`.
6. Rebuild Report 2 PDFs from v2 tidy extracts; retag when frozen.

---

## 7. Compute estimate

**Position-only (v1-like, HF-only):** from frozen v1 M3 order of magnitude —
~0.5–1 h wall per session (GPFA/LDS/Isomap dominate); ~1.5–2 h wall for ten
sessions in parallel (~6 h serial-equivalent). Extend-d similar. HF-only
reduces unit counts → same ballpark or faster.

**Multi-target (v2 primary + secondary):** latents fit **once** per
(session, method, d, fold) and are reused; extra cost is mostly readouts ×
targets (Ridge/kNN / multinomial / EMA grids / A5 / A13 per target). Relative
to position-only, expect roughly **~1.5–2.5×** wall if latents dominate, or
up to **~3×** if many d-grid readout refits are not amortized.

Budget (cohort + bridge union, parallel sessions): **~4–10 h** wall for the
full primary+secondary pass; **~15–30 h** if forced serial. Extend-d /
plateau still primarily on **x, y** (primary); secondary d-selection is
per-target on the cached latent grids where possible.

---

## 8. tmux / jobs

- Session name: `r2v2`
- Long jobs (&gt;~5 min): `tmux new -s r2v2` (or attach) and `| tee` logs under
  `outputs/real_quadrant/...`
- Phase 0 eligibility already ran in `r2v2` (~5.6 min including v1 repro).
- Never write under `hippo-dq/outputs/data_quality/matched/`.

---

## 9. Output directories (gitignored `outputs/`)

| Path | Role |
| --- | --- |
| `outputs/real_quadrant/phase0/` | Phase 0 eligibility JSON + log (local) |
| `outputs/real_quadrant/m3_v2/` (proposed) | HF-only M3 cohort + session fits |
| `outputs/real_quadrant/m3_v2/bridge/` (proposed) | v1-session bridge runs / flags |
| `outputs/real_quadrant/report2_v2/` (proposed) | Tidy extracts + PDF rebuild |
| `hippo-dq/.../matched/` | READ-ONLY matched sims (Report 4 owns) |

Exact dirname can be adjusted at implementation; keep v1 `m3/` / `report2/`
untouched.

---

## 10. Phase 0b — secondary targets (decision 2026-10-11)

Locked before any v2 result. **x, y remain PRIMARY.** Everything below is a
pre-registered **SECONDARY** analysis.

### 10.1 Variables

| Group | Variable | Notes |
| --- | --- | --- |
| Spatial (primary) | `x`, `y` (cm) | Same as v1 |
| Spatial (secondary) | `distance_to_wall` (cm) | Nearest distance to room-polygon edges; numpy only; **clip at 0 outside** |
| Spatial (secondary) | `wall_distance_bin` | `{near_wall <10, middle 10–30, center ≥30}` cm |
| Proprioceptive | `speed` (cm/s) | **Primary speed target:** backward-difference `|Δ(X,Y)|/Δt` over **250 ms** from the same `X`/`Y` (cm) as the position target, on the decoder grid (see §10.2) |
| Proprioceptive | `acceleration` | `dspeed/dt` of that 250 ms position-derived speed on the decoder grid |
| Proprioceptive | `head_direction` | Decode `(sin θ, cos θ)`; θ in radians |
| Proprioceptive | `movement_state` | `{still <2, slow 2–10, fast ≥10}` cm/s from the **position-derived** speed |

**Dropped — `spatial_context`:** deferred: no valid definition for irregular
(non-rectangular) arenas; wall information is carried by `distance_to_wall`
and `wall_distance_bin`. Same drop applies to the Report 1 mirror. Decision
2026-10-11 option **C**.

### 10.2 Phase 0b checks (report)

| # | Check | Result |
| --- | --- | --- |
| 1 | Head direction source | **PASS.** Column `HD` in `postions_dataset.csv` (degrees on disk → convert to rad). Matches `AnalyzedLog.mat` `S.PC_HD.Angle` (pose stream with left/right ear + nose; primary tracking stream `PC_HeadPos`). Distinct from movement direction `PC_MD`. **Not movement-derived.** Mask where `valid==False` or `HD` non-finite. |
| 2 | Speed definition + short-window QC | **Primary speed target = 250 ms backward position-derived speed** (not tracker `V`). See §10.2 definition. Tracker `V`/`PC_V` is a **sensitivity row** only. **Features stay causal `[t−W, t)` as v1.** |
| 3 | Arena geometry | Cohort room-A polygons are **not rectangular** → drove the `spatial_context` drop (§10.1). `distance_to_wall` to polygon edges remains well-defined. |
| 4 | Class balance | Local file only. `wall_distance_bin=center` is &lt;5% on essentially all cohort splits (merge → `middle`). `movement_state=fast` occasionally &lt;5%. |

**Speed target definition (decision after QC):**

- **Primary `speed`:** on the decoder grid, backward-difference
  `hypot(X(t)−X(t−250 ms), Y(t)−Y(t−250 ms)) / Δt` using the **same**
  `postions_dataset.csv` `X`,`Y` (cm) as the position target (room-local as
  used for decode). Mask with `valid` / finite as other targets. This speed
  feeds **`acceleration`** and **`movement_state`**.
- **Sensitivity row (speed only):** provided tracker `V` ≡ `AnalyzedLog`
  `S.PC_V` on the primary `PC_HeadPos` stream. Stored as a bare `float64`
  vector with **no window/formula field** in the package; exact upstream
  smoothing unknown. Per-session agreement vs the 250 ms position-derived
  speed is in `V2_COHORT.local.md` (r ≥ 0.8 on only a minority of cohort
  sessions → not used as the primary target).
- **Definition difference:** primary speed is an explicit causal-looking
  backward window on decode `X`/`Y`; `PC_V` is an opaque tracker estimate
  associated with head position, not guaranteed to match that window.

### 10.3 Pre-registered analysis rules (frozen on approval)

- Same features, trims, splits, masks, HF cohort, and bridge as v2 Phase 0.
  `valid==False` masks **all** targets; HD also masked where tracking HD is
  missing / non-finite.
- **Latents are target-independent:** fit once per `(session, method, d, fold)`,
  cache under `hippo-r2v2/outputs/`, reuse for every target. **`d` selected by
  inner CV per target.**
- **Continuous** (`x`,`y`,`distance_to_wall`,`speed`,`acceleration`): Ridge +
  kNN; normalized error = median `|err|` / median `|err|` of the constant train
  predictor (train **mean** for `x`,`y` as v1; train **median** for scalar
  targets). EMA (τ grid as v1, chosen on inner CV) on outputs.
- **`head_direction`:** Ridge/kNN on `(sin, cos)`; angle = `atan2`; score =
  median absolute circular error / floor of the constant train circular mean.
  EMA on sin/cos, **never** on the angle.
- **Categorical** (`wall_distance_bin`, `movement_state`): multinomial logistic
  regression (linear) + kNN classifier; score =
  `(1 − balanced_accuracy) / (1 − 1/k)` so 1 = chance like the other metrics.
  EMA on class probabilities.
- **Minimum class rule (fixed now):** a class with &lt; 5% of valid samples in
  **train or test** is merged into its neighbour (ordered bins) or the variable
  is reported as not evaluable for that session.
- **Beyond-position / beyond-speed control:** `distance_to_wall` and
  `wall_distance_bin` also computed from **decoded** `x`,`y`;
  `movement_state` from **decoded** speed. Report direct minus derived (sign
  counts over animals).
- **Stats:** same as v1 (N = animals, sign counts, no p-values). Per variable,
  one pre-registered contrast set: `raw_smooth` vs each method, and direct vs
  derived where applicable. Everything else descriptive.
- **A5 and A13** audits run **per target** (A13 must reach the floor for each).

### 10.4 Remaining notes (non-blocking)

1. **`wall_distance_bin=center`:** expect routine &lt;5% → merge into `middle`
   under the fixed rule (often a 2-class problem). Accepted with the merge rule.
2. Tracker `V` / `PC_V` remains available as a **speed sensitivity** readout
   only; it does not drive acceleration or movement_state.

---

## 11. Approval

`V2_PLAN.md` **APPROVED** 2026-10-11 (HF-only cohort; secondary targets;
`spatial_context` dropped; primary speed = 250 ms backward position-derived).
Implementation and decode proceed from this document. Matched MANIFEST had no
gaps in Phase 0.
