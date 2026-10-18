# Live Report 2 run — do-not-touch list

Fingerprint taken **2026-10-05T07:56:17Z** (Phase 0). Re-check under audit Q6
at Phase 2 start/end. Raw dump:
`outputs/data_quality/_phase0/live_fingerprint.txt`.
Output mtime baseline:
`outputs/data_quality/_phase0/live_outputs_mtime_baseline.txt`.

## Process

| Field | Value |
| ----- | ----- |
| Host | `bionet-neuroai` |
| tmux | session `m2`, pane `m2:0.0`, pane_pid `3231769` (bash) |
| PID | **1177494** |
| Children | none at snapshot (no loky/joblib worker PIDs; in-process threads) |
| Command | `python -m analysis.real_quadrant.run_m2` |
| Started (wall) | ~2026-10-05 10:38 local; etime ~18 min at snapshot |
| Status | `Rl+`, high CPU (~800–940%), LDS stage writing `lds.json` |
| exe | `/usr/bin/python3.12` (resolved via venv shim) |
| Python used | `/home/gmwager/projects/hippo/.hippo/bin/python` |
| VIRTUAL_ENV | `/home/gmwager/projects/hippo/.hippo` |
| cwd | `/home/gmwager/projects/hippo` |
| HIPPO_DATA_ROOT | `/data/ecephys/Shai_Daniel_12032026` |

Full argv at start (from tmux capture, read-only):

```text
cd ~/projects/hippo && source .hippo/bin/activate && \
  python -m analysis.real_quadrant.run_m2 2>&1 | tee outputs/real_quadrant/m2_run.log
```

Earlier failed attempts in the same pane (HIPPO_DATA_ROOT unset; then
`_final_fit_predict` unpack error) are historical only. Current process is the
third start after those fixes.

## Checkout (live)

| Field | Value |
| ----- | ----- |
| Path | `/home/gmwager/projects/hippo` |
| Branch | `offline-decoding` (tracks `origin/offline-decoding`) |
| HEAD | `24cc6022ca802b946d9b7a1c741e9919ae6d49c6` |
| Tip message | `Fix M2 figure-contract unpack and version-aware resume keys.` |
| Dirty files | **none** (`git status --porcelain` empty) |

**Do not** `checkout` / `switch` / `stash` / `pull` / `reset` / `rebase` /
`merge` / `clean`, and **do not edit any file** under this path.

## Memory / CPU

| Metric | Value |
| ------ | ----- |
| VmRSS | ~2.8 GiB at snapshot (was ~1.3 GiB earlier in LDS) |
| VmHWM / peak RSS | **~2.8 GiB** (2839388 kB) |
| VmPeak | ~15.0 GiB virtual |
| Threads / tasks | **191** (no separate worker processes) |
| Affinity | cores 0–127 |
| OMP/OPENBLAS/MKL env | **unset** (uncapped BLAS threads) |
| Spawns workers? | Not observed as child PIDs; threaded BLAS / internal pools only |

## Open files (snapshot)

Sparse during compute. Observed fds:

- stdin: `/dev/pts/43` (tmux pane)
- stdout/stderr: pipe → `tee outputs/real_quadrant/m2_run.log`
- matplotlib font files under `.hippo/lib/...`

No lab data files held open at snapshot time (lazy / already closed). Lab root
is still a dependency via `HIPPO_DATA_ROOT`.

## Paths this run owns (never write here)

### Outputs

- `/home/gmwager/projects/hippo/outputs/real_quadrant/` (entire tree)
- `/home/gmwager/projects/hippo/outputs/real_quadrant/m2/`
- `/home/gmwager/projects/hippo/outputs/real_quadrant/m1/` (prior; leave alone)
- `/home/gmwager/projects/hippo/outputs/real_quadrant/cache/` (`.npz` causal-count caches)
- `/home/gmwager/projects/hippo/outputs/real_quadrant/manifest.jsonl`
- `/home/gmwager/projects/hippo/outputs/real_quadrant/m2_run.log`

### Code / env the live process can re-import

- `/home/gmwager/projects/hippo/` working tree (cwd on `sys.path`)
- `/home/gmwager/projects/hippo/.hippo/` (venv; shared — **no pip**)
- Any module under `analysis/`, `realtime/`, `hippo/`, `hippo_sim/`, `configs/`

### Lab data (read-only for everyone)

- `/data/ecephys/Shai_Daniel_12032026/` — never write caches beside data

### Config entry

- Module: `analysis.real_quadrant.run_m2`
- `OUT_ROOT = <live>/outputs/real_quadrant/m2`
- Adapter defaults (if imported from live tree):
  - cache → `<live>/outputs/real_quadrant/cache`
  - manifest → `<live>/outputs/real_quadrant/manifest.jsonl`

## Import / install mode

Not an editable `pip install -e` egg-link. Resolution is **cwd / source tree
on `sys.path`** plus the `.hippo` site-packages for third-party deps. Changing
files under the live checkout changes what future imports see.

## Sibling / other sessions

- tmux `backup` (unrelated bash)
- Unrelated user `eviatar` SNN jobs — ignore; do not compete aggressively

## Report 4 isolation

| Item | Path |
| ---- | ---- |
| Worktree | `/home/gmwager/projects/hippo-dq` |
| Branch | `data-quality` @ same commit `24cc602` |
| Our outputs only | `/home/gmwager/projects/hippo-dq/outputs/data_quality/` |
| Run with | `cd /home/gmwager/projects/hippo-dq && PYTHONPATH=/home/gmwager/projects/hippo-dq` |
| Shared venv (read-only use) | `/home/gmwager/projects/hippo/.hippo/bin/python` — never `pip install` |

## Q6 watch list

1. PID 1177494 still running, or exited on its own (never restart).
2. Live `HEAD` still `24cc602…` and porcelain still empty.
3. No new/changed mtimes under live `outputs/real_quadrant/` caused by us
   (compare to `_phase0/live_outputs_mtime_baseline.txt`; live run may advance
   its own files — that is expected).
4. `python -c "import analysis.real_quadrant.adapter as a; print(a.__file__)"`
   with live cwd still resolves into `/home/gmwager/projects/hippo/...`.

## Amendment (Phase 0→1 gate, 2026-10-05)

Live M2 also writes fitted models via hardcoded
`realtime.quadrant_n5_run.OUTPUT_ROOT` →

- `/home/gmwager/projects/hippo/outputs/quadrant_n5/models/seed_0/real/<method>/`

Timestamps align with M2 method completion (LDS ~10:53, Isomap ~11:06).
**Add this tree to the do-not-touch / find-exclusion list.**

`outputs/quadrant_n5/seed_*/sim/` was **not** modified (newest still 2026-09-24)
and remains safe to copy read-only into the worktree.

## Live output dirs (do-not-touch; isolation-find exclusions)

1. `/home/gmwager/projects/hippo/outputs/real_quadrant/` — M2/M3 reports, cache, manifest, logs
2. `/home/gmwager/projects/hippo/outputs/quadrant_n5/models/seed_{0..5}/real/` — fitted models
   written by live `analyze_source` via hardcoded `OUTPUT_ROOT` (not all of `models/`).
   M3 writes **seeds 0–5** under `…/models/seed_i/real/<method>/`.

Isolation gate (M3-start marker `report2_m3_start.marker`): exclude exactly these two trees.
`outputs/quadrant_n5/seed_*/sim/` is **not** a live output dir (untouched since 2026-09-24).

## Status update 2026-10-05T11:21 local

Live PID 1177494 **ended on its own** after successful M2 completion
(`wall_time_s≈2444`, methods raw…gpfa all present in `m2_report.json`).
Do not restart. Isolation rules for the checkout working tree remain until
the user says otherwise; output dirs above stay do-not-write for Report 4.


## Live Report 2 process: M3 — ENDED

> Previous M2 PID 1177494 ended successfully earlier the same day.
> **M3 ended on its own 2026-10-05 15:12 local** (wall ~13:31–15:12; finished
> **9/10** sessions). `pgrep run_m3` empty as of 2026-10-06. Do not restart.
> Isolation rules for Report 2 **output dirs** remain (do not write into them).

| Field | Value |
| ----- | ----- |
| tmux | session **`m3`**, pane `m3:0.0`, pane_pid `2096784` (bash) |
| PID | **2096787** (`python -u -m analysis.real_quadrant.run_m3`) |
| Workers | 6× `--worker-job` children (PIDs 2105795–2105800 at fingerprint) |
| Start | **2026-10-05 13:31:37** local (`lstart`) |
| Command | `cd /home/gmwager/projects/hippo && source .hippo/bin/activate && export HIPPO_DATA_ROOT=... OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 && test -z "$(git status --porcelain)" && python -u -m analysis.real_quadrant.run_m3 2>&1 \| tee outputs/real_quadrant/m3_run.log` |
| cwd | `/home/gmwager/projects/hippo` |
| Python | `/home/gmwager/projects/hippo/.hippo/bin/python` (exe → `/usr/bin/python3.12`) |
| HIPPO_DATA_ROOT | `/data/ecephys/Shai_Daniel_12032026` |
| Thread env | `OMP/MKL/OPENBLAS_NUM_THREADS=8` |
| Checkout branch | `offline-decoding` |
| HEAD at fingerprint | **`b68606eb9e154f0aa8b319ac151130fc6b8c0697`** (NOT 24cc602) |
| Dirty files | **none** |
| Outputs (do-not-touch) | `outputs/real_quadrant/` (includes **`m3/`**, cache, manifest, `m3_run.log`) |
| Also still do-not-touch | `outputs/quadrant_n5/models/seed_{0..5}/real/` — **confirmed:** M3 writes fitted models here for seeds **0–5** |

### Isolation marker

`outputs/data_quality/_phase0/report2_m3_start.marker` mtime = M3 start
(2026-10-05 13:31:37 +0300).

### CRITICAL — code drift vs Report 4 worktree base

Report 4 worktree remains detached at **`24cc602`**. Live checkout has advanced
**4 commits** to `b68606e`. Files that differ in adapter / analyze_source path:

- `analysis/real_quadrant/adapter.py`
- `analysis/real_quadrant/session_select.py`
- `realtime/quadrant_n5_run.py`

Plus new M3/posthoc modules. **Lab-path identity with Report 4 is broken until
reconciled.** Do not assume byte-identical loaders.

## Isolation re-check (M3-start, -type f) — 2026-10-05

Marker: `outputs/data_quality/_phase0/report2_m3_start.marker` (mtime = M3 start).

```bash
find /home/gmwager/projects/hippo -type f -newer <marker> \
  -not -path '*/.git/*' -not -path '.../.git' \
  -not -path '.../outputs/real_quadrant' -not -path '.../outputs/real_quadrant/*' \
  -not -path '.../outputs/quadrant_n5/models/seed_*/real' \
  -not -path '.../outputs/quadrant_n5/models/seed_*/real/*'
```

**COUNT = 0** (clean).

Separate listing of new files under `models/seed_{1..5}/` newer than marker:
**only** `real/` (19 files/seed × 5; methods dm/pca/raw/raw_lag so far). No non-`real/`
writes into Report 1 sim model trees. `seed_0/real/` also advancing (same pattern).

### Status 2026-10-06

M3 PID 2096787 **ended** (9/10 sessions, `run_results.json` at 15:12).
No Report 2 process running. Output trees under `outputs/real_quadrant/` and
`outputs/quadrant_n5/models/seed_*/real/` remain do-not-write for Report 4.
Worktree move to `b68606e` follows in this session.

## Status 2026-10-07 (Phase 3 start)

- `pgrep -af real_quadrant`: **no Report 2 / M3 / M2 process** (idle).
- Live checkout HEAD: **`2638755`** on `offline-decoding` (porcelain clean at fingerprint).
- Report 4 worktree: **`b68606e`** detached at `/home/gmwager/projects/hippo-dq`.
- Unrelated tmux `sim_extend_d` finished (seed extend job EXIT=0); not a Report 2 run.
- Load ~37–41, MemAvailable ~590 GiB, nproc=128.
- Isolation marker for this phase: `outputs/data_quality/_phase0/report4_phase3_start.marker`.
- **Still do-not-write:** `outputs/real_quadrant/`, `outputs/quadrant_n5/models/seed_*/real/`.
- §1 rules unchanged.

## Status 2026-10-07T08:23Z (catch-up complete → Phase 3)

- `pgrep` for `analysis.real_quadrant` / `run_m[123]`: **none** (idle).
- Live HEAD: `2638755` on `offline-decoding`. Porcelain: untracked `analysis/real_quadrant/report/` only (not a Report 4 write).
- Worktree: `b68606e` at `/home/gmwager/projects/hippo-dq`. Matched arm frozen: `matched/MANIFEST.json` n=47.
- Isolation find newer than `report4_phase3_start.marker`: **COUNT=4** = those untracked report helpers under live tree; **no** writes under `outputs/real_quadrant/` or `models/seed_*/real/` by Report 4.
- MemAvailable ≈ 573 GiB, nproc=128. §1 rules unchanged. Do-not-write trees unchanged.

