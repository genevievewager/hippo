# History rewrite plan (do NOT execute until approved)

Purge committed virtualenv and bytecode junk from **all** branches. Keep
`docs/overleaf/` in history (outdated but fine). Repo stays **public**.

**Local `.git` size before rewrite (2026-09-30):** ~5.4 GB.

---

## 0. Decide fate of `architecture-hardening`

Only branch not merged into `master` / `quadrant-n5`.

```bash
git log master..architecture-hardening --stat
```

**One commit ahead of `master`:** `ee062a2f` — *Tighten AnalysisConfig ownership
so UI/CLI cannot silently diverge.*

Adds (~4k lines): `realtime/quadrant_experiment.py`,
`realtime/representation_registry.py`, pipeline invariant/artifact hardening,
Quadrant Comparison UI, AnalysisConfig tests, cursor rules,
`docs/PIPELINE_CONTRACTS.md`, and (on that branch) `docs/ARCHITECTURE.md`.

**Overlap with `quadrant-n5`:** the same theme was landed as WIP commit
`1100d380` on `quadrant-n5` (then merged hygiene from master). The two commits
are **not** identical — treat `architecture-hardening` as a parallel fork of
the AnalysisConfig work, not as unique production code already missing from
`quadrant-n5`. Diff the tips before deciding:

```bash
git diff quadrant-n5...architecture-hardening --stat
```

**Your call before rewrite:** merge remaining unique bits → `quadrant-n5`,
keep the branch as-is through the rewrite, or delete the branch (and its
worktree) after confirming nothing unique is needed.

Worktree today: `/tmp/hippo-arch` @ `ee062a2f`.

---

## 1. Pre-rewrite checklist (local)

```bash
# From the main clone (e.g. ~/projects/hippo), on a clean tree:
git status -sb

# Remove extra worktrees (paths as of 2026-09-30):
git worktree remove /tmp/hippo-arch
git worktree remove /tmp/hippo-n5-verify
# If remove fails because of local changes: git worktree remove --force …

git worktree list   # should show only the main checkout

# Confirm every branch you care about is pushed:
git fetch --all --prune
git branch -vv
# Expected remotes (adjust if you kept/deleted architecture-hardening):
#   master, quadrant-n5, cleanup/structure, architecture-hardening?
git push --all origin
git push --tags origin

# Mirror backup OUTSIDE the repo (example path — change as you like):
git clone --mirror \
  git@github.com:genevievewager/hippo.git \
  /home/gmwager/backups/hippo.git.mirror-$(date -u +%Y%m%dT%H%M%SZ)

# Record size before rewrite:
du -sh .git
du -sh /home/gmwager/backups/hippo.git.mirror-*
```

Also note tag: `quadrant-n5-figures-v1` (must be force-pushed after rewrite).

---

## 2. Install git-filter-repo (if needed)

```bash
python3 -m pip install --user git-filter-repo
# or: sudo apt install git-filter-repo
git filter-repo --version
```

---

## 3. Purge paths from ALL branches

**Purge only:**

- `.hippo/` (entire virtualenv tree)
- any `__pycache__/` path segment / `*.pyc`
- `.DS_Store`

**Do not purge:** `docs/overleaf/` (leave in history).

Run from a **fresh clone of the mirror backup** (safer than rewriting the
daily working copy), or from the main repo only after the mirror exists:

```bash
# Example: rewrite a disposable clone created from the mirror
git clone /home/gmwager/backups/hippo.git.mirror-YYYYMMDDTHHMMSSZ \
  /tmp/hippo-rewrite
cd /tmp/hippo-rewrite

# Size before
du -sh .git | tee /tmp/hippo-rewrite-size-before.txt

git filter-repo \
  --invert-paths \
  --path .hippo \
  --path-glob '**/__pycache__/**' \
  --path-glob '**/*.pyc' \
  --path .DS_Store \
  --path-glob '**/.DS_Store'

# If path-glob support differs on your git-filter-repo version, equivalent:
# git filter-repo --invert-paths \
#   --path .hippo \
#   --path-glob '*' --path-match '__pycache__'   # verify flags with --help
# Prefer verifying with a dry-run analysis first:
# git filter-repo --analyze
```

Re-add `origin` (filter-repo removes remotes):

```bash
git remote add origin git@github.com:genevievewager/hippo.git
git fetch origin
```

---

## 4. Verify rewritten history

```bash
du -sh .git | tee /tmp/hippo-rewrite-size-after.txt
# Expect a large drop from ~5.4G

# No .hippo paths left in any commit:
git log --all --oneline -- .hippo/   # must print nothing
git rev-list --all --objects | rg '\.hippo/|__pycache__|\.DS_Store' | head
# (second command should be empty or only non-matching noise)

# docs/overleaf still present in history (optional spot-check):
git log --all --oneline -- docs/overleaf/ | head

# Tests on rewritten tips (recreate venv; do not restore old .hippo from backup):
git checkout master
python3 -m venv .hippo
source .hippo/bin/activate
pip install -U pip
pip install -r requirements.lock
python -m pytest tests/ -q          # default excludes @pytest.mark.slow

git checkout quadrant-n5
# reuse or recreate .hippo the same way
python -m pytest tests/ -q

# Secret scan (optional but recommended):
# gitleaks detect --source . -v
```

---

## 5. Force-push all branches and tags

**Destructive. Only after verify.**

```bash
# Still in the rewritten clone, with origin set:
git push --force --all origin
git push --force --tags origin
```

Branches that must be updated on the remote (as of prep):

- `master`
- `quadrant-n5`
- `cleanup/structure`
- `architecture-hardening` (if kept)
- tag `quadrant-n5-figures-v1`

There is still **no** `pressure-test-reports` branch on origin; nothing to push
there unless you create it later.

---

## 6. Every clone must be deleted and re-cloned

Old clones retain the pre-rewrite objects and can accidentally
`git push --force` the junk history back. Delete and re-clone:

| Clone | Action |
|-------|--------|
| This machine `~/projects/hippo` | Remove or rename; `git clone git@github.com:genevievewager/hippo.git` |
| `/tmp/hippo-arch`, `/tmp/hippo-n5-verify` | Already removed in §1; do not reattach old worktrees |
| **bionet-neuroai** (pressure-test / Cursor Remote-SSH) | Delete the old clone; re-clone; rebuild env; reinstall hooks; re-check deploy key (below) |
| Any laptop / CI / colleague clone | Same: delete + re-clone |

### bionet-neuroai re-setup

```bash
# On bionet-neuroai, after deleting the old tree:
git clone git@github.com:genevievewager/hippo.git
cd hippo
python3 -m venv .hippo
source .hippo/bin/activate
pip install -r requirements.lock

# Prefer tracked hooks via core.hooksPath (hooks live in scripts/hooks/):
git config core.hooksPath scripts/hooks
# Ensure scripts/hooks/pre-commit is executable (it is in-repo).

# Deploy key: confirm SSH still authenticates to THIS repo only
ssh -T git@github.com
git fetch origin
git push origin HEAD:refs/heads/_deploy_key_probe && \
  git push origin --delete _deploy_key_probe
# If that fails, re-run scripts/setup_agent_sync.sh and re-add the
# printed PUBLIC key as a write-enabled deploy key on genevievewager/hippo.
```

Optional: reinstall the scheduled pressure-test runner via
`scripts/setup_agent_sync.sh` (creates timer/cron that publishes to
`pressure-test-reports` — branch still absent until first successful publish).

---

## 7. Docs / pins that break when commit hashes change

Update these **after** the rewrite (hashes will change):

| Location | Pin |
|----------|-----|
| `agents/quadrant_n5/SPEC.md` | Companion note: pressure-test built @ `01b86a3` |
| `realtime/quadrant_n5.py` | `SEEDS_0_4_PROVENANCE_SHA = "775fa1c3…"` (code provenance of seeds 0–4) |
| `agents/quadrant_n5/export_predictions.py` | Frozen `config_sha256` `4da17a5d…`; `EVAL_HASH`; uses `SEEDS_0_4_PROVENANCE_SHA` |
| `tests/test_quadrant_n5_predictions.py` | Asserts frozen `config_sha256` `4da17a5d…` |
| `agents/quadrant_n5/CURSOR_PROMPT_phase7_predictions_and_panels.md` | Mentions config `4da17a5d…`, seeds code `775fa1c3…` |
| `agents/quadrant_n5/figures/figures.py` | Mentions config / seeds / `report_code_sha eb442e9f` |

**Content hashes** (`config_sha256` of `configs/quadrant_n5.yaml`) do **not**
change from a history rewrite unless that file’s blob changes. **Git commit
SHAs** (`01b86a3`, `775fa1c3`, `eb442e9f`, tag targets) **do** change — re-resolve
with `git log` / `git rev-parse` after the rewrite and update the docs.

---

## 8. Rollback

If anything looks wrong **before** force-push, abandon `/tmp/hippo-rewrite` and
keep using the untouched mirror.

If force-push already happened, restore from the mirror:

```bash
cd /home/gmwager/backups/hippo.git.mirror-YYYYMMDDTHHMMSSZ
git push --force --all origin
git push --force --tags origin
```

Then re-clone all working copies again.

## Appendix — executed rewrite (2026-09-30, pass 2)

Filter-repo purged `.hippo/`, `__pycache__/`, `.DS_Store`, and `outputs/`.
Commit-map: `.git/filter-repo/commit-map`.

| pre-rewrite | post-rewrite |
|-------------|--------------|
| `01b86a34…` (`01b86a3`) | `8c01b734b40e2f0069b3e16fea56dd756fbe9955` |
| `775fa1c38063a1ecea07df28c64174beea8f122e` | `dcc78ed74194aa1d3e8fa7e1e25e90e02b26265e` |
| `eb442e9f5d3137e1bfb5d9ec960f128f25fec32e` | `566b94353c2467147f90f839603c86e9f2102c56` |
| tag tip `5a477fcf5170f6975f89b6c42858ca7b8094e8c8` | `2b8c46e917fca7d288ebe6df185babb3ff28f8c4` |

`4da17a5d…` config content hash left unchanged.

Tag `quadrant-n5-figures-v1` peeled commit is `2b8c46e9…` (pre-rewrite: `5a477fcf…`).

