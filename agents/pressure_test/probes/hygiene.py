"""Repository and IP hygiene.

This probe answers one question: *what has this repository committed that it
should not have, and what would leave the lab if the remote were readable?*

Everything here is local. It runs `git` against the working tree and never
contacts the network.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..findings import Finding, Probe, Severity

# Anything matching these is presumed to be derived, private, or enormous.
_VENV_MARKERS = ("pyvenv.cfg", "site-packages/", "/bin/activate")
_DATA_EXT = {".npy", ".npz", ".dat", ".bin", ".h5", ".hdf5", ".nwb", ".mat", ".parquet"}
_SECRET_PAT = re.compile(
    r"(BEGIN (RSA|OPENSSH|DSA|EC|PGP) PRIVATE KEY|"
    r"AKIA[0-9A-Z]{16}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"sk-[A-Za-z0-9]{32,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,})"
)
_PERSONAL_PATH = re.compile(r"(/home/[a-z][a-z0-9_-]*/|/Users/[A-Za-z][A-Za-z0-9_-]*/)")


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=120, check=False,
        ).stdout
    except Exception:
        return ""


class HygieneProbe(Probe):
    name = "hygiene"
    description = "What the repository is carrying that it should not be"

    def checks(self):
        yield from self.check(self._tracked_virtualenv)
        yield from self.check(self._gitignore_is_ineffective)
        yield from self.check(self._large_tracked_files)
        yield from self.check(self._tracked_secrets)
        yield from self.check(self._tracked_recording_data)
        yield from self.check(self._personal_paths)
        yield from self.check(self._remote_summary)

    def _tracked(self) -> list[str]:
        out = _git(self.ctx.repo_root, "ls-files")
        return [ln for ln in out.splitlines() if ln.strip()]

    # -------------------------------------------------------------------

    def _tracked_virtualenv(self):
        files = [f for f in self._tracked() if any(m in f for m in _VENV_MARKERS)]
        if not files:
            return []
        roots = sorted({f.split("/", 1)[0] for f in files})
        all_under = [f for f in self._tracked() if f.split("/", 1)[0] in roots]
        return [
            Finding(
                probe=self.name,
                title="A Python virtualenv is committed to the repository",
                severity=Severity.HIGH,
                category="hygiene",
                where=", ".join(roots),
                detail=(
                    f"{len(all_under)} tracked files live under {roots}, including "
                    "site-packages and compiled .so files. Every clone and every CI job "
                    "pays for it, the binaries are pinned to one interpreter and "
                    "architecture so they are actively misleading on a different "
                    "machine, and the history can never be made smaller without a "
                    "rewrite."
                ),
                evidence={"venv_roots": roots, "tracked_files_under": len(all_under),
                          "sample": files[:5]},
                suggestion=(
                    "git rm -r --cached <root>, commit, then rewrite history with "
                    "git-filter-repo to reclaim the object store. requirements.txt "
                    "already pins what is needed."
                ),
            )
        ]

    def _gitignore_is_ineffective(self):
        gi = self.ctx.repo_root / ".gitignore"
        if not gi.exists():
            return []
        patterns = [
            ln.strip().rstrip("/")
            for ln in gi.read_text(errors="ignore").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        tracked = self._tracked()
        venv = self._venv_roots()
        # Matches inside a committed virtualenv are real but not separately
        # actionable: purging the venv resolves them all at once. Counting them
        # in the headline number makes the finding look bigger than the work.
        outside_venv = [f for f in tracked if f.split("/", 1)[0] not in venv]
        violated: dict[str, int] = {}
        violated_outside: dict[str, int] = {}
        def _count(files: list[str], pat: str) -> int:
            if pat.startswith("*"):
                suffix = pat[1:]
                return sum(1 for f in files if f.endswith(suffix))
            # Directory-aligned matching only. Substring matching inflates the
            # count wildly: `external` would otherwise match every path that
            # merely contains the word.
            return sum(
                1 for f in files
                if f == pat
                or f.startswith(pat + "/")
                or f"/{pat}/" in f
                or f.endswith("/" + pat)
            )

        for pat in patterns:
            n = _count(tracked, pat)
            if n:
                violated[pat] = n
                m = _count(outside_venv, pat)
                if m:
                    violated_outside[pat] = m
        if not violated:
            return []
        return [
            Finding(
                probe=self.name,
                title=".gitignore lists paths that are already tracked",
                severity=Severity.MEDIUM,
                category="hygiene",
                where=".gitignore",
                detail=(
                    "gitignore only affects untracked files. These patterns were added "
                    "after the files were committed, so they are silently doing nothing "
                    "— which is worse than not having them, because the repository "
                    "looks clean on inspection.\n\n"
                    + (
                        f"Outside the committed virtualenv the live count is "
                        f"{sum(violated_outside.values())} files across "
                        f"{len(violated_outside)} pattern(s); everything else resolves "
                        "when the venv is purged."
                        if violated_outside else
                        "Every match is inside the committed virtualenv, so purging it "
                        "clears all of these at once."
                    )
                ),
                evidence={
                    "pattern_to_tracked_file_count": violated,
                    "pattern_to_tracked_outside_venv": violated_outside,
                },
                suggestion="git rm -r --cached for each, then commit.",
            )
        ]

    def _large_tracked_files(self):
        """Measure git BLOB sizes, not working-tree stat().

        stat() follows symlinks, so a venv's bin/python -> /usr/bin/python3.12
        is reported at the size of the system interpreter even though git
        stores only the short link target. That inflates the number and makes
        the finding easy to dismiss.
        """
        raw = _git(
            self.ctx.repo_root, "ls-tree", "-r", "-l", "-z", "HEAD"
        )
        big: list[tuple[str, int]] = []
        total_tracked = 0
        for entry in [e for e in raw.split("\0") if e.strip()]:
            try:
                meta, path = entry.split("\t", 1)
                parts = meta.split()
                mode, otype, size = parts[0], parts[1], parts[3]
            except (ValueError, IndexError):
                continue
            if otype != "blob" or mode == "120000":     # 120000 == symlink
                continue
            try:
                n = int(size)
            except ValueError:
                continue
            total_tracked += n
            if n > 5_000_000:
                big.append((path, n))
        if not big:
            return []
        big.sort(key=lambda kv: -kv[1])
        total_mb = sum(s for _, s in big) / 1e6
        venv = self._venv_roots()
        outside = [(p, s) for p, s in big if p.split("/", 1)[0] not in venv]
        return [
            Finding(
                probe=self.name,
                title=f"{len(big)} tracked blobs exceed 5 MB",
                severity=Severity.MEDIUM if total_mb < 200 else Severity.HIGH,
                category="hygiene",
                detail=(
                    f"{total_mb:.1f} MB in blobs over 5 MB, out of "
                    f"{total_tracked / 1e6:.0f} MB tracked in total. Large binaries in "
                    "git are permanent: every future clone downloads every historical "
                    "version of each one. "
                    + (
                        f"{len(outside)} of them are outside the committed virtualenv, "
                        "so they survive a venv purge and need handling separately."
                        if outside else
                        "All of them are inside the committed virtualenv, so purging it "
                        "resolves this finding too."
                    )
                ),
                evidence={
                    "largest": [{"path": p, "mb": round(s / 1e6, 2)} for p, s in big[:10]],
                    "outside_venv": [{"path": p, "mb": round(s / 1e6, 2)}
                                     for p, s in outside[:10]],
                    "total_tracked_mb": round(total_tracked / 1e6, 1),
                },
                suggestion="Move to the outputs/ directory or an artifact store.",
            )
        ]

    def _venv_roots(self) -> set[str]:
        return {
            f.split("/", 1)[0]
            for f in self._tracked()
            if any(m in f for m in _VENV_MARKERS)
        }

    def _tracked_secrets(self):
        out = []
        # Vendored third-party code is full of base64 blobs and test fixtures
        # that trip credential regexes. Scanning it produces noise, not signal.
        vendored = self._venv_roots() | {"node_modules"}
        for f in self._tracked():
            if f.split("/", 1)[0] in vendored:
                continue
            fp = self.ctx.repo_root / f
            if fp.suffix in {".png", ".pdf", ".so", ".pyc", ".zip", ".npy"}:
                continue
            try:
                if fp.stat().st_size > 2_000_000:
                    continue
                text = fp.read_text(errors="ignore")
            except OSError:
                continue
            m = _SECRET_PAT.search(text)
            if m:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Possible credential committed in `{f}`",
                        severity=Severity.CRITICAL,
                        category="hygiene",
                        where=f,
                        detail=(
                            "A pattern matching a private key or API token is present "
                            "in a tracked file. Treat it as compromised and rotate it — "
                            "removing the file does not remove it from history."
                        ),
                        evidence={"match_kind": m.group(0)[:20] + "..."},
                    )
                )
        return out

    def _tracked_recording_data(self):
        hits = [
            f for f in self._tracked()
            if Path(f).suffix.lower() in _DATA_EXT and not f.startswith(".hippo/")
        ]
        if not hits:
            return []
        return [
            Finding(
                probe=self.name,
                title="Recording-format data files are tracked in git",
                severity=Severity.HIGH,
                category="hygiene",
                detail=(
                    "Files in recording/array formats are committed. If any of these "
                    "contain real animal data, they are in the repository's history and "
                    "are exposed to anyone who can read the remote."
                ),
                evidence={"files": hits[:20], "count": len(hits)},
                suggestion=(
                    "Keep real recordings outside the repository entirely; reference "
                    "them by path from a config that is itself gitignored."
                ),
            )
        ]

    def _personal_paths(self):
        out: dict[str, int] = {}
        for f in self._tracked():
            if not f.endswith((".py", ".md", ".sh", ".yaml", ".yml", ".json", ".cfg")):
                continue
            if f.startswith(".hippo/"):
                continue
            try:
                text = (self.ctx.repo_root / f).read_text(errors="ignore")
            except OSError:
                continue
            n = len(_PERSONAL_PATH.findall(text))
            if n:
                out[f] = n
        if not out:
            return []
        return [
            Finding(
                probe=self.name,
                title="Absolute home-directory paths are hard-coded in tracked files",
                severity=Severity.LOW,
                category="hygiene",
                detail=(
                    "These break on any machine but the author's, and they disclose "
                    "usernames and directory layout to anyone reading the repository."
                ),
                evidence={"file_to_occurrences": dict(sorted(out.items(),
                                                            key=lambda kv: -kv[1])[:10])},
                suggestion="Resolve from __file__ or an env var, as ui/app.py does.",
            )
        ]

    def _remote_summary(self):
        url = _git(self.ctx.repo_root, "remote", "get-url", "origin").strip()
        if not url:
            return [
                Finding(
                    probe=self.name,
                    title="No `origin` remote configured",
                    severity=Severity.LOW,
                    category="hygiene",
                    detail="Automatic sync cannot run without a remote.",
                )
            ]
        uses_https = url.startswith("https://")
        return [
            Finding(
                probe=self.name,
                title="Remote configuration recorded",
                severity=Severity.INFO,
                category="hygiene",
                detail=(
                    f"origin = {url}\n"
                    + (
                        "This is an HTTPS remote. Unattended pushes from the server "
                        "would need a stored token; a read/write deploy key over SSH "
                        "keeps the credential machine-scoped and revocable instead."
                        if uses_https else
                        "SSH remote — suitable for deploy-key based unattended sync."
                    )
                ),
                evidence={"origin": url, "transport": "https" if uses_https else "ssh"},
            )
        ]
