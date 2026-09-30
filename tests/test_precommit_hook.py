"""Tests for scripts/hooks/check_no_junk.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "hooks" / "check_no_junk.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("check_no_junk", HOOK)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hook = _load_hook()


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    """Minimal fake repo tree with the three fixtures present."""
    for rel in hook.ALLOWLIST:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture\n", encoding="utf-8")
    return tmp_path


def test_rejects_csv_under_data(repo_root: Path) -> None:
    target = repo_root / "data" / "secret.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    errors = hook.check_paths(["data/secret.csv"], root=repo_root)
    assert errors
    assert any("blocked path" in e for e in errors)


def test_rejects_file_over_5mb(repo_root: Path) -> None:
    target = repo_root / "big.bin"
    target.write_bytes(b"\0" * (5 * 1024 * 1024 + 1))
    errors = hook.check_paths(["big.bin"], root=repo_root)
    assert errors
    assert any("bytes" in e for e in errors)


@pytest.mark.parametrize("rel", sorted(hook.ALLOWLIST))
def test_accepts_fixtures(repo_root: Path, rel: str) -> None:
    errors = hook.check_paths([rel], root=repo_root)
    assert errors == []


def test_accepts_agents_quadrant_n5(repo_root: Path) -> None:
    """Root /quadrant_n5/ is blocked; agents/quadrant_n5/ must stay allowed."""
    rel = "agents/quadrant_n5/SPEC.md"
    target = repo_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# ok\n", encoding="utf-8")
    assert hook.check_paths([rel], root=repo_root) == []


def test_rejects_root_quadrant_n5(repo_root: Path) -> None:
    rel = "quadrant_n5/report.json"
    target = repo_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}\n", encoding="utf-8")
    errors = hook.check_paths([rel], root=repo_root)
    assert errors
    assert any("blocked path" in e for e in errors)