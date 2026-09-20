"""Finding model shared by every probe.

A Finding is an *observation about the pipeline*, not a pytest assertion.
Probes never raise; they return Findings so a single run can surface many
independent problems at once.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Callable, Iterator


class Severity(IntEnum):
    """Ordered so `sorted(..., reverse=True)` puts the worst first."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name


# What kind of failure this is. Used for grouping in the report.
CATEGORIES = (
    "silent-wrong-answer",   # returns a plausible but incorrect value
    "crash",                 # raises where it should handle or fail clearly
    "causality",             # future information reachable from a causal path
    "leakage",               # test data influencing a fit
    "latency",               # realtime budget violated
    "contract",              # documented behaviour not enforced
    "ingest",                # real lab data cannot flow through
    "ui-robustness",         # UI errors / crashes
    "ui-usability",          # UI works but misleads or dead-ends the user
    "hygiene",               # repo / IP / operational hygiene
)


@dataclass
class Finding:
    probe: str
    title: str
    severity: Severity
    category: str
    detail: str
    where: str = ""                      # module:function or file path
    evidence: dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""
    repro: str = ""                      # copy-pasteable snippet
    # Whether a real caller can currently reach this. A defect that is correct
    # in isolation but unreachable today is still worth fixing — but it should
    # not compete for attention with one that fires on the live path now.
    reachability: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = int(self.severity)
        d["severity_label"] = self.severity.label
        return d


@dataclass
class ProbeResult:
    name: str
    findings: list[Finding] = field(default_factory=list)
    checks_run: int = 0
    duration_s: float = 0.0
    skipped: str = ""                    # non-empty reason => probe did not run
    crashed: str = ""                    # traceback if the probe itself broke

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checks_run": self.checks_run,
            "duration_s": round(self.duration_s, 3),
            "skipped": self.skipped,
            "crashed": self.crashed,
            "findings": [f.to_dict() for f in self.findings],
        }


class Probe:
    """Base class. Subclasses implement `checks()` yielding Findings."""

    name = "probe"
    description = ""

    def __init__(self, ctx: "RunContext"):
        self.ctx = ctx
        self._checks_run = 0

    def checks(self) -> Iterator[Finding]:  # pragma: no cover - interface
        raise NotImplementedError

    # -- helpers available to every probe -------------------------------

    def check(self, fn: Callable[[], Iterator[Finding] | list[Finding] | None]):
        """Run one check, converting an unexpected exception into a Finding.

        A probe that crashes is itself a signal, but it must not take down
        the other checks in the same probe.
        """
        self._checks_run += 1
        try:
            out = fn()
            if out:
                yield from out
        except Exception:
            yield Finding(
                probe=self.name,
                title=f"Probe check `{getattr(fn, '__name__', fn)}` raised",
                severity=Severity.LOW,
                category="contract",
                detail=(
                    "The pressure-test check itself crashed. This is usually a "
                    "harness bug or an API that changed shape, not necessarily a "
                    "pipeline defect — but it means this check did not run."
                ),
                evidence={"traceback": traceback.format_exc()[-2000:]},
            )

    def run(self) -> ProbeResult:
        t0 = time.perf_counter()
        res = ProbeResult(name=self.name)
        try:
            res.findings = list(self.checks())
        except Exception:
            res.crashed = traceback.format_exc()[-4000:]
        res.checks_run = self._checks_run
        res.duration_s = time.perf_counter() - t0
        return res


@dataclass
class RunContext:
    """Everything a probe may need, resolved once."""

    repo_root: Any
    experiment_dir: Any = None           # real dataset, if the operator gave one
    quick: bool = False                  # skip slow checks
    seed: int = 0
    latency_budget_ms: float = 20.0      # per-decode-step budget
    ui_timeout_s: float = 120.0

    def env(self) -> dict[str, Any]:
        import importlib

        vers: dict[str, str] = {}
        for mod in ("numpy", "pandas", "scipy", "sklearn", "streamlit", "plotly", "joblib"):
            try:
                vers[mod] = getattr(importlib.import_module(mod), "__version__", "?")
            except Exception:
                vers[mod] = "missing"
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "host": platform.node(),
            "packages": vers,
            "repo_root": str(self.repo_root),
            "experiment_dir": str(self.experiment_dir) if self.experiment_dir else None,
            "quick": self.quick,
            "latency_budget_ms": self.latency_budget_ms,
        }


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, sort_keys=False)
