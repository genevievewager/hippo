"""Orchestration and CLI.

    python -m agents.pressure_test                     # everything
    python -m agents.pressure_test --quick             # fast subset, for pre-commit
    python -m agents.pressure_test --probes ingest,ui
    python -m agents.pressure_test --fail-on high      # non-zero exit for CI/cron

Exit codes
    0  nothing at or above --fail-on
    1  findings at or above --fail-on
    2  the run itself failed

Data handling: every probe is fed synthetic, seeded data generated in-process.
Nothing reads a recording, and nothing leaves the machine. `--experiment-dir`
is the single exception and is opt-in; even then the contents are only read,
never copied into the report beyond shapes and counts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .findings import ProbeResult, RunContext, Severity
from . import report as report_mod

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_probes():
    from .probes.correctness import CorrectnessProbe
    from .probes.causality import CausalityProbe
    from .probes.ingest import IngestProbe
    from .probes.latency import LatencyProbe
    from .probes.ui import UIProbe
    from .probes.hygiene import HygieneProbe

    return {
        p.name: p
        for p in (
            CorrectnessProbe, CausalityProbe, IngestProbe,
            LatencyProbe, UIProbe, HygieneProbe,
        )
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agents.pressure_test",
        description="Pressure-test the hippocampal BCI pipeline and UI.",
    )
    p.add_argument("--probes", default="all",
                   help="comma-separated probe names, or 'all' (default)")
    p.add_argument("--quick", action="store_true",
                   help="skip the slow scaling checks; suitable for a pre-commit hook")
    p.add_argument("--out", default="reports/pressure_test",
                   help="output directory for the report pair (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latency-budget-ms", type=float, default=20.0,
                   help="per-decode-step budget used by the latency probe")
    p.add_argument("--experiment-dir", default=None,
                   help="optional real experiment directory to additionally validate")
    p.add_argument("--fail-on", default="none",
                   choices=["none", "info", "low", "medium", "high", "critical"],
                   help="exit non-zero if any finding is at or above this severity")
    p.add_argument("--repo-root", default=str(_REPO_ROOT))
    p.add_argument("--quiet", action="store_true")
    return p


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    registry = _load_probes()
    if args.probes.strip().lower() == "all":
        selected = list(registry.values())
    else:
        names = [n.strip() for n in args.probes.split(",") if n.strip()]
        unknown = [n for n in names if n not in registry]
        if unknown:
            print(f"unknown probe(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"available: {', '.join(registry)}", file=sys.stderr)
            return 2
        selected = [registry[n] for n in names]

    ctx = RunContext(
        repo_root=repo_root,
        experiment_dir=Path(args.experiment_dir) if args.experiment_dir else None,
        quick=args.quick,
        seed=args.seed,
        latency_budget_ms=args.latency_budget_ms,
    )

    results: list[ProbeResult] = []
    t0 = time.perf_counter()
    for cls in selected:
        if not args.quiet:
            print(f"  · {cls.name:<12} {cls.description}", file=sys.stderr, flush=True)
        try:
            results.append(cls(ctx).run())
        except Exception as exc:  # a probe that cannot even construct
            r = ProbeResult(name=cls.name)
            r.crashed = f"{type(exc).__name__}: {exc}"
            results.append(r)
    elapsed = time.perf_counter() - t0

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    env = ctx.env()
    env["elapsed_s"] = round(elapsed, 2)
    env["probes_run"] = [c.name for c in selected]

    json_path = report_mod.write_json(out_dir / "report.json", env, results)
    md_path = report_mod.write_markdown(out_dir / "report.md", env, results)

    if not args.quiet:
        print(report_mod.console_summary(results))
        print(f"  {md_path}")
        print(f"  {json_path}")
        print()

    if args.fail_on == "none":
        return 0
    threshold = Severity[args.fail_on.upper()]
    worst = max(
        (f.severity for r in results for f in r.findings),
        default=Severity.INFO,
    )
    has_any = any(f.severity >= threshold for r in results for f in r.findings)
    if has_any:
        if not args.quiet:
            print(
                f"FAIL: findings at or above {threshold.label} "
                f"(worst: {worst.label})",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
