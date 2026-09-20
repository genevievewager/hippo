"""Report rendering: one JSON for machines, one Markdown for humans."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from .findings import Finding, ProbeResult, Severity

_ICON = {
    Severity.CRITICAL: "CRITICAL",
    Severity.HIGH: "HIGH",
    Severity.MEDIUM: "MEDIUM",
    Severity.LOW: "LOW",
    Severity.INFO: "INFO",
}


def _all_findings(results: list[ProbeResult]) -> list[Finding]:
    out: list[Finding] = []
    for r in results:
        out.extend(r.findings)
    return sorted(out, key=lambda f: (-int(f.severity), f.probe, f.title))


def summarize(results: list[ProbeResult]) -> dict[str, int]:
    counts = {s.label: 0 for s in Severity}
    for f in _all_findings(results):
        counts[f.severity.label] += 1
    return counts


def write_json(path: Path, env: dict[str, Any], results: list[ProbeResult]) -> Path:
    payload = {
        "schema": "hippo.pressure_test/1",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": env,
        "summary": summarize(results),
        "probes": [r.to_dict() for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def _fmt_evidence(ev: dict[str, Any]) -> str:
    if not ev:
        return ""
    body = json.dumps(ev, indent=2, default=str)
    if len(body) > 1800:
        body = body[:1800] + "\n  ... (truncated; see the JSON report)"
    return f"\n<details><summary>evidence</summary>\n\n```json\n{body}\n```\n</details>\n"


def write_markdown(path: Path, env: dict[str, Any], results: list[ProbeResult]) -> Path:
    findings = _all_findings(results)
    counts = summarize(results)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append("# Pipeline pressure test")
    lines.append("")
    lines.append(f"`{env.get('host','?')}` · {now} · python {env.get('python','?')}")
    lines.append("")

    order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
    head = " · ".join(f"**{counts[s.label]}** {s.label.lower()}" for s in order)
    lines.append(head)
    lines.append("")

    if counts["CRITICAL"] or counts["HIGH"]:
        lines.append(
            "> Critical and high findings are ones that produce a wrong number or a "
            "missed decode step without raising. They are not style issues."
        )
        lines.append("")

    # Probe roll-up.
    lines.append("| probe | checks | findings | time |")
    lines.append("|---|---:|---:|---:|")
    for r in results:
        status = r.skipped or ("CRASHED" if r.crashed else str(len(r.findings)))
        lines.append(f"| {r.name} | {r.checks_run} | {status} | {r.duration_s:.1f}s |")
    lines.append("")

    for sev in order:
        group = [f for f in findings if f.severity is sev]
        if not group:
            continue
        lines.append(f"## {_ICON[sev]} ({len(group)})")
        lines.append("")
        for f in group:
            lines.append(f"### {f.title}")
            lines.append("")
            meta = [f"`{f.category}`"]
            if f.where:
                meta.append(f"`{f.where}`")
            meta.append(f"probe: `{f.probe}`")
            lines.append(" · ".join(meta))
            lines.append("")
            lines.append(f.detail)
            lines.append("")
            if f.reachability:
                lines.append(f"**Reachable today?** {f.reachability}")
                lines.append("")
            if f.repro:
                lines.append("**Reproduce**")
                lines.append("")
                lines.append("```python")
                lines.append(f.repro)
                lines.append("```")
                lines.append("")
            if f.suggestion:
                lines.append(f"**Fix** — {f.suggestion}")
                lines.append("")
            ev = _fmt_evidence(f.evidence)
            if ev:
                lines.append(ev)
            lines.append("")

    crashed = [r for r in results if r.crashed]
    if crashed:
        lines.append("## Probe failures")
        lines.append("")
        lines.append("These probes did not complete, so their area is unmeasured.")
        lines.append("")
        for r in crashed:
            lines.append(f"### {r.name}")
            lines.append("")
            lines.append("```")
            lines.append(r.crashed[-1500:])
            lines.append("```")
            lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(env, indent=2, default=str))
    lines.append("```")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def console_summary(results: list[ProbeResult]) -> str:
    counts = summarize(results)
    findings = _all_findings(results)
    out = [
        "",
        "pressure test — "
        + ", ".join(
            f"{counts[s.label]} {s.label.lower()}"
            for s in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM,
                      Severity.LOW, Severity.INFO)
        ),
        "",
    ]
    for f in findings:
        if f.severity >= Severity.MEDIUM:
            out.append(f"  [{f.severity.label:<8}] {f.title}")
            if f.where:
                out.append(f"             {f.where}")
    out.append("")
    return "\n".join(out)
