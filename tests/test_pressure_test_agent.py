"""The agent must not rot, and must not cry wolf.

These tests check the harness itself, not the pipeline: that the oracle is
right, that a probe which breaks degrades instead of exploding, and that the
report round-trips. Pipeline findings belong in the report, not here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agents.pressure_test.findings import Finding, Probe, RunContext, Severity
from agents.pressure_test import report as report_mod
from agents.pressure_test import synth


def test_oracle_counts_match_a_hand_worked_example():
    ss = synth.SpikeSet(
        "hand",
        pd.DataFrame({"time": [0.05, 0.15, 0.25, 0.35], "unit_id": [1, 2, 1, 1]}),
        [1, 2],
        "time",
        "unit_id",
    )
    # [0.1, 0.3): unit 1 has one spike (0.25), unit 2 has one (0.15).
    assert ss.truth_counts(0.1, 0.3).tolist() == [1.0, 1.0]
    # Half-open: the left edge is in, the right edge is out.
    assert ss.truth_counts(0.05, 0.25).tolist() == [1.0, 1.0]
    assert ss.truth_counts(0.0, 0.05).tolist() == [0.0, 0.0]


def test_oracle_shares_no_code_with_the_pipeline():
    import inspect

    src = inspect.getsource(synth.SpikeSet.truth_counts)
    # Compare code only: the docstring names these on purpose.
    body = src.replace(synth.SpikeSet.truth_counts.__doc__ or "", "")
    assert "searchsorted" not in body
    assert "realtime" not in body


def test_adversarial_sets_are_deterministic():
    a = synth.adversarial_sets(7)
    b = synth.adversarial_sets(7)
    assert [s.name for s in a] == [s.name for s in b]
    for x, y in zip(a, b):
        assert x.frame.equals(y.frame), x.name


def test_phy_schemas_use_vendor_column_names():
    ks = synth.phy_like("phy_spike_time_cluster_id")
    assert "cluster_id" in ks.frame.columns
    assert "unit_id" not in ks.frame.columns
    # Phy cluster ids are sparse and never 0..N-1.
    assert ks.unit_ids != list(range(len(ks.unit_ids)))


def test_a_broken_check_becomes_a_finding_not_a_crash(tmp_path):
    class Exploding(Probe):
        name = "exploding"

        def checks(self):
            def boom():
                raise RuntimeError("deliberate")

            yield from self.check(boom)

    res = Exploding(RunContext(repo_root=tmp_path)).run()
    assert res.crashed == ""
    assert len(res.findings) == 1
    assert res.findings[0].severity is Severity.LOW
    assert "raised" in res.findings[0].title


def test_report_round_trips(tmp_path):
    from agents.pressure_test.findings import ProbeResult

    r = ProbeResult(name="demo", checks_run=1)
    r.findings = [
        Finding(
            probe="demo",
            title="a finding",
            severity=Severity.HIGH,
            category="contract",
            detail="because",
            evidence={"n": 1},
        )
    ]
    j = report_mod.write_json(tmp_path / "r.json", {"host": "t"}, [r])
    m = report_mod.write_markdown(tmp_path / "r.md", {"host": "t"}, [r])
    payload = json.loads(j.read_text())
    assert payload["summary"]["HIGH"] == 1
    assert payload["probes"][0]["findings"][0]["severity_label"] == "HIGH"
    assert "a finding" in m.read_text()


@pytest.mark.parametrize("probe_name", ["correctness", "causality", "ingest"])
def test_probe_runs_without_crashing(probe_name, tmp_path):
    from agents.pressure_test.runner import _load_probes, _REPO_ROOT

    cls = _load_probes()[probe_name]
    res = cls(RunContext(repo_root=_REPO_ROOT, quick=True)).run()
    assert res.crashed == "", res.crashed
    assert res.checks_run > 0
    # A check that silently stopped working looks exactly like a clean run, so
    # require that no check degraded into the harness-error finding.
    harness_errors = [f for f in res.findings if "Probe check" in f.title]
    assert not harness_errors, [f.evidence for f in harness_errors]


def test_findings_carry_enough_to_act_on(tmp_path):
    """Every HIGH+ finding needs a location and a suggested fix."""
    from agents.pressure_test.runner import _load_probes, _REPO_ROOT

    ctx = RunContext(repo_root=_REPO_ROOT, quick=True)
    for name in ("correctness", "ingest"):
        for f in _load_probes()[name](ctx).run().findings:
            if f.severity >= Severity.HIGH:
                assert f.where, f.title
                assert f.detail.strip(), f.title
