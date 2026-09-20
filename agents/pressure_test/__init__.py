"""Pressure-test agent for the hippocampal BCI pipeline and UI.

Entry point::

    python -m agents.pressure_test
"""

from .findings import Finding, ProbeResult, RunContext, Severity  # noqa: F401

__all__ = ["Finding", "ProbeResult", "RunContext", "Severity"]
