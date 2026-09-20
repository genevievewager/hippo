"""Headless Streamlit driving: does it work, and does it help?

Two different questions, deliberately kept apart:

  robustness  — the page renders with no exception, in the states a user will
                actually arrive in (nothing configured yet; a dataset selected
                but no features; a stale upstream stage).

  usability   — the page, when it cannot do anything yet, *says what to do
                next*. A blank page, a bare "No data", or a raw exception is a
                dead end. For a seven-page pipeline where later pages inherit
                state from earlier ones, dead ends are the dominant usability
                failure mode.

Uses streamlit.testing.v1.AppTest: no browser, no display, no ports.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import textwrap
from pathlib import Path

from ..findings import Finding, Probe, Severity

# The pipeline order the README documents. Position matters: page N is
# expected to explain what page N-1 owes it.
PAGES: list[tuple[str, str]] = [
    ("home", "Home"),
    ("experiment_setup", "Experiment Setup"),
    ("neural_simulation", "Neural Simulation"),
    ("feature_explorer", "Feature Construction"),
    ("manifold_explorer", "Latent Representations"),
    ("decoder_benchmark", "Decoder Benchmark"),
    ("realtime_replay", "Realtime Replay"),
    ("live_deployment", "Live Deployment"),
]

# Text that means "we know you're stuck, here's the way out".
_GUIDANCE = re.compile(
    r"(go to|open|start (with|on)|first|then|use the|select a|generate a|"
    r"run the|create a|on the .* page|experiment setup|feature construction)",
    re.I,
)

# Text a user should never be shown.
_LEAKED_INTERNALS = re.compile(
    r"(Traceback \(most recent call last\)|KeyError|AttributeError|TypeError:|"
    r"IndexError|NoneType|/home/|/Users/|site-packages)",
)

_HARNESS = """
import sys
from pathlib import Path
sys.path.insert(0, {repo!r})

import streamlit as st
from ui import state
from ui.views import {module} as _view

OUTPUTS_ROOT = Path({outputs!r})
state.init_session_state(OUTPUTS_ROOT)
for k, v in {overrides!r}.items():
    st.session_state[k] = v
_view.render(OUTPUTS_ROOT)
"""


class UIProbe(Probe):
    name = "ui"
    description = "Headless render of every pipeline page, empty and mid-pipeline"

    def checks(self):
        yield from self.check(self._render_all_pages_empty)
        yield from self.check(self._dead_end_guidance)
        yield from self.check(self._no_leaked_internals)
        yield from self.check(self._destructive_actions_need_confirmation)
        yield from self.check(self._expensive_actions_are_explicit)
        yield from self.check(self._stale_upstream_state)
        yield from self.check(self._render_census)

    # -- harness ---------------------------------------------------------

    def _run_page(self, module: str, *, outputs: Path, overrides: dict | None = None):
        from streamlit.testing.v1 import AppTest

        script = textwrap.dedent(
            _HARNESS.format(
                repo=str(self.ctx.repo_root),
                module=module,
                outputs=str(outputs),
                overrides=overrides or {},
            )
        )
        at = AppTest.from_string(script)
        at.run(timeout=self.ctx.ui_timeout_s)
        return at

    @staticmethod
    def _all_text(at) -> str:
        parts: list[str] = []
        for attr in ("markdown", "text", "info", "warning", "error", "success",
                     "caption", "header", "subheader", "title"):
            try:
                parts.extend(str(getattr(e, "value", "")) for e in getattr(at, attr))
            except Exception:
                pass
        return "\n".join(parts)

    def _empty_outputs(self) -> Path:
        d = Path(tempfile.mkdtemp(prefix="hippo_ui_probe_"))
        self._tmpdirs.append(d)
        return d

    _tmpdirs: list[Path] = []

    def run(self):  # cleanup around the base implementation
        self._tmpdirs = []
        try:
            return super().run()
        finally:
            for d in self._tmpdirs:
                shutil.rmtree(d, ignore_errors=True)

    # -- checks ----------------------------------------------------------

    def _render_all_pages_empty(self):
        """The first-run state: no datasets, nothing configured."""
        out = []
        outputs = self._empty_outputs()
        for module, title in PAGES:
            try:
                at = self._run_page(module, outputs=outputs)
            except Exception as exc:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Page `{title}` could not be rendered at all",
                        severity=Severity.HIGH,
                        category="ui-robustness",
                        where=f"ui/views/{module}.py",
                        detail=f"Harness raised {type(exc).__name__}: {exc}",
                    )
                )
                continue
            if at.exception:
                first = at.exception[0]
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Page `{title}` raises on a fresh install",
                        severity=Severity.HIGH,
                        category="ui-robustness",
                        where=f"ui/views/{module}.py",
                        detail=(
                            "Rendered with an empty outputs/ directory — the state a "
                            "new user or a new lab machine starts in. The page throws "
                            "instead of explaining that no dataset exists yet."
                        ),
                        evidence={
                            "exception": str(getattr(first, "value", first))[:800],
                            "n_exceptions": len(at.exception),
                        },
                    )
                )
        return out

    def _dead_end_guidance(self):
        """A page that can't proceed must name the page that unblocks it."""
        out = []
        outputs = self._empty_outputs()
        for i, (module, title) in enumerate(PAGES):
            if module in {"home", "experiment_setup"}:
                continue  # entry points; nothing upstream to point at
            try:
                at = self._run_page(module, outputs=outputs)
            except Exception:
                continue
            if at.exception:
                continue  # already reported as robustness
            text = self._all_text(at)
            n_widgets = sum(
                len(getattr(at, a, []))
                for a in ("button", "selectbox", "slider", "multiselect",
                          "number_input", "text_input", "checkbox", "radio")
            )
            if text.strip() and _GUIDANCE.search(text):
                continue
            upstream = PAGES[i - 1][1]
            if not text.strip() and n_widgets == 0:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Page `{title}` renders blank with no dataset",
                        severity=Severity.MEDIUM,
                        category="ui-usability",
                        where=f"ui/views/{module}.py",
                        detail=(
                            "No text and no controls. A user landing here cannot tell "
                            "whether the page is broken, still loading, or waiting on "
                            "an earlier stage."
                        ),
                        suggestion=(
                            f"`st.info` naming the blocker and the fix: "
                            f'"No active dataset. Generate or load one on **{upstream}**, '
                            'then return here."'
                        ),
                    )
                )
            else:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Page `{title}` blocks without saying what to do next",
                        severity=Severity.LOW,
                        category="ui-usability",
                        where=f"ui/views/{module}.py",
                        detail=(
                            "The page renders in its empty state but no message points "
                            "at the upstream page that produces what it needs. In a "
                            "seven-stage pipeline where each page inherits the previous "
                            "page's committed state, this is where users get stuck."
                        ),
                        evidence={"rendered_text_head": text[:400]},
                        suggestion=f"Name **{upstream}** explicitly in the empty state.",
                    )
                )
        return out

    def _no_leaked_internals(self):
        """User-facing errors must not be raw Python or absolute paths."""
        out = []
        outputs = self._empty_outputs()
        for module, title in PAGES:
            try:
                at = self._run_page(module, outputs=outputs)
            except Exception:
                continue
            messages = []
            for attr in ("error", "warning"):
                try:
                    messages += [str(getattr(e, "value", "")) for e in getattr(at, attr)]
                except Exception:
                    pass
            leaked = [m for m in messages if _LEAKED_INTERNALS.search(m)]
            if leaked:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Page `{title}` shows internal detail in a user message",
                        severity=Severity.LOW,
                        category="ui-usability",
                        where=f"ui/views/{module}.py",
                        detail=(
                            "An st.error/st.warning contains a Python exception name, "
                            "traceback, or absolute filesystem path. Users read this as "
                            "'the tool is broken' rather than 'I need to do X first'."
                        ),
                        evidence={"messages": leaked[:3]},
                        suggestion=(
                            "Catch the specific exception, log the detail, and show a "
                            "sentence describing the cause and the next action."
                        ),
                    )
                )
        return out

    def _expensive_actions_are_explicit(self):
        """A widget change must never launch a Full sweep."""
        import inspect

        out = []
        try:
            from ui.views import decoder_benchmark as mod
        except Exception:
            return out
        src = inspect.getsource(mod)
        requested = "REQUESTED" in src or "requested" in src
        has_opt_in = re.search(r"checkbox\(|Full|full_sweep|confirm", src) is not None
        if not (requested and has_opt_in):
            out.append(
                Finding(
                    probe=self.name,
                    title="Full benchmark may not be behind an explicit opt-in",
                    severity=Severity.MEDIUM,
                    category="ui-usability",
                    where="ui/views/decoder_benchmark.py",
                    detail=(
                        "Streamlit re-runs the whole script on every widget change. "
                        "The README states a Full F x E x D x W sweep must require an "
                        "opt-in checkbox plus an explicit Run, but the action-flag and "
                        "opt-in pattern was not detected in this page. An accidental "
                        "multi-hour sweep is the most expensive UI mistake available."
                    ),
                    suggestion=(
                        "Keep the `*_REQUESTED` flag + consume-once pattern from "
                        "ui/state.py on every expensive path, and assert it in a test."
                    ),
                )
            )
        return out

    def _stale_upstream_state(self):
        """Point a page at a dataset directory that does not exist.

        This is not a contrived state: an experiment directory can be moved,
        deleted, or live on a mount that is not currently attached, while the
        session still holds the old path.
        """
        out = []
        outputs = self._empty_outputs()
        ghost = str(outputs / "experiment_that_was_deleted")
        from ui import state as ui_state

        overrides = {ui_state.KEY_ACTIVE_DATASET: ghost}
        for module, title in PAGES:
            if module == "home":
                continue
            try:
                at = self._run_page(module, outputs=outputs, overrides=overrides)
            except Exception as exc:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Page `{title}` cannot render with a missing dataset path",
                        severity=Severity.MEDIUM,
                        category="ui-robustness",
                        where=f"ui/views/{module}.py",
                        detail=f"Harness raised {type(exc).__name__}: {exc}",
                    )
                )
                continue
            if at.exception:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Page `{title}` raises when the active dataset is gone",
                        severity=Severity.MEDIUM,
                        category="ui-robustness",
                        where=f"ui/views/{module}.py",
                        detail=(
                            "The session pointed at an experiment directory that no "
                            "longer exists (moved, deleted, or on an unmounted share) "
                            "and the page threw rather than reporting that the dataset "
                            "is missing and offering to pick another."
                        ),
                        evidence={
                            "active_dataset": ghost,
                            "exception": str(
                                getattr(at.exception[0], "value", at.exception[0])
                            )[:600],
                        },
                        suggestion=(
                            "Validate the active dataset path once in "
                            "state.get_active_dataset() and clear it with an "
                            "explanatory st.warning when it does not resolve."
                        ),
                    )
                )
        return out

    def _render_census(self):
        """Positive evidence: what each page actually produced when rendered."""
        outputs = self._empty_outputs()
        census: dict[str, dict] = {}
        for module, title in PAGES:
            try:
                at = self._run_page(module, outputs=outputs)
            except Exception as exc:
                census[title] = {"rendered": False, "error": f"{type(exc).__name__}"}
                continue
            widgets = {
                a: len(getattr(at, a, []))
                for a in ("button", "selectbox", "slider", "multiselect",
                          "number_input", "text_input", "checkbox", "radio")
            }
            census[title] = {
                "rendered": True,
                "exceptions": len(at.exception),
                "text_chars": len(self._all_text(at)),
                "widgets": {k: v for k, v in widgets.items() if v},
            }
        return [
            Finding(
                probe=self.name,
                title="Page render census",
                severity=Severity.INFO,
                category="ui-robustness",
                detail=(
                    "Every pipeline page was rendered headlessly against an empty "
                    "outputs/ directory. Recorded so that a page going blank, losing "
                    "its controls, or starting to throw shows up as a diff between "
                    "runs rather than being noticed by a user."
                ),
                evidence={"pages": census},
            )
        ]

    def _destructive_actions_need_confirmation(self):
        """Nothing should delete an experiment on a single click."""
        import inspect

        out = []
        risky = re.compile(r"(shutil\.rmtree|os\.remove|\.unlink\(|rm -rf)")
        for module, title in PAGES:
            try:
                mod = __import__(f"ui.views.{module}", fromlist=["render"])
                src = inspect.getsource(mod)
            except Exception:
                continue
            if not risky.search(src):
                continue
            if not re.search(r"(confirm|are you sure|checkbox|type the name)", src, re.I):
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Page `{title}` deletes files without a confirmation step",
                        severity=Severity.MEDIUM,
                        category="ui-usability",
                        where=f"ui/views/{module}.py",
                        detail=(
                            "A destructive filesystem call appears with no confirmation "
                            "guard. Streamlit reruns make single-click deletion easy to "
                            "trigger by accident, and experiment directories hold hours "
                            "of compute."
                        ),
                    )
                )
        return out
