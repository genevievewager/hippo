"""Realtime budget.

Closed-loop hippocampal BCI work lives or dies on the tail, not the mean: a
p99 that exceeds `update_dt` means dropped decode steps, and a decoder that
drops steps under high firing rates drops them exactly when the animal is most
active — i.e. when the signal is there.

So this probe measures the hot path at escalating spike rates and buffer
depths, and reports the *scaling*, not just one number.
"""

from __future__ import annotations

import gc
import time

import numpy as np

from ..findings import Finding, Probe, Severity
from ..synth import poisson_spikes


def _percentile_ms(samples: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=float) * 1e3, q))


class LatencyProbe(Probe):
    name = "latency"
    description = "Per-decode-step cost and how it scales with load"

    def checks(self):
        yield from self.check(self._buffer_counts_at_scaling)
        yield from self.check(self._buffer_extend_scaling)
        yield from self.check(self._offline_matrix_scaling)

    # -- helpers ---------------------------------------------------------

    def _time_calls(self, fn, n: int) -> list[float]:
        gc.disable()
        try:
            out = []
            for _ in range(n):
                t0 = time.perf_counter()
                fn()
                out.append(time.perf_counter() - t0)
            return out
        finally:
            gc.enable()

    # -- checks ----------------------------------------------------------

    def _buffer_counts_at_scaling(self):
        """counts_at is a linear scan: cost grows with retained history."""
        from realtime.live.spike_buffer import CausalSpikeBuffer

        out: list[Finding] = []
        budget = self.ctx.latency_budget_ms
        scaling: dict[str, dict[str, float]] = {}
        configs = (
            [(64, 20.0, 2.0)] if self.ctx.quick
            else [(64, 20.0, 2.0), (256, 20.0, 2.0), (256, 50.0, 5.0), (384, 80.0, 5.0)]
        )
        for n_units, rate_hz, history_s in configs:
            ss = poisson_spikes(
                n_units=n_units, duration_s=history_s + 2.0, rate_hz=rate_hz,
                seed=self.ctx.seed,
            )
            buf = CausalSpikeBuffer(ss.unit_ids, history_s=history_s)
            df = ss.frame
            buf.extend(df["time"].to_numpy(float), df["unit_id"].to_numpy(int))
            t_now = float(df["time"].max())
            reps = 60 if self.ctx.quick else 200
            samples = self._time_calls(lambda: buf.counts_at(t_now, 0.25), reps)
            key = f"{n_units}u@{rate_hz:g}Hz/{history_s:g}s"
            scaling[key] = {
                "buffered_spikes": float(buf.n_spikes),
                "p50_ms": _percentile_ms(samples, 50),
                "p99_ms": _percentile_ms(samples, 99),
            }

        worst = max(scaling.items(), key=lambda kv: kv[1]["p99_ms"])
        if worst[1]["p99_ms"] > budget:
            out.append(
                Finding(
                    probe=self.name,
                    title="counts_at exceeds the realtime budget under load",
                    severity=Severity.HIGH,
                    category="latency",
                    where="realtime/live/spike_buffer.py:counts_at",
                    detail=(
                        f"p99 reached {worst[1]['p99_ms']:.2f} ms at {worst[0]} against "
                        f"a {budget:g} ms budget. counts_at scans the retained deque in "
                        "Python for every decode step, so cost is O(spikes in history), "
                        "not O(spikes in window) — raising history_s makes every step "
                        "slower even though the window did not change."
                    ),
                    evidence={"scaling": scaling, "budget_ms": budget},
                    suggestion=(
                        "Hold the buffer as two numpy arrays with a ring index and use "
                        "searchsorted, or bin incrementally into fixed update_dt bins "
                        "and keep a running window sum. Both make the step O(units)."
                    ),
                )
            )
        else:
            # Under budget today. The useful number is where it stops being so:
            # cost is linear in retained spikes, so fit and extrapolate.
            xs = np.array([v["buffered_spikes"] for v in scaling.values()])
            ys = np.array([v["p99_ms"] for v in scaling.values()])
            headroom = ""
            projected = None
            if xs.size >= 2 and np.ptp(xs) > 0:
                slope, intercept = np.polyfit(xs, ys, 1)
                if slope > 0:
                    projected = float((budget - intercept) / slope)
                    headroom = (
                        f" Cost is linear in retained spikes at "
                        f"{slope * 1000:.3f} ms per 1000 buffered spikes, so the "
                        f"{budget:g} ms budget is reached around "
                        f"{projected:,.0f} buffered spikes — roughly "
                        f"{projected / max(xs.max(), 1):.1f}x the heaviest load tested. "
                        "Doubling history_s or unit count halves that headroom, and "
                        "history_s is a tuning knob that looks unrelated to latency."
                    )
            out.append(
                Finding(
                    probe=self.name,
                    title="counts_at latency measured within budget",
                    severity=Severity.INFO,
                    category="latency",
                    where="realtime/live/spike_buffer.py:counts_at",
                    detail=(
                        f"Worst p99 {worst[1]['p99_ms']:.2f} ms at {worst[0]} against a "
                        f"{budget:g} ms budget.{headroom}"
                    ),
                    evidence={
                        "scaling": scaling,
                        "budget_ms": budget,
                        "projected_budget_crossing_buffered_spikes": projected,
                    },
                    suggestion=(
                        "Record this table each run; a change in slope is the early "
                        "warning, not the first dropped decode step."
                    ),
                )
            )

        # Superlinear growth is the thing to catch early, budget or not.
        keys = list(scaling)
        if len(keys) >= 2:
            first, last = scaling[keys[0]], scaling[keys[-1]]
            spike_ratio = last["buffered_spikes"] / max(first["buffered_spikes"], 1.0)
            time_ratio = last["p50_ms"] / max(first["p50_ms"], 1e-9)
            if spike_ratio > 1.5 and time_ratio > 1.3 * spike_ratio:
                out.append(
                    Finding(
                        probe=self.name,
                        title="Decode-step cost grows faster than spike count",
                        severity=Severity.MEDIUM,
                        category="latency",
                        where="realtime/live/spike_buffer.py",
                        detail=(
                            f"Buffered spikes grew {spike_ratio:.1f}x but median step "
                            f"time grew {time_ratio:.1f}x. Superlinear scaling means the "
                            "loop degrades sharply exactly during high-rate epochs."
                        ),
                        evidence={"scaling": scaling},
                    )
                )
        return out

    def _buffer_extend_scaling(self):
        """extend() appends in a Python loop, once per spike."""
        from realtime.live.spike_buffer import CausalSpikeBuffer

        ss = poisson_spikes(n_units=256, duration_s=1.0, rate_hz=40.0, seed=self.ctx.seed)
        times = ss.frame["time"].to_numpy(float)
        units = ss.frame["unit_id"].to_numpy(int)
        chunk = max(1, times.size // 50)          # ~50 polls of one second

        def run():
            buf = CausalSpikeBuffer(ss.unit_ids, history_s=2.0)
            for i in range(0, times.size, chunk):
                buf.extend(times[i:i + chunk], units[i:i + chunk])

        reps = 3 if self.ctx.quick else 10
        samples = self._time_calls(run, reps)
        per_spike_us = float(np.median(samples)) / max(times.size, 1) * 1e6
        if per_spike_us > 1.0:
            return [
                Finding(
                    probe=self.name,
                    title="Spike ingestion cost is per-spike Python overhead",
                    severity=Severity.MEDIUM,
                    category="latency",
                    where="realtime/live/spike_buffer.py:extend",
                    detail=(
                        f"{per_spike_us:.2f} us of Python per spike on ingest "
                        f"({times.size} spikes in 1 s of 256-unit, 40 Hz data). "
                        "extend() converts to numpy and then discards that by looping "
                        "and appending to two deques one element at a time."
                    ),
                    evidence={
                        "spikes_per_second": int(times.size),
                        "median_total_ms": float(np.median(samples)) * 1e3,
                        "per_spike_us": per_spike_us,
                    },
                    suggestion=(
                        "Append the numpy arrays into a preallocated ring buffer rather "
                        "than element-wise into deques."
                    ),
                )
            ]
        return []

    def _offline_matrix_scaling(self):
        """The training-side matrix build is a per-decode-time Python loop."""
        from realtime.spike_binner import build_causal_spike_matrix

        if self.ctx.quick:
            return []
        ss = poisson_spikes(n_units=200, duration_s=60.0, rate_hz=10.0, seed=self.ctx.seed)
        times = np.arange(1.0, 59.0, 0.02)        # 50 Hz decode over a minute
        t0 = time.perf_counter()
        build_causal_spike_matrix(ss.frame, ss.unit_ids, times, 0.25)
        elapsed = time.perf_counter() - t0
        per_step_ms = elapsed / times.size * 1e3
        if per_step_ms > self.ctx.latency_budget_ms / 4:
            return [
                Finding(
                    probe=self.name,
                    title="Offline feature build is slow enough to bottleneck sweeps",
                    severity=Severity.LOW,
                    category="latency",
                    where="realtime/spike_binner.py:build_causal_spike_matrix",
                    detail=(
                        f"{elapsed:.1f} s for {times.size} decode steps over 200 units "
                        f"({per_step_ms:.2f} ms/step). The F x E x D x W sweep rebuilds "
                        "this per window, so the UI's Full benchmark multiplies it."
                    ),
                    evidence={
                        "n_decode_steps": int(times.size),
                        "n_units": 200,
                        "elapsed_s": elapsed,
                        "per_step_ms": per_step_ms,
                    },
                    suggestion=(
                        "The inner loop rebuilds a Python dict lookup per window. "
                        "Vectorise with a single np.add.at over (bin_index, unit_index) "
                        "or cache per-W count matrices keyed by the existing hash."
                    ),
                )
            ]
        return []
