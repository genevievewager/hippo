"""Function-level correctness against an independent oracle.

The rule this probe enforces: *no function on the decode path may return a
plausible-looking wrong number.* It must either be right, or fail loudly.

Every count produced by the repo is compared against `SpikeSet.truth_counts`,
a brute-force oracle that shares no code with the pipeline.
"""

from __future__ import annotations

import inspect
import numpy as np
import pandas as pd

from ..findings import Finding, Probe, Severity
from ..synth import SpikeSet, adversarial_sets, poisson_spikes, units_frame


class CorrectnessProbe(Probe):
    name = "correctness"
    description = "Spike counting and feature assembly vs. an independent oracle"

    def checks(self):
        yield from self.check(self._causal_matrix_vs_oracle)
        yield from self.check(self._count_window_vs_oracle)
        yield from self.check(self._count_window_contract_enforcement)
        yield from self.check(self._buffer_vs_oracle)
        yield from self.check(self._buffer_prune_on_glitch)
        yield from self.check(self._unit_id_alignment)
        yield from self.check(self._offline_online_agreement)
        yield from self.check(self._nan_and_inf_propagation)
        yield from self.check(self._buffer_nan_poisoning)
        yield from self.check(self._transform_one_prev_counts_consistency)
        yield from self.check(self._live_decoder_double_counting)

    # -- oracle comparisons ---------------------------------------------

    def _windows(self, ss: SpikeSet) -> list[tuple[float, float]]:
        df = ss.canonical()
        if df.empty:
            return [(0.0, 1.0)]
        lo = float(df["time"].min())
        hi = float(min(df["time"].max(), lo + 5.0))
        span = max(hi - lo, 1e-6)
        return [
            (lo, hi),                                   # everything
            (lo + 0.2 * span, lo + 0.6 * span),         # interior slice
            (lo - 1.0, lo),                             # entirely before
            (hi, hi + 1.0),                             # entirely after
            (lo + 0.5 * span, lo + 0.5 * span),         # zero-width
        ]

    def _causal_matrix_vs_oracle(self):
        from realtime.spike_binner import build_causal_spike_matrix

        out = []
        for ss in adversarial_sets(self.ctx.seed):
            df = ss.frame.rename(columns={ss.time_col: "time", ss.unit_col: "unit_id"})
            for t_start, t_end in self._windows(ss):
                w = t_end - t_start
                if w <= 0:
                    continue
                got = build_causal_spike_matrix(df, ss.unit_ids, np.array([t_end]), w)[0]
                want = ss.truth_counts(t_start, t_end)
                if not np.array_equal(got, want):
                    out.append(
                        Finding(
                            probe=self.name,
                            title=f"build_causal_spike_matrix miscounts on `{ss.name}` input",
                            severity=Severity.CRITICAL,
                            category="silent-wrong-answer",
                            where="realtime/spike_binner.py:build_causal_spike_matrix",
                            detail=(
                                f"Window [{t_start:.4f}, {t_end:.4f}) over {ss.note}. "
                                f"Returned counts differ from brute-force truth with no "
                                f"error raised. Every downstream feature, representation "
                                f"and decoder inherits this."
                            ),
                            evidence={
                                "input": ss.name,
                                "returned": got.tolist(),
                                "truth": want.tolist(),
                                "delta": (got - want).tolist(),
                            },
                            suggestion=(
                                "Either sort defensively or validate and raise. Silently "
                                "accepting unsorted input is the failure mode here."
                            ),
                        )
                    )
                    break  # one finding per input class is enough
        return out

    def _count_window_vs_oracle(self):
        from realtime.spike_binner import count_spikes_in_window

        out = []
        for ss in adversarial_sets(self.ctx.seed):
            df = ss.frame.rename(columns={ss.time_col: "time", ss.unit_col: "unit_id"})
            for t_start, t_end in self._windows(ss):
                if t_end <= t_start:
                    continue
                try:
                    got = count_spikes_in_window(df, ss.unit_ids, t_start, t_end)
                except (ValueError, TypeError):
                    # Refusing input it cannot count correctly is the desired
                    # behaviour. The failure mode this probe exists to catch is
                    # returning a wrong number, not raising.
                    break
                want = ss.truth_counts(t_start, t_end)
                if not np.array_equal(got, want):
                    out.append(
                        Finding(
                            probe=self.name,
                            title=f"count_spikes_in_window miscounts on `{ss.name}` input",
                            severity=Severity.CRITICAL,
                            category="silent-wrong-answer",
                            where="realtime/spike_binner.py:count_spikes_in_window",
                            detail=(
                                f"Window [{t_start:.4f}, {t_end:.4f}) over {ss.note}. "
                                "The function's docstring states spike times must be "
                                "sorted ascending, but nothing enforces it: it runs "
                                "np.searchsorted on whatever it is given and returns a "
                                "wrong count with no warning."
                            ),
                            evidence={
                                "input": ss.name,
                                "returned": got.tolist(),
                                "truth": want.tolist(),
                                "delta": (got - want).tolist(),
                            },
                            repro=(
                                "import pandas as pd\n"
                                "from realtime.spike_binner import count_spikes_in_window\n"
                                "s = pd.DataFrame({'time':[0.9,0.1,0.5,0.7,0.3],"
                                "'unit_id':[1]*5})\n"
                                "count_spikes_in_window(s, [1], 0.2, 0.6)  "
                                "# -> [3.], truth is [2.]"
                            ),
                            reachability=(
                                "Indirectly. Every offline caller sorts first "
                                "(data_loading.load_simulation_data, "
                                "build_causal_spike_matrix, RealTimeDecoder.replay). "
                                "The one unsorted-capable feeder is the live path: "
                                "LiveDecoder -> CausalSpikeBuffer.as_dataframe() -> "
                                "NeuralFeatureExtractor -> count_spikes_in_window, and "
                                "as_dataframe returns arrival order. That order is "
                                "sorted today, and stops being sorted permanently once "
                                "the buffer takes a NaN or glitched sample."
                            ),
                            suggestion=(
                                "Add an O(1) monotonicity guard at the top "
                                "(`times[:-1] <= times[1:]` checked once per frame, or a "
                                "`_sorted` flag set by the loader) and raise "
                                "PipelineInvariantError otherwise."
                            ),
                        )
                    )
                    break
        return out

    def _count_window_contract_enforcement(self):
        """The docstring makes a promise. Is it a checked promise?"""
        from realtime.spike_binner import count_spikes_in_window

        src = inspect.getsource(count_spikes_in_window)
        documents_sorted = "sorted" in (count_spikes_in_window.__doc__ or "").lower()
        guards = any(
            tok in src
            for tok in ("is_monotonic", "np.all(np.diff", "assert_sorted", "np.diff(times)",
                        "times[:-1] <= times[1:]", "raise ValueError")
        )
        if documents_sorted and not guards:
            return [
                Finding(
                    probe=self.name,
                    title="Documented sortedness precondition is unenforced",
                    severity=Severity.HIGH,
                    category="contract",
                    where="realtime/spike_binner.py:count_spikes_in_window",
                    detail=(
                        "The docstring requires ascending spike times and defers the "
                        "responsibility to callers. No caller is verified, and one of "
                        "them (CausalSpikeBuffer.as_dataframe) can produce unsorted "
                        "frames. An unchecked precondition on the hot path of a "
                        "closed-loop BCI is a correctness hazard, not a style issue."
                    ),
                    suggestion=(
                        "Promote it to a checked invariant in "
                        "realtime/pipeline_invariants.py, e.g. "
                        "`assert_spike_times_sorted(times, context=...)`."
                    ),
                )
            ]
        return []

    def _buffer_vs_oracle(self):
        from realtime.live.spike_buffer import CausalSpikeBuffer

        out = []
        for ss in adversarial_sets(self.ctx.seed):
            if ss.name in {"empty", "clock_glitch"}:
                continue  # covered separately
            df = ss.frame.rename(columns={ss.time_col: "time", ss.unit_col: "unit_id"})
            if df.empty:
                continue
            buf = CausalSpikeBuffer(ss.unit_ids, history_s=1e9)
            # Feed in arrival order, one chunk per spike: worst case for a
            # buffer that assumes monotonic arrival.
            for t, u in zip(df["time"].to_numpy(dtype=float), df["unit_id"].to_numpy()):
                buf.extend([t], [int(u)])
            # Evaluate at an interior time: the buffer's early-`break` only
            # bites when later-arriving samples sit before the cutoff in the
            # deque, which is the normal case mid-recording.
            lo, hi = float(df["time"].min()), float(df["time"].max())
            t_end = lo + 0.7 * (hi - lo)
            w = 0.5
            got = buf.counts_at(t_end, w)
            want = ss.truth_counts(t_end - w, t_end)
            if not np.array_equal(got, want):
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"CausalSpikeBuffer.counts_at miscounts on `{ss.name}` arrival",
                        severity=Severity.CRITICAL,
                        category="silent-wrong-answer",
                        where="realtime/live/spike_buffer.py:counts_at",
                        detail=(
                            "counts_at scans the deque linearly and `break`s on the "
                            "first sample at or after t. That is only valid if the "
                            "buffer is time-ordered. Live sorters deliver "
                            "slightly out-of-order events, so the scan stops early and "
                            "silently undercounts the causal window — this is the "
                            "vector that feeds the live decoder."
                        ),
                        evidence={
                            "arrival_pattern": ss.note,
                            "returned": got.tolist(),
                            "truth": want.tolist(),
                            "delta": (got - want).tolist(),
                        },
                        repro=(
                            "from realtime.live.spike_buffer import CausalSpikeBuffer\n"
                            "b = CausalSpikeBuffer([1], history_s=10.0)\n"
                            "b.extend([0.1],[1]); b.extend([0.9],[1]); b.extend([0.2],[1])\n"
                            "b.counts_at(0.5, 0.5)  # -> [1.], truth is [2.]"
                        ),
                        reachability=(
                            "Not through ReplaySpikeStream, which sorts on connect and "
                            "hands out contiguous cursor slices. Fully reachable "
                            "through any live adapter, and reachable on the replay path "
                            "too once a NaN or glitched sample corrupts the deque order."
                        ),
                        suggestion=(
                            "Keep the buffer sorted on insert (bisect.insort on a list, "
                            "or a heap), or drop the `break` and scan the whole retained "
                            "history. Do not assume monotonic arrival from a sorter."
                        ),
                    )
                )
        return out

    def _buffer_prune_on_glitch(self):
        from realtime.live.spike_buffer import CausalSpikeBuffer

        buf = CausalSpikeBuffer([1], history_s=2.0)
        buf.extend([1.0, 1.1, 1.2], [1, 1, 1])
        before = buf.n_spikes
        buf.extend([1.0e9], [1])          # one glitched sample
        after = buf.n_spikes
        if after < before:
            return [
                Finding(
                    probe=self.name,
                    title="A single bad timestamp flushes the entire causal buffer",
                    severity=Severity.HIGH,
                    category="silent-wrong-answer",
                    where="realtime/live/spike_buffer.py:_prune",
                    detail=(
                        "`_prune` takes the max timestamp of the *incoming chunk* as "
                        "'now' and discards everything older than now - history_s. One "
                        "uninitialised or glitched sample therefore evicts all real "
                        "history. The decoder then runs on a near-empty window and "
                        "keeps emitting predictions as if nothing happened."
                    ),
                    evidence={
                        "spikes_before_glitch": before,
                        "spikes_after_glitch": after,
                        "glitch_timestamp": 1.0e9,
                    },
                    suggestion=(
                        "Prune against the decoder's clock (the `t` passed to "
                        "counts_at), not against incoming data, and reject or clamp "
                        "samples more than a few history_s beyond it."
                    ),
                )
            ]
        return []

    def _unit_id_alignment(self):
        """Counts must stay aligned to unit_ids order, whatever the ids are."""
        from realtime.spike_binner import build_causal_spike_matrix

        out = []
        ss = poisson_spikes(n_units=4, duration_s=3.0, rate_hz=12.0, seed=self.ctx.seed)
        remap = {u: v for u, v in zip(ss.unit_ids, [108, 3, 47, 11])}  # sparse, unordered
        df = ss.frame.copy()
        df["unit_id"] = df["unit_id"].map(remap)
        ids = [108, 3, 47, 11]                       # deliberately NOT sorted
        got = build_causal_spike_matrix(df, ids, np.array([3.0]), 3.0)[0]
        want = np.array(
            [float((df["unit_id"] == u).sum()) for u in ids]
        )
        if not np.array_equal(got, want):
            out.append(
                Finding(
                    probe=self.name,
                    title="Count columns misalign for sparse / unsorted unit ids",
                    severity=Severity.CRITICAL,
                    category="silent-wrong-answer",
                    where="realtime/spike_binner.py:build_causal_spike_matrix",
                    detail=(
                        "Phy cluster ids are arbitrary and sparse (e.g. 3, 11, 47, 108) "
                        "and the caller's ordering is not guaranteed sorted. If column "
                        "order does not follow unit_ids exactly, every unit's activity "
                        "is attributed to the wrong cell."
                    ),
                    evidence={"unit_ids": ids, "returned": got.tolist(), "truth": want.tolist()},
                )
            )
        return out

    def _offline_online_agreement(self):
        """The offline matrix and the live buffer must agree bit for bit."""
        from realtime.spike_binner import build_causal_spike_matrix
        from realtime.live.spike_buffer import CausalSpikeBuffer

        ss = poisson_spikes(n_units=8, duration_s=6.0, rate_hz=10.0, seed=self.ctx.seed)
        df = ss.frame
        w, dt = 0.25, 0.05
        times = np.arange(1.0, 5.0, dt)
        offline = build_causal_spike_matrix(df, ss.unit_ids, times, w)

        buf = CausalSpikeBuffer(ss.unit_ids, history_s=max(2.0, 4 * w))
        online = np.zeros_like(offline)
        cursor = 0
        arr_t = df["time"].to_numpy(dtype=float)
        arr_u = df["unit_id"].to_numpy(dtype=int)
        for i, t in enumerate(times):
            nxt = int(np.searchsorted(arr_t, t, side="left"))
            if nxt > cursor:
                buf.extend(arr_t[cursor:nxt], arr_u[cursor:nxt])
                cursor = nxt
            online[i] = buf.counts_at(float(t), w)

        if not np.allclose(offline, online):
            bad = int(np.argmax(np.abs(offline - online).sum(axis=1)))
            return [
                Finding(
                    probe=self.name,
                    title="Offline training features and live features disagree",
                    severity=Severity.CRITICAL,
                    category="silent-wrong-answer",
                    where="spike_binner.build_causal_spike_matrix vs live/spike_buffer.counts_at",
                    detail=(
                        "A decoder trained on the offline matrix is evaluated on the "
                        "live buffer's vectors. If the two count differently for the "
                        "same spikes and the same window, every reported offline "
                        "accuracy overstates live accuracy, and the gap will look like "
                        "a science problem rather than a plumbing one."
                    ),
                    evidence={
                        "n_decode_steps": int(times.size),
                        "n_mismatched_steps": int((~np.isclose(offline, online)).any(axis=1).sum()),
                        "worst_step_index": bad,
                        "worst_step_time_s": float(times[bad]),
                        "offline_row": offline[bad].tolist(),
                        "online_row": online[bad].tolist(),
                    },
                    suggestion=(
                        "Make this an assertion in the test suite, not just a probe: "
                        "the two implementations should be the single source of truth "
                        "for [t-W, t) or one should call the other."
                    ),
                )
            ]
        return []

    def _nan_and_inf_propagation(self):
        """NaN/Inf must not silently become a finite-looking feature."""
        from realtime.spike_binner import build_causal_spike_matrix

        out = []
        ss = poisson_spikes(n_units=4, duration_s=3.0, seed=self.ctx.seed)
        df = ss.frame.copy()
        df.loc[df.index[:3], "time"] = np.nan
        try:
            got = build_causal_spike_matrix(df, ss.unit_ids, np.array([2.0]), 1.0)
            finite = bool(np.isfinite(got).all())
            if finite:
                out.append(
                    Finding(
                        probe=self.name,
                        title="NaN spike timestamps are absorbed without error",
                        severity=Severity.MEDIUM,
                        category="contract",
                        where="realtime/spike_binner.py:build_causal_spike_matrix",
                        detail=(
                            "Three NaN timestamps produced a fully finite count matrix "
                            "and no warning. Corrupt sorter output should be rejected at "
                            "the boundary, not quietly absorbed into training data."
                        ),
                        evidence={"returned": got.tolist()},
                        suggestion="Validate `np.isfinite(times).all()` at load time.",
                    )
                )
        except Exception as exc:  # raising is the acceptable behaviour
            out.append(
                Finding(
                    probe=self.name,
                    title="NaN timestamps raise (good) but with an unclear error",
                    severity=Severity.INFO,
                    category="contract",
                    where="realtime/spike_binner.py",
                    detail=f"Raised {type(exc).__name__}: {exc}",
                    suggestion="Raise PipelineInvariantError naming the offending column.",
                )
            )
        return out

    # -- failure modes that persist across decode steps -------------------

    def _buffer_nan_poisoning(self):
        """One NaN timestamp is permanent. Both buffer guards are NaN-false.

        `if ts < t0: continue` does not skip NaN and `if ts >= t: break` does
        not stop on it, so the sample falls through to the increment on every
        call. Because it also sits at the head of the deque, `_prune`'s
        `self._times[0] < cutoff` is permanently False and the buffer never
        prunes again.
        """
        from realtime.live.spike_buffer import CausalSpikeBuffer

        out = []
        buf = CausalSpikeBuffer([1], history_s=0.25)
        try:
            buf.extend([float("nan")], [1])
        except (ValueError, TypeError):
            # Rejecting it on insert is the correct behaviour — nothing further
            # to test, the sample never entered the buffer.
            return []
        buf.extend([1.0, 1.1, 1.2, 1.3, 1.4], [1, 1, 1, 1, 1])

        near = float(buf.counts_at(1.5, 0.25)[0])          # truth: 2 (1.3, 1.4)
        far = float(buf.counts_at(100.0, 0.25)[0])          # truth: 0
        retained_before = buf.n_spikes
        buf.extend(list(np.arange(2.0, 7.0, 0.01)), [1] * 500)
        retained_after = buf.n_spikes

        if near != 2.0 or far != 0.0:
            out.append(
                Finding(
                    probe=self.name,
                    title="A single NaN timestamp biases every future decode step",
                    severity=Severity.CRITICAL,
                    category="silent-wrong-answer",
                    where="realtime/live/spike_buffer.py:counts_at",
                    detail=(
                        "NaN compares False against both guards in the scan, so the "
                        "sample is neither skipped nor terminates the loop — it is "
                        "counted into the window at every t, forever. The decoder does "
                        "not fail; it reports a permanent +1 on that unit and keeps "
                        "running. For a closed-loop policy reading that unit, this is "
                        "a constant offset that looks like a real firing rate."
                    ),
                    evidence={
                        "counts_at(1.5, W=0.25)": near, "truth": 2.0,
                        "counts_at(100.0, W=0.25)": far, "truth_far": 0.0,
                    },
                    repro=(
                        "from realtime.live.spike_buffer import CausalSpikeBuffer\n"
                        "b = CausalSpikeBuffer([1], history_s=0.25)\n"
                        "b.extend([float('nan')], [1])\n"
                        "b.extend([1.0,1.1,1.2,1.3,1.4], [1]*5)\n"
                        "b.counts_at(100.0, 0.25)  # -> [1.], truth is [0.]"
                    ),
                    reachability=(
                        "Yes on any live adapter. Not reachable through "
                        "ReplaySpikeStream, which sorts on connect."
                    ),
                    suggestion=(
                        "Reject non-finite timestamps at extend() with a clear error. "
                        "A NaN in a spike train is corrupt acquisition, never data."
                    ),
                )
            )
        if retained_after > retained_before + 400:
            out.append(
                Finding(
                    probe=self.name,
                    title="A non-finite or far-future head sample disables pruning permanently",
                    severity=Severity.HIGH,
                    category="silent-wrong-answer",
                    where="realtime/live/spike_buffer.py:_prune",
                    detail=(
                        "_prune only ever inspects `self._times[0]`. Once a NaN or a "
                        "glitched timestamp occupies the head, that comparison is "
                        "permanently False, so the buffer grows without bound for the "
                        "rest of the session and history_s stops meaning anything. "
                        "Unbounded growth also makes counts_at slower every step — the "
                        "latency probe's linear scaling becomes a leak."
                    ),
                    evidence={
                        "history_s": 0.25,
                        "retained_after_500_more_spikes": retained_after,
                        "expected_order_of_magnitude": "tens",
                    },
                    suggestion=(
                        "Prune in a loop against the decoder clock and drop non-finite "
                        "samples on insert rather than trusting the head element."
                    ),
                )
            )
        return out

    def _transform_one_prev_counts_consistency(self):
        """The two realtime branches must maintain the same kind of state.

        transform_one(spikes_df=...) stores the assembled FEATURE vector as
        _prev_counts; transform_one(counts=...) stores actual COUNTS. The
        value is fed back as prev_counts on the next call, so the two paths
        disagree about what that history means.
        """
        import inspect
        from realtime.observation import ObservationTransformer

        src = inspect.getsource(ObservationTransformer.transform_one)
        stores_raw = "self._prev_counts = raw" in src
        stores_counts = "self._prev_counts = counts_2d" in src
        if not (stores_raw and stores_counts):
            return []

        detail = (
            "In the spikes_df branch `self._prev_counts = raw`, where `raw` is the "
            "assembled feature vector (1 x n_features). In the counts branch it is "
            "`counts_2d` (1 x n_units). That value is passed back as `prev_counts` on "
            "the following call and overrides the extractor's own correct history, so "
            "the two documented realtime entry points maintain incompatible state."
        )
        # Try to demonstrate it end to end with a derivative-bearing feature set.
        evidence: dict = {"stores_feature_vector_as_prev_counts": True}
        try:
            from realtime.pipeline_artifacts import ObservationConfig
            from ..synth import poisson_spikes

            ss = poisson_spikes(n_units=5, duration_s=4.0, seed=self.ctx.seed)
            obs = ObservationConfig(
                window_s=0.25, update_dt=0.05, source_spikes="sorted",
                feature_set="counts_dynamics", feature_type="counts",
            )
            tr = ObservationTransformer(obs, unit_ids=ss.unit_ids)
            X, _ = tr.extract_raw(ss.frame, np.arange(1.0, 3.0, 0.05))
            tr.fit(np.asarray(X, dtype=float))
            tr.transform_one(ss.frame, 2.0)
            tr.transform_one(ss.frame, 2.05)          # second call consumes history
            evidence["second_call"] = "succeeded"
        except Exception as exc:
            evidence["second_call_error"] = f"{type(exc).__name__}: {exc}"
            detail += (
                f"\n\nDemonstrated: a second transform_one() on a feature set with a "
                f"count derivative raises {type(exc).__name__} — the shape check "
                f"catches the mismatch, so it fails loudly rather than corrupting "
                f"features. That is the lucky case, not the designed one."
            )
        return [
            Finding(
                probe=self.name,
                title="transform_one keeps different state on its two realtime paths",
                severity=Severity.MEDIUM,
                category="contract",
                where="realtime/observation.py:ObservationTransformer.transform_one",
                detail=detail,
                evidence=evidence,
                reachability=(
                    "Latent. No production caller uses the spikes_df branch today — "
                    "only tests/test_pipeline_plumbing.py, with feature_set='counts', "
                    "which never consumes prev_counts. LiveDecoder has the mirror-image "
                    "correct version. It surfaces the first time anyone runs a "
                    "derivative-bearing feature set through the documented "
                    "ObservationTransformer realtime path."
                ),
                suggestion=(
                    "Store counts in both branches, or pass None and let the extractor "
                    "own its history — it already maintains a correct copy."
                ),
            )
        ]

    def _live_decoder_double_counting(self):
        """Counts derived twice per step, from two different code paths."""
        import inspect

        try:
            from realtime import live_decoder as mod
        except Exception:
            return []
        src = inspect.getsource(mod)
        uses_counts_at = "counts_at(" in src
        uses_as_dataframe = "as_dataframe()" in src
        if not (uses_counts_at and uses_as_dataframe):
            return []
        return [
            Finding(
                probe=self.name,
                title="LiveDecoder derives the same counts twice by two different routes",
                severity=Severity.LOW,
                category="contract",
                where="realtime/live_decoder.py",
                detail=(
                    "One step takes counts from CausalSpikeBuffer.counts_at (linear "
                    "scan with an early break) and also re-derives them inside "
                    "extract_at from buffer.as_dataframe() via count_spikes_in_window "
                    "(searchsorted). The two agree on clean data and diverge under "
                    "exactly the buffer-corruption conditions reported above — so the "
                    "logged spikes-in-window and the features fed to the decoder can "
                    "come from different numbers. The failure then presents "
                    "inconsistently instead of as an obvious zeroing."
                ),
                suggestion=(
                    "Compute the count vector once per step and pass it down. It is "
                    "also the cheaper of the two."
                ),
            )
        ]
