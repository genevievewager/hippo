"""Causality and train/test leakage.

Two properties a closed-loop hippocampal BCI must hold or it is not a BCI:

  C1  A feature at decode time t depends only on spikes in [t - W, t).
      Not (t - W, t]. Not [t - W, t]. A single sample of future leakage
      inflates every decoding result and cannot exist on hardware.

  C2  Nothing fitted (F transforms, E representations, scalers) may see the
      held-out partition. The repo keys caches on `train_frac` and `seed`
      precisely to prevent this; the probe verifies the guarantee empirically
      rather than trusting the cache key.

Both are checked by *perturbation*: change only the future, or change only
the test rows, and assert the output does not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..findings import Finding, Probe, Severity
from ..synth import poisson_spikes


class CausalityProbe(Probe):
    name = "causality"
    description = "Future-information leakage and train/test contamination"

    def checks(self):
        found = False
        for f in self.check(self._future_spike_perturbation):
            found = True
            yield f
        for f in self.check(self._boundary_convention):
            found = True
            yield f
        for f in self.check(self._observation_transformer_future_independence):
            found = True
            yield f
        for f in self.check(self._fit_sees_only_train):
            found = True
            yield f
        for f in self.check(self._transform_is_stateless_across_calls):
            found = True
            yield f
        if not found:
            # A clean causality run is a result, not an absence of one. Record
            # it so a regression is visible as a change rather than as noise.
            yield Finding(
                probe=self.name,
                title="No causal leakage detected",
                severity=Severity.INFO,
                category="causality",
                detail=(
                    "Perturbing only the future left every feature row unchanged; the "
                    "half-open [t-W, t) convention held at the boundary across all "
                    "three counting implementations; poisoning only the held-out rows "
                    "left the fitted F transform unchanged; and transform() was "
                    "history-independent. This is the part of the pipeline that is "
                    "hardest to get right and it is currently correct."
                ),
            )

    # -- C1: the future must not matter --------------------------------

    def _future_spike_perturbation(self):
        """Inject a burst strictly after t; features at t must not move."""
        from realtime.spike_binner import build_causal_spike_matrix

        ss = poisson_spikes(n_units=6, duration_s=10.0, rate_hz=8.0, seed=self.ctx.seed)
        w, times = 0.3, np.arange(1.0, 5.0, 0.05)
        base = build_causal_spike_matrix(ss.frame, ss.unit_ids, times, w)

        future = pd.DataFrame(
            {
                "time": np.linspace(5.2, 9.0, 800),
                "unit_id": np.tile(ss.unit_ids, 800 // len(ss.unit_ids) + 1)[:800],
            }
        )
        perturbed = pd.concat([ss.frame, future], ignore_index=True).sort_values(
            "time", kind="mergesort"
        )
        after = build_causal_spike_matrix(perturbed, ss.unit_ids, times, w)

        if not np.array_equal(base, after):
            n = int((~np.isclose(base, after)).any(axis=1).sum())
            return [
                Finding(
                    probe=self.name,
                    title="Features at time t change when only the future changes",
                    severity=Severity.CRITICAL,
                    category="causality",
                    where="realtime/spike_binner.py:build_causal_spike_matrix",
                    detail=(
                        f"800 spikes were added strictly after the last decode time "
                        f"(t >= 5.2 s, decode times end at {times[-1]:.2f} s). "
                        f"{n} of {times.size} feature rows changed. Any decoding result "
                        "built on this is not realisable on hardware."
                    ),
                    evidence={"n_rows_changed": n, "n_rows": int(times.size)},
                )
            ]
        return []

    def _boundary_convention(self):
        """Half-open [t-W, t): a spike exactly at t is future, at t-W is past."""
        from realtime.spike_binner import (
            build_causal_spike_matrix,
            count_spikes_in_window,
        )
        from realtime.live.spike_buffer import CausalSpikeBuffer

        out = []
        t, w = 2.0, 0.5
        cases = {
            "spike exactly at t (must be EXCLUDED)": (t, 0.0),
            "spike exactly at t-W (must be INCLUDED)": (t - w, 1.0),
            "spike one ulp before t (must be INCLUDED)": (np.nextafter(t, 0.0), 1.0),
        }
        for label, (spike_t, expected) in cases.items():
            df = pd.DataFrame({"time": [spike_t], "unit_id": [1]})
            impls = {
                "build_causal_spike_matrix": float(
                    build_causal_spike_matrix(df, [1], np.array([t]), w)[0, 0]
                ),
                "count_spikes_in_window": float(
                    count_spikes_in_window(df, [1], t - w, t)[0]
                ),
            }
            buf = CausalSpikeBuffer([1], history_s=10.0)
            buf.extend([spike_t], [1])
            impls["CausalSpikeBuffer.counts_at"] = float(buf.counts_at(t, w)[0])

            wrong = {k: v for k, v in impls.items() if v != expected}
            if wrong:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Window boundary convention violated: {label}",
                        severity=Severity.HIGH,
                        category="causality",
                        where="realtime spike counting implementations",
                        detail=(
                            "The three counting paths must agree on the half-open "
                            "convention [t-W, t). Including a spike at exactly t means "
                            "the decoder sees a sample it would not have on hardware; "
                            "disagreement between paths means offline and live features "
                            "differ at every bin boundary."
                        ),
                        evidence={"expected": expected, "returned": impls},
                    )
                )
        return out

    def _observation_transformer_future_independence(self):
        """Same perturbation, but through the real ObservationTransformer."""
        from realtime.pipeline_artifacts import ObservationConfig
        from realtime.observation import ObservationTransformer

        ss = poisson_spikes(n_units=6, duration_s=10.0, rate_hz=8.0, seed=self.ctx.seed)
        times = np.arange(1.0, 5.0, 0.1)
        try:
            obs = ObservationConfig(
                window_s=0.3, update_dt=0.1, source_spikes="sorted",
                feature_set="counts", feature_type="counts",
            )
        except TypeError:
            return [
                Finding(
                    probe=self.name,
                    title="ObservationConfig signature not as expected by the probe",
                    severity=Severity.INFO,
                    category="contract",
                    detail="Skipping end-to-end causality check; update the harness.",
                )
            ]

        def raw(frame: pd.DataFrame) -> np.ndarray:
            tr = ObservationTransformer(obs, unit_ids=ss.unit_ids)
            X, _ = tr.extract_raw(frame, times)
            return np.asarray(X, dtype=float)

        base = raw(ss.frame)
        fut = pd.DataFrame(
            {"time": np.linspace(6.0, 9.5, 500),
             "unit_id": np.tile(ss.unit_ids, 500 // len(ss.unit_ids) + 1)[:500]}
        )
        after = raw(pd.concat([ss.frame, fut], ignore_index=True).sort_values("time"))

        if not np.allclose(base, after, equal_nan=True):
            return [
                Finding(
                    probe=self.name,
                    title="ObservationTransformer.extract_raw is not future-independent",
                    severity=Severity.CRITICAL,
                    category="causality",
                    where="realtime/observation.py:extract_raw",
                    detail=(
                        "Adding spikes strictly after the last decode time changed the "
                        "observation matrix. This is the object that owns W for both the "
                        "training and the realtime path."
                    ),
                    evidence={
                        "n_rows_changed": int((~np.isclose(base, after)).any(axis=1).sum()),
                    },
                )
            ]
        return []

    # -- C2: the test partition must not touch a fit --------------------

    def _fit_sees_only_train(self):
        """Corrupt only the test rows; a train-only fit must be unchanged."""
        from realtime.pipeline_artifacts import ObservationConfig
        from realtime.observation import ObservationTransformer

        ss = poisson_spikes(n_units=8, duration_s=12.0, rate_hz=9.0, seed=self.ctx.seed)
        times = np.arange(1.0, 10.0, 0.1)
        obs = ObservationConfig(
            window_s=0.25, update_dt=0.1, source_spikes="sorted",
            feature_set="counts", feature_type="zscore_counts",
        )
        tr = ObservationTransformer(obs, unit_ids=ss.unit_ids)
        X, _ = tr.extract_raw(ss.frame, times)
        X = np.asarray(X, dtype=float)

        n = X.shape[0]
        train_mask = np.zeros(n, dtype=bool)
        train_mask[: int(0.7 * n)] = True
        test_mask = ~train_mask

        def fitted_params(mat: np.ndarray):
            t = ObservationTransformer(obs, unit_ids=ss.unit_ids)
            t.fit(mat, train_mask=train_mask)
            f = t.f_transform
            return {
                k: np.asarray(getattr(f, k)).round(10).tolist()
                for k in ("mean_", "scale_", "var_", "n_features_in_")
                if getattr(f, k, None) is not None
            }

        clean = fitted_params(X)
        poisoned = X.copy()
        poisoned[test_mask] *= 1000.0          # only the held-out rows
        dirty = fitted_params(poisoned)

        if clean and clean != dirty:
            return [
                Finding(
                    probe=self.name,
                    title="Fitted F transform is influenced by held-out data",
                    severity=Severity.CRITICAL,
                    category="leakage",
                    where="realtime/observation.py:ObservationTransformer.fit",
                    detail=(
                        "Multiplying only the test rows by 1000 changed the fitted "
                        "scaler parameters. Held-out performance is then optimistic by "
                        "an unknown amount, and deployment selection is choosing on a "
                        "contaminated score."
                    ),
                    evidence={"clean_params": clean, "params_after_test_poison": dirty},
                )
            ]
        if not clean:
            return [
                Finding(
                    probe=self.name,
                    title="Leakage check could not read fitted parameters",
                    severity=Severity.LOW,
                    category="leakage",
                    detail=(
                        "The F transform exposes no mean_/scale_/var_ attributes, so "
                        "train-only fitting could not be verified by inspection. "
                        "Consider exposing fitted state via get_metadata() so this is "
                        "auditable."
                    ),
                )
            ]
        return []

    def _transform_is_stateless_across_calls(self):
        """transform() must be a pure function of its input."""
        from realtime.pipeline_artifacts import ObservationConfig
        from realtime.observation import ObservationTransformer

        ss = poisson_spikes(n_units=6, duration_s=8.0, seed=self.ctx.seed)
        times = np.arange(1.0, 6.0, 0.1)
        obs = ObservationConfig(
            window_s=0.25, update_dt=0.1, source_spikes="sorted",
            feature_set="counts", feature_type="zscore_counts",
        )
        tr = ObservationTransformer(obs, unit_ids=ss.unit_ids)
        X, _ = tr.extract_raw(ss.frame, times)
        tr.fit(np.asarray(X, dtype=float))
        a = tr.transform(np.asarray(X, dtype=float))
        _ = tr.transform(np.asarray(X, dtype=float) * 7.0)     # interleaved call
        b = tr.transform(np.asarray(X, dtype=float))
        if not np.allclose(a, b, equal_nan=True):
            return [
                Finding(
                    probe=self.name,
                    title="transform() output depends on call history",
                    severity=Severity.HIGH,
                    category="contract",
                    where="realtime/observation.py:ObservationTransformer.transform",
                    detail=(
                        "The same input produced different output after an unrelated "
                        "intervening call, so the transform carries hidden state. In "
                        "the UI, where pages are re-run on every widget interaction, "
                        "this makes results depend on click order."
                    ),
                    evidence={"max_abs_diff": float(np.nanmax(np.abs(a - b)))},
                )
            ]
        return []
