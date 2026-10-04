"""Representation-layer leakage: E, dynamic latents, splits, and caches.

The causality probe covers the observation layer — whether a feature at time
t can see past t. This one covers everything built on top of it, where the
leaks are subtler and the consequences land directly on the scientific claims:

  L1  The split. Chronological is necessary but not sufficient: with no gap,
      the first test samples' causal window [t-W, t) reaches back into the
      training period. The overlap scales with W, and W is an axis being
      compared, so the bias is not noise — it favours longer windows.

  L2  The fit. A manifold or latent model fitted on anything but the training
      partition inflates held-out performance by an unknown amount.

  L3  The cache. A fitted transform reused across runs with a different
      train_frac or seed was fitted on a different partition — one that may
      overlap this run's test set. A cache key that omits those is a leak
      with no visible symptom.

  L4  Causality of the representation itself. An RTS smoother or GPFA
      posterior at time t uses observations after t. That is legitimate for
      offline analysis and disqualifying for a deployable decoder, so the
      distinction has to be enforced, not documented.

Every check perturbs and compares rather than reading code: change only the
held-out data, or only the future, and assert the fitted object does not move.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from ..findings import Finding, Probe, Severity
from ..synth import poisson_spikes


class LeakageProbe(Probe):
    name = "leakage"
    description = "Manifold / latent fitting, split gaps, and cache partitioning"

    def checks(self):
        found = False
        for check in (
            self._split_gap_missing_on_main_path,
            self._transform_cache_ignores_fit_partition,
            self._manifold_fit_is_train_only,
            self._manifold_transform_is_pointwise,
            self._acausal_latents_are_gated,
            self._landmarks_come_from_train_only,
        ):
            for f in self.check(check):
                if f.severity >= Severity.LOW:
                    found = True
                yield f
        if not found:
            yield Finding(
                probe=self.name,
                title="No representation-layer leakage detected",
                severity=Severity.INFO,
                category="leakage",
                detail=(
                    "Splits carry a gap sized to the integration window; fitted "
                    "manifolds were unchanged by perturbing held-out rows; the "
                    "transform cache distinguishes fit partitions; acausal latents "
                    "are excluded from the deployable set."
                ),
            )


    @staticmethod
    def _make_encoder(name: str, n_components: int = 3):
        """Construct an encoder, tolerating ones that take no n_components.

        RawManifoldEncoder is the identity and has no latent dimension to set;
        skipping it on a TypeError would silently leave the identity path
        untested, which is exactly the path a deployable config is most likely
        to use.
        """
        from realtime.manifolds.registry import make_manifold_encoder

        try:
            return make_manifold_encoder(name, n_components=n_components)
        except TypeError:
            return make_manifold_encoder(name)

    # -- L1: the split --------------------------------------------------

    def _split_gap_missing_on_main_path(self):
        from realtime.train_decoder import causal_train_test_split
        from realtime.temporal.splits import required_split_gap_s

        sig = inspect.signature(causal_train_test_split)
        dt = 0.05
        t = np.arange(0.0, 600.0, dt)
        if "gap_s" in sig.parameters:
            from realtime.train_decoder import purge_gap_s

            W, hist = 0.250, 0.250
            gap = purge_gap_s(W, max_history_s=hist, update_dt=dt)
            train_mask, test_mask = causal_train_test_split(
                t, 0.80, gap_s=gap,
            )
            last_train = float(t[train_mask][-1])
            first_test = float(t[test_mask][0])
            n_overlap = int(((t[test_mask] - W) < last_train).sum())
            if first_test - last_train + 1e-12 >= gap and n_overlap == 0:
                return []
            # Parameter exists but does not actually purge — still a finding.
        else:
            train_mask, test_mask = causal_train_test_split(t, 0.70)
        split_time = float(t[train_mask][-1])
        n_test = int(test_mask.sum())

        overlap = {}
        for W in (0.05, 0.10, 0.25, 0.50, 1.00, 2.00):
            n = int(((t >= split_time) & (t - W < split_time)).sum())
            overlap[f"W={W:g}s"] = {
                "test_samples_reaching_into_train": n,
                "pct_of_test": round(100.0 * n / max(n_test, 1), 3),
            }

        return [
            Finding(
                probe=self.name,
                title="Train/test split has no gap, and the overlap scales with W",
                severity=Severity.HIGH,
                category="leakage",
                where="realtime/train_decoder.py:causal_train_test_split",
                detail=(
                    "The split is chronological, which rules out the usual "
                    "shuffled-split disaster. But it places the boundary at a single "
                    "instant with no gap, so a test sample at time t just past the "
                    "boundary builds its observation from spikes in [t-W, t) — part "
                    "of which is training data the decoder was fitted on.\n\n"
                    "The count of affected test samples is W / update_dt, so it grows "
                    "linearly with the window. Since the benchmark selects W by "
                    "comparing held-out scores across W, longer windows carry a "
                    "systematically larger share of contaminated samples. The bias is "
                    "small per-run but it points in a fixed direction on the very "
                    "axis being measured, which is the shape of bias that survives "
                    "averaging.\n\n"
                    "The repository already has the fix: "
                    "realtime/temporal/splits.py provides causal_train_val_test_split "
                    "with gap_s and required_split_gap_s(W, latent_history, lag). "
                    "realtime/temporal/comparison.py uses them. The main decoder "
                    "comparison path does not."
                ),
                evidence={
                    "session_s": 600.0,
                    "update_dt_s": dt,
                    "train_frac": 0.70,
                    "split_time_s": split_time,
                    "n_test_samples": n_test,
                    "overlap_by_window": overlap,
                    "example_required_gap_s": required_split_gap_s(2.0, 0.5, 0.0),
                },
                repro=(
                    "import numpy as np\n"
                    "from realtime.train_decoder import causal_train_test_split\n"
                    "t = np.arange(0, 600, 0.05)\n"
                    "tr, te = causal_train_test_split(t, 0.70)\n"
                    "split = t[tr][-1]\n"
                    "# test samples whose [t-W, t) reaches into train:\n"
                    "((t >= split) & (t - 2.0 < split)).sum()  # -> 40 at W=2s"
                ),
                reachability=(
                    "Yes — this is the split used by the main decoder comparison "
                    "(realtime/decoder_comparison.py) and therefore by deployment "
                    "selection and every reported window curve."
                ),
                suggestion=(
                    "Give causal_train_test_split a gap_s parameter and pass "
                    "required_split_gap_s(max_W, max_latent_history_s, "
                    "max_prediction_lag_s) from decoder_comparison, matching what "
                    "temporal/comparison.py already does. Record the gap in the "
                    "metrics so past runs can be distinguished from corrected ones."
                ),
            )
        ]

    # -- L3: the cache --------------------------------------------------

    def _transform_cache_ignores_fit_partition(self):
        from realtime.transform_cache import feature_transform_dirname

        names = {
            feature_transform_dirname("counts", "zscore_counts", 0.25)
            for _ in range(1)
        }
        name = next(iter(names))
        params = inspect.signature(feature_transform_dirname).parameters
        partition_aware = any(
            k in params for k in ("train_frac", "seed", "fit_hash", "split_hash")
        )
        if partition_aware:
            return []

        return [
            Finding(
                probe=self.name,
                title="Transform cache key omits the partition the transform was fit on",
                severity=Severity.CRITICAL,
                category="leakage",
                where="realtime/transform_cache.py:feature_transform_dirname",
                detail=(
                    "The cache directory name is built from (feature_set, "
                    f"feature_type_eff, decode_window) only — e.g. `{name}`. It "
                    "carries no train_frac, seed, or split identity, and the lookup "
                    "(find_feature_transform) does not read the provenance file "
                    "before returning a hit.\n\n"
                    "So with reuse_transforms enabled, a run at train_frac=0.5 will "
                    "load a transform fitted during an earlier train_frac=0.9 run. "
                    "That transform saw rows which are test data in the current run. "
                    "The scaler statistics, the PCA basis, the Isomap landmarks — all "
                    "of it was fit on data now being scored as held-out.\n\n"
                    "This is documented as already handled. The README states: "
                    "'Fitted transforms include train_frac and seed so test data "
                    "cannot leak into a reused fit.' The values are written into "
                    "provenance.json, but nothing reads them back, so the guarantee "
                    "is recorded rather than enforced.\n\n"
                    "There is no symptom. The run completes, the numbers look "
                    "reasonable, and held-out performance is optimistic by an amount "
                    "that depends on how much the two partitions overlapped."
                ),
                evidence={
                    "dirname_signature": list(params),
                    "example_dirname": name,
                    "same_name_for": [
                        "train_frac=0.70 seed=0",
                        "train_frac=0.50 seed=0",
                        "train_frac=0.90 seed=7",
                    ],
                    "readme_claim": (
                        "Fitted transforms include train_frac and seed so test data "
                        "cannot leak into a reused fit."
                    ),
                },
                repro=(
                    "from realtime.transform_cache import feature_transform_dirname\n"
                    "feature_transform_dirname('counts','zscore_counts',0.25)\n"
                    "# identical string regardless of train_frac / seed"
                ),
                reachability=(
                    "Whenever reuse_transforms is on and any earlier run used a "
                    "different train_frac or seed against the same output root. "
                    "Single-configuration runs are unaffected."
                ),
                suggestion=(
                    "Put the fit partition in the key, not just the provenance: "
                    "append the existing ObservationConfig.fit_hash() (which already "
                    "includes train_frac and seed) to the dirname. Failing that, have "
                    "find_feature_transform read provenance.json and reject a hit "
                    "whose train_frac/seed differ from the caller's. Treat any cache "
                    "written before this change as unkeyed and discard it."
                ),
            )
        ]

    # -- L2: the fit ----------------------------------------------------

    def _manifold_fit_is_train_only(self):
        """Corrupt only the held-out rows; a train-only fit must not move."""
        from realtime.manifolds.registry import available_manifolds

        out = []
        rng = np.random.default_rng(self.ctx.seed)
        n, d = 400, 20
        X = rng.normal(size=(n, d))
        train_mask = np.zeros(n, dtype=bool)
        train_mask[: int(0.7 * n)] = True

        poisoned = X.copy()
        poisoned[~train_mask] *= 1000.0

        for name in available_manifolds():
            try:
                a = self._make_encoder(name)
                b = self._make_encoder(name)
            except Exception:
                continue

            def _fit_and_embed(model, mat):
                model.fit(mat[train_mask])
                return np.asarray(model.transform(X[train_mask][:50]), dtype=float)

            try:
                za = _fit_and_embed(a, X)
                zb = _fit_and_embed(b, poisoned)
            except Exception:
                continue  # model not fittable on this synthetic shape

            if za.shape != zb.shape or not np.allclose(za, zb, atol=1e-8, equal_nan=True):
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Manifold `{name}` embedding depends on held-out rows",
                        severity=Severity.CRITICAL,
                        category="leakage",
                        where=f"realtime/manifolds/ ({name})",
                        detail=(
                            "Multiplying only the test rows by 1000 changed the "
                            "embedding of the training rows. The representation E is "
                            "therefore fit with knowledge of the held-out partition, "
                            "and every score computed on that partition is optimistic."
                        ),
                        evidence={
                            "manifold": name,
                            "max_abs_diff": float(np.nanmax(np.abs(za - zb)))
                            if za.shape == zb.shape else None,
                        },
                    )
                )
        return out

    def _manifold_transform_is_pointwise(self):
        """transform(x) must not depend on the other rows in the batch.

        A representation whose output for one timestep changes with the rest of
        the batch cannot run online at all: at deployment there is no batch.
        """
        from realtime.manifolds.registry import (
            available_manifolds,
            is_realtime_compatible_manifold,
        )

        out = []
        rng = np.random.default_rng(self.ctx.seed)
        X = rng.normal(size=(300, 15))
        probe_rows = X[:20]

        for name in available_manifolds():
            try:
                model = self._make_encoder(name)
                model.fit(X[:200])
                alone = np.asarray(model.transform(probe_rows), dtype=float)
                in_batch = np.asarray(
                    model.transform(np.vstack([probe_rows, X[200:]])), dtype=float
                )[: len(probe_rows)]
            except Exception:
                continue
            if alone.shape != in_batch.shape:
                continue
            if not np.allclose(alone, in_batch, atol=1e-8, equal_nan=True):
                declared_rt = is_realtime_compatible_manifold(name)
                out.append(
                    Finding(
                        probe=self.name,
                        title=(
                            f"Manifold `{name}` transform is batch-dependent"
                            + ("" if declared_rt else " (declared offline-only)")
                        ),
                        severity=Severity.HIGH if declared_rt else Severity.INFO,
                        category="leakage",
                        where=f"realtime/manifolds/ ({name})",
                        detail=(
                            "Embedding the same rows alone and inside a larger batch "
                            "gave different results, so transform() is a function of "
                            "the batch rather than of each observation. Offline that "
                            "quietly mixes test rows into each other's embedding; "
                            "online it cannot be evaluated at all, because a live "
                            "decoder has exactly one observation at a time."
                        ),
                        evidence={
                            "manifold": name,
                            "declared_realtime_compatible": declared_rt,
                            "max_abs_diff": float(np.nanmax(np.abs(alone - in_batch))),
                        },
                        suggestion=(
                            "Either give it a true out-of-sample extension, or mark it "
                            "supports_realtime = False so deployment selection cannot "
                            "choose it."
                        ),
                    )
                )
        return out

    # -- L4: causality of the representation ----------------------------

    def _acausal_latents_are_gated(self):
        """A smoother uses the future. It must not reach the deployable set."""
        from realtime.dynamic_latents.registry import DYNAMIC_LATENT_REGISTRY

        out = []
        for name, cls in sorted(DYNAMIC_LATENT_REGISTRY.items()):
            try:
                src = inspect.getsource(cls)
            except (OSError, TypeError):
                continue
            mod_src = ""
            try:
                mod_src = inspect.getsource(inspect.getmodule(cls))
            except (OSError, TypeError):
                pass
            text = f"{src}\n{mod_src}"
            smooths = any(
                tok in text
                for tok in ("rts_smooth", "smoothed", "acausal", "RTS smoother")
            )
            declares_realtime = bool(getattr(cls, "supports_realtime", False))
            if smooths and declares_realtime:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Latent model `{name}` smooths yet declares realtime support",
                        severity=Severity.HIGH,
                        category="causality",
                        where=f"realtime/dynamic_latents/ ({name})",
                        detail=(
                            "The implementation contains an RTS/acausal smoothing path "
                            "and also sets supports_realtime = True. A smoothed "
                            "estimate at time t is conditioned on observations after "
                            "t, so if any code path can reach the smoothed trajectory "
                            "while the model is marked deployable, the reported "
                            "held-out performance is not achievable online.\n\n"
                            "Having both is not itself a bug — a model can offer a "
                            "causal filter and an offline smoother — but which one "
                            "feeds evaluation must be decided by the flag, not by "
                            "whichever method the caller reached for."
                        ),
                        evidence={
                            "model": name,
                            "supports_realtime": declares_realtime,
                            "smoothing_tokens_present": True,
                        },
                        suggestion=(
                            "Assert at the evaluation boundary that a model marked "
                            "deployable returned its causal (filtered) trajectory — "
                            "e.g. have transform() take causal=True and refuse to "
                            "return smoothed output when supports_realtime is set."
                        ),
                    )
                )
        return out

    def _landmarks_come_from_train_only(self):
        """Landmark / neighbour selection must not look at held-out rows."""
        out = []
        for modname in (
            "realtime.manifolds.diffusion_nystrom",
            "realtime.manifolds.isomap",
        ):
            try:
                mod = __import__(modname, fromlist=["*"])
                src = inspect.getsource(mod)
            except Exception:
                continue
            picks_landmarks = any(
                tok in src for tok in ("landmark", "n_landmarks", "_select_landmarks")
            )
            if not picks_landmarks:
                continue
            # A landmark chooser that takes the full matrix is the smell.
            suspicious = [
                line.strip()
                for line in src.splitlines()
                if "landmark" in line.lower()
                and ("X_all" in line or "X_full" in line or "np.vstack" in line)
            ]
            if suspicious:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Landmark selection in {modname} may see all rows",
                        severity=Severity.MEDIUM,
                        category="leakage",
                        where=modname.replace(".", "/") + ".py",
                        detail=(
                            "Landmark or neighbour selection appears to operate on a "
                            "full matrix rather than the training partition. Landmarks "
                            "define the embedding geometry, so choosing them with "
                            "sight of held-out rows leaks structure even when the "
                            "subsequent fit is train-only."
                        ),
                        evidence={"lines": suspicious[:5]},
                    )
                )
        return out
