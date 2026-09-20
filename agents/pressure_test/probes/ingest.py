"""Can real sorted output actually get in?

The lab path is Kilosort/Phy: spike times and cluster ids, with vendor column
names and arbitrary sparse cluster ids. This probe feeds those shapes to the
entry points and reports where they bounce off.

It never reads real recordings — only synthetic frames wearing the vendor's
column names — so it is safe to run anywhere.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from ..findings import Finding, Probe, Severity
from ..synth import PHY_SCHEMAS, phy_like


class IngestProbe(Probe):
    name = "ingest"
    description = "Kilosort/Phy sorted-spike ingest readiness"

    def checks(self):
        yield from self.check(self._replay_stream_schemas)
        yield from self.check(self._binner_schemas)
        yield from self.check(self._buffer_schemas)
        yield from self.check(self._stub_streams_declared)
        yield from self.check(self._sample_index_vs_seconds)
        yield from self.check(self._loader_silently_drops_all_units)

    # -------------------------------------------------------------------

    @staticmethod
    def _diagnose_stream_failure(schema: str, columns: list[str], exc: Exception) -> str:
        """Name the actual failing stage — do not assume one cause fits all.

        connect() can fail in two distinct places, and the fix differs:
          (a) unit ids are read via `.get("unit_id", .get("unit"))` BEFORE the
              vendor columns are renamed, so a frame whose unit column is
              `cluster_id` yields None and dies in int();
          (b) the rename alias tuples themselves are missing the vendor name,
              so the column is never normalised and sort_values("time") raises
              KeyError.
        """
        has_unit_alias = any(c in columns for c in ("unit_id", "unit"))
        has_time_alias = any(c in columns for c in ("time", "time_s", "spike_time", "t"))
        if isinstance(exc, TypeError) and not has_unit_alias:
            return (
                "Root cause is ordering: connect() reads unit ids via "
                "`self._spikes.get('unit_id', self._spikes.get('unit'))` BEFORE it "
                "renames vendor columns, so this frame yields None and fails in "
                "int(). Moving the rename above the unit-id extraction fixes it."
            )
        if isinstance(exc, KeyError) and not has_time_alias:
            return (
                "Root cause is a missing alias, not ordering: the rename tuple in "
                "connect() covers only ('time_s', 'spike_time', 't'), so this "
                "frame's time column is never normalised and sort_values('time') "
                "raises. Note realtime/spike_binner.py DOES accept this name — the "
                "two alias lists have drifted apart."
            )
        if not has_unit_alias and not has_time_alias:
            return (
                "Neither the time nor the unit column name appears in connect()'s "
                "alias tuples, so reordering alone would not fix this: the vendor "
                "names have to be added to the alias table as well."
            )
        return f"Failed with {type(exc).__name__}; stage not automatically classified."

    def _replay_stream_schemas(self):
        from realtime.live.spike_stream import ReplaySpikeStream

        out = []
        for schema, _, note in PHY_SCHEMAS:
            ss = phy_like(schema, seed=self.ctx.seed)
            stream = ReplaySpikeStream(ss.frame.copy())
            try:
                stream.connect()
                ids = stream.list_unit_ids()
                if sorted(int(i) for i in ids) != sorted(ss.unit_ids):
                    out.append(
                        Finding(
                            probe=self.name,
                            title=f"ReplaySpikeStream loses unit ids for `{schema}`",
                            severity=Severity.HIGH,
                            category="ingest",
                            where="realtime/live/spike_stream.py:ReplaySpikeStream.connect",
                            detail=(
                                f"{note}. connect() succeeded but reported units "
                                f"{ids} instead of {ss.unit_ids}."
                            ),
                            evidence={"schema": schema, "reported": list(ids),
                                      "expected": ss.unit_ids},
                        )
                    )
            except Exception as exc:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"ReplaySpikeStream cannot load `{schema}` sorted output",
                        severity=Severity.CRITICAL if "cluster" in schema else Severity.HIGH,
                        category="ingest",
                        where="realtime/live/spike_stream.py:ReplaySpikeStream.connect",
                        detail=(
                            f"{note}. connect() raised {type(exc).__name__}: {exc}\n\n"
                            + self._diagnose_stream_failure(
                                schema, list(ss.frame.columns), exc
                            )
                        ),
                        reachability=(
                            "Yes, the moment a Phy/Kilosort export is loaded through "
                            "ReplaySpikeStream with a spikes frame rather than an "
                            "experiment_dir. The in-repo caller "
                            "(ui/services/live_deployment.py) passes experiment_dir "
                            "and reads simulator CSVs, so this fires on the first real "
                            "lab dataset, not on the simulated path."
                        ),
                        evidence={"schema": schema, "columns": list(ss.frame.columns),
                                  "error": f"{type(exc).__name__}: {exc}"},
                        repro=(
                            "import pandas as pd\n"
                            "from realtime.live.spike_stream import ReplaySpikeStream\n"
                            "phy = pd.DataFrame({'spike_time':[0.1,0.2,0.3],"
                            "'cluster_id':[7,7,8]})\n"
                            "ReplaySpikeStream(phy).connect()  # TypeError"
                        ),
                        suggestion=(
                            "Normalise column names first, then derive unit ids. Better: "
                            "put the alias table in one module "
                            "(spike_binner._resolve_spike_columns already has one) and "
                            "have every entry point call it."
                        ),
                    )
                )
        return out

    def _binner_schemas(self):
        """spike_binner has its own alias list — do the two agree?"""
        from realtime.spike_binner import _resolve_spike_columns, build_causal_spike_matrix

        out = []
        for schema, _, note in PHY_SCHEMAS:
            ss = phy_like(schema, seed=self.ctx.seed)
            try:
                _resolve_spike_columns(ss.frame)
                X = build_causal_spike_matrix(
                    ss.frame.rename(columns={ss.time_col: "time", ss.unit_col: "unit_id"}),
                    ss.unit_ids, np.array([3.9]), 4.0,
                )
                if X.sum() == 0:
                    out.append(
                        Finding(
                            probe=self.name,
                            title=f"Binner returns all-zero counts for `{schema}`",
                            severity=Severity.HIGH,
                            category="ingest",
                            where="realtime/spike_binner.py",
                            detail=f"{note}. Accepted without error but counted nothing.",
                            evidence={"schema": schema},
                        )
                    )
            except Exception as exc:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"Binner rejects `{schema}` column naming",
                        severity=Severity.MEDIUM,
                        category="ingest",
                        where="realtime/spike_binner.py:_resolve_spike_columns",
                        detail=(
                            f"{note}. Raised {type(exc).__name__}: {exc}. Note this "
                            "module maintains its own alias list, separate from the ones "
                            "in spike_stream.py and spike_buffer.py — three lists that "
                            "can and do drift apart."
                        ),
                        evidence={"schema": schema, "columns": list(ss.frame.columns)},
                        suggestion="Single shared alias resolver used by all three.",
                    )
                )
        return out

    def _buffer_schemas(self):
        from realtime.live.spike_buffer import CausalSpikeBuffer

        out = []
        for schema, _, note in PHY_SCHEMAS:
            ss = phy_like(schema, seed=self.ctx.seed)
            buf = CausalSpikeBuffer(ss.unit_ids, history_s=10.0)
            try:
                buf.extend_dataframe(ss.frame)
                if buf.n_spikes != len(ss.frame):
                    out.append(
                        Finding(
                            probe=self.name,
                            title=f"CausalSpikeBuffer drops spikes for `{schema}`",
                            severity=Severity.HIGH,
                            category="ingest",
                            where="realtime/live/spike_buffer.py:extend_dataframe",
                            detail=f"{note}. Ingested {buf.n_spikes} of {len(ss.frame)}.",
                            evidence={"schema": schema},
                        )
                    )
            except Exception as exc:
                out.append(
                    Finding(
                        probe=self.name,
                        title=f"CausalSpikeBuffer rejects `{schema}`",
                        severity=Severity.MEDIUM,
                        category="ingest",
                        where="realtime/live/spike_buffer.py:extend_dataframe",
                        detail=f"{note}. Raised {type(exc).__name__}: {exc}",
                        evidence={"schema": schema, "columns": list(ss.frame.columns)},
                    )
                )
        return out

    def _stub_streams_declared(self):
        """A stub is fine. A stub that isn't obviously a stub is not."""
        import realtime.live.spike_stream as mod

        out = []
        for name, obj in vars(mod).items():
            if not inspect.isclass(obj) or not issubclass(obj, mod.SpikeStream):
                continue
            if obj is mod.SpikeStream:
                continue
            src = inspect.getsource(obj)
            if "NotImplementedError" not in src:
                continue
            out.append(
                Finding(
                    probe=self.name,
                    title=f"`{name}` is an unimplemented acquisition path",
                    severity=Severity.MEDIUM,
                    category="ingest",
                    where=f"realtime/live/spike_stream.py:{name}",
                    detail=(
                        "connect() raises NotImplementedError. Sorted-spike ingest is "
                        "covered by PhySpikeStream (including follow=True, which "
                        "re-reads a sort as it grows), so this is no longer the only "
                        "path in — but streaming acquisition straight from Open Ephys "
                        "still does not exist, and the live-vs-replay gap (jitter, "
                        "dropouts, cluster drift, arrival reordering) remains "
                        "unexercised against hardware."
                    ),
                    evidence={"class": name},
                    suggestion=(
                        "Next: a ReplaySpikeStream/PhySpikeStream wrapper that injects "
                        "jitter, dropouts and out-of-order arrival, so the loop is "
                        "tested against non-ideal timing before hardware day."
                    ),
                )
            )
        return out

    def _sample_index_vs_seconds(self):
        """Phy ships sample indices. Seconds vs samples is a silent 30000x error.

        The gate belongs at ingest, where a sampling rate exists — not in the
        binner, which has no way to know what units it was handed. So this
        checks that the Phy loader refuses an implausible conversion rather
        than returning a well-formed empty session.
        """
        import tempfile
        from pathlib import Path

        out = []
        try:
            from realtime.live.phy_stream import PhyIngestError, load_phy_spikes
        except ImportError:
            return [
                Finding(
                    probe=self.name,
                    title="No Phy ingest path exists",
                    severity=Severity.HIGH,
                    category="ingest",
                    where="realtime/live/",
                    detail=(
                        "Kilosort writes spike_times.npy as sample indices. With no "
                        "loader that records a sampling rate, those indices reach the "
                        "binner as seconds and produce an empty but perfectly "
                        "well-formed count matrix — a decoder trained on silence."
                    ),
                )
            ]

        d = Path(tempfile.mkdtemp(prefix="hippo_fs_probe_"))
        fs = 30000.0
        rng = np.random.default_rng(self.ctx.seed)
        t = np.sort(rng.uniform(0.0, 120.0, 2000))
        np.save(d / "spike_times.npy", (t * fs).astype(np.int64))
        np.save(d / "spike_clusters.npy", rng.choice([3, 11], 2000).astype(np.int32))
        (d / "params.py").write_text(f"sample_rate = {fs}\n")

        # Correct rate must work.
        try:
            _, meta = load_phy_spikes(d, accepted_groups=None)
            ok = 100.0 < meta["duration_s"] < 140.0
        except Exception as exc:
            out.append(
                Finding(
                    probe=self.name,
                    title="Phy loader rejects a well-formed sort",
                    severity=Severity.HIGH,
                    category="ingest",
                    where="realtime/live/phy_stream.py:load_phy_spikes",
                    detail=f"A valid 120 s sort at 30 kHz raised {type(exc).__name__}: {exc}",
                )
            )
            return out
        if not ok:
            out.append(
                Finding(
                    probe=self.name,
                    title="Phy loader mis-converts sample indices to seconds",
                    severity=Severity.CRITICAL,
                    category="ingest",
                    where="realtime/live/phy_stream.py:load_phy_spikes",
                    detail=(
                        f"A 120 s sort at 30 kHz came back as "
                        f"{meta['duration_s']:.6g} s."
                    ),
                    evidence={"meta": meta},
                )
            )

        # A wrong rate must raise, not yield an empty-but-valid session.
        for bad_fs, label in ((1.0, "indices treated as seconds"),
                              (3.0e7, "rate too high by 1000x")):
            try:
                _, bad_meta = load_phy_spikes(d, fs=bad_fs, accepted_groups=None)
            except PhyIngestError:
                continue
            except Exception:
                continue
            out.append(
                Finding(
                    probe=self.name,
                    title=f"Implausible sampling rate accepted ({label})",
                    severity=Severity.CRITICAL,
                    category="ingest",
                    where="realtime/live/phy_stream.py:load_phy_spikes",
                    detail=(
                        f"fs={bad_fs} produced a session of "
                        f"{bad_meta['duration_s']:.6g} s without raising. A wrong "
                        "sampling rate scales every timestamp by a constant factor; "
                        "the windows still fill or still empty, and nothing "
                        "downstream can detect it."
                    ),
                    evidence={"fs": bad_fs, "meta": bad_meta},
                )
            )
        return out

    def _loader_silently_drops_all_units(self):
        """A units table without simulator annotations must not silently zero out.

        load_simulation_data annotates units and then filters to
        analysis-eligible ones. A units.csv lacking the region / cell-type
        columns marks every unit `unknown` -> `include_in_decoder=False`, so
        the filter removes all of them and the loader returns an empty spike
        frame with no warning. Sorted output from Kilosort/Phy has no such
        annotations, so this is the shape real lab data arrives in.
        """
        import json
        import tempfile
        from pathlib import Path

        out = []
        d = Path(tempfile.mkdtemp(prefix="hippo_ingest_probe_"))
        pd.DataFrame(
            {"time": [0.0, 0.05, 0.10], "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0],
             "speed": [0.0, 1.0, 1.0], "head_direction": [0.0, 0.1, 0.2]}
        ).to_csv(d / "behavior.csv", index=False)
        # Exactly what a Phy export gives you: ids, nothing else.
        pd.DataFrame({"unit_id": [1, 2]}).to_csv(d / "units.csv", index=False)
        pd.DataFrame(
            {"unit_id": [1, 2, 1, 2], "spike_time_s": [0.08, 0.02, 0.03, 0.09]}
        ).to_csv(d / "spikes_ground_truth.csv", index=False)
        (d / "summary.json").write_text(json.dumps({"session_duration_s": 0.1}))

        from realtime.data_loading import load_simulation_data

        try:
            data = load_simulation_data(d, spike_source="ground_truth")
        except ValueError:
            # Refusing to return an empty analysis set is the correct behaviour;
            # the failure mode this check exists for is returning one silently.
            return []
        n_units = len(data["unit_ids"])
        n_spikes = len(data["spikes_df"])
        if n_units == 0 or n_spikes == 0:
            out.append(
                Finding(
                    probe=self.name,
                    title="Loader silently returns zero units for an unannotated units.csv",
                    severity=Severity.CRITICAL,
                    category="ingest",
                    where="realtime/data_loading.py:load_simulation_data",
                    detail=(
                        "A units.csv carrying only unit_id — which is all a Phy or "
                        "Kilosort export gives you — is annotated region_canonical="
                        "'unknown', include_in_decoder=False for every unit. "
                        "filter_unit_ids_for_analysis then drops all of them, and the "
                        "loader returns an empty spike frame and an empty unit list "
                        "without raising or warning.\n\n"
                        "Downstream this looks like a science result, not a load "
                        "failure: features are all-zero, decoders train on nothing, "
                        "and the reported accuracy is whatever chance is for that "
                        "target. n_units_excluded records the drop, but nothing reads "
                        "it and nothing surfaces it."
                    ),
                    evidence={
                        "unit_ids_returned": list(data["unit_ids"]),
                        "spikes_returned": n_spikes,
                        "n_units_excluded": data.get("n_units_excluded"),
                        "units_csv_columns": ["unit_id"],
                    },
                    repro=(
                        "# units.csv with only a unit_id column\n"
                        "data = load_simulation_data(d, spike_source='ground_truth')\n"
                        "len(data['unit_ids'])  # -> 0, no warning"
                    ),
                    reachability=(
                        "Yes, on the first real sorted dataset. The simulator writes "
                        "the region columns, so the synthetic path never sees this; "
                        "a Phy export has none of them."
                    ),
                    suggestion=(
                        "Raise when the analysis filter removes every unit while the "
                        "raw table was non-empty — that state is never a valid result. "
                        "Separately decide the semantics for unannotated units: either "
                        "include them by default, or require an explicit region mapping "
                        "at ingest. Silently dropping is the one option that cannot be "
                        "right."
                    ),
                )
            )
        return out
