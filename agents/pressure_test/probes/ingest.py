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
                    severity=Severity.HIGH,
                    category="ingest",
                    where=f"realtime/live/spike_stream.py:{name}",
                    detail=(
                        "connect() raises NotImplementedError, so there is currently no "
                        "live acquisition: everything labelled 'realtime' is replay of "
                        "stored spikes. That is a sound way to build, but it means the "
                        "live-vs-replay gap (jitter, dropouts, cluster drift, arrival "
                        "reordering) has never been exercised."
                    ),
                    evidence={"class": name},
                    suggestion=(
                        "Since the lab path is Kilosort/Phy, the highest-value next "
                        "adapter is a Phy-directory stream that tails "
                        "spike_times.npy / spike_clusters.npy, plus a ReplaySpikeStream "
                        "wrapper that injects jitter and drops so the loop is tested "
                        "against non-ideal arrival before hardware day."
                    ),
                )
            )
        return out

    def _sample_index_vs_seconds(self):
        """Phy ships sample indices. Seconds vs samples is a silent 30000x error."""
        from realtime.spike_binner import build_causal_spike_matrix

        fs = 30000.0
        samples = np.array([30000, 45000, 60000, 90000], dtype=np.int64)  # 1.0..3.0 s
        df = pd.DataFrame({"time": samples, "unit_id": [1, 1, 2, 2]})
        X = build_causal_spike_matrix(df, [1, 2], np.array([2.0]), 1.0)
        if np.isfinite(X).all() and X.sum() == 0:
            return [
                Finding(
                    probe=self.name,
                    title="Sample-index timestamps are accepted as seconds",
                    severity=Severity.HIGH,
                    category="ingest",
                    where="realtime/spike_binner.py / realtime/data_loading.py",
                    detail=(
                        "`spike_times.npy` from Kilosort holds sample indices, not "
                        "seconds. Passing them through yields an empty but perfectly "
                        "well-formed count matrix — no error, no warning, just a "
                        "decoder that trains on silence. Nothing in the ingest path "
                        "records a sampling rate or sanity-checks the time range."
                    ),
                    evidence={
                        "fs_hz": fs,
                        "timestamps_passed": samples.tolist(),
                        "counts_returned": X.tolist(),
                    },
                    suggestion=(
                        "Require an explicit `fs` (or `time_units`) at the ingest "
                        "boundary and assert the resulting duration is plausible "
                        "(e.g. 1 s < span < 24 h) before anything downstream runs."
                    ),
                )
            ]
        return []

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

        data = load_simulation_data(d, spike_source="ground_truth")
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
