"""Kilosort/Phy ingest.

The property under test throughout: a wrong or missing sampling rate must
raise, never produce a well-formed empty result. Sample indices read as
seconds is a 30,000x error that leaves every causal window empty, and an
empty window is indistinguishable from a quiet recording.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from realtime.live.phy_stream import (
    PhyIngestError,
    PhySpikeStream,
    build_units_table,
    load_phy_spikes,
    read_cluster_groups,
    read_sample_rate,
)

FS = 30000.0


def make_phy_dir(
    tmp_path: Path,
    *,
    duration_s: float = 120.0,
    n_spikes: int = 4000,
    fs: float = FS,
    write_params: bool = True,
    groups: dict[int, str] | None = None,
    seed: int = 0,
) -> Path:
    d = tmp_path / "phy"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, duration_s, n_spikes))
    clusters = rng.choice([3, 11, 47, 108], n_spikes).astype(np.int32)
    np.save(d / "spike_times.npy", (t * fs).astype(np.int64))
    np.save(d / "spike_clusters.npy", clusters)
    if write_params:
        (d / "params.py").write_text(
            "dat_path = 'raw.bin'\nn_channels_dat = 384\n"
            f"sample_rate = {fs}\ndtype = 'int16'\n"
        )
    groups = groups or {3: "good", 11: "good", 47: "mua", 108: "noise"}
    pd.DataFrame(
        {"cluster_id": list(groups), "group": list(groups.values())}
    ).to_csv(d / "cluster_group.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"cluster_id": [3, 11, 47, 108], "depth": [3400.0, 3100.0, 2600.0, 900.0]}
    ).to_csv(d / "cluster_info.tsv", sep="\t", index=False)
    return d


def test_reads_sample_rate_from_params(tmp_path):
    d = make_phy_dir(tmp_path)
    assert read_sample_rate(d) == pytest.approx(FS)


def test_missing_params_refuses_rather_than_guessing(tmp_path):
    d = make_phy_dir(tmp_path, write_params=False)
    with pytest.raises(PhyIngestError, match="params.py"):
        load_phy_spikes(d)


def test_sample_indices_are_converted_to_seconds(tmp_path):
    d = make_phy_dir(tmp_path, duration_s=120.0)
    df, meta = load_phy_spikes(d)
    assert meta["fs_hz"] == pytest.approx(FS)
    assert meta["duration_s"] == pytest.approx(120.0, abs=1.0)
    # Seconds, not sample indices.
    assert df["time"].max() < 200.0


def test_wrong_sampling_rate_raises_not_empties(tmp_path):
    """The whole point: a bad fs must not yield a plausible empty session."""
    d = make_phy_dir(tmp_path, duration_s=120.0)
    with pytest.raises(PhyIngestError, match="plausible range"):
        load_phy_spikes(d, fs=1.0)          # indices treated as seconds
    with pytest.raises(PhyIngestError, match="plausible range"):
        load_phy_spikes(d, fs=30_000_000.0)  # absurdly high


def test_spikes_are_sorted_ascending(tmp_path):
    d = make_phy_dir(tmp_path)
    df, _ = load_phy_spikes(d)
    t = df["time"].to_numpy()
    assert np.all(t[:-1] <= t[1:])


def test_quality_filter_keeps_only_accepted_groups(tmp_path):
    d = make_phy_dir(tmp_path)
    good, _ = load_phy_spikes(d, accepted_groups=("good",))
    assert set(good["unit_id"].unique()) == {3, 11}

    with_mua, _ = load_phy_spikes(d, accepted_groups=("good", "mua"))
    assert set(with_mua["unit_id"].unique()) == {3, 11, 47}

    everything, _ = load_phy_spikes(d, accepted_groups=None)
    assert set(everything["unit_id"].unique()) == {3, 11, 47, 108}


def test_all_clusters_filtered_out_raises(tmp_path):
    d = make_phy_dir(tmp_path, groups={3: "noise", 11: "noise", 47: "noise", 108: "noise"})
    with pytest.raises(PhyIngestError, match="No clusters remain"):
        load_phy_spikes(d, accepted_groups=("good",))


def test_mismatched_file_lengths_raise(tmp_path):
    d = make_phy_dir(tmp_path)
    np.save(d / "spike_clusters.npy", np.array([1, 2, 3], dtype=np.int32))
    with pytest.raises(PhyIngestError, match="parallel"):
        load_phy_spikes(d)


def test_sparse_cluster_ids_are_preserved(tmp_path):
    """Phy cluster ids are arbitrary and never 0..N-1."""
    d = make_phy_dir(tmp_path)
    df, _ = load_phy_spikes(d, accepted_groups=None)
    assert sorted(df["unit_id"].unique()) == [3, 11, 47, 108]


def test_stream_replays_in_causal_chunks(tmp_path):
    d = make_phy_dir(tmp_path, duration_s=60.0)
    s = PhySpikeStream(d)
    s.connect()
    assert s.connected
    assert s.list_unit_ids() == [3, 11]

    first = s.get_new_spikes(up_to_time=10.0)
    second = s.get_new_spikes(up_to_time=20.0)
    assert len(first) and len(second)
    # No overlap, no gap, strictly forward.
    assert first["time"].max() <= second["time"].min()
    assert first["time"].max() < 10.0
    assert second["time"].max() < 20.0
    # Re-polling the same instant yields nothing.
    assert s.get_new_spikes(up_to_time=20.0).empty


def test_stream_feeds_the_causal_buffer(tmp_path):
    """End to end: Phy on disk -> buffer -> count vector."""
    from realtime.live.spike_buffer import CausalSpikeBuffer

    d = make_phy_dir(tmp_path, duration_s=60.0, n_spikes=6000)
    s = PhySpikeStream(d)
    s.connect()
    buf = CausalSpikeBuffer(s.list_unit_ids(), history_s=2.0)

    counts = None
    for t in np.arange(1.0, 20.0, 0.25):
        chunk = s.get_new_spikes(up_to_time=float(t))
        if not chunk.empty:
            buf.extend_dataframe(chunk)
        counts = buf.counts_at(float(t), 0.25)
    assert counts is not None
    assert counts.shape == (len(s.list_unit_ids()),)
    assert counts.sum() > 0


def test_units_table_maps_depth_to_region(tmp_path):
    d = make_phy_dir(tmp_path)
    depth_map = [
        (3200.0, 3600.0, "subiculum"),
        (2400.0, 3200.0, "dentate_gyrus"),
        (0.0, 2400.0, "visual_cortex"),
    ]
    units = build_units_table(d, [3, 11, 47, 108], depth_to_region=depth_map)
    by_id = units.set_index("unit_id")["region"].to_dict()
    assert by_id[3] == "subiculum"
    assert by_id[11] == "dentate_gyrus"
    assert by_id[108] == "visual_cortex"


def test_units_table_without_a_region_source_is_unknown(tmp_path):
    """Phy has no anatomy. Inventing one would be worse than refusing."""
    d = make_phy_dir(tmp_path)
    units = build_units_table(d, [3, 11])
    assert set(units["region"]) == {"unknown"}


def test_cluster_groups_read_from_ks_label_when_uncurated(tmp_path):
    d = make_phy_dir(tmp_path)
    (d / "cluster_group.tsv").unlink()
    pd.DataFrame(
        {"cluster_id": [3, 11, 47, 108], "KSLabel": ["good", "good", "mua", "mua"]}
    ).to_csv(d / "cluster_KSLabel.tsv", sep="\t", index=False)
    groups = read_cluster_groups(d)
    assert groups["quality_source"].iloc[0] == "cluster_KSLabel.tsv"
    assert set(groups.loc[groups["quality"] == "good", "unit_id"]) == {3, 11}
