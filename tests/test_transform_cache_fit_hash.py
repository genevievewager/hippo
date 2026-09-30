"""Transform-cache isolation: fit_hash key + provenance + unkeyed refuse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from realtime.pipeline_artifacts import ObservationConfig, hash_train_indices, write_provenance
from realtime.transform_cache import (
    feature_transform_dirname,
    find_feature_transform,
    find_feature_transform_in_roots,
    save_feature_transform_checkpoint,
)


def _obs(**overrides) -> ObservationConfig:
    kwargs = dict(
        window_s=0.250,
        update_dt=0.050,
        feature_set="counts",
        feature_type="counts",
        source_spikes="sorted",
        simulation_run_id="toy",
        seed=0,
        train_frac=0.70,
        train_index_hash="aaaaaaaaaaaaaaaa",
        session_s=600.0,
        config_hash="bbbbbbbbbbbbbbbb",
    )
    kwargs.update(overrides)
    return ObservationConfig(**kwargs)


def _write_keyed(
    root: Path,
    observation: ObservationConfig,
    *,
    provenance_fit_hash: str | None = None,
) -> Path:
    fit = observation.fit_hash()
    name = feature_transform_dirname(
        observation.feature_set,
        observation.feature_type,
        observation.window_s,
        fit_hash=fit,
    )
    d = root / "models" / "feature_transforms" / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "feature_set": observation.feature_set,
        "feature_type_eff": observation.feature_type,
        "decode_window_s": observation.window_s,
        "fit_hash": fit,
    }))
    write_provenance(d, {
        "kind": "SpikeFeatureTransformer",
        "fit_hash": fit if provenance_fit_hash is None else provenance_fit_hash,
        "train_index_hash": observation.train_index_hash,
        "session_s": observation.session_s,
        "config_hash": observation.config_hash,
        "seed": observation.seed,
        "train_frac": observation.train_frac,
    })
    return d


def test_fit_hash_covers_required_partition_fields():
    base = _obs()
    variants = {
        "seed": _obs(seed=1),
        "source_spikes": _obs(source_spikes="ground_truth"),
        "train_index_hash": _obs(train_index_hash="cccccccccccccccc"),
        "train_frac": _obs(train_frac=0.50),
        "session_s": _obs(session_s=300.0),
        "window_s": _obs(window_s=0.500),
        "update_dt": _obs(update_dt=0.025),
        "config_hash": _obs(config_hash="dddddddddddddddd"),
    }
    for name, other in variants.items():
        assert other.fit_hash() != base.fit_hash(), name
        assert feature_transform_dirname(
            other.feature_set, other.feature_type, other.window_s,
            fit_hash=other.fit_hash(),
        ) != feature_transform_dirname(
            base.feature_set, base.feature_type, base.window_s,
            fit_hash=base.fit_hash(),
        ), name


def test_seed_only_difference_never_shares_cache_hit(tmp_path: Path):
    a = _obs(seed=0)
    b = _obs(seed=1)
    assert a.train_frac == b.train_frac
    assert a.train_index_hash == b.train_index_hash
    root = tmp_path / "cmp"
    _write_keyed(root, a)
    assert find_feature_transform(
        root,
        feature_set="counts",
        feature_type_eff="counts",
        decode_window=0.250,
        fit_hash=a.fit_hash(),
    ) is not None
    assert find_feature_transform(
        root,
        feature_set="counts",
        feature_type_eff="counts",
        decode_window=0.250,
        fit_hash=b.fit_hash(),
    ) is None


def test_train_frac_only_difference_never_shares_cache_hit(tmp_path: Path):
    times = np.arange(0.0, 10.0, 0.05)
    n = len(times)
    n_a = int(round(0.70 * n))
    n_b = int(round(0.50 * n))
    mask_a = np.zeros(n, dtype=bool)
    mask_a[:n_a] = True
    mask_b = np.zeros(n, dtype=bool)
    mask_b[:n_b] = True
    a = _obs(train_frac=0.70, train_index_hash=hash_train_indices(mask_a))
    b = _obs(train_frac=0.50, train_index_hash=hash_train_indices(mask_b))
    assert a.seed == b.seed
    assert a.train_index_hash != b.train_index_hash
    root = tmp_path / "cmp"
    _write_keyed(root, a)
    assert find_feature_transform(
        root,
        feature_set="counts",
        feature_type_eff="counts",
        decode_window=0.250,
        fit_hash=a.fit_hash(),
    ) is not None
    assert find_feature_transform(
        root,
        feature_set="counts",
        feature_type_eff="counts",
        decode_window=0.250,
        fit_hash=b.fit_hash(),
    ) is None


def test_provenance_mismatch_refuses_and_is_a_miss(tmp_path: Path):
    a = _obs(seed=0)
    root = tmp_path / "cmp"
    _write_keyed(root, a, provenance_fit_hash="not-the-real-fit-hash")
    assert find_feature_transform(
        root,
        feature_set="counts",
        feature_type_eff="counts",
        decode_window=0.250,
        fit_hash=a.fit_hash(),
    ) is None


def test_unkeyed_dir_is_never_returned_even_with_matching_provenance(tmp_path: Path):
    a = _obs()
    root = tmp_path / "cmp"
    unkeyed = root / "models" / "feature_transforms" / "counts__counts_w0250ms"
    unkeyed.mkdir(parents=True)
    (unkeyed / "meta.json").write_text("{}")
    write_provenance(unkeyed, {"fit_hash": a.fit_hash()})
    assert find_feature_transform(
        root,
        feature_set="counts",
        feature_type_eff="counts",
        decode_window=0.250,
        fit_hash=a.fit_hash(),
    ) is None
    assert find_feature_transform_in_roots(
        [root],
        feature_set="counts",
        feature_type_eff="counts",
        decode_window=0.250,
        fit_hash=a.fit_hash(),
    ) is None


def test_save_requires_fit_hash(tmp_path: Path):
    class _Dummy:
        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "dummy").write_text("ok")

    try:
        save_feature_transform_checkpoint(
            _Dummy(),
            tmp_path / "cmp",
            feature_set="counts",
            feature_type_eff="counts",
            decode_window=0.250,
        )
    except ValueError as exc:
        assert "fit_hash" in str(exc)
    else:
        raise AssertionError("unkeyed write must be refused")
