"""Unit tests for Phase 8 kNN pressure helpers (no full experiment)."""
import numpy as np

from agents.quadrant_n5.knn_pressure import (
    _circular_distance,
    _circular_mean,
    _gain_table,
    _shuffle_gain_table,
    build_strata_masks,
    evaluate_cell_type_criterion,
    evaluate_criteria,
    evaluate_path_integration,
    train_mask_excluding_delta,
)
import pandas as pd


def test_train_mask_excluding_delta_last_seconds():
    t = np.arange(0.0, 10.0, 1.0)
    train = np.ones(len(t), dtype=bool)
    keep0 = train_mask_excluding_delta(t, train, 0.0)
    assert keep0.sum() == 10
    keep3 = train_mask_excluding_delta(t, train, 3.0)
    # t_end = 9; keep t <= 6
    assert list(t[keep3]) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_circular_mean_and_distance():
    # Two opposite angles → mean undefined-ish but atan2 of sum may be either;
    # use a tight cluster around π/2.
    angs = np.array([np.pi / 2 - 0.1, np.pi / 2, np.pi / 2 + 0.1])
    mu = _circular_mean(angs)
    assert abs(mu - np.pi / 2) < 0.05
    d = _circular_distance(np.array([np.pi / 2 + np.pi]), mu)
    assert abs(d[0] - np.pi) < 1e-6 or abs(d[0] - 0) < 1e-6 or d[0] > np.pi / 2


def test_strata_marks_heading_atypical():
    # 4 train samples in one bin heading east; one test heading west.
    y = np.array([[10.0, 10.0]] * 5)
    heading = np.array([0.0, 0.05, -0.05, 0.02, np.pi])  # last is west
    speed = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    train_ok = np.array([True, True, True, True, False])
    eval_mask = np.array([False, False, False, False, True])
    s = build_strata_masks(y, heading, speed, train_ok, eval_mask)
    assert s["atypical"][0]
    assert s["heading_atypical"][0]
    assert s["heading_dist_rad"][0] > np.pi / 2 - 1e-6


def test_evaluate_criteria_shape():
    excl = []
    for s in range(5):
        for d in (0.0, 30.0):
            excl.append(dict(
                seed=s, source="sorted", method="lds", decoder="knn",
                delta_s=d, median_err=10.0 + (0.5 if d == 30 else 0.0),
            ))
    nbr = [dict(seed=s, source="sorted", method="lds", median_abs_dt_s=100.0) for s in range(5)]
    st = []
    for s in range(5):
        st.append(dict(seed=s, source="sorted", method="lds", decoder="knn",
                       stratum="atypical", median_err=8.0))
        st.append(dict(seed=s, source="sorted", method="pca", decoder="knn",
                       stratum="atypical", median_err=12.0))
    c = evaluate_criteria(pd.DataFrame(excl), pd.DataFrame(nbr), pd.DataFrame(st))
    assert c["exclusion_delta30"]["status"] == "PASS"
    assert c["neighbour_median_dt"]["status"] == "PASS"
    assert c["strata_lds_vs_pca_atypical"]["status"] == "PASS"
    assert c["cell_type_grid_bvc"]["status"] == "PENDING"
    assert c["overall_1_to_3"] == "PASS"
    assert c["phase8_passed"] is False


def test_evaluate_cell_type_criterion_pass_fail():
    rows = []
    for s in range(5):
        # LDS better on seeds 0–3; PCA better on seed 4 → 4/5 PASS
        pca_err = 20.0
        lds_err = 10.0 if s < 4 else 25.0
        for method, err in (("pca", pca_err), ("lds", lds_err)):
            rows.append(dict(
                seed=s, source="sorted", subset="grid_bvc", method=method,
                decoder="knn", median_err=err,
            ))
            rows.append(dict(
                seed=s, source="sorted", subset="hd_speed", method=method,
                decoder="knn", median_err=err + 5,
            ))
    ct = evaluate_cell_type_criterion(pd.DataFrame(rows))
    assert ct["status"] == "PASS"
    assert ct["n_lds_better"] == 4
    assert ct["sign_count"]["LDS_better"] == 4
    assert ct["sign_count"]["PCA_better"] == 1
    c = evaluate_criteria(
        pd.DataFrame([dict(seed=0, source="sorted", method="lds", decoder="knn",
                           delta_s=0.0, median_err=1.0)]),
        pd.DataFrame([dict(seed=0, source="sorted", method="lds", median_abs_dt_s=1.0)]),
        pd.DataFrame([dict(seed=0, source="sorted", method="lds", decoder="knn",
                           stratum="atypical", median_err=1.0)]),
        cell_type=pd.DataFrame(rows),
    )
    assert c["cell_type_grid_bvc"]["status"] == "PASS"


def test_gain_table_and_shuffle_gain_table_accept_int_seeds():
    """Regression: sorted(int(int(s) for s in ...)) raised TypeError on unique()."""
    rows = []
    for s in range(2):
        for subset, pca_e, lds_e in (("grid_bvc", 30.0, 25.0), ("all", 20.0, 10.0)):
            for method, err in (("pca", pca_e), ("lds", lds_e)):
                for dec in ("ridge", "knn"):
                    rows.append(dict(
                        seed=s, subset=subset, method=method, decoder=dec,
                        median_err=err, shuffle_offset_s=None,
                    ))
    mech = pd.DataFrame(rows)
    gains = _gain_table(mech)
    assert set(gains.seed.tolist()) == {0, 1}
    assert "interaction" in set(gains.method)
    # seed column as int64 from CSV round-trip
    import io
    mech2 = pd.read_csv(io.StringIO(mech.to_csv(index=False)))
    gains2 = _gain_table(mech2)
    assert len(gains2) == len(gains)

    shuf_rows = []
    for s in range(2):
        for offset in (60.0, 120.0):
            for method, err in (("pca", 22.0), ("lds", 18.0)):
                for dec in ("ridge", "knn"):
                    shuf_rows.append(dict(
                        seed=s, method=method, decoder=dec,
                        shuffle_offset_s=offset, median_err=err,
                    ))
    shuf = pd.DataFrame(shuf_rows)
    sg = _shuffle_gain_table(shuf, mech)
    assert set(sg.seed.tolist()) == {0, 1}
    assert len(sg) == 8  # 2 seeds × 2 methods × 2 decoders (mean over offsets)


def test_evaluate_path_integration_pass():
    rows = []
    for s in range(5):
        for subset, err_pca, err_lds in (
            ("grid_bvc", 30.0, 25.0),
            ("all", 20.0 if s < 4 else 28.0, 15.0 if s < 4 else 24.0),
        ):
            for method, err in (("pca", err_pca), ("lds", err_lds)):
                for dec in ("ridge", "knn"):
                    rows.append(dict(
                        seed=s, source="sorted", subset=subset, method=method,
                        decoder=dec, median_err=err, shuffle_offset_s=None,
                    ))
    mech = pd.DataFrame(rows)
    gains = _gain_table(mech)
    shuf = gains[(gains.decoder == "knn") & (gains.method == "lds")].copy()
    shuf["gain_all_minus_grid_bvc"] = shuf["gain_all_minus_grid_bvc"] * 0.2
    a, b = evaluate_path_integration(mech, shuf)
    assert a["n_lds_gain_negative"] >= 4
    assert b["status"] == "PASS"
