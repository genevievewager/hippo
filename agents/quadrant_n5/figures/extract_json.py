"""Build tidy source-data CSVs from the quadrant_n5 result JSONs (primary source)."""
import json, os, csv, sys
import numpy as np, pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "outputs/quadrant_n5"
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)
HASHES = set()
pop = []
REPS = ["raw", "raw_lag", "pca", "dm", "lds", "isomap", "gpfa"]
err, floors, shifts, lc, rep_rows = [], [], [], [], []
for s in range(5):
    for src in ("sorted", "ground_truth"):
        ss = json.load(open(f"{ROOT}/seed_{s}/{src}/source_summary.json"))
        floors.append(dict(seed=s, source=src, floor_median=ss["floor"]["median"],
                           floor_mean=ss["floor"]["mean"], test_bins=ss["coverage"].get("n_test_occupied_bins"),
                           train_bins=ss["coverage"].get("n_train_occupied_bins"),
                           coverage=ss["coverage"]["fraction_test_in_train_occupied_bins"]))
        for rep in REPS:
            p = f"{ROOT}/seed_{s}/{src}/{rep}.json"
            by = {x["method"]: x for x in ss["methods"]}
            m = json.load(open(p)) if os.path.exists(p) else dict(by[rep], seed_streams=ss["seed_streams"],
                                                                  config_sha256=ss["config_sha256"], seed_index=s)
            assert m["seed_index"] == s and m["method"] == rep
            HASHES.add(m["config_sha256"])
            a = m["a13"]
            err.append(dict(seed=s, source=src, rep=rep, d=m["primary_d"],
                            ridge=m["ridge"]["median"], ridge_mean=m["ridge"]["mean"], ridge_p90=m["ridge"]["p90"],
                            knn=m["knn"]["median"], knn_mean=m["knn"]["mean"], knn_p90=m["knn"]["p90"],
                            ridge_alpha=m["ridge_alpha"], knn_k=m["knn_k"],
                            inner_cv_ridge=m.get("inner_cv_ridge_median"), inner_cv_knn=m.get("inner_cv_knn_median"),
                            a13_ridge=a["ridge_median_minus_floor"], a13_knn=a["knn_median_minus_floor"],
                            a13_ridge_pass=a["ridge_pass"], a13_knn_pass=a["knn_pass"],
                            n_units=m["n_units"], n_train=m["n_train"], n_eval=m["n_eval"],
                            data_seed=m["seed_streams"]["data_seed"]))
            if "shifts" in ss.get("a13", {}).get(rep, {}):
                for sh in ss["a13"][rep]["shifts"]:
                    shifts.append(dict(seed=s, source=src, rep=rep, shift_s=sh["shift_s"],
                                       ridge_minus_floor=sh["ridge_minus_floor"], knn_minus_floor=sh["knn_minus_floor"]))
        if src == "sorted":
            for lab, nn in ss["n_units_by_cell_type"].items():
                pop.append(dict(seed=s, kind="cell_type", label=lab, n=nn))
            for lab, nn in ss["n_units_by_region"].items():
                pop.append(dict(seed=s, kind="region", label=lab, n=nn))
            for r in ss.get("learning_curves", []):
                lc.append(dict(seed=s, rep=r["method"], frac=r["frac"], n_train=r["n_train"],
                               ridge=r["ridge"]["median"]))
    rp = json.load(open(f"{ROOT}/seed_{s}/replay/sorted_summary.json"))
    for m in rp["methods"]:
        rep_rows.append(dict(seed=s, rep=m["method"], a9=m["a9_label"], z_inf=m["max_abs_step_vs_batch"],
                             p50_ms=m["step_ms_p50"], p99_ms=m["step_ms_p99"], max_ms=m["step_ms_max"],
                             frac_over_budget=m["frac_over_budget"], a11_yhat_inf_cm=m["a11_max_abs_pred"],
                             phase3_vs_replay=m["phase3_vs_replay_offline_ridge"]))
aud = json.load(open(f"{ROOT}/audit_summary.json"))
arows = [dict(seed=x["seed_index"], check=r["check"], status=r["status"], note=r["note"])
         for x in aud["seeds"] for r in x["rows"]]
assert len(HASHES) == 1, f"results mix config hashes: {HASHES}"
for name, rows in [("population", pop), ("errors", err), ("floor", floors), ("a13_shifts", shifts), ("learning_curves", lc),
                   ("replay", rep_rows), ("audit", arows)]:
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"data_{name}.csv"), index=False)
    print(name, len(rows))
# behaviour: downsample to 1 Hz for trajectory panels (test segment is last 20%)
beh = []
for s in range(5):
    b = pd.read_csv(f"{ROOT}/seed_{s}/sim/behavior.csv")
    b = b.iloc[::4][["time_s", "x_cm", "y_cm"]].copy(); b["seed"] = s
    beh.append(b)
pd.concat(beh).to_csv(os.path.join(OUT, "data_behavior_5hz.csv"), index=False)
cfg = HASHES.pop()
rp0 = json.load(open(f"{ROOT}/seed_0/replay/sorted_summary.json"))
meta = dict(config_sha256=cfg,
            seeds_code_sha=aud["seeds"][0].get("seeds_0_4_code_sha", ""),
            report_code_sha=rp0.get("git_sha", ""),
            probe_track_sha256=rp0.get("probe_track_sha256", ""),
            latency_budget_ms=rp0.get("latency_budget_ms"))
json.dump(meta, open(os.path.join(OUT, "data_meta.json"), "w"), indent=1)
print("behavior ok; config", cfg[:12])

# ---- Phase 7: predictions.npz / d_sweep.json (skip if absent) ----
ARENA = 100.0
CX, CY = ARENA / 2.0, ARENA / 2.0
N_BINS = 10
WINDOW_S = 60.0
PRED_METHODS = ["pca", "dm", "lds", "raw"]
ALL_DEC = ("ridge", "knn")


def _bin_ix(xy):
    return np.clip((np.asarray(xy) / ARENA * N_BINS).astype(int), 0, N_BINS - 1)


win_rows, map_rows, pull_rows, cdf_rows, dsweep_rows, jump_rows = [], [], [], [], [], []
cdf_pool = {}
JUMP_CM = 20.0
for s in range(5):
    for src in ("sorted", "ground_truth"):
        npz_path = f"{ROOT}/seed_{s}/{src}/predictions.npz"
        if not os.path.isfile(npz_path):
            continue
        blob = np.load(npz_path)
        t = np.asarray(blob["t_s"], dtype=float)
        y = np.asarray(blob["y_true"], dtype=float)
        t0 = float(t[0])
        win = (t >= t0) & (t < t0 + WINDOW_S)
        true_step = np.linalg.norm(np.diff(y, axis=0), axis=1)
        true_jump = float((true_step > JUMP_CM).mean()) if len(true_step) else 0.0
        jump_rows.append(dict(
            seed=s, source=src, method="true", decoder="true",
            jump_rate=true_jump, jump_thresh_cm=JUMP_CM, n_steps=int(len(true_step)),
        ))
        for method in REPS:
            for dec in ALL_DEC:
                key = f"pred_{method}_{dec}"
                if key not in blob.files:
                    continue
                pred = np.asarray(blob[key], dtype=float)
                err = np.linalg.norm(pred - y, axis=1)
                step = np.linalg.norm(np.diff(pred, axis=0), axis=1)
                jump_rows.append(dict(
                    seed=s, source=src, method=method, decoder=dec,
                    jump_rate=float((step > JUMP_CM).mean()) if len(step) else 0.0,
                    jump_thresh_cm=JUMP_CM, n_steps=int(len(step)),
                    true_jump_rate=true_jump,
                ))
                # window traces (PCA/DM/LDS/raw only — Fig 7)
                if method in PRED_METHODS:
                    for i in np.where(win)[0]:
                        win_rows.append(dict(
                            seed=s, source=src, method=method, decoder=dec,
                            t_s=float(t[i]), x_true=float(y[i, 0]), y_true=float(y[i, 1]),
                            x_pred=float(pred[i, 0]), y_pred=float(pred[i, 1]),
                            err_cm=float(err[i]),
                        ))
                # spatial error maps
                bx, by = _bin_ix(y[:, 0]), _bin_ix(y[:, 1])
                for ix in range(N_BINS):
                    for iy in range(N_BINS):
                        m = (bx == ix) & (by == iy)
                        n = int(m.sum())
                        map_rows.append(dict(
                            seed=s, source=src, method=method, decoder=dec,
                            bin_x=ix, bin_y=iy, n=n,
                            median_err=float(np.median(err[m])) if n else np.nan,
                        ))
                # centre-pull
                d_true = np.linalg.norm(y - np.array([CX, CY]), axis=1)
                d_pred = np.linalg.norm(pred - np.array([CX, CY]), axis=1)
                X = np.column_stack([np.ones(len(d_true)), d_true])
                coef, _, _, _ = np.linalg.lstsq(X, d_pred, rcond=None)
                pull_rows.append(dict(
                    seed=s, source=src, method=method, decoder=dec,
                    kind="slope", intercept=float(coef[0]), slope=float(coef[1]),
                    bin_lo=np.nan, bin_hi=np.nan, n=int(len(d_true)),
                    mean_d_true=np.nan, mean_d_pred=np.nan,
                ))
                edges = np.arange(0.0, 55.0, 5.0)
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = (d_true >= lo) & (d_true < hi)
                    pull_rows.append(dict(
                        seed=s, source=src, method=method, decoder=dec,
                        kind="bin", intercept=np.nan, slope=np.nan,
                        bin_lo=float(lo), bin_hi=float(hi), n=int(m.sum()),
                        mean_d_true=float(d_true[m].mean()) if m.any() else np.nan,
                        mean_d_pred=float(d_pred[m].mean()) if m.any() else np.nan,
                    ))
                # CDF pool (across seeds later)
                cdf_pool.setdefault((src, method, dec), []).append(err)
        # d_sweep
        ds_path = f"{ROOT}/seed_{s}/{src}/d_sweep.json"
        if os.path.isfile(ds_path):
            sw = json.load(open(ds_path))
            if sw.get("selection_rule") == "phase3_per_fold_refit":
                for r in sw.get("rows") or []:
                    dsweep_rows.append(dict(
                        seed=s, source=src, method=r["method"], d=int(r["d"]),
                        ridge_median=r["ridge_median"], knn_median=r["knn_median"],
                        ridge_alpha=r["ridge_alpha"], knn_k=r["knn_k"],
                        inner_cv_ridge_median=r["inner_cv_ridge_median"],
                        inner_cv_knn_median=r["inner_cv_knn_median"],
                        selected=bool(r.get("selected")),
                        selection_rule=sw["selection_rule"],
                    ))

qs = np.linspace(0.0, 1.0, 101)
for (src, method, dec), chunks in cdf_pool.items():
    pooled = np.concatenate(chunks)
    for q, v in zip(qs, np.quantile(pooled, qs)):
        cdf_rows.append(dict(
            source=src, method=method, decoder=dec, q=float(q), err_cm=float(v),
        ))

for name, rows in [
    ("predictions_window", win_rows),
    ("error_maps", map_rows),
    ("center_pull", pull_rows),
    ("error_cdf", cdf_rows),
    ("d_sweep", dsweep_rows),
    ("jump_rate", jump_rows),
]:
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"data_{name}.csv"), index=False)
        print(name, len(rows))
    else:
        print(name, "skipped (absent)")

# ---- Phase 8: knn_pressure (skip if absent) ----
PRESS = os.path.join(ROOT, "knn_pressure")
if os.path.isdir(PRESS):
    for src_name, dest_name in (
        ("exclusion.csv", "knn_exclusion"),
        ("neighbour_diag.csv", "knn_neighbour"),
        ("strata.csv", "knn_strata"),
    ):
        path = os.path.join(PRESS, src_name)
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df.to_csv(os.path.join(OUT, f"data_{dest_name}.csv"), index=False)
            print(dest_name, len(df))
    crit_path = os.path.join(PRESS, "criteria.json")
    if os.path.isfile(crit_path):
        import shutil
        shutil.copy(crit_path, os.path.join(OUT, "data_knn_criteria.json"))
        print("knn_criteria copied")
else:
    print("knn_pressure skipped (absent)")
