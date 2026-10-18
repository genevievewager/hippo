"""Discriminator: what gives the simulation away (SPEC §5.3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_METRICS = (
    "firing_rate_hz",
    "isi_cv",
    "burst_fraction",
    "fano_factor",
    "participation_ratio",
    "mean_pairwise_corr",
    "synchrony_frac",
    "spatial_info_median",
    "theta_mod_index_median",
    "decode_error_over_floor",
)


def _pivot_features(df: pd.DataFrame, metrics: tuple[str, ...] = FEATURE_METRICS) -> pd.DataFrame:
    """One row per (dataset, session) with median metric values as features."""
    sub = df[df["metric"].isin(metrics)].copy()
    if sub.empty:
        return pd.DataFrame()
    g = (
        sub.groupby(["dataset", "animal|seed", "session", "metric"], as_index=False)["value"]
        .median()
    )
    wide = g.pivot_table(
        index=["dataset", "animal|seed", "session"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    return wide


def _group_label_shuffle(
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle binary labels at the group level (one label per group)."""
    uniq = np.unique(groups)
    lab_map = {g: int(y[groups == g][0]) for g in uniq}
    vals = np.array(list(lab_map.values()), dtype=int)
    rng.shuffle(vals)
    for g, v in zip(uniq, vals):
        lab_map[g] = int(v)
    return np.array([lab_map[g] for g in groups], dtype=int)


def _cv_auc_scores(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    stratified: bool,
) -> list[float]:
    n_splits = min(int(n_splits), len(np.unique(groups)))
    if n_splits < 2:
        return []
    if stratified:
        try:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
            splits = splitter.split(X, y, groups)
        except Exception:
            # Fall back if class counts cannot support stratified groups.
            splitter = GroupKFold(n_splits=n_splits)
            splits = splitter.split(X, y, groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, groups)
    scores: list[float] = []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model.fit(X[tr], y[tr])
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X[te])[:, 1]
        else:
            p = model.decision_function(X[te])
        scores.append(float(roc_auc_score(y[te], p)))
    return scores


def run_discriminator(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    rng = rng or np.random.default_rng(0)
    wide = _pivot_features(df)
    if wide.empty or len(wide) < 4:
        return {"status": "insufficient", "n_rows": int(len(wide))}

    y = (wide["dataset"].astype(str).str.startswith("sim")).astype(int).to_numpy()
    # Matched arm reuses lab session names for sim_matched — session alone would
    # merge opposite labels into one group and collapse the null to NaN.
    groups = (
        wide["dataset"].astype(str) + "::" + wide["session"].astype(str)
    ).to_numpy()
    feat_cols = [c for c in wide.columns if c not in {"dataset", "animal|seed", "session"}]
    X = np.array(wide[feat_cols].to_numpy(dtype=float), copy=True)
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isfinite(col)
        fill = float(np.median(col[m])) if m.any() else 0.0
        col[~m] = fill
        X[:, j] = col

    n_groups = int(len(np.unique(groups)))
    n_splits = min(int(n_splits), n_groups)
    if n_splits < 2:
        return {"status": "too_few_groups", "n_groups": n_groups}

    def _cv_auc(model, *, stratified: bool) -> float:
        scores = _cv_auc_scores(
            model, X, y, groups, n_splits=n_splits, stratified=stratified,
        )
        return float(np.mean(scores)) if scores else float("nan")

    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=0),
    )
    hgb = HistGradientBoostingClassifier(max_depth=3, random_state=0)
    auc_logit = _cv_auc(logit, stratified=False)
    auc_hgb = _cv_auc(hgb, stratified=False)
    auc_logit_strat = _cv_auc(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)),
        stratified=True,
    )

    # Null A: group-level label shuffle + GroupKFold (legacy; can be NaN when folds
    # lack both classes after shuffle — common on matched arm).
    y_shuf = _group_label_shuffle(y, groups, rng)
    scores_null = _cv_auc_scores(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)),
        X, y_shuf, groups, n_splits=n_splits, stratified=False,
    )
    auc_null = float(np.mean(scores_null)) if scores_null else float("nan")

    # Null B: same group-level shuffle + StratifiedGroupKFold (primary for matched).
    scores_null_s = _cv_auc_scores(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)),
        X, y_shuf, groups, n_splits=n_splits, stratified=True,
    )
    auc_null_strat = float(np.mean(scores_null_s)) if scores_null_s else float("nan")

    # permutation importance on full fit (logit); also |coef| when AUC saturates
    logit.fit(X, y)
    base = float(roc_auc_score(y, logit.predict_proba(X)[:, 1]))
    importances = []
    for j, name in enumerate(feat_cols):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        pj = float(roc_auc_score(y, logit.predict_proba(Xp)[:, 1]))
        importances.append({"feature": name, "auc_drop": base - pj})
    try:
        coefs = np.abs(logit.named_steps["logisticregression"].coef_).ravel()
        for j, name in enumerate(feat_cols):
            importances[j]["abs_coef"] = float(coefs[j]) if j < len(coefs) else float("nan")
    except Exception:
        for im in importances:
            im["abs_coef"] = float("nan")
    importances.sort(
        key=lambda d: (d["auc_drop"], d.get("abs_coef") or 0.0),
        reverse=True,
    )
    importances_full = [dict(im) for im in importances]

    # ablation until AUC < 0.6 (GroupKFold, same as primary AUC)
    names = list(feat_cols)
    remaining_idx = list(range(len(names)))
    ablation = []
    X_full = X.copy()
    while remaining_idx:
        X_work = X_full[:, remaining_idx]
        auc = float(np.mean(_cv_auc_scores(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0)),
            X_work, y, groups, n_splits=n_splits, stratified=False,
        ) or [float("nan")]))
        ablation.append({
            "n_features": len(remaining_idx),
            "auc": auc,
            "features": [names[i] for i in remaining_idx],
        })
        if not np.isfinite(auc) or auc < 0.6 or len(remaining_idx) <= 1:
            break
        order = [im["feature"] for im in importances_full]
        present = {names[i] for i in remaining_idx}
        drop_name = next(n for n in order if n in present)
        drop_i = names.index(drop_name)
        remaining_idx = [i for i in remaining_idx if i != drop_i]

    return {
        "status": "ok",
        "n_rows": int(wide.shape[0]),
        "n_features": len(FEATURE_METRICS),
        "auc_logit": auc_logit,
        "auc_hgb": auc_hgb,
        "auc_logit_stratified_group": auc_logit_strat,
        "auc_group_shuffle_null": auc_null,
        "auc_group_shuffle_null_stratified": auc_null_strat,
        "null_note": (
            "Both nulls shuffle labels at group (session) level. "
            "auc_group_shuffle_null uses GroupKFold; "
            "auc_group_shuffle_null_stratified uses StratifiedGroupKFold. "
            "Groups are dataset::session so matched arms do not merge opposite labels."
        ),
        "importances": importances_full,
        "ablation": ablation,
    }
