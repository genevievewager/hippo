"""Dataset-level union of decoder comparison metrics across partial runs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

UNION_CSV_NAME = "all_comparison_metrics.csv"

_KEY_COLS = (
    "spike_source",
    "target_name",
    "decoder_name",
    "feature_set",
    "feature_mode",
    "embedding_type",
    "decode_window_s",
    "manifold_n_components",
    "n_neighbors",
    "n_landmarks",
)


def union_csv_path(experiment_dir: Path) -> Path:
    return Path(experiment_dir) / "decoder_comparison" / UNION_CSV_NAME


def config_key_parts(row: Mapping[str, Any] | pd.Series) -> list[str]:
    """Normalized identity fields used for union keys and ``config_id``."""
    getter = row.get if hasattr(row, "get") else None
    parts: list[str] = []
    for col in _KEY_COLS:
        if getter is not None:
            value = getter(col)
        else:
            value = row[col] if col in row else None
        if col == "decode_window_s":
            try:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    parts.append("")
                else:
                    parts.append(f"{float(value):.3f}")
            except (TypeError, ValueError):
                parts.append("")
        else:
            parts.append(_norm_key_value(value))
    return parts


def canonical_config_key(row: Mapping[str, Any] | pd.Series) -> str:
    """Pipe-joined identity string; same combo always yields the same key."""
    return "|".join(config_key_parts(row))


def _slug_token(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return text or "x"


def make_config_id(row: Mapping[str, Any] | pd.Series) -> str:
    """Deterministic filesystem-safe id for one evaluated configuration.

    Not a UUID. The same identity fields always resolve to the same id.
    A short SHA1 suffix keeps truncated slugs unique.
    """
    parts = config_key_parts(row)
    source, target, decoder, feature_set, _mode, embedding, window, k, nn, n_land = parts
    tokens = [
        source or "src",
        target or "tgt",
        feature_set or "fs",
        embedding or "emb",
        decoder or "dec",
        f"w{window}" if window else "w",
        f"k{k}" if k else "k",
    ]
    if nn:
        tokens.append(f"nn{nn}")
    if n_land:
        tokens.append(f"nl{n_land}")
    slug = "__".join(_slug_token(t) for t in tokens)
    digest = hashlib.sha1(canonical_config_key(row).encode("utf-8")).hexdigest()[:10]
    if len(slug) > 120:
        slug = slug[:120].rstrip("_")
    return f"{slug}__{digest}"


def ensure_config_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``config_id`` from identity columns (legacy rows)."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    if "config_id" not in out.columns:
        out["config_id"] = ""
    missing = out["config_id"].isna() | (out["config_id"].astype(str).str.strip() == "")
    if missing.any():
        filled = [
            make_config_id(out.loc[idx])
            for idx in out.index[missing]
        ]
        out.loc[missing, "config_id"] = filled
    return out


def _norm_key_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.6g}"
    return str(value).strip()


def _key_series(df: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    n = len(df)
    index = df.index
    for col in _KEY_COLS:
        if col not in df.columns:
            parts.append(pd.Series([""] * n, index=index))
            continue
        if col == "decode_window_s":
            nums = pd.to_numeric(df[col], errors="coerce").round(3)
            parts.append(nums.map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}"))
        else:
            parts.append(df[col].map(_norm_key_value))
    key = parts[0].astype(str)
    for p in parts[1:]:
        key = key + "|" + p.astype(str)
    return key


def merge_comparison_metrics(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame | None,
) -> pd.DataFrame:
    """Concat ``incoming`` over ``existing``; keep newest row per combo key."""
    if incoming is None or incoming.empty:
        return existing.copy() if existing is not None and not existing.empty else pd.DataFrame()
    if existing is None or existing.empty:
        return incoming.copy()
    old = existing.copy()
    new = incoming.copy()
    old["_merge_rank"] = 0
    new["_merge_rank"] = 1
    cat = pd.concat([old, new], ignore_index=True)
    cat["_combo_key"] = _key_series(cat)
    cat = cat.sort_values("_merge_rank")
    cat = cat.drop_duplicates("_combo_key", keep="last")
    return cat.drop(columns=["_merge_rank", "_combo_key"]).reset_index(drop=True)


def collect_comparison_metric_csvs(experiment_dir: Path) -> list[Path]:
    root = Path(experiment_dir) / "decoder_comparison"
    if not root.exists():
        return []
    union = union_csv_path(experiment_dir)
    paths: list[Path] = []
    for path in sorted(root.rglob("decoder_comparison_metrics.csv")):
        if path.resolve() == union.resolve():
            continue
        paths.append(path)
    return paths


def load_or_collect_union(experiment_dir: Path) -> pd.DataFrame:
    """Prefer ``all_comparison_metrics.csv``, else glob and merge per-run CSVs."""
    union_path = union_csv_path(experiment_dir)
    if union_path.exists() and union_path.stat().st_size > 0:
        try:
            df = pd.read_csv(union_path)
            if not df.empty:
                return df
        except (OSError, pd.errors.EmptyDataError):
            pass
    merged = pd.DataFrame()
    for path in collect_comparison_metric_csvs(experiment_dir):
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if frame.empty:
            continue
        merged = merge_comparison_metrics(merged, frame)
    return merged


def persist_merged_comparison_metrics(
    incoming: pd.DataFrame,
    *,
    output_dir: Path,
    experiment_dir: Path,
    spike_source: str | None = None,
) -> pd.DataFrame:
    """Merge this run into the dataset union and rewrite the run CSV.

    Returns the union (optionally filtered to ``spike_source``) that should be
    used for ``best_decoder_by_target`` and the run-level metrics file.
    """
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)
    prior = load_or_collect_union(experiment_dir)
    run_path = output_dir / "decoder_comparison_metrics.csv"
    if run_path.exists() and run_path.stat().st_size > 0:
        try:
            prior_run = pd.read_csv(run_path)
        except (OSError, pd.errors.EmptyDataError):
            prior_run = pd.DataFrame()
        if not prior_run.empty:
            prior = merge_comparison_metrics(prior, prior_run)
    union = merge_comparison_metrics(prior, incoming)
    union = ensure_config_ids(union)

    union_path = union_csv_path(experiment_dir)
    union_path.parent.mkdir(parents=True, exist_ok=True)
    union.to_csv(union_path, index=False)

    scoped = union
    if spike_source and "spike_source" in union.columns and not union.empty:
        matched = union[union["spike_source"].astype(str) == str(spike_source)]
        if not matched.empty:
            scoped = matched
    output_dir.mkdir(parents=True, exist_ok=True)
    scoped.to_csv(output_dir / "decoder_comparison_metrics.csv", index=False)
    return scoped.reset_index(drop=True)
