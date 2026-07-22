"""Causal neural-manifold feature transformers for realtime decoding.

Manifold transforms are fit only on the training portion of a session and then
applied to held-out / realtime spike-count vectors. At test time the transform
is frozen — it never uses future spikes or future behavior.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

# Feature modes understood by the decoder comparison / realtime pipeline.
IDENTITY_FEATURE_MODES = ("counts", "rates")
MANIFOLD_FEATURE_MODES = (
    "global_pca",
    "region_pca",
    "layer_pca",
    "cell_type_pca",
    "rate_model_pca",
    "global_isomap",
    "global_isomap_distilled",
)
ALL_FEATURE_MODES = IDENTITY_FEATURE_MODES + MANIFOLD_FEATURE_MODES

# Offline-only static modes: evaluated in comparison, not auto-deployed realtime.
# Distilled Isomap is realtime-eligible when latency/distortion gates pass.
OFFLINE_ONLY_FEATURE_MODES = frozenset({"global_isomap"})

QUICK_FEATURE_MODES = ("counts", "global_pca", "region_pca")
FULL_FEATURE_MODES = ALL_FEATURE_MODES

GROUPING_COLUMN = {
    "global_pca": None,
    "global_isomap": None,
    "global_isomap_distilled": None,
    "region_pca": "region",
    "layer_pca": "layer",
    "cell_type_pca": "cell_type",
    "rate_model_pca": "rate_model",  # falls back to ratinabox_class
}

DEFAULT_ISOMAP_N_NEIGHBORS = 10


def is_manifold_feature_mode(feature_mode: str) -> bool:
    return feature_mode in MANIFOLD_FEATURE_MODES


def resolve_feature_modes(
    feature_modes: tuple[str, ...] | list[str] | None,
    *,
    max_models: str = "quick",
) -> tuple[str, ...]:
    """Resolve feature-mode list; apply quick/full defaults when empty."""
    if feature_modes:
        return tuple(dict.fromkeys(feature_modes))
    if max_models == "full":
        return FULL_FEATURE_MODES
    return QUICK_FEATURE_MODES


def grouping_for_feature_mode(feature_mode: str) -> str | None:
    return GROUPING_COLUMN.get(feature_mode)


def manifold_type_for_feature_mode(feature_mode: str) -> str:
    if feature_mode in IDENTITY_FEATURE_MODES:
        return "none"
    if feature_mode.endswith("_pca") or feature_mode == "global_pca":
        return "pca"
    if feature_mode == "global_isomap_distilled" or feature_mode.endswith(
        "_isomap_distilled"
    ):
        return "isomap_distilled"
    if feature_mode.endswith("_isomap") or feature_mode == "global_isomap":
        return "isomap"
    return feature_mode


def is_realtime_compatible_feature_mode(feature_mode: str) -> bool:
    """Return False for offline-only modes such as standard Isomap."""
    return feature_mode not in OFFLINE_ONLY_FEATURE_MODES


class IdentityFeatures(BaseEstimator, TransformerMixin):
    """Pass-through feature mode for raw counts or rates."""

    def __init__(self, feature_mode: str = "counts", decode_window: float = 0.250):
        if feature_mode not in IDENTITY_FEATURE_MODES:
            raise ValueError(f"IdentityFeatures only supports {IDENTITY_FEATURE_MODES}")
        self.feature_mode = feature_mode
        self.decode_window = float(decode_window)
        self.n_features_in_: int | None = None
        self.n_features_out_: int | None = None

    def fit(self, X: np.ndarray, y: Any = None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = int(X.shape[1])
        self.n_features_out_ = self.n_features_in_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.feature_mode == "rates":
            return X / self.decode_window
        return X

    def get_metadata(self) -> dict[str, Any]:
        return {
            "manifold_type": "none",
            "manifold_grouping": None,
            "manifold_n_components": None,
            "actual_n_features": self.n_features_out_,
            "explained_variance_ratio": None,
            "groups": [],
        }

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "meta.json", "w") as f:
            json.dump({
                "class_name": "IdentityFeatures",
                "feature_mode": self.feature_mode,
                "decode_window": self.decode_window,
                "n_features_in": self.n_features_in_,
                "n_features_out": self.n_features_out_,
            }, f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> IdentityFeatures:
        with open(Path(input_dir) / "meta.json") as f:
            meta = json.load(f)
        obj = cls(feature_mode=meta["feature_mode"], decode_window=meta["decode_window"])
        obj.n_features_in_ = meta.get("n_features_in")
        obj.n_features_out_ = meta.get("n_features_out")
        return obj


class GlobalPCAManifold(BaseEstimator, TransformerMixin):
    """Fit PCA on all units together."""

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = int(n_components)
        self.random_state = random_state
        self.pca_: PCA | None = None
        self.actual_n_components_: int | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None):
        X = np.asarray(X, dtype=float)
        n_samples, n_units = X.shape
        n_comp = min(self.n_components, n_units, max(1, n_samples - 1))
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca_.fit(X)
        self.actual_n_components_ = int(self.pca_.n_components_)
        self.explained_variance_ratio_ = np.asarray(self.pca_.explained_variance_ratio_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("GlobalPCAManifold must be fit before transform")
        return self.pca_.transform(np.asarray(X, dtype=float))

    def get_metadata(self) -> dict[str, Any]:
        return {
            "manifold_type": "pca",
            "manifold_grouping": None,
            "manifold_n_components": self.n_components,
            "actual_n_features": self.actual_n_components_,
            "explained_variance_ratio": (
                self.explained_variance_ratio_.tolist()
                if self.explained_variance_ratio_ is not None else None
            ),
            "explained_variance_sum": (
                float(np.sum(self.explained_variance_ratio_))
                if self.explained_variance_ratio_ is not None else None
            ),
            "groups": [{
                "group_name": "all",
                "n_units": int(self.pca_.n_features_in_) if self.pca_ is not None else None,
                "n_components": self.actual_n_components_,
                "explained_variance_sum": (
                    float(np.sum(self.explained_variance_ratio_))
                    if self.explained_variance_ratio_ is not None else None
                ),
                "explained_variance_by_component": (
                    self.explained_variance_ratio_.tolist()
                    if self.explained_variance_ratio_ is not None else None
                ),
            }],
        }

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pca_, output_dir / "pca.joblib")
        with open(output_dir / "meta.json", "w") as f:
            json.dump({
                "class_name": "GlobalPCAManifold",
                "n_components": self.n_components,
                "actual_n_components": self.actual_n_components_,
                "random_state": self.random_state,
                "explained_variance_ratio": (
                    self.explained_variance_ratio_.tolist()
                    if self.explained_variance_ratio_ is not None else None
                ),
            }, f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> GlobalPCAManifold:
        input_dir = Path(input_dir)
        with open(input_dir / "meta.json") as f:
            meta = json.load(f)
        obj = cls(
            n_components=meta["n_components"],
            random_state=meta.get("random_state", 42),
        )
        obj.pca_ = joblib.load(input_dir / "pca.joblib")
        obj.actual_n_components_ = meta.get("actual_n_components")
        ev = meta.get("explained_variance_ratio")
        obj.explained_variance_ratio_ = np.asarray(ev) if ev is not None else None
        return obj


class GroupwisePCAManifold(BaseEstimator, TransformerMixin):
    """Fit separate PCA models per unit group and concatenate latents."""

    def __init__(
        self,
        group_labels: list[str] | np.ndarray,
        n_components: int = 3,
        grouping_name: str = "region",
        random_state: int = 42,
    ):
        self.group_labels = [str(g) for g in group_labels]
        self.n_components = int(n_components)
        self.grouping_name = grouping_name
        self.random_state = random_state
        self.group_order_: list[str] = []
        self.group_indices_: dict[str, np.ndarray] = {}
        self.group_pcas_: dict[str, PCA] = {}
        self.group_n_components_: dict[str, int] = {}
        self.group_explained_: dict[str, np.ndarray] = {}
        self.actual_n_features_: int | None = None

    def fit(self, X: np.ndarray, y: Any = None):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != len(self.group_labels):
            raise ValueError(
                f"X has {X.shape[1]} units but group_labels has {len(self.group_labels)}"
            )
        n_samples = X.shape[0]
        labels = np.asarray(self.group_labels)
        self.group_order_ = sorted(set(labels.tolist()))
        self.group_indices_ = {}
        self.group_pcas_ = {}
        self.group_n_components_ = {}
        self.group_explained_ = {}
        total = 0
        for group in self.group_order_:
            idx = np.where(labels == group)[0]
            if len(idx) == 0:
                continue
            self.group_indices_[group] = idx
            n_comp = min(self.n_components, len(idx), max(1, n_samples - 1))
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            pca.fit(X[:, idx])
            self.group_pcas_[group] = pca
            self.group_n_components_[group] = int(pca.n_components_)
            self.group_explained_[group] = np.asarray(pca.explained_variance_ratio_)
            total += int(pca.n_components_)
        self.actual_n_features_ = total
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.group_pcas_:
            raise RuntimeError("GroupwisePCAManifold must be fit before transform")
        X = np.asarray(X, dtype=float)
        parts = []
        for group in self.group_order_:
            if group not in self.group_pcas_:
                continue
            idx = self.group_indices_[group]
            parts.append(self.group_pcas_[group].transform(X[:, idx]))
        if not parts:
            raise RuntimeError("No group PCA models available")
        return np.concatenate(parts, axis=1)

    def get_metadata(self) -> dict[str, Any]:
        groups = []
        for group in self.group_order_:
            if group not in self.group_pcas_:
                continue
            ev = self.group_explained_[group]
            groups.append({
                "group_name": group,
                "n_units": int(len(self.group_indices_[group])),
                "n_components": self.group_n_components_[group],
                "explained_variance_sum": float(np.sum(ev)),
                "explained_variance_by_component": ev.tolist(),
            })
        return {
            "manifold_type": "pca",
            "manifold_grouping": self.grouping_name,
            "manifold_n_components": self.n_components,
            "actual_n_features": self.actual_n_features_,
            "explained_variance_ratio": None,
            "explained_variance_sum": (
                float(np.mean([g["explained_variance_sum"] for g in groups]))
                if groups else None
            ),
            "groups": groups,
        }

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "group_labels": self.group_labels,
            "group_order": self.group_order_,
            "group_indices": self.group_indices_,
            "group_pcas": self.group_pcas_,
            "group_n_components": self.group_n_components_,
            "group_explained": self.group_explained_,
            "actual_n_features": self.actual_n_features_,
            "n_components": self.n_components,
            "grouping_name": self.grouping_name,
            "random_state": self.random_state,
        }, output_dir / "groupwise_pca.joblib")
        with open(output_dir / "meta.json", "w") as f:
            json.dump({
                "class_name": "GroupwisePCAManifold",
                "grouping_name": self.grouping_name,
                "n_components": self.n_components,
                "actual_n_features": self.actual_n_features_,
                "group_n_components": self.group_n_components_,
                "random_state": self.random_state,
            }, f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> GroupwisePCAManifold:
        payload = joblib.load(Path(input_dir) / "groupwise_pca.joblib")
        obj = cls(
            group_labels=payload["group_labels"],
            n_components=payload["n_components"],
            grouping_name=payload["grouping_name"],
            random_state=payload.get("random_state", 42),
        )
        obj.group_order_ = payload["group_order"]
        obj.group_indices_ = payload["group_indices"]
        obj.group_pcas_ = payload["group_pcas"]
        obj.group_n_components_ = payload["group_n_components"]
        obj.group_explained_ = payload["group_explained"]
        obj.actual_n_features_ = payload["actual_n_features"]
        return obj


class UMAPManifold(BaseEstimator, TransformerMixin):
    """Optional nonlinear embedding; skipped if umap-learn is unavailable."""

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = int(n_components)
        self.random_state = random_state
        self._umap = None
        self.available_ = False

    def fit(self, X: np.ndarray, y: Any = None):
        try:
            import umap  # type: ignore
        except ImportError:
            warnings.warn("umap-learn not installed; UMAPManifold unavailable", stacklevel=2)
            self.available_ = False
            return self
        self._umap = umap.UMAP(
            n_components=self.n_components, random_state=self.random_state,
        )
        self._umap.fit(np.asarray(X, dtype=float))
        self.available_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.available_ or self._umap is None:
            raise RuntimeError("UMAPManifold is unavailable")
        return self._umap.transform(np.asarray(X, dtype=float))


class IsomapManifold(BaseEstimator, TransformerMixin):
    """Static-decoding Isomap wrapper (offline analysis only).

    Delegates to ``IsomapManifoldEncoder`` with training-only preprocessing
    (sqrt counts + standardization + optional PCA precompression). Tagged
    ``realtime_compatible=False`` — do not auto-deploy into closed-loop replay.
    """

    realtime_compatible = False
    deployment_tag = "offline_analysis_only"

    def __init__(
        self,
        n_components: int = 3,
        n_neighbors: int = DEFAULT_ISOMAP_N_NEIGHBORS,
        *,
        transform: str = "sqrt_counts",
        standardize: bool = True,
        pre_pca_enabled: bool = True,
        pre_pca_n_components: int = 50,
        require_connected_graph: bool = True,
        allow_largest_component_only: bool = False,
        minimum_largest_component_fraction: float = 0.95,
        n_jobs: int | None = -1,
        random_state: int = 42,
    ):
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.transform_name = transform
        self.standardize = bool(standardize)
        self.pre_pca_enabled = bool(pre_pca_enabled)
        self.pre_pca_n_components = int(pre_pca_n_components)
        self.require_connected_graph = bool(require_connected_graph)
        self.allow_largest_component_only = bool(allow_largest_component_only)
        self.minimum_largest_component_fraction = float(
            minimum_largest_component_fraction
        )
        self.n_jobs = n_jobs
        self.random_state = int(random_state)
        self._encoder = None

    def fit(self, X: np.ndarray, y: Any = None):
        from realtime.manifolds.isomap import IsomapManifoldEncoder

        self._encoder = IsomapManifoldEncoder(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            transform=self.transform_name,
            standardize=self.standardize,
            pre_pca_enabled=self.pre_pca_enabled,
            pre_pca_n_components=self.pre_pca_n_components,
            require_connected_graph=self.require_connected_graph,
            allow_largest_component_only=self.allow_largest_component_only,
            minimum_largest_component_fraction=self.minimum_largest_component_fraction,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._encoder.fit(np.asarray(X, dtype=float))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("IsomapManifold must be fit before transform")
        return self._encoder.transform(np.asarray(X, dtype=float))

    def get_metadata(self) -> dict[str, Any]:
        if self._encoder is None:
            return {
                "manifold_type": "isomap",
                "manifold_grouping": None,
                "manifold_n_components": self.n_components,
                "actual_n_features": None,
                "n_neighbors": self.n_neighbors,
                "realtime_compatible": self.realtime_compatible,
                "deployment_tag": self.deployment_tag,
                "explained_variance_ratio": None,
                "groups": [],
            }
        return self._encoder.get_metadata()

    def save(self, output_dir: Path) -> None:
        if self._encoder is None:
            raise RuntimeError("IsomapManifold must be fit before save")
        output_dir = Path(output_dir)
        self._encoder.save(output_dir)
        # Overlay static class_name for load_feature_transformer dispatch.
        with open(output_dir / "meta.json") as f:
            meta = json.load(f)
        meta["class_name"] = "IsomapManifold"
        meta["feature_mode"] = "global_isomap"
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> IsomapManifold:
        from realtime.manifolds.isomap import IsomapManifoldEncoder

        enc = IsomapManifoldEncoder.load(input_dir)
        obj = cls(
            n_components=enc.n_components,
            n_neighbors=enc.n_neighbors,
            transform=enc.transform_name,
            standardize=enc.standardize,
            pre_pca_enabled=enc.pre_pca_enabled,
            pre_pca_n_components=enc.pre_pca_n_components,
            require_connected_graph=enc.require_connected_graph,
            allow_largest_component_only=enc.allow_largest_component_only,
            minimum_largest_component_fraction=enc.minimum_largest_component_fraction,
            n_jobs=enc.n_jobs,
            random_state=enc.random_state,
        )
        obj._encoder = enc
        return obj


def _group_labels_from_units(
    units_df: pd.DataFrame,
    unit_ids: list[int] | np.ndarray,
    grouping_col: str,
) -> list[str] | None:
    """Return group label per unit_id column order, or None if column missing."""
    units_df = units_df.copy()
    col = grouping_col
    if col not in units_df.columns:
        if grouping_col == "rate_model" and "ratinabox_class" in units_df.columns:
            col = "ratinabox_class"
        else:
            return None
    unit_ids = list(unit_ids)
    indexed = units_df.set_index("unit_id")
    labels = []
    for uid in unit_ids:
        if uid not in indexed.index:
            labels.append("unknown")
        else:
            labels.append(str(indexed.loc[uid, col]))
    return labels


def make_feature_transformer(
    feature_mode: str,
    *,
    decode_window: float,
    n_components: int = 3,
    units_df: pd.DataFrame | None = None,
    unit_ids: list[int] | np.ndarray | None = None,
    random_state: int = 42,
    n_neighbors: int = DEFAULT_ISOMAP_N_NEIGHBORS,
    isomap_pre_pca_enabled: bool = True,
    isomap_pre_pca_n_components: int = 50,
    isomap_require_connected_graph: bool = True,
    n_jobs: int | None = -1,
) -> Any:
    """
    Construct an unfitted feature transformer for a feature mode.

    Returns None when a requested groupwise mode cannot be built (missing metadata).
    """
    if feature_mode in IDENTITY_FEATURE_MODES:
        return IdentityFeatures(feature_mode=feature_mode, decode_window=decode_window)

    if feature_mode == "global_pca":
        return GlobalPCAManifold(n_components=n_components, random_state=random_state)

    if feature_mode == "global_isomap":
        return IsomapManifold(
            n_components=n_components,
            n_neighbors=n_neighbors,
            pre_pca_enabled=isomap_pre_pca_enabled,
            pre_pca_n_components=isomap_pre_pca_n_components,
            require_connected_graph=isomap_require_connected_graph,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    if feature_mode == "global_isomap_distilled":
        from realtime.manifolds.isomap_distilled_features import IsomapDistilledManifold

        return IsomapDistilledManifold(
            n_components=n_components,
            n_neighbors=n_neighbors,
            pre_pca_enabled=isomap_pre_pca_enabled,
            pre_pca_n_components=isomap_pre_pca_n_components,
            require_connected_graph=isomap_require_connected_graph,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    grouping = grouping_for_feature_mode(feature_mode)
    if grouping is None:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    if units_df is None or unit_ids is None:
        raise ValueError(f"{feature_mode} requires units_df and unit_ids")
    labels = _group_labels_from_units(units_df, unit_ids, grouping)
    if labels is None:
        warnings.warn(
            f"Skipping feature mode {feature_mode!r}: units.csv missing "
            f"grouping column {grouping!r}",
            stacklevel=2,
        )
        return None
    return GroupwisePCAManifold(
        group_labels=labels,
        n_components=n_components,
        grouping_name=grouping,
        random_state=random_state,
    )


def load_feature_transformer(input_dir: Path) -> Any:
    """Load a saved Identity / GlobalPCA / GroupwisePCA / Isomap transformer."""
    input_dir = Path(input_dir)
    with open(input_dir / "meta.json") as f:
        meta = json.load(f)
    name = meta.get("class_name")
    if name == "IdentityFeatures":
        return IdentityFeatures.load(input_dir)
    if name == "GlobalPCAManifold":
        return GlobalPCAManifold.load(input_dir)
    if name == "GroupwisePCAManifold":
        return GroupwisePCAManifold.load(input_dir)
    if name in ("IsomapManifold", "IsomapManifoldEncoder"):
        return IsomapManifold.load(input_dir)
    if name == "IsomapDistilledManifold":
        from realtime.manifolds.isomap_distilled_features import IsomapDistilledManifold

        return IsomapDistilledManifold.load(input_dir)
    raise ValueError(f"Unknown manifold class in {input_dir}: {name}")


def manifold_transform_dirname(
    feature_mode: str,
    decode_window: float,
    n_components: int | None,
    n_neighbors: int | None = None,
) -> str:
    """Stable directory name for a fitted manifold transform."""
    w_ms = int(round(float(decode_window) * 1000))
    if n_components is None:
        return f"{feature_mode}_w{w_ms:04d}ms"
    if n_neighbors is not None and (
        feature_mode in ("global_isomap", "global_isomap_distilled")
        or feature_mode.endswith("_isomap")
        or feature_mode.endswith("_isomap_distilled")
    ):
        return f"{feature_mode}_k{int(n_components)}_nn{int(n_neighbors)}_w{w_ms:04d}ms"
    return f"{feature_mode}_k{int(n_components)}_w{w_ms:04d}ms"
