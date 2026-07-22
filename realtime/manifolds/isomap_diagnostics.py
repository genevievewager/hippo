"""Graph-connectivity and neighborhood-quality diagnostics for Isomap."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


class DisconnectedGraphError(ValueError):
    """Raised when the Isomap neighbor graph fails connectivity requirements."""


def neighbor_graph_from_isomap(isomap) -> sparse.csr_matrix:
    """Return the symmetric neighbor adjacency used by a fitted sklearn Isomap."""
    if not hasattr(isomap, "nbrs_") or isomap.nbrs_ is None:
        raise RuntimeError("Isomap has no fitted nearest-neighbor graph")
    graph = isomap.nbrs_.kneighbors_graph(mode="connectivity")
    # Symmetrize: undirected geodesic graph
    graph = graph.maximum(graph.T)
    return sparse.csr_matrix(graph)


def compute_graph_diagnostics(
    isomap,
    X_train: np.ndarray,
    *,
    dense_degree_fraction_threshold: float = 0.25,
) -> dict[str, Any]:
    """Compute connectivity / degree diagnostics for a fitted Isomap model.

    Parameters
    ----------
    isomap:
        Fitted ``sklearn.manifold.Isomap`` instance.
    X_train:
        Training observations used to fit Isomap, shape ``[n_samples, n_features]``.
    dense_degree_fraction_threshold:
        Flag graphs whose mean degree exceeds this fraction of ``n_samples``.
    """
    X_train = np.asarray(X_train, dtype=float)
    n_samples = int(X_train.shape[0])
    graph = neighbor_graph_from_isomap(isomap)
    n_components, labels = connected_components(graph, directed=False)

    counts = np.bincount(labels, minlength=n_components)
    largest_label = int(np.argmax(counts)) if n_components > 0 else 0
    largest_size = int(counts[largest_label]) if n_components > 0 else 0
    largest_frac = float(largest_size / max(n_samples, 1))

    degrees = np.asarray(graph.sum(axis=1)).ravel()
    # Duplicate / near-duplicate population vectors inflate neighborhood density.
    # Hash rounded rows for a cheap duplicate fraction estimate.
    rounded = np.round(X_train, decimals=6)
    _, unique_counts = np.unique(rounded, axis=0, return_counts=True)
    n_duplicate_rows = int(np.sum(unique_counts[unique_counts > 1]))
    duplicate_fraction = float(n_duplicate_rows / max(n_samples, 1))

    geo = getattr(isomap, "dist_matrix_", None)
    if geo is not None:
        finite_frac = float(np.isfinite(geo).mean())
    else:
        finite_frac = float("nan")

    mean_degree = float(np.mean(degrees)) if len(degrees) else 0.0
    degree_fraction = mean_degree / max(n_samples, 1)
    dense_flag = bool(degree_fraction >= dense_degree_fraction_threshold)

    return {
        "n_connected_components": int(n_components),
        "graph_connected": bool(n_components == 1),
        "largest_component_label": largest_label,
        "largest_component_size": largest_size,
        "largest_component_fraction": largest_frac,
        "min_node_degree": int(np.min(degrees)) if len(degrees) else 0,
        "median_node_degree": float(np.median(degrees)) if len(degrees) else 0.0,
        "max_node_degree": int(np.max(degrees)) if len(degrees) else 0,
        "mean_node_degree": mean_degree,
        "mean_degree_over_n_samples": float(degree_fraction),
        "dense_graph_flag": dense_flag,
        "fraction_duplicate_observations": duplicate_fraction,
        "geodesic_distance_finite_fraction": finite_frac,
        "n_train_samples": n_samples,
        "component_labels": labels,
    }


def validate_graph_connectivity(
    diagnostics: dict[str, Any],
    *,
    require_connected_graph: bool = True,
    allow_largest_component_only: bool = False,
    minimum_largest_component_fraction: float = 0.95,
) -> tuple[bool, str | None]:
    """Return ``(accepted, exclusion_reason)`` for a fitted Isomap graph.

    When ``require_connected_graph`` is True and the graph is disconnected:
    - reject unless ``allow_largest_component_only`` and the largest component
      covers at least ``minimum_largest_component_fraction`` of training samples.
    """
    connected = bool(diagnostics.get("graph_connected", False))
    largest_frac = float(diagnostics.get("largest_component_fraction", 0.0))

    if connected:
        return True, None

    if not require_connected_graph:
        return True, None

    if allow_largest_component_only and largest_frac >= minimum_largest_component_fraction:
        return True, None

    reason = (
        f"disconnected_isomap_graph: n_components="
        f"{diagnostics.get('n_connected_components')}, "
        f"largest_component_fraction={largest_frac:.3f}. "
        "Increase n_neighbors or enable allow_largest_component_only."
    )
    return False, reason
