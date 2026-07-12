"""Causal Bayesian place decoder from population spike counts.

Fits Poisson rate maps over a spatial grid using training positions, then
decodes position via MAP/expected posterior. Optional exponential temporal
smoothing uses only past posteriors (never future spikes or future behavior).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from realtime.decoding_targets import (
    WALL_DISTANCE_MIDDLE_CM,
    WALL_DISTANCE_NEAR_CM,
    classify_wall_distance_bin,
    distance_to_wall,
)
from realtime.train_decoder import classify_spatial_context


class BayesianPlaceDecoder(BaseEstimator, RegressorMixin):
    """Decode (x, y) from spike counts with a Poisson Bayesian place model."""

    def __init__(
        self,
        n_bins: int = 20,
        occupancy_floor: float = 1e-3,
        rate_floor: float = 1e-3,
        smooth: bool = False,
        smooth_alpha: float = 0.7,
        arena_bounds: tuple[float, float, float, float] | None = None,
    ):
        self.n_bins = n_bins
        self.occupancy_floor = occupancy_floor
        self.rate_floor = rate_floor
        self.smooth = smooth
        self.smooth_alpha = smooth_alpha
        self.arena_bounds = arena_bounds

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "occupancy_floor": self.occupancy_floor,
            "rate_floor": self.rate_floor,
            "smooth": self.smooth,
            "smooth_alpha": self.smooth_alpha,
            "arena_bounds": self.arena_bounds,
        }

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("BayesianPlaceDecoder expects y with shape (n, 2) for (x, y)")

        if self.arena_bounds is None:
            pad = 1e-6
            self.x_min_ = float(y[:, 0].min() - pad)
            self.x_max_ = float(y[:, 0].max() + pad)
            self.y_min_ = float(y[:, 1].min() - pad)
            self.y_max_ = float(y[:, 1].max() + pad)
        else:
            self.x_min_, self.x_max_, self.y_min_, self.y_max_ = self.arena_bounds

        self.n_units_ = X.shape[1]
        self.x_edges_ = np.linspace(self.x_min_, self.x_max_, self.n_bins + 1)
        self.y_edges_ = np.linspace(self.y_min_, self.y_max_, self.n_bins + 1)
        self.x_centers_ = 0.5 * (self.x_edges_[:-1] + self.x_edges_[1:])
        self.y_centers_ = 0.5 * (self.y_edges_[:-1] + self.y_edges_[1:])

        bx = np.clip(np.digitize(y[:, 0], self.x_edges_) - 1, 0, self.n_bins - 1)
        by = np.clip(np.digitize(y[:, 1], self.y_edges_) - 1, 0, self.n_bins - 1)

        occupancy = np.zeros((self.n_bins, self.n_bins), dtype=float)
        spike_sums = np.zeros((self.n_bins, self.n_bins, self.n_units_), dtype=float)
        for i in range(len(X)):
            occupancy[bx[i], by[i]] += 1.0
            spike_sums[bx[i], by[i]] += X[i]

        occupancy = np.maximum(occupancy, self.occupancy_floor)
        self.rate_maps_ = np.maximum(spike_sums / occupancy[:, :, None], self.rate_floor)
        prior = occupancy / occupancy.sum()
        self.log_prior_ = np.log(np.maximum(prior, self.occupancy_floor))
        self._posterior_state_ = None
        return self

    def _log_likelihood(self, counts: np.ndarray) -> np.ndarray:
        # Poisson log-likelihood summed over units for each spatial bin.
        # log P(c|λ) = sum_u [c_u log λ_u - λ_u - log(c_u!)] ; factorial ignored (const)
        rates = self.rate_maps_
        log_rates = np.log(rates)
        # shape: (n_bins, n_bins)
        return (counts[None, None, :] * log_rates).sum(axis=2) - rates.sum(axis=2)

    def _posterior(self, counts: np.ndarray) -> np.ndarray:
        log_post = self._log_likelihood(counts) + self.log_prior_
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()
        return post

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = np.zeros((len(X), 2), dtype=float)
        state = None if not self.smooth else self._posterior_state_
        alpha = float(self.smooth_alpha)
        for i, counts in enumerate(X):
            post = self._posterior(counts)
            if self.smooth:
                if state is None:
                    state = post
                else:
                    state = alpha * post + (1.0 - alpha) * state
                    state /= state.sum()
                used = state
            else:
                used = post
            # Expected position under posterior
            px = used.sum(axis=1)
            py = used.sum(axis=0)
            preds[i, 0] = float(np.dot(px, self.x_centers_))
            preds[i, 1] = float(np.dot(py, self.y_centers_))
        if self.smooth:
            self._posterior_state_ = state
        return preds

    def reset_smoothing(self) -> None:
        self._posterior_state_ = None


class BayesianPlaceDerivedDecoder(BaseEstimator, ClassifierMixin):
    """Derive spatial categorical labels from a Bayesian place decoder."""

    def __init__(
        self,
        derived_target: str = "spatial_context",
        n_bins: int = 20,
        smooth: bool = False,
        smooth_alpha: float = 0.7,
        arena_bounds: tuple[float, float, float, float] | None = None,
        near_cm: float = WALL_DISTANCE_NEAR_CM,
        middle_cm: float = WALL_DISTANCE_MIDDLE_CM,
    ):
        self.derived_target = derived_target
        self.n_bins = n_bins
        self.smooth = smooth
        self.smooth_alpha = smooth_alpha
        self.arena_bounds = arena_bounds
        self.near_cm = near_cm
        self.middle_cm = middle_cm

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "derived_target": self.derived_target,
            "n_bins": self.n_bins,
            "smooth": self.smooth,
            "smooth_alpha": self.smooth_alpha,
            "arena_bounds": self.arena_bounds,
            "near_cm": self.near_cm,
            "middle_cm": self.middle_cm,
        }

    def fit(self, X: np.ndarray, y=None, *, position_xy: np.ndarray | None = None):
        if position_xy is None:
            raise ValueError(
                "BayesianPlaceDerivedDecoder.fit requires position_xy=(n,2) training positions"
            )
        self.place_ = BayesianPlaceDecoder(
            n_bins=self.n_bins,
            smooth=self.smooth,
            smooth_alpha=self.smooth_alpha,
            arena_bounds=self.arena_bounds,
        )
        self.place_.fit(X, position_xy)
        self.x_min_ = self.place_.x_min_
        self.x_max_ = self.place_.x_max_
        self.y_min_ = self.place_.y_min_
        self.y_max_ = self.place_.y_max_
        # Classes for sklearn compatibility
        if self.derived_target == "spatial_context":
            self.classes_ = np.array(["center", "wall", "corner"], dtype=object)
        elif self.derived_target == "wall_distance_bin":
            self.classes_ = np.array(["near_wall", "middle", "center"], dtype=object)
        else:
            raise ValueError(f"Unsupported derived_target: {self.derived_target}")
        return self

    def _labels_from_xy(self, xy: np.ndarray) -> np.ndarray:
        x = xy[:, 0]
        y = xy[:, 1]
        if self.derived_target == "spatial_context":
            return classify_spatial_context(
                x, y, self.x_min_, self.x_max_, self.y_min_, self.y_max_,
            )
        dist = distance_to_wall(x, y, self.x_min_, self.x_max_, self.y_min_, self.y_max_)
        return classify_wall_distance_bin(dist, self.near_cm, self.middle_cm)

    def predict(self, X: np.ndarray) -> np.ndarray:
        xy = self.place_.predict(X)
        return self._labels_from_xy(xy)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Soft one-hot from predicted label (no calibrated class probabilities).
        labels = self.predict(X)
        proba = np.zeros((len(labels), len(self.classes_)), dtype=float)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for i, lab in enumerate(labels):
            proba[i, class_to_idx[lab]] = 1.0
        return proba


class BayesianDistanceToWallDecoder(BaseEstimator, RegressorMixin):
    """Decode distance-to-wall via Bayesian place decoding then geometry."""

    def __init__(
        self,
        n_bins: int = 20,
        smooth: bool = False,
        smooth_alpha: float = 0.7,
        arena_bounds: tuple[float, float, float, float] | None = None,
    ):
        self.n_bins = n_bins
        self.smooth = smooth
        self.smooth_alpha = smooth_alpha
        self.arena_bounds = arena_bounds

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "smooth": self.smooth,
            "smooth_alpha": self.smooth_alpha,
            "arena_bounds": self.arena_bounds,
        }

    def fit(self, X: np.ndarray, y=None, *, position_xy: np.ndarray | None = None):
        if position_xy is None:
            raise ValueError("BayesianDistanceToWallDecoder.fit requires position_xy")
        self.place_ = BayesianPlaceDecoder(
            n_bins=self.n_bins,
            smooth=self.smooth,
            smooth_alpha=self.smooth_alpha,
            arena_bounds=self.arena_bounds,
        )
        self.place_.fit(X, position_xy)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        xy = self.place_.predict(X)
        dist = distance_to_wall(
            xy[:, 0], xy[:, 1],
            self.place_.x_min_, self.place_.x_max_,
            self.place_.y_min_, self.place_.y_max_,
        )
        return dist.reshape(-1, 1)
