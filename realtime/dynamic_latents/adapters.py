"""Sklearn-compatible wrappers that plug dynamic latents into F×E search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from realtime.dynamic_latents.gpfa import GPFAModel
from realtime.dynamic_latents.lds import LinearDynamicalSystem
from realtime.dynamic_latents.registry import is_dynamic_latent, make_dynamic_latent


class DynamicLatentEmbedding(BaseEstimator, TransformerMixin):
    """Adapter so dynamic models share the manifold_features save/load API.

    For decoder comparison, call :meth:`transform` with ``reset=False`` on the
    test partition after transforming train (warm-start / no future leakage).
    """

    def __init__(
        self,
        model_name: str = "global_lds",
        n_components: int = 5,
        *,
        update_dt: float = 0.025,
        decode_window: float | None = None,
        feature_set: str | None = None,
        spike_source: str | None = None,
        random_state: int = 42,
        causal_default: bool = True,
        **model_kwargs: Any,
    ):
        if not is_dynamic_latent(model_name):
            raise ValueError(f"Unknown dynamic latent embedding: {model_name}")
        self.model_name = str(model_name)
        self.n_components = int(n_components)
        self.update_dt = float(update_dt)
        self.decode_window = decode_window
        self.feature_set = feature_set
        self.spike_source = spike_source
        self.random_state = int(random_state)
        self.causal_default = bool(causal_default)
        self.model_kwargs = dict(model_kwargs)
        self.model_: Any | None = None

    @property
    def supports_realtime(self) -> bool:
        if self.model_ is not None:
            return bool(self.model_.supports_realtime)
        from realtime.dynamic_latents.registry import DYNAMIC_LATENT_REGISTRY

        return bool(getattr(DYNAMIC_LATENT_REGISTRY[self.model_name], "supports_realtime", False))

    @property
    def realtime_compatible(self) -> bool:
        return self.supports_realtime

    def fit(self, X: np.ndarray, y: Any = None):
        kwargs = {
            "n_components": self.n_components,
            "update_dt": self.update_dt,
            "decode_window": self.decode_window,
            "feature_set": self.feature_set,
            "spike_source": self.spike_source,
            "random_state": self.random_state,
            **self.model_kwargs,
        }
        self.model_ = make_dynamic_latent(self.model_name, **kwargs)
        self.model_.fit(np.asarray(X, dtype=float))
        return self

    def transform(
        self,
        X: np.ndarray,
        *,
        causal: bool | None = None,
        reset: bool = True,
    ) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("DynamicLatentEmbedding must be fit before transform")
        use_causal = self.causal_default if causal is None else bool(causal)
        # GPFA ignores causal_default and prefers smoothed offline.
        if self.model_name == "gpfa" and causal is None:
            use_causal = False
        return self.model_.transform(
            np.asarray(X, dtype=float),
            causal=use_causal,
            reset=reset,
        )

    def step(self, x_t: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("DynamicLatentEmbedding must be fit before step")
        return self.model_.step(x_t)

    def reset_state(self) -> None:
        if self.model_ is not None:
            self.model_.reset_state()

    def reconstruct(self, Z: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("DynamicLatentEmbedding must be fit before reconstruct")
        return self.model_.reconstruct(Z)

    def get_metadata(self) -> dict[str, Any]:
        if self.model_ is None:
            return {
                "manifold_type": self.model_name,
                "representation_family": "dynamic",
                "manifold_n_components": self.n_components,
                "actual_n_features": None,
                "realtime_compatible": self.supports_realtime,
                "supports_realtime": self.supports_realtime,
                "supports_causal_transform": self.model_name == "global_lds",
                "groups": [],
            }
        return self.model_.get_metadata()

    def save(self, output_dir: Path) -> None:
        if self.model_ is None:
            raise RuntimeError("DynamicLatentEmbedding must be fit before save")
        output_dir = Path(output_dir)
        self.model_.save(output_dir)
        # Overlay adapter class_name for load_feature_transformer dispatch.
        with open(output_dir / "meta.json") as f:
            meta = json.load(f)
        meta["class_name"] = "DynamicLatentEmbedding"
        meta["model_name"] = self.model_name
        meta["n_components"] = self.n_components
        meta["update_dt"] = self.update_dt
        meta["decode_window"] = self.decode_window
        meta["feature_set"] = self.feature_set
        meta["spike_source"] = self.spike_source
        meta["random_state"] = self.random_state
        meta["causal_default"] = self.causal_default
        meta["inner_class_name"] = type(self.model_).__name__
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load(cls, input_dir: Path) -> DynamicLatentEmbedding:
        input_dir = Path(input_dir)
        with open(input_dir / "meta.json") as f:
            meta = json.load(f)
        name = meta.get("model_name") or meta.get("model_type") or "global_lds"
        obj = cls(
            model_name=name,
            n_components=meta.get("n_components", meta.get("latent_dimension", 5)),
            update_dt=meta.get("update_dt", 0.025),
            decode_window=meta.get("decode_window"),
            feature_set=meta.get("feature_set"),
            spike_source=meta.get("spike_source"),
            random_state=meta.get("random_state", 42),
            causal_default=meta.get("causal_default", name != "gpfa"),
        )
        inner = meta.get("inner_class_name") or meta.get("class_name")
        if name == "gpfa" or inner == "GPFAModel":
            obj.model_ = GPFAModel.load(input_dir)
        else:
            obj.model_ = LinearDynamicalSystem.load(input_dir)
        return obj
