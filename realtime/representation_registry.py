"""Map existing representation *implementations* onto scientific quadrants.

A quadrant is a scientific category (linearity × temporal type). A method is
one implementation of that category. This registry does **not** duplicate
algorithms: ``make_feature_transformer`` / dynamic-latent classes remain the
authoritative computation.

Canonical quadrant ids::

    linear_static | nonlinear_static | linear_dynamic | nonlinear_dynamic

Legacy UI ids (``static_linear``, …) are accepted as aliases so existing
pages keep working. Unknown names infer ``quadrant="unknown"`` rather than
guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from realtime.dynamic_latents.registry import (
    DYNAMIC_LATENT_REGISTRY,
    FUTURE_DYNAMIC_LATENTS,
    dynamic_latent_capabilities,
    is_dynamic_latent,
)
from realtime.manifold_features import (
    OFFLINE_ONLY_FEATURE_MODES,
    is_realtime_compatible_feature_mode,
)
from realtime.search_space import ALL_EMBEDDING_TYPES, resolve_manifold_alias

LINEARITY_LINEAR = "linear"
LINEARITY_NONLINEAR = "nonlinear"
TEMPORAL_STATIC = "static"
TEMPORAL_DYNAMIC = "dynamic"

QUADRANT_LINEAR_STATIC = "linear_static"
QUADRANT_NONLINEAR_STATIC = "nonlinear_static"
QUADRANT_LINEAR_DYNAMIC = "linear_dynamic"
QUADRANT_NONLINEAR_DYNAMIC = "nonlinear_dynamic"
QUADRANT_UNKNOWN = "unknown"

CANONICAL_QUADRANTS: tuple[str, ...] = (
    QUADRANT_LINEAR_STATIC,
    QUADRANT_NONLINEAR_STATIC,
    QUADRANT_LINEAR_DYNAMIC,
    QUADRANT_NONLINEAR_DYNAMIC,
)

# Legacy UI / publication-plot ids → canonical scientific ids.
LEGACY_QUADRANT_TO_CANONICAL: dict[str, str] = {
    "static_linear": QUADRANT_LINEAR_STATIC,
    "static_nonlinear": QUADRANT_NONLINEAR_STATIC,
    "dynamic_linear": QUADRANT_LINEAR_DYNAMIC,
    "dynamic_nonlinear": QUADRANT_NONLINEAR_DYNAMIC,
}

CANONICAL_TO_LEGACY_QUADRANT: dict[str, str] = {
    v: k for k, v in LEGACY_QUADRANT_TO_CANONICAL.items()
}

QUADRANT_DISPLAY_LABELS: dict[str, str] = {
    QUADRANT_LINEAR_STATIC: "Linear + Static",
    QUADRANT_NONLINEAR_STATIC: "Nonlinear + Static",
    QUADRANT_LINEAR_DYNAMIC: "Linear + Dynamic",
    QUADRANT_NONLINEAR_DYNAMIC: "Nonlinear + Dynamic",
    QUADRANT_UNKNOWN: "Unknown",
}

# Default representative used in a *controlled* four-quadrant experiment.
# Nonlinear-dynamic has no valid implementation yet — do not fabricate one.
DEFAULT_QUADRANT_METHODS: dict[str, str | None] = {
    QUADRANT_LINEAR_STATIC: "global_pca",
    QUADRANT_NONLINEAR_STATIC: "diffusion_nystrom",
    QUADRANT_LINEAR_DYNAMIC: "global_lds",
    QUADRANT_NONLINEAR_DYNAMIC: None,
}


@dataclass(frozen=True)
class RepresentationSpec:
    """Metadata for one representation implementation (not a new algorithm)."""

    representation_name: str
    linearity: str
    temporal_type: str
    quadrant: str
    realtime_capable: bool
    offline_only: bool
    display_name: str
    implemented: bool = True
    description: str = ""
    notes: str = ""
    aliases: tuple[str, ...] = ()
    requires_anatomy: bool = False
    requires_simulation_rates: bool = False
    supervised: bool = False
    scientifically_ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation_name": self.representation_name,
            "linearity": self.linearity,
            "temporal_type": self.temporal_type,
            "quadrant": self.quadrant,
            "realtime_capable": self.realtime_capable,
            "offline_only": self.offline_only,
            "display_name": self.display_name,
            "implemented": self.implemented,
            "description": self.description,
            "notes": self.notes,
            "aliases": list(self.aliases),
            "requires_anatomy": self.requires_anatomy,
            "requires_simulation_rates": self.requires_simulation_rates,
            "supervised": self.supervised,
            "scientifically_ambiguous": self.scientifically_ambiguous,
        }


def _static_rt(name: str) -> tuple[bool, bool]:
    """Return (realtime_capable, offline_only) for a static embedding."""
    emb = resolve_manifold_alias(name)
    if emb == "global_isomap":
        return False, True
    rt = is_realtime_compatible_feature_mode(emb) and emb not in OFFLINE_ONLY_FEATURE_MODES
    return bool(rt), not bool(rt)


def _dynamic_rt(name: str) -> tuple[bool, bool]:
    if name not in DYNAMIC_LATENT_REGISTRY:
        return False, True
    caps = dynamic_latent_capabilities(name)
    rt = bool(caps["supports_realtime"])
    return rt, not rt


def _spec(
    name: str,
    *,
    linearity: str,
    temporal_type: str,
    display_name: str,
    description: str = "",
    notes: str = "",
    aliases: tuple[str, ...] = (),
    implemented: bool = True,
    requires_anatomy: bool = False,
    requires_simulation_rates: bool = False,
    supervised: bool = False,
    scientifically_ambiguous: bool = False,
    realtime_capable: bool | None = None,
    offline_only: bool | None = None,
) -> RepresentationSpec:
    if temporal_type == TEMPORAL_DYNAMIC and implemented and name in DYNAMIC_LATENT_REGISTRY:
        rt, off = _dynamic_rt(name)
    elif implemented and temporal_type == TEMPORAL_STATIC:
        rt, off = _static_rt(name)
    else:
        rt, off = False, True
    if realtime_capable is not None:
        rt = bool(realtime_capable)
    if offline_only is not None:
        off = bool(offline_only)
    quadrant = f"{linearity}_{temporal_type}" if linearity in {
        LINEARITY_LINEAR, LINEARITY_NONLINEAR,
    } and temporal_type in {TEMPORAL_STATIC, TEMPORAL_DYNAMIC} else QUADRANT_UNKNOWN
    if scientifically_ambiguous and not implemented:
        quadrant = QUADRANT_UNKNOWN
    return RepresentationSpec(
        representation_name=name,
        linearity=linearity,
        temporal_type=temporal_type,
        quadrant=quadrant,
        realtime_capable=rt,
        offline_only=off,
        display_name=display_name,
        implemented=implemented,
        description=description,
        notes=notes,
        aliases=aliases,
        requires_anatomy=requires_anatomy,
        requires_simulation_rates=requires_simulation_rates,
        supervised=supervised,
        scientifically_ambiguous=scientifically_ambiguous,
    )


_UNKNOWN_SPEC = RepresentationSpec(
    representation_name="",
    linearity="",
    temporal_type="",
    quadrant=QUADRANT_UNKNOWN,
    realtime_capable=False,
    offline_only=True,
    display_name="Unknown",
    implemented=False,
    description="Quadrant could not be inferred safely from the representation name.",
    notes="Older artifacts without a known embedding name are marked unknown rather than guessed.",
)


# Implemented methods. Computation stays in existing encoder / latent classes.
_IMPLEMENTED: tuple[RepresentationSpec, ...] = (
    _spec(
        "identity",
        linearity=LINEARITY_LINEAR,
        temporal_type=TEMPORAL_STATIC,
        display_name="Counts / identity",
        aliases=("counts", "none", "no_manifold"),
        description="Pass-through observation vector (no additional embedding).",
        notes="Linear static baseline: x_t^(W) itself. UI label `counts` aliases to identity.",
        realtime_capable=True,
        offline_only=False,
    ),
    RepresentationSpec(
        representation_name="raw_lag",
        linearity=LINEARITY_LINEAR,
        temporal_type=TEMPORAL_STATIC,
        quadrant=QUADRANT_UNKNOWN,
        realtime_capable=True,
        offline_only=False,
        display_name="Raw + lagged history",
        implemented=True,
        description="Z-scored counts stacked with the previous 5 causal steps.",
        notes=(
            "History control, not a quadrant member. Separates 'has a dynamics "
            "model' from 'can see the past.' First n_lags samples of a series "
            "are invalid."
        ),
        aliases=("lagged_raw", "raw_lags"),
    ),
    _spec(
        "global_pca",
        linearity=LINEARITY_LINEAR,
        temporal_type=TEMPORAL_STATIC,
        display_name="Global PCA",
        description="Linear PCA of the population observation (train-only fit).",
    ),
    _spec(
        "region_pca",
        linearity=LINEARITY_LINEAR,
        temporal_type=TEMPORAL_STATIC,
        display_name="Regional PCA",
        description="Per-region PCA concatenated into one latent vector.",
        notes="Requires regional unit metadata. Missing anatomy is a skip, not a fabricated embedding.",
        requires_anatomy=True,
    ),
    _spec(
        "layer_pca",
        linearity=LINEARITY_LINEAR,
        temporal_type=TEMPORAL_STATIC,
        display_name="Layer PCA",
        description="Per-layer PCA concatenated into one latent vector.",
        notes=(
            "Linear static, same class as global/region PCA. Intentionally omitted from "
            "the public Latent Representations picker; available in Advanced / full search."
        ),
        requires_anatomy=True,
    ),
    _spec(
        "cell_type_pca",
        linearity=LINEARITY_LINEAR,
        temporal_type=TEMPORAL_STATIC,
        display_name="Cell-type PCA",
        description="Per-cell-type PCA concatenated into one latent vector.",
        notes="Requires cell-type labels. Not a default quadrant representative.",
        requires_anatomy=True,
    ),
    _spec(
        "rate_model_pca",
        linearity=LINEARITY_LINEAR,
        temporal_type=TEMPORAL_STATIC,
        display_name="Rate-model PCA",
        description="PCA of rate-model features rather than spike-count observations.",
        notes=(
            "Classified linear_static from the PCA step. Input features may be "
            "simulation-specific (ground-truth rates). Not used as a default quadrant method."
        ),
        requires_simulation_rates=True,
        scientifically_ambiguous=True,
    ),
    _spec(
        "pls",
        linearity=LINEARITY_LINEAR,
        temporal_type=TEMPORAL_STATIC,
        display_name="PLS (supervised)",
        description="Partial least squares: linear latent directions using the behavioral target.",
        notes=(
            "Ambiguous as a *representation* in a controlled E comparison: the embedding "
            "is supervised (uses y). Classified linear_static from the linear mapping, "
            "but excluded from default quadrant methods so the comparison isolates E, not y-leakage."
        ),
        supervised=True,
        scientifically_ambiguous=True,
    ),
    RepresentationSpec(
        representation_name="bayesian_place_tuning",
        linearity="",
        temporal_type=TEMPORAL_STATIC,
        quadrant=QUADRANT_UNKNOWN,
        realtime_capable=False,
        offline_only=True,
        display_name="Bayesian place tuning",
        implemented=True,
        description="Place-field / encoding-model features, not a population latent.",
        notes=(
            "Not the same scientific object as PCA/Isomap/LDS. Left implemented for the "
            "existing decoder zoo, but quadrant is marked unknown so it is never silently "
            "treated as a linear-static manifold."
        ),
        scientifically_ambiguous=True,
    ),
    _spec(
        "global_isomap",
        linearity=LINEARITY_NONLINEAR,
        temporal_type=TEMPORAL_STATIC,
        display_name="Isomap (classic)",
        description="Nonlinear isometric manifold embedding (offline / out-of-sample limited).",
        notes="Offline diagnostic. Distilled Isomap is the deployable Nyström-style student.",
    ),
    _spec(
        "global_isomap_distilled",
        linearity=LINEARITY_NONLINEAR,
        temporal_type=TEMPORAL_STATIC,
        display_name="Distilled Isomap",
        description="Student map distilled from Isomap for causal / realtime application.",
    ),
    _spec(
        "diffusion_nystrom",
        linearity=LINEARITY_NONLINEAR,
        temporal_type=TEMPORAL_STATIC,
        display_name="Diffusion maps + Nyström",
        description="Nonlinear diffusion-map embedding with Nyström out-of-sample extension.",
    ),
    _spec(
        "global_lds",
        linearity=LINEARITY_LINEAR,
        temporal_type=TEMPORAL_DYNAMIC,
        display_name="Global LDS",
        description="Linear dynamical system / Kalman latent state (causal filter for realtime).",
    ),
    _spec(
        "gpfa",
        linearity=LINEARITY_LINEAR,
        temporal_type=TEMPORAL_DYNAMIC,
        display_name="GPFA (offline)",
        description="Linear-Gaussian factor analysis with AR(1)/GP temporal priors.",
        notes=(
            "Classified linear_dynamic from the existing implementation: linear loadings C "
            "plus Gaussian latents. Primary inference is acausal RTS smoothing "
            "(supports_realtime=False). This is not a nonlinear-dynamic method."
        ),
    ),
)

_UNIMPLEMENTED: tuple[RepresentationSpec, ...] = tuple(
    RepresentationSpec(
        representation_name=name,
        linearity=LINEARITY_NONLINEAR if name in {
            "lfads", "switching_lds", "recurrent_slds",
        } else LINEARITY_LINEAR,
        temporal_type=TEMPORAL_DYNAMIC,
        quadrant=(
            QUADRANT_NONLINEAR_DYNAMIC
            if name in {"lfads", "switching_lds", "recurrent_slds"}
            else QUADRANT_LINEAR_DYNAMIC
        ),
        realtime_capable=False,
        offline_only=True,
        display_name=name.replace("_", " ").title() + " (not implemented)",
        implemented=False,
        description="Reserved future dynamic latent. Not registered for computation.",
        notes=(
            "Placeholder only. The UI must show this quadrant as unavailable rather than "
            "substituting a different method or fabricating results."
            if name in {"lfads", "switching_lds", "recurrent_slds"}
            else "Reserved linear-dynamic variant; not implemented."
        ),
    )
    for name in FUTURE_DYNAMIC_LATENTS
)


def _index_specs(specs: Iterable[RepresentationSpec]) -> dict[str, RepresentationSpec]:
    out: dict[str, RepresentationSpec] = {}
    for spec in specs:
        out[spec.representation_name] = spec
        for alias in spec.aliases:
            out[alias] = spec
    return out


_SPECS: dict[str, RepresentationSpec] = _index_specs((*_IMPLEMENTED, *_UNIMPLEMENTED))


def canonicalize_quadrant(qid: str | None) -> str:
    """Accept legacy or canonical ids; unknown strings stay unknown."""
    raw = str(qid or "").strip()
    if not raw:
        return QUADRANT_UNKNOWN
    if raw in CANONICAL_QUADRANTS or raw == QUADRANT_UNKNOWN:
        return raw
    if raw in LEGACY_QUADRANT_TO_CANONICAL:
        return LEGACY_QUADRANT_TO_CANONICAL[raw]
    return QUADRANT_UNKNOWN


def legacy_quadrant_id(qid: str | None) -> str:
    canon = canonicalize_quadrant(qid)
    return CANONICAL_TO_LEGACY_QUADRANT.get(canon, canon)


def resolve_representation_name(name: str) -> str:
    """Canonical embedding_type (CLI `counts` → `identity`)."""
    return resolve_manifold_alias(str(name or "").strip())


def get_spec(name: str, *, missing: str = "unknown") -> RepresentationSpec:
    key = str(name or "").strip()
    if key in _SPECS:
        return _SPECS[key]
    resolved = resolve_representation_name(key)
    if resolved in _SPECS:
        return _SPECS[resolved]
    if missing == "raise":
        raise KeyError(f"Unknown representation {name!r}")
    return RepresentationSpec(
        representation_name=resolved or key,
        linearity=_UNKNOWN_SPEC.linearity,
        temporal_type=_UNKNOWN_SPEC.temporal_type,
        quadrant=QUADRANT_UNKNOWN,
        realtime_capable=False,
        offline_only=True,
        display_name=resolved or key or "Unknown",
        implemented=False,
        description=_UNKNOWN_SPEC.description,
        notes=_UNKNOWN_SPEC.notes,
        scientifically_ambiguous=True,
    )


def infer_quadrant(name: str | None) -> str:
    """Infer quadrant from a representation name. Never guesses an unimplemented class."""
    if name is None or str(name).strip() == "":
        return QUADRANT_UNKNOWN
    return get_spec(str(name)).quadrant


def is_implemented(name: str) -> bool:
    spec = get_spec(name)
    return bool(spec.implemented and spec.representation_name)


def methods_for_quadrant(
    qid: str,
    *,
    implemented_only: bool = True,
    public_ui_only: bool = False,
) -> tuple[str, ...]:
    """Return representation names in a quadrant (canonical embedding_type)."""
    canon = canonicalize_quadrant(qid)
    names: list[str] = []
    seen: set[str] = set()
    for spec in _IMPLEMENTED if implemented_only else (*_IMPLEMENTED, *_UNIMPLEMENTED):
        if spec.quadrant != canon:
            continue
        if implemented_only and not spec.implemented:
            continue
        if public_ui_only and spec.representation_name in {
            "layer_pca", "cell_type_pca", "rate_model_pca", "pls",
        }:
            continue
        if spec.scientifically_ambiguous and spec.quadrant == QUADRANT_UNKNOWN:
            continue
        name = spec.representation_name
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return tuple(names)


def default_method_for_quadrant(qid: str) -> str | None:
    return DEFAULT_QUADRANT_METHODS.get(canonicalize_quadrant(qid))


def implemented_nonlinear_dynamic_methods() -> tuple[str, ...]:
    return methods_for_quadrant(QUADRANT_NONLINEAR_DYNAMIC, implemented_only=True)


def nonlinear_dynamic_available() -> bool:
    return bool(implemented_nonlinear_dynamic_methods())


def embedding_metadata_columns(embedding_type: str) -> dict[str, Any]:
    """Columns to stamp onto a decoder-comparison metric row (additive)."""
    spec = get_spec(embedding_type)
    return {
        "quadrant": spec.quadrant,
        "linearity": spec.linearity or "",
        "temporal_type": spec.temporal_type or "",
        "realtime_capable": bool(spec.realtime_capable) if spec.implemented else False,
        "offline_only": bool(spec.offline_only),
        "representation_display_name": spec.display_name,
        "quadrant_label": QUADRANT_DISPLAY_LABELS.get(spec.quadrant, spec.quadrant),
    }


def annotate_metrics_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Fill quadrant metadata on metric dicts. Existing values are kept unless overwrite."""
    out: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        name = row.get("embedding_type") or row.get("manifold") or row.get("feature_mode")
        meta = embedding_metadata_columns(str(name or ""))
        for key, value in meta.items():
            if overwrite or key not in row or row[key] in (None, ""):
                row[key] = value
        if not overwrite and str(row.get("quadrant") or "") not in {
            *CANONICAL_QUADRANTS, QUADRANT_UNKNOWN,
        }:
            row["quadrant"] = canonicalize_quadrant(row.get("quadrant")) or infer_quadrant(
                str(name or "")
            )
        out.append(row)
    return out


def list_catalog(*, include_unimplemented: bool = False) -> list[dict[str, Any]]:
    specs = list(_IMPLEMENTED)
    if include_unimplemented:
        specs.extend(_UNIMPLEMENTED)
    seen: set[str] = set()
    catalog: list[dict[str, Any]] = []
    for spec in specs:
        if spec.representation_name in seen:
            continue
        seen.add(spec.representation_name)
        catalog.append(spec.to_dict())
    return catalog


def known_embedding_names() -> frozenset[str]:
    names = set(ALL_EMBEDDING_TYPES)
    names.update(_SPECS.keys())
    return frozenset(names)


def representation_fit_dependencies(
    *,
    observation_hash: str,
    embedding_type: str,
    n_components: int,
    seed: int,
    train_frac: float,
) -> dict[str, Any]:
    """Scientific dependencies of a fitted E. Decoder name is intentionally absent."""
    return {
        "observation_hash": str(observation_hash),
        "embedding_type": resolve_representation_name(embedding_type),
        "n_components": int(n_components),
        "seed": int(seed),
        "train_frac": float(train_frac),
    }


def is_known_dynamic_implementation(name: str) -> bool:
    return is_dynamic_latent(resolve_representation_name(name))
