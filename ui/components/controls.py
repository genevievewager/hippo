"""Shared Streamlit control widgets."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.services.comparison import (
    DECODE_WINDOW_LABELS,
    UI_FEATURE_SET_OPTIONS,
    UI_MANIFOLD_OPTIONS,
    format_decode_window,
)
from ui.services.datasets import DatasetInfo, available_spike_sources, inspect_dataset, list_datasets
from ui import state


def render_active_dataset_sidebar(outputs_root: Path) -> None:
    """Read-only summary of the global active dataset (change on Experiment Setup)."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Dataset")
    current = state.get_active_dataset()
    if current is None or not Path(current).exists():
        st.sidebar.caption("None selected — set on **Experiment Setup**.")
        return

    selected = Path(current)
    info = inspect_dataset(selected)
    summary = info.summary or {}
    st.sidebar.markdown(
        f"**{selected.name}**  \n"
        f"{summary.get('session_duration_s', '—')} s · "
        f"{summary.get('n_units', '—')} units  \n"
        f"spikes: "
        f"{'sorted' if info.has_sorted_spikes else ''}"
        f"{' + GT' if info.has_ground_truth_spikes else ''}"
    )
    bits = []
    if info.has_decoder_comparison:
        bits.append("decoder analyses")
    if info.has_realtime:
        bits.append("realtime")
    n_figs = 0
    fig_dir = selected / "figures"
    if fig_dir.exists():
        n_figs = len(list(fig_dir.rglob("*.png")))
    if n_figs:
        bits.append(f"{n_figs} figures")
    if bits:
        st.sidebar.caption(" · ".join(bits))
    st.sidebar.caption("Change on **Experiment Setup**.")


def dataset_selector(
    outputs_root: Path,
    *,
    key: str = "dataset_selector",
    label: str = "Dataset",
    allow_change: bool = True,
) -> Path | None:
    """Use the global active dataset; optionally allow changing it on-page."""
    active = state.get_active_dataset()
    datasets = list_datasets(outputs_root)
    if not datasets:
        st.warning(f"No datasets found under `{outputs_root}`. Generate one in Experiment Setup.")
        return None

    if not allow_change and active is not None and Path(active).exists():
        st.caption(f"Using active dataset: `{Path(active).name}`")
        return Path(active)

    labels = [d.name for d in datasets]
    index = 0
    if active is not None:
        for i, d in enumerate(datasets):
            if d.path.resolve() == Path(active).resolve() or d.name == Path(active).name:
                index = i
                break
    choice = st.selectbox(label, labels, index=index, key=key)
    selected = next(d for d in datasets if d.name == choice)
    state.set_active_dataset(selected.path)
    return selected.path


def require_active_dataset(outputs_root: Path) -> Path | None:
    """Return the global active dataset or prompt the user to set one on Experiment Setup."""
    active = state.get_active_dataset()
    if active is None or not Path(active).exists():
        st.info("No active dataset. Choose one on **Experiment Setup**.")
        return None
    st.caption(
        f"**Active dataset:** `{Path(active).name}` · change on **Experiment Setup**"
    )
    return Path(active)


def active_spike_source(
    dataset: Path | None,
    *,
    readonly: bool = True,
    key: str = "spike_source",
) -> str:
    """Read or set the global spike source (canonical control on Neural Simulation)."""
    sources = available_spike_sources(dataset) if dataset else ["sorted"]
    if not sources:
        sources = ["sorted"]
    if readonly:
        value = state.resolve_spike_source(dataset, sources=sources)
        st.caption(
            f"**Spike source:** `{value}` · change on **Neural Simulation**"
        )
        return value
    default = state.resolve_spike_source(dataset, sources=sources)
    idx = sources.index(default) if default in sources else 0
    value = st.selectbox("Spike source", sources, index=idx, key=key)
    state.set_spike_source(value)
    return value


def spike_source_selector(
    dataset: Path | None,
    *,
    key: str = "spike_source",
) -> str:
    """Canonical spike-source picker (Neural Simulation only)."""
    return active_spike_source(dataset, readonly=False, key=key)


def render_context_banner(dataset: Path | None, spike_source: str) -> None:
    """Single-line analysis context for consumer pages."""
    if dataset is None:
        return
    st.caption(
        f"**Dataset:** `{dataset.name}` · **Spike source:** `{spike_source}` · "
        "set on **Experiment Setup** / **Neural Simulation**"
    )


def feature_set_multiselect(
    *,
    key: str = "feature_sets",
    defaults: tuple[str, ...] | None = None,
    label: str = "Feature sets",
) -> list[str]:
    defaults = defaults or UI_FEATURE_SET_OPTIONS
    return st.multiselect(
        label,
        options=list(UI_FEATURE_SET_OPTIONS),
        default=[d for d in defaults if d in UI_FEATURE_SET_OPTIONS],
        key=key,
        help="Neural feature extraction families (upstream of manifold embeds).",
    )


def manifold_multiselect(
    *,
    key: str = "manifolds",
    defaults: tuple[str, ...] | None = None,
    options: tuple[str, ...] | list[str] | None = None,
    label: str = "Manifolds / embeddings",
) -> list[str]:
    from ui.services.comparison import UI_MANIFOLD_OPTIONS
    from ui.services.representations import format_representation_label

    opts = list(options) if options is not None else list(UI_MANIFOLD_OPTIONS)
    defaults = defaults or tuple(opts[:3])
    return st.multiselect(
        label,
        options=opts,
        default=[d for d in defaults if d in opts],
        format_func=format_representation_label,
        key=key,
        help=(
            "`counts` means identity (no manifold) — same CLI alias as "
            "`--manifolds counts`. Badges reflect realtime/causal capability."
        ),
    )


def representation_family_selector(*, key: str = "rep_family") -> str:
    """High-level Static manifold vs Dynamic latent state selector."""
    choice = st.radio(
        "Representation Type",
        options=["Static manifold", "Dynamic latent state"],
        horizontal=True,
        key=key,
        help=(
            "Static: x_t → z_t. Dynamic: z_(t-1), x_t → z_t. "
            "Downstream behavioral decoders are shared."
        ),
    )
    return "dynamic" if choice.startswith("Dynamic") else "static"


def render_architecture_diagram(family: str) -> None:
    """Simple schematic that changes with the selected pathway."""
    if family == "dynamic":
        st.code(
            "                 ┌───────────────┐\n"
            "                 │ Previous z_t  │\n"
            "                 └──────┬────────┘\n"
            "                        ↓\n"
            "Spikes → Features → Dynamic State Model → z_t\n"
            "                                      ↓\n"
            "                                   Decoder\n"
            "                                      ↓\n"
            "                                   Behavior",
            language="text",
        )
    else:
        st.code(
            "Spikes\n"
            "  ↓\n"
            "Features\n"
            "  ↓\n"
            "Static Manifold\n"
            "  ↓\n"
            "Decoder\n"
            "  ↓\n"
            "Behavior",
            language="text",
        )


def decode_window_multiselect(
    *,
    key: str = "decode_windows",
    defaults: list[float] | None = None,
    label: str = "Decode windows",
) -> list[float]:
    options = list(DECODE_WINDOW_LABELS.keys())
    defaults = defaults if defaults is not None else options
    labels = [format_decode_window(w) for w in options]
    label_to_val = dict(zip(labels, options))
    default_labels = [format_decode_window(w) for w in defaults if w in options]
    chosen = st.multiselect(
        label,
        options=labels,
        default=default_labels,
        key=key,
    )
    return [float(label_to_val[c]) for c in chosen]


def gated_decode_window_selector(
    cached_windows: list[float] | tuple[float, ...],
    *,
    key: str,
    defaults: list[float] | None = None,
    label: str = "Decode windows",
    multiple: bool = True,
    disabled_help: str | None = None,
) -> list[float]:
    """Show all canonical W; enable only those Feature Construction has cached.

    Uncached options are visible but disabled. Returns only enabled selections.
    """
    from realtime.transform_cache import window_ms

    options = list(DECODE_WINDOW_LABELS.keys())
    cached_ms = {window_ms(w) for w in cached_windows}
    defaults = list(defaults) if defaults is not None else options
    default_ms = {window_ms(w) for w in defaults}
    grey_help = disabled_help or "Generate this window on Feature Construction first."

    st.markdown(f"**{label}**")
    selected: list[float] = []
    cols = st.columns(len(options))
    for col, w in zip(cols, options):
        enabled = window_ms(w) in cached_ms
        with col:
            checked = st.checkbox(
                format_decode_window(w),
                value=bool(enabled and window_ms(w) in default_ms),
                disabled=not enabled,
                key=f"{key}_{window_ms(w):04d}",
                help=(None if enabled else grey_help),
            )
        if checked and enabled:
            selected.append(float(w))

    if not cached_ms:
        st.caption("Generate windows on **Feature Construction** first.")
    else:
        missing = [
            format_decode_window(w)
            for w in options
            if window_ms(w) not in cached_ms
        ]
        if missing:
            st.caption("Not yet generated (grey): " + ", ".join(missing))

    if not multiple and selected:
        pick = st.radio(
            "Use window",
            options=selected,
            format_func=format_decode_window,
            key=f"{key}_pick",
            horizontal=True,
        )
        return [float(pick)]
    return selected


def decoder_multiselects(
    *,
    key_prefix: str = "decoders",
    continuous_defaults: list[str] | tuple[str, ...] | None = None,
    categorical_defaults: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Pick continuous + categorical decoder families for a benchmark run."""
    from ui.services.comparison import (
        UI_CATEGORICAL_DECODER_OPTIONS,
        UI_CONTINUOUS_DECODER_OPTIONS,
        UI_DEFAULT_CATEGORICAL_DECODERS,
        UI_DEFAULT_CONTINUOUS_DECODERS,
    )

    cont_opts = list(UI_CONTINUOUS_DECODER_OPTIONS)
    cat_opts = list(UI_CATEGORICAL_DECODER_OPTIONS)
    cont_default = list(continuous_defaults) if continuous_defaults is not None else list(
        UI_DEFAULT_CONTINUOUS_DECODERS
    )
    cat_default = list(categorical_defaults) if categorical_defaults is not None else list(
        UI_DEFAULT_CATEGORICAL_DECODERS
    )
    cont_default = [d for d in cont_default if d in cont_opts]
    cat_default = [d for d in cat_default if d in cat_opts]

    c1, c2 = st.columns(2)
    with c1:
        continuous = st.multiselect(
            "Continuous decoders",
            options=cont_opts,
            default=cont_default,
            key=f"{key_prefix}_continuous",
            help="Used for position, speed, HD, wall distance, …",
        )
    with c2:
        categorical = st.multiselect(
            "Categorical decoders",
            options=cat_opts,
            default=cat_default,
            key=f"{key_prefix}_categorical",
            help="Used for spatial context, movement state, wall bins, …",
        )
    return list(dict.fromkeys([*continuous, *categorical]))


def metric_row(items: dict[str, object]) -> None:
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items.items()):
        col.metric(label, value)


def dataset_summary_cards(info: DatasetInfo) -> None:
    summary = info.summary or {}
    metric_row({
        "Duration (s)": summary.get("session_duration_s", "—"),
        "Behavior dt": summary.get("behavior_dt", "—"),
        "Units": summary.get("n_units", "—"),
        "Regions": len(summary.get("present_regions") or []),
    })
