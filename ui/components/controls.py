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
    """Show / change the global active dataset in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Dataset")
    datasets = list_datasets(outputs_root)
    if not datasets:
        st.sidebar.caption("No datasets under `outputs/` yet. Generate one in Experiment Setup.")
        return

    labels = [d.name for d in datasets]
    current = state.get_active_dataset()
    index = 0
    if current is not None:
        for i, d in enumerate(datasets):
            if d.path.resolve() == Path(current).resolve() or d.name == Path(current).name:
                index = i
                break
    choice = st.sidebar.selectbox(
        "Dataset",
        labels,
        index=index,
        key="sidebar_active_dataset",
        help="Shared across all pages.",
    )
    selected = next(d for d in datasets if d.name == choice)
    if state.get_active_dataset() is None or Path(state.get_active_dataset()).resolve() != selected.path.resolve():
        state.set_active_dataset(selected.path)

    info = inspect_dataset(selected.path)
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
    fig_dir = selected.path / "figures"
    if fig_dir.exists():
        n_figs = len(list(fig_dir.rglob("*.png")))
    if n_figs:
        bits.append(f"{n_figs} figures")
    if bits:
        st.sidebar.caption(" · ".join(bits))


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


def spike_source_selector(
    dataset: Path | None,
    *,
    key: str = "spike_source",
) -> str:
    sources = available_spike_sources(dataset) if dataset else ["sorted"]
    if not sources:
        sources = ["sorted"]
    default = st.session_state.get(state.KEY_SPIKE_SOURCE, sources[0])
    idx = sources.index(default) if default in sources else 0
    value = st.selectbox("Spike source", sources, index=idx, key=key)
    st.session_state[state.KEY_SPIKE_SOURCE] = value
    return value


def feature_set_multiselect(
    *,
    key: str = "feature_sets",
    defaults: tuple[str, ...] | None = None,
) -> list[str]:
    defaults = defaults or UI_FEATURE_SET_OPTIONS
    return st.multiselect(
        "Feature sets",
        options=list(UI_FEATURE_SET_OPTIONS),
        default=list(defaults),
        key=key,
        help="Neural feature extraction families (upstream of manifold embeds).",
    )


def manifold_multiselect(
    *,
    key: str = "manifolds",
    defaults: tuple[str, ...] | None = None,
    options: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    from ui.services.comparison import UI_MANIFOLD_OPTIONS
    from ui.services.representations import format_representation_label

    opts = list(options) if options is not None else list(UI_MANIFOLD_OPTIONS)
    defaults = defaults or tuple(opts[:3])
    return st.multiselect(
        "Manifolds / embeddings",
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
) -> list[float]:
    options = list(DECODE_WINDOW_LABELS.keys())
    defaults = defaults if defaults is not None else options
    labels = [format_decode_window(w) for w in options]
    label_to_val = dict(zip(labels, options))
    default_labels = [format_decode_window(w) for w in defaults if w in options]
    chosen = st.multiselect(
        "Decode windows",
        options=labels,
        default=default_labels,
        key=key,
    )
    return [float(label_to_val[c]) for c in chosen]


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
