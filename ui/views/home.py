"""Page: Home — project overview curated from the README."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render(_outputs_root: Path) -> None:
    st.header("Hippocampal Neuropixels Simulation and Decoding")
    st.markdown(
        "End-to-end hippocampal BCI simulation and deployment testbed. "
        "Open-field behavior and hippocampal population rates come from "
        "RatInABox. Neuropixels acquisition degradation and spike sorting "
        "produce the deployable spike source. Causal neural features (`F`) "
        "are mapped to a static or dynamic representation (`E`), then decoded "
        "(`D`) over causal windows (`W`). Deployment selection, realtime "
        "replay, and a frozen live bundle follow. The UI is a thin client "
        "over the same scientific backends as the CLI."
    )

    st.subheader("Design space")
    st.markdown(
        "The complete BCI design space spans `F × E × D × W × C`, where:"
    )
    st.markdown(
        """
- `F` = neural observation construction from spikes
- `E` = population-state representation applied to that observation
- `D` = behavioral decoder
- `W` = causal spike integration window
- `C` = closed-loop rule
"""
    )
    st.markdown(
        "The core decoder benchmark searches `F × E × D × W`. Closed-loop "
        "rule `C` is evaluated on decoded predictions and registry replay. "
        "Selection uses **sorted spikes only**; ground-truth spikes are "
        "diagnostic / non-deployable. Offline Isomap and GPFA can appear in "
        "comparison but cannot auto-win closed-loop deployment. "
        "`diffusion_nystrom` is the deployable nonlinear embedding."
    )

    st.subheader("Representation classes")
    st.markdown(
        "Latent Representations, Decoder Benchmark, and Realtime Replay "
        "organize `E` as a 2×2 grid (linearity × temporal dynamics):"
    )
    st.markdown(
        """
| | Linear | Nonlinear |
|---|---|---|
| **Static** | counts, global PCA, region PCA | diffusion maps + Nyström, distilled Isomap, Isomap (offline) |
| **Dynamic** | LDS , GPFA (offline) | not implemented |
"""
    )
    st.markdown(
        "Realtime Replay runs the three **realtime-capable** cells: "
        "`global_pca`, `diffusion_nystrom`, and `global_lds`. GPFA and "
        "classic Isomap remain offline diagnostics."
    )

    st.subheader("Purpose")
    st.markdown(
        "Identify a hippocampal BCI decoder under causal "
        "and realtime constraints: compare static manifolds and dynamic "
        "latents under the same decoder zoo, select deployable configurations "
        "on sorted-spike held-out performance, export a lab transplant "
        "registry, and evaluate closed-loop policies on that registry."
    )

    st.subheader("Scientific questions")
    st.markdown(
        """
1. Which neural features retain behaviorally useful information in the hippocampal formation?
2. Do anatomically structured or nonlinear population representations improve variable-specific decoding over raw population activity?
3. Do dynamic latent-state representations outperform static neural manifolds under causal constraints?
4. How much causal neural history is optimal for different behavioral variables?
5. Which configurations are deployable after sorted-spike evaluation and realtime-compatibility constraints?
6. Which decoded neural variables remain robust enough under recording degradation to support closed-loop BCI operation?
"""
    )

    st.subheader("UI workflow")
    st.markdown(
        "Use **Experiment Setup** to generate or load a dataset — that sets the "
        "shared **Active Dataset** for every downstream page. On **Neural "
        "Simulation**, choose **Spike source** (sorted vs ground truth); "
        "Feature Construction, Latent Representations, Decoder Benchmark, and "
        "Realtime Replay show those choices as read-only context banners. "
        "Live Open Ephys is a stub; Replay on Live Deployment uses stored "
        "sorted spikes. Simulation-trained bundles are **pipeline tests**, "
        "not scientifically validated behavioral decoding."
    )
    st.markdown(
        """
| Page | Role |
|------|------|
| Experiment Setup | Generate or load a dataset; sets **Active Dataset** |
| Neural Simulation | Inspect rates, spikes, sorting quality; sets **Spike source** |
| Feature Construction | Build and inspect observation sets `F` |
| Latent Representations | Fit and cache embeddings `E` in the 2×2 grid |
| Decoder Benchmark | Search `D × W` (continuous and discrete jobs); reuse cached transforms |
| Realtime Replay | Closed-loop comparison of the three realtime-capable representations |
| Live Deployment | Pack a frozen `F → E → D` bundle; Replay now, Open Ephys later |
"""
    )
