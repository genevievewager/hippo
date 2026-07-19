"""PDF compilation helpers for experiment figures.

Public API used by ``run_visualizations.py`` and the decoder workflow.
Implementation lives in ``compile_figures_pdf.py``; this module is the
preferred import path.
"""

from __future__ import annotations

from pathlib import Path

from visualization.compile_figures_pdf import (
    compile_figures_pdf as _compile_figures_pdf,
)


def compile_figures_pdf(
    figures_dir: Path | None = None,
    output_pdf: Path | None = None,
    *,
    experiment_dir: Path | None = None,
) -> Path:
    """
    Compile all PNGs under ``figures_dir`` into a sectioned PDF.

    Parameters
    ----------
    figures_dir
        Directory containing PNGs (searched recursively). Defaults to
        ``experiment_dir / "figures"`` when ``experiment_dir`` is given.
    output_pdf
        Destination path. Defaults to ``figures_dir / "output.pdf"``.
    experiment_dir
        Optional experiment root used to resolve ``figures_dir``.

    Returns
    -------
    Path
        Path to the written PDF (typically ``figures/output.pdf``).
    """
    if figures_dir is None:
        if experiment_dir is None:
            raise ValueError("Provide figures_dir or experiment_dir")
        figures_dir = Path(experiment_dir) / "figures"
    return _compile_figures_pdf(Path(figures_dir), output_pdf=output_pdf)
