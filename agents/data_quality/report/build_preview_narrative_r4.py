#!/usr/bin/env python3
"""Preview build: story + Fig1 + Fig2 only (narrative v6). Numbers frozen at v5."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pypdf import PdfWriter

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.report.anon_util import anonymize_data_dir, leak_check
from agents.data_quality.report.build_story_narrative_r4 import write_story_narrative
from agents.data_quality.report.number_audit import audit
from agents.data_quality.report.overlap_audit import audit_pdf
from agents.data_quality.report.page_geometry import place_figure_on_a4

ROOT = Path(__file__).resolve().parents[3]
V5 = DATA_QUALITY_ROOT / "reports" / "report4a_v5"
OUT = assert_dq_path(DATA_QUALITY_ROOT / "reports" / "report4a_v6_preview")
DATA = OUT / "data"
DATA_ANON = OUT / "data_anon"
BEHINFO = OUT / "behavior_info"
FIGS = OUT / "figs"
FIGS_ANON = OUT / "figs_anon"
PAGES = OUT / "pages"
PAGES_ANON = OUT / "pages_anon"
DELIVER = Path("/home/gmwager/projects/hippo/tmp_r4_review/v6")


def _run(cmd: list[str]) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    pip_local = str(ROOT / ".pip_local")
    env = {**os.environ}
    env["PYTHONPATH"] = f"{ROOT}:{pip_local}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def _anon_behinfo(src: Path, dst: Path, sess_map: dict[str, str]) -> None:
    """Rewrite session/animal columns in behavior_info CSVs/JSON (no data_meta)."""
    import pandas as pd
    from agents.data_quality.report.anon_util import ANIMAL_CODES, _rewrite_str, _walk

    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if not p.is_file():
            continue
        if p.suffix == ".json":
            raw = json.loads(p.read_text())
            (dst / p.name).write_text(json.dumps(_walk(raw, sess_map), indent=2, default=str) + "\n")
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].map(
                        lambda v: _rewrite_str(str(v), sess_map) if isinstance(v, str) else v
                    )
            if "animal" in df.columns:
                df["animal"] = df["animal"].map(lambda a: ANIMAL_CODES.get(str(a), str(a)))
            df.to_csv(dst / p.name, index=False)
        else:
            # skip large binary pools in preview
            if p.suffix in (".npz", ".npy", ".pkl"):
                continue
            shutil.copy2(p, dst / p.name)


def _reuse_v5() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if DATA.exists():
        shutil.rmtree(DATA)
    shutil.copytree(V5 / "data", DATA)
    if BEHINFO.exists():
        shutil.rmtree(BEHINFO)
    shutil.copytree(V5 / "behavior_info", BEHINFO)
    if DATA_ANON.exists():
        shutil.rmtree(DATA_ANON)
    sess_map = anonymize_data_dir(DATA, DATA_ANON)
    # also pull sessions from behinfo CSVs
    import pandas as pd
    from agents.data_quality.report.anon_util import build_session_map
    for name in ("encoding_diag_units.csv", "decode_rows.csv", "drift_noise_units.csv"):
        p = BEHINFO / name
        if p.is_file():
            df = pd.read_csv(p)
            if "session" in df.columns:
                sess_map.update(build_session_map(df["session"].astype(str).tolist()))
    beh_anon = OUT / "behavior_info_anon"
    if beh_anon.exists():
        shutil.rmtree(beh_anon)
    _anon_behinfo(BEHINFO, beh_anon, sess_map)


def main() -> None:
    _reuse_v5()
    FIGS.mkdir(parents=True, exist_ok=True)
    FIGS_ANON.mkdir(parents=True, exist_ok=True)
    PAGES.mkdir(parents=True, exist_ok=True)
    PAGES_ANON.mkdir(parents=True, exist_ok=True)

    beh_anon = OUT / "behavior_info_anon"

    # Render from anon tables so figure PDF text layer has no raw IDs
    _run([
        sys.executable, "-m", "agents.data_quality.report.figures_narrative_r4",
        "--data", str(DATA_ANON),
        "--behinfo", str(beh_anon),
        "--out", str(FIGS_ANON),
        "--only", "Fig1,Fig2",
    ])
    # Also keep non-anon figs for debugging
    _run([
        sys.executable, "-m", "agents.data_quality.report.figures_narrative_r4",
        "--data", str(DATA),
        "--behinfo", str(BEHINFO),
        "--out", str(FIGS),
        "--only", "Fig1,Fig2",
    ])

    headline = FIGS_ANON / "Fig1a_headline.png"
    story_pdf = PAGES_ANON / "00_story.pdf"
    write_story_narrative(
        DATA_ANON, story_pdf,
        headline_png=headline if headline.is_file() else None,
        behinfo=beh_anon if beh_anon.is_dir() else None,
    )

    # Place figures onto A4
    for stem in ("Fig1_behavior_info", "Fig2_tuning_drift"):
        src = FIGS_ANON / f"{stem}.pdf"
        dst = PAGES_ANON / f"{stem}.pdf"
        place_figure_on_a4(src, dst)

    # Merge preview PDF (story, Fig1, Fig2)
    writer = PdfWriter()
    order = [
        PAGES_ANON / "00_story.pdf",
        PAGES_ANON / "Fig1_behavior_info.pdf",
        PAGES_ANON / "Fig2_tuning_drift.pdf",
    ]
    for p in order:
        from pypdf import PdfReader
        writer.add_page(PdfReader(str(p)).pages[0])
    preview = OUT / "report4a_preview_anon.pdf"
    with preview.open("wb") as f:
        writer.write(f)

    # Audits
    leak = leak_check([preview, DATA_ANON / "data_meta.json"])
    (OUT / "anon_leak_check.json").write_text(json.dumps(leak, indent=2) + "\n")

    num = audit(DATA, OUT / "number_audit.json", preview)
    # number audit allowlist uses DATA (non-anon) + sibling behavior_info
    # Ensure behavior_info is sibling of DATA
    overlap = audit_pdf(preview, pages_png_dir=PAGES_ANON / "png")

    summary = {
        "preview_pdf": str(preview),
        "n_pages": 3,
        "leak_ok": leak.get("ok"),
        "leak_hits": leak.get("n_hits"),
        "number_ok": num.get("ok"),
        "number_n_failures": num.get("n_failures"),
        "number_failures": num.get("failures", [])[:20],
        "overlap_ok": overlap.get("ok"),
        "overlap_failures": (overlap.get("failures") or [])[:20],
    }
    (OUT / "BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    # Deliver PNGs + PDF into hippo review folder
    DELIVER.mkdir(parents=True, exist_ok=True)
    shutil.copy2(preview, DELIVER / "report4a_preview_anon.pdf")
    png_src = PAGES_ANON / "png"
    if png_src.is_dir():
        for i, name in enumerate(("story", "Fig1", "Fig2"), 1):
            src = png_src / f"page-{i:02d}.png"
            if not src.is_file():
                src = png_src / f"page-{i}.png"
            if src.is_file():
                shutil.copy2(src, DELIVER / f"preview_{name}.png")
    (DELIVER / "preview_BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copy2(OUT / "anon_leak_check.json", DELIVER / "preview_anon_leak_check.json")
    shutil.copy2(OUT / "number_audit.json", DELIVER / "preview_number_audit.json")
    if (PAGES_ANON / "png").is_dir() or overlap:
        (DELIVER / "preview_overlap_audit.json").write_text(json.dumps(overlap, indent=2) + "\n")

    if not (leak.get("ok") and num.get("ok") and overlap.get("ok")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
