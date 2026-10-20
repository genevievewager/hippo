#!/usr/bin/env python3
"""v7 preview: Fig1 + Fig2 only. Numbers frozen at v5. No story page yet."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.report.anon_util import anonymize_data_dir, leak_check, build_session_map
from agents.data_quality.report.extract_rasters_1s import extract_rasters_1s
from agents.data_quality.report.metric_registry_v7 import audit_duplicates, write_registry
from agents.data_quality.report.number_audit import audit
from agents.data_quality.report.overlap_audit import audit_pdf
from agents.data_quality.report.page_geometry import place_figure_on_a4
from agents.data_quality.report.whitespace_audit import audit_pngs

ROOT = Path(__file__).resolve().parents[3]
V5 = DATA_QUALITY_ROOT / "reports" / "report4a_v5"
STEP1 = DATA_QUALITY_ROOT / "reports" / "report4a_v3" / "_step1_behavior_info"
OUT = assert_dq_path(DATA_QUALITY_ROOT / "reports" / "report4a_v7_preview")
DATA = OUT / "data"
DATA_ANON = OUT / "data_anon"
BEHINFO = OUT / "behavior_info"
BEH_ANON = OUT / "behavior_info_anon"
FIGS_ANON = OUT / "figs_anon"
PAGES_ANON = OUT / "pages_anon"
DELIVER = Path("/home/gmwager/projects/hippo/tmp_r4_review/v7")


def _run(cmd: list[str]) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    pip_local = str(ROOT / ".pip_local")
    env = {**os.environ}
    env["PYTHONPATH"] = f"{ROOT}:{pip_local}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def _anon_behinfo(src: Path, dst: Path, sess_map: dict[str, str]) -> None:
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
        elif p.suffix in (".npz", ".npy", ".pkl"):
            continue
        else:
            shutil.copy2(p, dst / p.name)


def _prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if DATA.exists():
        shutil.rmtree(DATA)
    shutil.copytree(V5 / "data", DATA)
    if BEHINFO.exists():
        shutil.rmtree(BEHINFO)
    shutil.copytree(V5 / "behavior_info", BEHINFO)
    # frozen step1 behaviour fracs / HD medians
    if STEP1.is_dir():
        for name in ("step1_session_distributions.csv", "step1_dataset_pooled.csv"):
            src = STEP1 / name
            if src.is_file():
                shutil.copy2(src, DATA / name)
    # display rasters (metrics unchanged)
    meta = json.loads((DATA / "data_meta.json").read_text())
    session = meta.get("example_session") or "1150_09102024_2rooms"
    if not (DATA / "data_rasters_1s.json").is_file():
        extract_rasters_1s(DATA, session)

    if DATA_ANON.exists():
        shutil.rmtree(DATA_ANON)
    sess_map = anonymize_data_dir(DATA, DATA_ANON)
    import pandas as pd
    for name in ("encoding_diag_units.csv", "decode_rows.csv", "drift_sep_sessions.csv",
                 "step1_session_distributions.csv"):
        p = (BEHINFO if name != "step1_session_distributions.csv" else DATA) / name
        if p.is_file() and "session" in pd.read_csv(p, nrows=1).columns:
            sess_map.update(build_session_map(pd.read_csv(p)["session"].astype(str).tolist()))
    if BEH_ANON.exists():
        shutil.rmtree(BEH_ANON)
    _anon_behinfo(BEHINFO, BEH_ANON, sess_map)
    # also anon step1 in data_anon (already via anonymize_data_dir)


def main() -> None:
    _prepare()
    FIGS_ANON.mkdir(parents=True, exist_ok=True)
    PAGES_ANON.mkdir(parents=True, exist_ok=True)
    write_registry(OUT / "metric_registry.json")

    _run([
        sys.executable, "-m", "agents.data_quality.report.figures_dense_v7",
        "--data", str(DATA_ANON),
        "--behinfo", str(BEH_ANON),
        "--step1", str(DATA_ANON),
        "--out", str(FIGS_ANON),
        "--only", "Fig1,Fig2",
    ])

    for stem in ("Fig1_spikes_behaviour", "Fig2_tuning_drift"):
        place_figure_on_a4(FIGS_ANON / f"{stem}.pdf", PAGES_ANON / f"{stem}.pdf")

    writer = PdfWriter()
    order = [
        PAGES_ANON / "Fig1_spikes_behaviour.pdf",
        PAGES_ANON / "Fig2_tuning_drift.pdf",
    ]
    for p in order:
        writer.add_page(PdfReader(str(p)).pages[0])
    preview = OUT / "report4a_v7_preview_anon.pdf"
    with preview.open("wb") as f:
        writer.write(f)

    # Audits
    leak = leak_check([preview, DATA_ANON / "data_meta.json"])
    num = audit(DATA, OUT / "number_audit.json", preview)
    overlap = audit_pdf(preview, pages_png_dir=PAGES_ANON / "png")

    # whitespace on delivered PNGs
    pngs = sorted((PAGES_ANON / "png").glob("page-*.png"))
    white = audit_pngs(pngs)

    # duplicate registry audit from page text
    page_texts = {}
    reader = PdfReader(str(preview))
    for i, page in enumerate(reader.pages):
        name = "Fig1" if i == 0 else "Fig2"
        page_texts[name] = page.extract_text() or ""
    dup = audit_duplicates(page_texts)

    summary = {
        "preview_pdf": str(preview),
        "n_pages": 2,
        "leak_ok": leak.get("ok"),
        "number_ok": num.get("ok"),
        "number_failures": (num.get("failures") or [])[:15],
        "overlap_ok": overlap.get("ok"),
        "overlap_failures": (overlap.get("failures") or [])[:15],
        "whitespace_ok": white.get("ok"),
        "whitespace_pages": white.get("pages"),
        "duplicate_ok": dup.get("ok"),
        "duplicate_failures": dup.get("failures"),
    }
    (OUT / "BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "anon_leak_check.json").write_text(json.dumps(leak, indent=2) + "\n")
    (OUT / "whitespace_audit.json").write_text(json.dumps(white, indent=2) + "\n")
    (OUT / "duplicate_audit.json").write_text(json.dumps(dup, indent=2) + "\n")
    if overlap:
        (OUT / "overlap_audit.json").write_text(json.dumps(overlap, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)

    DELIVER.mkdir(parents=True, exist_ok=True)
    shutil.copy2(preview, DELIVER / "report4a_v7_preview_anon.pdf")
    for i, name in enumerate(("Fig1", "Fig2"), 1):
        for cand in (PAGES_ANON / "png" / f"page-{i:02d}.png", PAGES_ANON / "png" / f"page-{i}.png"):
            if cand.is_file():
                shutil.copy2(cand, DELIVER / f"preview_{name}.png")
                break
    # also copy native fig PNGs (higher res matplotlib)
    for stem, name in (("Fig1_spikes_behaviour", "Fig1_native"), ("Fig2_tuning_drift", "Fig2_native")):
        src = FIGS_ANON / f"{stem}.png"
        if src.is_file():
            shutil.copy2(src, DELIVER / f"{name}.png")
    (DELIVER / "preview_BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in ("anon_leak_check", "number_audit", "overlap_audit", "whitespace_audit", "duplicate_audit"):
        src = OUT / f"{name}.json"
        if src.is_file():
            shutil.copy2(src, DELIVER / f"preview_{name}.json")

    ok = all([
        leak.get("ok"), num.get("ok"), overlap.get("ok"),
        white.get("ok"), dup.get("ok"),
    ])
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
