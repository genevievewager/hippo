#!/usr/bin/env python3
"""Build full Report 4a v8 — prose page + Figs 1–5 + S1 + Appendix (8 pages)."""

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
from agents.data_quality.report.build_story_v8 import audit_story_prose, write_story_v8
from agents.data_quality.report.extract_rasters_1s import extract_rasters_1s
from agents.data_quality.report.metric_registry_v8 import audit_duplicates, write_registry
from agents.data_quality.report.number_audit import audit
from agents.data_quality.report.overlap_audit import audit_pdf
from agents.data_quality.report.page_geometry import place_figure_on_a4
from agents.data_quality.report.whitespace_audit import audit_pngs

ROOT = Path(__file__).resolve().parents[3]
V5 = DATA_QUALITY_ROOT / "reports" / "report4a_v5"
STEP1 = DATA_QUALITY_ROOT / "reports" / "report4a_v3" / "_step1_behavior_info"
OUT = assert_dq_path(DATA_QUALITY_ROOT / "reports" / "report4a_v8")
DATA = OUT / "data"
DATA_ANON = OUT / "data_anon"
BEHINFO = OUT / "behavior_info"
BEH_ANON = OUT / "behavior_info_anon"
FIGS = OUT / "figs"
FIGS_ANON = OUT / "figs_anon"
PAGES = OUT / "pages"
PAGES_ANON = OUT / "pages_anon"
DELIVER = Path("/home/gmwager/projects/hippo/tmp_r4_review/v8")
REPORT4 = Path("/home/gmwager/projects/hippo/reports/report4")

PAGE_ORDER = [
    ("story", None),
    ("Fig1", "Fig1_behavior_info.pdf"),
    ("Fig2", "Fig2_tuning_drift.pdf"),
    ("Fig3", "Fig3_spikes_behaviour.pdf"),
    ("Fig4", "Fig4_whats_right.pdf"),
    ("Fig5", "Fig5_R_todo.pdf"),
    ("FigS1", "FigS1_extras.pdf"),
    ("Appendix", "Appendix_gates.pdf"),
]
FIG_ONLY = "Fig1,Fig2,Fig3,Fig4,Fig5,FigS1,Appendix"


def _run(cmd: list[str]) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    pip_local = str(ROOT / ".pip_local")
    env = {**os.environ, "MPLBACKEND": "Agg"}
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


def _prepare() -> Path | None:
    OUT.mkdir(parents=True, exist_ok=True)
    if DATA.exists():
        shutil.rmtree(DATA)
    shutil.copytree(V5 / "data", DATA)
    if BEHINFO.exists():
        shutil.rmtree(BEHINFO)
    shutil.copytree(V5 / "behavior_info", BEHINFO)
    if STEP1.is_dir():
        for name in ("step1_session_distributions.csv", "step1_dataset_pooled.csv"):
            src = STEP1 / name
            if src.is_file():
                shutil.copy2(src, DATA / name)
    meta = json.loads((DATA / "data_meta.json").read_text())
    session = meta.get("example_session") or "1150_09102024_2rooms"
    if not (DATA / "data_rasters_1s.json").is_file():
        extract_rasters_1s(DATA, session)

    ladder = BEHINFO / "n_curve_ladder.csv"
    ladder_path = ladder if ladder.is_file() else None

    if DATA_ANON.exists():
        shutil.rmtree(DATA_ANON)
    sess_map = anonymize_data_dir(DATA, DATA_ANON)
    import pandas as pd
    for name in (
        "encoding_diag_units.csv", "decode_rows.csv", "drift_sep_sessions.csv",
        "step1_session_distributions.csv", "n_curve.csv", "n_curve_ladder.csv",
    ):
        p = BEHINFO / name if not name.startswith("step1") else DATA / name
        if name.startswith("n_curve"):
            p = BEHINFO / name
        if p.is_file() and "session" in pd.read_csv(p, nrows=1).columns:
            sess_map.update(build_session_map(pd.read_csv(p)["session"].astype(str).tolist()))
    if BEH_ANON.exists():
        shutil.rmtree(BEH_ANON)
    _anon_behinfo(BEHINFO, BEH_ANON, sess_map)
    return ladder_path


def _render(data: Path, beh: Path, figs: Path, ladder: Path | None) -> None:
    figs.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "agents.data_quality.report.figures_dense_v8",
        "--data", str(data), "--behinfo", str(beh), "--step1", str(data),
        "--out", str(figs), "--only", FIG_ONLY,
    ]
    if ladder and ladder.is_file():
        cmd.extend(["--ncurve-ladder", str(ladder)])
    _run(cmd)


def _make_pages(data: Path, beh: Path, figs: Path, pages: Path) -> list[tuple[str, Path]]:
    pages.mkdir(parents=True, exist_ok=True)
    story = pages / "00_story.pdf"
    write_story_v8(data, story)
    out: list[tuple[str, Path]] = [("story", story)]
    for key, stem in PAGE_ORDER[1:]:
        src = figs / stem
        if not src.is_file():
            print(json.dumps({"missing_fig": stem}), flush=True)
            continue
        dst = pages / f"{key}.pdf"
        place_figure_on_a4(src, dst)
        out.append((key, dst))
    return out


def _assemble(page_paths: list[tuple[str, Path]], out_pdf: Path) -> Path:
    w = PdfWriter()
    for _key, p in page_paths:
        w.add_page(PdfReader(str(p)).pages[0])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as f:
        w.write(f)
    return out_pdf


def main() -> int:
    print(json.dumps({"build_full_v8_start": True, "out": str(OUT)}), flush=True)
    ladder = _prepare()
    write_registry(OUT / "metric_registry.json")

    if FIGS.exists():
        shutil.rmtree(FIGS)
    _render(DATA, BEHINFO, FIGS, ladder)

    if FIGS_ANON.exists():
        shutil.rmtree(FIGS_ANON)
    ladder_anon = BEH_ANON / "n_curve_ladder.csv"
    _render(DATA_ANON, BEH_ANON, FIGS_ANON, ladder_anon if ladder_anon.is_file() else None)

    if PAGES.exists():
        shutil.rmtree(PAGES)
    pages = _make_pages(DATA, BEHINFO, FIGS, PAGES)
    pdf = _assemble(pages, OUT / "report4a_full.pdf")

    if PAGES_ANON.exists():
        shutil.rmtree(PAGES_ANON)
    pages_a = _make_pages(DATA_ANON, BEH_ANON, FIGS_ANON, PAGES_ANON)
    pdf_anon = _assemble(pages_a, OUT / "report4a_full_anon.pdf")

    # Audits
    png_dir = PAGES / "png"
    if png_dir.exists():
        shutil.rmtree(png_dir)
    png_dir.mkdir(parents=True)
    subprocess.check_call(["pdftoppm", "-png", "-r", "120", str(pdf), str(png_dir / "page")])
    pngs = sorted(png_dir.glob("page-*.png"))

    prose = audit_story_prose(PAGES / "00_story.pdf")
    overlap = audit_pdf(pdf, pages_png_dir=png_dir)
    num = audit(DATA, OUT / "number_audit.json", pdf)
    white = audit_pngs(pngs)
    leak = leak_check([pdf_anon, DATA_ANON / "data_meta.json", *FIGS_ANON.glob("*.pdf")])
    reader = PdfReader(str(pdf))
    keys = [k for k, _ in pages]
    page_texts = {k: (reader.pages[i].extract_text() or "") for i, k in enumerate(keys)}
    dup = audit_duplicates(page_texts)

    summary = {
        "version": "v8",
        "pdf": str(pdf),
        "pdf_anon": str(pdf_anon),
        "n_pages": len(reader.pages),
        "ladder": bool(ladder and ladder.is_file()),
        "prose_ok": prose.get("ok"),
        "prose": prose,
        "leak_ok": leak.get("ok"),
        "number_ok": num.get("ok"),
        "number_failures": (num.get("failures") or [])[:15],
        "overlap_ok": overlap.get("ok"),
        "overlap_n_failures": overlap.get("n_failures"),
        "overlap_failures": (overlap.get("failures") or [])[:15],
        "whitespace_ok": white.get("ok"),
        "whitespace_pages": [
            {k: p.get(k) for k in ("file", "blank_frac", "ok")} for p in (white.get("pages") or [])
        ],
        "duplicate_ok": dup.get("ok"),
        "duplicate_failures": dup.get("failures"),
        "delivered": False,
    }
    (OUT / "overlap_audit.json").write_text(json.dumps(overlap, indent=2, default=str) + "\n")
    (OUT / "whitespace_audit.json").write_text(json.dumps(white, indent=2) + "\n")
    (OUT / "duplicate_audit.json").write_text(json.dumps(dup, indent=2) + "\n")
    (OUT / "anon_leak_check.json").write_text(json.dumps(leak, indent=2) + "\n")
    (OUT / "prose_audit.json").write_text(json.dumps(prose, indent=2) + "\n")

    ok = all(summary[k] for k in (
        "prose_ok", "leak_ok", "number_ok", "overlap_ok", "whitespace_ok", "duplicate_ok",
    ))
    DELIVER.mkdir(parents=True, exist_ok=True)
    REPORT4.mkdir(parents=True, exist_ok=True)
    if ok:
        # reports/report4: PDFs only
        for src in (pdf, pdf_anon):
            shutil.copy2(src, REPORT4 / src.name)
            shutil.copy2(src, DELIVER / src.name)
        for p in pngs:
            shutil.copy2(p, DELIVER / p.name)
        if (png_dir / "page-01.png").is_file():
            shutil.copy2(png_dir / "page-01.png", DELIVER / "page1.png")
        elif (png_dir / "page-1.png").is_file():
            shutil.copy2(png_dir / "page-1.png", DELIVER / "page1.png")
        fig1_png = FIGS / "Fig1_behavior_info.png"
        if fig1_png.is_file():
            shutil.copy2(fig1_png, DELIVER / "Fig1_native.png")
        for name in (
            "number_audit.json", "anon_leak_check.json",
            "overlap_audit.json", "whitespace_audit.json", "duplicate_audit.json",
            "prose_audit.json",
        ):
            src = OUT / name
            if src.is_file():
                shutil.copy2(src, DELIVER / name)
        summary["delivered"] = True
    else:
        summary["deliver_blocked"] = "one or more audits failed"
        for name in (
            "number_audit.json", "anon_leak_check.json", "overlap_audit.json",
            "whitespace_audit.json", "duplicate_audit.json", "prose_audit.json",
        ):
            src = OUT / name
            if src.is_file():
                shutil.copy2(src, DELIVER / name)

    (OUT / "BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (DELIVER / "BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0 if summary.get("delivered") else 1


if __name__ == "__main__":
    raise SystemExit(main())
