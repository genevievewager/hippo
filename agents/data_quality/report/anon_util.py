#!/usr/bin/env python3
"""Anonymise animal IDs / session names for Report 4a delivery + leak check."""

from __future__ import annotations

import json
import re
from pathlib import Path

from agents.data_quality.report.extract_tidy import ANIMAL_CODES

SESSION_RE = re.compile(r"\b(\d{4})_(\d{8})(?:_\w+)?\b")
ANIMAL_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in ANIMAL_CODES) + r")\b")


def build_session_map(sessions: list[str]) -> dict[str, str]:
    """Map raw session ids → A_S1, B_S2, … (per-animal counters)."""
    counters: dict[str, int] = {}
    out: dict[str, str] = {}
    for s in sorted(set(sessions)):
        if not s or str(s).startswith("seed_"):
            continue
        animal = str(s).split("_", 1)[0]
        code = ANIMAL_CODES.get(animal, animal)
        counters[code] = counters.get(code, 0) + 1
        out[str(s)] = f"{code}_S{counters[code]}"
    return out


def _rewrite_str(s: str, sess_map: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        full = m.group(0)
        if full in sess_map:
            return sess_map[full]
        # bare animal_date without trailing token — still map if known prefix
        for raw, anon in sess_map.items():
            if full == raw or full.startswith(raw.split("_")[0] + "_"):
                # try exact
                if full in sess_map:
                    return sess_map[full]
        animal = m.group(1)
        code = ANIMAL_CODES.get(animal, animal)
        # unknown full session: animal code + S?
        return f"{code}_S?"

    s2 = SESSION_RE.sub(repl, s)
    # remaining bare animal ids
    s2 = ANIMAL_RE.sub(lambda m: ANIMAL_CODES.get(m.group(1), m.group(1)), s2)
    return s2


def _walk(obj, sess_map: dict[str, str]):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = _rewrite_str(str(k), sess_map) if SESSION_RE.search(str(k)) or ANIMAL_RE.search(str(k)) else k
            # remapping per_animal keys
            if k in ANIMAL_CODES:
                nk = ANIMAL_CODES[k]
            out[nk] = _walk(v, sess_map)
        return out
    if isinstance(obj, list):
        return [_walk(x, sess_map) for x in obj]
    if isinstance(obj, str):
        return _rewrite_str(obj, sess_map)
    return obj


def collect_sessions_from_meta(meta: dict) -> list[str]:
    found: list[str] = []
    raw = json.dumps(meta)
    found.extend(m.group(0) for m in SESSION_RE.finditer(raw))
    ex = meta.get("example_session")
    if ex:
        found.append(str(ex))
    q3 = meta.get("q3") or {}
    for r in q3.get("path_fail_matched") or []:
        if r.get("session"):
            found.append(str(r["session"]))
    neu = meta.get("boost_neutrality") or {}
    for s in neu.get("sessions") or []:
        if isinstance(s, dict) and s.get("session"):
            found.append(str(s["session"]))
    return found


def anonymize_meta(meta: dict, sess_map: dict[str, str] | None = None) -> dict:
    if sess_map is None:
        sess_map = build_session_map(collect_sessions_from_meta(meta))
    out = _walk(meta, sess_map)
    out["anon"] = True
    # Do not retain raw animal/session keys (leak check scans JSON text).
    out["animal_codes"] = {v: v for v in ANIMAL_CODES.values()}
    out.pop("session_map", None)
    return out


def anonymize_data_dir(src: Path, dst: Path) -> dict[str, str]:
    """Copy CSVs/JSON from src→dst with animal/session ids rewritten."""
    import shutil

    import pandas as pd

    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "data_meta.json").read_text())
    sess_map = build_session_map(collect_sessions_from_meta(meta))
    # also pull sessions from session metrics
    sess_csv = src / "data_session_metrics.csv"
    if sess_csv.is_file():
        df = pd.read_csv(sess_csv)
        if "session" in df.columns:
            sess_map.update(build_session_map(df["session"].astype(str).tolist()))

    for p in src.iterdir():
        if not p.is_file():
            continue
        if p.suffix == ".json":
            raw = json.loads(p.read_text())
            if p.name == "data_meta.json":
                (dst / p.name).write_text(json.dumps(anonymize_meta(raw, sess_map), indent=2, default=str) + "\n")
            else:
                (dst / p.name).write_text(json.dumps(_walk(raw, sess_map), indent=2, default=str) + "\n")
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].map(
                        lambda v: _rewrite_str(str(v), sess_map) if isinstance(v, str) else v
                    )
            if "animal" in df.columns:
                df["animal"] = df["animal"].map(
                    lambda a: ANIMAL_CODES.get(str(a), str(a))
                )
            df.to_csv(dst / p.name, index=False)
        else:
            shutil.copy2(p, dst / p.name)
    return sess_map


def leak_check(paths: list[Path]) -> dict:
    """Fail if any file still contains raw session or animal ID patterns."""
    animal_pat = re.compile(r"\b(" + "|".join(re.escape(a) for a in ANIMAL_CODES) + r")\b")
    sess_pat = re.compile(r"\b\d{4}_\d{8}\b")
    hits = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            # binary PDF: extract with pypdf if possible
            if p.suffix.lower() == ".pdf":
                try:
                    from pypdf import PdfReader
                    text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
                except Exception:
                    continue
            else:
                continue
        for m in sess_pat.finditer(text):
            hits.append({"file": str(p), "kind": "session", "match": m.group(0)})
        for m in animal_pat.finditer(text):
            # allow if already anonymised context? bare digits animal ids are leaks
            hits.append({"file": str(p), "kind": "animal", "match": m.group(0)})
    return {"ok": len(hits) == 0, "n_hits": len(hits), "hits": hits[:50]}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="+", help="Files/dirs to leak-check")
    args = ap.parse_args()
    paths: list[Path] = []
    for c in args.check or []:
        p = Path(c)
        if p.is_dir():
            paths.extend(p.rglob("*"))
        else:
            paths.append(p)
    res = leak_check(paths)
    print(json.dumps(res, indent=2))
    raise SystemExit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
