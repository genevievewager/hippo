#!/usr/bin/env python3
"""Generate matched sims for eligible room-A segments (n_jobs capped).

Discovery = Report 2 ``list_2room_sessions`` + ``_passes_room_a_exclusions``.
Queue ordered longest retained duration first. Resumable (skips yield_ok).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.session_discovery import eligible_room_a_sessions


TRIM_START_S = 60.0
TRIM_END_S = 10.0


def _mem_available_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def _retained_s(session: str, data_root: Path) -> float:
    cfg = yaml.safe_load((data_root / session / "config.yaml").read_text())
    t0, t1 = [float(x) for x in cfg["preprocessing"]["map_rooms"]["rooms"]["A"]["range"]]
    return max(0.0, (t1 - t0) - TRIM_START_S - TRIM_END_S)


def _recording_day(session: str) -> str:
    parts = session.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return session


def _one(session: str) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ["DQ_PILOT_SESSION"] = session
    from agents.data_quality.run_matched_pilot import main as pilot_main

    t0 = time.perf_counter()
    pilot_main()
    report = json.loads(
        (DATA_QUALITY_ROOT / "matched" / f"{session}_A" / "pilot_report.json").read_text()
    )
    report["fleet_wall_s"] = time.perf_counter() - t0
    report["recording_day"] = _recording_day(session)
    return report


def _yield_ok_on_disk(session: str) -> bool:
    rep = DATA_QUALITY_ROOT / "matched" / f"{session}_A" / "pilot_report.json"
    if not rep.is_file():
        return False
    d = json.loads(rep.read_text())
    yo = d.get("yield_ok") or {}
    if not (yo.get("MEC") and yo.get("SUB")):
        return False
    req = int(d.get("required_counts", {}).get("MEC", 0))
    return int(d.get("sim_yield", {}).get("MEC", 0)) >= req


def main() -> None:
    n_jobs = int(os.environ.get("DQ_N_JOBS", "2"))
    mem_floor = float(os.environ.get("DQ_MEM_FLOOR_GIB", "189"))
    data_root = Path(os.environ["HIPPO_DATA_ROOT"])
    sessions = eligible_room_a_sessions()

    todo = [s for s in sessions if not _yield_ok_on_disk(s)]
    todo.sort(key=lambda s: _retained_s(s, data_root), reverse=True)

    out_path = assert_dq_path(DATA_QUALITY_ROOT / "matched" / "fleet_status.json")
    prior_done = []
    if out_path.is_file():
        try:
            prior = json.loads(out_path.read_text())
            prior_done = list(prior.get("done") or [])
        except Exception:
            prior_done = []
    have = {d.get("session") for d in prior_done}
    for s in sessions:
        if s in have or s in todo:
            continue
        if _yield_ok_on_disk(s):
            rep = json.loads(
                (DATA_QUALITY_ROOT / "matched" / f"{s}_A" / "pilot_report.json").read_text()
            )
            prior_done.append(
                {
                    "session": s,
                    "gen_s": rep.get("generation_wall_s"),
                    "yield_ok": rep.get("yield_ok"),
                    "sim_yield": rep.get("sim_yield"),
                    "recording_day": _recording_day(s),
                    "from_disk": True,
                }
            )

    status = {
        "n_total_eligible": len(sessions),
        "n_todo": len(todo),
        "n_jobs": n_jobs,
        "done": prior_done,
        "failed": [],
        "todo_ordered_longest_first": [
            {
                "session": s,
                "retained_s": _retained_s(s, data_root),
                "recording_day": _recording_day(s),
            }
            for s in todo
        ],
    }
    print(
        json.dumps(
            {
                "start_todo": len(todo),
                "n_jobs": n_jobs,
                "first": todo[:3],
                "n_done_seeded": len(prior_done),
            }
        ),
        flush=True,
    )
    out_path.write_text(json.dumps(status, indent=2) + "\n")

    if not todo:
        status["message"] = "nothing_to_do"
        out_path.write_text(json.dumps(status, indent=2) + "\n")
        return

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_one, s): s for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            avail = _mem_available_gib()
            if avail < mem_floor:
                status["failed"].append(
                    {"session": s, "error": f"mem_floor {avail:.1f}<{mem_floor}"}
                )
                out_path.write_text(json.dumps(status, indent=2) + "\n")
                raise SystemExit(f"memory watchdog abort: {avail:.1f} GiB")
            try:
                rep = fut.result()
                status["done"].append(
                    {
                        "session": s,
                        "gen_s": rep.get("generation_wall_s") or rep.get("fleet_wall_s"),
                        "yield_ok": rep.get("yield_ok"),
                        "sim_yield": rep.get("sim_yield"),
                        "recording_day": rep.get("recording_day") or _recording_day(s),
                    }
                )
                print(json.dumps({"finished": s, "ok": rep.get("yield_ok")}), flush=True)
            except Exception as e:
                status["failed"].append({"session": s, "error": f"{type(e).__name__}: {e}"})
                print(json.dumps({"failed": s, "error": str(e)}), flush=True)
            out_path.write_text(json.dumps(status, indent=2) + "\n")

    status["n_done"] = len(status["done"])
    status["n_failed"] = len(status["failed"])
    out_path.write_text(json.dumps(status, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fleet_complete": True,
                "n_done": len(status["done"]),
                "n_failed": len(status["failed"]),
            }
        )
    )


if __name__ == "__main__":
    main()
