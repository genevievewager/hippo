#!/usr/bin/env python3
"""Post-checkout smoke + Q3 aggregates (M3 eligibility)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from analysis.real_quadrant.session_select import select_m3_cohort
from agents.data_quality.io.adapter_wrap import build_segment_bundle_dq
from agents.data_quality.io.analyze_wrap import patch_output_root
from agents.data_quality.io.paths import DATA_QUALITY_ROOT
from agents.data_quality.matching.session_discovery import eligibility_by_animal
import analysis.real_quadrant.adapter as adapter
import realtime.quadrant_n5_run as qrun


def main() -> None:
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    out = DATA_QUALITY_ROOT / "_phase0" / "post_b68606e_smoke_q3.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "adapter_file": adapter.__file__,
        "qrun_file": qrun.__file__,
        "has_ratemap_gate": hasattr(adapter, "RateMapStabilityError"),
        "OUTPUT_ROOT_before": str(qrun.OUTPUT_ROOT),
    }
    assert adapter.__file__.startswith("/home/gmwager/projects/hippo-dq/"), adapter.__file__
    assert qrun.__file__.startswith("/home/gmwager/projects/hippo-dq/"), qrun.__file__

    with patch_output_root() as dest:
        report["OUTPUT_ROOT_patched"] = str(qrun.OUTPUT_ROOT)
        report["analyze_wrap_ok"] = str(qrun.OUTPUT_ROOT).startswith(str(DATA_QUALITY_ROOT))
        assert report["analyze_wrap_ok"], qrun.OUTPUT_ROOT
    report["OUTPUT_ROOT_after"] = str(qrun.OUTPUT_ROOT)

    # M3 cohort (2/animal) — smoke uses one selected session
    cohort = select_m3_cohort(root)
    report["m3_cohort"] = {
        "n_selected": cohort["n_selected_sessions"],
        "n_animals": cohort["n_animals"],
        "selected": cohort["selected_sessions"],
    }
    primary = [r["session"] for r in cohort["selected_sessions"]]
    smoke_session = primary[0] if primary else "1150_27102024_2rooms"
    bundle = build_segment_bundle_dq(smoke_session, "A", require_ratemap_stability=True)
    meta = bundle.get("meta") if isinstance(bundle.get("meta"), dict) else {}
    report["smoke"] = {
        "session": smoke_session,
        "n_units": int(len(bundle["unit_ids"])),
        "segment_t0": float(bundle["segment_t0"]),
        "segment_t1": float(bundle["segment_t1"]),
        "valid_frac": float(meta.get("valid_frac_retained", float("nan"))),
        "ratemap_flagged": bool(
            (meta.get("ratemap_stability") or {}).get("flagged", False)
        ),
    }

    # Q3: full eligible set (not just M3's 2/animal) via imported session_select
    elig = eligibility_by_animal(root)
    report["q3_eligibility_by_animal"] = {
        "n_2rooms": elig["n_2rooms"],
        "n_animals": elig["n_animals"],
        "n_eligible": elig["n_eligible"],
        "per_animal": [
            {
                "animal": a["animal"],
                "total": a["total"],
                "eligible": a["eligible"],
                "excluded": a["excluded"],
                "exclude_reasons": a["exclude_reasons"],
            }
            for a in elig["animals"]
        ],
        "note": "Must match M3 log eligible/excluded counts (all eligible, not 2/animal cohort)",
    }
    # Write full eligibility artifact too
    (DATA_QUALITY_ROOT / "_phase0" / "eligibility_by_animal.json").write_text(
        json.dumps(elig, indent=2, default=str) + "\n"
    )

    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "smoke_q3_ok": True,
                "out": str(out),
                "n_m3_selected": len(primary),
                "n_eligible_all": elig["n_eligible"],
                "n_animals": elig["n_animals"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
