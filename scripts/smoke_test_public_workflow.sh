#!/usr/bin/env bash
# Short public-workflow smoke test matching the README happy path.
# Validates simulation → decode (sorted deployable) → visualize contracts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .hippo/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .hippo/bin/activate
fi

OUT="${SMOKE_OUT:-outputs/smoke_public}"
export SMOKE_OUT="$OUT"
rm -rf "$OUT"

echo "=== [1/3] Simulation ==="
python run_simulation.py \
  --output "$OUT" \
  --trajectory lab_npx2_default \
  --seed 1 \
  --duration 10

python - <<PY
from pathlib import Path
from realtime.output_contracts import assert_simulation_outputs
assert_simulation_outputs(Path("$OUT"))
print("simulation contract OK")
PY

echo "=== [2/3] Decode (quick profile, skip in-decode viz) ==="
python run_decoder.py \
  --input "$OUT" \
  --output "$OUT" \
  --profile quick \
  --skip-visualization

python - <<PY
from pathlib import Path
from realtime.output_contracts import assert_decode_outputs
assert_decode_outputs(Path("$OUT"))
# Sorted-only registry
import json
reg = json.loads((Path("$OUT") / "models" / "best_realtime_decoders.json").read_text())
assert reg.get("spike_source") == "sorted"
assert reg.get("deployable") is True
for t, cfg in reg.get("targets", {}).items():
    assert cfg.get("spike_source") == "sorted"
    assert cfg.get("oracle_non_deployable") is False
    assert "global_isomap" not in str(cfg.get("selected_feature_mode"))
print("decode/deployment contract OK (sorted-only, no classic Isomap)")
PY

echo "=== [3/3] Visualizations (read-only) ==="
# Snapshot mtimes of model/comparison artifacts before viz
python - <<'PY'
from pathlib import Path
import json, os, time
out = Path(os.environ.get("SMOKE_OUT", "outputs/smoke_public"))
watch = [
    out / "models" / "best_realtime_decoders.json",
    out / "decoder_comparison" / "sorted" / "decoder_comparison_metrics.csv",
    out / "deployment_decoder_selection" / "all_sorted_window_scores.csv",
]
stamp = {str(p): p.stat().st_mtime_ns for p in watch if p.exists()}
(out / ".smoke_pre_viz_mtimes.json").write_text(json.dumps(stamp))
print(f"recorded {len(stamp)} artifact mtimes before viz")
PY

python run_visualizations.py \
  --experiment "$OUT" \
  --all \
  --compile-pdf

python - <<'PY'
from pathlib import Path
import json, os
from realtime.output_contracts import assert_visualization_outputs
out = Path(os.environ.get("SMOKE_OUT", "outputs/smoke_public"))
assert_visualization_outputs(out)
stamp = json.loads((out / ".smoke_pre_viz_mtimes.json").read_text())
for path, mtime in stamp.items():
    p = Path(path)
    assert p.exists(), path
    assert p.stat().st_mtime_ns == mtime, f"viz mutated {path}"
print("visualization contract OK (PDF present; no model/comparison mutation)")
PY

echo "=== smoke_test_public_workflow PASSED ==="
