#!/usr/bin/env bash
# Helper to clone / launch Andy Peters' Neuropixels Trajectory Explorer.
# Does not deeply integrate MATLAB; prints clear instructions when unavailable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DIR="${ROOT}/external/neuropixels_trajectory_explorer"
REPO_URL="https://github.com/petersaj/neuropixels_trajectory_explorer"

echo "Neuropixels Trajectory Explorer helper"
echo "Project root: ${ROOT}"
echo

if [[ ! -d "${EXT_DIR}" ]]; then
  echo "Trajectory Explorer not found under external/."
  echo "Clone it with:"
  echo
  echo "  git clone ${REPO_URL} ${EXT_DIR}"
  echo
  echo "Wiki: https://github.com/petersaj/neuropixels_trajectory_explorer/wiki"
  echo "Allen page: https://brain-map.org/community-partnership/community-tools/neuropixels-trajectory-explorer"
  exit 1
fi

echo "Found: ${EXT_DIR}"

if command -v matlab >/dev/null 2>&1; then
  echo "MATLAB detected. Attempting to launch neuropixels_trajectory_explorer..."
  # Prefer the main entry script if present.
  ENTRY=""
  for candidate in \
      "neuropixels_trajectory_explorer.m" \
      "neuropixels_trajectory_explorer/neuropixels_trajectory_explorer.m"
  do
    if [[ -f "${EXT_DIR}/${candidate}" ]]; then
      ENTRY="${candidate}"
      break
    fi
  done
  if [[ -z "${ENTRY}" ]]; then
    ENTRY="$(find "${EXT_DIR}" -maxdepth 2 -name 'neuropixels_trajectory_explorer*.m' | head -n 1 || true)"
  fi
  if [[ -n "${ENTRY}" ]]; then
    ENTRY_DIR="$(cd "$(dirname "${EXT_DIR}/${ENTRY}")" && pwd)"
    ENTRY_NAME="$(basename "${ENTRY}")"
    ENTRY_FUNC="${ENTRY_NAME%.m}"
    matlab -nodisplay -nosplash -nodesktop -r "addpath('${ENTRY_DIR}'); try, ${ENTRY_FUNC}; catch ME, disp(getReport(ME)); end"
  else
    echo "Could not find neuropixels_trajectory_explorer*.m — open MATLAB and run the GUI from:"
    echo "  ${EXT_DIR}"
  fi
else
  echo "MATLAB not on PATH."
  echo
  echo "Manual launch:"
  echo "  1. Open MATLAB (or the standalone app shipped with the repo, if available)"
  echo "  2. addpath('${EXT_DIR}')  # or the folder containing the .m entry script"
  echo "  3. Run neuropixels_trajectory_explorer"
  echo "  4. Set your lab insertion AP/ML/depth/angles"
  echo "  5. Inspect the path through Allen CCF"
  echo "  6. Save/export the trajectory (File → Save in the GUI writes a .mat),"
  echo "     or manually write a region-depth CSV matching:"
  echo "       data/probe_trajectories/lab_insertion_001.csv"
  echo "  7. Point this project at the export:"
  echo
  echo "  python run_simulation.py \\"
  echo "    --output outputs/lab_trajectory_001 \\"
  echo "    --trajectory-config configs/probe_trajectory.yaml \\"
  echo "    --trajectory-export data/probe_trajectories/lab_insertion_001.csv \\"
  echo "    --cell-capture-config configs/cell_capture.yaml"
fi
