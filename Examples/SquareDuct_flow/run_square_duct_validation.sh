#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS=4
STEPS=6000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      JOBS="$2"; shift 2;;
    --steps)
      STEPS="$2"; shift 2;;
    *)
      echo "Unknown option: $1" >&2
      exit 2;;
  esac
done

ART_DIR="${SCRIPT_DIR}/_artifacts"
CASE_DIR="${ART_DIR}/cases"

(
  cd "${SCRIPT_DIR}"
  python3 - <<'PY'
from pathlib import Path
import shutil

root = Path("_artifacts")
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True, exist_ok=True)
(root / "cases").mkdir(parents=True, exist_ok=True)
for rel in ["Backtrace.0", "__pycache__", "main3d.gnu.ex", "tmp_build_dir"]:
    path = Path(rel)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
PY

  make -j"${JOBS}" DEBUG=FALSE USE_MPI=FALSE USE_OMP=FALSE \
    > "${ART_DIR}/square_duct_build.log" 2>&1

  ./main3d.gnu.ex inputs \
    amr.plot_file="${CASE_DIR}/32x32/plt" \
    amr.plot_int=1000 amr.chk_int=-1 \
    max_step="${STEPS}" stop_time="${STEPS}" \
    > "${ART_DIR}/square_duct_run_32x32.log" 2>&1

  ./main3d.gnu.ex inputs \
    amr.n_cell='12 48 48' \
    amr.blocking_factor_x=4 \
    amr.max_grid_size=48 \
    lbm.prescribed_force='6.666666666666667e-6 0.0 0.0' \
    amr.plot_file="${CASE_DIR}/48x48/plt" \
    amr.plot_int=1500 amr.chk_int=-1 \
    max_step="$((2 * STEPS))" stop_time="${STEPS}" \
    > "${ART_DIR}/square_duct_run_48x48.log" 2>&1
)

cat <<EOF
Square-duct runs completed.

Raw cases:
  ${CASE_DIR}/32x32
  ${CASE_DIR}/48x48

Open:
  ${SCRIPT_DIR}/Visualization.ipynb

Optional CLI validation:
  python3 validate_square_duct.py --inputs inputs --plot-root ${CASE_DIR}/32x32
  python3 validate_square_duct.py --inputs inputs --plot-root ${CASE_DIR}/48x48 --exact-force-x 1.0e-5
EOF
