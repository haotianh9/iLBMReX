#!/usr/bin/env bash
set -euo pipefail

# Test ladder:
#   T1: pure LBM forcing sanity (no IBM)          -> Examples/ForceValidation
#   T2: short cylinder stability sanity           -> this folder
#   T3: long Re=100 cylinder validation gate      -> this folder
#   T4: long Re=200 secondary trend check         -> this folder
#
# Usage examples:
#   ./validate_cylinder.sh
#   ./validate_cylinder.sh --quick
#   ./validate_cylinder.sh --nprocs 12 --omp 1 --jobs 12

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FV_DIR="${ROOT_DIR}/Examples/ForceValidation"
CYL_DIR="${SCRIPT_DIR}"

NPROCS=12
OMP_THREADS=1
JOBS=12
QUICK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nprocs)
      NPROCS="$2"; shift 2;;
    --omp)
      OMP_THREADS="$2"; shift 2;;
    --jobs)
      JOBS="$2"; shift 2;;
    --quick)
      QUICK=1; shift;;
    *)
      echo "Unknown option: $1" >&2
      exit 2;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
ART_DIR="${CYL_DIR}/_artifacts/test_ladder_${TS}"
mkdir -p "${ART_DIR}"

echo "[test_ladder] root=${ROOT_DIR}"
echo "[test_ladder] artifacts=${ART_DIR}"
echo "[test_ladder] nprocs=${NPROCS} omp=${OMP_THREADS} jobs=${JOBS} quick=${QUICK}"

fail() {
  echo "[test_ladder][FAIL] $*" >&2
  exit 1
}

require_file() {
  local f="$1"
  [[ -f "${f}" ]] || fail "missing file: ${f}"
}

#############################################
# T1: pure forcing sanity (no IBM)
#############################################
echo "[T1] Build ForceValidation"
(
  cd "${FV_DIR}"
  make -j"${JOBS}" DEBUG=FALSE > "${ART_DIR}/t1_build.log" 2>&1
)

echo "[T1] Run ForceValidation"
(
  cd "${FV_DIR}"
  OMP_NUM_THREADS="${OMP_THREADS}" mpirun -n "${NPROCS}" ./main2d.gnu.MPI.ex inputs \
    > "${ART_DIR}/t1_run.log" 2>&1
)

if rg -qi "nan|abort|error" "${ART_DIR}/t1_run.log"; then
  fail "T1 detected NaN/Abort/Error in run log"
fi

python3 - "${ART_DIR}/t1_run.log" > "${ART_DIR}/t1_metrics.txt" <<'PY'
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text()
pat = re.compile(r"<u>=([0-9eE+\-.]+)\s+pred=([0-9eE+\-.]+)")
errs = []
for m in pat.finditer(log):
    u = float(m.group(1))
    p = float(m.group(2))
    errs.append(abs(u - p))
if not errs:
    print("samples=0")
    print("max_abs_err=nan")
    sys.exit(3)
mx = max(errs)
mean = sum(errs) / len(errs)
print(f"samples={len(errs)}")
print(f"max_abs_err={mx:.12e}")
print(f"mean_abs_err={mean:.12e}")
# Gate: forcing response should be very close to prediction.
sys.exit(0 if mx <= 2.0e-7 else 4)
PY
case $? in
  0) ;;
  3) fail "T1 could not parse force_validation output";;
  4) fail "T1 forcing error exceeds threshold (max_abs_err > 2e-7)";;
  *) fail "T1 parser failed";;
esac

echo "[T1] PASS"

#############################################
# T2: short cylinder stability sanity
#############################################
echo "[T2] Build Cylinder flow case"
(
  cd "${CYL_DIR}"
  make -j"${JOBS}" > "${ART_DIR}/t2_build.log" 2>&1
)

echo "[T2] Run short cylinder sanity (2000 steps)"
(
  cd "${CYL_DIR}"
  rm -f force.dat
  OMP_NUM_THREADS="${OMP_THREADS}" mpirun -n "${NPROCS}" ./main2d.gnu.MPI.OMP.ex inputs \
    max_step=2000 stop_time=2000 amr.plot_int=-1 \
    > "${ART_DIR}/t2_run.log" 2>&1
  cp force.dat "${ART_DIR}/t2_force.dat"
)

require_file "${ART_DIR}/t2_force.dat"
rg -q "Coarse STEP 2000 ends" "${ART_DIR}/t2_run.log" || fail "T2 did not reach step 2000"
if rg -qi "nan|abort|error" "${ART_DIR}/t2_run.log"; then
  fail "T2 detected NaN/Abort/Error in run log"
fi

(
  cd "${CYL_DIR}"
  python3 analyze_force.py --force "${ART_DIR}/t2_force.dat" --discard-frac 0.5 \
    > "${ART_DIR}/t2_analyze.txt"
)

echo "[T2] PASS"

#############################################
# T3: long Re=100 validation gate (primary)
#############################################
if [[ "${QUICK}" -eq 0 ]]; then
  echo "[T3] Run long cylinder validation (50000 steps, Re=100 default)"
  (
    cd "${CYL_DIR}"
    rm -f force.dat
    OMP_NUM_THREADS="${OMP_THREADS}" mpirun -n "${NPROCS}" ./main2d.gnu.MPI.OMP.ex inputs \
      max_step=50000 stop_time=50000 amr.plot_int=-1 \
      > "${ART_DIR}/t3_run.log" 2>&1
    cp force.dat "${ART_DIR}/t3_force.dat"
  )

  require_file "${ART_DIR}/t3_force.dat"
  rg -q "Coarse STEP 50000 ends" "${ART_DIR}/t3_run.log" || fail "T3 did not reach step 50000"
  if rg -qi "nan|abort|error" "${ART_DIR}/t3_run.log"; then
    fail "T3 detected NaN/Abort/Error in run log"
  fi

  (
    cd "${CYL_DIR}"
    python3 analyze_force.py --force "${ART_DIR}/t3_force.dat" --source me --discard-frac 0.5 \
      > "${ART_DIR}/t3_analyze.txt"
  )

  rg -q "match_Cd=True match_St=True" "${ART_DIR}/t3_analyze.txt" || fail "T3 metric gate failed"
  rg -q "periodic_lift=True" "${ART_DIR}/t3_analyze.txt" || fail "T3 periodic-lift gate failed"
  echo "[T3] PASS"

  #############################################
  # T4: long Re=200 secondary trend check
  #############################################
  echo "[T4] Run long cylinder trend check (50000 steps, Re=200 override)"
  (
    cd "${CYL_DIR}"
    rm -f force.dat
    OMP_NUM_THREADS="${OMP_THREADS}" mpirun -n "${NPROCS}" ./main2d.gnu.MPI.OMP.ex inputs \
      lbmPhysicalParameters.nu=0.0048 max_step=50000 stop_time=50000 amr.plot_int=-1 \
      > "${ART_DIR}/t4_run.log" 2>&1
    cp force.dat "${ART_DIR}/t4_force.dat"
  )

  require_file "${ART_DIR}/t4_force.dat"
  rg -q "Coarse STEP 50000 ends" "${ART_DIR}/t4_run.log" || fail "T4 did not reach step 50000"
  if rg -qi "nan|abort|error" "${ART_DIR}/t4_run.log"; then
    fail "T4 detected NaN/Abort/Error in run log"
  fi

  (
    cd "${CYL_DIR}"
    python3 analyze_force.py --force "${ART_DIR}/t4_force.dat" --source me --discard-frac 0.5 \
      --cd-min 1.2 --cd-max 2.2 --st-min 0.18 --st-max 0.24 \
      > "${ART_DIR}/t4_analyze.txt"
  )
  echo "[T4] COMPLETE (report-only secondary check)"
else
  echo "[T3/T4] SKIPPED (--quick)"
fi

echo
echo "[test_ladder] COMPLETE"
echo "[test_ladder] artifacts=${ART_DIR}"
echo "[test_ladder] key files:"
echo "  ${ART_DIR}/t1_metrics.txt"
echo "  ${ART_DIR}/t2_analyze.txt"
if [[ "${QUICK}" -eq 0 ]]; then
  echo "  ${ART_DIR}/t3_analyze.txt"
  echo "  ${ART_DIR}/t4_analyze.txt"
fi
