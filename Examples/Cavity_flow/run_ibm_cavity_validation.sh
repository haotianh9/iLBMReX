#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p _artifacts
rm -rf out_bc out_ibm

if ! ls ./main2d.*.ex >/dev/null 2>&1; then
  make -j12
fi

exe="$(ls -1 ./main2d.*.ex | head -n1)"

export OMP_NUM_THREADS=1

mpirun -n 12 "${exe}" inputs_bc_ref > _artifacts/cavity_bc.log 2>&1
mpirun -n 12 "${exe}" inputs_ibm_box > _artifacts/cavity_ibm.log 2>&1

python3 compare_ibm_vs_bc.py --bc-root out_bc --ibm-root out_ibm --json-out _artifacts/cavity_ibm_vs_bc.json > _artifacts/cavity_compare.log 2>&1

echo "Validation complete. See _artifacts/cavity_ibm_vs_bc.json"
