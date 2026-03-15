#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Dict, Tuple

import numpy as np
import yt


def latest_plotfile(root: str) -> str:
    cands = [p for p in glob.glob(os.path.join(root, 'plt*')) if os.path.isdir(p)]
    if not cands:
        raise FileNotFoundError(f'No plotfiles found under {root}')
    cands.sort()
    return cands[-1]


def load_midplane_field(plotfile: str, field: str) -> np.ndarray:
    ds = yt.load(plotfile)
    dims = tuple(int(v) for v in ds.domain_dimensions)
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    arr = cg[("boxlib", field)].to_ndarray()

    if arr.ndim == 3:
        k = arr.shape[2] // 2
        arr2 = arr[:, :, k]
    elif arr.ndim == 2:
        arr2 = arr
    else:
        raise RuntimeError(f'Unexpected field shape for {field}: {arr.shape}')

    return np.asarray(arr2, dtype=np.float64)


def rel_err(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    d = a - b
    l2 = np.linalg.norm(d)
    l2_ref = max(np.linalg.norm(b), 1.0e-14)
    linf = np.max(np.abs(d))
    linf_ref = max(np.max(np.abs(b)), 1.0e-14)
    return float(l2 / l2_ref), float(linf / linf_ref)


def compare(bc_plot: str, ibm_plot: str) -> Dict[str, float]:
    ux_bc = load_midplane_field(bc_plot, 'ux')
    uy_bc = load_midplane_field(bc_plot, 'uy')
    ux_ibm = load_midplane_field(ibm_plot, 'ux')
    uy_ibm = load_midplane_field(ibm_plot, 'uy')

    if ux_bc.shape != ux_ibm.shape:
        # Overset IBM cavity support:
        # BC reference may be 64x64 while IBM domain can be taller (e.g. 64x96)
        # with an immersed plate at y=64. Compare only the lower cavity region.
        nxb, nyb = ux_bc.shape
        nxi, nyi = ux_ibm.shape
        if nxb == nxi and nyi >= nyb:
            ux_ibm = ux_ibm[:, :nyb]
            uy_ibm = uy_ibm[:, :nyb]
        else:
            raise RuntimeError(f'Shape mismatch: BC {ux_bc.shape}, IBM {ux_ibm.shape}')

    nx, ny = ux_bc.shape
    ix = nx // 2
    iy = ny // 2

    # Cavity conventions:
    # u(y) on vertical centerline x=L/2
    # v(x) on horizontal centerline y=L/2
    u_vert_bc = ux_bc[ix, :]
    u_vert_ibm = ux_ibm[ix, :]
    v_hori_bc = uy_bc[:, iy]
    v_hori_ibm = uy_ibm[:, iy]

    u_l2, u_linf = rel_err(u_vert_ibm, u_vert_bc)
    v_l2, v_linf = rel_err(v_hori_ibm, v_hori_bc)

    out = {
        'nx': int(nx),
        'ny': int(ny),
        'u_centerline_l2_rel': u_l2,
        'u_centerline_linf_rel': u_linf,
        'v_centerline_l2_rel': v_l2,
        'v_centerline_linf_rel': v_linf,
        'u_center_bc': float(ux_bc[ix, iy]),
        'u_center_ibm': float(ux_ibm[ix, iy]),
        'v_center_bc': float(uy_bc[ix, iy]),
        'v_center_ibm': float(uy_ibm[ix, iy]),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare IBM cavity vs BC cavity centerlines.')
    parser.add_argument('--bc-root', default='out_bc', help='Directory containing BC plotfiles (default: out_bc)')
    parser.add_argument('--ibm-root', default='out_ibm', help='Directory containing IBM plotfiles (default: out_ibm)')
    parser.add_argument('--json-out', default='_artifacts/cavity_ibm_vs_bc.json', help='Output JSON summary path')
    args = parser.parse_args()

    bc_plot = latest_plotfile(args.bc_root)
    ibm_plot = latest_plotfile(args.ibm_root)

    metrics = compare(bc_plot, ibm_plot)
    metrics['bc_plot'] = bc_plot
    metrics['ibm_plot'] = ibm_plot

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
