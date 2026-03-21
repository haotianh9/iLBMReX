#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

try:
    import yt
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit("validate_square_duct.py requires yt to read AMReX plotfiles") from exc


def latest_plotfile(root: str) -> str:
    cands = [
        str(path) for path in sorted(Path(root).glob("plt*"))
        if path.is_dir() and path.name[3:].isdigit()
    ]
    if not cands:
        raise FileNotFoundError(f"No plotfiles found under {root}")
    return cands[-1]


def parse_inputs(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            lhs, rhs = [part.strip() for part in line.split("=", 1)]
            vals: list[float] = []
            for tok in rhs.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
            out[lhs] = vals
    return out


def load_fields(plotfile: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = yt.load(plotfile)
    dims = tuple(int(v) for v in ds.domain_dimensions)
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)

    rho = np.asarray(cg[("boxlib", "rho")].to_ndarray(), dtype=np.float64)
    ux = np.asarray(cg[("boxlib", "ux")].to_ndarray(), dtype=np.float64)

    left = np.asarray(ds.domain_left_edge, dtype=np.float64)
    right = np.asarray(ds.domain_right_edge, dtype=np.float64)
    dims_arr = np.asarray(dims, dtype=np.int64)
    dx = (right - left) / dims_arr

    return rho, ux, left, right, dx


def square_duct_exact(y: np.ndarray,
                      z: np.ndarray,
                      a: float,
                      force_x: float,
                      mu: float,
                      n_terms: int) -> np.ndarray:
    yy = y[:, None]
    zz = z[None, :]
    out = np.zeros((len(y), len(z)), dtype=np.float64)
    pref = 16.0 * force_x * a * a / (mu * math.pi ** 3)

    for n in range(1, 2 * n_terms, 2):
        alpha = n * math.pi / (2.0 * a)
        x = alpha * a
        # Overflow-safe evaluation of cosh(alpha z) / cosh(alpha a).
        ratio = (
            np.exp(alpha * (zz - a)) + np.exp(-alpha * (zz + a))
        ) / (1.0 + np.exp(-2.0 * x))
        sign = -1.0 if (((n - 1) // 2) % 2) else 1.0
        out += sign * (1.0 / (n ** 3)) * (1.0 - ratio) * np.cos(alpha * yy)

    return pref * out


def centerline_from_plane(uavg: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, float]:
    k = int(np.argmin(np.abs(z)))
    return uavg[:, k], float(z[k])


def wall_slip_metrics(y: np.ndarray,
                      centerline_num: np.ndarray,
                      centerline_exact: np.ndarray,
                      a: float) -> dict[str, float]:
    y_wall_lo = -a
    y_wall_hi = a

    slope_num_lo = float((centerline_num[1] - centerline_num[0]) / (y[1] - y[0]))
    slope_ex_lo = float((centerline_exact[1] - centerline_exact[0]) / (y[1] - y[0]))
    u_wall_num_lo = float(centerline_num[0] + slope_num_lo * (y_wall_lo - y[0]))
    u_wall_ex_lo = float(centerline_exact[0] + slope_ex_lo * (y_wall_lo - y[0]))

    slope_num_hi = float((centerline_num[-1] - centerline_num[-2]) / (y[-1] - y[-2]))
    slope_ex_hi = float((centerline_exact[-1] - centerline_exact[-2]) / (y[-1] - y[-2]))
    u_wall_num_hi = float(centerline_num[-1] + slope_num_hi * (y_wall_hi - y[-1]))
    u_wall_ex_hi = float(centerline_exact[-1] + slope_ex_hi * (y_wall_hi - y[-1]))

    slip_len_lo = float(u_wall_num_lo / max(abs(slope_num_lo), 1.0e-16))
    slip_len_hi = float(u_wall_num_hi / max(abs(slope_num_hi), 1.0e-16))

    return {
        "u_wall_num_lo": u_wall_num_lo,
        "u_wall_exact_lo": u_wall_ex_lo,
        "u_wall_num_hi": u_wall_num_hi,
        "u_wall_exact_hi": u_wall_ex_hi,
        "wall_slip_len_lo": slip_len_lo,
        "wall_slip_len_hi": slip_len_hi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the 3D square-duct plotfile against the analytical profile."
    )
    parser.add_argument("--inputs", default="inputs", help="Inputs file used for the run")
    parser.add_argument("--plot-root", default="out_square_duct",
                        help="Directory containing plotfiles (default: out_square_duct)")
    parser.add_argument("--json-out", default="_artifacts/square_duct_metrics.json",
                        help="Output JSON summary path")
    parser.add_argument("--n-terms", type=int, default=200,
                        help="Number of odd-series terms (default: 200)")
    parser.add_argument("--exact-force-x", type=float, default=None,
                        help="Override the analytical forcing term G in the exact solution")
    parser.add_argument("--max-e2", type=float, default=None,
                        help="Optional validation gate on the relative L2 error")
    args = parser.parse_args()

    cfg = parse_inputs(Path(args.inputs))
    nu = cfg["lbmPhysicalParameters.nu"][0]
    force_x = cfg["lbm.prescribed_force"][0]
    exact_force_x = args.exact_force_x if args.exact_force_x is not None else force_x

    plotfile = latest_plotfile(args.plot_root)
    rho, ux, left, right, dx = load_fields(plotfile)

    y = left[1] + (np.arange(ux.shape[1], dtype=np.float64) + 0.5) * dx[1]
    z = left[2] + (np.arange(ux.shape[2], dtype=np.float64) + 0.5) * dx[2]
    a = 0.5 * (right[1] - left[1])

    rho_mean = float(np.mean(rho))
    mu = rho_mean * nu

    uavg = np.mean(ux, axis=0)
    uexact = square_duct_exact(y, z, a, exact_force_x, mu, args.n_terms)

    diff = uavg - uexact
    e2 = float(np.linalg.norm(diff) / max(np.linalg.norm(uexact), 1.0e-16))
    linf = float(np.max(np.abs(diff)) / max(np.max(np.abs(uexact)), 1.0e-16))

    centerline_num, z_centerline = centerline_from_plane(uavg, z)
    centerline_exact = square_duct_exact(y, np.array([z_centerline]), a, exact_force_x, mu, args.n_terms)[:, 0]
    centerline_diff = centerline_num - centerline_exact
    centerline_l2 = float(
        np.linalg.norm(centerline_diff) / max(np.linalg.norm(centerline_exact), 1.0e-16)
    )
    centerline_linf = float(
        np.max(np.abs(centerline_diff)) / max(np.max(np.abs(centerline_exact)), 1.0e-16)
    )
    wall_metrics = wall_slip_metrics(y, centerline_num, centerline_exact, a)

    j0 = int(np.argmin(np.abs(y)))
    k0 = int(np.argmin(np.abs(z)))
    y_center_num = float(y[j0])
    z_center_num = float(z[k0])
    u_center_num = float(uavg[j0, k0])

    u_center_exact = float(
        square_duct_exact(
            np.array([y_center_num]),
            np.array([z_center_num]),
            a,
            exact_force_x,
            mu,
            args.n_terms,
        )[0, 0]
    )

    metrics = {
        "plotfile": plotfile,
        "nx": int(ux.shape[0]),
        "ny": int(ux.shape[1]),
        "nz": int(ux.shape[2]),
        "dx": float(dx[0]),
        "dy": float(dx[1]),
        "dz": float(dx[2]),
        "nu": float(nu),
        "rho_mean": rho_mean,
        "force_x": float(force_x),
        "exact_force_x": float(exact_force_x),
        "a": float(a),
        "u_avg_l2_rel": e2,
        "u_avg_linf_rel": linf,
        "centerline_z": float(z_centerline),
        "centerline_l2_rel": centerline_l2,
        "centerline_linf_rel": centerline_linf,
        "u_center_y": y_center_num,
        "u_center_z": z_center_num,
        "u_center_exact": u_center_exact,
        "u_center_numeric": u_center_num,
        "u_max_exact": float(np.max(uexact)),
        "u_max_numeric": float(np.max(uavg)),
        **wall_metrics,
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    if args.max_e2 is not None and e2 > args.max_e2:
        raise SystemExit(
            f"Relative L2 error {e2:.6e} exceeds the requested threshold {args.max_e2:.6e}"
        )


if __name__ == "__main__":
    main()
