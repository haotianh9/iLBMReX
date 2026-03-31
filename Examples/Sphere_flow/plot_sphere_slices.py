#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yt


def latest_plotfile(root: Path) -> Path:
    plots = sorted(p for p in root.glob("plt*") if p.is_dir())
    if not plots:
        raise FileNotFoundError(f"No plotfiles found under {root}")
    return plots[-1]


def build_covering_grid(ds: yt.Dataset, dims: np.ndarray, smooth: bool, fields: list[tuple[str, str]]) -> yt.data_objects.construction_data_containers.YTCoveringGrid:
    if smooth and hasattr(ds, "smoothed_covering_grid"):
        try:
            return ds.smoothed_covering_grid(
                level=int(ds.index.max_level),
                left_edge=ds.domain_left_edge,
                dims=dims,
                fields=fields,
                num_ghost_zones=0,
            )
        except Exception:
            pass
    return ds.covering_grid(
        level=int(ds.index.max_level),
        left_edge=ds.domain_left_edge,
        dims=dims,
        num_ghost_zones=0,
    )


def load_velocity(
    plotfile: Path, smooth: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    ds = yt.load(str(plotfile))
    base_dims = np.array(ds.domain_dimensions, dtype=int)
    max_lev = int(ds.index.max_level)
    dims = base_dims * (int(ds.refine_by) ** max_lev)
    cg = build_covering_grid(ds, dims, smooth, [("boxlib", "ux"), ("boxlib", "uy"), ("boxlib", "uz")])

    ux = np.array(cg[("boxlib", "ux")], dtype=np.float64)
    uy = np.array(cg[("boxlib", "uy")], dtype=np.float64)
    uz = np.array(cg[("boxlib", "uz")], dtype=np.float64)
    lo = np.array(ds.domain_left_edge, dtype=np.float64)
    hi = np.array(ds.domain_right_edge, dtype=np.float64)
    return ux, uy, uz, float(ds.current_time), lo, hi, np.array(dims, dtype=int)


def compute_vorticity_midplane(ux: np.ndarray, uy: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    nx, ny, nz = ux.shape
    dx = (hi[0] - lo[0]) / float(nx)
    dy = (hi[1] - lo[1]) / float(ny)
    dz = (hi[2] - lo[2]) / float(nz)

    dux_dx, dux_dy, dux_dz = np.gradient(ux, dx, dy, dz, edge_order=2)
    duy_dx, duy_dy, duy_dz = np.gradient(uy, dx, dy, dz, edge_order=2)

    mid_k = nz // 2
    vor_z = duy_dx[:, :, mid_k] - dux_dy[:, :, mid_k]
    return vor_z


def load_midplane_raw(plotfile: Path, field: str, smooth: bool) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    ds = yt.load(str(plotfile))
    base_dims = np.array(ds.domain_dimensions, dtype=int)
    max_lev = int(ds.index.max_level)
    dims = base_dims * (int(ds.refine_by) ** max_lev)
    cg = build_covering_grid(ds, dims, smooth, [("boxlib", field)])

    data = np.array(cg[("boxlib", field)], dtype=np.float64)
    if data.ndim != 3:
        raise RuntimeError(f"Expected 3D field for {field}, got shape {data.shape}")

    mid_k = data.shape[2] // 2
    slice2d = data[:, :, mid_k]

    lo = np.array(ds.domain_left_edge[:2], dtype=np.float64)
    hi = np.array(ds.domain_right_edge[:2], dtype=np.float64)
    return slice2d, float(ds.current_time), lo, hi


def plot_fields(plotfile: Path, output: Path, vor_source: str, smooth: bool) -> None:
    ux, uy, uz, t, lo3, hi3, dims = load_velocity(plotfile, smooth)
    mid_k = ux.shape[2] // 2
    speed = np.sqrt(ux[:, :, mid_k] ** 2 + uy[:, :, mid_k] ** 2 + uz[:, :, mid_k] ** 2)

    lo = lo3[:2]
    hi = hi3[:2]

    if vor_source == "computed":
        vor = compute_vorticity_midplane(ux, uy, lo3, hi3)
    else:
        vor, _, _, _ = load_midplane_raw(plotfile, "vor", smooth)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    extent = [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]

    im0 = axes[0].imshow(vor.T, origin="lower", extent=extent, cmap="RdBu_r")
    axes[0].set_title(f"Vorticity (mid-z)\n{plotfile.name}, t={t:.1f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="vor")

    im1 = axes[1].imshow(speed.T, origin="lower", extent=extent, cmap="viridis")
    axes[1].set_title(f"Speed |u| (mid-z)\n{plotfile.name}, t={t:.1f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="|u|")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mid-plane vorticity and speed for Sphere flow.")
    parser.add_argument("--plot-root", type=Path, default=Path("out_sphere"))
    parser.add_argument("--plotfile", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("_artifacts/sphere_slices.png"))
    parser.add_argument(
        "--vor-source",
        choices=["computed", "raw"],
        default="computed",
        help="Use computed vorticity from ux/uy or raw 'vor' field from plotfile.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable smoothed covering grid interpolation across AMR levels.",
    )
    args = parser.parse_args()

    plotfile = args.plotfile or latest_plotfile(args.plot_root)
    plot_fields(plotfile, args.output, args.vor_source, smooth=not args.no_smooth)
    print(f"Wrote {args.output} from {plotfile}")


if __name__ == "__main__":
    main()
