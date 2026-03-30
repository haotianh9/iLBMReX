#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import imageio.v2 as imageio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yt


def parse_inputs(inputs_path: Path) -> tuple[float, float, float, float]:
    vals: dict[str, float | None] = {
        "ibm.x0": None,
        "ibm.y0": None,
        "ibm.R": None,
        "lbmPhysicalParameters.U0": None,
    }
    with inputs_path.open() as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            lhs, rhs = [p.strip() for p in line.split("=", 1)]
            if lhs in vals:
                vals[lhs] = float(rhs.split()[0])

    missing = [k for k, v in vals.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing keys in {inputs_path}: {missing}")
    return (
        float(vals["ibm.x0"]),
        float(vals["ibm.y0"]),
        float(vals["ibm.R"]),
        float(vals["lbmPhysicalParameters.U0"]),
    )


def list_plotfiles(root: Path) -> list[Path]:
    plots = [p for p in root.glob("plt*") if p.is_dir() and re.fullmatch(r"plt\d+", p.name)]
    if not plots:
        raise FileNotFoundError(f"No plotfiles found under {root}")
    plots.sort(key=lambda p: int(p.name[3:]))
    return plots


def finest_resolution_2d(ds) -> tuple[int, int]:
    level = int(ds.index.max_level)
    dims = np.array(ds.domain_dimensions, dtype=np.int64)
    if level > 0:
        ref = int(np.prod(np.asarray(ds.ref_factors[:level], dtype=np.int64)))
        dims *= ref
    return int(dims[0]), int(dims[1])


def compute_vorticity_from_velocity(
    ux: np.ndarray, uy: np.ndarray, extent: list[float]
) -> np.ndarray:
    nx, ny = ux.shape
    dx = (extent[1] - extent[0]) / float(nx)
    dy = (extent[3] - extent[2]) / float(ny)
    dudy = np.gradient(ux, dy, axis=1, edge_order=2)
    dvdx = np.gradient(uy, dx, axis=0, edge_order=2)
    return np.asarray(dvdx - dudy, dtype=np.float64)


def load_vorticity(plotfile: Path) -> tuple[np.ndarray, list[float], float]:
    ds = yt.load(str(plotfile))

    xlo = float(ds.domain_left_edge[0])
    xhi = float(ds.domain_right_edge[0])
    ylo = float(ds.domain_left_edge[1])
    yhi = float(ds.domain_right_edge[1])
    zlo = float(ds.domain_left_edge[2])
    zhi = float(ds.domain_right_edge[2])
    nx, ny = finest_resolution_2d(ds)

    zmid = 0.5 * (zlo + zhi)
    xmid = 0.5 * (xlo + xhi)
    ymid = 0.5 * (ylo + yhi)
    sl = ds.slice(2, zmid)
    frb = sl.to_frb(
        width=(xhi - xlo, "code_length"),
        height=(yhi - ylo, "code_length"),
        resolution=(nx, ny),
        center=(xmid, ymid, zmid),
    )

    extent = [xlo, xhi, ylo, yhi]
    # yt FRB returns image arrays with shape (ny, nx). Transpose to (nx, ny)
    # so x is axis-0 and y is axis-1, consistent with downstream operators.
    ux = np.asarray(frb[("boxlib", "ux")], dtype=np.float64).T
    uy = np.asarray(frb[("boxlib", "uy")], dtype=np.float64).T
    vor = compute_vorticity_from_velocity(ux, uy, extent)
    time_lbm = float(ds.current_time)
    return vor, extent, time_lbm


def compute_vlim(plotfiles: list[Path], q: float) -> float:
    vmax = 0.0
    for pf in plotfiles:
        vor, _, _ = load_vorticity(pf)
        local = float(np.quantile(np.abs(vor), q))
        vmax = max(vmax, local)
    return max(vmax, 1.0e-12)


def make_movie(
    plot_root: Path,
    inputs: Path,
    output: Path,
    fps: int,
    quantile: float,
    stride: int,
    interpolation: str,
) -> None:
    x0, y0, radius, u0 = parse_inputs(inputs)
    diameter = 2.0 * radius
    plotfiles = list_plotfiles(plot_root)
    plotfiles = plotfiles[:: max(1, stride)]
    vlim = compute_vlim(plotfiles, quantile)

    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output,
        fps=fps,
        macro_block_size=1,
        codec="libx264",
        ffmpeg_params=["-crf", "10", "-preset", "slow", "-pix_fmt", "yuv444p"],
    ) as writer:
        for pf in plotfiles:
            vor, extent, t_lbm = load_vorticity(pf)
            t_conv = t_lbm * u0 / diameter

            fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
            im = ax.imshow(
                vor.T,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-vlim,
                vmax=vlim,
                interpolation=interpolation,
                aspect="equal",
            )
            ax.add_patch(mpatches.Circle((x0, y0), radius, fill=False, edgecolor="black", linewidth=1.2))
            ax.set_title(f"{plot_root.name} {pf.name}  (tU/D={t_conv:.2f})")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.9, label="vorticity")

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)
            plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Make cylinder vorticity movie from AMReX plotfiles.")
    ap.add_argument("--plot-root", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, default=Path("inputs"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--quantile", type=float, default=0.995)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--interpolation",
        choices=["nearest", "bilinear"],
        default="nearest",
        help="imshow interpolation used for each frame (nearest matches static figures).",
    )
    args = ap.parse_args()

    make_movie(
        plot_root=args.plot_root,
        inputs=args.inputs,
        output=args.output,
        fps=args.fps,
        quantile=args.quantile,
        stride=args.stride,
        interpolation=args.interpolation,
    )
    print(f"Wrote movie: {args.output}")


if __name__ == "__main__":
    main()
