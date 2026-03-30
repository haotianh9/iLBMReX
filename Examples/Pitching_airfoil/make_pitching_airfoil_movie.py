#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import yt

from plot_pitching_airfoil_results import (
    airfoil_outline,
    latest_plotfiles,
    load_fields2d,
    parse_header_constants,
    parse_inputs,
    pitch_angle,
)


warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
yt.set_log_level("error")


def parse_xlim_ylim(raw: str) -> tuple[float, float]:
    vals = [float(x) for x in raw.split()]
    if len(vals) != 2:
        raise ValueError(f"Expected two values, got: {raw!r}")
    return vals[0], vals[1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a pitching-airfoil vorticity movie from AMReX plotfiles."
    )
    ap.add_argument("--plot-root", type=Path, default=Path("out_pitching_airfoil_19T_dense"))
    ap.add_argument("--inputs", type=Path, default=Path("inputs"))
    ap.add_argument("--geometry-header", type=Path, default=Path("IBMUserDefinedGeometry.H"))
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("_artifacts/plots_19T_dense/pitching_airfoil_vorticity_evolution.mp4"),
    )
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument(
        "--vorticity-limit",
        type=float,
        default=0.01,
        help="Symmetric vorticity color limit for the movie (uses +/- this value).",
    )
    ap.add_argument("--xlim", type=str, default="55 260")
    ap.add_argument("--ylim", type=str, default="45 147")
    ap.add_argument(
        "--include-endpoint",
        action="store_true",
        help="Include the final endpoint frame (disabled by default).",
    )
    args = ap.parse_args()

    inp = parse_inputs(args.inputs)
    par = parse_header_constants(args.geometry_header)
    plotfiles = latest_plotfiles(args.plot_root)
    if not args.include_endpoint and len(plotfiles) > 1:
        plotfiles = plotfiles[:-1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    xlim = parse_xlim_ylim(args.xlim)
    ylim = parse_xlim_ylim(args.ylim)

    vmax = max(abs(args.vorticity_limit), 1.0e-12)
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    extent0, _, _, vor0 = load_fields2d(plotfiles[0])
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    im = ax.imshow(
        vor0.T,
        origin="lower",
        extent=extent0,
        cmap="coolwarm",
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    poly = ax.fill([0.0], [0.0], facecolor="black", edgecolor="white", linewidth=0.7)[0]
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("vorticity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(False)

    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=2200)
    with writer.saving(fig, str(args.output), dpi=170):
        for pf in plotfiles:
            extent, _, _, vor = load_fields2d(pf)
            step = int(pf.name.replace("plt", ""))
            t_conv = par["pitch_frequency"] * step
            angle = float(pitch_angle(np.asarray([t_conv], dtype=np.float64), par)[0])

            im.set_data(vor.T)
            im.set_extent(extent)

            xp, yp = airfoil_outline(
                inp["ibm.x0"],
                inp["ibm.y0"],
                par["airfoil_chord"],
                par["airfoil_thickness"],
                angle,
            )
            poly.set_xy(np.column_stack([xp, yp]))
            ax.set_title(
                f"{pf.name}: t*U/c={t_conv:.2f}, pitch={math.degrees(angle):.1f} deg"
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"Wrote movie: {args.output}")
    print(f"Frames: {len(plotfiles)}")
    print(f"|omega| color limit: +/-{vmax:g}")


if __name__ == "__main__":
    main()
