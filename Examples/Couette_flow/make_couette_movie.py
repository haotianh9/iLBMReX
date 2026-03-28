#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import yt


def exact_couette(y: np.ndarray, t: float, u_top: float, height: float,
                  nu: float, terms: int = 600) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    base = u_top * y / height
    n = np.arange(1, terms + 1, dtype=float)[:, None]
    series = (
        (2.0 * u_top / np.pi)
        * ((-1.0) ** n / n)
        * np.sin(n * np.pi * y[None, :] / height)
        * np.exp(-(n * n) * (np.pi ** 2) * nu * t / (height * height))
    ).sum(axis=0)
    return base + series


def load_snapshot(plotfile: Path) -> dict[str, np.ndarray | float | str]:
    ds = yt.load(str(plotfile))
    ds.force_periodicity()
    level = ds.index.max_level
    ref = 2 ** level
    dims = np.array(ds.domain_dimensions, dtype=int) * ref
    cg = ds.smoothed_covering_grid(
        level=level, left_edge=ds.domain_left_edge, dims=dims
    )

    ux = np.array(cg[("boxlib", "ux")])[:, :, 0]
    x = ds.domain_left_edge[0].d + (np.arange(dims[0]) + 0.5) * (
        ds.domain_width[0].d / dims[0]
    )
    y = ds.domain_left_edge[1].d + (np.arange(dims[1]) + 0.5) * (
        ds.domain_width[1].d / dims[1]
    )
    return {
        "plotfile": plotfile.name,
        "time": float(ds.current_time),
        "x": x,
        "y": y,
        "ux": ux,
    }


def make_movie(plot_root: Path, output: Path, fps: int, u_top: float,
               height: float, nu: float) -> None:
    plotfiles = sorted(plot_root.glob("plt*"))
    if not plotfiles:
        raise FileNotFoundError(f"No plotfiles found under {plot_root}")

    snapshots = [load_snapshot(p) for p in plotfiles]
    output.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output, fps=fps) as writer:
        for snap in snapshots:
            ux = snap["ux"]
            x = snap["x"]
            y = snap["y"]
            time = float(snap["time"])

            profile_num = ux.mean(axis=0)
            profile_exact = exact_couette(
                y, time, u_top=u_top, height=height, nu=nu
            )

            fig, axes = plt.subplots(
                1, 2, figsize=(10.5, 4.8), constrained_layout=True
            )

            im = axes[0].imshow(
                ux.T,
                origin="lower",
                extent=[x[0], x[-1], y[0], y[-1]],
                cmap="viridis",
                vmin=0.0,
                vmax=u_top,
                aspect="equal",
            )
            axes[0].set_title(f"$u_x(x,y)$ at t = {time:.1f}")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            fig.colorbar(im, ax=axes[0], shrink=0.88, label="$u_x$")

            axes[1].plot(profile_exact, y, "k--", linewidth=2.0,
                         label="Exact transient")
            axes[1].plot(profile_num, y, color="#1f77b4", linewidth=2.0,
                         label="Numerical x-average")
            axes[1].set_xlim(-0.01, 1.05 * u_top)
            axes[1].set_ylim(y[0], y[-1])
            axes[1].set_xlabel("$u_x$")
            axes[1].set_ylabel("y")
            axes[1].set_title("Couette profile")
            axes[1].legend(loc="lower right")
            axes[1].grid(alpha=0.25)

            fig.suptitle(
                "Couette flow start-up on a fixed AMR hierarchy", fontsize=14
            )

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Couette-flow movie from AMReX plotfiles."
    )
    parser.add_argument(
        "--plot-root",
        type=Path,
        default=Path("out_validation"),
        help="Directory containing Couette plotfiles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_artifacts/couette_ux_evolution.mp4"),
        help="Output movie path.",
    )
    parser.add_argument("--fps", type=int, default=4, help="Movie frame rate.")
    parser.add_argument("--u-top", type=float, default=0.1,
                        help="Top-wall speed.")
    parser.add_argument("--height", type=float, default=64.0,
                        help="Channel height.")
    parser.add_argument("--nu", type=float, default=0.064,
                        help="Kinematic viscosity.")
    args = parser.parse_args()

    make_movie(args.plot_root, args.output, args.fps, args.u_top, args.height,
               args.nu)
    print(f"Wrote movie to {args.output}")


if __name__ == "__main__":
    main()
