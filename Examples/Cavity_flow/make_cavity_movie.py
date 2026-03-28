#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import yt

VOR_CLIM = 1.0e-2


def list_plotfiles(root: Path) -> list[Path]:
    plots = sorted(p for p in root.glob("plt*") if p.is_dir())
    if not plots:
        raise FileNotFoundError(f"No plotfiles found under {root}")
    return plots


def load_state2d(plotfile: Path) -> dict[str, np.ndarray | float]:
    ds = yt.load(str(plotfile))
    base_dims = np.array(ds.domain_dimensions, dtype=int)
    max_lev = int(ds.index.max_level)
    dims = base_dims * (int(ds.refine_by) ** max_lev)
    cg = ds.covering_grid(level=max_lev, left_edge=ds.domain_left_edge, dims=dims, num_ghost_zones=0)

    vor = np.array(cg[("boxlib", "vor")], dtype=np.float64)
    if vor.ndim == 3:
        vor = vor[:, :, vor.shape[2] // 2]

    lo = np.array(ds.domain_left_edge[:2], dtype=np.float64)
    hi = np.array(ds.domain_right_edge[:2], dtype=np.float64)
    nx, ny = vor.shape
    dx = (hi[0] - lo[0]) / float(nx)
    dy = (hi[1] - lo[1]) / float(ny)
    xc = lo[0] + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    yc = lo[1] + (np.arange(ny, dtype=np.float64) + 0.5) * dy

    return {
        "time": float(ds.current_time),
        "vor": vor,
        "lo": lo,
        "hi": hi,
        "xc": xc,
        "yc": yc,
    }


def crop_ibm_to_cavity(bc: dict, ibm: dict) -> tuple[dict, dict]:
    if bc["vor"].shape == ibm["vor"].shape:
        return bc, ibm

    nxb, nyb = bc["vor"].shape
    nxi, nyi = ibm["vor"].shape
    if nxb == nxi and nyi >= nyb:
        ibm = ibm.copy()
        ibm["vor"] = ibm["vor"][:, :nyb]
        ibm["yc"] = ibm["yc"][:nyb]
        ibm["hi"] = np.array([ibm["hi"][0], bc["hi"][1]], dtype=np.float64)
        return bc, ibm

    raise RuntimeError(f"Shape mismatch: BC {bc['vor'].shape}, IBM {ibm['vor'].shape}")


def pair_plotfiles(bc_root: Path, ibm_root: Path) -> list[tuple[Path, Path]]:
    bc_plots = list_plotfiles(bc_root)
    ibm_plots = list_plotfiles(ibm_root)

    ibm_by_name = {p.name: p for p in ibm_plots}
    pairs: list[tuple[Path, Path]] = []
    for bc_plot in bc_plots:
        ibm_plot = ibm_by_name.get(bc_plot.name)
        if ibm_plot is None:
            continue
        pairs.append((bc_plot, ibm_plot))

    if not pairs:
        raise RuntimeError("No matching BC/IBM plotfile names found")

    return pairs


def make_movie(bc_root: Path, ibm_root: Path, output: Path, fps: int) -> None:
    pairs = pair_plotfiles(bc_root, ibm_root)
    frames: list[tuple[dict, dict]] = []

    for bc_plot, ibm_plot in pairs:
        bc = load_state2d(bc_plot)
        ibm = load_state2d(ibm_plot)
        bc, ibm = crop_ibm_to_cavity(bc, ibm)
        frames.append((bc, ibm))

    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps) as writer:
        for bc, ibm in frames:
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
            for ax, title, data in (
                (axes[0], "BC cavity vorticity", bc["vor"]),
                (axes[1], "IBM cavity vorticity", ibm["vor"]),
            ):
                im = ax.imshow(
                    data.T,
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-VOR_CLIM,
                    vmax=VOR_CLIM,
                    extent=[float(bc["lo"][0]), float(bc["hi"][0]), float(bc["lo"][1]), float(bc["hi"][1])],
                )
                ax.set_title(f"{title} (t={bc['time']:.0f})")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                fig.colorbar(im, ax=ax, shrink=0.88, label="vorticity")

            fig.suptitle("Cavity flow: BC vs IBM", fontsize=14)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a BC-vs-IBM cavity vorticity movie.")
    parser.add_argument("--bc-root", type=Path, default=Path("out_bc_long"))
    parser.add_argument("--ibm-root", type=Path, default=Path("out_ibm_long"))
    parser.add_argument("--output", type=Path, default=Path("_artifacts/cavity_vorticity_evolution.mp4"))
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    make_movie(args.bc_root, args.ibm_root, args.output, args.fps)
    print(f"Wrote movie to {args.output}")


if __name__ == "__main__":
    main()
