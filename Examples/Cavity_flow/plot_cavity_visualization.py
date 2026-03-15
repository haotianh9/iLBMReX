#!/usr/bin/env python3
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import yt


def latest_plotfile(root: str) -> str:
    cands = [p for p in glob.glob(os.path.join(root, "plt*")) if os.path.isdir(p)]
    if not cands:
        raise FileNotFoundError(f"No plotfiles found under {root}")
    cands.sort()
    return cands[-1]


def load_field2d(plotfile: str, name: str) -> np.ndarray:
    ds = yt.load(plotfile)
    dims = tuple(int(v) for v in ds.domain_dimensions)
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)
    arr = cg[("boxlib", name)].to_ndarray()
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]
    return np.asarray(arr, dtype=np.float64)


def draw_triplet(ux: np.ndarray, uy: np.ndarray, vor: np.ndarray, title: str, out_png: str) -> None:
    speed = np.sqrt(ux * ux + uy * uy)
    extent = [0, ux.shape[0], 0, ux.shape[1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    fields = [("ux", ux), ("uy", uy), ("vorticity", vor)]
    for ax, (name, arr) in zip(axes, fields):
        im = ax.imshow(arr.T, origin="lower", extent=extent, cmap="viridis")
        ax.set_title(f"{title}: {name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.9)

    fig.suptitle(f"{title} (|u| max={speed.max():.4f})")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def draw_centerlines(ux_bc: np.ndarray, uy_bc: np.ndarray, ux_ibm: np.ndarray, uy_ibm: np.ndarray, out_png: str) -> None:
    if ux_bc.shape != ux_ibm.shape:
        nxb, nyb = ux_bc.shape
        nxi, nyi = ux_ibm.shape
        if nxb == nxi and nyi >= nyb:
            ux_ibm = ux_ibm[:, :nyb]
            uy_ibm = uy_ibm[:, :nyb]
        else:
            raise RuntimeError(f"Shape mismatch: BC {ux_bc.shape}, IBM {ux_ibm.shape}")

    nx, ny = ux_bc.shape
    ix = nx // 2
    iy = ny // 2
    y = np.arange(ny)
    x = np.arange(nx)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    axes[0].plot(ux_bc[ix, :], y, lw=2, label="BC")
    axes[0].plot(ux_ibm[ix, :], y, "--", lw=2, label="IBM")
    axes[0].set_title("u(y) at x=L/2")
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, uy_bc[:, iy], lw=2, label="BC")
    axes[1].plot(x, uy_ibm[:, iy], "--", lw=2, label="IBM")
    axes[1].set_title("v(x) at y=L/2")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cavity BC vs IBM fields and centerlines.")
    parser.add_argument("--bc-root", default="out_bc")
    parser.add_argument("--ibm-root", default="out_ibm")
    parser.add_argument("--out-dir", default="_artifacts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bc_plot = latest_plotfile(args.bc_root)
    ibm_plot = latest_plotfile(args.ibm_root)

    ux_bc = load_field2d(bc_plot, "ux")
    uy_bc = load_field2d(bc_plot, "uy")
    vor_bc = load_field2d(bc_plot, "vor")

    ux_ibm = load_field2d(ibm_plot, "ux")
    uy_ibm = load_field2d(ibm_plot, "uy")
    vor_ibm = load_field2d(ibm_plot, "vor")

    draw_triplet(ux_bc, uy_bc, vor_bc, "BC cavity", os.path.join(args.out_dir, "cavity_bc_fields.png"))
    draw_triplet(ux_ibm, uy_ibm, vor_ibm, "IBM cavity", os.path.join(args.out_dir, "cavity_ibm_fields.png"))
    draw_centerlines(
        ux_bc, uy_bc, ux_ibm, uy_ibm, os.path.join(args.out_dir, "cavity_centerlines_bc_vs_ibm.png")
    )

    print(f"BC plotfile : {bc_plot}")
    print(f"IBM plotfile: {ibm_plot}")
    print(f"Wrote: {os.path.join(args.out_dir, 'cavity_bc_fields.png')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'cavity_ibm_fields.png')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'cavity_centerlines_bc_vs_ibm.png')}")


if __name__ == "__main__":
    main()
