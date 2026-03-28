#!/usr/bin/env python3
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import yt

VOR_CLIM = 1.0e-2


def latest_plotfile(root: str) -> str:
    cands = [p for p in glob.glob(os.path.join(root, "plt*")) if os.path.isdir(p)]
    if not cands:
        raise FileNotFoundError(f"No plotfiles found under {root}")
    cands.sort()
    return cands[-1]


def latest_common_plotfiles(bc_root: str, ibm_root: str) -> tuple[str, str]:
    bc_plots = [p for p in glob.glob(os.path.join(bc_root, "plt*")) if os.path.isdir(p)]
    ibm_plots = [p for p in glob.glob(os.path.join(ibm_root, "plt*")) if os.path.isdir(p)]
    if not bc_plots:
        raise FileNotFoundError(f"No plotfiles found under {bc_root}")
    if not ibm_plots:
        raise FileNotFoundError(f"No plotfiles found under {ibm_root}")

    bc_by_name = {os.path.basename(p): p for p in bc_plots}
    common = sorted(name for name in bc_by_name if os.path.isdir(os.path.join(ibm_root, name)))
    if not common:
        raise RuntimeError(f"No common BC/IBM plotfiles found under {bc_root} and {ibm_root}")

    name = common[-1]
    return bc_by_name[name], os.path.join(ibm_root, name)


def load_state2d(plotfile: str) -> dict[str, np.ndarray | float]:
    ds = yt.load(plotfile)
    base_dims = np.array(ds.domain_dimensions, dtype=int)
    max_lev = int(ds.index.max_level)
    dims = base_dims * (int(ds.refine_by) ** max_lev)
    cg = ds.covering_grid(level=max_lev, left_edge=ds.domain_left_edge, dims=dims, num_ghost_zones=0)

    ux = cg[("boxlib", "ux")].to_ndarray()
    uy = cg[("boxlib", "uy")].to_ndarray()
    vor = cg[("boxlib", "vor")].to_ndarray()
    if ux.ndim == 3:
        k = ux.shape[2] // 2
        ux = ux[:, :, k]
        uy = uy[:, :, k]
        vor = vor[:, :, k]

    lo = np.array(ds.domain_left_edge[:2], dtype=np.float64)
    hi = np.array(ds.domain_right_edge[:2], dtype=np.float64)
    nx, ny = ux.shape
    dx = (hi[0] - lo[0]) / float(nx)
    dy = (hi[1] - lo[1]) / float(ny)
    xc = lo[0] + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    yc = lo[1] + (np.arange(ny, dtype=np.float64) + 0.5) * dy

    return {
        "time": float(ds.current_time),
        "ux": np.asarray(ux, dtype=np.float64),
        "uy": np.asarray(uy, dtype=np.float64),
        "vor": np.asarray(vor, dtype=np.float64),
        "lo": lo,
        "hi": hi,
        "xc": xc,
        "yc": yc,
    }


def crop_ibm_to_cavity(bc: dict, ibm: dict) -> tuple[dict, dict]:
    if bc["ux"].shape == ibm["ux"].shape:
        return bc, ibm

    nxb, nyb = bc["ux"].shape
    nxi, nyi = ibm["ux"].shape
    if nxb == nxi and nyi >= nyb:
        ibm = ibm.copy()
        for key in ("ux", "uy", "vor"):
            ibm[key] = ibm[key][:, :nyb]
        ibm["yc"] = ibm["yc"][:nyb]
        ibm["hi"] = np.array([ibm["hi"][0], bc["hi"][1]], dtype=np.float64)
        return bc, ibm

    raise RuntimeError(f"Shape mismatch: BC {bc['ux'].shape}, IBM {ibm['ux'].shape}")


def draw_vorticity(vor: np.ndarray, lo: np.ndarray, hi: np.ndarray, title: str, out_png: str, clim: float) -> None:
    extent = [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]
    clim = max(float(clim), 1.0e-14)

    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    im = ax.imshow(
        vor.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-clim,
        vmax=clim,
    )
    ax.set_title(f"{title}: vorticity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.9, label="vorticity")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def draw_centerlines(bc: dict, ibm: dict, out_png: str) -> None:
    xmid = 0.5 * (bc["lo"][0] + bc["hi"][0])
    ymid = 0.5 * (bc["lo"][1] + bc["hi"][1])

    ix_bc = int(np.argmin(np.abs(bc["xc"] - xmid)))
    ix_ibm = int(np.argmin(np.abs(ibm["xc"] - xmid)))
    iy_bc = int(np.argmin(np.abs(bc["yc"] - ymid)))
    iy_ibm = int(np.argmin(np.abs(ibm["yc"] - ymid)))

    u_bc = bc["ux"][ix_bc, :]
    v_bc = bc["uy"][:, iy_bc]
    u_ibm = np.interp(bc["yc"], ibm["yc"], ibm["ux"][ix_ibm, :])
    v_ibm = np.interp(bc["xc"], ibm["xc"], ibm["uy"][:, iy_ibm])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    axes[0].plot(u_bc, bc["yc"], lw=2, label="BC")
    axes[0].plot(u_ibm, bc["yc"], "--", lw=2, label="IBM")
    axes[0].set_title("u(y) at x=L/2")
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(bc["xc"], v_bc, lw=2, label="BC")
    axes[1].plot(bc["xc"], v_ibm, "--", lw=2, label="IBM")
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
    bc_plot, ibm_plot = latest_common_plotfiles(args.bc_root, args.ibm_root)

    bc = load_state2d(bc_plot)
    ibm = load_state2d(ibm_plot)
    bc, ibm = crop_ibm_to_cavity(bc, ibm)

    draw_vorticity(
        bc["vor"], bc["lo"], bc["hi"], "BC cavity", os.path.join(args.out_dir, "cavity_bc_vorticity.png"), VOR_CLIM
    )
    draw_vorticity(
        ibm["vor"], ibm["lo"], ibm["hi"], "IBM cavity", os.path.join(args.out_dir, "cavity_ibm_vorticity.png"), VOR_CLIM
    )
    draw_centerlines(bc, ibm, os.path.join(args.out_dir, "cavity_centerlines_bc_vs_ibm.png"))

    print(f"BC plotfile : {bc_plot}")
    print(f"IBM plotfile: {ibm_plot}")
    print(f"Wrote: {os.path.join(args.out_dir, 'cavity_bc_vorticity.png')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'cavity_ibm_vorticity.png')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'cavity_centerlines_bc_vs_ibm.png')}")


if __name__ == "__main__":
    main()
