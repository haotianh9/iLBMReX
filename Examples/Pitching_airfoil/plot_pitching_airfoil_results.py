#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import math
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yt
yt.set_log_level("error")


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
GEOM_HDR = ROOT / "IBMUserDefinedGeometry.H"
FORCE_FILE = ROOT / "force.dat"
PLOT_ROOT = ROOT / "out_pitching_airfoil"
OUT_DIR = ROOT / "_artifacts" / "plots"


def parse_inputs(path: Path) -> dict[str, float]:
    keys = {
        "ibm.x0": None,
        "ibm.y0": None,
        "lbmPhysicalParameters.U0": None,
    }
    with path.open() as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            lhs, rhs = [part.strip() for part in line.split("=", 1)]
            if lhs in keys:
                keys[lhs] = float(rhs.split()[0])
    missing = [k for k, v in keys.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing inputs keys in {path}: {missing}")
    return {k: float(v) for k, v in keys.items()}


def parse_header_constants(path: Path) -> dict[str, float]:
    names = [
        "airfoil_chord",
        "airfoil_thickness",
        "pitch_mean",
        "pitch_amplitude",
        "pitch_frequency",
        "pitch_phase",
    ]
    text = path.read_text()
    vals: dict[str, float] = {}
    for name in names:
        m = re.search(
            rf"{name}\s*=\s*amrex::Real\(([-+0-9.eE]+)\)", text, re.MULTILINE
        )
        if not m:
            raise RuntimeError(f"Could not parse {name} from {path}")
        vals[name] = float(m.group(1))
    return vals


def latest_plotfiles(root: Path) -> list[Path]:
    cands = [p for p in root.glob("plt*") if p.is_dir() and re.fullmatch(r"plt\d+", p.name)]
    cands.sort()
    if not cands:
        raise FileNotFoundError(f"No plotfiles found under {root}")
    return cands


def finest_covering_grid(ds):
    level = int(ds.index.max_level)
    dims = np.array(ds.domain_dimensions, dtype=np.int64)
    if level > 0:
        ref = int(np.prod(np.asarray(ds.ref_factors[:level], dtype=np.int64)))
        dims *= ref
    ds.force_periodicity()
    return ds.smoothed_covering_grid(
        level=level,
        left_edge=ds.domain_left_edge,
        dims=tuple(int(v) for v in dims),
        num_ghost_zones=0,
    )


def load_fields2d(plotfile: Path) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    ds = yt.load(str(plotfile))
    cg = finest_covering_grid(ds)

    def field(name: str) -> np.ndarray:
        arr = cg[("boxlib", name)].to_ndarray()
        if arr.ndim == 3:
            arr = arr[:, :, arr.shape[2] // 2]
        return np.asarray(arr, dtype=np.float64)

    extent = [
        float(ds.domain_left_edge[0]),
        float(ds.domain_right_edge[0]),
        float(ds.domain_left_edge[1]),
        float(ds.domain_right_edge[1]),
    ]
    return extent, field("ux"), field("uy"), field("vor")


def load_force(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with path.open("rb") as fh:
        for raw in fh:
            line = raw.decode("ascii", errors="ignore").replace("\x00", " ").strip()
            if not line or line.startswith("#"):
                continue
            vals = []
            for tok in line.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
            if len(vals) >= 23:
                rows.append(vals)
    if not rows:
        raise RuntimeError(f"No readable force rows found in {path}")
    arr = np.asarray(rows, dtype=np.float64)
    time = arr[:, 1]
    cd = arr[:, 20]
    cl = arr[:, 21]
    t_conv = arr[:, 22]
    return time, t_conv, cd, cl


def airfoil_outline(
    x0: float,
    y0: float,
    chord: float,
    thickness: float,
    angle: float,
    n_half: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    beta = np.linspace(0.0, math.pi, n_half + 1)
    x = 0.5 * chord * (1.0 - np.cos(beta))
    xc = np.clip(x / chord, 0.0, 1.0)
    yt = 5.0 * thickness * chord * (
        0.2969 * np.sqrt(xc)
        - 0.1260 * xc
        - 0.3516 * xc * xc
        + 0.2843 * xc * xc * xc
        - 0.1036 * xc * xc * xc * xc
    )
    xr = np.concatenate([x, x[-2:0:-1]])
    yr = np.concatenate([yt, -yt[-2:0:-1]])
    c = math.cos(angle)
    s = math.sin(angle)
    xp = x0 + c * xr - s * yr
    yp = y0 + s * xr + c * yr
    return xp, yp


def pitch_angle(t_conv: np.ndarray, par: dict[str, float]) -> np.ndarray:
    return par["pitch_mean"] + par["pitch_amplitude"] * np.sin(
        2.0 * math.pi * t_conv + par["pitch_phase"]
    )


def pick_snapshot_indices(
    plotfiles: list[Path], count: int = 4, include_endpoint: bool = False
) -> list[Path]:
    if not plotfiles:
        return []

    # For periodic pitching cases, the very last saved frame is often an endpoint
    # (integer period) and can be visually misleading for the cycle panel.
    pool = plotfiles if include_endpoint or len(plotfiles) == 1 else plotfiles[:-1]

    if len(pool) <= count:
        return pool
    return pool[-count:]


def plot_vorticity_cycle(
    out_png: Path,
    plotfiles: list[Path],
    inp: dict[str, float],
    par: dict[str, float],
    vorticity_limit: float,
) -> None:
    fields = []
    vmax = max(abs(vorticity_limit), 1.0e-12)
    for pf in plotfiles:
        extent, ux, uy, vor = load_fields2d(pf)
        fields.append((pf, extent, ux, uy, vor))
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    for axi, (pf, extent, ux, uy, vor) in zip(ax.ravel(), fields):
        step = int(pf.name.replace("plt", ""))
        t_conv = par["pitch_frequency"] * step
        angle = float(pitch_angle(np.asarray([t_conv]), par)[0])
        im = axi.imshow(
            vor.T,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            norm=norm,
            aspect="equal",
            interpolation="nearest",
        )
        xp, yp = airfoil_outline(
            inp["ibm.x0"],
            inp["ibm.y0"],
            par["airfoil_chord"],
            par["airfoil_thickness"],
            angle,
        )
        axi.fill(xp, yp, facecolor="black", edgecolor="white", linewidth=0.7)
        axi.set_xlim(40.0, 320.0)
        axi.set_ylim(25.0, 167.0)
        axi.set_title(
            f"{pf.name}: t*U/c={t_conv:.2f}, pitch={math.degrees(angle):.1f} deg"
        )
        axi.set_xlabel("x")
        axi.set_ylabel("y")
        axi.grid(False)

    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.92)
    cbar.set_label("vorticity")
    fig.suptitle("Pitching NACA 0012: vorticity over the last saved cycle")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_final_streamlines(
    out_png: Path,
    plotfile: Path,
    inp: dict[str, float],
    par: dict[str, float],
    vorticity_limit: float,
) -> None:
    extent, ux, uy, vor = load_fields2d(plotfile)
    speed = np.sqrt(ux * ux + uy * uy)
    step = int(plotfile.name.replace("plt", ""))
    t_conv = par["pitch_frequency"] * step
    angle = float(pitch_angle(np.asarray([t_conv]), par)[0])

    nx, ny = ux.shape
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    vmax = max(abs(vorticity_limit), 1.0e-12)
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    im = ax.imshow(
        vor.T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )
    skip = max(nx // 160, 1)
    ax.streamplot(
        x[::skip],
        y[::skip],
        ux[::skip, ::skip].T,
        uy[::skip, ::skip].T,
        color=np.clip(speed[::skip, ::skip].T, 0.0, np.percentile(speed, 99.0)),
        cmap="viridis",
        density=1.5,
        linewidth=0.8,
        arrowsize=0.7,
    )
    xp, yp = airfoil_outline(
        inp["ibm.x0"],
        inp["ibm.y0"],
        par["airfoil_chord"],
        par["airfoil_thickness"],
        angle,
    )
    ax.fill(xp, yp, facecolor="black", edgecolor="white", linewidth=0.7)
    ax.set_xlim(55.0, 260.0)
    ax.set_ylim(45.0, 147.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Final snapshot {plotfile.name}: streamlines over vorticity, pitch={math.degrees(angle):.1f} deg"
    )
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("vorticity")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_force_history(out_png: Path, t_conv: np.ndarray, cd: np.ndarray, cl: np.ndarray, par: dict[str, float]) -> None:
    angle_deg = np.degrees(pitch_angle(t_conv, par))
    tail_mask = t_conv >= max(t_conv[-1] - 3.0, t_conv[0])

    fig, ax = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True, sharex="col")

    ax[0, 0].plot(t_conv, angle_deg, color="tab:purple", lw=1.0)
    ax[1, 0].plot(t_conv, cd, color="tab:blue", lw=0.9)
    ax[2, 0].plot(t_conv, cl, color="tab:orange", lw=0.9)

    ax[0, 1].plot(t_conv[tail_mask], angle_deg[tail_mask], color="tab:purple", lw=1.0)
    ax[1, 1].plot(t_conv[tail_mask], cd[tail_mask], color="tab:blue", lw=0.9)
    ax[2, 1].plot(t_conv[tail_mask], cl[tail_mask], color="tab:orange", lw=0.9)

    ax[0, 0].set_title("Full run")
    ax[0, 1].set_title("Last three pitching periods")
    ax[0, 0].set_ylabel("pitch [deg]")
    ax[1, 0].set_ylabel("Cd (ME)")
    ax[2, 0].set_ylabel("Cl (ME)")
    ax[2, 0].set_xlabel("convective time t*U/c")
    ax[2, 1].set_xlabel("convective time t*U/c")

    for axi in ax.ravel():
        axi.grid(alpha=0.25)

    fig.suptitle("Pitching-airfoil motion and force history")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot pitching-airfoil flow snapshots and force history."
    )
    ap.add_argument("--inputs", type=Path, default=INPUTS)
    ap.add_argument("--geometry-header", type=Path, default=GEOM_HDR)
    ap.add_argument("--force-file", type=Path, default=FORCE_FILE)
    ap.add_argument("--plot-root", type=Path, default=PLOT_ROOT)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--vorticity-limit",
        type=float,
        default=0.01,
        help="Symmetric vorticity color limit for all vorticity plots (+/- value).",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inp = parse_inputs(args.inputs)
    par = parse_header_constants(args.geometry_header)
    plotfiles = latest_plotfiles(args.plot_root)
    _, t_conv, cd, cl = load_force(args.force_file)
    chosen = pick_snapshot_indices(plotfiles, count=4)
    final_snapshot = pick_snapshot_indices(plotfiles, count=1)[-1]

    plot_vorticity_cycle(
        args.output_dir / "pitching_airfoil_vorticity_cycle.png",
        chosen,
        inp,
        par,
        args.vorticity_limit,
    )
    plot_final_streamlines(
        args.output_dir / "pitching_airfoil_final_streamlines.png",
        final_snapshot,
        inp,
        par,
        args.vorticity_limit,
    )
    plot_force_history(
        args.output_dir / "pitching_airfoil_force_history.png", t_conv, cd, cl, par
    )

    print("wrote:")
    for name in [
        "pitching_airfoil_vorticity_cycle.png",
        "pitching_airfoil_final_streamlines.png",
        "pitching_airfoil_force_history.png",
    ]:
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
