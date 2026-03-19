#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yt


def _parse_floats(line: str) -> list[float]:
    vals = []
    for tok in line.split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def load_force(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for raw in path.open("rb"):
        line = raw.decode("ascii", errors="ignore").replace("\x00", " ").strip()
        if not line or line.startswith("#"):
            continue
        vals = _parse_floats(line)
        if len(vals) < 6:
            continue
        rows.append(vals)

    if not rows:
        raise RuntimeError(f"No readable force rows found in {path}")

    ncols = max(len(r) for r in rows)
    if ncols < 22:
        raise RuntimeError(f"Unexpected force format in {path}: max columns = {ncols}")

    t = np.asarray([r[22] if len(r) > 22 else r[1] for r in rows], dtype=np.float64)
    cd = np.asarray([r[20] if len(r) > 20 else r[4] for r in rows], dtype=np.float64)
    cl = np.asarray([r[21] if len(r) > 21 else r[5] for r in rows], dtype=np.float64)
    return t, cd, cl


def trimmed_ylim(y: np.ndarray, ignore_frac: float, pad_frac: float = 0.08) -> tuple[float, float]:
    n = len(y)
    if n == 0:
        raise RuntimeError("Cannot determine y limits from an empty series")
    start = min(max(int(ignore_frac * n), 0), max(n - 1, 0))
    core = y[start:]
    lo = float(np.min(core))
    hi = float(np.max(core))
    span = max(hi - lo, 1.0e-8)
    pad = pad_frac * span
    return lo - pad, hi + pad


def resolve_force(explicit: str | None, pattern: str) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(Path("_artifacts").glob(pattern))
    return candidates[-1] if candidates else None


def infer_tag(path: Path, fallback: str) -> str:
    name = path.stem
    if name == "force":
        return "current"
    if name.startswith("force_"):
        return name[len("force_"):]
    return fallback


def parse_cylinder_geometry(inputs_path: Path) -> tuple[float, float, float]:
    keys = {"ibm.x0": None, "ibm.y0": None, "ibm.R": None}
    with inputs_path.open() as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            lhs, rhs = [part.strip() for part in line.split("=", 1)]
            if lhs in keys:
                keys[lhs] = float(rhs.split()[0])
    missing = [k for k, v in keys.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing cylinder geometry keys in {inputs_path}: {missing}")
    return keys["ibm.x0"], keys["ibm.y0"], keys["ibm.R"]


def latest_plotfile(root: Path) -> Path:
    cands = []
    for p in root.glob("plt*"):
        if p.is_dir() and re.fullmatch(r"plt\d+", p.name):
            cands.append(p)
    if not cands:
        raise FileNotFoundError(f"No plotfile directories like pltNNNNN under {root}")
    cands.sort()
    return cands[-1]


def load_fields2d(plotfile: Path) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    ds = yt.load(str(plotfile))
    dims = tuple(int(v) for v in ds.domain_dimensions)
    cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=dims)

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


def add_cylinder(ax: plt.Axes, x0: float, y0: float, r: float) -> None:
    patch = mpatches.Circle((x0, y0), r, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(patch)


def plot_time_hist(
    out_png: Path,
    title: str,
    t: np.ndarray,
    cd: np.ndarray,
    cl: np.ndarray,
    cd_ylim_ignore_frac: float,
) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    ax[0].plot(t, cd, lw=0.9, color="tab:blue")
    ax[0].set_ylabel("Cd (ME)")
    ax[0].set_ylim(*trimmed_ylim(cd, cd_ylim_ignore_frac))
    ax[0].grid(alpha=0.25)
    ax[0].set_title(title)

    ax[1].plot(t, cl, lw=0.9, color="tab:orange")
    ax[1].set_xlabel("Convective time t*U/D")
    ax[1].set_ylabel("Cl (ME)")
    ax[1].grid(alpha=0.25)

    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_spectrum(out_png: Path, title: str, t: np.ndarray, cl: np.ndarray) -> None:
    x = cl - np.mean(cl)
    dt = (t[-1] - t[0]) / max(len(t) - 1, 1)
    f = np.fft.rfftfreq(len(x), d=dt)
    amp = np.abs(np.fft.rfft(x))

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(f, amp, lw=1.0)
    ax.set_xlim(0.0, 0.4)
    ax.set_xlabel("Strouhal number St")
    ax.set_ylabel("|FFT(Cl)|")
    ax.grid(alpha=0.25)
    ax.set_title(title)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_overlay(
    out_png: Path,
    t1: np.ndarray,
    cd1: np.ndarray,
    cl1: np.ndarray,
    label1: str,
    t2: np.ndarray,
    cd2: np.ndarray,
    cl2: np.ndarray,
    label2: str,
    cd_ylim_ignore_frac: float,
) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    ax[0].plot(t1, cd1, lw=0.9, label=label1)
    ax[0].plot(t2, cd2, lw=0.9, label=label2)
    ax[0].set_ylabel("Cd (ME)")
    lo1, hi1 = trimmed_ylim(cd1, cd_ylim_ignore_frac)
    lo2, hi2 = trimmed_ylim(cd2, cd_ylim_ignore_frac)
    ax[0].set_ylim(min(lo1, lo2), max(hi1, hi2))
    ax[0].grid(alpha=0.25)
    ax[0].legend()
    ax[0].set_title(f"Cylinder Flow: {label1} vs {label2}")

    ax[1].plot(t1, cl1, lw=0.9, label=label1)
    ax[1].plot(t2, cl2, lw=0.9, label=label2)
    ax[1].set_xlabel("Convective time t*U/D")
    ax[1].set_ylabel("Cl (ME)")
    ax[1].grid(alpha=0.25)
    ax[1].legend()

    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_flow_fields(
    out_png: Path,
    title: str,
    plotfile: Path,
    x0: float,
    y0: float,
    r: float,
) -> None:
    extent, ux, uy, vor = load_fields2d(plotfile)
    speed = np.sqrt(ux * ux + uy * uy)
    vmax_vor = float(np.max(np.abs(vor)))
    vor_norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax_vor, vmax=vmax_vor)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fields = [
        ("ux", ux, "viridis", None),
        ("uy", uy, "viridis", None),
        ("|u|", speed, "viridis", None),
        ("vorticity", vor, "coolwarm", vor_norm),
    ]
    for axi, (name, arr, cmap, norm) in zip(ax.ravel(), fields):
        im = axi.imshow(arr.T, origin="lower", extent=extent, cmap=cmap, norm=norm, aspect="equal")
        add_cylinder(axi, x0, y0, r)
        axi.set_title(name)
        axi.set_xlabel("x")
        axi.set_ylabel("y")
        fig.colorbar(im, ax=axi, shrink=0.9)

    fig.suptitle(f"{title}\n{plotfile.name}")
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot cylinder force histories and flow fields.")
    ap.add_argument("--re100-force", default=None)
    ap.add_argument("--re200-force", default=None)
    ap.add_argument("--plot-root", default="out_cylinder")
    ap.add_argument("--inputs", default="inputs")
    ap.add_argument("--out-dir", default="_artifacts")
    ap.add_argument(
        "--cd-ylim-ignore-frac",
        type=float,
        default=0.05,
        help="Ignore the first fraction of Cd samples when selecting the Cd y-range.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p100 = resolve_force(args.re100_force, "force_re100_*.dat")
    p200 = resolve_force(args.re200_force, "force_re200_*.dat")
    current = Path("force.dat")
    if p100 is None and p200 is None and current.exists():
        p100 = current

    if p100 is not None:
        t100, cd100, cl100 = load_force(p100)
        tag100 = infer_tag(p100, "re100")
        plot_time_hist(
            out_dir / f"{tag100}_cd_cl_time.png",
            f"{tag100}: Cd/Cl vs Convective Time",
            t100,
            cd100,
            cl100,
            args.cd_ylim_ignore_frac,
        )
        plot_spectrum(
            out_dir / f"{tag100}_cl_spectrum.png",
            f"{tag100}: Cl Spectrum",
            t100,
            cl100,
        )
        print(out_dir / f"{tag100}_cd_cl_time.png")
        print(out_dir / f"{tag100}_cl_spectrum.png")
    else:
        t100 = cd100 = cl100 = None
        tag100 = None

    if p200 is not None:
        t200, cd200, cl200 = load_force(p200)
        tag200 = infer_tag(p200, "re200")
        plot_time_hist(
            out_dir / f"{tag200}_cd_cl_time.png",
            f"{tag200}: Cd/Cl vs Convective Time",
            t200,
            cd200,
            cl200,
            args.cd_ylim_ignore_frac,
        )
        plot_spectrum(
            out_dir / f"{tag200}_cl_spectrum.png",
            f"{tag200}: Cl Spectrum",
            t200,
            cl200,
        )
        print(out_dir / f"{tag200}_cd_cl_time.png")
        print(out_dir / f"{tag200}_cl_spectrum.png")
    else:
        t200 = cd200 = cl200 = None
        tag200 = None

    if p100 is not None and p200 is not None:
        plot_overlay(
            out_dir / f"{tag100}_{tag200}_cd_cl_overlay.png",
            t100,
            cd100,
            cl100,
            tag100,
            t200,
            cd200,
            cl200,
            tag200,
            args.cd_ylim_ignore_frac,
        )
        print(out_dir / f"{tag100}_{tag200}_cd_cl_overlay.png")

    plot_root = Path(args.plot_root)
    if plot_root.exists():
        plotfile = latest_plotfile(plot_root)
        x0, y0, r = parse_cylinder_geometry(Path(args.inputs))
        plot_flow_fields(
            out_dir / "latest_flow_fields.png",
            "Cylinder Flow Fields",
            plotfile,
            x0,
            y0,
            r,
        )
        print(out_dir / "latest_flow_fields.png")


if __name__ == "__main__":
    main()
