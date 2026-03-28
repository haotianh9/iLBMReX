from pathlib import Path
import argparse

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt


def load_composite_vorticity(plotfile: Path):
    ds = yt.load(str(plotfile))
    ds.force_periodicity()

    level = ds.index.max_level
    ref = 2 ** level
    dims = np.array(ds.domain_dimensions, dtype=int) * ref
    cg = ds.smoothed_covering_grid(level=level, left_edge=ds.domain_left_edge, dims=dims)

    vor = np.array(cg[("boxlib", "vor")])[:, :, 0]
    xlo, ylo = ds.domain_left_edge.d[:2]
    xhi, yhi = ds.domain_right_edge.d[:2]

    return {
        "plotfile": plotfile.name,
        "time": float(ds.current_time),
        "vor": vor,
        "extent": [xlo, xhi, ylo, yhi],
    }


def render_frame(record, vmax):
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    im = ax.imshow(
        record["vor"].T,
        extent=record["extent"],
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title(f"{record['plotfile']}   t = {record['time']:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("vorticity")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Create a Taylor-Green vorticity movie from AMReX plotfiles.")
    parser.add_argument("--plot-root", default="out_validation", help="Directory containing plotfiles.")
    parser.add_argument(
        "--output",
        default="_artifacts/taylor_green_vorticity_evolution.mp4",
        help="Output movie path.",
    )
    parser.add_argument("--fps", type=int, default=4, help="Frames per second.")
    args = parser.parse_args()

    plot_root = Path(args.plot_root)
    plotfiles = sorted(plot_root.glob("plt*"))
    if not plotfiles:
        raise FileNotFoundError(f"No plotfiles found in {plot_root}")

    records = [load_composite_vorticity(plotfile) for plotfile in plotfiles]
    vmax = max(float(np.max(np.abs(record["vor"]))) for record in records)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output, fps=args.fps) as writer:
        for record in records:
            writer.append_data(render_frame(record, vmax))

    print(f"Wrote movie to {output}")


if __name__ == "__main__":
    main()
