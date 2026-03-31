#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import yt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def latest_plotfile(root: Path) -> Path:
    plots = sorted(p for p in root.glob("plt*") if p.is_dir())
    if not plots:
        raise FileNotFoundError(f"No plotfiles found under {root}")
    return plots[-1]


def load_velocity(plotfile: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    ds = yt.load(str(plotfile))
    base_dims = np.array(ds.domain_dimensions, dtype=int)
    max_lev = int(ds.index.max_level)
    dims = base_dims * (int(ds.refine_by) ** max_lev)
    cg = ds.covering_grid(level=max_lev, left_edge=ds.domain_left_edge, dims=dims, num_ghost_zones=0)

    ux = np.array(cg[("boxlib", "ux")], dtype=np.float64)
    uy = np.array(cg[("boxlib", "uy")], dtype=np.float64)
    uz = np.array(cg[("boxlib", "uz")], dtype=np.float64)

    lo = np.array(ds.domain_left_edge, dtype=np.float64)
    hi = np.array(ds.domain_right_edge, dtype=np.float64)
    return ux, uy, uz, float(ds.current_time), lo, hi


def parse_sphere_params(inputs_path: Path) -> tuple[float, float, float, float] | None:
    if not inputs_path.exists():
        return None
    text = inputs_path.read_text().splitlines()
    vals = {}
    pattern = re.compile(r"^\s*ibm\.(x0|y0|z0|R)\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)")
    for line in text:
        m = pattern.match(line)
        if m:
            vals[m.group(1)] = float(m.group(2))
    if all(k in vals for k in ("x0", "y0", "z0", "R")):
        return vals["x0"], vals["y0"], vals["z0"], vals["R"]
    return None


def parse_u0(inputs_path: Path) -> float | None:
    if not inputs_path.exists():
        return None
    pattern = re.compile(r"^\s*lbmPhysicalParameters\.U0\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)")
    for line in inputs_path.read_text().splitlines():
        m = pattern.match(line)
        if m:
            return float(m.group(1))
    return None


def add_sphere(ax, center: tuple[float, float, float], radius: float) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    x0, y0, z0 = center
    x = x0 + radius * np.outer(np.cos(u), np.sin(v))
    y = y0 + radius * np.outer(np.sin(u), np.sin(v))
    z = z0 + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x,
        y,
        z,
        color="#4a4a4a",
        alpha=1.0,
        linewidth=0.2,
        edgecolor="#333333",
        shade=True,
        zorder=0,
    )


def add_scale_bar_and_triad(
    ax,
    lo: np.ndarray,
    hi: np.ndarray,
    sphere: tuple[float, float, float, float] | None,
) -> None:
    xr = hi[0] - lo[0]
    yr = hi[1] - lo[1]
    zr = hi[2] - lo[2]
    if xr <= 0 or yr <= 0 or zr <= 0:
        return

    margin = 0.08

    # --- Coordinate triad: bottom-right (data coords), far from sphere/wake ---
    triad_base = np.array([hi[0] - margin * xr, lo[1] + margin * yr, lo[2] + margin * zr])
    triad_len = 0.12 * min(xr, yr, zr)
    ox, oy, oz = triad_base
    ax.quiver(ox, oy, oz, triad_len, 0.0, 0.0, color="#444444", linewidth=2, arrow_length_ratio=0.15)
    ax.quiver(ox, oy, oz, 0.0, triad_len, 0.0, color="#444444", linewidth=2, arrow_length_ratio=0.15)
    ax.quiver(ox, oy, oz, 0.0, 0.0, triad_len, color="#444444", linewidth=2, arrow_length_ratio=0.15)
    label_offset = 1.35 * triad_len
    ax.text(ox + label_offset, oy, oz, "x", color="#444444", fontsize=9, ha="left", va="center")
    ax.text(ox, oy + label_offset, oz, "y", color="#444444", fontsize=9, ha="center", va="bottom")
    ax.text(ox, oy, oz + label_offset, "z", color="#444444", fontsize=9, ha="center", va="bottom")

    # Scale bar removed per request.


def compute_qcriterion(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    du_dx, du_dy, du_dz = np.gradient(ux, dx, dy, dz, edge_order=2)
    dv_dx, dv_dy, dv_dz = np.gradient(uy, dx, dy, dz, edge_order=2)
    dw_dx, dw_dy, dw_dz = np.gradient(uz, dx, dy, dz, edge_order=2)

    s11 = du_dx
    s22 = dv_dy
    s33 = dw_dz
    s12 = 0.5 * (du_dy + dv_dx)
    s13 = 0.5 * (du_dz + dw_dx)
    s23 = 0.5 * (dv_dz + dw_dy)

    o12 = 0.5 * (du_dy - dv_dx)
    o13 = 0.5 * (du_dz - dw_dx)
    o23 = 0.5 * (dv_dz - dw_dy)

    s2 = s11 * s11 + s22 * s22 + s33 * s33 + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)
    o2 = 2.0 * (o12 * o12 + o13 * o13 + o23 * o23)

    return 0.5 * (o2 - s2)


def compute_lambda2(ux: np.ndarray, uy: np.ndarray, uz: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    du_dx, du_dy, du_dz = np.gradient(ux, dx, dy, dz, edge_order=2)
    dv_dx, dv_dy, dv_dz = np.gradient(uy, dx, dy, dz, edge_order=2)
    dw_dx, dw_dy, dw_dz = np.gradient(uz, dx, dy, dz, edge_order=2)

    s11 = du_dx
    s22 = dv_dy
    s33 = dw_dz
    s12 = 0.5 * (du_dy + dv_dx)
    s13 = 0.5 * (du_dz + dw_dx)
    s23 = 0.5 * (dv_dz + dw_dy)

    o12 = 0.5 * (du_dy - dv_dx)
    o13 = 0.5 * (du_dz - dw_dx)
    o23 = 0.5 * (dv_dz - dw_dy)

    n = ux.size
    s = np.empty((n, 3, 3), dtype=np.float64)
    o = np.empty((n, 3, 3), dtype=np.float64)

    s[..., 0, 0] = s11.ravel()
    s[..., 1, 1] = s22.ravel()
    s[..., 2, 2] = s33.ravel()
    s[..., 0, 1] = s[..., 1, 0] = s12.ravel()
    s[..., 0, 2] = s[..., 2, 0] = s13.ravel()
    s[..., 1, 2] = s[..., 2, 1] = s23.ravel()

    o[..., 0, 0] = 0.0
    o[..., 1, 1] = 0.0
    o[..., 2, 2] = 0.0
    o[..., 0, 1] = o12.ravel()
    o[..., 1, 0] = -o12.ravel()
    o[..., 0, 2] = o13.ravel()
    o[..., 2, 0] = -o13.ravel()
    o[..., 1, 2] = o23.ravel()
    o[..., 2, 1] = -o23.ravel()

    j = np.matmul(s, s) + np.matmul(o, o)
    eigvals = np.linalg.eigvalsh(j)
    lambda2 = eigvals[:, 1]
    return lambda2.reshape(ux.shape)


def select_iso(data: np.ndarray, method: str, iso: float | None, percentile: float | None) -> float:
    if iso is not None:
        return iso
    if method == "q":
        pos = data[data > 0]
        if pos.size == 0:
            return float(np.percentile(data, 95))
        return float(np.percentile(pos, percentile if percentile is not None else 95))
    neg = data[data < 0]
    if neg.size == 0:
        return float(np.percentile(data, 5))
    return float(np.percentile(neg, percentile if percentile is not None else 5))


def parse_list(arg: str, cast) -> list:
    items = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if chunk:
            items.append(cast(chunk))
    return items


def normalize_list(items: list, n: int, default) -> list:
    if not items:
        return [default] * n
    if len(items) >= n:
        return items[:n]
    # Extend using the last value.
    return items + [items[-1]] * (n - len(items))


def select_isos(data: np.ndarray, method: str, percentiles: list[float]) -> list[float]:
    if method == "q":
        pos = data[data > 0]
        if pos.size == 0:
            return [float(np.percentile(data, p)) for p in percentiles]
        return [float(np.percentile(pos, p)) for p in percentiles]
    neg = data[data < 0]
    if neg.size == 0:
        return [float(np.percentile(data, p)) for p in percentiles]
    return [float(np.percentile(neg, p)) for p in percentiles]


def apply_iso_floor(isos: list[float], method: str, min_fraction: float) -> list[float]:
    if not isos or min_fraction <= 0.0:
        return isos
    if method == "q":
        top = max(isos)
        if not np.isfinite(top) or top <= 0.0:
            return isos
        limit = top * min_fraction
        return [max(v, limit) for v in isos]
    # lambda2: negative values, keep away from zero
    bottom = min(isos)
    if not np.isfinite(bottom) or bottom >= 0.0:
        return isos
    limit = bottom * min_fraction  # still negative
    return [min(v, limit) for v in isos]


def spread_isos_if_needed(isos: list[float], method: str, min_fraction: float) -> list[float]:
    if len(isos) < 2:
        return isos
    # Detect duplicates after clamping (rounded to avoid float noise).
    rounded = {round(v, 12) for v in isos}
    if len(rounded) == len(isos):
        return isos

    if method == "q":
        max_iso = max(isos)
        if not np.isfinite(max_iso) or max_iso <= 0.0:
            return isos
        min_iso = max_iso * max(min_fraction, 1.0e-6)
        # Geometric spacing keeps separation while preserving scale.
        vals = np.geomspace(min_iso, max_iso, num=len(isos))
        return [float(v) for v in vals]

    # lambda2: negative values, keep away from zero
    min_iso = min(isos)
    if not np.isfinite(min_iso) or min_iso >= 0.0:
        return isos
    max_iso = min_iso * max(min_fraction, 1.0e-6)  # closer to zero (still negative)
    vals = np.geomspace(abs(min_iso), abs(max_iso), num=len(isos))
    return [-float(v) for v in vals]


def plot_isosurface(
    volume: np.ndarray,
    isos: list[float],
    colors: list[str],
    alphas: list[float],
    lo: np.ndarray,
    hi: np.ndarray,
    stride: int,
    output: Path,
    title: str,
    sphere: tuple[float, float, float, float] | None,
) -> None:
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="3d")

    scale = np.array(
        [
            (hi[0] - lo[0]) / (volume.shape[0] - 1),
            (hi[1] - lo[1]) / (volume.shape[1] - 1),
            (hi[2] - lo[2]) / (volume.shape[2] - 1),
        ]
    )

    data_min = float(np.nanmin(volume))
    data_max = float(np.nanmax(volume))

    for iso, color, alpha in zip(isos, colors, alphas):
        if not np.isfinite(iso) or iso <= data_min or iso >= data_max:
            # Skip levels outside the data range (no surface to extract).
            continue
        verts, faces, _, _ = measure.marching_cubes(volume, level=iso)
        verts = verts * scale + lo
        mesh = Poly3DCollection(verts[faces], alpha=alpha, facecolor=color, edgecolor="none")
        ax.add_collection3d(mesh)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    # Match the requested view (az=119.4, elev=9.0).
    ax.view_init(elev=9.0, azim=119.4)
    # Keep physical units equal across axes.
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))

    if sphere is not None:
        add_sphere(ax, (sphere[0], sphere[1], sphere[2]), sphere[3])

    add_scale_bar_and_triad(ax, lo, hi, sphere)

    # Clean background: hide axes, grids, and panes.
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis._axinfo["grid"]["linewidth"] = 0
    ax.set_axis_off()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def write_interactive_html(
    volume: np.ndarray,
    isos: list[float],
    colors: list[str],
    alphas: list[float],
    lo: np.ndarray,
    hi: np.ndarray,
    title: str,
    sphere: tuple[float, float, float, float] | None,
    output: Path,
    view_azim: float,
    view_elev: float,
    view_dist: float,
) -> None:
    import plotly.graph_objects as go
    import plotly.io as pio

    data_min = float(np.nanmin(volume))
    data_max = float(np.nanmax(volume))
    scale = np.array(
        [
            (hi[0] - lo[0]) / (volume.shape[0] - 1),
            (hi[1] - lo[1]) / (volume.shape[1] - 1),
            (hi[2] - lo[2]) / (volume.shape[2] - 1),
        ]
    )

    traces: list[go.BaseTraceType] = []
    for iso, color, alpha in zip(isos, colors, alphas):
        if not np.isfinite(iso) or iso <= data_min or iso >= data_max:
            continue
        verts, faces, _, _ = measure.marching_cubes(volume, level=iso)
        verts = verts * scale + lo
        traces.append(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=alpha,
                name=f"iso={iso:.2e}",
                hoverinfo="skip",
            )
        )

    if sphere is not None:
        x0, y0, z0, R = sphere
        nu, nv = 48, 24
        u = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
        v = np.linspace(0.0, np.pi, nv)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        x = x0 + R * np.cos(uu) * np.sin(vv)
        y = y0 + R * np.sin(uu) * np.sin(vv)
        z = z0 + R * np.cos(vv)
        verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        faces_i: list[int] = []
        faces_j: list[int] = []
        faces_k: list[int] = []
        for i in range(nu):
            i2 = (i + 1) % nu
            for j in range(nv - 1):
                idx = i * nv + j
                idx_i2 = i2 * nv + j
                idx_j2 = i * nv + (j + 1)
                idx_i2_j2 = i2 * nv + (j + 1)
                faces_i.extend([idx, idx])
                faces_j.extend([idx_i2, idx_i2_j2])
                faces_k.extend([idx_i2_j2, idx_j2])
        traces.append(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces_i,
                j=faces_j,
                k=faces_k,
                color="#4a4a4a",
                opacity=1.0,
                name="sphere",
                hoverinfo="skip",
            )
        )

    az = np.deg2rad(view_azim)
    el = np.deg2rad(view_elev)
    eye_dir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = eye_dir * float(view_dist)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        width=1000,
        height=700,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                up=dict(x=0.0, y=0.0, z=1.0),
                center=dict(x=0.0, y=0.0, z=0.0),
                eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
            ),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        showlegend=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    div = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        div_id="qplot",
        config={"displaylogo": False, "responsive": True},
    )
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <style>
      body {{ margin: 0; background: #ffffff; }}
      #angle_readout {{
        position: absolute;
        top: 12px;
        left: 12px;
        padding: 6px 10px;
        font-family: Arial, sans-serif;
        font-size: 13px;
        color: #333333;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #dddddd;
        border-radius: 4px;
        z-index: 10;
      }}
    </style>
  </head>
  <body>
    <div id="angle_readout">az=0°, elev=0°, dist=0</div>
    {div}
    <script>
      const gd = document.getElementById('qplot');
      function updateAngles() {{
        const cam = (gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene.camera) ? gd._fullLayout.scene.camera : {{}};
        const eye = cam.eye || {{x: 0, y: 0, z: 0}};
        const r = Math.sqrt(eye.x*eye.x + eye.y*eye.y + eye.z*eye.z) || 0.0;
        const az = Math.atan2(eye.y, eye.x) * 180 / Math.PI;
        const el = Math.atan2(eye.z, Math.sqrt(eye.x*eye.x + eye.y*eye.y)) * 180 / Math.PI;
        document.getElementById('angle_readout').innerText =
          `az=${{az.toFixed(1)}}°, elev=${{el.toFixed(1)}}°, dist=${{r.toFixed(2)}}`;
      }}
      gd.on('plotly_relayout', updateAngles);
      gd.on('plotly_relayouting', updateAngles);
      gd.on('plotly_afterplot', updateAngles);
      updateAngles();
    </script>
  </body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Q-criterion or lambda2 isosurface for Sphere flow.")
    parser.add_argument("--plot-root", type=Path, default=Path("out_sphere"))
    parser.add_argument("--plotfile", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("_artifacts/sphere_qcriterion.png"))
    parser.add_argument("--html", type=Path, default=None, help="Optional interactive HTML output")
    parser.add_argument("--method", choices=["q", "lambda2"], default="q")
    parser.add_argument("--iso", type=float, default=None)
    parser.add_argument("--percentile", type=float, default=None)
    parser.add_argument("--isos", type=str, default=None, help="Comma-separated iso values")
    parser.add_argument("--percentiles", type=str, default=None, help="Comma-separated percentiles for iso selection")
    parser.add_argument("--colors", type=str, default="#cfcfcf,#4a90e2,#e74c3c", help="Comma-separated colors")
    parser.add_argument("--alphas", type=str, default="0.35,0.25,0.25", help="Comma-separated alpha values")
    parser.add_argument("--thresholds-file", type=Path, default=None, help="Cache iso thresholds for consistent movies")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--inputs", type=Path, default=Path("inputs"), help="Inputs file to read ibm.x0/y0/z0/R")
    parser.add_argument("--view-azim", type=float, default=119.4)
    parser.add_argument("--view-elev", type=float, default=9.0)
    parser.add_argument("--view-dist", type=float, default=1.51)
    parser.add_argument(
        "--iso-min-fraction",
        type=float,
        default=0.05,
        help="Clamp iso values to at least this fraction of the strongest iso (reduces near-zero noise).",
    )
    args = parser.parse_args()

    plotfile = args.plotfile or latest_plotfile(args.plot_root)
    sphere = parse_sphere_params(args.inputs)
    u0 = parse_u0(args.inputs)
    ux, uy, uz, t, lo, hi = load_velocity(plotfile)

    if args.stride > 1:
        ux = ux[::args.stride, ::args.stride, ::args.stride]
        uy = uy[::args.stride, ::args.stride, ::args.stride]
        uz = uz[::args.stride, ::args.stride, ::args.stride]

    nx, ny, nz = ux.shape
    dx = (hi[0] - lo[0]) / float(nx)
    dy = (hi[1] - lo[1]) / float(ny)
    dz = (hi[2] - lo[2]) / float(nz)

    t_label = f"t={t:.1f}"
    if sphere is not None and u0 is not None and sphere[3] > 0:
        t_conv = t * u0 / (2.0 * sphere[3])
        t_label = f"tU/D = {t_conv:.2f}"

    if args.method == "q":
        data = compute_qcriterion(ux, uy, uz, dx, dy, dz)
        title = f"Q-criterion isosurfaces ({t_label})"
    else:
        data = compute_lambda2(ux, uy, uz, dx, dy, dz)
        title = f"Lambda2 isosurfaces ({t_label})"

    if args.isos:
        isos = parse_list(args.isos, float)
    elif args.percentiles:
        percentiles = parse_list(args.percentiles, float)
        isos = select_isos(data, args.method, percentiles)
    elif args.iso is not None or args.percentile is not None:
        isos = [select_iso(data, args.method, args.iso, args.percentile)]
    else:
        default_percentiles = [95.0, 75.0, 50.0]
        isos = select_isos(data, args.method, default_percentiles)

    if args.thresholds_file is not None:
        thresholds_path = args.thresholds_file
        if thresholds_path.exists():
            try:
                payload = json.loads(thresholds_path.read_text())
                if payload.get("method") == args.method and "isos" in payload:
                    isos = [float(v) for v in payload["isos"]]
            except Exception:
                pass
        else:
            thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            thresholds_path.write_text(
                json.dumps({"method": args.method, "isos": isos, "iso_min_fraction": args.iso_min_fraction})
            )

    # Clamp tiny iso values so near-zero noise doesn't create boundary speckles.
    clamped_isos = apply_iso_floor(isos, args.method, args.iso_min_fraction)
    spaced_isos = spread_isos_if_needed(clamped_isos, args.method, args.iso_min_fraction)
    if spaced_isos != isos:
        isos = spaced_isos
        if args.thresholds_file is not None:
            thresholds_path = args.thresholds_file
            thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            thresholds_path.write_text(
                json.dumps({"method": args.method, "isos": isos, "iso_min_fraction": args.iso_min_fraction})
            )

    colors = normalize_list(parse_list(args.colors, str), len(isos), "#f2f2f2")
    alphas = normalize_list(parse_list(args.alphas, float), len(isos), 0.25)
    # Draw larger (lower iso) surfaces first so small/high-iso surfaces stay visible.
    order = sorted(zip(isos, colors, alphas), key=lambda t: t[0])
    isos, colors, alphas = zip(*order)
    isos = list(isos)
    colors = list(colors)
    alphas = list(alphas)
    plot_isosurface(data, isos, colors, alphas, lo, hi, args.stride, args.output, title, sphere)
    iso_report = ", ".join(f"{v:.3e}" for v in isos)
    print(f"Wrote {args.output} from {plotfile} (isos={iso_report})")
    if args.html is not None:
        write_interactive_html(
            data,
            isos,
            colors,
            alphas,
            lo,
            hi,
            title,
            sphere,
            args.html,
            args.view_azim,
            args.view_elev,
            args.view_dist,
        )


if __name__ == "__main__":
    main()
