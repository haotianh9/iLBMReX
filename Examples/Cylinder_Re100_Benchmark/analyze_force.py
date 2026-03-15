#!/usr/bin/env python3
import argparse
import math
from pathlib import Path


def _parse_floats(line):
    out = []
    for tok in line.split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def read_force(path: Path, source: str):
    # Current format:
    # 0:step 1:time 2:Fx_tc 3:Fy_tc 4:Cd 5:Cl
    # 6:Fx_raw 7:Fy_raw 8:Cd_raw 9:Cl_raw
    # 10:Fx_eul 11:Fy_eul 12:Cd_eul 13:Cl_eul
    # 14:Fx_marker 15:Fy_marker 16:Cd_marker 17:Cl_marker
    # 18:Fx_me 19:Fy_me 20:Cd_me 21:Cl_me 22:t_conv
    # Legacy format (still supported):
    # 0:step 1:time 2:Fx 3:Fy 4:Cd 5:Cl ... 18:t_conv
    cols = {"primary": (4, 5), "eulerian": (12, 13), "marker": (16, 17), "me": (20, 21)}
    c_cd, c_cl = cols[source]

    rows = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vals = _parse_floats(s)
        if len(vals) < 6:
            continue
        step = int(vals[0])
        time_lbm = vals[1]
        if len(vals) > 22:
            time_conv = vals[22]
        elif len(vals) > 18:
            time_conv = vals[18]
        else:
            time_conv = None
        if c_cl < len(vals):
            cd = vals[c_cd]
            cl = vals[c_cl]
        else:
            cd = vals[4]
            cl = vals[5]
        rows.append((step, time_lbm, time_conv, cd, cl))
    return rows


def local_maxima(ys):
    idx = []
    for i in range(1, len(ys) - 1):
        if ys[i] > ys[i - 1] and ys[i] >= ys[i + 1]:
            idx.append(i)
    return idx


def estimate_st_peak(times, cls, d, u, min_sep_period_frac, amp_frac):
    cmean = sum(cls) / len(cls)
    cvar = sum((x - cmean) ** 2 for x in cls) / len(cls)
    cstd = math.sqrt(max(cvar, 0.0))
    if cstd < 1.0e-12:
        return None, 0, "lift variance too small"

    # Strouhal in [0.05, 0.5] => period in [d/(0.5u), d/(0.05u)]
    t_min_period = d / (0.5 * u)
    min_sep = min_sep_period_frac * t_min_period
    amp_thr = amp_frac * cstd

    pidx_raw = local_maxima(cls)
    pidx = []
    last_t = -1.0e100
    for i in pidx_raw:
        if cls[i] - cmean < amp_thr:
            continue
        if times[i] - last_t < min_sep:
            continue
        pidx.append(i)
        last_t = times[i]

    if len(pidx) < 3:
        return None, len(pidx), "insufficient filtered peaks"

    peak_t = [times[i] for i in pidx]
    periods = [peak_t[i + 1] - peak_t[i] for i in range(len(peak_t) - 1)]
    tmean = sum(periods) / len(periods)
    if tmean <= 0:
        return None, len(periods), "non-positive period"

    f = 1.0 / tmean
    st = f * d / u
    return st, len(periods), ""


def estimate_st_fft(times, cls, d, u, st_band_min, st_band_max):
    try:
        import numpy as np
    except Exception:
        return None, "numpy unavailable"

    if len(times) < 16:
        return None, "too few points for fft"

    dt = (times[-1] - times[0]) / max(len(times) - 1, 1)
    if dt <= 0:
        return None, "non-positive dt"

    x = np.asarray(cls, dtype=float)
    x = x - x.mean()
    if float(np.sqrt(np.mean(x * x))) < 1.0e-12:
        return None, "lift variance too small"

    f = np.fft.rfftfreq(len(x), d=dt)
    X = np.abs(np.fft.rfft(x))
    if len(X) < 3:
        return None, "insufficient fft bins"

    st = f * d / u
    band = (st >= st_band_min) & (st <= st_band_max)
    if len(band) != len(X):
        return None, "internal fft size mismatch"
    if not np.any(band):
        return None, "no fft bins in requested St band"

    ib = np.where(band)[0]
    j = int(ib[np.argmax(X[ib])])
    if j <= 0 or j >= len(f):
        return None, "invalid dominant bin in St band"

    st_dom = float(st[j])
    return st_dom, ""


def main():
    ap = argparse.ArgumentParser(description="Analyze Cd/St from force.dat")
    ap.add_argument("--force", default="force.dat")
    ap.add_argument("--source", choices=["primary", "eulerian", "marker", "me"], default="primary")
    ap.add_argument("--u0", type=float, default=0.03)
    ap.add_argument("--diameter", type=float, default=32.0)
    # Keep enough tail samples for FFT St resolution on O(1e4)-step runs.
    ap.add_argument("--discard-frac", type=float, default=0.25)
    ap.add_argument("--time-basis", choices=["auto", "lbm", "conv"], default="auto")
    ap.add_argument("--cd-min", type=float, default=1.20)
    ap.add_argument("--cd-max", type=float, default=1.80)
    ap.add_argument("--st-min", type=float, default=0.14)
    ap.add_argument("--st-max", type=float, default=0.20)
    ap.add_argument("--cl-rms-steady-max", type=float, default=1.0e-3)
    ap.add_argument("--cl-rms-min-periodic", type=float, default=2.0e-2)
    ap.add_argument("--peak-min-sep-frac", type=float, default=0.4)
    ap.add_argument("--peak-amp-frac", type=float, default=0.4)
    ap.add_argument("--st-band-min", type=float, default=0.05)
    ap.add_argument("--st-band-max", type=float, default=0.50)
    args = ap.parse_args()

    rows = read_force(Path(args.force), args.source)
    if len(rows) < 20:
        raise SystemExit("Not enough samples in force.dat")

    n0 = int(len(rows) * args.discard_frac)
    sub = rows[n0:]
    has_conv = all(r[2] is not None for r in sub)
    if args.time_basis == "conv" and not has_conv:
        raise SystemExit("Requested --time-basis=conv but t_conv column is missing.")

    use_conv = (args.time_basis == "conv") or (args.time_basis == "auto" and has_conv)
    if use_conv:
        times = [r[2] for r in sub]
        d_eff = 1.0
        u_eff = 1.0
    else:
        times = [r[1] for r in sub]
        d_eff = args.diameter
        u_eff = args.u0

    cds = [r[3] for r in sub]
    cls = [r[4] for r in sub]

    cd_mean = sum(cds) / len(cds)
    cl_mean = sum(cls) / len(cls)
    cl_rms = math.sqrt(sum((x - cl_mean) ** 2 for x in cls) / len(cls))

    st_peak, nperiod, peak_msg = estimate_st_peak(
        times, cls, d_eff, u_eff, args.peak_min_sep_frac, args.peak_amp_frac
    )
    st_fft, fft_msg = estimate_st_fft(
        times, cls, d_eff, u_eff, args.st_band_min, args.st_band_max
    )

    # Prefer peak-based St when resolved and inside the expected St band;
    # otherwise fallback to the band-limited FFT estimate.
    use_peak = (
        st_peak is not None
        and (args.st_band_min <= st_peak <= args.st_band_max)
    )
    if use_peak:
        st = st_peak
        st_source = "peak"
    elif st_fft is not None:
        st = st_fft
        st_source = "fft_fallback"
    else:
        st = None
        st_source = "none"

    print(
        f"samples_total={len(rows)} samples_used={len(sub)} discard_frac={args.discard_frac} source={args.source} time_basis={'conv' if use_conv else 'lbm'}"
    )
    print(f"Cd_mean={cd_mean:.6f}")
    print(f"Cl_mean={cl_mean:.6f}")
    print(f"Cl_rms={cl_rms:.6f}")
    periodic_lift = cl_rms >= args.cl_rms_min_periodic
    if periodic_lift:
        print(f"periodic_lift=True (Cl_rms >= {args.cl_rms_min_periodic})")
    else:
        print(f"periodic_lift=False (Cl_rms < {args.cl_rms_min_periodic})")
    if len(cds) >= 50:
        nfit = max(50, len(cds) // 5)
        i0 = len(cds) - nfit
        x = list(range(nfit))
        y = cds[i0:]
        xm = sum(x) / nfit
        ym = sum(y) / nfit
        den = sum((xi - xm) ** 2 for xi in x)
        slope = 0.0 if den == 0.0 else sum((x[i] - xm) * (y[i] - ym) for i in range(nfit)) / den
        print(f"Cd_slope_last{nfit}={slope:.6e}")
    if st is None:
        print(
            f"St=NaN (peak_reason='{peak_msg if peak_msg else 'n/a'}', fft_reason='{fft_msg if fft_msg else 'n/a'}')"
        )
    else:
        if st_source == "peak":
            print(f"St={st:.6f} source=peak periods_used={nperiod}")
        else:
            print(f"St={st:.6f} source=fft_fallback")

    cd_ok = args.cd_min <= cd_mean <= args.cd_max
    if st is None:
        st_ok = False
        confidence = "low"
    else:
        st_ok = periodic_lift and (args.st_min <= st <= args.st_max)
        confidence = "high" if (st_source == "peak" and periodic_lift) else ("medium" if periodic_lift else "low")

    print(f"reference_Cd_range=[{args.cd_min},{args.cd_max}] reference_St_range=[{args.st_min},{args.st_max}]")
    print(f"match_Cd={cd_ok} match_St={st_ok} confidence={confidence}")


if __name__ == "__main__":
    main()
