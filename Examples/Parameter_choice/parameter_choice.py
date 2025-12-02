"""
lbm_ib_design_2d3d.py

Design LBM + diffusive IBM parameters from a target Reynolds number,
for 2D (D2Q9) and 3D (D3Q19 / D3Q27) lattices.

All variables are in lattice units with dx = dt = 1.
"""

from dataclasses import dataclass
from typing import Dict, List
import math

# ----------------------------------------------------------------------
# Lattice definitions
# ----------------------------------------------------------------------

STENCILS = {
    "D2Q9":  {"dim": 2, "cs_sq": 1.0 / 3.0},
    "D3Q19": {"dim": 3, "cs_sq": 1.0 / 3.0},
    "D3Q27": {"dim": 3, "cs_sq": 1.0 / 3.0},
}


@dataclass
class Candidate:
    scheme: str
    L: int
    U: float
    nu: float
    tau_nu: float
    cost: float
    extras: Dict[str, float]
    cs: float  # lattice speed of sound

    @property
    def Ma(self) -> float:
        return self.U / self.cs


# ----------------------------------------------------------------------
# Cost model
# ----------------------------------------------------------------------

def compute_cost(L: int, U: float, dim: int = 2) -> float:
    """
    Very crude cost model in lattice units:
        cost ~ (number_of_cells) * (number_of_timesteps_to_cross_L)
             ~ L^dim * (L / U)
             = L^(dim + 1) / U

    Here we assume dx = dt = 1, and domain size ~ L in each direction.
    """
    if U <= 0:
        return float("inf")
    return (L ** (dim + 1)) / U


# ----------------------------------------------------------------------
# IBM (diffusive immersed boundary) helper
# ----------------------------------------------------------------------

def ibm_alpha_recommendation(dim: int, dt: float = 1.0, dx: float = 1.0) -> Dict[str, float]:
    """
    For an explicit diffusion step in d dimensions with central differences,
    a typical CFL constraint is:
        dt * alpha / dx^2 <= 1 / (2 d)

    With dx = dt = 1, this gives alpha <= 1 / (2 d).

    For immersed boundary diffusion (or regularization), this is a rough
    upper bound; in practice, staying well below it is safer.

    Returns a dict with:
        - alpha_theoretical_max
        - alpha_safe_max      (20% of theoretical)
        - alpha_preferred     (50% of safe_max)
        - alpha_min           (10% of preferred)
    """
    if dim <= 0:
        raise ValueError("dim must be >= 1")

    alpha_theoretical_max = (1.0 / (2.0 * dim)) * (dx ** 2) / dt
    alpha_safe_max = 0.2 * alpha_theoretical_max
    alpha_preferred = 0.5 * alpha_safe_max
    alpha_min = 0.1 * alpha_preferred

    return dict(
        alpha_theoretical_max=alpha_theoretical_max,
        alpha_safe_max=alpha_safe_max,
        alpha_preferred=alpha_preferred,
        alpha_min=alpha_min,
    )


# ----------------------------------------------------------------------
# Main search routine
# ----------------------------------------------------------------------

def find_lbm_ib_parameters(
    Re: float,
    stencil: str = "D2Q9",
    Ma_max: float = 0.1,
    L_min: int = 10,
    L_max: int = 200,
    tau_min: float = 0.51,
    tau_max: float = 1.9,
    tau_step: float = 0.01,
) -> Dict[str, List[Candidate]]:
    """
    Scan through (L, tau_nu) and assemble feasible LBM parameter sets
    for SRT, TRT, and MRT that satisfy:

        Re = U * L / nu
        nu = c_s^2 * (tau_nu - 0.5)
        U  <= Ma_max * c_s
        tau_min <= tau_nu <= tau_max

    for a given lattice stencil (D2Q9, D3Q19, D3Q27).

    Returns a dict mapping scheme name -> list[Candidate].
    """

    if stencil not in STENCILS:
        raise ValueError(f"Unknown stencil '{stencil}'. Choose from {list(STENCILS.keys())}.")

    info = STENCILS[stencil]
    dim = info["dim"]
    cs_sq = info["cs_sq"]
    cs = math.sqrt(cs_sq)

    schemes: Dict[str, List[Candidate]] = {"SRT": [], "TRT": [], "MRT": []}
    U_max = Ma_max * cs

    tau = tau_min
    while tau <= tau_max + 1e-12:
        nu = cs_sq * (tau - 0.5)
        if nu <= 0:
            tau += tau_step
            continue

        for L in range(L_min, L_max + 1):
            # Deduce U from Reynolds number
            U = Re * nu / L
            if U <= 0 or U > U_max:
                continue

            cost = compute_cost(L, U, dim=dim)

            # ------------------------------
            # SRT candidate
            # ------------------------------
            omega = 1.0 / tau
            srt_extra = {"omega": omega, "stencil": stencil}
            schemes["SRT"].append(
                Candidate(
                    scheme="SRT",
                    L=L,
                    U=U,
                    nu=nu,
                    tau_nu=tau,
                    cost=cost,
                    extras=srt_extra,
                    cs=cs,
                )
            )

            # ------------------------------
            # TRT candidate
            # ------------------------------
            # Use a "magic" parameter Lambda to choose bulk/even relaxation.
            # A common choice is Lambda ~ 3/16 or 1/4; we take 1/4.
            Lambda = 0.25
            a = 1.0 / tau - 0.5
            if abs(a) > 1e-12:
                inv_tau_bulk_minus_half = Lambda / a
                inv_tau_bulk = inv_tau_bulk_minus_half + 0.5
                if inv_tau_bulk > 0:
                    tau_bulk = 1.0 / inv_tau_bulk
                    # Require tau_bulk to also be in a reasonable range
                    if tau_min <= tau_bulk <= tau_max:
                        trt_extra = {
                            "tau_shear": tau,
                            "tau_bulk": tau_bulk,
                            "Lambda": Lambda,
                            "stencil": stencil,
                        }
                        schemes["TRT"].append(
                            Candidate(
                                scheme="TRT",
                                L=L,
                                U=U,
                                nu=nu,
                                tau_nu=tau,
                                cost=cost,
                                extras=trt_extra,
                                cs=cs,
                            )
                        )

            # ------------------------------
            # MRT candidate
            # ------------------------------
            # For simplicity, we pick standard-ish relaxation rates for
            # non-hydrodynamic moments and only tune the shear rate s_nu
            # via tau_nu, i.e. s_nu = 1 / tau_nu.
            s_nu = 1.0 / tau
            # These are generic, dimension-agnostic placeholders; you can
            # tune them later for your specific moment-basis.
            s_bulk = 1.2
            s_e = 1.64
            s_eps = 1.54
            s_q = 1.9

            mrt_extra = {
                "s_nu": s_nu,
                "s_bulk": s_bulk,
                "s_e": s_e,
                "s_eps": s_eps,
                "s_q": s_q,
                "stencil": stencil,
            }
            schemes["MRT"].append(
                Candidate(
                    scheme="MRT",
                    L=L,
                    U=U,
                    nu=nu,
                    tau_nu=tau,
                    cost=cost,
                    extras=mrt_extra,
                    cs=cs,
                )
            )

        tau += tau_step

    return schemes


# ----------------------------------------------------------------------
# Utility: summarize candidates & pick cheapest
# ----------------------------------------------------------------------

def summarize_candidates(cands: List[Candidate], n_best: int = 5) -> None:
    if not cands:
        print("  No feasible combinations found.")
        return

    L_vals = [c.L for c in cands]
    U_vals = [c.U for c in cands]
    nu_vals = [c.nu for c in cands]
    Ma_vals = [c.Ma for c in cands]

    print(f"  L range : [{min(L_vals)}, {max(L_vals)}]")
    print(f"  U range : [{min(U_vals):.4e}, {max(U_vals):.4e}]")
    print(f"  nu range: [{min(nu_vals):.4e}, {max(nu_vals):.4e}]")
    print(f"  Ma range: [{min(Ma_vals):.4e}, {max(Ma_vals):.4e}]")

    # sort by increasing cost
    cands_sorted = sorted(cands, key=lambda c: c.cost)
    print(f"\n  {n_best} lowest-cost combinations:")
    print("    (L, U, nu, tau_nu, Ma, cost)")
    for c in cands_sorted[:n_best]:
        print(
            f"    L={c.L:4d}, U={c.U:8.4e}, nu={c.nu:8.4e}, "
            f"tau={c.tau_nu:5.3f}, Ma={c.Ma:6.3f}, cost={c.cost:10.3e}"
        )
        if c.extras:
            print(f"      extras: {c.extras}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Design LBM + diffusive IBM parameters from a target Reynolds number."
    )
    parser.add_argument("Re", type=float, help="Target Reynolds number (Re = U*L/nu).")
    parser.add_argument(
        "--stencil",
        type=str,
        choices=list(STENCILS.keys()),
        default="D2Q9",
        help="LBM stencil: D2Q9 (2D), D3Q19 or D3Q27 (3D).",
    )
    parser.add_argument(
        "--Ma_max",
        type=float,
        default=0.1,
        help="Maximum Mach number (default: 0.1).",
    )
    parser.add_argument(
        "--L_min",
        type=int,
        default=10,
        help="Minimum characteristic length in lattice nodes (default: 10).",
    )
    parser.add_argument(
        "--L_max",
        type=int,
        default=200,
        help="Maximum characteristic length in lattice nodes (default: 200).",
    )
    parser.add_argument(
        "--tau_min",
        type=float,
        default=0.51,
        help="Minimum shear relaxation time tau_nu (default: 0.51).",
    )
    parser.add_argument(
        "--tau_max",
        type=float,
        default=1.9,
        help="Maximum shear relaxation time tau_nu (default: 1.9).",
    )
    parser.add_argument(
        "--tau_step",
        type=float,
        default=0.01,
        help="Step size for scanning tau_nu (default: 0.01).",
    )
    args = parser.parse_args()

    info = STENCILS[args.stencil]
    dim = info["dim"]
    cs_sq = info["cs_sq"]
    cs = math.sqrt(cs_sq)

    print(f"=== LBM + diffusive IBM parameter search ===")
    print(f"  Re       = {args.Re}")
    print(f"  stencil  = {args.stencil}  (dim = {dim}, c_s^2 = {cs_sq:.5f}, c_s = {cs:.5f})")
    print(f"  Ma_max   = {args.Ma_max}")
    print(f"  L range  = [{args.L_min}, {args.L_max}]")
    print(f"  tau_nu   in [{args.tau_min}, {args.tau_max}] with step {args.tau_step}")
    print(f"  dx = dt  = 1")
    print()

    schemes = find_lbm_ib_parameters(
        Re=args.Re,
        stencil=args.stencil,
        Ma_max=args.Ma_max,
        L_min=args.L_min,
        L_max=args.L_max,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_step=args.tau_step,
    )

    alpha_info = ibm_alpha_recommendation(dim=dim)
    print(f"IBM (diffusive coupling) alpha recommendations (dx = dt = 1, dim = {dim}):")
    for k, v in alpha_info.items():
        print(f"  {k:20s}: {v:.3e}")
    print()

    for scheme_name, cands in schemes.items():
        print(f"--- Scheme: {scheme_name} ---")
        summarize_candidates(cands, n_best=5)
        print()

    # Also, pick the single globally cheapest combo among all schemes
    all_cands = [c for clist in schemes.values() for c in clist]
    if all_cands:
        best_overall = min(all_cands, key=lambda c: c.cost)
        print("=== Lowest-cost combination over all schemes ===")
        print(
            f"  Scheme={best_overall.scheme}, L={best_overall.L}, "
            f"U={best_overall.U:.4e}, nu={best_overall.nu:.4e}, "
            f"tau_nu={best_overall.tau_nu:.3f}, Ma={best_overall.Ma:.3f}, "
            f"cost={best_overall.cost:.3e}"
        )
        if best_overall.extras:
            print(f"    extras: {best_overall.extras}")
    else:
        print("No feasible combinations at all; try relaxing constraints.")


if __name__ == "__main__":
    main()
