#include "IBM/IBForceEval.H"

#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cmath>

namespace IBForceEval {
using namespace amrex;

namespace {

AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE Real
signed_distance(Real x, Real y, Real z, LevelSetParams const &par) noexcept {
#if (AMREX_SPACEDIM == 3)
  const Real r = std::sqrt((x - par.x0) * (x - par.x0) + (y - par.y0) * (y - par.y0) +
                           (z - par.z0) * (z - par.z0));
#else
  amrex::ignore_unused(z);
  const Real r = std::sqrt((x - par.x0) * (x - par.x0) + (y - par.y0) * (y - par.y0));
#endif
  return r - par.R;
}

} // namespace

GpuArray<Real, 3>
ComputeMomentumExchangeBodyForce(MultiFab const &f_cc, Geometry const &geom,
                                 Vector<Real> const &dirx,
                                 Vector<Real> const &diry,
                                 Vector<Real> const &dirz, int ndir,
                                 LevelSetParams const &par,
                                 bool one_sided_bounceback) {
  GpuArray<Real, 3> Fbody{{Real(0.0), Real(0.0), Real(0.0)}};
  if (ndir <= 0 || f_cc.nComp() < ndir) {
    return Fbody;
  }

  // Opposite-direction map from discrete velocities.
  std::vector<int> opp(ndir, -1);
  for (int q = 0; q < ndir; ++q) {
    for (int p = 0; p < ndir; ++p) {
      if (std::abs(dirx[p] + dirx[q]) < 1.e-12 &&
          std::abs(diry[p] + diry[q]) < 1.e-12 &&
          std::abs(dirz[p] + dirz[q]) < 1.e-12) {
        opp[q] = p;
        break;
      }
    }
  }

  const auto dom = geom.Domain();
  const int ilo = dom.smallEnd(0), ihi = dom.bigEnd(0);
  const int jlo = dom.smallEnd(1), jhi = dom.bigEnd(1);
#if (AMREX_SPACEDIM == 3)
  const int klo = dom.smallEnd(2), khi = dom.bigEnd(2);
#else
  const int klo = 0, khi = 0;
#endif

  const auto plo = geom.ProbLoArray();
  const auto dx = geom.CellSizeArray();
  const auto periodic = geom.isPeriodicArray();

  Real fx_fluid = Real(0.0);
  Real fy_fluid = Real(0.0);
  Real fz_fluid = Real(0.0);

  for (MFIter mfi(f_cc, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto const f = f_cc[mfi].const_array();

    const int kbeg =
#if (AMREX_SPACEDIM == 3)
        bx.smallEnd(2);
#else
        0;
#endif
    const int kend =
#if (AMREX_SPACEDIM == 3)
        bx.bigEnd(2);
#else
        0;
#endif
    for (int k = kbeg; k <= kend; ++k) {
      for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
          const Real xc = plo[0] + (Real(i) + Real(0.5)) * dx[0];
          const Real yc = plo[1] + (Real(j) + Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM == 3)
          const Real zc = plo[2] + (Real(k) + Real(0.5)) * dx[2];
#else
          const Real zc = Real(0.0);
#endif
          const bool fluid_here = (signed_distance(xc, yc, zc, par) > Real(0.0));
          if (!fluid_here) {
            continue;
          }

          for (int q = 0; q < ndir; ++q) {
            // Skip rest direction and unmatched opposite pairs.
            if (std::abs(dirx[q]) < 1.e-12 && std::abs(diry[q]) < 1.e-12 &&
                std::abs(dirz[q]) < 1.e-12) {
              continue;
            }
            if (opp[q] < 0) {
              continue;
            }

            int di = static_cast<int>(std::llround(dirx[q]));
            int dj = static_cast<int>(std::llround(diry[q]));
            int dk = static_cast<int>(std::llround(dirz[q]));

            int in = i + di;
            int jn = j + dj;
            int kn = k + dk;

            if (periodic[0]) {
              const int nx = ihi - ilo + 1;
              while (in < ilo) in += nx;
              while (in > ihi) in -= nx;
            } else if (in < ilo || in > ihi) {
              continue;
            }
            if (periodic[1]) {
              const int ny = jhi - jlo + 1;
              while (jn < jlo) jn += ny;
              while (jn > jhi) jn -= ny;
            } else if (jn < jlo || jn > jhi) {
              continue;
            }
#if (AMREX_SPACEDIM == 3)
            if (periodic[2]) {
              const int nz = khi - klo + 1;
              while (kn < klo) kn += nz;
              while (kn > khi) kn -= nz;
            } else if (kn < klo || kn > khi) {
              continue;
            }
#else
            amrex::ignore_unused(klo, khi);
            kn = 0;
#endif

            const Real xn = plo[0] + (Real(in) + Real(0.5)) * dx[0];
            const Real yn = plo[1] + (Real(jn) + Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM == 3)
            const Real zn = plo[2] + (Real(kn) + Real(0.5)) * dx[2];
#else
            const Real zn = Real(0.0);
#endif
            const bool solid_next = (signed_distance(xn, yn, zn, par) <= Real(0.0));
            if (!solid_next) {
              continue;
            }

            const int qo = opp[q];
            const Real f_out = f(i, j, k, q);
            Real dm = Real(0.0);
            if (one_sided_bounceback) {
              // One-sided stationary-wall momentum exchange approximation:
              //   dF = 2 * f_q(x_f) * c_q
              // Useful when interior-solid populations are not physically
              // meaningful (e.g. marker direct-forcing IBM).
              dm = Real(2.0) * f_out;
            } else {
              // Two-sided link-wise form (Ladd / Mei / Wen):
              //   dF = (f_q(x_f) + f_opp(x_s)) * c_q
              const Real f_in = f(in, jn, kn, qo);
              dm = f_out + f_in;
            }
            fx_fluid += dm * dirx[q];
            fy_fluid += dm * diry[q];
            fz_fluid += dm * dirz[q];
          }
        }
      }
    }
  }

  ParallelDescriptor::ReduceRealSum(fx_fluid);
  ParallelDescriptor::ReduceRealSum(fy_fluid);
  ParallelDescriptor::ReduceRealSum(fz_fluid);

  // Return reaction force on the body.
  Fbody[0] = -fx_fluid;
  Fbody[1] = -fy_fluid;
  Fbody[2] = -fz_fluid;
  return Fbody;
}

} // namespace IBForceEval
