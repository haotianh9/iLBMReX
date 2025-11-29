#include "IBDiffuseLS.H"
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>

using namespace amrex;

static AMREX_GPU_HOST_DEVICE inline Real Hsmooth(Real s, Real eps) {
  if (s <= -eps)
    return Real(0.0);
  if (s >= eps)
    return Real(1.0);
  Real t = Real(0.5) + Real(0.5) * s / eps +
           Real(0.5) / M_PI * std::sin(M_PI * s / eps);
  return t;
}

void IBDiffuseLS::update_forcing(int lev, LevelSetManager &ls,
                                 MultiFab const &ucc, MultiFab const &vcc,
                                 MultiFab const &wcc, MultiFab &Fx,
                                 MultiFab &Fy, MultiFab &Fz, Real alpha) const {
  auto const &phi = ls.phi_at(lev);
  const Real eps = Real(2.0) * m_geom.CellSize(0); // smooth thickness
  amrex::Print() << "IBDiffuseLS::update_forcing at lev=" << lev
                 << " eps=" << eps << "\n";
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(Fx, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();

    auto ph = phi[mfi].const_array();
    auto ux = ucc[mfi].const_array();
    auto uy = vcc[mfi].const_array();
    auto uz = wcc[mfi].const_array();

    auto fxw = Fx[mfi].array();
    auto fyw = Fy[mfi].array();
    auto fzw = Fz[mfi].array();

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real chi = Hsmooth(-ph(i, j, k), eps); // ≈1 in solid, 0 in fluid
      fxw(i, j, k) = alpha * chi * (Real(0.0) - ux(i, j, k));
      fyw(i, j, k) = alpha * chi * (Real(0.0) - uy(i, j, k));
      fzw(i, j, k) = alpha * chi * (Real(0.0) - uz(i, j, k));

      // const amrex::Real eps = 1e-14;

      // if ((amrex::Math::abs(fxw(i, j, k)) > eps) ||
      //     (amrex::Math::abs(fyw(i, j, k)) > eps) ||
      //     (amrex::Math::abs(fzw(i, j, k)) > eps)) {

      //   amrex::Print() << "Forcing at i,j,k: " << i << "," << j << "," << k
      //                  << " chi=" << chi << " fx=" << fxw(i, j, k)
      //                  << " fy=" << fyw(i, j, k) << " fz=" << fzw(i, j, k)
      //                  << "\n";
      // }
    });
  }

  Fx.FillBoundary(m_geom.periodicity());
  Fy.FillBoundary(m_geom.periodicity());
  Fz.FillBoundary(m_geom.periodicity());
}
