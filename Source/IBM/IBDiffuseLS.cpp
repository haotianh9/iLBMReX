#include "IBM/IBDiffuseLS.H"

#include <AMReX_Array4.H>
#include <AMReX_MFIter.H>
#include <AMReX_Print.H>

using namespace amrex;

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
min_cell_size(Geometry const &geom) noexcept {
  const auto dx = geom.CellSizeArray();
  Real hmin = dx[0];
#if (AMREX_SPACEDIM >= 2)
  hmin = (dx[1] < hmin) ? dx[1] : hmin;
#endif
#if (AMREX_SPACEDIM == 3)
  hmin = (dx[2] < hmin) ? dx[2] : hmin;
#endif
  return hmin;
}

} // namespace

void IBDiffuseLS::update_forcing(int lev, LevelSetManager &ls,
                                 MultiFab const &ucc, MultiFab const &vcc,
                                 MultiFab const &wcc, MultiFab &Fx,
                                 MultiFab &Fy, MultiFab &Fz, Real alpha,
                                 Real eps_ratio) const {
  MultiFab const &phi_cc = ls.phi_at(lev);

  // Layout checks
  AMREX_ALWAYS_ASSERT(phi_cc.boxArray() == Fx.boxArray());
  AMREX_ALWAYS_ASSERT(phi_cc.DistributionMap() == Fx.DistributionMap());
  AMREX_ALWAYS_ASSERT(ucc.boxArray() == Fx.boxArray());
  AMREX_ALWAYS_ASSERT(vcc.boxArray() == Fx.boxArray());
  AMREX_ALWAYS_ASSERT(wcc.boxArray() == Fx.boxArray());

  const Real eps_phys = eps_ratio * min_cell_size(m_geom);

  Fx.setVal(Real(0.0));
  Fy.setVal(Real(0.0));
  Fz.setVal(Real(0.0));

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(Fx, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();

    auto const &ph = phi_cc.const_array(mfi);
    auto const &u = ucc.const_array(mfi);
    auto const &v = vcc.const_array(mfi);
    auto const &w = wcc.const_array(mfi);

    auto const &fx = Fx.array(mfi);
    auto const &fy = Fy.array(mfi);
    auto const &fz = Fz.array(mfi);

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      // Solid mask: Hsmooth(-phi) is ~1 inside (phi<0), ~0 outside.
      const Real chi = Hsmooth(-ph(i, j, k, 0), eps_phys);
      if (chi <= Real(0.0))
        return;

      // Stationary solid: u_wall = 0.
      fx(i, j, k, 0) = alpha * chi * (Real(0.0) - u(i, j, k, 0));
      fy(i, j, k, 0) = alpha * chi * (Real(0.0) - v(i, j, k, 0));
      fz(i, j, k, 0) = alpha * chi * (Real(0.0) - w(i, j, k, 0));
    });
  }

  Fx.FillBoundary(m_geom.periodicity());
  Fy.FillBoundary(m_geom.periodicity());
  Fz.FillBoundary(m_geom.periodicity());
}
