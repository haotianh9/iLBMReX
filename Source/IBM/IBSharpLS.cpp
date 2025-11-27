#include "IBSharpLS.H"
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>

using namespace amrex;

void IBSharpLS::update_forcing(int lev, LevelSetManager &ls,
                               MultiFab const &ucc, MultiFab const &vcc,
                               MultiFab const &wcc, MultiFab &Fx, MultiFab &Fy,
                               MultiFab &Fz, Real dt) const {
  auto const &phi = ls.phi_at(lev);
  const Real eps = Real(2.0) * m_geom.CellSize(0); // thin |phi|<=eps band

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(Fx, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.fabbox();

    auto ph = phi[mfi].const_array();
    auto ux = ucc[mfi].const_array();
    auto uy = vcc[mfi].const_array();
    auto uz = wcc[mfi].const_array();

    auto fxw = Fx[mfi].array();
    auto fyw = Fy[mfi].array();
    auto fzw = Fz[mfi].array();

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real band = (std::abs(ph(i, j, k)) <= eps) ? Real(1.0) : Real(0.0);
      // direct forcing: f ≈ (u_target - u)/dt with u_target = 0
      fxw(i, j, k) += band * (-ux(i, j, k) / dt);
      fyw(i, j, k) += band * (-uy(i, j, k) / dt);
      fzw(i, j, k) += band * (-uz(i, j, k) / dt);
    });
  }

  Fx.FillBoundary(m_geom.periodicity());
  Fy.FillBoundary(m_geom.periodicity());
  Fz.FillBoundary(m_geom.periodicity());
}
