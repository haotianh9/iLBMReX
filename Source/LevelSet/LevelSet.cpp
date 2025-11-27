#include "LevelSet.H"
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>

using namespace amrex;

void LevelSetManager::build_from_cylinder(int lev, Geometry const &gm,
                                          LSParams const &p) {
  auto &phi = m_phi[lev];

  const auto problo = gm.ProbLoArray();
  const auto dx = gm.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.fabbox();
    auto ph = phi[mfi].array();

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real x = problo[0] + (Real(i) + Real(0.5)) * dx[0];
      Real y = problo[1] + (Real(j) + Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM == 3)
      Real z = problo[2] + (Real(k) + Real(0.5)) * dx[2];
#else
            Real z = Real(0.0);
#endif
      Real r = std::sqrt((x - p.x0) * (x - p.x0) + (y - p.y0) * (y - p.y0)
#if (AMREX_SPACEDIM == 3)
                         + (z - p.z0) * (z - p.z0)
#endif
      );
      ph(i, j, k) = r - p.R; // signed distance (outside positive)
    });
  }
  phi.FillBoundary(gm.periodicity());
}
