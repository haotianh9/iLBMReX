#include "LevelSet/LevelSet.H"
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Print.H>
#include <cmath>

using namespace amrex;

void LevelSetManager::define_level(int lev, const amrex::BoxArray &ba,
                                   const amrex::DistributionMapping &dm,
                                   int ngrow) {
  if (lev >= static_cast<int>(m_phi.size())) {
    m_phi.resize(lev + 1);
  }

  if (!m_phi[lev] || m_phi[lev]->boxArray() != ba ||
      m_phi[lev]->DistributionMap() != dm || m_phi[lev]->nGrow() != ngrow) {
    m_phi[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 1, ngrow);
    m_phi[lev]->setVal(0.0);
  }
}

bool LevelSetManager::has_level(int lev) const {
  return lev < m_phi.size() && static_cast<bool>(m_phi[lev]);
}

MultiFab &LevelSetManager::phi_at(int lev) {
  AMREX_ALWAYS_ASSERT(has_level(lev));
  return *m_phi[lev];
}

MultiFab const &LevelSetManager::phi_at(int lev) const {
  AMREX_ALWAYS_ASSERT(has_level(lev));
  return *m_phi[lev];
}

void LevelSetManager::build_from_cylinder(int lev, Geometry const &geom,
                                          LevelSetParams const &par) {
  AMREX_ALWAYS_ASSERT(has_level(lev));
  MultiFab &phi = *m_phi[lev];

  const auto problo = geom.ProbLoArray();
  const auto dx = geom.CellSizeArray();

  const Real x0 = par.x0;
  const Real y0 = par.y0;
  const Real z0 = par.z0;
  const Real R = par.R;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(phi, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto phi_arr = phi[mfi].array();

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real x = problo[0] + (Real(i) + Real(0.5)) * dx[0];
      Real y = problo[1] + (Real(j) + Real(0.5)) * dx[1];

#if (AMREX_SPACEDIM == 3)
      Real z = problo[2] + (Real(k) + Real(0.5)) * dx[2];
      Real r = std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) +
                         (z - z0) * (z - z0));
#else
            Real r = std::sqrt( (x - x0)*(x - x0)
                              + (y - y0)*(y - y0) );
#endif
      // Signed distance: positive outside, negative inside the cylinder
      phi_arr(i, j, k, 0) = r - R;
      // amrex::Print() << "LevelSetManager: phi_arr(" << i << "," << j << ","
      // << k << ") = "
      //              << phi_arr(i, j, k, 0) << "\n";
    });
  }

  Print() << "LevelSetManager: built cylinder level-set on level " << lev
          << " with R = " << R << "\n";
}
