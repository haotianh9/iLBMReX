#include <AmrCoreLBM.H>
#include <Kernels.H>
using namespace amrex;
// Advance pdf at a single level for a single time step, update flux registers
void AmrCoreLBM::CollideAndStreamAtLevel(int lev, Real time, Real dt_lev,
                                         int /*iteration*/, int /*ncycle*/) {
  constexpr int Nghost = 1;
  std::swap(
      f_new[lev],
      f_old[lev]); // why using std swap instead of using amrex::MultiFab::Swap?

  double tauinv = 2.0 / (6.0 * nu + 1.0); // 1/tau
  double omtauinv = 1.0 - tauinv;         // 1 - 1/tau
  const Real dx = geom[lev].CellSize(0);
  const Real dy = geom[lev].CellSize(1);
  const Real dz = (AMREX_SPACEDIM == 2) ? Real(1.0) : geom[lev].CellSize(2);
  AMREX_D_TERM(Real dtdx = dt_lev / dx;, Real dtdy = dt_lev / dy;
               , Real dtdz = dt_lev / dz);

  MultiFab &f_new_fab = f_new[lev];
  MultiFab &f_old_fab = f_old[lev];
  MultiFab &ux_fab = ux[lev];
  MultiFab &uy_fab = uy[lev];
  MultiFab &rho_fab = rho[lev];

  MultiFab fluxes[AMREX_SPACEDIM];

  if (do_reflux) {
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      BoxArray ba = grids[lev];
      ba.surroundingNodes(i);
      fluxes[i].define(ba, dmap[lev], ndir, 0);
    }
  }
  // State with ghost cells
  MultiFab f_new_border(grids[lev], dmap[lev], f_new_fab.nComp(), nghost);
  FillPatch(lev, time, f_new_border, 0, f_new_border.nComp(),
            FillPatchType::fillpatch_class);
  MultiFab f_old_border(grids[lev], dmap[lev], f_old_fab.nComp(), nghost);
  FillPatch(lev, time, f_old_border, 0, f_old_border.nComp(),
            FillPatchType::fillpatch_class);
  MultiFab ux_border(grids[lev], dmap[lev], ux_fab.nComp(), nghost);
  FillPatch(lev, time, ux_border, 0, ux_border.nComp(),
            FillPatchType::fillpatch_class);
  MultiFab uy_border(grids[lev], dmap[lev], uy_fab.nComp(), nghost);
  FillPatch(lev, time, uy_border, 0, uy_border.nComp(),
            FillPatchType::fillpatch_class);
  MultiFab rho_border(grids[lev], dmap[lev], rho_fab.nComp(), nghost);
  FillPatch(lev, time, rho_border, 0, rho_border.nComp(),
            FillPatchType::fillpatch_class);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  {
    FArrayBox tmpfab;
    for (MFIter mfi(f_new_fab, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();
      const Box &gbx = amrex::grow(bx, 1);
      Array4<Real const> f_old_arr = f_old_border.const_array(mfi);
      Array4<Real const> ux_arr = ux_border.const_array(mfi);
      Array4<Real const> uy_arr = uy_border.const_array(mfi);
      Array4<Real const> rho_arr = rho_border.const_array(mfi);
      Array4<Real> f_new_arr = f_new_fab.array(mfi);
      amrex::launch(amrex::grow(gbx, 1), [=] AMREX_GPU_DEVICE(const Box &tbx) {
        collide_stream(tbx, f_new_arr, f_old_arr, rho_arr, ux_arr, uy_arr,
                       tauinv, omtauinv, ndir, dirx, diry,
#if (AMREX_SPACEDIM == 3)
                       dirz,
#endif
                       wi);
      });
    }
  }
}
