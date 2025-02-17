#include <AmrCoreLBM.H>
#include <Kernels.H>
using namespace amrex;
// Advance pdf at a single level for a single time step, update flux registers
void AmrCoreLBM::AdvanceAtLevel(int lev, Real time, Real dt_lev,
                                int /*iteration*/, int /*ncycle*/) {

  std::swap(
      f_new[lev],
      f_old[lev]); // why using std swap instead of using amrex::MultiFab::Swap?

  double tauinv = tau[lev];       // 1/tau
  double omtauinv = 1.0 - tauinv; // 1 - 1/tau
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
      Array4<Real> f_new_arr = f_new_fab.array(mfi);
      Array4<Real> f_old_arr = f_old_fab.array(mfi);
      Array4<Real> ux_arr = ux_fab.array(mfi);
      Array4<Real> uy_arr = uy_fab.array(mfi);
      Array4<Real> rho_arr = rho_fab.array(mfi);

      amrex::launch(amrex::grow(gbx, 1), [=] AMREX_GPU_DEVICE(const Box &tbx) {
        // amrex::Print() << "tbx: " << tbx << std::endl;
        // amrex::Print() << "stream" << std::endl;
        stream(tbx, f_new_arr, f_old_arr, ndir, nghost, dirx, diry,
#if (AMREX_SPACEDIM == 3)
               dirz,
#endif
               wi);
        // amrex::Print() << "calculate velocity and density" << std::endl;
        calculate_macro_velo_rho(tbx, f_new_arr, rho_arr, ux_arr, uy_arr, ndir,
                                 nghost, dirx, diry,
#if (AMREX_SPACEDIM == 3)
                                 dirz,
#endif
                                 wi);
        // amrex::Print() << "collide" << std::endl;
        collide(tbx, f_new_arr, rho_arr, ux_arr, uy_arr, tauinv, omtauinv, ndir,
                nghost, dirx, diry,
#if (AMREX_SPACEDIM == 3)
                dirz,
#endif
                wi);
      });
    }
  }
}
