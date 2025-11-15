#include "DebugNaN.H"
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PhysBCFunct.H>
#include <AmrCoreLBM.H>
#include <IBMomentum.H>
#include <Kernels.H>
using namespace amrex;

void AmrCoreLBM::AdvancePhiAtLevel(int lev, Real time, Real dt_lev,
                                   int /*iteration*/, int /*ncycle*/) {
  // std::swap(f_old[lev], f_new[lev]);
  // std::swap(macro_old[lev], macro_new[lev]);

  std::swap(f_new[lev], f_old[lev]);
  std::swap(macro_new[lev], macro_old[lev]);

  MultiFab &sF_new = f_new[lev];
  MultiFab &sM_new = macro_new[lev];

  MultiFab sfSborder(grids[lev], dmap[lev], sF_new.nComp(), sF_new.nGrow());
  MultiFab smSborder(grids[lev], dmap[lev], sM_new.nComp(), sM_new.nGrow());
  amrex::Print() << "begin filling patch on level " << lev << "\n";
  FillPatchMesoscopic(lev, time, sfSborder, 0, sfSborder.nComp());
  FillPatchMacro(lev, time, smSborder, 0, smSborder.nComp());
  dbg::AssertFinite(sfSborder, "f (sfSborder) before collide");
  dbg::AssertFinite(smSborder, "macro (smSborder) before collide");

  // --- per-level force fields (zeros if no IBM) -------------------------
  MultiFab fx(grids[lev], dmap[lev], 1, nghost);
  MultiFab fy(grids[lev], dmap[lev], 1, nghost);
  MultiFab fz(grids[lev], dmap[lev], 1, nghost);
  fx.setVal(0.0);
  fy.setVal(0.0);
  fz.setVal(0.0);
  fx.FillBoundary(geom[lev].periodicity());
  fy.FillBoundary(geom[lev].periodicity());
  fz.FillBoundary(geom[lev].periodicity());
  // amrex::Print() << "finest level: " << finestLevel() << "\n";
  // amrex::Print() << (m_use_cylinder != 0)  << "\n";
  // amrex::Print() << ( lev == finestLevel())  << "\n";
  // amrex::Print() << (m_ibm.size()>lev)  << "\n";
  // amrex::Print() << (m_ibm.size()) << "\n";
  // amrex::Print() << (m_ibm[lev]) << "\n";

  // If you only want IBM on the finest level, keep lev==finestLevel() here:
  if (m_use_cylinder != 0 && (lev == finestLevel()) && m_ibm.size() > lev &&
      m_ibm[lev]) {
    auto &ib = *m_ibm[lev];

    // Alias ghost-filled macro for velocity interpolation (no copies)
    MultiFab ucc(smSborder, amrex::make_alias, 1, 1);
    MultiFab vcc(smSborder, amrex::make_alias, 2, 1);
    MultiFab wcc(smSborder, amrex::make_alias, 3,
                 1); // ok in 2D; ignored by IBM code
    amrex::Print() << "begin ib interpolate and forcing on level " << lev
                   << "\n";
    ib.interpolate(ucc, vcc, wcc);
    ib.compute_direct_forcing();
    ib.spread(fx, fy, fz); // fz stays zero in 2D implementation
  }
  amrex::Real tempdx = xCellSize[lev];
  amrex::Real tempdy = yCellSize[lev];
  amrex::Real tempdz = zCellSize[lev];
  amrex::Real temptau = tau[lev];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

  for (MFIter mfi(sF_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

    const Box &vbx = mfi.validbox();
    const Box &tbx = mfi.tilebox();
    const Box &gtbx = amrex::grow(tbx, nghost);

    Array4<Real> rho_old = smSborder[mfi].array(0);
    Array4<Real> u_old = smSborder[mfi].array(1);
    Array4<Real> v_old = smSborder[mfi].array(2);
    Array4<Real> w_old = smSborder[mfi].array(3);

    Array4<Real> statein = sfSborder[mfi].array();

    Array4<Real> rho = sM_new[mfi].array(0);
    Array4<Real> u = sM_new[mfi].array(1);
    Array4<Real> v = sM_new[mfi].array(2);
    Array4<Real> w = sM_new[mfi].array(3);
    Array4<Real> vor = sM_new[mfi].array(4);
    Array4<Real> P = sM_new[mfi].array(5);

    Array4<Real> stateout = sF_new[mfi].array();

    // ---  per-tile views of force fields ---------------------------------
    Array4<Real const> fx_arr = fx[mfi].const_array();
    Array4<Real const> fy_arr = fy[mfi].const_array();
    Array4<Real const> fz_arr = fz[mfi].const_array(); // zeros in 2D

    amrex::ParallelFor(
        gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          amrex::Vector<amrex::Real> tempMac(nmac);
          tempMac[0] = rho_old(i, j, k);
          tempMac[1] = u_old(i, j, k);
          tempMac[2] = v_old(i, j, k);
          tempMac[3] = w_old(i, j, k);

          amrex::Vector<amrex::Real> feq(ndir);
          for (int i_dir = 0; i_dir < ndir; ++i_dir) {
            amrex::Vector<amrex::Real> tempMes(4);
            tempMes[0] = wi[i_dir];
            tempMes[1] = dirx[i_dir];
            tempMes[2] = diry[i_dir];
            tempMes[3] = dirz[i_dir];
            feq[i_dir] = feqFunction(tempMes, tempMac);
          }

          //  collide(i, j, k, statein, feq, ndir, temptau);
          // Guo-forced collide in-place on statein (post-collision)
          collide_forced(i, j, k, statein, feq, ndir, temptau, wi, dirx, diry,
                         dirz, rho_old(i, j, k), u_old(i, j, k), v_old(i, j, k),
                         w_old(i, j, k), fx_arr(i, j, k), fy_arr(i, j, k),
                         fz_arr(i, j, k));
        });
    dbg::AssertFinite(sfSborder, "f after collide");

    amrex::ParallelFor(
        gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // Ensure that the calculation area is within the valid range
          if (vbx.contains(i, j, k)) {
            stream(i, j, k, stateout, statein, dirx, diry, dirz, ndir);
          }
        });
    dbg::AssertFinite(sF_new, "f after stream");

    amrex::ParallelFor(
        gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // Ensure that the calculation area is within the valid range
          if (vbx.contains(i, j, k)) {
            // calculateMacro(i, j, k, stateout, rho, u, v, w, ndir, dirx, diry,
            // dirz);
            calculateMacroForcing(i, j, k, stateout, rho, u, v, w, fx_arr,
                                  fy_arr, fz_arr, ndir, dirx, diry, dirz);
            visPara(i, j, k, rho, u, v, vor, P, tempdx, tempdy, T0);
          }
        });
    dbg::AssertFinite(sM_new, "macro after macro update");
  }
}
