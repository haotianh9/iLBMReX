#include <AMReX_FillPatchUtil.H>
#include <AMReX_PhysBCFunct.H>
#include <AmrCoreLBM.H>
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

  FillPatchMesoscopic(lev, time, sfSborder, 0, sfSborder.nComp());
  FillPatchMacro(lev, time, smSborder, 0, smSborder.nComp());

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


    amrex::ParallelFor(gtbx,
                       [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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

                         collide(i, j, k, statein, feq, ndir, temptau);
                       });



    amrex::ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j,
                                                  int k) noexcept {
      amrex::Vector<amrex::Real> tempMac(nmac);
      tempMac[0] = rho_old(i, j, k);
      tempMac[1] = u_old(i, j, k);
      tempMac[2] = v_old(i, j, k);
      tempMac[3] = w_old(i, j, k);

      // Ensure that the calculation area is within the valid range
      if (vbx.contains(i, j, k)) {
        stream(i, j, k, stateout, statein, dirx, diry, dirz, ndir);

        calculateMacro(i, j, k, stateout, rho, u, v, w, ndir, dirx, diry, dirz);

        visPara(i, j, k, rho, u, v, vor, P, tempdx, tempdy, T0);
      }
    });
  }
}
