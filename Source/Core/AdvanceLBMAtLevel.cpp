#include "AmrCoreLBM.H"
#include "Kernels.H"
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PhysBCFunct.H>

// Level-set & IBM backends
#include "IBM/IBDiffuseLS.H"
#include "IBM/IBSharpLS.H"
#include "LevelSet/LevelSet.H"

using namespace amrex;

void AmrCoreLBM::AdvancePhiAtLevel(int lev, Real time, Real dt_lev,
                                   int /*iteration*/, int /*ncycle*/) {
  // swap time slots
  std::swap(f_new[lev], f_old[lev]);
  std::swap(macro_new[lev], macro_old[lev]);

  MultiFab &sF_new = f_new[lev];
  MultiFab &sM_new = macro_new[lev];

  MultiFab sfSborder(grids[lev], dmap[lev], sF_new.nComp(), sF_new.nGrow());
  MultiFab smSborder(grids[lev], dmap[lev], sM_new.nComp(), sM_new.nGrow());

  FillPatchMesoscopic(lev, time, sfSborder, 0, sfSborder.nComp());
  FillPatchMacro(lev, time, smSborder, 0, smSborder.nComp());

  const Real tempdx = xCellSize[lev];
  const Real tempdy = yCellSize[lev];
#if (AMREX_SPACEDIM == 3)
  const Real tempdz = zCellSize[lev];
#else
  const Real tempdz = Real(0.0);
#endif
  const Real temptau = tau[lev];

  // Body force (IBM) arrays: cell-centered, 1 comp
  MultiFab fx_cc(grids[lev], dmap[lev], 1, 0);
  MultiFab fy_cc(grids[lev], dmap[lev], 1, 0);
  MultiFab fz_cc(grids[lev], dmap[lev], 1, 0);
  fx_cc.setVal(0.0);
  fy_cc.setVal(0.0);
  fz_cc.setVal(0.0);

  // ---- Level-set driven IBM (only on finest) ----
  if (m_use_cylinder && lev == finestLevel() && m_ls) {
    // alias velocity components from macro staging
    MultiFab ucc(smSborder, amrex::make_alias, 1, 1); // u
    MultiFab vcc(smSborder, amrex::make_alias, 2, 1); // v
#if (AMREX_SPACEDIM == 3)
    MultiFab wcc(smSborder, amrex::make_alias, 3, 1); // w
#else
    MultiFab wcc(grids[lev], dmap[lev], 1, 0); // dummy w in 2D
    wcc.setVal(0.0);
#endif
    if (m_ib_method == 1 && m_ibd) {
      // Diffuse: Brinkman penalization in smoothed solid mask
      m_ibd->update_forcing(lev, *m_ls, ucc, vcc, wcc, fx_cc, fy_cc, fz_cc,
                            m_diff_par.alpha);
    } else if (m_ib_method == 2 && m_ibs) {
      // Sharp: direct forcing in |phi|<=eps band
      m_ibs->update_forcing(lev, *m_ls, ucc, vcc, wcc, fx_cc, fy_cc, fz_cc,
                            dt_lev);
    }
  }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(sF_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &vbx = mfi.validbox();
    const Box &tbx = mfi.tilebox();
    const Box &gtbx = amrex::grow(tbx, nghost);

    // staging views
    Array4<Real> rho_old = smSborder[mfi].array(0);
    Array4<Real> u_old = smSborder[mfi].array(1);
    Array4<Real> v_old = smSborder[mfi].array(2);
    Array4<Real> w_old = smSborder[mfi].array(3);

    Array4<Real> statein = sfSborder[mfi].array(); // f*
    Array4<Real> stateout = sF_new[mfi].array();   // post-stream

    Array4<Real> rho = sM_new[mfi].array(0);
    Array4<Real> u = sM_new[mfi].array(1);
    Array4<Real> v = sM_new[mfi].array(2);
    Array4<Real> w = sM_new[mfi].array(3);
    Array4<Real> vor = sM_new[mfi].array(4);
    Array4<Real> P = sM_new[mfi].array(5);

    // local force views
    auto fx = fx_cc[mfi].array(0);
    auto fy = fy_cc[mfi].array(0);
    auto fz = fz_cc[mfi].array(0);
    const int nd = ndir;  // capture number of directions
    auto dirx_loc = dirx; // OK for CPU path (AMREX_GPU_MAX_THREADS=0)
    auto diry_loc = diry;
    auto dirz_loc = dirz;
    auto wi_loc = wi;
    // -------- Collide (BGK) --------
    amrex::ParallelFor(
        gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // local macros from last stage
          amrex::Vector<amrex::Real> mac(4);
          mac[0] = rho_old(i, j, k);
          mac[1] = u_old(i, j, k);
          mac[2] = v_old(i, j, k);
          mac[3] = w_old(i, j, k);
          // build feq

          amrex::Vector<amrex::Real> tempMes(4);
          amrex::Vector<amrex::Real> feq_loc(ndir);

          for (int q = 0; q < ndir; ++q) {
            tempMes[0] = wi[q];
            tempMes[1] = dirx[q];
            tempMes[2] = diry[q];
            tempMes[3] = dirz[q];
            feq_loc[q] = feqFunction(tempMes, mac);
          }
           collide(i, j, k, statein, feq_loc, ndir, temptau);
          // collide_forced(i, j, k, statein, feq_loc, ndir, temptau, wi_loc,
          //                dirx_loc, diry_loc, dirz_loc, mac[0], mac[1], mac[2],
          //                mac[3], fx(i,j,k), fy(i,j,k), fz(i,j,k));
        });

    // // -------- Stream --------
    amrex::ParallelFor(
        gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          if (vbx.contains(i, j, k)) {
            stream(i, j, k, stateout, statein, dirx, diry, dirz, ndir);
          }
        });

    // -------- Macros w/ forcing + diagnostics --------
    amrex::ParallelFor(
        gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // Ensure that the calculation area is within the valid range
          if (vbx.contains(i, j, k)) {
            calculateMacroForcing(i, j, k, stateout, rho, u, v, w, fx, fy, fz,
                                  ndir, dirx, diry, dirz);
            // calculateMacro(i, j, k, stateout, rho, u, v, w, ndir, dirx, diry,
            // dirz);
            // visPara(i, j, k, rho, u, v, vor, P, tempdx, tempdy, T0);
          }
        });
  }
}
