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
  MultiFab *sForcingPtr = nullptr;
  if (m_use_cylinder && forcing.size() > lev) {
    sForcingPtr = &forcing[lev];
  }

  // local IBM scratch:
  MultiFab fx_cc(grids[lev], dmap[lev], 1, nghost);
  MultiFab fy_cc(grids[lev], dmap[lev], 1, nghost);
  MultiFab fz_cc(grids[lev], dmap[lev], 1, nghost);
  fx_cc.setVal(0.0);
  fy_cc.setVal(0.0);
  fz_cc.setVal(0.0);
  MultiFab sfSborder(grids[lev], dmap[lev], sF_new.nComp(), sF_new.nGrow());
  MultiFab smSborder(grids[lev], dmap[lev], sM_new.nComp(), sM_new.nGrow());
  FillPatchMesoscopic(lev, time, sfSborder, 0, sfSborder.nComp());
  FillPatchMacro(lev, time, smSborder, 0, smSborder.nComp());

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
    // amrex::Print() << "AdvancePhiAtLevel: lev=" << lev << " dt_lev=" <<
    // dt_lev
    //                << " updating IBM forcing\n";
    if (m_ib_method == 1 && m_ibd) {
      // Diffuse: Brinkman penalization in smoothed solid mask
      m_ibd->update_forcing(lev, *m_ls, ucc, vcc, wcc, fx_cc, fy_cc, fz_cc,
                            m_diff_par.alpha,m_diff_par.eps);
    } else if (m_ib_method == 2 && m_ibs) {
      // Sharp: direct forcing in |phi|<=eps band
      m_ibs->update_forcing(lev, *m_ls, ucc, vcc, wcc, fx_cc, fy_cc, fz_cc,
                            dt_lev);
    }
  }
  // amrex::Print() << "AdvancePhiAtLevel lev=" << lev
  //                << "  sF_new.nComp=" << sF_new.nComp()
  //                << "  sM_new.nComp=" << sM_new.nComp();

  // if (forcing.size() > lev) {
  //   amrex::Print() << "AdvancePhiAtLevel: lev=" << lev << " dt_lev=" <<
  //   dt_lev
  //                  << " forcing nComp=" << forcing[lev].nComp()
  //                  << " nGrow=" << forcing[lev].nGrow() << "\n";
  // }
  // amrex::Print() << "\n";

  // AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
  //     sM_new.nComp() > 0 && sF_new.nComp() > 0,
  //     "macro_new[lev] or f_new[lev] has zero components!");

  std::unique_ptr<MultiFab> sForcingborder;
  if (sForcingPtr) {
    sForcingborder = std::make_unique<MultiFab>(
        grids[lev], dmap[lev], sForcingPtr->nComp(), sForcingPtr->nGrow());
  }
  // if (sForcing) {
  //     sForcingborder = std::make_unique<MultiFab>(
  //         grids[lev], dmap[lev], sForcing.nComp(),
  //                           sForcing.nGrow());
  // }

  // ... fill fx_cc, fy_cc, fz_cc from IBM (m_ibd/m_ibs) ...

  // pack into forcing[lev] on valid region
  if (sForcingPtr) {
    MultiFab &sForcing = *sForcingPtr;
    for (MFIter mfi(sM_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      // amrex::Print() << "MFIter lev=" << lev
      //                << " smSborder.nComp=" << smSborder.nComp()
      //                << " sfSborder.nComp=" << sfSborder.nComp()
      //                << " box=" << mfi.validbox() << "\n";
      const Box &vbx = mfi.validbox();

      auto Fx = sForcing[mfi].array(0);
      auto Fy = sForcing[mfi].array(1);
      auto Fz = sForcing[mfi].array(2);

      auto fx = fx_cc[mfi].const_array();
      auto fy = fy_cc[mfi].const_array();
      auto fz = fz_cc[mfi].const_array();

      amrex::ParallelFor(vbx,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                           Fx(i, j, k) = fx(i, j, k, 0);
                           Fy(i, j, k) = fy(i, j, k, 0);
                           Fz(i, j, k) = fz(i, j, k, 0);
                           // amrex::Print() << "Packing forcing at i,j,k: " <<
                           // i << "," << j << ","
                           //                << k << " is " << Fx(i, j, k) <<
                           //                "," << Fy(i, j, k) <<
                           //                ","
                           //                << Fz(i, j, k) << "\n";
                         });
    }
  }
  // FillPatchForcing(lev, time, sForcingborder, 0, sForcingborder.nComp());
  if (sForcingborder) {
    FillPatchForcing(lev, time, *sForcingborder, 0, sForcingborder->nComp());
  }
  {
    Real maxFx = fx_cc.norm0(0, 0, true);
    Real maxFy = fy_cc.norm0(0, 0, true);
    Real maxF = std::max(maxFx, maxFy);
    amrex::Print() << "t = " << time << " max |F| = " << maxF
                   << " (Fx=" << maxFx << ", Fy=" << maxFy << ")\n";
  }
  const Real tempdx = xCellSize[lev];
  const Real tempdy = yCellSize[lev];
#if (AMREX_SPACEDIM == 3)
  const Real tempdz = zCellSize[lev];
#else
  const Real tempdz = Real(0.0);
#endif
  const Real temptau = tau[lev];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(sF_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &vbx = mfi.validbox();
    const Box &tbx = mfi.tilebox();
    Box gtbx = mfi.growntilebox(nghost); // <-- safe grown box, clamped to FAB

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
          // collide(i, j, k, statein, feq_loc, ndir, temptau);
          collide_forced(i, j, k, statein, feq_loc, ndir, temptau, wi_loc,
                         dirx_loc, diry_loc, dirz_loc, mac[0], mac[1], mac[2],
                         mac[3], fx(i, j, k), fy(i, j, k), fz(i, j, k));

          // amrex::Real fx_loc = 0.0;
          // amrex::Real fy_loc = 0.0;
          // amrex::Real fz_loc = 0.0;

          // // Inside ParallelFor, instead of passing fx(i,j,k) etc:
          // collide_forced(i, j, k, statein, feq_loc, ndir, temptau, wi_loc,
          //                dirx_loc, diry_loc, dirz_loc, mac[0], mac[1],
          //                mac[2], mac[3], fx_loc, fy_loc, fz_loc);
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
            visPara(i, j, k, rho, u, v, vor, P, tempdx, tempdy, T0);
          }
        });

    if (m_use_cylinder && m_ls) {
      const auto &phi_cc = m_ls->phi_at(lev);
      auto phi = phi_cc[mfi].const_array();

      Real eps_loc = m_ls_par.eps * xCellSize[lev];

      amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j,
                                                   int k) noexcept {
        Real chi = Hsmooth(-phi(i, j, k, 0), eps_loc);

        // Deep inside the solid: overwrite state with rest equilibrium
        if (chi > Real(0.99)) {
          Real rho0 = 1.0; // or just 1.0_rt
          rho(i, j, k) = rho0;
          u(i, j, k) = Real(0.0);
          v(i, j, k) = Real(0.0);
          w(i, j, k) = Real(0.0);
          vor(i, j, k) = Real(0.0);
          P(i, j, k) = rho0 / 3.0_rt;

          for (int q = 0; q < ndir; ++q) {
            Real cu = dirx[q] * u(i, j, k) + diry[q] * v(i, j, k)
#if (AMREX_SPACEDIM == 3)
                      + dirz[q] * w(i, j, k)
#endif
                ;
            Real feq =
                wi_loc[q] * rho0 *
                (Real(1.0) + Real(3.0) * cu + Real(4.5) * cu * cu -
                 Real(1.5) * (u(i, j, k) * u(i, j, k) + v(i, j, k) * v(i, j, k)
#if (AMREX_SPACEDIM == 3)
                              + w(i, j, k) * w(i, j, k)
#endif
                                  ));
            stateout(i, j, k, q) = feq;
          }
        }
      });
    }
  }

#ifdef AMREX_DEBUG
  {
    Real max_rho = macro_new[lev].norm0(0, 0, true);
    Real max_u = macro_new[lev].norm0(1, 0, true);
    Real max_v = macro_new[lev].norm0(2, 0, true);
    Real max_f = f_new[lev].norm0(0, 0, true); // component 0 is enough

    Real maxFx = 0.0;
    if (m_use_cylinder && forcing.size() > lev) {
      maxFx = forcing[lev].norm0(0, 0, true);
    }

    if (amrex::ParallelDescriptor::IOProcessor()) {
      amrex::Print() << "[lev " << lev << "] diag: "
                     << "max rho=" << max_rho << " max |u| ~ "
                     << std::max(max_u, max_v) << " max f=" << max_f
                     << " max Fx=" << maxFx << "\n";
    }
  }
#endif
}
