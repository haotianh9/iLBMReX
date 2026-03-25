#include "AmrCoreLBM.H"

#ifndef LBM_USE_IBM
#define LBM_USE_IBM 1
#endif

#include "Kernels.H"
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PhysBCFunct.H>
#include <iomanip>

#if LBM_USE_IBM
#include "IBM/IBMarkerDF.H"
#include "LevelSet/LevelSet.H"
#endif


using namespace amrex;

void AmrCoreLBM::AdvancePhiAtLevel(int lev, Real time, Real dt_lev,
                                   int /*iteration*/, int /*ncycle*/) {
  // swap time slots
  std::swap(f_new[lev], f_old[lev]);
  std::swap(macro_new[lev], macro_old[lev]);

  MultiFab &sF_new = f_new[lev];
  MultiFab &sM_new = macro_new[lev];
  MultiFab *sForcingPtr = nullptr;

  if (m_use_cylinder && lev == finestLevel()) {
    if (forcing.size() > lev && forcing[lev].ok() &&
        forcing[lev].nComp() == 3) {
      sForcingPtr = &forcing[lev];
    }
  } else {
    // enforce "0 anywhere else"
    if (m_use_cylinder && forcing.size() > lev && forcing[lev].ok()) {
      forcing[lev].setVal(0.0);
    }
  }

  // local IBM scratch:
  MultiFab fx_cc(grids[lev], dmap[lev], 1, nghost);
  MultiFab fy_cc(grids[lev], dmap[lev], 1, nghost);
  MultiFab fz_cc(grids[lev], dmap[lev], 1, nghost);
  fx_cc.setVal(0.0);
  fy_cc.setVal(0.0);
  fz_cc.setVal(0.0);

  // ---- Prescribed forcing (simple validation / regression test) ----
  // Applies everywhere on all levels unless you override it via IBM.
  if (m_use_prescribed_force) {
    fx_cc.setVal(m_prescribed_force[0]);
    fy_cc.setVal(m_prescribed_force[1]);
    fz_cc.setVal(m_prescribed_force[2]);
  }
  MultiFab sfSborder(grids[lev], dmap[lev], sF_new.nComp(), sF_new.nGrow());
  MultiFab smSborder(grids[lev], dmap[lev], sM_new.nComp(), sM_new.nGrow());
  FillPatchMesoscopic(lev, time, sfSborder, 0, sfSborder.nComp());
  FillPatchMacro(lev, time, smSborder, 0, smSborder.nComp());

  // Alias macro fields from the macro staging MultiFab.
  MultiFab rhocc(smSborder, amrex::make_alias, 0, 1); // rho
  MultiFab ucc(smSborder, amrex::make_alias, 1, 1);   // u
  MultiFab vcc(smSborder, amrex::make_alias, 2, 1);   // v
#if (AMREX_SPACEDIM == 3)
  MultiFab wcc(smSborder, amrex::make_alias, 3, 1);   // w
#else
  MultiFab wcc(grids[lev], dmap[lev], 1, 0); // dummy w in 2D
  wcc.setVal(0.0);
#endif

#if LBM_USE_IBM
  std::unique_ptr<MultiFab> prevFxAlias;
  std::unique_ptr<MultiFab> prevFyAlias;
  std::unique_ptr<MultiFab> prevFzAlias;

  if (m_use_cylinder && lev == finestLevel()) {
    EnsureIBMBackend(lev, dmap[lev], grids[lev]);
  }

  // ---- Marker DF IBM (no level-set required; only on finest) ----
  if (lev == finestLevel() && m_ib_method == IBMMethodMarker && m_ibm) {
    if (sForcingPtr && sForcingPtr->nComp() >= 3) {
      prevFxAlias = std::make_unique<MultiFab>(*sForcingPtr, amrex::make_alias, 0, 1);
      prevFyAlias = std::make_unique<MultiFab>(*sForcingPtr, amrex::make_alias, 1, 1);
      prevFzAlias = std::make_unique<MultiFab>(*sForcingPtr, amrex::make_alias, 2, 1);
    }
    // IAMReX-style multi-direct-forcing with delta-kernel
    m_ibm->update_forcing(rhocc, ucc, vcc, wcc, fx_cc, fy_cc, fz_cc, dt_lev,
                          prevFxAlias.get(), prevFyAlias.get(),
                          prevFzAlias.get());
  }
#endif // LBM_USE_IBM
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

  if (sForcingborder) {
    FillPatchForcing(lev, time, *sForcingborder, 0, sForcingborder->nComp());
  }
  if (m_force_eval_debug > 0) {
    // NOTE: `local=true` is per-rank and can print zero on IO rank even when
    // other ranks have nonzero forcing. Report both local and global norms.
    const Real maxFx_local = fx_cc.norm0(0, 0, true);
    const Real maxFy_local = fy_cc.norm0(0, 0, true);
    const Real maxFx_global = fx_cc.norm0(0, 0, false);
    const Real maxFy_global = fy_cc.norm0(0, 0, false);
    const Real maxF_global = std::max(maxFx_global, maxFy_global);

    Real maxFx_packed = Real(0.0);
    Real maxFy_packed = Real(0.0);
    if (sForcingPtr) {
      maxFx_packed = sForcingPtr->norm0(0, 0, false);
      maxFy_packed = sForcingPtr->norm0(1, 0, false);
    }
    const Real maxF_packed = std::max(maxFx_packed, maxFy_packed);

    amrex::Print() << std::scientific << std::setprecision(6)
                   << "[diag_force] lev=" << lev
                   << " step=" << (istep[lev] + 1)
                   << " t=" << time
                   << " cc_max|F|_global=" << maxF_global
                   << " (Fx=" << maxFx_global << ", Fy=" << maxFy_global << ")"
                   << " cc_max|F|_local=" << std::max(maxFx_local, maxFy_local)
                   << " packed_max|F|_global=" << maxF_packed
                   << " (Fx=" << maxFx_packed << ", Fy=" << maxFy_packed
                   << ")\n"
                   << std::defaultfloat;
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
  // We update ghost cells in `gtbx`, so tiled iteration would create
  // overlapping grown regions and collide some cells multiple times.
  for (MFIter mfi(sF_new, false); mfi.isValid(); ++mfi) {
    const Box &vbx = mfi.validbox();
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

    const int ndir_l = ndir;
    auto const *dirx_d = dirx_dev.data();
    auto const *diry_d = diry_dev.data();
    auto const *dirz_d = dirz_dev.data();
    auto const *wi_d = wi_dev.data();
    // -------- Collide (BGK) --------
    amrex::ParallelFor(
        gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // local macros from last stage
          const amrex::Real rho_loc = rho_old(i, j, k);
          const amrex::Real u_loc = u_old(i, j, k);
          const amrex::Real v_loc = v_old(i, j, k);
          const amrex::Real w_loc = w_old(i, j, k);
          // build feq

          amrex::Real feq_loc[BCBuf::MAX_NDIR];

          for (int q = 0; q < ndir_l; ++q) {
            feq_loc[q] = feqFunction(wi_d[q], dirx_d[q], diry_d[q], dirz_d[q],
                                     rho_loc, u_loc, v_loc, w_loc);
          }
          collide_forced(i, j, k, statein, feq_loc, ndir_l, temptau, wi_d,
                         dirx_d, diry_d, dirz_d, rho_loc, u_loc, v_loc, w_loc,
                         fx(i, j, k), fy(i, j, k), fz(i, j, k));

          // amrex::Real fx_loc = 0.0;
          // amrex::Real fy_loc = 0.0;
          // amrex::Real fz_loc = 0.0;

          // // Inside ParallelFor, instead of passing fx(i,j,k) etc:
          // collide_forced(i, j, k, statein, feq_loc, ndir, temptau, wi_loc,
          //                dirx_loc, diry_loc, dirz_loc, mac[0], mac[1],
          //                mac[2], mac[3], fx_loc, fy_loc, fz_loc);
        });

    const amrex::Box domain = geom[lev].Domain();
    const amrex::BCRec bc_rec =
        bcsMesoscopic.empty() ? amrex::BCRec() : bcsMesoscopic[0];

    // For `user_1` walls, impose stationary halfway bounce-back by populating
    // ghost cells with the fully reversed post-collision population from the
    // boundary-adjacent fluid cell that will pull from that ghost location.
    amrex::ParallelFor(
        gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          if (vbx.contains(i, j, k)) {
            return;
          }

          int flip_x = 0;
          int flip_y = 0;
#if (AMREX_SPACEDIM == 3)
          int flip_z = 0;
#endif
          bool has_bounce_back = false;
          bool unsupported_face = false;

          if (i < domain.smallEnd(0)) {
            if (bc_rec.lo(0) == amrex::BCType::user_1) {
              flip_x = 1;
              has_bounce_back = true;
            } else {
              unsupported_face = true;
            }
          } else if (i > domain.bigEnd(0)) {
            if (bc_rec.hi(0) == amrex::BCType::user_1) {
              flip_x = 1;
              has_bounce_back = true;
            } else {
              unsupported_face = true;
            }
          }

          if (j < domain.smallEnd(1)) {
            if (bc_rec.lo(1) == amrex::BCType::user_1) {
              flip_y = 1;
              has_bounce_back = true;
            } else {
              unsupported_face = true;
            }
          } else if (j > domain.bigEnd(1)) {
            if (bc_rec.hi(1) == amrex::BCType::user_1) {
              flip_y = 1;
              has_bounce_back = true;
            } else {
              unsupported_face = true;
            }
          }

#if (AMREX_SPACEDIM == 3)
          if (k < domain.smallEnd(2)) {
            if (bc_rec.lo(2) == amrex::BCType::user_1) {
              flip_z = 1;
              has_bounce_back = true;
            } else {
              unsupported_face = true;
            }
          } else if (k > domain.bigEnd(2)) {
            if (bc_rec.hi(2) == amrex::BCType::user_1) {
              flip_z = 1;
              has_bounce_back = true;
            } else {
              unsupported_face = true;
            }
          }
#endif

          if (!has_bounce_back || unsupported_face) {
            return;
          }

          for (int q = 0; q < ndir_l; ++q) {
            const int itgt = i + static_cast<int>(dirx_d[q]);
            const int jtgt = j + static_cast<int>(diry_d[q]);
#if (AMREX_SPACEDIM == 3)
            const int ktgt = k + static_cast<int>(dirz_d[q]);
#else
            const int ktgt = k;
#endif

            if (!vbx.contains(itgt, jtgt, ktgt)) {
              continue;
            }

            const int rx = -static_cast<int>(dirx_d[q]);
            const int ry = -static_cast<int>(diry_d[q]);
#if (AMREX_SPACEDIM == 3)
            const int rz = -static_cast<int>(dirz_d[q]);
#else
            const int rz = 0;
#endif

            int q_reflect = q;
            for (int p = 0; p < ndir_l; ++p) {
              if (static_cast<int>(dirx_d[p]) == rx &&
                  static_cast<int>(diry_d[p]) == ry &&
                  static_cast<int>(dirz_d[p]) == rz) {
                q_reflect = p;
                break;
              }
            }

            statein(i, j, k, q) = statein(itgt, jtgt, ktgt, q_reflect);
          }
        });

    // // -------- Stream --------
    amrex::ParallelFor(
        gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          if (vbx.contains(i, j, k)) {
            stream(i, j, k, stateout, statein, dirx_d, diry_d, dirz_d, ndir_l);
          }
        });

    // -------- Macros w/ forcing + diagnostics --------
    amrex::ParallelFor(
        gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          // Ensure that the calculation area is within the valid range
          if (vbx.contains(i, j, k)) {
            calculateMacroForcing(i, j, k, stateout, rho, u, v, w, fx, fy, fz,
                                  ndir_l, dirx_d, diry_d, dirz_d);
            // calculateMacro(i, j, k, stateout, rho, u, v, w, ndir, dirx, diry,
            // dirz);
          }
        });

  }
  const int step_lev = istep[lev];
  const bool do_vorticity = (((step_lev + 1) % regrid_int) == 0) ||
                            ((plot_int > 0) && (((step_lev + 1) % plot_int) == 0));

  if (do_vorticity) {

    macro_new[lev].FillBoundary(geom[lev].periodicity());

    const Real T0_local = T0; // <--- NEW: copy the member value
    for (MFIter mfi(sF_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &vbx = mfi.validbox();
      const Box &tbx = mfi.tilebox();
      Box gtbx = mfi.growntilebox(nghost); // <-- safe grown box, clamped to FAB
      Array4<Real> rho = sM_new[mfi].array(0);
      Array4<Real> u = sM_new[mfi].array(1);
      Array4<Real> v = sM_new[mfi].array(2);
      Array4<Real> w = sM_new[mfi].array(3);
      Array4<Real> vor = sM_new[mfi].array(4);
      Array4<Real> P = sM_new[mfi].array(5);

      amrex::ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j,
                                                    int k) noexcept {
        // Ensure that the calculation area is within the
        // valid range
        if (vbx.contains(i, j, k)) {

          visPara(i, j, k, rho, u, v, vor, P, tempdx, tempdy, tempdz, T0_local);
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
    if (m_use_cylinder && forcing.size() > lev &&
        forcing[lev].boxArray().size() > 0) {
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
