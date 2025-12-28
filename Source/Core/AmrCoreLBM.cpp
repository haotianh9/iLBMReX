
#include <AMReX_FillPatchUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>

#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include <AmrCoreLBM.H>
#include <Kernels.H>

#include "DebugNaN.H"

#include "IBM/IBDiffuseLS.H"
#include "IBM/IBMarkerDF.H"
#include "IBM/IBSharpLS.H"
#include "LevelSet/LevelSet.H"
#include <AMReX_GpuContainers.H>

#include <fstream>
#include <iomanip> // optional, for nicer formatting

using namespace amrex;

// constructor - reads in parameters from inputs file
//             - sizes multilevel arrays and data structures
//             - initializes BCRec boundary condition object
AmrCoreLBM::AmrCoreLBM() {

  ReadParameters();

  // Geometry on all levels has been defined already.

  // No valid BoxArray and DistributionMapping have been defined.
  // But the arrays for them have been resized.

  int nlevs_max = max_level + 1;

  istep.resize(nlevs_max, 0);
  nsubsteps.resize(nlevs_max, 1);

  for (int lev = 1; lev <= max_level; ++lev) {
    nsubsteps[lev] = MaxRefRatio(lev - 1);
  }

  t_new.resize(nlevs_max, 0.0);
  t_old.resize(nlevs_max, -1.e100);
  dt.resize(nlevs_max, 1.e100);

  xCellSize.resize(nlevs_max, 0.0);
  yCellSize.resize(nlevs_max, 0.0);
  zCellSize.resize(nlevs_max, 0.0);

  for (int lev = 0; lev <= max_level; ++lev) {
    xCellSize[lev] = geom[lev].CellSize(0);
    yCellSize[lev] = geom[lev].CellSize(1);
    zCellSize[lev] = geom[lev].CellSize(2);
  }
  tau_base = 3.0 * xCellSize[0] * nu + 0.5;

  tau.resize(nlevs_max, 0.0);
  dt[0] = xCellSize[0];
  tau[0] = tau_base;

  for (int lev = 1; lev <= max_level; ++lev) {
    dt[lev] = xCellSize[lev];
    tau[lev] = 0.5 + nsubsteps[lev] * (tau[lev - 1] - 0.5);
  }

  f_new.resize(nlevs_max);
  f_old.resize(nlevs_max);
  macro_new.resize(nlevs_max);
  macro_old.resize(nlevs_max);

  if (m_use_cylinder) {
    amrex::Print() << "Using IBM with method = " << m_ib_method << std::endl;
    // just one forcing
    forcing.resize(nlevs_max);
  }
  // set up boundary conditions

  using namespace BCVals;

  int bc_lo[AMREX_SPACEDIM];
  int bc_hi[AMREX_SPACEDIM];

  // 1) Define bc_lo/bc_hi and bcval for both periodic and non-periodic cases
  if (!(geom[0].isAllPeriodic())) {

    amrex::Vector<int> temp0, temp1;
    ParmParse pp("amrbc");

    pp.getarr("bc_lo", temp0);
    pp.getarr("bc_hi", temp1);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      bc_lo[idim] = temp0[idim];
      bc_hi[idim] = temp1[idim];

      if (bc_lo[idim] == amrex::BCType::ext_dir) {
        std::string dir = std::to_string(idim);
        pp.query(("bc_lo_" + dir + "_rho_val").c_str(),
                 BCVals::bc_lo_rho_val[idim]);
        pp.query(("bc_lo_" + dir + "_ux_val").c_str(),
                 BCVals::bc_lo_ux_val[idim]);
        pp.query(("bc_lo_" + dir + "_uy_val").c_str(),
                 BCVals::bc_lo_uy_val[idim]);
      }
      if (bc_hi[idim] == amrex::BCType::ext_dir) {
        std::string dir = std::to_string(idim);
        pp.query(("bc_hi_" + dir + "_rho_val").c_str(),
                 BCVals::bc_hi_rho_val[idim]);
        pp.query(("bc_hi_" + dir + "_ux_val").c_str(),
                 BCVals::bc_hi_ux_val[idim]);
        pp.query(("bc_hi_" + dir + "_uy_val").c_str(),
                 BCVals::bc_hi_uy_val[idim]);
      }
    }

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      bcval.lo_rho[idim] = BCVals::bc_lo_rho_val[idim];
      bcval.lo_ux[idim] = BCVals::bc_lo_ux_val[idim];
      bcval.lo_uy[idim] = BCVals::bc_lo_uy_val[idim];
      bcval.hi_rho[idim] = BCVals::bc_hi_rho_val[idim];
      bcval.hi_ux[idim] = BCVals::bc_hi_ux_val[idim];
      bcval.hi_uy[idim] = BCVals::bc_hi_uy_val[idim];
    }

  } else {
    // Fully periodic: still must provide valid BCRec vectors for FillPatch.
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      bc_lo[idim] = amrex::BCType::int_dir;
      bc_hi[idim] = amrex::BCType::int_dir;

      // Not used for int_dir, but keep defined.
      bcval.lo_rho[idim] = 0.0_rt;
      bcval.lo_ux[idim] = 0.0_rt;
      bcval.lo_uy[idim] = 0.0_rt;
      bcval.hi_rho[idim] = 0.0_rt;
      bcval.hi_ux[idim] = 0.0_rt;
      bcval.hi_uy[idim] = 0.0_rt;
    }
  }

  // 2) ALWAYS build BCRec vectors (required even if domain is periodic)
  bcsMesoscopic.resize(ndir);
  for (int c = 0; c < ndir; ++c) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      bcsMesoscopic[c].setLo(idim, bc_lo[idim]);
      bcsMesoscopic[c].setHi(idim, bc_hi[idim]);
    }
  }

  bcsMacro.resize(nmac);
  for (int c = 0; c < nmac; ++c) {
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      bcsMacro[c].setLo(idim, bc_lo[idim]);
      bcsMacro[c].setHi(idim, bc_hi[idim]);
    }
  }

  amrex::Print() << "AmrCoreLBM::AmrCoreLBM() finished." << std::endl;
}

AmrCoreLBM::~AmrCoreLBM() {}

// advance solution to final time
void AmrCoreLBM::Evolve() {

  Real cur_time = t_new[0];
  int last_plot_file_step = 0;

  amrex::Real nu = 0.064;
  amrex::Real H = 64.0;
  std::vector<amrex::Real> target_tstar = {0.005, 0.05, 0.5};
  std::vector<bool> tstar_written(target_tstar.size(), false);

  for (int step = istep[0]; step < max_step && cur_time < stop_time; ++step) {
    amrex::Print() << "\nCoarse STEP " << step + 1 << " starts ..."
                   << std::endl;

    int lev = 0;
    int iteration = 1;

    timeStepWithSubcycling(lev, cur_time, iteration);

    cur_time += dt[0];

    // sum rho to check conservation
    Real sum_rho = macro_new[0].sum(0);

    if (m_use_cylinder && m_force_interval > 0 &&
        ((step + 1) % m_force_interval == 0)) {
      ComputeIBForce(cur_time, step + 1);
    }

    amrex::Print() << "Coarse STEP " << step + 1 << " ends."
                   << " TIME = " << cur_time << " DT = " << dt[0]
                   << " Sum(Rho) = " << sum_rho << std::endl;

    // sync up time
    for (lev = 0; lev <= finest_level; ++lev) {
      t_new[lev] = cur_time;
    }

    if (plot_int > 0 && (step + 1) % plot_int == 0) {
      last_plot_file_step = step + 1;
      WritePlotFile();
    }

    if (chk_int > 0 && (step + 1) % chk_int == 0) {
      WriteCheckpointFile();
    }

#ifdef AMREX_MEM_PROFILING
    {
      std::ostringstream ss;
      ss << "[STEP " << step + 1 << "]";
      MemProfiler::report(ss.str());
    }
#endif

    if (cur_time >= stop_time - 1.e-6 * dt[0])
      break;

    Real t_star = nu * cur_time / (H * H);
    for (int i = 0; i < target_tstar.size(); ++i) {
      if (!tstar_written[i] && std::abs(t_star - target_tstar[i]) < 1e-5) {
        amrex::Print() << ">>> Output at t* = " << t_star
                       << " (t = " << cur_time << ")" << std::endl;
        std::string tag = "_tstar" + std::to_string(i);
        WritePlotFile();
        tstar_written[i] = true;
      }
    }
  }

  if (plot_int > 0 && istep[0] > last_plot_file_step) {
    WritePlotFile();
  }
}

void AmrCoreLBM::ComputeIBForce(Real time, int step) const {
  if (!m_use_cylinder) {
    return; // no IBM
  }

  const int lev = finestLevel(); // IBM is active only on finest level
  if (lev < 0)
    return;

  if (lev >= static_cast<int>(forcing.size())) {
    return;
  }

  if (forcing[lev].nComp() < AMREX_SPACEDIM) {
    return;
  }

  // --------------------------------------------------------------------
  // 1) Integrate body-force density over the domain
  // --------------------------------------------------------------------
  // Sum only over the *valid domain* (no ghost cells)
  const amrex::Box &domain = geom[lev].Domain();

  Real Fx_sum = forcing[lev].sum(domain, 0, false); // region, comp, local=false
  Real Fy_sum = forcing[lev].sum(domain, 1, false);
#if (AMREX_SPACEDIM == 3)
  Real Fz_sum = forcing[lev].sum(domain, 2, false);
#else
  Real Fz_sum = 0.0;
#endif

  const auto dx = geom[lev].CellSizeArray();
  Real dV = dx[0];
#if (AMREX_SPACEDIM >= 2)
  dV *= dx[1];
#endif
#if (AMREX_SPACEDIM == 3)
  dV *= dx[2];
#endif

  // Total force on fluid (integral of body-force density)
  Real Fx_fluid = Fx_sum * dV;
  Real Fy_fluid = Fy_sum * dV;
  Real Fz_fluid = Fz_sum * dV;

  // Reaction on the body
  Real Fx_body = -Fx_fluid;
  Real Fy_body = -Fy_fluid;
  Real Fz_body = -Fz_fluid;

  // --------------------------------------------------------------------
  // 2) Drag / lift coefficients
  //
  //   Cd = Fx_body / (0.5 * rho_ref * U_ref^2 * D)
  //   Cl = Fy_body / (0.5 * rho_ref * U_ref^2 * D)
  //
  // Here:
  //   - rho_ref ≈ 1 (LBM)
  //   - U_ref   = U0 (already set in ReadParameters)
  //   - D       = 2*R, with R from the cylinder level-set params
  // --------------------------------------------------------------------
  Real rho_ref = 1.0_rt;            // standard LBM choice
  Real U_ref = U0;                  // set earlier in ReadParameters()
  Real D_ref = 2.0_rt * m_ls_par.R; // cylinder diameter

  Real Cd = 0.0_rt;
  Real Cl = 0.0_rt;

  Real denom = 0.5_rt * rho_ref * U_ref * U_ref * D_ref;
  if (denom > 0.0_rt) {
    Cd = Fx_body / denom;
    Cl = Fy_body / denom;
  }

  // --------------------------------------------------------------------
  // 3) Print + append everything to force.dat
  // --------------------------------------------------------------------
  if (amrex::ParallelDescriptor::IOProcessor()) {

    static std::ofstream ofs;
    static bool initialized = false;

    if (!initialized) {
      ofs.open("force.dat");
      if (!ofs) {
        amrex::Print() << "WARNING: could not open force.dat\n";
        return;
      }
      ofs << "# step  time  Fx_body  Fy_body  Cd  Cl\n";
      initialized = true;
    }

    ofs << step << "  " << time << "  " << Fx_body << "  " << Fy_body << "  "
        << Cd << "  " << Cl << "\n";
  }
}

// initializes multilevel data
void AmrCoreLBM::InitData() {
  if (restart_chkfile == "") {
    // start simulation from the beginning
    const Real time = 0.0;
    InitFromScratch(time);

    AverageDown();
    if (m_use_cylinder && m_ls) {
      for (int lev = 0; lev <= finest_level; ++lev) {
        BuildLevelSetOnLevel(lev);
      }
    }

    /*
if (chk_int > 0) {
WriteCheckpointFile();
}
    */
  } else {
    // restart from a checkpoint
    // ReadCheckpointFile();
  }

  if (plot_int > 0) {

    WritePlotFile();
  }
}

// Make a new level using provided BoxArray and DistributionMapping and
// fill with interpolated coarse level data.
// overrides the pure virtual function in AmrCore
void AmrCoreLBM::MakeNewLevelFromCoarse(int lev, Real time, const BoxArray &ba,
                                        const DistributionMapping &dm) {

  const int f_ncomp = f_new[lev - 1].nComp();
  const int f_nghost = f_new[lev - 1].nGrow();
  const int m_ncomp = macro_new[lev - 1].nComp();
  const int m_nghost = macro_new[lev - 1].nGrow();

  f_new[lev].define(ba, dm, f_ncomp, f_nghost);
  f_old[lev].define(ba, dm, f_ncomp, f_nghost);
  macro_new[lev].define(ba, dm, m_ncomp, m_nghost);
  macro_old[lev].define(ba, dm, m_ncomp, m_nghost);

  t_new[lev] = time;
  t_old[lev] = time - 1.e200;

  // This part requires special attention. For AMR stability, do not
  // interpolate the mesoscopic populations f_i from coarse to fine. Instead,
  // initialize macros first and reconstruct f_i from equilibrium.
  FillCoarsePatchMacro(lev, time, macro_new[lev], 0, m_ncomp);
  MultiFab::Copy(macro_old[lev], macro_new[lev], 0, 0, m_ncomp, m_nghost);

  {
    auto const *wi_d = wi_dev.data();
    auto const *cx_d = dirx_dev.data();
    auto const *cy_d = diry_dev.data();
    auto const *cz_d = dirz_dev.data();
    const int ndir_l = ndir;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(f_new[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

      const Box &bx = mfi.fabbox();

      auto const rho = macro_new[lev][mfi].const_array(0);
      auto const ux = macro_new[lev][mfi].const_array(1);
      auto const uy = macro_new[lev][mfi].const_array(2);
      auto const uz = macro_new[lev][mfi].const_array(3);

      auto const f = f_new[lev][mfi].array();
      auto const fo = f_old[lev][mfi].array();

      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j,
                                                  int k) noexcept {
        Real r = amrex::max(rho(i, j, k), 1.e-12_rt);
        Real u0 = ux(i, j, k);
        Real v0 = uy(i, j, k);
        Real w0 = uz(i, j, k);
        Real u2 = u0 * u0 + v0 * v0 + w0 * w0;

        for (int q = 0; q < ndir_l; ++q) {
          Real cu = cx_d[q] * u0 + cy_d[q] * v0 + cz_d[q] * w0;
          Real feq = wi_d[q] * r *
                     (1.0_rt + 3.0_rt * cu + 4.5_rt * cu * cu - 1.5_rt * u2);
          f(i, j, k, q) = feq;
          fo(i, j, k, q) = feq;
        }
      });
    }
  }

  // --- Level-set φ on this level ---
  if (m_use_cylinder) {
    // --- Level-set φ on this level ---
    BuildLevelSetOnLevel(lev);
    const int nforce = 3;
    forcing[lev].define(ba, dm, nforce, m_nghost);
    forcing[lev].setVal(0.0);
  }

  // --- IBM object only for current finest (optional) ---
  if (m_use_cylinder && lev == finestLevel()) {
    if (m_ib_method == 1) {
      m_ibd = std::make_unique<IBDiffuseLS>(geom[lev]);
      m_ibs.reset();
      m_ibm.reset();
    } else if (m_ib_method == 2) {
      m_ibs = std::make_unique<IBSharpLS>(geom[lev]);
      m_ibd.reset();
      m_ibm.reset();
    } else if (m_ib_method == 3) {
      m_ibm = std::make_unique<IBMarkerDF>(geom[lev], dm, ba, m_ls_par.x0,
                                           m_ls_par.y0, m_ls_par.z0, m_ls_par.R,
                                           m_marker_par);
      m_ibd.reset();
      m_ibs.reset();
    } else {
      m_ibd.reset();
      m_ibs.reset();
      m_ibm.reset();
    }
  }
}

// Remake an existing level using provided BoxArray and DistributionMapping and
// fill with existing fine and coarse data.
// overrides the pure virtual function in AmrCore
void AmrCoreLBM::RemakeLevel(int lev, Real time, const BoxArray &ba,
                             const DistributionMapping &dm) {

  const int f_ncomp = f_new[lev].nComp();
  const int f_nghost = f_new[lev].nGrow();
  const int m_ncomp = macro_new[lev].nComp();
  const int m_nghost = macro_new[lev].nGrow();

  MultiFab newF_state(ba, dm, f_ncomp, f_nghost);
  MultiFab oldF_state(ba, dm, f_ncomp, f_nghost);
  MultiFab newM_state(ba, dm, m_ncomp, m_nghost);
  MultiFab oldM_state(ba, dm, m_ncomp, m_nghost);
  // Fill macroscopic state first (time-consistent), then reconstruct mesoscopic
  // populations from equilibrium. This is significantly more robust across AMR
  // regrids/remaps than interpolating f_i directly.
  FillPatchMacro(lev, time, newM_state, 0, m_ncomp);
  MultiFab::Copy(oldM_state, newM_state, 0, 0, m_ncomp, m_nghost);

  {
    auto const *wi_d = wi_dev.data();
    auto const *cx_d = dirx_dev.data();
    auto const *cy_d = diry_dev.data();
    auto const *cz_d = dirz_dev.data();
    const int ndir_l = ndir;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(newF_state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

      const Box &bx = mfi.fabbox();

      auto const rho = newM_state[mfi].const_array(0);
      auto const ux = newM_state[mfi].const_array(1);
      auto const uy = newM_state[mfi].const_array(2);
      auto const uz = newM_state[mfi].const_array(3);

      auto const f = newF_state[mfi].array();
      auto const fo = oldF_state[mfi].array();

      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j,
                                                  int k) noexcept {
        Real r = amrex::max(rho(i, j, k), 1.e-12_rt);
        Real u0 = ux(i, j, k);
        Real v0 = uy(i, j, k);
        Real w0 = uz(i, j, k);
        Real u2 = u0 * u0 + v0 * v0 + w0 * w0;

        for (int q = 0; q < ndir_l; ++q) {
          Real cu = cx_d[q] * u0 + cy_d[q] * v0 + cz_d[q] * w0;
          Real feq = wi_d[q] * r *
                     (1.0_rt + 3.0_rt * cu + 4.5_rt * cu * cu - 1.5_rt * u2);
          f(i, j, k, q) = feq;
          fo(i, j, k, q) = feq;
        }
      });
    }
  }

  std::swap(newF_state, f_new[lev]);
  std::swap(oldF_state, f_old[lev]);
  std::swap(newM_state, macro_new[lev]);
  std::swap(oldM_state, macro_old[lev]);
  // --- Level-set φ on this level ---
  if (m_use_cylinder) {
    // --- Level-set φ on this level ---
    BuildLevelSetOnLevel(lev);
    const int nforce = 3;
    forcing[lev].define(ba, dm, nforce, m_nghost);
    forcing[lev].setVal(0.0);
  }

  // --- IBM object only for current finest (optional) ---
  if (m_use_cylinder && lev == finestLevel()) {
    if (m_ib_method == 1) {
      m_ibd = std::make_unique<IBDiffuseLS>(geom[lev]);
      m_ibs.reset();
      m_ibm.reset();
    } else if (m_ib_method == 2) {
      m_ibs = std::make_unique<IBSharpLS>(geom[lev]);
      m_ibd.reset();
      m_ibm.reset();
    } else if (m_ib_method == 3) {
      m_ibm = std::make_unique<IBMarkerDF>(geom[lev], dm, ba, m_ls_par.x0,
                                           m_ls_par.y0, m_ls_par.z0, m_ls_par.R,
                                           m_marker_par);
      m_ibd.reset();
      m_ibs.reset();
    } else {
      m_ibd.reset();
      m_ibs.reset();
      m_ibm.reset();
    }
  }
}

// Delete level data
// overrides the pure virtual function in AmrCore
void AmrCoreLBM::ClearLevel(int lev) {
  f_new[lev].clear();
  f_old[lev].clear();
  macro_new[lev].clear();
  macro_old[lev].clear();
  if (m_use_cylinder && lev < static_cast<int>(forcing.size())) {
    forcing[lev].clear();
  }
}

// Make a new level from scratch using provided BoxArray and
// DistributionMapping. Only used during initialization. overrides the pure
// virtual function in AmrCore
void AmrCoreLBM::MakeNewLevelFromScratch(int lev, Real time, const BoxArray &ba,
                                         const DistributionMapping &dm) {

  amrex::Print() << "Making new level " << lev << std::endl;

  f_new[lev].define(ba, dm, ndir, nghost);
  f_old[lev].define(ba, dm, ndir, nghost);
  macro_new[lev].define(ba, dm, nmac, nghost);
  macro_old[lev].define(ba, dm, nmac, nghost);

  if (m_use_cylinder) {
    // --- Level-set φ on this level ---
    BuildLevelSetOnLevel(lev);
    const int nforce = 3;
    forcing[lev].define(ba, dm, nforce, nghost);
    forcing[lev].setVal(0.0);
  }

  // --- IBM object only for current finest (optional) ---
  if (m_use_cylinder && lev == finestLevel()) {
    if (m_ib_method == 1) {
      m_ibd = std::make_unique<IBDiffuseLS>(geom[lev]);
      m_ibs.reset();
      m_ibm.reset();
    } else if (m_ib_method == 2) {
      m_ibs = std::make_unique<IBSharpLS>(geom[lev]);
      m_ibd.reset();
      m_ibm.reset();
    } else if (m_ib_method == 3) {
      m_ibm = std::make_unique<IBMarkerDF>(geom[lev], dm, ba, m_ls_par.x0,
                                           m_ls_par.y0, m_ls_par.z0, m_ls_par.R,
                                           m_marker_par);
      m_ibd.reset();
      m_ibs.reset();
    } else {
      m_ibd.reset();
      m_ibs.reset();
      m_ibm.reset();
    }
  }

  MultiFab &cur = macro_new[lev];

  const auto problo = Geom(lev).ProbLoArray();
  const auto probhi = Geom(lev).ProbHiArray();
  const auto dx = Geom(lev).CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(cur, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

    Array4<Real> rho = cur[mfi].array(0);
    Array4<Real> u = cur[mfi].array(1);
    Array4<Real> v = cur[mfi].array(2);
    Array4<Real> w = cur[mfi].array(3);
    Array4<Real> vor = cur[mfi].array(4);
    const Box &box = mfi.fabbox();

    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      initdata(box, rho, u, v, w, vor, problo, probhi, dx, nu);
    });
  }

  MultiFab &curMacro = macro_new[lev];
  amrex::Real tempdx = xCellSize[lev];
  amrex::Real tempdy = yCellSize[lev];
  amrex::Real tempdz = zCellSize[lev];
  Real T0_local = T0;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(curMacro, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

    const Box &vbx = mfi.validbox();
    const Box &tbx = mfi.tilebox();
    const Box &gtbx = amrex::grow(tbx, nghost);

    Array4<Real> rho = curMacro[mfi].array(0);
    Array4<Real> u = curMacro[mfi].array(1);
    Array4<Real> v = curMacro[mfi].array(2);
    Array4<Real> vor = curMacro[mfi].array(4);
    Array4<Real> P = curMacro[mfi].array(5);
    amrex::ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j,
                                                  int k) noexcept {
      if (vbx.contains(i, j, k)) {
        visPara(i, j, k, rho, u, v, vor, P, tempdx, tempdy, tempdz, T0_local);
      }
    });
  }

  MultiFab::Copy(macro_old[lev], macro_new[lev], 0, 0, nmac, nghost);

  InitEquilibrium();
}

// tag all cells for refinement
// overrides the pure virtual function in AmrCore
void AmrCoreLBM::ErrorEst(int lev, TagBoxArray &tags, Real /*time*/,
                          int /*ngrow*/) {

  static bool first = true;
  static Vector<Real> thresholdRatio;

  // only do this during the first call to ErrorEst
  if (first) {
    first = false;
    // read in an array of "phierr", which is the tagging threshold
    // in this example, we tag values of "phi" which are greater than phierr
    // for that particular level
    // in subroutine state_error, you could use more elaborate tagging, such
    // as more advanced logical expressions, or gradients, etc.
    ParmParse pp("amrvorth");
    int n = pp.countval("thresholdRatio");
    if (n > 0) {
      pp.getarr("thresholdRatio", thresholdRatio, 0, n);
    }

    // Ensure thresholdRatio is defined for all levels [0..max_level]
    if (thresholdRatio.empty()) {
      thresholdRatio.resize(max_level + 1, 1.0_rt);
    } else if (static_cast<int>(thresholdRatio.size()) < max_level + 1) {
      thresholdRatio.resize(max_level + 1, thresholdRatio.back());
    }
  }

  MultiFab &curMacro = macro_new[lev];

  // Max vorticity (component 4) on this level
  amrex::Real vortMax = curMacro.max(4);
  if (m_use_cylinder) {
    // Level set φ for this level
    MultiFab &phi_mf = m_ls->phi_at(lev);

    // Characteristic cell size (assuming dx = dy = dz for now)
    amrex::Real dx_min = xCellSize[lev];

    //    const int clearval = TagBox::CLEAR;
    const int tagval = TagBox::SET;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {

      for (MFIter mfi(curMacro, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box &bx = mfi.validbox();

        // vorticity: macro_new[lev] component 4
        auto const &vort = curMacro[mfi].const_array(4);
        // level set φ: component 0 of m_ls->phi_at(lev)

        auto const &phi_arr = phi_mf[mfi].const_array(0);

        auto const &tagfab = tags[mfi].array();

        Real threshold = thresholdRatio[lev] * vortMax;
        int n_cells_band = 5; // you can set 3–5 as you like

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              // Tag by vorticity
              vorticity_tagging(i, j, k, tagfab, vort, threshold, tagval);

              // Tag by level-set band around structure
              levelset_tagging(i, j, k, tagfab, phi_arr, dx_min, n_cells_band,
                               tagval);
            });
      }
    }

  } else {

    // Characteristic cell size (assuming dx = dy = dz for now)
    amrex::Real dx_min = xCellSize[lev];

    //    const int clearval = TagBox::CLEAR;
    const int tagval = TagBox::SET;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {

      for (MFIter mfi(curMacro, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box &bx = mfi.validbox();

        // vorticity: macro_new[lev] component 4
        auto const &vort = curMacro[mfi].const_array(4);
        // level set φ: component 0 of m_ls->phi_at(lev)

        auto const &tagfab = tags[mfi].array();

        Real threshold = thresholdRatio[lev] * vortMax;
        int n_cells_band = 5; // you can set 3–5 as you like

        amrex::ParallelFor(
            bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              // Tag by vorticity
              vorticity_tagging(i, j, k, tagfab, vort, threshold, tagval);
            });
      }
    }
  }
}

// read in some parameters from inputs file
void AmrCoreLBM::ReadParameters() {
  {
    ParmParse pp; // Traditionally, max_step and stop_time do not have prefix.
    pp.query("max_step", max_step);
    pp.query("stop_time", stop_time);
  }
  amrex::Print() << " max_step = " << max_step << std::endl;
  amrex::Print() << " stop_time = " << stop_time << std::endl;
  {
    ParmParse pp("amr"); // Traditionally, these have prefix, amr.

    pp.query("regrid_int", regrid_int);
    pp.query("plot_file", plot_file);
    pp.query("plot_int", plot_int);
    pp.query("chk_file", chk_file);
    pp.query("chk_int", chk_int);
    pp.query("restart", restart_chkfile);
  }

  {
    if (AMREX_SPACEDIM == 2) {
      ParmParse pp("lbm2d");

      pp.query("ndir", ndir);
      pp.queryarr("wis", wis);
      pp.queryarr("wim", wim);
      pp.queryarr("dirx", dirx);
      pp.queryarr("diry", diry);
      pp.queryarr("dirz", dirz);

      wi.resize(wis.size());
      for (size_t idvs = 0; idvs < wis.size(); idvs++) {
        wi[idvs] = wis[idvs] / wim[idvs];
      }
    }

    if (AMREX_SPACEDIM == 3) {
      ParmParse pp("lbm3d");

      pp.query("ndir", ndir);
      pp.queryarr("wis", wis);
      pp.queryarr("wim", wim);
      pp.queryarr("dirx", dirx);
      pp.queryarr("diry", diry);
      pp.queryarr("dirz", dirz);
      wi.resize(wis.size());
      for (size_t idvs = 0; idvs < wis.size(); idvs++) {
        wi[idvs] = wis[idvs] / wim[idvs];
      }
    }

    {
      ParmParse pp("lbmPhysicalParameters");
      pp.query("nu", nu);
      pp.query("nmac", nmac);
      U0 = 0.1_rt;
      pp.query("U0", U0);
    }
    {
      amrex::ParmParse pp_ibm("ibm");

      // optional toggle (defaults are OK if missing)
      pp_ibm.query("use_cylinder", m_use_cylinder);

      std::string method = "none";
      pp_ibm.query("method", method);
      if (method == "diffuse") {
        m_ib_method = 1;
      } else if (method == "sharp") {
        m_ib_method = 2;
      } else if (method == "marker" || method == "iamr_marker") {
        m_ib_method = 3;
      } else {
        m_ib_method = 0;
      }

      // geometry: tie to your existing cylinder knobs (x0,y0,z0,R)
      pp_ibm.query("eps", m_diff_par.eps);
      // pp_ibm.query("alpha", m_ls_par.alpha);
      pp_ibm.query("alpha", m_diff_par.alpha); // <-- use diffuse params here
      pp_ibm.query("x0", m_ls_par.x0);
      pp_ibm.query("y0", m_ls_par.y0);
      pp_ibm.query("z0", m_ls_par.z0);
      pp_ibm.query("R", m_ls_par.R);

      // IAMReX-style marker DF parameters
      pp_ibm.query("delta_type", m_marker_par.delta_type); // 0:4pt, 1:3pt
      pp_ibm.query("loop_ns", m_marker_par.loop_ns);
      pp_ibm.query("n_marker", m_marker_par.n_marker);
      pp_ibm.query("rd", m_marker_par.rd);
      pp_ibm.query("ubx", m_marker_par.ubx);
      pp_ibm.query("uby", m_marker_par.uby);
      pp_ibm.query("ubz", m_marker_par.ubz);
      pp_ibm.query("omx", m_marker_par.omx);
      pp_ibm.query("omy", m_marker_par.omy);
      pp_ibm.query("omz", m_marker_par.omz);
      pp_ibm.query("verbose", m_marker_par.verbose);
      pp_ibm.query("force_interval", m_force_interval);

      amrex::Print() << "IBM parameters: use_cylinder = " << m_use_cylinder
                     << ", method = " << method << std::endl;
      amrex::Print() << "               x0 = " << m_ls_par.x0
                     << ", y0 = " << m_ls_par.y0 << ", z0 = " << m_ls_par.z0
                     << ", R = " << m_ls_par.R
                     << ", alpha = " << m_diff_par.alpha << std::endl;

      // create managers
      m_ls = std::make_unique<LevelSetManager>();
      m_ibd.reset();
      m_ibs.reset();
      m_ibm.reset();
    }

    int n_phi = (m_use_cylinder ? 1 : 0);
    amrex::Print() << "Number signed distance functions = " << n_phi
                   << std::endl;
    totalNumberVar = nmac + ndir + n_phi;
    amrex::Print() << "Total number of variables = " << totalNumberVar
                   << std::endl;
    wi_dev.resize(ndir);
    dirx_dev.resize(ndir);
    diry_dev.resize(ndir);
    dirz_dev.resize(ndir);

    amrex::Gpu::copy(amrex::Gpu::hostToDevice, wi.begin(), wi.end(),
                     wi_dev.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, dirx.begin(), dirx.end(),
                     dirx_dev.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, diry.begin(), diry.end(),
                     diry_dev.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, dirz.begin(), dirz.end(),
                     dirz_dev.begin());

    const Geometry &geom = Geom(0);
    L0 = geom.ProbHi(0) - geom.ProbLo(0);

    T0 = L0 / U0;
  }
}

// set covered coarse cells to be the average of overlying fine cells
void AmrCoreLBM::AverageDown() {
  for (int lev = finest_level - 1; lev >= 0; --lev) {
    amrex::average_down(f_new[lev + 1], f_new[lev], geom[lev + 1], geom[lev], 0,
                        f_new[lev].nComp(), refRatio(lev));
    amrex::average_down(macro_new[lev + 1], macro_new[lev], geom[lev + 1],
                        geom[lev], 0, macro_new[lev].nComp(), refRatio(lev));
  }
}

void AmrCoreLBM::AverageDownTo(int crse_lev) {
  amrex::average_down(f_new[crse_lev + 1], f_new[crse_lev], geom[crse_lev + 1],
                      geom[crse_lev], 0, f_new[crse_lev].nComp(),
                      refRatio(crse_lev));
  amrex::average_down(macro_new[crse_lev + 1], macro_new[crse_lev],
                      geom[crse_lev + 1], geom[crse_lev], 0,
                      macro_new[crse_lev].nComp(), refRatio(crse_lev));
}

// more flexible version of AverageDown() that lets you average down across
// multiple levels

void AmrCoreLBM::FillPatchMesoscopic(int lev, amrex::Real time,
                                     amrex::MultiFab &mf, int icomp,
                                     int ncomp) {
  if (lev == 0) {
    Vector<MultiFab *> smf;
    Vector<Real> stime;
    GetDataMesoscopic(0, time, smf, stime);

    GpuBndryFuncFab<mesoscopicBcFill> gpu_bndry_func(mesoscopicBcFill{bcval});
    PhysBCFunct<GpuBndryFuncFab<mesoscopicBcFill>> physbc(
        geom[lev], bcsMesoscopic, gpu_bndry_func);

    amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                geom[lev], physbc, 0);
  } else {
    Vector<MultiFab *> cmf, fmf;
    Vector<Real> ctime, ftime;
    GetDataMesoscopic(lev - 1, time, cmf, ctime);
    GetDataMesoscopic(lev, time, fmf, ftime);

    Interpolater *mapper = &cell_cons_interp;

    GpuBndryFuncFab<mesoscopicBcFill> gpu_bndry_func(mesoscopicBcFill{bcval});
    PhysBCFunct<GpuBndryFuncFab<mesoscopicBcFill>> cphysbc(
        geom[lev - 1], bcsMesoscopic, gpu_bndry_func);
    PhysBCFunct<GpuBndryFuncFab<mesoscopicBcFill>> fphysbc(
        geom[lev], bcsMesoscopic, gpu_bndry_func);

    amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime, 0, icomp, ncomp,
                              geom[lev - 1], geom[lev], cphysbc, 0, fphysbc, 0,
                              refRatio(lev - 1), mapper, bcsMesoscopic, 0);
    // AMR stability: f_i are not conserved scalars. Interpolating them across
    // coarse-fine interfaces can introduce nonphysical ghost-cell populations.
    // Here we overwrite only ghost cells with equilibrium reconstructed from a
    // time-consistent macroscopic FillPatch.
    if (mf.nGrow() > 0 && icomp == 0 && ncomp == ndir) {
      MultiFab tmpM(mf.boxArray(), mf.DistributionMap(), nmac, mf.nGrow());
      FillPatchMacro(lev, time, tmpM, 0, nmac);

      auto const *wi_d = wi_dev.data();
      auto const *cx_d = dirx_dev.data();
      auto const *cy_d = diry_dev.data();
      auto const *cz_d = dirz_dev.data();
      const int ndir_l = ndir;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box &vbx = mfi.validbox();
        const Box &gbx = mfi.fabbox();

        const int ilo = vbx.smallEnd(0), ihi = vbx.bigEnd(0);
        const int jlo = vbx.smallEnd(1), jhi = vbx.bigEnd(1);
#if (AMREX_SPACEDIM == 3)
        const int klo = vbx.smallEnd(2), khi = vbx.bigEnd(2);
#else
        const int klo = 0, khi = 0;
#endif

        auto const rho = tmpM[mfi].const_array(0);
        auto const ux = tmpM[mfi].const_array(1);
        auto const uy = tmpM[mfi].const_array(2);
        auto const uz = tmpM[mfi].const_array(3);

        auto const f = mf[mfi].array();

        amrex::ParallelFor(
            gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              const bool in_valid = (i >= ilo && i <= ihi) &&
                                    (j >= jlo && j <= jhi) &&
#if (AMREX_SPACEDIM == 3)
                                    (k >= klo && k <= khi);
#else
              true;
#endif

              if (!in_valid) {
                Real r = amrex::max(rho(i, j, k), 1.e-12_rt);
                Real u0 = ux(i, j, k);
                Real v0 = uy(i, j, k);
                Real w0 = uz(i, j, k);
                Real u2 = u0 * u0 + v0 * v0 + w0 * w0;

                for (int q = 0; q < ndir_l; ++q) {
                  Real cu = cx_d[q] * u0 + cy_d[q] * v0 + cz_d[q] * w0;
                  Real feq =
                      wi_d[q] * r *
                      (1.0_rt + 3.0_rt * cu + 4.5_rt * cu * cu - 1.5_rt * u2);
                  f(i, j, k, q) = feq;
                }
              }
            });
      }
    }
  }
}

void AmrCoreLBM::FillCoarsePatchMesoscopic(int lev, amrex::Real time,
                                           amrex::MultiFab &mf, int icomp,
                                           int ncomp) {
  BL_ASSERT(lev > 0);

  Vector<MultiFab *> cmf;
  Vector<Real> ctime;
  GetDataMesoscopic(lev - 1, time, cmf, ctime);
  Interpolater *mapper = &cell_cons_interp;

  if (cmf.size() != 1) {
    amrex::Abort("FillCoarsePatchMesoscopic: how did this happen?");
  }

  GpuBndryFuncFab<mesoscopicBcFill> gpu_bndry_func(mesoscopicBcFill{bcval});
  PhysBCFunct<GpuBndryFuncFab<mesoscopicBcFill>> cphysbc(
      geom[lev - 1], bcsMesoscopic, gpu_bndry_func);
  PhysBCFunct<GpuBndryFuncFab<mesoscopicBcFill>> fphysbc(
      geom[lev], bcsMesoscopic, gpu_bndry_func);

  amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp,
                               geom[lev - 1], geom[lev], cphysbc, 0, fphysbc, 0,
                               refRatio(lev - 1), mapper, bcsMesoscopic, 0);
}

// utility to copy in data from phi_old and/or phi_new into another multifab
void AmrCoreLBM::GetDataMesoscopic(int lev, amrex::Real time,
                                   amrex::Vector<amrex::MultiFab *> &data,
                                   amrex::Vector<amrex::Real> &datatime) {
  data.clear();
  datatime.clear();

  // Handle uninitialized old-time (you set t_old = time - 1.e200 on new levels)
  // In that case, only "new" data is meaningful.
  if (t_old[lev] < -1.e100_rt) {
    data.push_back(&f_new[lev]);
    datatime.push_back(t_new[lev]);
    return;
  }

  const amrex::Real dtloc = t_new[lev] - t_old[lev];
  const amrex::Real teps =
      amrex::max(1.e-12_rt, amrex::Math::abs(dtloc) * 1.e-3_rt);

  if (time >= t_new[lev] - teps && time <= t_new[lev] + teps) {
    data.push_back(&f_new[lev]);
    datatime.push_back(t_new[lev]);
  } else if (time >= t_old[lev] - teps && time <= t_old[lev] + teps) {
    data.push_back(&f_old[lev]);
    datatime.push_back(t_old[lev]);
  } else {
    data.push_back(&f_old[lev]);
    data.push_back(&f_new[lev]);
    datatime.push_back(t_old[lev]);
    datatime.push_back(t_new[lev]);
  }
}

// Advance a level by dt
// (includes a recursive call for finer levels)
void AmrCoreLBM::timeStepWithSubcycling(int lev, Real time, int iteration) {

  if (regrid_int > 0) // We may need to regrid
  {

    // help keep track of whether a level was already regridded
    // from a coarser level call to regrid
    static Vector<int> last_regrid_step(max_level + 1, 0);

    // regrid changes level "lev+1" so we don't regrid on max_level
    // also make sure we don't regrid fine levels again if
    // it was taken care of during a coarser regrid
    if (lev < max_level && istep[lev] > last_regrid_step[lev]) {
      if (istep[lev] % regrid_int == 0) {
        // regrid could add newly refine levels (if finest_level < max_level)
        // so we save the previous finest level index
        int old_finest = finest_level;
        regrid(lev, time);

        // mark that we have regridded this level already
        for (int k = lev; k <= finest_level; ++k) {
          last_regrid_step[k] = istep[k];
        }

        // if there are newly created levels, set the time step
        for (int k = old_finest + 1; k <= finest_level; ++k) {
          dt[k] = dt[k - 1] / MaxRefRatio(k - 1);
        }
      }
    }
  }

  if (Verbose()) {
    amrex::Print() << "[Level " << lev << " step " << istep[lev] + 1 << "] ";
    amrex::Print() << "ADVANCE with time = " << t_new[lev]
                   << " dt = " << dt[lev] << std::endl;
  }

  // Advance a single level for a single time step, and update flux registers

  t_old[lev] = t_new[lev];
  t_new[lev] += dt[lev];

  AdvancePhiAtLevel(lev, time, dt[lev], iteration, nsubsteps[lev]);

  ++istep[lev];

  if (Verbose()) {
    amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
    amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
  }

  if (lev < finest_level) {
    // recursive call for next-finer level
    for (int i = 1; i <= nsubsteps[lev + 1]; ++i) {
      timeStepWithSubcycling(lev + 1, time + (i - 1) * dt[lev + 1], i);
    }

    AverageDownTo(lev); // average lev+1 down to lev
  }
}

// get plotfile name
std::string AmrCoreLBM::PlotFileName(int lev) const {
  return amrex::Concatenate(plot_file, lev, 5);
}

amrex::Vector<std::unique_ptr<amrex::MultiFab>> AmrCoreLBM::PlotFileMF() const {
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> plot_mfs;

  // Component indices in macro_new
  constexpr int rho_comp = 0;
  constexpr int ux_comp = 1;
  constexpr int uy_comp = 2;
  constexpr int uz_comp = 3;
  constexpr int vor_comp = 4;
  constexpr int P_comp = 5;

  const int n_macro_out = 6; // rho, ux, uy, uz, vor, Pressure
  const int f_first_comp = n_macro_out;

  const bool want_phi = (m_use_cylinder && m_ls);

  for (int lev = 0; lev <= finest_level; ++lev) {
    amrex::Print() << "Preparing plotfile data at level " << lev << "\n";

    AMREX_ALWAYS_ASSERT(macro_new[lev].nComp() > P_comp);
    AMREX_ALWAYS_ASSERT(f_new[lev].nComp() >= ndir);

    // Do we have phi on this level?
    bool have_phi_on_lev = (want_phi && m_ls->has_level(lev));

    int n_phi = have_phi_on_lev ? 1 : 0;
    int ncomp_out = n_macro_out + ndir + n_phi;

    // Allocate output MultiFab with desired layout
    plot_mfs.push_back(
        std::make_unique<amrex::MultiFab>(grids[lev], dmap[lev], ncomp_out, 0));
    amrex::MultiFab &out = *plot_mfs.back();

    // ---- Copy macroscopic fields ----
    amrex::MultiFab::Copy(out, macro_new[lev], rho_comp, rho_comp, 1, 0);
    amrex::MultiFab::Copy(out, macro_new[lev], ux_comp, ux_comp, 1, 0);
    amrex::MultiFab::Copy(out, macro_new[lev], uy_comp, uy_comp, 1, 0);
    amrex::MultiFab::Copy(out, macro_new[lev], uz_comp, uz_comp, 1, 0);
    amrex::MultiFab::Copy(out, macro_new[lev], vor_comp, vor_comp, 1, 0);
    amrex::MultiFab::Copy(out, macro_new[lev], P_comp, P_comp, 1, 0);

    // ---- Copy all ndir distribution functions ----
    for (int i = 0; i < ndir; ++i) {
      amrex::MultiFab::Copy(out, f_new[lev], i, f_first_comp + i, 1, 0);
    }

    // ---- Copy level-set phi, if present, using ParallelCopy ----
    if (have_phi_on_lev) {
      const amrex::MultiFab &phi_src = m_ls->phi_at(lev);
      const int phi_dest_comp = out.nComp() - 1;

      // Scratch MF with the *same* BA & DM as out
      amrex::MultiFab phi_cc(out.boxArray(), out.DistributionMap(), 1, 0);
      phi_cc.setVal(0.0);

      // This handles different BoxArray / DistributionMapping internally
      phi_cc.ParallelCopy(phi_src, 0, 0, 1, 0, 0, Geom(lev).periodicity());

      amrex::MultiFab::Copy(out, phi_cc, 0, phi_dest_comp, 1, 0);
    }

    amrex::Print() << "m_use_cylinder = " << m_use_cylinder
                   << ", m_ls = " << (m_ls ? "not null" : "null")
                   << ", phi_written = " << (have_phi_on_lev ? "yes" : "no")
                   << std::endl;
  }

  return plot_mfs;
}

// set plotfile variable names
Vector<std::string> AmrCoreLBM::PlotFileVarNames() const {
  Vector<std::string> names;

  // Use level 0 as reference (all levels have the same nComp layout)
  const int nmacro = macro_new[0].nComp();
  const int nmeso = f_new[0].nComp();
  const bool have_phi = (m_use_cylinder && m_ls && m_ls->has_level(0));
  const int nphi = have_phi ? 1 : 0;

  // --- Macroscopic variables ---
  if (nmacro > 0)
    names.push_back("rho");
  if (nmacro > 1)
    names.push_back("ux");
  if (nmacro > 2)
    names.push_back("uy");
  if (nmacro > 3)
    names.push_back("uz");
  if (nmacro > 4)
    names.push_back("vor");
  if (nmacro > 5)
    names.push_back("Pressure");

  // Any extra macro components beyond the first 6 get generic names
  for (int m = 6; m < nmacro; ++m) {
    names.push_back("macro_" + std::to_string(m));
  }

  // --- Mesoscopic distribution functions f_i ---
  for (int i = 0; i < nmeso; ++i) {
    names.push_back("f_new_" + std::to_string(i));
  }

  // --- Level-set field φ, if present ---
  if (have_phi) {
    names.push_back("phi");
  }

  const int expected = nmacro + nmeso + nphi;
  if (static_cast<int>(names.size()) != expected) {
    amrex::Print() << "WARNING: PlotFileVarNames: names.size() = "
                   << names.size() << " but expected " << expected << "\n";
  } else {
    amrex::Print() << "PlotFileVarNames: total vars = " << names.size() << "\n";
  }

  return names;
}

// write plotfile to disk
void AmrCoreLBM::WritePlotFile() const {

  // Create MultiFabs for output
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> mf_ptrs = PlotFileMF();

  // Convert unique_ptr to raw pointers for AMReX
  amrex::Vector<const amrex::MultiFab *> mf;
  for (const auto &uptr : mf_ptrs) {
    mf.push_back(uptr.get()); // Extract raw pointer
  }
  const std::string &plotfilename = PlotFileName(istep[0]);
  Vector<std::string> varnames = PlotFileVarNames();

  // Write plot file
  amrex::WriteMultiLevelPlotfile(plotfilename, finest_level + 1, mf, varnames,
                                 Geom(), t_new[0], istep, refRatio());
}

void AmrCoreLBM::WriteCheckpointFile() const {
  /*
      // chk00010            write a checkpoint file with this root directory
      // chk00010/Header     this contains information you need to save (e.g.,
     finest_level, t_new, etc.) and also
      //                     the BoxArrays at each level
      // chk00010/Level_0/
      // chk00010/Level_1/
      // etc.                these subdirectories will hold the MultiFab data at
     each level of refinement

      // checkpoint file name, e.g., chk00010
      const std::string& checkpointname = amrex::Concatenate(chk_file,istep[0]);

      amrex::Print() << "Writing checkpoint " << checkpointname << "\n";

      const int nlevels = finest_level+1;

      // ---- prebuild a hierarchy of directories
      // ---- dirName is built first.  if dirName exists, it is renamed.  then
     build
      // ---- dirName/subDirPrefix_0 .. dirName/subDirPrefix_nlevels-1
      // ---- if callBarrier is true, call ParallelDescriptor::Barrier()
      // ---- after all directories are built
      // ---- ParallelDescriptor::IOProcessor() creates the directories
      amrex::PreBuildDirectorHierarchy(checkpointname, "Level_", nlevels, true);

      // write Header file
     if (ParallelDescriptor::IOProcessor()) {

         std::string HeaderFileName(checkpointname + "/Header");
         VisMF::IO_Buffer io_buffer(VisMF::IO_Buffer_Size);
         std::ofstream HeaderFile;
         HeaderFile.rdbuf()->pubsetbuf(io_buffer.dataPtr(), io_buffer.size());
         HeaderFile.open(HeaderFileName.c_str(), std::ofstream::out   |
                                                 std::ofstream::trunc |
                                                 std::ofstream::binary);
         if( ! HeaderFile.good()) {
             amrex::FileOpenFailed(HeaderFileName);
         }

         HeaderFile.precision(17);

         // write out title line
         HeaderFile << "Checkpoint file for AmrCoreLBM\n";

         // write out finest_level
         HeaderFile << finest_level << "\n";

         // write out array of istep
         for (int i = 0; i < istep.size(); ++i) {
             HeaderFile << istep[i] << " ";
         }
         HeaderFile << "\n";

         // write out array of dt
         for (int i = 0; i < dt.size(); ++i) {
             HeaderFile << dt[i] << " ";
         }
         HeaderFile << "\n";

         // write out array of t_new
         for (int i = 0; i < t_new.size(); ++i) {
             HeaderFile << t_new[i] << " ";
         }
         HeaderFile << "\n";

         // write the BoxArray at each level
         for (int lev = 0; lev <= finest_level; ++lev) {
             boxArray(lev).writeOn(HeaderFile);
             HeaderFile << '\n';
         }
     }

     // write the MultiFab data to, e.g., chk00010/Level_0/
     for (int lev = 0; lev <= finest_level; ++lev) {
         VisMF::Write(macro_new[lev],amrex::MultiFabFileFullPrefix(lev,
     checkpointname, "Level_", "phi"));
     }
  */
}

namespace {
// utility to skip to next line in Header
void GotoNextLine(std::istream &is) {
  constexpr std::streamsize bl_ignore_max{100000};
  is.ignore(bl_ignore_max, '\n');
}
} // namespace

void AmrCoreLBM::ReadCheckpointFile() {
  /*
      amrex::Print() << "Restart from checkpoint " << restart_chkfile << "\n";

      // Header
      std::string File(restart_chkfile + "/Header");

      VisMF::IO_Buffer io_buffer(VisMF::GetIOBufferSize());

      Vector<char> fileCharPtr;
      ParallelDescriptor::ReadAndBcastFile(File, fileCharPtr);
      std::string fileCharPtrString(fileCharPtr.dataPtr());
      std::istringstream is(fileCharPtrString, std::istringstream::in);

      std::string line, word;

      // read in title line
      std::getline(is, line);

      // read in finest_level
      is >> finest_level;
      GotoNextLine(is);

      // read in array of istep
      std::getline(is, line);
      {
          std::istringstream lis(line);
          int i = 0;
          while (lis >> word) {
              istep[i++] = std::stoi(word);
          }
      }

      // read in array of dt
      std::getline(is, line);
      {
          std::istringstream lis(line);
          int i = 0;
          while (lis >> word) {
              dt[i++] = std::stod(word);
          }
      }

      // read in array of t_new
      std::getline(is, line);
      {
          std::istringstream lis(line);
          int i = 0;
          while (lis >> word) {
              t_new[i++] = std::stod(word);
          }
      }

      for (int lev = 0; lev <= finest_level; ++lev) {

          // read in level 'lev' BoxArray from Header
          BoxArray ba;
          ba.readFrom(is);
          GotoNextLine(is);

          // create a distribution mapping
          DistributionMapping dm { ba, ParallelDescriptor::NProcs() };

          // set BoxArray grids and DistributionMapping dmap in AMReX_AmrMesh.H
     class SetBoxArray(lev, ba); SetDistributionMap(lev, dm);

          // build MultiFab and FluxRegister data
          int ncomp = 1;
          int nghost = 0;
          phi_old[lev].define(grids[lev], dmap[lev], ncomp, nghost);
          phi_new[lev].define(grids[lev], dmap[lev], ncomp, nghost);

          if (lev > 0 && do_reflux) {
              flux_reg[lev].reset(new FluxRegister(grids[lev], dmap[lev],
     refRatio(lev-1), lev, ncomp));
          }

          // build face velocity MultiFabs
          for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
          {
              facevel[lev][idim] =
     MultiFab(amrex::convert(ba,IntVect::TheDimensionVector(idim)), dm, 1, 1);
          }
      }

      // read in the MultiFab data
      for (int lev = 0; lev <= finest_level; ++lev) {
          VisMF::Read(phi_new[lev],
                      amrex::MultiFabFileFullPrefix(lev, restart_chkfile,
     "Level_", "phi"));
      }

  */
}

void AmrCoreLBM::InitEquilibrium() {

  // Device-accessible stencil data
  auto const *wi_d = wi_dev.data();
  auto const *cx_d = dirx_dev.data();
  auto const *cy_d = diry_dev.data();
  auto const *cz_d = dirz_dev.data();
  const int ndir_l = ndir;

  for (int lev = finest_level; lev >= 0; --lev) {
    MultiFab &curF = f_new[lev];
    MultiFab &curFo = f_old[lev];
    MultiFab &curMacro = macro_new[lev];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(curF, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

      const Box &bx = mfi.fabbox();

      auto const rho = curMacro[mfi].const_array(0);
      auto const ux = curMacro[mfi].const_array(1);
      auto const uy = curMacro[mfi].const_array(2);
      auto const uz = curMacro[mfi].const_array(3);

      auto const f = curF[mfi].array();
      auto const fo = curFo[mfi].array();

      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j,
                                                  int k) noexcept {
        Real r = amrex::max(rho(i, j, k), 1.e-12_rt);
        Real u0 = ux(i, j, k);
        Real v0 = uy(i, j, k);
        Real w0 = uz(i, j, k);
        Real u2 = u0 * u0 + v0 * v0 + w0 * w0;

        for (int q = 0; q < ndir_l; ++q) {
          Real cu = cx_d[q] * u0 + cy_d[q] * v0 + cz_d[q] * w0;
          Real feq = wi_d[q] * r *
                     (1.0_rt + 3.0_rt * cu + 4.5_rt * cu * cu - 1.5_rt * u2);
          f(i, j, k, q) = feq;
          fo(i, j, k, q) = feq;
        }
      });
    }
  }
}

void AmrCoreLBM::FillPatchMacro(int lev, amrex::Real time, amrex::MultiFab &mf,
                                int icomp, int ncomp) {

  if (lev == 0) {
    Vector<MultiFab *> smf;
    Vector<Real> stime;
    GetDataMacro(0, time, smf, stime);

    GpuBndryFuncFab<macroBcFill> gpu_bndry_func(macroBcFill{bcval});
    PhysBCFunct<GpuBndryFuncFab<macroBcFill>> physbc(geom[lev], bcsMacro,
                                                     gpu_bndry_func);

    amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                geom[lev], physbc, 0);
  } else {
    Vector<MultiFab *> cmf, fmf;
    Vector<Real> ctime, ftime;
    GetDataMacro(lev - 1, time, cmf, ctime);
    GetDataMacro(lev, time, fmf, ftime);

    Interpolater *mapper = &cell_cons_interp;

    GpuBndryFuncFab<macroBcFill> gpu_bndry_func(macroBcFill{bcval});
    PhysBCFunct<GpuBndryFuncFab<macroBcFill>> cphysbc(geom[lev - 1], bcsMacro,
                                                      gpu_bndry_func);
    PhysBCFunct<GpuBndryFuncFab<macroBcFill>> fphysbc(geom[lev], bcsMacro,
                                                      gpu_bndry_func);

    amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime, 0, icomp, ncomp,
                              geom[lev - 1], geom[lev], cphysbc, 0, fphysbc, 0,
                              refRatio(lev - 1), mapper, bcsMacro, 0);
  }
}

void AmrCoreLBM::FillPatchForcing(int lev, amrex::Real time,
                                  amrex::MultiFab &mf, int icomp, int ncomp) {
  // Just copy from current forcing[lev]; no time interpolation
  amrex::Vector<amrex::MultiFab *> smf(1, &forcing[lev]);
  amrex::Vector<amrex::Real> stime(1, t_new[lev]);

  GpuBndryFuncFab<macroBcFill> gpu_bndry_func(macroBcFill{bcval});
  PhysBCFunct<GpuBndryFuncFab<macroBcFill>> physbc(geom[lev], bcsMacro,
                                                   gpu_bndry_func);

  amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp, geom[lev],
                              physbc, 0);
}

void AmrCoreLBM::FillCoarsePatchMacro(int lev, amrex::Real time,
                                      amrex::MultiFab &mf, int icomp,
                                      int ncomp) {
  BL_ASSERT(lev > 0);

  Vector<MultiFab *> cmf;
  Vector<Real> ctime;
  GetDataMacro(lev - 1, time, cmf, ctime);
  Interpolater *mapper = &cell_cons_interp;

  if (cmf.size() != 1) {
    amrex::Abort("FillCoarsePatchMacro: how did this happen?");
  }

  GpuBndryFuncFab<macroBcFill> gpu_bndry_func(macroBcFill{bcval});
  PhysBCFunct<GpuBndryFuncFab<macroBcFill>> cphysbc(geom[lev - 1], bcsMacro,
                                                    gpu_bndry_func);
  PhysBCFunct<GpuBndryFuncFab<macroBcFill>> fphysbc(geom[lev], bcsMacro,
                                                    gpu_bndry_func);

  amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp,
                               geom[lev - 1], geom[lev], cphysbc, 0, fphysbc, 0,
                               refRatio(lev - 1), mapper, bcsMacro, 0);
}

void AmrCoreLBM::GetDataMacro(int lev, Real time, Vector<MultiFab *> &data,
                              Vector<Real> &datatime) {
  data.clear();
  datatime.clear();

  // Handle uninitialized/sentinel time (e.g. t_old = time - 1.e200 on new
  // levels). In that case, only the "new" state is meaningful.
  constexpr Real SENT = 1.e50_rt;
  if (amrex::Math::abs(t_old[lev]) > SENT ||
      amrex::Math::abs(t_new[lev]) > SENT) {
    data.push_back(&macro_new[lev]);
    datatime.push_back(t_new[lev]);
    return;
  }

  const Real dtloc = amrex::Math::abs(t_new[lev] - t_old[lev]);
  const Real teps = amrex::max(1.e-12_rt, dtloc * 1.e-3_rt);

  if (time >= t_new[lev] - teps && time <= t_new[lev] + teps) {
    data.push_back(&macro_new[lev]);
    datatime.push_back(t_new[lev]);
  } else if (time >= t_old[lev] - teps && time <= t_old[lev] + teps) {
    data.push_back(&macro_old[lev]);
    datatime.push_back(t_old[lev]);
  } else {
    data.push_back(&macro_old[lev]);
    data.push_back(&macro_new[lev]);
    datatime.push_back(t_old[lev]);
    datatime.push_back(t_new[lev]);
  }
}

void AmrCoreLBM::BuildLevelSetOnLevel(int lev) {
  if (!m_use_cylinder || !m_ls)
    return;

  // Use the *macro* MultiFab as the reference for BA/DM
  const BoxArray &ba = macro_new[lev].boxArray();
  const DistributionMapping &dm = macro_new[lev].DistributionMap();
  amrex::Print() << "BuildLevelSetOnLevel: level " << lev
                 << " (nghost = " << nghost << ")\n";
  // (Re)define storage for this level and rebuild φ
  m_ls->define_level(lev, ba, dm, /*ng=*/nghost);
  m_ls->build_from_cylinder(lev, geom[lev], m_ls_par);

  amrex::Print() << "BuildLevelSetOnLevel: level " << lev
                 << " (|BA| = " << ba.size() << ", R = " << m_ls_par.R << ")\n";
}
