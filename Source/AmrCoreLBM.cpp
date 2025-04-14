
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

#include <AMReX_GpuContainers.H>

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

  int bc_lo[AMREX_SPACEDIM];
  int bc_hi[AMREX_SPACEDIM];

  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {

    if (Geom(0).isPeriodic()[idim] == 1) {
      bc_lo[idim] = bc_hi[idim] = BCType::int_dir;
    } else {
      bc_lo[idim] = bc_hi[idim] = BCType::foextrap;
    }
  }

  bcsMesoscopic.resize(ndir); // Setup 1-component
  for (int idir = 0; idir < ndir; ++idir) {

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      // lo-side BCs
      if (bc_lo[idim] ==
              BCType::int_dir || // periodic uses "internal Dirichlet"
          bc_lo[idim] == BCType::foextrap || // first-order extrapolation
          bc_lo[idim] == BCType::ext_dir) {  // external Dirichlet
        bcsMesoscopic[idir].setLo(idim, bc_lo[idim]);
      } else {
        amrex::Abort("Invalid bc_lo");
      }

      // hi-side BCSs
      if (bc_hi[idim] ==
              BCType::int_dir || // periodic uses "internal Dirichlet"
          bc_hi[idim] == BCType::foextrap || // first-order extrapolation
          bc_hi[idim] == BCType::ext_dir) {  // external Dirichlet
        bcsMesoscopic[idir].setHi(idim, bc_hi[idim]);
      } else {
        amrex::Abort("Invalid bc_hi");
      }
    }
  }

  bcsMacro.resize(nmac); // Setup 1-component
  for (int idir = 0; idir < nmac; ++idir) {

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      // lo-side BCs
      if (bc_lo[idim] ==
              BCType::int_dir || // periodic uses "internal Dirichlet"
          bc_lo[idim] == BCType::foextrap || // first-order extrapolation
          bc_lo[idim] == BCType::ext_dir) {  // external Dirichlet
        bcsMacro[idir].setLo(idim, bc_lo[idim]);
      } else {
        amrex::Abort("Invalid bc_lo");
      }

      // hi-side BCSs
      if (bc_hi[idim] ==
              BCType::int_dir || // periodic uses "internal Dirichlet"
          bc_hi[idim] == BCType::foextrap || // first-order extrapolation
          bc_hi[idim] == BCType::ext_dir) {  // external Dirichlet
        bcsMacro[idir].setHi(idim, bc_hi[idim]);
      } else {
        amrex::Abort("Invalid bc_hi");
      }
    }
  }
}

AmrCoreLBM::~AmrCoreLBM() {}

// advance solution to final time
void AmrCoreLBM::Evolve() {

  Real cur_time = t_new[0];
  int last_plot_file_step = 0;

  for (int step = istep[0]; step < max_step && cur_time < stop_time; ++step) {
    amrex::Print() << "\nCoarse STEP " << step + 1 << " starts ..."
                   << std::endl;

    int lev = 0;
    int iteration = 1;

    timeStepWithSubcycling(lev, cur_time, iteration);

    cur_time += dt[0];

    // sum phi to check conservation
    Real sum_phi = macro_new[0].sum(0);

    amrex::Print() << "Coarse STEP " << step + 1 << " ends."
                   << " TIME = " << cur_time << " DT = " << dt[0]
                   << " Sum(Phi) = " << sum_phi << std::endl;

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
  }

  if (plot_int > 0 && istep[0] > last_plot_file_step) {
    WritePlotFile();
  }
}

// initializes multilevel data
void AmrCoreLBM::InitData() {
  if (restart_chkfile == "") {
    // start simulation from the beginning
    const Real time = 0.0;
    InitFromScratch(time);

    AverageDown();

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

  // This part requires special attention, both macroscopic - mesoscopic need to
  // be initialized
  FillCoarsePatchMesoscopic(lev, time, f_new[lev], 0, f_ncomp);
  MultiFab::Copy(f_old[lev], f_new[lev], 0, 0, f_ncomp, f_nghost);

  FillCoarsePatchMacro(lev, time, macro_new[lev], 0, m_ncomp);
  MultiFab::Copy(macro_old[lev], macro_new[lev], 0, 0, m_ncomp, m_nghost);
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

  FillPatchMesoscopic(lev, time, newF_state, 0, f_ncomp);
  FillPatchMacro(lev, time, newM_state, 0, m_ncomp);

  MultiFab::Copy(oldF_state, newF_state, 0, 0, f_ncomp, f_nghost);
  MultiFab::Copy(oldM_state, newM_state, 0, 0, m_ncomp, m_nghost);

  std::swap(newF_state, f_new[lev]);
  std::swap(oldF_state, f_old[lev]);
  std::swap(newM_state, macro_new[lev]);
  std::swap(oldM_state, macro_old[lev]);
}

// Delete level data
// overrides the pure virtual function in AmrCore
void AmrCoreLBM::ClearLevel(int lev) {
  f_new[lev].clear();
  f_old[lev].clear();
  macro_new[lev].clear();
  macro_old[lev].clear();
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

    amrex::launch(box, [=] AMREX_GPU_DEVICE(Box const &tbx) {
      initdata(tbx, rho, u, v, w, vor, problo, probhi, dx, nu);
    });
  }

  MultiFab &curMacro = macro_new[lev];
  amrex::Real tempdx = xCellSize[lev];
  amrex::Real tempdy = yCellSize[lev];
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
    amrex::ParallelFor(
        gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          if (vbx.contains(i, j, k)) {
            visPara(i, j, k, rho, u, v, vor, P, tempdx, tempdy, T0);
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
    ParmParse pp("adv");
    int n = pp.countval("thresholdRatio");
    if (n > 0) {
      pp.getarr("thresholdRatio", thresholdRatio, 0, n);
    }
  }

  amrex::Real vortMax = macro_new[lev].max(4);

  MultiFab &curMacro = macro_new[lev];

  //    const int clearval = TagBox::CLEAR;
  const int tagval = TagBox::SET;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  {

    for (MFIter mfi(curMacro, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.validbox();
      Array4<const Real> statefab = curMacro[mfi].array(4);
      auto const &tagfab = tags.array(mfi);

      Real threshold = thresholdRatio[lev] * vortMax;

      amrex::ParallelFor(
          bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            state_error(i, j, k, tagfab, statefab, threshold, tagval);
          });
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
    }

    totalNumberVar = ndir + nmac;

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

    // The characteristic velocity is hard-coded when it is initialized, so I
    // will define it here directly.
    U0 = 0.02;
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

// more flexible version of AverageDown() that lets you average down across
// multiple levels
void AmrCoreLBM::AverageDownTo(int crse_lev) {
  amrex::average_down(f_new[crse_lev + 1], f_new[crse_lev], geom[crse_lev + 1],
                      geom[crse_lev], 0, f_new[crse_lev].nComp(),
                      refRatio(crse_lev));
  amrex::average_down(macro_new[crse_lev + 1], macro_new[crse_lev],
                      geom[crse_lev + 1], geom[crse_lev], 0,
                      macro_new[crse_lev].nComp(), refRatio(crse_lev));
}

// compute a new multifab by coping in phi from valid region and filling ghost
// cells works for single level and 2-level cases (fill fine grid ghost by
// interpolating from coarse)
void AmrCoreLBM::FillPatchMesoscopic(int lev, Real time, MultiFab &mf,
                                     int icomp, int ncomp) {
  if (lev == 0) {
    Vector<MultiFab *> smf;
    Vector<Real> stime;
    GetDataMesoscopic(0, time, smf, stime);

    if (Gpu::inLaunchRegion()) {
      GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
      PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> physbc(geom[lev], bcsMesoscopic,
                                                       gpu_bndry_func);
      amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                  geom[lev], physbc, 0);
    } else {
      CpuBndryFuncFab bndry_func(
          nullptr); // Without EXT_DIR, we can pass a nullptr.
      PhysBCFunct<CpuBndryFuncFab> physbc(geom[lev], bcsMesoscopic, bndry_func);
      amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                  geom[lev], physbc, 0);
    }
  } else {
    Vector<MultiFab *> cmf, fmf;
    Vector<Real> ctime, ftime;
    GetDataMesoscopic(lev - 1, time, cmf, ctime);
    GetDataMesoscopic(lev, time, fmf, ftime);

    Interpolater *mapper = &cell_cons_interp;

    if (Gpu::inLaunchRegion()) {
      GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
      PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> cphysbc(
          geom[lev - 1], bcsMesoscopic, gpu_bndry_func);
      PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> fphysbc(
          geom[lev], bcsMesoscopic, gpu_bndry_func);

      amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime, 0, icomp,
                                ncomp, geom[lev - 1], geom[lev], cphysbc, 0,
                                fphysbc, 0, refRatio(lev - 1), mapper,
                                bcsMesoscopic, 0);
    } else {
      CpuBndryFuncFab bndry_func(
          nullptr); // Without EXT_DIR, we can pass a nullptr.
      PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev - 1], bcsMesoscopic,
                                           bndry_func);
      PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev], bcsMesoscopic,
                                           bndry_func);

      amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime, 0, icomp,
                                ncomp, geom[lev - 1], geom[lev], cphysbc, 0,
                                fphysbc, 0, refRatio(lev - 1), mapper,
                                bcsMesoscopic, 0);
    }
  }
}

// fill an entire multifab by interpolating from the coarser level
// this comes into play when a new level of refinement appears
void AmrCoreLBM::FillCoarsePatchMesoscopic(int lev, Real time, MultiFab &mf,
                                           int icomp, int ncomp) {
  BL_ASSERT(lev > 0);

  Vector<MultiFab *> cmf;
  Vector<Real> ctime;
  GetDataMesoscopic(lev - 1, time, cmf, ctime);
  Interpolater *mapper = &cell_cons_interp;

  if (cmf.size() != 1) {
    amrex::Abort("FillCoarsePatchMesoscopic: how did this happen?");
  }

  if (Gpu::inLaunchRegion()) {
    GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
    PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> cphysbc(
        geom[lev - 1], bcsMesoscopic, gpu_bndry_func);
    PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> fphysbc(geom[lev], bcsMesoscopic,
                                                      gpu_bndry_func);

    amrex::InterpFromCoarseLevel(
        mf, time, *cmf[0], 0, icomp, ncomp, geom[lev - 1], geom[lev], cphysbc,
        0, fphysbc, 0, refRatio(lev - 1), mapper, bcsMesoscopic, 0);
  } else {
    CpuBndryFuncFab bndry_func(
        nullptr); // Without EXT_DIR, we can pass a nullptr.
    PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev - 1], bcsMesoscopic,
                                         bndry_func);
    PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev], bcsMesoscopic, bndry_func);

    amrex::InterpFromCoarseLevel(
        mf, time, *cmf[0], 0, icomp, ncomp, geom[lev - 1], geom[lev], cphysbc,
        0, fphysbc, 0, refRatio(lev - 1), mapper, bcsMesoscopic, 0);
  }
}

// utility to copy in data from phi_old and/or phi_new into another multifab
void AmrCoreLBM::GetDataMesoscopic(int lev, Real time, Vector<MultiFab *> &data,
                                   Vector<Real> &datatime) {

  data.clear();
  datatime.clear();

  const Real teps = (t_new[lev] - t_old[lev]) * 1.e-3;

  if (time > t_new[lev] - teps && time < t_new[lev] + teps) {
    data.push_back(&f_new[lev]);
    datatime.push_back(t_new[lev]);
  } else if (time > t_old[lev] - teps && time < t_old[lev] + teps) {
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

// put together an array of multifabs for writing
amrex::Vector<std::unique_ptr<amrex::MultiFab>> AmrCoreLBM::PlotFileMF() const {
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> plot_mfs;

  constexpr int rho_comp = 0;
  constexpr int ux_comp = 1;
  constexpr int uy_comp = 2;
  constexpr int uz_comp = 3;
  constexpr int vor_comp = 4;
  constexpr int P_comp = 4;
  constexpr int f_comp = 5;

  for (int lev = 0; lev <= finest_level; ++lev) {

    AMREX_ALWAYS_ASSERT(macro_new[lev].nComp() >= uz_comp);
    AMREX_ALWAYS_ASSERT(f_new[lev].nComp() >= ndir);
    // Allocate MultiFab with 12 components (rho, ux, uy, and f_old[0-8])
    plot_mfs.push_back(std::make_unique<amrex::MultiFab>(grids[lev], dmap[lev],
                                                         totalNumberVar, 0));

    // Copy rho
    amrex::MultiFab::Copy(*plot_mfs[lev], macro_new[lev], rho_comp, rho_comp, 1,
                          0);

    // Copy ux
    amrex::MultiFab::Copy(*plot_mfs[lev], macro_new[lev], ux_comp, ux_comp, 1,
                          0);

    // Copy uy
    amrex::MultiFab::Copy(*plot_mfs[lev], macro_new[lev], uy_comp, uy_comp, 1,
                          0);

    // Copy uz
    amrex::MultiFab::Copy(*plot_mfs[lev], macro_new[lev], uz_comp, uz_comp, 1,
                          0);

    // Copy vor
    amrex::MultiFab::Copy(*plot_mfs[lev], macro_new[lev], vor_comp, vor_comp, 1,
                          0);

    // Copy pressure
    amrex::MultiFab::Copy(*plot_mfs[lev], macro_new[lev], P_comp, P_comp, 1, 0);

    // Copy all 9 components of f_old
    for (int i = 0; i < ndir; ++i) {
      amrex::MultiFab::Copy(*plot_mfs[lev], f_new[lev], i, f_comp + i, 1, 0);
    }
  }

  return plot_mfs;
}

// set plotfile variable names
Vector<std::string> AmrCoreLBM::PlotFileVarNames() const {
  Vector<std::string> names;
  names.push_back("rho");
  names.push_back("ux");
  names.push_back("uy");
  names.push_back("uz");
  names.push_back("vor");
  names.push_back("Pressure");

  for (int i = 0; i < ndir; ++i) {
    names.push_back("f_new_" + std::to_string(i));
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

  for (int lev = finest_level; lev >= 0; --lev) {
    MultiFab &curF = f_new[lev];
    MultiFab &curFo = f_old[lev];
    MultiFab &curMacro = macro_new[lev];

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(curF, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

      const Box &bx = mfi.fabbox();
      Array4<Real> rho = curMacro[mfi].array(0);
      Array4<Real> u = curMacro[mfi].array(1);
      Array4<Real> v = curMacro[mfi].array(2);
      Array4<Real> w = curMacro[mfi].array(3);
      Array4<Real> f = curF[mfi].array();
      Array4<Real> fo = curFo[mfi].array();

      amrex::ParallelFor(bx,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                           amrex::Vector<amrex::Real> tempMac(nmac);
                           tempMac[0] = rho(i, j, k);
                           tempMac[1] = u(i, j, k);
                           tempMac[2] = v(i, j, k);
                           tempMac[3] = w(i, j, k);

                           for (unsigned int i_dir = 0; i_dir < ndir; ++i_dir) {

                             amrex::Vector<amrex::Real> tempMes(4);
                             tempMes[0] = wi[i_dir];
                             tempMes[1] = dirx[i_dir];
                             tempMes[2] = diry[i_dir];
                             tempMes[3] = dirz[i_dir];

                             f(i, j, k, i_dir) = feqFunction(tempMes, tempMac);
                             fo(i, j, k, i_dir) = f(i, j, k, i_dir);
                           }
                         });
    }
  }
}

void AmrCoreLBM::FillPatchMacro(int lev, Real time, MultiFab &mf, int icomp,
                                int ncomp) {
  if (lev == 0) {
    Vector<MultiFab *> smf;
    Vector<Real> stime;
    GetDataMacro(0, time, smf, stime);

    if (Gpu::inLaunchRegion()) {
      GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
      PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> physbc(geom[lev], bcsMacro,
                                                       gpu_bndry_func);
      amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                  geom[lev], physbc, 0);
    } else {
      CpuBndryFuncFab bndry_func(
          nullptr); // Without EXT_DIR, we can pass a nullptr.
      PhysBCFunct<CpuBndryFuncFab> physbc(geom[lev], bcsMacro, bndry_func);
      amrex::FillPatchSingleLevel(mf, time, smf, stime, 0, icomp, ncomp,
                                  geom[lev], physbc, 0);
    }
  } else {
    Vector<MultiFab *> cmf, fmf;
    Vector<Real> ctime, ftime;
    GetDataMacro(lev - 1, time, cmf, ctime);
    GetDataMacro(lev, time, fmf, ftime);

    Interpolater *mapper = &cell_cons_interp;

    if (Gpu::inLaunchRegion()) {
      GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
      PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> cphysbc(geom[lev - 1], bcsMacro,
                                                        gpu_bndry_func);
      PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> fphysbc(geom[lev], bcsMacro,
                                                        gpu_bndry_func);

      amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime, 0, icomp,
                                ncomp, geom[lev - 1], geom[lev], cphysbc, 0,
                                fphysbc, 0, refRatio(lev - 1), mapper, bcsMacro,
                                0);
    } else {
      CpuBndryFuncFab bndry_func(
          nullptr); // Without EXT_DIR, we can pass a nullptr.
      PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev - 1], bcsMacro, bndry_func);
      PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev], bcsMacro, bndry_func);

      amrex::FillPatchTwoLevels(mf, time, cmf, ctime, fmf, ftime, 0, icomp,
                                ncomp, geom[lev - 1], geom[lev], cphysbc, 0,
                                fphysbc, 0, refRatio(lev - 1), mapper, bcsMacro,
                                0);
    }
  }
}

void AmrCoreLBM::FillCoarsePatchMacro(int lev, Real time, MultiFab &mf,
                                      int icomp, int ncomp) {
  BL_ASSERT(lev > 0);

  Vector<MultiFab *> cmf;
  Vector<Real> ctime;
  GetDataMacro(lev - 1, time, cmf, ctime);
  Interpolater *mapper = &cell_cons_interp;

  if (cmf.size() != 1) {
    amrex::Abort("FillCoarsePatchMacro: how did this happen?");
  }

  if (Gpu::inLaunchRegion()) {
    GpuBndryFuncFab<AmrCoreFill> gpu_bndry_func(AmrCoreFill{});
    PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> cphysbc(geom[lev - 1], bcsMacro,
                                                      gpu_bndry_func);
    PhysBCFunct<GpuBndryFuncFab<AmrCoreFill>> fphysbc(geom[lev], bcsMacro,
                                                      gpu_bndry_func);

    amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp,
                                 geom[lev - 1], geom[lev], cphysbc, 0, fphysbc,
                                 0, refRatio(lev - 1), mapper, bcsMacro, 0);
  } else {
    CpuBndryFuncFab bndry_func(
        nullptr); // Without EXT_DIR, we can pass a nullptr.
    PhysBCFunct<CpuBndryFuncFab> cphysbc(geom[lev - 1], bcsMacro, bndry_func);
    PhysBCFunct<CpuBndryFuncFab> fphysbc(geom[lev], bcsMacro, bndry_func);

    amrex::InterpFromCoarseLevel(mf, time, *cmf[0], 0, icomp, ncomp,
                                 geom[lev - 1], geom[lev], cphysbc, 0, fphysbc,
                                 0, refRatio(lev - 1), mapper, bcsMacro, 0);
  }
}

void AmrCoreLBM::GetDataMacro(int lev, Real time, Vector<MultiFab *> &data,
                              Vector<Real> &datatime) {

  data.clear();
  datatime.clear();

  const Real teps = (t_new[lev] - t_old[lev]) * 1.e-3;

  if (time > t_new[lev] - teps && time < t_new[lev] + teps) {
    data.push_back(&macro_old[lev]);
    datatime.push_back(t_new[lev]);
  } else if (time > t_old[lev] - teps && time < t_old[lev] + teps) {
    data.push_back(&macro_old[lev]);
    datatime.push_back(t_old[lev]);
  } else {
    data.push_back(&macro_old[lev]);
    data.push_back(&macro_old[lev]);
    datatime.push_back(t_old[lev]);
    datatime.push_back(t_new[lev]);
  }
}
