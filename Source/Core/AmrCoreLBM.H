#ifndef AmrCoreAdv_H_
#define AmrCoreAdv_H_

#include <limits>
#include <memory>
#include <string>

#ifdef AMREX_USE_OMP
#include <omp.h>
#endif

#include "BC/bc_val.H"
#include <AMReX_AmrCore.H>
#include <AMReX_BCRec.H>
#include <AMReX_FillPatcher.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_TableData.H>
#include <memory> // for std::unique_ptr

#include "IBM/IBDiffuseLS.H"   // DiffuseIBParams, IBDiffuseLS
#include "IBM/IBSharpLS.H"     // SharpIBParams,   IBSharpLS
#include "LevelSet/LevelSet.H" // LevelSetParams, LevelSetManager

// ---- Forward decls; full definitions live in the .cpps
class LevelSetManager;
class IBDiffuseLS;
class IBSharpLS;

class AmrCoreLBM : public amrex::AmrCore {
public:
  ////////////////
  // public member functions

  // constructor - reads in parameters from inputs file
  //             - sizes multilevel arrays and data structures
  AmrCoreLBM();
  virtual ~AmrCoreLBM();

  // advance solution to final time
  void Evolve();

  // integrated IBM force on the immersed body 
  void ComputeIBForce(amrex::Real time, int step) const;

  // initializes multilevel data
  void InitData();

  // Make a new level using provided BoxArray and DistributionMapping and
  // fill with interpolated coarse level data.
  // overrides the pure virtual function in AmrCore
  virtual void
  MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray &ba,
                         const amrex::DistributionMapping &dm) override;

  // Remake an existing level using provided BoxArray and DistributionMapping
  // and fill with existing fine and coarse data. overrides the pure virtual
  // function in AmrCore
  virtual void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray &ba,
                           const amrex::DistributionMapping &dm) override;

  // Delete level data
  // overrides the pure virtual function in AmrCore
  virtual void ClearLevel(int lev) override;

  // Make a new level from scratch using provided BoxArray and
  // DistributionMapping. Only used during initialization. overrides the pure
  // virtual function in AmrCore
  virtual void
  MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray &ba,
                          const amrex::DistributionMapping &dm) override;

  // tag all cells for refinement
  // overrides the pure virtual function in AmrCore
  virtual void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time,
                        int ngrow) override;

  // Advance phi at a single level for a single time step, update flux registers
  void AdvancePhiAtLevel(int lev, amrex::Real time, amrex::Real dt_lev,
                         int iteration, int ncycle);

  // Define the advection velocity as the curl of a scalar field
  void DefineVelocityAtLevel(int lev, amrex::Real time);

  void DefineVelocityAllLevels(amrex::Real time);

  // compute dt from CFL considerations
  amrex::Real EstTimeStep(int lev, amrex::Real time);
  // --- Level-set & IBM parameter bundles (MUST be complete types here) ---

private:
  ////////////////
  // private member functions

  // read in some parameters from inputs file
  void ReadParameters();

  // set covered coarse cells to be the average of overlying fine cells
  void AverageDown();

  // void AverageDownSingle ();
  // more flexible version of AverageDown() that lets you average down across
  // multiple levels
  void AverageDownTo(int crse_lev);

  // compute a new multifab by coping in phi from valid region and filling ghost
  // cells works for single level and 2-level cases (fill fine grid ghost by
  // interpolating from coarse)
  void FillPatchMesoscopic(int lev, amrex::Real time, amrex::MultiFab &mf,
                           int icomp, int ncomp);

  // fill an entire multifab by interpolating from the coarser level
  // this comes into play when a new level of refinement appears
  void FillCoarsePatchMesoscopic(int lev, amrex::Real time, amrex::MultiFab &mf,
                                 int icomp, int ncomp);

  // utility to copy in data from phi_old and/or phi_new into another multifab
  // void GetDataMesoscopic (int lev, amrex::Real time,
  // amrex::Vector<amrex::MultiFab*>& data,amrex::Vector<amrex::Real>&
  // datatime);
  // void GetDataMacro(int lev, amrex::Real time,
  // amrex::Vector<amrex::MultiFab*>& data,amrex::Vector<amrex::Real>&
  // datatime);

  void GetDataMesoscopic(int lev, amrex::Real time,
                         amrex::Vector<amrex::MultiFab *> &data,
                         amrex::Vector<amrex::Real> &datatime);
  // Advance a level by dt - includes a recursive call for finer levels
  void timeStepWithSubcycling(int lev, amrex::Real time, int iteration);

  // Advance all levels by the same dt
  void timeStepNoSubcycling(amrex::Real time, int iteration);

  // a wrapper for EstTimeStep
  void ComputeDt();

  // get plotfile name
  std::string PlotFileName(int lev) const;

  // put together an array of multifabs for writing
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> PlotFileMF() const;

  // set plotfile variables names
  amrex::Vector<std::string> PlotFileVarNames() const;

  // write plotfile to disk
  void WritePlotFile() const;

  // write checkpoint file to disk
  void WriteCheckpointFile() const;

  // read checkpoint file from disk
  void ReadCheckpointFile();

  void check_ghost_cells_nonzero(const amrex::MultiFab &mf, int comp, int lev);
  void check_ghost_cells_nonzero2(const amrex::MultiFab &mf, int comp, int lev);

  void BuildLevelSetOnLevel(int lev);
  ////////////////
  // private data members

  amrex::Vector<int> istep;     // which step?
  amrex::Vector<int> nsubsteps; // how many substeps on each level?

  // keep track of old time, new time, and time step at each level
  amrex::Vector<amrex::Real> t_new;
  amrex::Vector<amrex::Real> t_old;
  amrex::Vector<amrex::Real> dt;
  amrex::Vector<amrex::Real> tau;
  // array of multifabs to store the solution at each level of refinement
  // after advancing a level we use "swap".

  amrex::Vector<amrex::MultiFab> f_new;
  amrex::Vector<amrex::MultiFab> f_old;
  amrex::Vector<amrex::MultiFab> macro_new;
  amrex::Vector<amrex::MultiFab> macro_old;

  amrex::Vector<amrex::MultiFab> forcing;

  // this is essentially a 2*DIM integer array storing the physical boundary
  // condition types at the lo/hi walls in each direction
  amrex::Vector<amrex::BCRec> bcsMesoscopic; // 1-component
  amrex::Vector<amrex::BCRec> bcsMacro;
  // stores fluxes at coarse-fine interface for synchronization
  // this will be sized "nlevs_max+1"
  // NOTE: the flux register associated with flux_reg[lev] is associated
  // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
  // therefore flux_reg[0] and flux_reg[nlevs_max] are never actually
  // used in the reflux operation

  // Velocity on all faces at all levels

  ////////////////
  // runtime parameters

  // maximum number of steps and stop time
  int max_step = std::numeric_limits<int>::max();
  amrex::Real stop_time = std::numeric_limits<amrex::Real>::max();

  // if >= 0 we restart from a checkpoint
  std::string restart_chkfile = "";

  // how often each level regrids the higher levels of refinement
  // (after a level advances that many time steps)
  int regrid_int = 2;

  // hyperbolic refluxing as part of multilevel synchronization
  int do_reflux = 1;

  // do we subcycle in time?
  int do_subcycle = 1;

  // plotfile prefix and frequency
  std::string plot_file{"plt"};
  int plot_int = -1;

  // checkpoint prefix and frequency
  std::string chk_file{"chk"};
  int chk_int = -1;

  // LBM specific parameter

  // LBM stencil size (default per dimension)
#if (AMREX_SPACEDIM == 2)
  int ndir{9}; // D2Q9
#else
  int ndir{19}; // D3Q19
#endif
  amrex::Vector<amrex::Real> wis;
  amrex::Vector<amrex::Real> wim;
  amrex::Vector<amrex::Real> wi;
  amrex::Vector<amrex::Real> dirx;
  amrex::Vector<amrex::Real> diry;
  amrex::Vector<amrex::Real> dirz;

  amrex::Gpu::DeviceVector<amrex::Real> wi_dev;
  amrex::Gpu::DeviceVector<amrex::Real> dirx_dev;
  amrex::Gpu::DeviceVector<amrex::Real> diry_dev;
  amrex::Gpu::DeviceVector<amrex::Real> dirz_dev;

  amrex::Real nu;
  amrex::Real tau_base;
  amrex::Real T0, L0, U0;

  amrex::Vector<amrex::Real> xCellSize, yCellSize, zCellSize;

  int nmac;
  const int nghost = 3;

  void InitEquilibrium();

  int totalNumberVar;

  void FillPatchMacro(int lev, amrex::Real time, amrex::MultiFab &mf, int icomp,
                      int ncomp);

  void FillPatchForcing(int lev, amrex::Real time, amrex::MultiFab &mf,
                        int icomp, int ncomp);

  void FillCoarsePatchMacro(int lev, amrex::Real time, amrex::MultiFab &mf,
                            int icomp, int ncomp);

  void GetDataMacro(int lev, amrex::Real time,
                    amrex::Vector<amrex::MultiFab *> &data,
                    amrex::Vector<amrex::Real> &datatime);

  BCBuf bcval;

  // BC/geometry switches
  // IBM (cylinder) runtime knobs — default values and OFF by default
  int m_use_cylinder{0}; // 0 = off (default), 1 = on
  int m_ib_method{0};    // 0: none, 1: diffuse, 2: sharp

  // Parameter bundles
  LevelSetParams m_ls_par{};
  DiffuseIBParams m_diff_par{};
  SharpIBParams m_sharp_par{};

  // Backends (opaque handles)
  std::unique_ptr<LevelSetManager> m_ls;
  std::unique_ptr<IBDiffuseLS> m_ibd;
  std::unique_ptr<IBSharpLS> m_ibs;

  // state per level
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_state; // distribution f
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_mac;   // rho,u
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_force; // IB forcing

  // misc
  int m_nlev_alloc{0};
};

#endif
