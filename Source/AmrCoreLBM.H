#ifndef AmrCoreLBM_H_
#define AmrCoreLBM_H_

#include <AMReX_Config.H>
#ifdef AMREX_PARTICLES
#   include <AMReX_AmrParticles.H>
#   include <AMReX_Particles.H>
#endif

#include <AMReX_AmrCore.H>
#include <AMReX_BCRec.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_FillPatcher.H>
#include <AMReX_BC_TYPES.H>
#ifdef AMREX_USE_OMP
#   include <omp.h>
#endif

#include <limits>
#include <memory>
#include <string>



class AmrCoreLBM
    : public amrex::AmrCore
{
public:

    ////////////////
    // public member functions

    // constructor - reads in parameters from inputs file
    //             - sizes multilevel arrays and data structures
    AmrCoreLBM ();
    virtual ~AmrCoreLBM();

    // advance solution to final time
    void Evolve ();

    // initializes multilevel data
    void InitData ();

    // Make a new level using provided BoxArray and DistributionMapping and
    // fill with interpolated coarse level data.
    // overrides the pure virtual function in AmrCore
    virtual void MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba,
                                         const amrex::DistributionMapping& dm) override;

    // Remake an existing level using provided BoxArray and DistributionMapping and
    // fill with existing fine and coarse data.
    // overrides the pure virtual function in AmrCore
    virtual void RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba,
                              const amrex::DistributionMapping& dm) override;

    // Delete level data
    // overrides the pure virtual function in AmrCore
    virtual void ClearLevel (int lev) override;

    // Make a new level from scratch using provided BoxArray and DistributionMapping.
    // Only used during initialization.
    // overrides the pure virtual function in AmrCore
    virtual void MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& ba,
                                          const amrex::DistributionMapping& dm) override;

    // tag all cells for refinement
    // overrides the pure virtual function in AmrCore
    virtual void ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;

    // Advance pdf at a single level for a single time step, update flux registers
    void AdvanceAtLevel (int lev, amrex::Real time, amrex::Real dt_lev, int /*iteration*/, int /*ncycle*/);

    // Advance pdf at all levels for a single time step
    void  AdvanceAllLevels (amrex::Real time, amrex::Real dt_lev, int iteration);

    void ComputeDensityVelocity (int lev);


private:

    ////////////////
    // private member functions

    // read in some parameters from inputs file
    void ReadParameters();

    // set covered coarse cells to be the average of overlying fine cells
    void AverageDown ();

    // more flexible version of AverageDown() that lets you average down across multiple levels
    void AverageDownTo (int crse_lev);

    enum class FillPatchType { fillpatch_class, fillpatch_function };

    // compute a new multifab by coping in phi from valid region and filling ghost cells
    // works for single level and 2-level cases (fill fine grid ghost by interpolating from coarse)
    void FillPatch (int lev, amrex::Real time, amrex::MultiFab& mf, int icomp, int ncomp, FillPatchType fptype);

    // fill an entire multifab by interpolating from the coarser level
    // this comes into play when a new level of refinement appears
    void FillCoarsePatch (int lev, amrex::Real time, amrex::MultiFab& mf, int icomp, int ncomp);

    // utility to copy in data from phi_old and/or phi_new into another multifab
    void GetData (int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data,
                  amrex::Vector<amrex::Real>& datatime);

    // Advance a level by dt - includes a recursive call for finer levels
    void timeStepWithSubcycling (int lev, amrex::Real time, int iteration);

    // // Advance all levels by the same dt
    void timeStepNoSubcycling (amrex::Real time, int iteration);

    // initialize probability density function based on given IC of velocity and density fields
    void InitEquilibrium () ;

    // get plotfile name
    std::string PlotFileName (int lev) const;

    // put together an array of multifabs for writing
    amrex::Vector<const amrex::MultiFab*> PlotFileMF () const;

    // set plotfile variables names
    amrex::Vector<std::string> PlotFileVarNames () const;

    

    // write plotfile to disk
    void WritePlotFile () const;

    // write checkpoint file to disk
    void WriteCheckpointFile () const;

    // read checkpoint file from disk
    void ReadCheckpointFile ();

    ////////////////
    // private data members
#if (AMREX_SPACEDIM == 2)
    // D2Q9 parameters
    static constexpr unsigned int ndir = 9;
    static constexpr double w0 = 4.0 / 9.0;  // zero weight
    static constexpr double ws = 1.0 / 9.0;  // adjacent weight
    static constexpr double wd = 1.0 / 36.0; // diagonal weight
    static constexpr double wi[] = {w0, ws, ws, ws, ws, wd, wd, wd, wd};
    static constexpr int dirx[] ={0, 1, 0, -1, 0, 1, -1, -1, 1};
    static constexpr int diry[] ={0, 0, 1, 0, -1, 1, 1, -1, -1};
#elif (AMREX_SPACEDIM == 3)
    // D3Q27 parameters
    static constexpr unsigned int ndir = 27;
    static constexpr double w0 = 8.0 / 27.0;  // zero weight
    static constexpr double ws = 2.0 / 27.0;  // adjacent weight
    static constexpr double wd = 1.0 / 54.0;  // diagonal weight
    static constexpr double wv = 1.0 / 216.0; // vertex weight
    static constexpr double wi[] = {
        w0, ws, ws, ws, ws, ws, ws, wd, wd, wd, wd, wd, wd, wd, wd, wd, wd, wd, wd, 
        wv, wv, wv, wv, wv, wv, wv, wv
    };
    static constexpr int dirx[] ={
        0, +1, -1, 0, 0, 0, 0, +1, -1, +1, -1, 0, 0, +1, -1, +1, -1, 0, 0, +1, -1, 
        +1, -1, +1, -1, -1, +1
    };
    static constexpr int diry[] ={
        0, 0, 0, +1, -1, 0, 0, +1, -1, 0, 0, +1, -1, -1, +1, 0, 0, +1, -1, +1, -1, 
        +1, -1, -1, +1, +1, -1
    };
    static constexpr int dirz[] ={
        0, 0, 0, 0, 0, +1, -1, 0, 0, +1, -1, +1, -1, 0, 0, -1, +1, -1, +1, +1, -1, 
        -1, +1, +1, -1, +1, -1
    };
#endif
    amrex::Vector<int> istep;      // which step?
    amrex::Vector<int> nsubsteps;  // how many substeps on each level?

    // keep track of old time, new time, and time step at each level
    amrex::Vector<amrex::Real> t_new;
    amrex::Vector<amrex::Real> t_old;
    amrex::Vector<amrex::Real> dt;
    amrex::Vector<amrex::Real> tau; // relaxation time at each level
    

    // array of multifabs to store the solution at each level of refinement
    // after advancing a level we use "swap".
    amrex::Vector<amrex::MultiFab> f_old;
    amrex::Vector<amrex::MultiFab> f_new;
    amrex::Vector<amrex::MultiFab> ux;
    amrex::Vector<amrex::MultiFab> uy;
    amrex::Vector<amrex::MultiFab> rho;
    


    // this is essentially a 2*DIM integer array storing the physical boundary
    // condition types at the lo/hi walls in each direction
    amrex::Vector<amrex::BCRec> bcs;  // 1-component

    // stores fluxes at coarse-fine interface for synchronization
    // this will be sized "nlevs_max+1"
    // NOTE: the flux register associated with flux_reg[lev] is associated
    // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
    // therefore flux_reg[0] and flux_reg[nlevs_max] are never actually
    // used in the reflux operation
    amrex::Vector<std::unique_ptr<amrex::FluxRegister> > flux_reg;

    // This is for fillpatch during timestepping, but not for regridding.
    amrex::Vector<std::unique_ptr<amrex::FillPatcher<amrex::MultiFab>>> fillpatcher;


    // Velocity on all faces at all levels
    amrex::Vector< amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> > facevel;


    const int nghost = 1;

    ////////////////
    // runtime parameters

    // maximum number of steps and stop time
    int max_step = std::numeric_limits<int>::max();
    amrex::Real stop_time = std::numeric_limits<amrex::Real>::max();

    // if >= 0 we restart from a checkpoint
    std::string restart_chkfile = "";


    double nu = 1.0 / 6.0;
    double tau_base = 3.0 * nu + 0.5; // at the coarsest level
    
    
    // how often each level regrids the higher levels of refinement
    // (after a level advances that many time steps)
    int regrid_int = 2;

    // hyperbolic refluxing as part of multilevel synchronization
    int do_reflux = 1;

    // do we subcycle in time?
    int do_subcycle = 1;

    // plotfile prefix and frequency
    std::string plot_file {"plt"};
    int plot_int = -1;

    // checkpoint prefix and frequency
    std::string chk_file {"chk"};
    int chk_int = -1;
    
};

#endif
