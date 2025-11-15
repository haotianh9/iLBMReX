#include <iostream>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>
#include <AmrCoreLBM.H>

// -------- FPE (CPU) helper: use AMReX if present; otherwise fallback -----
#if __has_include(<AMReX_FPE.H>)
  #include <AMReX_FPE.H>
  #define HAVE_AMREX_FPE 1
#else
  #define HAVE_AMREX_FPE 0
#endif

#if !HAVE_AMREX_FPE
  #if defined(__linux__)
    #include <fenv.h>
    inline void enable_fpe_traps_fallback() {
      // Trap invalid, divide-by-zero, overflow (ignore underflow/inexact).
      feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
    }
  #else
    inline void enable_fpe_traps_fallback() {}
  #endif
#endif
// ------------------------------------------------------------------------

using namespace amrex;

int main (int argc, char* argv[])
{
  amrex::Initialize(argc, argv);

#if HAVE_AMREX_FPE
  amrex::InitializeFPE();     // if your AMReX provides it
#else
  enable_fpe_traps_fallback(); // otherwise use glibc fallback on Linux
#endif

  {
    BL_PROFILE("main()");

    AmrCoreLBM amr_core_lbm;
    amr_core_lbm.InitData();
    amr_core_lbm.Evolve();
  }

#if HAVE_AMREX_FPE
  amrex::FinalizeFPE();
#endif
  amrex::Finalize();
  return 0;
}
