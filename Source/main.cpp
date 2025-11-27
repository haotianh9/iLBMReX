#include "AmrCoreLBM.H"
#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>

using namespace amrex;

int main(int argc, char *argv[]) {
  amrex::Initialize(argc, argv);
  {
    BL_PROFILE("main");
    AmrCoreLBM sim; // your ctor reads inputs / allocs containers
    sim.InitData(); // your existing initializer
    sim.Evolve();   // your existing time integrator
  }
  amrex::Finalize();
  return 0;
}
