
#include <iostream>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>

#include <AmrCoreLBM.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    {
        // timer for profiling
        BL_PROFILE("main()");

        // wallclock time
        const auto strt_total = amrex::second();

        // constructor - reads in parameters from inputs file
        //             - sizes multilevel arrays and data structures
        AmrCoreLBM amr_core_lbm;

        // initialize AMR data
        amr_core_lbm.InitData();

        // advance solution to final time
        amr_core_lbm.Evolve();

        // wallclock time
        auto end_total = amrex::second() - strt_total;

        if (amr_core_lbm.Verbose()) {
            // print wallclock time
            ParallelDescriptor::ReduceRealMax(end_total ,ParallelDescriptor::IOProcessorNumber());
            amrex::Print() << "\nTotal Time: " << end_total << '\n';
        }
    }

    amrex::Finalize();
}
