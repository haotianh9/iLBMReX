// DebugNaN.H  (patched)
#ifndef DEBUG_NAN_H_
#define DEBUG_NAN_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Reduce.H>
#include <AMReX_Print.H>

namespace dbg {

// Check that all values in [comp .. comp+ncomp-1] are finite; abort if not.
inline void AssertFinite (amrex::MultiFab const& mf, int comp, int ncomp, std::string tag)
{
    using namespace amrex;

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<long>       reduce_data(reduce_op);

    const int ncomp_loc = ncomp;

    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& arr = mf.const_array(mfi);
        Box const&  bx  = mfi.tilebox();

        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i,int j,int k) -> GpuTuple<long>
            {
                for (int n=0; n<ncomp_loc; ++n) {
                    amrex::Real v = arr(i,j,k, comp+n);
                    if (!amrex::Math::isfinite(v)) return {1L}; // found bad
                }
                return {0L};
            });
    }

    // Extract the scalar from the tuple and MPI-reduce it
    auto hv  = reduce_data.value();      // GpuTuple<long>
    long bad = amrex::get<0>(hv);
    amrex::ParallelDescriptor::ReduceLongSum(bad);

    if (bad) {
        amrex::Abort("NaN/Inf detected in " + tag);
    }
}

inline void AssertFinite (amrex::MultiFab const& mf, std::string tag)
{
    AssertFinite(mf, 0, mf.nComp(), std::move(tag));
}

} // namespace dbg

#endif
