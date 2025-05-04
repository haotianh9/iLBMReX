#ifndef TAGGING_H
#define TAGGING_H

#include <AMReX_Array4.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void state_error(int i, int j, int k, amrex::Array4<char> const &tag,
                 amrex::Array4<amrex::Real const> const &state,
                 amrex::Real threshold, char tagval) {

  if (std::fabs(state(i, j, k)) > threshold)
    tag(i, j, k) = tagval;
}

#endif
