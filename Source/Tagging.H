#ifndef TAGGING_H
#define TAGGING_H

#include <AMReX_Array4.H>
#include <cmath>

// ---------------------------------------------
// Tagging based on vorticity (or any scalar)
// ---------------------------------------------
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void vorticity_tagging(int i, int j, int k, amrex::Array4<char> const &tag,
                       amrex::Array4<amrex::Real const> const &state,
                       amrex::Real threshold, char tagval) {
  if (std::fabs(state(i, j, k)) > threshold) {
    tag(i, j, k) = tagval;
  }
}

// ---------------------------------------------
// Tagging based on signed-distance level set φ:
//   φ = 0 on interface, |φ| = distance
// We refine a band |φ| < (n_cells_band + 0.5)*dx_min.
// ---------------------------------------------
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void levelset_tagging(int i, int j, int k, amrex::Array4<char> const &tag,
                      amrex::Array4<amrex::Real const> const &phi,
                      amrex::Real dx_min, int n_cells_band, char tagval) {
  amrex::Real dist = phi(i, j, k); // signed distance → abs

  if (dist < (static_cast<amrex::Real>(n_cells_band) + 0.5) * dx_min) {
    tag(i, j, k) = tagval;
  }
}

#endif
