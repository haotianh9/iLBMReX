#ifndef collideH
#define collideH

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void collide(int i, int j, int k,
             // amrex::Array4<amrex::Real>  const & f_out,
             amrex::Array4<amrex::Real> const &f_old,
             amrex::Real const *feq, int const ndir,
             amrex::Real const tau) {
  // amrex::Print() << tau << std::endl;
  // amrex::Print() << "Collide at (" << i << "," << j << "," << k << ")\n";

  for (int i_dir = 0; i_dir < ndir; ++i_dir) {

    // amrex::Print() << "f_old before: " << f_old(i, j, k, i_dir) << std::endl;
    // amrex::Print() << "feq: " << feq[i_dir] << std::endl;
    f_old(i, j, k, i_dir) = 1.0 * f_old(i, j, k, i_dir) -
                            (f_old(i, j, k, i_dir) - feq[i_dir]) / tau;
    // amrex::Print() << "f_old after: " << f_old(i, j, k, i_dir) << std::endl;
  }
}

#endif
