#ifndef streamH
#define streamH

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void stream(int i, int j, int k, amrex::Array4<amrex::Real> const &f_out,
            amrex::Array4<amrex::Real> const &f_old,
            amrex::Real const *dirx, amrex::Real const *diry,
            amrex::Real const *dirz, int const ndir) {
  int ip, jp, kp;

  for (int i_dir = 0; i_dir < ndir; ++i_dir) {
    ip = i - static_cast<int>(dirx[i_dir]);
    jp = j - static_cast<int>(diry[i_dir]);
    kp = k - static_cast<int>(dirz[i_dir]);

    f_out(i, j, k, i_dir) = f_old(ip, jp, kp, i_dir);
  }
}

#endif
