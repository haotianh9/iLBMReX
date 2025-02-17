#ifndef cal_u_rho_H_
#define cal_u_rho_H_

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>






AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void calculate_macro_velo_rho(amrex::Box const &bx,
  amrex::Array4<amrex::Real const> const &f_old_arrs,
  amrex::Array4<amrex::Real > const &rho_arrs,
  amrex::Array4<amrex::Real > const &ux_arrs,
  amrex::Array4<amrex::Real > const &uy_arrs, int const ndir, int const nghost,
            const int *dirx, const int *diry,
#if (AMREX_SPACEDIM == 3)
            const int *dirz,
#endif
            const double *wi) {
  using namespace amrex;

  const auto lo = amrex::lbound(bx);
  const auto hi = amrex::ubound(bx);

  // cal rho u
  for (int k = lo.z+nghost; k <= hi.z-nghost; ++k) {
    for (int j = lo.y+nghost; j <= hi.y-nghost; ++j) {
      for (int i = lo.x+nghost; i <= hi.x-nghost; ++i) {
        double rho_temp = 0.0;
        double ux_temp = 0.0;
        double uy_temp = 0.0;
        for (int i_dir = 0; i_dir < ndir; ++i_dir) {
          int imd = i - dirx[i_dir];
          int jmd = j - diry[i_dir];
          rho_temp += f_old_arrs(imd, jmd, k, i_dir);
          ux_temp += dirx[i_dir] * f_old_arrs(imd, jmd, k, i_dir);
          uy_temp += diry[i_dir] * f_old_arrs(imd, jmd, k, i_dir);
// need to find the box index of  imd,jmd,kmd , no, find the corresponding
// ghost cell in my own box
#if (AMREX_SPACEDIM == 3)
          uz_temp += dirz[i_dir] * f_old_arrs(imd, jmd, kmd, i_dir);
#endif
        }
        rho_arrs(i, j, k) = rho_temp;
        ux_arrs(i, j, k) = ux_temp / rho_temp;
        uy_arrs(i, j, k) = uy_temp / rho_temp;
#if (AMREX_SPACEDIM == 3)
        uz_arrs(i, j, k) = uz_temp / rho_temp;
#endif

      }
    }
  }
}




#endif
