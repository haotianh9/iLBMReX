#ifndef collide_H_
#define collide_H_

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void collide(amrex::Box const &bx,
             amrex::Array4<amrex::Real> const &f_new_arrs,
             amrex::Array4<amrex::Real const> const &rho_arrs,
             amrex::Array4<amrex::Real const> const &ux_arrs,
             amrex::Array4<amrex::Real const> const &uy_arrs,
             double const tauinv, double const omtauinv, int const ndir, int const nghost,
             const int *dirx, const int *diry,
#if (AMREX_SPACEDIM == 3)
             const int *dirz,
#endif
             const double *wi) {
  using namespace amrex;

  const auto lo = amrex::lbound(bx);
  const auto hi = amrex::ubound(bx);
  // collide
  #if (AMREX_SPACEDIM == 3)
  for (int k = lo.z+nghost; k <= hi.z-nghost; ++k) {
  #else
  for (int k = lo.z; k <= hi.z; ++k) {
   #endif 
    for (int j = lo.y+nghost; j <= hi.y-nghost; ++j) {
      for (int i = lo.x+nghost; i <= hi.x-nghost; ++i) {
        for (int i_dir = 0; i_dir < ndir; ++i_dir) {

          double cidotu = dirx[i_dir] * ux_arrs(i, j, k) +
                          diry[i_dir] * uy_arrs(i, j, k)
#if (AMREX_SPACEDIM == 3)
                          + dirz[i_dir] * uz_arrs(i, j, k)
#endif
              ;
          // calculate equilibrium
          double feq = wi[i_dir] * rho_arrs(i, j, k) *
                       (1.0 + 3.0 * cidotu + 4.5 * cidotu * cidotu -
                        1.5 * (ux_arrs(i, j, k) * ux_arrs(i, j, k) +
                               uy_arrs(i, j, k) * uy_arrs(i, j, k)
#if (AMREX_SPACEDIM == 3)
                               + uz_arrs(i, j, k) * uz_arrs(i, j, k)
#endif
                                   ));

          // relax to equilibrium
          // amrex::Print() << "f_new_arrs: " << f_new_arrs(i, j, k, i_dir) << std::endl;
          // amrex::Print() << "tauinv: " << tauinv << std::endl;  
          // amrex::Print() << "feq: " << feq << std::endl;
          // amrex::Print() << "omtauinv: " << omtauinv << std::endl;
          f_new_arrs(i, j, k, i_dir) = omtauinv * f_new_arrs(i, j, k, i_dir) + tauinv * feq;
          // amrex::Print() << "f_new_arrs: " << f_new_arrs(i, j, k, i_dir) << std::endl;
        }
      }
    }
  }
}

#endif
