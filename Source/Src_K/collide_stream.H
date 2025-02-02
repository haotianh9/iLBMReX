#ifndef collide_stream_H_
#define collide_stream_H_

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void collide_stream(amrex::Box const& bx,
             amrex::Array4<amrex::Real> const& f_new_arrs,
             amrex::Array4<amrex::Real > const& f_old_arrs,
             amrex::Array4<amrex::Real const> const& rho_arrs,
             amrex::Array4<amrex::Real const> const& ux_arrs,
             amrex::Array4<amrex::Real const> const& uy_arrs,
             double const tauinv,
             double const omtauinv,
             int const ndir,
             const int* dirx,
             const int* diry,
            #if (AMREX_SPACEDIM == 3)
             const int* dirz,
            #endif
             const double* wi)
{
    using namespace amrex;

    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);
    // collide
    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
               for ( int i_dir = 0; i_dir < ndir; ++i_dir) {

                double cidotu = dirx[i_dir] * ux_arrs(i,j,k) +
                                diry[i_dir] * uy_arrs(i,j,k);
                // calculate equilibrium
                double feq = wi[i_dir] * rho_arrs(i,j,k) *
                            (1.0 + 3.0 * cidotu + 4.5 * cidotu * cidotu -
                                1.5 * (ux_arrs(i,j,k)* ux_arrs(i,j,k) +
                                    uy_arrs(i,j,k) * uy_arrs(i,j,k)));

                // relax to equilibrium
                f_old_arrs(i, j, k, i_dir) =
                    omtauinv * f_old_arrs(i, j, k, i_dir) + tauinv * feq;
                }
            }
        }
    }
    // stream
     for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                for ( int i_dir = 0; i_dir < ndir; ++i_dir) {
          int imd = i - dirx[i_dir];
          int jmd = j - diry[i_dir];

      // need to find the box index of  imd,jmd,kmd , no, find the corresponding
      // ghost cell in my own box
#if (AMREX_SPACEDIM == 2)
          f_new_arrs(i, j, k, i_dir) = f_old_arrs(imd, jmd, k, i_dir);
#elif (AMREX_SPACEDIM == 3)
                                int kmd = k-dirz[i_dir];
                                f_new_arrs(i,j,k,i_dir) = f_old_arrs(imd,jmd,kmd,i_dir);
#endif
       
        }
                 }
        }
    }    
}

#endif
