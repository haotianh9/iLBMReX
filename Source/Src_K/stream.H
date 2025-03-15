#ifndef stream_H_
#define stream_H_

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>




AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void stream(amrex::Box const &bx, amrex::Array4<amrex::Real> const &f_new_arrs,
            amrex::Array4<amrex::Real const> const &f_old_arrs, int const ndir, int const nghost,
            const int *dirx, const int *diry,
#if (AMREX_SPACEDIM == 3)
            const int *dirz,
#endif
const double *wi   ) {
  using namespace amrex;

  const auto lo = amrex::lbound(bx);
  const auto hi = amrex::ubound(bx);

  // stream
  #if (AMREX_SPACEDIM == 3)
  for (int k = lo.z+nghost; k <= hi.z-nghost; ++k) {
  #else
  for (int k = lo.z; k <= hi.z; ++k) {
   #endif 
  //  amrex::Print()  << lo.y+nghost << " " << hi.y-nghost<< " " << nghost << std::endl;
    for (int j = lo.y+nghost; j <= hi.y-nghost; ++j) {
      for (int i = lo.x+nghost; i <= hi.x-nghost; ++i) {
        for (int i_dir = 0; i_dir < ndir; ++i_dir) {
          int imd = i - dirx[i_dir];
          int jmd = j - diry[i_dir];


#if (AMREX_SPACEDIM == 2)
          
          f_new_arrs(i, j, k, i_dir) = f_old_arrs(imd, jmd, k, i_dir);
          // if (i_dir ==1){
          //   amrex::Print() << "stream: " << i << " " << j << " " << imd << " " << jmd<< " " << f_old_arrs(imd, jmd, k, i_dir)  << std::endl;
          // }
          
#elif (AMREX_SPACEDIM == 3)
          int kmd = k - dirz[i_dir];
          f_new_arrs(i, j, k, i_dir) = f_old_arrs(imd, jmd, kmd, i_dir);
#endif
        }
      }
    }
  }
}



#endif
