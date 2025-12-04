#ifndef PROB_H_
#define PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>



AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
initdata(amrex::Box const& bx, amrex::Array4<amrex::Real> const& rho,amrex::Array4<amrex::Real> const& ux, amrex::Array4<amrex::Real> const& uy, amrex::Array4<amrex::Real> const& uz,amrex::Array4<amrex::Real> const& vor,amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_hi,
         amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,double nu)
{
    using namespace amrex;
    double t=0;
    double u_max = (double)0.02;	
    double rho0 = 1.0;
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);
    double kx = 2.0 * M_PI / (prob_hi[0]-prob_lo[0]);
    double ky = 2.0 * M_PI / (prob_hi[1]-prob_lo[1]);
    double td = 1.0 / (nu * (kx * kx + ky * ky));
     //amrex::Print() << "prob_lo[0]: " << prob_lo[0] << " prob_hi[0]: " << prob_hi[0] << std::endl;
     //amrex::Print() << "prob_lo[1]: " << prob_lo[1] << " prob_hi[1]: " << prob_hi[1] << std::endl;
    // amrex::Print() << "dx[0]: " << dx[0] << " dx[1]: " << dx[1] << std::endl;
    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {             
            for (int i = lo.x; i <= hi.x; ++i) {

            
                 rho(i, j, k)  =1.0;
                 ux(i, j, k)  =0.0;
                 uy(i, j, k)  =0.0;
                 uz(i, j, k)  =0.0;
                 
            }
        }
    }
}

#endif
