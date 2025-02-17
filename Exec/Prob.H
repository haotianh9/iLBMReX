#ifndef PROB_H_
#define PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>



AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
initdata(amrex::Box const& bx, amrex::Array4<amrex::Real> const& rho,amrex::Array4<amrex::Real> const& ux, amrex::Array4<amrex::Real> const& uy,amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
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
    // amrex::Print() << "prob_lo[0]: " << prob_lo[0] << " prob_hi[0]: " << prob_hi[0] << std::endl;
    // amrex::Print() << "prob_lo[1]: " << prob_lo[1] << " prob_hi[1]: " << prob_hi[1] << std::endl;
    // amrex::Print() << "dx[0]: " << dx[0] << " dx[1]: " << dx[1] << std::endl;
    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            Real Y = prob_lo[1] + (0.5+j) * dx[1];
            for (int i = lo.x; i <= hi.x; ++i) {
                Real X= prob_lo[0] + (0.5+i) * dx[0];
                ux(i, j, k) =  -u_max * sqrt(ky / kx) * cos(kx * X) * sin(ky * Y) * exp(-1.0 * t / td);
                uy(i, j, k) =u_max * sqrt(kx / ky) * sin(kx * X) * cos(ky * Y) * exp(-1.0 * t / td);
                 double P = -0.25 * rho0 * u_max * u_max *
               ((ky / kx) * cos(2.0 * kx * X) + (kx / ky) * cos(2.0 * ky * Y)) *
               exp(-2.0 * t / td);
                rho(i, j, k) = rho0 + 3.0 * P;
            }
        }
    }
}

#endif
