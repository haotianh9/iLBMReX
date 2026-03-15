#ifndef PROB_H_
#define PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <cmath>



AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void initdata(int i, int j, int k,
              amrex::Array4<amrex::Real> const &rho,
              amrex::Array4<amrex::Real> const &ux,
              amrex::Array4<amrex::Real> const &uy,
              amrex::Array4<amrex::Real> const &uz,
              amrex::Array4<amrex::Real> const &vor,
              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi,
              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx, double nu)
{
    using namespace amrex;
    amrex::ignore_unused(vor, nu);
    // Keep initialization consistent with uniform benchmark inflow.
    const double u0 = 0.03;
    rho(i, j, k) = 1.0;
    ux(i, j, k) = u0;
    uy(i, j, k) = 0.0;
    uz(i, j, k) = 0.0;
}

#endif
