#ifndef PROB_H_
#define PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

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
              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
              double nu) {
    amrex::ignore_unused(vor, prob_lo, prob_hi, dx, nu);
    rho(i, j, k) = 1.0;
    ux(i, j, k) = 0.0;
    uy(i, j, k) = 0.0;
    uz(i, j, k) = 0.0;
}

#endif
