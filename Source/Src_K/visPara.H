#ifndef visParaH
#define visParaH

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void visPara(int i, int j, int k, amrex::Array4<amrex::Real const> const &rho,
             amrex::Array4<amrex::Real const> const &u,
             amrex::Array4<amrex::Real const> const &v,
             amrex::Array4<amrex::Real> const &vorticity,
             amrex::Array4<amrex::Real> const &Pressure, amrex::Real const dx,
             amrex::Real const dy, amrex::Real const T0) {
  using namespace amrex;

  Pressure(i, j, k) = (rho(i, j, k) - 1.0) / (3.0 * 0.5 * 0.02 * 0.02);

  // This is just a test based on the flow rate
  vorticity(i, j, k) = u(i, j, k);
}

#endif
