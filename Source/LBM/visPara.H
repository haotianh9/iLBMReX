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
             amrex::Real const dy, amrex::Real const dz, amrex::Real const T0) {
  using namespace amrex;

  // Isothermal LBM pressure perturbation (in lattice units).
  Pressure(i, j, k) = (rho(i, j, k) - 1.0) / 3.0;

  // This is just a test based on the flow rate
  Real dv_dx = (v(i + 1, j, k) - v(i - 1, j, k)) / (2.0 * dx);
  Real du_dy = (u(i, j + 1, k) - u(i, j - 1, k)) / (2.0 * dy);

  vorticity(i, j, k) = dv_dx - du_dy;
}

#endif
