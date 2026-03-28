#ifndef visParaH
#define visParaH

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real vorticityDiagnostic(
    int i, int j, int k, amrex::Array4<amrex::Real const> const &u,
    amrex::Array4<amrex::Real const> const &v,
    amrex::Array4<amrex::Real const> const &w, amrex::Real const dx,
    amrex::Real const dy, amrex::Real const dz) {
  using namespace amrex;

  Real dv_dx = (v(i + 1, j, k) - v(i - 1, j, k)) / (2.0 * dx);
  Real du_dy = (u(i, j + 1, k) - u(i, j - 1, k)) / (2.0 * dy);

#if (AMREX_SPACEDIM == 3)
  Real du_dz = (u(i, j, k + 1) - u(i, j, k - 1)) / (2.0 * dz);
  Real dw_dx = (w(i + 1, j, k) - w(i - 1, j, k)) / (2.0 * dx);
  Real dv_dz = (v(i, j, k + 1) - v(i, j, k - 1)) / (2.0 * dz);
  Real dw_dy = (w(i, j + 1, k) - w(i, j - 1, k)) / (2.0 * dy);

  Real omega_x = dw_dy - dv_dz;
  Real omega_y = du_dz - dw_dx;
  Real omega_z = dv_dx - du_dy;

  return std::sqrt(omega_x * omega_x + omega_y * omega_y +
                   omega_z * omega_z);
#else
  amrex::ignore_unused(w, dz);
  return dv_dx - du_dy;
#endif
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void visPara(int i, int j, int k, amrex::Array4<amrex::Real const> const &rho,
             amrex::Array4<amrex::Real const> const &u,
             amrex::Array4<amrex::Real const> const &v,
             amrex::Array4<amrex::Real const> const &w,
             amrex::Array4<amrex::Real> const &vorticity,
             amrex::Array4<amrex::Real> const &Pressure, amrex::Real const dx,
             amrex::Real const dy, amrex::Real const dz, amrex::Real const T0) {
  using namespace amrex;

  // Isothermal LBM pressure perturbation (in lattice units).
  Pressure(i, j, k) = (rho(i, j, k) - 1.0) / 3.0;

  amrex::ignore_unused(T0);
  vorticity(i, j, k) = vorticityDiagnostic(i, j, k, u, v, w, dx, dy, dz);
}

#endif
