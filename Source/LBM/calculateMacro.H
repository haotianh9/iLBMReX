#ifndef calculateMacroH
#define calculateMacroH

#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void calculateMacro(int i, int j, int k,
                    amrex::Array4<amrex::Real const> const &f_new_arrs,
                    amrex::Array4<amrex::Real> const &rho,
                    amrex::Array4<amrex::Real> const &u,
                    amrex::Array4<amrex::Real> const &v,
                    amrex::Array4<amrex::Real> const &w, int const ndir,
                    amrex::Vector<amrex::Real> const dirx,
                    amrex::Vector<amrex::Real> const diry,
                    amrex::Vector<amrex::Real> const dirz) {
  using namespace amrex;

  amrex::Real rho_temp, u_temp, v_temp, w_temp;
  rho_temp = 0.0;
  u_temp = 0.0;
  v_temp = 0.0;
  w_temp = 0.0;

  for (int i_dir = 0; i_dir < ndir; ++i_dir) {
    rho_temp += f_new_arrs(i, j, k, i_dir);
    u_temp += dirx[i_dir] * f_new_arrs(i, j, k, i_dir);
    v_temp += diry[i_dir] * f_new_arrs(i, j, k, i_dir);
    w_temp += dirz[i_dir] * f_new_arrs(i, j, k, i_dir);
  }
  rho(i, j, k) = rho_temp;
  u(i, j, k) = u_temp / rho_temp;
  v(i, j, k) = v_temp / rho_temp;
  w(i, j, k) = w_temp / rho_temp;
}

#endif