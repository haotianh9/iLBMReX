#ifndef BC_VALUES_H_
#define BC_VALUES_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

#include <cstdio>

namespace BCVals {
extern amrex::Vector<amrex::Real> bc_lo_rho_val;
extern amrex::Vector<amrex::Real> bc_lo_ux_val;
extern amrex::Vector<amrex::Real> bc_lo_uy_val;
extern amrex::Vector<amrex::Real> bc_lo_uz_val;

extern amrex::Vector<amrex::Real> bc_hi_rho_val;
extern amrex::Vector<amrex::Real> bc_hi_ux_val;
extern amrex::Vector<amrex::Real> bc_hi_uy_val;
extern amrex::Vector<amrex::Real> bc_hi_uz_val;
} // namespace BCVals

struct BCBuf {
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lo_rho, lo_ux, lo_uy;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lo_uz;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> hi_rho, hi_ux, hi_uy;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> hi_uz;

  int ndir = 0;
  static constexpr int MAX_NDIR = 27;
  amrex::GpuArray<amrex::Real, MAX_NDIR> dirx{};
  amrex::GpuArray<amrex::Real, MAX_NDIR> diry{};
  amrex::GpuArray<amrex::Real, MAX_NDIR> dirz{};
  amrex::GpuArray<amrex::Real, MAX_NDIR> wi{};
};

#endif
