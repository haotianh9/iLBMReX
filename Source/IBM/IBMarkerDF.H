#pragma once

#include "IBM/MarkerIBParams.H"

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Config.H>

// IBMarkerDF is implemented using AMReX particles (ParticleContainer).
// When AMReX is built without particle support, we provide a small stub
// implementation that aborts if used.

#if defined(AMREX_PARTICLES) && AMREX_PARTICLES

// Prefer the umbrella header when available; it pulls in the correct set of
// particle headers for the AMReX version in use.
#if defined(__has_include)
#  if __has_include(<AMReX_Particles.H>)
#    include <AMReX_Particles.H>
#  else
#    include <AMReX_TypeTraits.H>
#    include <AMReX_ParticleUtil.H>
#    include <AMReX_ParticleMesh.H>
#    include <AMReX_ParticleContainer.H>
#    include <AMReX_ParIter.H>
#  endif
#else
#  include <AMReX_Particles.H>
#endif

#include <array>
#include <memory>

// IAMReX-style marker-based direct-forcing immersed boundary method.
//
// This ports the core coupling stages from IAMReX's DiffusedIB_Parallel:
//  - Lagrangian markers stored in an AMReX ParticleContainer
//  - Euler->marker interpolation and marker->Euler spreading using
//  discrete-delta kernels
//  - Multi-direct-forcing iterations with in-loop velocity correction (and
//  accumulation of total Eulerian force)
//
// This is intentionally scoped to a single rigid body (the existing cylinder
// geometry in your code).

class IBMarkerDF {
public:
  // Marker attributes (matches IAMReX ordering for interoperability)
  enum P_ATTR_REAL {
    U_Marker = 0,
    V_Marker,
    W_Marker,
    Fx_Marker,
    Fy_Marker,
    Fz_Marker,
    Mx_Marker,
    My_Marker,
    Mz_Marker,
    num_Real
  };

  enum P_ATTR_INT { M_ID = 0, num_Int };

  using MarkerContainer = amrex::ParticleContainer<0, 0, num_Real, num_Int>;
  using MarkerParIter = amrex::ParIter<0, 0, num_Real, num_Int>;

  IBMarkerDF(amrex::Geometry const &geom, amrex::DistributionMapping const &dm,
             amrex::BoxArray const &ba, amrex::Real x0, amrex::Real y0,
             amrex::Real z0, amrex::Real R, MarkerIBParams const &par);

  // Computes IBM body force on the Eulerian grid:
  //   - Inputs:  ucc/vcc/wcc are cell-centered velocities
  //   - Outputs: Fx/Fy/Fz are cell-centered force densities (same
  //   units/convention as your existing IBM)
  void update_forcing(amrex::MultiFab const &rhocc,
                      amrex::MultiFab const &ucc,
                      amrex::MultiFab const &vcc,
                      amrex::MultiFab const &wcc,
                      amrex::MultiFab &Fx,
                      amrex::MultiFab &Fy,
                      amrex::MultiFab &Fz,
                      amrex::Real dt) const;

  // Access for diagnostics
  [[nodiscard]] amrex::Long num_markers() const;

private:
  void build_markers(amrex::DistributionMapping const &dm,
                     amrex::BoxArray const &ba);

  void update_lagrangian_marker() const;

  void velocity_interpolation(amrex::MultiFab const &EulerVel,
                              int euler_vel_comp, int delta_type) const;

  void compute_lagrangian_force(amrex::Real dt) const;

  void force_spreading(amrex::MultiFab &EulerForce, int euler_force_comp,
                       int delta_type) const;

  void velocity_correction(amrex::MultiFab &EulerVel,
                           amrex::MultiFab const &EulerForce,
                           int euler_vel_comp, int euler_force_comp,
                           amrex::Real dt) const;

private:
  amrex::Geometry m_geom;

  // Rigid body (single) parameters
  amrex::Real m_x0{0.0}, m_y0{0.0}, m_z0{0.0};
  amrex::Real m_R{0.0};

  MarkerIBParams m_par;

  // Marker kernel data (IAMReX-style golden spiral on a sphere)
  // We keep device copies so UpdateLagrangianMarker can run on GPU.
  mutable amrex::Gpu::DeviceVector<amrex::Real> m_phiK_d;
  mutable amrex::Gpu::DeviceVector<amrex::Real> m_thetaK_d;
  int m_start_id{0};
  amrex::Real m_dv{0.0};

  std::unique_ptr<MarkerContainer> m_markers;
};

#else

#include <AMReX.H>

// Stub implementation when AMReX particles are not enabled.
// This allows the codebase to build for non-particle examples (e.g. pure LBM)
// while giving a clear runtime error if marker IBM is selected.
class IBMarkerDF {
public:
  IBMarkerDF(amrex::Geometry const &geom, amrex::DistributionMapping const &dm,
             amrex::BoxArray const &ba, amrex::Real x0, amrex::Real y0,
             amrex::Real z0, amrex::Real R, MarkerIBParams const &par) {
    amrex::ignore_unused(geom, dm, ba, x0, y0, z0, R, par);
    amrex::Abort(
        "IBMarkerDF (marker IBM) requires AMReX particles. Rebuild with USE_PARTICLES=TRUE (GNUmake) or enable AMReX particles in your build system.");
  }

  void update_forcing(amrex::MultiFab const&, amrex::MultiFab const&, amrex::MultiFab const&,
                      amrex::MultiFab const&, amrex::MultiFab&, amrex::MultiFab&,
                      amrex::MultiFab&, amrex::Real) const {
    amrex::Abort("IBMarkerDF::update_forcing called but AMReX particles are disabled.");
  }

  [[nodiscard]] amrex::Long num_markers() const { return 0; }
};

#endif // (AMREX_PARTICLES && AMREX_PARTICLES)
