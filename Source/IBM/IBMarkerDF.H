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
#include <vector>

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
                      amrex::Real dt,
                      amrex::Real time,
                      amrex::MultiFab const *prev_Fx = nullptr,
                      amrex::MultiFab const *prev_Fy = nullptr,
                      amrex::MultiFab const *prev_Fz = nullptr) const;

  // Access for diagnostics
  [[nodiscard]] amrex::Long num_markers() const;
  [[nodiscard]] amrex::GpuArray<amrex::Real,3> last_marker_force_sum() const;

public:
  void build_markers(amrex::DistributionMapping const &dm,
                     amrex::BoxArray const &ba);

  void update_lagrangian_marker(amrex::Real time) const;

  void velocity_interpolation(amrex::MultiFab const &EulerVel,
                              int euler_vel_comp, int delta_type) const;

  void compute_lagrangian_force_explicit(amrex::Real dt,
                                         amrex::Real time) const;
  void compute_lagrangian_force_ivc(amrex::Real dt,
                                    amrex::Real time) const;

  void force_spreading(amrex::MultiFab &EulerForce, int euler_force_comp,
                       int delta_type) const;

  void velocity_correction(amrex::MultiFab &EulerVel,
                           amrex::MultiFab const &EulerForce,
                           int euler_vel_comp, int euler_force_comp,
                           amrex::Real dt) const;

  void gather_marker_state(std::vector<int> &ids,
                           std::vector<amrex::Real> &data,
                           amrex::Real time) const;

  void scatter_marker_force(std::vector<amrex::Real> const &fx_global,
                            std::vector<amrex::Real> const &fy_global,
                            std::vector<amrex::Real> const &fz_global) const;

  void build_ivc_operator(std::vector<amrex::Real> const &x_global,
                          std::vector<amrex::Real> const &y_global,
                          std::vector<amrex::Real> const &z_global) const;

  static bool lu_factorize(std::vector<amrex::Real> &a,
                           std::vector<int> &pivot, int n,
                           amrex::Real tiny);

  static void lu_solve(std::vector<amrex::Real> const &a,
                       std::vector<int> const &pivot,
                       std::vector<amrex::Real> const &rhs,
                       std::vector<amrex::Real> &x, int n);

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
  mutable amrex::Gpu::DeviceVector<amrex::Real> m_xref_d;
  mutable amrex::Gpu::DeviceVector<amrex::Real> m_yref_d;
  mutable amrex::Gpu::DeviceVector<amrex::Real> m_zref_d;
  std::vector<amrex::Real> m_phiK_h;
  std::vector<amrex::Real> m_thetaK_h;
  std::vector<amrex::Real> m_xref_h;
  std::vector<amrex::Real> m_yref_h;
  std::vector<amrex::Real> m_zref_h;
  int m_start_id{0};
  amrex::Real m_dv{0.0};
  amrex::Real m_wall_tol{0.0};

  std::unique_ptr<MarkerContainer> m_markers;

  // Sum over last update_forcing call:
  //   \int a_ib dV = sum_markers(F_marker * dv)
  // Stored per component (x,y,z).
  mutable amrex::GpuArray<amrex::Real,3> m_last_marker_force_sum{{amrex::Real(0.0),
                                                                   amrex::Real(0.0),
                                                                   amrex::Real(0.0)}};

  // Cached IVC operator (stationary geometry): LU-factorized dense matrix.
  mutable bool m_ivc_operator_ready{false};
  mutable int m_ivc_nmark{0};
  mutable std::vector<amrex::Real> m_ivc_lu;
  mutable std::vector<int> m_ivc_pivot;
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
                      amrex::MultiFab&, amrex::Real, amrex::Real,
                      amrex::MultiFab const* = nullptr,
                      amrex::MultiFab const* = nullptr,
                      amrex::MultiFab const* = nullptr) const {
    amrex::Abort("IBMarkerDF::update_forcing called but AMReX particles are disabled.");
  }

  [[nodiscard]] amrex::Long num_markers() const { return 0; }
  [[nodiscard]] amrex::GpuArray<amrex::Real,3> last_marker_force_sum() const {
    return {{amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)}};
  }
};

#endif // (AMREX_PARTICLES && AMREX_PARTICLES)
