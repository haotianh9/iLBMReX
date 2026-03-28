#include "IBM/IBMarkerDF.H"

#if defined(AMREX_PARTICLES) && AMREX_PARTICLES

#include <AMReX_Array4.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
// NOTE: this AMReX tree does not ship AMReX_ParallelFor.H; ParallelFor lives in
// GpuLaunch.
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#if defined(__has_include)
#  if __has_include("IBMUserDefinedGeometry.H")
#    include "IBMUserDefinedGeometry.H"
#    define LBM_HAVE_USER_DEFINED_IBM_GEOMETRY 1
#  endif
#endif

#ifndef LBM_HAVE_USER_DEFINED_IBM_GEOMETRY
#  define LBM_HAVE_USER_DEFINED_IBM_GEOMETRY 0
#endif

using namespace amrex;

#if !LBM_HAVE_USER_DEFINED_IBM_GEOMETRY
namespace lbm_user_ibm_geometry {

inline const char *label() noexcept { return "user_defined"; }

inline void build_markers(amrex::Geometry const &geom, MarkerIBParams const &par,
                          amrex::Real h,
                          std::vector<amrex::Real> &x_ref,
                          std::vector<amrex::Real> &y_ref,
                          std::vector<amrex::Real> &z_ref,
                          amrex::Real &dv, amrex::Real &wall_tol) {
  amrex::ignore_unused(geom, par, h, x_ref, y_ref, z_ref, dv, wall_tol);
  amrex::Abort(
      "marker_geometry = user_defined requires IBMUserDefinedGeometry.H in the example directory.");
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
marker_state(int idx, amrex::Real time, MarkerIBParams const &par,
             amrex::Real x0, amrex::Real y0, amrex::Real z0,
             amrex::Real const *xref, amrex::Real const *yref,
             amrex::Real const *zref, amrex::Real &x, amrex::Real &y,
             amrex::Real &z, amrex::Real &ux, amrex::Real &uy,
             amrex::Real &uz) noexcept {
  amrex::ignore_unused(idx, time, par, xref, yref, zref);
  x = x0;
  y = y0;
  z = z0;
  ux = amrex::Real(0.0);
  uy = amrex::Real(0.0);
  uz = amrex::Real(0.0);
}

} // namespace lbm_user_ibm_geometry
#endif

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
delta_function(Real xf, Real xp, Real h, int type) noexcept {
  // Matches IAMReX DiffusedIB_Parallel::deltaFunction (value/h scaling).
  const Real rr = Math::abs(xf - xp) / h;
  Real value = amrex::Real(0.0);

  if (type == 0) { // FOUR_POINT_IB
    if (rr < amrex::Real(1.0)) {
      value = amrex::Real(0.125) * (amrex::Real(3.0) - amrex::Real(2.0) * rr +
                          std::sqrt(amrex::Real(1.0) + amrex::Real(4.0) * rr - amrex::Real(4.0) * rr * rr));
    } else if (rr < amrex::Real(2.0)) {
      value = amrex::Real(0.125) * (amrex::Real(5.0) - amrex::Real(2.0) * rr -
                          std::sqrt(-amrex::Real(7.0) + amrex::Real(12.0) * rr - amrex::Real(4.0) * rr * rr));
    } else {
      value = amrex::Real(0.0);
    }
  } else { // THREE_POINT_IB
    if (rr < amrex::Real(0.5)) {
      value = (amrex::Real(1.0) + std::sqrt(amrex::Real(1.0) - amrex::Real(3.0) * rr * rr)) / amrex::Real(3.0);
    } else if (rr < amrex::Real(1.5)) {
      value = (amrex::Real(5.0) - amrex::Real(3.0) * rr -
               std::sqrt(-amrex::Real(2.0) + amrex::Real(6.0) * rr - amrex::Real(3.0) * rr * rr)) /
              amrex::Real(6.0);
    } else {
      value = amrex::Real(0.0);
    }
  }

  return value / h;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
delta_type_to_kernel(int delta_type) noexcept {
  // Accept both IAMReX enum-style (0/1) and explicit stencil-size (4/3) input.
  // Internally, delta_function expects: 0 => 4-pt, 1 => 3-pt.
  if (delta_type == 4 || delta_type == 0) {
    return 0;
  }
  if (delta_type == 3 || delta_type == 1) {
    return 1;
  }
  // Fallback to the more robust 4-pt kernel on invalid input.
  return 0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
omega_cross_r(const Real om[3], Real rx, Real ry, Real rz, Real &cx, Real &cy,
              Real &cz) noexcept {
  cx = om[1] * rz - om[2] * ry;
  cy = om[2] * rx - om[0] * rz;
  cz = om[0] * ry - om[1] * rx;
}

AMREX_FORCE_INLINE Real overlap_1d(Real xa, Real xb, Real h, int kernel_type,
                                   Real plo, int ilo, int ihi) noexcept {
  const int ia = static_cast<int>(std::floor((xa - plo) / h));
  const int ib = static_cast<int>(std::floor((xb - plo) / h));

  int lo = amrex::max(ia - 2, ib - 2);
  int hi = amrex::min(ia + 2, ib + 2);
  lo = amrex::max(lo, ilo);
  hi = amrex::min(hi, ihi);

  if (lo > hi) {
    return amrex::Real(0.0);
  }

  Real sum = amrex::Real(0.0);
  for (int i = lo; i <= hi; ++i) {
    const Real xc = plo + (Real(i) + amrex::Real(0.5)) * h;
    const Real da = delta_function(xc, xa, h, kernel_type);
    const Real db = delta_function(xc, xb, h, kernel_type);
    sum += da * db * h;
  }
  return sum;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
box_target_velocity(Real x, Real y, Real z, MarkerIBParams const &par,
                    Real wall_tol, Real &ux, Real &uy, Real &uz) noexcept {
  amrex::ignore_unused(x, z);
  ux = amrex::Real(0.0);
  uy = amrex::Real(0.0);
  uz = amrex::Real(0.0);

  // Cavity lid by default: only the yhi face moves.
  if (amrex::Math::abs(y - par.box_yhi) <= wall_tol) {
    ux = par.box_lid_ux;
    uy = par.box_lid_uy;
    uz = par.box_lid_uz;
  }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
rigid_target_velocity(Real x, Real y, Real z, Real x0, Real y0, Real z0,
                      MarkerIBParams const &par, Real wall_tol, Real time,
                      Real &ux, Real &uy, Real &uz) noexcept {
  amrex::ignore_unused(time);
  if (par.geometry_type == MarkerIBParams::GeometryBox) {
    box_target_velocity(x, y, z, par, wall_tol, ux, uy, uz);
    return;
  }

  Real om[3] = {par.omx, par.omy, par.omz};
  const Real rx = x - x0;
  const Real ry = y - y0;
  const Real rz = z - z0;
  Real urx, ury, urz;
  omega_cross_r(om, rx, ry, rz, urx, ury, urz);
  ux = par.ubx + urx;
  uy = par.uby + ury;
  uz = par.ubz + urz;
}

} // namespace

IBMarkerDF::IBMarkerDF(Geometry const &geom, DistributionMapping const &dm,
                       BoxArray const &ba, Real x0, Real y0, Real z0, Real R,
                       MarkerIBParams const &par)
    : m_geom(geom), m_x0(x0), m_y0(y0), m_z0(z0), m_R(R), m_par(par) {
  build_markers(dm, ba);
}

Long IBMarkerDF::num_markers() const {
  return m_markers ? m_markers->TotalNumberOfParticles() : 0;
}

amrex::GpuArray<amrex::Real,3> IBMarkerDF::last_marker_force_sum() const {
  return m_last_marker_force_sum;
}

void IBMarkerDF::build_markers(DistributionMapping const &dm,
                               BoxArray const &ba) {
  m_markers = std::make_unique<MarkerContainer>(m_geom, dm, ba);

  const auto dx = m_geom.CellSizeArray();
  const Real h = dx[0];
  const Real pi = Math::pi<Real>();

  int Ml = m_par.n_marker;
  Real dv = amrex::Real(0.0);
  const bool use_box = (m_par.geometry_type == MarkerIBParams::GeometryBox);
  const bool use_custom =
      (m_par.geometry_type == MarkerIBParams::GeometryUserDefined);

  std::vector<Real> phi_h;
  std::vector<Real> theta_h;
  std::vector<Real> x_ref;
  std::vector<Real> y_ref;
  std::vector<Real> z_ref;

  if (use_box) {
#if (AMREX_SPACEDIM != 2)
    amrex::Abort(
        "IBMarkerDF: box marker geometry is currently implemented for 2D only.");
#else
    const auto problo = m_geom.ProbLoArray();
    const auto probhi = m_geom.ProbHiArray();

    Real xlo = m_par.box_xlo;
    Real xhi = m_par.box_xhi;
    Real ylo = m_par.box_ylo;
    Real yhi = m_par.box_yhi;

    if (xhi <= xlo) {
      xlo = problo[0];
      xhi = probhi[0];
    }
    if (yhi <= ylo) {
      ylo = problo[1];
      yhi = probhi[1];
    }

    // Keep markers strictly inside the particle domain.
    // AMReX particle coordinates on non-periodic domains are effectively
    // half-open at prob_hi, so markers exactly at prob_hi are dropped on
    // Redistribute.
    const Real epsx = amrex::max(amrex::Real(1.0e-6) * dx[0],
                                 amrex::Real(1.0e-12));
    const Real epsy = amrex::max(amrex::Real(1.0e-6) * dx[1],
                                 amrex::Real(1.0e-12));
    xlo = amrex::max(xlo, problo[0] + epsx);
    xhi = amrex::min(xhi, probhi[0] - epsx);
    ylo = amrex::max(ylo, problo[1] + epsy);
    yhi = amrex::min(yhi, probhi[1] - epsy);

    if (xhi <= xlo || yhi <= ylo) {
      amrex::Abort("IBMarkerDF: invalid in-domain box bounds after clamping.");
    }

    // Persist effective bounds so lid targeting and diagnostics use the actual
    // marker geometry.
    m_par.box_xlo = xlo;
    m_par.box_xhi = xhi;
    m_par.box_ylo = ylo;
    m_par.box_yhi = yhi;

    Real ds = (m_par.box_ds > amrex::Real(0.0)) ? m_par.box_ds : h;
    ds = amrex::max(ds, amrex::Real(1.0e-12));

    const int nx =
        amrex::max(2, static_cast<int>(std::round((xhi - xlo) / ds)) + 1);
    const int ny =
        amrex::max(2, static_cast<int>(std::round((yhi - ylo) / ds)) + 1);
    const Real dsx = (xhi - xlo) / Real(nx - 1);
    const Real dsy = (yhi - ylo) / Real(ny - 1);

    const bool lid_only = (m_par.box_lid_only != 0);
    if (lid_only) {
      x_ref.reserve(static_cast<std::size_t>(nx + 1));
    } else {
      x_ref.reserve(static_cast<std::size_t>(2 * nx + 2 * amrex::max(ny - 2, 0) +
                                             1));
    }
    y_ref.reserve(x_ref.capacity());
    z_ref.reserve(x_ref.capacity());

    // Keep index 0 unused so marker id() can be used as a direct index.
    x_ref.push_back(amrex::Real(0.0));
    y_ref.push_back(amrex::Real(0.0));
    z_ref.push_back(amrex::Real(0.0));

    auto add_marker = [&](Real x, Real y) {
      x_ref.push_back(x);
      y_ref.push_back(y);
      z_ref.push_back(amrex::Real(0.0));
    };

    if (lid_only) {
      // Lid-only forcing: only place markers on the top wall.
      for (int i = 0; i < nx; ++i) {
        add_marker(xlo + Real(i) * dsx, yhi);
      }
    } else {
      // Full cavity box: all four walls.
      // Bottom and top walls (include corners).
      for (int i = 0; i < nx; ++i) {
        add_marker(xlo + Real(i) * dsx, ylo);
      }
      for (int i = 0; i < nx; ++i) {
        add_marker(xlo + Real(i) * dsx, yhi);
      }
      // Left and right walls (exclude corners to avoid duplicates).
      for (int j = 1; j < ny - 1; ++j) {
        const Real y = ylo + Real(j) * dsy;
        add_marker(xlo, y);
        add_marker(xhi, y);
      }
    }

    Ml = static_cast<int>(x_ref.size()) - 1;
    AMREX_ALWAYS_ASSERT(Ml > 0);

    const Real perimeter = lid_only
                               ? (xhi - xlo)
                               : amrex::Real(2.0) * ((xhi - xlo) + (yhi - ylo));
    const Real ds_eff = perimeter / Real(Ml);
    dv = ds_eff * h;

    m_wall_tol = (m_par.box_wall_tol > amrex::Real(0.0))
                     ? m_par.box_wall_tol
                     : amrex::max(amrex::Real(0.25) * amrex::min(dsx, dsy),
                                  amrex::Real(1.0e-12));

    m_xref_d.resize(Ml + 1);
    m_yref_d.resize(Ml + 1);
    m_zref_d.resize(Ml + 1);
    m_xref_h = x_ref;
    m_yref_h = y_ref;
    m_zref_h = z_ref;
    Gpu::copy(Gpu::hostToDevice, x_ref.begin(), x_ref.end(), m_xref_d.begin());
    Gpu::copy(Gpu::hostToDevice, y_ref.begin(), y_ref.end(), m_yref_d.begin());
    Gpu::copy(Gpu::hostToDevice, z_ref.begin(), z_ref.end(), m_zref_d.begin());

    m_phiK_d.resize(0);
    m_thetaK_d.resize(0);
    m_phiK_h.clear();
    m_thetaK_h.clear();
#endif
  } else if (use_custom) {
    lbm_user_ibm_geometry::build_markers(m_geom, m_par, h, x_ref, y_ref, z_ref,
                                         dv, m_wall_tol);
    Ml = static_cast<int>(x_ref.size()) - 1;
    AMREX_ALWAYS_ASSERT(Ml > 0);

    m_xref_d.resize(Ml + 1);
    m_yref_d.resize(Ml + 1);
    m_zref_d.resize(Ml + 1);
    m_xref_h = x_ref;
    m_yref_h = y_ref;
    m_zref_h = z_ref;
    Gpu::copy(Gpu::hostToDevice, x_ref.begin(), x_ref.end(), m_xref_d.begin());
    Gpu::copy(Gpu::hostToDevice, y_ref.begin(), y_ref.end(), m_yref_d.begin());
    Gpu::copy(Gpu::hostToDevice, z_ref.begin(), z_ref.end(), m_zref_d.begin());

    m_phiK_d.resize(0);
    m_thetaK_d.resize(0);
    m_phiK_h.clear();
    m_thetaK_h.clear();
  } else {
#if (AMREX_SPACEDIM == 2)
    if (Ml <= 0) {
      Ml = static_cast<int>(std::ceil(amrex::Real(2.0) * pi * m_R / h));
      Ml = std::max(Ml, 16);
    }
    // Approximate annulus area of thickness h.
    dv = (amrex::Real(2.0) * pi * m_R * h) / static_cast<Real>(Ml);

    // theta=pi/2 -> z=0, and phi is the polar angle.
    phi_h.assign(Ml + 1, amrex::Real(0.0));
    theta_h.assign(Ml + 1, amrex::Real(0.5) * pi);
    for (int i = 1; i <= Ml; ++i) {
      phi_h[i] = amrex::Real(2.0) * pi * Real(i - 1) / Real(Ml);
    }
#else
    // IAMReX-like heuristic: build markers on a thin shell at radius R.
    if (Ml <= 0) {
      const Real rd = m_par.rd;
      const Real a1 = Math::powi<3>(m_R - (rd - amrex::Real(0.5)) * h);
      const Real a2 = Math::powi<3>(m_R - (rd + amrex::Real(0.5)) * h);
      const Real denom = amrex::Real(3.0) * h * h * h / amrex::Real(4.0) / pi;
      Ml = static_cast<int>((a1 - a2) / denom);
      Ml = std::max(Ml, 32);
    }
    {
      const Real rd = m_par.rd;
      const Real a1 = Math::powi<3>(m_R - (rd - amrex::Real(0.5)) * h);
      const Real a2 = Math::powi<3>(m_R - (rd + amrex::Real(0.5)) * h);
      dv = (a1 - a2) / (amrex::Real(3.0) * Real(Ml) / amrex::Real(4.0) / pi);
    }

    // Golden-spiral parameterization (ported from IAMReX), but shifted to be
    // 1-based so we can index with particle.id() directly.
    phi_h.assign(Ml + 1, amrex::Real(0.0));
    theta_h.assign(Ml + 1, amrex::Real(0.0));
    Real phi = amrex::Real(0.0);
    for (int marker_index = 0; marker_index < Ml; ++marker_index) {
      const Real Hk = -amrex::Real(1.0) +
                      amrex::Real(2.0) * Real(marker_index) /
                          (Real(Ml) - amrex::Real(1.0));
      const Real theta = std::acos(Hk);
      if (marker_index == 0 || marker_index == Ml - 1) {
        phi = amrex::Real(0.0);
      } else {
        phi = std::fmod(phi + amrex::Real(3.809) / std::sqrt(Real(Ml)) /
                                  std::sqrt(amrex::Real(1.0) - Hk * Hk),
                        amrex::Real(2.0) * pi);
      }
      const int idx = marker_index + 1;
      phi_h[idx] = phi;
      theta_h[idx] = theta;
    }
#endif

    m_wall_tol = amrex::Real(0.0);

    m_phiK_d.resize(Ml + 1);
    m_thetaK_d.resize(Ml + 1);
    m_phiK_h = phi_h;
    m_thetaK_h = theta_h;
    Gpu::copy(Gpu::hostToDevice, phi_h.begin(), phi_h.end(), m_phiK_d.begin());
    Gpu::copy(Gpu::hostToDevice, theta_h.begin(), theta_h.end(),
              m_thetaK_d.begin());

    m_xref_d.resize(0);
    m_yref_d.resize(0);
    m_zref_d.resize(0);
    m_xref_h.clear();
    m_yref_h.clear();
    m_zref_h.clear();
  }

  m_dv = dv;

  // Insert markers (on IOProcessor) into a single tile, then Redistribute.
  auto &ptile = m_markers->GetParticles(0)[{0, 0}];

  if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
    for (int i = 1; i <= Ml; ++i) {
      MarkerContainer::ParticleType p;
      p.id() = i;
      p.cpu() = ParallelDescriptor::MyProc();
      p.pos(0) = m_x0;
      p.pos(1) = m_y0;
#if (AMREX_SPACEDIM == 3)
      p.pos(2) = m_z0;
#endif

      std::array<ParticleReal, num_Real> attr{};
      std::array<int, num_Int> idarr{};
      idarr[M_ID] = i; // stable local marker index for reference arrays

      ptile.push_back(p);
      ptile.push_back_real(attr);
      ptile.push_back_int(idarr);
    }
  }

  m_markers->Redistribute();

  // We insert markers with ids 1..Ml and store the stable marker index in M_ID.
  // Avoid host-side particle-id scans here because GPU particle data may live
  // in device memory after Redistribute.
  m_start_id = 0;

  update_lagrangian_marker(amrex::Real(0.0));

  if (m_par.verbose) {
    amrex::Print() << "[IBMarkerDF] Built " << num_markers()
                   << " markers, dv=" << m_dv
                   << " geometry="
                   << ((m_par.geometry_type == MarkerIBParams::GeometryBox)
                           ? "box"
                           : ((m_par.geometry_type ==
                               MarkerIBParams::GeometryUserDefined)
                                  ? lbm_user_ibm_geometry::label()
                                  : "cylinder"))
                   << " delta_type=" << m_par.delta_type
                   << " loop_ns=" << m_par.loop_ns
                   << " id_range=[1," << Ml
                   << "] start_id=" << m_start_id;
    if (m_par.geometry_type == MarkerIBParams::GeometryBox) {
      amrex::Print() << " box_bounds=(" << m_par.box_xlo << "," << m_par.box_xhi
                     << ")x(" << m_par.box_ylo << "," << m_par.box_yhi << ")";
    }
    amrex::Print() << "\n";
  }
}

void IBMarkerDF::update_lagrangian_marker(Real time) const {
  // IAMReX semantics: update positions and reset per-marker arrays to zero,
  // then Redistribute.
  const auto *phiK = m_phiK_d.data();
  const auto *thetaK = m_thetaK_d.data();
  const auto *xref = m_xref_d.data();
  const auto *yref = m_yref_d.data();
  const auto *zref = m_zref_d.data();
  const Real x0 = m_x0;
  const Real y0 = m_y0;
#if (AMREX_SPACEDIM == 3)
  const Real z0 = m_z0;
#else
  const Real z0 = amrex::Real(0.0);
#endif
  const Real R = m_R;
  const int start_id = m_start_id;
  const int use_box =
      (m_par.geometry_type == MarkerIBParams::GeometryBox) ? 1 : 0;
  const int use_custom =
      (m_par.geometry_type == MarkerIBParams::GeometryUserDefined) ? 1 : 0;
  const int max_idx = (use_box != 0 || use_custom != 0)
                          ? static_cast<int>(m_xref_d.size()) - 1
                          : static_cast<int>(m_phiK_d.size()) - 1;
  const int max_idx_ref = static_cast<int>(m_xref_d.size()) - 1;

  for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
    auto *particles = pti.GetArrayOfStructs().data();
    const Long np = pti.numParticles();

    auto &soa = pti.GetStructOfArrays();
    auto const *midP = soa.GetIntData(M_ID).data();
    auto *uP = soa.GetRealData(U_Marker).data();
    auto *vP = soa.GetRealData(V_Marker).data();
    auto *wP = soa.GetRealData(W_Marker).data();
    auto *fxP = soa.GetRealData(Fx_Marker).data();
    auto *fyP = soa.GetRealData(Fy_Marker).data();
    auto *fzP = soa.GetRealData(Fz_Marker).data();
    auto *mxP = soa.GetRealData(Mx_Marker).data();
    auto *myP = soa.GetRealData(My_Marker).data();
    auto *mzP = soa.GetRealData(Mz_Marker).data();

    const bool renorm = (m_par.renormalize_delta != 0) &&
                        (m_par.coupling_method != MarkerIBParams::CouplingIVC);
    const MarkerIBParams par = m_par;
    ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      const int m_id = particles[i].id();
      int idx = midP[i];
      if (idx <= 0) {
        idx = m_id - start_id;
      }
      if (max_idx > 0 && (idx < 1 || idx > max_idx)) {
        idx = ((idx - 1) % max_idx + max_idx) % max_idx + 1;
      }

      if (use_box != 0) {
        particles[i].pos(0) = xref[idx];
        particles[i].pos(1) = yref[idx];
#if (AMREX_SPACEDIM == 3)
        particles[i].pos(2) = zref[idx];
#endif
        mxP[i] = amrex::Real(0.0);
        myP[i] = amrex::Real(0.0);
        mzP[i] = amrex::Real(0.0);
      } else if (use_custom != 0) {
        int ridx = idx;
        if (max_idx_ref > 0 && (ridx < 1 || ridx > max_idx_ref)) {
          ridx = ((ridx - 1) % max_idx_ref + max_idx_ref) % max_idx_ref + 1;
        }
        Real px = x0;
        Real py = y0;
        Real pz = z0;
        Real utx = amrex::Real(0.0);
        Real uty = amrex::Real(0.0);
        Real utz = amrex::Real(0.0);
        lbm_user_ibm_geometry::marker_state(ridx, time, par, x0, y0, z0,
                                            xref, yref, zref, px, py, pz, utx,
                                            uty, utz);
        particles[i].pos(0) = px;
        particles[i].pos(1) = py;
#if (AMREX_SPACEDIM == 3)
        particles[i].pos(2) = pz;
#endif
        mxP[i] = utx;
        myP[i] = uty;
        mzP[i] = utz;
      } else {
        // Position on the sphere/circle.
        particles[i].pos(0) =
            x0 + R * std::sin(thetaK[idx]) * std::cos(phiK[idx]);
        particles[i].pos(1) =
            y0 + R * std::sin(thetaK[idx]) * std::sin(phiK[idx]);
#if (AMREX_SPACEDIM == 3)
        particles[i].pos(2) = z0 + R * std::cos(thetaK[idx]);
#endif
        mxP[i] = amrex::Real(0.0);
        myP[i] = amrex::Real(0.0);
        mzP[i] = amrex::Real(0.0);
      }

      uP[i] = amrex::Real(0.0);
      vP[i] = amrex::Real(0.0);
      wP[i] = amrex::Real(0.0);
      fxP[i] = amrex::Real(0.0);
      fyP[i] = amrex::Real(0.0);
      fzP[i] = amrex::Real(0.0);
    });
  }

  m_markers->Redistribute();
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void velocity_interpolation_cic(
    P const &p, Real &Up, Real &Vp, Real &Wp, Array4<Real const> const &E,
    int EulerVIndex, GpuArray<Real, AMREX_SPACEDIM> const &plo,
    GpuArray<Real, AMREX_SPACEDIM> const &dx, int type,
    int ilo, int ihi, int jlo, int jhi, int klo, int khi,
    bool renormalize_delta, Real* partition_out) noexcept {
  // AMREX_D_TERM does token concatenation; include operators in args 2/3.
  // In 2D: dx[0]*dx[1]. In 3D: dx[0]*dx[1]*dx[2].
  const Real d = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

  const Real lx = (p.pos(0) - plo[0]) / dx[0];
  const Real ly = (p.pos(1) - plo[1]) / dx[1];
#if (AMREX_SPACEDIM == 3)
  const Real lz = (p.pos(2) - plo[2]) / dx[2];
#else
  const Real lz = amrex::Real(0.0);
#endif

  const int i0 = static_cast<int>(std::floor(lx));
  const int j0 = static_cast<int>(std::floor(ly));
  const int k0 = static_cast<int>(std::floor(lz));
#if (AMREX_SPACEDIM < 3)
  amrex::ignore_unused(k0);
#endif

  Real u = amrex::Real(0.0), v = amrex::Real(0.0), w = amrex::Real(0.0);
  Real part = amrex::Real(0.0);

  // IAMReX DiffusedIB style: always traverse a 5-point stencil in each
  // direction; the delta kernel itself determines effective support.
  for (int ii = -2; ii < 3; ++ii) {
    const Real xi = plo[0] + (Real(i0 + ii) + amrex::Real(0.5)) * dx[0];
    const Real tU = delta_function(xi, p.pos(0), dx[0], type);

    for (int jj = -2; jj < 3; ++jj) {
      const Real yi = plo[1] + (Real(j0 + jj) + amrex::Real(0.5)) * dx[1];
      const Real tV = delta_function(yi, p.pos(1), dx[1], type);

#if (AMREX_SPACEDIM == 3)
      for (int kk = -2; kk < 3; ++kk) {
        const Real zi = plo[2] + (Real(k0 + kk) + amrex::Real(0.5)) * dx[2];
        const Real tW = delta_function(zi, p.pos(2), dx[2], type);

        const int ip = i0 + ii;
        const int jp = j0 + jj;
        const int kp = k0 + kk;
        if (ip < ilo || ip > ihi || jp < jlo || jp > jhi || kp < klo ||
            kp > khi) {
          continue;
        }
        const Real delta_value = tU * tV * tW;
        part += delta_value * d;
        u += delta_value * E(ip, jp, kp, EulerVIndex + 0) * d;
        v += delta_value * E(ip, jp, kp, EulerVIndex + 1) * d;
        w += delta_value * E(ip, jp, kp, EulerVIndex + 2) * d;
      }
#else
      const int ip = i0 + ii;
      const int jp = j0 + jj;
      if (ip < ilo || ip > ihi || jp < jlo || jp > jhi) {
        continue;
      }
      const Real delta_value = tU * tV;
      part += delta_value * d;
      u += delta_value * E(ip, jp, 0, EulerVIndex + 0) * d;
      v += delta_value * E(ip, jp, 0, EulerVIndex + 1) * d;
      w += amrex::Real(0.0);
#endif
    }
  }

  if (partition_out != nullptr) {
    *partition_out = part;
  }
  if (renormalize_delta && part > amrex::Real(0.0)) {
    u /= part;
    v /= part;
    w /= part;
  }

  Up = u;
  Vp = v;
  Wp = w;
}

void IBMarkerDF::velocity_interpolation(MultiFab const &EulerVel,
                                        int euler_vel_comp,
                                        int delta_type) const {
  // Fill boundary before interpolation (IAMReX semantics).
  MultiFab &E = const_cast<MultiFab &>(EulerVel);
  E.FillBoundary(euler_vel_comp, 3, m_geom.periodicity());

  const auto dx = m_geom.CellSizeArray();
  const auto plo = m_geom.ProbLoArray();

  for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
    auto *particles = pti.GetArrayOfStructs().data();
    const Long np = pti.numParticles();

    auto &soa = pti.GetStructOfArrays();
    auto *uP = soa.GetRealData(U_Marker).data();
    auto *vP = soa.GetRealData(V_Marker).data();
    auto *wP = soa.GetRealData(W_Marker).data();

    auto const &Efab = E[pti].const_array();
    const Box ebx = E[pti].box();
    const int ilo = ebx.smallEnd(0), ihi = ebx.bigEnd(0);
    const int jlo = ebx.smallEnd(1), jhi = ebx.bigEnd(1);
#if (AMREX_SPACEDIM == 3)
    const int klo = ebx.smallEnd(2), khi = ebx.bigEnd(2);
#else
    const int klo = 0, khi = 0;
#endif
    const int kernel_type = delta_type_to_kernel(delta_type);

    const bool renorm = (m_par.renormalize_delta != 0) &&
                        (m_par.coupling_method != MarkerIBParams::CouplingIVC);
    ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      Real Up = amrex::Real(0.0), Vp = amrex::Real(0.0), Wp = amrex::Real(0.0);
      velocity_interpolation_cic(particles[i], Up, Vp, Wp, Efab, euler_vel_comp,
                                 plo, dx, kernel_type, ilo, ihi, jlo, jhi, klo,
                                 khi, renorm, nullptr);
      uP[i] += Up;
      vP[i] += Vp;
      wP[i] += Wp;
    });
  }
}

void IBMarkerDF::compute_lagrangian_force_explicit(Real dt, Real time) const {
  // Matches IAMReX: F = (Ub + omega x r - U_marker) / dt
  const Real relax =
      amrex::max(amrex::Real(0.0), amrex::min(amrex::Real(1.0), m_par.mdf_relax));
  const bool use_diag_correction =
      (m_par.coupling_method == MarkerIBParams::CouplingExplicitDiag);
  const Real diag_eps = amrex::max(m_par.explicit_diag_eps, amrex::Real(1.0e-30));

  const auto dx = m_geom.CellSizeArray();
  const auto plo = m_geom.ProbLoArray();
  const auto dom = m_geom.Domain();
  const int ilo = dom.smallEnd(0), ihi = dom.bigEnd(0);
  const int jlo = dom.smallEnd(1), jhi = dom.bigEnd(1);
#if (AMREX_SPACEDIM == 3)
  const int klo = dom.smallEnd(2), khi = dom.bigEnd(2);
#else
  const int klo = 0, khi = 0;
#endif
  const int kernel_type = delta_type_to_kernel(m_par.delta_type);

  const Real x0 = m_x0;
  const Real y0 = m_y0;
  const Real dv = m_dv;
#if (AMREX_SPACEDIM == 3)
  const Real z0 = m_z0;
#else
  const Real z0 = amrex::Real(0.0);
#endif
  const Real wall_tol = m_wall_tol;
  const MarkerIBParams par = m_par;
  const int use_custom =
      (m_par.geometry_type == MarkerIBParams::GeometryUserDefined) ? 1 : 0;

  for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
    auto *particles = pti.GetArrayOfStructs().data();
    const Long np = pti.numParticles();

    auto &soa = pti.GetStructOfArrays();
    auto const *uP = soa.GetRealData(U_Marker).data();
    auto const *vP = soa.GetRealData(V_Marker).data();
    auto const *wP = soa.GetRealData(W_Marker).data();
    auto *fxP = soa.GetRealData(Fx_Marker).data();
    auto *fyP = soa.GetRealData(Fy_Marker).data();
    auto *fzP = soa.GetRealData(Fz_Marker).data();
    auto const *txP = soa.GetRealData(Mx_Marker).data();
    auto const *tyP = soa.GetRealData(My_Marker).data();
    auto const *tzP = soa.GetRealData(Mz_Marker).data();

    ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      Real Ux = amrex::Real(0.0);
      Real Uy = amrex::Real(0.0);
      Real Uz = amrex::Real(0.0);
      const Real px = particles[i].pos(0);
      const Real py = particles[i].pos(1);
#if (AMREX_SPACEDIM == 3)
      const Real pz = particles[i].pos(2);
#else
      const Real pz = amrex::Real(0.0);
#endif
      if (use_custom != 0) {
        Ux = txP[i];
        Uy = tyP[i];
        Uz = tzP[i];
      } else {
        rigid_target_velocity(px, py, pz, x0, y0, z0, par, wall_tol, time, Ux,
                              Uy, Uz);
      }

      // IAMReX direct-forcing form: acceleration that drives marker velocity
      // toward the rigid-body target over one physical step.
      // Under-relaxation can improve MDF fixed-point stability at long times.
      Real denom = dt;
      if (use_diag_correction) {
        // Diagonal implicit correction:
        //   M_kk = dv * sum_x delta_h(x-Xk)^2 dV
        // and F_k = (Ub-Uk)/(dt*M_kk).
        const Real dV = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);
        const Real lx = (particles[i].pos(0) - plo[0]) / dx[0];
        const Real ly = (particles[i].pos(1) - plo[1]) / dx[1];
#if (AMREX_SPACEDIM == 3)
        const Real lz = (particles[i].pos(2) - plo[2]) / dx[2];
#else
        const Real lz = amrex::Real(0.0);
#endif
        const int i0 = static_cast<int>(std::floor(lx));
        const int j0 = static_cast<int>(std::floor(ly));
        const int k0 = static_cast<int>(std::floor(lz));
#if (AMREX_SPACEDIM < 3)
        amrex::ignore_unused(k0);
#endif

        Real sum_delta2 = amrex::Real(0.0);
        for (int ii = -2; ii < 3; ++ii) {
          const int ip = i0 + ii;
          if (ip < ilo || ip > ihi) {
            continue;
          }
          const Real xi = plo[0] + (Real(ip) + amrex::Real(0.5)) * dx[0];
          const Real tU = delta_function(xi, particles[i].pos(0), dx[0], kernel_type);
          for (int jj = -2; jj < 3; ++jj) {
            const int jp = j0 + jj;
            if (jp < jlo || jp > jhi) {
              continue;
            }
            const Real yi = plo[1] + (Real(jp) + amrex::Real(0.5)) * dx[1];
            const Real tV = delta_function(yi, particles[i].pos(1), dx[1], kernel_type);
#if (AMREX_SPACEDIM == 3)
            for (int kk = -2; kk < 3; ++kk) {
              const int kp = k0 + kk;
              if (kp < klo || kp > khi) {
                continue;
              }
              const Real zi = plo[2] + (Real(kp) + amrex::Real(0.5)) * dx[2];
              const Real tW = delta_function(zi, particles[i].pos(2), dx[2], kernel_type);
              const Real delta_value = tU * tV * tW;
              sum_delta2 += delta_value * delta_value * dV;
            }
#else
            const Real delta_value = tU * tV;
            sum_delta2 += delta_value * delta_value * dV;
#endif
          }
        }
        const Real mkk = dv * sum_delta2;
        denom = dt * amrex::max(mkk, diag_eps);
      }

      fxP[i] += relax * (Ux - uP[i]) / denom;
      fyP[i] += relax * (Uy - vP[i]) / denom;
      fzP[i] += relax * (Uz - wP[i]) / denom;
    });
  }
}

void IBMarkerDF::gather_marker_state(std::vector<int> &ids,
                                     std::vector<Real> &data,
                                     Real time) const {
  ids.clear();
  data.clear();

  const bool use_box = (m_par.geometry_type == MarkerIBParams::GeometryBox);
  const bool use_custom =
      (m_par.geometry_type == MarkerIBParams::GeometryUserDefined);
  const Real x0 = m_x0;
  const Real y0 = m_y0;
#if (AMREX_SPACEDIM == 3)
  const Real z0 = m_z0;
#else
  const Real z0 = amrex::Real(0.0);
#endif

  for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
    const Long np = pti.numParticles();
    auto &soa = pti.GetStructOfArrays();
    auto const &mid_arr = soa.GetIntData(M_ID);

    auto const &u_arr = soa.GetRealData(U_Marker);
    auto const &v_arr = soa.GetRealData(V_Marker);
    auto const &w_arr = soa.GetRealData(W_Marker);

    std::vector<int> mid_host(static_cast<std::size_t>(np), 0);
    std::vector<Real> u_host(static_cast<std::size_t>(np), Real(0.0));
    std::vector<Real> v_host(static_cast<std::size_t>(np), Real(0.0));
    std::vector<Real> w_host(static_cast<std::size_t>(np), Real(0.0));

    Gpu::copy(Gpu::deviceToHost, mid_arr.begin(), mid_arr.begin() + np,
              mid_host.begin());
    Gpu::copy(Gpu::deviceToHost, u_arr.begin(), u_arr.begin() + np,
              u_host.begin());
    Gpu::copy(Gpu::deviceToHost, v_arr.begin(), v_arr.begin() + np,
              v_host.begin());
    Gpu::copy(Gpu::deviceToHost, w_arr.begin(), w_arr.begin() + np,
              w_host.begin());

    for (Long n = 0; n < np; ++n) {
      const int idx = mid_host[n];
      if (idx <= 0) {
        continue;
      }

      Real px = x0;
      Real py = y0;
#if (AMREX_SPACEDIM == 3)
      Real pz = z0;
#else
      Real pz = amrex::Real(0.0);
#endif

      if (use_box) {
        if (idx < static_cast<int>(m_xref_h.size())) {
          px = m_xref_h[idx];
          py = m_yref_h[idx];
#if (AMREX_SPACEDIM == 3)
          pz = m_zref_h[idx];
#endif
        }
      } else if (use_custom) {
        Real utx = amrex::Real(0.0);
        Real uty = amrex::Real(0.0);
        Real utz = amrex::Real(0.0);
        lbm_user_ibm_geometry::marker_state(
            idx, time, m_par, x0, y0, z0,
            m_xref_h.data(), m_yref_h.data(), m_zref_h.data(),
            px, py, pz, utx, uty, utz);
      } else if (idx < static_cast<int>(m_phiK_h.size()) &&
                 idx < static_cast<int>(m_thetaK_h.size())) {
        const Real phi = m_phiK_h[idx];
        const Real theta = m_thetaK_h[idx];
        px = x0 + m_R * std::sin(theta) * std::cos(phi);
        py = y0 + m_R * std::sin(theta) * std::sin(phi);
#if (AMREX_SPACEDIM == 3)
        pz = z0 + m_R * std::cos(theta);
#endif
      }

      ids.push_back(idx);
      data.push_back(px);
      data.push_back(py);
      data.push_back(pz);
      data.push_back(u_host[n]);
      data.push_back(v_host[n]);
      data.push_back(w_host[n]);
    }
  }
}

void IBMarkerDF::scatter_marker_force(std::vector<Real> const &fx_global,
                                      std::vector<Real> const &fy_global,
                                      std::vector<Real> const &fz_global) const {
  const int nmark = static_cast<int>(fx_global.size());
  if (nmark == 0) {
    return;
  }

  for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
    const Long np = pti.numParticles();
    auto &soa = pti.GetStructOfArrays();
    auto const &mid_arr = soa.GetIntData(M_ID);
    auto &fx_arr = soa.GetRealData(Fx_Marker);
    auto &fy_arr = soa.GetRealData(Fy_Marker);
    auto &fz_arr = soa.GetRealData(Fz_Marker);

    std::vector<int> mid_host(static_cast<std::size_t>(np), 0);
    std::vector<Real> fx_host(static_cast<std::size_t>(np), Real(0.0));
    std::vector<Real> fy_host(static_cast<std::size_t>(np), Real(0.0));
    std::vector<Real> fz_host(static_cast<std::size_t>(np), Real(0.0));

    Gpu::copy(Gpu::deviceToHost, mid_arr.begin(), mid_arr.begin() + np,
              mid_host.begin());
    Gpu::copy(Gpu::deviceToHost, fx_arr.begin(), fx_arr.begin() + np,
              fx_host.begin());
    Gpu::copy(Gpu::deviceToHost, fy_arr.begin(), fy_arr.begin() + np,
              fy_host.begin());
    Gpu::copy(Gpu::deviceToHost, fz_arr.begin(), fz_arr.begin() + np,
              fz_host.begin());

    for (Long n = 0; n < np; ++n) {
      const int idx = mid_host[n] - 1;
      if (idx >= 0 && idx < nmark) {
        fx_host[n] += fx_global[idx];
        fy_host[n] += fy_global[idx];
        fz_host[n] += fz_global[idx];
      }
    }

    Gpu::copy(Gpu::hostToDevice, fx_host.begin(), fx_host.end(),
              fx_arr.begin());
    Gpu::copy(Gpu::hostToDevice, fy_host.begin(), fy_host.end(),
              fy_arr.begin());
    Gpu::copy(Gpu::hostToDevice, fz_host.begin(), fz_host.end(),
              fz_arr.begin());
  }
}

bool IBMarkerDF::lu_factorize(std::vector<Real> &a, std::vector<int> &pivot,
                              int n, Real tiny) {
  pivot.assign(n, 0);
  for (int k = 0; k < n; ++k) {
    int p = k;
    Real maxv = std::abs(a[k * n + k]);
    for (int i = k + 1; i < n; ++i) {
      const Real v = std::abs(a[i * n + k]);
      if (v > maxv) {
        maxv = v;
        p = i;
      }
    }
    if (maxv <= tiny) {
      return false;
    }

    pivot[k] = p;
    if (p != k) {
      for (int j = 0; j < n; ++j) {
        std::swap(a[k * n + j], a[p * n + j]);
      }
    }

    const Real akk = a[k * n + k];
    for (int i = k + 1; i < n; ++i) {
      a[i * n + k] /= akk;
      const Real lik = a[i * n + k];
      for (int j = k + 1; j < n; ++j) {
        a[i * n + j] -= lik * a[k * n + j];
      }
    }
  }
  return true;
}

void IBMarkerDF::lu_solve(std::vector<Real> const &a,
                          std::vector<int> const &pivot,
                          std::vector<Real> const &rhs,
                          std::vector<Real> &x, int n) {
  x = rhs;

  for (int k = 0; k < n; ++k) {
    const int p = pivot[k];
    if (p != k) {
      std::swap(x[k], x[p]);
    }
  }

  for (int i = 1; i < n; ++i) {
    Real sum = x[i];
    for (int j = 0; j < i; ++j) {
      sum -= a[i * n + j] * x[j];
    }
    x[i] = sum;
  }

  for (int i = n - 1; i >= 0; --i) {
    Real sum = x[i];
    for (int j = i + 1; j < n; ++j) {
      sum -= a[i * n + j] * x[j];
    }
    x[i] = sum / a[i * n + i];
  }
}

void IBMarkerDF::build_ivc_operator(std::vector<Real> const &x_global,
                                    std::vector<Real> const &y_global,
                                    std::vector<Real> const &z_global) const {
  const int nmark = static_cast<int>(x_global.size());
  m_ivc_operator_ready = false;
  m_ivc_nmark = nmark;

  if (nmark <= 0) {
    return;
  }

  const auto dx = m_geom.CellSizeArray();
  const auto plo = m_geom.ProbLoArray();
  const auto dom = m_geom.Domain();
  const int ilo = dom.smallEnd(0), ihi = dom.bigEnd(0);
  const int jlo = dom.smallEnd(1), jhi = dom.bigEnd(1);
#if (AMREX_SPACEDIM == 3)
  const int klo = dom.smallEnd(2), khi = dom.bigEnd(2);
#else
  const int klo = 0, khi = 0;
#endif

  const int kernel_type = delta_type_to_kernel(m_par.delta_type);

  m_ivc_lu.assign(static_cast<std::size_t>(nmark) * static_cast<std::size_t>(nmark),
                  amrex::Real(0.0));

  for (int i = 0; i < nmark; ++i) {
    for (int j = i; j < nmark; ++j) {
      const Real sx = overlap_1d(x_global[i], x_global[j], dx[0], kernel_type,
                                 plo[0], ilo, ihi);
      const Real sy = overlap_1d(y_global[i], y_global[j], dx[1], kernel_type,
                                 plo[1], jlo, jhi);
#if (AMREX_SPACEDIM == 3)
      const Real sz = overlap_1d(z_global[i], z_global[j], dx[2], kernel_type,
                                 plo[2], klo, khi);
#else
      const Real sz = amrex::Real(1.0);
#endif
      Real val = m_dv * sx * sy * sz;
      if (i == j) {
        val += m_par.ivc_diag_reg;
      }
      m_ivc_lu[i * nmark + j] = val;
      m_ivc_lu[j * nmark + i] = val;
    }
  }

  const Real tiny = amrex::max(m_par.ivc_diag_reg * amrex::Real(1.0e-4),
                               amrex::Real(1.0e-30));
  m_ivc_operator_ready = lu_factorize(m_ivc_lu, m_ivc_pivot, nmark, tiny);
  m_ivc_x_cached = x_global;
  m_ivc_y_cached = y_global;
  m_ivc_z_cached = z_global;

  if (m_par.ivc_verbose > 0 && ParallelDescriptor::IOProcessor()) {
    amrex::Print() << "  [ibm_marker_ivc] operator build n=" << nmark
                   << " kernel=" << ((kernel_type == 1) ? "3pt" : "4pt")
                   << " diag_reg=" << m_par.ivc_diag_reg
                   << " ready=" << (m_ivc_operator_ready ? 1 : 0) << "\n";
  }
}

void IBMarkerDF::compute_lagrangian_force_ivc(Real dt, Real time) const {
  const int nmark = static_cast<int>(num_markers());
  if (nmark <= 0) {
    return;
  }

  if (m_par.renormalize_delta != 0 && m_par.ivc_verbose > 0 &&
      ParallelDescriptor::IOProcessor()) {
    amrex::Print() << "  [ibm_marker_ivc] warning: renormalized delta is disabled"
                   << " for IVC consistency; set ibm.delta_renorm=0.\n";
  }

  std::vector<int> ids_local;
  std::vector<Real> marker_local;
  gather_marker_state(ids_local, marker_local, time);

  const int local_n = static_cast<int>(ids_local.size());
  const int nprocs = ParallelDescriptor::NProcs();
  std::vector<int> counts(nprocs, 0);
  std::vector<int> displs(nprocs, 0);
  std::vector<int> counts6(nprocs, 0);
  std::vector<int> displs6(nprocs, 0);

#ifdef AMREX_USE_MPI
  MPI_Allgather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT,
                ParallelDescriptor::Communicator());
#else
  counts[0] = local_n;
#endif

  int total_n = 0;
  for (int p = 0; p < nprocs; ++p) {
    displs[p] = total_n;
    total_n += counts[p];
    counts6[p] = counts[p] * 6;
    displs6[p] = displs[p] * 6;
  }

  std::vector<int> ids_all(total_n, 0);
  std::vector<Real> marker_all(static_cast<std::size_t>(total_n) * 6U,
                               amrex::Real(0.0));

#ifdef AMREX_USE_MPI
  MPI_Allgatherv(ids_local.data(), local_n, MPI_INT, ids_all.data(), counts.data(),
                 displs.data(), MPI_INT, ParallelDescriptor::Communicator());
  MPI_Allgatherv(marker_local.data(), local_n * 6,
                 ParallelDescriptor::Mpi_typemap<Real>::type(),
                 marker_all.data(), counts6.data(), displs6.data(),
                 ParallelDescriptor::Mpi_typemap<Real>::type(),
                 ParallelDescriptor::Communicator());
#else
  ids_all = ids_local;
  marker_all = marker_local;
#endif

  if (total_n != nmark && ParallelDescriptor::IOProcessor()) {
    amrex::Print() << "  [ibm_marker_ivc] warning: gathered markers=" << total_n
                   << " expected=" << nmark << "\n";
  }

  std::vector<Real> x_global(nmark, amrex::Real(0.0));
  std::vector<Real> y_global(nmark, amrex::Real(0.0));
  std::vector<Real> z_global(nmark, amrex::Real(0.0));
  std::vector<Real> u_global(nmark, amrex::Real(0.0));
  std::vector<Real> v_global(nmark, amrex::Real(0.0));
  std::vector<Real> w_global(nmark, amrex::Real(0.0));
  std::vector<int> marker_id_global(static_cast<std::size_t>(nmark), 0);

  for (int n = 0; n < total_n; ++n) {
    const int idx = ids_all[n] - 1;
    if (idx < 0 || idx >= nmark) {
      continue;
    }
    const int o = 6 * n;
    x_global[idx] = marker_all[o + 0];
    y_global[idx] = marker_all[o + 1];
    z_global[idx] = marker_all[o + 2];
    u_global[idx] = marker_all[o + 3];
    v_global[idx] = marker_all[o + 4];
    w_global[idx] = marker_all[o + 5];
    marker_id_global[idx] = ids_all[n];
  }

  for (int i = 0; i < nmark; ++i) {
    if (marker_id_global[i] <= 0) {
      marker_id_global[i] = i + 1;
    }
  }

  bool geometry_changed =
      (static_cast<int>(m_ivc_x_cached.size()) != nmark) ||
      (static_cast<int>(m_ivc_y_cached.size()) != nmark) ||
      (static_cast<int>(m_ivc_z_cached.size()) != nmark);
  if (!geometry_changed) {
    constexpr Real rebuild_tol = amrex::Real(1.0e-12);
    for (int i = 0; i < nmark; ++i) {
      if (amrex::Math::abs(x_global[i] - m_ivc_x_cached[i]) > rebuild_tol ||
          amrex::Math::abs(y_global[i] - m_ivc_y_cached[i]) > rebuild_tol ||
          amrex::Math::abs(z_global[i] - m_ivc_z_cached[i]) > rebuild_tol) {
        geometry_changed = true;
        break;
      }
    }
  }
  if (!m_ivc_operator_ready || m_ivc_nmark != nmark ||
      m_par.ivc_rebuild_matrix > 0 || geometry_changed) {
    build_ivc_operator(x_global, y_global, z_global);
  }

  if (!m_ivc_operator_ready) {
    if (ParallelDescriptor::IOProcessor()) {
      amrex::Print() << "  [ibm_marker_ivc] LU factorization failed; falling back"
                     << " to explicit direct forcing.\n";
    }
    compute_lagrangian_force_explicit(dt, time);
    return;
  }

  std::vector<Real> fx_global(nmark, amrex::Real(0.0));
  std::vector<Real> fy_global(nmark, amrex::Real(0.0));
  std::vector<Real> fz_global(nmark, amrex::Real(0.0));

  if (ParallelDescriptor::IOProcessor()) {
    std::vector<Real> bx(nmark, amrex::Real(0.0));
    std::vector<Real> by(nmark, amrex::Real(0.0));
    std::vector<Real> bz(nmark, amrex::Real(0.0));

    const Real wall_tol = m_wall_tol;

    const bool use_custom =
        (m_par.geometry_type == MarkerIBParams::GeometryUserDefined);

    for (int i = 0; i < nmark; ++i) {
      Real Ux = amrex::Real(0.0);
      Real Uy = amrex::Real(0.0);
      Real Uz = amrex::Real(0.0);
      if (use_custom) {
        Real px = m_x0;
        Real py = m_y0;
        Real pz = m_z0;
        lbm_user_ibm_geometry::marker_state(
            marker_id_global[i], time, m_par, m_x0, m_y0, m_z0, m_xref_h.data(),
            m_yref_h.data(), m_zref_h.data(), px, py, pz, Ux, Uy, Uz);
      } else {
        rigid_target_velocity(x_global[i], y_global[i], z_global[i], m_x0, m_y0,
                              m_z0, m_par, wall_tol, time, Ux, Uy, Uz);
      }

      // Match explicit direct forcing scaling for the IVC solve.
      bx[i] = (Ux - u_global[i]) / dt;
      by[i] = (Uy - v_global[i]) / dt;
      bz[i] = (Uz - w_global[i]) / dt;
    }

    lu_solve(m_ivc_lu, m_ivc_pivot, bx, fx_global, nmark);
    lu_solve(m_ivc_lu, m_ivc_pivot, by, fy_global, nmark);
    lu_solve(m_ivc_lu, m_ivc_pivot, bz, fz_global, nmark);

    if (m_par.ivc_verbose > 0) {
      Real fmax = 0.0;
      for (int i = 0; i < nmark; ++i) {
        const Real fmag = std::sqrt(fx_global[i] * fx_global[i] +
                                    fy_global[i] * fy_global[i] +
                                    fz_global[i] * fz_global[i]);
        fmax = amrex::max(fmax, fmag);
      }
      amrex::Print() << "  [ibm_marker_ivc] solved n=" << nmark
                     << " max|F|=" << fmax << "\n";
    }
  }

#ifdef AMREX_USE_MPI
  MPI_Bcast(fx_global.data(), nmark, ParallelDescriptor::Mpi_typemap<Real>::type(),
            ParallelDescriptor::IOProcessorNumber(),
            ParallelDescriptor::Communicator());
  MPI_Bcast(fy_global.data(), nmark, ParallelDescriptor::Mpi_typemap<Real>::type(),
            ParallelDescriptor::IOProcessorNumber(),
            ParallelDescriptor::Communicator());
  MPI_Bcast(fz_global.data(), nmark, ParallelDescriptor::Mpi_typemap<Real>::type(),
            ParallelDescriptor::IOProcessorNumber(),
            ParallelDescriptor::Communicator());
#endif

  scatter_marker_force(fx_global, fy_global, fz_global);
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void force_spreading_cic(
    P const &p, Real fxP, Real fyP, Real fzP, Array4<Real> const &E,
    int EulerFIndex, GpuArray<Real, AMREX_SPACEDIM> const &plo,
    GpuArray<Real, AMREX_SPACEDIM> const &dx, int type,
    int ilo, int ihi, int jlo, int jhi, int klo, int khi,
    bool renormalize_delta, Real* partition_out) noexcept {
#if (AMREX_SPACEDIM < 3)
  amrex::ignore_unused(fzP);
#endif
  const Real lx = (p.pos(0) - plo[0]) / dx[0];
  const Real ly = (p.pos(1) - plo[1]) / dx[1];
#if (AMREX_SPACEDIM == 3)
  const Real lz = (p.pos(2) - plo[2]) / dx[2];
#else
  const Real lz = amrex::Real(0.0);
#endif

  const int i0 = static_cast<int>(std::floor(lx));
  const int j0 = static_cast<int>(std::floor(ly));
  const int k0 = static_cast<int>(std::floor(lz));
#if (AMREX_SPACEDIM < 3)
  amrex::ignore_unused(k0);
#endif

  const Real d = AMREX_D_TERM(dx[0], * dx[1], * dx[2]);
  Real part = amrex::Real(0.0);

  // First pass: compute kernel partition (needed for optional renormalization)
  // IAMReX DiffusedIB style: fixed 5-point traversal in each dimension.
  for (int ii = -2; ii < 3; ++ii) {
    const Real xi = plo[0] + (Real(i0 + ii) + amrex::Real(0.5)) * dx[0];
    const Real tU = delta_function(xi, p.pos(0), dx[0], type);

    for (int jj = -2; jj < 3; ++jj) {
      const Real yi = plo[1] + (Real(j0 + jj) + amrex::Real(0.5)) * dx[1];
      const Real tV = delta_function(yi, p.pos(1), dx[1], type);

#if (AMREX_SPACEDIM == 3)
      for (int kk = -2; kk < 3; ++kk) {
        const Real zi = plo[2] + (Real(k0 + kk) + amrex::Real(0.5)) * dx[2];
        const Real tW = delta_function(zi, p.pos(2), dx[2], type);

        const int ip = i0 + ii;
        const int jp = j0 + jj;
        const int kp = k0 + kk;
        if (ip < ilo || ip > ihi || jp < jlo || jp > jhi || kp < klo ||
            kp > khi) {
          continue;
        }
        const Real delta_value = tU * tV * tW;
        part += delta_value * d;
      }
#else
      const Real delta_value = tU * tV;
      const int ip = i0 + ii;
      const int jp = j0 + jj;
      const int kp = 0;
      if (ip < ilo || ip > ihi || jp < jlo || jp > jhi || kp < klo ||
          kp > khi) {
        continue;
      }
      part += delta_value * d;
#endif
    }
  }

  if (partition_out != nullptr) {
    *partition_out = part;
  }
  Real scale = amrex::Real(1.0);
  if (renormalize_delta && part > amrex::Real(0.0)) {
    scale = amrex::Real(1.0) / part;
  }

  // Second pass: spread with optional renormalization
  for (int ii = -2; ii < 3; ++ii) {
    const Real xi = plo[0] + (Real(i0 + ii) + amrex::Real(0.5)) * dx[0];
    const Real tU = delta_function(xi, p.pos(0), dx[0], type);

    for (int jj = -2; jj < 3; ++jj) {
      const Real yi = plo[1] + (Real(j0 + jj) + amrex::Real(0.5)) * dx[1];
      const Real tV = delta_function(yi, p.pos(1), dx[1], type);

#if (AMREX_SPACEDIM == 3)
      for (int kk = -2; kk < 3; ++kk) {
        const Real zi = plo[2] + (Real(k0 + kk) + amrex::Real(0.5)) * dx[2];
        const Real tW = delta_function(zi, p.pos(2), dx[2], type);

        const int ip = i0 + ii;
        const int jp = j0 + jj;
        const int kp = k0 + kk;
        if (ip < ilo || ip > ihi || jp < jlo || jp > jhi || kp < klo ||
            kp > khi) {
          continue;
        }
        const Real delta_value = (tU * tV * tW) * scale;
        Gpu::Atomic::AddNoRet(&E(ip, jp, kp, EulerFIndex + 0),
                              delta_value * fxP);
        Gpu::Atomic::AddNoRet(&E(ip, jp, kp, EulerFIndex + 1),
                              delta_value * fyP);
        Gpu::Atomic::AddNoRet(&E(ip, jp, kp, EulerFIndex + 2),
                              delta_value * fzP);
      }
#else
      const Real delta_value = (tU * tV) * scale;
      const int ip = i0 + ii;
      const int jp = j0 + jj;
      const int kp = 0;
      if (ip < ilo || ip > ihi || jp < jlo || jp > jhi || kp < klo ||
          kp > khi) {
        continue;
      }
      Gpu::Atomic::AddNoRet(&E(ip, jp, 0, EulerFIndex + 0), delta_value * fxP);
      Gpu::Atomic::AddNoRet(&E(ip, jp, 0, EulerFIndex + 1), delta_value * fyP);
#endif
    }
  }
}

void IBMarkerDF::force_spreading(MultiFab &EulerForce, int euler_force_comp,
                                 int delta_type) const {
  const Real dv = m_dv;
  const Real x0 = m_x0;
  const Real y0 = m_y0;
#if (AMREX_SPACEDIM == 3)
  const Real z0 = m_z0;
#endif

  const auto dx = m_geom.CellSizeArray();
  const auto plo = m_geom.ProbLoArray();

  for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
    auto *particles = pti.GetArrayOfStructs().data();
    const Long np = pti.numParticles();

    auto &soa = pti.GetStructOfArrays();
    auto *fxP = soa.GetRealData(Fx_Marker).data();
    auto *fyP = soa.GetRealData(Fy_Marker).data();
    auto *fzP = soa.GetRealData(Fz_Marker).data();
    auto *mxP = soa.GetRealData(Mx_Marker).data();
    auto *myP = soa.GetRealData(My_Marker).data();
    auto *mzP = soa.GetRealData(Mz_Marker).data();

    auto const &E = EulerForce[pti].array();
    const Box ebx = EulerForce[pti].box();
    const int ilo = ebx.smallEnd(0), ihi = ebx.bigEnd(0);
    const int jlo = ebx.smallEnd(1), jhi = ebx.bigEnd(1);
#if (AMREX_SPACEDIM == 3)
    const int klo = ebx.smallEnd(2), khi = ebx.bigEnd(2);
#else
    const int klo = 0, khi = 0;
#endif
    const int kernel_type = delta_type_to_kernel(delta_type);
    const bool renorm = (m_par.renormalize_delta != 0);

    ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      // IAMReX multiplies by dv before spreading.
      Real fx = fxP[i] * dv;
      Real fy = fyP[i] * dv;
      Real fz = fzP[i] * dv;

      const Real rx = particles[i].pos(0) - x0;
      const Real ry = particles[i].pos(1) - y0;
#if (AMREX_SPACEDIM == 3)
      const Real rz = particles[i].pos(2) - z0;
#else
      const Real rz = amrex::Real(0.0);
#endif

      // Store moment about body center (r x F).
      mxP[i] = ry * fz - rz * fy;
      myP[i] = rz * fx - rx * fz;
      mzP[i] = rx * fy - ry * fx;

      force_spreading_cic(particles[i], fx, fy, fz, E, euler_force_comp, plo,
                          dx, kernel_type, ilo, ihi, jlo, jhi, klo, khi,
                          renorm, nullptr);
    });
  }

  EulerForce.SumBoundary(euler_force_comp, 3, m_geom.periodicity());
}

void IBMarkerDF::velocity_correction(MultiFab &EulerVel,
                                     MultiFab const &EulerForce,
                                     int euler_vel_comp, int euler_force_comp,
                                     Real dt) const {
  MultiFab::Saxpy(EulerVel, dt, EulerForce, euler_force_comp, euler_vel_comp, 3,
                  0);
}

void IBMarkerDF::update_forcing(MultiFab const &rhocc, MultiFab const &ucc,
                                MultiFab const &vcc, MultiFab const &wcc,
                                MultiFab &Fx, MultiFab &Fy, MultiFab &Fz,
                                Real dt, Real time, MultiFab const *prev_Fx,
                                MultiFab const *prev_Fy,
                                MultiFab const *prev_Fz) const {
  // Build Euler velocity scratch: (u,v,w)
  // IMPORTANT: We need at least two ghost cells for the 3-pt/4-pt delta
  // stencil. Copy as much of (u,v,w) as available, including overlap with
  // source ghost cells, then fill remaining periodic ghosts.
  constexpr int ng = 2;
  MultiFab EulerVel(ucc.boxArray(), ucc.DistributionMap(), 3, ng);
  MultiFab EulerForceIter(ucc.boxArray(), ucc.DistributionMap(), 3, ng);
  MultiFab EulerForceTotal(ucc.boxArray(), ucc.DistributionMap(), 3, ng);

  EulerVel.setVal(0.0);
  EulerForceIter.setVal(0.0);
  EulerForceTotal.setVal(0.0);

  for (MFIter mfi(EulerVel, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    auto const u = ucc[mfi].const_array();
    auto const v = vcc[mfi].const_array();
    auto const w = wcc[mfi].const_array();
    auto ev = EulerVel[mfi].array();
    const bool has_prev_force =
        (prev_Fx != nullptr && prev_Fy != nullptr && prev_Fz != nullptr);
    amrex::ignore_unused(has_prev_force);

    Box cpy = mfi.fabbox() & ucc[mfi].box() & vcc[mfi].box();
#if (AMREX_SPACEDIM == 3)
    cpy &= wcc[mfi].box();
#endif
    ParallelFor(cpy, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      ev(i, j, k, 0) = u(i, j, k, 0);
      ev(i, j, k, 1) = v(i, j, k, 0);
#if (AMREX_SPACEDIM == 3)
      ev(i, j, k, 2) = w(i, j, k, 0);
#else
      ev(i, j, k, 2) = amrex::Real(0.0);
#endif
    });
  }

  // Fill ghost cells before the first interpolation.
  EulerVel.FillBoundary(0, 3, m_geom.periodicity());

  // Reset per-call force accumulator.
  m_last_marker_force_sum[0] = amrex::Real(0.0);
  m_last_marker_force_sum[1] = amrex::Real(0.0);
  m_last_marker_force_sum[2] = amrex::Real(0.0);

  const bool use_ivc = (m_par.coupling_method == MarkerIBParams::CouplingIVC);
  // Explicit direct forcing uses IAMReX loop_ns iterations. IVC solves the
  // marker force implicitly once per step.
  const int nloop = use_ivc ? 1 : std::max(m_par.loop_ns, 1);
  for (int it = 0; it < nloop; ++it) {
    EulerForceIter.setVal(0.0);

    update_lagrangian_marker(time);
    velocity_interpolation(EulerVel, 0, m_par.delta_type);

    // Optional diagnostics: kernel partition (should be ~1) and slip at markers
    // BEFORE applying the forcing for this MDF iteration.
    if (m_par.debug_kernel_partition > 0) {
#ifndef AMREX_USE_GPU
      const auto plo = m_geom.ProbLoArray();
      const auto dx = m_geom.CellSizeArray();
      const auto dom = m_geom.Domain();
      const int ilo = dom.smallEnd(0), ihi = dom.bigEnd(0);
      const int jlo = dom.smallEnd(1), jhi = dom.bigEnd(1);
#if (AMREX_SPACEDIM == 3)
      const int klo = dom.smallEnd(2), khi = dom.bigEnd(2);
#else
      const int klo = 0, khi = 0;
#endif
      const int kernel_type = delta_type_to_kernel(m_par.delta_type);
      amrex::Real pmin = 1.0e200, pmax = -1.0e200, psum = 0.0;
      amrex::Long pn = 0;

      // NOTE: marker particles are stored in MarkerContainer (see IBMarkerDF.H)
      auto part_for = [=] AMREX_GPU_HOST_DEVICE (const MarkerContainer::ParticleType& p) noexcept {
        const Real x1 = p.pos(0);
        const Real x2 = p.pos(1);
#if (AMREX_SPACEDIM == 3)
        const Real x3 = p.pos(2);
#else
        const Real x3 = 0.0;
#endif
        const Real dV = AMREX_D_TERM(dx[0], * dx[1], * dx[2]);
        int ic = static_cast<int>(amrex::Math::floor((x1 - plo[0]) / dx[0]));
        int jc = static_cast<int>(amrex::Math::floor((x2 - plo[1]) / dx[1]));
#if (AMREX_SPACEDIM == 3)
        int kc = static_cast<int>(amrex::Math::floor((x3 - plo[2]) / dx[2]));
#else
        int kc = 0;
#endif
        int i0 = ic - 1;
        int i1 = (kernel_type == 1) ? (ic + 1) : (ic + 2);
        int j0 = jc - 1;
        int j1 = (kernel_type == 1) ? (jc + 1) : (jc + 2);
#if (AMREX_SPACEDIM == 3)
        int k0 = kc - 1;
        int k1 = (kernel_type == 1) ? (kc + 1) : (kc + 2);
#else
        int k0 = 0;
        int k1 = 0;
#endif
        i0 = amrex::max(i0, ilo);
        i1 = amrex::min(i1, ihi);
        j0 = amrex::max(j0, jlo);
        j1 = amrex::min(j1, jhi);
#if (AMREX_SPACEDIM == 3)
        k0 = amrex::max(k0, klo);
        k1 = amrex::min(k1, khi);
#endif

        Real part = 0.0;
        for (int i = i0; i <= i1; ++i) {
          // delta_function signature: (xf, xp, h, type)
          // Here: xf = marker coord, xp = cell-center coord.
          const Real tU = delta_function(x1, (plo[0] + (i + 0.5) * dx[0]), dx[0], kernel_type);
          for (int j = j0; j <= j1; ++j) {
            const Real tV = delta_function(x2, (plo[1] + (j + 0.5) * dx[1]), dx[1], kernel_type);
#if (AMREX_SPACEDIM == 3)
            for (int k = k0; k <= k1; ++k) {
              const Real tW = delta_function(x3, (plo[2] + (k + 0.5) * dx[2]), dx[2], kernel_type);
              part += (tU * tV * tW) * dV;
            }
#else
            part += (tU * tV) * dV;
#endif
          }
        }
        return part;
      };

      for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
        const auto& aos = pti.GetArrayOfStructs();
        const int np = pti.numParticles();
        for (int n = 0; n < np; ++n) {
          const amrex::Real part = part_for(aos[n]);
          pmin = amrex::min(pmin, part);
          pmax = amrex::max(pmax, part);
          psum += part;
          ++pn;
        }
      }

      amrex::ParallelDescriptor::ReduceRealMin(pmin);
      amrex::ParallelDescriptor::ReduceRealMax(pmax);
      amrex::ParallelDescriptor::ReduceRealSum(psum);
      amrex::ParallelDescriptor::ReduceLongSum(pn);

      if (amrex::ParallelDescriptor::IOProcessor()) {
        const amrex::Real pmean = (pn > 0) ? (psum / static_cast<amrex::Real>(pn)) : 0.0;
        amrex::Print() << "  [ibm_marker_df] it=" << it
                       << " delta_partition(min/mean/max)=(" << pmin << "," << pmean << "," << pmax << ")"
                       << " kernel=" << ((kernel_type == 1) ? "3pt" : "4pt")
                       << " renorm=" << ((m_par.renormalize_delta != 0) ? 1 : 0)
                       << " n=" << pn << "\n";
      }
#else
      if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  [ibm_marker_df] it=" << it
                       << " (kernel partition diagnostics disabled on GPU build)\n";
      }
#endif
    }

    if (m_par.verbose > 0) {
	#ifndef AMREX_USE_GPU
		amrex::Real slip_max = 0.0;
		amrex::Real slip_sum = 0.0;
		amrex::Long slip_n = 0;

		// NOTE: For the current debug path (rigid, stationary cylinder), the
		// desired marker velocity is zero everywhere. If/when we add moving
		// rigid bodies, replace (ud,vd,wd) with rigid-body kinematics.
		for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
				auto& soa = pti.GetStructOfArrays();
				const auto& u_arr  = soa.GetRealData(U_Marker);
				const auto& v_arr  = soa.GetRealData(V_Marker);
				const auto& w_arr  = soa.GetRealData(W_Marker);
				const auto& fx_arr = soa.GetRealData(Fx_Marker);
				const auto& fy_arr = soa.GetRealData(Fy_Marker);
				const auto& fz_arr = soa.GetRealData(Fz_Marker);
				const int np = pti.numParticles();
				for (int n = 0; n < np; ++n) {
					const amrex::Real u = u_arr[n];
					const amrex::Real v = v_arr[n];
					const amrex::Real w = w_arr[n];
					const amrex::Real umag = std::sqrt(u*u + v*v + w*w);
					slip_max = std::max(slip_max, umag);
					slip_sum += umag;
					if (umag > 1.0e-12) ++slip_n;
				}
			}
		amrex::ParallelDescriptor::ReduceRealMax(slip_max);
		amrex::ParallelDescriptor::ReduceRealSum(slip_sum);
		amrex::ParallelDescriptor::ReduceLongSum(slip_n);

		if (amrex::ParallelDescriptor::IOProcessor()) {
			const amrex::Real slip_mean = (slip_n > 0) ? (slip_sum / static_cast<amrex::Real>(slip_n)) : 0.0;
					amrex::Print() << "  [ibm_marker_df] it=" << it
					               << " slip_max=" << slip_max
					               << " slip_mean=" << slip_mean << "\n";
		}
		#else
			if (amrex::ParallelDescriptor::IOProcessor()) {
				amrex::Print() << "  [ibm_marker_df] it=" << it
				               << " (marker slip diagnostics disabled on GPU build)\n";
			}
		#endif
    }

    if (m_par.verbose > 0 &&
        m_par.geometry_type == MarkerIBParams::GeometryBox) {
#ifndef AMREX_USE_GPU
      amrex::Real y_min = std::numeric_limits<amrex::Real>::max();
      amrex::Real y_max = std::numeric_limits<amrex::Real>::lowest();
      amrex::Long n_lid = 0;
      amrex::Long n_tot = 0;
      for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
        auto const *particles = pti.GetArrayOfStructs().data();
        const auto np2 = pti.numParticles();
        for (int n = 0; n < np2; ++n) {
          const amrex::Real y = particles[n].pos(1);
          y_min = amrex::min(y_min, y);
          y_max = amrex::max(y_max, y);
          if (amrex::Math::abs(y - m_par.box_yhi) <= m_wall_tol) {
            ++n_lid;
          }
          ++n_tot;
        }
      }
      amrex::ParallelDescriptor::ReduceRealMin(y_min);
      amrex::ParallelDescriptor::ReduceRealMax(y_max);
      amrex::ParallelDescriptor::ReduceLongSum(n_lid);
      amrex::ParallelDescriptor::ReduceLongSum(n_tot);
      if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  [ibm_marker_df] it=" << it
                       << " box_y_range=(" << y_min << "," << y_max << ")"
                       << " lid_markers=" << n_lid << "/" << n_tot
                       << " box_yhi=" << m_par.box_yhi
                       << " wall_tol=" << m_wall_tol
                       << " lid_ux=" << m_par.box_lid_ux << "\n";
      }
#endif
    }

    if (use_ivc) {
      compute_lagrangian_force_ivc(dt, time);
    } else {
      compute_lagrangian_force_explicit(dt, time);
    }
    force_spreading(EulerForceIter, 0, m_par.delta_type);

    // IAMR-consistent force extraction path: accumulate marker integral
    // sum(F_marker * dv), where F_marker is the marker acceleration.
    {
      amrex::Real sx = 0.0, sy = 0.0, sz = 0.0;
      for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
        const auto &ptile = pti.GetParticleTile();
        const auto np2 = ptile.numParticles();
        const auto &soa = ptile.GetStructOfArrays();
        auto const &fx_arr = soa.GetRealData(Fx_Marker);
        auto const &fy_arr = soa.GetRealData(Fy_Marker);
        auto const &fz_arr = soa.GetRealData(Fz_Marker);
#ifdef AMREX_USE_GPU
        std::vector<amrex::Real> fx_host(np2);
        std::vector<amrex::Real> fy_host(np2);
        std::vector<amrex::Real> fz_host(np2);
        Gpu::copy(Gpu::deviceToHost, fx_arr.begin(), fx_arr.begin() + np2,
                  fx_host.begin());
        Gpu::copy(Gpu::deviceToHost, fy_arr.begin(), fy_arr.begin() + np2,
                  fy_host.begin());
        Gpu::copy(Gpu::deviceToHost, fz_arr.begin(), fz_arr.begin() + np2,
                  fz_host.begin());
        for (Long n = 0; n < np2; ++n) {
          sx += fx_host[n] * m_dv;
          sy += fy_host[n] * m_dv;
          sz += fz_host[n] * m_dv;
        }
#else
        for (int n = 0; n < np2; ++n) {
          sx += fx_arr[n] * m_dv;
          sy += fy_arr[n] * m_dv;
          sz += fz_arr[n] * m_dv;
        }
#endif
      }
      amrex::ParallelDescriptor::ReduceRealSum(sx);
      amrex::ParallelDescriptor::ReduceRealSum(sy);
      amrex::ParallelDescriptor::ReduceRealSum(sz);
      m_last_marker_force_sum[0] += sx;
      m_last_marker_force_sum[1] += sy;
      m_last_marker_force_sum[2] += sz;
    }

	    // Global force balance diagnostics: compare Lagrangian sum(F*dv) to
	    // Eulerian sum(f*dV) for the *current MDF iteration*.
	    if (m_par.debug_force_balance > 0) {
	    #ifndef AMREX_USE_GPU
	      amrex::Real m_fx = 0.0, m_fy = 0.0, m_fz = 0.0;
	      {
                // Iterate over marker particles at level 0.
                // (IBMarkerDF itself is not a ParticleContainer.)
                for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
	          const auto &ptile = pti.GetParticleTile();
	          const auto np2 = ptile.numParticles();
	          const auto &soa = ptile.GetStructOfArrays();
	          auto const &fx_arr = soa.GetRealData(Fx_Marker);
	          auto const &fy_arr = soa.GetRealData(Fy_Marker);
	          auto const &fz_arr = soa.GetRealData(Fz_Marker);
	          for (int n = 0; n < np2; ++n) {
	            m_fx += fx_arr[n] * m_dv;
	            m_fy += fy_arr[n] * m_dv;
	            m_fz += fz_arr[n] * m_dv;
	          }
	        }
	      }
	      amrex::ParallelDescriptor::ReduceRealSum(m_fx);
	      amrex::ParallelDescriptor::ReduceRealSum(m_fy);
	      amrex::ParallelDescriptor::ReduceRealSum(m_fz);

	      const auto dxloc = m_geom.CellSizeArray();
	      const amrex::Real dV = AMREX_D_TERM(dxloc[0], * dxloc[1], * dxloc[2]);
	      amrex::Real e_fx = 0.0, e_fy = 0.0, e_fz = 0.0;
	      for (amrex::MFIter mfi(EulerForceIter, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	        const amrex::Box& bx = mfi.tilebox();
	        auto ef = EulerForceIter[mfi].const_array();
	        amrex::Loop(bx, [&](int i, int j, int k) noexcept {
	          e_fx += ef(i, j, k, 0) * dV;
	          e_fy += ef(i, j, k, 1) * dV;
	          e_fz += ef(i, j, k, 2) * dV;
	        });
	      }
	      amrex::ParallelDescriptor::ReduceRealSum(e_fx);
	      amrex::ParallelDescriptor::ReduceRealSum(e_fy);
	      amrex::ParallelDescriptor::ReduceRealSum(e_fz);

	      if (amrex::ParallelDescriptor::IOProcessor()) {
	        amrex::Print() << "  [ibm_marker_df] it=" << it
	                       << " force_balance marker(sum(F*dv))=(" << m_fx << "," << m_fy << "," << m_fz << ")"
	                       << " euler(sum(f*dV))=(" << e_fx << "," << e_fy << "," << e_fz << ")\n";
	      }
	    #else
	      if (amrex::ParallelDescriptor::IOProcessor()) {
	        amrex::Print() << "  [ibm_marker_df] it=" << it
	                       << " (force balance diagnostics disabled on GPU build)\n";
	      }
	    #endif
	    }

    // Velocity correction is applied in-loop, and we accumulate total Eulerian
    // force.
    velocity_correction(EulerVel, EulerForceIter, 0, 0, dt);
    MultiFab::Add(EulerForceTotal, EulerForceIter, 0, 0, 3, 0);
  }

  // Export to your existing (Fx,Fy,Fz) convention.
  // IMPORTANT (LBM coupling): Your LBM kernels are advanced with an implicit
  // lattice time step of 1.0 (dt does not enter collide/stream). The direct
  // forcing formulation above computes an Eulerian acceleration field f such
  // that a Navier–Stokes update would apply u^{n+1}=u^n+dt*f.
  // Therefore we export dt*f so that the LBM update applies the same impulse
  // per time step and does not blow up when dt_lev = O(dx) is small.
  for (MFIter mfi(Fx, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    // IMPORTANT:
    //   Fx/Fy/Fz are typically allocated with nghost >= the state ghost width,
    //   but rhocc/EulerForceTotal usually only have the state ghost width.
    //   Using fabbox() (includes Fx ghost cells) can therefore drive indices
    //   outside rhocc/ef (e.g. i=-3 when rhocc only goes down to -2), which
    //   triggers AMReX bounds assertions in DEBUG.
    //   We only need to write forces on the valid region; ghost cells are
    //   handled by FillBoundary / BC machinery upstream.
    const Box &bx = mfi.tilebox();
    auto fx = Fx[mfi].array();
    auto fy = Fy[mfi].array();
    auto fz = Fz[mfi].array();
    auto ef = EulerForceTotal[mfi].const_array();
    auto rho = rhocc[mfi].const_array();

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      // LBM Guo forcing expects (dt * force_density) in lattice units.
      // EulerForceTotal stores the acceleration field a, so force_density = rho*a.
      fx(i, j, k, 0) = dt * rho(i, j, k, 0) * ef(i, j, k, 0);
      fy(i, j, k, 0) = dt * rho(i, j, k, 0) * ef(i, j, k, 1);
      fz(i, j, k, 0) = dt * rho(i, j, k, 0) * ef(i, j, k, 2);
    });
  }

  Fx.FillBoundary(m_geom.periodicity());
  Fy.FillBoundary(m_geom.periodicity());
  Fz.FillBoundary(m_geom.periodicity());
}

#endif // (AMREX_PARTICLES && AMREX_PARTICLES)
