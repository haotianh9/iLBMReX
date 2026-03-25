#ifndef PITCHING_AIRFOIL_IBM_USER_DEFINED_GEOMETRY_H_
#define PITCHING_AIRFOIL_IBM_USER_DEFINED_GEOMETRY_H_

#include "IBM/MarkerIBParams.H"

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_Math.H>

#include <cmath>
#include <vector>

namespace lbm_user_ibm_geometry {

inline constexpr amrex::Real airfoil_chord = amrex::Real(20.0);
inline constexpr amrex::Real airfoil_thickness = amrex::Real(0.12);
inline constexpr amrex::Real pitch_mean = amrex::Real(0.0);
inline constexpr amrex::Real pitch_amplitude =
    amrex::Real(0.2617993877991494);
inline constexpr amrex::Real pitch_frequency = amrex::Real(0.0015);
inline constexpr amrex::Real pitch_phase = amrex::Real(0.0);

inline const char *label() noexcept { return "pitching_airfoil"; }

inline bool refinement_region(amrex::Real x0, amrex::Real y0, amrex::Real z0,
                              MarkerIBParams const &par, amrex::Real dx,
                              amrex::Real &xlo, amrex::Real &xhi,
                              amrex::Real &ylo, amrex::Real &yhi,
                              amrex::Real &zlo, amrex::Real &zhi) noexcept {
  amrex::ignore_unused(par);

#if (AMREX_SPACEDIM != 2)
  amrex::ignore_unused(x0, y0, z0, dx, xlo, xhi, ylo, yhi, zlo, zhi);
  return false;
#else
  const amrex::Real chord =
      amrex::max(airfoil_chord, amrex::Real(1.0e-12));
  const amrex::Real half_thickness =
      amrex::Real(0.5) * airfoil_thickness * chord;
  const amrex::Real sin_amp = std::sin(amrex::Math::abs(pitch_amplitude));

  // Keep the full swept airfoil envelope on the finest AMR level.
  // The leading edge is fixed at (x0, y0); the largest vertical excursion
  // occurs near the trailing edge at the pitch-amplitude extremes.
  const amrex::Real x_extent = chord;
  const amrex::Real y_extent = chord * sin_amp + half_thickness;
  const amrex::Real pad = amrex::max(amrex::Real(6.0) * dx, amrex::Real(2.0));

  xlo = x0 - pad;
  xhi = x0 + x_extent + pad;
  ylo = y0 - y_extent - pad;
  yhi = y0 + y_extent + pad;
  zlo = z0 - pad;
  zhi = z0 + pad;
  return true;
#endif
}

inline void build_markers(amrex::Geometry const &geom, MarkerIBParams const &par,
                          amrex::Real h,
                          std::vector<amrex::Real> &x_ref,
                          std::vector<amrex::Real> &y_ref,
                          std::vector<amrex::Real> &z_ref,
                          amrex::Real &dv, amrex::Real &wall_tol) {
  amrex::ignore_unused(geom);

#if (AMREX_SPACEDIM != 2)
  amrex::ignore_unused(par, h, x_ref, y_ref, z_ref, dv, wall_tol);
  amrex::Abort(
      "Pitching_airfoil user geometry is implemented for 2D only.");
#else
  const amrex::Real chord =
      amrex::max(airfoil_chord, amrex::Real(1.0e-12));
  const amrex::Real thick =
      amrex::max(airfoil_thickness, amrex::Real(1.0e-12));

  const int n_half =
      (par.n_marker > 0)
          ? amrex::max(par.n_marker / 2, 32)
          : amrex::max(
                static_cast<int>(
                    std::ceil(amrex::Real(1.5) * chord / h)),
                48);

  x_ref.clear();
  y_ref.clear();
  z_ref.clear();
  x_ref.reserve(static_cast<std::size_t>(2 * n_half + 2));
  y_ref.reserve(x_ref.capacity());
  z_ref.reserve(x_ref.capacity());

  // Keep index 0 unused so marker id() can be used as a direct index.
  x_ref.push_back(amrex::Real(0.0));
  y_ref.push_back(amrex::Real(0.0));
  z_ref.push_back(amrex::Real(0.0));

  auto thickness_profile = [=](amrex::Real x) noexcept {
    const amrex::Real xc = amrex::min(
        amrex::max(x / chord, amrex::Real(0.0)), amrex::Real(1.0));
    const amrex::Real sqrt_x =
        std::sqrt(amrex::max(xc, amrex::Real(0.0)));
    return amrex::Real(5.0) * thick * chord *
           (amrex::Real(0.2969) * sqrt_x - amrex::Real(0.1260) * xc -
            amrex::Real(0.3516) * xc * xc +
            amrex::Real(0.2843) * xc * xc * xc -
            amrex::Real(0.1036) * xc * xc * xc * xc);
  };

  const amrex::Real pi = amrex::Math::pi<amrex::Real>();
  for (int i = 0; i <= n_half; ++i) {
    const amrex::Real beta = pi * amrex::Real(i) / amrex::Real(n_half);
    const amrex::Real x = amrex::Real(0.5) * chord *
                          (amrex::Real(1.0) - std::cos(beta));
    x_ref.push_back(x);
    y_ref.push_back(thickness_profile(x));
    z_ref.push_back(amrex::Real(0.0));
  }

  for (int i = n_half - 1; i >= 1; --i) {
    const amrex::Real beta = pi * amrex::Real(i) / amrex::Real(n_half);
    const amrex::Real x = amrex::Real(0.5) * chord *
                          (amrex::Real(1.0) - std::cos(beta));
    x_ref.push_back(x);
    y_ref.push_back(-thickness_profile(x));
    z_ref.push_back(amrex::Real(0.0));
  }

  const int ml = static_cast<int>(x_ref.size()) - 1;
  AMREX_ALWAYS_ASSERT(ml > 0);

  amrex::Real perimeter = amrex::Real(0.0);
  for (int i = 1; i <= ml; ++i) {
    const int inext = (i == ml) ? 1 : (i + 1);
    const amrex::Real dxs = x_ref[inext] - x_ref[i];
    const amrex::Real dys = y_ref[inext] - y_ref[i];
    perimeter += std::sqrt(dxs * dxs + dys * dys);
  }

  dv = (perimeter / amrex::Real(ml)) * h;
  wall_tol = amrex::Real(0.0);
#endif
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
marker_state(int idx, amrex::Real time, MarkerIBParams const &par,
             amrex::Real x0, amrex::Real y0, amrex::Real z0,
             amrex::Real const *xref, amrex::Real const *yref,
             amrex::Real const *zref, amrex::Real &x, amrex::Real &y,
             amrex::Real &z, amrex::Real &ux, amrex::Real &uy,
             amrex::Real &uz) noexcept {
  amrex::ignore_unused(par);

  const amrex::Real twopi =
      amrex::Real(2.0) * amrex::Math::pi<amrex::Real>();
  const amrex::Real phase = twopi * pitch_frequency * time + pitch_phase;
  const amrex::Real angle =
      pitch_mean + pitch_amplitude * std::sin(phase);
  const amrex::Real omega =
      twopi * pitch_frequency * pitch_amplitude * std::cos(phase);

  const amrex::Real xr = xref[idx];
  const amrex::Real yr = yref[idx];
  const amrex::Real zr = zref[idx];

  const amrex::Real c = std::cos(angle);
  const amrex::Real s = std::sin(angle);

  x = x0 + c * xr - s * yr;
  y = y0 + s * xr + c * yr;
  z = z0 + zr;

  ux = -omega * (y - y0);
  uy = omega * (x - x0);
  uz = amrex::Real(0.0);
}

} // namespace lbm_user_ibm_geometry

#endif
