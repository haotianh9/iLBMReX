#ifndef IBMOMENTUM_H_
#define IBMOMENTUM_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Utility.H>
#include <cmath>

namespace amrex {

struct IBMarker {
  Real x{0}, y{0}, z{0};
  Real Ux{0}, Uy{0}, Uz{0}; // interpolated Eulerian velocity at marker
};

// 4-pt Peskin 1D delta (support |r|<=2)
AMREX_FORCE_INLINE
Real delta1D_peskin(Real r) {
  r = std::abs(r);
  if (r < 1.0) {
    return 0.125 * (3.0 - 2.0 * r + std::sqrt(1.0 + 4.0 * r - 4.0 * r * r));
  } else if (r < 2.0) {
    return 0.125 * (5.0 - 2.0 * r - std::sqrt(-7.0 + 12.0 * r - 4.0 * r * r));
  } else {
    return 0.0;
  }
}

class IBMomentum {
public:
  IBMomentum(const Geometry &geom, Real x0, Real y0, Real z0, Real radius,
             Real ds, Real alpha)
      : m_geom(geom), m_alpha(alpha) {
    build_circle(x0, y0, z0, radius, ds);
  }

  const Vector<IBMarker> &markers() const { return m_markers; }

  // Interpolate cell-centered (u,v,w) to markers with 4-pt kernel
  void interpolate(const MultiFab &u_cc, const MultiFab &v_cc,
                   const MultiFab &w_cc);

  // Compute Lagrangian force F_L = alpha (U_target - U_interp), here U_target=0
  void compute_direct_forcing();

  // Spread marker forces to Eulerian force fields (fx,fy,fz), zero them first
  void spread(MultiFab &fx_cc, MultiFab &fy_cc, MultiFab &fz_cc) const;

private:
  Geometry m_geom;
  Vector<IBMarker> m_markers;
  Real m_alpha{2.0}; // direct-forcing gain
  Real m_ds{1.0};    // arc-length spacing of markers

  void build_circle(Real x0, Real y0, Real z0, Real R, Real ds);
};

inline void IBMomentum::build_circle(Real x0, Real y0, Real z0, Real R,
                                     Real ds) {
  const int dim = AMREX_SPACEDIM;
  m_ds = ds;
  int N = std::max(8, int(std::round(2.0 * M_PI * R / ds)));
  m_markers.resize(N);
  for (int k = 0; k < N; ++k) {
    Real th = (2.0 * M_PI) * (Real(k) / Real(N));
    m_markers[k].x = x0 + R * std::cos(th);
    m_markers[k].y = y0 + R * std::sin(th);
    if (dim == 3)
      m_markers[k].z = z0; // spanwise-invariant cylinder
  }
}
inline void IBMomentum::interpolate(const MultiFab &u_cc, const MultiFab &v_cc,
                                    const MultiFab &w_cc) {
  // Zero marker samples
  for (auto &p : m_markers) {
    p.Ux = 0;
    p.Uy = 0;
    p.Uz = 0;
  }

  const auto dx = m_geom.CellSizeArray();
  const auto plo = m_geom.ProbLoArray();
  const Box &domain = m_geom.Domain();

  // Loop tiles and accumulate contributions from any marker stencil cells that
  // lie in the tile
  for (MFIter mfi(u_cc); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto uarr = u_cc.const_array(mfi);
    auto varr = v_cc.const_array(mfi);
#if (AMREX_SPACEDIM == 3)
    auto warr = w_cc.const_array(mfi);
#endif
    for (auto &p : m_markers) {
      // stencil center cell
#if (AMREX_SPACEDIM == 3)
      IntVect ic(int(std::floor((p.x - plo[0]) / dx[0])),
                 int(std::floor((p.y - plo[1]) / dx[1])),
                 int(std::floor((p.z - plo[2]) / dx[2])));
#else
      IntVect ic(int(std::floor((p.x - plo[0]) / dx[0])),
                 int(std::floor((p.y - plo[1]) / dx[1])));
#endif

#if (AMREX_SPACEDIM == 3)
      for (int kz = -2; kz <= 2; ++kz)
#endif
        for (int jy = -2; jy <= 2; ++jy)
          for (int ix = -2; ix <= 2; ++ix) {
#if (AMREX_SPACEDIM == 3)
            IntVect iv = ic + IntVect(ix, jy, kz);
#else
          IntVect iv = ic + IntVect(ix, jy);
#endif
            if (!domain.contains(iv) || !bx.contains(iv))
              continue;

            const int ii = iv[0];
            const int jj = iv[1];
#if (AMREX_SPACEDIM == 3)
            const int kk = iv[2];
#else
          const int kk = 0;
#endif
            amrex::Real xc = plo[0] + (ii + amrex::Real(0.5)) * dx[0];
            amrex::Real yc = plo[1] + (jj + amrex::Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM == 3)
            amrex::Real zc = plo[2] + (kk + amrex::Real(0.5)) * dx[2];
#endif
            amrex::Real rx = (p.x - xc) / dx[0];
            amrex::Real ry = (p.y - yc) / dx[1];
#if (AMREX_SPACEDIM == 3)
            amrex::Real rz = (p.z - zc) / dx[2];
#endif
            amrex::Real delta = delta1D_peskin(rx) * delta1D_peskin(ry);
#if (AMREX_SPACEDIM == 3)
            delta *= delta1D_peskin(rz);
#endif
            amrex::Real vol = dx[0] * dx[1];
#if (AMREX_SPACEDIM == 3)
            vol *= dx[2];
#endif
            p.Ux += uarr(ii, jj, kk) * delta * vol;
            p.Uy += varr(ii, jj, kk) * delta * vol;
#if (AMREX_SPACEDIM == 3)
            p.Uz += warr(ii, jj, kk) * delta * vol;
#endif
          }
    }
  }
}

inline void IBMomentum::compute_direct_forcing() {
  for (auto &p : m_markers) {
    p.Ux = m_alpha * (0.0 - p.Ux);
    p.Uy = m_alpha * (0.0 - p.Uy);
    if (AMREX_SPACEDIM == 3)
      p.Uz = m_alpha * (0.0 - p.Uz);
  }
}
inline void IBMomentum::spread(MultiFab &fx_cc, MultiFab &fy_cc,
                               MultiFab &fz_cc) const {
  fx_cc.setVal(0.0);
  fy_cc.setVal(0.0);
#if (AMREX_SPACEDIM == 3)
  fz_cc.setVal(0.0);
#endif

  const auto dx = m_geom.CellSizeArray();
  const auto plo = m_geom.ProbLoArray();
  const Box &domain = m_geom.Domain();

  for (MFIter mfi(fx_cc); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto arrx = fx_cc.array(mfi);
    auto arry = fy_cc.array(mfi);
#if (AMREX_SPACEDIM == 3)
    auto arrz = fz_cc.array(mfi);
#endif
    for (auto const &p : m_markers) {
#if (AMREX_SPACEDIM == 3)
      IntVect ic(int(std::floor((p.x - plo[0]) / dx[0])),
                 int(std::floor((p.y - plo[1]) / dx[1])),
                 int(std::floor((p.z - plo[2]) / dx[2])));
#else
      IntVect ic(int(std::floor((p.x - plo[0]) / dx[0])),
                 int(std::floor((p.y - plo[1]) / dx[1])));
#endif

#if (AMREX_SPACEDIM == 3)
      for (int kz = -2; kz <= 2; ++kz)
#endif
        for (int jy = -2; jy <= 2; ++jy)
          for (int ix = -2; ix <= 2; ++ix) {
#if (AMREX_SPACEDIM == 3)
            IntVect iv = ic + IntVect(ix, jy, kz);
#else
          IntVect iv = ic + IntVect(ix, jy);
#endif
            if (!domain.contains(iv) || !bx.contains(iv))
              continue;

            const int ii = iv[0];
            const int jj = iv[1];
#if (AMREX_SPACEDIM == 3)
            const int kk = iv[2];
#else
          const int kk = 0;
#endif
            amrex::Real xc = plo[0] + (ii + amrex::Real(0.5)) * dx[0];
            amrex::Real yc = plo[1] + (jj + amrex::Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM == 3)
            amrex::Real zc = plo[2] + (kk + amrex::Real(0.5)) * dx[2];
#endif
            amrex::Real rx = (p.x - xc) / dx[0];
            amrex::Real ry = (p.y - yc) / dx[1];
#if (AMREX_SPACEDIM == 3)
            amrex::Real rz = (p.z - zc) / dx[2];
#endif
            amrex::Real delta = delta1D_peskin(rx) * delta1D_peskin(ry);
#if (AMREX_SPACEDIM == 3)
            delta *= delta1D_peskin(rz);
#endif
            amrex::Real wgt = delta * m_ds; // discrete line element weight
            // NOTE: p.U* holds the FORCE after compute_direct_forcing()
            arrx(ii, jj, kk) += p.Ux * wgt;
            arry(ii, jj, kk) += p.Uy * wgt;
#if (AMREX_SPACEDIM == 3)
            arrz(ii, jj, kk) += p.Uz * wgt;
#endif
          }
    }
  }
}

} // namespace amrex

#endif // IBMOMENTUM_H_
