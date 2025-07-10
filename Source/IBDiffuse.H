// IBDiffuse.H
#ifndef IAMR_IBDIFFUSE_H_
#define IAMR_IBDIFFUSE_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Utility.H> // for amrex::Vector
#include <cmath>

/// Forward‐declare your driver
class AmrCoreLBM;

namespace amrex {

/// A single Lagrangian marker
struct IBPoint {
  Real x, y;
#if (AMREX_SPACEDIM == 3)
  Real z;
#else
  Real z = 0.0;
#endif
  Real ds; // marker spacing
  IBPoint(Real _x, Real _y, Real _z, Real _ds) : x(_x), y(_y), z(_z), ds(_ds) {}
};

/// Simple 3‐vector for rigid‐body motion
struct Real3 {
  Real x, y, z;
};

/// Cross‐product
inline Real3 cross(const Real3 &a, const Real3 &b) {
  return Real3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x};
}

/**
 Diffusive immersed‐boundary penalty coupling.
   - interpToMarkers: interpolate C → C_k at each marker
   - spreadForcing:  spread F_k = α(C_b − C_k) back to Eulerian ∂C/∂t RHS
   - prescribed kinematics via Li et al. (2024) Eqs. (12)–(13)
 :contentReference[oaicite:5]{index=5}
*/
class IBDiffuse {
public:
  /// Construct with geometry, Lagrangian points, penalty α, and target C_b
  IBDiffuse(const Geometry &geom, const Vector<IBPoint> &pts, Real alpha,
            Real Cb);

  /// Spread the IB forcing into the scalar RHS (delta_rhs).  Must be called
  /// *before* your call to diffuse_scalar so that delta_rhs is included.
  void spreadForcing(const MultiFab &C, MultiFab *delta_rhs,
                     int rhsComp = 0) const;

  /// Interpolate C (one component) onto the IB markers.
  void interpToMarkers(const MultiFab &C, Vector<Real> &Cmark,
                       int comp = 0) const;

  /// Set rigid‐body state at time tⁿ
  void setPrescribedKinematics(const Real3 &Xr_n, const Real3 &Ur_n,
                               const Real3 &Wr_n);

  /// Advance markers & centroid via Li et al. (2024) Eqs. (12)–(13)
  void advancePrescribedKinematics(const Real3 &Ur_np1, const Real3 &Wr_np1,
                                   Real dt);

private:
  Vector<IBPoint> d_ib_pts;
  Real d_alpha;
  Real d_Cb;
  Geometry d_geom;

  // Rigid‐body at t^n
  Real3 d_Xr_n{0, 0, 0};
  Real3 d_Ur_n{0, 0, 0};
  Real3 d_Wr_n{0, 0, 0};

  // Desired marker velocities U_d at t^n
  mutable Vector<Real3> d_Ud;

  // Peskin 4‐point delta
  Real delta1d(Real r, Real h) const;
  Real delta3d(Real rx, Real ry, Real rz) const;
};

} // namespace amrex

#endif // IAMR_IBDIFFUSE_H_
