// IBDiffuse.cpp
#include "IBDiffuse.H"
using namespace amrex;

IBDiffuse::IBDiffuse(const Geometry &geom, const Vector<IBPoint> &pts,
                     Real alpha, Real Cb)
    : d_ib_pts(pts), d_alpha(alpha), d_Cb(Cb), d_geom(geom) {}

Real IBDiffuse::delta1d(Real r, Real h) const {
  r = std::abs(r) / h;
  if (r < 1.0)
    return (1.0 + std::cos(M_PI * r)) / (2.0 * h);
  else if (r < 2.0)
    return (1.0 + std::cos(M_PI * r / 2.0)) / (4.0 * h);
  else
    return 0.0;
}

Real IBDiffuse::delta3d(Real rx, Real ry, Real rz) const {
  const auto dx = d_geom.CellSizeArray();
  Real dz = (AMREX_SPACEDIM == 3 ? dx[2] : dx[1]);
  return delta1d(rx, dx[0]) * delta1d(ry, dx[1]) * delta1d(rz, dz);
}

void IBDiffuse::interpToMarkers(const MultiFab &C, Vector<Real> &Cmark,
                                int comp) const {
  Cmark.resize(d_ib_pts.size());
  const auto plo = d_geom.ProbLoArray();
  const auto dx = d_geom.CellSizeArray();

  for (std::size_t i = 0; i < d_ib_pts.size(); ++i) {
    const auto &p = d_ib_pts[i];
    // build nearest‐cell index
#if (AMREX_SPACEDIM == 3)
    int ix = static_cast<int>((p.x - plo[0]) / dx[0]);
    int iy = static_cast<int>((p.y - plo[1]) / dx[1]);
    int iz = static_cast<int>((p.z - plo[2]) / dx[2]);
    IntVect ic(ix, iy, iz);
#else
    int ix = static_cast<int>((p.x - plo[0]) / dx[0]);
    int iy = static_cast<int>((p.y - plo[1]) / dx[1]);
    IntVect ic(ix, iy);
#endif

    Real Ck = 0.0;
    // support ±2 cells
#if (AMREX_SPACEDIM == 3)
    for (int dz = -2; dz <= 2; ++dz) {
      for (int dy = -2; dy <= 2; ++dy) {
        for (int dx_ = -2; dx_ <= 2; ++dx_) {
          IntVect iv = ic + IntVect(dx_, dy, dz);
#else
    for (int dy = -2; dy <= 2; ++dy) {
      for (int dx_ = -2; dx_ <= 2; ++dx_) {
        IntVect iv = ic + IntVect(dx_, dy);
#endif
          if (!d_geom.Domain().contains(iv))
            continue;

          // physical offset from marker to cell‐center
          Real xr = plo[0] + (iv[0] + 0.5) * dx[0] - p.x;
          Real yr = plo[1] + (iv[1] + 0.5) * dx[1] - p.y;
#if (AMREX_SPACEDIM == 3)
          Real zr = plo[2] + (iv[2] + 0.5) * dx[2] - p.z;
#else
        Real zr = 0.0;
#endif
          Real w = delta3d(xr, yr, zr);
          Ck += C.const_array(comp)(iv, 0) * w * p.ds;
        }
      }
#if (AMREX_SPACEDIM == 3)
    }
  }
}
#endif

Cmark[i] = Ck;
}
}

void IBDiffuse::spreadForcing(const MultiFab &C, MultiFab *delta_rhs,
                              int rhsComp) const {
  // 1) interpolate once
  Vector<Real> Cmark;
  interpToMarkers(C, Cmark, rhsComp);

  // 2) zero out that component of delta_rhs
  delta_rhs->setVal(0.0, rhsComp, 1, delta_rhs->nGrow());

  // 3) loop over tiles & markers, spread Fk = α(Cb − Ck)
  const auto plo = d_geom.ProbLoArray();
  const auto dx = d_geom.CellSizeArray();

  for (MFIter mfi(*delta_rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    auto Ffab = delta_rhs->array(mfi);
    Box bx = mfi.fabbox();
    Box ext = amrex::grow(bx, 2);

    for (std::size_t i = 0; i < d_ib_pts.size(); ++i) {
      const auto &p = d_ib_pts[i];
      Real Fk = d_alpha * (d_Cb - Cmark[i]);

      // nearest‐cell index
#if (AMREX_SPACEDIM == 3)
      int ix = static_cast<int>((p.x - plo[0]) / dx[0]);
      int iy = static_cast<int>((p.y - plo[1]) / dx[1]);
      int iz = static_cast<int>((p.z - plo[2]) / dx[2]);
      IntVect ic(ix, iy, iz);
#else
      int ix = static_cast<int>((p.x - plo[0]) / dx[0]);
      int iy = static_cast<int>((p.y - plo[1]) / dx[1]);
      IntVect ic(ix, iy);
#endif

#if (AMREX_SPACEDIM == 3)
      for (int dz = -2; dz <= 2; ++dz) {
        for (int dy = -2; dy <= 2; ++dy) {
          for (int dx_ = -2; dx_ <= 2; ++dx_) {
            IntVect iv = ic + IntVect(dx_, dy, dz);
#else
      for (int dy = -2; dy <= 2; ++dy) {
        for (int dx_ = -2; dx_ <= 2; ++dx_) {
          IntVect iv = ic + IntVect(dx_, dy);
#endif
            if (!ext.contains(iv))
              continue;

            Real xr = plo[0] + (iv[0] + 0.5) * dx[0] - p.x;
            Real yr = plo[1] + (iv[1] + 0.5) * dx[1] - p.y;
#if (AMREX_SPACEDIM == 3)
            Real zr = plo[2] + (iv[2] + 0.5) * dx[2] - p.z;
#else
          Real zr = 0.0;
#endif
            Real w = delta3d(xr, yr, zr);
            Ffab(iv, rhsComp) += Fk * w * p.ds;
          }
        }
#if (AMREX_SPACEDIM == 3)
      }
    }
  }
#endif
}
}
}

void IBDiffuse::setPrescribedKinematics(const Real3 &Xr_n, const Real3 &Ur_n,
                                        const Real3 &Wr_n) {
  d_Xr_n = Xr_n;
  d_Ur_n = Ur_n;
  d_Wr_n = Wr_n;
  d_Ud.assign(d_ib_pts.size(), Real3{0, 0, 0});
}

void IBDiffuse::advancePrescribedKinematics(const Real3 &Ur_np1,
                                            const Real3 &Wr_np1, Real dt) {
  // (12) compute U_d
  for (std::size_t l = 0; l < d_ib_pts.size(); ++l) {
    const auto &p = d_ib_pts[l];
    Real3 R{p.x - d_Xr_n.x, p.y - d_Xr_n.y, p.z - d_Xr_n.z};
    Real3 cr = cross(d_Wr_n, R);
    d_Ud[l] = {d_Ur_n.x + cr.x, d_Ur_n.y + cr.y, d_Ur_n.z + cr.z};
  }
  // advect
  for (std::size_t l = 0; l < d_ib_pts.size(); ++l) {
    auto &p = d_ib_pts[l];
    p.x += dt * d_Ud[l].x;
    p.y += dt * d_Ud[l].y;
    p.z += dt * d_Ud[l].z;
  }
  // (13) update centroid
  Real3 Xr_np1{d_Xr_n.x + 0.5 * dt * (d_Ur_n.x + Ur_np1.x),
               d_Xr_n.y + 0.5 * dt * (d_Ur_n.y + Ur_np1.y),
               d_Xr_n.z + 0.5 * dt * (d_Ur_n.z + Ur_np1.z)};
  d_Xr_n = Xr_np1;
  d_Ur_n = Ur_np1;
  d_Wr_n = Wr_np1;
}
