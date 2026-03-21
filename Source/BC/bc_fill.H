#ifndef BCFILL_H
#define BCFILL_H
#include "bc_val.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

#include <cstdio>

#define AMREX_USE_GPU_PRINTF

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
nearestInteriorCellAndBoundaryHits(const amrex::IntVect &iv,
                                   amrex::IntVect &iv_mirror,
                                   amrex::GeometryData const &geom,
                                   amrex::GpuArray<int, AMREX_SPACEDIM> &lo_hit,
                                   amrex::GpuArray<int, AMREX_SPACEDIM> &hi_hit) {
  const auto &domain = geom.Domain();

  iv_mirror = iv;
  for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
    lo_hit[dim] = 0;
    hi_hit[dim] = 0;
  }

  for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
    const int coord = iv[dim];
    const int lo = domain.smallEnd(dim);
    const int hi = domain.bigEnd(dim);

    if (coord < lo) {
      iv_mirror[dim] = lo;
      lo_hit[dim] = 1;
    } else if (coord > hi) {
      iv_mirror[dim] = hi;
      hi_hit[dim] = 1;
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
bool
firstBoundaryFaceOfType(amrex::GpuArray<int, AMREX_SPACEDIM> const &lo_hit,
                        amrex::GpuArray<int, AMREX_SPACEDIM> const &hi_hit,
                        amrex::BCRec const *bcr, int bcomp, int bc_type,
                        int &bc_dim, bool &bc_on_lo_side) {
  bc_dim = -1;
  bc_on_lo_side = true;

  for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
    if (lo_hit[dim] && bcr[bcomp].lo(dim) == bc_type) {
      bc_dim = dim;
      bc_on_lo_side = true;
      return true;
    }
    if (hi_hit[dim] && bcr[bcomp].hi(dim) == bc_type) {
      bc_dim = dim;
      bc_on_lo_side = false;
      return true;
    }
  }

  return false;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
applyPrescribedMacroState(amrex::IntVect const &iv,
                          amrex::Array4<amrex::Real> const &data, int dcomp,
                          int numcomp, BCBuf const &bcval, int bc_dim,
                          bool bc_on_lo_side) {
  if (bc_on_lo_side) {
    if (numcomp > 0)
      data(iv, dcomp + 0) = bcval.lo_rho[bc_dim];
    if (numcomp > 1)
      data(iv, dcomp + 1) = bcval.lo_ux[bc_dim];
    if (numcomp > 2)
      data(iv, dcomp + 2) = bcval.lo_uy[bc_dim];
    if (numcomp > 3)
      data(iv, dcomp + 3) = bcval.lo_uz[bc_dim];
  } else {
    if (numcomp > 0)
      data(iv, dcomp + 0) = bcval.hi_rho[bc_dim];
    if (numcomp > 1)
      data(iv, dcomp + 1) = bcval.hi_ux[bc_dim];
    if (numcomp > 2)
      data(iv, dcomp + 2) = bcval.hi_uy[bc_dim];
    if (numcomp > 3)
      data(iv, dcomp + 3) = bcval.hi_uz[bc_dim];
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
applyMirroredMacroState(amrex::IntVect const &iv,
                        amrex::IntVect const &iv_mirror,
                        amrex::Array4<amrex::Real> const &data, int dcomp,
                        int numcomp, BCBuf const &bcval, int bc_dim,
                        bool bc_on_lo_side) {
  const amrex::Real rho_b =
      bc_on_lo_side ? bcval.lo_rho[bc_dim] : bcval.hi_rho[bc_dim];
  const amrex::Real u_b =
      bc_on_lo_side ? bcval.lo_ux[bc_dim] : bcval.hi_ux[bc_dim];
  const amrex::Real v_b =
      bc_on_lo_side ? bcval.lo_uy[bc_dim] : bcval.hi_uy[bc_dim];
  const amrex::Real w_b =
      bc_on_lo_side ? bcval.lo_uz[bc_dim] : bcval.hi_uz[bc_dim];

  if (numcomp > 0) {
    data(iv, dcomp + 0) =
        amrex::max(amrex::Real(1.0e-20),
                   amrex::Real(2.0) * rho_b - data(iv_mirror, dcomp + 0));
  }
  if (numcomp > 1) {
    data(iv, dcomp + 1) =
        amrex::Real(2.0) * u_b - data(iv_mirror, dcomp + 1);
  }
  if (numcomp > 2) {
    data(iv, dcomp + 2) =
        amrex::Real(2.0) * v_b - data(iv_mirror, dcomp + 2);
  }
  if (numcomp > 3) {
    data(iv, dcomp + 3) =
        amrex::Real(2.0) * w_b - data(iv_mirror, dcomp + 3);
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
copyMacroFromMirror(amrex::IntVect const &iv, amrex::IntVect const &iv_mirror,
                    amrex::Array4<amrex::Real> const &data, int dcomp,
                    int numcomp) {
  for (int n = 0; n < numcomp; ++n) {
    data(iv, dcomp + n) = data(iv_mirror, dcomp + n);
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
copyMesoscopicFromMirror(amrex::IntVect const &iv,
                         amrex::IntVect const &iv_mirror,
                         amrex::Array4<amrex::Real> const &f, int fcomp,
                         int ndir) {
  for (int n = 0; n < ndir; ++n) {
    f(iv, fcomp + n) = f(iv_mirror, fcomp + n);
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
applyExtDirMesoscopic(amrex::IntVect const &iv,
                      amrex::IntVect const &iv_mirror,
                      amrex::Array4<amrex::Real> const &f, int fcomp, int ndir,
                      BCBuf const &bcval, int bc_dim, bool bc_on_lo_side) {
  amrex::Real fnp[BCBuf::MAX_NDIR], feqnp[BCBuf::MAX_NDIR],
      feq[BCBuf::MAX_NDIR];
  amrex::Real rhonp = 0.0, unp = 0.0, vnp = 0.0, wnp = 0.0;

  for (int n = 0; n < ndir; ++n) {
    fnp[n] = f(iv_mirror, fcomp + n);
    rhonp += fnp[n];
    unp += bcval.dirx[n] * fnp[n];
    vnp += bcval.diry[n] * fnp[n];
    wnp += bcval.dirz[n] * fnp[n];
  }

  rhonp = amrex::max(rhonp, amrex::Real(1.0e-20));
  unp /= rhonp;
  vnp /= rhonp;
  wnp /= rhonp;

  const amrex::Real rho_b =
      bc_on_lo_side ? bcval.lo_rho[bc_dim] : bcval.hi_rho[bc_dim];
  const amrex::Real u_b =
      bc_on_lo_side ? bcval.lo_ux[bc_dim] : bcval.hi_ux[bc_dim];
  const amrex::Real v_b =
      bc_on_lo_side ? bcval.lo_uy[bc_dim] : bcval.hi_uy[bc_dim];
  const amrex::Real w_b =
      bc_on_lo_side ? bcval.lo_uz[bc_dim] : bcval.hi_uz[bc_dim];

  for (int n = 0; n < ndir; ++n) {
    amrex::Real eu_np =
        bcval.dirx[n] * unp + bcval.diry[n] * vnp + bcval.dirz[n] * wnp;
    amrex::Real uv_np = unp * unp + vnp * vnp + wnp * wnp;
    feqnp[n] = bcval.wi[n] * rhonp *
               (1.0 + 3.0 * eu_np + 4.5 * eu_np * eu_np - 1.5 * uv_np);

    amrex::Real eu_b =
        bcval.dirx[n] * u_b + bcval.diry[n] * v_b + bcval.dirz[n] * w_b;
    amrex::Real uv_b = u_b * u_b + v_b * v_b + w_b * w_b;
    feq[n] = bcval.wi[n] * rho_b *
             (1.0 + 3.0 * eu_b + 4.5 * eu_b * eu_b - 1.5 * uv_b);

    f(iv, fcomp + n) = feq[n] + (fnp[n] - feqnp[n]);
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
applyFaceMirroredMesoscopic(amrex::IntVect const &iv,
                            amrex::IntVect const &iv_mirror,
                            amrex::Array4<amrex::Real> const &f, int fcomp,
                            int ndir, BCBuf const &bcval, int bc_dim,
                            bool bc_on_lo_side) {
  amrex::Real fnp[BCBuf::MAX_NDIR], feqnp[BCBuf::MAX_NDIR],
      feqg[BCBuf::MAX_NDIR];
  amrex::Real rhonp = 0.0, unp = 0.0, vnp = 0.0, wnp = 0.0;

  for (int n = 0; n < ndir; ++n) {
    fnp[n] = f(iv_mirror, fcomp + n);
    rhonp += fnp[n];
    unp += bcval.dirx[n] * fnp[n];
    vnp += bcval.diry[n] * fnp[n];
    wnp += bcval.dirz[n] * fnp[n];
  }

  rhonp = amrex::max(rhonp, amrex::Real(1.0e-20));
  unp /= rhonp;
  vnp /= rhonp;
  wnp /= rhonp;

  const amrex::Real rho_b =
      bc_on_lo_side ? bcval.lo_rho[bc_dim] : bcval.hi_rho[bc_dim];
  const amrex::Real u_b =
      bc_on_lo_side ? bcval.lo_ux[bc_dim] : bcval.hi_ux[bc_dim];
  const amrex::Real v_b =
      bc_on_lo_side ? bcval.lo_uy[bc_dim] : bcval.hi_uy[bc_dim];
  const amrex::Real w_b =
      bc_on_lo_side ? bcval.lo_uz[bc_dim] : bcval.hi_uz[bc_dim];

  const amrex::Real rho_g =
      amrex::max(amrex::Real(1.0e-20), amrex::Real(2.0) * rho_b - rhonp);
  const amrex::Real u_g = amrex::Real(2.0) * u_b - unp;
  const amrex::Real v_g = amrex::Real(2.0) * v_b - vnp;
  const amrex::Real w_g = amrex::Real(2.0) * w_b - wnp;

  for (int n = 0; n < ndir; ++n) {
    amrex::Real eu_np =
        bcval.dirx[n] * unp + bcval.diry[n] * vnp + bcval.dirz[n] * wnp;
    amrex::Real uv_np = unp * unp + vnp * vnp + wnp * wnp;
    feqnp[n] = bcval.wi[n] * rhonp *
               (1.0 + 3.0 * eu_np + 4.5 * eu_np * eu_np - 1.5 * uv_np);

    amrex::Real eu_g =
        bcval.dirx[n] * u_g + bcval.diry[n] * v_g + bcval.dirz[n] * w_g;
    amrex::Real uv_g = u_g * u_g + v_g * v_g + w_g * w_g;
    feqg[n] = bcval.wi[n] * rho_g *
              (1.0 + 3.0 * eu_g + 4.5 * eu_g * eu_g - 1.5 * uv_g);

    f(iv, fcomp + n) = feqg[n] + (fnp[n] - feqnp[n]);
  }
}

struct macroBcFill {
  BCBuf bcval;

  AMREX_GPU_DEVICE
  void operator()(const amrex::IntVect &iv,
                  amrex::Array4<amrex::Real> const &data, const int dcomp,
                  const int numcomp, amrex::GeometryData const &geom,
                  const amrex::Real /*time*/, const amrex::BCRec *bcr,
                  const int bcomp, const int /*orig_comp*/) const {

    amrex::IntVect iv_mirror;
    amrex::GpuArray<int, AMREX_SPACEDIM> lo_hit{}, hi_hit{};
    nearestInteriorCellAndBoundaryHits(iv, iv_mirror, geom, lo_hit, hi_hit);

    int bc_dim = -1;
    bool bc_on_lo_side = true;

    if (firstBoundaryFaceOfType(lo_hit, hi_hit, bcr, bcomp,
                                amrex::BCType::user_2, bc_dim,
                                bc_on_lo_side)) {
      copyMacroFromMirror(iv, iv_mirror, data, dcomp, numcomp);
      applyMirroredMacroState(iv, iv_mirror, data, dcomp, numcomp, bcval,
                              bc_dim, bc_on_lo_side);
      return;
    }

    if (firstBoundaryFaceOfType(lo_hit, hi_hit, bcr, bcomp,
                                amrex::BCType::user_1, bc_dim,
                                bc_on_lo_side) ||
        firstBoundaryFaceOfType(lo_hit, hi_hit, bcr, bcomp,
                                amrex::BCType::ext_dir, bc_dim,
                                bc_on_lo_side)) {
      copyMacroFromMirror(iv, iv_mirror, data, dcomp, numcomp);
      applyPrescribedMacroState(iv, data, dcomp, numcomp, bcval, bc_dim,
                                bc_on_lo_side);
      return;
    }
  }
};

struct mesoscopicBcFill {
  BCBuf bcval;
  AMREX_GPU_DEVICE
  void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &f,
                  const int fcomp, const int numcomp,
                  amrex::GeometryData const &geom, const amrex::Real /*time*/,
                  const amrex::BCRec *bcr, const int bcomp,
                  const int /*orig_comp*/) const {

    const int ndir = amrex::min(numcomp, bcval.ndir);
    if (ndir <= 0) {
      return;
    }

    amrex::IntVect iv_mirror;
    amrex::GpuArray<int, AMREX_SPACEDIM> lo_hit{}, hi_hit{};
    nearestInteriorCellAndBoundaryHits(iv, iv_mirror, geom, lo_hit, hi_hit);

    int bc_dim = -1;
    bool bc_on_lo_side = true;

    if (firstBoundaryFaceOfType(lo_hit, hi_hit, bcr, bcomp,
                                amrex::BCType::user_2, bc_dim,
                                bc_on_lo_side)) {
      applyFaceMirroredMesoscopic(iv, iv_mirror, f, fcomp, ndir, bcval,
                                  bc_dim, bc_on_lo_side);
      return;
    }

    if (firstBoundaryFaceOfType(lo_hit, hi_hit, bcr, bcomp,
                                amrex::BCType::user_1, bc_dim,
                                bc_on_lo_side)) {
      copyMesoscopicFromMirror(iv, iv_mirror, f, fcomp, ndir);
      return;
    }

    if (firstBoundaryFaceOfType(lo_hit, hi_hit, bcr, bcomp,
                                amrex::BCType::ext_dir, bc_dim,
                                bc_on_lo_side)) {
      applyExtDirMesoscopic(iv, iv_mirror, f, fcomp, ndir, bcval, bc_dim,
                            bc_on_lo_side);
      return;
    }
  }
};
#endif
