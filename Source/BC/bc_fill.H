#ifndef BCFILL_H
#define BCFILL_H
#include "bc_val.H"
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

#include <cstdio>

#define AMREX_USE_GPU_PRINTF

struct macroBcFill {
  BCBuf bcval;

  AMREX_GPU_DEVICE
  void operator()(const amrex::IntVect &iv,
                  amrex::Array4<amrex::Real> const &data, const int dcomp,
                  const int numcomp, amrex::GeometryData const &geom,
                  const amrex::Real /*time*/, const amrex::BCRec *bcr,
                  const int bcomp, const int /*orig_comp*/) const {

    const auto &domain = geom.Domain();

    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
      int coord = iv[dim];
      int lo = domain.smallEnd(dim);
      int hi = domain.bigEnd(dim);

      if (coord < lo && bcr[bcomp].lo(dim) == amrex::BCType::ext_dir) {
        amrex::IntVect iv_mirror = iv;
        iv_mirror[dim] = amrex::min(amrex::max(iv[dim] + 1, lo), hi);
        for (int n = 0; n < numcomp; ++n) {
          data(iv, dcomp + n) = data(iv_mirror, dcomp + n);
        }
        if (numcomp > 0) data(iv, dcomp + 0) = bcval.lo_rho[dim];
        if (numcomp > 1) data(iv, dcomp + 1) = bcval.lo_ux[dim];
        if (numcomp > 2) data(iv, dcomp + 2) = bcval.lo_uy[dim];
        if (numcomp > 3) data(iv, dcomp + 3) = bcval.lo_uz[dim];
        return;
      }

      if (coord > hi && bcr[bcomp].hi(dim) == amrex::BCType::ext_dir) {
        amrex::IntVect iv_mirror = iv;
        iv_mirror[dim] = amrex::min(amrex::max(iv[dim] - 1, lo), hi);
        for (int n = 0; n < numcomp; ++n) {
          data(iv, dcomp + n) = data(iv_mirror, dcomp + n);
        }
        if (numcomp > 0) data(iv, dcomp + 0) = bcval.hi_rho[dim];
        if (numcomp > 1) data(iv, dcomp + 1) = bcval.hi_ux[dim];
        if (numcomp > 2) data(iv, dcomp + 2) = bcval.hi_uy[dim];
        if (numcomp > 3) data(iv, dcomp + 3) = bcval.hi_uz[dim];
        return;
      }
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

    const auto &domain = geom.Domain();

    const int ndir = amrex::min(numcomp, bcval.ndir);
    if (ndir <= 0) {
      return;
    }

    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
      int coord = iv[dim];
      int lo = domain.smallEnd(dim);
      int hi = domain.bigEnd(dim);

      if (coord < lo && bcr[bcomp].lo(dim) == amrex::BCType::ext_dir) {

        amrex::IntVect iv_mirror = iv;
        iv_mirror[dim] = iv[dim] + 1;
        iv_mirror[dim] = amrex::min(amrex::max(iv_mirror[dim], lo), hi);

        amrex::Real fnp[BCBuf::MAX_NDIR], feqnp[BCBuf::MAX_NDIR], feq[BCBuf::MAX_NDIR];
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

        for (int n = 0; n < ndir; ++n) {
          amrex::Real eu_np = bcval.dirx[n] * unp + bcval.diry[n] * vnp +
                              bcval.dirz[n] * wnp;
          amrex::Real uv_np = unp * unp + vnp * vnp + wnp * wnp;
          feqnp[n] = bcval.wi[n] * rhonp *
                     (1.0 + 3.0 * eu_np + 4.5 * eu_np * eu_np - 1.5 * uv_np);

          amrex::Real u_b = bcval.lo_ux[dim];
          amrex::Real v_b = bcval.lo_uy[dim];
          amrex::Real w_b = bcval.lo_uz[dim];
          amrex::Real eu_b = bcval.dirx[n] * u_b + bcval.diry[n] * v_b +
                             bcval.dirz[n] * w_b;
          amrex::Real uv_b = u_b * u_b + v_b * v_b + w_b * w_b;
          feq[n] = bcval.wi[n] * bcval.lo_rho[dim] *
                   (1.0 + 3.0 * eu_b + 4.5 * eu_b * eu_b - 1.5 * uv_b);

          f(iv, fcomp + n) = feq[n] + (fnp[n] - feqnp[n]);
        }
        return;
      }

      if (coord > hi && bcr[bcomp].hi(dim) == amrex::BCType::ext_dir) {

        amrex::IntVect iv_mirror = iv;
        iv_mirror[dim] = iv[dim] - 1;
        iv_mirror[dim] = amrex::min(amrex::max(iv_mirror[dim], lo), hi);

        amrex::Real fnp[BCBuf::MAX_NDIR], feqnp[BCBuf::MAX_NDIR], feq[BCBuf::MAX_NDIR];
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

        for (int n = 0; n < ndir; ++n) {
          amrex::Real eu_np = bcval.dirx[n] * unp + bcval.diry[n] * vnp +
                              bcval.dirz[n] * wnp;
          amrex::Real uv_np = unp * unp + vnp * vnp + wnp * wnp;
          feqnp[n] = bcval.wi[n] * rhonp *
                     (1.0 + 3.0 * eu_np + 4.5 * eu_np * eu_np - 1.5 * uv_np);

          amrex::Real u_b = bcval.hi_ux[dim];
          amrex::Real v_b = bcval.hi_uy[dim];
          amrex::Real w_b = bcval.hi_uz[dim];
          amrex::Real eu_b = bcval.dirx[n] * u_b + bcval.diry[n] * v_b +
                             bcval.dirz[n] * w_b;
          amrex::Real uv_b = u_b * u_b + v_b * v_b + w_b * w_b;
          feq[n] = bcval.wi[n] * bcval.hi_rho[dim] *
                   (1.0 + 3.0 * eu_b + 4.5 * eu_b * eu_b - 1.5 * uv_b);

          f(iv, fcomp + n) = feq[n] + (fnp[n] - feqnp[n]);
        }
        return;
      }
    }
  }
};
#endif
