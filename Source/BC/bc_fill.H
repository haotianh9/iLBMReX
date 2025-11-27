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
                  const int /*numcomp*/, amrex::GeometryData const &geom,
                  const amrex::Real /*time*/, const amrex::BCRec *bcr,
                  const int bcomp, const int /*orig_comp*/) const {

    const auto &domain = geom.Domain();

    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
      int coord = iv[dim];
      int lo = domain.smallEnd(dim);
      int hi = domain.bigEnd(dim);

      if (coord < lo && bcr[bcomp].lo(dim) == amrex::BCType::ext_dir) {
        data(iv, dcomp + 0) = bcval.lo_rho[dim];
        data(iv, dcomp + 1) = bcval.lo_ux[dim];
        data(iv, dcomp + 2) = bcval.lo_uy[dim];
        return;
      }

      if (coord > hi && bcr[bcomp].hi(dim) == amrex::BCType::ext_dir) {
        data(iv, dcomp + 0) = bcval.hi_rho[dim];
        data(iv, dcomp + 1) = bcval.hi_ux[dim];
        data(iv, dcomp + 2) = bcval.hi_uy[dim];
        return;
      }
    }
  }
};

struct mesoscopicBcFill {
  BCBuf bcval;

  AMREX_GPU_DEVICE
  void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &f,
                  const int fcomp, const int /*numcomp*/,
                  amrex::GeometryData const &geom, const amrex::Real /*time*/,
                  const amrex::BCRec *bcr, const int bcomp,
                  const int /*orig_comp*/) const {

    const auto &domain = geom.Domain();

    constexpr int ndir = 9;
    static constexpr int ex[ndir] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    static constexpr int ey[ndir] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    static constexpr amrex::Real wi[ndir] = {
        4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
      int coord = iv[dim];
      int lo = domain.smallEnd(dim);
      int hi = domain.bigEnd(dim);

      if (coord < lo && bcr[bcomp].lo(dim) == amrex::BCType::ext_dir) {

        amrex::IntVect iv_mirror = iv;
        iv_mirror[dim] = iv[dim] + 1;
        iv_mirror[dim] = amrex::min(amrex::max(iv_mirror[dim], lo), hi);

        amrex::Real fnp[ndir], feqnp[ndir], feq[ndir];
        amrex::Real rhonp = 0.0, unp = 0.0, vnp = 0.0;

        for (int n = 0; n < ndir; ++n) {
          fnp[n] = f(iv_mirror, fcomp + n);
          rhonp += fnp[n];
          unp += ex[n] * fnp[n];
          vnp += ey[n] * fnp[n];
        }

        unp /= rhonp;
        vnp /= rhonp;

        for (int n = 0; n < ndir; ++n) {
          amrex::Real eu_np = ex[n] * unp + ey[n] * vnp;
          amrex::Real uv_np = unp * unp + vnp * vnp;
          feqnp[n] = wi[n] * rhonp *
                     (1.0 + 3.0 * eu_np + 4.5 * eu_np * eu_np - 1.5 * uv_np);

          amrex::Real u_b = bcval.lo_ux[dim];
          amrex::Real v_b = bcval.lo_uy[dim];
          amrex::Real eu_b = ex[n] * u_b + ey[n] * v_b;
          amrex::Real uv_b = u_b * u_b + v_b * v_b;
          feq[n] = wi[n] * bcval.lo_rho[dim] *
                   (1.0 + 3.0 * eu_b + 4.5 * eu_b * eu_b - 1.5 * uv_b);

          f(iv, fcomp + n) = feq[n] + (fnp[n] - feqnp[n]);
        }
        return;
      }

      if (coord > hi && bcr[bcomp].hi(dim) == amrex::BCType::ext_dir) {

        amrex::IntVect iv_mirror = iv;
        iv_mirror[dim] = iv[dim] - 1;
        iv_mirror[dim] = amrex::min(amrex::max(iv_mirror[dim], lo), hi);

        amrex::Real fnp[ndir], feqnp[ndir], feq[ndir];
        amrex::Real rhonp = 0.0, unp = 0.0, vnp = 0.0;

        for (int n = 0; n < ndir; ++n) {
          fnp[n] = f(iv_mirror, fcomp + n);
          rhonp += fnp[n];
          unp += ex[n] * fnp[n];
          vnp += ey[n] * fnp[n];
        }

        unp /= rhonp;
        vnp /= rhonp;

        for (int n = 0; n < ndir; ++n) {
          amrex::Real eu_np = ex[n] * unp + ey[n] * vnp;
          amrex::Real uv_np = unp * unp + vnp * vnp;
          feqnp[n] = wi[n] * rhonp *
                     (1.0 + 3.0 * eu_np + 4.5 * eu_np * eu_np - 1.5 * uv_np);

          amrex::Real u_b = bcval.hi_ux[dim];
          amrex::Real v_b = bcval.hi_uy[dim];
          amrex::Real eu_b = ex[n] * u_b + ey[n] * v_b;
          amrex::Real uv_b = u_b * u_b + v_b * v_b;
          feq[n] = wi[n] * bcval.hi_rho[dim] *
                   (1.0 + 3.0 * eu_b + 4.5 * eu_b * eu_b - 1.5 * uv_b);

          f(iv, fcomp + n) = feq[n] + (fnp[n] - feqnp[n]);
        }
        return;
      }
    }
  }
};
#endif
