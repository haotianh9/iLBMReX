#ifndef BCFILL_H
#define BCFILL_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>
#include <bc_val.H>
#include <cstdio>

#define AMREX_USE_GPU_PRINTF

struct macroBcFill {

  AMREX_GPU_DEVICE
  void operator()(const amrex::IntVect &iv,
                  amrex::Array4<amrex::Real> const &data, const int dcomp,
                  const int /*numcomp*/, amrex::GeometryData const &geom,
                  const amrex::Real /*time*/, const amrex::BCRec *bcr,
                  const int bcomp, const int /*orig_comp*/) const {
    // do something for external Dirichlet (BCType::ext_dir)
    const int i = iv[0];
    const int j = iv[1];
    const auto &domain = geom.Domain();

    const int ilo = domain.smallEnd(0);
    const int ihi = domain.bigEnd(0);
    const int jlo = domain.smallEnd(1);
    const int jhi = domain.bigEnd(1);

    bool is_bc = false;
    bool is_left = false;
    bool is_down = false;
    bool is_right = false;
    bool is_top = false;

    if (i < ilo && bcr[bcomp].lo(0) == amrex::BCType::ext_dir) {
      is_bc = true;
      is_left = true;
    }
    if (i > ihi && bcr[bcomp].hi(0) == amrex::BCType::ext_dir) {
      is_bc = true;
      is_right = true;
    }
    if (j < jlo && bcr[bcomp].lo(1) == amrex::BCType::ext_dir) {
      is_bc = true;
      is_down = true;
    }
    if (j > jhi && bcr[bcomp].hi(1) == amrex::BCType::ext_dir) {
      is_bc = true;
      is_top = true;
    }

    if (is_bc) {
      // if (is_left) {
      //   data(iv, dcomp + 0) = BCVals::bc_lo_rho_val[0];
      //   data(iv, dcomp + 1) = BCVals::bc_lo_ux_val[0];
      //   data(iv, dcomp + 2) = BCVals::bc_lo_uy_val[0];
      // }

      // if (is_right) {
      //   data(iv, dcomp + 0) = BCVals::bc_hi_rho_val[0];
      //   data(iv, dcomp + 1) = BCVals::bc_hi_ux_val[0];
      //   data(iv, dcomp + 2) = BCVals::bc_hi_uy_val[0];
      // }

      if (is_down) {
        data(iv, dcomp + 0) = BCVals::bc_lo_rho_val[1];
        data(iv, dcomp + 1) = BCVals::bc_lo_ux_val[1];
        data(iv, dcomp + 2) = BCVals::bc_lo_uy_val[1];
        // data(iv, dcomp + 0) = 1.0;
        // data(iv, dcomp + 1) = 0.0;
        // data(iv, dcomp + 2) = 0.0;
      }

      if (is_top) {
        data(iv, dcomp + 0) = BCVals::bc_hi_rho_val[1];
        data(iv, dcomp + 1) = BCVals::bc_hi_ux_val[1];
        data(iv, dcomp + 2) = BCVals::bc_hi_uy_val[1];
        // data(iv, dcomp + 0) = 1.0;
        // data(iv, dcomp + 1) = 0.1;
        // data(iv, dcomp + 2) = 0.0;
      }
    }
  }
};

struct mesoscopicBcFill {

  AMREX_GPU_DEVICE
  void operator()(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &f,
                  const int fcomp, const int /*numcomp*/,
                  amrex::GeometryData const &geom, const amrex::Real /*time*/,
                  const amrex::BCRec *bcr, const int bcomp,
                  const int /*orig_comp*/) const {

    const int i = iv[0];
    const int j = iv[1];
    const auto &domain = geom.Domain();

    const int ilo = domain.smallEnd(0);
    const int ihi = domain.bigEnd(0);
    const int jlo = domain.smallEnd(1);
    const int jhi = domain.bigEnd(1);

    bool is_bc = false;
    bool is_left = false;
    bool is_down = false;
    bool is_right = false;
    bool is_top = false;

    constexpr int ndir = 9;
    static constexpr int ex[ndir] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    static constexpr int ey[ndir] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    static constexpr amrex::Real wi[ndir] = {
        4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

    if (i < ilo && bcr[bcomp].lo(0) == amrex::BCType::ext_dir) {
      is_bc = true;
      is_left = true;
    }
    if (i > ihi && bcr[bcomp].hi(0) == amrex::BCType::ext_dir) {
      is_bc = true;
      is_right = true;
    }
    if (j < jlo && bcr[bcomp].lo(1) == amrex::BCType::ext_dir) {
      is_bc = true;
      is_down = true;
    }
    if (j > jhi && bcr[bcomp].hi(1) == amrex::BCType::ext_dir) {
      is_bc = true;
      is_top = true;
    }

    if (is_bc) {

      if (is_down) {

        amrex::Real rhonp, feq[ndir], fnp[ndir];
        for (int n = 0; n < ndir; ++n) {
          int mirror_i = amrex::min(amrex::max(i, ilo), ihi);
          int mirror_j = amrex::min(amrex::max(j + 1, jlo), jhi);
          fnp[n] = f(amrex::IntVect(mirror_i, mirror_j), fcomp + n);
        }

        rhonp = 0.0;
        for (int n = 0; n < ndir; ++n) {
          rhonp += fnp[n];
        }

        for (int n = 0; n < ndir; ++n) {
          amrex::Real rho_boundary = BCVals::bc_lo_rho_val[1];
          amrex::Real u_boundary = BCVals::bc_lo_ux_val[1];
          amrex::Real v_boundary = BCVals::bc_lo_uy_val[1];

          amrex::Real eu_boundary = ex[n] * u_boundary + ey[n] * v_boundary;
          amrex::Real uv_boundary =
              u_boundary * u_boundary + v_boundary * v_boundary;

          feq[n] = wi[n] * rho_boundary *
                   (1.0 + 3.0 * eu_boundary + 4.5 * eu_boundary * eu_boundary -
                    1.5 * uv_boundary);

          f(iv, fcomp + n) = feq[n];
        }

        f(iv, fcomp + 2) = f(iv, fcomp + 4);
        f(iv, fcomp + 5) = f(iv, fcomp + 7);
        f(iv, fcomp + 6) = f(iv, fcomp + 8);
      }

      if (is_top) {
        amrex::Real fnp[ndir], feqnp[ndir], feq[ndir];
        amrex::Real rhonp, unp, vnp;

        for (int n = 0; n < ndir; ++n) {
          int mirror_i = amrex::min(amrex::max(i, ilo), ihi);
          int mirror_j = amrex::min(amrex::max(j - 1, jlo), jhi);
          fnp[n] = f(amrex::IntVect(mirror_i, mirror_j), fcomp + n);
        }

        rhonp = 0.0;
        unp = 0.0;
        vnp = 0.0;
        for (int n = 0; n < ndir; ++n) {
          rhonp += fnp[n];
          unp += ex[n] * fnp[n];
          vnp += ey[n] * fnp[n];
        }
        unp = unp / rhonp;
        vnp = vnp / rhonp;

        for (int n = 0; n < ndir; ++n) {
          amrex::Real eu_np = ex[n] * unp + ey[n] * vnp;
          amrex::Real uv_np = unp * unp + vnp * vnp;

          feqnp[n] = wi[n] * rhonp *
                     (1.0 + 3.0 * eu_np + 4.5 * eu_np * eu_np - 1.5 * uv_np);
          amrex::Real rho_boundary = BCVals::bc_hi_rho_val[1];
          amrex::Real u_boundary = BCVals::bc_hi_ux_val[1];
          amrex::Real v_boundary = BCVals::bc_hi_uy_val[1];

          amrex::Real eu_boundary = ex[n] * u_boundary + ey[n] * v_boundary;
          amrex::Real uv_boundary =
              u_boundary * u_boundary + v_boundary * v_boundary;

          feq[n] = wi[n] * rho_boundary *
                   (1.0 + 3.0 * eu_boundary + 4.5 * eu_boundary * eu_boundary -
                    1.5 * uv_boundary);

          f(iv, fcomp + n) = feq[n] + (fnp[n] - feqnp[n]);
        }
      }
    }
  }
};
#endif
