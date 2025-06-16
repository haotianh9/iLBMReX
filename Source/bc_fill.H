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
    // const int i = iv[0];
    // const int j = iv[1];
    const auto &domain = geom.Domain();

    // const int ilo = domain.smallEnd(0);
    // const int ihi = domain.bigEnd(0);
    // const int jlo = domain.smallEnd(1);
    // const int jhi = domain.bigEnd(1);

    bool is_dirichlet_bc = false;
    int bc_dir = 0;
    bool is_lo = false;
    bool is_hi = false;

    for (int dim_idx = 0; dim_idx < AMREX_SPACEDIM; ++dim_idx) {
      if (iv[dim_idx] < domain.smallEnd(dim_idx) &&
          bcr[bcomp].lo(dim_idx) == amrex::BCType::ext_dir) {
        is_dirichlet_bc = true;
        bc_dir = dim_idx;
        is_lo = true;
      }
      if (iv[dim_idx] > domain.bigEnd(dim_idx) &&
          bcr[bcomp].hi(dim_idx) == amrex::BCType::ext_dir) {
        is_dirichlet_bc = true;
        bc_dir = dim_idx;
        is_hi = true;
      }
    }

    if (is_dirichlet_bc) {
      // printf("iv: %d %d, is_dirichlet_bc: %d, is_lo: %d, is_hi: %d, bc_dir:
      // %d\n", iv[0],iv[1],   is_dirichlet_bc, is_lo, is_hi, bc_dir);
      if (is_lo) {
        data(iv, dcomp + 0) = BCVals::bc_lo_rho_val[bc_dir];
        data(iv, dcomp + 1) = BCVals::bc_lo_ux_val[bc_dir];
        data(iv, dcomp + 2) = BCVals::bc_lo_uy_val[bc_dir];
        // data(iv, dcomp + 0) = 1.0;
        // data(iv, dcomp + 1) = 0.0;
        // data(iv, dcomp + 2) = 0.0;
      }

      if (is_hi) {
        data(iv, dcomp + 0) = BCVals::bc_hi_rho_val[bc_dir];
        data(iv, dcomp + 1) = BCVals::bc_hi_ux_val[bc_dir];
        data(iv, dcomp + 2) = BCVals::bc_hi_uy_val[bc_dir];
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
    // Inamuro et al. Velocity BC Method (2002)
    const int i = iv[0];
    const int j = iv[1];
    const auto &domain = geom.Domain();

    const int ilo = domain.smallEnd(0);
    const int ihi = domain.bigEnd(0);
    const int jlo = domain.smallEnd(1);
    const int jhi = domain.bigEnd(1);

    constexpr int ndir = 9;
    static constexpr int ex[ndir] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    static constexpr int ey[ndir] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    static constexpr amrex::Real wi[ndir] = {
        4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

    bool is_dirichlet_bc = false;
    int bc_dir = 0;
    bool is_lo = false;
    bool is_hi = false;

    for (int dim_idx = 0; dim_idx < AMREX_SPACEDIM; ++dim_idx) {
      if (iv[dim_idx] < domain.smallEnd(dim_idx) &&
          bcr[bcomp].lo(dim_idx) == amrex::BCType::ext_dir) {
        is_dirichlet_bc = true;
        bc_dir = dim_idx;
        is_lo = true;
      }
      if (iv[dim_idx] > domain.bigEnd(dim_idx) &&
          bcr[bcomp].hi(dim_idx) == amrex::BCType::ext_dir) {
        is_dirichlet_bc = true;
        bc_dir = dim_idx;
        is_hi = true;
      }
    }

    if (is_dirichlet_bc) {

      if (is_lo) {
        amrex::Real fnp[ndir], feqnp[ndir], feq[ndir];
        amrex::Real rhonp, unp, vnp;

        for (int n = 0; n < ndir; ++n) {
          int mirror_i = 0, mirror_j = 0;
          if (bc_dir == 0) {
            mirror_i = amrex::min(amrex::max(i + 1, ilo), ihi);
            mirror_j = amrex::min(amrex::max(j, jlo), jhi);
          }

          else if (bc_dir == 1) {
            mirror_i = amrex::min(amrex::max(i, ilo), ihi);
            mirror_j = amrex::min(amrex::max(j + 1, jlo), jhi);
          }

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
        amrex::Real rho_boundary = BCVals::bc_lo_rho_val[bc_dir];
        amrex::Real u_boundary = BCVals::bc_lo_ux_val[bc_dir];
        amrex::Real v_boundary = BCVals::bc_lo_uy_val[bc_dir];
        for (int n = 0; n < ndir; ++n) {

          amrex::Real eu_np = ex[n] * unp + ey[n] * vnp;
          amrex::Real uv_np = unp * unp + vnp * vnp;

          feqnp[n] = wi[n] * rhonp *
                     (1.0 + 3.0 * eu_np + 4.5 * eu_np * eu_np - 1.5 * uv_np);

          amrex::Real eu_boundary = ex[n] * u_boundary + ey[n] * v_boundary;
          amrex::Real uv_boundary =
              u_boundary * u_boundary + v_boundary * v_boundary;

          feq[n] = wi[n] * rho_boundary *
                   (1.0 + 3.0 * eu_boundary + 4.5 * eu_boundary * eu_boundary -
                    1.5 * uv_boundary);

          f(iv, fcomp + n) = feq[n] + (fnp[n] - feqnp[n]);
        }

        // f(iv, fcomp + 2) = f(iv, fcomp + 4);
        // f(iv, fcomp + 5) = f(iv, fcomp + 7);
        // f(iv, fcomp + 6) = f(iv, fcomp + 8);
      }

      if (is_hi) {
        amrex::Real fnp[ndir], feqnp[ndir], feq[ndir];
        amrex::Real rhonp, unp, vnp;

        for (int n = 0; n < ndir; ++n) {
          int mirror_i = 0, mirror_j = 0;
          if (bc_dir == 0) {
            mirror_i = amrex::min(amrex::max(i - 1, ilo), ihi);
            mirror_j = amrex::min(amrex::max(j, jlo), jhi);
          }

          else if (bc_dir == 1) {
            mirror_i = amrex::min(amrex::max(i, ilo), ihi);
            mirror_j = amrex::min(amrex::max(j - 1, jlo), jhi);
          }

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
        amrex::Real rho_boundary = BCVals::bc_hi_rho_val[bc_dir];
        amrex::Real u_boundary = BCVals::bc_hi_ux_val[bc_dir];
        amrex::Real v_boundary = BCVals::bc_hi_uy_val[bc_dir];

        for (int n = 0; n < ndir; ++n) {
          amrex::Real eu_np = ex[n] * unp + ey[n] * vnp;
          amrex::Real uv_np = unp * unp + vnp * vnp;

          feqnp[n] = wi[n] * rhonp *
                     (1.0 + 3.0 * eu_np + 4.5 * eu_np * eu_np - 1.5 * uv_np);

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
