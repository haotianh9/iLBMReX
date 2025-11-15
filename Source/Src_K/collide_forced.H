#ifndef COLLIDE_FORCED_H_
#define COLLIDE_FORCED_H_

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void collide_forced(int i, int j, int k,
                    amrex::Array4<amrex::Real> const &f_old,
                    amrex::Vector<amrex::Real> const &feq_i, // length=ndir
                    int ndir, amrex::Real tau,
                    amrex::Vector<amrex::Real> const &wi,
                    amrex::Vector<amrex::Real> const &dirx,
                    amrex::Vector<amrex::Real> const &diry,
                    amrex::Vector<amrex::Real> const &dirz, amrex::Real rho,
                    amrex::Real ux, amrex::Real uy, amrex::Real uz,
                    amrex::Real fx, amrex::Real fy, amrex::Real fz) {
  constexpr amrex::Real cs2 = 1.0 / 3.0;
  const amrex::Real fac = (1.0 - 0.5 / tau);

  for (int q = 0; q < ndir; ++q) {
    amrex::Real eiu = dirx[q] * ux + diry[q] * uy + dirz[q] * uz;
    amrex::Real termx = (dirx[q] - ux) / cs2 + (eiu / (cs2 * cs2)) * dirx[q];
    amrex::Real termy = (diry[q] - uy) / cs2 + (eiu / (cs2 * cs2)) * diry[q];
    amrex::Real termz = (dirz[q] - uz) / cs2 + (eiu / (cs2 * cs2)) * dirz[q];
    amrex::Real Si = wi[q] * fac * (termx * fx + termy * fy + termz * fz);

    f_old(i, j, k, q) =
        f_old(i, j, k, q) - (f_old(i, j, k, q) - feq_i[q]) / tau + Si;
  }
}

#endif
