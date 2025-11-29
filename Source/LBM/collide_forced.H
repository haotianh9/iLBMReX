#ifndef COLLIDE_FORCED_H_
#define COLLIDE_FORCED_H_

#include "collide.H" // make sure this is included somewhere
#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void collide_forced(int i, int j, int k,
                    amrex::Array4<amrex::Real> const &f_old,
                    amrex::Vector<amrex::Real> const &feq_i, // length = ndir
                    int ndir, amrex::Real tau,
                    amrex::Vector<amrex::Real> const &wi,
                    amrex::Vector<amrex::Real> const &dirx,
                    amrex::Vector<amrex::Real> const &diry,
                    amrex::Vector<amrex::Real> const &dirz, amrex::Real rho,
                    amrex::Real ux, amrex::Real uy, amrex::Real uz,
                    amrex::Real fx, amrex::Real fy, amrex::Real fz) {
  // 1) Do the usual BGK collision using your existing kernel
  collide(i, j, k, f_old, feq_i, ndir, tau);

  // 2) If forcing is effectively zero, we are done:
  if (fx == 0.0 && fy == 0.0 && fz == 0.0) {
    return;
  }

  // 3) Guo forcing increment
  constexpr amrex::Real cs2 = 1.0 / 3.0;
  const amrex::Real fac = 1.0 - 0.5 / tau; // Guo prefactor (1 - 1/(2τ))

  for (int q = 0; q < ndir; ++q) {
    amrex::Real cx = dirx[q];
    amrex::Real cy = diry[q];
    amrex::Real cz = dirz[q];

    amrex::Real eiu = cx * ux + cy * uy + cz * uz;

    amrex::Real termx = (cx - ux) / cs2 + (eiu / (cs2 * cs2)) * cx;
    amrex::Real termy = (cy - uy) / cs2 + (eiu / (cs2 * cs2)) * cy;
#if (AMREX_SPACEDIM == 3)
    amrex::Real termz = (cz - uz) / cs2 + (eiu / (cs2 * cs2)) * cz;
#else
    amrex::Real termz = 0.0;
#endif

    amrex::Real Si = wi[q] * fac * (termx * fx + termy * fy + termz * fz);

    // add forcing *increment* on top of BGK result
    f_old(i, j, k, q) += Si;
  }
}

#endif
