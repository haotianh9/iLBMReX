#ifndef feqH
#define feqH

#include <AMReX_Array4.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE

amrex::Real feqFunction(amrex::Vector<amrex::Real> mesoscopic,
                        amrex::Vector<amrex::Real> macro) {
  amrex::Real result;
  amrex::Real eu, uv;
  amrex::Real wi, dirx, diry, dirz, rho, u, v, w;

  wi = mesoscopic[0];
  dirx = mesoscopic[1];
  diry = mesoscopic[2];
  dirz = mesoscopic[3];

  rho = macro[0];
  u = macro[1];
  v = macro[2];
  w = macro[3];

  eu = dirx * u + diry * v
#if (AMREX_SPACEDIM == 3)
       + dirz * w
#endif
      ;
  uv = u * u + v * v
#if (AMREX_SPACEDIM == 3)
       + w * w
#endif
      ;

  result = wi * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);

  return result;
}

#endif
