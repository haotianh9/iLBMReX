#ifndef feqH
#define feqH

#include <AMReX_Array4.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE

amrex::Real feqFunction(amrex::Real wi, amrex::Real dirx, amrex::Real diry,
                        amrex::Real dirz, amrex::Real rho, amrex::Real u,
                        amrex::Real v, amrex::Real w) {
  amrex::Real result;
  amrex::Real eu, uv;

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
