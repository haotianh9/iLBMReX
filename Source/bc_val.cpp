#include "bc_val.H"

namespace BCVals {
    amrex::Vector<amrex::Real> bc_lo_rho_val(AMREX_SPACEDIM, 0.0);
    amrex::Vector<amrex::Real> bc_lo_ux_val(AMREX_SPACEDIM, 0.0);
    amrex::Vector<amrex::Real> bc_lo_uy_val(AMREX_SPACEDIM, 0.0);

    amrex::Vector<amrex::Real> bc_hi_rho_val(AMREX_SPACEDIM, 0.0);
    amrex::Vector<amrex::Real> bc_hi_ux_val(AMREX_SPACEDIM, 0.0);
    amrex::Vector<amrex::Real> bc_hi_uy_val(AMREX_SPACEDIM, 0.0);
}

