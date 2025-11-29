#pragma once
#include "LevelSet/LevelSet.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>


AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real Hsmooth(amrex::Real s, amrex::Real eps) noexcept
{
    using amrex::Real;
    if (s <= -eps) return Real(0.0);
    if (s >=  eps) return Real(1.0);

    constexpr Real pi = Real(3.14159265358979323846);
    Real t = Real(0.5) + Real(0.5) * s / eps
           + Real(0.5) / M_PI * std::sin(M_PI * s / eps);
    return t;
}


struct DiffuseIBParams {
  amrex::Real alpha{2.0};
};

class IBDiffuseLS {
public:
  explicit IBDiffuseLS(amrex::Geometry const &gm) : m_geom(gm) {}

  // Stationary solid (u_wall = 0). Writes cell-centered body force (Fx,Fy,Fz)
  void update_forcing(int lev, LevelSetManager &ls, amrex::MultiFab const &ucc,
                      amrex::MultiFab const &vcc, amrex::MultiFab const &wcc,
                      amrex::MultiFab &Fx, amrex::MultiFab &Fy,
                      amrex::MultiFab &Fz, amrex::Real alpha) const;

private:
  amrex::Geometry m_geom;
};
