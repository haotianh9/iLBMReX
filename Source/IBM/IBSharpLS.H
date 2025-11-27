#pragma once
#include "LevelSet/LevelSet.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

struct SharpIBParams {
  amrex::Real ds{0.7}; /* reserved for later */
};

class IBSharpLS {
public:
  explicit IBSharpLS(amrex::Geometry const &gm) : m_geom(gm) {}

  // Simple direct-forcing in thin |phi|<=eps band; target wall velocity = 0
  void update_forcing(int lev, LevelSetManager &ls, amrex::MultiFab const &ucc,
                      amrex::MultiFab const &vcc, amrex::MultiFab const &wcc,
                      amrex::MultiFab &Fx, amrex::MultiFab &Fy,
                      amrex::MultiFab &Fz, amrex::Real dt) const;

private:
  amrex::Geometry m_geom;
};
