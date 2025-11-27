#pragma once
#include "LevelSet/LevelSet.H"
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

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
