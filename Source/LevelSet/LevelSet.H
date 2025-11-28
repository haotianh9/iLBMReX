#pragma once
#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

struct LSParams {
  amrex::Real x0{120.0}, y0{60.0}, z0{0.0}, R{12.0}, eps{1.5}, alpha{2.0};
  int reinit_niter{0}; // reserved for future
};

class LevelSetManager {
public:
  LevelSetManager() = default;

  void define_level(int lev, const amrex::BoxArray &ba,
                    const amrex::DistributionMapping &dm, int ng = 1) {
    if ((int)m_phi.size() <= lev)
      m_phi.resize(lev + 1);
    m_phi[lev].define(ba, dm, 1, ng);
  }

  bool has_level(int lev) const {
    return (int)m_phi.size() > lev && m_phi[lev].ok();
  }

  amrex::MultiFab &phi_at(int lev) { return m_phi[lev]; }
  amrex::MultiFab const &phi_at(int lev) const { return m_phi[lev]; }

  // Build signed-distance φ = distance-to-center - R
  void build_from_cylinder(int lev, amrex::Geometry const &gm,
                           LSParams const &p);

private:
  amrex::Vector<amrex::MultiFab> m_phi;
};
