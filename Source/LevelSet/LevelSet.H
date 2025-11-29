#ifndef LEVELSET_H_
#define LEVELSET_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

struct LevelSetParams {
  amrex::Real eps = 1.5;   // smoothing length
  amrex::Real alpha = 2.0; // scaling factor, not strictly needed
  amrex::Real x0 = 12.0;   // cylinder center x
  amrex::Real y0 = 60.0;   // cylinder center y
  amrex::Real z0 = 0.0;    // cylinder center z (for 3D)
  amrex::Real R = 12.0;    // cylinder radius
};

class LevelSetManager {
public:
  LevelSetManager() = default;

  // Allocate φ on this level
  void define_level(int lev, amrex::BoxArray const &ba,
                    amrex::DistributionMapping const &dm, int ng);

  // Build signed distance φ from a cylinder
  void build_from_cylinder(int lev, amrex::Geometry const &geom,
                           LevelSetParams const &par);

  // Query & accessors used in AmrCoreLBM
  bool has_level(int lev) const;

  amrex::MultiFab &phi_at(int lev);
  amrex::MultiFab const &phi_at(int lev) const;

private:
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_phi;
};

#endif // LEVELSET_H_
