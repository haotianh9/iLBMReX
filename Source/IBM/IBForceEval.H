#pragma once

#include "LevelSet/LevelSet.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

namespace IBForceEval {

// Link-wise momentum-exchange force on the body (reaction), using a geometric
// fluid/solid indicator from the cylinder/sphere signed distance.
//
// Inputs:
//   - f_cc: cell-centered LBM populations (ndir components)
//   - geom:   level geometry
//   - dirx/y/z: discrete velocity set entries (expected in {-1,0,1})
//   - ndir: number of directions
//   - par: cylinder/sphere center and radius
//
// Returns:
//   - body-force components (Fx,Fy,Fz) in lattice-force units per step.
amrex::GpuArray<amrex::Real, 3>
ComputeMomentumExchangeBodyForce(amrex::MultiFab const &f_cc,
                                 amrex::Geometry const &geom,
                                 amrex::Vector<amrex::Real> const &dirx,
                                 amrex::Vector<amrex::Real> const &diry,
                                 amrex::Vector<amrex::Real> const &dirz,
                                 int ndir,
                                 LevelSetParams const &par,
                                 bool one_sided_bounceback = false);

} // namespace IBForceEval
