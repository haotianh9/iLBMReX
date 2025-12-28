#pragma once

#include <AMReX_REAL.H>

// Parameters for IAMReX-style marker-based direct-forcing IBM.
//
// This is intentionally kept as a POD to simplify I/O via ParmParse.
struct MarkerIBParams {
  // Discrete delta kernel type (matches IAMReX enum ordering)
  //   0 = FOUR_POINT_IB
  //   1 = THREE_POINT_IB
  int delta_type{0};

  // Number of multi-direct-forcing iterations (IAMReX: loop_ns / loop_solid)
  int loop_ns{2};

  // Optional override for number of Lagrangian markers.
  // If 0, a heuristic is used (IAMReX-like for spheres).
  int n_marker{0};

  // IAMReX 'rd' parameter used in their Ml/dv formula for spherical shells.
  // (Defaults to 0.0; keep it configurable for reproducibility.)
  amrex::Real rd{0.0};

  // Rigid-body kinematics (prescribed here; your code currently uses a fixed
  // body).
  amrex::Real ubx{0.0};
  amrex::Real uby{0.0};
  amrex::Real ubz{0.0};
  amrex::Real omx{0.0};
  amrex::Real omy{0.0};
  amrex::Real omz{0.0};

  // Diagnostics
  int verbose{0};
};
