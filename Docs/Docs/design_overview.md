# Design overview

iLBMReX combines three ingredients: the lattice Boltzmann method (LBM),
marker‑based immersed‑boundary (IB) coupling and block‑structured
adaptive mesh refinement (AMR) provided by the AMReX library. This
overview summarizes the key components and design choices.

## AMR hierarchy

The solver inherits from `amrex::AmrCore` through the class
`AmrCoreLBM`. This driver manages a hierarchy of refinement levels with
user‑specified maximum level and refinement ratios. At each regridding
step the solver tags cells for refinement based primarily on vorticity and,
for marker/cylinder IBM cases, optional level-set/body proximity criteria.
The current time-integration path advances with AMR subcycling by refinement
ratio (`timeStepWithSubcycling`).

## Lattice Boltzmann update

Within each time step the solver performs the following operations in
sequence:

1. **FillPatch stage:** synchronize mesoscopic/macroscopic state and ghost
   cells for the current level.
2. **Immersed-boundary force construction (if enabled on finest level):**
   interpolate Eulerian velocity to markers, compute marker forces (explicit
   or IVC), and spread to Eulerian forcing fields.
3. **Collision:** apply BGK relaxation, using the forcing term in the
   `collide_forced` kernel.
4. **Boundary and ghost handling:** fill internal/periodic boundaries and apply
   supported wall treatments (including `user_1` bounce-back handling).
5. **Streaming:** pull-stream distributions into `f_new`.
6. **Macroscopic update:** reconstruct `rho`, velocity, and derived fields
   (vorticity and pressure component in `macro_new`).

These operations are implemented with AMReX CPU/GPU parallel-for constructs.

## Immersed‑boundary coupling

iLBMReX implements a direct‑forcing IB method using Lagrangian markers
attached to immersed bodies. Markers exist on the finest AMR level to
simplify interpolation and force spreading. Supported geometries include
circle/sphere shells (`marker_geometry = cylinder`), axis-aligned marker boxes
(`marker_geometry = box`, currently 2D-only), and user-defined geometries via
`IBMUserDefinedGeometry.H` in an example directory.
Rigid-body motion can be prescribed through translational/rotational inputs
(`ub*`, `om*`), or fully custom marker-state logic in the user-defined
geometry hook.

## Diagnostics and I/O

The solver writes plotfiles at user‑defined intervals. Each plotfile
contains macroscopic fields (density, velocity components), vorticity
magnitude, pressure, all `f_i` distribution components, and optionally level-set
`phi` (for cylinder/level-set paths). IB force fields are not currently written
as dedicated plotfile components; integrated force diagnostics are written to a
separate text file (default `force.dat`) when `ibm.force_interval > 0`.
Checkpoint files are also supported. Post-processing can be performed with
AMReX-compatible tools (e.g. ParaView, VisIt, yt).
