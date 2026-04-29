Algorithm and Software
======================

iLBMReX combines three key ingredients: the lattice Boltzmann method (LBM),
marker-based immersed-boundary (IB) coupling and block-structured adaptive mesh
refinement (AMR) provided by AMReX. This chapter summarizes the key components
and design choices.

AMR Hierarchy
~~~~~~~~~~~~~

The solver inherits from ``amrex::AmrCore`` through the class ``AmrCoreLBM``.
This driver manages a hierarchy of refinement levels with user-specified maximum
level and refinement ratios. At each regridding step the solver tags cells for
refinement based primarily on vorticity and, for marker/cylinder IBM cases,
optional level-set/body proximity criteria. The current time integration path
advances with AMR subcycling by refinement ratio.

Lattice Boltzmann Update
~~~~~~~~~~~~~~~~~~~~~~~~

Within each time step the solver performs the following operations in sequence:

1. **FillPatch stage:** synchronize mesoscopic/macroscopic state and ghost cells.

2. **Immersed-boundary force construction (finest level if enabled):**
   interpolate Eulerian velocity to markers, compute marker forces
   (explicit or IVC), and spread them back to Eulerian forcing fields.

3. **Collision:** apply BGK relaxation with forcing through the
   ``collide_forced`` kernel.

4. **Boundary and ghost handling:** fill internal/periodic boundaries and apply
   supported wall treatments (including ``user_1`` bounce-back handling).

5. **Streaming:** pull-stream distributions into the new state.

6. **Macroscopic update:** reconstruct density, velocity, vorticity, and pressure
   components.

These operations are implemented with AMReX CPU/GPU parallel-for constructs.

Immersed-Boundary Coupling
~~~~~~~~~~~~~~~~~~~~~~~~~~

iLBMReX implements a direct-forcing IB method using Lagrangian markers attached
to immersed bodies. Markers exist on the finest AMR level to simplify interpolation
and force spreading. Supported marker geometries include circle/sphere shells
(``marker_geometry = cylinder``), axis-aligned boxes (``marker_geometry = box``,
currently 2D-only), and user-defined marker sets via
``IBMUserDefinedGeometry.H``.

Diagnostics and I/O
~~~~~~~~~~~~~~~~~~~

The solver writes plotfiles at user-defined intervals. Each plotfile contains
macroscopic fields (density, velocity components), vorticity magnitude, pressure
diagnostics, and all distribution-function components. For cylinder/level-set
paths, ``phi`` is also written. IB force fields are not currently written as
dedicated plotfile components; integrated force diagnostics are written to a
separate text file (default ``force.dat``) when ``ibm.force_interval > 0``.
Checkpoint files are also supported. Post-processing can be performed with
AMReX-compatible tools (e.g. ParaView, VisIt, yt).
