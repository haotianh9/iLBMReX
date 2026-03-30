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
refinement based on vorticity, density gradients and (for IB cases) proximity to
immersed boundaries. Subcycling in time is optional and can reduce computational
cost by taking larger time steps on coarser levels.

Lattice Boltzmann Update
~~~~~~~~~~~~~~~~~~~~~~~~

Within each time step the solver performs the following operations in sequence:

1. **Collision:** Apply the Bhatnagar–Gross–Krook (BGK) relaxation operator to
   compute post-collision distribution functions.

2. **Immersed-boundary forcing:** If IB markers are present, interpolate velocity
   from the Eulerian grid to the Lagrangian markers, compute forcing from body
   motion and spread forces back to the grid.

3. **Streaming:** Shift distribution functions along discrete lattice directions.

4. **Boundary conditions:** Apply bounce-back or no-slip conditions on solid walls,
   periodic boundaries on domain edges, and moving wall conditions where specified.

5. **Macroscopic update:** Compute density and velocity moments from the distribution
   functions; compute derived quantities such as vorticity or pressure perturbation.

These operations are implemented as portable CUDA/HIP/CPU kernels using AMReX's
parallel-for constructs.

Immersed-Boundary Coupling
~~~~~~~~~~~~~~~~~~~~~~~~~~

iLBMReX implements a direct-forcing IB method using Lagrangian markers attached
to immersed bodies. Markers exist on the finest AMR level to simplify interpolation
and force spreading. Supported geometries include circles, spheres, axis-aligned
boxes and user-defined marker sets. Moving bodies are specified via time-dependent
position and velocity functions read from the inputs file. Tagging for IB cases
ensures that the region surrounding the body remains on the finest level throughout
the simulation.

Diagnostics and I/O
~~~~~~~~~~~~~~~~~~~

The solver writes plotfiles at user-defined intervals. Each plotfile contains
macroscopic fields (density, velocity components), vorticity magnitude, pressure
perturbation and force contributions. For IB cases, integrated lift and drag forces
are written to a separate text file. Checkpoint files are also supported.
Post-processing can be performed with AMReX's built-in tools or with generic
visualization software (e.g. ParaView or VisIt).
