Software Architecture
=====================

This chapter describes the iLBMReX software architecture. For detailed implementation
information, consult the following Markdown documentation in the parent ``Docs/``
directory:

* ``design_overview.md`` - Overview of the solver architecture
* ``api_reference.md`` - Reference to key classes and functions

Key Components
~~~~~~~~~~~~~~

**AmrCoreLBM Driver**
  Inherits from ``amrex::AmrCore`` and manages the AMR hierarchy, regridding,
  and time advancement. Handles synchronization and interpolation between AMR levels.

**Lattice Boltzmann Kernel**
  Implements the BGK collision operator, streaming, boundary conditions, and
  macroscopic moment calculations using AMReX's parallel-for constructs.

**Immersed Boundary Module**
  Manages Lagrangian markers, performs Eulerian-Lagrangian interpolation,
  computes immersed boundary forces, and spreads forces back to the grid.

**I/O and Diagnostics**
  Writes plotfiles in AMReX native format, checkpoint files for restarting,
  and integrated diagnostics (lift, drag) for immersed boundary simulations.

GPU Implementation
~~~~~~~~~~~~~~~~~~

Computational kernels use AMReX ``ParallelFor`` constructs and GPU-device
annotations, enabling CPU execution and configured AMReX GPU backends. The
provided GNUmake examples expose CUDA builds through ``USE_CUDA``.

Source Code Organization
~~~~~~~~~~~~~~~~~~~~~~~~~

The source code is organized in the ``Source/`` directory with the following
major components:

* ``Core/`` - AMR driver, time advancement, I/O, and input parsing
* ``LBM/`` - Collision, streaming, forcing, and macroscopic kernels
* ``BC/`` - Physical boundary-condition helpers and prescribed boundary values
* ``IBM/`` - Marker-based immersed-boundary implementation and force diagnostics
* ``LevelSet/`` - Signed-distance support for cylinder-style geometry/refinement
* ``Utils/`` - Utility helpers
