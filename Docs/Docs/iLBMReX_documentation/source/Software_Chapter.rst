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

All computational kernels are implemented as portable CUDA/HIP/CPU code using
AMReX's ``AMREX_FOR_*`` macros and ``ParallelFor`` constructs, enabling
efficient execution on both CPU and GPU devices.

Source Code Organization
~~~~~~~~~~~~~~~~~~~~~~~~~

The source code is organized in the ``Source/`` directory with the following
major components:

* ``Solver/`` - Main solver driver and time advancement
* ``DataTypes/`` - Data structure and field definitions
* ``Utils/`` - Utility functions and I/O routines
* ``IB/`` - Immersed boundary implementation
* ``Init/`` - Problem initialization and input parsing
