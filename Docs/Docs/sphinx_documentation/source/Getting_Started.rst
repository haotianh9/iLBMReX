.. _Chap:GettingStarted:

Getting Started
===============

This guide explains how to install the required dependencies and build the
iLBMReX solver from source.

Prerequisites
~~~~~~~~~~~~~

Before compiling iLBMReX you will need:

* A C++14-capable compiler, such as GCC ≥ 9 or Clang ≥ 11
* GNU Make
* An MPI implementation (e.g. OpenMPI or MPICH) for parallel runs
* The AMReX library (https://amrex-codes.github.io). This repository includes
  an AMReX submodule under ``./amrex`` used by default in ``Make.LBM``
* A CUDA toolkit if you enable CUDA builds (``USE_CUDA = TRUE`` in an example
  ``GNUmakefile``)

Installing AMReX
~~~~~~~~~~~~~~~~

Follow the official installation guidelines at the AMReX website:
https://amrex-codes.github.io

When building with GPU support, select the appropriate backend. If you do not
use the bundled ``./amrex`` submodule, set ``AMREX_HOME`` so the GNUmake build
can locate AMReX.

Building iLBMReX
~~~~~~~~~~~~~~~~

Clone this repository and build one of the example problems using the provided
GNUmake configuration:

.. code-block:: bash

   git clone --recurse-submodules https://github.com/haotianh9/lattice_boltzmann_method.git
   cd lattice_boltzmann_method/Examples/Cylinder_flow
   make -j

If AMReX is external to this repository:

.. code-block:: bash

   make AMREX_HOME=/path/to/amrex -j

This typically generates ``main2d.gnu.ex`` for 2D examples (or
``main3d.gnu.ex`` for 3D examples). CUDA variants are generated when enabled in
the example ``GNUmakefile``.

Running a Test Case
~~~~~~~~~~~~~~~~~~~

After building, run the 2D cylinder wake example:

.. code-block:: bash

   ./main2d.gnu.ex inputs

This reads the ``inputs`` file in the current directory, runs the solver and
writes plotfiles to a subdirectory such as ``out_cylinder``. You can visualize
these using tools like Python's ``yt`` package, ParaView or VisIt.

Advanced Options
~~~~~~~~~~~~~~~~

Build and runtime controls come from each example ``GNUmakefile`` and
``inputs`` file:

* ``DIM = 2`` or ``DIM = 3`` selects dimensionality
* ``USE_MPI``, ``USE_OMP``, ``USE_CUDA``, and ``DEBUG`` are compile-time toggles
* Marker IBM runs require particle support; set ``USE_PARTICLES = TRUE`` when
  ``ibm.method = 1``
* AMR and physics settings are configured in ``inputs`` (for example
  ``amr.max_level``, ``amr.ref_ratio``, ``amr.regrid_int``,
  ``lbmPhysicalParameters.nu``)

Refer to the design overview documentation for architecture details and to the
``Source/`` directory for implementation details.
