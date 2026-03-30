.. _Chap:GettingStarted:

Getting Started
===============

This guide explains how to install the required dependencies and build the
iLBMReX solver from source.

Prerequisites
~~~~~~~~~~~~~

Before compiling iLBMReX you will need:

* A C++14-capable compiler, such as GCC ≥ 9 or Clang ≥ 11
* CMake ≥ 3.22
* An MPI implementation (e.g. OpenMPI or MPICH) for parallel runs
* The AMReX library (https://amrex-codes.github.io). Optionally, build
  AMReX with GPU support if you intend to run iLBMReX on NVIDIA or AMD GPUs
* A CUDA toolkit (for NVIDIA GPUs) or ROCm (for AMD GPUs) when building
  with GPU support

Installing AMReX
~~~~~~~~~~~~~~~~

Follow the official installation guidelines at the AMReX website:
https://amrex-codes.github.io

When building with GPU support, select the appropriate backend (CUDA or HIP).
Once installed, set the ``AMREX_HOME`` environment variable so that CMake can
locate AMReX if it is not installed in a standard location.

Building iLBMReX
~~~~~~~~~~~~~~~~

Clone this repository and build one of the example problems using the provided
CMake configuration:

.. code-block:: bash

   git clone https://github.com/haotianh9/lattice_boltzmann_method.git
   cd lattice_boltzmann_method/Examples/Cylinder_flow
   cmake ..
   make -j

The build system will automatically detect available compilers and GPU support,
generating executables such as ``main2d.gnu.ex`` for CPU runs and
``main2d.gnu.CUDA.ex`` for GPU runs (if a compatible CUDA/HIP compiler is detected).

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

The root ``CMakeLists.txt`` file exposes numerous options. You can enable or
disable GPU support (``WITH_GPU``), choose the GPU backend (``AMREX_GPU_BACKEND``),
enable AMR subcycling (``WITH_SUBCYCLING``), and build in debug mode
(``CMAKE_BUILD_TYPE=Debug``). Refer to the design overview documentation for
more information about the solver architecture, and consult the source code in
the ``Source/`` directory for implementation details.
