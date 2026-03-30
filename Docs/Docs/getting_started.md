# Getting started

This guide explains how to install the required dependencies and build the
iLBMReX solver from source.

## Prerequisites

Before compiling iLBMReX you will need:

* A C++14‑capable compiler, such as GCC ≥ 9 or Clang ≥ 11.
* CMake ≥ 3.22.
* An MPI implementation (e.g. OpenMPI or MPICH) for parallel runs.
* The AMReX library (https://amrex-codes.github.io). Optionally, build
  AMReX with GPU support if you intend to run iLBMReX on NVIDIA or AMD
  GPUs.
* A CUDA toolkit (for NVIDIA GPUs) or ROCm (for AMD GPUs) when building
  with GPU support.

## Installing AMReX

Follow the official installation guidelines at the AMReX website
(<https://amrex-codes.github.io>). When building with GPU support,
select the appropriate backend (CUDA or HIP). Once installed, set the
`AMREX_HOME` environment variable so that CMake can locate AMReX if it
is not installed in a standard location.

## Building iLBMReX

Clone this repository and build the example problems using the provided
Makefiles.  Each example directory under ``Examples/`` contains a
Makefile that automatically locates AMReX and builds both CPU and
GPU executables (if a CUDA or HIP compiler is available).

For instance, to build the cylinder‑flow example:

```sh
git clone https://github.com/haotianh9/lattice_boltzmann_method.git
cd lattice_boltzmann_method/Examples/Cylinder_flow
make -j        # builds main2d and (if available) GPU executables
```

The default target builds an executable named `main2d.gnu.ex` for CPU
runs.  When a compatible GPU compiler is detected (e.g. ``nvcc`` or
``hipcc``), the Makefile also builds a GPU version (usually
``main2d.gnu.CUDA.ex`` or similar).  You can run the build with
``make VERBOSE=1`` to see which commands are executed.

## Running a test case

From within the example directory you can run the 2D cylinder case
directly after building:

```sh
./main2d.gnu.ex inputs
```

This reads the `inputs` file in the current directory, runs the solver
and writes plotfiles to a subdirectory such as `out_cylinder`.  Use
tools like Python’s `yt` package, ParaView or VisIt to visualise the
results.

## Advanced options

The root `CMakeLists.txt` file exposes numerous options. You can enable
or disable GPU support (`WITH_GPU`), choose the GPU backend
(`AMREX_GPU_BACKEND`), enable AMR subcycling (`WITH_SUBCYCLING`), and
build in debug mode (`CMAKE_BUILD_TYPE=Debug`). Refer to
`Docs/design_overview.md` for more information about the solver
architecture and to the source code in the `Source/` directory for
implementation details.