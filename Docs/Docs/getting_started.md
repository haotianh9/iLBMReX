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

Following the official guidelines in (https://amrex-codes.github.io).

After installation, set the `AMREX_HOME` environment variable to point
to the installation directory if the installer does not place AMReX in a
standard location.

## Building iLBMReX

Clone this repository and create a build directory:

```sh
git clone https://github.com/haotianh9/lattice_boltzmann_method.git
cd lattice_boltzmann_method
cd Examples
cd Cylinder_flow
cmake .. 
make -j
```



This will generate one or more executables, typically `main2d.gnu.ex` and
`main2d.gnu.CUDA.ex`, depending on the compilation setting.

## Running a test case

To run the 2D cylinder wake example, execute from the build directory:

```sh
cd ./Examples/Cylinder_flow
./main2d.gnu.ex inputs
```

Output plotfiles will be written to a new directory in
`Examples/Cylinder_flow/out_cylinder`. You can visualize these using python's
`yt` package or with general‑purpose visualization software such as
ParaView or VisIt.

