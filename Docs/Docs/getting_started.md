# Getting started

This guide explains how to install the required dependencies and build the
iLBMReX solver from source.

## Prerequisites

Before compiling iLBMReX you will need:

* A C++14‑capable compiler, such as GCC ≥ 9 or Clang ≥ 11.
* GNU Make.
* An MPI implementation (e.g. OpenMPI or MPICH) for parallel runs.
* The AMReX library (https://amrex-codes.github.io). This repository includes
  an AMReX submodule under `./amrex`, which is the default used by `Make.LBM`.
* A CUDA toolkit if you enable CUDA builds (`USE_CUDA = TRUE` in an example
  `GNUmakefile`).

## Installing AMReX

Follow the official installation guidelines at the AMReX website
(<https://amrex-codes.github.io>). When building with GPU support,
select the appropriate backend. If you are not using the bundled `./amrex`
submodule, set `AMREX_HOME` so the GNUmake build can find AMReX.

## Building iLBMReX

Clone this repository and build the example problems using the provided
Makefiles. Each example directory under `Examples/` contains a
`GNUmakefile` that includes `Make.LBM` and builds an executable for that case.

For instance, to build the cylinder‑flow example:

```sh
git clone --recurse-submodules https://github.com/haotianh9/lattice_boltzmann_method.git
cd lattice_boltzmann_method/Examples/Cylinder_flow
make -j
```

If AMReX is external to this repository, provide it explicitly:

```sh
make AMREX_HOME=/path/to/amrex -j
```

The default target usually builds `main2d.gnu.ex` (or `main3d.gnu.ex` for 3D
cases). You can run `make VERBOSE=1` to inspect the full compile and link
commands.

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

Build/runtime knobs are controlled through each example's `GNUmakefile` and
its `inputs` file:

* `DIM = 2` or `DIM = 3` selects dimensionality.
* `USE_MPI`, `USE_OMP`, `USE_CUDA`, and `DEBUG` are compile-time toggles.
* IBM marker runs require particle support; set `USE_PARTICLES = TRUE` for
  cases that enable marker IBM (`ibm.method = 1`).
* AMR behavior (e.g. `amr.max_level`, `amr.ref_ratio`, `amr.regrid_int`) and
  physics parameters (e.g. `lbmPhysicalParameters.nu`) are configured in
  `inputs`.

Refer to `Docs/Docs/design_overview.md` and the implementation under
`Source/` for details.
