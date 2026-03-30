<div align="center">
  <h1>iLBMReX</h1>
<!-- <img src="./README_figures/IAMReX.png" alt="title" width="300"> -->
<!-- <p align="center">
  <a href="https://joss.theoj.org/papers/10.21105/joss.08080">
  <img src="https://img.shields.io/badge/JOSS-10.21105%2Fjoss.08080-green" alt="JOSS">
  </a>
  <a href="https://pubs.aip.org/aip/pof/article-abstract/36/11/113335/3320064/An-open-source-adaptive-solver-for-particle">
  <img src="https://img.shields.io/badge/Physics%20of%20Fluids-10.1063%2F5.0236509-blue" alt="Physics of Fluids">
  </a>
  <a href="https://arc.aiaa.org/doi/abs/10.2514/6.2025-1865">
  <img src="https://img.shields.io/badge/AIAA-10.2514%2F6.2025--1865-orange" alt="AIAA">
  </a>
</p> -->

  <p><strong>Immersed-Boundary Lattice Boltzmann Solver with Adaptive Mesh Refinement</strong></p>

[Overview](#overview) -
[Sample Cases](#sample-cases) -
[Getting started](#getting-started) -
[State of the Field](#state-of-the-field) -
[Get Help](#get-help) -
[Contribute](#contribute) -
[Citation](#citation) -
[License](#license) -
[Acknowledgements](#acknowledgements) -
[Contact](#contact)

</div>

## Overview

iLBMReX is an open-source fluid-structure interaction solver that couples the lattice Boltzmann method (LBM) with a marker-based immersed-boundary (IB) scheme and block-structured adaptive mesh refinement (AMR) based on the AMReX library. It targets two- and three-dimensional flows at low to intermediate and Reynolds numbers where localized mesh refinement around moving boundaries or wake structures yields significant efficiency gains.

The solver is implemented in C++ with portable GPU kernels via the AMReX back-end and supports dynamic regridding, checkpointing, and run-time diagnostics. Example simulations included in this repository cover canonical validation cases such as Couette flow and Taylor-Green vortex decay, immersed-boundary benchmarks such as flow past a cylinder, an immersed cavity box, a pitching NACA 0012 airfoil, and flow over a sphere, as well as a three-dimensional duct flow.

iLBMReX is intended as research infrastructure: it enables users to develop, validate, and compare adaptive IB-LBM algorithms on CPU and GPU hardware. The code base leverages the marker-based direct-forcing IB coupling from IAMReX while retaining the collision-streaming simplicity of LBM. By embedding the solver within AMReX's multilevel mesh hierarchy, users can focus on modeling, algorithmic, and physical questions rather than implementing AMR and parallelization from scratch.

## Sample Cases

The `Examples/` directory contains input files and post-processing scripts for a variety of test problems:

- **Taylor-Green vortex decay**: transient vortex decay with an analytic solution, useful for validation.
- **Couette flow**: planar shear flow with an analytic solution, useful for validation.
- **Square duct (3D)**: steady laminar flow in a square duct that demonstrates the solver's 3D capabilities.
- **Cylinder wake**: flow past a circular cylinder at moderate Reynolds number with IB markers attached to the finest AMR level; outputs include lift and drag coefficients.
- **Immersed cavity box**: IB markers representing a box inside a cavity; tests 3D IB coupling.
- **Pitching NACA 0012 airfoil**: prescribed periodic pitching motion in 2D; illustrates support for moving geometries.
- **Flow over a sphere**: 3D flow past a sphere; illustrates IBM support in 3D flows.

Each example includes an `inputs` file and, when applicable, Python scripts for visualizing output fields from AMReX plotfiles.

## Getting started

This guide explains how to install the required dependencies and build the
iLBMReX solver from source.

## Prerequisites

Before compiling iLBMReX you will need:

* A C++14-capable compiler, such as GCC >= 9 or Clang >= 11.
* CMake >= 3.22.
* An MPI implementation (e.g. OpenMPI or MPICH) for parallel runs.
* The AMReX library (<https://amrex-codes.github.io>). Optionally, build
  AMReX with GPU support if you intend to run iLBMReX on NVIDIA or AMD
  GPUs.
* A CUDA toolkit (for NVIDIA GPUs) or ROCm (for AMD GPUs) when building
  with GPU support.

## Installing AMReX

Follow the official AMReX installation guidelines at
<https://amrex-codes.github.io>.

After installation, set the `AMREX_HOME` environment variable to point
to the installation directory if the installer does not place AMReX in a
standard location.

## Building iLBMReX

Clone this repository and build the cylinder-flow example:

```sh
git clone https://github.com/haotianh9/lattice_boltzmann_method.git
cd lattice_boltzmann_method
cd Examples
cd Cylinder_flow
cmake ..
make -j
```

This will generate one or more executables, typically `main2d.gnu.ex` and
`main2d.gnu.CUDA.ex`, depending on the dimensionality compiled.

## Running a test case

To run the 2D cylinder wake example, execute:

```sh
cd ./Examples/Cylinder_flow
./main2d.gnu.ex inputs
```

Output plotfiles will be written to a new directory in
`Examples/Cylinder_flow/out_cylinder`. You can visualize these using
Python's `yt` package or with general-purpose visualization software
such as ParaView or VisIt.

## State of the Field

Adaptive immersed-boundary LBM solvers are less common than either uniform-grid LBM codes or adaptive finite-volume / finite-difference IB solvers. Established LBM frameworks such as Palabos and OpenLB provide mature collision-streaming infrastructures, but dynamic AMR is not their primary focus. Conversely, IBAMR and IAMReX support adaptive immersed-boundary simulations for the incompressible Navier-Stokes equations, but use projection-based finite-volume or finite-difference methods.

iLBMReX offers a different combination: a BGK LBM solver with immersed-boundary coupling, dynamic AMR through AMReX, and portable GPU execution within a single code base. Researchers who need both AMR and the algorithmic simplicity of LBM can use iLBMReX as a starting point for developing and validating new methods.

## Get Help

To ask questions or report issues, please open a GitHub issue on this repository. For general discussion or feature requests, GitHub Discussions may be added in the future.

When reporting a bug, please include information about your platform, compiler, and problem setup.

## Contribute

Contributions of all kinds are welcome, including documentation improvements, bug fixes, new test problems, and new solver capabilities.

To contribute:

- submit a pull request against the main branch;
- make sure new features include tests or examples when appropriate; and
- follow the development workflow described in [CONTRIBUTING.md](CONTRIBUTING.md).

If you use iLBMReX in your own GitHub projects, consider adding `iLBMReX` as a repository topic to help others discover related work.

## Citation

If you use iLBMReX in your research, please cite the JOSS paper associated with this software:

```bibtex
<!-- @article{Hang2026iLBMReX,
  author = {Hang, Haotian and Zhang, Feihu and Zeng, Yadong},
  title   = {iLBMReX: an adaptive immersed-boundary lattice Boltzmann solver built on AMReX},
  journal = {Journal of Open Source Software},
  year    = {2026},
  doi     = {},
} -->
```

Please also check `JOSS_paper/paper.bib` for the most up-to-date citation information.

## License

This project is released under the terms of the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Acknowledgements

We thank the AMReX development team for providing the underlying mesh, I/O, and GPU infrastructure, and the IAMReX developers for helpful discussions on immersed-boundary implementations. We also acknowledge the broader open-source community whose contributions to AMReX, Palabos, OpenLB, and IBAMR helped make this work possible.

## Contact

For questions about the software or collaboration opportunities, please contact Haotian Hang at `hanghaotian@gmail.com`.
