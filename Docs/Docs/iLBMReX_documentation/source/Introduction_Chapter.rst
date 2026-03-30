Introduction
===================

iLBMReX is an open-source fluid-structure interaction solver that couples the
lattice Boltzmann method (LBM) with a marker-based immersed-boundary (IB) scheme
and block-structured adaptive mesh refinement (AMR) based on the AMReX library.

It targets two- and three-dimensional flows at low to intermediate Reynolds numbers
where localized mesh refinement around moving boundaries or wake structures yields
significant efficiency gains.

The solver is implemented in C++ with portable GPU kernels via the AMReX back-end
and supports dynamic regridding, checkpointing, and run-time diagnostics.

Key Features
-----------

* **Lattice Boltzmann Method (LBM):** BGK collision operator with portable CUDA/HIP/CPU kernels
* **Immersed-Boundary Coupling:** Marker-based direct-forcing approach for moving geometries
* **Adaptive Mesh Refinement (AMR):** Block-structured AMR via AMReX with dynamic regridding
* **GPU Support:** Portable GPU kernels via AMReX back-end (CUDA, HIP, or CPU execution)
* **Diagnostic Output:** Plotfiles compatible with ParaView, VisIt, and the yt package
* **Moving Boundaries:** Support for time-dependent position and velocity functions
* **Parallel I/O:** Checkpointing and plotfile output

Research Infrastructure
-----------------------

iLBMReX is intended as research infrastructure that enables users to develop, validate,
and compare adaptive IB-LBM algorithms on CPU and GPU hardware. The code base leverages
the marker-based direct-forcing immersed boundary coupling while retaining the
collision-streaming simplicity of LBM. By embedding the solver within AMReX's multilevel
mesh hierarchy, users can focus on modeling, algorithmic, and physical questions rather
than implementing AMR and parallelization from scratch.

Example Simulations
-------------------

Example simulations included in this repository cover:

* **Canonical validation cases:** Couette flow and Taylor-Green vortex decay
* **Immersed-boundary benchmarks:** Flow past a cylinder, flow over a sphere, pitching NACA 0012 airfoil
* **3D cases:** Immersed cavity box and duct flow

Each example includes an inputs file and, when applicable, Python scripts for
visualizing output fields from AMReX plotfiles.

For Getting Started
-------------------

For installation instructions, building the solver, and running test cases,
see the :ref:`Getting Started <Chap:GettingStarted>` section.

For algorithm and software details, see the :ref:`Algorithm and Software <Chap:Algorithm>`
section or consult the Markdown documentation in the parent ``Docs/`` directory.
