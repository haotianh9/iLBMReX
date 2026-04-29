Introduction
===================

iLBMReX is an open-source fluid-structure interaction solver that couples the
lattice Boltzmann method (LBM) with a marker-based immersed-boundary (IB) scheme
and block-structured adaptive mesh refinement (AMR) based on the AMReX library.

It targets two- and three-dimensional flows at low to intermediate Reynolds numbers
where localized mesh refinement around moving boundaries or wake structures yields
significant efficiency gains.

Key Features
-----------

* **Lattice Boltzmann Method (LBM):** BGK collision operator with AMReX CPU/GPU kernels
* **Immersed-Boundary Coupling:** Marker-based direct-forcing approach for moving geometries
* **Adaptive Mesh Refinement (AMR):** Block-structured AMR via AMReX with dynamic regridding
* **GPU Support:** Portable GPU kernels via AMReX back-end; the provided GNUmake examples expose CUDA builds
* **Diagnostic Output:** Plotfiles compatible with ParaView, VisIt, and the yt package
* **Moving Boundaries:** Support for prescribed marker kinematics through example-local geometry hooks
* **Parallel I/O:** CheckPointing and plotfile output with AMReX native I/O

Code Philosophy
---------------

iLBMReX aims to provide research infrastructure for developing and validating adaptive
IB-LBM algorithms on both CPU and GPU hardware. It leverages the marker-based direct-forcing
immersed boundary approach while retaining the collision-streaming simplicity of LBM.
By embedding the solver within AMReX's multilevel mesh hierarchy, users can focus on
modeling, algorithmic, and physical questions rather than implementing AMR and parallelization
from scratch.

The iLBMReX source code is available at
https://github.com/haotianh9/lattice_boltzmann_method

For questions or bug reports, please open an issue on GitHub:
https://github.com/haotianh9/lattice_boltzmann_method/issues
