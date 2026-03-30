.. iLBMReX documentation master file, created by
   sphinx-quickstart on Sun Nov 12 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _ilbmrex_doc_index:

Welcome to iLBMReX's documentation!
===================================

iLBMReX is an open-source fluid-structure interaction solver that couples the
lattice Boltzmann method (LBM) with a marker-based immersed-boundary (IB) scheme
and block-structured adaptive mesh refinement (AMR) based on the AMReX library.
It targets two- and three-dimensional flows at low to intermediate Reynolds numbers
where localized mesh refinement around moving boundaries or wake structures yields
significant efficiency gains.

The solver is implemented in C++ with portable GPU kernels via the AMReX back-end
and supports dynamic regridding, checkpointing, and run-time diagnostics.

The iLBMReX source code is available at
https://github.com/haotianh9/lattice_boltzmann_method

.. note::
   For quick start instructions, example problems, and design notes, please see
   the Markdown documentation in the main Docs folder.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Introduction_Chapter
   Getting_Started
   SetupAndRunning
   Visualization_Chapter
   AlgorithmAndSoftware
   Software
   FluidEquations
   Debugging
   Contributing

.. toctree::
   :caption: References

   references

For licensing information, please see the LICENSE file in the
iLBMReX home directory.
