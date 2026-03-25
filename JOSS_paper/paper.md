---
title: "iLBMReX: an adaptive immersed‑boundary lattice Boltzmann solver built on AMReX"
authors:
  - Haotian Hang
  - Feihu Zhang
  - Yadong Zeng
date: 2026-03-25
---

## Summary

The **iLBMReX** project is a new adaptive fluid–structure interaction solver that combines a lattice Boltzmann method (LBM) with immersed‑boundary (IB) techniques and block‑structured adaptive mesh refinement (AMR).  Whereas existing open‑source IB–LBM codes typically operate on static uniform or multi‑resolution grids and AMR‑based immersed‑boundary solvers such as IAMReX [@Li2024Open] rely on finite‑volume Navier–Stokes discretisations, iLBMReX bridges these domains by implementing a BGK‑collision LBM on AMReX’s multi‑level hierarchy and providing a validated marker‑based IB backend for coupling immersed bodies to the fluid.  The code supports two‑ and three‑dimensional simulations on GPU‑accelerated architectures, handles dynamic regridding, and includes benchmarking and analysis tools for lift/drag evaluation.  It therefore opens a new avenue for studies that require both the low Mach‑number accuracy of LBM and the efficiency of dynamic mesh adaptation, e.g. fluid structure interaciton, dynamic stall, biological propulsion.

iLBMReX is written in modern C++, leverages the AMReX framework for parallelism and AMR, and uses `amrex::MultiFab` data structures to store distribution functions and macroscopic variables on each refinement level.  Runtime parameters define the lattice stencil (D2Q9 or D3Q19) and physical properties such as viscosity and relaxation time.  The solver advances the distribution function through collision and streaming while updating macroscopic density, velocity, vorticity and pressure. An immersed‑boundary forcing field may be applied during the collision step. The code includes built‑in diagnostics for vorticity and pressure perturbation and supports checkpointing, subcycling and flexible boundary conditions.


## Statement of Need

Adaptive immersed‑boundary lattice Boltzmann solvers have received little attention compared to finite‑volume or finite‑difference adaptive solvers.  Leading IB–LBM codes such as **Palabos** and **OpenLB** provide static multi‑block refinement or uniform grids, but neither offers generic, dynamic AMR coupled to immersed boundaries.  Conversely, AMR frameworks like **IAMReX** [@Li2024Open] support immersed boundaries but solve the incompressible Navier–Stokes equations using projection methods rather than LBM.  Researchers wishing to study fluid–structure interactions at intermediate Reynolds numbers must therefore choose between the kinetic advantages of LBM and the adaptivity offered by AMR solvers.  iLBMReX addresses this gap by delivering a general‑purpose IB‑LBM solver with dynamic mesh refinement.  It is suited for problems where the flow features evolve in time and space (e.g., vortical wakes from pitching airfoils) and where local mesh resolution improves accuracy while minimising computational cost.  The code also targets GPU‑accelerated systems through AMReX’s portable parallelism, enabling large‑scale simulations on modern heterogeneous hardware.


## Implementation and Features

### Lattice Boltzmann solver with AMR

At its core, iLBMReX implements a single‑relaxation‑time (BGK) LBM on nested grids.  The main driver class `AmrCoreLBM` derives from `amrex::AmrCore` and manages refinement levels, time stepping, checkpointing and plotfile output.  For each level, the solver stores distribution functions (`f_old`/`f_new`) and macroscopic variables (`macro_old`/`macro_new`) in `amrex::MultiFab` arrays.  A recursive `timeStepWithSubcycling` routine advances coarse and fine levels, while `ErrorEst` tags cells for refinement based on user‑specified criteria.  Subcycling in time is optional; a non‑subcycling mode propagates a single time step across all levels.  During each time step, ghost‑cell halos are filled via AMReX utilities, after which a collision–streaming–forcing loop is executed on each tile of each level (implemented in `AdvancePhiAtLevel`).  The collision uses the BGK operator with external forcing, and the streaming follows the discrete lattice directions `dirx`, `diry` and `dirz`.  Boundary conditions, including bounce‑back, periodic and prescribed‐force walls, are implemented via user‑defined `BCType` flags and ghost‑cell reflection.

### Immersed‑boundary modules

iLBMReX currently supports a **marker‑based direct‑forcing IBM** enabled with `ibm.method = 1` (`0` disables IBM; the legacy aliases `marker` and `iamr_marker` are still accepted).  Inspired by IAMReX’s multi‑direct forcing approach and earlier immersed‑boundary lattice Boltzmann formulations [@Wu2009implicit], this backend uses Lagrangian markers constrained to the finest grid.  Markers interpolate velocity from the Eulerian mesh and spread Lagrangian forces using a smoothed delta kernel; a force iteration enforces the desired boundary motion.  The `IBMarkerDF` class (defined in `IBMarkerDF.H`) ports the core coupling stages from IAMReX’s diffused immersed‑boundary implementation and uses AMReX particle containers for marker storage.  Since markers live only on the finest level, memory usage remains modest despite adaptivity.

The solver reports integrated body forces and drag/lift coefficients from the coupled forcing field and from the Lagrangian marker reaction accumulated by the marker IBM.  These diagnostics are written during runtime and are used in the current validation cases.

### Diagnostics, forcing and GPU support

During each time step the solver calculates macroscopic fields (density, velocity, vorticity and pressure) from the post‑streaming distribution functions.  A diagnostic kernel (`visPara`) computes vorticity and pressure perturbation using finite differences on the GPU.  Body forces can be applied on all levels (controlled by `m_use_prescribed_force`), and the solver includes hooks for Guo forcing.  For validation or regression tests, forcing terms and IB forcing can be output for analysis.  The code uses AMReX’s GPU porting layer throughout; collision, streaming and force spreading kernels are decorated with `AMREX_GPU_DEVICE` macros and therefore execute on CUDA or HIP devices when available, while falling back to OpenMP on CPUs.

### Example problems and extensibility

The repository contains example input files demonstrating canonical flows: lid‑driven cavity and Couette flows, laminar cylinder flow and square‑duct Poiseuille flow.  Boundary conditions, physical parameters and IB options are specified in `inputs_*` files.  For example, enabling the marker IBM with a rectangular obstacle requires setting `ibm.method = 1` and providing geometry parameters for the box.  Two additional test cases—a two‑dimensional pitching airfoil and a three‑dimensional sphere—are under development and will be included before submission.  Users can extend the code by adding new immersed‑boundary geometries, force models or adaptive tagging criteria without modifying the core LBM integrator.  Contributions are welcome via pull requests, and detailed instructions for running and analysing test cases are provided in the repository.


## Relation to Previous Work

iLBMReX draws on several strands of research.  Its immersed‑boundary implementation is adapted from the IAMReX framework [@Li2024Open], which combines multi‑direct‑forcing IBM with AMR for finite‑volume solvers.  In the marker IBM, iLBMReX uses the same Lagrangian forcing iteration and spatial hashing used in IAMReX, but couples them to an LBM fluid solver instead of a projection‑based Navier–Stokes solver.  The code also adopts the practice of restricting markers to the finest level, reducing the memory footprint and improving cache locality.

Compared with general‑purpose LBM libraries, iLBMReX uniquely offers dynamic AMR and IB coupling.  **Palabos** and **OpenLB** provide static multi‑block refinement or uniform grids but do not implement dynamic AMR with immersed boundaries.  LAMBReX demonstrates LBM on AMReX but lacks IB support.  By porting IAMReX’s IBM techniques into an LBM solver, iLBMReX combines the low‑Mach advantages of LBM with AMReX’s adaptive capabilities; for the corresponding finite‑volume IAMReX framework, see [@Li2024Open].  The resulting software enables efficient simulation of intermediate‑Reynolds‑number FSI problems that would otherwise require either coarse uniform grids or more complex finite‑volume solvers.


## Acknowledgements

The authors thank discussions with the IAMReX developers, whose framework [@Li2024Open] informed the implementation of the marker IBM. The authors thank Yaning Wang for hep discussion on validation of IB-LBM. 
We also acknowledge the AMReX development team at Lawrence Berkeley National Laboratory for creating the infrastructure that underpins iLBMReX. 
This work has been inspired by the open‑source lattice Boltzmann and immersed‑boundary communities, and we hope iLBMReX will, in turn, contribute to that ecosystem.
