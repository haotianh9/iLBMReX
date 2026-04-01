---
title: "iLBMReX: an adaptive immersed-boundary lattice Boltzmann solver built on AMReX"
tags:

  - Fluid dynamics
  - Computational Fluid dynamics
  - Lattice Boltzmann Method (LBM)
  - Immersed Boundary Method (IBM)
  - Adaptive Mesh Refinement (AMR)
  - AMReX
authors:
  - name: Haotian Hang
    orcid: 0000-0001-5217-8124
    corresponding: true 
    affiliation: "1,2"
  - name: Feihu Zhang
    orcid: 0000-0000-0000-0000
    affiliation: "3"
  - name: Yadong Zeng
    orcid: 0000-0000-0000-0000
    affiliation: "4"
affiliations:
  - index: 1
    name: "Institut de Recherche sur les Phénomènes Hors Équilibre, AMU, CNRS, Marseille, 13013, France"
  - index: 2
    name: "Department of Aerospace and Mechanical Engineering, University of Southern California, 854 Downey Way, Los Angeles, CA 90089, USA"
  - index: 3
    name: "School of Aeronautics and Astronautics, Sun Yat-sen University, Guangzhou 510275, China"
  - index: 4
    name: "Affiliation to be completed before submission"
date: 25 March 2026
bibliography: paper.bib
---

# Summary

iLBMReX is an open-source fluid--structure-interaction solver that combines the lattice Boltzmann method (LBM), immersed-boundary (IB) coupling, and block-structured adaptive mesh refinement (AMR). It is designed for researchers who need local mesh refinement around moving bodies and unsteady flow fields, but who also want the algorithmic simplicity and low-Mach-number efficiency of LBM, whose predominantly local operations make it inherently well matched to massively parallel GPU architectures [@Bhatnagar1954model; @Kruger2017lattice]. Built on AMReX [@AMReX_JOSS], the software supports two- and three-dimensional simulations, dynamic regridding, checkpointing, and GPU-accelerated execution. The current code base includes 2D and 3D validation cases, immersed-boundary examples for flow past a cylinder, an immersed cavity box, a pitching NACA 0012 airfoil, and flow over a sphere, together with diagnostics for integrated body forces such as lift and drag.

The software targets problems such as immersed-body wakes, moving-boundary flows, and biological or engineering propulsion in which important flow structures are spatially localized and evolve in time. By combining AMR with a marker-based immersed-boundary backend inside an LBM solver, iLBMReX provides a capability that is still uncommon in open research software.

# Statement of need

Adaptive immersed-boundary LBM solvers remain much less common than adaptive finite-volume or finite-difference solvers. In practice, researchers often face a trade-off. General-purpose open-source LBM libraries such as Palabos [@Latt2021Palabos] and OpenLB [@Krause2021OpenLB] offer mature LBM infrastructures, while classical LBM refinement studies have largely focused on composite-grid, grid-refinement, or multi-block strategies rather than fully dynamic AMR [@Lin2000lattice; @Dupuis2003theory; @Peng2006application]. On the other hand, immersed-boundary AMR frameworks such as IBAMR [@IBAMR_software; @Griffith2007adaptive] and IAMReX [@Li2025IAMReX] support adaptive immersed-boundary simulations, but they solve the incompressible Navier--Stokes equations with projection-based finite-volume or finite-difference methods rather than LBM.

This gap matters for researchers studying intermediate-Reynolds-number fluid--structure interaction, dynamic stall, or bio-inspired propulsion. These problems often benefit from LBM because of its simple collision--streaming structure and efficient treatment of complex boundaries, but they also benefit from dynamic mesh adaptation because the important flow features occupy only part of the domain. iLBMReX addresses this need by providing a reusable research code for adaptive IB--LBM simulations on modern CPU and GPU hardware.

# State of the field

The main scholarly contribution of iLBMReX is not to replace broad LBM ecosystems such as Palabos or OpenLB, nor to replicate the full scope of mature AMR immersed-boundary solvers such as IBAMR or IAMReX. Rather, it is designed to fill the gap at the intersection of these tool classes. Palabos and OpenLB are strong choices for users seeking established LBM infrastructures in which dynamic AMR is not the primary design focus [@Latt2021Palabos; @Krause2021OpenLB]. IBAMR and IAMReX, by contrast, are strong choices for users interested in adaptive immersed-boundary simulations based on incompressible Navier--Stokes solvers [@IBAMR_software; @Griffith2007adaptive; @Li2025IAMReX]. iLBMReX offers a different combination: an LBM solver with immersed boundaries, dynamic AMR through AMReX, and portable GPU execution within a single code base [@AMReX_JOSS]. This combination is intended to enable efficient fluid--structure interaction simulations at intermediate Reynolds numbers, while also creating opportunities for optimization and data-driven methods in this regime.

This "build rather than contribute elsewhere" justification is therefore methodological. The design goals of iLBMReX are centered on AMR-aware LBM development, IB--LBM coupling, and experimentation with adaptive kinetic solvers. Those goals differ from extending an existing uniform-grid LBM package or from adapting a projection-based AMR code to reproduce LBM behavior. The package is intended for researchers who specifically need this combination for algorithm development, validation, and simulation studies.

# Software design

At the core of iLBMReX is a single-relaxation-time BGK lattice Boltzmann solver implemented on AMReX's multilevel mesh hierarchy [@Bhatnagar1954model; @AMReX_JOSS]. The software design balances three competing priorities: preserving the simple and local update pattern of LBM, supporting dynamic refinement and optional subcycling across AMR levels, and coupling immersed bodies without discarding AMReX's parallel data structures. The main driver class, `AmrCoreLBM`, inherits from `amrex::AmrCore` and manages level creation, tagging, time stepping, checkpointing, and plotfile generation. Distribution functions and macroscopic variables are stored in `amrex::MultiFab` objects so that collision, streaming, and diagnostic kernels can operate consistently across refinement levels.

The current immersed-boundary implementation uses a marker-based direct-forcing approach enabled with `ibm.method = 1`. This choice favors a relatively transparent and extensible coupling strategy over a more specialized body representation. Lagrangian markers are restricted to the finest grid, where they interpolate velocity from the Eulerian mesh and spread forces back through a smoothed delta kernel. The implementation supports circular and spherical bodies, axis-aligned marker boxes, and example-local user-defined marker geometries such as the pitching NACA 0012 airfoil. The implementation was informed by the immersed-boundary strategy used in IAMReX [@Li2025IAMReX] as well as earlier immersed-boundary LBM formulations [@Wu2009implicit; @Peng2006application]. Restricting markers to the finest level keeps the coupling logic manageable and helps control memory use in adaptive runs.

A related design decision is that AMR tagging for immersed-boundary cases is not based only on flow diagnostics such as vorticity. Instead, the solver also tags a geometry-aware region around the immersed body so that the body remains covered by the finest AMR level during regridding. For moving geometries, this region can be defined conservatively over the full motion envelope; in the pitching-airfoil example, the entire swept NACA 0012 chord is kept on the finest level throughout the prescribed pitching cycle. This keeps the Eulerian--Lagrangian coupling localized, reproducible, and less sensitive to transient changes in wake strength.

A second design choice is the consistent use of AMReX's portable parallelism layer for compute-intensive kernels [@AMReX_JOSS]. Collision, streaming, force spreading, and diagnostics are written so that the same code path can run on CUDA- or HIP-enabled devices and fall back to threaded CPU execution. The solver also exposes diagnostics for macroscopic fields, vorticity, pressure perturbation, integrated body forces, and Guo-style forcing terms [@Guo2002forcing]. This combination makes the code suitable both for production simulations and for methodological studies of adaptive IB--LBM algorithms.

# Research impact statement

iLBMReX is positioned as research infrastructure for adaptive IB--LBM development. Its immediate impact is enabling reproducible studies in a part of the research software landscape where dynamic AMR, immersed boundaries, and LBM are not commonly available together in open source form. The repository already includes canonical examples such as Couette flow, Taylor--Green vortex decay, flow past a cylinder, square-duct flow in 3D, an immersed cavity box, a pitching NACA 0012 airfoil, and flow over a sphere, together with runtime diagnostics for immersed-boundary forces and standard flow quantities. These materials provide a concrete starting point for validation, regression testing, and method comparison.

The near-term significance of the software is also practical. Because the code is built on AMReX [@AMReX_JOSS], researchers can investigate adaptive LBM strategies on heterogeneous hardware without first building their own AMR and GPU infrastructure. Because the immersed-boundary backend is integrated directly into the solver, users can study fluid--structure problems involving moving boundaries, flow control, and biological propulsion. This lowers the barrier for research on intermediate-Reynolds-number fluid--structure interaction and on new adaptive kinetic algorithms.

# AI usage disclosure

Generative AI tools were used for both code debugging of the code base and language editing during preparation of this paper. The authors reviewed and verified all technical descriptions, citations, and software claims in the final manuscript. No statement in this paper should be interpreted as an unverified AI-generated scientific claim.

# Acknowledgements

The authors thank the IAMReX developers, whose framework [@Li2025IAMReX] informed the implementation of the marker-based immersed-boundary module. The authors also thank Yaning Wang for helpful discussions on IB--LBM validation. We acknowledge the AMReX development team [@AMReX_JOSS] at Lawrence Berkeley National Laboratory for creating the software infrastructure that underpins iLBMReX.

No financial support applies for the preparation of this manuscript.
