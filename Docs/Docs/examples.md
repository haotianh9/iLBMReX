# Example problems

This document describes the example input files supplied with iLBMReX.
Each problem resides in a subdirectory of `Examples/` and can be run by
passing its `inputs` file to the appropriate executable. Feel free to
use these as templates when setting up your own simulations.

## Couette flow (2D)

This canonical test simulates laminar shear flow between two parallel
moving plates. The analytic solution provides profiles for density and
velocity, allowing you to compute the L₂ error and verify second‑order
accuracy. To run:

```sh
./iLBMReX2D ../Examples/Couette/inputs
```

## Taylor–Green vortex (2D)

This decaying vortex problem tests the solver’s ability to resolve
transient structures. It is often used to measure convergence rates in
the time integration and spatial discretization. Plots of kinetic
energy versus time should show the expected decay rate.

## Cylinder wake (2D)

Flow past a circular cylinder at a moderate Reynolds number. The case
uses marker‑based IB coupling to represent the cylinder and tags a
refinement region around the body. Outputs include lift and drag
coefficients and the Strouhal number. Run it with:

```sh
./iLBMReX2D ../Examples/Cylinder/inputs
```

## Square duct (3D)

Steady laminar flow through a square duct demonstrates the 3D solver,
adaptive refinement and GPU acceleration. It can be run as follows:

```sh
./iLBMReX3D ../Examples/Duct/inputs
```

## Immersed cavity box (3D)

This case simulates flow around a rectangular box inside a cavity. It
demonstrates the robustness of the IB coupling in three dimensions. See
the `Examples/ImmersedBox` directory for inputs.

## Pitching NACA 0012 airfoil (2D)

This test exercises support for moving immersed boundaries. The
NACA 0012 airfoil undergoes sinusoidal pitching. To ensure the
Lagrangian markers remain on the finest AMR level throughout the cycle,
the solver tags the entire swept chord for refinement. See
`Examples/PitchingAirfoil` for details.

## Flow over a sphere (3D)

Flow past a sphere at a moderate Reynolds number resembles the cylinder
case but in three dimensions. Diagnostics include lift and drag
coefficients. The inputs reside in the `Examples/Sphere` directory.