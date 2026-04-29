# Example Problems

This document describes the example problems supplied with iLBMReX.
Each problem resides in a subdirectory of `Examples/` and can be run by
building the executable and passing the `inputs` file. Feel free to
use these as templates when setting up your own simulations.

## Couette Flow (2D)

This canonical test simulates laminar shear flow between two parallel
moving plates. The analytic solution provides profiles for density and
velocity, allowing you to compute the L₂ error and verify second-order
accuracy. To run:

```sh
cd Examples/Couette_flow
make -j
./main2d.gnu.ex inputs
```

## Taylor–Green Vortex (2D)

This decaying vortex problem tests the solver's ability to resolve
transient structures. It is often used to measure convergence rates in
time integration and spatial discretization. Plots of kinetic energy
versus time should show the expected decay rate.

```sh
cd Examples/Taylor_Green_Vortex
make -j
./main2d.gnu.ex inputs
```

## Cylinder Wake (2D)

Flow past a circular cylinder at a moderate Reynolds number. The case
uses marker-based IB coupling to represent the cylinder and tags a
refinement region around the body. Outputs include lift and drag
coefficients and the Strouhal number. Run it with:

```sh
cd Examples/Cylinder_flow
make -j
./main2d.gnu.ex inputs
```

## Square Duct (3D)

Steady laminar flow through a square duct demonstrates the 3D solver and
body-force-driven internal flow setup. It can be run as follows:

```sh
cd Examples/SquareDuct_flow
make -j
./main3d.gnu.ex inputs
```

## Cavity Flow with Obstacle (2D)

This directory includes two 2D cavity variants:

- `inputs_bc_ref`: moving lid imposed with physical boundary conditions.
- `inputs_ibm_box`: moving lid imposed with marker IBM box geometry.

```sh
cd Examples/Cavity_flow
make -j
./main2d.gnu.ex inputs_bc_ref
./main2d.gnu.ex inputs_ibm_box
```

## Pitching NACA 0012 Airfoil (2D)

This test exercises support for moving immersed boundaries. The
NACA 0012 airfoil undergoes sinusoidal pitching. To ensure the
Lagrangian markers remain on the finest AMR level throughout the cycle,
the solver tags the entire swept chord for refinement.

```sh
cd Examples/Pitching_airfoil
make -j
./main2d.gnu.ex inputs
```

## Flow Over a Sphere (3D)

Flow past a sphere at a moderate Reynolds number resembles the cylinder
case but in three dimensions. Diagnostics include lift and drag
coefficients. This case demonstrates the solver's capability for
3D wake modeling with adaptive mesh refinement.

```sh
cd Examples/Sphere_flow
make -j
./main3d.gnu.ex inputs
```

## GPU Execution

If you have built a GPU variant, the executable name includes the backend
(for example, `main2d.gnu.CUDA.ex`).
Run them the same way:

```sh
./main2d.gnu.CUDA.ex inputs
```

## Additional Examples

The `Examples/` directory also includes:

- **ForceValidation** - Uniform-body-force validation in a periodic box (checks forcing implementation)
- **Parameter_choice** - Studies on optimal parameter selection for LBM stability

## Visualizing Results

Examples generate AMReX plotfiles (e.g., `plt_00000`, `plt_00100`, etc.).
Checkpoint files are produced when `amr.chk_int > 0`. Visualize outputs using:

- **ParaView**: Load the plotfiles directly
- **VisIt**: Supports AMReX format natively
- **yt**: Python package for scientific visualization
- **Amrvis**: Specialized AMR visualization tool

Many examples include Python scripts (e.g., `plot_*.py`) for custom visualization
and data extraction from the plotfiles.
