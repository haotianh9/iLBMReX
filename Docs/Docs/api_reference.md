# API reference (brief)

The current release of iLBMReX is both a usable library and a learning tool for lattice-boltzmann method and immersed boudnary methods in the framework of AMRES. Nevertheless,
advanced users and developers may find the following classes and
functions of interest:

* **`AmrCoreLBM`** – the main driver class. It manages the AMR hierarchy,
  time stepping, collision and streaming operations, regridding and
  output.
* **`LBM::collide()`** – implements the single‑relaxation‑time BGK
  collision operator on each AMR level.
* **`LBM::stream()`** – performs streaming of the distribution functions
  along the discrete velocity directions.
* **`IBM::force()`** – contains routines for interpolating Eulerian
  velocities to Lagrangian markers and spreading forces back to the grid.
* **`Utilities::Tagging`** – implements refinement tagging criteria for
  vorticity, density gradients and immersed bodies.
* **`Diagnostics`** – gathers integrated forces and writes time series
  to file.

Full details of these classes can be found in the source code under
`Source/`. 