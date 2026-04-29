Lattice Boltzmann Method
========================

The iLBMReX solver uses the lattice Boltzmann method (LBM) to evolve the flow.  Instead of directly advancing the Navier–Stokes equations, LBM evolves a set of particle distribution functions :math:`f_i` on a regular lattice and recovers macroscopic quantities through simple moment sums.  This section summarises the variables, discrete velocity sets, equilibrium distribution, governing equation, macroscopic moments and the basic update cycle used in the code.

Fluid variables
---------------

At each cell the solver stores a population of distribution functions for each discrete lattice direction.  The primary variables are:

.. list-table::
   :header-rows: 1

   * - Variable
     - Definition
   * - :math:`f_i`
     - Distribution function in lattice direction :math:`i` (with :math:`i = 0, \dots, N_\mathrm{dir}-1`)
   * - :math:`\rho`
     - Fluid density computed as :math:`\rho = \sum_i f_i`
   * - :math:`\boldsymbol{u}`
     - Fluid velocity determined from the first moment: :math:`\rho\,\boldsymbol{u} = \sum_i \boldsymbol{c}_i f_i`
   * - :math:`\boldsymbol{F}`
     - External/body force per unit volume (e.g., gravity or immersed–boundary forcing)

Discrete velocity sets
----------------------

The code supports both two- and three-dimensional lattices. In 2D the examples
use D2Q9. In 3D, the lattice is selected from the input file; current examples
include D3Q19 (square duct) and D3Q27 (sphere/cavity-style inputs). Each
direction has an associated velocity vector :math:`\boldsymbol{c}_i` and weight
:math:`w_i`. The lattice sound speed is :math:`c_s = 1/\sqrt{3}` for these
lattices.

Equilibrium distribution
-----------------------

In the BGK collision model the distribution functions relax toward a local equilibrium :math:`f_i^{\mathrm{eq}}` that depends on the macroscopic density and velocity.  The equilibrium distribution for isothermal flows is given by

.. math::
   f_i^{\mathrm{eq}} = w_i \, \rho \left[ 1 + \frac{\boldsymbol{c}_i\cdot \boldsymbol{u}}{c_s^2} + \frac{(\boldsymbol{c}_i\cdot \boldsymbol{u})^2}{2 c_s^4} - \frac{\boldsymbol{u}\cdot \boldsymbol{u}}{2 c_s^2} \right],

where :math:`w_i` are the lattice weights and :math:`\rho` and :math:`\boldsymbol{u}` are the local density and velocity, respectively.

Lattice Boltzmann equation
--------------------------

The evolution of the distribution functions is governed by the discrete Boltzmann equation with a single–relaxation–time (BGK) collision operator and a forcing term :math:`F_i`:

.. math::
   f_i(\boldsymbol{x} + \boldsymbol{c}_i \Delta t, \, t + \Delta t) - f_i(\boldsymbol{x}, t) = -\frac{1}{\tau} \bigl[ f_i(\boldsymbol{x}, t) - f_i^{\mathrm{eq}}(\boldsymbol{x}, t) \bigr] + \Delta t\, F_i.

Here :math:`\tau` is the relaxation time.  It controls the viscosity through

.. math::
   \nu = c_s^2 \left( \tau - \frac{1}{2} \right) \Delta t,

where :math:`\nu` is the kinematic viscosity.  The term :math:`F_i` accounts for external forcing and is computed using the Guo forcing scheme:

.. math::
   F_i = \left(1 - \frac{1}{2\tau}\right) w_i \left[ \frac{\boldsymbol{c}_i - \boldsymbol{u}}{c_s^2} + \frac{(\boldsymbol{c}_i\cdot \boldsymbol{u}) \boldsymbol{c}_i}{c_s^4} \right] \cdot \boldsymbol{F},

where :math:`\boldsymbol{F}` is the macroscopic force per unit volume.

Macroscopic quantities
----------------------

After the collision and streaming stages, macroscopic quantities are obtained by summing over the updated distributions:

.. math::
   \rho = \sum_i f_i, \qquad \rho\,\boldsymbol{u} = \sum_i \boldsymbol{c}_i f_i + \frac{1}{2}\boldsymbol{F}.

Derived quantities such as vorticity magnitude and pressure perturbation are computed from the velocity and density fields for diagnostics and adaptive mesh refinement.

Boundary conditions
-------------------

The solver supports periodic boundaries, AMReX ``ext_dir``/``foextrap`` style
physical boundaries, and user wall treatments such as bounce-back
(``user_1``) or face-mirrored prescribed wall states (``user_2``). For
immersed boundaries the body force :math:`\boldsymbol{F}` is provided by the
immersed-boundary algorithm described in :ref:`ImmersedBoundary`.

Time integration
----------------

Each time step on a given AMR level proceeds through the following stages:

1. **FillPatch and forcing:** mesoscopic/macroscopic ghost cells are filled and optional prescribed or immersed-boundary forcing is constructed.
2. **Collision:** local distributions :math:`f_i` are relaxed toward equilibrium using the BGK operator with the Guo forcing increment.
3. **Boundary and ghost handling:** internal, periodic, and supported physical boundary ghost cells are synchronized.
4. **Streaming:** post-collision distributions are pull-streamed along the discrete velocity directions.
5. **Macroscopic update:** density and velocity are recomputed from the streamed distributions, including the half-step forcing correction; derived diagnostics are refreshed for output and regridding.

When adaptive mesh refinement is enabled, the above operations are performed on each refinement level with inter‑level synchronization.
