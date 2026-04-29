Fluid Variables and Equations
==============================

Fluid Variables
---------------

.. table::

   +-----------------------+--------------------------------------------------+
   | Variable              | Definition                                       |
   +=======================+==================================================+
   | :math:`\rho`          | Fluid density                                    |
   +-----------------------+--------------------------------------------------+
   | :math:`U`             | Fluid velocity                                   |
   +-----------------------+--------------------------------------------------+
   | :math:`f_i`           | Lattice distribution functions                   |
   +-----------------------+--------------------------------------------------+
   | :math:`\nu`           | Kinematic viscosity                              |
   +-----------------------+--------------------------------------------------+
   | :math:`\tau`          | Relaxation time (BGK collision operator)         |
   +-----------------------+--------------------------------------------------+
   | :math:`{\bf F}`       | External forces (e.g., immersed boundary)        |
   +-----------------------+--------------------------------------------------+

Governing Equations
-------------------

iLBMReX solves the incompressible Navier-Stokes equations using the lattice
Boltzmann method (LBM). The governing equations are derived from conservation
principles:

**Conservation of Mass:**

.. math::

   \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho U) = 0

For incompressible flow, this reduces to:

.. math::

   \nabla \cdot U = 0

**Conservation of Momentum:**

.. math::

   \frac{\partial (\rho U)}{\partial t} + \nabla \cdot (\rho U U) + \nabla p = \nu \nabla^2 (\rho U) + {\bf F}

where:

* :math:`p` is pressure
* :math:`\nu` is kinematic viscosity
* :math:`{\bf F}` represents external forces (immersed boundary forcing, body forces)

Although these equations are solved implicitly through the lattice Boltzmann
method via the distribution functions :math:`f_i`, they recover these
macroscopic conservation laws in the Chapman-Enskog limit.

.. _FluidEquations:LBM:

Lattice Boltzmann Method
------------------------

The lattice Boltzmann method solves the discrete Boltzmann equation:

.. math::

   f_i(\mathbf{x} + \mathbf{e}_i \Delta t, t + \Delta t) - f_i(\mathbf{x}, t)
   = -\frac{1}{\tau} \left[ f_i(\mathbf{x}, t) - f_i^{eq}(\mathbf{x}, t) \right] + F_i

where:

* :math:`f_i` are the distribution functions at lattice sites
* :math:`\mathbf{e}_i` are the discrete lattice velocities
* :math:`\tau` is the relaxation time (related to viscosity by :math:`\nu = c_s^2(\tau - 0.5)\Delta t`)
* :math:`f_i^{eq}` are the equilibrium distribution functions
* :math:`F_i` is the forcing term (for immersed boundaries and external forces)

**Macroscopic quantities** are computed as moments of the distribution functions:

.. math::

   \rho = \sum_i f_i

   \rho U = \sum_i f_i \mathbf{e}_i + \frac{1}{2}{\bf F}

   p = c_s^2 \rho

where :math:`c_s = 1/\sqrt{3}` is the lattice sound speed.

Time Integration
----------------

iLBMReX uses the BGK collision operator with explicit time advancement:

1. **FillPatch and forcing**: synchronize ghost cells and construct any prescribed or immersed-boundary force
2. **Collision**: update distribution functions based on local equilibrium and the Guo forcing term
3. **Boundary and ghost handling**: synchronize internal, periodic, and supported physical boundary ghost cells
4. **Streaming**: pull-stream distributions from neighboring lattice sites
5. **Macroscopic update**: recover density, velocity, and diagnostics

This process is repeated for each time step and AMR refinement level.

Incompressibility Constraint
-----------------------------

To enforce the incompressibility constraint :math:`\nabla \cdot U = 0`, iLBMReX
uses a lattice Boltzmann formulation with zero velocity divergence built into
the equilibrium distribution functions. This ensures mass conservation is
automatically satisfied during the simulation.

External Forces and Immersed Boundaries
----------------------------------------

External forces (such as those from immersed boundaries) are incorporated through
the forcing term :math:`F_i` in the lattice Boltzmann equation. For the
direct-forcing immersed boundary method, the force is computed at Lagrangian
marker points and spread to the Eulerian grid using a smooth interpolation kernel.

The immersed boundary forcing is applied via:

.. math::

   F_i = \left( 1 - \frac{1}{2\tau} \right) w_i
   \left[
   \frac{\mathbf{e}_i - U}{c_s^2}
   + \frac{(\mathbf{e}_i \cdot U)\mathbf{e}_i}{c_s^4}
   \right] \cdot \mathbf{F}

where :math:`w_i` are the lattice weights and :math:`\mathbf{F}` is the
force density at the grid point.

Physical Parameters
-------------------

Key physical parameters in iLBMReX:

* **Kinematic viscosity**: :math:`\nu` (set in inputs as ``lbmPhysicalParameters.nu``)
* **Reference inlet/body velocity**: set where needed as ``lbmPhysicalParameters.U0`` or boundary-condition values
* **Domain size**: Physical domain dimensions (set in inputs)
* **Grid resolution**: Base grid cells and refinement ratio (set in inputs)

The Reynolds number is defined as:

.. math::

   Re = \frac{U L}{\nu}

where :math:`U` is a characteristic velocity, :math:`L` is a characteristic length,
and :math:`\nu` is the kinematic viscosity.

For more details on iLBMReX-specific parameters and options, see the example input
files in ``Examples/`` and the :ref:`Chap:SetupAndRunning` section.
