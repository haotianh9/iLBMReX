.. _ImmersedBoundary:

Immersed‑Boundary Coupling
==========================

The immersed‑boundary (IB) method couples Lagrangian markers attached to solid bodies with the Eulerian fluid represented by the lattice Boltzmann equations.  iLBMReX employs a direct‑forcing IB approach in which body forces are computed on the immersed surface and then spread to the fluid lattice to enforce the no‑slip condition.

Overview
--------

The IB algorithm operates on the finest refinement level where the body surface is discretised by a set of markers.  At each time step the following operations are performed:

#. **Interpolation:** The Eulerian velocity field obtained from the previous collision and streaming stages is interpolated to the marker positions using a discrete delta kernel to obtain the current fluid velocity on the surface.
#. **Desired velocity:** The desired velocity of each marker is computed from either a prescribed rigid‑body motion or by integrating the translational and rotational equations of motion for freely moving bodies.
#. **Lagrangian force:** The difference between the desired and interpolated velocities, scaled by the time step, defines a Lagrangian force density on each marker.
#. **Spreading:** The marker forces are spread back to the Eulerian mesh using the same discrete delta kernel to produce a body‑force field :math:`\boldsymbol{F}`.
#. **Fluid update:** The body force is incorporated into the Guo forcing term of the lattice Boltzmann equation during the next collision stage (see :ref:`FluidEquations:LBM`).  The distribution functions themselves are not directly modified.
#. **Marker update:** The marker positions are advanced using the desired velocity.  For freely moving bodies this velocity is obtained by solving Newton’s equations, as described below.

Multi‑direct forcing
--------------------

To improve enforcement of the no‑slip condition the direct‑forcing IB scheme may be iterated several times within a time step.  Let :math:`\boldsymbol{u}^{(0)}` denote the Eulerian velocity after the collision stage and :math:`\boldsymbol{F}^{(0)} = 0`.  For iteration :math:`m = 1, \dots, N_s`:

1. **Interpolate:** Compute the marker velocity :math:`\mathbf{U}^{(m-1)}( \mathbf{X}_l )` by interpolating the current Eulerian velocity :math:`\boldsymbol{u}^{(m-1)}` to the marker positions.
2. **Force update:** Evaluate the Lagrangian force increment

   .. math::

      \mathbf{F}^{(m)}_l = \mathbf{F}^{(m-1)}_l + \frac{ \mathbf{U}^d( \mathbf{X}_l ) - \mathbf{U}^{(m-1)}( \mathbf{X}_l )}{\Delta t},

   where :math:`\mathbf{U}^d` is the desired marker velocity defined by the rigid‑body kinematics.
3. **Spreading:** Spread :math:`\mathbf{F}^{(m)}_l` back to the Eulerian mesh using the discrete delta kernel to obtain the body‑force field :math:`\boldsymbol{F}^{(m)}`.
4. **Eulerian update:** Update the Eulerian velocity (equivalently the momentum moments) via

   .. math::

      \boldsymbol{u}^{(m)} = \boldsymbol{u}^{(0)} + \Delta t\, \boldsymbol{F}^{(m)}.


In iLBMReX the force field :math:`\boldsymbol{F}^{(m)}` is used exclusively through the Guo forcing term in the lattice Boltzmann collision operator.  Empirically two or three sub‑iterations (:math:`N_s \in [2,3]`) are sufficient to obtain accurate no‑slip enforcement.

Rigid‑body motion
-----------------

For a prescribed motion the marker positions and velocities are known functions of time.  For freely moving bodies the translational and rotational equations of motion are solved concurrently with the fluid:

.. math::

   m \frac{d \mathbf{U}_r}{dt} = \int_{S_b} \boldsymbol{F}\, dS + m\, \boldsymbol{g}, \qquad
   I \frac{d \boldsymbol{\Omega}}{dt} = \int_{S_b} \mathbf{r} \times \boldsymbol{F}\, dS,

where :math:`m` and :math:`I` are the mass and moment of inertia of the body, :math:`\boldsymbol{g}` is gravity, :math:`\mathbf{r}` is the vector from the centre of mass to the marker position, and :math:`S_b` is the body surface.  The surface integrals are approximated by summing over the marker forces.  The updated translational velocity :math:`\mathbf{U}_r` and angular velocity :math:`\boldsymbol{\Omega}` are then used to compute the desired marker velocity :math:`\mathbf{U}^d` for the next iteration.

Implementation considerations
----------------------------

The IB coupling is implemented as portable CUDA/HIP/CPU kernels using AMReX’s `ParallelFor` constructs.  Markers are stored only on the finest AMR level to reduce memory consumption, and ghost‑cell synchronization ensures that body forces influence coarser levels correctly.  When multiple bodies are present the coupling loop is executed sequentially for each body, and the accumulated force field accounts for all immersed objects.