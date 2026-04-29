.. _ImmersedBoundary:

Immersed‑Boundary Coupling
==========================

The immersed‑boundary (IB) method couples Lagrangian markers attached to solid bodies with the Eulerian fluid represented by the lattice Boltzmann equations.  iLBMReX employs a direct‑forcing IB approach in which body forces are computed on the immersed surface and then spread to the fluid lattice to enforce the no‑slip condition.

Overview
--------

The IB algorithm operates on the finest refinement level where the body surface is discretised by a set of markers.  At each time step the following operations are performed:

#. **Interpolation:** The Eulerian velocity field obtained from the previous collision and streaming stages is interpolated to the marker positions using a discrete delta kernel to obtain the current fluid velocity on the surface.
#. **Desired velocity:** The desired velocity of each marker is computed from prescribed rigid-body inputs, marker-box lid inputs, or an example-local user-defined geometry hook.
#. **Lagrangian force:** The difference between the desired and interpolated velocities, scaled by the time step, defines a Lagrangian force density on each marker.
#. **Spreading:** The marker forces are spread back to the Eulerian mesh using the same discrete delta kernel to produce a body‑force field :math:`\boldsymbol{F}`.
#. **Fluid update:** The body force is incorporated into the Guo forcing term of the lattice Boltzmann equation during the next collision stage (see :ref:`FluidEquations:LBM`).  The distribution functions themselves are not directly modified.
#. **Marker update:** Built-in circle/sphere and box geometries rebuild marker positions from their prescribed geometry. User-defined geometries can provide time-dependent marker positions through ``IBMUserDefinedGeometry.H``.

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

Prescribed motion
-----------------

The current marker backend supports prescribed kinematics.  For built-in
circle/sphere geometries, ``ibm.ubx``, ``ibm.uby``, ``ibm.ubz`` and
``ibm.omx``, ``ibm.omy``, ``ibm.omz`` define the target translational and
angular marker velocity used in the force computation.  For marker boxes,
``ibm.box_lid_ux``, ``ibm.box_lid_uy`` and ``ibm.box_lid_uz`` define the
moving-lid target velocity.  More general time-dependent marker positions and
velocities are supplied by adding ``IBMUserDefinedGeometry.H`` in the example
directory, as done by the pitching-airfoil example.

Freely moving rigid-body dynamics are not solved by the current code base.

Implementation considerations
----------------------------

The IB coupling is implemented with AMReX CPU/GPU ``ParallelFor`` constructs.
Markers are stored on the finest AMR level to keep interpolation and spreading
localized.  The current implementation is scoped to one marker body/backend per
simulation.
