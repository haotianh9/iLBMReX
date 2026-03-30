Direct-Forcing Immersed Boundary Method
=========================================

iLBMReX implements a direct-forcing immersed-boundary (IBM) method for simulating
fluid-structure interaction with moving boundaries. This section describes the
API and implementation of the direct-forcing approach.

Overview
--------

The direct-forcing IBM in iLBMReX uses a marker-based representation of immersed bodies.
For each moving body:

1. **Marker points** are placed on the body surface and distributed on the finest
   AMR level to ensure accurate coupling.

2. **Velocity interpolation** interpolates the fluid velocity from Eulerian grid
   points to Lagrangian marker points using a smooth interpolation kernel.

3. **Force computation** calculates the force required to maintain the desired body
   velocity (typically prescribed kinematics or rigid body dynamics).

4. **Force spreading** distributes the computed force back to the Eulerian grid
   using the Guo forcing term in the lattice Boltzmann collision operator.

The approach maintains the simplicity and efficiency of the lattice Boltzmann method
while enabling accurate simulation of arbitrary moving body geometries.

Supported Geometries
--------------------

The current implementation supports:

* **Circles and spheres** - Marker points distributed over surface
* **Axis-aligned boxes** - Rectangular immersed boundaries
* **User-defined geometries** - Custom marker distributions via input files
* **Moving boundaries** - Time-dependent position and velocity specified via functions

AMR Integration
---------------

Markers are maintained exclusively on the finest AMR level to:

* Simplify interpolation and force spreading operations
* Avoid handling multiple levels of marker representation
* Ensure consistent sub-cycling behavior across AMR levels

Regridding tags the region surrounding immersed bodies to maintain markers on
the finest level throughout the simulation.

API Documentation
------------------

Detailed API documentation is generated from source code in ``Source/IB/``.
Key classes and functions include:

.. doxygengroup:: DFIBM
   :project: iLBMReX
   :members:
