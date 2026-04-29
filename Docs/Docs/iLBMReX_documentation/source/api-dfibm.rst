Direct-Forcing Immersed Boundary Method
=========================================

iLBMReX implements a marker-based direct-forcing immersed-boundary (IBM) method.
This section describes the current implementation surface.

Overview
--------

The direct-forcing IBM in iLBMReX uses a marker-based representation of immersed bodies.
For the active marker backend:

1. **Marker points** are placed on the body surface and distributed on the finest
   AMR level to ensure accurate coupling.

2. **Velocity interpolation** interpolates the fluid velocity from Eulerian grid
   points to Lagrangian marker points using a smooth interpolation kernel.

3. **Force computation** calculates the force required to maintain the desired
   marker velocity (prescribed by built-in geometry inputs or by a user-defined
   geometry hook).

4. **Force spreading** distributes the computed force to the Eulerian grid; the
   LBM update then incorporates it through the Guo forcing term.

The approach maintains the simplicity and efficiency of the lattice Boltzmann method
while enabling prescribed immersed geometries.

Supported Geometries
--------------------

The current implementation supports:

* **Circles and spheres** - Marker points distributed over surface
* **Axis-aligned boxes** - 2D rectangular marker boundaries
* **User-defined geometries** - Custom marker distributions and kinematics via
  ``IBMUserDefinedGeometry.H`` in the example directory

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

Detailed API documentation is generated from source code in ``Source/IBM/``.
Key classes and functions include:

.. doxygengroup:: DFIBM
   :project: iLBMReX
   :members:
