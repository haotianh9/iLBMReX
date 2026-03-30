.. _GeometryRepresentation:

Geometry Representation
=======================

iLBMReX supports two mechanisms for describing solid bodies embedded within the fluid domain: a marker‑based immersed‑boundary representation and a level‑set representation.  Both approaches are used solely to define the geometry and to tag cells for adaptive mesh refinement.  The actual enforcement of the no‑slip condition is handled by the immersed‑boundary forcing algorithm (see :ref:`ImmersedBoundary`).

Level‑set field
---------------

For simple stationary bodies such as a circular cylinder the geometry can be represented by a signed‑distance level‑set field :math:`\phi(\boldsymbol{x})`.  The zero contour of :math:`\phi` corresponds to the body surface, positive values indicate the fluid region and negative values indicate the interior of the body.  For example, a cylinder of radius :math:`R` centered at :math:`(x_0, y_0)` is described by

.. math::

   \phi(x,y) = \sqrt{(x - x_0)^2 + (y - y_0)^2} - R.

This field is initialised on each refinement level and does not evolve in time for stationary bodies.  If a body moves the centre coordinates :math:`(x_0(t), y_0(t), z_0(t))` are updated at each time step and the signed‑distance field is re‑computed.

The level‑set field is not used to solve two‑phase flow problems in iLBMReX; it exists purely for geometry definition and to support adaptive mesh refinement near complex boundaries.

Use with adaptive mesh refinement
---------------------------------

When block‑structured AMR is enabled the solver tags cells for refinement based on vorticity magnitude, density gradients and proximity to immersed boundaries.  For level‑set geometries the signed‑distance field is used to identify cells near the solid surface.  Cells with :math:`|\phi| < \epsilon` for a user‑specified threshold :math:`\epsilon` are tagged to remain on the finest level.  This ensures that the boundary is resolved even when the bulk flow can be computed on a coarser grid.

Marker‑based representation
---------------------------

For moving and deformable bodies a marker‑based representation is employed.  A set of Lagrangian markers is placed on the body surface and these markers move with the body.  The markers are used by the immersed‑boundary algorithm to apply forcing to the fluid and to update the rigid‑body or structural dynamics.  Markers are stored only on the finest AMR level to minimise memory usage.  The geometry representation via markers is described in more detail in :ref:`ImmersedBoundary`.
