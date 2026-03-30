.. role:: cpp(code)
   :language: c++


.. _Chap:SetupAndRunning:

Creating and Running Your Own Problem
=====================================

This section covers how to set up your own problem with iLBMReX and
provides information on controlling the simulation parameters and algorithm options.

Problem Setup Overview
----------------------

In iLBMReX, a combination of coded routines and input parameters provide the
problem setup. Input parameters control the algorithm behavior and simulation
execution. Typically, input parameters are collected into an ``inputs`` file,
which is specified on the command line. Input parameters can also be specified
directly on the command line, overriding values in the inputs file.

iLBMReX uses AMReX's ``ParmParse`` class infrastructure to read input parameters.
For detailed information on ``ParmParse``, see the AMReX documentation:
:ref:`amrex:sec:basics:parmparse`.

Example Input File
~~~~~~~~~~~~~~~~~~

Input files specify:

* Domain size and geometry
* Refinement criteria and AMR parameters
* Time stepping and output intervals
* Immersed body geometries and motion
* Physical parameters (viscosity, density)
* Boundary conditions

See the example problems in ``Examples/`` directory for complete input files.
Each example directory includes an ``inputs`` file that can be used as a template
for creating your own simulations.

Running iLBMReX
---------------

To run an iLBMReX simulation:

.. code-block:: shell

   ./main2d.gnu.ex inputs

where ``main2d.gnu.ex`` is the executable and ``inputs`` is the input file.

If you have built the GPU version:

.. code-block:: shell

   ./main2d.gnu.CUDA.ex inputs

Output and Diagnostics
~~~~~~~~~~~~~~~~~~~~~~

iLBMReX generates the following output:

* **Plotfiles**: AMReX plotfile format at user-specified intervals (``amr.plot_int``)
* **Checkpoint files**: AMReX checkpoint format for restart capability (``amr.chk_int``)
* **Diagnostic files**: For immersed boundary cases, integrated lift/drag forces

Plotfiles can be visualized with:

* ParaView
* VisIt
* Python ``yt`` package
* Amrvis

Customizing Your Problem
-------------------------

To create a custom problem:

1. **Create a new directoryunder ``Examples/``**

2. **Copy an input file** from a similar example (e.g., Couette, Cylinder)

3. **Modify the input parameters** for your problem:

   - Domain dimensions and refinement
   - Immersed body geometry (if needed)
   - Initial and boundary conditions
   - Physical parameters (viscosity, domain size)

4. **Build the executable** in your example directory:

   .. code-block:: shell

      make -j

5. **Run the simulation**:

   .. code-block:: shell

      ./main2d.gnu.ex inputs

Key Input Parameters
~~~~~~~~~~~~~~~~~~~~

**AMR Parameters:**

* ``amr.max_level`` - Maximum refinement level
* ``amr.ref_ratio`` - Refinement ratio between levels
* ``amr.regrid_int`` - Regridding interval
* ``amr.n_error_buf`` - Buffer cells around tagged regions

**Time Stepping:**

* ``n_cell`` - Base grid resolution in each direction
* ``max_step`` - Maximum number of time steps
* ``stop_time`` - Simulation end time
* ``dt`` - Initial time step (or CFL-based if adaptive)

**Immersed Boundary:**

* ``ib.bodies`` - Number of immersed bodies
* ``ib.body.N.geometry`` - Geometry type (sphere, circle, box, or custom)
* ``ib.body.N.center`` - Body center position
* ``ib.body.N.velocity`` - Body velocity (prescribed or rigid body dynamics)

**Physical Parameters:**

* ``lbm.nu`` - Kinematic viscosity
* ``lbm.density`` - Fluid density

For complete documentation on all input parameters, see the example input files
in the ``Examples/`` directory and refer to the source code in ``Source/Init/`` .

Git Workflow for Development
----------------------------

Before making any code changes, we recommend creating a new git branch:

.. code-block:: shell

   git checkout main
   git checkout -b <branch_name>

where ``<branch_name>`` reflects your changes (e.g., ``couette_flow``, ``new_ib_geometry``).

To pull in updates from the remote repository:

.. code-block:: shell

   git fetch origin
   git merge origin/main

This keeps your local branch in sync with the latest development.

For more information, see the :ref:`Chap:Contributing` section.
