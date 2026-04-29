Software
=========

The comprehensive software architecture of iLBMReX is described in the Markdown
documentation provided in the parent ``Docs/`` directory:

* ``design_overview.md`` - Architecture and design decisions
* ``api_reference.md`` - Key classes and functions reference

For implementation details on the following topics, please consult the source
code in the ``Source/`` directory or the Markdown documentation listed above:

**Primary Components:**

#. **AMR Hierarchy Management** - AmrCoreLBM driver built on amrex::AmrCore
#. **Lattice Boltzmann Kernels** - BGK collision, streaming, and moment calculations
#. **Immersed Boundary Coupling** - Lagrangian marker management and force spreading
#. **I/O and Diagnostics** - Plotfile output and integrated force diagnostics
#. **GPU Kernels** - AMReX CPU/GPU implementation; CUDA builds are exposed by the provided GNUmake examples

For more details on AMReX itself, see the official documentation:
https://amrex-codes.github.io/amrex/docs_html/
