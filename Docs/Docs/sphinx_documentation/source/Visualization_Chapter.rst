Visualization
=============

There are a large number of tools that can be used to read iLBMReX or AMReX data
and make plots. Here we give a brief overview of some visualization approaches.

For more information on visualization tools and AMReX data formats, see the
AMReX documentation: https://amrex-codes.github.io/amrex/docs_html/

Supported Tools
~~~~~~~~~~~~~~~

**ParaView**
    A general-purpose open-source visualization software that supports AMReX
    plotfile format natively. ParaView can be downloaded from:
    https://www.paraview.org/

**VisIt**
    Another general-purpose visualization tool with native support for AMReX
    data formats. More information available at:
    https://wci.llnl.gov/simulation/computer-codes/visit

**yt**
    A Python-based analysis and visualization toolkit that works well with
    AMReX plotfiles. Installation and documentation available at:
    https://yt-project.org/

**Amrvis**
    A custom visualization tool developed specifically for hierarchical AMR data.
    More information available at:
    https://ccse.lbl.gov/Software/Amrvis/

Output Data
~~~~~~~~~~~

iLBMReX writes plotfiles at user-specified intervals containing:

* Macroscopic fields: density, velocity components
* Derived fields: vorticity magnitude, pressure perturbation
* Force fields: immersed boundary force contributions (for IB cases)
* Integrated diagnostics: lift and drag coefficients (in separate text files)

Each plotfile is a directory containing AMReX-format data that can be read by
any of the visualization tools listed above.
