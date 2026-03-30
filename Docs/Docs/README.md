# Documentation for iLBMReX

This directory contains supplementary materials for the iLBMReX
lattice Boltzmann solver. It complements the repository
`README.md` by providing detailed build instructions, tutorials and
design notes.

## Contents

* `getting_started.md` – step-by-step instructions for installing
  dependencies, compiling the solver and running a first example.
* `examples.md` – descriptions of the example problems included in the
  repository with notes on expected results.
* `design_overview.md` – a high-level description of the solver
  architecture, including the AMR hierarchy, lattice Boltzmann update
  sequence and immersed-boundary coupling strategy.
* `api_reference.md` – a brief reference to key classes and functions used
  in the solver.
* `../JOSS_paper/` – the manuscript prepared for submission to the Journal of
  Open Source Software, including `paper.md` and the accompanying
  `paper.bib` (located in the repository root).

## Building the documentation

At present, the documentation consists of plain Markdown files. You can
read them directly or import them into a static site generator (e.g.
Sphinx or MkDocs) if you wish to produce HTML output. A simple
`pandoc` script is included in the root of the repository to convert
`paper.md` into a PDF for review.

## Contributing

We welcome documentation improvements! Issues and pull requests that
enhance clarity, add examples or document new features are greatly
appreciated.