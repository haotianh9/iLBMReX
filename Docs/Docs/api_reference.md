# API reference (brief)

This project is primarily an executable solver, but several core classes and
functions are useful entry points for developers:

* **`AmrCoreLBM`** (`Source/Core/AmrCoreLBM.H`) - main AMR/LBM driver.
  Owns hierarchy state, time stepping, regridding, plotfile/checkpoint I/O,
  and IBM orchestration.
* **`AmrCoreLBM::AdvancePhiAtLevel`** (`Source/Core/AdvanceLBMAtLevel.cpp`) -
  per-level update routine that performs forcing, collision, streaming, and
  macroscopic recovery.
* **`collide`** (`Source/LBM/collide.H`) and **`collide_forced`**
  (`Source/LBM/collide_forced.H`) - BGK collision kernels (with and without
  forcing terms).
* **`stream`** (`Source/LBM/stream.H`) - pull-streaming kernel for the
  distribution functions.
* **`calculateMacro`** (`Source/LBM/calculateMacro.H`) and
  **`calculateMacroForcing`** (`Source/LBM/calculateMacroForcing.H`) -
  macroscopic variable reconstruction kernels.
* **`IBMarkerDF`** (`Source/IBM/IBMarkerDF.H/.cpp`) - marker-based
  direct-forcing immersed-boundary backend (interpolation, forcing solve,
  spreading).
* **`MarkerIBParams`** (`Source/IBM/MarkerIBParams.H`) - IBM parameter bundle
  parsed from `ibm.*` inputs keys.
* **`vorticity_tagging`** and **`levelset_tagging`** (`Source/Tagging.H`) -
  AMR tagging helpers used by `AmrCoreLBM::ErrorEst`.

For runtime parameter details, see `AmrCoreLBM::ReadParameters()` in
`Source/Core/AmrCoreLBM.cpp`.
