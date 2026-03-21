PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = FALSE

DIM        = 3

COMP       = gcc

USE_MPI    = FALSE
USE_OMP    = FALSE
USE_CUDA   = FALSE

Bpack   := ./Make.package
Blocs   := .

include Make.LBM

format:
	@echo "Formatting source code..."
	@find $(foreach dir, $(Bdirs), $(TOP)/$(dir)) -type f \( -name "*.cpp" -o -name "*.H" -o -name "*.h" -o -name "*.hpp" \) \
		-exec clang-format -i {} +
	@echo "Formatting complete."
