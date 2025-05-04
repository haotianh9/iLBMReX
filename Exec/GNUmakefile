
PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = TRUE


DIM        = 2
#DIM       = 3

COMP	   = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE 
USE_CUDA   = FALSE

Bpack   := ./Make.package 
Blocs   := . 

include Make.LBM
 
format:
	@echo "Formatting source code..."
	@find $(foreach dir, $(Bdirs), $(TOP)/$(dir)) -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
		-exec clang-format -i {} +
	@echo "Formatting complete."