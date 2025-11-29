PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = TRUE

DIM        = 2
#DIM       = 3

COMP	   = gcc

USE_MPI    = TRUE
USE_OMP    = TRUE 
USE_CUDA   = FALSE

USE_PARTICLES = TRUE

Bpack   := ./Make.package 
Blocs   := . 

include Make.LBM
 
format:
	@echo "Formatting source code..."
	@find $(foreach dir, $(Bdirs), $(TOP)/$(dir)) -type f \( -name "*.cpp"  -o -name "*.H" -o -name "*.h" -o -name "*.hpp" \) \
		-exec clang-format -i {} +
	@echo "Formatting complete."
