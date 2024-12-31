
PRECISION  = DOUBLE
PROFILE    = FALSE

DEBUG      = TRUE
# DEBUG      = FALSE

DIM        = 2
#DIM       = 3

COMP	   = gnu

USE_MPI    = TRUE
USE_OMP    = FALSE 
USE_CUDA   = FALSE

Bpack   := ./Make.package 
Blocs   := . 

include Make.LBM

.PHONY: format
format: $(MY_SRC_FILES)
	@echo "Formatting project-specific files..."
	$(foreach file,$(MY_SRC_FILES),$(FORMATTER) $(FORMAT_OPTS) $(file) && echo "Formatted $(file)";)

.PHONY: check-format
check-format: $(MY_SRC_FILES)
	@echo "Checking formatting of project-specific files..."
	@$(foreach file,$(MY_SRC_FILES),$(FORMATTER) --dry-run --Werror $(file) || exit 1;)

