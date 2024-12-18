# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= /home/haotian/Work_LBM/amrex

DEBUG        = True
USE_MPI      = FALSE
USE_OMP      = FALSE
COMP         = gnu
DIM          = 2

# Include necessary AMReX and project-specific configurations
include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include ../Src/Make.package
VPATH_LOCATIONS  += ../Src
INCLUDE_LOCATIONS += ../Src
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Tools/GNUMake/Make.rules

# Define formatting rules for your project files only
FORMATTER = clang-format
FORMAT_OPTS = -i
SRC_EXTENSIONS = cpp h
MY_SRC_FILES = $(shell find ../Src -type f \( $(foreach ext,$(SRC_EXTENSIONS),-name '*.$(ext)') \))

.PHONY: format
format: $(MY_SRC_FILES)
	@echo "Formatting project-specific files..."
	$(foreach file,$(MY_SRC_FILES),$(FORMATTER) $(FORMAT_OPTS) $(file) && echo "Formatted $(file)";)

.PHONY: check-format
check-format: $(MY_SRC_FILES)
	@echo "Checking formatting of project-specific files..."
	@$(foreach file,$(MY_SRC_FILES),$(FORMATTER) --dry-run --Werror $(file) || exit 1;)

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  format       - Format project-specific source files in ../Src."
	@echo "  check-format - Check formatting of project-specific source files."
	@echo "  clean        - Clean up temporary files (if needed)."
	@echo "  all          - Default build target (from AMReX rules)."
