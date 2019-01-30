# Unix commands.
PYTHON := python
NVCC_COMPILE := nvcc -c -o
RM_RF := rm -rf

# Library compilation rules.
NVCC_FLAGS := -x cu -Xcompiler -fPIC -shared

# File structure.
BUILD_DIR := build
INCLUDE_DIRS := include
TORCH_FFI_BUILD := build_ffi.py
SEARCHSORTED_CUDA_KERNEL := $(BUILD_DIR)/searchsorted_cuda_kernel.so

INCLUDE_FLAGS := $(foreach d, $(INCLUDE_DIRS), -I$d)

NVCC_RESULT := $(shell which nvcc 2> NULL)
NVCC_TEST := $(notdir $(NVCC_RESULT))

ifeq ($(NVCC_TEST),nvcc)
        TARGETS := cpu gpu
else
        TARGETS := cpu
endif

TORCH_FFI_TARGET_GPU := searchsorted/searchsorted_wrapper/_searchsorted_cuda_wrapper.so
TORCH_FFI_TARGET_CPU := searchsorted/searchsorted_wrapper/_searchsorted_cpu_wrapper.so

all: $(TARGETS)
	rm NULL
cpu: $(TORCH_FFI_TARGET_CPU)
gpu: $(TORCH_FFI_TARGET_GPU)

$(TORCH_FFI_TARGET_GPU): $(SEARCHSORTED_CUDA_KERNEL) $(TORCH_FFI_BUILD)
	$(PYTHON) $(TORCH_FFI_BUILD)
	$(TORCH_FFI_TARGET)

$(TORCH_FFI_TARGET_CPU): $(TORCH_FFI_BUILD)
	$(PYTHON) $(TORCH_FFI_BUILD)

$(BUILD_DIR)/%.so: src_searchsorted/src_cuda/%.cu
	@ mkdir -p $(BUILD_DIR)
	# Separate cpp shared library that will be loaded to the extern C ffi
	$(NVCC_COMPILE) $@ $? $(NVCC_FLAGS) $(INCLUDE_FLAGS)

clean:
	$(RM_RF) $(BUILD_DIR) $(SEARCHSORTED_CUDA_KERNEL)
