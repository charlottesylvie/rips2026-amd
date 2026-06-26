HIPCC ?= /opt/rocm/bin/hipcc
GPU_ARCH ?= gfx1151
PROGRAM ?= sssp_hip
BUILD_DIR ?= build

CPPFLAGS ?=
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wno-sign-compare \
            -Wno-unused-parameter -fno-fast-math
HIPFLAGS ?= --offload-arch=$(GPU_ARCH) -mno-wavefrontsize64
LDFLAGS ?=
LDLIBS ?=

SOURCES := kernel.hip csr_graph.cpp support.cpp
OBJECTS := $(BUILD_DIR)/kernel.o $(BUILD_DIR)/csr_graph.o \
           $(BUILD_DIR)/support.o
DEPS := $(OBJECTS:.o=.d)
HEADERS := common.h csr_graph.h support.h wl.h

.PHONY: all clean check smoke print-config

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
	$(HIPCC) $(HIPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/kernel.o: kernel.hip $(HEADERS) | $(BUILD_DIR)
	$(HIPCC) $(CPPFLAGS) $(CXXFLAGS) $(HIPFLAGS) -MMD -MP \
		-x hip -c $< -o $@

$(BUILD_DIR)/csr_graph.o: csr_graph.cpp common.h csr_graph.h | $(BUILD_DIR)
	$(HIPCC) $(CPPFLAGS) $(CXXFLAGS) $(HIPFLAGS) -MMD -MP \
		-x hip -c $< -o $@

$(BUILD_DIR)/support.o: support.cpp common.h csr_graph.h support.h | $(BUILD_DIR)
	$(HIPCC) $(CPPFLAGS) $(CXXFLAGS) $(HIPFLAGS) -MMD -MP \
		-x hip -c $< -o $@

check:
	python3 -m py_compile tools/*.py
	python3 tools/source_check.py .
	python3 tools/self_test.py

smoke: $(PROGRAM)
	mkdir -p $(BUILD_DIR)/smoke
	python3 tools/make_tiny_gr.py $(BUILD_DIR)/smoke/tiny.gr
	ADDS_RUNS=1 ./$(PROGRAM) -g 0 -s 0 \
		-o $(BUILD_DIR)/smoke/tiny.out \
		$(BUILD_DIR)/smoke/tiny.gr \
		3>$(BUILD_DIR)/smoke/tiny.bench
	python3 tools/check_tiny_output.py $(BUILD_DIR)/smoke/tiny.out
	@echo "Benchmark record:"
	@cat $(BUILD_DIR)/smoke/tiny.bench

print-config:
	@echo "PROGRAM=$(PROGRAM)"
	@echo "HIPCC=$(HIPCC)"
	@echo "GPU_ARCH=$(GPU_ARCH)"
	@echo "CXXFLAGS=$(CXXFLAGS)"
	@echo "HIPFLAGS=$(HIPFLAGS)"

clean:
	rm -rf $(BUILD_DIR) $(PROGRAM)

-include $(DEPS)
