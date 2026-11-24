CXX = g++
MPICXX = mpicxx

# Proposal Bag. 6.5: -O3 -march=native -fopenmp, strictly NO -ffast-math
CXXFLAGS = -O3 -march=native -Wall -Wextra -std=c++17
OMPFLAGS = -fopenmp
INCLUDES = -Isrc

USE_OPENBLAS ?= 0
BLAS_LIBS ?= -lopenblas
ifeq ($(USE_OPENBLAS),1)
	CXXFLAGS += -DUSE_OPENBLAS
	LDLIBS += $(BLAS_LIBS)
endif

BUILD_DIR = build

COMMON_GEMM_SRCS = src/common/gemm.cpp
COMMON_ATTN_SRCS = src/common/gemm.cpp src/common/block_attention.cpp
V2_SRCS = $(COMMON_ATTN_SRCS) src/v2_openmp/v2_openmp.cpp
V4_SRCS = $(COMMON_ATTN_SRCS) src/v4_hybrid/v4_hybrid.cpp

.PHONY: all setup check test-gemm test-v2 test-v4 test test-all bench-gemm bench-v2 bench clean help

all: help

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

setup:
	mkdir -p $(BUILD_DIR)

check:
	@echo "Checking development environment..."
	@$(CXX) --version
	@$(MPICXX) --version

$(BUILD_DIR)/test_gemm: tests/test_gemm.cpp $(COMMON_GEMM_SRCS) src/common/gemm.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) tests/test_gemm.cpp $(COMMON_GEMM_SRCS) -o $@ $(LDLIBS)

$(BUILD_DIR)/test_v2_openmp: tests/test_v2_openmp.cpp $(V2_SRCS) src/common/gemm.hpp src/common/attention_types.hpp src/common/block_attention.hpp src/v2_openmp/v2_openmp.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) tests/test_v2_openmp.cpp $(V2_SRCS) -o $@ $(LDLIBS)

$(BUILD_DIR)/test_v4_hybrid: tests/test_v4_hybrid.cpp $(V4_SRCS) src/common/gemm.hpp src/common/attention_types.hpp src/common/block_attention.hpp src/v4_hybrid/v4_hybrid.hpp | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -DOMPI_SKIP_MPICXX $(OMPFLAGS) $(INCLUDES) tests/test_v4_hybrid.cpp $(V4_SRCS) -o $@ $(LDLIBS)

$(BUILD_DIR)/bench_gemm: benchmarks/gemm/bench_gemm.cpp $(COMMON_GEMM_SRCS) src/common/gemm.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) benchmarks/gemm/bench_gemm.cpp $(COMMON_GEMM_SRCS) -o $@ $(LDLIBS)

$(BUILD_DIR)/bench_v2_scaling: benchmarks/gemm/bench_v2_scaling.cpp $(V2_SRCS) src/common/gemm.hpp src/common/attention_types.hpp src/common/block_attention.hpp src/v2_openmp/v2_openmp.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) benchmarks/gemm/bench_v2_scaling.cpp $(V2_SRCS) -o $@ $(LDLIBS)

test-gemm: $(BUILD_DIR)/test_gemm
	OPENBLAS_NUM_THREADS=1 ./$(BUILD_DIR)/test_gemm

test-v2: $(BUILD_DIR)/test_v2_openmp
	OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores ./$(BUILD_DIR)/test_v2_openmp

test-v4: $(BUILD_DIR)/test_v4_hybrid
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores mpirun --map-by slot:PE=2 --bind-to core -np 2 ./$(BUILD_DIR)/test_v4_hybrid

test: test-gemm test-v2

test-all: test test-v4

bench-gemm: $(BUILD_DIR)/bench_gemm
	OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores ./$(BUILD_DIR)/bench_gemm

bench-v2: $(BUILD_DIR)/bench_v2_scaling
	OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores ./$(BUILD_DIR)/bench_v2_scaling

bench: bench-gemm bench-v2

help:
	@echo "Ring Attention HPC - Available Commands"
	@echo "  make setup        - Create build directory"
	@echo "  make check        - Check compiler availability"
	@echo "  make test         - Build and run unit tests (GEMM + V2 OpenMP)"
	@echo "  make test-v4      - Build and run V4 Hybrid MPI+OpenMP unit tests (2x2 cores)"
	@echo "  make test-all     - Build and run all unit tests (GEMM + V2 + V4)"
	@echo "  make test-gemm    - Build and run GEMM unit tests"
	@echo "  make test-v2      - Build and run V2 OpenMP unit tests"
	@echo "  make bench        - Build and run GEMM + V2 strong-scaling benchmarks"
	@echo "  make bench-gemm   - Build and run GEMM FLOP/s benchmark"
	@echo "  make bench-v2     - Build and run V2 OpenMP strong-scaling benchmark"
	@echo "  make USE_OPENBLAS=1 <target> - Build using cblas_sgemm backend"
	@echo "  make clean        - Remove build directory"

clean:
	rm -rf $(BUILD_DIR)
