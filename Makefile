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
V0_SRCS = src/v0_naive/v0_naive.cpp
V1_SRCS = src/v1_blockwise/v1_blockwise.cpp
V2_SRCS = $(COMMON_ATTN_SRCS) src/v2_openmp/v2_openmp.cpp
V3_SRCS = src/v3_ring_mpi/v3_ring_mpi.cpp
V4_SRCS = $(COMMON_ATTN_SRCS) src/v4_hybrid/v4_hybrid.cpp
ATTN_SRCS = $(COMMON_ATTN_SRCS) $(V0_SRCS) $(V1_SRCS) src/v2_openmp/v2_openmp.cpp $(V3_SRCS) src/v4_hybrid/v4_hybrid.cpp src/main.cpp

.PHONY: all setup check test-gemm test-v2 test-v4 test-cli test test-all bench-gemm bench-v2 bench clean help

all: $(BUILD_DIR)/attn $(BUILD_DIR)/test_gemm $(BUILD_DIR)/test_v2_openmp $(BUILD_DIR)/test_v4_hybrid $(BUILD_DIR)/bench_gemm $(BUILD_DIR)/bench_v2_scaling

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

setup:
	mkdir -p $(BUILD_DIR)

check:
	@echo "Checking development environment..."
	@$(CXX) --version
	@$(MPICXX) --version

$(BUILD_DIR)/attn: $(ATTN_SRCS) src/common/gemm.hpp src/common/attention_types.hpp src/common/block_attention.hpp src/common/memory_utils.hpp src/v0_naive/v0_naive.hpp src/v1_blockwise/v1_blockwise.hpp src/v2_openmp/v2_openmp.hpp src/v3_ring_mpi/v3_ring_mpi.hpp src/v4_hybrid/v4_hybrid.hpp | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -DOMPI_SKIP_MPICXX $(OMPFLAGS) $(INCLUDES) $(ATTN_SRCS) -o $@ $(LDLIBS)

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

test-cli: $(BUILD_DIR)/attn
	OPENBLAS_NUM_THREADS=1 ./$(BUILD_DIR)/attn --version v0 --s 512 --d 64 --seed 1 --verify
	OPENBLAS_NUM_THREADS=1 ./$(BUILD_DIR)/attn --version v1 --s 512 --d 64 --c 128 --tile 64 --seed 1 --verify
	OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores ./$(BUILD_DIR)/attn --version v2 --s 512 --d 64 --c 128 --tile 64 --threads 4 --seed 1 --verify
	OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores mpirun --bind-to core -np 4 ./$(BUILD_DIR)/attn --version v3 --s 1024 --d 64 --c 256 --tile 64 --seed 1 --verify
	OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores mpirun --bind-to core -np 4 ./$(BUILD_DIR)/attn --version v3 --s 1024 --d 64 --c 256 --tile 64 --seed 1 --blocking --verify
	OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores mpirun --bind-to core -np 4 ./$(BUILD_DIR)/attn --version v3 --s 1024 --d 64 --c 256 --tile 64 --seed 1 --stress-logits --verify
	OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores mpirun --bind-to core -np 4 ./$(BUILD_DIR)/attn --version v3 --s 1024 --d 64 --c 256 --tile 64 --seed 1 --no-comm
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores mpirun --map-by slot:PE=2 --bind-to core -np 2 ./$(BUILD_DIR)/attn --version v4 --s 512 --d 64 --c 256 --tile 64 --threads 2 --seed 1 --verify

test: test-gemm test-v2

test-all: test test-v4 test-cli

bench-gemm: $(BUILD_DIR)/bench_gemm
	OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores ./$(BUILD_DIR)/bench_gemm

bench-v2: $(BUILD_DIR)/bench_v2_scaling
	OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores ./$(BUILD_DIR)/bench_v2_scaling

bench: bench-gemm bench-v2

help:
	@echo "Ring Attention HPC - Available Commands"
	@echo "  make all          - Build attn CLI, unit tests, and benchmarks"
	@echo "  make setup        - Create build directory"
	@echo "  make check        - Check compiler availability"
	@echo "  make test         - Build and run unit tests (GEMM + V2 OpenMP)"
	@echo "  make test-v4      - Build and run V4 Hybrid MPI+OpenMP unit tests (2x2 cores)"
	@echo "  make test-cli     - Verify unified ./build/attn CLI across V0, V1, V2, V3, V4"
	@echo "  make test-all     - Build and run all unit tests + unified CLI verification"
	@echo "  make test-gemm    - Build and run GEMM unit tests"
	@echo "  make test-v2      - Build and run V2 OpenMP unit tests"
	@echo "  make bench        - Build and run GEMM + V2 strong-scaling benchmarks"
	@echo "  make bench-gemm   - Build and run GEMM FLOP/s benchmark"
	@echo "  make bench-v2     - Build and run V2 OpenMP strong-scaling benchmark"
	@echo "  make USE_OPENBLAS=1 <target> - Build using cblas_sgemm backend"
	@echo "  make clean        - Remove build directory"

clean:
	rm -rf $(BUILD_DIR)
