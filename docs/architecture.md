# System Architecture

## Overview

The project implements four main versions of attention plus one optional hybrid extension:

- **V0**: Naive Attention (FP64 ground truth and FP32 baseline)
- **V1**: Blockwise Attention with Online Softmax (serial)
- **V2**: Blockwise Attention with OpenMP (shared-memory)
- **V3**: Ring Attention with MPI (distributed ring, overlap communication/computation)
- **V4 (Optional)**: Hybrid MPI + OpenMP

## Shared Kernel & Online Softmax Layer (`src/common/`)

All versions share a unified compute kernel and blockwise Online Softmax primitives (detailed in [docs/common-api.md](common-api.md)) so that computational FLOP/s rate $F$ is consistent across experiments:

- **`src/common/gemm.hpp` / `gemm.cpp`**: Cache-blocked row-major FP32 GEMM (`gemm_nt_fp32`, `gemm_nn_fp32`) using a $48 \times 64 \times 64$ L1d-friendly blocking scheme and an AVX2/FMA $6 \times 16$ micro-kernel, with an optional `-DUSE_OPENBLAS` backend (`cblas_sgemm` with `OPENBLAS_NUM_THREADS=1`).
- **`src/common/attention_types.hpp`**: Shared `AttentionConfig`, `ErrorMetrics`, and validation/FP64 error helpers.
- **`src/common/block_attention.hpp` / `block_attention.cpp`**: Tiled ($b_q \times b_k$) Online Softmax primitives (`init_block_state`, `attend_block_online_softmax`, `finalize_block_output`) maintaining per-row running $(m, \ell, O)$ with $O(b_q b_k)$ scratchpad memory.

## Data Flow

1. Generate $Q, K, V \in \mathbb{R}^{s \times d}$ tensors (or load tokenized DNA sequence blocks).
2. Divide input sequences into blocks of size $c$ and internal tiles $b_q \times b_k$.
3. Compute attention using the selected implementation (V0–V4).
4. Verify numerical correctness against FP64 reference (target relative L2 error $< 10^{-5}$).
5. Collect performance, strong/weak scaling, and memory (`VmHWM` / `getrusage`) metrics into `results/raw/`.
6. Analyze experimental results and bottleneck behavior.

## Parallelization

- **V2 (`src/v2_openmp/`)**: Uses `#pragma omp parallel` across the outer query block loop (`qi_blk`). Each thread writes strictly to disjoint slices of `O` and keeps its own private `(m_state, l_state, tile_buf)` buffers, eliminating race conditions and synchronization overhead inside the Online Softmax reduction.
- **V3 (`src/v3_ring_mpi/`)**: Uses MPI for ring-based communication of contiguous double-buffered `[K|V]` blocks (`2 * c * d` floats) with non-blocking `MPI_Isend` / `MPI_Irecv` overlapped with `attend_block_online_softmax`.
- **V4 (`src/v4_hybrid/`)**: Two-level parallelism:
  - *Inter-rank (MPI Ring)*: Each rank holds local $Q_r \in \mathbb{R}^{c \times d}$ and rotates `[K|V]` buffers (`kv_cur`, `kv_next`) around the ring using `MPI_Isend` / `MPI_Irecv` + `MPI_Waitall`.
  - *Intra-rank (OpenMP)*: Inside each ring step, OpenMP threads parallelize across query sub-blocks of $Q_r$ using thread-private tile scratchpads and disjoint `(O_local, m_state, l_state)` row slices. Initialized with `MPI_THREAD_FUNNELED` so all MPI calls occur strictly on the master thread outside parallel regions.

## Reproducibility & Environment Rules

- Compile with `-O3 -march=native -Wall -Wextra -std=c++17` (plus `-fopenmp`); **never use `-ffast-math`** (preserves `exp()` accuracy for Online Softmax).
- Pin threads/ranks to physical cores: `OPENBLAS_NUM_THREADS=1`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, and `mpirun --bind-to core`.
