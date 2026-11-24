
# System Architecture

## Overview

The project implements four main versions of attention:

- V0: Naive Attention
- V1: Blockwise Attention with Online Softmax
- V2: Blockwise Attention with OpenMP
- V3: Ring Attention with MPI

V4 (Hybrid MPI + OpenMP) is optional.

## Data Flow

1. Generate Q, K, and V tensors.
2. Divide input sequences into blocks.
3. Compute attention using the selected implementation.
4. Verify numerical correctness.
5. Collect performance and memory metrics.
6. Analyze experimental results.

## Parallelization

- V2 uses OpenMP for shared-memory parallelism.
- V3 uses MPI for ring-based communication.
- V4 combines MPI and OpenMP.
