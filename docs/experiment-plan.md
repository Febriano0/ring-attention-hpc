
# Experiment Plan

## Parameters

| Parameter | Planned Values |
|---|---|
| Head dimension (d) | 64, optional 128 |
| Block size (c) | 128, 256, 512, 1024 |
| MPI processes / threads | 1, 2, 4, optional 8 |
| Precision | FP32 |
| Reference precision | FP64 |
| Repetitions | At least 5 |

## Metrics

1. Numerical correctness
2. Peak memory per process
3. Execution time
4. Speedup
5. Parallel efficiency
6. Strong scaling
7. Weak scaling
8. Communication-computation overlap

## Benchmark

- GEMM benchmark for computational performance (FLOP/s).
- MPI ping-pong benchmark for communication bandwidth.
- Blocking versus non-blocking communication comparison.

## Results

All raw experimental results will be stored in CSV format under `results/raw/`.

Plots will be stored under `results/figures/`.
