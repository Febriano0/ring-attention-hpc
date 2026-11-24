# Experiment Plan: Serial vs HPC Attention

## Objective and Scope

The primary objective is to measure the performance impact of parallel and distributed attention on the same host and workload. The project is not intended to evaluate biological validity or clinical performance.

Use the following comparisons for different questions:

- **V1 vs V3**: serial blockwise attention versus distributed Ring Attention. This is the preferred comparison for assessing the effect of MPI parallelism while keeping the blockwise method and numerical input consistent.
- **V0 vs V3**: naive baseline versus optimized HPC implementation. This is an end-to-end before/after comparison; its speedup combines algorithmic and parallelization changes.
- **V2 vs V3**: shared-memory OpenMP versus distributed MPI scaling.

The FASTQ/DNA application is an optional workload demonstration. The attention kernel benchmarks use generated numerical inputs and do not require downloaded biological data or model training.

## Demo Version Coverage

The CLI application demo (`scripts/demo_cli.sh`) can run any combination of V0–V4 and runs all five by default. V0/V1 are serial, V2 uses OpenMP, V3 uses MPI, and V4 combines MPI with OpenMP. V2 uses one process with the configured thread count; V3 uses the configured MPI rank count; V4 uses both rank and thread counts.

The Streamlit demo intentionally remains a two-column V0-vs-V3 demonstration. V0-vs-V3 shows the combined effect of changing the algorithm and using Ring MPI. Use V1-vs-V3 in the CLI for the more focused serial-blockwise versus distributed comparison.

## Parameters

| Parameter | Planned Values |
|---|---|
| Sequence length (`s`) | Use matching values for the compared versions; increase for memory/scaling tests |
| Head dimension (`d`) | 64; optional 128 |
| Block size (`c`) | 128, 256, 512, 1024, subject to `s = P * c` for V3/V4 |
| MPI processes (`P`) | 1, 2, 4; optional 8 when resources permit |
| OpenMP threads | 1, 2, 4; optional 8 when resources permit |
| Precision | FP32 |
| Numerical reference | FP64 |
| Repetitions | At least 5 timed repetitions, plus warm-up |

Use the same `s`, `d`, tile size, seed, host, and compiler settings when comparing V1 and V3. V3 requires `s` to be divisible by the number of MPI processes and uses `c = s / P` in the CLI experiment.

## Measurement Procedure

1. Build once using the documented compiler flags and run `make test-all`.
2. Pin the workload to a consistent set of CPU cores where supported. Keep `OPENBLAS_NUM_THREADS=1` to avoid nested BLAS threading.
3. Use the same input seed and parameters for the serial and MPI runs. Run a warm-up and at least five timed repetitions.
4. Record the runtime, process count, thread count, block size, sequence length, compiler, host, and operating system alongside results.
5. Repeat runs on the same machine under comparable load. Do not mix WSL2 and native Linux measurements in one result series without labeling the environments.

For the focused serial-versus-MPI comparison, the README gives directly runnable V1/V3 commands. `scripts/run_experiments.py --reps 5` runs the broader experiment collection, including memory, scaling, overlap, and V0-vs-V3 application measurements.

## Metrics

1. **Execution time**: report the repeated-run timing (preferably median and spread).
2. **Speedup**: `T_serial / T_parallel` for matching workloads.
3. **Parallel efficiency**: `speedup / P`; state the baseline and process count used.
4. **Peak memory**: report per-process measurements and distinguish peak RSS from allocated-buffer estimates.
5. **Strong scaling**: hold total `s` constant while increasing `P`.
6. **Weak scaling**: increase total `s` with `P` while holding per-process work approximately constant.
7. **Communication/computation overlap**: compare blocking, non-blocking, and no-communication modes; use the same problem size.
8. **Numerical error**: compare output with the FP64 reference to ensure performance changes preserve the expected result.
9. **Microbenchmarks**: GEMM throughput (FLOP/s) and MPI ping-pong bandwidth help interpret compute and communication limits; they are supporting measurements, not end-to-end inference latency.

## Benchmark Commands and Outputs

```bash
make test-all
python3 scripts/run_experiments.py --reps 5
python3 scripts/plot_results.py
```

Raw CSV files are written under `results/raw/`; plots are written under `results/figures/`. Re-run on the target machine before treating checked-in result files as representative of that machine.

## Environment Limitations

Linux and Linux HPC systems are the primary benchmark environments. Windows users can run the Linux toolchain under WSL2, but should label those results as WSL2 and keep the repository inside the WSL filesystem. The current Makefile, memory accounting, and shell scripts are not a native Windows build path. WSL2 runs on one Windows host and is not a substitute for multi-node MPI cluster testing.
