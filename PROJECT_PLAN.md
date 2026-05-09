# GPU Memory Bandwidth Benchmarker

## Goal

Build a CUDA benchmark that measures effective GPU memory bandwidth across several access patterns, then compare the results against the theoretical peak bandwidth of the GPU.

This project is meant to prove that you understand:

- CUDA kernel launch configuration
- Global memory reads and writes
- Coalesced vs non-coalesced memory access
- Shared memory behavior
- Bank conflicts
- Timing with CUDA events
- Basic profiling with Nsight Compute
- Interpreting bandwidth numbers against hardware specs

## Final Deliverable

At the end, the project should contain:

- CUDA kernels for different memory access patterns
- A command-line benchmark executable
- Clean bandwidth calculations in GB/s
- A results table printed by the program
- Nsight Compute validation notes
- Ahrs, and comparison to theoretical peak

Suggested final repository structure:

```text
CUDA_Project/
├── CMakeLists.txt
├── README.md
├── PROJECT_PLAN.md
├── src/
│   ├── main.cu
│   ├── benchmark.cu
│   ├── benchmark.cuh
│   ├── kernels.cu
│   └── kernels.cuh
├── include/
│   └── cuda_check.cuh
├── results/
│   ├── bandwidth_results.md
│   └── nsight_notes.md
└── scripts/
    └── run_benchmarks.sh
```

## Measurement Formula

Effective bandwidth:

```text
bandwidth = bytes_transferred / elapsed_time
```

For GB/s:

```text
GB/s = bytes_transferred / elapsed_seconds / 1e9
```

For a simple copy-style benchmark:

```cpp
out[i] = in[i];
```

Each element performs:

- One global memory read
- One global memory write

So total bytes transferred:

```text
bytes = num_elements * sizeof(float) * 2
```

Use CUDA events for timing:

```cpp
cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
```

## Phase 1: Baseline Global Memory Benchmark

Estimated time: 2 days

### Objective

Write a simple CUDA benchmark that measures sequential global memory read/write bandwidth.

### Tasks

- Create a minimal CUDA project.
- Allocate large input and output arrays on the GPU.
- Initialize input data.
- Write a baseline copy kernel:

```cpp
out[i] = in[i];
```

- Use CUDA events to measure elapsed kernel time.
- Run multiple iterations and report the average time.
- Compute effective bandwidth manually.
- Print:
  - GPU name
  - Number of elements
  - Total bytes transferred
  - Average elapsed time
  - Effective bandwidth in GB/s

### Important Details

- Use a large enough buffer to exceed cache size, such as 256 MB, 512 MB, or 1 GB.
- Warm up the GPU before timing.
- Avoid including allocation or host-device copy time in the measured kernel time.
- Check CUDA errors after every CUDA API call and kernel launch.

### Deliverable

A working benchmark that prints something like:

```text
Pattern              Size       Time (ms)     Bandwidth (GB/s)
Sequential copy      512 MB     0.78          688.3
```

### Validation

Find the theoretical memory bandwidth of your GPU from NVIDIA's official spec page.

Record:

- GPU model
- Memory type
- Memory bus width
- Memory data rate
- Theoretical bandwidth
- Measured bandwidth
- Percentage of theoretical peak

Formula:

```text
percent_of_peak = measured_bandwidth / theoretical_peak * 100
```

## Phase 2: Coalescing Study

Estimated time: 2 days

### Objective

Measure how memory access patterns affect global memory bandwidth.

### Kernels to Implement

1. Coalesced access

```cpp
out[i] = in[i];
```

2. Strided access

```cpp
out[i] = in[i * stride];
```

3. Offset access

```cpp
out[i] = in[i + offset];
```

4. Reverse access

```cpp
out[i] = in[n - 1 - i];
```

### Tasks

- Add a benchmark mode for each access pattern.
- Test several strides:

```text
1, 2, 4, 8, 16, 32
```

- Keep the benchmark size consistent across patterns.
- Print all results in one table.
- Explain which patterns are coalesced and which are not.

### Nsight Compute Validation

Run Nsight Compute on selected kernels:

```bash
ncu ./bandwidth_benchmark
```

Useful metrics:

```text
l1tex__t_bytes.sum
dram__bytes.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
sm__throughput.avg.pct_of_peak_sustained_elapsed
```

### Deliverable

A results table like:

```text
Pattern             Stride     Bandwidth (GB/s)     % of Peak
Coalesced           1          720.4                82.1
Strided             2          390.8                44.5
Strided             4          210.2                23.9
Strided             8          112.7                12.8
Reverse             1          705.5                80.4
```

## Phase 3: Shared Memory Study

Estimated time: 3 days

### Objective

Use shared memory as a scratchpad and measure how tile size and bank conflicts affect performance.

### Kernels to Implement

1. Shared memory tiled copy without intentional bank conflicts

```cpp
shared[threadIdx.x] = in[i];
__syncthreads();
out[i] = shared[threadIdx.x];
```

2. Shared memory access with configurable stride

```cpp
shared[(threadIdx.x * stride) % blockDim.x] = in[i];
__syncthreads();
out[i] = shared[(threadIdx.x * stride) % blockDim.x];
```

3. Shared memory tiled copy with padding

```cpp
__shared__ float tile[TILE_SIZE + 1];
```

### Tasks

- Test different block sizes:

```text
128, 256, 512, 1024
```

- Test different shared-memory strides:

```text
1, 2, 4, 8, 16, 32
```

- Compare with the Phase 1 global memory baseline.
- Record whether shared memory improves, hurts, or does not change performance.
- Use Nsight Compute to inspect shared memory behavior.

### Nsight Compute Metrics

Useful metrics:

```text
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
sm__warps_active.avg.pct_of_peak_sustained_active
smsp__sass_average_branch_targets_threads_uniform.pct
```

### Occupancy Notes

Record how block size affects occupancy.

Useful command:

```bash
ncu --section Occupancy ./bandwidth_benchmark
```

Record:

- Block size
- Registers per thread
- Shared memory per block
- Theoretical occupancy
- Achieved occupancy
- Bandwidth

### Deliverable

A table like:

```text
Kernel                  Block     Stride     Bandwidth (GB/s)     Bank Conflicts
Shared tiled            256       1          690.1                Low
Shared conflict         256       2          510.4                Medium
Shared conflict         256       16         180.9                High
Shared padded           256       16         540.7                Lower
```

## Phase 4: Polish and Documentation

Estimated time: 1 day

### Objective

Turn the project into something clean enough to show on GitHub or a CV.

### Tasks

- Add command-line arguments:

```text
--size-mb
--iterations
--pattern
--stride
--block-size
```

- Print a clean summary table.
- Add a README.
- Add a results file with your measured numbers.
- Add Nsight Compute notes.
- Add build and run instructions.
- Push to GitHub.

### README Sections

Recommended README structure:

```text
# GPU Memory Bandwidth Benchmarker

## Overview
## Why This Project Matters
## Hardware
## Build
## Run
## Methodology
## Results
## Nsight Compute Validation
## What I Learned
## Future Work
```

### Final Results Table

The README should include:

```text
Pattern             Config             GB/s      % of Peak
Sequential copy     block=256          ...
Strided             stride=2           ...
Strided             stride=4           ...
Shared tiled        block=256          ...
Shared conflict     stride=16          ...
Shared padded       stride=16          ...
```

## Suggested Implementation Order

1. Create the CMake project.
2. Add CUDA error-checking helpers.
3. Add GPU device info printing.
4. Implement the baseline kernel.
5. Add timing with CUDA events.
6. Add repeated iterations and averaging.
7. Add coalescing kernels.
8. Add shared memory kernels.
9. Add command-line options.
10. Add formatted result tables.
11. Profile with Nsight Compute.
12. Write README and results notes.

## Recommended First Milestone

Before adding every pattern, aim for this minimal working version:

```text
Builds successfully
Runs one baseline copy kernel
Prints GPU name
Prints elapsed time
Prints effective GB/s
Checks CUDA errors
```

Once that works, the rest of the project becomes incremental.

## Common Mistakes to Avoid

- Measuring `cudaMalloc` or `cudaMemcpy` time instead of kernel time.
- Forgetting to synchronize before reading timing results.
- Using arrays too small to represent real memory bandwidth.
- Reporting single-run timing instead of averaged timing.
- Forgeting that copy kernels transfer both read and write bytes.
- Comparing measured GB/s to the wrong GPU spec.
- Assuming shared memory is always faster.
- Ignoring compiler optimization effects.

## Stretch Goals

After the main project works, consider adding:

- `float`, `float2`, `float4`, and `int4` vectorized memory access comparisons
- Read-only benchmark vs write-only benchmark vs copy benchmark
- Pinned host memory transfer benchmark
- Unified memory benchmark
- Multi-GPU detection
- CSV export
- Python plotting script
- GitHub Actions build check

## Success Criteria

The project is successful when:

- The benchmark builds cleanly.
- The baseline bandwidth is reasonably close to expected hardware performance.
- Strided access shows a clear performance drop.
- Shared memory experiments show measurable differences.
- Results are documented with methodology.
- The README explains what the numbers mean, not just what they are.

