# Bandwidth Benchmark Results

## Run Summary

This file documents the current benchmark results for the CUDA memory bandwidth project.

Date: 2026-05-10  
GPU: NVIDIA GeForce MX250  
Reported global memory: 1.99988 GB  
Kernels: sequential copy, strided copy, offset copy, reverse copy  
Block size: 256  
Iterations: 100

The benchmark currently measures four kernels:

```cpp
output[idx] = input[idx];
```

and

```cpp
output[idx] = input[idx * stride];
```

and

```cpp
output[idx] = input[idx + offset];
```

and

```cpp
output[idx] = input[num_elements - 1 - idx];
```

For each output element, the kernel performs:

- one global memory read
- one global memory write

Effective bandwidth is computed as:

```text
bandwidth = (2 * bytes) / time
```

where `time` is the average kernel execution time.

## Sequential Results

| Size (MB) | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: |
| 10 | 0.431892 | 48.5573 |
| 32 | 1.36412 | 49.1957 |
| 64 | 2.72588 | 49.2384 |
| 128 | 5.44591 | 49.2912 |
| 256 | 10.89040 | 49.2974 |

## Strided Results

Test size: `32 MB`

| Pattern | Stride | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| Strided | 2 | 2.01950 | 33.2304 |
| Strided | 4 | 3.34089 | 20.0871 |
| Strided | 8 | 6.03623 | 11.1177 |
| Strided | 16 | 6.04769 | 11.0966 |
| Strided | 32 | 6.07290 | 11.0505 |

## Offset Access

Test size: `32 MB`

| Pattern | Offset | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| Offset | 2 | 1.36865 | 49.0330 |
| Offset | 4 | 1.36841 | 49.0414 |
| Offset | 8 | 1.37065 | 48.9612 |
| Offset | 16 | 1.36745 | 49.0759 |
| Offset | 32 | 1.36547 | 49.1470 |

## Reverse Access

Test size: `32 MB`

| Pattern | Parameter | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| Reverse | 2 | 1.36419 | 49.1931 |
| Reverse | 4 | 1.36417 | 49.1938 |
| Reverse | 8 | 1.36393 | 49.2027 |
| Reverse | 16 | 1.36410 | 49.1964 |
| Reverse | 32 | 1.36429 | 49.1898 |

### Observation

Reverse access performs almost the same as sequential access.
Although the direction is reversed, adjacent threads still access adjacent memory locations, so memory coalescing remains effective.

## Observations

- The sequential benchmark stabilizes quickly after `32 MB`.
- Sequential bandwidth stays very close to `49.2 GB/s`, which suggests the timing and byte-counting are consistent.
- The `10 MB` case is slightly lower, which is expected because smaller problem sizes are more sensitive to launch overhead and timing noise.
- Strided access shows the expected drop in effective bandwidth as stride increases.
- The biggest drop happens between stride `1` and stride `8`, which is a strong sign that memory coalescing is now clearly affecting performance.
- Strides `8`, `16`, and `32` cluster near `11.1 GB/s`, suggesting the access pattern has become inefficient enough that additional stride increases do not change throughput much on this GPU.
- Offset access stays near `49 GB/s` for all tested offsets, which shows that shifting the starting position does not meaningfully hurt coalescing in this benchmark.
- Reverse access stays essentially equal to sequential access, which is what we expect when warps still touch adjacent memory locations in reverse order.

## Interpretation

This is a strong result for the first two phases of the project.

What the numbers suggest:

- the helper refactor preserved benchmark behavior
- the kernel launch configuration is reasonable
- timing with CUDA events is working correctly
- the bandwidth formula is being applied consistently
- the sequential benchmark size is large enough to produce stable results
- the strided benchmark is clearly demonstrating the cost of reduced coalescing
- the offset benchmark shows that changing the base address is very different from changing the spacing between threads
- the reverse benchmark is a nice confirmation that access direction alone is not enough to break coalescing

For quick comparison against the sequential `32 MB` baseline:

| Case | Parameter | Bandwidth (GB/s) | Relative to Sequential |
| --- | ---: | ---: | ---: |
| Sequential | 1 | 49.1957 | 100.0% |
| Strided | 2 | 33.2304 | 67.5% |
| Strided | 4 | 20.0871 | 40.8% |
| Strided | 8 | 11.1177 | 22.6% |
| Strided | 16 | 11.0966 | 22.6% |
| Strided | 32 | 11.0505 | 22.5% |
| Offset | 2 | 49.0330 | 99.7% |
| Offset | 4 | 49.0414 | 99.7% |
| Offset | 8 | 48.9612 | 99.5% |
| Offset | 16 | 49.0759 | 99.8% |
| Offset | 32 | 49.1470 | 99.9% |
| Reverse | 2 | 49.1931 | 100.0% |
| Reverse | 4 | 49.1938 | 100.0% |
| Reverse | 8 | 49.2027 | 100.0% |
| Reverse | 16 | 49.1964 | 100.0% |
| Reverse | 32 | 49.1898 | 100.0% |

The next steps are:

- compare the sequential baseline against the theoretical peak bandwidth of the exact MX250 memory variant
- profile the sequential and high-stride kernels with Nsight Compute
- add shared-memory experiments for the next phase

## Notes

- These numbers reflect device-to-device kernel copy bandwidth only.
- `cudaMalloc`, `cudaMemset`, and setup work are not included in the timed region.
- This benchmark does not yet validate output correctness on the host side.
- The strided benchmark allocates a larger input buffer so that `input[idx * stride]` stays in bounds.
- The offset benchmark also allocates a larger input buffer so that `input[idx + offset]` stays in bounds.
