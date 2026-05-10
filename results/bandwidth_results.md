# Bandwidth Benchmark Results

## Run Summary

This file documents the current benchmark results for the CUDA memory bandwidth project.

Date: 2026-05-10  
GPU: NVIDIA GeForce MX250  
Reported global memory: 1.99988 GB  
Kernels: sequential copy, strided copy, offset copy, reverse copy, shared tiled copy, shared conflict copy  
Block size: 256  
Iterations: 100

The benchmark currently measures six kernels:

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

and

```cpp
shared[threadIdx.x] = input[idx];
__syncthreads();
output[idx] = shared[threadIdx.x];
```

and

```cpp
shared[shared_index] = input[idx];
__syncthreads();
output[idx] = shared[shared_index];
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
| 10 | 0.429988 | 48.7724 |
| 32 | 1.36429 | 49.1898 |
| 64 | 2.72494 | 49.2554 |
| 128 | 5.44632 | 49.2875 |
| 256 | 10.89080 | 49.2959 |

## Strided Results

Test size: `32 MB`

| Pattern | Param | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| Strided | 2 | 2.01984 | 33.2248 |
| Strided | 4 | 3.34000 | 20.0925 |
| Strided | 8 | 6.03652 | 11.1171 |
| Strided | 16 | 6.04865 | 11.0949 |
| Strided | 32 | 6.07426 | 11.0481 |

## Offset Access

Test size: `32 MB`

| Pattern | Param | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| Offset | 2 | 1.36826 | 49.0469 |
| Offset | 4 | 1.36901 | 49.0201 |
| Offset | 8 | 1.37020 | 48.9773 |
| Offset | 16 | 1.36741 | 49.0774 |
| Offset | 32 | 1.36553 | 49.1448 |

## Reverse Access

Test size: `32 MB`

| Pattern | Parameter | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| Reverse | 1 | 1.36418 | 49.1934 |

### Observation

Reverse access performs almost the same as sequential access.
Although the direction is reversed, adjacent threads still access adjacent memory locations, so memory coalescing remains effective.

## Shared Memory Tile

Test size: `32 MB`

| Pattern | Parameter | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| SharedTile | 32 | 2.54595 | 26.3591 |
| SharedTile | 64 | 1.39067 | 48.2564 |
| SharedTile | 128 | 1.38473 | 48.4633 |
| SharedTile | 256 | 1.38690 | 48.3878 |
| SharedTile | 512 | 1.40759 | 47.6764 |
| SharedTile | 1024 | 1.45693 | 46.0619 |

### Observation

Most shared-memory tiled runs are slightly slower than the direct sequential copy, and the `32`-thread case is a large outlier.
That is still a reasonable result because this kernel adds shared-memory traffic and a synchronization barrier without reducing global memory traffic.

## Shared Memory Conflict

Test size: `32 MB`

| Pattern | Parameter | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| SharedConf | 1 | 1.49623 | 44.8520 |
| SharedConf | 2 | 1.54150 | 43.5348 |
| SharedConf | 4 | 1.55292 | 43.2147 |
| SharedConf | 8 | 1.66702 | 40.2568 |
| SharedConf | 16 | 1.65706 | 40.4988 |
| SharedConf | 32 | 1.65728 | 40.4933 |

### Observation

The shared-conflict kernel is consistently slower than the plain shared-tile kernel.
As the shared stride increases, bandwidth drops into the `40-45 GB/s` range, which is consistent with shared-memory bank conflicts adding overhead.

## Observations

- The sequential benchmark stabilizes quickly after `32 MB`.
- Sequential bandwidth stays very close to `49.2 GB/s`, which suggests the timing and byte-counting are consistent.
- The `10 MB` case is slightly lower, which is expected because smaller problem sizes are more sensitive to launch overhead and timing noise.
- Strided access shows the expected drop in effective bandwidth as stride increases.
- The biggest drop happens between stride `1` and stride `8`, which is a strong sign that memory coalescing is now clearly affecting performance.
- Strides `8`, `16`, and `32` cluster near `11.1 GB/s`, suggesting the access pattern has become inefficient enough that additional stride increases do not change throughput much on this GPU.
- Offset access stays near `49 GB/s` for all tested offsets, which shows that shifting the starting position does not meaningfully hurt coalescing in this benchmark.
- Reverse access stays essentially equal to sequential access, which is what we expect when warps still touch adjacent memory locations in reverse order.
- Most shared-memory tile results land in the `46-48.5 GB/s` range, which is below the direct sequential baseline.
- The `32`-thread shared-memory case drops sharply to `26.38 GB/s`, which makes block size an important variable for the next phase.
- Shared-memory conflict results are lower than plain shared-tile results across all tested strides.
- The shared-conflict kernel drops from `44.85 GB/s` at stride `1` to about `40.5 GB/s` at strides `16` and `32`, which is a useful first sign of bank-conflict cost.

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
- the shared-memory results give you a strong setup for Phase 3, because now you have a baseline showing that shared memory needs a reason to win
- the shared-conflict results give you your first direct evidence that shared-memory access pattern matters, not just whether shared memory is used

For quick comparison against the sequential `32 MB` baseline:

| Case | Parameter | Bandwidth (GB/s) | Relative to Sequential |
| --- | ---: | ---: | ---: |
| Sequential | 1 | 49.1898 | 100.0% |
| Strided | 2 | 33.2248 | 67.5% |
| Strided | 4 | 20.0925 | 40.8% |
| Strided | 8 | 11.1171 | 22.6% |
| Strided | 16 | 11.0949 | 22.6% |
| Strided | 32 | 11.0481 | 22.5% |
| Offset | 2 | 49.0469 | 99.7% |
| Offset | 4 | 49.0201 | 99.7% |
| Offset | 8 | 48.9773 | 99.6% |
| Offset | 16 | 49.0774 | 99.8% |
| Offset | 32 | 49.1448 | 99.9% |
| Reverse | 1 | 49.1934 | 100.0% |
| SharedTile | 32 | 26.3591 | 53.6% |
| SharedTile | 64 | 48.2564 | 98.1% |
| SharedTile | 128 | 48.4633 | 98.5% |
| SharedTile | 256 | 48.3878 | 98.4% |
| SharedTile | 512 | 47.6764 | 96.9% |
| SharedTile | 1024 | 46.0619 | 93.6% |
| SharedConf | 1 | 44.8520 | 91.2% |
| SharedConf | 2 | 43.5348 | 88.5% |
| SharedConf | 4 | 43.2147 | 87.9% |
| SharedConf | 8 | 40.2568 | 81.8% |
| SharedConf | 16 | 40.4988 | 82.3% |
| SharedConf | 32 | 40.4933 | 82.3% |

The next steps are:

- compare the sequential baseline against the theoretical peak bandwidth of the exact MX250 memory variant
- profile the sequential and high-stride kernels with Nsight Compute
- add bank-conflict and padding experiments to the shared-memory phase
- compare `SharedTile` and `SharedConf` with Nsight Compute bank-conflict metrics

## Notes

- These numbers reflect device-to-device kernel copy bandwidth only.
- `cudaMalloc`, `cudaMemset`, and setup work are not included in the timed region.
- This benchmark does not yet validate output correctness on the host side.
- The strided benchmark allocates a larger input buffer so that `input[idx * stride]` stays in bounds.
- The offset benchmark also allocates a larger input buffer so that `input[idx + offset]` stays in bounds.
- The shared-memory tiled kernel still performs one global read and one global write per element, but also adds shared-memory operations and a synchronization point.
- The shared-conflict kernel uses a strided shared-memory index to intentionally stress bank behavior while keeping the global-memory access pattern unchanged.
