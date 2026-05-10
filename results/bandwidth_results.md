# Bandwidth Benchmark Results

## Run Summary

This file documents the current benchmark results for the CUDA memory bandwidth project.

Date: 2026-05-10  
GPU: NVIDIA GeForce MX250  
Reported global memory: 1.99988 GB  
Kernels: sequential copy, strided copy, offset copy, reverse copy, shared tiled copy, shared conflict copy, shared no-coalescing copy  
Block size: 256  
Iterations: 100

The benchmark currently measures seven kernels:

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

and

```cpp
shared[tid] = input[idx];
__syncthreads();
output[idx] = shared[(tid * shared_stride) % blockDim.x];
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
| 10 | 0.429138 | 48.8690 |
| 32 | 1.36394 | 49.2023 |
| 64 | 2.72516 | 49.2513 |
| 128 | 5.44525 | 49.2971 |
| 256 | 10.89030 | 49.2981 |

## Strided Results

Test size: `32 MB`

| Pattern | Param | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| Strided | 2 | 2.02004 | 33.2215 |
| Strided | 4 | 3.33984 | 20.0935 |
| Strided | 8 | 6.03685 | 11.1165 |
| Strided | 16 | 6.04873 | 11.0947 |
| Strided | 32 | 6.07329 | 11.0498 |

## Offset Access

Test size: `32 MB`

| Pattern | Param | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| Offset | 2 | 1.36864 | 49.0333 |
| Offset | 4 | 1.36867 | 49.0322 |
| Offset | 8 | 1.37054 | 48.9652 |
| Offset | 16 | 1.36731 | 49.0811 |
| Offset | 32 | 1.36659 | 49.1068 |

## Reverse Access

Test size: `32 MB`

| Pattern | Parameter | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| Reverse | 1 | 1.36451 | 49.1816 |

### Observation

Reverse access performs almost the same as sequential access.
Although the direction is reversed, adjacent threads still access adjacent memory locations, so memory coalescing remains effective.

## Shared Memory Tile

Test size: `32 MB`

| Pattern | Parameter | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| SharedTile | 32 | 2.56379 | 26.1757 |
| SharedTile | 64 | 1.39178 | 48.2180 |
| SharedTile | 128 | 1.38525 | 48.4454 |
| SharedTile | 256 | 1.38780 | 48.3564 |
| SharedTile | 512 | 1.40799 | 47.6629 |
| SharedTile | 1024 | 1.45645 | 46.0772 |

### Observation

Most shared-memory tiled runs are slightly slower than the direct sequential copy, and the `32`-thread case is a large outlier.
That is still a reasonable result because this kernel adds shared-memory traffic and a synchronization barrier without reducing global memory traffic.

## Shared Memory Conflict

Test size: `32 MB`

| Pattern | Parameter | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| SharedConf | 1 | 1.38085 | 48.5995 |
| SharedConf | 2 | 1.39020 | 48.2727 |
| SharedConf | 4 | 1.39041 | 48.2656 |
| SharedConf | 8 | 1.41710 | 47.3564 |
| SharedConf | 16 | 1.41983 | 47.2655 |
| SharedConf | 32 | 1.41990 | 47.2631 |

### Observation

The corrected shared-conflict kernel is now much closer to the plain shared-tile kernel because it no longer has shared-memory write aliasing between threads.
There is still a measurable drop as the shared stride increases, but it is much smaller than before, which makes this a cleaner bank-conflict experiment.

## Shared Memory No-Coalescing

Test size: `32 MB`

| Pattern | Parameter | Avg Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: | ---: |
| SharedNoCol | 1 | 1.38787 | 48.3539 |
| SharedNoCol | 2 | 1.38470 | 48.4644 |
| SharedNoCol | 4 | 1.39780 | 48.0103 |
| SharedNoCol | 8 | 1.49245 | 44.9656 |
| SharedNoCol | 16 | 2.18313 | 30.7398 |
| SharedNoCol | 32 | 4.47946 | 14.9815 |

### Observation

The shared no-coalescing kernel stays close to the tiled baseline for small strides, then degrades sharply as the shared stride increases.
The drop to `30.74 GB/s` at stride `16` and `14.98 GB/s` at stride `32` suggests a much stronger penalty than the plain shared-conflict kernel.

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
- Shared-memory conflict results are now only slightly lower than plain shared-tile results across the tested strides.
- The corrected shared-conflict kernel drops from `48.60 GB/s` at stride `1` to about `47.26 GB/s` at strides `16` and `32`, which is a milder but cleaner signal.
- The shared no-coalescing kernel is mild at strides `1-4`, then drops hard at strides `8`, `16`, and `32`.
- `SharedNoCol` falls much more sharply than `SharedConf`, which suggests the readback access pattern is now creating a stronger shared-memory bottleneck.

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
- the shared no-coalescing results make the shared-memory story even clearer: once the shared-memory access pattern gets bad enough, throughput can collapse dramatically

For quick comparison against the sequential `32 MB` baseline:

| Case | Parameter | Bandwidth (GB/s) | Relative to Sequential |
| --- | ---: | ---: | ---: |
| Sequential | 1 | 49.2023 | 100.0% |
| Strided | 2 | 33.2215 | 67.5% |
| Strided | 4 | 20.0935 | 40.9% |
| Strided | 8 | 11.1165 | 22.6% |
| Strided | 16 | 11.0947 | 22.5% |
| Strided | 32 | 11.0498 | 22.5% |
| Offset | 2 | 49.0333 | 99.7% |
| Offset | 4 | 49.0322 | 99.7% |
| Offset | 8 | 48.9652 | 99.5% |
| Offset | 16 | 49.0811 | 99.8% |
| Offset | 32 | 49.1068 | 99.8% |
| Reverse | 1 | 49.1816 | 100.0% |
| SharedTile | 32 | 26.1757 | 53.2% |
| SharedTile | 64 | 48.2180 | 98.0% |
| SharedTile | 128 | 48.4454 | 98.5% |
| SharedTile | 256 | 48.3564 | 98.3% |
| SharedTile | 512 | 47.6629 | 96.9% |
| SharedTile | 1024 | 46.0772 | 93.7% |
| SharedConf | 1 | 48.5995 | 98.8% |
| SharedConf | 2 | 48.2727 | 98.1% |
| SharedConf | 4 | 48.2656 | 98.1% |
| SharedConf | 8 | 47.3564 | 96.2% |
| SharedConf | 16 | 47.2655 | 96.1% |
| SharedConf | 32 | 47.2631 | 96.1% |
| SharedNoCol | 1 | 48.3539 | 98.3% |
| SharedNoCol | 2 | 48.4644 | 98.5% |
| SharedNoCol | 4 | 48.0103 | 97.6% |
| SharedNoCol | 8 | 44.9656 | 91.4% |
| SharedNoCol | 16 | 30.7398 | 62.5% |
| SharedNoCol | 32 | 14.9815 | 30.4% |

The next steps are:

- compare the sequential baseline against the theoretical peak bandwidth of the exact MX250 memory variant
- profile the sequential and high-stride kernels with Nsight Compute
- add bank-conflict and padding experiments to the shared-memory phase
- compare `SharedTile` and `SharedConf` with Nsight Compute bank-conflict metrics
- compare `SharedConf` and `SharedNoCol` to see which shared-memory access pattern is more destructive on this GPU

## Notes

- These numbers reflect device-to-device kernel copy bandwidth only.
- `cudaMalloc`, `cudaMemset`, and setup work are not included in the timed region.
- This benchmark does not yet validate output correctness on the host side.
- The strided benchmark allocates a larger input buffer so that `input[idx * stride]` stays in bounds.
- The offset benchmark also allocates a larger input buffer so that `input[idx + offset]` stays in bounds.
- The shared-memory tiled kernel still performs one global read and one global write per element, but also adds shared-memory operations and a synchronization point.
- The shared-conflict kernel uses a strided shared-memory index to intentionally stress bank behavior while keeping the global-memory access pattern unchanged.
- The shared no-coalescing kernel keeps the global-memory load simple, but uses a strided shared-memory readback pattern that becomes increasingly expensive as the stride grows.
