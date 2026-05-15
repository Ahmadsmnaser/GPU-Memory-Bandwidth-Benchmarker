# ⚡ GPU Memory Bandwidth Benchmarker

A CUDA benchmark project that measures effective GPU memory bandwidth across multiple memory access patterns, then compares how access behavior affects throughput.

The project focuses on:

- 🚀 CUDA kernel launch configuration
- 🧠 Global memory reads and writes
- 📏 Coalesced vs non-coalesced global memory access
- 📦 Shared memory usage
- 🧩 Shared memory stride behavior
- ⏱️ CUDA event timing
- 📊 Bandwidth calculation in GB/s
- 🔍 Performance interpretation from benchmark results

---

## 🔎 Overview

This benchmark measures how fast a GPU can move data inside device memory using simple copy-style kernels.

The baseline kernel is:

```cpp
output[idx] = input[idx];
```

This represents a coalesced global-memory copy.

The project then compares this baseline against:

- strided global-memory reads
- offset global-memory reads
- reverse global-memory reads
- shared-memory tiled copy
- shared-memory modulo conflict copy
- shared-memory strided copy without collisions

The goal is not just to print numbers, but to explain what the numbers mean.

---

## 🎯 Why This Project Matters

GPU performance depends heavily on memory access patterns.

Two kernels may perform the same number of operations, but if one accesses memory in a coalesced way and the other accesses memory with large strides, their performance can be very different.

This project demonstrates that clearly:

- ✅ Sequential access reaches about **49 GB/s**
- 📉 Strided global-memory access drops sharply
- 🔁 Offset and reverse access stay close to sequential performance
- ⚠️ Shared memory does not automatically improve performance
- 🧱 Bad shared-memory access patterns can significantly reduce throughput

---

## 🖥️ Hardware

Current benchmark machine:

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce MX250 |
| Reported global memory | 1.99988 GB |
| CUDA timing method | CUDA Events |
| Default block size | 256 |
| Iterations per benchmark | 100 |

> Note: theoretical peak bandwidth comparison is still TODO because MX250 has multiple memory variants. The exact memory type/data rate should be verified before reporting percentage of peak.

---

## ✨ Current Features

Implemented kernels:

| Pattern | Description |
|---|---|
| Sequential | Coalesced global-memory copy |
| Strided | Global-memory read with `input[idx * stride]` |
| Offset | Global-memory read with `input[idx + offset]` |
| Reverse | Global-memory read from the end toward the start |
| SharedTile | Global → shared → global tiled copy |
| SharedConf | Shared-memory modulo-based conflict stress test |
| SharedNoCol | Shared-memory strided copy without shared index collisions |

---

## 📁 Repository Structure

Suggested current structure:

```text
CUDA_Project/
├── README.md
├── PROJECT_PLAN.md
├── src/
│   └── benchmark.cu
├── results/
│   └── bandwidth_results.md
└── scripts/
    └── run_benchmarks.sh
```

Possible future structure:

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

---

## 🛠️ Build

From the `src/` directory:

```bash
nvcc benchmark.cu -o BM
```

Run:

```bash
./BM
```

Example:

```bash
cd CUDA_Project/src
nvcc benchmark.cu -o BM
./BM
```

---

## 🧪 Methodology

### ⏱️ Timing

The benchmark uses CUDA events:

```cpp
cudaEventRecord(start);

for (int i = 0; i < iterations; i++) {
    kernel<<<gridSize, blockSize>>>(...);
}

cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
```

Only kernel execution is timed.

The following are **not** included in the timed region:

- `cudaMalloc`
- `cudaMemset`
- CUDA event creation
- setup work
- cleanup work

### 📐 Bandwidth Formula

For a copy-style kernel:

```cpp
output[idx] = input[idx];
```

Each output element performs:

- one global memory read
- one global memory write

So the transferred bytes are counted as:

```text
bytes_transferred = 2 * output_size_bytes
```

Effective bandwidth:

```text
GB/s = bytes_transferred / elapsed_seconds / 1e9
```

---

## ⚙️ Kernels

### ✅ Sequential Copy

```cpp
output[idx] = input[idx];
```

Adjacent threads access adjacent memory locations.

Expected behavior: high bandwidth.

---

### 📉 Strided Global-Memory Copy

```cpp
output[idx] = input[idx * stride];
```

Adjacent threads access memory locations separated by `stride`.

Expected behavior: bandwidth drops as stride increases.

---

### ➡️ Offset Global-Memory Copy

```cpp
output[idx] = input[idx + offset];
```

The starting position changes, but adjacent threads still access adjacent memory locations.

Expected behavior: close to sequential bandwidth.

---

### 🔄 Reverse Copy

```cpp
output[idx] = input[num_elements - 1 - idx];
```

Adjacent threads access neighboring memory locations in reverse order.

Expected behavior: close to sequential bandwidth.

---

### 📦 Shared-Memory Tiled Copy

```cpp
shared[threadIdx.x] = input[idx];
__syncthreads();
output[idx] = shared[threadIdx.x];
```

This adds shared-memory operations and synchronization.

Expected behavior: not necessarily faster than sequential copy because global-memory traffic is not reduced.

---

### 🧨 Shared-Memory Modulo Conflict Copy

```cpp
int shared_index = (threadIdx.x * shared_stride) % blockDim.x;

shared[shared_index] = input[idx];
__syncthreads();
output[idx] = shared[shared_index];
```

This stresses shared-memory indexing behavior.

Note: this kernel may cause multiple threads to access the same shared-memory location, so it is useful as a stress test but not as a clean bank-conflict-only benchmark.

---

### 🧩 Shared-Memory Strided Copy Without Collisions

```cpp
int shared_index = threadIdx.x * shared_stride;

shared[shared_index] = input[idx];
__syncthreads();
output[idx] = shared[shared_index];
```

Dynamic shared memory is allocated as:

```cpp
sharedMemBytes = blockSize * shared_stride * sizeof(float);
```

This avoids multiple threads writing to the same shared-memory location while still creating a strided shared-memory access pattern.

---

## 📊 Results Summary

### 🟢 Phase 1: Sequential Global-Memory Baseline

| Size (MB) | Avg Time (ms) | Bandwidth (GB/s) |
|---:|---:|---:|
| 10 | 0.429343 | 48.8456 |
| 32 | 1.36458 | 49.1791 |
| 64 | 2.72465 | 49.2606 |
| 128 | 5.44492 | 49.3002 |
| 256 | 10.89000 | 49.2994 |

Sequential bandwidth stabilizes around **49.2 GB/s** after 32 MB.

The 256 MB result is used as the main sequential baseline:

```text
Sequential baseline: 49.2994 GB/s
```

---

### 🔵 Phase 2: Global-Memory Access Pattern Study

Test size: **32 MB**

| Pattern | Parameter | Avg Time (ms) | Bandwidth (GB/s) |
|---|---:|---:|---:|
| Sequential | 1 | 1.36458 | 49.1791 |
| Strided | 2 | 2.01954 | 33.2297 |
| Strided | 4 | 3.33984 | 20.0935 |
| Strided | 8 | 6.03605 | 11.1180 |
| Strided | 16 | 6.04921 | 11.0938 |
| Strided | 32 | 6.07281 | 11.0507 |
| Offset | 2 | 1.36922 | 49.0124 |
| Offset | 4 | 1.36884 | 49.0260 |
| Offset | 8 | 1.37045 | 48.9685 |
| Offset | 16 | 1.36764 | 49.0689 |
| Offset | 32 | 1.36652 | 49.1094 |
| Reverse | 1 | 1.36524 | 49.1554 |

#### Interpretation

Strided access significantly reduces bandwidth because adjacent threads access distant memory addresses.

Offset access remains close to sequential performance because adjacent threads still access adjacent memory locations.

Reverse access also remains close to sequential performance because access direction alone does not break coalescing.

The key lesson:

> Memory direction is not the problem. Spacing between neighboring threads' memory addresses is the real performance killer.

---

### 🟣 Phase 3: Shared-Memory Study

Test size: **32 MB**

#### 📦 Shared-Memory Tiled Copy

| Pattern | Block Size | Avg Time (ms) | Bandwidth (GB/s) |
|---|---:|---:|---:|
| SharedTile | 32 | 2.56401 | 26.1734 |
| SharedTile | 64 | 1.39265 | 48.1879 |
| SharedTile | 128 | 1.38642 | 48.4043 |
| SharedTile | 256 | 1.38866 | 48.3265 |
| SharedTile | 512 | 1.40897 | 47.6296 |
| SharedTile | 1024 | 1.45733 | 46.0493 |

Most SharedTile runs are slightly slower than sequential copy. This is expected because the sequential copy is already coalesced, while SharedTile adds shared-memory operations and a synchronization barrier.

The 32-thread block result is an outlier and should not be used as the representative shared-memory baseline.

---

#### 🧨 Shared-Memory Modulo Conflict Copy

| Pattern | Shared Stride | Avg Time (ms) | Bandwidth (GB/s) |
|---|---:|---:|---:|
| SharedConf | 1 | 1.49641 | 44.8465 |
| SharedConf | 2 | 1.52980 | 43.8676 |
| SharedConf | 4 | 1.54450 | 43.4502 |
| SharedConf | 8 | 1.66743 | 40.2469 |
| SharedConf | 16 | 1.65755 | 40.4868 |
| SharedConf | 32 | 1.65685 | 40.5038 |

The modulo-based SharedConf kernel is consistently slower than SharedTile, but it may mix bank-conflict behavior with shared-index collisions.

---

#### 🧩 Shared-Memory Strided Copy Without Collisions

| Pattern | Shared Stride | Avg Time (ms) | Bandwidth (GB/s) |
|---|---:|---:|---:|
| SharedNoCol | 1 | 1.38660 | 48.3982 |
| SharedNoCol | 2 | 1.38492 | 48.4569 |
| SharedNoCol | 4 | 1.39821 | 47.9962 |
| SharedNoCol | 8 | 1.49331 | 44.9397 |
| SharedNoCol | 16 | 2.18315 | 30.7395 |
| SharedNoCol | 32 | 4.47997 | 14.9798 |

This is the cleaner shared-memory bank-behavior experiment.

The result is clear:

- strides 1, 2, and 4 stay close to SharedTile
- stride 8 begins to degrade
- stride 16 drops hard
- stride 32 collapses to about 15 GB/s

This suggests a strong shared-memory bottleneck caused by the strided shared-memory access pattern.

---

## 📈 Relative Performance vs Sequential 32 MB Baseline

Baseline:

```text
Sequential 32 MB = 49.1791 GB/s
```

| Case | Parameter | Bandwidth (GB/s) | Relative to Sequential |
|---|---:|---:|---:|
| Sequential | 1 | 49.1791 | 100.0% |
| Strided | 2 | 33.2297 | 67.6% |
| Strided | 4 | 20.0935 | 40.9% |
| Strided | 8 | 11.1180 | 22.6% |
| Strided | 16 | 11.0938 | 22.6% |
| Strided | 32 | 11.0507 | 22.5% |
| Offset | 2 | 49.0124 | 99.7% |
| Offset | 4 | 49.0260 | 99.7% |
| Offset | 8 | 48.9685 | 99.6% |
| Offset | 16 | 49.0689 | 99.8% |
| Offset | 32 | 49.1094 | 99.9% |
| Reverse | 1 | 49.1554 | 100.0% |
| SharedTile | 32 | 26.1734 | 53.2% |
| SharedTile | 64 | 48.1879 | 98.0% |
| SharedTile | 128 | 48.4043 | 98.4% |
| SharedTile | 256 | 48.3265 | 98.3% |
| SharedTile | 512 | 47.6296 | 96.8% |
| SharedTile | 1024 | 46.0493 | 93.6% |
| SharedConf | 1 | 44.8465 | 91.2% |
| SharedConf | 2 | 43.8676 | 89.2% |
| SharedConf | 4 | 43.4502 | 88.4% |
| SharedConf | 8 | 40.2469 | 81.8% |
| SharedConf | 16 | 40.4868 | 82.3% |
| SharedConf | 32 | 40.5038 | 82.4% |
| SharedNoCol | 1 | 48.3982 | 98.4% |
| SharedNoCol | 2 | 48.4569 | 98.5% |
| SharedNoCol | 4 | 47.9962 | 97.6% |
| SharedNoCol | 8 | 44.9397 | 91.4% |
| SharedNoCol | 16 | 30.7395 | 62.5% |
| SharedNoCol | 32 | 14.9798 | 30.5% |

---

## 🧠 Key Takeaways

1. ✅ Sequential global-memory copy reaches a stable bandwidth of about **49 GB/s**.
2. 📉 Strided global-memory access heavily reduces effective bandwidth.
3. ➡️ Offset access does not significantly reduce bandwidth.
4. 🔄 Reverse access performs close to sequential access.
5. ⚠️ Shared memory does not automatically improve performance.
6. 📦 SharedTile is slightly slower than sequential copy because it adds synchronization and shared-memory traffic.
7. 🧨 The modulo-based shared-conflict kernel is useful as a stress test, but it is not the cleanest bank-conflict measurement.
8. 🧩 The no-collision shared-memory stride benchmark shows the clearest shared-memory degradation.
9. 📉 SharedNoCol stride 32 reaches only about **30.5%** of the sequential 32 MB baseline.
10. 🎯 Access pattern matters more than the raw number of operations.

---

## ⚠️ Current Limitations

- 🧪 Output correctness is not yet validated on the host side.
- 🔬 Nsight Compute has not yet been used to verify hardware-level metrics.
- 📌 Theoretical peak bandwidth has not yet been verified for this exact MX250 memory variant.
- 🖥️ The benchmark currently uses a single GPU.
- 🗂️ The code is still in a single CUDA source file.
- ⌨️ Command-line arguments are not implemented yet.
- 📄 CSV export is not implemented yet.

---

## 🔬 Nsight Compute Validation Plan

Future profiling should compare selected kernels:

```bash
ncu ./BM
```

Recommended kernels to inspect:

- Sequential 32 MB or 256 MB
- Strided stride 8 or 32
- SharedTile block 256
- SharedNoCol stride 16 or 32

Useful metrics:

```text
dram__throughput.avg.pct_of_peak_sustained_elapsed
l1tex__t_bytes.sum
dram__bytes.sum
sm__throughput.avg.pct_of_peak_sustained_elapsed
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
sm__warps_active.avg.pct_of_peak_sustained_active
```

---
