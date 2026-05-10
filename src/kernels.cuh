#pragma once

#include <cstddef>
#include <cuda_runtime.h>

enum BenchmarkPattern
{
    kSequential = 0,
    kStrided = 1,
    kOffset = 2,
    kReverse = 3,
    kSharedTile = 4,
    kSharedConf = 5,
    kSharedNoCol = 6
};

__global__ void initialize_input_kernel(float *data, size_t num_elements);

__global__ void copy_kernel(const float *input, float *output, size_t num_elements);

__global__ void stride_copy_kernel(
    const float *input,
    float *output,
    size_t num_elements,
    size_t stride);

__global__ void offset_copy_kernel(
    const float *input,
    float *output,
    size_t num_elements,
    size_t offset);

__global__ void reverse_copy_kernel(
    const float *input,
    float *output,
    size_t num_elements);

__global__ void shared_tiled_copy_kernel(
    const float *input,
    float *output,
    size_t num_elements);

__global__ void shared_conflict_copy_kernel(
    const float *input,
    float *output,
    size_t num_elements,
    size_t shared_stride);

__global__ void shared_conflict_copy_kernel_without_collisions(
    const float *input,
    float *output,
    size_t num_elements,
    size_t shared_stride);
