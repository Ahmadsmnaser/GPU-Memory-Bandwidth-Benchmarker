#include "kernels.cuh"

__global__ void initialize_input_kernel(float *data, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        data[idx] = static_cast<float>(idx);
    }
}

__global__ void copy_kernel(const float *input, float *output, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        output[idx] = input[idx];
    }
}

__global__ void stride_copy_kernel(
    const float *input,
    float *output,
    size_t num_elements,
    size_t stride)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        output[idx] = input[idx * stride];
    }
}

__global__ void offset_copy_kernel(
    const float *input,
    float *output,
    size_t num_elements,
    size_t offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        output[idx] = input[idx + offset];
    }
}

__global__ void reverse_copy_kernel(
    const float *input,
    float *output,
    size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        output[idx] = input[num_elements - 1 - idx];
    }
}

__global__ void shared_tiled_copy_kernel(
    const float *input,
    float *output,
    size_t num_elements)
{
    extern __shared__ float shared[];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < num_elements)
    {
        shared[tid] = input[idx];
    }

    __syncthreads();

    if (idx < num_elements)
    {
        output[idx] = shared[tid];
    }
}

__global__ void shared_conflict_copy_kernel(
    const float *input,
    float *output,
    size_t num_elements,
    size_t shared_stride)
{
    extern __shared__ float shared[];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int shared_index = (tid * static_cast<int>(shared_stride)) % blockDim.x;

    if (idx < num_elements)
    {
        shared[tid] = input[idx];
    }

    __syncthreads();

    if (idx < num_elements)
    {
        output[idx] = shared[shared_index];
    }
}

__global__ void shared_conflict_copy_kernel_without_collisions(
    const float *input,
    float *output,
    size_t num_elements,
    size_t shared_stride)
{
    extern __shared__ float shared[];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    size_t shared_index = static_cast<size_t>(tid) * shared_stride;

    if (idx < num_elements)
    {
        shared[shared_index] = input[idx];
    }

    __syncthreads();

    if (idx < num_elements)
    {
        output[idx] = shared[shared_index];
    }
}
