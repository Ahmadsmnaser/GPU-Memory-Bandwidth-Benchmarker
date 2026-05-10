#include <cmath>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "kernels.cuh"

namespace
{
constexpr int kInitializationBlockSize = 256;

bool check_cuda(cudaError_t err, const char *message)
{
    if (err != cudaSuccess)
    {
        std::cerr << message << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

bool print_gpu_info()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (!check_cuda(err, "Failed to get CUDA device count"))
    {
        return false;
    }

    std::cout << "CUDA Device Count: " << device_count << std::endl;
    if (device_count == 0)
    {
        std::cerr << "No CUDA-capable GPU found." << std::endl;
        return false;
    }

    for (int device_index = 0; device_index < device_count; ++device_index)
    {
        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, device_index);
        if (!check_cuda(err, "Failed to get GPU properties"))
        {
            return false;
        }

        double global_mem_gb =
            static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);

        std::cout << "GPU " << device_index << ": " << prop.name << std::endl;
        std::cout << "Global Memory Size: " << global_mem_gb << " GB" << std::endl;
    }

    return true;
}

size_t get_input_bytes(BenchmarkPattern pattern, size_t num_elements, size_t param)
{
    if (pattern == kStrided)
    {
        return num_elements * param * sizeof(float);
    }

    if (pattern == kOffset)
    {
        return (num_elements + param) * sizeof(float);
    }

    return num_elements * sizeof(float);
}

size_t get_shared_memory_bytes(BenchmarkPattern pattern, int block_size, size_t param)
{
    if (pattern == kSharedTile || pattern == kSharedConf)
    {
        return static_cast<size_t>(block_size) * sizeof(float);
    }

    if (pattern == kSharedNoCol)
    {
        return static_cast<size_t>(block_size) * param * sizeof(float);
    }

    return 0;
}

bool launch_benchmark_kernel(
    BenchmarkPattern pattern,
    int grid_size,
    int block_size,
    const float *dev_input,
    float *dev_output,
    size_t num_elements,
    size_t param)
{
    if (pattern == kSequential)
    {
        copy_kernel<<<grid_size, block_size>>>(dev_input, dev_output, num_elements);
        return true;
    }

    if (pattern == kStrided)
    {
        stride_copy_kernel<<<grid_size, block_size>>>(
            dev_input,
            dev_output,
            num_elements,
            param);
        return true;
    }

    if (pattern == kOffset)
    {
        offset_copy_kernel<<<grid_size, block_size>>>(
            dev_input,
            dev_output,
            num_elements,
            param);
        return true;
    }

    if (pattern == kReverse)
    {
        reverse_copy_kernel<<<grid_size, block_size>>>(dev_input, dev_output, num_elements);
        return true;
    }

    if (pattern == kSharedTile)
    {
        size_t shared_mem_bytes = get_shared_memory_bytes(pattern, block_size, param);
        shared_tiled_copy_kernel<<<grid_size, block_size, shared_mem_bytes>>>(
            dev_input,
            dev_output,
            num_elements);
        return true;
    }

    if (pattern == kSharedConf)
    {
        size_t shared_mem_bytes = get_shared_memory_bytes(pattern, block_size, param);
        shared_conflict_copy_kernel<<<grid_size, block_size, shared_mem_bytes>>>(
            dev_input,
            dev_output,
            num_elements,
            param);
        return true;
    }

    if (pattern == kSharedNoCol)
    {
        size_t shared_mem_bytes = get_shared_memory_bytes(pattern, block_size, param);
        shared_conflict_copy_kernel_without_collisions<<<grid_size, block_size, shared_mem_bytes>>>(
            dev_input,
            dev_output,
            num_elements,
            param);
        return true;
    }

    return false;
}

float expected_value(
    BenchmarkPattern pattern,
    size_t output_index,
    size_t num_elements,
    int block_size,
    size_t param)
{
    size_t block_base = (output_index / static_cast<size_t>(block_size)) * static_cast<size_t>(block_size);
    size_t tid = output_index % static_cast<size_t>(block_size);

    if (pattern == kSequential || pattern == kSharedTile)
    {
        return static_cast<float>(output_index);
    }

    if (pattern == kStrided)
    {
        return static_cast<float>(output_index * param);
    }

    if (pattern == kOffset)
    {
        return static_cast<float>(output_index + param);
    }

    if (pattern == kReverse)
    {
        return static_cast<float>(num_elements - 1 - output_index);
    }

    if (pattern == kSharedConf)
    {
        size_t shared_index = (tid * param) % static_cast<size_t>(block_size);
        return static_cast<float>(block_base + shared_index);
    }

    if (pattern == kSharedNoCol)
    {
        return static_cast<float>(output_index);
    }

    return -1.0f;
}

bool validate_output(
    BenchmarkPattern pattern,
    const float *dev_output,
    size_t num_elements,
    int block_size,
    size_t param)
{
    if (num_elements == 0)
    {
        return true;
    }

    std::vector<size_t> sample_indices;
    sample_indices.push_back(0);

    if (num_elements > 2)
    {
        sample_indices.push_back(num_elements / 2);
    }

    if (num_elements > 1)
    {
        sample_indices.push_back(num_elements - 1);
    }

    std::vector<float> host_output(sample_indices.size(), 0.0f);
    for (size_t i = 0; i < sample_indices.size(); ++i)
    {
        cudaError_t err = cudaMemcpy(
            &host_output[i],
            dev_output + sample_indices[i],
            sizeof(float),
            cudaMemcpyDeviceToHost);
        if (!check_cuda(err, "Failed to copy validation sample to host"))
        {
            return false;
        }
    }

    for (size_t i = 0; i < sample_indices.size(); ++i)
    {
        float expected = expected_value(pattern, sample_indices[i], num_elements, block_size, param);
        if (std::fabs(host_output[i] - expected) > 0.5f)
        {
            std::cerr << "Validation failed at output index " << sample_indices[i]
                      << ": expected " << expected
                      << ", got " << host_output[i] << std::endl;
            return false;
        }
    }

    return true;
}

size_t get_param_to_print(BenchmarkPattern pattern, int block_size, size_t param)
{
    if (pattern == kSharedTile)
    {
        return static_cast<size_t>(block_size);
    }

    return param;
}

bool run_benchmark(
    const char *pattern_name,
    BenchmarkPattern pattern,
    int size_mb,
    int block_size,
    int iterations,
    size_t param)
{
    size_t output_bytes = static_cast<size_t>(size_mb) * 1024 * 1024;
    size_t num_elements = output_bytes / sizeof(float);
    size_t input_bytes = get_input_bytes(pattern, num_elements, param);
    size_t input_elements = input_bytes / sizeof(float);

    float *dev_input = nullptr;
    float *dev_output = nullptr;

    cudaError_t err = cudaMalloc(&dev_input, input_bytes);
    if (!check_cuda(err, "cudaMalloc input failed"))
    {
        return false;
    }

    err = cudaMalloc(&dev_output, output_bytes);
    if (!check_cuda(err, "cudaMalloc output failed"))
    {
        cudaFree(dev_input);
        return false;
    }

    int init_grid_size =
        static_cast<int>((input_elements + kInitializationBlockSize - 1) / kInitializationBlockSize);
    initialize_input_kernel<<<init_grid_size, kInitializationBlockSize>>>(dev_input, input_elements);
    err = cudaGetLastError();
    if (!check_cuda(err, "Input initialization launch failed"))
    {
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    err = cudaDeviceSynchronize();
    if (!check_cuda(err, "Input initialization execution failed"))
    {
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    err = cudaMemset(dev_output, 0, output_bytes);
    if (!check_cuda(err, "cudaMemset output failed"))
    {
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    int grid_size = static_cast<int>((num_elements + block_size - 1) / block_size);
    if (!launch_benchmark_kernel(pattern, grid_size, block_size, dev_input, dev_output, num_elements, param))
    {
        std::cerr << "Unknown benchmark pattern." << std::endl;
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    err = cudaGetLastError();
    if (!check_cuda(err, "Warm-up kernel launch failed"))
    {
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    err = cudaDeviceSynchronize();
    if (!check_cuda(err, "Warm-up kernel execution failed"))
    {
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    cudaEvent_t start;
    cudaEvent_t stop;

    err = cudaEventCreate(&start);
    if (!check_cuda(err, "Failed to create start event"))
    {
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    err = cudaEventCreate(&stop);
    if (!check_cuda(err, "Failed to create stop event"))
    {
        cudaEventDestroy(start);
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    err = cudaEventRecord(start);
    if (!check_cuda(err, "Failed to record start event"))
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        if (!launch_benchmark_kernel(pattern, grid_size, block_size, dev_input, dev_output, num_elements, param))
        {
            std::cerr << "Unknown benchmark pattern." << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(dev_input);
            cudaFree(dev_output);
            return false;
        }
    }

    err = cudaGetLastError();
    if (!check_cuda(err, "Benchmark kernel launch failed"))
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    err = cudaEventRecord(stop);
    if (!check_cuda(err, "Failed to record stop event"))
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    err = cudaEventSynchronize(stop);
    if (!check_cuda(err, "Failed to synchronize stop event"))
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    float total_milliseconds = 0.0f;
    err = cudaEventElapsedTime(&total_milliseconds, start, stop);
    if (!check_cuda(err, "Failed to calculate elapsed time"))
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    if (!validate_output(pattern, dev_output, num_elements, block_size, param))
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    float avg_milliseconds = total_milliseconds / iterations;
    if (avg_milliseconds <= 0.0f)
    {
        std::cerr << "Invalid timing result." << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    double bytes_transferred = static_cast<double>(num_elements) * sizeof(float) * 2.0;
    double seconds = avg_milliseconds / 1000.0;
    double bandwidth_gbps = bytes_transferred / seconds / 1e9;

    std::cout << pattern_name << "\t"
              << size_mb << "\t\t"
              << get_param_to_print(pattern, block_size, param) << "\t"
              << avg_milliseconds << "\t\t"
              << bandwidth_gbps << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dev_input);
    cudaFree(dev_output);
    return true;
}
} // namespace

int main()
{
    if (!print_gpu_info())
    {
        return -1;
    }

    int block_size = 256;
    int iterations = 100;

    std::cout << "\n=== CUDA Memory Bandwidth Benchmark ===\n";
    std::cout << "Block size: " << block_size << "\n";
    std::cout << "Iterations: " << iterations << "\n\n";
    std::cout << "Pattern\t\tSize(MB)\tParam\tAvg Time(ms)\tBandwidth(GB/s)\n";
    std::cout << "-----------------------------------------------------------------------\n";

    int baseline_sizes_mb[] = {10, 32, 64, 128, 256};
    for (int size_mb : baseline_sizes_mb)
    {
        run_benchmark("Sequential", kSequential, size_mb, block_size, iterations, 1);
    }

    std::cout << "\n";

    int test_size_mb = 32;
    int strides[] = {2, 4, 8, 16, 32};
    for (int stride : strides)
    {
        run_benchmark("Strided   ", kStrided, test_size_mb, block_size, iterations, stride);
    }

    std::cout << "\n";

    for (int offset : strides)
    {
        run_benchmark("Offset    ", kOffset, test_size_mb, block_size, iterations, offset);
    }

    std::cout << "\n";

    run_benchmark("Reverse   ", kReverse, test_size_mb, block_size, iterations, 1);

    std::cout << "\n";

    int shared_tile_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int shared_block_size : shared_tile_sizes)
    {
        run_benchmark("SharedTile", kSharedTile, test_size_mb, shared_block_size, iterations, 1);
    }

    std::cout << "\n";

    int shared_strides[] = {1, 2, 4, 8, 16, 32};
    for (int shared_stride : shared_strides)
    {
        run_benchmark("SharedConf", kSharedConf, test_size_mb, block_size, iterations, shared_stride);
    }

    std::cout << "\n";

    for (int shared_stride : shared_strides)
    {
        run_benchmark("SharedNoCol", kSharedNoCol, test_size_mb, block_size, iterations, shared_stride);
    }

    std::cout << "\nBenchmark completed.\n";
    return 0;
}
