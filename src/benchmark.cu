#include <iostream>
#include <cuda_runtime.h>

// Sequential copy kernel: coalesced access
__global__ void copy_kernel(const float *input, float *output, size_t num_elements)
{
    // Global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid out-of-bounds access
    if (idx < num_elements)
    {
        output[idx] = input[idx];
    }
}

// Strided copy kernel: non-coalesced / less-coalesced access
__global__ void stride_copy_kernel(
    const float *input,
    float *output,
    size_t num_elements,
    size_t stride)
{
    // Global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // input must be allocated with num_elements * stride capacity
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
    // Global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid out-of-bounds access
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
        // Load data into shared memory tile
        shared[tid] = input[idx];
    }
    __syncthreads();

    if (idx < num_elements)
    {
        // Write data from shared memory to output
        output[idx] = shared[tid];
    }
}

// Helper function to check CUDA errors
bool check_cuda(cudaError_t err, const char *message)
{
    if (err != cudaSuccess)
    {
        std::cerr << message << ": " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

// Helper function to print GPU info
bool print_gpu_info()
{
    int deviceCount = 0;

    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (!check_cuda(err, "Failed to get CUDA device count"))
    {
        return false;
    }

    std::cout << "CUDA Device Count: " << deviceCount << std::endl;

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable GPU found." << std::endl;
        return false;
    }

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop{};

        err = cudaGetDeviceProperties(&prop, i);
        if (!check_cuda(err, "Failed to get GPU properties"))
        {
            return false;
        }

        double globalMemGB =
            static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);

        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "Global Memory Size: " << globalMemGB << " GB" << std::endl;
    }

    return true;
}

bool launch_benchmark_kernel(
    int pattern,
    int gridSize,
    int blockSize,
    const float *dev_input,
    float *dev_output,
    size_t num_elements,
    size_t stride)
{
    if (pattern == 0)
    {
        copy_kernel<<<gridSize, blockSize>>>(dev_input, dev_output, num_elements);
        return true;
    }

    if (pattern == 1)
    {
        stride_copy_kernel<<<gridSize, blockSize>>>(
            dev_input,
            dev_output,
            num_elements,
            stride);
        return true;
    }

    if (pattern == 2)
    {
        offset_copy_kernel<<<gridSize, blockSize>>>(
            dev_input,
            dev_output,
            num_elements,
            stride);
        return true;
    }

    if (pattern == 3)
    {
        reverse_copy_kernel<<<gridSize, blockSize>>>(dev_input, dev_output, num_elements);
        return true;
    }

    return false;
}

// Generic benchmark helper
// pattern:
//   0 = sequential copy
//   1 = strided copy
//   2 = offset copy
//   3 = reverse copy
bool run_benchmark(
    const char *patternName,
    int pattern,
    int sizeMB,
    int blockSize,
    int iterations,
    size_t stride)
{
    size_t outputBytes = static_cast<size_t>(sizeMB) * 1024 * 1024;
    size_t num_elements = outputBytes / sizeof(float);

    // For strided and offset access, input must be larger to avoid out-of-bounds access
    size_t inputBytes = outputBytes;
    if (pattern == 1 || pattern == 2)
    {
        inputBytes = outputBytes * stride;
    }

    float *dev_input = nullptr;
    float *dev_output = nullptr;

    cudaError_t err = cudaMalloc(&dev_input, inputBytes);
    if (!check_cuda(err, "cudaMalloc input failed"))
    {
        return false;
    }

    err = cudaMalloc(&dev_output, outputBytes);
    if (!check_cuda(err, "cudaMalloc output failed"))
    {
        cudaFree(dev_input);
        return false;
    }

    // Initialize GPU buffers
    err = cudaMemset(dev_input, 1, inputBytes);
    if (!check_cuda(err, "cudaMemset input failed"))
    {
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    err = cudaMemset(dev_output, 0, outputBytes);
    if (!check_cuda(err, "cudaMemset output failed"))
    {
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    int gridSize =
        static_cast<int>((num_elements + blockSize - 1) / blockSize);

    // Warm-up kernel to avoid first-launch overhead
    if (!launch_benchmark_kernel(
            pattern,
            gridSize,
            blockSize,
            dev_input,
            dev_output,
            num_elements,
            stride))
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

    // Timed benchmark loop
    for (int i = 0; i < iterations; i++)
    {
        if (!launch_benchmark_kernel(
                pattern,
                gridSize,
                blockSize,
                dev_input,
                dev_output,
                num_elements,
                stride))
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

    float totalMilliseconds = 0.0f;

    err = cudaEventElapsedTime(&totalMilliseconds, start, stop);
    if (!check_cuda(err, "Failed to calculate elapsed time"))
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    float avgMilliseconds = totalMilliseconds / iterations;

    if (avgMilliseconds <= 0.0f)
    {
        std::cerr << "Invalid timing result." << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(dev_input);
        cudaFree(dev_output);
        return false;
    }

    // Effective bandwidth:
    // one global read + one global write
    double bytesTransferred =
        static_cast<double>(num_elements) * sizeof(float) * 2.0;

    double seconds = avgMilliseconds / 1000.0;
    double bandwidthGBps = bytesTransferred / seconds / 1e9;

    std::cout << patternName << "\t"
              << sizeMB << "\t\t"
              << stride << "\t"
              << avgMilliseconds << "\t\t"
              << bandwidthGBps << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_input);
    cudaFree(dev_output);

    return true;
}

int main()
{
    if (!print_gpu_info())
    {
        return -1;
    }

    int blockSize = 256;
    int iterations = 100;

    std::cout << "\n=== CUDA Memory Bandwidth Benchmark ===\n";
    std::cout << "Block size: " << blockSize << "\n";
    std::cout << "Iterations: " << iterations << "\n\n";

    std::cout << "Pattern\t\tSize(MB)\tStride\tAvg Time(ms)\tBandwidth(GB/s)\n";
    std::cout << "-----------------------------------------------------------------------\n";

    // Phase 1: baseline sequential copy over several sizes
    int baselineSizesMB[] = {10, 32, 64, 128, 256};

    for (int sizeMB : baselineSizesMB)
    {
        run_benchmark(
            "Sequential",
            0,
            sizeMB,
            blockSize,
            iterations,
            1);
    }

    std::cout << "\n";

    // Phase 2: strided access, fixed size for fair comparison
    int testSizeMB = 32;
    int strides[] = {2, 4, 8, 16, 32};

    for (int stride : strides)
    {
        run_benchmark(
            "Strided   ",
            1,
            testSizeMB,
            blockSize,
            iterations,
            stride);
    }
    std::cout << "\n";

    for (int offset : strides)
    {
        run_benchmark(
            "Offset    ",
            2,
            testSizeMB,
            blockSize,
            iterations,
            offset);
    }
    std::cout << "\n";

    for (int stride : strides)
    {
        run_benchmark(
            "Reverse   ",
            3,
            testSizeMB,
            blockSize,
            iterations,
            stride);
    }

    return 0;
}
