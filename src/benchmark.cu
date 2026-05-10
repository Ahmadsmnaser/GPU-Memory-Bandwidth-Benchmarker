#include <iostream>
#include <cuda_runtime.h>

// Simple CUDA kernel to copy data from input to output
__global__ void copy_kernel(const float *input, float *output, size_t num_elements)
{
    // Global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against out-of-bounds access
    if (idx < num_elements)
    {
        output[idx] = input[idx];
    }
}

int main()
{
    int deviceCount = 0;

    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get CUDA device count: "
                  << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    std::cout << "CUDA Device Count: " << deviceCount << std::endl;

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable GPU found." << std::endl;
        return -1;
    }

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop{};

        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to get properties for GPU " << i << ": "
                      << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        double globalMemGB =
            static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);

        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "Global Memory Size: " << globalMemGB << " GB" << std::endl;
    }

    int blockSize = 256;
    int iterations = 100;

    // Test sizes in MB
    int sizesMB[] = {10, 32, 64, 128, 256};

    std::cout << "\n=== Sequential Copy Bandwidth Benchmark ===\n";
    std::cout << "Block size: " << blockSize << "\n";
    std::cout << "Iterations: " << iterations << "\n\n";

    std::cout << "Size(MB)\tAvg Time(ms)\tBandwidth(GB/s)\n";
    std::cout << "------------------------------------------------\n";

    for (int sizeMB : sizesMB)
    {
        size_t bytes = static_cast<size_t>(sizeMB) * 1024 * 1024;
        size_t num_elements = bytes / sizeof(float);

        float *dev_input = nullptr;
        float *dev_output = nullptr;

        // Allocate input buffer on GPU
        err = cudaMalloc(&dev_input, bytes);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaMalloc input failed for " << sizeMB
                      << " MB: " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        // Allocate output buffer on GPU
        err = cudaMalloc(&dev_output, bytes);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaMalloc output failed for " << sizeMB
                      << " MB: " << cudaGetErrorString(err) << std::endl;
            cudaFree(dev_input);
            continue;
        }

        // Initialize input buffer
        err = cudaMemset(dev_input, 1, bytes);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaMemset input failed for " << sizeMB
                      << " MB: " << cudaGetErrorString(err) << std::endl;
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        // Initialize output buffer
        err = cudaMemset(dev_output, 0, bytes);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaMemset output failed for " << sizeMB
                      << " MB: " << cudaGetErrorString(err) << std::endl;
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        int gridSize =
            static_cast<int>((num_elements + blockSize - 1) / blockSize);

        // Warm-up kernel launch to avoid first-launch overhead
        copy_kernel<<<gridSize, blockSize>>>(dev_input, dev_output, num_elements);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Warm-up kernel launch failed for " << sizeMB
                      << " MB: " << cudaGetErrorString(err) << std::endl;
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            std::cerr << "Warm-up kernel execution failed for " << sizeMB
                      << " MB: " << cudaGetErrorString(err) << std::endl;
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        cudaEvent_t start, stop;

        err = cudaEventCreate(&start);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to create start event: "
                      << cudaGetErrorString(err) << std::endl;
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        err = cudaEventCreate(&stop);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to create stop event: "
                      << cudaGetErrorString(err) << std::endl;
            cudaEventDestroy(start);
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        // Start timing
        err = cudaEventRecord(start);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to record start event: "
                      << cudaGetErrorString(err) << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        // Timed benchmark loop
        for (int i = 0; i < iterations; i++)
        {
            copy_kernel<<<gridSize, blockSize>>>(dev_input, dev_output, num_elements);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Benchmark kernel launch failed for " << sizeMB
                      << " MB: " << cudaGetErrorString(err) << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        // Stop timing
        err = cudaEventRecord(stop);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to record stop event: "
                      << cudaGetErrorString(err) << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        err = cudaEventSynchronize(stop);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to synchronize stop event: "
                      << cudaGetErrorString(err) << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        float totalMilliseconds = 0.0f;

        err = cudaEventElapsedTime(&totalMilliseconds, start, stop);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed to calculate elapsed time: "
                      << cudaGetErrorString(err) << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        float avgMilliseconds = totalMilliseconds / iterations;

        if (avgMilliseconds <= 0.0f)
        {
            std::cerr << "Invalid timing result for " << sizeMB << " MB." << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(dev_input);
            cudaFree(dev_output);
            continue;
        }

        // Copy kernel transfers: one read + one write
        double bytesTransferred =
            static_cast<double>(num_elements) * sizeof(float) * 2.0;

        double seconds = avgMilliseconds / 1000.0;
        double bandwidthGBps = bytesTransferred / seconds / 1e9;

        std::cout << sizeMB << "\t\t"
                  << avgMilliseconds << "\t\t"
                  << bandwidthGBps << "\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(dev_input);
        cudaFree(dev_output);
    }

    return 0;
}