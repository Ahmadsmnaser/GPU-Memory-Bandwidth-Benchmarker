#include <iostream>
#include <cuda_runtime.h>   

// Simple CUDA kernel to copy data from input to output
__global__ void copy_kernel(const float* input , float* output , size_t num_elements){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements){
        output[idx] = input[idx];
    }
}



int main(){
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "CUDA Device Count: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; i++){
        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            std::cerr << "Failed to get properties for GPU " << i << "." << std::endl;
            return -1;
        }
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        double GlobalMemGB = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "Global Memory Size: " << GlobalMemGB << " GB" << std::endl;
    }

    size_t bytes = 1024 * 1024 * 10; // 10 MB

    size_t num_elements = bytes / sizeof(float);
    std::cout << "Buffer size in bytes: " << bytes << " that equals: " << bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Number of float elements: " << num_elements << std::endl;

    float *dev_input = nullptr, *dev_output = nullptr;
    err = cudaMalloc(&dev_input, bytes);
    if(dev_input == nullptr || err != cudaSuccess){
        std::cerr << "Failed to allocate memory on the GPU for input: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }else{
        std::cout << "Successfully allocated memory on the GPU for input. " << std::endl;
    }
    err = cudaMalloc(&dev_output, bytes);
    if(dev_output == nullptr || err != cudaSuccess){
        std::cerr << "Failed to allocate memory on the GPU for output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(dev_input);
        return -1;
    }else{
        std::cout << "Successfully allocated memory on the GPU for output. " << std::endl;
    }

    err = cudaMemset(dev_input, 1, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize input buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(dev_input);
        cudaFree(dev_output);
        return -1;
    }
    err = cudaMemset(dev_output, 0, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize output buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(dev_input);
        cudaFree(dev_output);
        return -1;
    }

    int blockSize = 256;
    int gridSize = static_cast<int>((num_elements + blockSize - 1) / blockSize);

    std::cout << "Block size: " << blockSize << std::endl;
    std::cout << "Grid size: " << gridSize << std::endl;


    copy_kernel<<<gridSize, blockSize>>>(dev_input, dev_output, num_elements);

    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelErr) << std::endl;
        cudaFree(dev_input);
        std::cout << "Freed memory for input buffer." << std::endl;
        cudaFree(dev_output);
        std::cout << "Freed memory for output buffer." << std::endl;
        return -1;
    }

    cudaError_t syncErr = cudaDeviceSynchronize(); // Ensure kernel execution is complete
    if (syncErr != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(syncErr) << std::endl;
        cudaFree(dev_input);
        std::cout << "Freed memory for input buffer." << std::endl;
        cudaFree(dev_output);
        std::cout << "Freed memory for output buffer." << std::endl;
        return -1;
    }

    std::cout << "Kernel executed successfully." << std::endl;


    cudaFree(dev_input);
    std::cout << "Freed memory for input buffer." << std::endl;
    cudaFree(dev_output);
    std::cout << "Freed memory for output buffer." << std::endl;






    return 0;

}