// we will start the project with printing:
// GPU name
// CUDA device count
// Global memory size

#include <iostream>
#include <cuda_runtime.h>   


int main(){
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count." << std::endl;
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
        double GlobalMemGB = static_cast<double>(prop.totalGlobalMem) / (1024 * 1024 * 1024);
        std::cout << "Global Memory Size: " << GlobalMemGB << "GB" << std::endl;
    }

    size_t bytes = 1024 * 1024 * 10; // 10 MB
    float *dev_input = nullptr, *dev_output = nullptr;
    cudaError_t err = cudaMalloc(&dev_input, bytes);
    if(dev_input == nullptr || err != cudaSuccess){
        std::cerr << "Failed to allocate memory on the GPU for input." << std::endl;
        return -1;
    }else{
        std::cout << "Successfully allocated memory on the GPU for input." << std::endl;
    }
    err = cudaMalloc(&dev_output, bytes);
    if(dev_output == nullptr || err != cudaSuccess){
        std::cerr << "Failed to allocate memory on the GPU for output." << std::endl;
        cudaFree(dev_input);
        return -1;
    }else{
        std::cout << "Successfully allocated memory on the GPU for output." << std::endl;
    }




    cudaFree(dev_input);
    cudaFree(dev_output);






    return 0;

}