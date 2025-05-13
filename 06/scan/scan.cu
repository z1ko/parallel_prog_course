#define PPROG_IMPLEMENTATION

#include <random.cuh>
#include <timer.cuh>
#include <utils.cuh>

#include <chrono>
#include <thread>
#include <fstream>
#include <array>

// ===================================================================================
// CONFIGURATION

// ~1M elements
#define N (1 << 20)

#define BLOCK_SIZE 512

// ===================================================================================
// UTILITIES

#define cuda_check_error()                                                     \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

// ===================================================================================
// @@ IMPLEMENTATION

//@@ Sequential inclusive scan 
std::vector<int> scan_cpu(int* elements, int n) {
    
    std::vector<int> result(N);
    result[0] = elements[0];

    for (int i = 1; i < n; ++i)
        result[i] += result[i - 1] + elements[i];

    return result;
}

//@@ Write the inclusive scan fixup kernel for multi-block grids
__global__ void scan_fixup_gpu(int* result, int* scan_blocks, int n) {
  //@@...
}

//@@ Write the inclusive scan kernel using the up-down sweep algorithm
//   NOTE: 'blocks' can be NULL! 
__global__ void scan_gpu(int* elements, int* result, int* blocks, int n) {
    //@@...
}

// ===================================================================================

int main(int argc, char** argv) {

    std::vector<int> elements = random_array(1, 100, N);

    std::vector<int> result_cpu;
    std::vector<int> result_gpu;
    result_gpu.resize(N);

    // Allocate matrix in GPU memory
    int *d_elements, *d_result;
    cudaMalloc(&d_elements, N * sizeof(int));
    cudaMemcpy(d_elements, elements.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_result, N * sizeof(int));
    cudaMemset(d_result, 0, N * sizeof(int));
    cuda_check_error();

    // Each block is responsible for (2 * BLOCK_SIZE) elements
    const int block_count = ceil((float)N / (BLOCK_SIZE * 2));
    printf("block_count: %d\n", block_count);

    // Create result and auxiliary arrays
    int *d_blocks, *d_scan_blocks;
    cudaMalloc(&d_blocks, 2 * block_count * sizeof(int));
    cudaMalloc(&d_scan_blocks, 2 * block_count * sizeof(int));
    cuda_check_error();

    // ===================================================================================
    // SEQUENTIAL

    auto timer_cpu = timerCPU{};
    timer_cpu.start();

    // Sequential
    result_cpu = scan_cpu(elements.data(), N);

    timer_cpu.stop();

    // ===================================================================================
    // GPU NAIVE

    auto timer_gpu = timerGPU{};
    timer_gpu.start();

    // Local scan inside each block, also stores block largest value
    scan_gpu<<<block_count, BLOCK_SIZE>>>(d_elements, d_result, d_blocks, N);
    cudaDeviceSynchronize();

    // Scan of all the blocks largest values
    scan_gpu<<<1, BLOCK_SIZE>>>(d_blocks, d_scan_blocks, NULL, BLOCK_SIZE * 2);
    cudaDeviceSynchronize();
    
    // Fixes local scan values using the scan of all the blocks largest values
    scan_fixup_gpu<<<block_count, BLOCK_SIZE>>>(d_result, d_scan_blocks, N);
    cudaDeviceSynchronize();

    timer_gpu.stop();

    // Move result matrix to CPU memory
    cudaMemcpy(result_gpu.data(), d_result, N * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check_error();

    // ===================================================================================
    // TIMERS

    auto cpu_ms = timer_cpu.elapsed_ms();
    auto gpu_ms = timer_gpu.elapsed_ms();

    printf("reduction CPU\n\t%f ms\n", cpu_ms);

    printf("reduction GPU\n\t%f ms (speedup: cpu %.2fx)\n", 
        gpu_ms, cpu_ms / gpu_ms);

    // ===================================================================================
    // CHECK

    //for (int i = 0; i < N; ++i) {
    //    printf("%6d\t%6d\t%6d\n", elements[i], result_cpu[i], result_gpu[i]);
    //}

    bool ok_naive = (result_cpu == result_gpu);
    printf("Solution CPU vs GPU: %s\n", ok_naive ? "CORRECT" : "INCORRECT");

    // ===================================================================================

    return 0;
}
