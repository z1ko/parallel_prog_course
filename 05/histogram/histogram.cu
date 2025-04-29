#define PPROG_IMPLEMENTATION

#include <random.cuh>
#include <timer.cuh>
#include <utils.cuh>

#include <chrono>
#include <thread>
#include <fstream>

// ===================================================================================
// CONFIGURATION

#define BLOCK_SIZE 256
#define BLOCK_COUNT 24
#define NUM_BINS 128

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

//@@ Write sequential histogram kernel 
std::array<int, NUM_BINS> histogram_cpu(char* text, int N) {
    std::array<int, NUM_BINS> bins{0};

    //@@ ..

    return bins;
}

//@@ Write the histogram kernel without shared memory
__global__ void histogram_gpu_naive(char* text, int* bins, int N) {
    //@@ ..
}

//@@ Write the histogram kernel with privatized shared memory
__global__ void histogram_gpu_private(char* text, int* bins, int N) {
    //@@ ..
}

// ===================================================================================

int main(int argc, char** argv) {

    // Load big text file
    std::ifstream file("sherlock_holmes.txt", std::ios::ate);
    std::streamsize N = file.tellg();
    file.seekg(0, std::ios::beg);

    printf("loading file with %ld characters\n", N);

    constexpr unsigned int BUFFER_SIZE = 20e6; // 20 MB
    char* h_text = new char[BUFFER_SIZE];
    if (!file.read(h_text, N)) {
        printf("can't read file!\n");
        return -1;
    }

    //h_text[N] = '\0';
    //printf("%s\n", h_text);

    std::array<int, NUM_BINS> result_cpu;
    std::array<int, NUM_BINS> result_gpu_naive;
    std::array<int, NUM_BINS> result_gpu_private;

    // Allocate matrix in GPU memory
    char *d_text;
    cudaMalloc((void**)&d_text, N);
    cudaMemcpy(d_text, h_text, N, cudaMemcpyHostToDevice);
    cuda_check_error();

    // Create result matrix
    int *d_bins;
    cudaMalloc(&d_bins, NUM_BINS * sizeof(int));
    cuda_check_error();

    // ===================================================================================
    // SEQUENTIAL

    auto timer_cpu = timerCPU{};
    timer_cpu.start();

    // Sequential
    result_cpu = histogram_cpu(h_text, N);

    timer_cpu.stop();

    // ===================================================================================
    // GPU NAIVE

    auto timer_gpu_naive = timerGPU{};
    timer_gpu_naive.start();

    // Kernel launch
    histogram_gpu_naive<<<BLOCK_COUNT, BLOCK_SIZE>>>(d_text, d_bins, N);

    timer_gpu_naive.stop();

    // Move result matrix to CPU memory
    cudaMemcpy(result_gpu_naive.data(), d_bins, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check_error();

    // ===================================================================================
    // GPU WITH SHARED MEMORY

    cudaMemset(d_bins, 0, NUM_BINS * sizeof(int));
    cuda_check_error();

    auto timer_gpu_shmem = timerGPU{};
    timer_gpu_shmem.start();

    // Kernel launch
    histogram_gpu_private<<<BLOCK_COUNT, BLOCK_SIZE>>>(d_text, d_bins, N);

    timer_gpu_shmem.stop();

    // Move result matrix to CPU memory
    cudaMemcpy(result_gpu_private.data(), d_bins, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check_error();

    // ===================================================================================
    // VISUALIZE

    printf("%8s\t%8s\t%8s\n", "cpu", "naive", "private");
    for (int i = 0; i < NUM_BINS; ++i) {
        printf("%8d\t%8d\t%8d\n", result_cpu[i], result_gpu_naive[i], result_gpu_private[i]);
    }
    printf("\n");

    // ===================================================================================
    // TIMERS

    auto cpu_ms = timer_cpu.elapsed_ms();
    auto gpu_naive_ms = timer_gpu_naive.elapsed_ms();
    auto gpu_shmem_ms = timer_gpu_shmem.elapsed_ms();

    printf("histogram CPU\n\t%f ms\n", cpu_ms);

    printf("histogram GPU:NAIVE\n\t%f ms (speedup: cpu %.2fx)\n", 
        gpu_naive_ms, cpu_ms / gpu_naive_ms);

    printf("histogram GPU:SHMEM\n\t%f ms (speedup: cpu %.2fx, gpu_naive %.2fx)\n", 
        gpu_shmem_ms, cpu_ms / gpu_shmem_ms, gpu_naive_ms / gpu_shmem_ms);

    // ===================================================================================
    // CHECK

    bool ok_naive = (result_cpu == result_gpu_naive);
    printf("Solution CPU vs NAIVE: %s\n", ok_naive ? "CORRECT" : "INCORRECT");

    bool ok_shmem = (result_cpu == result_gpu_private);
    printf("Solution CPU vs SHMEM: %s\n", ok_shmem ? "CORRECT" : "INCORRECT");

    // ===================================================================================

    return 0;
}
