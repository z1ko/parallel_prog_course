#define PPROG_IMPLEMENTATION

#include <random.cuh>
#include <timer.cuh>
#include <utils.cuh>

#include <chrono>
#include <thread>

// ===================================================================================
// CONFIGURATION

#define N 5000 /*1 << 8*/
#define TILE_SIZE 32

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

// NOTE: use this function to visualize the matrices
void print_matrix(const std::vector<int>& matrix) {
    printf("\n");
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            printf("%3d ", matrix[y * N + x]);
        }
        printf("\n");
    }
}

// ===================================================================================
// @@ IMPLEMENTATION

//@@ Write sequential matrix transpose 
std::vector<int> transpose_cpu(const std::vector<int>& input, int n) {
    std::vector<int> output{};
    output.resize(input.size());

    //@@ ...

    return output;
}

//@@ Write the transpose kernel without shared memory
__global__ void transpose_gpu_naive(int* input, int* output, int n) {
    // @@ ...
}

//@@ Write the transpose kernel with shared memory
__global__ void transpose_gpu_shmem(int* input, int* output, int n) {
    // @@ ...
}

// ===================================================================================

int main(int argc, char** argv) {

    // Generates random matrix data
    printf("generating random data...\n");
    auto matrix = random_matrix<int>(N, N, 2000 /* SEED */);
    printf("done\n");

    std::vector<int> result_cpu{};
    result_cpu.resize(matrix.size());

    std::vector<int> result_gpu_naive{};
    result_gpu_naive.resize(matrix.size());

    std::vector<int> result_gpu_shmem{};
    result_gpu_shmem.resize(matrix.size());

    //print_matrix(matrix);

    // Allocate matrix in GPU memory
    int *d_matrix;
    cudaMalloc((void**)&d_matrix, matrix.size() * sizeof(int));
    cudaMemcpy(d_matrix, matrix.data(), matrix.size() * sizeof(int), cudaMemcpyHostToDevice);
    cuda_check_error();

    // Create result matrix
    int *d_result;
    cudaMalloc(&d_result, matrix.size() * sizeof(int));
    cuda_check_error();

    // Setup blocks and threads count
    dim3 threads = dim3(TILE_SIZE, TILE_SIZE);
    dim3 blocks = dim3(
        ((N) + TILE_SIZE - 1) / TILE_SIZE,
        ((N) + TILE_SIZE - 1) / TILE_SIZE
    );

    // ===================================================================================
    // SEQUENTIAL

    auto timer_cpu = timerCPU{};
    timer_cpu.start();

    // Sequential
    result_cpu = transpose_cpu(matrix, N);

    timer_cpu.stop();

    //print_matrix(result_cpu);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ===================================================================================
    // GPU NAIVE

    auto timer_gpu_naive = timerGPU{};
    timer_gpu_naive.start();

    // Kernel launch
    transpose_gpu_naive<<<blocks, threads>>>(d_matrix, d_result, N);

    timer_gpu_naive.stop();

    // Move result matrix to CPU memory
    cudaMemcpy(result_gpu_naive.data(), d_result, result_gpu_naive.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check_error();

    //print_matrix(result_gpu_naive);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ===================================================================================
    // GPU WITH SHARED MEMORY

    auto timer_gpu_shmem = timerGPU{};
    timer_gpu_shmem.start();

    // Kernel launch
    transpose_gpu_shmem<<<blocks, threads>>>(d_matrix, d_result, N);

    timer_gpu_shmem.stop();

    // Move result matrix to CPU memory
    cudaMemcpy(result_gpu_shmem.data(), d_result, result_gpu_shmem.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check_error();

    //print_matrix(result_gpu_shmem);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ===================================================================================
    // TIMERS

    auto cpu_ms = timer_cpu.elapsed_ms();
    auto gpu_naive_ms = timer_gpu_naive.elapsed_ms();
    auto gpu_shmem_ms = timer_gpu_shmem.elapsed_ms();

    printf("transpose CPU\n\t%f ms\n", cpu_ms);

    printf("transpose GPU:NAIVE\n\t%f ms (speedup: cpu %.2fx)\n", 
        gpu_naive_ms, cpu_ms / gpu_naive_ms);

    printf("transpose GPU:SHMEM\n\t%f ms (speedup: cpu %.2fx, gpu_naive %.2fx)\n", 
        gpu_shmem_ms, cpu_ms / gpu_shmem_ms, gpu_naive_ms / gpu_shmem_ms);

    // ===================================================================================
    // CHECK

    bool ok_naive = mat_check_result(result_cpu, result_gpu_naive, N);
    printf("Solution CPU vs NAIVE: %s\n", ok_naive ? "CORRECT" : "INCORRECT");

    bool ok_shmem = mat_check_result(result_cpu, result_gpu_shmem, N);
    printf("Solution CPU vs SHMEM: %s\n", ok_shmem ? "CORRECT" : "INCORRECT");

    // ===================================================================================

    // Free cuda memory
    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}
