#define PPROG_IMPLEMENTATION

#include <random.cuh>
#include <timer.cuh>
#include <utils.cuh>

// ===================================================================================
// CONFIGURATION

#define N 1024
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
            printf("%6d ", matrix[y * N + x]);
        }
        printf("\n");
    }
}

// ===================================================================================
// @@ IMPLEMENTATION

//@@ Implement the naive matrix multiplication
__global__ void matmul_gpu_naive(int *A, int* B, int *O, int n) {
    // @@ ...
}

//@@ Implement the tiled matrix multiplication using the shared memory
__global__ void matmul_gpu_shmem(int *A, int* B, int *O, int n) {
    // @@ ...
}

// ===================================================================================

int main(int argc, char** argv) {

    // Generates random matrix data
    printf("generating random data...\n");
    auto A = random_matrix<int>(N, N, 1999 /* RNG SEED */);
    auto B = random_matrix<int>(N, N, 2000 /* RNG SEED */);
    printf("done\n");

    //print_matrix(A);
    //print_matrix(B);

    std::vector<int> result_cpu{};
    result_cpu.resize(A.size());

    std::vector<int> result_gpu_shmem{};
    result_gpu_shmem.resize(A.size());

    std::vector<int> result_gpu_naive{};
    result_gpu_naive.resize(A.size());

    // Setup blocks and threads count
    dim3 threads = dim3(TILE_SIZE, TILE_SIZE);
    dim3 blocks = dim3(
        ((N) + TILE_SIZE - 1) / TILE_SIZE,
        ((N) + TILE_SIZE - 1) / TILE_SIZE
    );

    // Allocate all necessary matrices in GPU memory
    int *d_A, *d_B, *d_O;
    cudaMalloc((void**)&d_A, A.size() * sizeof(int));
    cudaMalloc((void**)&d_B, A.size() * sizeof(int));
    cudaMalloc((void**)&d_O, A.size() * sizeof(int));
    cuda_check_error();

    // Copy data from CPU to GPU
    cudaMemcpy(d_A, A.data(), A.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(int), cudaMemcpyHostToDevice);
    cuda_check_error();

    // ================================================================================================
    // CPU

    auto timer_cpu = timerCPU{};
    timer_cpu.start();

    result_cpu = matmul_cpu<int>(A, B, N, N, N);
    
    timer_cpu.stop();
    
    //print_matrix(result_cpu);

    // ================================================================================================
    // GPU NAIVE

    auto timer_gpu_naive = timerGPU{};
    timer_gpu_naive.start();

    matmul_gpu_naive<<<blocks, threads>>>(d_A, d_B, d_O, N);

    timer_gpu_naive.stop();
    cuda_check_error();

    // Move result matrix to CPU memory
    cudaMemcpy(result_gpu_naive.data(), d_O, result_gpu_naive.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check_error();

    //print_matrix(result_gpu_naive);

    // ================================================================================================
    // GPU SHARED MEMORY

    auto timer_gpu_shmem = timerGPU{};
    timer_gpu_shmem.start();

    // Invoke kernel
    matmul_gpu_shmem<<<blocks, threads>>>(d_A, d_B, d_O, N);
    
    timer_gpu_shmem.stop();

    // Move result matrix to CPU memory
    cudaMemcpy(result_gpu_shmem.data(), d_O, result_gpu_shmem.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check_error();

    //print_matrix(result_gpu_shmem);

    // ================================================================================================
    // TIMERS

    auto cpu_ms = timer_cpu.elapsed_ms();
    auto gpu_naive_ms = timer_gpu_naive.elapsed_ms();
    auto gpu_shmem_ms = timer_gpu_shmem.elapsed_ms();

    printf("transpose CPU\n\t%f ms\n", cpu_ms);

    printf("transpose GPU:NAIVE\n\t%f ms (speedup: cpu %.2fx)\n", 
        gpu_naive_ms, cpu_ms / gpu_naive_ms);

    printf("transpose GPU:SHMEM\n\t%f ms (speedup: cpu %.2fx, gpu_naive %.2fx)\n", 
        gpu_shmem_ms, cpu_ms / gpu_shmem_ms, gpu_naive_ms / gpu_shmem_ms);

    // ================================================================================================
    // CHECK

    bool ok_naive = mat_check_result(result_cpu, result_gpu_naive, N);
    printf("Solution CPU vs NAIVE: %s\n", ok_naive ? "CORRECT" : "INCORRECT");

    bool ok_shmem = mat_check_result(result_cpu, result_gpu_shmem, N);
    printf("Solution CPU vs SHMEM: %s\n", ok_shmem ? "CORRECT" : "INCORRECT");

    // Free cuda memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_O);

    return 0;
}
