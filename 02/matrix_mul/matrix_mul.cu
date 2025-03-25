
#define PPROG_IMPLEMENTATION
#include <random.cuh>
#include <timer.cuh>
#include <utils.cuh>

#define N 1024
#define BLOCK_SIZE 32
#define MEMORY_TRANSFER_TIMER 0

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
void print_matrix(const std::vector<float>& matrix) {
    printf("\n");
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            printf("%5.2f ", matrix[y * N + x]);
        }
        printf("\n");
    }
}

//@@ Write the matmul kernel
__global__ void matmul_gpu(float* A, float* B, float* O, int n) {
    // ...
}

//@@ Implement the CPU version of matmul
std::vector<float> matmul_cpu(const std::vector<float>& A, const std::vector<float>& B) {
    // ...
}

int main(int argc, char** argv) {

    // Generates random matrix data
    printf("generating random data...\n");
    auto A = random_square_matrix<float>(N, 1999);
    auto B = random_square_matrix<float>(N, 2000);
    printf("done\n");

    std::vector<float> result_gpu;
    result_gpu.resize(A.size());

    //print_matrix(A);
    //print_matrix(B);

    //@@ Setup blocks and threads count
    // ...


    //@@ Allocate all necessary matrices in GPU memory
    float *d_A, *d_B, *d_O;
    
    // ...

    auto timer_gpu = timerGPU{};

#if MEMORY_TRANSFER_TIMER == 1
    timer_gpu.start();
#endif

    //@@ Copy data from CPU to GPU
    // ...

#if MEMORY_TRANSFER_TIMER == 0
    timer_gpu.start();
#endif

    //@@ Invoke kernel
    // ...
    
#if MEMORY_TRANSFER_TIMER == 0
    timer_gpu.stop();
#endif

    //@@ Move result matrix to CPU memory
    // ...

#if MEMORY_TRANSFER_TIMER == 1
    timer_gpu.stop();
#endif

    //print_matrix(result);

    //@@ Free cuda memory
    // ...

    auto timer_cpu = timerCPU{};
    timer_cpu.start();
    auto solution_cpu = matmul_cpu(A, B);
    timer_cpu.stop();

    auto gpu_ms = timer_gpu.elapsed_ms();
    auto cpu_ms = timer_cpu.elapsed_ms();

    printf("GPU matmul elapsed time: %f ms\n", gpu_ms);
    printf("CPU matmul elapsed time: %f ms\n", cpu_ms);    
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);

    // Check solution
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            auto rval = result_gpu[y * N + x];
            auto cval = solution_cpu[y * N + x];
            if (!float_approx_equal(rval, cval, 1e-6)) {
                printf("invalid element at (%d, %d):\t: gpu = %f\tcpu = %f\n",
                    y, x, rval, cval);
                exit(1);
            }
        }
    }

    printf("Solution is similar!\n");
    return 0;
}
