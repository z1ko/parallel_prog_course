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

// How many elements?
#define N (1 << 24) // ~16 million elements
// The size of each segment of data
#define CHUNK_SIZE (1 << 22) // ~4 million elements

// How many streams do we have?
#define CUDA_STREAMS_COUNT 4

// The size of a block
#define BLOCK_SIZE 1024
// The size of the grid
#define GRID_SIZE ((CHUNK_SIZE  + BLOCK_SIZE - 1) / BLOCK_SIZE)

// ===================================================================================
// UTILITIES

#define cuda_check_error()                                     \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess)                                      \
    {                                                          \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      std::exit(EXIT_FAILURE);                                 \
    }                                                          \
  }

// ===================================================================================
// @@ IMPLEMENTATION

__global__ void vecAdd(int *in1, int *in2, int *out, int len)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < len)
    out[i] = in1[i] + in2[i];
}
// ===================================================================================

int main(int argc, char **argv)
{
  std::vector<int> A = pprog::rand::random_array<int>(0, 100, N, 2048);
  std::vector<int> B = pprog::rand::random_array<int>(0, 100, N, 6900);
  std::vector<int> O = std::vector<int>(N);

  // ===================================================================================

  cudaStream_t streams[CUDA_STREAMS_COUNT];

  int *dA[CUDA_STREAMS_COUNT];
  int *dB[CUDA_STREAMS_COUNT];
  int *dO[CUDA_STREAMS_COUNT];

  //@@ Create all streams

  //@@ Allocate all memory segments

  //@@ Stream operations

  //@@ Destroy all streams

  //@@ Free al memory

  // ===================================================================================
  // Check results

  bool valid = true;
  for (int i = 0; i < N; ++i)
  {
    if (A[i] + B[i] != O[i]) {
      printf("[%5i] %3i %3i | %3i | %3i\n", i, A[i], B[i], O[i], A[i] + B[i]);
      valid = false;
      break;
    }
  }

  printf("\nCORRECT? %s\n", valid ? "YES" : "NO");
  return 0;
}
