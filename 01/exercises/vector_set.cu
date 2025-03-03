#include <cstdio>

#define cuda_check_error()                                                     \
  {                                                                            \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

__global__ void kernel_vector_set(int *v, int value, int N) {
  auto gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid < N) {
    v[gid] = value;
  }
}

int main() {

  const int VALUE = 4;
  const int N = 4097;

  int *vector;

  //@@ Allocate managed memory

  cudaMallocManaged(&vector, N * sizeof(int));

  //@@ Initialize the grid and block dimensions

  const int threads = 128;
  const int blocks = (N + threads - 1) / threads;

  //@@ Launch kernel to set all elements of vector

  kernel_vector_set<<<blocks, threads>>>(vector, VALUE, N);
  cudaDeviceSynchronize();

  for (int i = 0; i < N; i++) {
    printf("[%d]: %d\n", i, vector[i]);
    if (vector[i] != VALUE) {
      printf(">< INCORRECT: not equal to %d", VALUE);
      return 1;
    }
  }

  //@@ Free memory

  cudaFree(vector);

  return 0;
}
