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

//@@ Write kernel for vector initialization

int main() {

  const int VALUE = 4;
  const int N = 4097;

  int *vector;

  //@@ Allocate managed memory

  //@@ Initialize the grid and block dimensions

  //@@ Launch kernel to set all elements of vector

  for (int i = 0; i < N; i++) {
    printf("[%d]: %d\n", i, vector[i]);
    if (vector[i] != VALUE) {
      printf(">< INCORRECT: not equal to %d", VALUE);
      return 1;
    }
  }

  //@@ Free memory

  return 0;
}
