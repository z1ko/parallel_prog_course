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
#define N (1 << 28) // ~256 million floats (~1 GB)

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

// ===================================================================================

void run_transfer_test(bool pinned)
{
  float *h_data = nullptr;
  float *d_data = nullptr;

  // Allocate memory on GPU
  cudaMalloc((void **)&d_data, N * sizeof(float));

  if (pinned)
  {
    //@@ Allocate pinned memory
  }
  else
  {
    // Normal allocation
    h_data = new float[N];
  }

  // Host to device
  timerGPU outbound_timer{};

  outbound_timer.start();
  {
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cuda_check_error();
  }
  outbound_timer.stop();
  printf("outbound: %.3f ms\n", outbound_timer.elapsed_ms());

  // Device to host
  timerGPU inbound_timer{};
  inbound_timer.start();
  {
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check_error();
  }
  inbound_timer.stop();
  printf(" inbound: %.3f ms\n", inbound_timer.elapsed_ms());

  if (pinned)
  {
    //@@ Free pinned memory
  }
  else
  {
    delete[] h_data;
  }

  cudaFree(d_data);
  cuda_check_error();
}

int main(int argc, char **argv)
{
  printf("========== Pageable Memory Test ==========\n");
  run_transfer_test(false);

  printf("========== Pinned Memory Test ==========\n");
  run_transfer_test(true);

  return 0;
}
