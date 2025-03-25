
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define PPROG_TIMER
#include <timer.cuh>

#define BLOCK_SIDE_LEN 32

//@@ Write the grayscale kernel
__global__ void grayscale_gpu(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    //...
}

//@@ Write the sequential version of grayscale
void grayscale_cpu(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    //...
}

int main(int argc, char** argv) {

    // Load image data from file
    int w, h, c;
    unsigned char *h_img = stbi_load("image.png", &w, &h, &c, 0);
    if(h_img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }

    const auto img_size = sizeof(unsigned char) * w * h * c;

    unsigned char* result_cpu = new unsigned char[img_size];
    
    auto timer_cpu = timerCPU{};
    timer_cpu.start();
    grayscale_cpu(h_img, result_cpu, w, h, c);
    timer_cpu.stop();

    if (!stbi_write_png("result_cpu.png", w, h, c, result_cpu, w * c)) {
        printf("Error in saving the image\n");
        exit(1);
    }
    
    delete result_cpu;

    unsigned char *d_img, *d_output;

    //@@ Allocate GPU memory
    //...
    
    //@@ Transfer memory from CPU to GPU
    //...

    //@@ Setup blocks and threads count
    //...

    auto timer_gpu = timerGPU{};
    timer_gpu.start();

    //@@ Invoke kernel
    //...

    timer_gpu.stop();

    //@@ Move result image to CPU memory
    //...

    // Save image to disk
    if (!stbi_write_png("result_gpu.png", w, h, c, h_img, w * c)) {
        printf("Error in saving the image\n");
        exit(1);
    }

    auto gpu_ms = timer_gpu.elapsed_ms();
    auto cpu_ms = timer_cpu.elapsed_ms();

    printf("GPU matmul elapsed time: %f ms\n", gpu_ms);
    printf("CPU matmul elapsed time: %f ms\n", cpu_ms);    
    printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);

    //@@ Free cuda memory
    //...

    stbi_image_free(h_img);
    return 0;
}
