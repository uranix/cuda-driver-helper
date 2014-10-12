#include "gpu_allocator.h"
#include "cuda_context.h"

#define LOOKUP_SYMBOL(name) GPU_ ## name = lookup(#name);

struct my_cuda_context : cuda_context {

    CUfunction GPU_sum;

    my_cuda_context() {
        load_module("kernel.cubin");
        LOOKUP_SYMBOL(sum);
    }

    void sum(size_t n, const float *a, const float *b, float *c) {
        dim3 grid((n + 255) / 256);
        dim3 block(256);
        launch(GPU_sum, grid, block, { &n, &a, &b, &c });
    }
};

#include <iostream>

int main() {
    try {
        my_cuda_context ctx;
        gpu_allocator<float> gpu;
        std::allocator<float> cpu;

        const int N = 10000;

        float *a = gpu.allocate(N);
        float *b = gpu.allocate(N);
        float *c = gpu.allocate(N);

        float *ha = cpu.allocate(N);
        float *hb = cpu.allocate(N);
        float *hc = cpu.allocate(N);

        for (int i = 0; i < N; i++) {
            ha[i] = i * i;
            hb[i] = i;
        }

        cuMemcpyHtoD((CUdeviceptr)a, ha, N * sizeof(float));
        cuMemcpyHtoD((CUdeviceptr)b, hb, N * sizeof(float));

        ctx.sum(N, a, b, c);

        cuMemcpyDtoH(hc, (CUdeviceptr)c, N * sizeof(float));

        for (int i = 0; i < 10; i++)
            std::cout << ha[i] << " + " << hb[i] << " = " << hc[i] << std::endl;
        std::cout << "..." << std::endl;
        for (int i = N - 10; i < N; i++)
            std::cout << ha[i] << " + " << hb[i] << " = " << hc[i] << std::endl;

        gpu.deallocate(a);
        gpu.deallocate(b);
        gpu.deallocate(c);

        cpu.deallocate(ha, 0);
        cpu.deallocate(hb, 0);
        cpu.deallocate(hc, 0);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Exception occured: " << e.what() << std::endl;
        return 1;
    }
}
