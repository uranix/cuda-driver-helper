#include "gpu_allocator.h"
#include "cuda_context.h"

namespace gpu = cuda_helper;

class my_cuda_context : public gpu::cuda_context {
    DECLARE_KERNEL(sum, my_cuda_context, GPU_sum);
    DECLARE_KERNEL(scal, my_cuda_context, GPU_scale);

public:
    my_cuda_context() {
        load_module("kernels.ptx");
    }

    void sum(size_t n, const float *a, const float *b, float *c) {
        gpu::dim3 grid((n + 255) / 256);
        gpu::dim3 block(256);
        GPU_sum(grid, block)({ &n, &a, &b, &c });
    }

    void scale(size_t n, float *a, float scale) {
        gpu::dim3 grid((n + 255) / 256);
        gpu::dim3 block(256);
        GPU_scale(grid, block)({ &n, &a, &scale });
    }
};

#include <iostream>

int main() {
    try {
        my_cuda_context ctx;
        gpu::allocator<float> gpu;
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

        ctx.memcpy_HtoD(a, ha, N * sizeof(float));
        ctx.memcpy_HtoD(b, hb, N * sizeof(float));

        ctx.sum(N, a, b, c);
        ctx.scale(N, c, .5);

        ctx.memcpy_DtoH(hc, c, N * sizeof(float));

        for (int i = 0; i < 10; i++)
            std::cout << "(" << ha[i] << " + " << hb[i] << ") * 0.5 = " << hc[i] << std::endl;
        std::cout << "..." << std::endl;
        for (int i = N - 10; i < N; i++)
            std::cout << "(" << ha[i] << " + " << hb[i] << ") * 0.5 = " << hc[i] << std::endl;

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
