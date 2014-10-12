#ifndef __CUDA_CONTEXT_H__
#define __CUDA_CONTEXT_H__

#include <cuda.h>
#include "cuda_helper.h"

#include <vector>

class cuda_context {
    CUcontext ctx;
    CUdevice dev;

    std::vector<CUmodule> modules;

protected:
    struct dim3 {
        unsigned int x, y, z;
        dim3 (unsigned int x, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) { }
    };

    struct param_holder {
        void *_data[64]; /* Lives on stack, not heap */
        param_holder(std::initializer_list<void *> init) {
            if (init.size() > 64)
                throw std::out_of_range("Number of parameters is too big for a kernel function");
            std::copy(init.begin(), init.end(), _data);
        }
        void **data() {
            return &_data[0];
        }
    };

    void launch(CUfunction f, dim3 grid, dim3 block, param_holder params, unsigned int dynShmem = 0, CUstream stream = 0) {
        CUDA_CHECK(cuLaunchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z, dynShmem, stream, params.data(), 0));
    }

    CUfunction lookup(const char *name, CUmodule hint = 0) {
        if (hint) {
            CUfunction f;
            CUresult res = cuModuleGetFunction(&f, hint, name);
            if (res == CUDA_SUCCESS)
                return f;
            if (res != CUDA_ERROR_NOT_FOUND)
                CUDA_CHECK(res);
        }
        for (auto m : modules) {
            CUfunction f;
            CUresult res = cuModuleGetFunction(&f, m, name);
            if (res == CUDA_SUCCESS)
                return f;
            if (res != CUDA_ERROR_NOT_FOUND)
                CUDA_CHECK(res);
        }
        throw std::runtime_error(std::string("No module with function `") + name + "' was loaded");
    }
public:
    cuda_context(const int devid = 0, unsigned int flags = CU_CTX_SCHED_AUTO, bool performInit = true) {
        if (performInit)
            CUDA_CHECK(cuInit(0));
        CUDA_CHECK(cuDeviceGet(&dev, devid));
        CUDA_CHECK(cuCtxCreate(&ctx, flags, dev));
    }

    CUmodule load_module(const char *file) {
        CUmodule mod;
        CUDA_CHECK(cuModuleLoad(&mod, file));
        modules.push_back(mod);
        return mod;
    }

    ~cuda_context() {
        /* No CUDA_CHECK's due to possible exceptions thrown */
        for (auto m : modules)
            cuModuleUnload(m);
        cuCtxDestroy(ctx);
    }
};

#endif
