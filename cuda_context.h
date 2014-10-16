#ifndef __CUDA_CONTEXT_H__
#define __CUDA_CONTEXT_H__

#include <cuda.h>
#include "cuda_helper.h"

#include <string>
#include <map>

namespace cuda_helper {

struct dim3 {
    unsigned int x, y, z;
    dim3 (unsigned int x, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) { }
};

class cuda_context {
    CUcontext ctx;
    CUdevice dev;

    std::map<std::string, CUmodule> modules;

protected:
    CUfunction lookup(const char *name, CUmodule hint = 0) const {
        if (hint) {
            CUfunction f;
            CUresult res = cuModuleGetFunction(&f, hint, name);
            if (res == CUDA_SUCCESS)
                return f;
            if (res != CUDA_ERROR_NOT_FOUND)
                throw cuda_error(std::string("Loading function `") + name + "' from given module failed", res);
        }
        for (auto m : modules) {
            CUfunction f;
            CUresult res = cuModuleGetFunction(&f, m.second, name);
            if (res == CUDA_SUCCESS)
                return f;
            if (res != CUDA_ERROR_NOT_FOUND)
                throw cuda_error(std::string("Loading function `") + name + "' from module `" + m.first + "' failed", res);
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
        CUresult err = cuModuleLoad(&mod, file);
        if (err != CUDA_SUCCESS)
            throw cuda_error(std::string("Loading module `") + file + "' failed", err);
        modules.push_back(std::pair<std::string, CUmodule>(file, mod));
        return mod;
    }

    void memcpy_DtoH(void *dst, const void *src, size_t bytes) {
        CUDA_CHECK(cuMemcpyDtoH(dst, (CUdeviceptr)src, bytes));
    }

    void memcpy_HtoD(void *dst, const void *src, size_t bytes) {
        CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr)dst, src, bytes));
    }

    ~cuda_context() {
        /* No CUDA_CHECK's due to possible exceptions being thrown */
        for (auto m : modules)
            cuModuleUnload(m);
        cuCtxDestroy(ctx);
    }
};

class configured_call {
    CUfunction f;
    dim3 grid;
    dim3 block;
    unsigned int shmem;
    CUstream stream;
public:
    configured_call(CUfunction f, dim3 grid, dim3 block, unsigned int shmem, CUstream stream)
        : f(f), grid(grid), block(block), shmem(shmem), stream(stream)
    { }

    class param_holder {
        void *_data[64]; /* Lives on stack, not heap */
    public:
        param_holder(std::initializer_list<void *> init) {
            if (init.size() > 64)
                throw std::out_of_range("Number of parameters is too big for a kernel function");
            std::copy(init.begin(), init.end(), _data);
        }
        void **data() {
            return _data;
        }
    };

    void operator()(param_holder params) const {
        CUDA_CHECK(cuLaunchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z, shmem, stream, params.data(), 0));
    }
};

}

#ifndef CUDA_HELPER_OUTERCLASS
#define CUDA_HELPER_OUTERCLASS(className, memberName) \
    reinterpret_cast<const className*>(reinterpret_cast<const unsigned char*>(this) - offsetof(className, memberName))
#endif

#define DECLARE_KERNEL(name, classname, member) class __kernel_ ## name { mutable CUfunction __f; const char *__name; \
    public: __kernel_ ## name() : __f(0), __name(#name) { } \
    cuda_helper::configured_call operator()(cuda_helper::dim3 grid, cuda_helper::dim3 block, unsigned int shmem = 0, CUstream stream = 0) const { \
        if (!__f) { __f = CUDA_HELPER_OUTERCLASS(classname, member)->lookup(__name); } \
        return cuda_helper::configured_call(__f, grid, block, shmem, stream); \
    } } member

#endif
