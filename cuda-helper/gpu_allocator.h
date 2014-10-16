#ifndef __GPU_ALLOCATOR_H__
#define __GPU_ALLOCATOR_H__

#include <cstddef>
#include <new>
#include <stdexcept>

#include <cuda.h>
#include "cuda_helper.h"

namespace cuda_helper {

template <typename T> class allocator {
    CUdevice device;
    void check_current_device(const char *msg = "CUDA device has changed since allocator construction") const {
        CUdevice _device;
        CUDA_CHECK(cuCtxGetDevice(&_device));
        if (device != _device)
            throw std::logic_error(msg);
    }
public:
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T &reference;
    typedef const T &const_reference;
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    allocator() {
        cuCtxGetDevice(&device);
    }

    allocator(const allocator &other) : device(other.device) {
        check_current_device("Cannot copy-construct an allocator for different device");
    }

    template <typename U>
    allocator(const allocator<U> &other) : device(other.device) {
        check_current_device("Cannot copy-construct an allocator for different device");
    }

    ~allocator() { }

    pointer address(reference r) const { return &r; }
    const_pointer address(const_reference r) const { return &r; }
    size_type max_size() const {
        check_current_device();
        size_t total_mem;
        CUDA_CHECK(cuDeviceTotalMem(&total_mem, device));

        return total_mem / sizeof(T);
    }
    template <typename U> struct rebind {
        typedef allocator<U> other;
    };
    bool operator!=(const allocator &other) const { return !(*this == other); }
    bool operator==(const allocator &other) const { return device == other.device; }
    T *allocate(const size_type n) const {
        if (n == 0)
#if __cplusplus < 201103L
            return NULL;
#else
            return nullptr;
#endif

        if (n > max_size())
            throw std::length_error("The requested allocation size is too big");

        CUdeviceptr p;
        /* TODO: throw away asynchronous errors */
        CUresult err = cuMemAlloc(&p, n * sizeof(value_type));
        if (err != CUDA_SUCCESS)
            throw std::bad_alloc();

        return reinterpret_cast<pointer>(p);
    }
    void deallocate(pointer p, size_type /* unnamed */ = 0) const {
        CUdeviceptr pv = reinterpret_cast<CUdeviceptr>(p);
        CUDA_CHECK(cuMemFree(pv));
    }
    template <typename U>
    pointer allocate(const size_t n, const U) const {
        return allocate(n);
    }

#if __cplusplus < 201103L
private:
    allocator &operator=(const allocator &);
public:
#else
    allocator &operator=(const allocator &) = delete;
#endif

    void construct(pointer, const reference) const {
        throw std::logic_error("Cannot construct an object from GPU memory on host");
    }

    void destroy(pointer) const {
        throw std::logic_error("Cannot destroy an object from GPU memory on host");
    }
};

}

#endif
