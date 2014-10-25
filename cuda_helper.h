#ifndef __CUDA_HELPER_H__
#define __CUDA_HELPER_H__

#include <cuda.h>
#include <stdexcept>

namespace cuda_helper {

const char *cuErrorString(CUresult err) {
    /* Valid for CUDA 5.5 */
#define CASE(x) case x: return #x
    switch (err) {
        CASE(CUDA_SUCCESS                             );
        CASE(CUDA_ERROR_INVALID_VALUE                 );
        CASE(CUDA_ERROR_OUT_OF_MEMORY                 );
        CASE(CUDA_ERROR_NOT_INITIALIZED               );
        CASE(CUDA_ERROR_DEINITIALIZED                 );
        CASE(CUDA_ERROR_PROFILER_DISABLED             );
        CASE(CUDA_ERROR_PROFILER_NOT_INITIALIZED      );
        CASE(CUDA_ERROR_PROFILER_ALREADY_STARTED      );
        CASE(CUDA_ERROR_PROFILER_ALREADY_STOPPED      );
        CASE(CUDA_ERROR_NO_DEVICE                     );
        CASE(CUDA_ERROR_INVALID_DEVICE                );
        CASE(CUDA_ERROR_INVALID_IMAGE                 );
        CASE(CUDA_ERROR_INVALID_CONTEXT               );
        CASE(CUDA_ERROR_CONTEXT_ALREADY_CURRENT       );
        CASE(CUDA_ERROR_MAP_FAILED                    );
        CASE(CUDA_ERROR_UNMAP_FAILED                  );
        CASE(CUDA_ERROR_ARRAY_IS_MAPPED               );
        CASE(CUDA_ERROR_ALREADY_MAPPED                );
        CASE(CUDA_ERROR_NO_BINARY_FOR_GPU             );
        CASE(CUDA_ERROR_ALREADY_ACQUIRED              );
        CASE(CUDA_ERROR_NOT_MAPPED                    );
        CASE(CUDA_ERROR_NOT_MAPPED_AS_ARRAY           );
        CASE(CUDA_ERROR_NOT_MAPPED_AS_POINTER         );
        CASE(CUDA_ERROR_ECC_UNCORRECTABLE             );
        CASE(CUDA_ERROR_UNSUPPORTED_LIMIT             );
        CASE(CUDA_ERROR_CONTEXT_ALREADY_IN_USE        );
        CASE(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED       );
        CASE(CUDA_ERROR_INVALID_SOURCE                );
        CASE(CUDA_ERROR_FILE_NOT_FOUND                );
        CASE(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND);
        CASE(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED     );
        CASE(CUDA_ERROR_OPERATING_SYSTEM              );
        CASE(CUDA_ERROR_INVALID_HANDLE                );
        CASE(CUDA_ERROR_NOT_FOUND                     );
        CASE(CUDA_ERROR_NOT_READY                     );
        CASE(CUDA_ERROR_LAUNCH_FAILED                 );
        CASE(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES       );
        CASE(CUDA_ERROR_LAUNCH_TIMEOUT                );
        CASE(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING );
        CASE(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED   );
        CASE(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED       );
        CASE(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE        );
        CASE(CUDA_ERROR_CONTEXT_IS_DESTROYED          );
        CASE(CUDA_ERROR_ASSERT                        );
        CASE(CUDA_ERROR_TOO_MANY_PEERS                );
        CASE(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED);
        CASE(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED    );
        CASE(CUDA_ERROR_NOT_PERMITTED                 );
        CASE(CUDA_ERROR_NOT_SUPPORTED                 );
        CASE(CUDA_ERROR_UNKNOWN                       );
        default: return "<unknown result code>";
    };
#undef CASE
}

class cuda_error : public std::runtime_error {
public:
    cuda_error(const std::string &msg, CUresult err) : std::runtime_error(msg + ". CUresult = " + cuErrorString(err)) { }
};

}

#define CUDA_HELPER_STRINGIZE_DETAIL(x) #x
#define CUDA_HELPER_STRINGIZE(x) CUDA_HELPER_STRINGIZE_DETAIL(x)

#define CUDA_CHECK(x) do { \
    CUresult __err = (x); \
    if (__err != CUDA_SUCCESS) \
        throw cuda_helper::cuda_error(std::string("Call `") + #x + "' on line " + CUDA_HELPER_STRINGIZE(__LINE__) + " in file " + __FILE__ + " failed", __err); \
} while (false)

#endif
