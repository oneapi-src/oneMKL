#ifndef _MKL_BLAS_CUBLAS_TASK_HPP_
#define _MKL_BLAS_CUBLAS_TASK_HPP_
#include <cublas_v2.h>
#include <cuda.h>
#include <complex>
#include <CL/sycl.hpp>
#include "oneapi/mkl/types.hpp"
#include "cublas_scope_handle.hpp"
#include <CL/sycl/detail/pi.hpp>

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

template <typename H, typename F>
static inline void host_task_internal(H &cgh, cl::sycl::queue queue, F f) {
    cgh.interop_task([f, queue](cl::sycl::interop_handler ih){
        auto sc = CublasScopedContextHandler(queue, ih);
        f(sc);
    });
}

template <typename H, typename F>
static inline void onemkl_cublas_host_task(H &cgh, cl::sycl::queue queue, F f) {
    (void)host_task_internal(cgh, queue, f);
}

} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
#endif // _MKL_BLAS_CUBLAS_TASK_HPP_
