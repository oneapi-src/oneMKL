#ifndef _MKL_BLAS_CUBLAS_TASK_HPP_
#define _MKL_BLAS_CUBLAS_TASK_HPP_
#include <cublas_v2.h>
#include <cuda.h>
#include <complex>
#include <CL/sycl.hpp>
#include "oneapi/mkl/types.hpp"
#ifndef __HIPSYCL__
#include "cublas_scope_handle.hpp"
#include <CL/sycl/detail/pi.hpp>
#else
#include "cublas_scope_handle_hipsycl.hpp"
namespace sycl {
using interop_handler = sycl::interop_handle;
}
#endif
namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

#ifdef __HIPSYCL__
template <typename H, typename F>
static inline void host_task_internal(H &cgh, sycl::queue queue, F f) {
    cgh.hipSYCL_enqueue_custom_operation([f, queue](sycl::interop_handle ih) {
        auto sc = CublasScopedContextHandler(queue, ih);
        f(sc);
    });
}
#else
template <typename H, typename F>
static inline void host_task_internal(H &cgh, sycl::queue queue, F f) {
    cgh.interop_task([f, queue](sycl::interop_handler ih) {
        auto sc = CublasScopedContextHandler(queue, ih);
        f(sc);
    });
}
#endif
template <typename H, typename F>
static inline void onemkl_cublas_host_task(H &cgh, sycl::queue queue, F f) {
    (void)host_task_internal(cgh, queue, f);
}

} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
#endif // _MKL_BLAS_CUBLAS_TASK_HPP_
