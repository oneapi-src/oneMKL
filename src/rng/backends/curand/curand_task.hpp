#ifndef _MKL_RNG_CURAND_TASK_HPP_
#define _MKL_RNG_CURAND_TASK_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

namespace oneapi {
namespace mkl {
namespace rng {
namespace curand {
#ifdef __HIPSYCL__
template <typename H, typename A, typename F>
static inline void host_task_internal(H &cgh, A acc, F f) {
    cgh.hipSYCL_enqueue_custom_operation([f, acc](sycl::interop_handle ih) {
        auto r_ptr =
            reinterpret_cast<typename A::value_type *>(ih.get_native_mem<sycl::backend::cuda>(acc));
        f(r_ptr);
    });
}

template <typename H, typename F>
static inline void host_task_internal(H &cgh, F f) {
    cgh.hipSYCL_enqueue_custom_operation([f](sycl::interop_handle ih) { f(ih); });
}
#else
template <typename H, typename A, typename F>
static inline void host_task_internal(H &cgh, A acc, F f) {
    cgh.host_task([f, acc](sycl::interop_handle ih) {
        auto r_ptr = reinterpret_cast<typename A::value_type *>(
            ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(acc));
        f(r_ptr);
    });
}

template <typename H, typename F>
static inline void host_task_internal(H &cgh, F f) {
    cgh.host_task([f](sycl::interop_handle ih) { f(ih); });
}
#endif
template <typename H, typename A, typename F>
static inline void onemkl_curand_host_task(H &cgh, A acc, F f) {
    host_task_internal(cgh, acc, f);
}

template <typename H, typename F>
static inline void onemkl_curand_host_task(H &cgh, F f) {
    host_task_internal(cgh, f);
}

} // namespace curand
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif
