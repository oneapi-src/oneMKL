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
static inline void host_task_internal(sycl::handler &cgh, A acc, sycl::queue, F f) {
    cgh.hipSYCL_enqueue_custom_operation([f, acc](sycl::interop_handle ih) {
        auto r_ptr =
            reinterpret_cast<typename A::value_type *>(ih.get_native_mem<sycl::backend::cuda>(acc));
        f(r_ptr);
    });
}
#else
template <typename H, typename A, typename F, typename NumberTyp>
static inline void host_task_internal(H &cgh, A acc, F f) {
    cgh.host_task([f, acc](sycl::interop_handle ih) {
        auto r_ptr = reinterpret_cast<typename A::value_type *>(
            ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(acc));
        f(r_ptr);
    });
}
#endif
template <typename H, typename A, typename F>
static inline void onemkl_curand_host_task(H &cgh, A acc, F f) {
    host_task_internal(cgh, acc, f);
}
} // namespace curand
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif
