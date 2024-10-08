/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#ifndef _ONEMATH_SRC_SPARSE_BLAS_BACKENDS_MKL_COMMON_MKL_HELPER_HPP_
#define _ONEMATH_SRC_SPARSE_BLAS_BACKENDS_MKL_COMMON_MKL_HELPER_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/exceptions.hpp"
#include "oneapi/math/sparse_blas/detail/helper_types.hpp"

#include "sparse_blas/enum_data_types.hpp"
#include "sparse_blas/macros.hpp"

namespace oneapi::mkl::sparse::detail {

/// Return whether a pointer is accessible on the host
template <typename T>
inline bool is_ptr_accessible_on_host(sycl::queue &queue, const T *host_or_device_ptr) {
    auto alloc_type = sycl::get_pointer_type(host_or_device_ptr, queue.get_context());
    return alloc_type == sycl::usm::alloc::host || alloc_type == sycl::usm::alloc::shared ||
           alloc_type == sycl::usm::alloc::unknown;
}

/// Throw an exception if the scalar is not accessible in the host
inline void check_ptr_is_host_accessible(const std::string &function_name,
                                         const std::string &scalar_name,
                                         bool is_ptr_accessible_on_host) {
    if (!is_ptr_accessible_on_host) {
        throw mkl::invalid_argument(
            "sparse_blas", function_name,
            "Scalar " + scalar_name + " must be accessible on the host for buffer functions.");
    }
}

/// Return a scalar on the host from a pointer to host or device memory
/// Used for USM functions
template <typename T>
inline T get_scalar_on_host(sycl::queue &queue, const T *host_or_device_ptr,
                            bool is_ptr_accessible_on_host) {
    if (is_ptr_accessible_on_host) {
        return *host_or_device_ptr;
    }
    T scalar;
    auto event = queue.copy(host_or_device_ptr, &scalar, 1);
    event.wait_and_throw();
    return scalar;
}

/// Merge multiple event dependencies into one
inline sycl::event collapse_dependencies(sycl::queue &queue,
                                         const std::vector<sycl::event> &dependencies) {
    if (dependencies.empty()) {
        return {};
    }
    else if (dependencies.size() == 1) {
        return dependencies[0];
    }

    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.host_task([=]() {});
    });
}

/// Convert \p value_type to template type argument and use it to call \p op_functor.
#define DISPATCH_MKL_OPERATION(function_name, value_type, op_functor, ...)                         \
    switch (value_type) {                                                                          \
        case detail::data_type::real_fp32: return op_functor<float>(__VA_ARGS__);                  \
        case detail::data_type::real_fp64: return op_functor<double>(__VA_ARGS__);                 \
        case detail::data_type::complex_fp32: return op_functor<std::complex<float>>(__VA_ARGS__); \
        case detail::data_type::complex_fp64:                                                      \
            return op_functor<std::complex<double>>(__VA_ARGS__);                                  \
        default:                                                                                   \
            throw oneapi::mkl::exception(                                                          \
                "sparse_blas", function_name,                                                      \
                "Internal error: unsupported type " + data_type_to_str(value_type));               \
    }

#define CHECK_DESCR_MATCH(descr, argument, optimize_func_name)                                    \
    do {                                                                                          \
        if (descr->last_optimized_##argument != argument) {                                       \
            throw mkl::invalid_argument(                                                          \
                "sparse_blas", __func__,                                                          \
                #argument " argument must match with the previous call to " #optimize_func_name); \
        }                                                                                         \
    } while (0)

} // namespace oneapi::mkl::sparse::detail

#endif // _ONEMATH_SRC_SPARSE_BLAS_BACKENDS_MKL_COMMON_MKL_HELPER_HPP_
