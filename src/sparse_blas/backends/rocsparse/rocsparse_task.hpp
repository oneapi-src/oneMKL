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

#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_TASKS_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_TASKS_HPP_

#include "rocsparse_handles.hpp"
#include "rocsparse_scope_handle.hpp"

/// This file provide a helper function to submit host_task using buffers or USM seamlessly

namespace oneapi::mkl::sparse::rocsparse {

template <typename T, typename Container>
auto get_value_accessor(sycl::handler &cgh, Container container) {
    auto buffer_ptr =
        reinterpret_cast<sycl::buffer<T, 1> *>(container->value_container.buffer_ptr.get());
    return buffer_ptr->template get_access<sycl::access::mode::read_write>(cgh);
}

template <typename T, typename... Ts>
auto get_fp_accessors(sycl::handler &cgh, Ts... containers) {
    return std::array<sycl::accessor<T, 1>, sizeof...(containers)>{ get_value_accessor<T>(
        cgh, containers)... };
}

template <typename T>
auto get_row_accessor(sycl::handler &cgh, matrix_handle_t smhandle) {
    auto buffer_ptr =
        reinterpret_cast<sycl::buffer<T, 1> *>(smhandle->row_container.buffer_ptr.get());
    return buffer_ptr->template get_access<sycl::access::mode::read_write>(cgh);
}

template <typename T>
auto get_col_accessor(sycl::handler &cgh, matrix_handle_t smhandle) {
    auto buffer_ptr =
        reinterpret_cast<sycl::buffer<T, 1> *>(smhandle->col_container.buffer_ptr.get());
    return buffer_ptr->template get_access<sycl::access::mode::read_write>(cgh);
}

template <typename T>
auto get_int_accessors(sycl::handler &cgh, matrix_handle_t smhandle) {
    return std::array<sycl::accessor<T, 1>, 2>{ get_row_accessor<T>(cgh, smhandle),
                                                get_col_accessor<T>(cgh, smhandle) };
}

template <typename Functor, typename... CaptureOnlyAcc>
void submit_host_task(sycl::handler &cgh, sycl::queue &queue, Functor functor,
                      CaptureOnlyAcc... capture_only_accessors) {
    // Only capture the accessors to ensure the dependencies are properly handled
    // The accessors's pointer have already been set to the native container types in previous functions
    cgh.host_task([functor, queue, capture_only_accessors...](sycl::interop_handle ih) {
        auto unused = std::make_tuple(capture_only_accessors...);
        (void)unused;
        auto sc = RocsparseScopedContextHandler(queue, ih);
        functor(sc);
    });
}

template <typename Functor, typename... CaptureOnlyAcc>
void submit_host_task_with_acc(sycl::handler &cgh, sycl::queue &queue, Functor functor,
                               sycl::accessor<std::uint8_t> workspace_placeholder_acc,
                               CaptureOnlyAcc... capture_only_accessors) {
    // Only capture the accessors to ensure the dependencies are properly handled
    // The accessors's pointer have already been set to the native container types in previous functions
    cgh.require(workspace_placeholder_acc);
    cgh.host_task([functor, queue, workspace_placeholder_acc,
                   capture_only_accessors...](sycl::interop_handle ih) {
        auto unused = std::make_tuple(capture_only_accessors...);
        (void)unused;
        auto sc = RocsparseScopedContextHandler(queue, ih);
        functor(sc, workspace_placeholder_acc);
    });
}

/// Helper submit functions to capture all accessors from the generic containers \p other_containers and ensure the dependencies of buffers are respected.
/// The accessors are not directly used as the underlying data pointer has already been captured in previous functions.
/// \p workspace_placeholder_acc is a placeholder accessor that will be bound to the cgh if not empty and given to the functor as a last argument
template <bool UseWorkspace, typename Functor, typename... Ts>
sycl::event dispatch_submit(const std::string &function_name, sycl::queue queue,
                            const std::vector<sycl::event> &dependencies, Functor functor,
                            matrix_handle_t sm_handle,
                            sycl::accessor<std::uint8_t, 1> workspace_placeholder_acc,
                            Ts... other_containers) {
    if (sm_handle->all_use_buffer()) {
        detail::data_type value_type = sm_handle->get_value_type();
        detail::data_type int_type = sm_handle->get_int_type();

#define ONEMKL_ROCSPARSE_SUBMIT(FP_TYPE, INT_TYPE)                                             \
    return queue.submit([&](sycl::handler &cgh) {                                              \
        cgh.depends_on(dependencies);                                                          \
        auto fp_accs = get_fp_accessors<FP_TYPE>(cgh, sm_handle, other_containers...);         \
        auto int_accs = get_int_accessors<INT_TYPE>(cgh, sm_handle);                           \
        if constexpr (UseWorkspace) {                                                          \
            submit_host_task_with_acc(cgh, queue, functor, workspace_placeholder_acc, fp_accs, \
                                      int_accs);                                               \
        }                                                                                      \
        else {                                                                                 \
            (void)workspace_placeholder_acc;                                                   \
            submit_host_task(cgh, queue, functor, fp_accs, int_accs);                          \
        }                                                                                      \
    })
#define ONEMKL_ROCSPARSE_SUBMIT_INT(FP_TYPE)            \
    if (int_type == detail::data_type::int32) {         \
        ONEMKL_ROCSPARSE_SUBMIT(FP_TYPE, std::int32_t); \
    }                                                   \
    else if (int_type == detail::data_type::int64) {    \
        ONEMKL_ROCSPARSE_SUBMIT(FP_TYPE, std::int64_t); \
    }

        if (value_type == detail::data_type::real_fp32) {
            ONEMKL_ROCSPARSE_SUBMIT_INT(float)
        }
        else if (value_type == detail::data_type::real_fp64) {
            ONEMKL_ROCSPARSE_SUBMIT_INT(double)
        }
        else if (value_type == detail::data_type::complex_fp32) {
            ONEMKL_ROCSPARSE_SUBMIT_INT(std::complex<float>)
        }
        else if (value_type == detail::data_type::complex_fp64) {
            ONEMKL_ROCSPARSE_SUBMIT_INT(std::complex<double>)
        }

#undef ONEMKL_ROCSPARSE_SUBMIT_INT
#undef ONEMKL_ROCSPARSE_SUBMIT

        throw oneapi::mkl::exception("sparse_blas", function_name,
                                     "Could not dispatch buffer kernel to a supported type");
    }
    else {
        // USM submit does not need to capture accessors
        if constexpr (!UseWorkspace) {
            return queue.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dependencies);
                submit_host_task(cgh, queue, functor);
            });
        }
        else {
            throw oneapi::mkl::exception("sparse_blas", function_name,
                                         "Internal error: Cannot use accessor workspace with USM");
        }
    }
}

template <typename Functor, typename... Ts>
sycl::event dispatch_submit(const std::string &function_name, sycl::queue queue, Functor functor,
                            matrix_handle_t sm_handle,
                            sycl::accessor<std::uint8_t, 1> workspace_placeholder_acc,
                            Ts... other_containers) {
    return dispatch_submit<true>(function_name, queue, {}, functor, sm_handle,
                                 workspace_placeholder_acc, other_containers...);
}

template <typename Functor, typename... Ts>
sycl::event dispatch_submit(const std::string &function_name, sycl::queue queue,
                            const std::vector<sycl::event> &dependencies, Functor functor,
                            matrix_handle_t sm_handle, Ts... other_containers) {
    return dispatch_submit<false>(function_name, queue, dependencies, functor, sm_handle, {},
                                  other_containers...);
}

template <typename Functor, typename... Ts>
sycl::event dispatch_submit(const std::string &function_name, sycl::queue queue, Functor functor,
                            matrix_handle_t sm_handle, Ts... other_containers) {
    return dispatch_submit<false>(function_name, queue, {}, functor, sm_handle, {},
                                  other_containers...);
}

} // namespace oneapi::mkl::sparse::rocsparse

#endif // _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_TASKS_HPP_
