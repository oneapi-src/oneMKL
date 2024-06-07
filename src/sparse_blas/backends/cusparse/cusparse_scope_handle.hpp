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
#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_SCOPE_HANDLE_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_SCOPE_HANDLE_HPP_

/**
 * @file Similar to cusparse_scope_handle.hpp
*/

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <thread>

#include "cusparse_error.hpp"
#include "cusparse_global_handle.hpp"
#include "cusparse_helper.hpp"

namespace oneapi::mkl::sparse::cusparse {

class CusparseScopedContextHandler {
    CUcontext original_;
    sycl::context *placedContext_;
    sycl::interop_handle &ih;
    bool needToRecover_;
    static thread_local cusparse_global_handle<pi_context> handle_helper;

    CUstream get_stream(const sycl::queue &queue);
    sycl::context get_context(const sycl::queue &queue);

public:
    CusparseScopedContextHandler(sycl::queue queue, sycl::interop_handle &ih);

    ~CusparseScopedContextHandler() noexcept(false);

    /**
     * @brief get_handle: creates the handle by implicitly impose the advice
     * given by nvidia for creating a cusparse_global_handle. (e.g. one cuStream per device
     * per thread).
     * @param queue sycl queue.
     * @return a pair of: cusparseHandle_t a handle to construct cusparse routines; and a CUDA stream
     */
    std::pair<cusparseHandle_t, CUstream> get_handle_and_stream(const sycl::queue &queue);

    /// See get_handle_and_stream
    cusparseHandle_t get_handle(const sycl::queue &queue);

    // This is a work-around function for reinterpret_casting the memory. This
    // will be fixed when SYCL-2020 has been implemented for Pi backend.
    template <typename AccT>
    inline void *get_mem(AccT acc) {
        auto cudaPtr = ih.get_native_mem<sycl::backend::ext_oneapi_cuda>(acc);
        return reinterpret_cast<void *>(cudaPtr);
    }

    template <typename T>
    inline void *get_mem(T *ptr) {
        return reinterpret_cast<void *>(ptr);
    }

    void wait_stream(const sycl::queue &queue) {
        cuStreamSynchronize(get_stream(queue));
    }
};

} // namespace oneapi::mkl::sparse::cusparse

#endif //_ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_SCOPE_HANDLE_HPP_
