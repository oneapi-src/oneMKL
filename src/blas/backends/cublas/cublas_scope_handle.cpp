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
#include "cublas_scope_handle.hpp"
#include <CL/sycl/detail/common.hpp>

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

cublas_handle::~cublas_handle() noexcept(false) {
    for (auto &handle_pair : cublas_handle_mapper_) {
        cublasStatus_t err;
        if (handle_pair.second != nullptr) {
            auto handle = handle_pair.second->exchange(nullptr);
            if (handle != nullptr) {
                CUBLAS_ERROR_FUNC(cublasDestroy, err, handle);
                handle = nullptr;
            }
            delete handle_pair.second;
            handle_pair.second = nullptr;
        }
    }
    cublas_handle_mapper_.clear();
}
/**
 * Inserts a new element in the map if its key is unique. This new element
 * is constructed in place using args as the arguments for the construction
 * of a value_type (which is an object of a pair type). The insertion only
 * takes place if no other element in the container has a key equivalent to
 * the one being emplaced (keys in a map container are unique).
 */
thread_local cublas_handle CublasScopedContextHandler::handle_helper = cublas_handle{};

CublasScopedContextHandler::CublasScopedContextHandler(cl::sycl::queue queue) {
    placedContext_ = queue.get_context();
    auto device = queue.get_device();
    auto desired = cl::sycl::get_native<cl::sycl::backend::cuda>(placedContext_);
    auto cudaDevice = cl::sycl::get_native<cl::sycl::backend::cuda>(device);
    CUresult err;
    CUDA_ERROR_FUNC(cuCtxGetCurrent, err, &original_);
    CUcontext primary;
    cuDevicePrimaryCtxRetain(&primary, cudaDevice);
    bool isPrimary = primary == desired;
    cuDevicePrimaryCtxRelease(cudaDevice);
    if (original_ != desired) {
        // Sets the desired context as the active one for the thread
        CUDA_ERROR_FUNC(cuCtxSetCurrent, err, desired);
        // No context is installed and the suggested context is primary
        // This is the most common case. We can activate the context in the
        // thread and leave it there until all the PI context referring to the
        // same underlying CUDA primary context are destroyed. This emulates
        // the behaviour of the CUDA runtime api, and avoids costly context
        // switches. No action is required on this side of the if.
        needToRecover_ = !(original_ == nullptr && isPrimary);
    }
}

CublasScopedContextHandler::~CublasScopedContextHandler() noexcept(false) {
    if (needToRecover_) {
        CUresult err;
        CUDA_ERROR_FUNC(cuCtxSetCurrent, err, original_);
    }
}

void ContextCallback(void *userData) {
    auto *ptr = static_cast<std::atomic<cublasHandle_t> **>(userData);
    if (!ptr) {
        return;
    }
    if (*ptr != nullptr) {
        auto handle = (*ptr)->exchange(nullptr);
        if (handle != nullptr) {
            cublasStatus_t err1;
            CUBLAS_ERROR_FUNC(cublasDestroy, err1, handle);
            handle = nullptr;
        }
        delete *ptr;
        *ptr = nullptr;
    }
}

cublasHandle_t CublasScopedContextHandler::get_handle(const cl::sycl::queue &queue) {
    auto piPlacedContext_ = reinterpret_cast<pi_context>(placedContext_.get());
    CUstream streamId = get_stream(queue);
    cublasStatus_t err;
    auto it = handle_helper.cublas_handle_mapper_.find(piPlacedContext_);
    if (it != handle_helper.cublas_handle_mapper_.end()) {
        if (it->second == nullptr) {
            handle_helper.cublas_handle_mapper_.erase(it);
        }
        else {
            auto handle = it->second->load();
            if (handle != nullptr) {
                cudaStream_t currentStreamId;
                CUBLAS_ERROR_FUNC(cublasGetStream, err, handle, &currentStreamId);
                if (currentStreamId != streamId) {
                    CUBLAS_ERROR_FUNC(cublasSetStream, err, handle, streamId);
                }
                return handle;
            }
            else {
                handle_helper.cublas_handle_mapper_.erase(it);
            }
        }
    }

    cublasHandle_t handle;

    CUBLAS_ERROR_FUNC(cublasCreate, err, &handle);
    CUBLAS_ERROR_FUNC(cublasSetStream, err, handle, streamId);

    auto insert_iter = handle_helper.cublas_handle_mapper_.insert(
        std::make_pair(piPlacedContext_, new std::atomic<cublasHandle_t>(handle)));

    auto ptr = &(insert_iter.first->second);

    sycl::detail::pi::contextSetExtendedDeleter(placedContext_, ContextCallback, ptr);

    return handle;
}

CUstream CublasScopedContextHandler::get_stream(const cl::sycl::queue &queue) {
    return cl::sycl::get_native<cl::sycl::backend::cuda>(queue);
}
cl::sycl::context CublasScopedContextHandler::get_context(const cl::sycl::queue &queue) {
    return queue.get_context();
}

} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
