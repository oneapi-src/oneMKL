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

#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_GLOBAL_HANDLE_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_GLOBAL_HANDLE_HPP_

/**
 * @file Similar to blas_handle.hpp
 * Provides a map from a ur_context_handle_t (or equivalent) to a rocsparse_handle.
 * @see rocsparse_scope_handle.hpp
*/

#include <atomic>
#include <unordered_map>

namespace oneapi::mkl::sparse::rocsparse {

template <typename T>
struct rocsparse_global_handle {
    using handle_container_t = std::unordered_map<T, std::atomic<rocsparse_handle> *>;
    handle_container_t rocsparse_global_handle_mapper_{};

    ~rocsparse_global_handle() noexcept(false) {
        for (auto &handle_pair : rocsparse_global_handle_mapper_) {
            if (handle_pair.second != nullptr) {
                auto handle = handle_pair.second->exchange(nullptr);
                if (handle != nullptr) {
                    ROCSPARSE_ERR_FUNC(rocsparse_destroy_handle, handle);
                    handle = nullptr;
                }
                else {
                    // if the handle is nullptr it means the handle was already
                    // destroyed by the ContextCallback and we're free to delete the
                    // atomic object.
                    delete handle_pair.second;
                }

                handle_pair.second = nullptr;
            }
        }
        rocsparse_global_handle_mapper_.clear();
    }
};

} // namespace oneapi::mkl::sparse::rocsparse

#endif // _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_GLOBAL_HANDLE_HPP_
