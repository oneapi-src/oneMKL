/*******************************************************************************
* Copyright 2023 Codeplay Software Ltd.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

// MKLCPU and MKLGPU backends include
// This include defines its own oneapi::mkl::sparse namespace with some of the types that are used here: matrix_handle_t, index_base, transpose, uolo, diag.
#include <oneapi/mkl/spblas.hpp>

// Includes are set up so that oneapi::mkl::sparse namespace refers to the MKLCPU and MKLGPU backends namespace (oneMKL product)
// in this file.
// oneapi::mkl::sparse::detail namespace refers to the oneMKL interface namespace.

#include "oneapi/mkl/sparse_blas/detail/helper_types.hpp"

namespace oneapi::mkl::sparse::detail {

inline auto get_handle(detail::matrix_handle **handle) {
    return reinterpret_cast<oneapi::mkl::sparse::matrix_handle_t *>(handle);
}

inline auto get_handle(detail::matrix_handle *handle) {
    return reinterpret_cast<oneapi::mkl::sparse::matrix_handle_t>(handle);
}

} // namespace oneapi::mkl::sparse::detail
