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

/*
This file lists functions matching those required by sparse_blas_function_table_t in
src/sparse_blas/function_table.hpp.

To use this:

#define WRAPPER_VERSION <Wrapper version number>
#define BACKEND         <Backend name eg. mklgpu>

extern "C" sparse_blas_function_table_t mkl_sparse_blas_table = {
    WRAPPER_VERSION,
#include "sparse_blas/backends/backend_wrappers.cxx"
};

Changes to this file should be matched to changes in sparse_blas/function_table.hpp. The required
function template instantiations must be added to backend_sparse_blas_instantiations.cxx.
*/

// clang-format off
oneapi::mkl::sparse::BACKEND::init_matrix_handle,
oneapi::mkl::sparse::BACKEND::release_matrix_handle,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::set_csr_data,
oneapi::mkl::sparse::BACKEND::optimize_gemv,
oneapi::mkl::sparse::BACKEND::optimize_trsv,
oneapi::mkl::sparse::BACKEND::gemv,
oneapi::mkl::sparse::BACKEND::gemv,
oneapi::mkl::sparse::BACKEND::gemv,
oneapi::mkl::sparse::BACKEND::gemv,
oneapi::mkl::sparse::BACKEND::gemv,
oneapi::mkl::sparse::BACKEND::gemv,
oneapi::mkl::sparse::BACKEND::gemv,
oneapi::mkl::sparse::BACKEND::gemv,
oneapi::mkl::sparse::BACKEND::trsv,
oneapi::mkl::sparse::BACKEND::trsv,
oneapi::mkl::sparse::BACKEND::trsv,
oneapi::mkl::sparse::BACKEND::trsv,
oneapi::mkl::sparse::BACKEND::trsv,
oneapi::mkl::sparse::BACKEND::trsv,
oneapi::mkl::sparse::BACKEND::trsv,
oneapi::mkl::sparse::BACKEND::trsv,
oneapi::mkl::sparse::BACKEND::gemm,
oneapi::mkl::sparse::BACKEND::gemm,
oneapi::mkl::sparse::BACKEND::gemm,
oneapi::mkl::sparse::BACKEND::gemm,
oneapi::mkl::sparse::BACKEND::gemm,
oneapi::mkl::sparse::BACKEND::gemm,
oneapi::mkl::sparse::BACKEND::gemm,
oneapi::mkl::sparse::BACKEND::gemm,
    // clang-format on
