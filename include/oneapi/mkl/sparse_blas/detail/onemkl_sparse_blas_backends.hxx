/***************************************************************************
*  Copyright(C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0(the "License");
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

// This file is meant to be included in each backend onemkl_sparse_blas_BACKEND.hpp files.
// It is used to exports each symbol to the onemkl_sparse_blas_BACKEND library.

ONEMKL_EXPORT void init_matrix_handle(sycl::queue &queue, matrix_handle_t *p_handle);

ONEMKL_EXPORT sycl::event release_matrix_handle(sycl::queue &queue, matrix_handle_t *p_handle,
                                                const std::vector<sycl::event> &dependencies = {});

template <typename fpType, typename intType>
ONEMKL_EXPORT std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>> set_csr_data(
    sycl::queue &queue, matrix_handle_t handle, intType num_rows, intType num_cols, intType nnz,
    index_base index, sycl::buffer<intType, 1> &row_ptr, sycl::buffer<intType, 1> &col_ind,
    sycl::buffer<fpType, 1> &val);

template <typename fpType, typename intType>
ONEMKL_EXPORT std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>, sycl::event>
set_csr_data(sycl::queue &queue, matrix_handle_t handle, intType num_rows, intType num_cols,
             intType nnz, index_base index, intType *row_ptr, intType *col_ind, fpType *val,
             const std::vector<sycl::event> &dependencies = {});

ONEMKL_EXPORT sycl::event optimize_gemm(sycl::queue &queue, transpose transpose_A,
                                        matrix_handle_t handle,
                                        const std::vector<sycl::event> &dependencies = {});

ONEMKL_EXPORT sycl::event optimize_gemv(sycl::queue &queue, transpose transpose_val,
                                        matrix_handle_t handle,
                                        const std::vector<sycl::event> &dependencies = {});

ONEMKL_EXPORT sycl::event optimize_trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val,
                                        diag diag_val, matrix_handle_t handle,
                                        const std::vector<sycl::event> &dependencies = {});

template <typename fpType>
ONEMKL_EXPORT std::enable_if_t<detail::is_fp_supported_v<fpType>> gemv(
    sycl::queue &queue, transpose transpose_val, const fpType alpha, matrix_handle_t A_handle,
    sycl::buffer<fpType, 1> &x, const fpType beta, sycl::buffer<fpType, 1> &y);

template <typename fpType>
ONEMKL_EXPORT std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemv(
    sycl::queue &queue, transpose transpose_val, const fpType alpha, matrix_handle_t A_handle,
    const fpType *x, const fpType beta, fpType *y,
    const std::vector<sycl::event> &dependencies = {});

template <typename fpType>
ONEMKL_EXPORT std::enable_if_t<detail::is_fp_supported_v<fpType>> trsv(
    sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
    matrix_handle_t A_handle, sycl::buffer<fpType, 1> &x, sycl::buffer<fpType, 1> &y);

template <typename fpType>
ONEMKL_EXPORT std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> trsv(
    sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
    matrix_handle_t A_handle, const fpType *x, fpType *y,
    const std::vector<sycl::event> &dependencies = {});

template <typename fpType>
ONEMKL_EXPORT std::enable_if_t<detail::is_fp_supported_v<fpType>> gemm(
    sycl::queue &queue, layout dense_matrix_layout, transpose transpose_A, transpose transpose_B,
    const fpType alpha, matrix_handle_t A_handle, sycl::buffer<fpType, 1> &B,
    const std::int64_t columns, const std::int64_t ldb, const fpType beta,
    sycl::buffer<fpType, 1> &C, const std::int64_t ldc);

template <typename fpType>
ONEMKL_EXPORT std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemm(
    sycl::queue &queue, layout dense_matrix_layout, transpose transpose_A, transpose transpose_B,
    const fpType alpha, matrix_handle_t A_handle, const fpType *B, const std::int64_t columns,
    const std::int64_t ldb, const fpType beta, fpType *C, const std::int64_t ldc,
    const std::vector<sycl::event> &dependencies = {});
