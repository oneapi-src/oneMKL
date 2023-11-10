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

// This file is meant to be included in each backend sparse_blas_ct.hpp files
// Each function calls the implementation from onemkl_sparse_blas_backends.hxx

#ifndef BACKEND
#error "BACKEND is not defined"
#endif

inline void init_matrix_handle(backend_selector<backend::BACKEND> selector,
                               matrix_handle_t *p_handle) {
    BACKEND::init_matrix_handle(selector.get_queue(), p_handle);
}

inline sycl::event release_matrix_handle(backend_selector<backend::BACKEND> selector,
                                         matrix_handle_t *p_handle,
                                         const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::release_matrix_handle(selector.get_queue(), p_handle, dependencies);
}

template <typename fpType, typename intType>
std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>> set_csr_data(
    backend_selector<backend::BACKEND> selector, matrix_handle_t handle, intType num_rows,
    intType num_cols, intType nnz, index_base index, sycl::buffer<intType, 1> &row_ptr,
    sycl::buffer<intType, 1> &col_ind, sycl::buffer<fpType, 1> &val) {
    BACKEND::set_csr_data(selector.get_queue(), handle, num_rows, num_cols, nnz, index, row_ptr,
                          col_ind, val);
}

template <typename fpType, typename intType>
std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>, sycl::event> set_csr_data(
    backend_selector<backend::BACKEND> selector, matrix_handle_t handle, intType num_rows,
    intType num_cols, intType nnz, index_base index, intType *row_ptr, intType *col_ind,
    fpType *val, const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::set_csr_data(selector.get_queue(), handle, num_rows, num_cols, nnz, index,
                                 row_ptr, col_ind, val, dependencies);
}

inline sycl::event optimize_gemm(backend_selector<backend::BACKEND> selector, transpose transpose_A,
                                 matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::optimize_gemm(selector.get_queue(), transpose_A, handle, dependencies);
}

inline sycl::event optimize_gemm(backend_selector<backend::BACKEND> selector, transpose transpose_A,
                                 transpose transpose_B, layout dense_matrix_layout,
                                 const std::int64_t columns, matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::optimize_gemm(selector.get_queue(), transpose_A, transpose_B,
                                  dense_matrix_layout, columns, handle, dependencies);
}

inline sycl::event optimize_gemv(backend_selector<backend::BACKEND> selector,
                                 transpose transpose_val, matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::optimize_gemv(selector.get_queue(), transpose_val, handle, dependencies);
}

inline sycl::event optimize_trsv(backend_selector<backend::BACKEND> selector, uplo uplo_val,
                                 transpose transpose_val, diag diag_val, matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::optimize_trsv(selector.get_queue(), uplo_val, transpose_val, diag_val, handle,
                                  dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemv(
    backend_selector<backend::BACKEND> selector, transpose transpose_val, const fpType alpha,
    matrix_handle_t A_handle, sycl::buffer<fpType, 1> &x, const fpType beta,
    sycl::buffer<fpType, 1> &y) {
    BACKEND::gemv(selector.get_queue(), transpose_val, alpha, A_handle, x, beta, y);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemv(
    backend_selector<backend::BACKEND> selector, transpose transpose_val, const fpType alpha,
    matrix_handle_t A_handle, const fpType *x, const fpType beta, fpType *y,
    const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::gemv(selector.get_queue(), transpose_val, alpha, A_handle, x, beta, y,
                         dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> trsv(
    backend_selector<backend::BACKEND> selector, uplo uplo_val, transpose transpose_val,
    diag diag_val, matrix_handle_t A_handle, sycl::buffer<fpType, 1> &x,
    sycl::buffer<fpType, 1> &y) {
    BACKEND::trsv(selector.get_queue(), uplo_val, transpose_val, diag_val, A_handle, x, y);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> trsv(
    backend_selector<backend::BACKEND> selector, uplo uplo_val, transpose transpose_val,
    diag diag_val, matrix_handle_t A_handle, const fpType *x, fpType *y,
    const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::trsv(selector.get_queue(), uplo_val, transpose_val, diag_val, A_handle, x, y,
                         dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemm(
    backend_selector<backend::BACKEND> selector, layout dense_matrix_layout, transpose transpose_A,
    transpose transpose_B, const fpType alpha, matrix_handle_t A_handle, sycl::buffer<fpType, 1> &B,
    const std::int64_t columns, const std::int64_t ldb, const fpType beta,
    sycl::buffer<fpType, 1> &C, const std::int64_t ldc) {
    BACKEND::gemm(selector.get_queue(), dense_matrix_layout, transpose_A, transpose_B, alpha,
                  A_handle, B, columns, ldb, beta, C, ldc);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemm(
    backend_selector<backend::BACKEND> selector, layout dense_matrix_layout, transpose transpose_A,
    transpose transpose_B, const fpType alpha, matrix_handle_t A_handle, const fpType *B,
    const std::int64_t columns, const std::int64_t ldb, const fpType beta, fpType *C,
    const std::int64_t ldc, const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::gemm(selector.get_queue(), dense_matrix_layout, transpose_A, transpose_B, alpha,
                         A_handle, B, columns, ldb, beta, C, ldc, dependencies);
}
