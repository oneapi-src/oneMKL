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
                               oneapi::mkl::sparse::matrix_handle_t *handle) {
    oneapi::mkl::sparse::BACKEND::init_matrix_handle(selector.get_queue(), handle);
}

inline sycl::event release_matrix_handle(backend_selector<backend::BACKEND> selector,
                                         oneapi::mkl::sparse::matrix_handle_t *handle,
                                         const std::vector<sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::BACKEND::release_matrix_handle(selector.get_queue(), handle,
                                                               dependencies);
}

template <typename fpType, typename intType>
std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>> set_csr_data(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::sparse::matrix_handle_t handle,
    intType num_rows, intType num_cols, oneapi::mkl::index_base index,
    sycl::buffer<intType, 1> &row_ptr, sycl::buffer<intType, 1> &col_ind,
    sycl::buffer<fpType, 1> &val) {
    oneapi::mkl::sparse::BACKEND::set_csr_data(selector.get_queue(), handle, num_rows, num_cols,
                                               index, row_ptr, col_ind, val);
}

template <typename fpType, typename intType>
std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>, sycl::event> set_csr_data(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::sparse::matrix_handle_t handle,
    intType num_rows, intType num_cols, oneapi::mkl::index_base index, intType *row_ptr,
    intType *col_ind, fpType *val, const std::vector<sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::BACKEND::set_csr_data(selector.get_queue(), handle, num_rows,
                                                      num_cols, index, row_ptr, col_ind, val,
                                                      dependencies);
}

inline sycl::event optimize_gemv(backend_selector<backend::BACKEND> selector,
                                 oneapi::mkl::transpose transpose_val,
                                 oneapi::mkl::sparse::matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies) {
    return oneapi::mkl::sparse::BACKEND::optimize_gemv(selector.get_queue(), transpose_val, handle,
                                                       dependencies);
}

inline sycl::event optimize_trmv(backend_selector<backend::BACKEND> selector,
                                 oneapi::mkl::uplo uplo_val, oneapi::mkl::transpose transpose_val,
                                 oneapi::mkl::diag diag_val,
                                 oneapi::mkl::sparse::matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies) {
    return oneapi::mkl::sparse::BACKEND::optimize_trmv(
        selector.get_queue(), uplo_val, transpose_val, diag_val, handle, dependencies);
}

inline sycl::event optimize_trsv(backend_selector<backend::BACKEND> selector,
                                 oneapi::mkl::uplo uplo_val, oneapi::mkl::transpose transpose_val,
                                 oneapi::mkl::diag diag_val,
                                 oneapi::mkl::sparse::matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies) {
    return oneapi::mkl::sparse::BACKEND::optimize_trsv(
        selector.get_queue(), uplo_val, transpose_val, diag_val, handle, dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemv(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::transpose transpose_val,
    const fpType alpha, oneapi::mkl::sparse::matrix_handle_t A_handle, sycl::buffer<fpType, 1> &x,
    const fpType beta, sycl::buffer<fpType, 1> &y) {
    oneapi::mkl::sparse::BACKEND::gemv(selector.get_queue(), transpose_val, alpha, A_handle, x,
                                       beta, y);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemv(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::transpose transpose_val,
    const fpType alpha, oneapi::mkl::sparse::matrix_handle_t A_handle, const fpType *x,
    const fpType beta, const fpType *y, const std::vector<sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::BACKEND::gemv(selector.get_queue(), transpose_val, alpha, A_handle,
                                              x, beta, y, dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemvdot(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::transpose transpose_val, fpType alpha,
    oneapi::mkl::sparse::matrix_handle_t A_handle, sycl::buffer<fpType, 1> &x, fpType beta,
    sycl::buffer<fpType, 1> &y, sycl::buffer<fpType, 1> &d) {
    oneapi::mkl::sparse::BACKEND::gemvdot(selector.get_queue(), transpose_val, alpha, A_handle, x,
                                          beta, y, d);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemvdot(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::transpose transpose_val, fpType alpha,
    oneapi::mkl::sparse::matrix_handle_t A_handle, fpType *x, fpType beta, fpType *y, fpType *d,
    const std::vector<sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::BACKEND::gemvdot(selector.get_queue(), transpose_val, alpha,
                                                 A_handle, x, beta, y, d, dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> symv(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::uplo uplo_val, fpType alpha,
    oneapi::mkl::sparse::matrix_handle_t A_handle, sycl::buffer<fpType, 1> &x, fpType beta,
    sycl::buffer<fpType, 1> &y) {
    oneapi::mkl::sparse::BACKEND::symv(selector.get_queue(), uplo_val, alpha, A_handle, x, beta, y);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> symv(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::uplo uplo_val, fpType alpha,
    oneapi::mkl::sparse::matrix_handle_t A_handle, fpType *x, fpType beta, fpType *y,
    const std::vector<sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::BACKEND::symv(selector.get_queue(), uplo_val, alpha, A_handle, x,
                                              beta, y, dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> trmv(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::uplo uplo_val,
    oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val, fpType alpha,
    oneapi::mkl::sparse::matrix_handle_t A_handle, sycl::buffer<fpType, 1> &x, fpType beta,
    sycl::buffer<fpType, 1> &y) {
    oneapi::mkl::sparse::BACKEND::trmv(selector.get_queue(), uplo_val, transpose_val, diag_val,
                                       alpha, A_handle, x, beta, y);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> trmv(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::uplo uplo_val,
    oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val, fpType alpha,
    oneapi::mkl::sparse::matrix_handle_t A_handle, fpType *x, fpType beta, fpType *y,
    const std::vector<sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::BACKEND::trmv(selector.get_queue(), uplo_val, transpose_val,
                                              diag_val, alpha, A_handle, x, beta, y, dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> trsv(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::uplo uplo_val,
    oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
    oneapi::mkl::sparse::matrix_handle_t A_handle, sycl::buffer<fpType, 1> &x,
    sycl::buffer<fpType, 1> &y) {
    oneapi::mkl::sparse::BACKEND::trsv(selector.get_queue(), uplo_val, transpose_val, diag_val,
                                       A_handle, x, y);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> trsv(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::uplo uplo_val,
    oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
    oneapi::mkl::sparse::matrix_handle_t A_handle, fpType *x, fpType *y,
    const std::vector<sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::BACKEND::trsv(selector.get_queue(), uplo_val, transpose_val,
                                              diag_val, A_handle, x, y, dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemm(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::layout dense_matrix_layout,
    oneapi::mkl::transpose transpose_A, oneapi::mkl::transpose transpose_B, const fpType alpha,
    oneapi::mkl::sparse::matrix_handle_t A_handle, sycl::buffer<fpType, 1> &B,
    const std::int64_t columns, const std::int64_t ldb, const fpType beta,
    sycl::buffer<fpType, 1> &C, const std::int64_t ldc) {
    oneapi::mkl::sparse::BACKEND::gemm(selector.get_queue(), dense_matrix_layout, transpose_A,
                                       transpose_B, alpha, A_handle, B, columns, ldb, beta, C, ldc);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemm(
    backend_selector<backend::BACKEND> selector, oneapi::mkl::layout dense_matrix_layout,
    oneapi::mkl::transpose transpose_A, oneapi::mkl::transpose transpose_B, const fpType alpha,
    oneapi::mkl::sparse::matrix_handle_t A_handle, const fpType *B, const std::int64_t columns,
    const std::int64_t ldb, const fpType beta, const fpType *C, const std::int64_t ldc,
    const std::vector<sycl::event> &dependencies = {}) {
    return oneapi::mkl::sparse::BACKEND::gemm(selector.get_queue(), dense_matrix_layout,
                                              transpose_A, transpose_B, alpha, A_handle, B, columns,
                                              ldb, beta, C, ldc, dependencies);
}