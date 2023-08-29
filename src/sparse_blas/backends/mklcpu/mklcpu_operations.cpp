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

// MKLCPU backend includes
// These includes define their own oneapi::mkl::sparse namespace with some of the types that are used here: matrix_handle_t, index_base.
#include <oneapi/mkl/spblas.hpp>

#include "oneapi/mkl/sparse_blas/detail/helper_types.hpp"
#include "oneapi/mkl/sparse_blas/detail/mklcpu/onemkl_sparse_blas_mklcpu.hpp"

// Include are set up so that oneapi::mkl::sparse::matrix_handle_t refers to the MKLCPU handle and
// oneapi::mkl::sparse::detail::matrix_handle* refers to the oneMKL interface handle in this file.

namespace oneapi::mkl::sparse::mklcpu {

sycl::event optimize_gemv(sycl::queue& /*queue*/, transpose /*transpose_val*/,
                          detail::matrix_handle* /*handle*/,
                          const std::vector<sycl::event>& /*dependencies*/) {
    throw unimplemented("SPARSE_BLAS", "optimize_gemv");
}

sycl::event optimize_trmv(sycl::queue& /*queue*/, uplo /*uplo_val*/, transpose /*transpose_val*/,
                          diag /*diag_val*/, detail::matrix_handle* /*handle*/,
                          const std::vector<sycl::event>& /*dependencies*/) {
    throw unimplemented("SPARSE_BLAS", "optimize_trmv");
}

sycl::event optimize_trsv(sycl::queue& /*queue*/, uplo /*uplo_val*/, transpose /*transpose_val*/,
                          diag /*diag_val*/, detail::matrix_handle* /*handle*/,
                          const std::vector<sycl::event>& /*dependencies*/) {
    throw unimplemented("SPARSE_BLAS", "optimize_trsv");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemv(
    sycl::queue& /*queue*/, transpose /*transpose_val*/, const fpType /*alpha*/,
    detail::matrix_handle* /*A_handle*/, sycl::buffer<fpType, 1>& /*x*/, const fpType /*beta*/,
    sycl::buffer<fpType, 1>& /*y*/) {
    throw unimplemented("SPARSE_BLAS", "gemv");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemv(
    sycl::queue& /*queue*/, transpose /*transpose_val*/, const fpType /*alpha*/,
    detail::matrix_handle* /*A_handle*/, const fpType* /*x*/, const fpType /*beta*/,
    const fpType* /*y*/, const std::vector<sycl::event>& /*dependencies*/) {
    throw unimplemented("SPARSE_BLAS", "gemv");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemvdot(
    sycl::queue& /*queue*/, transpose /*transpose_val*/, fpType /*alpha*/,
    detail::matrix_handle* /*A_handle*/, sycl::buffer<fpType, 1>& /*x*/, fpType /*beta*/,
    sycl::buffer<fpType, 1>& /*y*/, sycl::buffer<fpType, 1>& /*d*/) {
    throw unimplemented("SPARSE_BLAS", "gemvdot");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemvdot(
    sycl::queue& /*queue*/, transpose /*transpose_val*/, fpType /*alpha*/,
    detail::matrix_handle* /*A_handle*/, fpType* /*x*/, fpType /*beta*/, fpType* /*y*/,
    fpType* /*d*/, const std::vector<sycl::event>& /*dependencies*/) {
    throw unimplemented("SPARSE_BLAS", "gemvdot");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> symv(sycl::queue& /*queue*/, uplo /*uplo_val*/,
                                                         fpType /*alpha*/,
                                                         detail::matrix_handle* /*A_handle*/,
                                                         sycl::buffer<fpType, 1>& /*x*/,
                                                         fpType /*beta*/,
                                                         sycl::buffer<fpType, 1>& /*y*/) {
    throw unimplemented("SPARSE_BLAS", "symv");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> symv(
    sycl::queue& /*queue*/, uplo /*uplo_val*/, fpType /*alpha*/,
    detail::matrix_handle* /*A_handle*/, fpType* /*x*/, fpType /*beta*/, fpType* /*y*/,
    const std::vector<sycl::event>& /*dependencies*/) {
    throw unimplemented("SPARSE_BLAS", "symv");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> trmv(
    sycl::queue& /*queue*/, uplo /*uplo_val*/, transpose /*transpose_val*/, diag /*diag_val*/,
    fpType /*alpha*/, detail::matrix_handle* /*A_handle*/, sycl::buffer<fpType, 1>& /*x*/,
    fpType /*beta*/, sycl::buffer<fpType, 1>& /*y*/) {
    throw unimplemented("SPARSE_BLAS", "trmv");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> trmv(
    sycl::queue& /*queue*/, uplo /*uplo_val*/, transpose /*transpose_val*/, diag /*diag_val*/,
    fpType /*alpha*/, detail::matrix_handle* /*A_handle*/, fpType* /*x*/, fpType /*beta*/,
    fpType* /*y*/, const std::vector<sycl::event>& /*dependencies*/) {
    throw unimplemented("SPARSE_BLAS", "trmv");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> trsv(sycl::queue& /*queue*/, uplo /*uplo_val*/,
                                                         transpose /*transpose_val*/,
                                                         diag /*diag_val*/,
                                                         detail::matrix_handle* /*A_handle*/,
                                                         sycl::buffer<fpType, 1>& /*x*/,
                                                         sycl::buffer<fpType, 1>& /*y*/) {
    throw unimplemented("SPARSE_BLAS", "trsv");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> trsv(
    sycl::queue& /*queue*/, uplo /*uplo_val*/, transpose /*transpose_val*/, diag /*diag_val*/,
    detail::matrix_handle* /*A_handle*/, fpType* /*x*/, fpType* /*y*/,
    const std::vector<sycl::event>& /*dependencies*/) {
    throw unimplemented("SPARSE_BLAS", "trsv");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemm(
    sycl::queue& /*queue*/, layout /*dense_matrix_layout*/, transpose /*transpose_A*/,
    transpose /*transpose_B*/, const fpType /*alpha*/, detail::matrix_handle* /*A_handle*/,
    sycl::buffer<fpType, 1>& /*B*/, const std::int64_t /*columns*/, const std::int64_t /*ldb*/,
    const fpType /*beta*/, sycl::buffer<fpType, 1>& /*C*/, const std::int64_t /*ldc*/) {
    throw unimplemented("SPARSE_BLAS", "gemm");
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemm(
    sycl::queue& /*queue*/, layout /*dense_matrix_layout*/, transpose /*transpose_A*/,
    transpose /*transpose_B*/, const fpType /*alpha*/, detail::matrix_handle* /*A_handle*/,
    const fpType* /*B*/, const std::int64_t /*columns*/, const std::int64_t /*ldb*/,
    const fpType /*beta*/, const fpType* /*C*/, const std::int64_t /*ldc*/,
    const std::vector<sycl::event>& /*dependencies*/) {
    throw unimplemented("SPARSE_BLAS", "gemm");
}

#define INSTANTIATE_OPERATIONS(FP_TYPE)                                                           \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> gemv(                           \
        sycl::queue& queue, transpose transpose_val, const FP_TYPE alpha,                         \
        detail::matrix_handle* A_handle, sycl::buffer<FP_TYPE, 1>& x, const FP_TYPE beta,         \
        sycl::buffer<FP_TYPE, 1>& y);                                                             \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>, sycl::event> gemv(              \
        sycl::queue& queue, transpose transpose_val, const FP_TYPE alpha,                         \
        detail::matrix_handle* A_handle, const FP_TYPE* x, const FP_TYPE beta, const FP_TYPE* y,  \
        const std::vector<sycl::event>& dependencies);                                            \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> gemvdot(                        \
        sycl::queue& queue, transpose transpose_val, FP_TYPE alpha,                               \
        detail::matrix_handle* A_handle, sycl::buffer<FP_TYPE, 1>& x, FP_TYPE beta,               \
        sycl::buffer<FP_TYPE, 1>& y, sycl::buffer<FP_TYPE, 1>& d);                                \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>, sycl::event> gemvdot(           \
        sycl::queue& queue, transpose transpose_val, FP_TYPE alpha,                               \
        detail::matrix_handle* A_handle, FP_TYPE* x, FP_TYPE beta, FP_TYPE* y, FP_TYPE* d,        \
        const std::vector<sycl::event>& dependencies);                                            \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> symv(                           \
        sycl::queue& queue, uplo uplo_val, FP_TYPE alpha, detail::matrix_handle* A_handle,        \
        sycl::buffer<FP_TYPE, 1>& x, FP_TYPE beta, sycl::buffer<FP_TYPE, 1>& y);                  \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>, sycl::event> symv(              \
        sycl::queue& queue, uplo uplo_val, FP_TYPE alpha, detail::matrix_handle* A_handle,        \
        FP_TYPE* x, FP_TYPE beta, FP_TYPE* y, const std::vector<sycl::event>& dependencies);      \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> trmv(                           \
        sycl::queue& queue, uplo uplo_val, transpose transpose_val, diag diag_val, FP_TYPE alpha, \
        detail::matrix_handle* A_handle, sycl::buffer<FP_TYPE, 1>& x, FP_TYPE beta,               \
        sycl::buffer<FP_TYPE, 1>& y);                                                             \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>, sycl::event> trmv(              \
        sycl::queue& queue, uplo uplo_val, transpose transpose_val, diag diag_val, FP_TYPE alpha, \
        detail::matrix_handle* A_handle, FP_TYPE* x, FP_TYPE beta, FP_TYPE* y,                    \
        const std::vector<sycl::event>& dependencies);                                            \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> trsv(                           \
        sycl::queue& queue, uplo uplo_val, transpose transpose_val, diag diag_val,                \
        detail::matrix_handle* A_handle, sycl::buffer<FP_TYPE, 1>& x,                             \
        sycl::buffer<FP_TYPE, 1>& y);                                                             \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>, sycl::event> trsv(              \
        sycl::queue& queue, uplo uplo_val, transpose transpose_val, diag diag_val,                \
        detail::matrix_handle* A_handle, FP_TYPE* x, FP_TYPE* y,                                  \
        const std::vector<sycl::event>& dependencies);                                            \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> gemm(                           \
        sycl::queue& queue, layout dense_matrix_layout, transpose transpose_A,                    \
        transpose transpose_B, const FP_TYPE alpha, detail::matrix_handle* A_handle,              \
        sycl::buffer<FP_TYPE, 1>& B, const std::int64_t columns, const std::int64_t ldb,          \
        const FP_TYPE beta, sycl::buffer<FP_TYPE, 1>& C, const std::int64_t ldc);                 \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>, sycl::event> gemm(              \
        sycl::queue& queue, layout dense_matrix_layout, transpose transpose_A,                    \
        transpose transpose_B, const FP_TYPE alpha, detail::matrix_handle* A_handle,              \
        const FP_TYPE* B, const std::int64_t columns, const std::int64_t ldb, const FP_TYPE beta, \
        const FP_TYPE* C, const std::int64_t ldc, const std::vector<sycl::event>& dependencies);

INSTANTIATE_OPERATIONS(float)
INSTANTIATE_OPERATIONS(double)
INSTANTIATE_OPERATIONS(std::complex<float>)
INSTANTIATE_OPERATIONS(std::complex<double>)
#undef INSTANTIATE_OPERATIONS

} // namespace oneapi::mkl::sparse::mklcpu
