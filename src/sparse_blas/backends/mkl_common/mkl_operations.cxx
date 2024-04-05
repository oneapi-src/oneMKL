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

sycl::event optimize_gemm(sycl::queue& queue, transpose /*transpose_A*/,
                          detail::matrix_handle* /*handle*/,
                          const std::vector<sycl::event>& dependencies) {
    // TODO: Call to optimize_gemm with 2024.1 oneMKL release
    // Return an event depending on the dependencies
    return queue.submit([=](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.host_task([=]() { /* Empty kernel */ });
    });
}

sycl::event optimize_gemm(sycl::queue& queue, transpose /*transpose_A*/, transpose /*transpose_B*/,
                          layout /*dense_matrix_layout*/, const std::int64_t /*columns*/,
                          detail::matrix_handle* /*handle*/,
                          const std::vector<sycl::event>& dependencies) {
    // TODO: Call to optimize_gemm with 2024.1 oneMKL release
    // Return an event depending on the dependencies
    return queue.submit([=](sycl::handler& cgh) {
        cgh.depends_on(dependencies);
        cgh.host_task([=]() { /* Empty kernel */ });
    });
}

sycl::event optimize_gemv(sycl::queue& queue, transpose transpose_val,
                          detail::matrix_handle* handle,
                          const std::vector<sycl::event>& dependencies) {
    return oneapi::mkl::sparse::optimize_gemv(queue, transpose_val, detail::get_handle(handle),
                                              dependencies);
}

sycl::event optimize_trsv(sycl::queue& queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                          detail::matrix_handle* handle,
                          const std::vector<sycl::event>& dependencies) {
    // TODO: Remove this if condition once Intel oneMKL adds support for trans/conjtrans to optimize_trsv
    if (transpose_val != transpose::nontrans) {
        throw mkl::unimplemented("sparse_blas/backends/mkl", __FUNCTION__,
                                 "Transposed or conjugate trsv is not supported");
    }
    return oneapi::mkl::sparse::optimize_trsv(queue, uplo_val, transpose_val, diag_val,
                                              detail::get_handle(handle), dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemv(
    sycl::queue& queue, transpose transpose_val, const fpType alpha,
    detail::matrix_handle* A_handle, sycl::buffer<fpType, 1>& x, const fpType beta,
    sycl::buffer<fpType, 1>& y) {
    oneapi::mkl::sparse::gemv(queue, transpose_val, alpha, detail::get_handle(A_handle), x, beta,
                              y);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemv(
    sycl::queue& queue, transpose transpose_val, const fpType alpha,
    detail::matrix_handle* A_handle, const fpType* x, const fpType beta, fpType* y,
    const std::vector<sycl::event>& dependencies) {
    return oneapi::mkl::sparse::gemv(queue, transpose_val, alpha, detail::get_handle(A_handle), x,
                                     beta, y, dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> trsv(sycl::queue& queue, uplo uplo_val,
                                                         transpose transpose_val, diag diag_val,
                                                         detail::matrix_handle* A_handle,
                                                         sycl::buffer<fpType, 1>& x,
                                                         sycl::buffer<fpType, 1>& y) {
    // TODO: Remove this if condition once Intel oneMKL adds support for trans/conjtrans to trsv
    if (transpose_val != transpose::nontrans) {
        throw mkl::unimplemented("sparse_blas/backends/mkl", __FUNCTION__,
                                 "Transposed or conjugate trsv is not supported");
    }

    const fpType alpha = static_cast<fpType>(1);
    oneapi::mkl::sparse::trsv(queue, uplo_val, transpose_val, diag_val, alpha,
                              detail::get_handle(A_handle), x, y);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> trsv(
    sycl::queue& queue, uplo uplo_val, transpose transpose_val, diag diag_val,
    detail::matrix_handle* A_handle, const fpType* x, fpType* y,
    const std::vector<sycl::event>& dependencies) {
    // TODO: Remove this if condition once Intel oneMKL adds support for trans/conjtrans to trsv
    if (transpose_val != transpose::nontrans) {
        throw mkl::unimplemented("sparse_blas/backends/mkl", __FUNCTION__,
                                 "Transposed or conjugate trsv is not supported");
    }
    // TODO: Remove const_cast in future oneMKL release
    const fpType alpha = static_cast<fpType>(1);
    return oneapi::mkl::sparse::trsv(queue, uplo_val, transpose_val, diag_val, alpha,
                                     detail::get_handle(A_handle), const_cast<fpType*>(x), y,
                                     dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemm(
    sycl::queue& queue, layout dense_matrix_layout, transpose transpose_A, transpose transpose_B,
    const fpType alpha, detail::matrix_handle* A_handle, sycl::buffer<fpType, 1>& B,
    const std::int64_t columns, const std::int64_t ldb, const fpType beta,
    sycl::buffer<fpType, 1>& C, const std::int64_t ldc) {
    oneapi::mkl::sparse::gemm(queue, dense_matrix_layout, transpose_A, transpose_B, alpha,
                              detail::get_handle(A_handle), B, columns, ldb, beta, C, ldc);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemm(
    sycl::queue& queue, layout dense_matrix_layout, transpose transpose_A, transpose transpose_B,
    const fpType alpha, detail::matrix_handle* A_handle, const fpType* B,
    const std::int64_t columns, const std::int64_t ldb, const fpType beta, fpType* C,
    const std::int64_t ldc, const std::vector<sycl::event>& dependencies) {
    // TODO: Remove const_cast in future oneMKL release
    return oneapi::mkl::sparse::gemm(queue, dense_matrix_layout, transpose_A, transpose_B, alpha,
                                     detail::get_handle(A_handle), const_cast<fpType*>(B), columns,
                                     ldb, beta, C, ldc, dependencies);
}

#define INSTANTIATE_GEMV(FP_TYPE)                                                          \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> gemv(                    \
        sycl::queue& queue, transpose transpose_val, const FP_TYPE alpha,                  \
        detail::matrix_handle* A_handle, sycl::buffer<FP_TYPE, 1>& x, const FP_TYPE beta,  \
        sycl::buffer<FP_TYPE, 1>& y);                                                      \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>, sycl::event> gemv(       \
        sycl::queue& queue, transpose transpose_val, const FP_TYPE alpha,                  \
        detail::matrix_handle* A_handle, const FP_TYPE* x, const FP_TYPE beta, FP_TYPE* y, \
        const std::vector<sycl::event>& dependencies)

#define INSTANTIATE_TRSV(FP_TYPE)                                                    \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> trsv(              \
        sycl::queue& queue, uplo uplo_val, transpose transpose_val, diag diag_val,   \
        detail::matrix_handle* A_handle, sycl::buffer<FP_TYPE, 1>& x,                \
        sycl::buffer<FP_TYPE, 1>& y);                                                \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>, sycl::event> trsv( \
        sycl::queue& queue, uplo uplo_val, transpose transpose_val, diag diag_val,   \
        detail::matrix_handle* A_handle, const FP_TYPE* x, FP_TYPE* y,               \
        const std::vector<sycl::event>& dependencies)

#define INSTANTIATE_GEMM(FP_TYPE)                                                                 \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>> gemm(                           \
        sycl::queue& queue, layout dense_matrix_layout, transpose transpose_A,                    \
        transpose transpose_B, const FP_TYPE alpha, detail::matrix_handle* A_handle,              \
        sycl::buffer<FP_TYPE, 1>& B, const std::int64_t columns, const std::int64_t ldb,          \
        const FP_TYPE beta, sycl::buffer<FP_TYPE, 1>& C, const std::int64_t ldc);                 \
    template std::enable_if_t<detail::is_fp_supported_v<FP_TYPE>, sycl::event> gemm(              \
        sycl::queue& queue, layout dense_matrix_layout, transpose transpose_A,                    \
        transpose transpose_B, const FP_TYPE alpha, detail::matrix_handle* A_handle,              \
        const FP_TYPE* B, const std::int64_t columns, const std::int64_t ldb, const FP_TYPE beta, \
        FP_TYPE* C, const std::int64_t ldc, const std::vector<sycl::event>& dependencies)

FOR_EACH_FP_TYPE(INSTANTIATE_GEMV);
FOR_EACH_FP_TYPE(INSTANTIATE_TRSV);
FOR_EACH_FP_TYPE(INSTANTIATE_GEMM);

#undef INSTANTIATE_GEMV
#undef INSTANTIATE_TRSV
#undef INSTANTIATE_GEMM
