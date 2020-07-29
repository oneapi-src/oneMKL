/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <CL/sycl.hpp>

#include "mkl_internal_blas_gpu_wrappers.hpp"
#include "oneapi/mkl/blas/detail/mklgpu/onemkl_blas_mklgpu.hpp"
#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace mklgpu {

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     float alpha, const float *a, std::int64_t lda, const float *b,
                     std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b,
                                               ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     double alpha, const double *a, std::int64_t lda, const double *b,
                     std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b,
                                               ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b,
                                               ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b,
                                               ldb, beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, const float *b, std::int64_t ldb, float beta,
                     float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::symm(queue, left_right, upper_lower, m, n, alpha, a, lda,
                                               b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                     double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::symm(queue, left_right, upper_lower, m, n, alpha, a, lda,
                                               b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::symm(queue, left_right, upper_lower, m, n, alpha, a, lda,
                                               b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::symm(queue, left_right, upper_lower, m, n, alpha, a, lda,
                                               b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event hemm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hemm(queue, left_right, upper_lower, m, n, alpha, a, lda,
                                               b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event hemm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hemm(queue, left_right, upper_lower, m, n, alpha, a, lda,
                                               b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, float alpha,
                     const float *a, std::int64_t lda, float beta, float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta,
                                               c, ldc, dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, double alpha,
                     const double *a, std::int64_t lda, double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta,
                                               c, ldc, dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta,
                                               c, ldc, dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta,
                                               c, ldc, dependencies);
}

cl::sycl::event herk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, float alpha,
                     const std::complex<float> *a, std::int64_t lda, float beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta,
                                               c, ldc, dependencies);
}

cl::sycl::event herk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, double alpha,
                     const std::complex<double> *a, std::int64_t lda, double beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta,
                                               c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, float alpha,
                      const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
                      float beta, float *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b,
                                                ldb, beta, c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, double alpha,
                      const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                      double beta, double *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b,
                                                ldb, beta, c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                      const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                      std::complex<float> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b,
                                                ldb, beta, c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b,
                                                ldb, beta, c, ldc, dependencies);
}

cl::sycl::event her2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                      const std::complex<float> *b, std::int64_t ldb, float beta,
                      std::complex<float> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b,
                                                ldb, beta, c, ldc, dependencies);
}

cl::sycl::event her2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, double beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b,
                                                ldb, beta, c, ldc, dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, float *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trmm(queue, left_right, upper_lower, trans, unit_diag, m,
                                               n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, double *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trmm(queue, left_right, upper_lower, trans, unit_diag, m,
                                               n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trmm(queue, left_right, upper_lower, trans, unit_diag, m,
                                               n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trmm(queue, left_right, upper_lower, trans, unit_diag, m,
                                               n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, float *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trsm(queue, left_right, upper_lower, trans, unit_diag, m,
                                               n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, double *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trsm(queue, left_right, upper_lower, trans, unit_diag, m,
                                               n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trsm(queue, left_right, upper_lower, trans, unit_diag, m,
                                               n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trsm(queue, left_right, upper_lower, trans, unit_diag, m,
                                               n, alpha, a, lda, b, ldb, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y,
                                               incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda,
                     const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y,
                                               incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y,
                                               incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                     std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y,
                                               incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha, const float *a,
                     std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx,
                                               beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                     const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                     double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx,
                                               beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx,
                                               beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx,
                                               beta, y, incy, dependencies);
}

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::ger(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                              dependencies);
}

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                    double *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::ger(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                              dependencies);
}

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                               dependencies);
}

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                               dependencies);
}

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::geru(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                               dependencies);
}

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::geru(queue, m, n, alpha, x, incx, y, incy, a, lda,
                                               dependencies);
}

cl::sycl::event hbmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx,
                                               beta, y, incy, dependencies);
}

cl::sycl::event hbmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                     std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx,
                                               beta, y, incy, dependencies);
}

cl::sycl::event hemv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta,
                                               y, incy, dependencies);
}

cl::sycl::event hemv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta,
                                               y, incy, dependencies);
}

cl::sycl::event her(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                    float alpha, const std::complex<float> *x, std::int64_t incx,
                    std::complex<float> *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::her(queue, upper_lower, n, alpha, x, incx, a, lda,
                                              dependencies);
}

cl::sycl::event her(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                    double alpha, const std::complex<double> *x, std::int64_t incx,
                    std::complex<double> *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::her(queue, upper_lower, n, alpha, x, incx, a, lda,
                                              dependencies);
}

cl::sycl::event her2(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::her2(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                               lda, dependencies);
}

cl::sycl::event her2(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::her2(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                               lda, dependencies);
}

cl::sycl::event hpmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hpmv(queue, upper_lower, n, alpha, a, x, incx, beta, y,
                                               incy, dependencies);
}

cl::sycl::event hpmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hpmv(queue, upper_lower, n, alpha, a, x, incx, beta, y,
                                               incy, dependencies);
}

cl::sycl::event hpr(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                    float alpha, const std::complex<float> *x, std::int64_t incx,
                    std::complex<float> *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hpr(queue, upper_lower, n, alpha, x, incx, a,
                                              dependencies);
}

cl::sycl::event hpr(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                    double alpha, const std::complex<double> *x, std::int64_t incx,
                    std::complex<double> *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hpr(queue, upper_lower, n, alpha, x, incx, a,
                                              dependencies);
}

cl::sycl::event hpr2(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                               dependencies);
}

cl::sycl::event hpr2(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                               dependencies);
}

cl::sycl::event sbmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx,
                                               beta, y, incy, dependencies);
}

cl::sycl::event sbmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     std::int64_t k, double alpha, const double *a, std::int64_t lda,
                     const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx,
                                               beta, y, incy, dependencies);
}

cl::sycl::event symv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta,
                                               y, incy, dependencies);
}

cl::sycl::event symv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     double alpha, const double *a, std::int64_t lda, const double *x,
                     std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta,
                                               y, incy, dependencies);
}

cl::sycl::event syr(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                    float alpha, const float *x, std::int64_t incx, float *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syr(queue, upper_lower, n, alpha, x, incx, a, lda,
                                              dependencies);
}

cl::sycl::event syr(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                    double alpha, const double *x, std::int64_t incx, double *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syr(queue, upper_lower, n, alpha, x, incx, a, lda,
                                              dependencies);
}

cl::sycl::event syr2(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     float alpha, const float *x, std::int64_t incx, const float *y,
                     std::int64_t incy, float *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                               lda, dependencies);
}

cl::sycl::event syr2(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     double alpha, const double *x, std::int64_t incx, const double *y,
                     std::int64_t incy, double *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                               lda, dependencies);
}

cl::sycl::event spmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     float alpha, const float *a, const float *x, std::int64_t incx, float beta,
                     float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::spmv(queue, upper_lower, n, alpha, a, x, incx, beta, y,
                                               incy, dependencies);
}

cl::sycl::event spmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     double alpha, const double *a, const double *x, std::int64_t incx, double beta,
                     double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::spmv(queue, upper_lower, n, alpha, a, x, incx, beta, y,
                                               incy, dependencies);
}

cl::sycl::event spr(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                    float alpha, const float *x, std::int64_t incx, float *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::spr(queue, upper_lower, n, alpha, x, incx, a,
                                              dependencies);
}

cl::sycl::event spr(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                    double alpha, const double *x, std::int64_t incx, double *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::spr(queue, upper_lower, n, alpha, x, incx, a,
                                              dependencies);
}

cl::sycl::event spr2(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     float alpha, const float *x, std::int64_t incx, const float *y,
                     std::int64_t incy, float *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::spr2(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                               dependencies);
}

cl::sycl::event spr2(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, std::int64_t n,
                     double alpha, const double *x, std::int64_t incx, const double *y,
                     std::int64_t incy, double *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::spr2(queue, upper_lower, n, alpha, x, incx, y, incy, a,
                                               dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     std::int64_t k, const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda,
                                               x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     std::int64_t k, const double *a, std::int64_t lda, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda,
                                               x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     std::int64_t k, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda,
                                               x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     std::int64_t k, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda,
                                               x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     std::int64_t k, const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda,
                                               x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     std::int64_t k, const double *a, std::int64_t lda, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda,
                                               x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     std::int64_t k, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda,
                                               x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     std::int64_t k, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda,
                                               x, incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const float *a, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                               dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const double *a, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                               dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const std::complex<float> *a, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                               dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const std::complex<double> *a, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                               dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const float *a, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                               dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const double *a, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                               dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const std::complex<float> *a, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                               dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const std::complex<double> *a, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                               dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x,
                                               incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const double *a, std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x,
                                               incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x,
                                               incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x,
                                               incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const float *a, std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x,
                                               incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const double *a, std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x,
                                               incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x,
                                               incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t n,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x,
                                               incx, dependencies);
}

cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::dotc(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::dotc(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::dotu(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::dotu(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::iamax(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::iamax(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::iamax(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::iamax(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::iamin(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::iamin(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::iamin(queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::iamin(queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::asum(queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::asum(queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::asum(queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::asum(queue, n, x, incx, result, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                     std::int64_t incx, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                     std::int64_t incx, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x,
                           std::int64_t *incx, float **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::axpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                                     group_size, dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x,
                           std::int64_t *incx, double **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::axpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                                     group_size, dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::axpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                                     group_size, dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::axpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                                     group_size, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::copy(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::copy(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::copy(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::copy(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, float *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::dot(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                    const double *y, std::int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::dot(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, const float *x,
                       std::int64_t incx, const float *y, std::int64_t incy, float *result,
                       const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::sdsdot(queue, n, sb, x, incx, y, incy, result,
                                                 dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::dot(queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::nrm2(queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::nrm2(queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::nrm2(queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::nrm2(queue, n, x, incx, result, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                    std::int64_t incx, std::complex<float> *y, std::int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rot(queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                    std::int64_t incx, std::complex<double> *y, std::int64_t incy, double c,
                    double s, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rot(queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                    std::int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rot(queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
                    std::int64_t incy, double c, double s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rot(queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, float *a, float *b, float *c, float *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rotg(queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, double *a, double *b, double *c, double *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rotg(queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<float> *a, std::complex<float> *b,
                     float *c, std::complex<float> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rotg(queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<double> *a, std::complex<double> *b,
                     double *c, std::complex<double> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rotg(queue, a, b, c, s, dependencies);
}

cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy, float *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rotm(queue, n, x, incx, y, incy, param, dependencies);
}

cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy, double *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rotm(queue, n, x, incx, y, incy, param, dependencies);
}

cl::sycl::event rotmg(cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1,
                      float *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rotmg(queue, d1, d2, x1, y1, param, dependencies);
}

cl::sycl::event rotmg(cl::sycl::queue &queue, double *d1, double *d2, double *x1, double y1,
                      double *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::rotmg(queue, d1, d2, x1, y1, param, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::scal(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::scal(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::scal(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::scal(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::scal(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::scal(queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::swap(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::swap(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::swap(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::swap(queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose *transa,
                           oneapi::mkl::transpose *transb, std::int64_t *m, std::int64_t *n,
                           std::int64_t *k, float *alpha, const float **a, std::int64_t *lda,
                           const float **b, std::int64_t *ldb, float *beta, float **c,
                           std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     b, ldb, beta, c, ldc, group_count, group_size,
                                                     dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose *transa,
                           oneapi::mkl::transpose *transb, std::int64_t *m, std::int64_t *n,
                           std::int64_t *k, double *alpha, const double **a, std::int64_t *lda,
                           const double **b, std::int64_t *ldb, double *beta, double **c,
                           std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     b, ldb, beta, c, ldc, group_count, group_size,
                                                     dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose *transa,
                           oneapi::mkl::transpose *transb, std::int64_t *m, std::int64_t *n,
                           std::int64_t *k, std::complex<float> *alpha,
                           const std::complex<float> **a, std::int64_t *lda,
                           const std::complex<float> **b, std::int64_t *ldb,
                           std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     b, ldb, beta, c, ldc, group_count, group_size,
                                                     dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose *transa,
                           oneapi::mkl::transpose *transb, std::int64_t *m, std::int64_t *n,
                           std::int64_t *k, std::complex<double> *alpha,
                           const std::complex<double> **a, std::int64_t *lda,
                           const std::complex<double> **b, std::int64_t *ldb,
                           std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     b, ldb, beta, c, ldc, group_count, group_size,
                                                     dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                           oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                           std::int64_t k, float alpha, const float *a, std::int64_t lda,
                           std::int64_t stride_a, const float *b, std::int64_t ldb,
                           std::int64_t stride_b, float beta, float *c, std::int64_t ldc,
                           std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     stride_a, b, ldb, stride_b, beta, c, ldc,
                                                     stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                           oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                           std::int64_t k, double alpha, const double *a, std::int64_t lda,
                           std::int64_t stride_a, const double *b, std::int64_t ldb,
                           std::int64_t stride_b, double beta, double *c, std::int64_t ldc,
                           std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     stride_a, b, ldb, stride_b, beta, c, ldc,
                                                     stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                           oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                           std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<float> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                           std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     stride_a, b, ldb, stride_b, beta, c, ldc,
                                                     stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                           oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n,
                           std::int64_t k, std::complex<double> alpha,
                           const std::complex<double> *a, std::int64_t lda, std::int64_t stride_a,
                           const std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
                           std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                           std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                     stride_a, b, ldb, stride_b, beta, c, ldc,
                                                     stride_c, batch_size, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, std::int64_t n,
                      std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *b,
                      std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a,
                                                lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, std::int64_t n,
                      std::int64_t k, double alpha, const double *a, std::int64_t lda,
                      const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a,
                                                lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, std::int64_t n,
                      std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                      std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a,
                                                lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose transa, oneapi::mkl::transpose transb, std::int64_t n,
                      std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                      std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                      std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return oneapi::mkl::mklgpu::internal::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a,
                                                lda, b, ldb, beta, c, ldc, dependencies);
}

} // namespace mklgpu
} // namespace mkl
} // namespace oneapi
