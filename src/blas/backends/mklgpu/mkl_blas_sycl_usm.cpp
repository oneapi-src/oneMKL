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

#include "oneapi/mkl/blas/detail/mklgpu/onemkl_blas_mklgpu.hpp"
#include "oneapi/mkl/types.hpp"
#include "mkl_internal_blas_sycl_gpu.hpp"

namespace oneapi {
namespace mkl {
namespace mklgpu {
namespace column_major {

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     float alpha, const float *a, std::int64_t lda, const float *b,
                     std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     double alpha, const double *a, std::int64_t lda, const double *b,
                     std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, const float *b, std::int64_t ldb, float beta,
                     float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssymm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                     double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsymm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csymm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zsymm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event hemm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chemm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event hemm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhemm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, float alpha,
                     const float *a, std::int64_t lda, float beta, float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyrk_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, double alpha,
                     const double *a, std::int64_t lda, double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyrk_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csyrk_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zsyrk_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event herk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, float alpha,
                     const std::complex<float> *a, std::int64_t lda, float beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cherk_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event herk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, double alpha,
                     const std::complex<double> *a, std::int64_t lda, double beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zherk_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, float alpha,
                      const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
                      float beta, float *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyr2k_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, double alpha,
                      const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                      double beta, double *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyr2k_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                      const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                      std::complex<float> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csyr2k_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zsyr2k_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event her2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                      const std::complex<float> *b, std::int64_t ldb, float beta,
                      std::complex<float> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cher2k_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event her2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, double beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zher2k_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, float *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strmm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, double *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrmm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrmm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrmm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, float *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strsm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, double *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrsm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrsm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrsm_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha,
                                  a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda,
                     const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha,
                                  a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha,
                                  a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                     std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha,
                                  a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha, const float *a,
                     std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku,
                                  alpha, a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                     const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                     double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku,
                                  alpha, a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku,
                                  alpha, a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku,
                                  alpha, a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sger_sycl(&queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                 dependencies);
}

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                    double *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dger_sycl(&queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                 dependencies);
}

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgerc_sycl(&queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                  dependencies);
}

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgerc_sycl(&queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                  dependencies);
}

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgeru_sycl(&queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                  dependencies);
}

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgeru_sycl(&queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                  dependencies);
}

cl::sycl::event hbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event hbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event hemv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chemv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event hemv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhemv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event her(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cher_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, lda, dependencies);
}

cl::sycl::event her(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zher_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, lda, dependencies);
}

cl::sycl::event her2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cher2_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, lda, dependencies);
}

cl::sycl::event her2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zher2_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, lda, dependencies);
}

cl::sycl::event hpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chpmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x,
                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event hpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhpmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x,
                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event hpr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chpr_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, dependencies);
}

cl::sycl::event hpr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhpr_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, dependencies);
}

cl::sycl::event hpr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chpr2_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, dependencies);
}

cl::sycl::event hpr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhpr2_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, dependencies);
}

cl::sycl::event sbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
                     float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event sbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
                     double alpha, const double *a, std::int64_t lda, const double *x,
                     std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event spmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                     const float *a, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sspmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x,
                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event spmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                     const double *a, const double *x, std::int64_t incx, double beta, double *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dspmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x,
                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event spr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, float *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sspr_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, dependencies);
}

cl::sycl::event spr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, double *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dspr_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, dependencies);
}

cl::sycl::event spr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                     const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sspr2_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, dependencies);
}

cl::sycl::event spr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                     const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                     double *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dspr2_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, dependencies);
}

cl::sycl::event symv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                     float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssymv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event symv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                     double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsymv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event syr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, float *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyr_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, lda, dependencies);
}

cl::sycl::event syr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, double *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyr_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, lda, dependencies);
}

cl::sycl::event syr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                     const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyr2_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, lda, dependencies);
}

cl::sycl::event syr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                     const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                     double *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyr2_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, lda, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, const float *a,
                     std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, const double *a,
                     std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztbmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, const float *a,
                     std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stbsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, const double *a,
                     std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtbsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctbsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztbsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const float *a, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stpmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const double *a, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtpmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctpmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztpmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const float *a, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stpsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const double *a, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtpsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctpsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztpsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const float *a, std::int64_t lda,
                     float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const double *a, std::int64_t lda,
                     double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrmv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const float *a, std::int64_t lda,
                     float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const double *a, std::int64_t lda,
                     double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrsv_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::scasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dzasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                     std::int64_t incx, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::saxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                     std::int64_t incx, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::daxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::caxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zaxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::scopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dcopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ccopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zcopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, float *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sdot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                    const double *y, std::int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ddot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, const float *x,
                       std::int64_t incx, const float *y, std::int64_t incy, float *result,
                       const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sdsdot_sycl(&queue, n, sb, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsdot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cdotc_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdotc_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cdotu_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdotu_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::scnrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dznrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::snrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dnrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                    std::int64_t incx, std::complex<float> *y, std::int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csrot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                    std::int64_t incx, std::complex<double> *y, std::int64_t incy, double c,
                    double s, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdrot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                    std::int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
                    std::int64_t incy, double c, double s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, float *a, float *b, float *c, float *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, double *a, double *b, double *c, double *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<float> *a, std::complex<float> *b,
                     float *c, std::complex<float> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::crotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<double> *a, std::complex<double> *b,
                     double *c, std::complex<double> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zrotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy, float *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srotm_sycl(&queue, n, x, incx, y, incy, param, dependencies);
}

cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy, double *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drotm_sycl(&queue, n, x, incx, y, incy, param, dependencies);
}

cl::sycl::event rotmg(cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1,
                      float *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srotmg_sycl(&queue, d1, d2, x1, y1, param, dependencies);
}

cl::sycl::event rotmg(cl::sycl::queue &queue, double *d1, double *d2, double *x1, double y1,
                      double *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drotmg_sycl(&queue, d1, d2, x1, y1, param, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::isamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::idamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::icamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::izamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::isamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::idamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::icamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::izamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                           const float *a, std::int64_t lda, std::int64_t stride_a, const float *b,
                           std::int64_t ldb, std::int64_t stride_b, float beta, float *c,
                           std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemm_batch_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                                        ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda,
                                        stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                        batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                           const double *a, std::int64_t lda, std::int64_t stride_a,
                           const double *b, std::int64_t ldb, std::int64_t stride_b, double beta,
                           double *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemm_batch_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                                        ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda,
                                        stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                        batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<float> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                           std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemm_batch_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                                        ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda,
                                        stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                        batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<double> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                           std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemm_batch_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                                        ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda,
                                        stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                        batch_size, dependencies);
}

cl::sycl::event *coalesce_events(cl::sycl::queue &queue, std::vector<cl::sycl::event *> &prereqs) {
#ifdef _WIN64
    for (std::int64_t i = 0; i < prereqs.size(); i++)
        prereqs[i]->wait();
    return new cl::sycl::event();
#else
    if (prereqs.size() > 0) {
        return new cl::sycl::event(queue.submit([&](cl::sycl::handler &cgh) {
            for (std::int64_t i = 0; i < prereqs.size(); i++)
                cgh.depends_on(*prereqs[i]);
            cgh.single_task<class coalesce_events_kernel_colmajor>([]() {});
        }));
    }
    else
        return new cl::sycl::event();
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha,
                           const float **a, std::int64_t *lda, const float **b, std::int64_t *ldb,
                           float *beta, float **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(::oneapi::mkl::gpu::sgemm_batch_sycl(
            &queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa[i]), ::mkl::cblas_convert(transb[i]),
            m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size,
            group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, double *alpha,
                           const double **a, std::int64_t *lda, const double **b, std::int64_t *ldb,
                           double *beta, double **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(::oneapi::mkl::gpu::dgemm_batch_sycl(
            &queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa[i]), ::mkl::cblas_convert(transb[i]),
            m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size,
            group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<float> *alpha, const std::complex<float> **a,
                           std::int64_t *lda, const std::complex<float> **b, std::int64_t *ldb,
                           std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(::oneapi::mkl::gpu::cgemm_batch_sycl(
            &queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa[i]), ::mkl::cblas_convert(transb[i]),
            m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size,
            group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           std::int64_t *lda, const std::complex<double> **b, std::int64_t *ldb,
                           std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(::oneapi::mkl::gpu::zgemm_batch_sycl(
            &queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa[i]), ::mkl::cblas_convert(transb[i]),
            m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size,
            group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x,
                           std::int64_t *incx, float **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::saxpy_batch(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                                   dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x,
                           std::int64_t *incx, double **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::daxpy_batch(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                                   dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::caxpy_batch(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                                   dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zaxpy_batch(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                                   dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                      const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemmt_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                                   alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, double alpha, const double *a,
                      std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemmt_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                                   alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, std::complex<float> alpha,
                      const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                      std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemmt_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                                   alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemmt_sycl(&queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                                   alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

} //namespace column_major
namespace row_major {

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     float alpha, const float *a, std::int64_t lda, const float *b,
                     std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     double alpha, const double *a, std::int64_t lda, const double *b,
                     std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                     oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, const float *b, std::int64_t ldb, float beta,
                     float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssymm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                     double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsymm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csymm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zsymm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event hemm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chemm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event hemm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhemm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                  beta, c, ldc, dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, float alpha,
                     const float *a, std::int64_t lda, float beta, float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyrk_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, double alpha,
                     const double *a, std::int64_t lda, double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyrk_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csyrk_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zsyrk_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event herk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, float alpha,
                     const std::complex<float> *a, std::int64_t lda, float beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cherk_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event herk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                     oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, double alpha,
                     const std::complex<double> *a, std::int64_t lda, double beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zherk_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                  ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                  dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, float alpha,
                      const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
                      float beta, float *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyr2k_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k, double alpha,
                      const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                      double beta, double *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyr2k_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                      const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
                      std::complex<float> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csyr2k_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zsyr2k_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event her2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                      const std::complex<float> *b, std::int64_t ldb, float beta,
                      std::complex<float> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cher2k_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event her2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower,
                      oneapi::mkl::transpose trans, std::int64_t n, std::int64_t k,
                      std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, double beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zher2k_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta,
                                   c, ldc, dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, float *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strmm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, double *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrmm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrmm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrmm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, float *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strsm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, double *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrsm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     std::complex<float> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrsm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right,
                     oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                     oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     std::complex<double> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrsm_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                                  ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                                  ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                  dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha,
                                  a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda,
                     const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha,
                                  a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha,
                                  a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                     std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha,
                                  a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha, const float *a,
                     std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku,
                                  alpha, a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                     const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                     double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku,
                                  alpha, a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku,
                                  alpha, a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku,
                                  alpha, a, lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sger_sycl(&queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                 dependencies);
}

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                    double *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dger_sycl(&queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                 dependencies);
}

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgerc_sycl(&queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                  dependencies);
}

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgerc_sycl(&queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                  dependencies);
}

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgeru_sycl(&queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                  dependencies);
}

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgeru_sycl(&queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda,
                                  dependencies);
}

cl::sycl::event hbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event hbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event hemv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chemv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event hemv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhemv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event her(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cher_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, lda, dependencies);
}

cl::sycl::event her(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zher_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, lda, dependencies);
}

cl::sycl::event her2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cher2_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, lda, dependencies);
}

cl::sycl::event her2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zher2_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, lda, dependencies);
}

cl::sycl::event hpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chpmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x,
                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event hpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhpmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x,
                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event hpr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chpr_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, dependencies);
}

cl::sycl::event hpr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhpr_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, dependencies);
}

cl::sycl::event hpr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chpr2_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, dependencies);
}

cl::sycl::event hpr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhpr2_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, dependencies);
}

cl::sycl::event sbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
                     float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event sbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
                     double alpha, const double *a, std::int64_t lda, const double *x,
                     std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event spmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                     const float *a, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sspmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x,
                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event spmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                     const double *a, const double *x, std::int64_t incx, double beta, double *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dspmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x,
                                  incx, beta, y, incy, dependencies);
}

cl::sycl::event spr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, float *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sspr_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, dependencies);
}

cl::sycl::event spr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, double *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dspr_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, dependencies);
}

cl::sycl::event spr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                     const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sspr2_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, dependencies);
}

cl::sycl::event spr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                     const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                     double *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dspr2_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, dependencies);
}

cl::sycl::event symv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                     float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssymv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event symv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                     double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsymv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a,
                                  lda, x, incx, beta, y, incy, dependencies);
}

cl::sycl::event syr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, float *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyr_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, lda, dependencies);
}

cl::sycl::event syr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, double *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyr_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                 incx, a, lda, dependencies);
}

cl::sycl::event syr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
                     const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyr2_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, lda, dependencies);
}

cl::sycl::event syr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
                     const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                     double *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyr2_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x,
                                  incx, y, incy, a, lda, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, const float *a,
                     std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, const double *a,
                     std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztbmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, const float *a,
                     std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stbsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, const double *a,
                     std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtbsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctbsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztbsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, k, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const float *a, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stpmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const double *a, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtpmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctpmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztpmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const float *a, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::stpsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const double *a, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtpsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctpsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztpsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a, x,
                                  incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const float *a, std::int64_t lda,
                     float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const double *a, std::int64_t lda,
                     double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrmv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const float *a, std::int64_t lda,
                     float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const double *a, std::int64_t lda,
                     double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
                     oneapi::mkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrsv_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo),
                                  ::mkl::cblas_convert(trans), ::mkl::cblas_convert(diag), n, a,
                                  lda, x, incx, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::scasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dzasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                     std::int64_t incx, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::saxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                     std::int64_t incx, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::daxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::caxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zaxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::scopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dcopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ccopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zcopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, float *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sdot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                    const double *y, std::int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ddot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, const float *x,
                       std::int64_t incx, const float *y, std::int64_t incy, float *result,
                       const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sdsdot_sycl(&queue, n, sb, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsdot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cdotc_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdotc_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cdotu_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdotu_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::scnrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dznrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::snrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dnrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                    std::int64_t incx, std::complex<float> *y, std::int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csrot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                    std::int64_t incx, std::complex<double> *y, std::int64_t incy, double c,
                    double s, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdrot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                    std::int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
                    std::int64_t incy, double c, double s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, float *a, float *b, float *c, float *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, double *a, double *b, double *c, double *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<float> *a, std::complex<float> *b,
                     float *c, std::complex<float> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::crotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<double> *a, std::complex<double> *b,
                     double *c, std::complex<double> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zrotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy, float *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srotm_sycl(&queue, n, x, incx, y, incy, param, dependencies);
}

cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy, double *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drotm_sycl(&queue, n, x, incx, y, incy, param, dependencies);
}

cl::sycl::event rotmg(cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1,
                      float *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::srotmg_sycl(&queue, d1, d2, x1, y1, param, dependencies);
}

cl::sycl::event rotmg(cl::sycl::queue &queue, double *d1, double *d2, double *x1, double y1,
                      double *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::drotmg_sycl(&queue, d1, d2, x1, y1, param, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zdscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::isamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::idamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::icamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::izamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::isamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::idamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::icamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::izamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                           const float *a, std::int64_t lda, std::int64_t stride_a, const float *b,
                           std::int64_t ldb, std::int64_t stride_b, float beta, float *c,
                           std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemm_batch_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                                        ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda,
                                        stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                        batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                           const double *a, std::int64_t lda, std::int64_t stride_a,
                           const double *b, std::int64_t ldb, std::int64_t stride_b, double beta,
                           double *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemm_batch_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                                        ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda,
                                        stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                        batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<float> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                           std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemm_batch_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                                        ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda,
                                        stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                        batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<double> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                           std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemm_batch_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                                        ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda,
                                        stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                        batch_size, dependencies);
}

cl::sycl::event *coalesce_events(cl::sycl::queue &queue, std::vector<cl::sycl::event *> &prereqs) {
#ifdef _WIN64
    for (std::int64_t i = 0; i < prereqs.size(); i++)
        prereqs[i]->wait();
    return new cl::sycl::event();
#else
    if (prereqs.size() > 0) {
        return new cl::sycl::event(queue.submit([&](cl::sycl::handler &cgh) {
            for (std::int64_t i = 0; i < prereqs.size(); i++)
                cgh.depends_on(*prereqs[i]);
            cgh.single_task<class coalesce_events_kernel_rowmajor>([]() {});
        }));
    }
    else
        return new cl::sycl::event();
#endif
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha,
                           const float **a, std::int64_t *lda, const float **b, std::int64_t *ldb,
                           float *beta, float **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(::oneapi::mkl::gpu::sgemm_batch_sycl(
            &queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa[i]), ::mkl::cblas_convert(transb[i]),
            m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size,
            group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k, double *alpha,
                           const double **a, std::int64_t *lda, const double **b, std::int64_t *ldb,
                           double *beta, double **c, std::int64_t *ldc, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(::oneapi::mkl::gpu::dgemm_batch_sycl(
            &queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa[i]), ::mkl::cblas_convert(transb[i]),
            m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size,
            group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<float> *alpha, const std::complex<float> **a,
                           std::int64_t *lda, const std::complex<float> **b, std::int64_t *ldb,
                           std::complex<float> *beta, std::complex<float> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(::oneapi::mkl::gpu::cgemm_batch_sycl(
            &queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa[i]), ::mkl::cblas_convert(transb[i]),
            m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size,
            group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose *transa, transpose *transb,
                           std::int64_t *m, std::int64_t *n, std::int64_t *k,
                           std::complex<double> *alpha, const std::complex<double> **a,
                           std::int64_t *lda, const std::complex<double> **b, std::int64_t *ldb,
                           std::complex<double> *beta, std::complex<double> **c, std::int64_t *ldc,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(::oneapi::mkl::gpu::zgemm_batch_sycl(
            &queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa[i]), ::mkl::cblas_convert(transb[i]),
            m[i], n[i], k[i], alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size,
            group_size[i], dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x,
                           std::int64_t *incx, float **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::saxpy_batch(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                                   dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x,
                           std::int64_t *incx, double **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::daxpy_batch(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                                   dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::caxpy_batch(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                                   dependencies);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zaxpy_batch(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                                   dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                      const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemmt_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                                   alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, double alpha, const double *a,
                      std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemmt_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                                   alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, std::complex<float> alpha,
                      const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                      std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemmt_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                                   alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemmt_sycl(&queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                                   ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k,
                                   alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

} //namespace row_major
} //namespace mklgpu
} //namespace mkl
} //namespace oneapi
