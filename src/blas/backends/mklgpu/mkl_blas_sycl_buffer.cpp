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
#include "onemkl/blas/detail/mklgpu/onemkl_blas_mklgpu.hpp"
#include "onemkl/types.hpp"

namespace onemkl {
namespace mklgpu {

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                   ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                   ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                   ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                   ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
          cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                   ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                   beta, c, ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                   beta, c, ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                   beta, c, ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    onemkl::mklgpu::internal::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                   beta, c, ldc);
}

void hemm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::hemm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                   beta, c, ldc);
}

void hemm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    onemkl::mklgpu::internal::hemm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                                   beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc) {
    onemkl::mklgpu::internal::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    onemkl::mklgpu::internal::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void herk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc) {
    onemkl::mklgpu::internal::herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void herk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, double alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    onemkl::mklgpu::internal::herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
           std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                                    ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
           std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                                    ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                                    ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    onemkl::mklgpu::internal::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                                    ldc);
}

void her2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                                    ldc);
}

void her2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                                    ldc);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    onemkl::mklgpu::internal::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a,
                                   lda, b, ldb);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    onemkl::mklgpu::internal::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a,
                                   lda, b, ldb);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    onemkl::mklgpu::internal::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a,
                                   lda, b, ldb);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    onemkl::mklgpu::internal::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a,
                                   lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    onemkl::mklgpu::internal::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a,
                                   lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    onemkl::mklgpu::internal::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a,
                                   lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    onemkl::mklgpu::internal::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a,
                                   lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    onemkl::mklgpu::internal::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a,
                                   lda, b, ldb);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    onemkl::mklgpu::internal::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                   incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                   incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                   incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    onemkl::mklgpu::internal::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                                   incy);
}

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
         std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::ger(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
         std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::ger(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::geru(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::geru(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void hbmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void hbmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    onemkl::mklgpu::internal::hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void hemv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

void hemv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    onemkl::mklgpu::internal::hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

void her(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::her(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void her(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::her(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void her2(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::her2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void her2(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::her2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void hpmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::hpmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void hpmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    onemkl::mklgpu::internal::hpmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void hpr(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a) {
    onemkl::mklgpu::internal::hpr(queue, upper_lower, n, alpha, x, incx, a);
}

void hpr(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a) {
    onemkl::mklgpu::internal::hpr(queue, upper_lower, n, alpha, x, incx, a);
}

void hpr2(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a) {
    onemkl::mklgpu::internal::hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void hpr2(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a) {
    onemkl::mklgpu::internal::hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void sbmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void sbmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, std::int64_t k,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void symv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

void symv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

void syr(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a,
         std::int64_t lda) {
    onemkl::mklgpu::internal::syr(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void syr(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a,
         std::int64_t lda) {
    onemkl::mklgpu::internal::syr(queue, upper_lower, n, alpha, x, incx, a, lda);
}

void syr2(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void syr2(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    onemkl::mklgpu::internal::syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

void spmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::spmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void spmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::spmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

void spr(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a) {
    onemkl::mklgpu::internal::spr(queue, upper_lower, n, alpha, x, incx, a);
}

void spr(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a) {
    onemkl::mklgpu::internal::spr(queue, upper_lower, n, alpha, x, incx, a);
}

void spr2(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a) {
    onemkl::mklgpu::internal::spr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void spr2(cl::sycl::queue &queue, onemkl::uplo upper_lower, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a) {
    onemkl::mklgpu::internal::spr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
          onemkl::diag unit_diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    onemkl::mklgpu::internal::dotc(queue, n, x, incx, y, incy, result);
}

void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    onemkl::mklgpu::internal::dotc(queue, n, x, incx, y, incy, result);
}

void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    onemkl::mklgpu::internal::dotu(queue, n, x, incx, y, incy, result);
}

void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    onemkl::mklgpu::internal::dotu(queue, n, x, incx, y, incy, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    onemkl::mklgpu::internal::iamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    onemkl::mklgpu::internal::iamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    onemkl::mklgpu::internal::iamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    onemkl::mklgpu::internal::iamax(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    onemkl::mklgpu::internal::iamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    onemkl::mklgpu::internal::iamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    onemkl::mklgpu::internal::iamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    onemkl::mklgpu::internal::iamin(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    onemkl::mklgpu::internal::asum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    onemkl::mklgpu::internal::asum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    onemkl::mklgpu::internal::asum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    onemkl::mklgpu::internal::asum(queue, n, x, incx, result);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::axpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::axpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::axpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::axpy(queue, n, alpha, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::copy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::copy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::copy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::copy(queue, n, x, incx, y, incy);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &result) {
    onemkl::mklgpu::internal::dot(queue, n, x, incx, y, incy, result);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &result) {
    onemkl::mklgpu::internal::dot(queue, n, x, incx, y, incy, result);
}

void sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, cl::sycl::buffer<float, 1> &x,
            std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
            cl::sycl::buffer<float, 1> &result) {
    onemkl::mklgpu::internal::sdsdot(queue, n, sb, x, incx, y, incy, result);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &result) {
    onemkl::mklgpu::internal::dot(queue, n, x, incx, y, incy, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    onemkl::mklgpu::internal::nrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    onemkl::mklgpu::internal::nrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    onemkl::mklgpu::internal::nrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    onemkl::mklgpu::internal::nrm2(queue, n, x, incx, result);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
         std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
         float s) {
    onemkl::mklgpu::internal::rot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
         std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
         double c, double s) {
    onemkl::mklgpu::internal::rot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s) {
    onemkl::mklgpu::internal::rot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s) {
    onemkl::mklgpu::internal::rot(queue, n, x, incx, y, incy, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &b,
          cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<float, 1> &s) {
    onemkl::mklgpu::internal::rotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &b,
          cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<double, 1> &s) {
    onemkl::mklgpu::internal::rotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<std::complex<float>, 1> &s) {
    onemkl::mklgpu::internal::rotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<std::complex<double>, 1> &s) {
    onemkl::mklgpu::internal::rotg(queue, a, b, c, s);
}

void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &param) {
    onemkl::mklgpu::internal::rotm(queue, n, x, incx, y, incy, param);
}

void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &param) {
    onemkl::mklgpu::internal::rotm(queue, n, x, incx, y, incy, param);
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1, cl::sycl::buffer<float, 1> &d2,
           cl::sycl::buffer<float, 1> &x1, float y1, cl::sycl::buffer<float, 1> &param) {
    onemkl::mklgpu::internal::rotmg(queue, d1, d2, x1, y1, param);
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1, cl::sycl::buffer<double, 1> &d2,
           cl::sycl::buffer<double, 1> &x1, double y1, cl::sycl::buffer<double, 1> &param) {
    onemkl::mklgpu::internal::rotmg(queue, d1, d2, x1, y1, param);
}

void scal(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    onemkl::mklgpu::internal::scal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    onemkl::mklgpu::internal::scal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::scal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::scal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::scal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    onemkl::mklgpu::internal::scal(queue, n, alpha, x, incx);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::swap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::swap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::swap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    onemkl::mklgpu::internal::swap(queue, n, x, incx, y, incy);
}

void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<onemkl::transpose, 1> &transa,
                cl::sycl::buffer<onemkl::transpose, 1> &transb,
                cl::sycl::buffer<std::int64_t, 1> &m, cl::sycl::buffer<std::int64_t, 1> &n,
                cl::sycl::buffer<std::int64_t, 1> &k, cl::sycl::buffer<float, 1> &alpha,
                cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                cl::sycl::buffer<float, 1> &beta, cl::sycl::buffer<float, 1> &c,
                cl::sycl::buffer<std::int64_t, 1> &ldc, std::int64_t group_count,
                cl::sycl::buffer<std::int64_t, 1> &group_size) {
    onemkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                         beta, c, ldc, group_count, group_size);
}

void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<onemkl::transpose, 1> &transa,
                cl::sycl::buffer<onemkl::transpose, 1> &transb,
                cl::sycl::buffer<std::int64_t, 1> &m, cl::sycl::buffer<std::int64_t, 1> &n,
                cl::sycl::buffer<std::int64_t, 1> &k, cl::sycl::buffer<double, 1> &alpha,
                cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                cl::sycl::buffer<double, 1> &beta, cl::sycl::buffer<double, 1> &c,
                cl::sycl::buffer<std::int64_t, 1> &ldc, std::int64_t group_count,
                cl::sycl::buffer<std::int64_t, 1> &group_size) {
    onemkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                         beta, c, ldc, group_count, group_size);
}

void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<onemkl::transpose, 1> &transa,
                cl::sycl::buffer<onemkl::transpose, 1> &transb,
                cl::sycl::buffer<std::int64_t, 1> &m, cl::sycl::buffer<std::int64_t, 1> &n,
                cl::sycl::buffer<std::int64_t, 1> &k,
                cl::sycl::buffer<std::complex<float>, 1> &alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                cl::sycl::buffer<std::complex<float>, 1> &beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    onemkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                         beta, c, ldc, group_count, group_size);
}

void gemm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<onemkl::transpose, 1> &transa,
    cl::sycl::buffer<onemkl::transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<double>, 1> &alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<double>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<double>, 1> &beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    onemkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                         beta, c, ldc, group_count, group_size);
}

void gemm_batch(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, float beta,
                cl::sycl::buffer<float, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    onemkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b,
                                         ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    onemkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b,
                                         ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    onemkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b,
                                         ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    onemkl::mklgpu::internal::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b,
                                         ldb, stride_b, beta, c, ldc, stride_c, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<onemkl::side, 1> &left_right,
                cl::sycl::buffer<onemkl::uplo, 1> &upper_lower,
                cl::sycl::buffer<onemkl::transpose, 1> &trans,
                cl::sycl::buffer<onemkl::diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<float, 1> &alpha,
                cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    onemkl::mklgpu::internal::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                         alpha, a, lda, b, ldb, group_count, group_size);
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<onemkl::side, 1> &left_right,
                cl::sycl::buffer<onemkl::uplo, 1> &upper_lower,
                cl::sycl::buffer<onemkl::transpose, 1> &trans,
                cl::sycl::buffer<onemkl::diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<double, 1> &alpha,
                cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    onemkl::mklgpu::internal::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                         alpha, a, lda, b, ldb, group_count, group_size);
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<onemkl::side, 1> &left_right,
                cl::sycl::buffer<onemkl::uplo, 1> &upper_lower,
                cl::sycl::buffer<onemkl::transpose, 1> &trans,
                cl::sycl::buffer<onemkl::diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n,
                cl::sycl::buffer<std::complex<float>, 1> &alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    onemkl::mklgpu::internal::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                         alpha, a, lda, b, ldb, group_count, group_size);
}

void trsm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<onemkl::side, 1> &left_right,
    cl::sycl::buffer<onemkl::uplo, 1> &upper_lower, cl::sycl::buffer<onemkl::transpose, 1> &trans,
    cl::sycl::buffer<onemkl::diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::complex<double>, 1> &alpha,
    cl::sycl::buffer<std::complex<double>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    onemkl::mklgpu::internal::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                         alpha, a, lda, b, ldb, group_count, group_size);
}

void trsm_batch(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    onemkl::mklgpu::internal::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                         alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    onemkl::mklgpu::internal::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                         alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    onemkl::mklgpu::internal::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                         alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    onemkl::mklgpu::internal::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                         alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
}

void gemmt(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
           onemkl::transpose transb, std::int64_t n, std::int64_t k, float alpha,
           cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
           std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb,
                                    beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
           onemkl::transpose transb, std::int64_t n, std::int64_t k, double alpha,
           cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
           std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb,
                                    beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
           onemkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb,
                                    beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
           onemkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    onemkl::mklgpu::internal::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb,
                                    beta, c, ldc);
}

void gemm_ext(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<half, 1> &a, std::int64_t lda, cl::sycl::buffer<half, 1> &b,
              std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemm_ext(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                       c, ldc);
}

void gemm_ext(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
              onemkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
              cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
              cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
    onemkl::mklgpu::internal::gemm_ext(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao,
                                       b, ldb, bo, beta, c, ldc, co);
}

void gemm_ext(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
              std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                   ldc);
}

void gemm_ext(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
              cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
              std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                   ldc);
}

void gemm_ext(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
              std::int64_t ldc) {
    onemkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                   ldc);
}

void gemm_ext(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
              std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
              std::int64_t ldc) {
    onemkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                   ldc);
}

void gemm_ext(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
              cl::sycl::buffer<half, 1> &a, std::int64_t lda, cl::sycl::buffer<half, 1> &b,
              std::int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    onemkl::mklgpu::internal::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                   ldc);
}

} // namespace mklgpu
} // namespace onemkl
