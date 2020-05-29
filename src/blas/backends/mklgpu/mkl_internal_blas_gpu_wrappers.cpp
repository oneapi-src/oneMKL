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
#include <cstdint>

#include "include/allocator_helper.hpp"
#include "mkl_internal_blas_gpu_wrappers.hpp"
#include "mkl_internal_blas_sycl_gpu.hpp"

namespace onemkl {
namespace mklgpu {
namespace internal {

// Buffer APIs

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    mkl::gpu::sgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k, alpha,
                    a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    mkl::gpu::dgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k, alpha,
                    a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    mkl::gpu::cgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k, alpha,
                    a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    mkl::gpu::zgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k, alpha,
                    a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    mkl::gpu::ssymm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    mkl::gpu::dsymm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    mkl::gpu::csymm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    mkl::gpu::zsymm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void hemm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    mkl::gpu::chemm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void hemm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    mkl::gpu::zhemm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    mkl::gpu::ssyrk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    mkl::gpu::dsyrk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc) {
    mkl::gpu::csyrk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    mkl::gpu::zsyrk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void herk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc) {
    mkl::gpu::cherk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void herk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, std::int64_t n,
          std::int64_t k, double alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    mkl::gpu::zherk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
           std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    mkl::gpu::ssyr2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
           std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    mkl::gpu::dsyr2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    mkl::gpu::csyr2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    mkl::gpu::zsyr2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void her2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    mkl::gpu::cher2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void her2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    mkl::gpu::zher2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    mkl::gpu::strmm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    mkl::gpu::dtrmm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    mkl::gpu::ctrmm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    mkl::gpu::ztrmm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    mkl::gpu::strsm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    mkl::gpu::dtrsm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    mkl::gpu::ctrsm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    mkl::gpu::ztrsm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    mkl::gpu::sgemv(queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    mkl::gpu::dgemv(queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    mkl::gpu::cgemv(queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    mkl::gpu::zgemv(queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    mkl::gpu::sgbmv(queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                    incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    mkl::gpu::dgbmv(queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                    incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    mkl::gpu::cgbmv(queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                    incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    mkl::gpu::zgbmv(queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                    incy);
}

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
         std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    mkl::gpu::sger(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
         std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    mkl::gpu::dger(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    mkl::gpu::cgerc(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    mkl::gpu::zgerc(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    mkl::gpu::cgeru(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    mkl::gpu::zgeru(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void hbmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    mkl::gpu::chbmv(queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void hbmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    mkl::gpu::zhbmv(queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void hemv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    mkl::gpu::chemv(queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
}

void hemv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    mkl::gpu::zhemv(queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
}

void her(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    mkl::gpu::cher(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda);
}

void her(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    mkl::gpu::zher(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda);
}

void her2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    mkl::gpu::cher2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a, lda);
}

void her2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    mkl::gpu::zher2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a, lda);
}

void hpmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          std::int64_t incy) {
    mkl::gpu::chpmv(queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void hpmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    mkl::gpu::zhpmv(queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void hpr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a) {
    mkl::gpu::chpr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a);
}

void hpr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a) {
    mkl::gpu::zhpr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a);
}

void hpr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a) {
    mkl::gpu::chpr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a);
}

void hpr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a) {
    mkl::gpu::zhpr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a);
}

void sbmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    mkl::gpu::ssbmv(queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void sbmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    mkl::gpu::dsbmv(queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void spmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    mkl::gpu::sspmv(queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void spmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    mkl::gpu::dspmv(queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void spr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a) {
    mkl::gpu::sspr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a);
}

void spr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a) {
    mkl::gpu::dspr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a);
}

void spr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a) {
    mkl::gpu::sspr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a);
}

void spr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a) {
    mkl::gpu::dspr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a);
}

void symv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    mkl::gpu::ssymv(queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
}

void symv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    mkl::gpu::dsymv(queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
}

void syr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a,
         std::int64_t lda) {
    mkl::gpu::ssyr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda);
}

void syr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a,
         std::int64_t lda) {
    mkl::gpu::dsyr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda);
}

void syr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    mkl::gpu::ssyr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a, lda);
}

void syr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    mkl::gpu::dsyr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a, lda);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    mkl::gpu::stbmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    mkl::gpu::dtbmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    mkl::gpu::ctbmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    mkl::gpu::ztbmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    mkl::gpu::stbsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    mkl::gpu::dtbsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    mkl::gpu::ctbsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    mkl::gpu::ztbsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    mkl::gpu::stpmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    mkl::gpu::dtpmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    mkl::gpu::ctpmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    mkl::gpu::ztpmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    mkl::gpu::stpsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    mkl::gpu::dtpsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    mkl::gpu::ctpsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    mkl::gpu::ztpsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    mkl::gpu::strmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    mkl::gpu::dtrmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    mkl::gpu::ctrmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    mkl::gpu::ztrmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    mkl::gpu::strsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    mkl::gpu::dtrsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    mkl::gpu::ctrsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    mkl::gpu::ztrsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::scasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::dzasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::sasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::dasum(queue, n, x, incx, result);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    mkl::gpu::saxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    mkl::gpu::daxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    mkl::gpu::caxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    mkl::gpu::zaxpy(queue, n, alpha, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    mkl::gpu::scopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    mkl::gpu::dcopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    mkl::gpu::ccopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    mkl::gpu::zcopy(queue, n, x, incx, y, incy);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::sdot(queue, n, x, incx, y, incy, result);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::ddot(queue, n, x, incx, y, incy, result);
}

void sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, cl::sycl::buffer<float, 1> &x,
            std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
            cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::sdsdot(queue, n, sb, x, incx, y, incy, result);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::dsdot(queue, n, x, incx, y, incy, result);
}

void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    mkl::gpu::cdotc(queue, n, x, incx, y, incy, result);
}

void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    mkl::gpu::zdotc(queue, n, x, incx, y, incy, result);
}

void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    mkl::gpu::cdotu(queue, n, x, incx, y, incy, result);
}

void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    mkl::gpu::zdotu(queue, n, x, incx, y, incy, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::scnrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::dznrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::snrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::dnrm2(queue, n, x, incx, result);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
         std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
         float s) {
    mkl::gpu::csrot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
         std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
         double c, double s) {
    mkl::gpu::zdrot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s) {
    mkl::gpu::srot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s) {
    mkl::gpu::drot(queue, n, x, incx, y, incy, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &b,
          cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<float, 1> &s) {
    mkl::gpu::srotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &b,
          cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<double, 1> &s) {
    mkl::gpu::drotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<std::complex<float>, 1> &s) {
    mkl::gpu::crotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<std::complex<double>, 1> &s) {
    mkl::gpu::zrotg(queue, a, b, c, s);
}

void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &param) {
    mkl::gpu::srotm(queue, n, x, incx, y, incy, param);
}

void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &param) {
    mkl::gpu::drotm(queue, n, x, incx, y, incy, param);
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1, cl::sycl::buffer<float, 1> &d2,
           cl::sycl::buffer<float, 1> &x1, float y1, cl::sycl::buffer<float, 1> &param) {
    mkl::gpu::srotmg(queue, d1, d2, x1, y1, param);
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1, cl::sycl::buffer<double, 1> &d2,
           cl::sycl::buffer<double, 1> &x1, double y1, cl::sycl::buffer<double, 1> &param) {
    mkl::gpu::drotmg(queue, d1, d2, x1, y1, param);
}

void scal(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    mkl::gpu::sscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    mkl::gpu::dscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    mkl::gpu::cscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    mkl::gpu::zscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    mkl::gpu::csscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    mkl::gpu::zdscal(queue, n, alpha, x, incx);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    mkl::gpu::sswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    mkl::gpu::dswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    mkl::gpu::cswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    mkl::gpu::zswap(queue, n, x, incx, y, incy);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    mkl::gpu::isamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    mkl::gpu::idamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    mkl::gpu::icamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    mkl::gpu::izamax(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    mkl::gpu::isamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    mkl::gpu::idamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    mkl::gpu::icamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    mkl::gpu::izamin(queue, n, x, incx, result);
}

void gemm_batch(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, float beta,
                cl::sycl::buffer<float, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    mkl::gpu::sgemm_batch(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k,
                          alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                          batch_size);
}

void gemm_batch(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                cl::sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    mkl::gpu::dgemm_batch(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k,
                          alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                          batch_size);
}

void gemm_batch(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    mkl::gpu::cgemm_batch(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k,
                          alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                          batch_size);
}

void gemm_batch(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    mkl::gpu::zgemm_batch(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k,
                          alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                          batch_size);
}

void trsm_batch(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b,
                std::int64_t batch_size) {
    mkl::gpu::strsm_batch(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                          mkl::cblas_convert(trans), mkl::cblas_convert(unit_diag), m, n, alpha, a,
                          lda, stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    mkl::gpu::dtrsm_batch(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                          mkl::cblas_convert(trans), mkl::cblas_convert(unit_diag), m, n, alpha, a,
                          lda, stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    mkl::gpu::ctrsm_batch(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                          mkl::cblas_convert(trans), mkl::cblas_convert(unit_diag), m, n, alpha, a,
                          lda, stride_a, b, ldb, stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                onemkl::transpose trans, onemkl::diag unit_diag, std::int64_t m, std::int64_t n,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    mkl::gpu::ztrsm_batch(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                          mkl::cblas_convert(trans), mkl::cblas_convert(unit_diag), m, n, alpha, a,
                          lda, stride_a, b, ldb, stride_b, batch_size);
}

void gemmt(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
           onemkl::transpose transb, std::int64_t n, std::int64_t k, float alpha,
           cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
           std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    mkl::gpu::sgemmt(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                     mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
           onemkl::transpose transb, std::int64_t n, std::int64_t k, double alpha,
           cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
           std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    mkl::gpu::dgemmt(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                     mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
           onemkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    mkl::gpu::zgemmt(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                     mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
           onemkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    mkl::gpu::cgemmt(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                     mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
          cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    mkl::gpu::hgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k, alpha,
                    a, lda, b, ldb, beta, c, ldc);
}

void gemm_ext(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<half, 1> &a, std::int64_t lda, cl::sycl::buffer<half, 1> &b,
              std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    mkl::gpu::gemm_f16f16f32(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k,
                             alpha, a, lda, b, ldb, beta, c, ldc);
}

template <typename T_src, typename T_dest>
static inline void copy_mat(T_src &src, onemkl::transpose trans, int row, int col, int ld,
                            T_dest off, T_dest *&dest) {
    int i, j;
    if (trans == onemkl::transpose::nontrans) {
        for (j = 0; j < col; j++) {
            for (i = 0; i < row; i++) {
                dest[i + ld * j] = (T_dest)src[i + ld * j] - off;
            }
        }
    }
    else {
        for (i = 0; i < row; i++) {
            for (j = 0; j < col; j++) {
                dest[i * ld + j] = (T_dest)src[i * ld + j] - off;
            }
        }
    }
}

template <typename T_src, typename T_dest, typename T_off>
static inline void copy_mat(T_src &src, int row, int col, int ld, onemkl::offset off_kind,
                            T_off off, T_dest &dest) {
    using T_data = typename std::remove_reference<decltype(dest[0])>::type;
    int i, j;
    T_data tmp;

    if (off_kind == onemkl::offset::fix) {
        tmp = off[0];
        for (j = 0; j < col; j++) {
            for (i = 0; i < row; i++) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else if (off_kind == onemkl::offset::column) {
        for (j = 0; j < col; j++) {
            for (i = 0; i < row; i++) {
                tmp              = off[i];
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else {
        for (j = 0; j < col; j++) {
            tmp = off[j];
            for (i = 0; i < row; i++) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
}

void gemm_ext(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
              onemkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
              cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
              cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
              cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
    // DGEMM is used for reference implementation to maximize accuracy.
    // Optimized implementation for specific architectures will be added in future releases.
    std::int64_t sizea, sizeb, sizec;
    sizea      = (transa == onemkl::transpose::nontrans) ? lda * k : lda * m;
    sizeb      = (transb == onemkl::transpose::nontrans) ? ldb * n : ldb * k;
    sizec      = ldc * n;
    double *ad = (double *)onemkl::aligned_alloc(64, sizeof(double) * sizea);
    double *bd = (double *)onemkl::aligned_alloc(64, sizeof(double) * sizeb);
    double *cd = (double *)onemkl::aligned_alloc(64, sizeof(double) * sizec);
    {
        double alphad = alpha;
        double betad  = beta;
        double aod    = ao;
        double bod    = bo;
        auto acc_a    = a.template get_access<cl::sycl::access::mode::read>();
        auto acc_b    = b.template get_access<cl::sycl::access::mode::read>();
        auto acc_c    = c.template get_access<cl::sycl::access::mode::read_write>();
        auto acc_co   = co.template get_access<cl::sycl::access::mode::read_write>();
        copy_mat(acc_a, transa, m, k, lda, aod, ad);
        copy_mat(acc_b, transb, k, n, ldb, bod, bd);
        copy_mat(acc_c, onemkl::transpose::nontrans, m, n, ldc, 0.0, cd);
        cl::sycl::buffer<double, 1> A_buf(ad, sizea);
        cl::sycl::buffer<double, 1> B_buf(bd, sizeb);
        cl::sycl::buffer<double, 1> C_buf(cd, sizec);
        mkl::gpu::dgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k,
                        alphad, A_buf, lda, B_buf, ldb, betad, C_buf, ldc);
        auto acc_cd = C_buf.template get_access<cl::sycl::access::mode::read>();
        copy_mat(acc_cd, m, n, ldc, offsetc, acc_co, acc_c);
    }
    onemkl::aligned_free(ad);
    onemkl::aligned_free(bd);
    onemkl::aligned_free(cd);
}

// USM APIs

cl::sycl::event gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                     std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float *a,
                     std::int64_t lda, const float *b, std::int64_t ldb, float beta, float *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sgemm_sycl(&queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m,
                                n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                     std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double *a,
                     std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dgemm_sycl(&queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m,
                                n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                     std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                     std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cgemm_sycl(&queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m,
                                n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb,
                     std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                     std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zgemm_sycl(&queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m,
                                n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     std::int64_t m, std::int64_t n, float alpha, const float *a, std::int64_t lda,
                     const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ssymm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta,
                                c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     std::int64_t m, std::int64_t n, double alpha, const double *a,
                     std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dsymm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta,
                                c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     std::int64_t m, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                     std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::csymm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta,
                                c, ldc, dependencies);
}

cl::sycl::event symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     std::int64_t m, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                     std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zsymm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta,
                                c, ldc, dependencies);
}

cl::sycl::event hemm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     std::int64_t m, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                     std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::chemm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta,
                                c, ldc, dependencies);
}

cl::sycl::event hemm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     std::int64_t m, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                     std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zhemm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta,
                                c, ldc, dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                     std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                     float beta, float *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ssyrk_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                n, k, alpha, a, lda, beta, c, ldc, dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                     std::int64_t n, std::int64_t k, double alpha, const double *a,
                     std::int64_t lda, double beta, double *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dsyrk_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                n, k, alpha, a, lda, beta, c, ldc, dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                     std::int64_t n, std::int64_t k, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> beta,
                     std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::csyrk_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                n, k, alpha, a, lda, beta, c, ldc, dependencies);
}

cl::sycl::event syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                     std::int64_t n, std::int64_t k, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> beta,
                     std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zsyrk_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                n, k, alpha, a, lda, beta, c, ldc, dependencies);
}

cl::sycl::event herk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                     std::int64_t n, std::int64_t k, float alpha, const std::complex<float> *a,
                     std::int64_t lda, float beta, std::complex<float> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cherk_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                n, k, alpha, a, lda, beta, c, ldc, dependencies);
}

cl::sycl::event herk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                     std::int64_t n, std::int64_t k, double alpha, const std::complex<double> *a,
                     std::int64_t lda, double beta, std::complex<double> *c, std::int64_t ldc,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zherk_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                n, k, alpha, a, lda, beta, c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                      std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                      const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ssyr2k_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                 n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                      std::int64_t n, std::int64_t k, double alpha, const double *a,
                      std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dsyr2k_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                 n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                      std::int64_t n, std::int64_t k, std::complex<float> alpha,
                      const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                      std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::csyr2k_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                 n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                      std::int64_t n, std::int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zsyr2k_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                 n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event her2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                      std::int64_t n, std::int64_t k, std::complex<float> alpha,
                      const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                      std::int64_t ldb, float beta, std::complex<float> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cher2k_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                 n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event her2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans,
                      std::int64_t n, std::int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, double beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zher2k_sycl(&queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans),
                                 n, k, alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, float *b,
                     std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::strmm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                                mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda, double *b,
                     std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dtrmm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                                mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ctrmm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                                mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                dependencies);
}

cl::sycl::event trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ztrmm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                                mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, float *b,
                     std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::strsm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                                mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda, double *b,
                     std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dtrsm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                                mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ctrsm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                                mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                dependencies);
}

cl::sycl::event trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
                     onemkl::transpose transa, onemkl::diag unit_diag, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ztrsm_sycl(&queue, mkl::cblas_convert(left_right),
                                mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                                mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb,
                                dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sgemv_sycl(&queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx,
                                beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda,
                     const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dgemv_sycl(&queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx,
                                beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cgemv_sycl(&queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx,
                                beta, y, incy, dependencies);
}

cl::sycl::event gemv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                     std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zgemv_sycl(&queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx,
                                beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha, const float *a,
                     std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sgbmv_sycl(&queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x,
                                incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                     const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                     double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dgbmv_sycl(&queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x,
                                incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> beta, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cgbmv_sycl(&queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x,
                                incx, beta, y, incy, dependencies);
}

cl::sycl::event gbmv(cl::sycl::queue &queue, onemkl::transpose trans, std::int64_t m,
                     std::int64_t n, std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> beta, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zgbmv_sycl(&queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x,
                                incx, beta, y, incy, dependencies);
}

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sger_sycl(&queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
}

cl::sycl::event ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                    double *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dger_sycl(&queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
}

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cgerc_sycl(&queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
}

cl::sycl::event gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zgerc_sycl(&queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
}

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cgeru_sycl(&queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
}

cl::sycl::event geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zgeru_sycl(&queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
}

cl::sycl::event hbmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::int64_t k,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::chbmv_sycl(&queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx,
                                beta, y, incy, dependencies);
}

cl::sycl::event hbmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::int64_t k,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zhbmv_sycl(&queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx,
                                beta, y, incy, dependencies);
}

cl::sycl::event hemv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::chemv_sycl(&queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta,
                                y, incy, dependencies);
}

cl::sycl::event hemv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zhemv_sycl(&queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta,
                                y, incy, dependencies);
}

cl::sycl::event her(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
                    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cher_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda,
                               dependencies);
}

cl::sycl::event her(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
                    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                    std::int64_t lda, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zher_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda,
                               dependencies);
}

cl::sycl::event her2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cher2_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a,
                                lda, dependencies);
}

cl::sycl::event her2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zher2_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a,
                                lda, dependencies);
}

cl::sycl::event hpmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *a,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
                     std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::chpmv_sycl(&queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y,
                                incy, dependencies);
}

cl::sycl::event hpmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *a,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
                     std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zhpmv_sycl(&queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y,
                                incy, dependencies);
}

cl::sycl::event hpr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
                    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::chpr_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a,
                               dependencies);
}

cl::sycl::event hpr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
                    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zhpr_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a,
                               dependencies);
}

cl::sycl::event hpr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n,
                     std::complex<float> alpha, const std::complex<float> *x, std::int64_t incx,
                     const std::complex<float> *y, std::int64_t incy, std::complex<float> *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::chpr2_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a,
                                dependencies);
}

cl::sycl::event hpr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n,
                     std::complex<double> alpha, const std::complex<double> *x, std::int64_t incx,
                     const std::complex<double> *y, std::int64_t incy, std::complex<double> *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zhpr2_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a,
                                dependencies);
}

cl::sycl::event sbmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::int64_t k,
                     float alpha, const float *a, std::int64_t lda, const float *x,
                     std::int64_t incx, float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ssbmv_sycl(&queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx,
                                beta, y, incy, dependencies);
}

cl::sycl::event sbmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, std::int64_t k,
                     double alpha, const double *a, std::int64_t lda, const double *x,
                     std::int64_t incx, double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dsbmv_sycl(&queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx,
                                beta, y, incy, dependencies);
}

cl::sycl::event spmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, float alpha,
                     const float *a, const float *x, std::int64_t incx, float beta, float *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sspmv_sycl(&queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y,
                                incy, dependencies);
}

cl::sycl::event spmv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, double alpha,
                     const double *a, const double *x, std::int64_t incx, double beta, double *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dspmv_sycl(&queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y,
                                incy, dependencies);
}

cl::sycl::event spr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, float *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sspr_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a,
                               dependencies);
}

cl::sycl::event spr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, double *a,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dspr_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a,
                               dependencies);
}

cl::sycl::event spr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
                     const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sspr2_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a,
                                dependencies);
}

cl::sycl::event spr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
                     const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                     double *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dspr2_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a,
                                dependencies);
}

cl::sycl::event symv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, float alpha,
                     const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                     float beta, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ssymv_sycl(&queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta,
                                y, incy, dependencies);
}

cl::sycl::event symv(cl::sycl::queue &queue, onemkl::uplo uplo, std::int64_t n, double alpha,
                     const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                     double beta, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dsymv_sycl(&queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta,
                                y, incy, dependencies);
}

cl::sycl::event syr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
                    const float *x, std::int64_t incx, float *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ssyr_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda,
                               dependencies);
}

cl::sycl::event syr(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
                    const double *x, std::int64_t incx, double *a, std::int64_t lda,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dsyr_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda,
                               dependencies);
}

cl::sycl::event syr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, float alpha,
                     const float *x, std::int64_t incx, const float *y, std::int64_t incy, float *a,
                     std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ssyr2_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a,
                                lda, dependencies);
}

cl::sycl::event syr2(cl::sycl::queue &queue, onemkl::uplo upplo, std::int64_t n, double alpha,
                     const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                     double *a, std::int64_t lda,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dsyr2_sycl(&queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a,
                                lda, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, std::int64_t k, const float *a,
                     std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::stbmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, std::int64_t k, const double *a,
                     std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dtbmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ctbmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ztbmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, std::int64_t k, const float *a,
                     std::int64_t lda, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::stbsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, std::int64_t k, const double *a,
                     std::int64_t lda, double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dtbsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ctbsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, std::int64_t k,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ztbsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, k, a, lda, x, incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const float *a, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::stpmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const double *a, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dtpmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ctpmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ztpmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const float *a, float *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::stpsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const double *a, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dtpsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ctpsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ztpsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const float *a, std::int64_t lda, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::strmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const double *a, std::int64_t lda,
                     double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dtrmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ctrmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ztrmv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const float *a, std::int64_t lda, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::strsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const double *a, std::int64_t lda,
                     double *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dtrsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ctrsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans,
                     onemkl::diag diag, std::int64_t n, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ztrsv_sycl(&queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                                mkl::cblas_convert(diag), n, a, lda, x, incx, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::scasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dzasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event asum(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dasum_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                     std::int64_t incx, float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::saxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                     std::int64_t incx, double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::daxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *x, std::int64_t incx, std::complex<float> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::caxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *x, std::int64_t incx, std::complex<double> *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zaxpy_sycl(&queue, n, alpha, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::scopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dcopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ccopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event copy(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zcopy_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, float *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sdot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                    const double *y, std::int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::ddot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, const float *x,
                       std::int64_t incx, const float *y, std::int64_t incy, float *result,
                       const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sdsdot_sycl(&queue, n, sb, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dot(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                    const float *y, std::int64_t incy, double *result,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dsdot_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cdotc_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotc(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zdotc_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                     std::complex<float> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cdotu_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event dotu(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                     std::complex<double> *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zdotu_sycl(&queue, n, x, incx, y, incy, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                     std::int64_t incx, float *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::scnrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                     std::int64_t incx, double *result,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dznrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                     float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::snrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event nrm2(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                     double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dnrm2_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                    std::int64_t incx, std::complex<float> *y, std::int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::csrot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                    std::int64_t incx, std::complex<double> *y, std::int64_t incy, double c,
                    double s, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zdrot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                    std::int64_t incy, float c, float s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::srot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rot(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
                    std::int64_t incy, double c, double s,
                    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::drot_sycl(&queue, n, x, incx, y, incy, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, float *a, float *b, float *c, float *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::srotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, double *a, double *b, double *c, double *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::drotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<float> *a, std::complex<float> *b,
                     float *c, std::complex<float> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::crotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotg(cl::sycl::queue &queue, std::complex<double> *a, std::complex<double> *b,
                     double *c, std::complex<double> *s,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zrotg_sycl(&queue, a, b, c, s, dependencies);
}

cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy, float *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::srotm_sycl(&queue, n, x, incx, y, incy, param, dependencies);
}

cl::sycl::event rotm(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy, double *param,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::drotm_sycl(&queue, n, x, incx, y, incy, param, dependencies);
}

cl::sycl::event rotmg(cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1,
                      float *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::srotmg_sycl(&queue, d1, d2, x1, y1, param, dependencies);
}

cl::sycl::event rotmg(cl::sycl::queue &queue, double *d1, double *d2, double *x1, double y1,
                      double *param, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::drotmg_sycl(&queue, d1, d2, x1, y1, param, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, float *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, double *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                     std::complex<float> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                     std::complex<double> *x, std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, float alpha, std::complex<float> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::csscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event scal(cl::sycl::queue &queue, std::int64_t n, double alpha, std::complex<double> *x,
                     std::int64_t incx,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zdscal_sycl(&queue, n, alpha, x, incx, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
                     std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                     double *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event swap(cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                     std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zswap_sycl(&queue, n, x, incx, y, incy, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::isamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::idamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::icamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamax(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::izamax_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::isamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
                      std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::idamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::icamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event iamin(cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                      std::int64_t incx, std::int64_t *result,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::izamin_sycl(&queue, n, x, incx, result, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                           const float *a, std::int64_t lda, std::int64_t stride_a, const float *b,
                           std::int64_t ldb, std::int64_t stride_b, float beta, float *c,
                           std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sgemm_batch(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m,
                                 n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                 stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                           const double *a, std::int64_t lda, std::int64_t stride_a,
                           const double *b, std::int64_t ldb, std::int64_t stride_b, double beta,
                           double *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dgemm_batch(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m,
                                 n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                 stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, const std::complex<float> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<float> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                           std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cgemm_batch(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m,
                                 n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                 stride_c, batch_size, dependencies);
}

cl::sycl::event gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, const std::complex<double> *a,
                           std::int64_t lda, std::int64_t stride_a, const std::complex<double> *b,
                           std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                           std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
                           std::int64_t batch_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zgemm_batch(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m,
                                 n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc,
                                 stride_c, batch_size, dependencies);
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
            cgh.single_task<class coalesce_events_kernel>([]() {
            });
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
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(mkl::gpu::sgemm_batch(
            queue, mkl::cblas_convert(transa[i]), mkl::cblas_convert(transb[i]), m[i], n[i], k[i],
            alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size, group_size[i],
            dependencies));
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
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(mkl::gpu::dgemm_batch(
            queue, mkl::cblas_convert(transa[i]), mkl::cblas_convert(transb[i]), m[i], n[i], k[i],
            alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size, group_size[i],
            dependencies));
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
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(mkl::gpu::cgemm_batch(
            queue, mkl::cblas_convert(transa[i]), mkl::cblas_convert(transb[i]), m[i], n[i], k[i],
            alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size, group_size[i],
            dependencies));
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
        cl::sycl::event *gemm_batch_event = new cl::sycl::event(mkl::gpu::zgemm_batch(
            queue, mkl::cblas_convert(transa[i]), mkl::cblas_convert(transb[i]), m[i], n[i], k[i],
            alpha[i], a, lda[i], b, ldb[i], beta[i], c, ldc[i], total_group_size, group_size[i],
            dependencies));
        coalesced_events.push_back(gemm_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x,
                           std::int64_t *incx, float **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *axpy_batch_event = new cl::sycl::event(
            mkl::gpu::saxpy_batch_sycl(&queue, n[i], alpha[i], x, incx[i], y, incy[i],
                                       group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(axpy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x,
                           std::int64_t *incx, double **y, std::int64_t *incy,
                           std::int64_t group_count, std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *axpy_batch_event = new cl::sycl::event(
            mkl::gpu::daxpy_batch_sycl(&queue, n[i], alpha[i], x, incx[i], y, incy[i],
                                       group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(axpy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
                           const std::complex<float> **x, std::int64_t *incx,
                           std::complex<float> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *axpy_batch_event = new cl::sycl::event(
            mkl::gpu::caxpy_batch_sycl(&queue, n[i], alpha[i], x, incx[i], y, incy[i],
                                       group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(axpy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event axpy_batch(cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
                           const std::complex<double> **x, std::int64_t *incx,
                           std::complex<double> **y, std::int64_t *incy, std::int64_t group_count,
                           std::int64_t *group_size,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    std::vector<cl::sycl::event *> coalesced_events;
    coalesced_events.reserve(group_count);
    std::int64_t total_group_size = 0;
    for (std::int64_t i = 0; i < group_count; i++) {
        cl::sycl::event *axpy_batch_event = new cl::sycl::event(
            mkl::gpu::zaxpy_batch_sycl(&queue, n[i], alpha[i], x, incx[i], y, incy[i],
                                       group_size[i], total_group_size, dependencies));
        coalesced_events.push_back(axpy_batch_event);
        total_group_size += group_size[i];
    }
    return *coalesce_events(queue, coalesced_events);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                      const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::sgemmt_sycl(&queue, mkl::cblas_convert(upper_lower),
                                 mkl::cblas_convert(transa), mkl::cblas_convert(transb), n, k,
                                 alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, double alpha, const double *a,
                      std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::dgemmt_sycl(&queue, mkl::cblas_convert(upper_lower),
                                 mkl::cblas_convert(transa), mkl::cblas_convert(transb), n, k,
                                 alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, std::complex<float> alpha,
                      const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                      std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                      std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::cgemmt_sycl(&queue, mkl::cblas_convert(upper_lower),
                                 mkl::cblas_convert(transa), mkl::cblas_convert(transb), n, k,
                                 alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      std::int64_t n, std::int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, std::int64_t lda,
                      const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
                      std::complex<double> *c, std::int64_t ldc,
                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    return mkl::gpu::zgemmt_sycl(&queue, mkl::cblas_convert(upper_lower),
                                 mkl::cblas_convert(transa), mkl::cblas_convert(transb), n, k,
                                 alpha, a, lda, b, ldb, beta, c, ldc, dependencies);
}

} //namespace internal
} //namespace mklgpu
} //namespace onemkl
