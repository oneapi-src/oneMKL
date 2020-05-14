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

#include "include/allocator_helper.hpp"
#include "mkl_internal_blas_gpu_wrappers.hpp"
#include "mkl_internal_blas_sycl_gpu.hpp"

namespace onemkl {
namespace mklgpu {
namespace internal {

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb, int64_t m,
          int64_t n, int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
          cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
          int64_t ldc) {
    mkl::gpu::sgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k, alpha,
                    a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb, int64_t m,
          int64_t n, int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
          cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
          int64_t ldc) {
    mkl::gpu::dgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k, alpha,
                    a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb, int64_t m,
          int64_t n, int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    mkl::gpu::cgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k, alpha,
                    a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb, int64_t m,
          int64_t n, int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    mkl::gpu::zgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k, alpha,
                    a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, int64_t m,
          int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
          cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
          int64_t ldc) {
    mkl::gpu::ssymm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, int64_t m,
          int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
          cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
          int64_t ldc) {
    mkl::gpu::dsymm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, int64_t m,
          int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    mkl::gpu::csymm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, int64_t m,
          int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    mkl::gpu::zsymm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void hemm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, int64_t m,
          int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    mkl::gpu::chemm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void hemm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower, int64_t m,
          int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    mkl::gpu::zhemm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower), m, n,
                    alpha, a, lda, b, ldb, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
          int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, float beta,
          cl::sycl::buffer<float, 1> &c, int64_t ldc) {
    mkl::gpu::ssyrk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
          int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, double beta,
          cl::sycl::buffer<double, 1> &c, int64_t ldc) {
    mkl::gpu::dsyrk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
          int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          int64_t lda, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          int64_t ldc) {
    mkl::gpu::csyrk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
          int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          int64_t lda, std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
          int64_t ldc) {
    mkl::gpu::zsyrk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void herk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
          int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          float beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    mkl::gpu::cherk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void herk(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
          int64_t k, double alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          double beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    mkl::gpu::zherk(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                    a, lda, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
           int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
           int64_t ldc) {
    mkl::gpu::ssyr2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
           int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
           int64_t ldc) {
    mkl::gpu::dsyr2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
           int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    mkl::gpu::csyr2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
           int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    mkl::gpu::zsyr2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void her2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
           int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, float beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    mkl::gpu::cher2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void her2k(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose trans, int64_t n,
           int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, double beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    mkl::gpu::zher2k(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(trans), n, k, alpha,
                     a, lda, b, ldb, beta, c, ldc);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, int64_t m, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb) {
    mkl::gpu::strmm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, int64_t m, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
          int64_t ldb) {
    mkl::gpu::dtrmm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, int64_t m, int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb) {
    mkl::gpu::ctrmm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trmm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, int64_t m, int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb) {
    mkl::gpu::ztrmm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, int64_t m, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb) {
    mkl::gpu::strsm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, int64_t m, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
          int64_t ldb) {
    mkl::gpu::dtrsm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, int64_t m, int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb) {
    mkl::gpu::ctrsm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void trsm(cl::sycl::queue &queue, onemkl::side left_right, onemkl::uplo upper_lower,
          onemkl::transpose transa, onemkl::diag unit_diag, int64_t m, int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb) {
    mkl::gpu::ztrsm(queue, mkl::cblas_convert(left_right), mkl::cblas_convert(upper_lower),
                    mkl::cblas_convert(transa), mkl::cblas_convert(unit_diag), m, n, alpha, a, lda,
                    b, ldb);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, int64_t m, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    mkl::gpu::sgemv(queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, int64_t m, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    mkl::gpu::dgemv(queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, int64_t m, int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    mkl::gpu::cgemv(queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, onemkl::transpose trans, int64_t m, int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    mkl::gpu::zgemv(queue, mkl::cblas_convert(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, int64_t m, int64_t n, int64_t kl,
          int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
          cl::sycl::buffer<float, 1> &x, int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
          int64_t incy) {
    mkl::gpu::sgbmv(queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                    incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, int64_t m, int64_t n, int64_t kl,
          int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
          cl::sycl::buffer<double, 1> &x, int64_t incx, double beta, cl::sycl::buffer<double, 1> &y,
          int64_t incy) {
    mkl::gpu::dgbmv(queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                    incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, int64_t m, int64_t n, int64_t kl,
          int64_t ku, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    mkl::gpu::cgbmv(queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                    incy);
}

void gbmv(cl::sycl::queue &queue, onemkl::transpose trans, int64_t m, int64_t n, int64_t kl,
          int64_t ku, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    mkl::gpu::zgbmv(queue, mkl::cblas_convert(trans), m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                    incy);
}

void ger(cl::sycl::queue &queue, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
         int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<float, 1> &a,
         int64_t lda) {
    mkl::gpu::sger(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void ger(cl::sycl::queue &queue, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
         int64_t incx, cl::sycl::buffer<double, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &a,
         int64_t lda) {
    mkl::gpu::dger(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    mkl::gpu::cgerc(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    mkl::gpu::zgerc(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    mkl::gpu::cgeru(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    mkl::gpu::zgeru(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

void hbmv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    mkl::gpu::chbmv(queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void hbmv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    mkl::gpu::zhbmv(queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void hemv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    mkl::gpu::chemv(queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
}

void hemv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    mkl::gpu::zhemv(queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
}

void her(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    mkl::gpu::cher(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda);
}

void her(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    mkl::gpu::zher(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda);
}

void her2(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda) {
    mkl::gpu::cher2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a, lda);
}

void her2(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda) {
    mkl::gpu::zher2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a, lda);
}

void hpmv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          int64_t incy) {
    mkl::gpu::chpmv(queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void hpmv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    mkl::gpu::zhpmv(queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void hpr(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a) {
    mkl::gpu::chpr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a);
}

void hpr(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a) {
    mkl::gpu::zhpr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a);
}

void hpr2(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a) {
    mkl::gpu::chpr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a);
}

void hpr2(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a) {
    mkl::gpu::zhpr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a);
}

void sbmv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    mkl::gpu::ssbmv(queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void sbmv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    mkl::gpu::dsbmv(queue, mkl::cblas_convert(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void spmv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, int64_t incy) {
    mkl::gpu::sspmv(queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void spmv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, int64_t incy) {
    mkl::gpu::dspmv(queue, mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta, y, incy);
}

void spr(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &a) {
    mkl::gpu::sspr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a);
}

void spr(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &a) {
    mkl::gpu::dspr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a);
}

void spr2(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
          cl::sycl::buffer<float, 1> &a) {
    mkl::gpu::sspr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a);
}

void spr2(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
          int64_t incy, cl::sycl::buffer<double, 1> &a) {
    mkl::gpu::dspr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a);
}

void symv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    mkl::gpu::ssymv(queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
}

void symv(cl::sycl::queue &queue, onemkl::uplo uplo, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x, int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    mkl::gpu::dsymv(queue, mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx, beta, y, incy);
}

void syr(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &a, int64_t lda) {
    mkl::gpu::ssyr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda);
}

void syr(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &a,
         int64_t lda) {
    mkl::gpu::dsyr(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, a, lda);
}

void syr2(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
          cl::sycl::buffer<float, 1> &a, int64_t lda) {
    mkl::gpu::ssyr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a, lda);
}

void syr2(cl::sycl::queue &queue, onemkl::uplo upplo, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
          int64_t incy, cl::sycl::buffer<double, 1> &a, int64_t lda) {
    mkl::gpu::dsyr2(queue, mkl::cblas_convert(upplo), n, alpha, x, incx, y, incy, a, lda);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda,
          cl::sycl::buffer<float, 1> &x, int64_t incx) {
    mkl::gpu::stbmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda,
          cl::sycl::buffer<double, 1> &x, int64_t incx) {
    mkl::gpu::dtbmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    mkl::gpu::ctbmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    mkl::gpu::ztbmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda,
          cl::sycl::buffer<float, 1> &x, int64_t incx) {
    mkl::gpu::stbsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda,
          cl::sycl::buffer<double, 1> &x, int64_t incx) {
    mkl::gpu::dtbsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    mkl::gpu::ctbsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    mkl::gpu::ztbsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, int64_t incx) {
    mkl::gpu::stpmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, int64_t incx) {
    mkl::gpu::dtpmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    mkl::gpu::ctpmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    mkl::gpu::ztpmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, int64_t incx) {
    mkl::gpu::stpsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, int64_t incx) {
    mkl::gpu::dtpsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    mkl::gpu::ctpsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    mkl::gpu::ztpsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
          int64_t incx) {
    mkl::gpu::strmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    mkl::gpu::dtrmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    mkl::gpu::ctrmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    mkl::gpu::ztrmv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
          int64_t incx) {
    mkl::gpu::strsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    mkl::gpu::dtrsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    mkl::gpu::ctrsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, onemkl::uplo upplo, onemkl::transpose trans, onemkl::diag diag,
          int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    mkl::gpu::ztrsv(queue, mkl::cblas_convert(upplo), mkl::cblas_convert(trans),
                    mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::scasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::dzasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::sasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::dasum(queue, n, x, incx, result);
}

void axpy(cl::sycl::queue &queue, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy) {
    mkl::gpu::saxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          int64_t incx, cl::sycl::buffer<double, 1> &y, int64_t incy) {
    mkl::gpu::daxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    mkl::gpu::caxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    mkl::gpu::zaxpy(queue, n, alpha, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &y, int64_t incy) {
    mkl::gpu::scopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &y, int64_t incy) {
    mkl::gpu::dcopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    mkl::gpu::ccopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    mkl::gpu::zcopy(queue, n, x, incx, y, incy);
}

void dot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
         cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::sdot(queue, n, x, incx, y, incy, result);
}

void dot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
         cl::sycl::buffer<double, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::ddot(queue, n, x, incx, y, incy, result);
}

void sdsdot(cl::sycl::queue &queue, int64_t n, float sb, cl::sycl::buffer<float, 1> &x,
            int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
            cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::sdsdot(queue, n, sb, x, incx, y, incy, result);
}

void dot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
         cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::dsdot(queue, n, x, incx, y, incy, result);
}

void dotc(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    mkl::gpu::cdotc(queue, n, x, incx, y, incy, result);
}

void dotc(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    mkl::gpu::zdotc(queue, n, x, incx, y, incy, result);
}

void dotu(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    mkl::gpu::cdotu(queue, n, x, incx, y, incy, result);
}

void dotu(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    mkl::gpu::zdotu(queue, n, x, incx, y, incy, result);
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::scnrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::dznrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    mkl::gpu::snrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    mkl::gpu::dnrm2(queue, n, x, incx, result);
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
         int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy, float c,
         float s) {
    mkl::gpu::csrot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
         int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy, double c,
         double s) {
    mkl::gpu::zdrot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
         cl::sycl::buffer<float, 1> &y, int64_t incy, float c, float s) {
    mkl::gpu::srot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
         cl::sycl::buffer<double, 1> &y, int64_t incy, double c, double s) {
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

void rotm(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<float, 1> &param) {
    mkl::gpu::srotm(queue, n, x, incx, y, incy, param);
}

void rotm(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &param) {
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

void scal(cl::sycl::queue &queue, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          int64_t incx) {
    mkl::gpu::sscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          int64_t incx) {
    mkl::gpu::dscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    mkl::gpu::cscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    mkl::gpu::zscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx) {
    mkl::gpu::csscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx) {
    mkl::gpu::zdscal(queue, n, alpha, x, incx);
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &y, int64_t incy) {
    mkl::gpu::sswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &y, int64_t incy) {
    mkl::gpu::dswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy) {
    mkl::gpu::cswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy) {
    mkl::gpu::zswap(queue, n, x, incx, y, incy);
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    mkl::gpu::isamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    mkl::gpu::idamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    mkl::gpu::icamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    mkl::gpu::izamax(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    mkl::gpu::isamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
           cl::sycl::buffer<int64_t, 1> &result) {
    mkl::gpu::idamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    mkl::gpu::icamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           int64_t incx, cl::sycl::buffer<int64_t, 1> &result) {
    mkl::gpu::izamin(queue, n, x, incx, result);
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
    auto transa_acc     = transa.get_access<cl::sycl::access::mode::read>();
    auto transb_acc     = transb.get_access<cl::sycl::access::mode::read>();
    auto m_acc          = m.get_access<cl::sycl::access::mode::read>();
    auto n_acc          = n.get_access<cl::sycl::access::mode::read>();
    auto k_acc          = k.get_access<cl::sycl::access::mode::read>();
    auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>();
    auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>();
    auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>();
    auto beta_acc       = beta.get_access<cl::sycl::access::mode::read>();
    auto ldc_acc        = ldc.get_access<cl::sycl::access::mode::read>();
    auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>();
    int64_t stride_a, stride_b, stride_c, off_a = 0, off_b = 0, off_c = 0;
    for (int64_t i = 0; i < group_count; i++) {
        stride_a =
            (transa_acc[i] == transpose::nontrans) ? lda_acc[i] * k_acc[i] : lda_acc[i] * m_acc[i];
        stride_b =
            (transb_acc[i] == transpose::nontrans) ? ldb_acc[i] * n_acc[i] : ldb_acc[i] * k_acc[i];
        stride_c = ldc_acc[i] * n_acc[i];
        mkl::gpu::sgemm_batch(
            queue, mkl::cblas_convert(transa_acc[i]), mkl::cblas_convert(transb_acc[i]), m_acc[i],
            n_acc[i], k_acc[i], alpha_acc[i], a, lda_acc[i], stride_a, b, ldb_acc[i], stride_b,
            beta_acc[i], c, ldc_acc[i], stride_c, group_size_acc[i], off_a, off_b, off_c);
        off_a += stride_a * group_size_acc[i];
        off_b += stride_b * group_size_acc[i];
        off_c += stride_c * group_size_acc[i];
    }
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
    auto transa_acc     = transa.get_access<cl::sycl::access::mode::read>();
    auto transb_acc     = transb.get_access<cl::sycl::access::mode::read>();
    auto m_acc          = m.get_access<cl::sycl::access::mode::read>();
    auto n_acc          = n.get_access<cl::sycl::access::mode::read>();
    auto k_acc          = k.get_access<cl::sycl::access::mode::read>();
    auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>();
    auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>();
    auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>();
    auto beta_acc       = beta.get_access<cl::sycl::access::mode::read>();
    auto ldc_acc        = ldc.get_access<cl::sycl::access::mode::read>();
    auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>();
    int64_t stride_a, stride_b, stride_c, off_a = 0, off_b = 0, off_c = 0;
    for (int64_t i = 0; i < group_count; i++) {
        stride_a =
            (transa_acc[i] == transpose::nontrans) ? lda_acc[i] * k_acc[i] : lda_acc[i] * m_acc[i];
        stride_b =
            (transb_acc[i] == transpose::nontrans) ? ldb_acc[i] * n_acc[i] : ldb_acc[i] * k_acc[i];
        stride_c = ldc_acc[i] * n_acc[i];
        mkl::gpu::dgemm_batch(
            queue, mkl::cblas_convert(transa_acc[i]), mkl::cblas_convert(transb_acc[i]), m_acc[i],
            n_acc[i], k_acc[i], alpha_acc[i], a, lda_acc[i], stride_a, b, ldb_acc[i], stride_b,
            beta_acc[i], c, ldc_acc[i], stride_c, group_size_acc[i], off_a, off_b, off_c);
        off_a += stride_a * group_size_acc[i];
        off_b += stride_b * group_size_acc[i];
        off_c += stride_c * group_size_acc[i];
    }
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
    auto transa_acc     = transa.get_access<cl::sycl::access::mode::read>();
    auto transb_acc     = transb.get_access<cl::sycl::access::mode::read>();
    auto m_acc          = m.get_access<cl::sycl::access::mode::read>();
    auto n_acc          = n.get_access<cl::sycl::access::mode::read>();
    auto k_acc          = k.get_access<cl::sycl::access::mode::read>();
    auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>();
    auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>();
    auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>();
    auto beta_acc       = beta.get_access<cl::sycl::access::mode::read>();
    auto ldc_acc        = ldc.get_access<cl::sycl::access::mode::read>();
    auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>();
    int64_t stride_a, stride_b, stride_c, off_a = 0, off_b = 0, off_c = 0;
    for (int64_t i = 0; i < group_count; i++) {
        stride_a =
            (transa_acc[i] == transpose::nontrans) ? lda_acc[i] * k_acc[i] : lda_acc[i] * m_acc[i];
        stride_b =
            (transb_acc[i] == transpose::nontrans) ? ldb_acc[i] * n_acc[i] : ldb_acc[i] * k_acc[i];
        stride_c = ldc_acc[i] * n_acc[i];
        mkl::gpu::cgemm_batch(
            queue, mkl::cblas_convert(transa_acc[i]), mkl::cblas_convert(transb_acc[i]), m_acc[i],
            n_acc[i], k_acc[i], alpha_acc[i], a, lda_acc[i], stride_a, b, ldb_acc[i], stride_b,
            beta_acc[i], c, ldc_acc[i], stride_c, group_size_acc[i], off_a, off_b, off_c);
        off_a += stride_a * group_size_acc[i];
        off_b += stride_b * group_size_acc[i];
        off_c += stride_c * group_size_acc[i];
    }
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
    auto transa_acc     = transa.get_access<cl::sycl::access::mode::read>();
    auto transb_acc     = transb.get_access<cl::sycl::access::mode::read>();
    auto m_acc          = m.get_access<cl::sycl::access::mode::read>();
    auto n_acc          = n.get_access<cl::sycl::access::mode::read>();
    auto k_acc          = k.get_access<cl::sycl::access::mode::read>();
    auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>();
    auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>();
    auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>();
    auto beta_acc       = beta.get_access<cl::sycl::access::mode::read>();
    auto ldc_acc        = ldc.get_access<cl::sycl::access::mode::read>();
    auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>();
    int64_t stride_a, stride_b, stride_c, off_a = 0, off_b = 0, off_c = 0;
    for (int64_t i = 0; i < group_count; i++) {
        stride_a =
            (transa_acc[i] == transpose::nontrans) ? lda_acc[i] * k_acc[i] : lda_acc[i] * m_acc[i];
        stride_b =
            (transb_acc[i] == transpose::nontrans) ? ldb_acc[i] * n_acc[i] : ldb_acc[i] * k_acc[i];
        stride_c = ldc_acc[i] * n_acc[i];
        mkl::gpu::zgemm_batch(
            queue, mkl::cblas_convert(transa_acc[i]), mkl::cblas_convert(transb_acc[i]), m_acc[i],
            n_acc[i], k_acc[i], alpha_acc[i], a, lda_acc[i], stride_a, b, ldb_acc[i], stride_b,
            beta_acc[i], c, ldc_acc[i], stride_c, group_size_acc[i], off_a, off_b, off_c);
        off_a += stride_a * group_size_acc[i];
        off_b += stride_b * group_size_acc[i];
        off_c += stride_c * group_size_acc[i];
    }
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

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<onemkl::side, 1> &left_right,
                cl::sycl::buffer<onemkl::uplo, 1> &upper_lower,
                cl::sycl::buffer<onemkl::transpose, 1> &trans,
                cl::sycl::buffer<onemkl::diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<float, 1> &alpha,
                cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    auto side_acc       = left_right.get_access<cl::sycl::access::mode::read>();
    auto uplo_acc       = upper_lower.get_access<cl::sycl::access::mode::read>();
    auto trans_acc      = trans.get_access<cl::sycl::access::mode::read>();
    auto diag_acc       = unit_diag.get_access<cl::sycl::access::mode::read>();
    auto m_acc          = m.get_access<cl::sycl::access::mode::read>();
    auto n_acc          = n.get_access<cl::sycl::access::mode::read>();
    auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>();
    auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>();
    auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>();
    auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>();
    int64_t stride_a, stride_b, off_a = 0, off_b = 0;
    for (int64_t i = 0; i < group_count; i++) {
        stride_a = (side_acc[i] == side::left) ? lda_acc[i] * m_acc[i] : lda_acc[i] * n_acc[i];
        stride_b = ldb_acc[i] * n_acc[i];
        mkl::gpu::strsm_batch(queue, mkl::cblas_convert(side_acc[i]),
                              mkl::cblas_convert(uplo_acc[i]), mkl::cblas_convert(trans_acc[i]),
                              mkl::cblas_convert(diag_acc[i]), m_acc[i], n_acc[i], alpha_acc[i], a,
                              lda_acc[i], stride_a, b, ldb_acc[i], stride_b, group_size_acc[i],
                              off_a, off_b);
        off_a += stride_a * group_size_acc[i];
        off_b += stride_b * group_size_acc[i];
    }
}

void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<onemkl::side, 1> &left_right,
                cl::sycl::buffer<onemkl::uplo, 1> &upper_lower,
                cl::sycl::buffer<onemkl::transpose, 1> &trans,
                cl::sycl::buffer<onemkl::diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
                cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<double, 1> &alpha,
                cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
                cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
                std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    auto side_acc       = left_right.get_access<cl::sycl::access::mode::read>();
    auto uplo_acc       = upper_lower.get_access<cl::sycl::access::mode::read>();
    auto trans_acc      = trans.get_access<cl::sycl::access::mode::read>();
    auto diag_acc       = unit_diag.get_access<cl::sycl::access::mode::read>();
    auto m_acc          = m.get_access<cl::sycl::access::mode::read>();
    auto n_acc          = n.get_access<cl::sycl::access::mode::read>();
    auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>();
    auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>();
    auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>();
    auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>();
    int64_t stride_a, stride_b, off_a = 0, off_b = 0;
    for (int64_t i = 0; i < group_count; i++) {
        stride_a = (side_acc[i] == side::left) ? lda_acc[i] * m_acc[i] : lda_acc[i] * n_acc[i];
        stride_b = ldb_acc[i] * n_acc[i];
        mkl::gpu::dtrsm_batch(queue, mkl::cblas_convert(side_acc[i]),
                              mkl::cblas_convert(uplo_acc[i]), mkl::cblas_convert(trans_acc[i]),
                              mkl::cblas_convert(diag_acc[i]), m_acc[i], n_acc[i], alpha_acc[i], a,
                              lda_acc[i], stride_a, b, ldb_acc[i], stride_b, group_size_acc[i],
                              off_a, off_b);
        off_a += stride_a * group_size_acc[i];
        off_b += stride_b * group_size_acc[i];
    }
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
    auto side_acc       = left_right.get_access<cl::sycl::access::mode::read>();
    auto uplo_acc       = upper_lower.get_access<cl::sycl::access::mode::read>();
    auto trans_acc      = trans.get_access<cl::sycl::access::mode::read>();
    auto diag_acc       = unit_diag.get_access<cl::sycl::access::mode::read>();
    auto m_acc          = m.get_access<cl::sycl::access::mode::read>();
    auto n_acc          = n.get_access<cl::sycl::access::mode::read>();
    auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>();
    auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>();
    auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>();
    auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>();
    int64_t stride_a, stride_b, off_a = 0, off_b = 0;
    for (int64_t i = 0; i < group_count; i++) {
        stride_a = (side_acc[i] == side::left) ? lda_acc[i] * m_acc[i] : lda_acc[i] * n_acc[i];
        stride_b = ldb_acc[i] * n_acc[i];
        mkl::gpu::ctrsm_batch(queue, mkl::cblas_convert(side_acc[i]),
                              mkl::cblas_convert(uplo_acc[i]), mkl::cblas_convert(trans_acc[i]),
                              mkl::cblas_convert(diag_acc[i]), m_acc[i], n_acc[i], alpha_acc[i], a,
                              lda_acc[i], stride_a, b, ldb_acc[i], stride_b, group_size_acc[i],
                              off_a, off_b);
        off_a += stride_a * group_size_acc[i];
        off_b += stride_b * group_size_acc[i];
    }
}

void trsm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<onemkl::side, 1> &left_right,
    cl::sycl::buffer<onemkl::uplo, 1> &upper_lower, cl::sycl::buffer<onemkl::transpose, 1> &trans,
    cl::sycl::buffer<onemkl::diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::complex<double>, 1> &alpha,
    cl::sycl::buffer<std::complex<double>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    auto side_acc       = left_right.get_access<cl::sycl::access::mode::read>();
    auto uplo_acc       = upper_lower.get_access<cl::sycl::access::mode::read>();
    auto trans_acc      = trans.get_access<cl::sycl::access::mode::read>();
    auto diag_acc       = unit_diag.get_access<cl::sycl::access::mode::read>();
    auto m_acc          = m.get_access<cl::sycl::access::mode::read>();
    auto n_acc          = n.get_access<cl::sycl::access::mode::read>();
    auto alpha_acc      = alpha.get_access<cl::sycl::access::mode::read>();
    auto lda_acc        = lda.get_access<cl::sycl::access::mode::read>();
    auto ldb_acc        = ldb.get_access<cl::sycl::access::mode::read>();
    auto group_size_acc = group_size.get_access<cl::sycl::access::mode::read>();
    int64_t stride_a, stride_b, off_a = 0, off_b = 0;
    for (int64_t i = 0; i < group_count; i++) {
        stride_a = (side_acc[i] == side::left) ? lda_acc[i] * m_acc[i] : lda_acc[i] * n_acc[i];
        stride_b = ldb_acc[i] * n_acc[i];
        mkl::gpu::ztrsm_batch(queue, mkl::cblas_convert(side_acc[i]),
                              mkl::cblas_convert(uplo_acc[i]), mkl::cblas_convert(trans_acc[i]),
                              mkl::cblas_convert(diag_acc[i]), m_acc[i], n_acc[i], alpha_acc[i], a,
                              lda_acc[i], stride_a, b, ldb_acc[i], stride_b, group_size_acc[i],
                              off_a, off_b);
        off_a += stride_a * group_size_acc[i];
        off_b += stride_b * group_size_acc[i];
    }
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
           onemkl::transpose transb, int64_t n, int64_t k, float alpha,
           cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb,
           float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc) {
    mkl::gpu::sgemmt(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                     mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
           onemkl::transpose transb, int64_t n, int64_t k, double alpha,
           cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b, int64_t ldb,
           double beta, cl::sycl::buffer<double, 1> &c, int64_t ldc) {
    mkl::gpu::dgemmt(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                     mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
           onemkl::transpose transb, int64_t n, int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    mkl::gpu::zgemmt(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                     mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, onemkl::uplo upper_lower, onemkl::transpose transa,
           onemkl::transpose transb, int64_t n, int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    mkl::gpu::cgemmt(queue, mkl::cblas_convert(upper_lower), mkl::cblas_convert(transa),
                     mkl::cblas_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb, int64_t m,
          int64_t n, int64_t k, half alpha, cl::sycl::buffer<half, 1> &a, int64_t lda,
          cl::sycl::buffer<half, 1> &b, int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c,
          int64_t ldc) {
    mkl::gpu::hgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k, alpha,
                    a, lda, b, ldb, beta, c, ldc);
}

void gemm_ext(cl::sycl::queue &queue, onemkl::transpose transa, onemkl::transpose transb, int64_t m,
              int64_t n, int64_t k, float alpha, cl::sycl::buffer<half, 1> &a, int64_t lda,
              cl::sycl::buffer<half, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
              int64_t ldc) {
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
              onemkl::offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
              cl::sycl::buffer<int8_t, 1> &a, int64_t lda, int8_t ao,
              cl::sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo, float beta,
              cl::sycl::buffer<int32_t, 1> &c, int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
    // DGEMM is used for reference implementation to maximize accuracy.
    // Optimized implementation for specific architectures will be added in future releases.
    int64_t sizea, sizeb, sizec;
    sizea       = (transa == onemkl::transpose::nontrans) ? lda * k : lda * m;
    sizeb       = (transb == onemkl::transpose::nontrans) ? ldb * n : ldb * k;
    sizec       = ldc * n;
    double *ad  = (double *)onemkl::aligned_alloc(64, sizeof(double) * sizea);
    double *bd  = (double *)onemkl::aligned_alloc(64, sizeof(double) * sizeb);
    double *cd  = (double *)onemkl::aligned_alloc(64, sizeof(double) * sizec);
    double aod  = ao;
    double bod  = bo;
    auto acc_a  = a.template get_access<cl::sycl::access::mode::read>();
    auto acc_b  = b.template get_access<cl::sycl::access::mode::read>();
    auto acc_c  = c.template get_access<cl::sycl::access::mode::read_write>();
    auto acc_co = co.template get_access<cl::sycl::access::mode::read_write>();
    copy_mat(acc_a, transa, m, k, lda, aod, ad);
    copy_mat(acc_b, transb, k, n, ldb, bod, bd);
    copy_mat(acc_c, onemkl::transpose::nontrans, m, n, ldc, 0.0, cd);
    cl::sycl::buffer<double, 1> A_buf(ad, sizea);
    cl::sycl::buffer<double, 1> B_buf(bd, sizeb);
    cl::sycl::buffer<double, 1> C_buf(cd, sizec);
    mkl::gpu::dgemm(queue, mkl::cblas_convert(transa), mkl::cblas_convert(transb), m, n, k, alpha,
                    A_buf, lda, B_buf, ldb, beta, C_buf, ldc);
    auto acc_cd = C_buf.template get_access<cl::sycl::access::mode::read>();
    copy_mat(acc_cd, m, n, ldc, offsetc, acc_co, acc_c);
    onemkl::aligned_free(ad);
    onemkl::aligned_free(bd);
    onemkl::aligned_free(cd);
}

} //namespace internal
} //namespace mklgpu
} //namespace onemkl
