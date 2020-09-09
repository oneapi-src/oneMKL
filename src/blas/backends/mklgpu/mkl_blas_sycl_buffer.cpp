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

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::sgemm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::dgemm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::cgemm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zgemm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
          cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::hgemm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::gemm_f16f16f32(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                               ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta,
                               c, ldc);
}

void symm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::ssymm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::dsymm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::csymm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zsymm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void hemm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::chemm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void hemm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zhemm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::ssyrk(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::dsyrk(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::csyrk(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zsyrk(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void herk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc) {
    ::mkl::gpu::cherk(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void herk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zherk(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
           std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::ssyr2k(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
           std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::dsyr2k(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::csyr2k(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    ::mkl::gpu::zsyr2k(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void her2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::cher2k(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void her2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zher2k(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::strmm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::dtrmm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::ctrmm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::ztrmm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::strsm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::dtrsm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::ctrsm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::ztrsm(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::sgemv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dgemv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::cgemv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::zgemv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::sgbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku, alpha, a,
                      lda, x, incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dgbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku, alpha, a,
                      lda, x, incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::cgbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku, alpha, a,
                      lda, x, incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::zgbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku, alpha, a,
                      lda, x, incx, beta, y, incy);
}

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
         std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    ::mkl::gpu::sger(queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
         std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    ::mkl::gpu::dger(queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::cgerc(queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::zgerc(queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::cgeru(queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::zgeru(queue, MKL_COL_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void hbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::chbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void hbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::zhbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void hemv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::chemv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx,
                      beta, y, incy);
}

void hemv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::zhemv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx,
                      beta, y, incy);
}

void her(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::cher(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a, lda);
}

void her(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::zher(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a, lda);
}

void her2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::cher2(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a, lda);
}

void her2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::zher2(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a, lda);
}

void hpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::chpmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta,
                      y, incy);
}

void hpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::zhpmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta,
                      y, incy);
}

void hpr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a) {
    ::mkl::gpu::chpr(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a);
}

void hpr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a) {
    ::mkl::gpu::zhpr(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a);
}

void hpr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a) {
    ::mkl::gpu::chpr2(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a);
}

void hpr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a) {
    ::mkl::gpu::zhpr2(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a);
}

void sbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::ssbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void sbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dsbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void spmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::sspmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta,
                      y, incy);
}

void spmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dspmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta,
                      y, incy);
}

void spr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a) {
    ::mkl::gpu::sspr(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a);
}

void spr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a) {
    ::mkl::gpu::dspr(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a);
}

void spr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a) {
    ::mkl::gpu::sspr2(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a);
}

void spr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a) {
    ::mkl::gpu::dspr2(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a);
}

void symv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::ssymv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx,
                      beta, y, incy);
}

void symv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dsymv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx,
                      beta, y, incy);
}

void syr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a,
         std::int64_t lda) {
    ::mkl::gpu::ssyr(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a, lda);
}

void syr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a,
         std::int64_t lda) {
    ::mkl::gpu::dsyr(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a, lda);
}

void syr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    ::mkl::gpu::ssyr2(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a, lda);
}

void syr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    ::mkl::gpu::dsyr2(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a, lda);
}

void tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::stbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztbmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::stbsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtbsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctbsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztbsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::stpmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtpmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctpmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztpmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::stpsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtpsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctpsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztpsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::strmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtrmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctrmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztrmv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::strsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtrsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctrsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztrsv(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::scasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::dzasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::sasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::dasum(queue, n, x, incx, result);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::saxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::daxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::caxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::zaxpy(queue, n, alpha, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::scopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dcopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::ccopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::zcopy(queue, n, x, incx, y, incy);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::sdot(queue, n, x, incx, y, incy, result);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::ddot(queue, n, x, incx, y, incy, result);
}

void sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, cl::sycl::buffer<float, 1> &x,
            std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
            cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::sdsdot(queue, n, sb, x, incx, y, incy, result);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::dsdot(queue, n, x, incx, y, incy, result);
}

void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    ::mkl::gpu::cdotc(queue, n, x, incx, y, incy, result);
}

void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    ::mkl::gpu::zdotc(queue, n, x, incx, y, incy, result);
}

void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    ::mkl::gpu::cdotu(queue, n, x, incx, y, incy, result);
}

void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    ::mkl::gpu::zdotu(queue, n, x, incx, y, incy, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::scnrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::dznrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::snrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::dnrm2(queue, n, x, incx, result);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
         std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
         float s) {
    ::mkl::gpu::csrot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
         std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
         double c, double s) {
    ::mkl::gpu::zdrot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s) {
    ::mkl::gpu::srot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s) {
    ::mkl::gpu::drot(queue, n, x, incx, y, incy, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &b,
          cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<float, 1> &s) {
    ::mkl::gpu::srotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &b,
          cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<double, 1> &s) {
    ::mkl::gpu::drotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<std::complex<float>, 1> &s) {
    ::mkl::gpu::crotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<std::complex<double>, 1> &s) {
    ::mkl::gpu::zrotg(queue, a, b, c, s);
}

void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &param) {
    ::mkl::gpu::srotm(queue, n, x, incx, y, incy, param);
}

void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &param) {
    ::mkl::gpu::drotm(queue, n, x, incx, y, incy, param);
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1, cl::sycl::buffer<float, 1> &d2,
           cl::sycl::buffer<float, 1> &x1, float y1, cl::sycl::buffer<float, 1> &param) {
    ::mkl::gpu::srotmg(queue, d1, d2, x1, y1, param);
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1, cl::sycl::buffer<double, 1> &d2,
           cl::sycl::buffer<double, 1> &x1, double y1, cl::sycl::buffer<double, 1> &param) {
    ::mkl::gpu::drotmg(queue, d1, d2, x1, y1, param);
}

void scal(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    ::mkl::gpu::sscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    ::mkl::gpu::dscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::cscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::zscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::csscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::zdscal(queue, n, alpha, x, incx);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::sswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::cswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::zswap(queue, n, x, incx, y, incy);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::isamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::idamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::icamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::izamax(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::isamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::idamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::icamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::izamin(queue, n, x, incx, result);
}

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, float beta,
                cl::sycl::buffer<float, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    ::mkl::gpu::sgemm_batch(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                            ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, double beta, cl::sycl::buffer<double, 1> &c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    ::mkl::gpu::dgemm_batch(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                            ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    ::mkl::gpu::cgemm_batch(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                            ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    ::mkl::gpu::zgemm_batch(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                            ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    ::mkl::gpu::strsm_batch(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                            ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                            ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb,
                            stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    ::mkl::gpu::dtrsm_batch(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                            ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                            ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb,
                            stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    ::mkl::gpu::ctrsm_batch(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                            ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                            ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb,
                            stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    ::mkl::gpu::ztrsm_batch(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(left_right),
                            ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                            ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb,
                            stride_b, batch_size);
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, float alpha,
           cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
           std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::sgemmt(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, double alpha,
           cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
           std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::dgemmt(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    ::mkl::gpu::zgemmt(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::cgemmt(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}

void gemm_bias(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
               oneapi::mkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
               float alpha, cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
               cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
               cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc,
               cl::sycl::buffer<int32_t, 1> &co) {
    ::mkl::gpu::gemm_s8u8s32(queue, MKL_COL_MAJOR, ::mkl::cblas_convert(transa),
                             ::mkl::cblas_convert(transb), ::mkl::cblas_convert(offsetc), m, n, k,
                             alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co);
}

} //namespace column_major
namespace row_major {

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
          std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::sgemm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
          std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::dgemm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::cgemm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zgemm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
          cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::hgemm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<half, 1> &a,
          std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::gemm_f16f16f32(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                               ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, b, ldb, beta,
                               c, ldc);
}

void symm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::ssymm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::dsymm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::csymm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zsymm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void hemm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::chemm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void hemm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zhemm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::ssyrk(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::dsyrk(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::csyrk(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void syrk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, std::complex<double> beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zsyrk(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void herk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
          std::int64_t ldc) {
    ::mkl::gpu::cherk(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void herk(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
          cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zherk(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                      ::mkl::cblas_convert(trans), n, k, alpha, a, lda, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
           std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::ssyr2k(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
           std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::dsyr2k(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::csyr2k(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void syr2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    ::mkl::gpu::zsyr2k(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void her2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::cher2k(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void her2k(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, double beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::zher2k(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::strmm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::dtrmm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::ctrmm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trmm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::ztrmm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::strsm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::dtrsm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::ctrsm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void trsm(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose transa, oneapi::mkl::diag unit_diag, std::int64_t m,
          std::int64_t n, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    ::mkl::gpu::ztrsm(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                      ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(transa),
                      ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, b, ldb);
}

void gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::sgemv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dgemv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::cgemv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void gemv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::zgemv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::sgbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku, alpha, a,
                      lda, x, incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dgbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku, alpha, a,
                      lda, x, incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::cgbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku, alpha, a,
                      lda, x, incx, beta, y, incy);
}

void gbmv(cl::sycl::queue &queue, oneapi::mkl::transpose trans, std::int64_t m, std::int64_t n,
          std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::zgbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(trans), m, n, kl, ku, alpha, a,
                      lda, x, incx, beta, y, incy);
}

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
         std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    ::mkl::gpu::sger(queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
         std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    ::mkl::gpu::dger(queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::cgerc(queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::zgerc(queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::cgeru(queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::zgeru(queue, MKL_ROW_MAJOR, m, n, alpha, x, incx, y, incy, a, lda);
}

void hbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::chbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void hbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::zhbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void hemv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx, std::complex<float> beta,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::chemv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx,
                      beta, y, incy);
}

void hemv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::zhemv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx,
                      beta, y, incy);
}

void her(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::cher(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a, lda);
}

void her(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::zher(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a, lda);
}

void her2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::cher2(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a, lda);
}

void her2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    ::mkl::gpu::zher2(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a, lda);
}

void hpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::chpmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta,
                      y, incy);
}

void hpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
          std::int64_t incy) {
    ::mkl::gpu::zhpmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta,
                      y, incy);
}

void hpr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<float>, 1> &a) {
    ::mkl::gpu::chpr(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a);
}

void hpr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
         cl::sycl::buffer<std::complex<double>, 1> &a) {
    ::mkl::gpu::zhpr(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a);
}

void hpr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &a) {
    ::mkl::gpu::chpr2(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a);
}

void hpr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n,
          std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &a) {
    ::mkl::gpu::zhpr2(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a);
}

void sbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
          float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::ssbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void sbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, std::int64_t k,
          double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dsbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, k, alpha, a, lda, x,
                      incx, beta, y, incy);
}

void spmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::sspmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta,
                      y, incy);
}

void spmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dspmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, x, incx, beta,
                      y, incy);
}

void spr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a) {
    ::mkl::gpu::sspr(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a);
}

void spr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a) {
    ::mkl::gpu::dspr(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a);
}

void spr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a) {
    ::mkl::gpu::sspr2(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a);
}

void spr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a) {
    ::mkl::gpu::dspr2(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a);
}

void symv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::ssymv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx,
                      beta, y, incy);
}

void symv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dsymv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, a, lda, x, incx,
                      beta, y, incy);
}

void syr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
         cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &a,
         std::int64_t lda) {
    ::mkl::gpu::ssyr(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a, lda);
}

void syr(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
         cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &a,
         std::int64_t lda) {
    ::mkl::gpu::dsyr(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, a, lda);
}

void syr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx, cl::sycl::buffer<float, 1> &y,
          std::int64_t incy, cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    ::mkl::gpu::ssyr2(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a, lda);
}

void syr2(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, std::int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx, cl::sycl::buffer<double, 1> &y,
          std::int64_t incy, cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    ::mkl::gpu::dsyr2(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), n, alpha, x, incx, y, incy,
                      a, lda);
}

void tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::stbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztbmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
          std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::stbsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
          std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtbsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctbsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tbsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, std::int64_t k,
          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztbsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, k, a, lda, x, incx);
}

void tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::stpmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtpmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctpmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztpmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::stpsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtpsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctpsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void tpsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztpsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, x, incx);
}

void trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::strmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtrmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctrmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trmv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztrmv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
          cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    ::mkl::gpu::strsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
          cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    ::mkl::gpu::dtrsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ctrsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void trsv(cl::sycl::queue &queue, oneapi::mkl::uplo uplo, oneapi::mkl::transpose trans,
          oneapi::mkl::diag diag, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::ztrsv(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(uplo), ::mkl::cblas_convert(trans),
                      ::mkl::cblas_convert(diag), n, a, lda, x, incx);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::scasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::dzasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::sasum(queue, n, x, incx, result);
}

void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::dasum(queue, n, x, incx, result);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::saxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::daxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::caxpy(queue, n, alpha, x, incx, y, incy);
}

void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::zaxpy(queue, n, alpha, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::scopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dcopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::ccopy(queue, n, x, incx, y, incy);
}

void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::zcopy(queue, n, x, incx, y, incy);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::sdot(queue, n, x, incx, y, incy, result);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::ddot(queue, n, x, incx, y, incy, result);
}

void sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb, cl::sycl::buffer<float, 1> &x,
            std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
            cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::sdsdot(queue, n, sb, x, incx, y, incy, result);
}

void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::dsdot(queue, n, x, incx, y, incy, result);
}

void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    ::mkl::gpu::cdotc(queue, n, x, incx, y, incy, result);
}

void dotc(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    ::mkl::gpu::zdotc(queue, n, x, incx, y, incy, result);
}

void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<float>, 1> &result) {
    ::mkl::gpu::cdotu(queue, n, x, incx, y, incy, result);
}

void dotu(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
          cl::sycl::buffer<std::complex<double>, 1> &result) {
    ::mkl::gpu::zdotu(queue, n, x, incx, y, incy, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::scnrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::dznrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &result) {
    ::mkl::gpu::snrm2(queue, n, x, incx, result);
}

void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &result) {
    ::mkl::gpu::dnrm2(queue, n, x, incx, result);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
         std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
         float s) {
    ::mkl::gpu::csrot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
         std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
         double c, double s) {
    ::mkl::gpu::zdrot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
         cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c, float s) {
    ::mkl::gpu::srot(queue, n, x, incx, y, incy, c, s);
}

void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
         cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c, double s) {
    ::mkl::gpu::drot(queue, n, x, incx, y, incy, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &b,
          cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<float, 1> &s) {
    ::mkl::gpu::srotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &b,
          cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<double, 1> &s) {
    ::mkl::gpu::drotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
          cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
          cl::sycl::buffer<std::complex<float>, 1> &s) {
    ::mkl::gpu::crotg(queue, a, b, c, s);
}

void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
          cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
          cl::sycl::buffer<std::complex<double>, 1> &s) {
    ::mkl::gpu::zrotg(queue, a, b, c, s);
}

void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy, cl::sycl::buffer<float, 1> &param) {
    ::mkl::gpu::srotm(queue, n, x, incx, y, incy, param);
}

void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy, cl::sycl::buffer<double, 1> &param) {
    ::mkl::gpu::drotm(queue, n, x, incx, y, incy, param);
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1, cl::sycl::buffer<float, 1> &d2,
           cl::sycl::buffer<float, 1> &x1, float y1, cl::sycl::buffer<float, 1> &param) {
    ::mkl::gpu::srotmg(queue, d1, d2, x1, y1, param);
}

void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1, cl::sycl::buffer<double, 1> &d2,
           cl::sycl::buffer<double, 1> &x1, double y1, cl::sycl::buffer<double, 1> &param) {
    ::mkl::gpu::drotmg(queue, d1, d2, x1, y1, param);
}

void scal(cl::sycl::queue &queue, std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          std::int64_t incx) {
    ::mkl::gpu::sscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
          std::int64_t incx) {
    ::mkl::gpu::dscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::cscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::zscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::csscal(queue, n, alpha, x, incx);
}

void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    ::mkl::gpu::zdscal(queue, n, alpha, x, incx);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
          cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    ::mkl::gpu::sswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
          cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    ::mkl::gpu::dswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::cswap(queue, n, x, incx, y, incy);
}

void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
          std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    ::mkl::gpu::zswap(queue, n, x, incx, y, incy);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::isamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::idamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::icamax(queue, n, x, incx, result);
}

void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::izamax(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
           cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::isamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::idamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::icamin(queue, n, x, incx, result);
}

void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    ::mkl::gpu::izamin(queue, n, x, incx, result);
}

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, float beta,
                cl::sycl::buffer<float, 1> &c, std::int64_t ldc, std::int64_t stride_c,
                std::int64_t batch_size) {
    ::mkl::gpu::sgemm_batch(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                            ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, double beta, cl::sycl::buffer<double, 1> &c,
                std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
    ::mkl::gpu::dgemm_batch(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                            ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::complex<float> beta,
                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    ::mkl::gpu::cgemm_batch(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                            ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
}

void gemm_batch(cl::sycl::queue &queue, oneapi::mkl::transpose transa,
                oneapi::mkl::transpose transb, std::int64_t m, std::int64_t n, std::int64_t k,
                std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                std::int64_t lda, std::int64_t stride_a,
                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::complex<double> beta,
                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                std::int64_t stride_c, std::int64_t batch_size) {
    ::mkl::gpu::zgemm_batch(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                            ::mkl::cblas_convert(transb), m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    ::mkl::gpu::strsm_batch(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                            ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                            ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb,
                            stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                std::int64_t stride_b, std::int64_t batch_size) {
    ::mkl::gpu::dtrsm_batch(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                            ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                            ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb,
                            stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, std::complex<float> alpha,
                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    ::mkl::gpu::ctrsm_batch(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                            ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                            ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb,
                            stride_b, batch_size);
}

void trsm_batch(cl::sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                std::int64_t n, std::complex<double> alpha,
                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    ::mkl::gpu::ztrsm_batch(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(left_right),
                            ::mkl::cblas_convert(upper_lower), ::mkl::cblas_convert(trans),
                            ::mkl::cblas_convert(unit_diag), m, n, alpha, a, lda, stride_a, b, ldb,
                            stride_b, batch_size);
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, float alpha,
           cl::sycl::buffer<float, 1> &a, std::int64_t lda, cl::sycl::buffer<float, 1> &b,
           std::int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::sgemmt(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, double alpha,
           cl::sycl::buffer<double, 1> &a, std::int64_t lda, cl::sycl::buffer<double, 1> &b,
           std::int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::dgemmt(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    ::mkl::gpu::zgemmt(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::mkl::gpu::cgemmt(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(upper_lower),
                       ::mkl::cblas_convert(transa), ::mkl::cblas_convert(transb), n, k, alpha, a,
                       lda, b, ldb, beta, c, ldc);
}

void gemm_bias(cl::sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
               oneapi::mkl::offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
               float alpha, cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda, int8_t ao,
               cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb, uint8_t bo, float beta,
               cl::sycl::buffer<int32_t, 1> &c, std::int64_t ldc,
               cl::sycl::buffer<int32_t, 1> &co) {
    ::mkl::gpu::gemm_s8u8s32(queue, MKL_ROW_MAJOR, ::mkl::cblas_convert(transa),
                             ::mkl::cblas_convert(transb), ::mkl::cblas_convert(offsetc), m, n, k,
                             alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co);
}

} //namespace row_major
} //namespace mklgpu
} //namespace mkl
} //namespace oneapi
