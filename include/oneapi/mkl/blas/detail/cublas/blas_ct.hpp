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

//
// Generated based on include/oneapi/mkl/blas/detail/blas_ct_templates.hpp
//

#ifndef _DETAIL_CUBLAS_BLAS_HPP_
#define _DETAIL_CUBLAS_BLAS_HPP_

#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/detail/backends.hpp"
#include "oneapi/mkl/blas/detail/cublas/onemkl_blas_cublas.hpp"
#include "oneapi/mkl/blas/detail/blas_ct_templates.hpp"

namespace oneapi {
namespace mkl {
namespace blas {

template <>
void herk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           std::int64_t n, std::int64_t k, float alpha,
                           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                           float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                           std::int64_t ldc) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::cublas::herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <>
void herk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           std::int64_t n, std::int64_t k, double alpha,
                           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                           double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                           std::int64_t ldc) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::cublas::herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <>
void scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, float alpha,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    oneapi::mkl::cublas::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <>
void scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, double alpha,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    oneapi::mkl::cublas::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <>
void scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    oneapi::mkl::cublas::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <>
void scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    oneapi::mkl::cublas::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <>
void scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, float alpha,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    oneapi::mkl::cublas::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <>
void scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, double alpha,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    oneapi::mkl::cublas::scal(queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

template <>
void trmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::cublas::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <>
void trmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::cublas::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <>
void trmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n,
                           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::cublas::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <>
void trmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n,
                           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::cublas::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <>
void tpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::cublas::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <>
void tpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::cublas::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <>
void tpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n,
                           cl::sycl::buffer<std::complex<float>, 1> &a,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::cublas::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <>
void tpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n,
                           cl::sycl::buffer<std::complex<double>, 1> &a,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::cublas::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <>
void spr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                          cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<float, 1> &a) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    oneapi::mkl::cublas::spr(queue, upper_lower, n, alpha, x, incx, a);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

template <>
void spr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                          cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<double, 1> &a) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    oneapi::mkl::cublas::spr(queue, upper_lower, n, alpha, x, incx, a);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

template <>
void gemm_batch<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                                 std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                 cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                 std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                                 std::int64_t ldb, std::int64_t stride_b, float beta,
                                 cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
                                 std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                                    stride_b, beta, c, ldc, stride_c, batch_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size);
}

template <>
void gemm_batch<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                                 std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                                 cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                 std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                                 std::int64_t ldb, std::int64_t stride_b, double beta,
                                 cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                                 std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                                    stride_b, beta, c, ldc, stride_c, batch_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size);
}

template <>
void gemm_batch<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                                 std::int64_t m, std::int64_t n, std::int64_t k,
                                 std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                                 std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                                 cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                 std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                                    stride_b, beta, c, ldc, stride_c, batch_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size);
}

template <>
void gemm_batch<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                                 std::int64_t m, std::int64_t n, std::int64_t k,
                                 std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 std::int64_t stride_a,
                                 cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                 std::int64_t stride_b, std::complex<double> beta,
                                 cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                 std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
    oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                                    stride_b, beta, c, ldc, stride_c, batch_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size);
}

template <>
void syrk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           std::int64_t n, std::int64_t k, float alpha,
                           cl::sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
                           cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::cublas::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <>
void syrk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           std::int64_t n, std::int64_t k, double alpha,
                           cl::sycl::buffer<double, 1> &a, std::int64_t lda, double beta,
                           cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::cublas::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <>
void syrk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           std::int64_t n, std::int64_t k, std::complex<float> alpha,
                           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                           std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::cublas::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <>
void syrk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           std::int64_t n, std::int64_t k, std::complex<double> alpha,
                           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                           std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    oneapi::mkl::cublas::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

template <>
void her2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                           std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                           std::int64_t lda) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::cublas::her2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

template <>
void her2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                           std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                           std::int64_t lda) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::cublas::her2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

template <>
void hbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                           std::int64_t incx, std::complex<float> beta,
                           cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    hbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void hbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                           std::int64_t incx, std::complex<double> beta,
                           cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    hbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void rot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
                          float s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    oneapi::mkl::cublas::rot(queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

template <>
void rot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c,
                          double s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    oneapi::mkl::cublas::rot(queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

template <>
void rot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                          float c, float s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    oneapi::mkl::cublas::rot(queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

template <>
void rot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                          double c, double s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    oneapi::mkl::cublas::rot(queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

template <>
void axpy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, float alpha,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    oneapi::mkl::cublas::axpy(queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

template <>
void axpy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, double alpha,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    oneapi::mkl::cublas::axpy(queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

template <>
void axpy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    oneapi::mkl::cublas::axpy(queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

template <>
void axpy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    oneapi::mkl::cublas::axpy(queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

template <>
void sdsdot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, float sb,
                             cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                             cl::sycl::buffer<float, 1> &result) {
    sdsdot_precondition(queue, n, sb, x, incx, y, incy, result);
    oneapi::mkl::cublas::sdsdot(queue, n, sb, x, incx, y, incy, result);
    sdsdot_postcondition(queue, n, sb, x, incx, y, incy, result);
}

template <>
void gerc<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                           std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                           std::int64_t lda) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::cublas::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <>
void gerc<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                           std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                           std::int64_t lda) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::cublas::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <>
void syr2k<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                            std::int64_t n, std::int64_t k, float alpha,
                            cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                            cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                               ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void syr2k<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                            std::int64_t n, std::int64_t k, double alpha,
                            cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                            cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                               ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void syr2k<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                            std::int64_t n, std::int64_t k, std::complex<float> alpha,
                            cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                            std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                            std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                               ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void syr2k<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                            std::int64_t n, std::int64_t k, std::complex<double> alpha,
                            cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                            std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                            std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                               ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                           cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void gemv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                           cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void gemv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                           std::int64_t incx, std::complex<float> beta,
                           cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void gemv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                           std::int64_t incx, std::complex<double> beta,
                           cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void her<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    oneapi::mkl::cublas::her(queue, upper_lower, n, alpha, x, incx, a, lda);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

template <>
void her<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    oneapi::mkl::cublas::her(queue, upper_lower, n, alpha, x, incx, a, lda);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

template <>
void hpr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                          cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<std::complex<float>, 1> &a) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    oneapi::mkl::cublas::hpr(queue, upper_lower, n, alpha, x, incx, a);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

template <>
void hpr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                          cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<std::complex<double>, 1> &a) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    oneapi::mkl::cublas::hpr(queue, upper_lower, n, alpha, x, incx, a);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

template <>
void iamin<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                            std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::iamin(queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

template <>
void iamin<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                            std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::iamin(queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

template <>
void iamin<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                            cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                            cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::iamin(queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

template <>
void iamin<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                            cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                            cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::iamin(queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

template <>
void hpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                           std::int64_t incy) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    oneapi::mkl::cublas::hpmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

template <>
void hpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                           std::int64_t incy) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    oneapi::mkl::cublas::hpmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

template <>
void spmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                           cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                           std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
                           std::int64_t incy) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    oneapi::mkl::cublas::spmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

template <>
void spmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                           cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
                           std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y,
                           std::int64_t incy) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    oneapi::mkl::cublas::spmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

template <>
void gemm_ext<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<half, 1> &b, std::int64_t ldb, float beta,
                               cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::gemm_ext(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                  ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm_ext<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                               offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                               float alpha, cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda,
                               int8_t ao, cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb,
                               uint8_t bo, float beta, cl::sycl::buffer<int32_t, 1> &c,
                               std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
    gemm_ext_precondition(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo,
                          beta, c, ldc, co);
    oneapi::mkl::cublas::gemm_ext(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b,
                                  ldb, bo, beta, c, ldc, co);
    gemm_ext_postcondition(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo,
                           beta, c, ldc, co);
}

template <>
void gemm_ext<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                               cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::gemm_ext(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                  ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm_ext<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                               cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                               cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::gemm_ext(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                  ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm_ext<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::gemm_ext(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                  ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm_ext<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::gemm_ext(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                  ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm_ext<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
                               cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                               cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::gemm_ext(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                                  ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void swap<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    oneapi::mkl::cublas::swap(queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

template <>
void swap<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    oneapi::mkl::cublas::swap(queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

template <>
void swap<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    oneapi::mkl::cublas::swap(queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

template <>
void swap<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    oneapi::mkl::cublas::swap(queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

template <>
void geru<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                           std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                           std::int64_t lda) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::cublas::geru(queue, m, n, alpha, x, incx, y, incy, a, lda);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <>
void geru<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                           std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                           std::int64_t lda) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::cublas::geru(queue, m, n, alpha, x, incx, y, incy, a, lda);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <>
void nrm2<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<float, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::nrm2(queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

template <>
void nrm2<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<double, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::nrm2(queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

template <>
void nrm2<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::nrm2(queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

template <>
void nrm2<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::nrm2(queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

template <>
void gemm<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                           cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                           cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                           cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                           cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                           std::int64_t ldb, std::complex<float> beta,
                           cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                           std::int64_t ldb, std::complex<double> beta,
                           cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void gemm<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                           std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
                           cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                           cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void syr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                           cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::cublas::syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

template <>
void syr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                           cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::cublas::syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

template <>
void ger<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                          cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                          cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::cublas::ger(queue, m, n, alpha, x, incx, y, incy, a, lda);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <>
void ger<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                          cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                          cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    oneapi::mkl::cublas::ger(queue, m, n, alpha, x, incx, y, incy, a, lda);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

template <>
void trsm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    oneapi::mkl::cublas::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                              b, ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <>
void trsm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    oneapi::mkl::cublas::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                              b, ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <>
void trsm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                           std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    oneapi::mkl::cublas::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                              b, ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <>
void trsm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                           std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    oneapi::mkl::cublas::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                              b, ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <>
void dotu<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                           cl::sycl::buffer<std::complex<float>, 1> &result) {
    dotu_precondition(queue, n, x, incx, y, incy, result);
    oneapi::mkl::cublas::dotu(queue, n, x, incx, y, incy, result);
    dotu_postcondition(queue, n, x, incx, y, incy, result);
}

template <>
void dotu<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                           cl::sycl::buffer<std::complex<double>, 1> &result) {
    dotu_precondition(queue, n, x, incx, y, incy, result);
    oneapi::mkl::cublas::dotu(queue, n, x, incx, y, incy, result);
    dotu_postcondition(queue, n, x, incx, y, incy, result);
}

template <>
void hemm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           std::int64_t m, std::int64_t n, std::complex<float> alpha,
                           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                           std::int64_t ldc) {
    hemm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::hemm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c,
                              ldc);
    hemm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void hemm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           std::int64_t m, std::int64_t n, std::complex<double> alpha,
                           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                           std::int64_t ldc) {
    hemm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::hemm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c,
                              ldc);
    hemm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void hpr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                           std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    oneapi::mkl::cublas::hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

template <>
void hpr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                           std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    oneapi::mkl::cublas::hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

template <>
void gbmv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           std::int64_t kl, std::int64_t ku, float alpha,
                           cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                           cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void gbmv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           std::int64_t kl, std::int64_t ku, double alpha,
                           cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                           cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void gbmv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                           std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void gbmv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                           std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                           std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void tbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, std::int64_t k,
                           cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::cublas::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <>
void tbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, std::int64_t k,
                           cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::cublas::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <>
void tbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, std::int64_t k,
                           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::cublas::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <>
void tbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, std::int64_t k,
                           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::cublas::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <>
void symm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           std::int64_t m, std::int64_t n, float alpha,
                           cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                           cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c,
                              ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void symm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           std::int64_t m, std::int64_t n, double alpha,
                           cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                           cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c,
                              ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void symm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           std::int64_t m, std::int64_t n, std::complex<float> alpha,
                           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                           std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c,
                              ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void symm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           std::int64_t m, std::int64_t n, std::complex<double> alpha,
                           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                           std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c,
                              ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void dotc<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                           cl::sycl::buffer<std::complex<float>, 1> &result) {
    dotc_precondition(queue, n, x, incx, y, incy, result);
    oneapi::mkl::cublas::dotc(queue, n, x, incx, y, incy, result);
    dotc_postcondition(queue, n, x, incx, y, incy, result);
}

template <>
void dotc<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                           cl::sycl::buffer<std::complex<double>, 1> &result) {
    dotc_precondition(queue, n, x, incx, y, incy, result);
    oneapi::mkl::cublas::dotc(queue, n, x, incx, y, incy, result);
    dotc_postcondition(queue, n, x, incx, y, incy, result);
}

template <>
void syr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                          cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    oneapi::mkl::cublas::syr(queue, upper_lower, n, alpha, x, incx, a, lda);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

template <>
void syr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                          cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    oneapi::mkl::cublas::syr(queue, upper_lower, n, alpha, x, incx, a, lda);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

template <>
void trmm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    oneapi::mkl::cublas::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                              b, ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <>
void trmm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    oneapi::mkl::cublas::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                              b, ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <>
void trmm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                           std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    oneapi::mkl::cublas::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                              b, ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <>
void trmm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                           transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                           std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    oneapi::mkl::cublas::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                              b, ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

template <>
void rotmg<backend::cublas>(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
                            cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1,
                            float y1, cl::sycl::buffer<float, 1> &param) {
    rotmg_precondition(queue, d1, d2, x1, y1, param);
    oneapi::mkl::cublas::rotmg(queue, d1, d2, x1, y1, param);
    rotmg_postcondition(queue, d1, d2, x1, y1, param);
}

template <>
void rotmg<backend::cublas>(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
                            cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1,
                            double y1, cl::sycl::buffer<double, 1> &param) {
    rotmg_precondition(queue, d1, d2, x1, y1, param);
    oneapi::mkl::cublas::rotmg(queue, d1, d2, x1, y1, param);
    rotmg_postcondition(queue, d1, d2, x1, y1, param);
}

template <>
void tpsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::cublas::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <>
void tpsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::cublas::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <>
void tpsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n,
                           cl::sycl::buffer<std::complex<float>, 1> &a,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::cublas::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <>
void tpsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n,
                           cl::sycl::buffer<std::complex<double>, 1> &a,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    oneapi::mkl::cublas::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

template <>
void trsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::cublas::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <>
void trsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::cublas::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <>
void trsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n,
                           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::cublas::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <>
void trsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n,
                           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    oneapi::mkl::cublas::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

template <>
void copy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    oneapi::mkl::cublas::copy(queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

template <>
void copy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    oneapi::mkl::cublas::copy(queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

template <>
void copy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    oneapi::mkl::cublas::copy(queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

template <>
void copy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    oneapi::mkl::cublas::copy(queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

template <>
void hemv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                           std::int64_t incx, std::complex<float> beta,
                           cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void hemv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                           std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                           std::int64_t incx, std::complex<double> beta,
                           cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void gemmt<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                            transpose transb, std::int64_t n, std::int64_t k, float alpha,
                            cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                            cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc);
    oneapi::mkl::cublas::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb,
                               beta, c, ldc);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

template <>
void gemmt<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                            transpose transb, std::int64_t n, std::int64_t k, double alpha,
                            cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                            cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc);
    oneapi::mkl::cublas::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb,
                               beta, c, ldc);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

template <>
void gemmt<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                            transpose transb, std::int64_t n, std::int64_t k,
                            std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                            std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                            std::int64_t ldb, std::complex<float> beta,
                            cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc);
    oneapi::mkl::cublas::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb,
                               beta, c, ldc);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

template <>
void gemmt<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                            transpose transb, std::int64_t n, std::int64_t k,
                            std::complex<double> alpha,
                            cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                            std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                            std::int64_t ldc) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc);
    oneapi::mkl::cublas::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb,
                               beta, c, ldc);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

template <>
void asum<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<float, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::asum(queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

template <>
void asum<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<double, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::asum(queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

template <>
void asum<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::asum(queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

template <>
void asum<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::asum(queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

template <>
void sbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                           float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                           cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void sbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                           double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                           cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void tbsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, std::int64_t k,
                           cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::cublas::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <>
void tbsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, std::int64_t k,
                           cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::cublas::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <>
void tbsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, std::int64_t k,
                           cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::cublas::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <>
void tbsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                           diag unit_diag, std::int64_t n, std::int64_t k,
                           cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    oneapi::mkl::cublas::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

template <>
void spr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                           cl::sycl::buffer<float, 1> &a) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    oneapi::mkl::cublas::spr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

template <>
void spr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                           cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                           cl::sycl::buffer<double, 1> &a) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    oneapi::mkl::cublas::spr2(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

template <>
void iamax<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                            std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::iamax(queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

template <>
void iamax<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                            std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::iamax(queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

template <>
void iamax<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                            cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                            cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::iamax(queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

template <>
void iamax<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                            cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                            cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    oneapi::mkl::cublas::iamax(queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

template <>
void rotm<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                           cl::sycl::buffer<float, 1> &param) {
    rotm_precondition(queue, n, x, incx, y, incy, param);
    oneapi::mkl::cublas::rotm(queue, n, x, incx, y, incy, param);
    rotm_postcondition(queue, n, x, incx, y, incy, param);
}

template <>
void rotm<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                           std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                           cl::sycl::buffer<double, 1> &param) {
    rotm_precondition(queue, n, x, incx, y, incy, param);
    oneapi::mkl::cublas::rotm(queue, n, x, incx, y, incy, param);
    rotm_postcondition(queue, n, x, incx, y, incy, param);
}

template <>
void dot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                          cl::sycl::buffer<float, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    oneapi::mkl::cublas::dot(queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

template <>
void dot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                          std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                          cl::sycl::buffer<double, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    oneapi::mkl::cublas::dot(queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

template <>
void dot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                          std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                          cl::sycl::buffer<double, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    oneapi::mkl::cublas::dot(queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

template <>
void trsm_batch<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                 transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                 float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                 std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                                 std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            stride_a, b, ldb, stride_b, batch_size);
    oneapi::mkl::cublas::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                                    a, lda, stride_a, b, ldb, stride_b, batch_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             stride_a, b, ldb, stride_b, batch_size);
}

template <>
void trsm_batch<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                 transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                 double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                 std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                                 std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            stride_a, b, ldb, stride_b, batch_size);
    oneapi::mkl::cublas::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                                    a, lda, stride_a, b, ldb, stride_b, batch_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             stride_a, b, ldb, stride_b, batch_size);
}

template <>
void trsm_batch<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                 transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                 std::complex<float> alpha,
                                 cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                 std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                                 std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            stride_a, b, ldb, stride_b, batch_size);
    oneapi::mkl::cublas::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                                    a, lda, stride_a, b, ldb, stride_b, batch_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             stride_a, b, ldb, stride_b, batch_size);
}

template <>
void trsm_batch<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                 transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                 std::complex<double> alpha,
                                 cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                 std::int64_t stride_a,
                                 cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                 std::int64_t stride_b, std::int64_t batch_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            stride_a, b, ldb, stride_b, batch_size);
    oneapi::mkl::cublas::trsm_batch(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                                    a, lda, stride_a, b, ldb, stride_b, batch_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             stride_a, b, ldb, stride_b, batch_size);
}

template <>
void her2k<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                            std::int64_t n, std::int64_t k, std::complex<float> alpha,
                            cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                            float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                            std::int64_t ldc) {
    her2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                               ldc);
    her2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void her2k<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                            std::int64_t n, std::int64_t k, std::complex<double> alpha,
                            cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                            double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                            std::int64_t ldc) {
    her2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    oneapi::mkl::cublas::her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c,
                               ldc);
    her2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void rotg<backend::cublas>(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
                           cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
                           cl::sycl::buffer<float, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    oneapi::mkl::cublas::rotg(queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

template <>
void rotg<backend::cublas>(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
                           cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
                           cl::sycl::buffer<double, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    oneapi::mkl::cublas::rotg(queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

template <>
void rotg<backend::cublas>(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
                           cl::sycl::buffer<std::complex<float>, 1> &b,
                           cl::sycl::buffer<float, 1> &c,
                           cl::sycl::buffer<std::complex<float>, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    oneapi::mkl::cublas::rotg(queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

template <>
void rotg<backend::cublas>(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
                           cl::sycl::buffer<std::complex<double>, 1> &b,
                           cl::sycl::buffer<double, 1> &c,
                           cl::sycl::buffer<std::complex<double>, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    oneapi::mkl::cublas::rotg(queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

template <>
void symv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                           cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                           cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void symv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                           cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                           cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                           cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    oneapi::mkl::cublas::symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

// USM APIs

template <>
cl::sycl::event syr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      float alpha, const float *x, std::int64_t incx,
                                      const float *y, std::int64_t incy, float *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = oneapi::mkl::cublas::syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda,
                                          dependencies);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event syr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      double alpha, const double *x, std::int64_t incx,
                                      const double *y, std::int64_t incy, double *a,
                                      std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = oneapi::mkl::cublas::syr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda,
                                          dependencies);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, float alpha, float *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::scal(queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                      double *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::scal(queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      std::complex<float> alpha, std::complex<float> *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::scal(queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      std::complex<double> alpha, std::complex<double> *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::scal(queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                      std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::scal(queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event scal<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                      std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::scal(queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event trmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const float *a,
                                      std::int64_t lda, float *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx,
                                          dependencies);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event trmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const double *a,
                                      std::int64_t lda, double *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx,
                                          dependencies);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event trmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const std::complex<float> *a,
                                      std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx,
                                          dependencies);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event trmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const std::complex<double> *a,
                                      std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::trmv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx,
                                          dependencies);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const float *a, float *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                          dependencies);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const double *a, double *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                          dependencies);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const std::complex<float> *a,
                                      std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                          dependencies);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const std::complex<double> *a,
                                      std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tpmv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                          dependencies);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event spr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                     float alpha, const float *x, std::int64_t incx, float *a,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    auto done = oneapi::mkl::cublas::spr(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

template <>
cl::sycl::event spr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                     double alpha, const double *x, std::int64_t incx, double *a,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    auto done = oneapi::mkl::cublas::spr(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

template <>
cl::sycl::event hpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *a,
                                      const std::complex<float> *x, std::int64_t incx,
                                      std::complex<float> beta, std::complex<float> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::hpmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy,
                                          dependencies);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event hpmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *a,
                                      const std::complex<double> *x, std::int64_t incx,
                                      std::complex<double> beta, std::complex<double> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::hpmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy,
                                          dependencies);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event syrk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      std::int64_t n, std::int64_t k, float alpha, const float *a,
                                      std::int64_t lda, float beta, float *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = oneapi::mkl::cublas::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                          ldc, dependencies);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

template <>
cl::sycl::event syrk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      std::int64_t n, std::int64_t k, double alpha, const double *a,
                                      std::int64_t lda, double beta, double *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = oneapi::mkl::cublas::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                          ldc, dependencies);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

template <>
cl::sycl::event syrk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      std::complex<float> beta, std::complex<float> *c,
                                      std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = oneapi::mkl::cublas::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                          ldc, dependencies);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

template <>
cl::sycl::event syrk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      std::complex<double> beta, std::complex<double> *c,
                                      std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = oneapi::mkl::cublas::syrk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                          ldc, dependencies);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

template <>
cl::sycl::event her2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *x,
                                      std::int64_t incx, const std::complex<float> *y,
                                      std::int64_t incy, std::complex<float> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = oneapi::mkl::cublas::her2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda,
                                          dependencies);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event her2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *x,
                                      std::int64_t incx, const std::complex<double> *y,
                                      std::int64_t incy, std::complex<double> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = oneapi::mkl::cublas::her2(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda,
                                          dependencies);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event hbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::int64_t k, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      const std::complex<float> *x, std::int64_t incx,
                                      std::complex<float> beta, std::complex<float> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::cublas::hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                          incy, dependencies);
    hbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

template <>
cl::sycl::event hbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::int64_t k, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      const std::complex<double> *x, std::int64_t incx,
                                      std::complex<double> beta, std::complex<double> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::cublas::hbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                          incy, dependencies);
    hbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

template <>
cl::sycl::event rot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                                     std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                                     float c, float s,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rot_precondition(queue, n, x, incx, y, incy, c, s, dependencies);
    auto done = oneapi::mkl::cublas::rot(queue, n, x, incx, y, incy, c, s, dependencies);
    rot_postcondition(queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

template <>
cl::sycl::event rot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                     std::complex<double> *x, std::int64_t incx,
                                     std::complex<double> *y, std::int64_t incy, double c, double s,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rot_precondition(queue, n, x, incx, y, incy, c, s, dependencies);
    auto done = oneapi::mkl::cublas::rot(queue, n, x, incx, y, incy, c, s, dependencies);
    rot_postcondition(queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

template <>
cl::sycl::event rot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, float *x,
                                     std::int64_t incx, float *y, std::int64_t incy, float c,
                                     float s,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rot_precondition(queue, n, x, incx, y, incy, c, s, dependencies);
    auto done = oneapi::mkl::cublas::rot(queue, n, x, incx, y, incy, c, s, dependencies);
    rot_postcondition(queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

template <>
cl::sycl::event rot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, double *x,
                                     std::int64_t incx, double *y, std::int64_t incy, double c,
                                     double s,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rot_precondition(queue, n, x, incx, y, incy, c, s, dependencies);
    auto done = oneapi::mkl::cublas::rot(queue, n, x, incx, y, incy, c, s, dependencies);
    rot_postcondition(queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

template <>
cl::sycl::event axpy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, float alpha,
                                      const float *x, std::int64_t incx, float *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event axpy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, double alpha,
                                      const double *x, std::int64_t incx, double *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event axpy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *x,
                                      std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event axpy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *x,
                                      std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::axpy(queue, n, alpha, x, incx, y, incy, dependencies);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event axpy_batch<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x, std::int64_t *incx,
    float **y, std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    axpy_batch_precondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = oneapi::mkl::cublas::axpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                                group_size, dependencies);
    axpy_batch_postcondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

template <>
cl::sycl::event axpy_batch<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x, std::int64_t *incx,
    double **y, std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    axpy_batch_precondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = oneapi::mkl::cublas::axpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                                group_size, dependencies);
    axpy_batch_postcondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

template <>
cl::sycl::event axpy_batch<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
    const std::complex<float> **x, std::int64_t *incx, std::complex<float> **y, std::int64_t *incy,
    std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    axpy_batch_precondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = oneapi::mkl::cublas::axpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                                group_size, dependencies);
    axpy_batch_postcondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

template <>
cl::sycl::event axpy_batch<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
    const std::complex<double> **x, std::int64_t *incx, std::complex<double> **y,
    std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    axpy_batch_precondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = oneapi::mkl::cublas::axpy_batch(queue, n, alpha, x, incx, y, incy, group_count,
                                                group_size, dependencies);
    axpy_batch_postcondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

template <>
cl::sycl::event gerc<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *x,
                                      std::int64_t incx, const std::complex<float> *y,
                                      std::int64_t incy, std::complex<float> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done =
        oneapi::mkl::cublas::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event gerc<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *x,
                                      std::int64_t incx, const std::complex<double> *y,
                                      std::int64_t incy, std::complex<double> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done =
        oneapi::mkl::cublas::gerc(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event syr2k<backend::cublas>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    float alpha, const float *a, std::int64_t lda, const float *b, std::int64_t ldb, float beta,
    float *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = oneapi::mkl::cublas::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                           beta, c, ldc, dependencies);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

template <>
cl::sycl::event syr2k<backend::cublas>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    double alpha, const double *a, std::int64_t lda, const double *b, std::int64_t ldb, double beta,
    double *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = oneapi::mkl::cublas::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                           beta, c, ldc, dependencies);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

template <>
cl::sycl::event syr2k<backend::cublas>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = oneapi::mkl::cublas::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                           beta, c, ldc, dependencies);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

template <>
cl::sycl::event syr2k<backend::cublas>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = oneapi::mkl::cublas::syr2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                           beta, c, ldc, dependencies);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

template <>
cl::sycl::event gemv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                                      std::int64_t n, float alpha, const float *a, std::int64_t lda,
                                      const float *x, std::int64_t incx, float beta, float *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                                          dependencies);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event gemv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                                      std::int64_t n, double alpha, const double *a,
                                      std::int64_t lda, const double *x, std::int64_t incx,
                                      double beta, double *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                                          dependencies);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event gemv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                                      std::int64_t n, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      const std::complex<float> *x, std::int64_t incx,
                                      std::complex<float> beta, std::complex<float> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                                          dependencies);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event gemv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                                      std::int64_t n, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      const std::complex<double> *x, std::int64_t incx,
                                      std::complex<double> beta, std::complex<double> *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::gemv(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy,
                                          dependencies);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event her<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                     float alpha, const std::complex<float> *x, std::int64_t incx,
                                     std::complex<float> *a, std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done =
        oneapi::mkl::cublas::her(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event her<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                     double alpha, const std::complex<double> *x, std::int64_t incx,
                                     std::complex<double> *a, std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done =
        oneapi::mkl::cublas::her(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event hpr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                     float alpha, const std::complex<float> *x, std::int64_t incx,
                                     std::complex<float> *a,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    auto done = oneapi::mkl::cublas::hpr(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

template <>
cl::sycl::event hpr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                     double alpha, const std::complex<double> *x, std::int64_t incx,
                                     std::complex<double> *a,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    auto done = oneapi::mkl::cublas::hpr(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

template <>
cl::sycl::event iamin<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    iamin_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::iamin(queue, n, x, incx, result, dependencies);
    iamin_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event iamin<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    iamin_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::iamin(queue, n, x, incx, result, dependencies);
    iamin_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event iamin<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    iamin_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::iamin(queue, n, x, incx, result, dependencies);
    iamin_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event iamin<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    iamin_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::iamin(queue, n, x, incx, result, dependencies);
    iamin_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event gemm_batch<backend::cublas>(
    cl::sycl::queue &queue, transpose *transa, transpose *transb, std::int64_t *m, std::int64_t *n,
    std::int64_t *k, float *alpha, const float **a, std::int64_t *lda, const float **b,
    std::int64_t *ldb, float *beta, float **c, std::int64_t *ldc, std::int64_t group_count,
    std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size, dependencies);
    auto done =
        oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                        c, ldc, group_count, group_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

template <>
cl::sycl::event gemm_batch<backend::cublas>(
    cl::sycl::queue &queue, transpose *transa, transpose *transb, std::int64_t *m, std::int64_t *n,
    std::int64_t *k, double *alpha, const double **a, std::int64_t *lda, const double **b,
    std::int64_t *ldb, double *beta, double **c, std::int64_t *ldc, std::int64_t group_count,
    std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size, dependencies);
    auto done =
        oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                        c, ldc, group_count, group_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

template <>
cl::sycl::event gemm_batch<backend::cublas>(
    cl::sycl::queue &queue, transpose *transa, transpose *transb, std::int64_t *m, std::int64_t *n,
    std::int64_t *k, std::complex<float> *alpha, const std::complex<float> **a, std::int64_t *lda,
    const std::complex<float> **b, std::int64_t *ldb, std::complex<float> *beta,
    std::complex<float> **c, std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size, dependencies);
    auto done =
        oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                        c, ldc, group_count, group_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

template <>
cl::sycl::event gemm_batch<backend::cublas>(
    cl::sycl::queue &queue, transpose *transa, transpose *transb, std::int64_t *m, std::int64_t *n,
    std::int64_t *k, std::complex<double> *alpha, const std::complex<double> **a, std::int64_t *lda,
    const std::complex<double> **b, std::int64_t *ldb, std::complex<double> *beta,
    std::complex<double> **c, std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size, dependencies);
    auto done =
        oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                                        c, ldc, group_count, group_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

template <>
cl::sycl::event gemm_batch<backend::cublas>(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, float alpha, const float *a, std::int64_t lda, std::int64_t stride_a,
    const float *b, std::int64_t ldb, std::int64_t stride_b, float beta, float *c, std::int64_t ldc,
    std::int64_t stride_c, std::int64_t batch_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                                batch_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

template <>
cl::sycl::event gemm_batch<backend::cublas>(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, double alpha, const double *a, std::int64_t lda, std::int64_t stride_a,
    const double *b, std::int64_t ldb, std::int64_t stride_b, double beta, double *c,
    std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                                batch_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

template <>
cl::sycl::event gemm_batch<backend::cublas>(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    std::int64_t stride_a, const std::complex<float> *b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<float> beta, std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                                batch_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

template <>
cl::sycl::event gemm_batch<backend::cublas>(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    std::int64_t stride_a, const std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<double> beta, std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = oneapi::mkl::cublas::gemm_batch(queue, transa, transb, m, n, k, alpha, a, lda,
                                                stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                                batch_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

template <>
cl::sycl::event spmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      float alpha, const float *a, const float *x,
                                      std::int64_t incx, float beta, float *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::spmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy,
                                          dependencies);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event spmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      double alpha, const double *a, const double *x,
                                      std::int64_t incx, double beta, double *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::spmv(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy,
                                          dependencies);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event swap<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, float *x,
                                      std::int64_t incx, float *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    swap_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::swap(queue, n, x, incx, y, incy, dependencies);
    swap_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event swap<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, double *x,
                                      std::int64_t incx, double *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    swap_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::swap(queue, n, x, incx, y, incy, dependencies);
    swap_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event swap<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      std::complex<float> *x, std::int64_t incx,
                                      std::complex<float> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    swap_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::swap(queue, n, x, incx, y, incy, dependencies);
    swap_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event swap<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      std::complex<double> *x, std::int64_t incx,
                                      std::complex<double> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    swap_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::swap(queue, n, x, incx, y, incy, dependencies);
    swap_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event geru<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *x,
                                      std::int64_t incx, const std::complex<float> *y,
                                      std::int64_t incy, std::complex<float> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done =
        oneapi::mkl::cublas::geru(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event geru<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *x,
                                      std::int64_t incx, const std::complex<double> *y,
                                      std::int64_t incy, std::complex<double> *a, std::int64_t lda,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done =
        oneapi::mkl::cublas::geru(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event nrm2<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<float> *x, std::int64_t incx,
                                      float *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    nrm2_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::nrm2(queue, n, x, incx, result, dependencies);
    nrm2_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event nrm2<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<double> *x, std::int64_t incx,
                                      double *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    nrm2_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::nrm2(queue, n, x, incx, result, dependencies);
    nrm2_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event nrm2<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                      std::int64_t incx, float *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    nrm2_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::nrm2(queue, n, x, incx, result, dependencies);
    nrm2_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event nrm2<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, const double *x,
                                      std::int64_t incx, double *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    nrm2_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::nrm2(queue, n, x, incx, result, dependencies);
    nrm2_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event gemm<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                                      std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                      const float *a, std::int64_t lda, const float *b,
                                      std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::cublas::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                          beta, c, ldc, dependencies);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

template <>
cl::sycl::event gemm<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                                      std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                                      const double *a, std::int64_t lda, const double *b,
                                      std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::cublas::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                          beta, c, ldc, dependencies);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

template <>
cl::sycl::event gemm<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                                      std::int64_t m, std::int64_t n, std::int64_t k,
                                      std::complex<float> alpha, const std::complex<float> *a,
                                      std::int64_t lda, const std::complex<float> *b,
                                      std::int64_t ldb, std::complex<float> beta,
                                      std::complex<float> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::cublas::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                          beta, c, ldc, dependencies);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

template <>
cl::sycl::event gemm<backend::cublas>(cl::sycl::queue &queue, transpose transa, transpose transb,
                                      std::int64_t m, std::int64_t n, std::int64_t k,
                                      std::complex<double> alpha, const std::complex<double> *a,
                                      std::int64_t lda, const std::complex<double> *b,
                                      std::int64_t ldb, std::complex<double> beta,
                                      std::complex<double> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::cublas::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                                          beta, c, ldc, dependencies);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

template <>
cl::sycl::event herk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      std::int64_t n, std::int64_t k, float alpha,
                                      const std::complex<float> *a, std::int64_t lda, float beta,
                                      std::complex<float> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = oneapi::mkl::cublas::herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                          ldc, dependencies);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

template <>
cl::sycl::event herk<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      std::int64_t n, std::int64_t k, double alpha,
                                      const std::complex<double> *a, std::int64_t lda, double beta,
                                      std::complex<double> *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = oneapi::mkl::cublas::herk(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                                          ldc, dependencies);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

template <>
cl::sycl::event ger<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                     float alpha, const float *x, std::int64_t incx, const float *y,
                                     std::int64_t incy, float *a, std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done =
        oneapi::mkl::cublas::ger(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event ger<backend::cublas>(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                                     double alpha, const double *x, std::int64_t incx,
                                     const double *y, std::int64_t incy, double *a,
                                     std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done =
        oneapi::mkl::cublas::ger(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event trsm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      transpose trans, diag unit_diag, std::int64_t m,
                                      std::int64_t n, float alpha, const float *a, std::int64_t lda,
                                      float *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = oneapi::mkl::cublas::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                          alpha, a, lda, b, ldb, dependencies);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

template <>
cl::sycl::event trsm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      transpose trans, diag unit_diag, std::int64_t m,
                                      std::int64_t n, double alpha, const double *a,
                                      std::int64_t lda, double *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = oneapi::mkl::cublas::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                          alpha, a, lda, b, ldb, dependencies);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

template <>
cl::sycl::event trsm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      transpose trans, diag unit_diag, std::int64_t m,
                                      std::int64_t n, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      std::complex<float> *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = oneapi::mkl::cublas::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                          alpha, a, lda, b, ldb, dependencies);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

template <>
cl::sycl::event trsm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      transpose trans, diag unit_diag, std::int64_t m,
                                      std::int64_t n, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      std::complex<double> *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = oneapi::mkl::cublas::trsm(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                          alpha, a, lda, b, ldb, dependencies);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

template <>
cl::sycl::event dotu<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<float> *x, std::int64_t incx,
                                      const std::complex<float> *y, std::int64_t incy,
                                      std::complex<float> *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    dotu_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::cublas::dotu(queue, n, x, incx, y, incy, result, dependencies);
    dotu_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

template <>
cl::sycl::event dotu<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<double> *x, std::int64_t incx,
                                      const std::complex<double> *y, std::int64_t incy,
                                      std::complex<double> *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    dotu_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::cublas::dotu(queue, n, x, incx, y, incy, result, dependencies);
    dotu_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

template <>
cl::sycl::event hemm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      const std::complex<float> *b, std::int64_t ldb,
                                      std::complex<float> beta, std::complex<float> *c,
                                      std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hemm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::cublas::hemm(queue, left_right, upper_lower, m, n, alpha, a, lda, b,
                                          ldb, beta, c, ldc, dependencies);
    hemm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

template <>
cl::sycl::event hemm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      const std::complex<double> *b, std::int64_t ldb,
                                      std::complex<double> beta, std::complex<double> *c,
                                      std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hemm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::cublas::hemm(queue, left_right, upper_lower, m, n, alpha, a, lda, b,
                                          ldb, beta, c, ldc, dependencies);
    hemm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

template <>
cl::sycl::event hpr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *x,
                                      std::int64_t incx, const std::complex<float> *y,
                                      std::int64_t incy, std::complex<float> *a,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    auto done =
        oneapi::mkl::cublas::hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    return done;
}

template <>
cl::sycl::event hpr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *x,
                                      std::int64_t incx, const std::complex<double> *y,
                                      std::int64_t incy, std::complex<double> *a,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    auto done =
        oneapi::mkl::cublas::hpr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    return done;
}

template <>
cl::sycl::event gbmv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                                      std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
                                      const float *a, std::int64_t lda, const float *x,
                                      std::int64_t incx, float beta, float *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::cublas::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta,
                                          y, incy, dependencies);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

template <>
cl::sycl::event gbmv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                                      std::int64_t n, std::int64_t kl, std::int64_t ku,
                                      double alpha, const double *a, std::int64_t lda,
                                      const double *x, std::int64_t incx, double beta, double *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::cublas::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta,
                                          y, incy, dependencies);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

template <>
cl::sycl::event gbmv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                                      std::int64_t n, std::int64_t kl, std::int64_t ku,
                                      std::complex<float> alpha, const std::complex<float> *a,
                                      std::int64_t lda, const std::complex<float> *x,
                                      std::int64_t incx, std::complex<float> beta,
                                      std::complex<float> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::cublas::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta,
                                          y, incy, dependencies);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

template <>
cl::sycl::event gbmv<backend::cublas>(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                                      std::int64_t n, std::int64_t kl, std::int64_t ku,
                                      std::complex<double> alpha, const std::complex<double> *a,
                                      std::int64_t lda, const std::complex<double> *x,
                                      std::int64_t incx, std::complex<double> beta,
                                      std::complex<double> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::cublas::gbmv(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta,
                                          y, incy, dependencies);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

template <>
cl::sycl::event tbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, std::int64_t k,
                                      const float *a, std::int64_t lda, float *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                          incx, dependencies);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, std::int64_t k,
                                      const double *a, std::int64_t lda, double *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                          incx, dependencies);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, std::int64_t k,
                                      const std::complex<float> *a, std::int64_t lda,
                                      std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                          incx, dependencies);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, std::int64_t k,
                                      const std::complex<double> *a, std::int64_t lda,
                                      std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tbmv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                          incx, dependencies);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event symm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      std::int64_t m, std::int64_t n, float alpha, const float *a,
                                      std::int64_t lda, const float *b, std::int64_t ldb,
                                      float beta, float *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::cublas::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b,
                                          ldb, beta, c, ldc, dependencies);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

template <>
cl::sycl::event symm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      std::int64_t m, std::int64_t n, double alpha, const double *a,
                                      std::int64_t lda, const double *b, std::int64_t ldb,
                                      double beta, double *c, std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::cublas::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b,
                                          ldb, beta, c, ldc, dependencies);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

template <>
cl::sycl::event symm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      const std::complex<float> *b, std::int64_t ldb,
                                      std::complex<float> beta, std::complex<float> *c,
                                      std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::cublas::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b,
                                          ldb, beta, c, ldc, dependencies);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

template <>
cl::sycl::event symm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      const std::complex<double> *b, std::int64_t ldb,
                                      std::complex<double> beta, std::complex<double> *c,
                                      std::int64_t ldc,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = oneapi::mkl::cublas::symm(queue, left_right, upper_lower, m, n, alpha, a, lda, b,
                                          ldb, beta, c, ldc, dependencies);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

template <>
cl::sycl::event dotc<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<float> *x, std::int64_t incx,
                                      const std::complex<float> *y, std::int64_t incy,
                                      std::complex<float> *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    dotc_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::cublas::dotc(queue, n, x, incx, y, incy, result, dependencies);
    dotc_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

template <>
cl::sycl::event dotc<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<double> *x, std::int64_t incx,
                                      const std::complex<double> *y, std::int64_t incy,
                                      std::complex<double> *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    dotc_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::cublas::dotc(queue, n, x, incx, y, incy, result, dependencies);
    dotc_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

template <>
cl::sycl::event syr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                     float alpha, const float *x, std::int64_t incx, float *a,
                                     std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done =
        oneapi::mkl::cublas::syr(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event syr<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                     double alpha, const double *x, std::int64_t incx, double *a,
                                     std::int64_t lda,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done =
        oneapi::mkl::cublas::syr(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

template <>
cl::sycl::event trmm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      transpose trans, diag unit_diag, std::int64_t m,
                                      std::int64_t n, float alpha, const float *a, std::int64_t lda,
                                      float *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = oneapi::mkl::cublas::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                          alpha, a, lda, b, ldb, dependencies);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

template <>
cl::sycl::event trmm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      transpose trans, diag unit_diag, std::int64_t m,
                                      std::int64_t n, double alpha, const double *a,
                                      std::int64_t lda, double *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = oneapi::mkl::cublas::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                          alpha, a, lda, b, ldb, dependencies);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

template <>
cl::sycl::event trmm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      transpose trans, diag unit_diag, std::int64_t m,
                                      std::int64_t n, std::complex<float> alpha,
                                      const std::complex<float> *a, std::int64_t lda,
                                      std::complex<float> *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = oneapi::mkl::cublas::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                          alpha, a, lda, b, ldb, dependencies);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

template <>
cl::sycl::event trmm<backend::cublas>(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                      transpose trans, diag unit_diag, std::int64_t m,
                                      std::int64_t n, std::complex<double> alpha,
                                      const std::complex<double> *a, std::int64_t lda,
                                      std::complex<double> *b, std::int64_t ldb,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = oneapi::mkl::cublas::trmm(queue, left_right, upper_lower, trans, unit_diag, m, n,
                                          alpha, a, lda, b, ldb, dependencies);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

template <>
cl::sycl::event rotmg<backend::cublas>(
    cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1, float *param,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rotmg_precondition(queue, d1, d2, x1, y1, param, dependencies);
    auto done = oneapi::mkl::cublas::rotmg(queue, d1, d2, x1, y1, param, dependencies);
    rotmg_postcondition(queue, d1, d2, x1, y1, param, dependencies);
    return done;
}

template <>
cl::sycl::event rotmg<backend::cublas>(
    cl::sycl::queue &queue, double *d1, double *d2, double *x1, double y1, double *param,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rotmg_precondition(queue, d1, d2, x1, y1, param, dependencies);
    auto done = oneapi::mkl::cublas::rotmg(queue, d1, d2, x1, y1, param, dependencies);
    rotmg_postcondition(queue, d1, d2, x1, y1, param, dependencies);
    return done;
}

template <>
cl::sycl::event tpsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const float *a, float *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                          dependencies);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tpsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const double *a, double *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                          dependencies);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tpsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const std::complex<float> *a,
                                      std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                          dependencies);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tpsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const std::complex<double> *a,
                                      std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tpsv(queue, upper_lower, trans, unit_diag, n, a, x, incx,
                                          dependencies);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event trsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const float *a,
                                      std::int64_t lda, float *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx,
                                          dependencies);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event trsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const double *a,
                                      std::int64_t lda, double *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx,
                                          dependencies);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event trsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const std::complex<float> *a,
                                      std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx,
                                          dependencies);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event trsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, const std::complex<double> *a,
                                      std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::trsv(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx,
                                          dependencies);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event copy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                      std::int64_t incx, float *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    copy_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::copy(queue, n, x, incx, y, incy, dependencies);
    copy_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event copy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, const double *x,
                                      std::int64_t incx, double *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    copy_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::copy(queue, n, x, incx, y, incy, dependencies);
    copy_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event copy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<float> *x, std::int64_t incx,
                                      std::complex<float> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    copy_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::copy(queue, n, x, incx, y, incy, dependencies);
    copy_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event copy<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<double> *x, std::int64_t incx,
                                      std::complex<double> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    copy_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::copy(queue, n, x, incx, y, incy, dependencies);
    copy_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event hemv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::complex<float> alpha, const std::complex<float> *a,
                                      std::int64_t lda, const std::complex<float> *x,
                                      std::int64_t incx, std::complex<float> beta,
                                      std::complex<float> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                          incy, dependencies);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event hemv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::complex<double> alpha, const std::complex<double> *a,
                                      std::int64_t lda, const std::complex<double> *x,
                                      std::int64_t incx, std::complex<double> beta,
                                      std::complex<double> *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::hemv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                          incy, dependencies);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event gemmt<backend::cublas>(
    cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
    std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
    float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc, dependencies);
    auto done = oneapi::mkl::cublas::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda,
                                           b, ldb, beta, c, ldc, dependencies);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, dependencies);
    return done;
}

template <>
cl::sycl::event gemmt<backend::cublas>(
    cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
    std::int64_t k, double alpha, const double *a, std::int64_t lda, const double *b,
    std::int64_t ldb, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc, dependencies);
    auto done = oneapi::mkl::cublas::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda,
                                           b, ldb, beta, c, ldc, dependencies);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, dependencies);
    return done;
}

template <>
cl::sycl::event gemmt<backend::cublas>(
    cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc, dependencies);
    auto done = oneapi::mkl::cublas::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda,
                                           b, ldb, beta, c, ldc, dependencies);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, dependencies);
    return done;
}

template <>
cl::sycl::event gemmt<backend::cublas>(
    cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc, dependencies);
    auto done = oneapi::mkl::cublas::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda,
                                           b, ldb, beta, c, ldc, dependencies);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, dependencies);
    return done;
}

template <>
cl::sycl::event sbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::int64_t k, float alpha, const float *a, std::int64_t lda,
                                      const float *x, std::int64_t incx, float beta, float *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::cublas::sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                          incy, dependencies);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

template <>
cl::sycl::event sbmv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      std::int64_t k, double alpha, const double *a,
                                      std::int64_t lda, const double *x, std::int64_t incx,
                                      double beta, double *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = oneapi::mkl::cublas::sbmv(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                                          incy, dependencies);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

template <>
cl::sycl::event asum<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<float> *x, std::int64_t incx,
                                      float *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    asum_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::asum(queue, n, x, incx, result, dependencies);
    asum_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event asum<backend::cublas>(cl::sycl::queue &queue, std::int64_t n,
                                      const std::complex<double> *x, std::int64_t incx,
                                      double *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    asum_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::asum(queue, n, x, incx, result, dependencies);
    asum_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event asum<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                      std::int64_t incx, float *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    asum_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::asum(queue, n, x, incx, result, dependencies);
    asum_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event asum<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, const double *x,
                                      std::int64_t incx, double *result,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    asum_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::asum(queue, n, x, incx, result, dependencies);
    asum_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event tbsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, std::int64_t k,
                                      const float *a, std::int64_t lda, float *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                          incx, dependencies);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tbsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, std::int64_t k,
                                      const double *a, std::int64_t lda, double *x,
                                      std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                          incx, dependencies);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tbsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, std::int64_t k,
                                      const std::complex<float> *a, std::int64_t lda,
                                      std::complex<float> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                          incx, dependencies);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event tbsv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                      diag unit_diag, std::int64_t n, std::int64_t k,
                                      const std::complex<double> *a, std::int64_t lda,
                                      std::complex<double> *x, std::int64_t incx,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = oneapi::mkl::cublas::tbsv(queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                                          incx, dependencies);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

template <>
cl::sycl::event spr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      float alpha, const float *x, std::int64_t incx,
                                      const float *y, std::int64_t incy, float *a,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    auto done =
        oneapi::mkl::cublas::spr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    return done;
}

template <>
cl::sycl::event spr2<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      double alpha, const double *x, std::int64_t incx,
                                      const double *y, std::int64_t incy, double *a,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    auto done =
        oneapi::mkl::cublas::spr2(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    return done;
}

template <>
cl::sycl::event iamax<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    iamax_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::iamax(queue, n, x, incx, result, dependencies);
    iamax_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event iamax<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    iamax_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::iamax(queue, n, x, incx, result, dependencies);
    iamax_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event iamax<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    iamax_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::iamax(queue, n, x, incx, result, dependencies);
    iamax_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event iamax<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    iamax_precondition(queue, n, x, incx, result, dependencies);
    auto done = oneapi::mkl::cublas::iamax(queue, n, x, incx, result, dependencies);
    iamax_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

template <>
cl::sycl::event rotm<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, float *x,
                                      std::int64_t incx, float *y, std::int64_t incy, float *param,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rotm_precondition(queue, n, x, incx, y, incy, param, dependencies);
    auto done = oneapi::mkl::cublas::rotm(queue, n, x, incx, y, incy, param, dependencies);
    rotm_postcondition(queue, n, x, incx, y, incy, param, dependencies);
    return done;
}

template <>
cl::sycl::event rotm<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, double *x,
                                      std::int64_t incx, double *y, std::int64_t incy,
                                      double *param,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rotm_precondition(queue, n, x, incx, y, incy, param, dependencies);
    auto done = oneapi::mkl::cublas::rotm(queue, n, x, incx, y, incy, param, dependencies);
    rotm_postcondition(queue, n, x, incx, y, incy, param, dependencies);
    return done;
}

template <>
cl::sycl::event rotg<backend::cublas>(cl::sycl::queue &queue, float *a, float *b, float *c,
                                      float *s,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rotg_precondition(queue, a, b, c, s, dependencies);
    auto done = oneapi::mkl::cublas::rotg(queue, a, b, c, s, dependencies);
    rotg_postcondition(queue, a, b, c, s, dependencies);
    return done;
}

template <>
cl::sycl::event rotg<backend::cublas>(cl::sycl::queue &queue, double *a, double *b, double *c,
                                      double *s,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rotg_precondition(queue, a, b, c, s, dependencies);
    auto done = oneapi::mkl::cublas::rotg(queue, a, b, c, s, dependencies);
    rotg_postcondition(queue, a, b, c, s, dependencies);
    return done;
}

template <>
cl::sycl::event rotg<backend::cublas>(cl::sycl::queue &queue, std::complex<float> *a,
                                      std::complex<float> *b, float *c, std::complex<float> *s,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rotg_precondition(queue, a, b, c, s, dependencies);
    auto done = oneapi::mkl::cublas::rotg(queue, a, b, c, s, dependencies);
    rotg_postcondition(queue, a, b, c, s, dependencies);
    return done;
}

template <>
cl::sycl::event rotg<backend::cublas>(cl::sycl::queue &queue, std::complex<double> *a,
                                      std::complex<double> *b, double *c, std::complex<double> *s,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    rotg_precondition(queue, a, b, c, s, dependencies);
    auto done = oneapi::mkl::cublas::rotg(queue, a, b, c, s, dependencies);
    rotg_postcondition(queue, a, b, c, s, dependencies);
    return done;
}

template <>
cl::sycl::event sdsdot<backend::cublas>(
    cl::sycl::queue &queue, std::int64_t n, float sb, const float *x, std::int64_t incx,
    const float *y, std::int64_t incy, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    sdsdot_precondition(queue, n, sb, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::cublas::sdsdot(queue, n, sb, x, incx, y, incy, result, dependencies);
    sdsdot_postcondition(queue, n, sb, x, incx, y, incy, result, dependencies);
    return done;
}

template <>
cl::sycl::event her2k<backend::cublas>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, float beta, std::complex<float> *c,
    std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    her2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = oneapi::mkl::cublas::her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                           beta, c, ldc, dependencies);
    her2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

template <>
cl::sycl::event her2k<backend::cublas>(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, double beta, std::complex<double> *c,
    std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    her2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = oneapi::mkl::cublas::her2k(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                                           beta, c, ldc, dependencies);
    her2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

template <>
cl::sycl::event dot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                     std::int64_t incx, const float *y, std::int64_t incy,
                                     float *result,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    dot_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::cublas::dot(queue, n, x, incx, y, incy, result, dependencies);
    dot_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

template <>
cl::sycl::event dot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, const double *x,
                                     std::int64_t incx, const double *y, std::int64_t incy,
                                     double *result,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    dot_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::cublas::dot(queue, n, x, incx, y, incy, result, dependencies);
    dot_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

template <>
cl::sycl::event dot<backend::cublas>(cl::sycl::queue &queue, std::int64_t n, const float *x,
                                     std::int64_t incx, const float *y, std::int64_t incy,
                                     double *result,
                                     const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    dot_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done = oneapi::mkl::cublas::dot(queue, n, x, incx, y, incy, result, dependencies);
    dot_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

template <>
cl::sycl::event symv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      float alpha, const float *a, std::int64_t lda, const float *x,
                                      std::int64_t incx, float beta, float *y, std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                          incy, dependencies);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

template <>
cl::sycl::event symv<backend::cublas>(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                                      double alpha, const double *a, std::int64_t lda,
                                      const double *x, std::int64_t incx, double beta, double *y,
                                      std::int64_t incy,
                                      const cl::sycl::vector_class<cl::sycl::event> &dependencies) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = oneapi::mkl::cublas::symv(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                                          incy, dependencies);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

} // namespace blas
} // namespace mkl
} // namespace oneapi

#endif //_DETAIL_CUBLAS_BLAS_HPP_
