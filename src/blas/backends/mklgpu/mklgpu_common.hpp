/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef _MKL_INTERNAL_BLAS_SYCL_GPU_HPP_
#define _MKL_INTERNAL_BLAS_SYCL_GPU_HPP_

#include <CL/sycl.hpp>
#include <complex>

typedef enum { MKL_ROW_MAJOR = 101, MKL_COL_MAJOR = 102 } MKL_LAYOUT;

typedef enum { MKL_NOTRANS = 111, MKL_TRANS = 112, MKL_CONJTRANS = 113 } MKL_TRANSPOSE;

typedef enum { MKL_UPPER = 121, MKL_LOWER = 122 } MKL_UPLO;

typedef enum { MKL_NONUNIT = 131, MKL_UNIT = 132 } MKL_DIAG;

typedef enum { MKL_LEFT = 141, MKL_RIGHT = 142 } MKL_SIDE;

enum CBLAS_OFFSET { CblasRowOffset = 171, CblasColOffset = 172, CblasFixOffset = 173 };
typedef enum CBLAS_OFFSET CBLAS_OFFSET;

namespace mkl {

enum class transpose : char { nontrans = 0, trans = 1, conjtrans = 3, N = 0, T = 1, C = 3 };

inline MKL_TRANSPOSE cblas_convert(oneapi::mkl::transpose t) {
    if (t == oneapi::mkl::transpose::nontrans)
        return MKL_NOTRANS;
    if (t == oneapi::mkl::transpose::trans)
        return MKL_TRANS;
    if (t == oneapi::mkl::transpose::conjtrans)
        return MKL_CONJTRANS;
    return MKL_NOTRANS;
}

inline MKL_UPLO cblas_convert(oneapi::mkl::uplo u) {
    if (u == oneapi::mkl::uplo::upper)
        return MKL_UPPER;
    if (u == oneapi::mkl::uplo::lower)
        return MKL_LOWER;
    return MKL_UPPER;
}

inline MKL_DIAG cblas_convert(oneapi::mkl::diag d) {
    if (d == oneapi::mkl::diag::nonunit)
        return MKL_NONUNIT;
    if (d == oneapi::mkl::diag::unit)
        return MKL_UNIT;
    return MKL_NONUNIT;
}

inline MKL_SIDE cblas_convert(oneapi::mkl::side s) {
    if (s == oneapi::mkl::side::left)
        return MKL_LEFT;
    if (s == oneapi::mkl::side::right)
        return MKL_RIGHT;
    return MKL_LEFT;
}

inline CBLAS_OFFSET cblas_convert(oneapi::mkl::offset o) {
    if (o == oneapi::mkl::offset::fix)
        return CblasFixOffset;
    if (o == oneapi::mkl::offset::column)
        return CblasColOffset;
    return CblasRowOffset;
}
} // namespace mkl

namespace oneapi {
namespace mkl {
namespace gpu {

// Buffer APIs

void sgemm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
           int64_t m, int64_t n, int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
           int64_t ldc);

void dgemm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
           int64_t m, int64_t n, int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
           int64_t lda, cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta,
           cl::sycl::buffer<double, 1> &c, int64_t ldc);

void cgemm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
           int64_t m, int64_t n, int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zgemm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
           int64_t m, int64_t n, int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void ssymm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
           int64_t ldc);

void dsymm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
           int64_t ldc);

void csymm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zsymm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void chemm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zhemm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void ssyrk(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
           float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc);

void dsyrk(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           double beta, cl::sycl::buffer<double, 1> &c, int64_t ldc);

void csyrk(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zsyrk(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void cherk(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zherk(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, double alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, double beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void ssyr2k(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
            cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
            int64_t ldc);

void dsyr2k(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
            cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta,
            cl::sycl::buffer<double, 1> &c, int64_t ldc);

void csyr2k(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, std::complex<float> alpha,
            cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
            cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zsyr2k(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, std::complex<double> alpha,
            cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
            cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void cher2k(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, std::complex<float> alpha,
            cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, float beta,
            cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zher2k(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, std::complex<double> alpha,
            cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, double beta,
            cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void strmm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n, float alpha,
           cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb);

void dtrmm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n, double alpha,
           cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
           int64_t ldb);

void ctrmm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb);

void ztrmm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb);

void strsm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n, float alpha,
           cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb);

void dtrsm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n, double alpha,
           cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
           int64_t ldb);

void ctrsm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb);

void ztrsm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb);

void sgemv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
           int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, int64_t incy);

void dgemv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &x, int64_t incx, double beta,
           cl::sycl::buffer<double, 1> &y, int64_t incy);

void cgemv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zgemv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void sgbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           int64_t kl, int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &x, int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
           int64_t incy);

void dgbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           int64_t kl, int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &x, int64_t incx, double beta,
           cl::sycl::buffer<double, 1> &y, int64_t incy);

void cgbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           int64_t kl, int64_t ku, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zgbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           int64_t kl, int64_t ku, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void sger(cl::sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
          cl::sycl::buffer<float, 1> &a, int64_t lda);

void dger(cl::sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
          int64_t incy, cl::sycl::buffer<double, 1> &a, int64_t lda);

void cgerc(cl::sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zgerc(cl::sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void cgeru(cl::sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zgeru(cl::sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void chbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zhbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void chemv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zhemv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void cher(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zher(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void cher2(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zher2(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void chpmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zhpmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void chpr(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &a);

void zhpr(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &a);

void chpr2(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<float>, 1> &a);

void zhpr2(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<double>, 1> &a);

void ssbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k,
           float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
           int64_t incx, float beta, cl::sycl::buffer<float, 1> &y, int64_t incy);

void dsbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k,
           double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &x, int64_t incx, double beta,
           cl::sycl::buffer<double, 1> &y, int64_t incy);

void sspmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, float alpha,
           cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, int64_t incx, float beta,
           cl::sycl::buffer<float, 1> &y, int64_t incy);

void dspmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, double alpha,
           cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, int64_t incx,
           double beta, cl::sycl::buffer<double, 1> &y, int64_t incy);

void sspr(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &a);

void dspr(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &a);

void sspr2(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
           cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
           cl::sycl::buffer<float, 1> &a);

void dspr2(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
           cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
           int64_t incy, cl::sycl::buffer<double, 1> &a);

void ssymv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, float alpha,
           cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
           float beta, cl::sycl::buffer<float, 1> &y, int64_t incy);

void dsymv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, double alpha,
           cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
           int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, int64_t incy);

void ssyr(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &a, int64_t lda);

void dsyr(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &a,
          int64_t lda);

void ssyr2(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
           cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
           cl::sycl::buffer<float, 1> &a, int64_t lda);

void dsyr2(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
           cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
           int64_t incy, cl::sycl::buffer<double, 1> &a, int64_t lda);

void stbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &x, int64_t incx);

void dtbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &x, int64_t incx);

void ctbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztbmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void stbsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &x, int64_t incx);

void dtbsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &x, int64_t incx);

void ctbsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztbsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void stpmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
           int64_t incx);

void dtpmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
           int64_t incx);

void ctpmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztpmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void stpsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
           int64_t incx);

void dtpsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
           int64_t incx);

void ctpsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztpsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void strmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &x, int64_t incx);

void dtrmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &x, int64_t incx);

void ctrmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztrmv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void strsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &x, int64_t incx);

void dtrsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &x, int64_t incx);

void ctrsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztrsv(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void scasum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
            int64_t incx, cl::sycl::buffer<float, 1> &result);

void dzasum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
            int64_t incx, cl::sycl::buffer<double, 1> &result);

void sasum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<float, 1> &result);

void dasum(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
           cl::sycl::buffer<double, 1> &result);

void saxpy(cl::sycl::queue &queue, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
           int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy);

void daxpy(cl::sycl::queue &queue, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
           int64_t incx, cl::sycl::buffer<double, 1> &y, int64_t incy);

void caxpy(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zaxpy(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void scopy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<float, 1> &y, int64_t incy);

void dcopy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
           cl::sycl::buffer<double, 1> &y, int64_t incy);

void ccopy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zcopy(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void sdot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<float, 1> &result);

void ddot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &result);

void sdsdot(cl::sycl::queue &queue, int64_t n, float sb, cl::sycl::buffer<float, 1> &x,
            int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
            cl::sycl::buffer<float, 1> &result);

void dsdot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &result);

void cdotc(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<float>, 1> &result);

void zdotc(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<double>, 1> &result);

void cdotu(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<float>, 1> &result);

void zdotu(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<double>, 1> &result);

void scnrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
            int64_t incx, cl::sycl::buffer<float, 1> &result);

void dznrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
            int64_t incx, cl::sycl::buffer<double, 1> &result);

void snrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<float, 1> &result);

void dnrm2(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
           cl::sycl::buffer<double, 1> &result);

void csrot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy, float c,
           float s);

void zdrot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy, double c,
           double s);

void srot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
          cl::sycl::buffer<float, 1> &y, int64_t incy, float c, float s);

void drot(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
          cl::sycl::buffer<double, 1> &y, int64_t incy, double c, double s);

void srotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &b,
           cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<float, 1> &s);

void drotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &b,
           cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<double, 1> &s);

void crotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
           cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
           cl::sycl::buffer<std::complex<float>, 1> &s);

void zrotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
           cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<double, 1> &c,
           cl::sycl::buffer<std::complex<double>, 1> &s);

void srotm(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<float, 1> &param);

void drotm(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
           cl::sycl::buffer<double, 1> &y, int64_t incy, cl::sycl::buffer<double, 1> &param);

void srotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1, cl::sycl::buffer<float, 1> &d2,
            cl::sycl::buffer<float, 1> &x1, float y1, cl::sycl::buffer<float, 1> &param);

void drotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
            cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1, double y1,
            cl::sycl::buffer<double, 1> &param);

void sscal(cl::sycl::queue &queue, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
           int64_t incx);

void dscal(cl::sycl::queue &queue, int64_t n, double alpha, cl::sycl::buffer<double, 1> &x,
           int64_t incx);

void cscal(cl::sycl::queue &queue, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void zscal(cl::sycl::queue &queue, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void csscal(cl::sycl::queue &queue, int64_t n, float alpha,
            cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void zdscal(cl::sycl::queue &queue, int64_t n, double alpha,
            cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void sswap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
           cl::sycl::buffer<float, 1> &y, int64_t incy);

void dswap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
           cl::sycl::buffer<double, 1> &y, int64_t incy);

void cswap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zswap(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
           int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void isamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
            cl::sycl::buffer<int64_t, 1> &result);

void idamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
            cl::sycl::buffer<int64_t, 1> &result);

void icamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
            int64_t incx, cl::sycl::buffer<int64_t, 1> &result);

void izamax(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
            int64_t incx, cl::sycl::buffer<int64_t, 1> &result);

void isamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<float, 1> &x, int64_t incx,
            cl::sycl::buffer<int64_t, 1> &result);

void idamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<double, 1> &x, int64_t incx,
            cl::sycl::buffer<int64_t, 1> &result);

void icamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<float>, 1> &x,
            int64_t incx, cl::sycl::buffer<int64_t, 1> &result);

void izamin(cl::sycl::queue &queue, int64_t n, cl::sycl::buffer<std::complex<double>, 1> &x,
            int64_t incx, cl::sycl::buffer<int64_t, 1> &result);

void sgemm_batch(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha,
                 cl::sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                 cl::sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b, float beta,
                 cl::sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                 int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

void dgemm_batch(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, double alpha,
                 cl::sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                 cl::sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b, double beta,
                 cl::sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c, int64_t batch_size,
                 int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

void cgemm_batch(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, std::complex<float> alpha,
                 cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                 cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, int64_t stride_b,
                 std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc,
                 int64_t stride_c, int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0,
                 int64_t offset_c = 0);

void zgemm_batch(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, std::complex<double> alpha,
                 cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                 cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                 std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                 int64_t ldc, int64_t stride_c, int64_t batch_size, int64_t offset_a = 0,
                 int64_t offset_b = 0, int64_t offset_c = 0);

void strsm_batch(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                 MKL_UPLO upper_lower, MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m,
                 int64_t n, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
                 int64_t stride_a, cl::sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b,
                 int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0);

void dtrsm_batch(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                 MKL_UPLO upper_lower, MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m,
                 int64_t n, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
                 int64_t stride_a, cl::sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b,
                 int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0);

void ctrsm_batch(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                 MKL_UPLO upper_lower, MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m,
                 int64_t n, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                 int64_t lda, int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                 int64_t ldb, int64_t stride_b, int64_t batch_size, int64_t offset_a = 0,
                 int64_t offset_b = 0);

void ztrsm_batch(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                 MKL_UPLO upper_lower, MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m,
                 int64_t n, std::complex<double> alpha,
                 cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                 cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                 int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0);

void sgemmt(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
            int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta,
            cl::sycl::buffer<float, 1> &c, int64_t ldc);

void dgemmt(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, double alpha,
            cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
            int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, int64_t ldc);

void zgemmt(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, std::complex<double> alpha,
            cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
            cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void cgemmt(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, std::complex<float> alpha,
            cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
            cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void hgemm(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
           int64_t m, int64_t n, int64_t k, cl::sycl::half alpha,
           cl::sycl::buffer<cl::sycl::half, 1> &a, int64_t lda,
           cl::sycl::buffer<cl::sycl::half, 1> &b, int64_t ldb, cl::sycl::half beta,
           cl::sycl::buffer<cl::sycl::half, 1> &c, int64_t ldc);

void gemm_f16f16f32(cl::sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                    MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha,
                    cl::sycl::buffer<cl::sycl::half, 1> &a, int64_t lda,
                    cl::sycl::buffer<cl::sycl::half, 1> &b, int64_t ldb, float beta,
                    cl::sycl::buffer<float, 1> &c, int64_t ldc);

cl::sycl::event gemm_s8u8s32_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                                  MKL_TRANSPOSE transb, CBLAS_OFFSET offsetc, int64_t m, int64_t n,
                                  int64_t k, float alpha, cl::sycl::buffer<int8_t, 1> *a,
                                  int64_t lda, int8_t ao, cl::sycl::buffer<uint8_t, 1> *b,
                                  int64_t ldb, uint8_t bo, float beta,
                                  cl::sycl::buffer<int32_t, 1> *c, int64_t ldc,
                                  cl::sycl::buffer<int32_t, 1> *co, int64_t offset_a = 0,
                                  int64_t offset_b = 0, int64_t offset_c = 0,
                                  int64_t offset_co = 0);

// USM APIs

cl::sycl::event sgemm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                           MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha,
                           const float *a, int64_t lda, const float *b, int64_t ldb, float beta,
                           float *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event dgemm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                           MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, double alpha,
                           const double *a, int64_t lda, const double *b, int64_t ldb, double beta,
                           double *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event cgemm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                           MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                           std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                           const std::complex<float> *b, int64_t ldb, std::complex<float> beta,
                           std::complex<float> *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event zgemm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                           MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                           std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                           const std::complex<double> *b, int64_t ldb, std::complex<double> beta,
                           std::complex<double> *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event ssymm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, int64_t m, int64_t n, float alpha, const float *a,
                           int64_t lda, const float *b, int64_t ldb, float beta, float *c,
                           int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event dsymm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, int64_t m, int64_t n, double alpha,
                           const double *a, int64_t lda, const double *b, int64_t ldb, double beta,
                           double *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event csymm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, int64_t m, int64_t n, std::complex<float> alpha,
                           const std::complex<float> *a, int64_t lda, const std::complex<float> *b,
                           int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                           int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event zsymm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, int64_t m, int64_t n, std::complex<double> alpha,
                           const std::complex<double> *a, int64_t lda,
                           const std::complex<double> *b, int64_t ldb, std::complex<double> beta,
                           std::complex<double> *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event chemm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, int64_t m, int64_t n, std::complex<float> alpha,
                           const std::complex<float> *a, int64_t lda, const std::complex<float> *b,
                           int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                           int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event zhemm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, int64_t m, int64_t n, std::complex<double> alpha,
                           const std::complex<double> *a, int64_t lda,
                           const std::complex<double> *b, int64_t ldb, std::complex<double> beta,
                           std::complex<double> *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event ssyrk_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                           MKL_TRANSPOSE trans, int64_t n, int64_t k, float alpha, const float *a,
                           int64_t lda, float beta, float *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_c = 0);

cl::sycl::event dsyrk_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                           MKL_TRANSPOSE trans, int64_t n, int64_t k, double alpha, const double *a,
                           int64_t lda, double beta, double *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_c = 0);

cl::sycl::event csyrk_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                           MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<float> alpha,
                           const std::complex<float> *a, int64_t lda, std::complex<float> beta,
                           std::complex<float> *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_c = 0);

cl::sycl::event zsyrk_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                           MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<double> alpha,
                           const std::complex<double> *a, int64_t lda, std::complex<double> beta,
                           std::complex<double> *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_c = 0);

cl::sycl::event cherk_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                           MKL_TRANSPOSE trans, int64_t n, int64_t k, float alpha,
                           const std::complex<float> *a, int64_t lda, float beta,
                           std::complex<float> *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_c = 0);

cl::sycl::event zherk_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                           MKL_TRANSPOSE trans, int64_t n, int64_t k, double alpha,
                           const std::complex<double> *a, int64_t lda, double beta,
                           std::complex<double> *c, int64_t ldc,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_c = 0);

cl::sycl::event ssyr2k_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                            MKL_TRANSPOSE trans, int64_t n, int64_t k, float alpha, const float *a,
                            int64_t lda, const float *b, int64_t ldb, float beta, float *c,
                            int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                            int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event dsyr2k_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                            MKL_TRANSPOSE trans, int64_t n, int64_t k, double alpha,
                            const double *a, int64_t lda, const double *b, int64_t ldb, double beta,
                            double *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                            int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event csyr2k_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                            MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<float> alpha,
                            const std::complex<float> *a, int64_t lda, const std::complex<float> *b,
                            int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                            int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                            int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event zsyr2k_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                            MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<double> alpha,
                            const std::complex<double> *a, int64_t lda,
                            const std::complex<double> *b, int64_t ldb, std::complex<double> beta,
                            std::complex<double> *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                            int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event cher2k_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                            MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<float> alpha,
                            const std::complex<float> *a, int64_t lda, const std::complex<float> *b,
                            int64_t ldb, float beta, std::complex<float> *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                            int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event zher2k_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                            MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<double> alpha,
                            const std::complex<double> *a, int64_t lda,
                            const std::complex<double> *b, int64_t ldb, double beta,
                            std::complex<double> *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                            int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event strmm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                           int64_t m, int64_t n, float alpha, const float *a, int64_t lda, float *b,
                           int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0);

cl::sycl::event dtrmm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                           int64_t m, int64_t n, double alpha, const double *a, int64_t lda,
                           double *b, int64_t ldb,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0);

cl::sycl::event ctrmm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                           int64_t m, int64_t n, std::complex<float> alpha,
                           const std::complex<float> *a, int64_t lda, std::complex<float> *b,
                           int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0);

cl::sycl::event ztrmm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                           int64_t m, int64_t n, std::complex<double> alpha,
                           const std::complex<double> *a, int64_t lda, std::complex<double> *b,
                           int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0);

cl::sycl::event strsm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                           int64_t m, int64_t n, float alpha, const float *a, int64_t lda, float *b,
                           int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0);

cl::sycl::event dtrsm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                           int64_t m, int64_t n, double alpha, const double *a, int64_t lda,
                           double *b, int64_t ldb,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0);

cl::sycl::event ctrsm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                           int64_t m, int64_t n, std::complex<float> alpha,
                           const std::complex<float> *a, int64_t lda, std::complex<float> *b,
                           int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0);

cl::sycl::event ztrsm_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                           MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                           int64_t m, int64_t n, std::complex<double> alpha,
                           const std::complex<double> *a, int64_t lda, std::complex<double> *b,
                           int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                           int64_t offset_a = 0, int64_t offset_b = 0);

cl::sycl::event sgemv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans,
                           int64_t m, int64_t n, float alpha, const float *a, int64_t lda,
                           const float *x, int64_t incx, float beta, float *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dgemv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans,
                           int64_t m, int64_t n, double alpha, const double *a, int64_t lda,
                           const double *x, int64_t incx, double beta, double *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event cgemv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans,
                           int64_t m, int64_t n, std::complex<float> alpha,
                           const std::complex<float> *a, int64_t lda, const std::complex<float> *x,
                           int64_t incx, std::complex<float> beta, std::complex<float> *y,
                           int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zgemv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans,
                           int64_t m, int64_t n, std::complex<double> alpha,
                           const std::complex<double> *a, int64_t lda,
                           const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                           std::complex<double> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sgbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans,
                           int64_t m, int64_t n, int64_t kl, int64_t ku, float alpha,
                           const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                           float *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dgbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans,
                           int64_t m, int64_t n, int64_t kl, int64_t ku, double alpha,
                           const double *a, int64_t lda, const double *x, int64_t incx, double beta,
                           double *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event cgbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans,
                           int64_t m, int64_t n, int64_t kl, int64_t ku, std::complex<float> alpha,
                           const std::complex<float> *a, int64_t lda, const std::complex<float> *x,
                           int64_t incx, std::complex<float> beta, std::complex<float> *y,
                           int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zgbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans,
                           int64_t m, int64_t n, int64_t kl, int64_t ku, std::complex<double> alpha,
                           const std::complex<double> *a, int64_t lda,
                           const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                           std::complex<double> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sger_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n,
                          float alpha, const float *x, int64_t incx, const float *y, int64_t incy,
                          float *a, int64_t lda,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dger_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n,
                          double alpha, const double *x, int64_t incx, const double *y,
                          int64_t incy, double *a, int64_t lda,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event cgerc_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *x, int64_t incx,
                           const std::complex<float> *y, int64_t incy, std::complex<float> *a,
                           int64_t lda,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zgerc_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                           const std::complex<double> *y, int64_t incy, std::complex<double> *a,
                           int64_t lda,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event cgeru_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *x, int64_t incx,
                           const std::complex<float> *y, int64_t incy, std::complex<float> *a,
                           int64_t lda,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zgeru_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                           const std::complex<double> *y, int64_t incy, std::complex<double> *a,
                           int64_t lda,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event chbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                           int64_t lda, const std::complex<float> *x, int64_t incx,
                           std::complex<float> beta, std::complex<float> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zhbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                           int64_t lda, const std::complex<double> *x, int64_t incx,
                           std::complex<double> beta, std::complex<double> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event chemv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                           const std::complex<float> *x, int64_t incx, std::complex<float> beta,
                           std::complex<float> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zhemv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                           const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                           std::complex<double> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event cher_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                          float alpha, const std::complex<float> *x, int64_t incx,
                          std::complex<float> *a, int64_t lda,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zher_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                          double alpha, const std::complex<double> *x, int64_t incx,
                          std::complex<double> *a, int64_t lda,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event cher2_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *x, int64_t incx,
                           const std::complex<float> *y, int64_t incy, std::complex<float> *a,
                           int64_t lda,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zher2_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                           const std::complex<double> *y, int64_t incy, std::complex<double> *a,
                           int64_t lda,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event chpmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *a,
                           const std::complex<float> *x, int64_t incx, std::complex<float> beta,
                           std::complex<float> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zhpmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *a,
                           const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                           std::complex<double> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event chpr_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                          float alpha, const std::complex<float> *x, int64_t incx,
                          std::complex<float> *a,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zhpr_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                          double alpha, const std::complex<double> *x, int64_t incx,
                          std::complex<double> *a,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event chpr2_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                           std::complex<float> alpha, const std::complex<float> *x, int64_t incx,
                           const std::complex<float> *y, int64_t incy, std::complex<float> *a,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zhpr2_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                           std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                           const std::complex<double> *y, int64_t incy, std::complex<double> *a,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ssbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           int64_t k, float alpha, const float *a, int64_t lda, const float *x,
                           int64_t incx, float beta, float *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dsbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           int64_t k, double alpha, const double *a, int64_t lda, const double *x,
                           int64_t incx, double beta, double *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sspmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           float alpha, const float *a, const float *x, int64_t incx, float beta,
                           float *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dspmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           double alpha, const double *a, const double *x, int64_t incx,
                           double beta, double *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sspr_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                          float alpha, const float *x, int64_t incx, float *a,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dspr_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                          double alpha, const double *x, int64_t incx, double *a,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sspr2_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                           float alpha, const float *x, int64_t incx, const float *y, int64_t incy,
                           float *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dspr2_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                           double alpha, const double *x, int64_t incx, const double *y,
                           int64_t incy, double *a,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ssymv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           float alpha, const float *a, int64_t lda, const float *x, int64_t incx,
                           float beta, float *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dsymv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                           double alpha, const double *a, int64_t lda, const double *x,
                           int64_t incx, double beta, double *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ssyr_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                          float alpha, const float *x, int64_t incx, float *a, int64_t lda,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dsyr_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                          double alpha, const double *x, int64_t incx, double *a, int64_t lda,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ssyr2_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                           float alpha, const float *x, int64_t incx, const float *y, int64_t incy,
                           float *a, int64_t lda,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dsyr2_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                           double alpha, const double *x, int64_t incx, const double *y,
                           int64_t incy, double *a, int64_t lda,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event stbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, int64_t k, const float *a,
                           int64_t lda, float *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dtbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, int64_t k,
                           const double *a, int64_t lda, double *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ctbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, int64_t k,
                           const std::complex<float> *a, int64_t lda, std::complex<float> *x,
                           int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ztbmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, int64_t k,
                           const std::complex<double> *a, int64_t lda, std::complex<double> *x,
                           int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event stbsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, int64_t k, const float *a,
                           int64_t lda, float *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dtbsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, int64_t k,
                           const double *a, int64_t lda, double *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ctbsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, int64_t k,
                           const std::complex<float> *a, int64_t lda, std::complex<float> *x,
                           int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ztbsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, int64_t k,
                           const std::complex<double> *a, int64_t lda, std::complex<double> *x,
                           int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event stpmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, const float *a, float *x,
                           int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dtpmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, const double *a,
                           double *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ctpmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
                           const std::complex<float> *a, std::complex<float> *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ztpmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
                           const std::complex<double> *a, std::complex<double> *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event stpsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, const float *a, float *x,
                           int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dtpsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, const double *a,
                           double *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ctpsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
                           const std::complex<float> *a, std::complex<float> *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ztpsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
                           const std::complex<double> *a, std::complex<double> *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event strmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, const float *a,
                           int64_t lda, float *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dtrmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, const double *a,
                           int64_t lda, double *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ctrmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
                           const std::complex<float> *a, int64_t lda, std::complex<float> *x,
                           int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ztrmv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
                           const std::complex<double> *a, int64_t lda, std::complex<double> *x,
                           int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event strsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, const float *a,
                           int64_t lda, float *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dtrsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n, const double *a,
                           int64_t lda, double *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ctrsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
                           const std::complex<float> *a, int64_t lda, std::complex<float> *x,
                           int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ztrsv_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo,
                           MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
                           const std::complex<double> *a, int64_t lda, std::complex<double> *x,
                           int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event scasum_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<float> *x,
                            int64_t incx, float *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dzasum_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<double> *x,
                            int64_t incx, double *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sasum_sycl(cl::sycl::queue *queue, int64_t n, const float *x, int64_t incx,
                           float *result,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dasum_sycl(cl::sycl::queue *queue, int64_t n, const double *x, int64_t incx,
                           double *result,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event saxpy_sycl(cl::sycl::queue *queue, int64_t n, float alpha, const float *x,
                           int64_t incx, float *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event daxpy_sycl(cl::sycl::queue *queue, int64_t n, double alpha, const double *x,
                           int64_t incx, double *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event caxpy_sycl(cl::sycl::queue *queue, int64_t n, std::complex<float> alpha,
                           const std::complex<float> *x, int64_t incx, std::complex<float> *y,
                           int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zaxpy_sycl(cl::sycl::queue *queue, int64_t n, std::complex<double> alpha,
                           const std::complex<double> *x, int64_t incx, std::complex<double> *y,
                           int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event scopy_sycl(cl::sycl::queue *queue, int64_t n, const float *x, int64_t incx,
                           float *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dcopy_sycl(cl::sycl::queue *queue, int64_t n, const double *x, int64_t incx,
                           double *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ccopy_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<float> *x,
                           int64_t incx, std::complex<float> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zcopy_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<double> *x,
                           int64_t incx, std::complex<double> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sdot_sycl(cl::sycl::queue *queue, int64_t n, const float *x, int64_t incx,
                          const float *y, int64_t incy, float *result,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event ddot_sycl(cl::sycl::queue *queue, int64_t n, const double *x, int64_t incx,
                          const double *y, int64_t incy, double *result,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sdsdot_sycl(cl::sycl::queue *queue, int64_t n, float sb, const float *x,
                            int64_t incx, const float *y, int64_t incy, float *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dsdot_sycl(cl::sycl::queue *queue, int64_t n, const float *x, int64_t incx,
                           const float *y, int64_t incy, double *result,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event cdotc_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<float> *x,
                           int64_t incx, const std::complex<float> *y, int64_t incy,
                           std::complex<float> *result,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zdotc_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<double> *x,
                           int64_t incx, const std::complex<double> *y, int64_t incy,
                           std::complex<double> *result,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event cdotu_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<float> *x,
                           int64_t incx, const std::complex<float> *y, int64_t incy,
                           std::complex<float> *result,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zdotu_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<double> *x,
                           int64_t incx, const std::complex<double> *y, int64_t incy,
                           std::complex<double> *result,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event scnrm2_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<float> *x,
                            int64_t incx, float *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dznrm2_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<double> *x,
                            int64_t incx, double *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event snrm2_sycl(cl::sycl::queue *queue, int64_t n, const float *x, int64_t incx,
                           float *result,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dnrm2_sycl(cl::sycl::queue *queue, int64_t n, const double *x, int64_t incx,
                           double *result,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event csrot_sycl(cl::sycl::queue *queue, int64_t n, std::complex<float> *x, int64_t incx,
                           std::complex<float> *y, int64_t incy, float c, float s,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zdrot_sycl(cl::sycl::queue *queue, int64_t n, std::complex<double> *x, int64_t incx,
                           std::complex<double> *y, int64_t incy, double c, double s,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event srot_sycl(cl::sycl::queue *queue, int64_t n, float *x, int64_t incx, float *y,
                          int64_t incy, float c, float s,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event drot_sycl(cl::sycl::queue *queue, int64_t n, double *x, int64_t incx, double *y,
                          int64_t incy, double c, double s,
                          const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event srotg_sycl(cl::sycl::queue *queue, float *a, float *b, float *c, float *s,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event drotg_sycl(cl::sycl::queue *queue, double *a, double *b, double *c, double *s,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event crotg_sycl(cl::sycl::queue *queue, std::complex<float> *a, std::complex<float> *b,
                           float *c, std::complex<float> *s,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zrotg_sycl(cl::sycl::queue *queue, std::complex<double> *a, std::complex<double> *b,
                           double *c, std::complex<double> *s,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event srotm_sycl(cl::sycl::queue *queue, int64_t n, float *x, int64_t incx, float *y,
                           int64_t incy, float *param,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event drotm_sycl(cl::sycl::queue *queue, int64_t n, double *x, int64_t incx, double *y,
                           int64_t incy, double *param,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event srotmg_sycl(cl::sycl::queue *queue, float *d1, float *d2, float *x1, float y1,
                            float *param,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event drotmg_sycl(cl::sycl::queue *queue, double *d1, double *d2, double *x1, double y1,
                            double *param,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sscal_sycl(cl::sycl::queue *queue, int64_t n, float alpha, float *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dscal_sycl(cl::sycl::queue *queue, int64_t n, double alpha, double *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event cscal_sycl(cl::sycl::queue *queue, int64_t n, std::complex<float> alpha,
                           std::complex<float> *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zscal_sycl(cl::sycl::queue *queue, int64_t n, std::complex<double> alpha,
                           std::complex<double> *x, int64_t incx,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event csscal_sycl(cl::sycl::queue *queue, int64_t n, float alpha, std::complex<float> *x,
                            int64_t incx,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zdscal_sycl(cl::sycl::queue *queue, int64_t n, double alpha,
                            std::complex<double> *x, int64_t incx,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sswap_sycl(cl::sycl::queue *queue, int64_t n, float *x, int64_t incx, float *y,
                           int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event dswap_sycl(cl::sycl::queue *queue, int64_t n, double *x, int64_t incx, double *y,
                           int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event cswap_sycl(cl::sycl::queue *queue, int64_t n, std::complex<float> *x, int64_t incx,
                           std::complex<float> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zswap_sycl(cl::sycl::queue *queue, int64_t n, std::complex<double> *x, int64_t incx,
                           std::complex<double> *y, int64_t incy,
                           const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event isamax_sycl(cl::sycl::queue *queue, int64_t n, const float *x, int64_t incx,
                            int64_t *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event idamax_sycl(cl::sycl::queue *queue, int64_t n, const double *x, int64_t incx,
                            int64_t *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event icamax_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<float> *x,
                            int64_t incx, int64_t *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event izamax_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<double> *x,
                            int64_t incx, int64_t *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event isamin_sycl(cl::sycl::queue *queue, int64_t n, const float *x, int64_t incx,
                            int64_t *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event idamin_sycl(cl::sycl::queue *queue, int64_t n, const double *x, int64_t incx,
                            int64_t *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event icamin_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<float> *x,
                            int64_t incx, int64_t *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event izamin_sycl(cl::sycl::queue *queue, int64_t n, const std::complex<double> *x,
                            int64_t incx, int64_t *result,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sgemm_batch_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha,
                                 const float *a, int64_t lda, int64_t strideA, const float *b,
                                 int64_t ldb, int64_t strideB, float beta, float *c, int64_t ldc,
                                 int64_t strideC, int64_t group_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                                 int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event dgemm_batch_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                                 double alpha, const double *a, int64_t lda, int64_t strideA,
                                 const double *b, int64_t ldb, int64_t strideB, double beta,
                                 double *c, int64_t ldc, int64_t strideC, int64_t group_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                                 int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event cgemm_batch_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                                 std::complex<float> alpha, const std::complex<float> *a,
                                 int64_t lda, int64_t strideA, const std::complex<float> *b,
                                 int64_t ldb, int64_t strideB, std::complex<float> beta,
                                 std::complex<float> *c, int64_t ldc, int64_t strideC,
                                 int64_t group_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                                 int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event zgemm_batch_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                                 std::complex<double> alpha, const std::complex<double> *a,
                                 int64_t lda, int64_t strideA, const std::complex<double> *b,
                                 int64_t ldb, int64_t strideB, std::complex<double> beta,
                                 std::complex<double> *c, int64_t ldc, int64_t strideC,
                                 int64_t group_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                                 int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event sgemm_batch_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha,
                                 const float **a, int64_t lda, const float **b, int64_t ldb,
                                 float beta, float **c, int64_t ldc, int64_t offset_batch,
                                 int64_t group_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                                 int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event dgemm_batch_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                                 double alpha, const double **a, int64_t lda, const double **b,
                                 int64_t ldb, double beta, double **c, int64_t ldc,
                                 int64_t offset_batch, int64_t group_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                                 int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event cgemm_batch_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                                 std::complex<float> alpha, const std::complex<float> **a,
                                 int64_t lda, const std::complex<float> **b, int64_t ldb,
                                 std::complex<float> beta, std::complex<float> **c, int64_t ldc,
                                 int64_t offset_batch, int64_t group_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                                 int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event zgemm_batch_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                                 MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                                 std::complex<double> alpha, const std::complex<double> **a,
                                 int64_t lda, const std::complex<double> **b, int64_t ldb,
                                 std::complex<double> beta, std::complex<double> **c, int64_t ldc,
                                 int64_t offset_batch, int64_t group_size,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                                 int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event saxpy_batch_sycl(cl::sycl::queue *queue, std::int64_t n, float alpha,
                                 const float **x, std::int64_t incx, float **y, std::int64_t incy,
                                 std::int64_t batch_size, std::int64_t offset,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event daxpy_batch_sycl(cl::sycl::queue *queue, std::int64_t n, double alpha,
                                 const double **x, std::int64_t incx, double **y, std::int64_t incy,
                                 std::int64_t batch_size, std::int64_t offset,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event caxpy_batch_sycl(cl::sycl::queue *queue, std::int64_t n, std::complex<float> alpha,
                                 const std::complex<float> **x, std::int64_t incx,
                                 std::complex<float> **y, std::int64_t incy,
                                 std::int64_t batch_size, std::int64_t offset,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event zaxpy_batch_sycl(cl::sycl::queue *queue, std::int64_t n, std::complex<double> alpha,
                                 const std::complex<double> **x, std::int64_t incx,
                                 std::complex<double> **y, std::int64_t incy,
                                 std::int64_t batch_size, std::int64_t offset,
                                 const cl::sycl::vector_class<cl::sycl::event> &dependencies);

cl::sycl::event sgemmt_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                            MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t n, int64_t k,
                            float alpha, const float *a, int64_t lda, const float *b, int64_t ldb,
                            float beta, float *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                            int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event dgemmt_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                            MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t n, int64_t k,
                            double alpha, const double *a, int64_t lda, const double *b,
                            int64_t ldb, double beta, double *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                            int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event zgemmt_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                            MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t n, int64_t k,
                            std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                            const std::complex<double> *b, int64_t ldb, std::complex<double> beta,
                            std::complex<double> *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                            int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

cl::sycl::event cgemmt_sycl(cl::sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                            MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t n, int64_t k,
                            std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                            const std::complex<float> *b, int64_t ldb, std::complex<float> beta,
                            std::complex<float> *c, int64_t ldc,
                            const cl::sycl::vector_class<cl::sycl::event> &dependencies,
                            int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

} // namespace gpu
} // namespace mkl
} // namespace oneapi
#endif //_MKL_INTERNAL_BLAS_SYCL_GPU_HPP_
