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

#ifndef _MKL_INTERNAL_BLAS_SYCL_GPU_HPP_
#define _MKL_INTERNAL_BLAS_SYCL_GPU_HPP_

#include <CL/sycl.hpp>
#include <complex>

typedef enum { MKL_ROW_MAJOR = 101, MKL_COL_MAJOR = 102 } MKL_LAYOUT;

typedef enum { MKL_NOTRANS = 111, MKL_TRANS = 112, MKL_CONJTRANS = 113 } MKL_TRANSPOSE;

typedef enum { MKL_UPPER = 121, MKL_LOWER = 122 } MKL_UPLO;

typedef enum { MKL_NONUNIT = 131, MKL_UNIT = 132 } MKL_DIAG;

typedef enum { MKL_LEFT = 141, MKL_RIGHT = 142 } MKL_SIDE;

typedef enum {
    MKL_COMPACT_SSE    = 181,
    MKL_COMPACT_AVX    = 182,
    MKL_COMPACT_AVX512 = 183
} MKL_COMPACT_PACK;

enum CBLAS_OFFSET { CblasRowOffset = 171, CblasColOffset = 172, CblasFixOffset = 173 };
typedef enum CBLAS_OFFSET CBLAS_OFFSET;

namespace mkl {

inline MKL_TRANSPOSE cblas_convert(onemkl::transpose t) {
    if (t == onemkl::transpose::nontrans)
        return MKL_NOTRANS;
    if (t == onemkl::transpose::trans)
        return MKL_TRANS;
    if (t == onemkl::transpose::conjtrans)
        return MKL_CONJTRANS;
    return MKL_NOTRANS;
}

inline MKL_UPLO cblas_convert(onemkl::uplo u) {
    if (u == onemkl::uplo::upper)
        return MKL_UPPER;
    if (u == onemkl::uplo::lower)
        return MKL_LOWER;
    return MKL_UPPER;
}

inline MKL_DIAG cblas_convert(onemkl::diag d) {
    if (d == onemkl::diag::nonunit)
        return MKL_NONUNIT;
    if (d == onemkl::diag::unit)
        return MKL_UNIT;
    return MKL_NONUNIT;
}

inline MKL_SIDE cblas_convert(onemkl::side s) {
    if (s == onemkl::side::left)
        return MKL_LEFT;
    if (s == onemkl::side::right)
        return MKL_RIGHT;
    return MKL_LEFT;
}

namespace gpu {

// gemm

void sgemm(cl::sycl::queue &queue, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m, int64_t n,
           int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
           int64_t ldc);

void dgemm(cl::sycl::queue &queue, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m, int64_t n,
           int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
           int64_t ldc);

void cgemm(cl::sycl::queue &queue, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m, int64_t n,
           int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zgemm(cl::sycl::queue &queue, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m, int64_t n,
           int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

// symm

void ssymm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, int64_t m, int64_t n,
           float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b,
           int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc);

void dsymm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, int64_t m, int64_t n,
           double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
           int64_t ldc);

void csymm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, int64_t m, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zsymm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, int64_t m, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

// hemm

void chemm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, int64_t m, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zhemm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, int64_t m, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

// syrk
void ssyrk(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
           float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, float beta,
           cl::sycl::buffer<float, 1> &c, int64_t ldc);

void dsyrk(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
           double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda, double beta,
           cl::sycl::buffer<double, 1> &c, int64_t ldc);

void csyrk(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zsyrk(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

// herk

void cherk(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
           float alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda, float beta,
           cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zherk(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
           double alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda, double beta,
           cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

// syr2k

void ssyr2k(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
            float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &b,
            int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc);

void dsyr2k(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
            double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
            cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta,
            cl::sycl::buffer<double, 1> &c, int64_t ldc);

void csyr2k(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
            std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
            cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zsyr2k(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
            std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
            cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

// her2k

void cher2k(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
            std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, float beta,
            cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zher2k(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE trans, int64_t n, int64_t k,
            std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, double beta,
            cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

// trmm

void strmm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
           MKL_DIAG unit_diag, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
           int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb);

void dtrmm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
           MKL_DIAG unit_diag, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
           int64_t lda, cl::sycl::buffer<double, 1> &b, int64_t ldb);

void ctrmm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
           MKL_DIAG unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb);

void ztrmm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
           MKL_DIAG unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb);

// trsm
void strsm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
           MKL_DIAG unit_diag, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
           int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb);

void dtrsm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
           MKL_DIAG unit_diag, int64_t m, int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
           int64_t lda, cl::sycl::buffer<double, 1> &b, int64_t ldb);

void ctrsm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
           MKL_DIAG unit_diag, int64_t m, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb);

void ztrsm(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
           MKL_DIAG unit_diag, int64_t m, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb);

// Level2

void sgemv(cl::sycl::queue &queue, MKL_TRANSPOSE trans, int64_t m, int64_t n, float alpha,
           cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
           float beta, cl::sycl::buffer<float, 1> &y, int64_t incy);

void dgemv(cl::sycl::queue &queue, MKL_TRANSPOSE trans, int64_t m, int64_t n, double alpha,
           cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
           int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, int64_t incy);

void cgemv(cl::sycl::queue &queue, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zgemv(cl::sycl::queue &queue, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void sgbmv(cl::sycl::queue &queue, MKL_TRANSPOSE trans, int64_t m, int64_t n, int64_t kl,
           int64_t ku, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &x, int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
           int64_t incy);

void dgbmv(cl::sycl::queue &queue, MKL_TRANSPOSE trans, int64_t m, int64_t n, int64_t kl,
           int64_t ku, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &x, int64_t incx, double beta,
           cl::sycl::buffer<double, 1> &y, int64_t incy);

void cgbmv(cl::sycl::queue &queue, MKL_TRANSPOSE trans, int64_t m, int64_t n, int64_t kl,
           int64_t ku, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zgbmv(cl::sycl::queue &queue, MKL_TRANSPOSE trans, int64_t m, int64_t n, int64_t kl,
           int64_t ku, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void sger(cl::sycl::queue &queue, int64_t m, int64_t n, float alpha, cl::sycl::buffer<float, 1> &x,
          int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy, cl::sycl::buffer<float, 1> &a,
          int64_t lda);

void dger(cl::sycl::queue &queue, int64_t m, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
          int64_t incy, cl::sycl::buffer<double, 1> &a, int64_t lda);

void cgerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zgerc(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void cgeru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zgeru(cl::sycl::queue &queue, int64_t m, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void chbmv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, int64_t k, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zhbmv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, int64_t k, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void chemv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zhemv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void cher(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zher(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void cher2(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zher2(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void chpmv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx, std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
           int64_t incy);

void zhpmv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &a,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void chpr(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, float alpha,
          cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<float>, 1> &a);

void zhpr(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, double alpha,
          cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          cl::sycl::buffer<std::complex<double>, 1> &a);

void chpr2(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, std::complex<float> alpha,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<float>, 1> &a);

void zhpr2(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, std::complex<double> alpha,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           cl::sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           cl::sycl::buffer<std::complex<double>, 1> &a);

void ssbmv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, int64_t k, float alpha,
           cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
           float beta, cl::sycl::buffer<float, 1> &y, int64_t incy);

void dsbmv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, int64_t k, double alpha,
           cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
           int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, int64_t incy);

void sspmv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, float alpha,
           cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, int64_t incx, float beta,
           cl::sycl::buffer<float, 1> &y, int64_t incy);

void dspmv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, double alpha,
           cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, int64_t incx,
           double beta, cl::sycl::buffer<double, 1> &y, int64_t incy);

void sspr(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &a);

void dspr(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &a);

void sspr2(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, float alpha,
           cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
           cl::sycl::buffer<float, 1> &a);

void dspr2(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, double alpha,
           cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
           int64_t incy, cl::sycl::buffer<double, 1> &a);

void ssymv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, float alpha,
           cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx,
           float beta, cl::sycl::buffer<float, 1> &y, int64_t incy);

void dsymv(cl::sycl::queue &queue, MKL_UPLO uplo, int64_t n, double alpha,
           cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
           int64_t incx, double beta, cl::sycl::buffer<double, 1> &y, int64_t incy);

void ssyr(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, float alpha,
          cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &a, int64_t lda);

void dsyr(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, double alpha,
          cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &a,
          int64_t lda);

void ssyr2(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, float alpha,
           cl::sycl::buffer<float, 1> &x, int64_t incx, cl::sycl::buffer<float, 1> &y, int64_t incy,
           cl::sycl::buffer<float, 1> &a, int64_t lda);

void dsyr2(cl::sycl::queue &queue, MKL_UPLO upplo, int64_t n, double alpha,
           cl::sycl::buffer<double, 1> &x, int64_t incx, cl::sycl::buffer<double, 1> &y,
           int64_t incy, cl::sycl::buffer<double, 1> &a, int64_t lda);

void stbmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
           int64_t incx);

void dtbmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
           int64_t incx);

void ctbmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztbmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void stbsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           int64_t k, cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x,
           int64_t incx);

void dtbsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           int64_t k, cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
           int64_t incx);

void ctbsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztbsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           int64_t k, cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void stpmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, int64_t incx);

void dtpmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, int64_t incx);

void ctpmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx);

void ztpmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<std::complex<double>, 1> &a,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void stpsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x, int64_t incx);

void dtpsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x, int64_t incx);

void ctpsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::complex<float>, 1> &x,
           int64_t incx);

void ztpsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<std::complex<double>, 1> &a,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void strmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx);

void dtrmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
           int64_t incx);

void ctrmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztrmv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void strsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<float, 1> &a, int64_t lda, cl::sycl::buffer<float, 1> &x, int64_t incx);

void dtrsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &x,
           int64_t incx);

void ctrsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztrsv(cl::sycl::queue &queue, MKL_UPLO upplo, MKL_TRANSPOSE trans, MKL_DIAG diag, int64_t n,
           cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           cl::sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

// Level1

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

// batch api

void sgemm_batch(cl::sycl::queue &queue, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m,
                 int64_t n, int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
                 int64_t stride_a, cl::sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b,
                 float beta, cl::sycl::buffer<float, 1> &c, int64_t ldc, int64_t stride_c,
                 int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0,
                 int64_t offset_c = 0);

void dgemm_batch(cl::sycl::queue &queue, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m,
                 int64_t n, int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
                 int64_t stride_a, cl::sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b,
                 double beta, cl::sycl::buffer<double, 1> &c, int64_t ldc, int64_t stride_c,
                 int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0,
                 int64_t offset_c = 0);

void cgemm_batch(cl::sycl::queue &queue, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m,
                 int64_t n, int64_t k, std::complex<float> alpha,
                 cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                 cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, int64_t stride_b,
                 std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc,
                 int64_t stride_c, int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0,
                 int64_t offset_c = 0);

void zgemm_batch(cl::sycl::queue &queue, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m,
                 int64_t n, int64_t k, std::complex<double> alpha,
                 cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                 cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                 std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                 int64_t ldc, int64_t stride_c, int64_t batch_size, int64_t offset_a = 0,
                 int64_t offset_b = 0, int64_t offset_c = 0);

void strsm_batch(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower,
                 MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m, int64_t n, float alpha,
                 cl::sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                 cl::sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                 int64_t offset_a = 0, int64_t offset_b = 0);

void dtrsm_batch(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower,
                 MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m, int64_t n, double alpha,
                 cl::sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                 cl::sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                 int64_t offset_a = 0, int64_t offset_b = 0);

void ctrsm_batch(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower,
                 MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m, int64_t n,
                 std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                 int64_t lda, int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                 int64_t ldb, int64_t stride_b, int64_t batch_size, int64_t offset_a = 0,
                 int64_t offset_b = 0);

void ztrsm_batch(cl::sycl::queue &queue, MKL_SIDE left_right, MKL_UPLO upper_lower,
                 MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m, int64_t n,
                 std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                 int64_t lda, int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                 int64_t ldb, int64_t stride_b, int64_t batch_size, int64_t offset_a = 0,
                 int64_t offset_b = 0);

// BLAS like extension

void sgemmt(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
            int64_t lda, cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta,
            cl::sycl::buffer<float, 1> &c, int64_t ldc);

void dgemmt(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, double alpha,
            cl::sycl::buffer<double, 1> &a, int64_t lda, cl::sycl::buffer<double, 1> &b,
            int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c, int64_t ldc);

void zgemmt(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, std::complex<double> alpha,
            cl::sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
            cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void cgemmt(cl::sycl::queue &queue, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, std::complex<float> alpha,
            cl::sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
            cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
            cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void hgemm(cl::sycl::queue &queue, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m, int64_t n,
           int64_t k, half alpha, cl::sycl::buffer<half, 1> &a, int64_t lda,
           cl::sycl::buffer<half, 1> &b, int64_t ldb, half beta, cl::sycl::buffer<half, 1> &c,
           int64_t ldc);

void gemm_f16f16f32(cl::sycl::queue &queue, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t m,
                    int64_t n, int64_t k, float alpha, cl::sycl::buffer<half, 1> &a, int64_t lda,
                    cl::sycl::buffer<half, 1> &b, int64_t ldb, float beta,
                    cl::sycl::buffer<float, 1> &c, int64_t ldc);

void gemm_s8u8s32(cl::sycl::queue &queue, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
                  CBLAS_OFFSET offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                  cl::sycl::buffer<int8_t, 1> &a, int64_t lda, int8_t ao,
                  cl::sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo, float beta,
                  cl::sycl::buffer<int32_t, 1> &c, int64_t ldc, cl::sycl::buffer<int32_t, 1> &co);
} // namespace gpu
} // namespace mkl
#endif //_MKL_INTERNAL_BLAS_SYCL_GPU_HPP_
