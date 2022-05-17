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

#ifndef _MKLGPU_COMMON_HPP_
#define _MKLGPU_COMMON_HPP_

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <complex>

typedef enum { MKL_ROW_MAJOR = 101, MKL_COL_MAJOR = 102 } MKL_LAYOUT;

typedef enum { MKL_NOTRANS = 111, MKL_TRANS = 112, MKL_CONJTRANS = 113 } MKL_TRANSPOSE;

typedef enum { MKL_UPPER = 121, MKL_LOWER = 122 } MKL_UPLO;

typedef enum { MKL_NONUNIT = 131, MKL_UNIT = 132 } MKL_DIAG;

typedef enum { MKL_LEFT = 141, MKL_RIGHT = 142 } MKL_SIDE;

enum CBLAS_OFFSET { CblasRowOffset = 171, CblasColOffset = 172, CblasFixOffset = 173 };
typedef enum CBLAS_OFFSET CBLAS_OFFSET;

namespace oneapi {
namespace mkl {
namespace blas {
namespace mklgpu {

inline MKL_TRANSPOSE mkl_convert(oneapi::mkl::transpose t) {
    if (t == oneapi::mkl::transpose::nontrans)
        return MKL_NOTRANS;
    if (t == oneapi::mkl::transpose::trans)
        return MKL_TRANS;
    if (t == oneapi::mkl::transpose::conjtrans)
        return MKL_CONJTRANS;
    return MKL_NOTRANS;
}

inline MKL_UPLO mkl_convert(oneapi::mkl::uplo u) {
    if (u == oneapi::mkl::uplo::upper)
        return MKL_UPPER;
    if (u == oneapi::mkl::uplo::lower)
        return MKL_LOWER;
    return MKL_UPPER;
}

inline MKL_DIAG mkl_convert(oneapi::mkl::diag d) {
    if (d == oneapi::mkl::diag::nonunit)
        return MKL_NONUNIT;
    if (d == oneapi::mkl::diag::unit)
        return MKL_UNIT;
    return MKL_NONUNIT;
}

inline MKL_SIDE mkl_convert(oneapi::mkl::side s) {
    if (s == oneapi::mkl::side::left)
        return MKL_LEFT;
    if (s == oneapi::mkl::side::right)
        return MKL_RIGHT;
    return MKL_LEFT;
}

inline CBLAS_OFFSET mkl_convert(oneapi::mkl::offset o) {
    if (o == oneapi::mkl::offset::fix)
        return CblasFixOffset;
    if (o == oneapi::mkl::offset::column)
        return CblasColOffset;
    return CblasRowOffset;
}

} // namespace mklgpu
} // namespace blas
} // namespace mkl
} // namespace oneapi

namespace oneapi {
namespace mkl {
namespace gpu {

// Buffer APIs

void sgemm(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
           int64_t m, int64_t n, int64_t k, float alpha, sycl::buffer<float, 1> &a, int64_t lda,
           sycl::buffer<float, 1> &b, int64_t ldb, float beta, sycl::buffer<float, 1> &c,
           int64_t ldc);

void dgemm(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
           int64_t m, int64_t n, int64_t k, double alpha, sycl::buffer<double, 1> &a, int64_t lda,
           sycl::buffer<double, 1> &b, int64_t ldb, double beta, sycl::buffer<double, 1> &c,
           int64_t ldc);

void cgemm(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
           int64_t m, int64_t n, int64_t k, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zgemm(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
           int64_t m, int64_t n, int64_t k, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void ssymm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, float alpha, sycl::buffer<float, 1> &a, int64_t lda,
           sycl::buffer<float, 1> &b, int64_t ldb, float beta, sycl::buffer<float, 1> &c,
           int64_t ldc);

void dsymm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, double alpha, sycl::buffer<double, 1> &a, int64_t lda,
           sycl::buffer<double, 1> &b, int64_t ldb, double beta, sycl::buffer<double, 1> &c,
           int64_t ldc);

void csymm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
           std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zsymm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void chemm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
           std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zhemm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           int64_t m, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void ssyrk(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, float alpha, sycl::buffer<float, 1> &a, int64_t lda, float beta,
           sycl::buffer<float, 1> &c, int64_t ldc);

void dsyrk(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, double alpha, sycl::buffer<double, 1> &a, int64_t lda, double beta,
           sycl::buffer<double, 1> &c, int64_t ldc);

void csyrk(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c,
           int64_t ldc);

void zsyrk(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &a, int64_t lda, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

sycl::event ssyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, float alpha,
                             sycl::buffer<float, 1> *a, int64_t lda, int64_t stride_a, float beta,
                             sycl::buffer<float, 1> *c, int64_t ldc, int64_t stride_c,
                             int64_t batchsize, int64_t offset_a = 0, int64_t offset_c = 0);

sycl::event dsyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, double alpha,
                             sycl::buffer<double, 1> *a, int64_t lda, int64_t stride_a, double beta,
                             sycl::buffer<double, 1> *c, int64_t ldc, int64_t stride_c,
                             int64_t batchsize, int64_t offset_a = 0, int64_t offset_c = 0);

sycl::event csyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<float> alpha,
                             sycl::buffer<std::complex<float>, 1> *a, int64_t lda, int64_t stride_a,
                             std::complex<float> beta, sycl::buffer<std::complex<float>, 1> *c,
                             int64_t ldc, int64_t stride_c, int64_t batchsize, int64_t offset_a = 0,
                             int64_t offset_c = 0);

sycl::event zsyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<double> alpha,
                             sycl::buffer<std::complex<double>, 1> *a, int64_t lda,
                             int64_t stride_a, std::complex<double> beta,
                             sycl::buffer<std::complex<double>, 1> *c, int64_t ldc,
                             int64_t stride_c, int64_t batchsize, int64_t offset_a = 0,
                             int64_t offset_c = 0);

void cherk(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, float alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           float beta, sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zherk(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
           int64_t n, int64_t k, double alpha, sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, double beta, sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void ssyr2k(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, float alpha, sycl::buffer<float, 1> &a, int64_t lda,
            sycl::buffer<float, 1> &b, int64_t ldb, float beta, sycl::buffer<float, 1> &c,
            int64_t ldc);

void dsyr2k(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, double alpha, sycl::buffer<double, 1> &a, int64_t lda,
            sycl::buffer<double, 1> &b, int64_t ldb, double beta, sycl::buffer<double, 1> &c,
            int64_t ldc);

void csyr2k(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, std::complex<float> alpha,
            sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
            sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
            sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zsyr2k(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, std::complex<double> alpha,
            sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
            sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
            sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void cher2k(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, std::complex<float> alpha,
            sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
            sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, float beta,
            sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void zher2k(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE trans,
            int64_t n, int64_t k, std::complex<double> alpha,
            sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
            sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, double beta,
            sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void strmm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n, float alpha,
           sycl::buffer<float, 1> &a, int64_t lda, sycl::buffer<float, 1> &b, int64_t ldb);

void dtrmm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n, double alpha,
           sycl::buffer<double, 1> &a, int64_t lda, sycl::buffer<double, 1> &b, int64_t ldb);

void ctrmm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<float>, 1> &b, int64_t ldb);

void ztrmm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, int64_t ldb);

void strsm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n, float alpha,
           sycl::buffer<float, 1> &a, int64_t lda, sycl::buffer<float, 1> &b, int64_t ldb);

void dtrsm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n, double alpha,
           sycl::buffer<double, 1> &a, int64_t lda, sycl::buffer<double, 1> &b, int64_t ldb);

void ctrsm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<float>, 1> &b, int64_t ldb);

void ztrsm(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
           MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m, int64_t n,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, int64_t ldb);

void sgemv(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           float alpha, sycl::buffer<float, 1> &a, int64_t lda, sycl::buffer<float, 1> &x,
           int64_t incx, float beta, sycl::buffer<float, 1> &y, int64_t incy);

void dgemv(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           double alpha, sycl::buffer<double, 1> &a, int64_t lda, sycl::buffer<double, 1> &x,
           int64_t incx, double beta, sycl::buffer<double, 1> &y, int64_t incy);

void cgemv(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zgemv(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

sycl::event sgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, float alpha, sycl::buffer<float, 1> *a, int64_t lda,
                             int64_t stride_a, sycl::buffer<float, 1> *x, int64_t incx,
                             int64_t stride_x, float beta, sycl::buffer<float, 1> *y, int64_t incy,
                             int64_t stride_y, int64_t groupsize, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event dgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, double alpha, sycl::buffer<double, 1> *a, int64_t lda,
                             int64_t stride_a, sycl::buffer<double, 1> *x, int64_t incx,
                             int64_t stride_x, double beta, sycl::buffer<double, 1> *y,
                             int64_t incy, int64_t stride_y, int64_t groupsize,
                             int64_t offset_a = 0, int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event cgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, std::complex<float> alpha,
                             sycl::buffer<std::complex<float>, 1> *a, int64_t lda, int64_t stride_a,
                             sycl::buffer<std::complex<float>, 1> *x, int64_t incx,
                             int64_t stride_x, std::complex<float> beta,
                             sycl::buffer<std::complex<float>, 1> *y, int64_t incy,
                             int64_t stride_y, int64_t groupsize, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event zgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, std::complex<double> alpha,
                             sycl::buffer<std::complex<double>, 1> *a, int64_t lda,
                             int64_t stride_a, sycl::buffer<std::complex<double>, 1> *x,
                             int64_t incx, int64_t stride_x, std::complex<double> beta,
                             sycl::buffer<std::complex<double>, 1> *y, int64_t incy,
                             int64_t stride_y, int64_t groupsize, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event sdgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, sycl::buffer<float, 1> *a, int64_t lda, int64_t stride_a,
                             sycl::buffer<float, 1> *x, int64_t incx, int64_t stride_x,
                             sycl::buffer<float, 1> *c, int64_t ldc, int64_t stride_c,
                             int64_t groupsize, int64_t offset_a = 0, int64_t offset_x = 0,
                             int64_t offset_c = 0);

sycl::event ddgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, sycl::buffer<double, 1> *a, int64_t lda, int64_t stride_a,
                             sycl::buffer<double, 1> *x, int64_t incx, int64_t stride_x,
                             sycl::buffer<double, 1> *c, int64_t ldc, int64_t stride_c,
                             int64_t groupsize, int64_t offset_a = 0, int64_t offset_x = 0,
                             int64_t offset_c = 0);

sycl::event cdgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, sycl::buffer<std::complex<float>, 1> *a, int64_t lda,
                             int64_t stride_a, sycl::buffer<std::complex<float>, 1> *x,
                             int64_t incx, int64_t stride_x,
                             sycl::buffer<std::complex<float>, 1> *c, int64_t ldc, int64_t stride_c,
                             int64_t groupsize, int64_t offset_a = 0, int64_t offset_x = 0,
                             int64_t offset_c = 0);

sycl::event zdgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, sycl::buffer<std::complex<double>, 1> *a, int64_t lda,
                             int64_t stride_a, sycl::buffer<std::complex<double>, 1> *x,
                             int64_t incx, int64_t stride_x,
                             sycl::buffer<std::complex<double>, 1> *c, int64_t ldc,
                             int64_t stride_c, int64_t groupsize, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_c = 0);

void sgbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           int64_t kl, int64_t ku, float alpha, sycl::buffer<float, 1> &a, int64_t lda,
           sycl::buffer<float, 1> &x, int64_t incx, float beta, sycl::buffer<float, 1> &y,
           int64_t incy);

void dgbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           int64_t kl, int64_t ku, double alpha, sycl::buffer<double, 1> &a, int64_t lda,
           sycl::buffer<double, 1> &x, int64_t incx, double beta, sycl::buffer<double, 1> &y,
           int64_t incy);

void cgbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           int64_t kl, int64_t ku, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zgbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m, int64_t n,
           int64_t kl, int64_t ku, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void sger(sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n, float alpha,
          sycl::buffer<float, 1> &x, int64_t incx, sycl::buffer<float, 1> &y, int64_t incy,
          sycl::buffer<float, 1> &a, int64_t lda);

void dger(sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n, double alpha,
          sycl::buffer<double, 1> &x, int64_t incx, sycl::buffer<double, 1> &y, int64_t incy,
          sycl::buffer<double, 1> &a, int64_t lda);

void cgerc(sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zgerc(sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void cgeru(sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zgeru(sycl::queue &queue, MKL_LAYOUT layout, int64_t m, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void chbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zhbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void chemv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zhemv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void cher(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
          sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zher(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
          sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void cher2(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<float>, 1> &a, int64_t lda);

void zher2(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<double>, 1> &a, int64_t lda);

void chpmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zhpmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void chpr(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
          sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
          sycl::buffer<std::complex<float>, 1> &a);

void zhpr(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
          sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
          sycl::buffer<std::complex<double>, 1> &a);

void chpr2(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<float>, 1> &a);

void zhpr2(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<double>, 1> &a);

void ssbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k, float alpha,
           sycl::buffer<float, 1> &a, int64_t lda, sycl::buffer<float, 1> &x, int64_t incx,
           float beta, sycl::buffer<float, 1> &y, int64_t incy);

void dsbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k, double alpha,
           sycl::buffer<double, 1> &a, int64_t lda, sycl::buffer<double, 1> &x, int64_t incx,
           double beta, sycl::buffer<double, 1> &y, int64_t incy);

void sspmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, float alpha,
           sycl::buffer<float, 1> &a, sycl::buffer<float, 1> &x, int64_t incx, float beta,
           sycl::buffer<float, 1> &y, int64_t incy);

void dspmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, double alpha,
           sycl::buffer<double, 1> &a, sycl::buffer<double, 1> &x, int64_t incx, double beta,
           sycl::buffer<double, 1> &y, int64_t incy);

void sspr(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
          sycl::buffer<float, 1> &x, int64_t incx, sycl::buffer<float, 1> &a);

void dspr(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
          sycl::buffer<double, 1> &x, int64_t incx, sycl::buffer<double, 1> &a);

void sspr2(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
           sycl::buffer<float, 1> &x, int64_t incx, sycl::buffer<float, 1> &y, int64_t incy,
           sycl::buffer<float, 1> &a);

void dspr2(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
           sycl::buffer<double, 1> &x, int64_t incx, sycl::buffer<double, 1> &y, int64_t incy,
           sycl::buffer<double, 1> &a);

void ssymv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, float alpha,
           sycl::buffer<float, 1> &a, int64_t lda, sycl::buffer<float, 1> &x, int64_t incx,
           float beta, sycl::buffer<float, 1> &y, int64_t incy);

void dsymv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, double alpha,
           sycl::buffer<double, 1> &a, int64_t lda, sycl::buffer<double, 1> &x, int64_t incx,
           double beta, sycl::buffer<double, 1> &y, int64_t incy);

void ssyr(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
          sycl::buffer<float, 1> &x, int64_t incx, sycl::buffer<float, 1> &a, int64_t lda);

void dsyr(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
          sycl::buffer<double, 1> &x, int64_t incx, sycl::buffer<double, 1> &a, int64_t lda);

void ssyr2(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
           sycl::buffer<float, 1> &x, int64_t incx, sycl::buffer<float, 1> &y, int64_t incy,
           sycl::buffer<float, 1> &a, int64_t lda);

void dsyr2(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, double alpha,
           sycl::buffer<double, 1> &x, int64_t incx, sycl::buffer<double, 1> &y, int64_t incy,
           sycl::buffer<double, 1> &a, int64_t lda);

void stbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, sycl::buffer<float, 1> &a, int64_t lda,
           sycl::buffer<float, 1> &x, int64_t incx);

void dtbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, sycl::buffer<double, 1> &a, int64_t lda,
           sycl::buffer<double, 1> &x, int64_t incx);

void ctbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztbmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void stbsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, sycl::buffer<float, 1> &a, int64_t lda,
           sycl::buffer<float, 1> &x, int64_t incx);

void dtbsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, sycl::buffer<double, 1> &a, int64_t lda,
           sycl::buffer<double, 1> &x, int64_t incx);

void ctbsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztbsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, int64_t k, sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void stpmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<float, 1> &a, sycl::buffer<float, 1> &x,
           int64_t incx);

void dtpmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<double, 1> &a, sycl::buffer<double, 1> &x,
           int64_t incx);

void ctpmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<std::complex<float>, 1> &a,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztpmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<std::complex<double>, 1> &a,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void stpsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<float, 1> &a, sycl::buffer<float, 1> &x,
           int64_t incx);

void dtpsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<double, 1> &a, sycl::buffer<double, 1> &x,
           int64_t incx);

void ctpsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<std::complex<float>, 1> &a,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztpsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<std::complex<double>, 1> &a,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void strmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<float, 1> &a, int64_t lda,
           sycl::buffer<float, 1> &x, int64_t incx);

void dtrmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<double, 1> &a, int64_t lda,
           sycl::buffer<double, 1> &x, int64_t incx);

void ctrmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztrmv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void strsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<float, 1> &a, int64_t lda,
           sycl::buffer<float, 1> &x, int64_t incx);

void dtrsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<double, 1> &a, int64_t lda,
           sycl::buffer<double, 1> &x, int64_t incx);

void ctrsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void ztrsv(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
           MKL_DIAG diag, int64_t n, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void scasum(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
            sycl::buffer<float, 1> &result);

void dzasum(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
            sycl::buffer<double, 1> &result);

void sasum(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
           sycl::buffer<float, 1> &result);

void dasum(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
           sycl::buffer<double, 1> &result);

void saxpy(sycl::queue &queue, int64_t n, float alpha, sycl::buffer<float, 1> &x, int64_t incx,
           sycl::buffer<float, 1> &y, int64_t incy);

void daxpy(sycl::queue &queue, int64_t n, double alpha, sycl::buffer<double, 1> &x, int64_t incx,
           sycl::buffer<double, 1> &y, int64_t incy);

void caxpy(sycl::queue &queue, int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zaxpy(sycl::queue &queue, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void saxpy_batch(sycl::queue &queue, int64_t n, float alpha, sycl::buffer<float, 1> &x,
                 int64_t incx, int64_t stridex, sycl::buffer<float, 1> &y, int64_t incy,
                 int64_t stridey, int64_t batch_size);

void daxpy_batch(sycl::queue &queue, int64_t n, double alpha, sycl::buffer<double, 1> &x,
                 int64_t incx, int64_t stridex, sycl::buffer<double, 1> &y, int64_t incy,
                 int64_t stridey, int64_t batch_size);

void caxpy_batch(sycl::queue &queue, int64_t n, std::complex<float> alpha,
                 sycl::buffer<std::complex<float>, 1> &x, int64_t incx, int64_t stridex,
                 sycl::buffer<std::complex<float>, 1> &y, int64_t incy, int64_t stridey,
                 int64_t batch_size);

void zaxpy_batch(sycl::queue &queue, int64_t n, std::complex<double> alpha,
                 sycl::buffer<std::complex<double>, 1> &x, int64_t incx, int64_t stridex,
                 sycl::buffer<std::complex<double>, 1> &y, int64_t incy, int64_t stridey,
                 int64_t batch_size);

sycl::event saxpby_sycl(sycl::queue *queue, int64_t n, float alpha, sycl::buffer<float, 1> *x,
                        int64_t incx, float beta, sycl::buffer<float, 1> *y, int64_t incy);

sycl::event daxpby_sycl(sycl::queue *queue, int64_t n, double alpha, sycl::buffer<double, 1> *x,
                        int64_t incx, double beta, sycl::buffer<double, 1> *y, int64_t incy);

sycl::event caxpby_sycl(sycl::queue *queue, int64_t n, std::complex<float> alpha,
                        sycl::buffer<std::complex<float>, 1> *x, int64_t incx,
                        std::complex<float> beta, sycl::buffer<std::complex<float>, 1> *y,
                        int64_t incy);

sycl::event zaxpby_sycl(sycl::queue *queue, int64_t n, std::complex<double> alpha,
                        sycl::buffer<std::complex<double>, 1> *x, int64_t incx,
                        std::complex<double> beta, sycl::buffer<std::complex<double>, 1> *y,
                        int64_t incy);

void scopy(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
           sycl::buffer<float, 1> &y, int64_t incy);

void dcopy(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
           sycl::buffer<double, 1> &y, int64_t incy);

void ccopy(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zcopy(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

sycl::event scopy_batch_sycl(sycl::queue *queue, int64_t n, sycl::buffer<float, 1> *x, int64_t incx,
                             std::int64_t stridex, sycl::buffer<float, 1> *y, int64_t incy,
                             std::int64_t stridey, std::int64_t batch_size);

sycl::event ccopy_batch_sycl(sycl::queue *queue, int64_t n, sycl::buffer<std::complex<float>, 1> *x,
                             int64_t incx, std::int64_t stridex,
                             sycl::buffer<std::complex<float>, 1> *y, int64_t incy,
                             std::int64_t stridey, std::int64_t batch_size);

sycl::event zcopy_batch_sycl(sycl::queue *queue, int64_t n,
                             sycl::buffer<std::complex<double>, 1> *x, int64_t incx,
                             std::int64_t stridex, sycl::buffer<std::complex<double>, 1> *y,
                             int64_t incy, std::int64_t stridey, std::int64_t batch_size);

sycl::event dcopy_batch_sycl(sycl::queue *queue, int64_t n, sycl::buffer<double, 1> *x,
                             int64_t incx, std::int64_t stridex, sycl::buffer<double, 1> *y,
                             int64_t incy, std::int64_t stridey, std::int64_t batch_size);

void sdot(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
          sycl::buffer<float, 1> &y, int64_t incy, sycl::buffer<float, 1> &result);

void ddot(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
          sycl::buffer<double, 1> &y, int64_t incy, sycl::buffer<double, 1> &result);

void sdsdot(sycl::queue &queue, int64_t n, float sb, sycl::buffer<float, 1> &x, int64_t incx,
            sycl::buffer<float, 1> &y, int64_t incy, sycl::buffer<float, 1> &result);

void dsdot(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
           sycl::buffer<float, 1> &y, int64_t incy, sycl::buffer<double, 1> &result);

void cdotc(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<float>, 1> &result);

void zdotc(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<double>, 1> &result);

void cdotu(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<float>, 1> &result);

void zdotu(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy,
           sycl::buffer<std::complex<double>, 1> &result);

void scnrm2(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
            sycl::buffer<float, 1> &result);

void dznrm2(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
            sycl::buffer<double, 1> &result);

void snrm2(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
           sycl::buffer<float, 1> &result);

void dnrm2(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
           sycl::buffer<double, 1> &result);

void csrot(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy, float c, float s);

void zdrot(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy, double c, double s);

void srot(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
          sycl::buffer<float, 1> &y, int64_t incy, float c, float s);

void drot(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
          sycl::buffer<double, 1> &y, int64_t incy, double c, double s);

void srotg(sycl::queue &queue, sycl::buffer<float, 1> &a, sycl::buffer<float, 1> &b,
           sycl::buffer<float, 1> &c, sycl::buffer<float, 1> &s);

void drotg(sycl::queue &queue, sycl::buffer<double, 1> &a, sycl::buffer<double, 1> &b,
           sycl::buffer<double, 1> &c, sycl::buffer<double, 1> &s);

void crotg(sycl::queue &queue, sycl::buffer<std::complex<float>, 1> &a,
           sycl::buffer<std::complex<float>, 1> &b, sycl::buffer<float, 1> &c,
           sycl::buffer<std::complex<float>, 1> &s);

void zrotg(sycl::queue &queue, sycl::buffer<std::complex<double>, 1> &a,
           sycl::buffer<std::complex<double>, 1> &b, sycl::buffer<double, 1> &c,
           sycl::buffer<std::complex<double>, 1> &s);

void srotm(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
           sycl::buffer<float, 1> &y, int64_t incy, sycl::buffer<float, 1> &param);

void drotm(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
           sycl::buffer<double, 1> &y, int64_t incy, sycl::buffer<double, 1> &param);

void srotmg(sycl::queue &queue, sycl::buffer<float, 1> &d1, sycl::buffer<float, 1> &d2,
            sycl::buffer<float, 1> &x1, float y1, sycl::buffer<float, 1> &param);

void drotmg(sycl::queue &queue, sycl::buffer<double, 1> &d1, sycl::buffer<double, 1> &d2,
            sycl::buffer<double, 1> &x1, double y1, sycl::buffer<double, 1> &param);

void sscal(sycl::queue &queue, int64_t n, float alpha, sycl::buffer<float, 1> &x, int64_t incx);

void dscal(sycl::queue &queue, int64_t n, double alpha, sycl::buffer<double, 1> &x, int64_t incx);

void cscal(sycl::queue &queue, int64_t n, std::complex<float> alpha,
           sycl::buffer<std::complex<float>, 1> &x, int64_t incx);

void zscal(sycl::queue &queue, int64_t n, std::complex<double> alpha,
           sycl::buffer<std::complex<double>, 1> &x, int64_t incx);

void csscal(sycl::queue &queue, int64_t n, float alpha, sycl::buffer<std::complex<float>, 1> &x,
            int64_t incx);

void zdscal(sycl::queue &queue, int64_t n, double alpha, sycl::buffer<std::complex<double>, 1> &x,
            int64_t incx);

void sswap(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
           sycl::buffer<float, 1> &y, int64_t incy);

void dswap(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
           sycl::buffer<double, 1> &y, int64_t incy);

void cswap(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<float>, 1> &y, int64_t incy);

void zswap(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
           sycl::buffer<std::complex<double>, 1> &y, int64_t incy);

void isamax(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
            sycl::buffer<int64_t, 1> &result);

void idamax(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
            sycl::buffer<int64_t, 1> &result);

void icamax(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
            sycl::buffer<int64_t, 1> &result);

void izamax(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
            sycl::buffer<int64_t, 1> &result);

void isamin(sycl::queue &queue, int64_t n, sycl::buffer<float, 1> &x, int64_t incx,
            sycl::buffer<int64_t, 1> &result);

void idamin(sycl::queue &queue, int64_t n, sycl::buffer<double, 1> &x, int64_t incx,
            sycl::buffer<int64_t, 1> &result);

void icamin(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<float>, 1> &x, int64_t incx,
            sycl::buffer<int64_t, 1> &result);

void izamin(sycl::queue &queue, int64_t n, sycl::buffer<std::complex<double>, 1> &x, int64_t incx,
            sycl::buffer<int64_t, 1> &result);

void sgemm_batch(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
                 int64_t m, int64_t n, int64_t k, float alpha, sycl::buffer<float, 1> &a,
                 int64_t lda, int64_t stride_a, sycl::buffer<float, 1> &b, int64_t ldb,
                 int64_t stride_b, float beta, sycl::buffer<float, 1> &c, int64_t ldc,
                 int64_t stride_c, int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0,
                 int64_t offset_c = 0);

void dgemm_batch(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
                 int64_t m, int64_t n, int64_t k, double alpha, sycl::buffer<double, 1> &a,
                 int64_t lda, int64_t stride_a, sycl::buffer<double, 1> &b, int64_t ldb,
                 int64_t stride_b, double beta, sycl::buffer<double, 1> &c, int64_t ldc,
                 int64_t stride_c, int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0,
                 int64_t offset_c = 0);

void cgemm_batch(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
                 int64_t m, int64_t n, int64_t k, std::complex<float> alpha,
                 sycl::buffer<std::complex<float>, 1> &a, int64_t lda, int64_t stride_a,
                 sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, int64_t stride_b,
                 std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c, int64_t ldc,
                 int64_t stride_c, int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0,
                 int64_t offset_c = 0);

void zgemm_batch(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
                 int64_t m, int64_t n, int64_t k, std::complex<double> alpha,
                 sycl::buffer<std::complex<double>, 1> &a, int64_t lda, int64_t stride_a,
                 sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, int64_t stride_b,
                 std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c, int64_t ldc,
                 int64_t stride_c, int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0,
                 int64_t offset_c = 0);

void hgemm_batch(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
                 int64_t m, int64_t n, int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1> &a,
                 int64_t lda, int64_t stride_a, sycl::buffer<sycl::half, 1> &b, int64_t ldb,
                 int64_t stride_b, sycl::half beta, sycl::buffer<sycl::half, 1> &c, int64_t ldc,
                 int64_t stride_c, int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0,
                 int64_t offset_c = 0);

void strsm_batch(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
                 MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m, int64_t n, float alpha,
                 sycl::buffer<float, 1> &a, int64_t lda, int64_t stride_a,
                 sycl::buffer<float, 1> &b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                 int64_t offset_a = 0, int64_t offset_b = 0);

void dtrsm_batch(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
                 MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m, int64_t n, double alpha,
                 sycl::buffer<double, 1> &a, int64_t lda, int64_t stride_a,
                 sycl::buffer<double, 1> &b, int64_t ldb, int64_t stride_b, int64_t batch_size,
                 int64_t offset_a = 0, int64_t offset_b = 0);

void ctrsm_batch(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
                 MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m, int64_t n,
                 std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
                 int64_t stride_a, sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
                 int64_t stride_b, int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0);

void ztrsm_batch(sycl::queue &queue, MKL_LAYOUT layout, MKL_SIDE left_right, MKL_UPLO upper_lower,
                 MKL_TRANSPOSE trans, MKL_DIAG unit_diag, int64_t m, int64_t n,
                 std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
                 int64_t stride_a, sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
                 int64_t stride_b, int64_t batch_size, int64_t offset_a = 0, int64_t offset_b = 0);

void sgemmt(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, float alpha, sycl::buffer<float, 1> &a,
            int64_t lda, sycl::buffer<float, 1> &b, int64_t ldb, float beta,
            sycl::buffer<float, 1> &c, int64_t ldc);

void dgemmt(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, double alpha, sycl::buffer<double, 1> &a,
            int64_t lda, sycl::buffer<double, 1> &b, int64_t ldb, double beta,
            sycl::buffer<double, 1> &c, int64_t ldc);

void zgemmt(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, std::complex<double> alpha,
            sycl::buffer<std::complex<double>, 1> &a, int64_t lda,
            sycl::buffer<std::complex<double>, 1> &b, int64_t ldb, std::complex<double> beta,
            sycl::buffer<std::complex<double>, 1> &c, int64_t ldc);

void cgemmt(sycl::queue &queue, MKL_LAYOUT layout, MKL_UPLO upper_lower, MKL_TRANSPOSE transa,
            MKL_TRANSPOSE transb, int64_t n, int64_t k, std::complex<float> alpha,
            sycl::buffer<std::complex<float>, 1> &a, int64_t lda,
            sycl::buffer<std::complex<float>, 1> &b, int64_t ldb, std::complex<float> beta,
            sycl::buffer<std::complex<float>, 1> &c, int64_t ldc);

void hgemm(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, MKL_TRANSPOSE transb,
           int64_t m, int64_t n, int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1> &a,
           int64_t lda, sycl::buffer<sycl::half, 1> &b, int64_t ldb, sycl::half beta,
           sycl::buffer<sycl::half, 1> &c, int64_t ldc);

void gemm_f16f16f32(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                    MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha,
                    sycl::buffer<sycl::half, 1> &a, int64_t lda, sycl::buffer<sycl::half, 1> &b,
                    int64_t ldb, float beta, sycl::buffer<float, 1> &c, int64_t ldc);

void gemm_bf16bf16f32(sycl::queue &queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                      MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha,
                      sycl::buffer<bfloat16, 1> &a, int64_t lda, sycl::buffer<bfloat16, 1> &b,
                      int64_t ldb, float beta, sycl::buffer<float, 1> &c, int64_t ldc);

sycl::event gemm_s8s8s32_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                              MKL_TRANSPOSE transb, CBLAS_OFFSET offsetc, int64_t m, int64_t n,
                              int64_t k, float alpha, sycl::buffer<int8_t, 1> *a, int64_t lda,
                              int8_t ao, sycl::buffer<int8_t, 1> *b, int64_t ldb, int8_t bo,
                              float beta, sycl::buffer<int32_t, 1> *c, int64_t ldc,
                              sycl::buffer<int32_t, 1> *co, int64_t offset_a = 0,
                              int64_t offset_b = 0, int64_t offset_c = 0, int64_t offset_co = 0);

sycl::event gemm_s8u8s32_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                              MKL_TRANSPOSE transb, CBLAS_OFFSET offsetc, int64_t m, int64_t n,
                              int64_t k, float alpha, sycl::buffer<int8_t, 1> *a, int64_t lda,
                              int8_t ao, sycl::buffer<uint8_t, 1> *b, int64_t ldb, uint8_t bo,
                              float beta, sycl::buffer<int32_t, 1> *c, int64_t ldc,
                              sycl::buffer<int32_t, 1> *co, int64_t offset_a = 0,
                              int64_t offset_b = 0, int64_t offset_c = 0, int64_t offset_co = 0);

sycl::event gemm_u8s8s32_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                              MKL_TRANSPOSE transb, CBLAS_OFFSET offsetc, int64_t m, int64_t n,
                              int64_t k, float alpha, sycl::buffer<uint8_t, 1> *a, int64_t lda,
                              uint8_t ao, sycl::buffer<int8_t, 1> *b, int64_t ldb, int8_t bo,
                              float beta, sycl::buffer<int32_t, 1> *c, int64_t ldc,
                              sycl::buffer<int32_t, 1> *co, int64_t offset_a = 0,
                              int64_t offset_b = 0, int64_t offset_c = 0, int64_t offset_co = 0);

sycl::event gemm_u8u8s32_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                              MKL_TRANSPOSE transb, CBLAS_OFFSET offsetc, int64_t m, int64_t n,
                              int64_t k, float alpha, sycl::buffer<uint8_t, 1> *a, int64_t lda,
                              uint8_t ao, sycl::buffer<uint8_t, 1> *b, int64_t ldb, uint8_t bo,
                              float beta, sycl::buffer<int32_t, 1> *c, int64_t ldc,
                              sycl::buffer<int32_t, 1> *co, int64_t offset_a = 0,
                              int64_t offset_b = 0, int64_t offset_c = 0, int64_t offset_co = 0);

// USM APIs

sycl::event sgemm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                       MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha,
                       const float *a, int64_t lda, const float *b, int64_t ldb, float beta,
                       float *c, int64_t ldc, const std::vector<sycl::event> &dependencies,
                       int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event dgemm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                       MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, double alpha,
                       const double *a, int64_t lda, const double *b, int64_t ldb, double beta,
                       double *c, int64_t ldc, const std::vector<sycl::event> &dependencies,
                       int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event cgemm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                       MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                       std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                       const std::complex<float> *b, int64_t ldb, std::complex<float> beta,
                       std::complex<float> *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event zgemm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                       MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                       std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                       const std::complex<double> *b, int64_t ldb, std::complex<double> beta,
                       std::complex<double> *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event hgemm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                       MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, sycl::half alpha,
                       const sycl::half *a, int64_t lda, const sycl::half *b, int64_t ldb,
                       sycl::half beta, sycl::half *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event gemm_f16f16f32_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                                MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha,
                                const sycl::half *a, int64_t lda, const sycl::half *b, int64_t ldb,
                                float beta, float *c, int64_t ldc,
                                const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                                int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event gemm_bf16bf16f32_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                                  MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                                  float alpha, const bfloat16 *a, int64_t lda, const bfloat16 *b,
                                  int64_t ldb, float beta, float *c, int64_t ldc,
                                  const std::vector<sycl::event> &dependencies,
                                  int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event ssymm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, int64_t m, int64_t n, float alpha, const float *a,
                       int64_t lda, const float *b, int64_t ldb, float beta, float *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event dsymm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, int64_t m, int64_t n, double alpha, const double *a,
                       int64_t lda, const double *b, int64_t ldb, double beta, double *c,
                       int64_t ldc, const std::vector<sycl::event> &dependencies,
                       int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event csymm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, int64_t m, int64_t n, std::complex<float> alpha,
                       const std::complex<float> *a, int64_t lda, const std::complex<float> *b,
                       int64_t ldb, std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event zsymm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, int64_t m, int64_t n, std::complex<double> alpha,
                       const std::complex<double> *a, int64_t lda, const std::complex<double> *b,
                       int64_t ldb, std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event chemm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, int64_t m, int64_t n, std::complex<float> alpha,
                       const std::complex<float> *a, int64_t lda, const std::complex<float> *b,
                       int64_t ldb, std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event zhemm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, int64_t m, int64_t n, std::complex<double> alpha,
                       const std::complex<double> *a, int64_t lda, const std::complex<double> *b,
                       int64_t ldb, std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event ssyrk_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                       MKL_TRANSPOSE trans, int64_t n, int64_t k, float alpha, const float *a,
                       int64_t lda, float beta, float *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_c = 0);

sycl::event dsyrk_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                       MKL_TRANSPOSE trans, int64_t n, int64_t k, double alpha, const double *a,
                       int64_t lda, double beta, double *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_c = 0);

sycl::event csyrk_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                       MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<float> alpha,
                       const std::complex<float> *a, int64_t lda, std::complex<float> beta,
                       std::complex<float> *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_c = 0);

sycl::event zsyrk_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                       MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<double> alpha,
                       const std::complex<double> *a, int64_t lda, std::complex<double> beta,
                       std::complex<double> *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_c = 0);

sycl::event ssyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, float alpha, const float *a,
                             int64_t lda, int64_t stride_a, float beta, float *c, int64_t ldc,
                             int64_t stride_c, int64_t batchsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_c = 0);

sycl::event dsyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, double alpha,
                             const double *a, int64_t lda, int64_t stride_a, double beta, double *c,
                             int64_t ldc, int64_t stride_c, int64_t batchsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_c = 0);

sycl::event csyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<float> alpha,
                             const std::complex<float> *a, int64_t lda, int64_t stride_a,
                             std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                             int64_t stride_c, int64_t batchsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_c = 0);

sycl::event zsyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<double> alpha,
                             const std::complex<double> *a, int64_t lda, int64_t stride_a,
                             std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                             int64_t stride_c, int64_t batchsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_c = 0);

sycl::event ssyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, float alpha,
                             const float **a, int64_t lda, float beta, float **c, int64_t ldc,
                             int64_t offset_batch, int64_t batchsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_c = 0);

sycl::event dsyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, double alpha,
                             const double **a, int64_t lda, double beta, double **c, int64_t ldc,
                             int64_t offset_batch, int64_t batchsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_c = 0);

sycl::event csyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<float> alpha,
                             const std::complex<float> **a, int64_t lda, std::complex<float> beta,
                             std::complex<float> **c, int64_t ldc, int64_t offset_batch,
                             int64_t batchsize, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_c = 0);

sycl::event zsyrk_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                             MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<double> alpha,
                             const std::complex<double> **a, int64_t lda, std::complex<double> beta,
                             std::complex<double> **c, int64_t ldc, int64_t offset_batch,
                             int64_t batchsize, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_c = 0);

sycl::event cherk_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                       MKL_TRANSPOSE trans, int64_t n, int64_t k, float alpha,
                       const std::complex<float> *a, int64_t lda, float beta,
                       std::complex<float> *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_c = 0);

sycl::event zherk_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                       MKL_TRANSPOSE trans, int64_t n, int64_t k, double alpha,
                       const std::complex<double> *a, int64_t lda, double beta,
                       std::complex<double> *c, int64_t ldc,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_c = 0);

sycl::event ssyr2k_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                        MKL_TRANSPOSE trans, int64_t n, int64_t k, float alpha, const float *a,
                        int64_t lda, const float *b, int64_t ldb, float beta, float *c, int64_t ldc,
                        const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                        int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event dsyr2k_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                        MKL_TRANSPOSE trans, int64_t n, int64_t k, double alpha, const double *a,
                        int64_t lda, const double *b, int64_t ldb, double beta, double *c,
                        int64_t ldc, const std::vector<sycl::event> &dependencies,
                        int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event csyr2k_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                        MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<float> alpha,
                        const std::complex<float> *a, int64_t lda, const std::complex<float> *b,
                        int64_t ldb, std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                        const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                        int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event zsyr2k_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                        MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<double> alpha,
                        const std::complex<double> *a, int64_t lda, const std::complex<double> *b,
                        int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                        int64_t ldc, const std::vector<sycl::event> &dependencies,
                        int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event cher2k_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                        MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<float> alpha,
                        const std::complex<float> *a, int64_t lda, const std::complex<float> *b,
                        int64_t ldb, float beta, std::complex<float> *c, int64_t ldc,
                        const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                        int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event zher2k_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                        MKL_TRANSPOSE trans, int64_t n, int64_t k, std::complex<double> alpha,
                        const std::complex<double> *a, int64_t lda, const std::complex<double> *b,
                        int64_t ldb, double beta, std::complex<double> *c, int64_t ldc,
                        const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                        int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event strmm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m,
                       int64_t n, float alpha, const float *a, int64_t lda, float *b, int64_t ldb,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0);

sycl::event dtrmm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m,
                       int64_t n, double alpha, const double *a, int64_t lda, double *b,
                       int64_t ldb, const std::vector<sycl::event> &dependencies,
                       int64_t offset_a = 0, int64_t offset_b = 0);

sycl::event ctrmm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m,
                       int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                       int64_t lda, std::complex<float> *b, int64_t ldb,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0);

sycl::event ztrmm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m,
                       int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                       int64_t lda, std::complex<double> *b, int64_t ldb,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0);

sycl::event strsm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m,
                       int64_t n, float alpha, const float *a, int64_t lda, float *b, int64_t ldb,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0);

sycl::event dtrsm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m,
                       int64_t n, double alpha, const double *a, int64_t lda, double *b,
                       int64_t ldb, const std::vector<sycl::event> &dependencies,
                       int64_t offset_a = 0, int64_t offset_b = 0);

sycl::event ctrsm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m,
                       int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                       int64_t lda, std::complex<float> *b, int64_t ldb,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0);

sycl::event ztrsm_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                       MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag, int64_t m,
                       int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                       int64_t lda, std::complex<double> *b, int64_t ldb,
                       const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                       int64_t offset_b = 0);

sycl::event strsm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                             MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                             int64_t m, int64_t n, float alpha, const float **a, int64_t lda,
                             float **b, int64_t ldb, int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0);

sycl::event dtrsm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                             MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                             int64_t m, int64_t n, double alpha, const double **a, int64_t lda,
                             double **b, int64_t ldb, int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0);

sycl::event ctrsm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                             MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                             int64_t m, int64_t n, std::complex<float> alpha,
                             const std::complex<float> **a, int64_t lda, std::complex<float> **b,
                             int64_t ldb, int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0);

sycl::event ztrsm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                             MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                             int64_t m, int64_t n, std::complex<double> alpha,
                             const std::complex<double> **a, int64_t lda, std::complex<double> **b,
                             int64_t ldb, int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0);

sycl::event strsm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                             MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                             int64_t m, int64_t n, float alpha, const float *a, int64_t lda,
                             int64_t stridea, float *b, int64_t ldb, int64_t strideb,
                             int64_t batchsize, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_b = 0);

sycl::event dtrsm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                             MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                             int64_t m, int64_t n, double alpha, const double *a, int64_t lda,
                             int64_t stridea, double *b, int64_t ldb, int64_t strideb,
                             int64_t batchsize, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_b = 0);

sycl::event ctrsm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                             MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                             int64_t m, int64_t n, std::complex<float> alpha,
                             const std::complex<float> *a, int64_t lda, int64_t stridea,
                             std::complex<float> *b, int64_t ldb, int64_t strideb,
                             int64_t batchsize, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_b = 0);

sycl::event ztrsm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right,
                             MKL_UPLO upper_lower, MKL_TRANSPOSE transa, MKL_DIAG unit_diag,
                             int64_t m, int64_t n, std::complex<double> alpha,
                             const std::complex<double> *a, int64_t lda, int64_t stridea,
                             std::complex<double> *b, int64_t ldb, int64_t strideb,
                             int64_t batchsize, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_b = 0);

sycl::event sgemv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m,
                       int64_t n, float alpha, const float *a, int64_t lda, const float *x,
                       int64_t incx, float beta, float *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event dgemv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m,
                       int64_t n, double alpha, const double *a, int64_t lda, const double *x,
                       int64_t incx, double beta, double *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event cgemv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m,
                       int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                       int64_t lda, const std::complex<float> *x, int64_t incx,
                       std::complex<float> beta, std::complex<float> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event zgemv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m,
                       int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                       int64_t lda, const std::complex<double> *x, int64_t incx,
                       std::complex<double> beta, std::complex<double> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event sgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, float alpha, const float *a, int64_t lda, int64_t strideA,
                             const float *x, int64_t incx, int64_t stride_x, float beta, float *y,
                             int64_t incy, int64_t stride_y, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event dgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, double alpha, const double *a, int64_t lda, int64_t strideA,
                             const double *x, int64_t incx, int64_t stride_x, double beta,
                             double *y, int64_t incy, int64_t stride_y, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event cgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                             int64_t lda, int64_t strideA, const std::complex<float> *x,
                             int64_t incx, int64_t stride_x, std::complex<float> beta,
                             std::complex<float> *y, int64_t incy, int64_t stride_y,
                             int64_t groupsize, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event zgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                             int64_t lda, int64_t strideA, const std::complex<double> *x,
                             int64_t incx, int64_t stride_x, std::complex<double> beta,
                             std::complex<double> *y, int64_t incy, int64_t stride_y,
                             int64_t groupsize, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event sgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, float alpha, const float **a, int64_t lda, const float **x,
                             int64_t incx, float beta, float **y, int64_t incy,
                             int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event dgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, double alpha, const double **a, int64_t lda,
                             const double **x, int64_t incx, double beta, double **y, int64_t incy,
                             int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event cgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, std::complex<float> alpha, const std::complex<float> **a,
                             int64_t lda, const std::complex<float> **x, int64_t incx,
                             std::complex<float> beta, std::complex<float> **y, int64_t incy,
                             int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event zgemv_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa, int64_t m,
                             int64_t n, std::complex<double> alpha, const std::complex<double> **a,
                             int64_t lda, const std::complex<double> **x, int64_t incx,
                             std::complex<double> beta, std::complex<double> **y, int64_t incy,
                             int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_y = 0);

sycl::event sdgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, const float *a, int64_t lda, int64_t strideA,
                             const float *x, int64_t incx, int64_t stride_x, float *c, int64_t ldc,
                             int64_t stride_c, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_c = 0);

sycl::event ddgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, const double *a, int64_t lda, int64_t strideA,
                             const double *x, int64_t incx, int64_t stride_x, double *c,
                             int64_t ldc, int64_t stride_c, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_c = 0);

sycl::event cdgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, const std::complex<float> *a, int64_t lda, int64_t strideA,
                             const std::complex<float> *x, int64_t incx, int64_t stride_x,
                             std::complex<float> *c, int64_t ldc, int64_t stride_c,
                             int64_t groupsize, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_x = 0, int64_t offset_c = 0);

sycl::event zdgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, const std::complex<double> *a, int64_t lda, int64_t strideA,
                             const std::complex<double> *x, int64_t incx, int64_t stride_x,
                             std::complex<double> *c, int64_t ldc, int64_t stride_c,
                             int64_t groupsize, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_x = 0, int64_t offset_c = 0);

sycl::event sdgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, const float **a, int64_t lda, const float **x, int64_t incx,
                             float **c, int64_t ldc, int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_c = 0);

sycl::event ddgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, const double **a, int64_t lda, const double **x,
                             int64_t incx, double **c, int64_t ldc, int64_t offset_batch,
                             int64_t groupsize, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_x = 0, int64_t offset_c = 0);

sycl::event cdgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, const std::complex<float> **a, int64_t lda,
                             const std::complex<float> **x, int64_t incx, std::complex<float> **c,
                             int64_t ldc, int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_c = 0);

sycl::event zdgmm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_SIDE left_right, int64_t m,
                             int64_t n, const std::complex<double> **a, int64_t lda,
                             const std::complex<double> **x, int64_t incx, std::complex<double> **c,
                             int64_t ldc, int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_x = 0, int64_t offset_c = 0);

sycl::event sgbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m,
                       int64_t n, int64_t kl, int64_t ku, float alpha, const float *a, int64_t lda,
                       const float *x, int64_t incx, float beta, float *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event dgbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m,
                       int64_t n, int64_t kl, int64_t ku, double alpha, const double *a,
                       int64_t lda, const double *x, int64_t incx, double beta, double *y,
                       int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event cgbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m,
                       int64_t n, int64_t kl, int64_t ku, std::complex<float> alpha,
                       const std::complex<float> *a, int64_t lda, const std::complex<float> *x,
                       int64_t incx, std::complex<float> beta, std::complex<float> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event zgbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE trans, int64_t m,
                       int64_t n, int64_t kl, int64_t ku, std::complex<double> alpha,
                       const std::complex<double> *a, int64_t lda, const std::complex<double> *x,
                       int64_t incx, std::complex<double> beta, std::complex<double> *y,
                       int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event sger_sycl(sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n, float alpha,
                      const float *x, int64_t incx, const float *y, int64_t incy, float *a,
                      int64_t lda, const std::vector<sycl::event> &dependencies);

sycl::event dger_sycl(sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n, double alpha,
                      const double *x, int64_t incx, const double *y, int64_t incy, double *a,
                      int64_t lda, const std::vector<sycl::event> &dependencies);

sycl::event cgerc_sycl(sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n,
                       std::complex<float> alpha, const std::complex<float> *x, int64_t incx,
                       const std::complex<float> *y, int64_t incy, std::complex<float> *a,
                       int64_t lda, const std::vector<sycl::event> &dependencies);

sycl::event zgerc_sycl(sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n,
                       std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                       const std::complex<double> *y, int64_t incy, std::complex<double> *a,
                       int64_t lda, const std::vector<sycl::event> &dependencies);

sycl::event cgeru_sycl(sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n,
                       std::complex<float> alpha, const std::complex<float> *x, int64_t incx,
                       const std::complex<float> *y, int64_t incy, std::complex<float> *a,
                       int64_t lda, const std::vector<sycl::event> &dependencies);

sycl::event zgeru_sycl(sycl::queue *queue, MKL_LAYOUT layout, int64_t m, int64_t n,
                       std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                       const std::complex<double> *y, int64_t incy, std::complex<double> *a,
                       int64_t lda, const std::vector<sycl::event> &dependencies);

sycl::event chbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k,
                       std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                       const std::complex<float> *x, int64_t incx, std::complex<float> beta,
                       std::complex<float> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event zhbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k,
                       std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                       const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                       std::complex<double> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event chemv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                       std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                       const std::complex<float> *x, int64_t incx, std::complex<float> beta,
                       std::complex<float> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event zhemv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                       std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                       const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                       std::complex<double> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event cher_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
                      const std::complex<float> *x, int64_t incx, std::complex<float> *a,
                      int64_t lda, const std::vector<sycl::event> &dependencies);

sycl::event zher_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                      double alpha, const std::complex<double> *x, int64_t incx,
                      std::complex<double> *a, int64_t lda,
                      const std::vector<sycl::event> &dependencies);

sycl::event cher2_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                       std::complex<float> alpha, const std::complex<float> *x, int64_t incx,
                       const std::complex<float> *y, int64_t incy, std::complex<float> *a,
                       int64_t lda, const std::vector<sycl::event> &dependencies);

sycl::event zher2_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                       std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                       const std::complex<double> *y, int64_t incy, std::complex<double> *a,
                       int64_t lda, const std::vector<sycl::event> &dependencies);

sycl::event chpmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                       std::complex<float> alpha, const std::complex<float> *a,
                       const std::complex<float> *x, int64_t incx, std::complex<float> beta,
                       std::complex<float> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event zhpmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                       std::complex<double> alpha, const std::complex<double> *a,
                       const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                       std::complex<double> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event chpr_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
                      const std::complex<float> *x, int64_t incx, std::complex<float> *a,
                      const std::vector<sycl::event> &dependencies);

sycl::event zhpr_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                      double alpha, const std::complex<double> *x, int64_t incx,
                      std::complex<double> *a, const std::vector<sycl::event> &dependencies);

sycl::event chpr2_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                       std::complex<float> alpha, const std::complex<float> *x, int64_t incx,
                       const std::complex<float> *y, int64_t incy, std::complex<float> *a,
                       const std::vector<sycl::event> &dependencies);

sycl::event zhpr2_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                       std::complex<double> alpha, const std::complex<double> *x, int64_t incx,
                       const std::complex<double> *y, int64_t incy, std::complex<double> *a,
                       const std::vector<sycl::event> &dependencies);

sycl::event ssbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k,
                       float alpha, const float *a, int64_t lda, const float *x, int64_t incx,
                       float beta, float *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event dsbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, int64_t k,
                       double alpha, const double *a, int64_t lda, const double *x, int64_t incx,
                       double beta, double *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event sspmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, float alpha,
                       const float *a, const float *x, int64_t incx, float beta, float *y,
                       int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event dspmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                       double alpha, const double *a, const double *x, int64_t incx, double beta,
                       double *y, int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event sspr_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
                      const float *x, int64_t incx, float *a,
                      const std::vector<sycl::event> &dependencies);

sycl::event dspr_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                      double alpha, const double *x, int64_t incx, double *a,
                      const std::vector<sycl::event> &dependencies);

sycl::event sspr2_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                       float alpha, const float *x, int64_t incx, const float *y, int64_t incy,
                       float *a, const std::vector<sycl::event> &dependencies);

sycl::event dspr2_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                       double alpha, const double *x, int64_t incx, const double *y, int64_t incy,
                       double *a, const std::vector<sycl::event> &dependencies);

sycl::event ssymv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n, float alpha,
                       const float *a, int64_t lda, const float *x, int64_t incx, float beta,
                       float *y, int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event dsymv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO uplo, int64_t n,
                       double alpha, const double *a, int64_t lda, const double *x, int64_t incx,
                       double beta, double *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event ssyr_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n, float alpha,
                      const float *x, int64_t incx, float *a, int64_t lda,
                      const std::vector<sycl::event> &dependencies);

sycl::event dsyr_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                      double alpha, const double *x, int64_t incx, double *a, int64_t lda,
                      const std::vector<sycl::event> &dependencies);

sycl::event ssyr2_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                       float alpha, const float *x, int64_t incx, const float *y, int64_t incy,
                       float *a, int64_t lda, const std::vector<sycl::event> &dependencies);

sycl::event dsyr2_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, int64_t n,
                       double alpha, const double *x, int64_t incx, const double *y, int64_t incy,
                       double *a, int64_t lda, const std::vector<sycl::event> &dependencies);

sycl::event stbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, int64_t k, const float *a, int64_t lda, float *x,
                       int64_t incx, const std::vector<sycl::event> &dependencies);

sycl::event dtbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, int64_t k, const double *a, int64_t lda, double *x,
                       int64_t incx, const std::vector<sycl::event> &dependencies);

sycl::event ctbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, int64_t k, const std::complex<float> *a,
                       int64_t lda, std::complex<float> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event ztbmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, int64_t k, const std::complex<double> *a,
                       int64_t lda, std::complex<double> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event stbsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, int64_t k, const float *a, int64_t lda, float *x,
                       int64_t incx, const std::vector<sycl::event> &dependencies);

sycl::event dtbsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, int64_t k, const double *a, int64_t lda, double *x,
                       int64_t incx, const std::vector<sycl::event> &dependencies);

sycl::event ctbsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, int64_t k, const std::complex<float> *a,
                       int64_t lda, std::complex<float> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event ztbsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, int64_t k, const std::complex<double> *a,
                       int64_t lda, std::complex<double> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event stpmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const float *a, float *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event dtpmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const double *a, double *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event ctpmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const std::complex<float> *a,
                       std::complex<float> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event ztpmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const std::complex<double> *a,
                       std::complex<double> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event stpsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const float *a, float *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event dtpsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const double *a, double *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event ctpsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const std::complex<float> *a,
                       std::complex<float> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event ztpsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const std::complex<double> *a,
                       std::complex<double> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event strmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const float *a, int64_t lda, float *x,
                       int64_t incx, const std::vector<sycl::event> &dependencies);

sycl::event dtrmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const double *a, int64_t lda, double *x,
                       int64_t incx, const std::vector<sycl::event> &dependencies);

sycl::event ctrmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const std::complex<float> *a, int64_t lda,
                       std::complex<float> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event ztrmv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const std::complex<double> *a, int64_t lda,
                       std::complex<double> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event strsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const float *a, int64_t lda, float *x,
                       int64_t incx, const std::vector<sycl::event> &dependencies);

sycl::event dtrsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const double *a, int64_t lda, double *x,
                       int64_t incx, const std::vector<sycl::event> &dependencies);

sycl::event ctrsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const std::complex<float> *a, int64_t lda,
                       std::complex<float> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event ztrsv_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upplo, MKL_TRANSPOSE trans,
                       MKL_DIAG diag, int64_t n, const std::complex<double> *a, int64_t lda,
                       std::complex<double> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event scasum_sycl(sycl::queue *queue, int64_t n, const std::complex<float> *x, int64_t incx,
                        float *result, const std::vector<sycl::event> &dependencies);

sycl::event dzasum_sycl(sycl::queue *queue, int64_t n, const std::complex<double> *x, int64_t incx,
                        double *result, const std::vector<sycl::event> &dependencies);

sycl::event sasum_sycl(sycl::queue *queue, int64_t n, const float *x, int64_t incx, float *result,
                       const std::vector<sycl::event> &dependencies);

sycl::event dasum_sycl(sycl::queue *queue, int64_t n, const double *x, int64_t incx, double *result,
                       const std::vector<sycl::event> &dependencies);

sycl::event saxpy_sycl(sycl::queue *queue, int64_t n, float alpha, const float *x, int64_t incx,
                       float *y, int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event daxpy_sycl(sycl::queue *queue, int64_t n, double alpha, const double *x, int64_t incx,
                       double *y, int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event caxpy_sycl(sycl::queue *queue, int64_t n, std::complex<float> alpha,
                       const std::complex<float> *x, int64_t incx, std::complex<float> *y,
                       int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event zaxpy_sycl(sycl::queue *queue, int64_t n, std::complex<double> alpha,
                       const std::complex<double> *x, int64_t incx, std::complex<double> *y,
                       int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event saxpby_sycl(sycl::queue *queue, int64_t n, float alpha, const float *x, int64_t incx,
                        float beta, float *y, int64_t incy,
                        const std::vector<sycl::event> &dependencies);

sycl::event daxpby_sycl(sycl::queue *queue, int64_t n, double alpha, const double *x, int64_t incx,
                        double beta, double *y, int64_t incy,
                        const std::vector<sycl::event> &dependencies);

sycl::event caxpby_sycl(sycl::queue *queue, int64_t n, std::complex<float> alpha,
                        const std::complex<float> *x, int64_t incx, std::complex<float> beta,
                        std::complex<float> *y, int64_t incy,
                        const std::vector<sycl::event> &dependencies);

sycl::event zaxpby_sycl(sycl::queue *queue, int64_t n, std::complex<double> alpha,
                        const std::complex<double> *x, int64_t incx, std::complex<double> beta,
                        std::complex<double> *y, int64_t incy,
                        const std::vector<sycl::event> &dependencies);

sycl::event scopy_sycl(sycl::queue *queue, int64_t n, const float *x, int64_t incx, float *y,
                       int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event dcopy_sycl(sycl::queue *queue, int64_t n, const double *x, int64_t incx, double *y,
                       int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event ccopy_sycl(sycl::queue *queue, int64_t n, const std::complex<float> *x, int64_t incx,
                       std::complex<float> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event zcopy_sycl(sycl::queue *queue, int64_t n, const std::complex<double> *x, int64_t incx,
                       std::complex<double> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event scopy_batch_sycl(sycl::queue *queue, int64_t n, const float **x, int64_t incx,
                             float **y, int64_t incy, int64_t offset_batch, int64_t batch_size,
                             const std::vector<sycl::event> &dependencies);
sycl::event ccopy_batch_sycl(sycl::queue *queue, int64_t n, const std::complex<float> **x,
                             int64_t incx, std::complex<float> **y, int64_t incy,
                             int64_t offset_batch, int64_t batch_size,
                             const std::vector<sycl::event> &dependencies);
sycl::event zcopy_batch_sycl(sycl::queue *queue, int64_t n, const std::complex<double> **x,
                             int64_t incx, std::complex<double> **y, int64_t incy,
                             int64_t offset_batch, int64_t batch_size,
                             const std::vector<sycl::event> &dependencies);
sycl::event dcopy_batch_sycl(sycl::queue *queue, int64_t n, const double **x, int64_t incx,
                             double **y, int64_t incy, int64_t offset_batch, int64_t batch_size,
                             const std::vector<sycl::event> &dependencies);
sycl::event scopy_batch_sycl(sycl::queue *queue, int64_t n, const float *x, int64_t incx,
                             std::int64_t stridex, float *y, int64_t incy, std::int64_t stridey,
                             std::int64_t batch_size, const std::vector<sycl::event> &dependencies);
sycl::event ccopy_batch_sycl(sycl::queue *queue, int64_t n, const std::complex<float> *x,
                             int64_t incx, std::int64_t stridex, std::complex<float> *y,
                             int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                             const std::vector<sycl::event> &dependencies);
sycl::event zcopy_batch_sycl(sycl::queue *queue, int64_t n, const std::complex<double> *x,
                             int64_t incx, std::int64_t stridex, std::complex<double> *y,
                             int64_t incy, std::int64_t stridey, std::int64_t batch_size,
                             const std::vector<sycl::event> &dependencies);
sycl::event dcopy_batch_sycl(sycl::queue *queue, int64_t n, const double *x, int64_t incx,
                             std::int64_t stridex, double *y, int64_t incy, std::int64_t stridey,
                             std::int64_t batch_size, const std::vector<sycl::event> &dependencies);

sycl::event sdot_sycl(sycl::queue *queue, int64_t n, const float *x, int64_t incx, const float *y,
                      int64_t incy, float *result, const std::vector<sycl::event> &dependencies);

sycl::event ddot_sycl(sycl::queue *queue, int64_t n, const double *x, int64_t incx, const double *y,
                      int64_t incy, double *result, const std::vector<sycl::event> &dependencies);

sycl::event sdsdot_sycl(sycl::queue *queue, int64_t n, float sb, const float *x, int64_t incx,
                        const float *y, int64_t incy, float *result,
                        const std::vector<sycl::event> &dependencies);

sycl::event dsdot_sycl(sycl::queue *queue, int64_t n, const float *x, int64_t incx, const float *y,
                       int64_t incy, double *result, const std::vector<sycl::event> &dependencies);

sycl::event cdotc_sycl(sycl::queue *queue, int64_t n, const std::complex<float> *x, int64_t incx,
                       const std::complex<float> *y, int64_t incy, std::complex<float> *result,
                       const std::vector<sycl::event> &dependencies);

sycl::event zdotc_sycl(sycl::queue *queue, int64_t n, const std::complex<double> *x, int64_t incx,
                       const std::complex<double> *y, int64_t incy, std::complex<double> *result,
                       const std::vector<sycl::event> &dependencies);

sycl::event cdotu_sycl(sycl::queue *queue, int64_t n, const std::complex<float> *x, int64_t incx,
                       const std::complex<float> *y, int64_t incy, std::complex<float> *result,
                       const std::vector<sycl::event> &dependencies);

sycl::event zdotu_sycl(sycl::queue *queue, int64_t n, const std::complex<double> *x, int64_t incx,
                       const std::complex<double> *y, int64_t incy, std::complex<double> *result,
                       const std::vector<sycl::event> &dependencies);

sycl::event scnrm2_sycl(sycl::queue *queue, int64_t n, const std::complex<float> *x, int64_t incx,
                        float *result, const std::vector<sycl::event> &dependencies);

sycl::event dznrm2_sycl(sycl::queue *queue, int64_t n, const std::complex<double> *x, int64_t incx,
                        double *result, const std::vector<sycl::event> &dependencies);

sycl::event snrm2_sycl(sycl::queue *queue, int64_t n, const float *x, int64_t incx, float *result,
                       const std::vector<sycl::event> &dependencies);

sycl::event dnrm2_sycl(sycl::queue *queue, int64_t n, const double *x, int64_t incx, double *result,
                       const std::vector<sycl::event> &dependencies);

sycl::event csrot_sycl(sycl::queue *queue, int64_t n, std::complex<float> *x, int64_t incx,
                       std::complex<float> *y, int64_t incy, float c, float s,
                       const std::vector<sycl::event> &dependencies);

sycl::event zdrot_sycl(sycl::queue *queue, int64_t n, std::complex<double> *x, int64_t incx,
                       std::complex<double> *y, int64_t incy, double c, double s,
                       const std::vector<sycl::event> &dependencies);

sycl::event srot_sycl(sycl::queue *queue, int64_t n, float *x, int64_t incx, float *y, int64_t incy,
                      float c, float s, const std::vector<sycl::event> &dependencies);

sycl::event drot_sycl(sycl::queue *queue, int64_t n, double *x, int64_t incx, double *y,
                      int64_t incy, double c, double s,
                      const std::vector<sycl::event> &dependencies);

sycl::event srotg_sycl(sycl::queue *queue, float *a, float *b, float *c, float *s,
                       const std::vector<sycl::event> &dependencies);

sycl::event drotg_sycl(sycl::queue *queue, double *a, double *b, double *c, double *s,
                       const std::vector<sycl::event> &dependencies);

sycl::event crotg_sycl(sycl::queue *queue, std::complex<float> *a, std::complex<float> *b, float *c,
                       std::complex<float> *s, const std::vector<sycl::event> &dependencies);

sycl::event zrotg_sycl(sycl::queue *queue, std::complex<double> *a, std::complex<double> *b,
                       double *c, std::complex<double> *s,
                       const std::vector<sycl::event> &dependencies);

sycl::event srotm_sycl(sycl::queue *queue, int64_t n, float *x, int64_t incx, float *y,
                       int64_t incy, float *param, const std::vector<sycl::event> &dependencies);

sycl::event drotm_sycl(sycl::queue *queue, int64_t n, double *x, int64_t incx, double *y,
                       int64_t incy, double *param, const std::vector<sycl::event> &dependencies);

sycl::event srotmg_sycl(sycl::queue *queue, float *d1, float *d2, float *x1, float y1, float *param,
                        const std::vector<sycl::event> &dependencies);

sycl::event drotmg_sycl(sycl::queue *queue, double *d1, double *d2, double *x1, double y1,
                        double *param, const std::vector<sycl::event> &dependencies);

sycl::event sscal_sycl(sycl::queue *queue, int64_t n, float alpha, float *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event dscal_sycl(sycl::queue *queue, int64_t n, double alpha, double *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event cscal_sycl(sycl::queue *queue, int64_t n, std::complex<float> alpha,
                       std::complex<float> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event zscal_sycl(sycl::queue *queue, int64_t n, std::complex<double> alpha,
                       std::complex<double> *x, int64_t incx,
                       const std::vector<sycl::event> &dependencies);

sycl::event csscal_sycl(sycl::queue *queue, int64_t n, float alpha, std::complex<float> *x,
                        int64_t incx, const std::vector<sycl::event> &dependencies);

sycl::event zdscal_sycl(sycl::queue *queue, int64_t n, double alpha, std::complex<double> *x,
                        int64_t incx, const std::vector<sycl::event> &dependencies);

sycl::event sswap_sycl(sycl::queue *queue, int64_t n, float *x, int64_t incx, float *y,
                       int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event dswap_sycl(sycl::queue *queue, int64_t n, double *x, int64_t incx, double *y,
                       int64_t incy, const std::vector<sycl::event> &dependencies);

sycl::event cswap_sycl(sycl::queue *queue, int64_t n, std::complex<float> *x, int64_t incx,
                       std::complex<float> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event zswap_sycl(sycl::queue *queue, int64_t n, std::complex<double> *x, int64_t incx,
                       std::complex<double> *y, int64_t incy,
                       const std::vector<sycl::event> &dependencies);

sycl::event isamax_sycl(sycl::queue *queue, int64_t n, const float *x, int64_t incx,
                        int64_t *result, const std::vector<sycl::event> &dependencies);

sycl::event idamax_sycl(sycl::queue *queue, int64_t n, const double *x, int64_t incx,
                        int64_t *result, const std::vector<sycl::event> &dependencies);

sycl::event icamax_sycl(sycl::queue *queue, int64_t n, const std::complex<float> *x, int64_t incx,
                        int64_t *result, const std::vector<sycl::event> &dependencies);

sycl::event izamax_sycl(sycl::queue *queue, int64_t n, const std::complex<double> *x, int64_t incx,
                        int64_t *result, const std::vector<sycl::event> &dependencies);

sycl::event isamin_sycl(sycl::queue *queue, int64_t n, const float *x, int64_t incx,
                        int64_t *result, const std::vector<sycl::event> &dependencies);

sycl::event idamin_sycl(sycl::queue *queue, int64_t n, const double *x, int64_t incx,
                        int64_t *result, const std::vector<sycl::event> &dependencies);

sycl::event icamin_sycl(sycl::queue *queue, int64_t n, const std::complex<float> *x, int64_t incx,
                        int64_t *result, const std::vector<sycl::event> &dependencies);

sycl::event izamin_sycl(sycl::queue *queue, int64_t n, const std::complex<double> *x, int64_t incx,
                        int64_t *result, const std::vector<sycl::event> &dependencies);

sycl::event sgemm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                             MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha,
                             const float *a, int64_t lda, int64_t strideA, const float *b,
                             int64_t ldb, int64_t strideB, float beta, float *c, int64_t ldc,
                             int64_t strideC, int64_t group_size,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event dgemm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                             MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, double alpha,
                             const double *a, int64_t lda, int64_t strideA, const double *b,
                             int64_t ldb, int64_t strideB, double beta, double *c, int64_t ldc,
                             int64_t strideC, int64_t group_size,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event cgemm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                             MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                             std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                             int64_t strideA, const std::complex<float> *b, int64_t ldb,
                             int64_t strideB, std::complex<float> beta, std::complex<float> *c,
                             int64_t ldc, int64_t strideC, int64_t group_size,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event zgemm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                             MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                             std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                             int64_t strideA, const std::complex<double> *b, int64_t ldb,
                             int64_t strideB, std::complex<double> beta, std::complex<double> *c,
                             int64_t ldc, int64_t strideC, int64_t group_size,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event hgemm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                             MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                             sycl::half alpha, const sycl::half *a, int64_t lda, int64_t strideA,
                             const sycl::half *b, int64_t ldb, int64_t strideB, sycl::half beta,
                             sycl::half *c, int64_t ldc, int64_t strideC, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event sgemm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                             MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, float alpha,
                             const float **a, int64_t lda, const float **b, int64_t ldb, float beta,
                             float **c, int64_t ldc, int64_t offset_batch, int64_t group_size,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event dgemm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                             MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k, double alpha,
                             const double **a, int64_t lda, const double **b, int64_t ldb,
                             double beta, double **c, int64_t ldc, int64_t offset_batch,
                             int64_t group_size, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event cgemm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                             MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                             std::complex<float> alpha, const std::complex<float> **a, int64_t lda,
                             const std::complex<float> **b, int64_t ldb, std::complex<float> beta,
                             std::complex<float> **c, int64_t ldc, int64_t offset_batch,
                             int64_t group_size, const std::vector<sycl::event> &dependencies,
                             int64_t offset_a = 0, int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event zgemm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                             MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                             std::complex<double> alpha, const std::complex<double> **a,
                             int64_t lda, const std::complex<double> **b, int64_t ldb,
                             std::complex<double> beta, std::complex<double> **c, int64_t ldc,
                             int64_t offset_batch, int64_t group_size,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event hgemm_batch_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                             MKL_TRANSPOSE transb, int64_t m, int64_t n, int64_t k,
                             sycl::half alpha, const sycl::half **a, int64_t lda,
                             const sycl::half **b, int64_t ldb, sycl::half beta, sycl::half **c,
                             int64_t ldc, int64_t offset_batch, int64_t groupsize,
                             const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                             int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event saxpy_batch_sycl(sycl::queue *queue, int64_t n, float alpha, const float **x,
                             int64_t incx, float **y, int64_t incy, int64_t batch_size,
                             int64_t offset, const std::vector<sycl::event> &dependencies);

sycl::event daxpy_batch_sycl(sycl::queue *queue, int64_t n, double alpha, const double **x,
                             int64_t incx, double **y, int64_t incy, int64_t batch_size,
                             int64_t offset, const std::vector<sycl::event> &dependencies);

sycl::event zaxpy_batch_sycl(sycl::queue *queue, int64_t n, std::complex<double> alpha,
                             const std::complex<double> **x, int64_t incx, std::complex<double> **y,
                             int64_t incy, int64_t batch_size, int64_t offset,
                             const std::vector<sycl::event> &dependencies);

sycl::event caxpy_batch_sycl(sycl::queue *queue, int64_t n, std::complex<float> alpha,
                             const std::complex<float> **x, int64_t incx, std::complex<float> **y,
                             int64_t incy, int64_t batch_size, int64_t offset,
                             const std::vector<sycl::event> &dependencies);

sycl::event saxpy_batch_sycl(sycl::queue *queue, int64_t n, float alpha, const float *x,
                             int64_t incx, int64_t stridex, float *y, int64_t incy, int64_t stridey,
                             int64_t batch_size, const std::vector<sycl::event> &dependencies);

sycl::event daxpy_batch_sycl(sycl::queue *queue, int64_t n, double alpha, const double *x,
                             int64_t incx, int64_t stridex, double *y, int64_t incy,
                             int64_t stridey, int64_t batch_size,
                             const std::vector<sycl::event> &dependencies);

sycl::event caxpy_batch_sycl(sycl::queue *queue, int64_t n, std::complex<float> alpha,
                             const std::complex<float> *x, int64_t incx, int64_t stridex,
                             std::complex<float> *y, int64_t incy, int64_t stridey,
                             int64_t batch_size, const std::vector<sycl::event> &dependencies);

sycl::event zaxpy_batch_sycl(sycl::queue *queue, int64_t n, std::complex<double> alpha,
                             const std::complex<double> *x, int64_t incx, int64_t stridex,
                             std::complex<double> *y, int64_t incy, int64_t stridey,
                             int64_t batch_size, const std::vector<sycl::event> &dependencies);

sycl::event sgemmt_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                        MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t n, int64_t k,
                        float alpha, const float *a, int64_t lda, const float *b, int64_t ldb,
                        float beta, float *c, int64_t ldc,
                        const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                        int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event dgemmt_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                        MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t n, int64_t k,
                        double alpha, const double *a, int64_t lda, const double *b, int64_t ldb,
                        double beta, double *c, int64_t ldc,
                        const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                        int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event zgemmt_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                        MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t n, int64_t k,
                        std::complex<double> alpha, const std::complex<double> *a, int64_t lda,
                        const std::complex<double> *b, int64_t ldb, std::complex<double> beta,
                        std::complex<double> *c, int64_t ldc,
                        const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                        int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event cgemmt_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_UPLO upper_lower,
                        MKL_TRANSPOSE transa, MKL_TRANSPOSE transb, int64_t n, int64_t k,
                        std::complex<float> alpha, const std::complex<float> *a, int64_t lda,
                        const std::complex<float> *b, int64_t ldb, std::complex<float> beta,
                        std::complex<float> *c, int64_t ldc,
                        const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                        int64_t offset_b = 0, int64_t offset_c = 0);

sycl::event gemm_s8s8s32_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                              MKL_TRANSPOSE transb, CBLAS_OFFSET offsetc, int64_t m, int64_t n,
                              int64_t k, float alpha, const int8_t *a, int64_t lda, int8_t ao,
                              const int8_t *b, int64_t ldb, int8_t bo, float beta, int32_t *c,
                              int64_t ldc, const int32_t *co,
                              const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                              int64_t offset_b = 0, int64_t offset_c = 0, int64_t offset_co = 0);

sycl::event gemm_s8u8s32_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                              MKL_TRANSPOSE transb, CBLAS_OFFSET offsetc, int64_t m, int64_t n,
                              int64_t k, float alpha, const int8_t *a, int64_t lda, int8_t ao,
                              const uint8_t *b, int64_t ldb, uint8_t bo, float beta, int32_t *c,
                              int64_t ldc, const int32_t *co,
                              const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                              int64_t offset_b = 0, int64_t offset_c = 0, int64_t offset_co = 0);

sycl::event gemm_u8s8s32_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                              MKL_TRANSPOSE transb, CBLAS_OFFSET offsetc, int64_t m, int64_t n,
                              int64_t k, float alpha, const uint8_t *a, int64_t lda, uint8_t ao,
                              const int8_t *b, int64_t ldb, int8_t bo, float beta, int32_t *c,
                              int64_t ldc, const int32_t *co,
                              const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                              int64_t offset_b = 0, int64_t offset_c = 0, int64_t offset_co = 0);

sycl::event gemm_u8u8s32_sycl(sycl::queue *queue, MKL_LAYOUT layout, MKL_TRANSPOSE transa,
                              MKL_TRANSPOSE transb, CBLAS_OFFSET offsetc, int64_t m, int64_t n,
                              int64_t k, float alpha, const uint8_t *a, int64_t lda, uint8_t ao,
                              const uint8_t *b, int64_t ldb, uint8_t bo, float beta, int32_t *c,
                              int64_t ldc, const int32_t *co,
                              const std::vector<sycl::event> &dependencies, int64_t offset_a = 0,
                              int64_t offset_b = 0, int64_t offset_c = 0, int64_t offset_co = 0);

} // namespace gpu
} // namespace mkl
} // namespace oneapi

#endif //_MKLGPU_COMMON_HPP
