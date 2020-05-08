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

#ifndef _REFERENCE_BLAS_TEMPLATES_HPP__
#define _REFERENCE_BLAS_TEMPLATES_HPP__

#include <stdlib.h>
#include <complex>
#include "cblas.h"

extern "C" {

void csrot_(const int *N, void *X, const int *incX, void *Y, const int *incY, const float *c,
            const float *s);

void zdrot_(const int *N, void *X, const int *incX, void *Y, const int *incY, const double *c,
            const double *s);

void crotg_(void *a, void *b, const float *c, void *s);

void zrotg_(void *a, void *b, const double *c, void *s);
}

template <typename T_src, typename T_dest>
static inline void copy_mat(T_src &src, CBLAS_TRANSPOSE trans, int row, int col, int ld,
                            T_dest *&dest) {
    int i, j;
    if (trans == CblasNoTrans) {
        for (j = 0; j < col; j++) {
            for (i = 0; i < row; i++) {
                dest[i + ld * j] = (T_dest)src[i + ld * j];
            }
        }
    }
    else {
        for (i = 0; i < row; i++) {
            for (j = 0; j < col; j++) {
                dest[i * ld + j] = (T_dest)src[i * ld + j];
            }
        }
    }
}

template <typename T_src, typename T_dest>
static inline void copy_mat(T_src &src, CBLAS_TRANSPOSE trans, int row, int col, int ld, T_dest off,
                            T_dest *&dest) {
    int i, j;
    if (trans == CblasNoTrans) {
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
static inline void copy_mat(T_src &src, int row, int col, int ld, CBLAS_OFFSET off_kind, T_off off,
                            T_dest &dest) {
    using T_data = typename std::remove_reference<decltype(dest[0])>::type;
    int i, j;
    T_data tmp;

    if (off_kind == CblasFixOffset) {
        tmp = off[0];
        for (j = 0; j < col; j++) {
            for (i = 0; i < row; i++) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else if (off_kind == CblasColOffset) {
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

template <typename T_src, typename T_desc>
static inline void update_c(T_src &src, CBLAS_UPLO upper_lower, int row, int col, int ld,
                            T_desc *&dest) {
    int i, j;
    for (j = 0; j < col; j++) {
        for (i = 0; i < row; i++) {
            if (upper_lower == CblasUpper) {
                if (j >= i)
                    dest[i + ld * j] = (T_desc)src[i + ld * j];
                else
                    dest[i + ld * j] = (T_desc)0.0;
            }
            else {
                if (j <= i)
                    dest[i + ld * j] = (T_desc)src[i + ld * j];
                else
                    dest[i + ld * j] = (T_desc)0.0;
            }
        }
    }
}

/* Level 3 */

template <typename fp>
static void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n,
                 const int *k, const fp *alpha, const fp *a, const int *lda, const fp *b,
                 const int *ldb, const fp *beta, fp *c, const int *ldc);

template <>
void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n, const int *k,
          const half *alpha, const half *a, const int *lda, const half *b, const int *ldb,
          const half *beta, half *c, const int *ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizea, sizeb, sizec;
    const float alphaf = *alpha;
    const float betaf  = *beta;
    sizea              = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
    sizeb              = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
    sizec              = *ldc * *n;
    float *af          = (float *)aligned_alloc(64, sizeof(float) * sizea);
    float *bf          = (float *)aligned_alloc(64, sizeof(float) * sizeb);
    float *cf          = (float *)aligned_alloc(64, sizeof(float) * sizec);
    copy_mat(a, transa, *m, *k, *lda, af);
    copy_mat(b, transb, *k, *n, *ldb, bf);
    copy_mat(c, CblasNoTrans, *m, *n, *ldc, cf);
    cblas_sgemm(CblasColMajor, transa, transb, *m, *n, *k, alphaf, af, *lda, bf, *ldb, betaf, cf,
                *ldc);
    copy_mat(cf, CblasNoTrans, *m, *n, *ldc, c);
    free(af);
    free(bf);
    free(cf);
}

template <>
void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n, const int *k,
          const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
          const float *beta, float *c, const int *ldc) {
    cblas_sgemm(CblasColMajor, transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c,
                *ldc);
}

template <>
void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n, const int *k,
          const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
          const double *beta, double *c, const int *ldc) {
    cblas_dgemm(CblasColMajor, transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c,
                *ldc);
}

template <>
void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n, const int *k,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *b, const int *ldb, const std::complex<float> *beta,
          std::complex<float> *c, const int *ldc) {
    cblas_cgemm(CblasColMajor, transa, transb, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
}

template <>
void gemm(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n, const int *k,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *b, const int *ldb, const std::complex<double> *beta,
          std::complex<double> *c, const int *ldc) {
    cblas_zgemm(CblasColMajor, transa, transb, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
}

template <typename fp>
static void symm(CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
                 const fp *alpha, const fp *a, const int *lda, const fp *b, const int *ldb,
                 const fp *beta, fp *c, const int *ldc);

template <>
void symm(CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n, const float *alpha,
          const float *a, const int *lda, const float *b, const int *ldb, const float *beta,
          float *c, const int *ldc) {
    cblas_ssymm(CblasColMajor, left_right, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void symm(CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n, const double *alpha,
          const double *a, const int *lda, const double *b, const int *ldb, const double *beta,
          double *c, const int *ldc) {
    cblas_dsymm(CblasColMajor, left_right, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void symm(CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *b, const int *ldb, const std::complex<float> *beta,
          std::complex<float> *c, const int *ldc) {
    cblas_csymm(CblasColMajor, left_right, uplo, *m, *n, alpha, a, *lda, b, *ldb, beta, c, *ldc);
}

template <>
void symm(CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *b, const int *ldb, const std::complex<double> *beta,
          std::complex<double> *c, const int *ldc) {
    cblas_zsymm(CblasColMajor, left_right, uplo, *m, *n, alpha, a, *lda, b, *ldb, beta, c, *ldc);
}

template <typename fp>
static void syrk(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
                 const fp *alpha, const fp *a, const int *lda, const fp *beta, fp *c,
                 const int *ldc);

template <>
void syrk(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k, const float *alpha,
          const float *a, const int *lda, const float *beta, float *c, const int *ldc) {
    cblas_ssyrk(CblasColMajor, uplo, trans, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

template <>
void syrk(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k, const double *alpha,
          const double *a, const int *lda, const double *beta, double *c, const int *ldc) {
    cblas_dsyrk(CblasColMajor, uplo, trans, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

template <>
void syrk(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *beta, std::complex<float> *c, const int *ldc) {
    cblas_csyrk(CblasColMajor, uplo, trans, *n, *k, alpha, a, *lda, beta, c, *ldc);
}

template <>
void syrk(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *beta, std::complex<double> *c, const int *ldc) {
    cblas_zsyrk(CblasColMajor, uplo, trans, *n, *k, alpha, a, *lda, beta, c, *ldc);
}

template <typename fp>
static void hemm(CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
                 const fp *alpha, const fp *a, const int *lda, const fp *b, const int *ldb,
                 const fp *beta, fp *c, const int *ldc);

template <>
void hemm(CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *b, const int *ldb, const std::complex<float> *beta,
          std::complex<float> *c, const int *ldc) {
    cblas_chemm(CblasColMajor, left_right, uplo, *m, *n, alpha, a, *lda, b, *ldb, beta, c, *ldc);
}

template <>
void hemm(CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *b, const int *ldb, const std::complex<double> *beta,
          std::complex<double> *c, const int *ldc) {
    cblas_zhemm(CblasColMajor, left_right, uplo, *m, *n, alpha, a, *lda, b, *ldb, beta, c, *ldc);
}

template <typename fp_scalar, typename fp_data>
static void herk(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
                 const fp_scalar *alpha, const fp_data *a, const int *lda, const fp_scalar *beta,
                 fp_data *c, const int *ldc);

template <>
void herk(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k, const float *alpha,
          const std::complex<float> *a, const int *lda, const float *beta, std::complex<float> *c,
          const int *ldc) {
    cblas_cherk(CblasColMajor, uplo, trans, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

template <>
void herk(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k, const double *alpha,
          const std::complex<double> *a, const int *lda, const double *beta,
          std::complex<double> *c, const int *ldc) {
    cblas_zherk(CblasColMajor, uplo, trans, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

template <typename fp>
static void syr2k(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
                  const fp *alpha, const fp *a, const int *lda, const fp *b, const int *ldb,
                  const fp *beta, fp *c, const int *ldc);

template <>
void syr2k(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k, const float *alpha,
           const float *a, const int *lda, const float *b, const int *ldb, const float *beta,
           float *c, const int *ldc) {
    cblas_ssyr2k(CblasColMajor, uplo, trans, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void syr2k(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k, const double *alpha,
           const double *a, const int *lda, const double *b, const int *ldb, const double *beta,
           double *c, const int *ldc) {
    cblas_dsyr2k(CblasColMajor, uplo, trans, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void syr2k(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
           const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
           const std::complex<float> *b, const int *ldb, const std::complex<float> *beta,
           std::complex<float> *c, const int *ldc) {
    cblas_csyr2k(CblasColMajor, uplo, trans, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
}

template <>
void syr2k(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
           const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
           const std::complex<double> *b, const int *ldb, const std::complex<double> *beta,
           std::complex<double> *c, const int *ldc) {
    cblas_zsyr2k(CblasColMajor, uplo, trans, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
}

template <typename fp_scalar, typename fp_data>
static void her2k(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
                  const fp_data *alpha, const fp_data *a, const int *lda, const fp_data *b,
                  const int *ldb, const fp_scalar *beta, fp_data *c, const int *ldc);

template <>
void her2k(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
           const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
           const std::complex<float> *b, const int *ldb, const float *beta, std::complex<float> *c,
           const int *ldc) {
    cblas_cher2k(CblasColMajor, uplo, trans, *n, *k, alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void her2k(CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
           const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
           const std::complex<double> *b, const int *ldb, const double *beta,
           std::complex<double> *c, const int *ldc) {
    cblas_zher2k(CblasColMajor, uplo, trans, *n, *k, alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <typename fp>
static void trmm(CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa, CBLAS_DIAG diag,
                 const int *m, const int *n, const fp *alpha, const fp *a, const int *lda, fp *b,
                 const int *ldb);

template <>
void trmm(CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int *m,
          const int *n, const float *alpha, const float *a, const int *lda, float *b,
          const int *ldb) {
    cblas_strmm(CblasColMajor, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trmm(CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int *m,
          const int *n, const double *alpha, const double *a, const int *lda, double *b,
          const int *ldb) {
    cblas_dtrmm(CblasColMajor, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trmm(CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int *m,
          const int *n, const std::complex<float> *alpha, const std::complex<float> *a,
          const int *lda, std::complex<float> *b, const int *ldb) {
    cblas_ctrmm(CblasColMajor, side, uplo, transa, diag, *m, *n, alpha, a, *lda, b, *ldb);
}

template <>
void trmm(CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int *m,
          const int *n, const std::complex<double> *alpha, const std::complex<double> *a,
          const int *lda, std::complex<double> *b, const int *ldb) {
    cblas_ztrmm(CblasColMajor, side, uplo, transa, diag, *m, *n, alpha, a, *lda, b, *ldb);
}

template <typename fp>
static void trsm(CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa, CBLAS_DIAG diag,
                 const int *m, const int *n, const fp *alpha, const fp *a, const int *lda, fp *b,
                 const int *ldb);

template <>
void trsm(CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int *m,
          const int *n, const float *alpha, const float *a, const int *lda, float *b,
          const int *ldb) {
    cblas_strsm(CblasColMajor, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trsm(CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int *m,
          const int *n, const double *alpha, const double *a, const int *lda, double *b,
          const int *ldb) {
    cblas_dtrsm(CblasColMajor, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trsm(CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int *m,
          const int *n, const std::complex<float> *alpha, const std::complex<float> *a,
          const int *lda, std::complex<float> *b, const int *ldb) {
    cblas_ctrsm(CblasColMajor, side, uplo, transa, diag, *m, *n, alpha, a, *lda, b, *ldb);
}

template <>
void trsm(CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int *m,
          const int *n, const std::complex<double> *alpha, const std::complex<double> *a,
          const int *lda, std::complex<double> *b, const int *ldb) {
    cblas_ztrsm(CblasColMajor, side, uplo, transa, diag, *m, *n, alpha, a, *lda, b, *ldb);
}

/*  Level 2 */

template <typename fp>
static void gemv(CBLAS_TRANSPOSE trans, const int *m, const int *n, const fp *alpha, const fp *a,
                 const int *lda, const fp *x, const int *incx, const fp *beta, fp *y,
                 const int *incy);

template <>
void gemv(CBLAS_TRANSPOSE trans, const int *m, const int *n, const float *alpha, const float *a,
          const int *lda, const float *x, const int *incx, const float *beta, float *y,
          const int *incy) {
    cblas_sgemv(CblasColMajor, trans, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void gemv(CBLAS_TRANSPOSE trans, const int *m, const int *n, const double *alpha, const double *a,
          const int *lda, const double *x, const int *incx, const double *beta, double *y,
          const int *incy) {
    cblas_dgemv(CblasColMajor, trans, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void gemv(CBLAS_TRANSPOSE trans, const int *m, const int *n, const std::complex<float> *alpha,
          const std::complex<float> *a, const int *lda, const std::complex<float> *x,
          const int *incx, const std::complex<float> *beta, std::complex<float> *y,
          const int *incy) {
    cblas_cgemv(CblasColMajor, trans, *m, *n, alpha, a, *lda, x, *incx, beta, y, *incy);
}

template <>
void gemv(CBLAS_TRANSPOSE trans, const int *m, const int *n, const std::complex<double> *alpha,
          const std::complex<double> *a, const int *lda, const std::complex<double> *x,
          const int *incx, const std::complex<double> *beta, std::complex<double> *y,
          const int *incy) {
    cblas_zgemv(CblasColMajor, trans, *m, *n, alpha, a, *lda, x, *incx, beta, y, *incy);
}

template <typename fp>
static void gbmv(CBLAS_TRANSPOSE trans, const int *m, const int *n, int *kl, int *ku,
                 const fp *alpha, const fp *a, const int *lda, const fp *x, const int *incx,
                 const fp *beta, fp *y, const int *incy);

template <>
void gbmv(CBLAS_TRANSPOSE trans, const int *m, const int *n, int *kl, int *ku, const float *alpha,
          const float *a, const int *lda, const float *x, const int *incx, const float *beta,
          float *y, const int *incy) {
    cblas_sgbmv(CblasColMajor, trans, *m, *n, *kl, *ku, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void gbmv(CBLAS_TRANSPOSE trans, const int *m, const int *n, int *kl, int *ku, const double *alpha,
          const double *a, const int *lda, const double *x, const int *incx, const double *beta,
          double *y, const int *incy) {
    cblas_dgbmv(CblasColMajor, trans, *m, *n, *kl, *ku, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void gbmv(CBLAS_TRANSPOSE trans, const int *m, const int *n, int *kl, int *ku,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *x, const int *incx, const std::complex<float> *beta,
          std::complex<float> *y, const int *incy) {
    cblas_cgbmv(CblasColMajor, trans, *m, *n, *kl, *ku, alpha, a, *lda, x, *incx, beta, y, *incy);
}

template <>
void gbmv(CBLAS_TRANSPOSE trans, const int *m, const int *n, int *kl, int *ku,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *x, const int *incx, const std::complex<double> *beta,
          std::complex<double> *y, const int *incy) {
    cblas_zgbmv(CblasColMajor, trans, *m, *n, *kl, *ku, alpha, a, *lda, x, *incx, beta, y, *incy);
}

template <typename fp>
static void ger(const int *m, const int *n, const fp *alpha, const fp *x, const int *incx,
                const fp *y, const int *incy, fp *a, const int *lda);

template <>
void ger(const int *m, const int *n, const float *alpha, const float *x, const int *incx,
         const float *y, const int *incy, float *a, const int *lda) {
    cblas_sger(CblasColMajor, *m, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <>
void ger(const int *m, const int *n, const double *alpha, const double *x, const int *incx,
         const double *y, const int *incy, double *a, const int *lda) {
    cblas_dger(CblasColMajor, *m, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <typename fp>
static void gerc(const int *m, const int *n, const fp *alpha, const fp *x, const int *incx,
                 const fp *y, const int *incy, fp *a, const int *lda);

template <>
void gerc(const int *m, const int *n, const std::complex<float> *alpha,
          const std::complex<float> *x, const int *incx, const std::complex<float> *y,
          const int *incy, std::complex<float> *a, const int *lda) {
    cblas_cgerc(CblasColMajor, *m, *n, alpha, x, *incx, y, *incy, a, *lda);
}

template <>
void gerc(const int *m, const int *n, const std::complex<double> *alpha,
          const std::complex<double> *x, const int *incx, const std::complex<double> *y,
          const int *incy, std::complex<double> *a, const int *lda) {
    cblas_zgerc(CblasColMajor, *m, *n, alpha, x, *incx, y, *incy, a, *lda);
}

template <typename fp>
static void geru(const int *m, const int *n, const fp *alpha, const fp *x, const int *incx,
                 const fp *y, const int *incy, fp *a, const int *lda);

template <>
void geru(const int *m, const int *n, const std::complex<float> *alpha,
          const std::complex<float> *x, const int *incx, const std::complex<float> *y,
          const int *incy, std::complex<float> *a, const int *lda) {
    cblas_cgeru(CblasColMajor, *m, *n, alpha, x, *incx, y, *incy, a, *lda);
}

template <>
void geru(const int *m, const int *n, const std::complex<double> *alpha,
          const std::complex<double> *x, const int *incx, const std::complex<double> *y,
          const int *incy, std::complex<double> *a, const int *lda) {
    cblas_zgeru(CblasColMajor, *m, *n, alpha, x, *incx, y, *incy, a, *lda);
}

template <typename fp>
static void hbmv(CBLAS_UPLO upper_lower, const int *n, const int *k, const fp *alpha, const fp *a,
                 const int *lda, const fp *x, const int *incx, const fp *beta, fp *y,
                 const int *incy);

template <>
void hbmv(CBLAS_UPLO upper_lower, const int *n, const int *k, const std::complex<float> *alpha,
          const std::complex<float> *a, const int *lda, const std::complex<float> *x,
          const int *incx, const std::complex<float> *beta, std::complex<float> *y,
          const int *incy) {
    cblas_chbmv(CblasColMajor, upper_lower, *n, *k, alpha, a, *lda, x, *incx, beta, y, *incy);
}

template <>
void hbmv(CBLAS_UPLO upper_lower, const int *n, const int *k, const std::complex<double> *alpha,
          const std::complex<double> *a, const int *lda, const std::complex<double> *x,
          const int *incx, const std::complex<double> *beta, std::complex<double> *y,
          const int *incy) {
    cblas_zhbmv(CblasColMajor, upper_lower, *n, *k, alpha, a, *lda, x, *incx, beta, y, *incy);
}

template <typename fp>
static void hemv(CBLAS_UPLO upper_lower, const int *n, const fp *alpha, const fp *a, const int *lda,
                 const fp *x, const int *incx, const fp *beta, fp *y, const int *incy);

template <>
void hemv(CBLAS_UPLO upper_lower, const int *n, const std::complex<float> *alpha,
          const std::complex<float> *a, const int *lda, const std::complex<float> *x,
          const int *incx, const std::complex<float> *beta, std::complex<float> *y,
          const int *incy) {
    cblas_chemv(CblasColMajor, upper_lower, *n, alpha, a, *lda, x, *incx, beta, y, *incy);
}

template <>
void hemv(CBLAS_UPLO upper_lower, const int *n, const std::complex<double> *alpha,
          const std::complex<double> *a, const int *lda, const std::complex<double> *x,
          const int *incx, const std::complex<double> *beta, std::complex<double> *y,
          const int *incy) {
    cblas_zhemv(CblasColMajor, upper_lower, *n, alpha, a, *lda, x, *incx, beta, y, *incy);
}

template <typename fp_scalar, typename fp_data>
static void her(CBLAS_UPLO upper_lower, const int *n, const fp_scalar *alpha, const fp_data *x,
                const int *incx, fp_data *a, const int *lda);

template <>
void her(CBLAS_UPLO upper_lower, const int *n, const float *alpha, const std::complex<float> *x,
         const int *incx, std::complex<float> *a, const int *lda) {
    cblas_cher(CblasColMajor, upper_lower, *n, *alpha, x, *incx, a, *lda);
}

template <>
void her(CBLAS_UPLO upper_lower, const int *n, const double *alpha, const std::complex<double> *x,
         const int *incx, std::complex<double> *a, const int *lda) {
    cblas_zher(CblasColMajor, upper_lower, *n, *alpha, x, *incx, a, *lda);
}

template <typename fp>
static void her2(CBLAS_UPLO upper_lower, const int *n, const fp *alpha, const fp *x,
                 const int *incx, const fp *y, const int *incy, fp *a, const int *lda);

template <>
void her2(CBLAS_UPLO upper_lower, const int *n, const std::complex<float> *alpha,
          const std::complex<float> *x, const int *incx, const std::complex<float> *y,
          const int *incy, std::complex<float> *a, const int *lda) {
    cblas_cher2(CblasColMajor, upper_lower, *n, alpha, x, *incx, y, *incy, a, *lda);
}

template <>
void her2(CBLAS_UPLO upper_lower, const int *n, const std::complex<double> *alpha,
          const std::complex<double> *x, const int *incx, const std::complex<double> *y,
          const int *incy, std::complex<double> *a, const int *lda) {
    cblas_zher2(CblasColMajor, upper_lower, *n, alpha, x, *incx, y, *incy, a, *lda);
}

template <typename fp>
static void hpmv(CBLAS_UPLO upper_lower, const int *n, const fp *alpha, const fp *a, const fp *x,
                 const int *incx, const fp *beta, fp *y, const int *incy);

template <>
void hpmv(CBLAS_UPLO upper_lower, const int *n, const std::complex<float> *alpha,
          const std::complex<float> *a, const std::complex<float> *x, const int *incx,
          const std::complex<float> *beta, std::complex<float> *y, const int *incy) {
    cblas_chpmv(CblasColMajor, upper_lower, *n, alpha, a, x, *incx, beta, y, *incy);
}

template <>
void hpmv(CBLAS_UPLO upper_lower, const int *n, const std::complex<double> *alpha,
          const std::complex<double> *a, const std::complex<double> *x, const int *incx,
          const std::complex<double> *beta, std::complex<double> *y, const int *incy) {
    cblas_zhpmv(CblasColMajor, upper_lower, *n, alpha, a, x, *incx, beta, y, *incy);
}

template <typename fp_scalar, typename fp_data>
static void hpr(CBLAS_UPLO upper_lower, const int *n, const fp_scalar *alpha, const fp_data *x,
                const int *incx, fp_data *a);

template <>
void hpr(CBLAS_UPLO upper_lower, const int *n, const float *alpha, const std::complex<float> *x,
         const int *incx, std::complex<float> *a) {
    cblas_chpr(CblasColMajor, upper_lower, *n, *alpha, x, *incx, a);
}

template <>
void hpr(CBLAS_UPLO upper_lower, const int *n, const double *alpha, const std::complex<double> *x,
         const int *incx, std::complex<double> *a) {
    cblas_zhpr(CblasColMajor, upper_lower, *n, *alpha, x, *incx, a);
}

template <typename fp>
static void hpr2(CBLAS_UPLO upper_lower, const int *n, const fp *alpha, const fp *x,
                 const int *incx, const fp *y, const int *incy, fp *a);

template <>
void hpr2(CBLAS_UPLO upper_lower, const int *n, const std::complex<float> *alpha,
          const std::complex<float> *x, const int *incx, const std::complex<float> *y,
          const int *incy, std::complex<float> *a) {
    cblas_chpr2(CblasColMajor, upper_lower, *n, alpha, x, *incx, y, *incy, a);
}

template <>
void hpr2(CBLAS_UPLO upper_lower, const int *n, const std::complex<double> *alpha,
          const std::complex<double> *x, const int *incx, const std::complex<double> *y,
          const int *incy, std::complex<double> *a) {
    cblas_zhpr2(CblasColMajor, upper_lower, *n, alpha, x, *incx, y, *incy, a);
}

template <typename fp>
static void sbmv(CBLAS_UPLO upper_lower, const int *n, const int *k, const fp *alpha, const fp *a,
                 const int *lda, const fp *x, const int *incx, const fp *beta, fp *y,
                 const int *incy);

template <>
void sbmv(CBLAS_UPLO upper_lower, const int *n, const int *k, const float *alpha, const float *a,
          const int *lda, const float *x, const int *incx, const float *beta, float *y,
          const int *incy) {
    cblas_ssbmv(CblasColMajor, upper_lower, *n, *k, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void sbmv(CBLAS_UPLO upper_lower, const int *n, const int *k, const double *alpha, const double *a,
          const int *lda, const double *x, const int *incx, const double *beta, double *y,
          const int *incy) {
    cblas_dsbmv(CblasColMajor, upper_lower, *n, *k, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <typename fp>
static void symv(CBLAS_UPLO upper_lower, const int *n, const fp *alpha, const fp *a, const int *lda,
                 const fp *x, const int *incx, const fp *beta, fp *y, const int *incy);

template <>
void symv(CBLAS_UPLO upper_lower, const int *n, const float *alpha, const float *a, const int *lda,
          const float *x, const int *incx, const float *beta, float *y, const int *incy) {
    cblas_ssymv(CblasColMajor, upper_lower, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void symv(CBLAS_UPLO upper_lower, const int *n, const double *alpha, const double *a,
          const int *lda, const double *x, const int *incx, const double *beta, double *y,
          const int *incy) {
    cblas_dsymv(CblasColMajor, upper_lower, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <typename fp>
static void syr(CBLAS_UPLO upper_lower, const int *n, const fp *alpha, const fp *x, const int *incx,
                fp *a, const int *lda);

template <>
void syr(CBLAS_UPLO upper_lower, const int *n, const float *alpha, const float *x, const int *incx,
         float *a, const int *lda) {
    cblas_ssyr(CblasColMajor, upper_lower, *n, *alpha, x, *incx, a, *lda);
}

template <>
void syr(CBLAS_UPLO upper_lower, const int *n, const double *alpha, const double *x,
         const int *incx, double *a, const int *lda) {
    cblas_dsyr(CblasColMajor, upper_lower, *n, *alpha, x, *incx, a, *lda);
}

template <typename fp>
static void syr2(CBLAS_UPLO upper_lower, const int *n, const fp *alpha, const fp *x,
                 const int *incx, const fp *y, const int *incy, fp *a, const int *lda);

template <>
void syr2(CBLAS_UPLO upper_lower, const int *n, const float *alpha, const float *x, const int *incx,
          const float *y, const int *incy, float *a, const int *lda) {
    cblas_ssyr2(CblasColMajor, upper_lower, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <>
void syr2(CBLAS_UPLO upper_lower, const int *n, const double *alpha, const double *x,
          const int *incx, const double *y, const int *incy, double *a, const int *lda) {
    cblas_dsyr2(CblasColMajor, upper_lower, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <typename fp>
static void spmv(CBLAS_UPLO upper_lower, const int *n, const fp *alpha, const fp *a, const fp *x,
                 const int *incx, const fp *beta, fp *y, const int *incy);

template <>
void spmv(CBLAS_UPLO upper_lower, const int *n, const float *alpha, const float *a, const float *x,
          const int *incx, const float *beta, float *y, const int *incy) {
    cblas_sspmv(CblasColMajor, upper_lower, *n, *alpha, a, x, *incx, *beta, y, *incy);
}

template <>
void spmv(CBLAS_UPLO upper_lower, const int *n, const double *alpha, const double *a,
          const double *x, const int *incx, const double *beta, double *y, const int *incy) {
    cblas_dspmv(CblasColMajor, upper_lower, *n, *alpha, a, x, *incx, *beta, y, *incy);
}

template <typename fp>
static void spr(CBLAS_UPLO upper_lower, const int *n, const fp *alpha, const fp *x, const int *incx,
                fp *a);

template <>
void spr(CBLAS_UPLO upper_lower, const int *n, const float *alpha, const float *x, const int *incx,
         float *a) {
    cblas_sspr(CblasColMajor, upper_lower, *n, *alpha, x, *incx, a);
}

template <>
void spr(CBLAS_UPLO upper_lower, const int *n, const double *alpha, const double *x,
         const int *incx, double *a) {
    cblas_dspr(CblasColMajor, upper_lower, *n, *alpha, x, *incx, a);
}

template <typename fp>
static void spr2(CBLAS_UPLO upper_lower, const int *n, const fp *alpha, const fp *x,
                 const int *incx, const fp *y, const int *incy, fp *a);

template <>
void spr2(CBLAS_UPLO upper_lower, const int *n, const float *alpha, const float *x, const int *incx,
          const float *y, const int *incy, float *a) {
    cblas_sspr2(CblasColMajor, upper_lower, *n, *alpha, x, *incx, y, *incy, a);
}

template <>
void spr2(CBLAS_UPLO upper_lower, const int *n, const double *alpha, const double *x,
          const int *incx, const double *y, const int *incy, double *a) {
    cblas_dspr2(CblasColMajor, upper_lower, *n, *alpha, x, *incx, y, *incy, a);
}

template <typename fp>
static void tbmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
                 const int *k, const fp *a, const int *lda, fp *x, const int *incx);

template <>
void tbmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const int *k, const float *a, const int *lda, float *x, const int *incx) {
    cblas_stbmv(CblasColMajor, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const int *k, const double *a, const int *lda, double *x, const int *incx) {
    cblas_dtbmv(CblasColMajor, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const int *k, const std::complex<float> *a, const int *lda, std::complex<float> *x,
          const int *incx) {
    cblas_ctbmv(CblasColMajor, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const int *k, const std::complex<double> *a, const int *lda, std::complex<double> *x,
          const int *incx) {
    cblas_ztbmv(CblasColMajor, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <typename fp>
static void tbsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
                 const int *k, const fp *a, const int *lda, fp *x, const int *incx);

template <>
void tbsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const int *k, const float *a, const int *lda, float *x, const int *incx) {
    cblas_stbsv(CblasColMajor, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const int *k, const double *a, const int *lda, double *x, const int *incx) {
    cblas_dtbsv(CblasColMajor, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const int *k, const std::complex<float> *a, const int *lda, std::complex<float> *x,
          const int *incx) {
    cblas_ctbsv(CblasColMajor, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const int *k, const std::complex<double> *a, const int *lda, std::complex<double> *x,
          const int *incx) {
    cblas_ztbsv(CblasColMajor, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <typename fp>
static void tpmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
                 const fp *a, fp *x, const int *incx);

template <>
void tpmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const float *a, float *x, const int *incx) {
    cblas_stpmv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const double *a, double *x, const int *incx) {
    cblas_dtpmv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const std::complex<float> *a, std::complex<float> *x, const int *incx) {
    cblas_ctpmv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const std::complex<double> *a, std::complex<double> *x, const int *incx) {
    cblas_ztpmv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <typename fp>
static void tpsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
                 const fp *a, fp *x, const int *incx);

template <>
void tpsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const float *a, float *x, const int *incx) {
    cblas_stpsv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const double *a, double *x, const int *incx) {
    cblas_dtpsv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const std::complex<float> *a, std::complex<float> *x, const int *incx) {
    cblas_ctpsv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const std::complex<double> *a, std::complex<double> *x, const int *incx) {
    cblas_ztpsv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <typename fp>
static void trmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
                 const fp *a, const int *lda, fp *x, const int *incx);

template <>
void trmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const float *a, const int *lda, float *x, const int *incx) {
    cblas_strmv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const double *a, const int *lda, double *x, const int *incx) {
    cblas_dtrmv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const std::complex<float> *a, const int *lda, std::complex<float> *x, const int *incx) {
    cblas_ctrmv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trmv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const std::complex<double> *a, const int *lda, std::complex<double> *x, const int *incx) {
    cblas_ztrmv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <typename fp>
static void trsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
                 const fp *a, const int *lda, fp *x, const int *incx);

template <>
void trsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const float *a, const int *lda, float *x, const int *incx) {
    cblas_strsv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const double *a, const int *lda, double *x, const int *incx) {
    cblas_dtrsv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const std::complex<float> *a, const int *lda, std::complex<float> *x, const int *incx) {
    cblas_ctrsv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trsv(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int *n,
          const std::complex<double> *a, const int *lda, std::complex<double> *x, const int *incx) {
    cblas_ztrsv(CblasColMajor, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

/* Level 1 */

template <typename fp_data, typename fp_res>
static fp_res asum(const int *n, const fp_data *x, const int *incx);

template <>
float asum(const int *n, const float *x, const int *incx) {
    return cblas_sasum(*n, x, *incx);
}

template <>
double asum(const int *n, const double *x, const int *incx) {
    return cblas_dasum(*n, x, *incx);
}

template <>
float asum(const int *n, const std::complex<float> *x, const int *incx) {
    return cblas_scasum(*n, x, *incx);
}

template <>
double asum(const int *n, const std::complex<double> *x, const int *incx) {
    return cblas_dzasum(*n, x, *incx);
}

template <typename fp>
static void axpy(const int *n, const fp *alpha, const fp *x, const int *incx, fp *y,
                 const int *incy);

template <>
void axpy(const int *n, const float *alpha, const float *x, const int *incx, float *y,
          const int *incy) {
    cblas_saxpy(*n, *alpha, x, *incx, y, *incy);
}

template <>
void axpy(const int *n, const double *alpha, const double *x, const int *incx, double *y,
          const int *incy) {
    cblas_daxpy(*n, *alpha, x, *incx, y, *incy);
}

template <>
void axpy(const int *n, const std::complex<float> *alpha, const std::complex<float> *x,
          const int *incx, std::complex<float> *y, const int *incy) {
    cblas_caxpy(*n, alpha, x, *incx, y, *incy);
}

template <>
void axpy(const int *n, const std::complex<double> *alpha, const std::complex<double> *x,
          const int *incx, std::complex<double> *y, const int *incy) {
    cblas_zaxpy(*n, alpha, x, *incx, y, *incy);
}

template <typename fp>
static void copy(const int *n, const fp *x, const int *incx, fp *y, const int *incy);

template <>
void copy(const int *n, const float *x, const int *incx, float *y, const int *incy) {
    cblas_scopy(*n, x, *incx, y, *incy);
}
template <>
void copy(const int *n, const double *x, const int *incx, double *y, const int *incy) {
    cblas_dcopy(*n, x, *incx, y, *incy);
}
template <>
void copy(const int *n, const std::complex<float> *x, const int *incx, std::complex<float> *y,
          const int *incy) {
    cblas_ccopy(*n, x, *incx, y, *incy);
}
template <>
void copy(const int *n, const std::complex<double> *x, const int *incx, std::complex<double> *y,
          const int *incy) {
    cblas_zcopy(*n, x, *incx, y, *incy);
}

template <typename fp, typename fp_res>
static fp_res dot(const int *n, const fp *x, const int *incx, const fp *y, const int *incy);

template <>
float dot(const int *n, const float *x, const int *incx, const float *y, const int *incy) {
    return cblas_sdot(*n, x, *incx, y, *incy);
}

template <>
double dot(const int *n, const double *x, const int *incx, const double *y, const int *incy) {
    return cblas_ddot(*n, x, *incx, y, *incy);
}

template <>
double dot(const int *n, const float *x, const int *incx, const float *y, const int *incy) {
    return cblas_dsdot(*n, x, *incx, y, *incy);
}

static float sdsdot(const int *n, const float *sb, const float *x, const int *incx, const float *y,
                    const int *incy) {
    return cblas_sdsdot(*n, *sb, x, *incx, y, *incy);
}

template <typename fp, typename fp_res>
static fp_res nrm2(const int *n, const fp *x, const int *incx);

template <>
float nrm2(const int *n, const float *x, const int *incx) {
    return cblas_snrm2(*n, x, *incx);
}

template <>
double nrm2(const int *n, const double *x, const int *incx) {
    return cblas_dnrm2(*n, x, *incx);
}

template <>
float nrm2(const int *n, const std::complex<float> *x, const int *incx) {
    return cblas_scnrm2(*n, x, *incx);
}

template <>
double nrm2(const int *n, const std::complex<double> *x, const int *incx) {
    return cblas_dznrm2(*n, x, *incx);
}

template <typename fp, typename fp_scalar>
static void rot(const int *n, fp *x, const int *incx, fp *y, const int *incy, const fp_scalar *c,
                const fp_scalar *s);

template <>
void rot(const int *n, float *x, const int *incx, float *y, const int *incy, const float *c,
         const float *s) {
    cblas_srot(*n, x, *incx, y, *incy, *c, *s);
}

template <>
void rot(const int *n, double *x, const int *incx, double *y, const int *incy, const double *c,
         const double *s) {
    cblas_drot(*n, x, *incx, y, *incy, *c, *s);
}

template <>
void rot(const int *n, std::complex<float> *x, const int *incx, std::complex<float> *y,
         const int *incy, const float *c, const float *s) {
    csrot_(n, (void *)x, incx, (void *)y, incy, c, s);
}

template <>
void rot(const int *n, std::complex<double> *x, const int *incx, std::complex<double> *y,
         const int *incy, const double *c, const double *s) {
    zdrot_(n, (void *)x, incx, (void *)y, incy, c, s);
}

template <typename fp, typename fp_c>
static void rotg(fp *a, fp *b, fp_c *c, fp *s);

template <>
void rotg(float *a, float *b, float *c, float *s) {
    cblas_srotg(a, b, c, s);
}

template <>
void rotg(double *a, double *b, double *c, double *s) {
    cblas_drotg(a, b, c, s);
}

template <>
void rotg(std::complex<float> *a, std::complex<float> *b, float *c, std::complex<float> *s) {
    crotg_((void *)a, (void *)b, c, (void *)s);
}

template <>
void rotg(std::complex<double> *a, std::complex<double> *b, double *c, std::complex<double> *s) {
    zrotg_((void *)a, (void *)b, c, (void *)s);
}

template <typename fp>
static void rotm(const int *n, fp *x, const int *incx, fp *y, const int *incy, const fp *param);

template <>
void rotm(const int *n, float *x, const int *incx, float *y, const int *incy, const float *param) {
    cblas_srotm(*n, x, *incx, y, *incy, param);
}

template <>
void rotm(const int *n, double *x, const int *incx, double *y, const int *incy,
          const double *param) {
    cblas_drotm(*n, x, *incx, y, *incy, param);
}

template <typename fp>
static void rotmg(fp *d1, fp *d2, fp *x1, fp *y1, fp *param);

template <>
void rotmg(float *d1, float *d2, float *x1, float *y1, float *param) {
    cblas_srotmg(d1, d2, x1, *y1, param);
}

template <>
void rotmg(double *d1, double *d2, double *x1, double *y1, double *param) {
    cblas_drotmg(d1, d2, x1, *y1, param);
}

template <typename fp_scalar, typename fp_data>
static void scal(const int *n, const fp_scalar *alpha, fp_data *x, const int *incx);

template <>
void scal(const int *n, const float *alpha, float *x, const int *incx) {
    cblas_sscal(*n, *alpha, x, *incx);
}
template <>
void scal(const int *n, const double *alpha, double *x, const int *incx) {
    cblas_dscal(*n, *alpha, x, *incx);
}
template <>
void scal(const int *n, const std::complex<float> *alpha, std::complex<float> *x, const int *incx) {
    cblas_cscal(*n, alpha, x, *incx);
}
template <>
void scal(const int *n, const std::complex<double> *alpha, std::complex<double> *x,
          const int *incx) {
    cblas_zscal(*n, alpha, x, *incx);
}
template <>
void scal(const int *n, const float *alpha, std::complex<float> *x, const int *incx) {
    cblas_csscal(*n, *alpha, x, *incx);
}
template <>
void scal(const int *n, const double *alpha, std::complex<double> *x, const int *incx) {
    cblas_zdscal(*n, *alpha, x, *incx);
}

template <typename fp>
static void swap(const int *n, fp *x, const int *incx, fp *y, const int *incy);

template <>
void swap(const int *n, float *x, const int *incx, float *y, const int *incy) {
    cblas_sswap(*n, x, *incx, y, *incy);
}

template <>
void swap(const int *n, double *x, const int *incx, double *y, const int *incy) {
    cblas_dswap(*n, x, *incx, y, *incy);
}

template <>
void swap(const int *n, std::complex<float> *x, const int *incx, std::complex<float> *y,
          const int *incy) {
    cblas_cswap(*n, x, *incx, y, *incy);
}

template <>
void swap(const int *n, std::complex<double> *x, const int *incx, std::complex<double> *y,
          const int *incy) {
    cblas_zswap(*n, x, *incx, y, *incy);
}

template <typename fp>
static void dotc(fp *pres, const int *n, const fp *x, const int *incx, const fp *y,
                 const int *incy);

template <>
void dotc(std::complex<float> *pres, const int *n, const std::complex<float> *x, const int *incx,
          const std::complex<float> *y, const int *incy) {
    cblas_cdotc_sub(*n, x, *incx, y, *incy, pres);
}

template <>
void dotc(std::complex<double> *pres, const int *n, const std::complex<double> *x, const int *incx,
          const std::complex<double> *y, const int *incy) {
    cblas_zdotc_sub(*n, x, *incx, y, *incy, pres);
}

template <typename fp>
static void dotu(fp *pres, const int *n, const fp *x, const int *incx, const fp *y,
                 const int *incy);

template <>
void dotu(std::complex<float> *pres, const int *n, const std::complex<float> *x, const int *incx,
          const std::complex<float> *y, const int *incy) {
    cblas_cdotu_sub(*n, x, *incx, y, *incy, pres);
}

template <>
void dotu(std::complex<double> *pres, const int *n, const std::complex<double> *x, const int *incx,
          const std::complex<double> *y, const int *incy) {
    cblas_zdotu_sub(*n, x, *incx, y, *incy, pres);
}

template <typename fp>
static int iamax(const int *n, const fp *x, const int *incx);

template <>
int iamax(const int *n, const float *x, const int *incx) {
    return cblas_isamax(*n, x, *incx);
}

template <>
int iamax(const int *n, const double *x, const int *incx) {
    return cblas_idamax(*n, x, *incx);
}

template <>
int iamax(const int *n, const std::complex<float> *x, const int *incx) {
    return cblas_icamax(*n, x, *incx);
}

template <>
int iamax(const int *n, const std::complex<double> *x, const int *incx) {
    return cblas_izamax(*n, x, *incx);
}

inline float abs_val(float val) {
    return std::abs(val);
}

inline double abs_val(double val) {
    return std::abs(val);
}

inline float abs_val(std::complex<float> val) {
    return std::abs(val.real()) + std::abs(val.imag());
}

inline double abs_val(std::complex<double> val) {
    return std::abs(val.real()) + std::abs(val.imag());
}

template <typename fp>
static int iamin(const int *n, const fp *x, const int *incx);

template <>
int iamin(const int *n, const float *x, const int *incx) {
    if (*n < 1 || *incx < 1) {
        return 0;
    }
    int min_idx  = 0;
    auto min_val = abs_val(x[0]);

    for (int logical_i = 0; logical_i < *n; ++logical_i) {
        int i             = logical_i * std::abs(*incx);
        auto curr_val     = abs_val(x[i]);
        bool is_first_nan = std::isnan(curr_val) && !std::isnan(min_val);
        if (is_first_nan || curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

template <>
int iamin(const int *n, const double *x, const int *incx) {
    if (*n < 1 || *incx < 1) {
        return 0;
    }
    int min_idx  = 0;
    auto min_val = abs_val(x[0]);

    for (int logical_i = 0; logical_i < *n; ++logical_i) {
        int i             = logical_i * std::abs(*incx);
        auto curr_val     = abs_val(x[i]);
        bool is_first_nan = std::isnan(curr_val) && !std::isnan(min_val);
        if (is_first_nan || curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

template <>
int iamin(const int *n, const std::complex<float> *x, const int *incx) {
    if (*n < 1 || *incx < 1) {
        return 0;
    }
    int min_idx  = 0;
    auto min_val = abs_val(x[0]);

    for (int logical_i = 0; logical_i < *n; ++logical_i) {
        int i             = logical_i * std::abs(*incx);
        auto curr_val     = abs_val(x[i]);
        bool is_first_nan = std::isnan(curr_val) && !std::isnan(min_val);
        if (is_first_nan || curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

template <>
int iamin(const int *n, const std::complex<double> *x, const int *incx) {
    if (*n < 1 || *incx < 1) {
        return 0;
    }
    int min_idx  = 0;
    auto min_val = abs_val(x[0]);

    for (int logical_i = 0; logical_i < *n; ++logical_i) {
        int i             = logical_i * std::abs(*incx);
        auto curr_val     = abs_val(x[i]);
        bool is_first_nan = std::isnan(curr_val) && !std::isnan(min_val);
        if (is_first_nan || curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

/* Extensions */

template <typename fp>
static void gemm_ext(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n,
                     const int *k, const fp *alpha, const fp *a, const int *lda, const fp *b,
                     const int *ldb, const fp *beta, fp *c, const int *ldc);

template <>
void gemm_ext(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n,
              const int *k, const half *alpha, const half *a, const int *lda, const half *b,
              const int *ldb, const half *beta, half *c, const int *ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizea, sizeb, sizec;
    const float alphaf = *alpha;
    const float betaf  = *beta;
    sizea              = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
    sizeb              = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
    sizec              = *ldc * *n;
    float *af          = (float *)aligned_alloc(64, sizeof(float) * sizea);
    float *bf          = (float *)aligned_alloc(64, sizeof(float) * sizeb);
    float *cf          = (float *)aligned_alloc(64, sizeof(float) * sizec);
    copy_mat(a, transa, *m, *k, *lda, af);
    copy_mat(b, transb, *k, *n, *ldb, bf);
    copy_mat(c, CblasNoTrans, *m, *n, *ldc, cf);
    cblas_sgemm(CblasColMajor, transa, transb, *m, *n, *k, alphaf, af, *lda, bf, *ldb, betaf, cf,
                *ldc);
    copy_mat(cf, CblasNoTrans, *m, *n, *ldc, c);
    free(af);
    free(bf);
    free(cf);
}

template <>
void gemm_ext(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n,
              const int *k, const float *alpha, const float *a, const int *lda, const float *b,
              const int *ldb, const float *beta, float *c, const int *ldc) {
    cblas_sgemm(CblasColMajor, transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c,
                *ldc);
}

template <>
void gemm_ext(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n,
              const int *k, const double *alpha, const double *a, const int *lda, const double *b,
              const int *ldb, const double *beta, double *c, const int *ldc) {
    cblas_dgemm(CblasColMajor, transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c,
                *ldc);
}

template <>
void gemm_ext(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n,
              const int *k, const std::complex<float> *alpha, const std::complex<float> *a,
              const int *lda, const std::complex<float> *b, const int *ldb,
              const std::complex<float> *beta, std::complex<float> *c, const int *ldc) {
    cblas_cgemm(CblasColMajor, transa, transb, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
}

template <>
void gemm_ext(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n,
              const int *k, const std::complex<double> *alpha, const std::complex<double> *a,
              const int *lda, const std::complex<double> *b, const int *ldb,
              const std::complex<double> *beta, std::complex<double> *c, const int *ldc) {
    cblas_zgemm(CblasColMajor, transa, transb, *m, *n, *k, alpha, a, *lda, b, *ldb, beta, c, *ldc);
}

template <typename fpa, typename fpc>
static void gemm_ext(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n,
                     const int *k, const fpc *alpha, const fpa *a, const int *lda, const fpa *b,
                     const int *ldb, const fpc *beta, fpc *c, const int *ldc);

template <>
void gemm_ext(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m, const int *n,
              const int *k, const float *alpha, const half *a, const int *lda, const half *b,
              const int *ldb, const float *beta, float *c, const int *ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizea, sizeb;
    sizea     = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
    sizeb     = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
    float *af = (float *)aligned_alloc(64, sizeof(float) * sizea);
    float *bf = (float *)aligned_alloc(64, sizeof(float) * sizeb);
    copy_mat(a, transa, *m, *k, *lda, af);
    copy_mat(b, transb, *k, *n, *ldb, bf);
    cblas_sgemm(CblasColMajor, transa, transb, *m, *n, *k, *alpha, af, *lda, bf, *ldb, *beta, c,
                *ldc);
    free(af);
    free(bf);
}

template <typename fps, typename fpa, typename fpb, typename fpc>
static void gemm_ext(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc,
                     const int *m, const int *n, const int *k, const fps *alpha, const fpa *a,
                     const int *lda, const fpa *ao, const fpb *b, const int *ldb, const fpb *bo,
                     const fps *beta, fpc *c, const int *ldc, const fpc *co);

template <>
void gemm_ext(CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_OFFSET offsetc, const int *m,
              const int *n, const int *k, const float *alpha, const int8_t *a, const int *lda,
              const int8_t *ao, const uint8_t *b, const int *ldb, const uint8_t *bo,
              const float *beta, int32_t *c, const int *ldc, const int32_t *co) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizea, sizeb, sizec;
    sizea         = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
    sizeb         = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
    sizec         = *ldc * *n;
    double *ad    = (double *)aligned_alloc(64, sizeof(double) * sizea);
    double *bd    = (double *)aligned_alloc(64, sizeof(double) * sizeb);
    double *cd    = (double *)aligned_alloc(64, sizeof(double) * sizec);
    double alphad = *alpha;
    double betad  = *beta;
    double aod    = *ao;
    double bod    = *bo;
    copy_mat(a, transa, *m, *k, *lda, aod, ad);
    copy_mat(b, transb, *k, *n, *ldb, bod, bd);
    copy_mat(c, CblasNoTrans, *m, *n, *ldc, 0.0, cd);
    cblas_dgemm(CblasColMajor, transa, transb, *m, *n, *k, alphad, ad, *lda, bd, *ldb, betad, cd,
                *ldc);
    copy_mat(cd, *m, *n, *ldc, offsetc, co, c);
    free(ad);
    free(bd);
    free(cd);
}

template <typename fp>
static void gemmt(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                  const int *n, const int *k, const fp *alpha, const fp *a, const int *lda,
                  const fp *b, const int *ldb, const fp *beta, fp *c, const int *ldc);

template <>
void gemmt(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *n,
           const int *k, const float *alpha, const float *a, const int *lda, const float *b,
           const int *ldb, const float *beta, float *c, const int *ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizec;
    sizec     = *ldc * *n;
    float *cf = (float *)aligned_alloc(64, sizeof(float) * sizec);
    update_c(c, upper_lower, *n, *n, *ldc, cf);
    cblas_sgemm(CblasColMajor, transa, transb, *n, *n, *k, *alpha, a, *lda, b, *ldb, *beta, cf,
                *ldc);
    update_c(cf, upper_lower, *n, *n, *ldc, c);
}

template <>
void gemmt(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *n,
           const int *k, const double *alpha, const double *a, const int *lda, const double *b,
           const int *ldb, const double *beta, double *c, const int *ldc) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizec;
    sizec      = *ldc * *n;
    double *cf = (double *)aligned_alloc(64, sizeof(double) * sizec);
    update_c(c, upper_lower, *n, *n, *ldc, cf);
    cblas_dgemm(CblasColMajor, transa, transb, *n, *n, *k, *alpha, a, *lda, b, *ldb, *beta, cf,
                *ldc);
    update_c(cf, upper_lower, *n, *n, *ldc, c);
}

template <>
void gemmt(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *n,
           const int *k, const std::complex<float> *alpha, const std::complex<float> *a,
           const int *lda, const std::complex<float> *b, const int *ldb,
           const std::complex<float> *beta, std::complex<float> *c, const int *ldc) {
    // Not supported in NETLIB. CGEMM is used as reference.
    int sizec;
    sizec = *ldc * *n;
    std::complex<float> *cf =
        (std::complex<float> *)aligned_alloc(64, sizeof(std::complex<float>) * sizec);
    update_c(c, upper_lower, *n, *n, *ldc, cf);
    cblas_cgemm(CblasColMajor, transa, transb, *n, *n, *k, alpha, a, *lda, b, *ldb, beta, cf, *ldc);
    update_c(cf, upper_lower, *n, *n, *ldc, c);
}

template <>
void gemmt(CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *n,
           const int *k, const std::complex<double> *alpha, const std::complex<double> *a,
           const int *lda, const std::complex<double> *b, const int *ldb,
           const std::complex<double> *beta, std::complex<double> *c, const int *ldc) {
    // Not supported in NETLIB. ZGEMM is used as reference.
    int sizec;
    sizec = *ldc * *n;
    std::complex<double> *cf =
        (std::complex<double> *)aligned_alloc(64, sizeof(std::complex<double>) * sizec);
    update_c(c, upper_lower, *n, *n, *ldc, cf);
    cblas_zgemm(CblasColMajor, transa, transb, *n, *n, *k, alpha, a, *lda, b, *ldb, beta, cf, *ldc);
    update_c(cf, upper_lower, *n, *n, *ldc, c);
}

#endif /* header guard */
