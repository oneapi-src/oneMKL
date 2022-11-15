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

#ifndef _REFERENCE_BLAS_TEMPLATES_HPP__
#define _REFERENCE_BLAS_TEMPLATES_HPP__

#include <stdlib.h>
#include <complex>
#include "oneapi/mkl/exceptions.hpp"
#include "cblas.h"
#include "test_helper.hpp"

#ifdef __linux__
#include <dlfcn.h>
#define LIB_TYPE                void *
#define GET_LIB_HANDLE(libname) dlopen((libname), RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND)
#define GET_FUNC(lib, fn)       dlsym(lib, (fn))
#elif defined(_WIN64)
#include <windows.h>
#define LIB_TYPE                HINSTANCE
#define GET_LIB_HANDLE(libname) LoadLibrary(libname)
#define GET_FUNC(lib, fn)       GetProcAddress((lib), (fn))
#endif

extern "C" {
static LIB_TYPE h_libblas = NULL;
static LIB_TYPE blas_library() {
    if (h_libblas == NULL) {
        h_libblas = GET_LIB_HANDLE(REF_BLAS_LIBNAME);
        if (h_libblas == NULL) {
            throw oneapi::mkl::library_not_found("BLAS", "blas_library()",
                                                 "BLAS library not found.");
        }
    }
    return h_libblas;
}

static LIB_TYPE h_libcblas = NULL;
static LIB_TYPE cblas_library() {
    if (h_libcblas == NULL) {
        h_libcblas = GET_LIB_HANDLE(REF_CBLAS_LIBNAME);
        if (h_libcblas == NULL) {
            throw oneapi::mkl::library_not_found("CBLAS", "cblas_library()",
                                                 "CBLAS library not found.");
        }
    }
    return h_libcblas;
}

/* Level 3 */

static void (*cblas_sgemm_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                             const int m, const int n, const int k, const float alpha,
                             const float *a, const int lda, const float *b, const int ldb,
                             const float beta, float *c, const int ldc);
static void (*cblas_dgemm_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                             const int m, const int n, const int k, const double alpha,
                             const double *a, const int lda, const double *b, const int ldb,
                             const double beta, double *c, const int ldc);
static void (*cblas_cgemm_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                             const int m, const int n, const int k, const void *alpha,
                             const void *a, const int lda, const void *b, const int ldb,
                             const void *beta, void *c, const int ldc);
static void (*cblas_zgemm_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                             const int m, const int n, const int k, const void *alpha,
                             const void *a, const int lda, const void *b, const int ldb,
                             const void *beta, void *c, const int ldc);
static void (*cblas_ssymm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const float alpha, const float *a,
                             const int lda, const float *b, const int ldb, const float beta,
                             float *c, const int ldc);
static void (*cblas_dsymm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const double alpha, const double *a,
                             const int lda, const double *b, const int ldb, const double beta,
                             double *c, const int ldc);
static void (*cblas_csymm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const void *alpha, const void *a,
                             const int lda, const void *b, const int ldb, const void *beta, void *c,
                             const int ldc);
static void (*cblas_zsymm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const void *alpha, const void *a,
                             const int lda, const void *b, const int ldb, const void *beta, void *c,
                             const int ldc);
static void (*cblas_ssyrk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const float alpha, const float *a,
                             const int lda, const float beta, float *c, const int ldc);
static void (*cblas_dsyrk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const double alpha, const double *a,
                             const int lda, const double beta, double *c, const int ldc);
static void (*cblas_csyrk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const void *alpha, const void *a,
                             const int lda, const void *beta, void *c, const int ldc);
static void (*cblas_zsyrk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const void *alpha, const void *a,
                             const int lda, const void *beta, void *c, const int ldc);
static void (*cblas_chemm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const void *alpha, const void *a,
                             const int lda, const void *b, const int ldb, const void *beta, void *c,
                             const int ldc);
static void (*cblas_zhemm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                             const int m, const int n, const void *alpha, const void *a,
                             const int lda, const void *b, const int ldb, const void *beta, void *c,
                             const int ldc);
static void (*cblas_cherk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const float alpha, const void *a,
                             const int lda, const float beta, void *c, const int ldc);
static void (*cblas_zherk_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                             const int n, const int k, const double alpha, const void *a,
                             const int lda, const double beta, void *c, const int ldc);
static void (*cblas_ssyr2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const float alpha, const float *a,
                              const int lda, const float *b, const int ldb, const float beta,
                              float *c, const int ldc);
static void (*cblas_dsyr2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const double alpha, const double *a,
                              const int lda, const double *b, const int ldb, const double beta,
                              double *c, const int ldc);
static void (*cblas_csyr2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const void *alpha, const void *a,
                              const int lda, const void *b, const int ldb, const void *beta,
                              void *c, const int ldc);
static void (*cblas_zsyr2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const void *alpha, const void *a,
                              const int lda, const void *b, const int ldb, const void *beta,
                              void *c, const int ldc);
static void (*cblas_cher2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const void *alpha, const void *a,
                              const int lda, const void *b, const int ldb, const float beta,
                              void *c, const int ldc);
static void (*cblas_zher2k_p)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                              const int n, const int k, const void *alpha, const void *a,
                              const int lda, const void *b, const int ldb, const double beta,
                              void *c, const int ldc);
static void (*cblas_strmm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const float alpha, const float *a, const int lda, float *b,
                             const int ldb);
static void (*cblas_dtrmm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const double alpha, const double *a, const int lda, double *b,
                             const int ldb);
static void (*cblas_ctrmm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const void *alpha, const void *a, const int lda, void *b,
                             const int ldb);
static void (*cblas_ztrmm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const void *alpha, const void *a, const int lda, void *b,
                             const int ldb);
static void (*cblas_strsm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const float alpha, const float *a, const int lda, float *b,
                             const int ldb);
static void (*cblas_dtrsm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const double alpha, const double *a, const int lda, double *b,
                             const int ldb);
static void (*cblas_ctrsm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const void *alpha, const void *a, const int lda, void *b,
                             const int ldb);
static void (*cblas_ztrsm_p)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                             CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                             const void *alpha, const void *a, const int lda, void *b,
                             const int ldb);

static void cblas_sgemm_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                                const int m, const int n, const int k, const float alpha,
                                const float *a, const int lda, const float *b, const int ldb,
                                const float beta, float *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_sgemm_p == NULL)
            cblas_sgemm_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                          const int m, const int n, const int k, const float alpha, const float *a,
                          const int lda, const float *b, const int ldb, const float beta, float *c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_sgemm");
        if (cblas_sgemm_p != NULL)
            cblas_sgemm_p(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_dgemm_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                                const int m, const int n, const int k, const double alpha,
                                const double *a, const int lda, const double *b, const int ldb,
                                const double beta, double *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_dgemm_p == NULL)
            cblas_dgemm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa,
                                      CBLAS_TRANSPOSE transb, const int m, const int n, const int k,
                                      const double alpha, const double *a, const int lda,
                                      const double *b, const int ldb, const double beta, double *c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_dgemm");
        if (cblas_dgemm_p != NULL)
            cblas_dgemm_p(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_cgemm_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                                const int m, const int n, const int k, const void *alpha,
                                const void *a, const int lda, const void *b, const int ldb,
                                const void *beta, void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_cgemm_p == NULL)
            cblas_cgemm_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                          const int m, const int n, const int k, const void *alpha, const void *a,
                          const int lda, const void *b, const int ldb, const void *beta, void *c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_cgemm");
        if (cblas_cgemm_p != NULL)
            cblas_cgemm_p(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_zgemm_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                                const int m, const int n, const int k, const void *alpha,
                                const void *a, const int lda, const void *b, const int ldb,
                                const void *beta, void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zgemm_p == NULL)
            cblas_zgemm_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                          const int m, const int n, const int k, const void *alpha, const void *a,
                          const int lda, const void *b, const int ldb, const void *beta, void *c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_zgemm");
        if (cblas_zgemm_p != NULL)
            cblas_zgemm_p(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_ssymm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const float alpha, const float *a,
                                const int lda, const float *b, const int ldb, const float beta,
                                float *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_ssymm_p == NULL)
            cblas_ssymm_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int m,
                          const int n, const float alpha, const float *a, const int lda,
                          const float *b, const int ldb, const float beta, float *c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_ssymm");
        if (cblas_ssymm_p != NULL)
            cblas_ssymm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_dsymm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const double alpha, const double *a,
                                const int lda, const double *b, const int ldb, const double beta,
                                double *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_dsymm_p == NULL)
            cblas_dsymm_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int m,
                          const int n, const double alpha, const double *a, const int lda,
                          const double *b, const int ldb, const double beta, double *c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_dsymm");
        if (cblas_dsymm_p != NULL)
            cblas_dsymm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_csymm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const void *alpha, const void *a,
                                const int lda, const void *b, const int ldb, const void *beta,
                                void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_csymm_p == NULL)
            cblas_csymm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                      const int m, const int n, const void *alpha, const void *a,
                                      const int lda, const void *b, const int ldb, const void *beta,
                                      void *c, const int ldc))GET_FUNC(h_libcblas, "cblas_csymm");
        if (cblas_csymm_p != NULL)
            cblas_csymm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_zsymm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const void *alpha, const void *a,
                                const int lda, const void *b, const int ldb, const void *beta,
                                void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zsymm_p == NULL)
            cblas_zsymm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                      const int m, const int n, const void *alpha, const void *a,
                                      const int lda, const void *b, const int ldb, const void *beta,
                                      void *c, const int ldc))GET_FUNC(h_libcblas, "cblas_zsymm");
        if (cblas_zsymm_p != NULL)
            cblas_zsymm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_ssyrk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const float alpha, const float *a,
                                const int lda, const float beta, float *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_ssyrk_p == NULL)
            cblas_ssyrk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const float alpha, const float *a,
                                      const int lda, const float beta, float *c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_ssyrk");
        if (cblas_ssyrk_p != NULL)
            cblas_ssyrk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void cblas_dsyrk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const double alpha, const double *a,
                                const int lda, const double beta, double *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_dsyrk_p == NULL)
            cblas_dsyrk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const double alpha, const double *a,
                                      const int lda, const double beta, double *c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_dsyrk");
        if (cblas_dsyrk_p != NULL)
            cblas_dsyrk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void cblas_csyrk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const void *alpha, const void *a,
                                const int lda, const void *beta, void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_csyrk_p == NULL)
            cblas_csyrk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const void *alpha, const void *a,
                                      const int lda, const void *beta, void *c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_csyrk");
        if (cblas_csyrk_p != NULL)
            cblas_csyrk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void cblas_zsyrk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const void *alpha, const void *a,
                                const int lda, const void *beta, void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zsyrk_p == NULL)
            cblas_zsyrk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const void *alpha, const void *a,
                                      const int lda, const void *beta, void *c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_zsyrk");
        if (cblas_zsyrk_p != NULL)
            cblas_zsyrk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void cblas_chemm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const void *alpha, const void *a,
                                const int lda, const void *b, const int ldb, const void *beta,
                                void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_chemm_p == NULL)
            cblas_chemm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                      const int m, const int n, const void *alpha, const void *a,
                                      const int lda, const void *b, const int ldb, const void *beta,
                                      void *c, const int ldc))GET_FUNC(h_libcblas, "cblas_chemm");
        if (cblas_chemm_p != NULL)
            cblas_chemm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_zhemm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                const int m, const int n, const void *alpha, const void *a,
                                const int lda, const void *b, const int ldb, const void *beta,
                                void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zhemm_p == NULL)
            cblas_zhemm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo,
                                      const int m, const int n, const void *alpha, const void *a,
                                      const int lda, const void *b, const int ldb, const void *beta,
                                      void *c, const int ldc))GET_FUNC(h_libcblas, "cblas_zhemm");
        if (cblas_zhemm_p != NULL)
            cblas_zhemm_p(layout, left_right, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_cherk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const float alpha, const void *a,
                                const int lda, const float beta, void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_cherk_p == NULL)
            cblas_cherk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const float alpha, const void *a,
                                      const int lda, const float beta, void *c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_cherk");
        if (cblas_cherk_p != NULL)
            cblas_cherk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void cblas_zherk_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                const int n, const int k, const double alpha, const void *a,
                                const int lda, const double beta, void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zherk_p == NULL)
            cblas_zherk_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                      const int n, const int k, const double alpha, const void *a,
                                      const int lda, const double beta, void *c,
                                      const int ldc))GET_FUNC(h_libcblas, "cblas_zherk");
        if (cblas_zherk_p != NULL)
            cblas_zherk_p(layout, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
}

static void cblas_ssyr2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const float alpha, const float *a,
                                 const int lda, const float *b, const int ldb, const float beta,
                                 float *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_ssyr2k_p == NULL)
            cblas_ssyr2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const float alpha, const float *a, const int lda,
                          const float *b, const int ldb, const float beta, float *c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_ssyr2k");
        if (cblas_ssyr2k_p != NULL)
            cblas_ssyr2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_dsyr2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const double alpha, const double *a,
                                 const int lda, const double *b, const int ldb, const double beta,
                                 double *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_dsyr2k_p == NULL)
            cblas_dsyr2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const double alpha, const double *a, const int lda,
                          const double *b, const int ldb, const double beta, double *c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_dsyr2k");
        if (cblas_dsyr2k_p != NULL)
            cblas_dsyr2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_csyr2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const void *alpha, const void *a,
                                 const int lda, const void *b, const int ldb, const void *beta,
                                 void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_csyr2k_p == NULL)
            cblas_csyr2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const void *alpha, const void *a, const int lda,
                          const void *b, const int ldb, const void *beta, void *c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_csyr2k");
        if (cblas_csyr2k_p != NULL)
            cblas_csyr2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_zsyr2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const void *alpha, const void *a,
                                 const int lda, const void *b, const int ldb, const void *beta,
                                 void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zsyr2k_p == NULL)
            cblas_zsyr2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const void *alpha, const void *a, const int lda,
                          const void *b, const int ldb, const void *beta, void *c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_zsyr2k");
        if (cblas_zsyr2k_p != NULL)
            cblas_zsyr2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_cher2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const void *alpha, const void *a,
                                 const int lda, const void *b, const int ldb, const float beta,
                                 void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_cher2k_p == NULL)
            cblas_cher2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const void *alpha, const void *a, const int lda,
                          const void *b, const int ldb, const float beta, void *c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_cher2k");
        if (cblas_cher2k_p != NULL)
            cblas_cher2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_zher2k_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans,
                                 const int n, const int k, const void *alpha, const void *a,
                                 const int lda, const void *b, const int ldb, const double beta,
                                 void *c, const int ldc) {
    if (cblas_library() != NULL) {
        if (cblas_zher2k_p == NULL)
            cblas_zher2k_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int n,
                          const int k, const void *alpha, const void *a, const int lda,
                          const void *b, const int ldb, const double beta, void *c,
                          const int ldc))GET_FUNC(h_libcblas, "cblas_zher2k");
        if (cblas_zher2k_p != NULL)
            cblas_zher2k_p(layout, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

static void cblas_strmm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const float alpha, const float *a, const int lda, float *b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_strmm_p == NULL)
            cblas_strmm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const float alpha, const float *a, const int lda,
                                      float *b, const int ldb))GET_FUNC(h_libcblas, "cblas_strmm");
        if (cblas_strmm_p != NULL)
            cblas_strmm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_dtrmm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const double alpha, const double *a, const int lda, double *b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_dtrmm_p == NULL)
            cblas_dtrmm_p = (void (*)(
                CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
                CBLAS_DIAG diag, const int m, const int n, const double alpha, const double *a,
                const int lda, double *b, const int ldb))GET_FUNC(h_libcblas, "cblas_dtrmm");
        if (cblas_dtrmm_p != NULL)
            cblas_dtrmm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_ctrmm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const void *alpha, const void *a, const int lda, void *b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_ctrmm_p == NULL)
            cblas_ctrmm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const void *alpha, const void *a, const int lda,
                                      void *b, const int ldb))GET_FUNC(h_libcblas, "cblas_ctrmm");
        if (cblas_ctrmm_p != NULL)
            cblas_ctrmm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_ztrmm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const void *alpha, const void *a, const int lda, void *b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_ztrmm_p == NULL)
            cblas_ztrmm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const void *alpha, const void *a, const int lda,
                                      void *b, const int ldb))GET_FUNC(h_libcblas, "cblas_ztrmm");
        if (cblas_ztrmm_p != NULL)
            cblas_ztrmm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_strsm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const float alpha, const float *a, const int lda, float *b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_strsm_p == NULL)
            cblas_strsm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const float alpha, const float *a, const int lda,
                                      float *b, const int ldb))GET_FUNC(h_libcblas, "cblas_strsm");
        if (cblas_strsm_p != NULL)
            cblas_strsm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_dtrsm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const double alpha, const double *a, const int lda, double *b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_dtrsm_p == NULL)
            cblas_dtrsm_p = (void (*)(
                CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
                CBLAS_DIAG diag, const int m, const int n, const double alpha, const double *a,
                const int lda, double *b, const int ldb))GET_FUNC(h_libcblas, "cblas_dtrsm");
        if (cblas_dtrsm_p != NULL)
            cblas_dtrsm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_ctrsm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const void *alpha, const void *a, const int lda, void *b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_ctrsm_p == NULL)
            cblas_ctrsm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const void *alpha, const void *a, const int lda,
                                      void *b, const int ldb))GET_FUNC(h_libcblas, "cblas_ctrsm");
        if (cblas_ctrsm_p != NULL)
            cblas_ctrsm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

static void cblas_ztrsm_wrapper(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m, const int n,
                                const void *alpha, const void *a, const int lda, void *b,
                                const int ldb) {
    if (cblas_library() != NULL) {
        if (cblas_ztrsm_p == NULL)
            cblas_ztrsm_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
                                      CBLAS_TRANSPOSE transa, CBLAS_DIAG diag, const int m,
                                      const int n, const void *alpha, const void *a, const int lda,
                                      void *b, const int ldb))GET_FUNC(h_libcblas, "cblas_ztrsm");
        if (cblas_ztrsm_p != NULL)
            cblas_ztrsm_p(layout, side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
    }
}

/* Level 2 */

static void (*cblas_sgemv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             const float alpha, const float *a, const int lda, const float *x,
                             const int incx, const float beta, float *y, const int incy);
static void (*cblas_dgemv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             const double alpha, const double *a, const int lda, const double *x,
                             const int incx, const double beta, double *y, const int incy);
static void (*cblas_cgemv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             const void *alpha, const void *a, const int lda, const void *x,
                             const int incx, const void *beta, void *y, const int incy);
static void (*cblas_zgemv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             const void *alpha, const void *a, const int lda, const void *x,
                             const int incx, const void *beta, void *y, const int incy);
static void (*cblas_sgbmv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             int kl, int ku, const float alpha, const float *a, const int lda,
                             const float *x, const int incx, const float beta, float *y,
                             const int incy);
static void (*cblas_dgbmv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             int kl, int ku, const double alpha, const double *a, const int lda,
                             const double *x, const int incx, const double beta, double *y,
                             const int incy);
static void (*cblas_cgbmv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             int kl, int ku, const void *alpha, const void *a, const int lda,
                             const void *x, const int incx, const void *beta, void *y,
                             const int incy);
static void (*cblas_zgbmv_p)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                             int kl, int ku, const void *alpha, const void *a, const int lda,
                             const void *x, const int incx, const void *beta, void *y,
                             const int incy);
static void (*cblas_sger_p)(CBLAS_LAYOUT layout, const int m, const int n, const float alpha,
                            const float *x, const int incx, const float *y, const int incy,
                            float *a, const int lda);
static void (*cblas_dger_p)(CBLAS_LAYOUT layout, const int m, const int n, const double alpha,
                            const double *x, const int incx, const double *y, const int incy,
                            double *a, const int lda);
static void (*cblas_cgerc_p)(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                             const void *x, const int incx, const void *y, const int incy, void *a,
                             const int lda);
static void (*cblas_zgerc_p)(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                             const void *x, const int incx, const void *y, const int incy, void *a,
                             const int lda);
static void (*cblas_cgeru_p)(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                             const void *x, const int incx, const void *y, const int incy, void *a,
                             const int lda);
static void (*cblas_zgeru_p)(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                             const void *x, const int incx, const void *y, const int incy, void *a,
                             const int lda);
static void (*cblas_chbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n, const int k,
                             const void *alpha, const void *a, const int lda, const void *x,
                             const int incx, const void *beta, void *y, const int incy);
static void (*cblas_zhbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n, const int k,
                             const void *alpha, const void *a, const int lda, const void *x,
                             const int incx, const void *beta, void *y, const int incy);
static void (*cblas_chemv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void *alpha, const void *a, const int lda, const void *x,
                             const int incx, const void *beta, void *y, const int incy);
static void (*cblas_zhemv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void *alpha, const void *a, const int lda, const void *x,
                             const int incx, const void *beta, void *y, const int incy);
static void (*cblas_cher_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const float alpha, const void *x, const int incx, void *a,
                            const int lda);
static void (*cblas_zher_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const double alpha, const void *x, const int incx, void *a,
                            const int lda);
static void (*cblas_cher2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void *alpha, const void *x, const int incx, const void *y,
                             const int incy, void *a, const int lda);
static void (*cblas_zher2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void *alpha, const void *x, const int incx, const void *y,
                             const int incy, void *a, const int lda);
static void (*cblas_chpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void *alpha, const void *a, const void *x, const int incx,
                             const void *beta, void *y, const int incy);
static void (*cblas_zhpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void *alpha, const void *a, const void *x, const int incx,
                             const void *beta, void *y, const int incy);
static void (*cblas_chpr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const float alpha, const void *x, const int incx, void *a);
static void (*cblas_zhpr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const double alpha, const void *x, const int incx, void *a);
static void (*cblas_chpr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void *alpha, const void *x, const int incx, const void *y,
                             const int incy, void *a);
static void (*cblas_zhpr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const void *alpha, const void *x, const int incx, const void *y,
                             const int incy, void *a);
static void (*cblas_ssbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n, const int k,
                             const float alpha, const float *a, const int lda, const float *x,
                             const int incx, const float beta, float *y, const int incy);
static void (*cblas_dsbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n, const int k,
                             const double alpha, const double *a, const int lda, const double *x,
                             const int incx, const double beta, double *y, const int incy);
static void (*cblas_ssymv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const float alpha, const float *a, const int lda, const float *x,
                             const int incx, const float beta, float *y, const int incy);
static void (*cblas_dsymv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const double alpha, const double *a, const int lda, const double *x,
                             const int incx, const double beta, double *y, const int incy);
static void (*cblas_ssyr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const float alpha, const float *x, const int incx, float *a,
                            const int lda);
static void (*cblas_dsyr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const double alpha, const double *x, const int incx, double *a,
                            const int lda);
static void (*cblas_ssyr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const float alpha, const float *x, const int incx, const float *y,
                             const int incy, float *a, const int lda);
static void (*cblas_dsyr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const double alpha, const double *x, const int incx, const double *y,
                             const int incy, double *a, const int lda);
static void (*cblas_sspmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const float alpha, const float *a, const float *x, const int incx,
                             const float beta, float *y, const int incy);
static void (*cblas_dspmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const double alpha, const double *a, const double *x, const int incx,
                             const double beta, double *y, const int incy);
static void (*cblas_sspr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const float alpha, const float *x, const int incx, float *a);
static void (*cblas_dspr_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                            const double alpha, const double *x, const int incx, double *a);
static void (*cblas_sspr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const float alpha, const float *x, const int incx, const float *y,
                             const int incy, float *a);
static void (*cblas_dspr2_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                             const double alpha, const double *x, const int incx, const double *y,
                             const int incy, double *a);
static void (*cblas_stbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const float *a,
                             const int lda, float *x, const int incx);
static void (*cblas_dtbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const double *a,
                             const int lda, double *x, const int incx);
static void (*cblas_ctbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const void *a,
                             const int lda, void *x, const int incx);
static void (*cblas_ztbmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const void *a,
                             const int lda, void *x, const int incx);
static void (*cblas_stbsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const float *a,
                             const int lda, float *x, const int incx);
static void (*cblas_dtbsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const double *a,
                             const int lda, double *x, const int incx);
static void (*cblas_ctbsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const void *a,
                             const int lda, void *x, const int incx);
static void (*cblas_ztbsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const int k, const void *a,
                             const int lda, void *x, const int incx);
static void (*cblas_stpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const float *a, float *x,
                             const int incx);
static void (*cblas_dtpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const double *a, double *x,
                             const int incx);
static void (*cblas_ctpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                             const int incx);
static void (*cblas_ztpmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                             const int incx);
static void (*cblas_stpsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const float *a, float *x,
                             const int incx);
static void (*cblas_dtpsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const double *a, double *x,
                             const int incx);
static void (*cblas_ctpsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                             const int incx);
static void (*cblas_ztpsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                             const int incx);
static void (*cblas_strmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const float *a, const int lda,
                             float *x, const int incx);
static void (*cblas_dtrmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const double *a, const int lda,
                             double *x, const int incx);
static void (*cblas_ctrmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void *a, const int lda,
                             void *x, const int incx);
static void (*cblas_ztrmv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void *a, const int lda,
                             void *x, const int incx);
static void (*cblas_strsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const float *a, const int lda,
                             float *x, const int incx);
static void (*cblas_dtrsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const double *a, const int lda,
                             double *x, const int incx);
static void (*cblas_ctrsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void *a, const int lda,
                             void *x, const int incx);
static void (*cblas_ztrsv_p)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                             CBLAS_DIAG unit_diag, const int n, const void *a, const int lda,
                             void *x, const int incx);

static void cblas_sgemv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, const float alpha, const float *a, const int lda,
                                const float *x, const int incx, const float beta, float *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_sgemv_p == NULL)
            cblas_sgemv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                      const int n, const float alpha, const float *a, const int lda,
                                      const float *x, const int incx, const float beta, float *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_sgemv");
        if (cblas_sgemv_p != NULL)
            cblas_sgemv_p(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_dgemv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, const double alpha, const double *a, const int lda,
                                const double *x, const int incx, const double beta, double *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dgemv_p == NULL)
            cblas_dgemv_p = (void (*)(
                CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                const double alpha, const double *a, const int lda, const double *x, const int incx,
                const double beta, double *y, const int incy))GET_FUNC(h_libcblas, "cblas_dgemv");
        if (cblas_dgemv_p != NULL)
            cblas_dgemv_p(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_cgemv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, const void *alpha, const void *a, const int lda,
                                const void *x, const int incx, const void *beta, void *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_cgemv_p == NULL)
            cblas_cgemv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                      const int n, const void *alpha, const void *a, const int lda,
                                      const void *x, const int incx, const void *beta, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_cgemv");
        if (cblas_cgemv_p != NULL)
            cblas_cgemv_p(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_zgemv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, const void *alpha, const void *a, const int lda,
                                const void *x, const int incx, const void *beta, void *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zgemv_p == NULL)
            cblas_zgemv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                      const int n, const void *alpha, const void *a, const int lda,
                                      const void *x, const int incx, const void *beta, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zgemv");
        if (cblas_zgemv_p != NULL)
            cblas_zgemv_p(layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_sgbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, int kl, int ku, const float alpha, const float *a,
                                const int lda, const float *x, const int incx, const float beta,
                                float *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_sgbmv_p == NULL)
            cblas_sgbmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                          int kl, int ku, const float alpha, const float *a, const int lda,
                          const float *x, const int incx, const float beta, float *y,
                          const int incy))GET_FUNC(h_libcblas, "cblas_sgbmv");
        if (cblas_sgbmv_p != NULL)
            cblas_sgbmv_p(layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_dgbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, int kl, int ku, const double alpha, const double *a,
                                const int lda, const double *x, const int incx, const double beta,
                                double *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dgbmv_p == NULL)
            cblas_dgbmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                          int kl, int ku, const double alpha, const double *a, const int lda,
                          const double *x, const int incx, const double beta, double *y,
                          const int incy))GET_FUNC(h_libcblas, "cblas_dgbmv");
        if (cblas_dgbmv_p != NULL)
            cblas_dgbmv_p(layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_cgbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, int kl, int ku, const void *alpha, const void *a,
                                const int lda, const void *x, const int incx, const void *beta,
                                void *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_cgbmv_p == NULL)
            cblas_cgbmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                          int kl, int ku, const void *alpha, const void *a, const int lda,
                          const void *x, const int incx, const void *beta, void *y,
                          const int incy))GET_FUNC(h_libcblas, "cblas_cgbmv");
        if (cblas_cgbmv_p != NULL)
            cblas_cgbmv_p(layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_zgbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m,
                                const int n, int kl, int ku, const void *alpha, const void *a,
                                const int lda, const void *x, const int incx, const void *beta,
                                void *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zgbmv_p == NULL)
            cblas_zgbmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int m, const int n,
                          int kl, int ku, const void *alpha, const void *a, const int lda,
                          const void *x, const int incx, const void *beta, void *y,
                          const int incy))GET_FUNC(h_libcblas, "cblas_zgbmv");
        if (cblas_zgbmv_p != NULL)
            cblas_zgbmv_p(layout, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_sger_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const float alpha,
                               const float *x, const int incx, const float *y, const int incy,
                               float *a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_sger_p == NULL)
            cblas_sger_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const float alpha,
                          const float *x, const int incx, const float *y, const int incy, float *a,
                          const int lda))GET_FUNC(h_libcblas, "cblas_sger");
        if (cblas_sger_p != NULL)
            cblas_sger_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_dger_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const double alpha,
                               const double *x, const int incx, const double *y, const int incy,
                               double *a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_dger_p == NULL)
            cblas_dger_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const double alpha,
                          const double *x, const int incx, const double *y, const int incy,
                          double *a, const int lda))GET_FUNC(h_libcblas, "cblas_dger");
        if (cblas_dger_p != NULL)
            cblas_dger_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_cgerc_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                                const void *x, const int incx, const void *y, const int incy,
                                void *a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_cgerc_p == NULL)
            cblas_cgerc_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                          const void *x, const int incx, const void *y, const int incy, void *a,
                          const int lda))GET_FUNC(h_libcblas, "cblas_cgerc");
        if (cblas_cgerc_p != NULL)
            cblas_cgerc_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_zgerc_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                                const void *x, const int incx, const void *y, const int incy,
                                void *a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_zgerc_p == NULL)
            cblas_zgerc_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                          const void *x, const int incx, const void *y, const int incy, void *a,
                          const int lda))GET_FUNC(h_libcblas, "cblas_zgerc");
        if (cblas_zgerc_p != NULL)
            cblas_zgerc_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_cgeru_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                                const void *x, const int incx, const void *y, const int incy,
                                void *a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_cgeru_p == NULL)
            cblas_cgeru_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                          const void *x, const int incx, const void *y, const int incy, void *a,
                          const int lda))GET_FUNC(h_libcblas, "cblas_cgeru");
        if (cblas_cgeru_p != NULL)
            cblas_cgeru_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_zgeru_wrapper(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                                const void *x, const int incx, const void *y, const int incy,
                                void *a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_zgeru_p == NULL)
            cblas_zgeru_p =
                (void (*)(CBLAS_LAYOUT layout, const int m, const int n, const void *alpha,
                          const void *x, const int incx, const void *y, const int incy, void *a,
                          const int lda))GET_FUNC(h_libcblas, "cblas_zgeru");
        if (cblas_zgeru_p != NULL)
            cblas_zgeru_p(layout, m, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_chbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const int k, const void *alpha, const void *a, const int lda,
                                const void *x, const int incx, const void *beta, void *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_chbmv_p == NULL)
            cblas_chbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const int k, const void *alpha, const void *a, const int lda,
                                      const void *x, const int incx, const void *beta, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_chbmv");
        if (cblas_chbmv_p != NULL)
            cblas_chbmv_p(layout, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_zhbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const int k, const void *alpha, const void *a, const int lda,
                                const void *x, const int incx, const void *beta, void *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zhbmv_p == NULL)
            cblas_zhbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const int k, const void *alpha, const void *a, const int lda,
                                      const void *x, const int incx, const void *beta, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zhbmv");
        if (cblas_zhbmv_p != NULL)
            cblas_zhbmv_p(layout, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_chemv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void *alpha, const void *a, const int lda, const void *x,
                                const int incx, const void *beta, void *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_chemv_p == NULL)
            cblas_chemv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void *alpha, const void *a, const int lda,
                                      const void *x, const int incx, const void *beta, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_chemv");
        if (cblas_chemv_p != NULL)
            cblas_chemv_p(layout, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_zhemv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void *alpha, const void *a, const int lda, const void *x,
                                const int incx, const void *beta, void *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zhemv_p == NULL)
            cblas_zhemv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void *alpha, const void *a, const int lda,
                                      const void *x, const int incx, const void *beta, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zhemv");
        if (cblas_zhemv_p != NULL)
            cblas_zhemv_p(layout, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_cher_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const float alpha, const void *x, const int incx, void *a,
                               const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_cher_p == NULL)
            cblas_cher_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const float alpha, const void *x, const int incx, void *a,
                                     const int lda))GET_FUNC(h_libcblas, "cblas_cher");
        if (cblas_cher_p != NULL)
            cblas_cher_p(layout, upper_lower, n, alpha, x, incx, a, lda);
    }
}

static void cblas_zher_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const double alpha, const void *x, const int incx, void *a,
                               const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_zher_p == NULL)
            cblas_zher_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const double alpha, const void *x, const int incx, void *a,
                                     const int lda))GET_FUNC(h_libcblas, "cblas_zher");
        if (cblas_zher_p != NULL)
            cblas_zher_p(layout, upper_lower, n, alpha, x, incx, a, lda);
    }
}

static void cblas_cher2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void *alpha, const void *x, const int incx, const void *y,
                                const int incy, void *a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_cher2_p == NULL)
            cblas_cher2_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void *alpha, const void *x, const int incx,
                                      const void *y, const int incy, void *a,
                                      const int lda))GET_FUNC(h_libcblas, "cblas_cher2");
        if (cblas_cher2_p != NULL)
            cblas_cher2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_zher2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void *alpha, const void *x, const int incx, const void *y,
                                const int incy, void *a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_zher2_p == NULL)
            cblas_zher2_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void *alpha, const void *x, const int incx,
                                      const void *y, const int incy, void *a,
                                      const int lda))GET_FUNC(h_libcblas, "cblas_zher2");
        if (cblas_zher2_p != NULL)
            cblas_zher2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_chpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void *alpha, const void *a, const void *x, const int incx,
                                const void *beta, void *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_chpmv_p == NULL)
            cblas_chpmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void *alpha, const void *a, const void *x,
                                      const int incx, const void *beta, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_chpmv");
        if (cblas_chpmv_p != NULL)
            cblas_chpmv_p(layout, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    }
}

static void cblas_zhpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void *alpha, const void *a, const void *x, const int incx,
                                const void *beta, void *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zhpmv_p == NULL)
            cblas_zhpmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const void *alpha, const void *a, const void *x,
                                      const int incx, const void *beta, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zhpmv");
        if (cblas_zhpmv_p != NULL)
            cblas_zhpmv_p(layout, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    }
}

static void cblas_chpr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const float alpha, const void *x, const int incx, void *a) {
    if (cblas_library() != NULL) {
        if (cblas_chpr_p == NULL)
            cblas_chpr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const float alpha, const void *x, const int incx,
                                     void *a))GET_FUNC(h_libcblas, "cblas_chpr");
        if (cblas_chpr_p != NULL)
            cblas_chpr_p(layout, upper_lower, n, alpha, x, incx, a);
    }
}

static void cblas_zhpr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const double alpha, const void *x, const int incx, void *a) {
    if (cblas_library() != NULL) {
        if (cblas_zhpr_p == NULL)
            cblas_zhpr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const double alpha, const void *x, const int incx,
                                     void *a))GET_FUNC(h_libcblas, "cblas_zhpr");
        if (cblas_zhpr_p != NULL)
            cblas_zhpr_p(layout, upper_lower, n, alpha, x, incx, a);
    }
}

static void cblas_chpr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void *alpha, const void *x, const int incx, const void *y,
                                const int incy, void *a) {
    if (cblas_library() != NULL) {
        if (cblas_chpr2_p == NULL)
            cblas_chpr2_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                          const void *alpha, const void *x, const int incx, const void *y,
                          const int incy, void *a))GET_FUNC(h_libcblas, "cblas_chpr2");
        if (cblas_chpr2_p != NULL)
            cblas_chpr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a);
    }
}

static void cblas_zhpr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const void *alpha, const void *x, const int incx, const void *y,
                                const int incy, void *a) {
    if (cblas_library() != NULL) {
        if (cblas_zhpr2_p == NULL)
            cblas_zhpr2_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                          const void *alpha, const void *x, const int incx, const void *y,
                          const int incy, void *a))GET_FUNC(h_libcblas, "cblas_zhpr2");
        if (cblas_zhpr2_p != NULL)
            cblas_zhpr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a);
    }
}

static void cblas_ssbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const int k, const float alpha, const float *a, const int lda,
                                const float *x, const int incx, const float beta, float *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_ssbmv_p == NULL)
            cblas_ssbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const int k, const float alpha, const float *a, const int lda,
                                      const float *x, const int incx, const float beta, float *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_ssbmv");
        if (cblas_ssbmv_p != NULL)
            cblas_ssbmv_p(layout, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_dsbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const int k, const double alpha, const double *a, const int lda,
                                const double *x, const int incx, const double beta, double *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dsbmv_p == NULL)
            cblas_dsbmv_p = (void (*)(
                CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n, const int k,
                const double alpha, const double *a, const int lda, const double *x, const int incx,
                const double beta, double *y, const int incy))GET_FUNC(h_libcblas, "cblas_dsbmv");
        if (cblas_dsbmv_p != NULL)
            cblas_dsbmv_p(layout, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_ssymv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const float alpha, const float *a, const int lda, const float *x,
                                const int incx, const float beta, float *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_ssymv_p == NULL)
            cblas_ssymv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const float alpha, const float *a, const int lda,
                                      const float *x, const int incx, const float beta, float *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_ssymv");
        if (cblas_ssymv_p != NULL)
            cblas_ssymv_p(layout, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_dsymv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const double alpha, const double *a, const int lda, const double *x,
                                const int incx, const double beta, double *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dsymv_p == NULL)
            cblas_dsymv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const double alpha, const double *a, const int lda,
                                      const double *x, const int incx, const double beta, double *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_dsymv");
        if (cblas_dsymv_p != NULL)
            cblas_dsymv_p(layout, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    }
}

static void cblas_ssyr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const float alpha, const float *x, const int incx, float *a,
                               const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_ssyr_p == NULL)
            cblas_ssyr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const float alpha, const float *x, const int incx, float *a,
                                     const int lda))GET_FUNC(h_libcblas, "cblas_ssyr");
        if (cblas_ssyr_p != NULL)
            cblas_ssyr_p(layout, upper_lower, n, alpha, x, incx, a, lda);
    }
}

static void cblas_dsyr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const double alpha, const double *x, const int incx, double *a,
                               const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_dsyr_p == NULL)
            cblas_dsyr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const double alpha, const double *x, const int incx, double *a,
                                     const int lda))GET_FUNC(h_libcblas, "cblas_dsyr");
        if (cblas_dsyr_p != NULL)
            cblas_dsyr_p(layout, upper_lower, n, alpha, x, incx, a, lda);
    }
}

static void cblas_ssyr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const float alpha, const float *x, const int incx, const float *y,
                                const int incy, float *a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_ssyr2_p == NULL)
            cblas_ssyr2_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const float alpha, const float *x, const int incx,
                                      const float *y, const int incy, float *a,
                                      const int lda))GET_FUNC(h_libcblas, "cblas_ssyr2");
        if (cblas_ssyr2_p != NULL)
            cblas_ssyr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_dsyr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const double alpha, const double *x, const int incx,
                                const double *y, const int incy, double *a, const int lda) {
    if (cblas_library() != NULL) {
        if (cblas_dsyr2_p == NULL)
            cblas_dsyr2_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const double alpha, const double *x, const int incx,
                                      const double *y, const int incy, double *a,
                                      const int lda))GET_FUNC(h_libcblas, "cblas_dsyr2");
        if (cblas_dsyr2_p != NULL)
            cblas_dsyr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    }
}

static void cblas_sspmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const float alpha, const float *a, const float *x, const int incx,
                                const float beta, float *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_sspmv_p == NULL)
            cblas_sspmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const float alpha, const float *a, const float *x,
                                      const int incx, const float beta, float *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_sspmv");
        if (cblas_sspmv_p != NULL)
            cblas_sspmv_p(layout, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    }
}

static void cblas_dspmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const double alpha, const double *a, const double *x,
                                const int incx, const double beta, double *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dspmv_p == NULL)
            cblas_dspmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                      const double alpha, const double *a, const double *x,
                                      const int incx, const double beta, double *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_dspmv");
        if (cblas_dspmv_p != NULL)
            cblas_dspmv_p(layout, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    }
}

static void cblas_sspr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const float alpha, const float *x, const int incx, float *a) {
    if (cblas_library() != NULL) {
        if (cblas_sspr_p == NULL)
            cblas_sspr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const float alpha, const float *x, const int incx,
                                     float *a))GET_FUNC(h_libcblas, "cblas_sspr");
        if (cblas_sspr_p != NULL)
            cblas_sspr_p(layout, upper_lower, n, alpha, x, incx, a);
    }
}

static void cblas_dspr_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                               const double alpha, const double *x, const int incx, double *a) {
    if (cblas_library() != NULL) {
        if (cblas_dspr_p == NULL)
            cblas_dspr_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                     const double alpha, const double *x, const int incx,
                                     double *a))GET_FUNC(h_libcblas, "cblas_dspr");
        if (cblas_dspr_p != NULL)
            cblas_dspr_p(layout, upper_lower, n, alpha, x, incx, a);
    }
}

static void cblas_sspr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const float alpha, const float *x, const int incx, const float *y,
                                const int incy, float *a) {
    if (cblas_library() != NULL) {
        if (cblas_sspr2_p == NULL)
            cblas_sspr2_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                          const float alpha, const float *x, const int incx, const float *y,
                          const int incy, float *a))GET_FUNC(h_libcblas, "cblas_sspr2");
        if (cblas_sspr2_p != NULL)
            cblas_sspr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a);
    }
}

static void cblas_dspr2_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                                const double alpha, const double *x, const int incx,
                                const double *y, const int incy, double *a) {
    if (cblas_library() != NULL) {
        if (cblas_dspr2_p == NULL)
            cblas_dspr2_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int n,
                          const double alpha, const double *x, const int incx, const double *y,
                          const int incy, double *a))GET_FUNC(h_libcblas, "cblas_dspr2");
        if (cblas_dspr2_p != NULL)
            cblas_dspr2_p(layout, upper_lower, n, alpha, x, incx, y, incy, a);
    }
}

static void cblas_stbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const float *a,
                                const int lda, float *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_stbmv_p == NULL)
            cblas_stbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const float *a, const int lda, float *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_stbmv");
        if (cblas_stbmv_p != NULL)
            cblas_stbmv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_dtbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const double *a,
                                const int lda, double *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtbmv_p == NULL)
            cblas_dtbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const double *a, const int lda, double *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_dtbmv");
        if (cblas_dtbmv_p != NULL)
            cblas_dtbmv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_ctbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const void *a,
                                const int lda, void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctbmv_p == NULL)
            cblas_ctbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const void *a, const int lda, void *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_ctbmv");
        if (cblas_ctbmv_p != NULL)
            cblas_ctbmv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_ztbmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const void *a,
                                const int lda, void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztbmv_p == NULL)
            cblas_ztbmv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const void *a, const int lda, void *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_ztbmv");
        if (cblas_ztbmv_p != NULL)
            cblas_ztbmv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_stbsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const float *a,
                                const int lda, float *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_stbsv_p == NULL)
            cblas_stbsv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const float *a, const int lda, float *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_stbsv");
        if (cblas_stbsv_p != NULL)
            cblas_stbsv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_dtbsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const double *a,
                                const int lda, double *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtbsv_p == NULL)
            cblas_dtbsv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const double *a, const int lda, double *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_dtbsv");
        if (cblas_dtbsv_p != NULL)
            cblas_dtbsv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_ctbsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const void *a,
                                const int lda, void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctbsv_p == NULL)
            cblas_ctbsv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const void *a, const int lda, void *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_ctbsv");
        if (cblas_ctbsv_p != NULL)
            cblas_ctbsv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_ztbsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const int k, const void *a,
                                const int lda, void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztbsv_p == NULL)
            cblas_ztbsv_p = (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower,
                                      CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag, const int n,
                                      const int k, const void *a, const int lda, void *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_ztbsv");
        if (cblas_ztbsv_p != NULL)
            cblas_ztbsv_p(layout, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    }
}

static void cblas_stpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const float *a, float *x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_stpmv_p == NULL)
            cblas_stpmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const float *a, float *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_stpmv");
        if (cblas_stpmv_p != NULL)
            cblas_stpmv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_dtpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const double *a, double *x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtpmv_p == NULL)
            cblas_dtpmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const double *a, double *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_dtpmv");
        if (cblas_dtpmv_p != NULL)
            cblas_dtpmv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_ctpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctpmv_p == NULL)
            cblas_ctpmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ctpmv");
        if (cblas_ctpmv_p != NULL)
            cblas_ctpmv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_ztpmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztpmv_p == NULL)
            cblas_ztpmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ztpmv");
        if (cblas_ztpmv_p != NULL)
            cblas_ztpmv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_stpsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const float *a, float *x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_stpsv_p == NULL)
            cblas_stpsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const float *a, float *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_stpsv");
        if (cblas_stpsv_p != NULL)
            cblas_stpsv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_dtpsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const double *a, double *x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtpsv_p == NULL)
            cblas_dtpsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const double *a, double *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_dtpsv");
        if (cblas_dtpsv_p != NULL)
            cblas_dtpsv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_ctpsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctpsv_p == NULL)
            cblas_ctpsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ctpsv");
        if (cblas_ctpsv_p != NULL)
            cblas_ctpsv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_ztpsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                                const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztpsv_p == NULL)
            cblas_ztpsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void *a, void *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ztpsv");
        if (cblas_ztpsv_p != NULL)
            cblas_ztpsv_p(layout, upper_lower, trans, unit_diag, n, a, x, incx);
    }
}

static void cblas_strmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const float *a, const int lda,
                                float *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_strmv_p == NULL)
            cblas_strmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const float *a, const int lda,
                          float *x, const int incx))GET_FUNC(h_libcblas, "cblas_strmv");
        if (cblas_strmv_p != NULL)
            cblas_strmv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_dtrmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const double *a, const int lda,
                                double *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtrmv_p == NULL)
            cblas_dtrmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const double *a, const int lda,
                          double *x, const int incx))GET_FUNC(h_libcblas, "cblas_dtrmv");
        if (cblas_dtrmv_p != NULL)
            cblas_dtrmv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_ctrmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void *a, const int lda,
                                void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctrmv_p == NULL)
            cblas_ctrmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void *a, const int lda, void *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ctrmv");
        if (cblas_ctrmv_p != NULL)
            cblas_ctrmv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_ztrmv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void *a, const int lda,
                                void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztrmv_p == NULL)
            cblas_ztrmv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void *a, const int lda, void *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ztrmv");
        if (cblas_ztrmv_p != NULL)
            cblas_ztrmv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_strsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const float *a, const int lda,
                                float *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_strsv_p == NULL)
            cblas_strsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const float *a, const int lda,
                          float *x, const int incx))GET_FUNC(h_libcblas, "cblas_strsv");
        if (cblas_strsv_p != NULL)
            cblas_strsv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_dtrsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const double *a, const int lda,
                                double *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dtrsv_p == NULL)
            cblas_dtrsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const double *a, const int lda,
                          double *x, const int incx))GET_FUNC(h_libcblas, "cblas_dtrsv");
        if (cblas_dtrsv_p != NULL)
            cblas_dtrsv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_ctrsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void *a, const int lda,
                                void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ctrsv_p == NULL)
            cblas_ctrsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void *a, const int lda, void *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ctrsv");
        if (cblas_ctrsv_p != NULL)
            cblas_ctrsv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

static void cblas_ztrsv_wrapper(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                                CBLAS_DIAG unit_diag, const int n, const void *a, const int lda,
                                void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_ztrsv_p == NULL)
            cblas_ztrsv_p =
                (void (*)(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                          CBLAS_DIAG unit_diag, const int n, const void *a, const int lda, void *x,
                          const int incx))GET_FUNC(h_libcblas, "cblas_ztrsv");
        if (cblas_ztrsv_p != NULL)
            cblas_ztrsv_p(layout, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    }
}

/* Level 1 */

static float (*cblas_sasum_p)(const int n, const float *x, const int incx);
static double (*cblas_dasum_p)(const int n, const double *x, const int incx);
static float (*cblas_scasum_p)(const int n, const void *x, const int incx);
static double (*cblas_dzasum_p)(const int n, const void *x, const int incx);
static void (*cblas_saxpy_p)(const int n, const float alpha, const float *x, const int incx,
                             float *y, const int incy);
static void (*cblas_daxpy_p)(const int n, const double alpha, const double *x, const int incx,
                             double *y, const int incy);
static void (*cblas_caxpy_p)(const int n, const void *alpha, const void *x, const int incx, void *y,
                             const int incy);
static void (*cblas_zaxpy_p)(const int n, const void *alpha, const void *x, const int incx, void *y,
                             const int incy);
static void (*cblas_scopy_p)(const int n, const float *x, const int incx, float *y, const int incy);
static void (*cblas_dcopy_p)(const int n, const double *x, const int incx, double *y,
                             const int incy);
static void (*cblas_ccopy_p)(const int n, const void *x, const int incx, void *y, const int incy);
static void (*cblas_zcopy_p)(const int n, const void *x, const int incx, void *y, const int incy);
static float (*cblas_sdot_p)(const int n, const float *x, const int incx, const float *y,
                             const int incy);
static double (*cblas_ddot_p)(const int n, const double *x, const int incx, const double *y,
                              const int incy);
static double (*cblas_dsdot_p)(const int n, const float *x, const int incx, const float *y,
                               const int incy);
static float (*cblas_sdsdot_p)(const int n, const float sb, const float *x, const int incx,
                               const float *y, const int incy);
static float (*cblas_snrm2_p)(const int n, const float *x, const int incx);
static double (*cblas_dnrm2_p)(const int n, const double *x, const int incx);
static float (*cblas_scnrm2_p)(const int n, const void *x, const int incx);
static double (*cblas_dznrm2_p)(const int n, const void *x, const int incx);
static void (*cblas_srot_p)(const int n, float *x, const int incx, float *y, const int incy,
                            const float c, const float s);
static void (*cblas_drot_p)(const int n, double *x, const int incx, double *y, const int incy,
                            const double c, const double s);
static void (*csrot_p)(const int *n, void *x, const int *incx, void *y, const int *incy,
                       const float *c, const float *s);
static void (*zdrot_p)(const int *n, void *x, const int *incx, void *y, const int *incy,
                       const double *c, const double *s);
static void (*cblas_srotg_p)(float *a, float *b, float *c, float *s);
static void (*cblas_drotg_p)(double *a, double *b, double *c, double *s);
static void (*crotg_p)(void *a, void *b, float *c, void *s);
static void (*zrotg_p)(void *a, void *b, double *c, void *s);
static void (*cblas_srotm_p)(const int n, float *x, const int incx, float *y, const int incy,
                             const float *param);
static void (*cblas_drotm_p)(const int n, double *x, const int incx, double *y, const int incy,
                             const double *param);
static void (*cblas_srotmg_p)(float *d1, float *d2, float *x1, float y1, float *param);
static void (*cblas_drotmg_p)(double *d1, double *d2, double *x1, double y1, double *param);
static void (*cblas_sscal_p)(const int n, const float alpha, float *x, const int incx);
static void (*cblas_dscal_p)(const int n, const double alpha, double *x, const int incx);
static void (*cblas_cscal_p)(const int n, const void *alpha, void *x, const int incx);
static void (*cblas_zscal_p)(const int n, const void *alpha, void *x, const int incx);
static void (*cblas_csscal_p)(const int n, const float alpha, void *x, const int incx);
static void (*cblas_zdscal_p)(const int n, const double alpha, void *x, const int incx);
static void (*cblas_sswap_p)(const int n, float *x, const int incx, float *y, const int incy);
static void (*cblas_dswap_p)(const int n, double *x, const int incx, double *y, const int incy);
static void (*cblas_cswap_p)(const int n, void *x, const int incx, void *y, const int incy);
static void (*cblas_zswap_p)(const int n, void *x, const int incx, void *y, const int incy);
static void (*cblas_cdotc_sub_p)(const int n, const void *x, const int incx, const void *y,
                                 const int incy, void *pres);
static void (*cblas_zdotc_sub_p)(const int n, const void *x, const int incx, const void *y,
                                 const int incy, void *pres);
static void (*cblas_cdotu_sub_p)(const int n, const void *x, const int incx, const void *y,
                                 const int incy, void *pres);
static void (*cblas_zdotu_sub_p)(const int n, const void *x, const int incx, const void *y,
                                 const int incy, void *pres);
static int (*cblas_isamax_p)(const int n, const float *x, const int incx);
static int (*cblas_idamax_p)(const int n, const double *x, const int incx);
static int (*cblas_icamax_p)(const int n, const void *x, const int incx);
static int (*cblas_izamax_p)(const int n, const void *x, const int incx);

static float cblas_sasum_wrapper(const int n, const float *x, const int incx) {
    float sasum_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_sasum_p == NULL)
            cblas_sasum_p = (float (*)(const int n, const float *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_sasum");
        if (cblas_sasum_p != NULL)
            sasum_res = cblas_sasum_p(n, x, incx);
    }
    return sasum_res;
}

static double cblas_dasum_wrapper(const int n, const double *x, const int incx) {
    double dasum_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_dasum_p == NULL)
            cblas_dasum_p = (double (*)(const int n, const double *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_dasum");
        if (cblas_dasum_p != NULL)
            dasum_res = cblas_dasum_p(n, x, incx);
    }
    return dasum_res;
}

static float cblas_scasum_wrapper(const int n, const void *x, const int incx) {
    float scasum_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_scasum_p == NULL)
            cblas_scasum_p = (float (*)(const int n, const void *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_scasum");
        if (cblas_scasum_p != NULL)
            scasum_res = cblas_scasum_p(n, x, incx);
    }
    return scasum_res;
}

static double cblas_dzasum_wrapper(const int n, const void *x, const int incx) {
    double dzasum_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_dzasum_p == NULL)
            cblas_dzasum_p = (double (*)(const int n, const void *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_dzasum");
        if (cblas_dzasum_p != NULL)
            dzasum_res = cblas_dzasum_p(n, x, incx);
    }
    return dzasum_res;
}

static void cblas_saxpy_wrapper(const int n, const float alpha, const float *x, const int incx,
                                float *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_saxpy_p == NULL)
            cblas_saxpy_p =
                (void (*)(const int n, const float alpha, const float *x, const int incx, float *y,
                          const int incy))GET_FUNC(h_libcblas, "cblas_saxpy");
        if (cblas_saxpy_p != NULL)
            cblas_saxpy_p(n, alpha, x, incx, y, incy);
    }
}

static void cblas_daxpy_wrapper(const int n, const double alpha, const double *x, const int incx,
                                double *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_daxpy_p == NULL)
            cblas_daxpy_p =
                (void (*)(const int n, const double alpha, const double *x, const int incx,
                          double *y, const int incy))GET_FUNC(h_libcblas, "cblas_daxpy");
        if (cblas_daxpy_p != NULL)
            cblas_daxpy_p(n, alpha, x, incx, y, incy);
    }
}

static void cblas_caxpy_wrapper(const int n, const void *alpha, const void *x, const int incx,
                                void *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_caxpy_p == NULL)
            cblas_caxpy_p = (void (*)(const int n, const void *alpha, const void *x, const int incx,
                                      void *y, const int incy))GET_FUNC(h_libcblas, "cblas_caxpy");
        if (cblas_caxpy_p != NULL)
            cblas_caxpy_p(n, alpha, x, incx, y, incy);
    }
}

static void cblas_zaxpy_wrapper(const int n, const void *alpha, const void *x, const int incx,
                                void *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zaxpy_p == NULL)
            cblas_zaxpy_p = (void (*)(const int n, const void *alpha, const void *x, const int incx,
                                      void *y, const int incy))GET_FUNC(h_libcblas, "cblas_zaxpy");
        if (cblas_zaxpy_p != NULL)
            cblas_zaxpy_p(n, alpha, x, incx, y, incy);
    }
}

static void cblas_scopy_wrapper(const int n, const float *x, const int incx, float *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_scopy_p == NULL)
            cblas_scopy_p = (void (*)(const int n, const float *x, const int incx, float *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_scopy");
        if (cblas_scopy_p != NULL)
            cblas_scopy_p(n, x, incx, y, incy);
    }
}

static void cblas_dcopy_wrapper(const int n, const double *x, const int incx, double *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dcopy_p == NULL)
            cblas_dcopy_p = (void (*)(const int n, const double *x, const int incx, double *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_dcopy");
        if (cblas_dcopy_p != NULL)
            cblas_dcopy_p(n, x, incx, y, incy);
    }
}

static void cblas_ccopy_wrapper(const int n, const void *x, const int incx, void *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_ccopy_p == NULL)
            cblas_ccopy_p = (void (*)(const int n, const void *x, const int incx, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_ccopy");
        if (cblas_ccopy_p != NULL)
            cblas_ccopy_p(n, x, incx, y, incy);
    }
}

static void cblas_zcopy_wrapper(const int n, const void *x, const int incx, void *y,
                                const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zcopy_p == NULL)
            cblas_zcopy_p = (void (*)(const int n, const void *x, const int incx, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zcopy");
        if (cblas_zcopy_p != NULL)
            cblas_zcopy_p(n, x, incx, y, incy);
    }
}

static float cblas_sdot_wrapper(const int n, const float *x, const int incx, const float *y,
                                const int incy) {
    float sdot_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_sdot_p == NULL)
            cblas_sdot_p = (float (*)(const int n, const float *x, const int incx, const float *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_sdot");
        if (cblas_sdot_p != NULL)
            sdot_res = cblas_sdot_p(n, x, incx, y, incy);
    }
    return sdot_res;
}

static double cblas_ddot_wrapper(const int n, const double *x, const int incx, const double *y,
                                 const int incy) {
    double ddot_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_ddot_p == NULL)
            cblas_ddot_p =
                (double (*)(const int n, const double *x, const int incx, const double *y,
                            const int incy))GET_FUNC(h_libcblas, "cblas_ddot");
        if (cblas_ddot_p != NULL)
            ddot_res = cblas_ddot_p(n, x, incx, y, incy);
    }
    return ddot_res;
}

static double cblas_dsdot_wrapper(const int n, const float *x, const int incx, const float *y,
                                  const int incy) {
    double dsdot_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_dsdot_p == NULL)
            cblas_dsdot_p = (double (*)(const int n, const float *x, const int incx, const float *y,
                                        const int incy))GET_FUNC(h_libcblas, "cblas_dsdot");
        if (cblas_dsdot_p != NULL)
            dsdot_res = cblas_dsdot_p(n, x, incx, y, incy);
    }
    return dsdot_res;
}

static float cblas_sdsdot_wrapper(const int n, const float sb, const float *x, const int incx,
                                  const float *y, const int incy) {
    float sdsdot_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_sdsdot_p == NULL)
            cblas_sdsdot_p =
                (float (*)(const int n, const float sb, const float *x, const int incx,
                           const float *y, const int incy))GET_FUNC(h_libcblas, "cblas_sdsdot");
        if (cblas_sdsdot_p != NULL)
            sdsdot_res = cblas_sdsdot_p(n, sb, x, incx, y, incy);
    }
    return sdsdot_res;
}

static float cblas_snrm2_wrapper(const int n, const float *x, const int incx) {
    float snrm2_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_snrm2_p == NULL)
            cblas_snrm2_p = (float (*)(const int n, const float *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_snrm2");
        if (cblas_snrm2_p != NULL)
            snrm2_res = cblas_snrm2_p(n, x, incx);
    }
    return snrm2_res;
}

static double cblas_dnrm2_wrapper(const int n, const double *x, const int incx) {
    double dnrm2_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_dnrm2_p == NULL)
            cblas_dnrm2_p = (double (*)(const int n, const double *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_dnrm2");
        if (cblas_dnrm2_p != NULL)
            dnrm2_res = cblas_dnrm2_p(n, x, incx);
    }
    return dnrm2_res;
}

static float cblas_scnrm2_wrapper(const int n, const void *x, const int incx) {
    float scnrm2_res = 0.0f;
    if (cblas_library() != NULL) {
        if (cblas_scnrm2_p == NULL)
            cblas_scnrm2_p = (float (*)(const int n, const void *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_scnrm2");
        if (cblas_scnrm2_p != NULL)
            scnrm2_res = cblas_scnrm2_p(n, x, incx);
    }
    return scnrm2_res;
}

static double cblas_dznrm2_wrapper(const int n, const void *x, const int incx) {
    double dznrm2_res = 0.0;
    if (cblas_library() != NULL) {
        if (cblas_dznrm2_p == NULL)
            cblas_dznrm2_p = (double (*)(const int n, const void *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_dznrm2");
        if (cblas_dznrm2_p != NULL)
            dznrm2_res = cblas_dznrm2_p(n, x, incx);
    }
    return dznrm2_res;
}

static void cblas_srot_wrapper(const int n, float *x, const int incx, float *y, const int incy,
                               const float c, const float s) {
    if (cblas_library() != NULL) {
        if (cblas_srot_p == NULL)
            cblas_srot_p =
                (void (*)(const int n, float *x, const int incx, float *y, const int incy,
                          const float c, const float s))GET_FUNC(h_libcblas, "cblas_srot");
        if (cblas_srot_p != NULL)
            cblas_srot_p(n, x, incx, y, incy, c, s);
    }
}

static void cblas_drot_wrapper(const int n, double *x, const int incx, double *y, const int incy,
                               const double c, const double s) {
    if (cblas_library() != NULL) {
        if (cblas_drot_p == NULL)
            cblas_drot_p =
                (void (*)(const int n, double *x, const int incx, double *y, const int incy,
                          const double c, const double s))GET_FUNC(h_libcblas, "cblas_drot");
        if (cblas_drot_p != NULL)
            cblas_drot_p(n, x, incx, y, incy, c, s);
    }
}

static void csrot_wrapper(const int *n, void *x, const int *incx, void *y, const int *incy,
                          const float *c, const float *s) {
    if (blas_library() != NULL) {
        if (csrot_p == NULL)
            csrot_p = (void (*)(const int *n, void *x, const int *incx, void *y, const int *incy,
                                const float *c, const float *s))GET_FUNC(h_libblas, "csrot_");
        if (csrot_p == NULL)
            csrot_p = (void (*)(const int *n, void *x, const int *incx, void *y, const int *incy,
                                const float *c, const float *s))GET_FUNC(h_libblas, "CSROT");
        if (csrot_p != NULL)
            csrot_p(n, x, incx, y, incy, c, s);
    }
}

static void zdrot_wrapper(const int *n, void *x, const int *incx, void *y, const int *incy,
                          const double *c, const double *s) {
    if (blas_library() != NULL) {
        if (zdrot_p == NULL)
            zdrot_p = (void (*)(const int *n, void *x, const int *incx, void *y, const int *incy,
                                const double *c, const double *s))GET_FUNC(h_libblas, "zdrot_");
        if (zdrot_p == NULL)
            zdrot_p = (void (*)(const int *n, void *x, const int *incx, void *y, const int *incy,
                                const double *c, const double *s))GET_FUNC(h_libblas, "ZDROT");
        if (zdrot_p != NULL)
            zdrot_p(n, x, incx, y, incy, c, s);
    }
}

static void cblas_srotg_wrapper(float *a, float *b, float *c, float *s) {
    if (cblas_library() != NULL) {
        if (cblas_srotg_p == NULL)
            cblas_srotg_p = (void (*)(float *a, float *b, float *c, float *s))GET_FUNC(
                h_libcblas, "cblas_srotg");
        if (cblas_srotg_p != NULL)
            cblas_srotg_p(a, b, c, s);
    }
}

static void cblas_drotg_wrapper(double *a, double *b, double *c, double *s) {
    if (cblas_library() != NULL) {
        if (cblas_drotg_p == NULL)
            cblas_drotg_p = (void (*)(double *a, double *b, double *c, double *s))GET_FUNC(
                h_libcblas, "cblas_drotg");
        if (cblas_drotg_p != NULL)
            cblas_drotg_p(a, b, c, s);
    }
}

static void crotg_wrapper(void *a, void *b, float *c, void *s) {
    if (blas_library() != NULL) {
        if (crotg_p == NULL)
            crotg_p = (void (*)(void *a, void *b, float *c, void *s))GET_FUNC(h_libblas, "crotg_");
        if (crotg_p == NULL)
            crotg_p = (void (*)(void *a, void *b, float *c, void *s))GET_FUNC(h_libblas, "CROTG");
        if (crotg_p != NULL)
            crotg_p(a, b, c, s);
    }
}

static void zrotg_wrapper(void *a, void *b, double *c, void *s) {
    if (blas_library() != NULL) {
        if (zrotg_p == NULL)
            zrotg_p = (void (*)(void *a, void *b, double *c, void *s))GET_FUNC(h_libblas, "zrotg_");
        if (zrotg_p == NULL)
            zrotg_p = (void (*)(void *a, void *b, double *c, void *s))GET_FUNC(h_libblas, "ZROTG");
        if (zrotg_p != NULL)
            zrotg_p(a, b, c, s);
    }
}

static void cblas_srotm_wrapper(const int n, float *x, const int incx, float *y, const int incy,
                                const float *param) {
    if (cblas_library() != NULL) {
        if (cblas_srotm_p == NULL)
            cblas_srotm_p =
                (void (*)(const int n, float *x, const int incx, float *y, const int incy,
                          const float *param))GET_FUNC(h_libcblas, "cblas_srotm");
        if (cblas_srotm_p != NULL)
            cblas_srotm_p(n, x, incx, y, incy, param);
    }
}

static void cblas_drotm_wrapper(const int n, double *x, const int incx, double *y, const int incy,
                                const double *param) {
    if (cblas_library() != NULL) {
        if (cblas_drotm_p == NULL)
            cblas_drotm_p =
                (void (*)(const int n, double *x, const int incx, double *y, const int incy,
                          const double *param))GET_FUNC(h_libcblas, "cblas_drotm");
        if (cblas_drotm_p != NULL)
            cblas_drotm_p(n, x, incx, y, incy, param);
    }
}

static void cblas_srotmg_wrapper(float *d1, float *d2, float *x1, float y1, float *param) {
    if (cblas_library() != NULL) {
        if (cblas_srotmg_p == NULL)
            cblas_srotmg_p = (void (*)(float *d1, float *d2, float *x1, float y1,
                                       float *param))GET_FUNC(h_libcblas, "cblas_srotmg");
        if (cblas_srotmg_p != NULL)
            cblas_srotmg_p(d1, d2, x1, y1, param);
    }
}

static void cblas_drotmg_wrapper(double *d1, double *d2, double *x1, double y1, double *param) {
    if (cblas_library() != NULL) {
        if (cblas_drotmg_p == NULL)
            cblas_drotmg_p = (void (*)(double *d1, double *d2, double *x1, double y1,
                                       double *param))GET_FUNC(h_libcblas, "cblas_drotmg");
        if (cblas_drotmg_p != NULL)
            cblas_drotmg_p(d1, d2, x1, y1, param);
    }
}

static void cblas_sscal_wrapper(const int n, const float alpha, float *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_sscal_p == NULL)
            cblas_sscal_p = (void (*)(const int n, const float alpha, float *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_sscal");
        if (cblas_sscal_p != NULL)
            cblas_sscal_p(n, alpha, x, incx);
    }
}

static void cblas_dscal_wrapper(const int n, const double alpha, double *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_dscal_p == NULL)
            cblas_dscal_p = (void (*)(const int n, const double alpha, double *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_dscal");
        if (cblas_dscal_p != NULL)
            cblas_dscal_p(n, alpha, x, incx);
    }
}

static void cblas_cscal_wrapper(const int n, const void *alpha, void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_cscal_p == NULL)
            cblas_cscal_p = (void (*)(const int n, const void *alpha, void *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_cscal");
        if (cblas_cscal_p != NULL)
            cblas_cscal_p(n, alpha, x, incx);
    }
}

static void cblas_zscal_wrapper(const int n, const void *alpha, void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_zscal_p == NULL)
            cblas_zscal_p = (void (*)(const int n, const void *alpha, void *x,
                                      const int incx))GET_FUNC(h_libcblas, "cblas_zscal");
        if (cblas_zscal_p != NULL)
            cblas_zscal_p(n, alpha, x, incx);
    }
}

static void cblas_csscal_wrapper(const int n, const float alpha, void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_csscal_p == NULL)
            cblas_csscal_p = (void (*)(const int n, const float alpha, void *x,
                                       const int incx))GET_FUNC(h_libcblas, "cblas_csscal");
        if (cblas_csscal_p != NULL)
            cblas_csscal_p(n, alpha, x, incx);
    }
}

static void cblas_zdscal_wrapper(const int n, const double alpha, void *x, const int incx) {
    if (cblas_library() != NULL) {
        if (cblas_zdscal_p == NULL)
            cblas_zdscal_p = (void (*)(const int n, const double alpha, void *x,
                                       const int incx))GET_FUNC(h_libcblas, "cblas_zdscal");
        if (cblas_zdscal_p != NULL)
            cblas_zdscal_p(n, alpha, x, incx);
    }
}

static void cblas_sswap_wrapper(const int n, float *x, const int incx, float *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_sswap_p == NULL)
            cblas_sswap_p = (void (*)(const int n, float *x, const int incx, float *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_sswap");
        if (cblas_sswap_p != NULL)
            cblas_sswap_p(n, x, incx, y, incy);
    }
}

static void cblas_dswap_wrapper(const int n, double *x, const int incx, double *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_dswap_p == NULL)
            cblas_dswap_p = (void (*)(const int n, double *x, const int incx, double *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_dswap");
        if (cblas_dswap_p != NULL)
            cblas_dswap_p(n, x, incx, y, incy);
    }
}

static void cblas_cswap_wrapper(const int n, void *x, const int incx, void *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_cswap_p == NULL)
            cblas_cswap_p = (void (*)(const int n, void *x, const int incx, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_cswap");
        if (cblas_cswap_p != NULL)
            cblas_cswap_p(n, x, incx, y, incy);
    }
}

static void cblas_zswap_wrapper(const int n, void *x, const int incx, void *y, const int incy) {
    if (cblas_library() != NULL) {
        if (cblas_zswap_p == NULL)
            cblas_zswap_p = (void (*)(const int n, void *x, const int incx, void *y,
                                      const int incy))GET_FUNC(h_libcblas, "cblas_zswap");
        if (cblas_zswap_p != NULL)
            cblas_zswap_p(n, x, incx, y, incy);
    }
}

static void cblas_cdotc_sub_wrapper(const int n, const void *x, const int incx, const void *y,
                                    const int incy, void *pres) {
    if (cblas_library() != NULL) {
        if (cblas_cdotc_sub_p == NULL)
            cblas_cdotc_sub_p =
                (void (*)(const int n, const void *x, const int incx, const void *y, const int incy,
                          void *pres))GET_FUNC(h_libcblas, "cblas_cdotc_sub");
        if (cblas_cdotc_sub_p != NULL)
            cblas_cdotc_sub_p(n, x, incx, y, incy, pres);
    }
}

static void cblas_zdotc_sub_wrapper(const int n, const void *x, const int incx, const void *y,
                                    const int incy, void *pres) {
    if (cblas_library() != NULL) {
        if (cblas_zdotc_sub_p == NULL)
            cblas_zdotc_sub_p =
                (void (*)(const int n, const void *x, const int incx, const void *y, const int incy,
                          void *pres))GET_FUNC(h_libcblas, "cblas_zdotc_sub");
        if (cblas_zdotc_sub_p != NULL)
            cblas_zdotc_sub_p(n, x, incx, y, incy, pres);
    }
}

static void cblas_cdotu_sub_wrapper(const int n, const void *x, const int incx, const void *y,
                                    const int incy, void *pres) {
    if (cblas_library() != NULL) {
        if (cblas_cdotu_sub_p == NULL)
            cblas_cdotu_sub_p =
                (void (*)(const int n, const void *x, const int incx, const void *y, const int incy,
                          void *pres))GET_FUNC(h_libcblas, "cblas_cdotu_sub");
        if (cblas_cdotu_sub_p != NULL)
            cblas_cdotu_sub_p(n, x, incx, y, incy, pres);
    }
}

static void cblas_zdotu_sub_wrapper(const int n, const void *x, const int incx, const void *y,
                                    const int incy, void *pres) {
    if (cblas_library() != NULL) {
        if (cblas_zdotu_sub_p == NULL)
            cblas_zdotu_sub_p =
                (void (*)(const int n, const void *x, const int incx, const void *y, const int incy,
                          void *pres))GET_FUNC(h_libcblas, "cblas_zdotu_sub");
        if (cblas_zdotu_sub_p != NULL)
            cblas_zdotu_sub_p(n, x, incx, y, incy, pres);
    }
}

static int cblas_isamax_wrapper(const int n, const float *x, const int incx) {
    int isamax_res = 0;
    if (cblas_library() != NULL) {
        if (cblas_isamax_p == NULL)
            cblas_isamax_p = (int (*)(const int n, const float *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_isamax");
        if (cblas_isamax_p != NULL)
            isamax_res = cblas_isamax_p(n, x, incx);
    }
    return isamax_res;
}

static int cblas_idamax_wrapper(const int n, const double *x, const int incx) {
    int idamax_res = 0;
    if (cblas_library() != NULL) {
        if (cblas_idamax_p == NULL)
            cblas_idamax_p = (int (*)(const int n, const double *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_idamax");
        if (cblas_idamax_p != NULL)
            idamax_res = cblas_idamax_p(n, x, incx);
    }
    return idamax_res;
}

static int cblas_icamax_wrapper(const int n, const void *x, const int incx) {
    int icamax_res = 0;
    if (cblas_library() != NULL) {
        if (cblas_icamax_p == NULL)
            cblas_icamax_p = (int (*)(const int n, const void *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_icamax");
        if (cblas_icamax_p != NULL)
            icamax_res = cblas_icamax_p(n, x, incx);
    }
    return icamax_res;
}

static int cblas_izamax_wrapper(const int n, const void *x, const int incx) {
    int izamax_res = 0;
    if (cblas_library() != NULL) {
        if (cblas_izamax_p == NULL)
            cblas_izamax_p = (int (*)(const int n, const void *x, const int incx))GET_FUNC(
                h_libcblas, "cblas_izamax");
        if (cblas_izamax_p != NULL)
            izamax_res = cblas_izamax_p(n, x, incx);
    }
    return izamax_res;
}
}

inline bool isNonTranspose(CBLAS_TRANSPOSE trans) {
    return trans == CblasNoTrans;
}

template <typename T_src, typename T_dest>
static inline void copy_mat(T_src &src, CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int row,
                            int col, int ld, T_dest *&dest) {
    int i, j, Iend, Jend;
    if (layout == CblasColMajor) {
        Jend = isNonTranspose(trans) ? col : row;
        Iend = isNonTranspose(trans) ? row : col;
    }
    else {
        Jend = isNonTranspose(trans) ? row : col;
        Iend = isNonTranspose(trans) ? col : row;
    }

    for (j = 0; j < Jend; j++) {
        for (i = 0; i < Iend; i++) {
            dest[i + ld * j] = (T_dest)src[i + ld * j];
        }
    }
}

template <typename T_src, typename T_dest>
static inline void copy_mat(T_src &src, CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int row,
                            int col, int ld, T_dest off, T_dest *&dest) {
    int i, j, Iend, Jend;
    if (layout == CblasColMajor) {
        Jend = isNonTranspose(trans) ? col : row;
        Iend = isNonTranspose(trans) ? row : col;
    }
    else {
        Jend = isNonTranspose(trans) ? row : col;
        Iend = isNonTranspose(trans) ? col : row;
    }

    for (j = 0; j < Jend; j++) {
        for (i = 0; i < Iend; i++) {
            dest[i + ld * j] = (T_dest)src[i + ld * j] - off;
        }
    }
}

template <typename T_src, typename T_dest, typename T_off>
static inline void copy_mat(T_src &src, CBLAS_LAYOUT layout, int row, int col, int ld,
                            CBLAS_OFFSET off_kind, T_off off, T_dest &dest) {
    using T_data = typename std::remove_reference<decltype(dest[0])>::type;
    int i, j;
    T_data tmp;

    int Jend = (layout == CblasColMajor) ? col : row;
    int Iend = (layout == CblasColMajor) ? row : col;

    if (off_kind == CblasFixOffset) {
        tmp = off[0];
        for (j = 0; j < Jend; j++) {
            for (i = 0; i < Iend; i++) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else if (((off_kind == CblasColOffset) && (layout == CblasColMajor)) ||
             ((off_kind == CblasRowOffset) && (layout == CblasRowMajor))) {
        for (j = 0; j < Jend; j++) {
            for (i = 0; i < Iend; i++) {
                tmp = off[i];
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
    else {
        for (j = 0; j < Jend; j++) {
            tmp = off[j];
            for (i = 0; i < Iend; i++) {
                dest[i + ld * j] = tmp + (T_data)src[i + ld * j];
            }
        }
    }
}

template <typename T_src, typename T_desc>
static inline void update_c(T_src &src, CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, int row,
                            int col, int ld, T_desc *&dest) {
    int i, j;

    int Jend = (layout == CblasColMajor) ? col : row;
    int Iend = (layout == CblasColMajor) ? row : col;

    for (j = 0; j < Jend; j++) {
        for (i = 0; i < Iend; i++) {
            if (((upper_lower == CblasUpper) && (layout == CblasColMajor)) ||
                ((upper_lower == CblasLower) && (layout == CblasRowMajor))) {
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
static void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m,
                 const int *n, const int *k, const fp *alpha, const fp *a, const int *lda,
                 const fp *b, const int *ldb, const fp *beta, fp *c, const int *ldc);

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m,
          const int *n, const int *k, const sycl::half *alpha, const sycl::half *a, const int *lda,
          const sycl::half *b, const int *ldb, const sycl::half *beta, sycl::half *c,
          const int *ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizea, sizeb, sizec;
    const float alphaf = *alpha;
    const float betaf = *beta;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
        sizec = *ldc * *n;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
        sizec = *ldc * *m;
    }
    float *af = (float *)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizea);
    float *bf = (float *)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizeb);
    float *cf = (float *)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizec);
    copy_mat(a, layout, transa, *m, *k, *lda, af);
    copy_mat(b, layout, transb, *k, *n, *ldb, bf);
    copy_mat(c, layout, CblasNoTrans, *m, *n, *ldc, cf);
    cblas_sgemm_wrapper(layout, transa, transb, *m, *n, *k, alphaf, af, *lda, bf, *ldb, betaf, cf,
                        *ldc);
    copy_mat(cf, layout, CblasNoTrans, *m, *n, *ldc, c);
    oneapi::mkl::aligned_free(af);
    oneapi::mkl::aligned_free(bf);
    oneapi::mkl::aligned_free(cf);
}

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m,
          const int *n, const int *k, const float *alpha, const float *a, const int *lda,
          const float *b, const int *ldb, const float *beta, float *c, const int *ldc) {
    cblas_sgemm_wrapper(layout, transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c,
                        *ldc);
}

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m,
          const int *n, const int *k, const double *alpha, const double *a, const int *lda,
          const double *b, const int *ldb, const double *beta, double *c, const int *ldc) {
    cblas_dgemm_wrapper(layout, transa, transb, *m, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c,
                        *ldc);
}

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m,
          const int *n, const int *k, const std::complex<float> *alpha,
          const std::complex<float> *a, const int *lda, const std::complex<float> *b,
          const int *ldb, const std::complex<float> *beta, std::complex<float> *c, const int *ldc) {
    cblas_cgemm_wrapper(layout, transa, transb, *m, *n, *k, (const void *)alpha, (const void *)a,
                        *lda, (const void *)b, *ldb, (const void *)beta, (void *)c, *ldc);
}

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m,
          const int *n, const int *k, const std::complex<double> *alpha,
          const std::complex<double> *a, const int *lda, const std::complex<double> *b,
          const int *ldb, const std::complex<double> *beta, std::complex<double> *c,
          const int *ldc) {
    cblas_zgemm_wrapper(layout, transa, transb, *m, *n, *k, (const void *)alpha, (const void *)a,
                        *lda, (const void *)b, *ldb, (const void *)beta, (void *)c, *ldc);
}

template <typename fpa, typename fpc>
static void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m,
                 const int *n, const int *k, const fpc *alpha, const fpa *a, const int *lda,
                 const fpa *b, const int *ldb, const fpc *beta, fpc *c, const int *ldc);

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m,
          const int *n, const int *k, const float *alpha, const sycl::half *a, const int *lda,
          const sycl::half *b, const int *ldb, const float *beta, float *c, const int *ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizea, sizeb;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
    }
    float *af = (float *)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizea);
    float *bf = (float *)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizeb);
    copy_mat(a, layout, transa, *m, *k, *lda, af);
    copy_mat(b, layout, transb, *k, *n, *ldb, bf);
    cblas_sgemm_wrapper(layout, transa, transb, *m, *n, *k, *alpha, af, *lda, bf, *ldb, *beta, c,
                        *ldc);
    oneapi::mkl::aligned_free(af);
    oneapi::mkl::aligned_free(bf);
}

template <>
void gemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, const int *m,
          const int *n, const int *k, const float *alpha, const oneapi::mkl::bfloat16 *a,
          const int *lda, const oneapi::mkl::bfloat16 *b, const int *ldb, const float *beta,
          float *c, const int *ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizea, sizeb;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
    }
    float *af = (float *)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizea);
    float *bf = (float *)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizeb);
    copy_mat(a, layout, transa, *m, *k, *lda, af);
    copy_mat(b, layout, transb, *k, *n, *ldb, bf);
    cblas_sgemm_wrapper(layout, transa, transb, *m, *n, *k, *alpha, af, *lda, bf, *ldb, *beta, c,
                        *ldc);
    oneapi::mkl::aligned_free(af);
    oneapi::mkl::aligned_free(bf);
}

template <typename fp>
static void symm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m,
                 const int *n, const fp *alpha, const fp *a, const int *lda, const fp *b,
                 const int *ldb, const fp *beta, fp *c, const int *ldc);

template <>
void symm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
          const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
          const float *beta, float *c, const int *ldc) {
    cblas_ssymm_wrapper(layout, left_right, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void symm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
          const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
          const double *beta, double *c, const int *ldc) {
    cblas_dsymm_wrapper(layout, left_right, uplo, *m, *n, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void symm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *b, const int *ldb, const std::complex<float> *beta,
          std::complex<float> *c, const int *ldc) {
    cblas_csymm_wrapper(layout, left_right, uplo, *m, *n, (const void *)alpha, (const void *)a,
                        *lda, (const void *)b, *ldb, (const void *)beta, (void *)c, *ldc);
}

template <>
void symm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *b, const int *ldb, const std::complex<double> *beta,
          std::complex<double> *c, const int *ldc) {
    cblas_zsymm_wrapper(layout, left_right, uplo, *m, *n, (const void *)alpha, (const void *)a,
                        *lda, (const void *)b, *ldb, (const void *)beta, (void *)c, *ldc);
}

template <typename fp>
static void syrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n,
                 const int *k, const fp *alpha, const fp *a, const int *lda, const fp *beta, fp *c,
                 const int *ldc);

template <>
void syrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
          const float *alpha, const float *a, const int *lda, const float *beta, float *c,
          const int *ldc) {
    cblas_ssyrk_wrapper(layout, uplo, trans, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

template <>
void syrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
          const double *alpha, const double *a, const int *lda, const double *beta, double *c,
          const int *ldc) {
    cblas_dsyrk_wrapper(layout, uplo, trans, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

template <>
void syrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *beta, std::complex<float> *c, const int *ldc) {
    cblas_csyrk_wrapper(layout, uplo, trans, *n, *k, (const void *)alpha, (const void *)a, *lda,
                        (const void *)beta, (void *)c, *ldc);
}

template <>
void syrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *beta, std::complex<double> *c, const int *ldc) {
    cblas_zsyrk_wrapper(layout, uplo, trans, *n, *k, (const void *)alpha, (const void *)a, *lda,
                        (const void *)beta, (void *)c, *ldc);
}

template <typename fp>
static void hemm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m,
                 const int *n, const fp *alpha, const fp *a, const int *lda, const fp *b,
                 const int *ldb, const fp *beta, fp *c, const int *ldc);

template <>
void hemm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *b, const int *ldb, const std::complex<float> *beta,
          std::complex<float> *c, const int *ldc) {
    cblas_chemm_wrapper(layout, left_right, uplo, *m, *n, (const void *)alpha, (const void *)a,
                        *lda, (const void *)b, *ldb, (const void *)beta, (void *)c, *ldc);
}

template <>
void hemm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, CBLAS_UPLO uplo, const int *m, const int *n,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *b, const int *ldb, const std::complex<double> *beta,
          std::complex<double> *c, const int *ldc) {
    cblas_zhemm_wrapper(layout, left_right, uplo, *m, *n, (const void *)alpha, (const void *)a,
                        *lda, (const void *)b, *ldb, (const void *)beta, (void *)c, *ldc);
}

template <typename fp_scalar, typename fp_data>
static void herk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n,
                 const int *k, const fp_scalar *alpha, const fp_data *a, const int *lda,
                 const fp_scalar *beta, fp_data *c, const int *ldc);

template <>
void herk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
          const float *alpha, const std::complex<float> *a, const int *lda, const float *beta,
          std::complex<float> *c, const int *ldc) {
    cblas_cherk_wrapper(layout, uplo, trans, *n, *k, *alpha, (const void *)a, *lda, *beta,
                        (void *)c, *ldc);
}

template <>
void herk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
          const double *alpha, const std::complex<double> *a, const int *lda, const double *beta,
          std::complex<double> *c, const int *ldc) {
    cblas_zherk_wrapper(layout, uplo, trans, *n, *k, *alpha, (const void *)a, *lda, *beta,
                        (void *)c, *ldc);
}

template <typename fp>
static void syr2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n,
                  const int *k, const fp *alpha, const fp *a, const int *lda, const fp *b,
                  const int *ldb, const fp *beta, fp *c, const int *ldc);

template <>
void syr2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
           const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
           const float *beta, float *c, const int *ldc) {
    cblas_ssyr2k_wrapper(layout, uplo, trans, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void syr2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
           const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
           const double *beta, double *c, const int *ldc) {
    cblas_dsyr2k_wrapper(layout, uplo, trans, *n, *k, *alpha, a, *lda, b, *ldb, *beta, c, *ldc);
}

template <>
void syr2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
           const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
           const std::complex<float> *b, const int *ldb, const std::complex<float> *beta,
           std::complex<float> *c, const int *ldc) {
    cblas_csyr2k_wrapper(layout, uplo, trans, *n, *k, (const void *)alpha, (const void *)a, *lda,
                         (const void *)b, *ldb, (const void *)beta, (void *)c, *ldc);
}

template <>
void syr2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
           const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
           const std::complex<double> *b, const int *ldb, const std::complex<double> *beta,
           std::complex<double> *c, const int *ldc) {
    cblas_zsyr2k_wrapper(layout, uplo, trans, *n, *k, (const void *)alpha, (const void *)a, *lda,
                         (const void *)b, *ldb, (const void *)beta, (void *)c, *ldc);
}

template <typename fp_scalar, typename fp_data>
static void her2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n,
                  const int *k, const fp_data *alpha, const fp_data *a, const int *lda,
                  const fp_data *b, const int *ldb, const fp_scalar *beta, fp_data *c,
                  const int *ldc);

template <>
void her2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
           const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
           const std::complex<float> *b, const int *ldb, const float *beta, std::complex<float> *c,
           const int *ldc) {
    cblas_cher2k_wrapper(layout, uplo, trans, *n, *k, (const void *)alpha, (const void *)a, *lda,
                         (const void *)b, *ldb, *beta, (void *)c, *ldc);
}

template <>
void her2k(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, const int *n, const int *k,
           const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
           const std::complex<double> *b, const int *ldb, const double *beta,
           std::complex<double> *c, const int *ldc) {
    cblas_zher2k_wrapper(layout, uplo, trans, *n, *k, (const void *)alpha, (const void *)a, *lda,
                         (const void *)b, *ldb, *beta, (void *)c, *ldc);
}

template <typename fp>
static void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
                 CBLAS_DIAG diag, const int *m, const int *n, const fp *alpha, const fp *a,
                 const int *lda, fp *b, const int *ldb);

template <>
void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int *m, const int *n, const float *alpha, const float *a,
          const int *lda, float *b, const int *ldb) {
    cblas_strmm_wrapper(layout, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int *m, const int *n, const double *alpha, const double *a,
          const int *lda, double *b, const int *ldb) {
    cblas_dtrmm_wrapper(layout, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int *m, const int *n, const std::complex<float> *alpha,
          const std::complex<float> *a, const int *lda, std::complex<float> *b, const int *ldb) {
    cblas_ctrmm_wrapper(layout, side, uplo, transa, diag, *m, *n, (const void *)alpha,
                        (const void *)a, *lda, (void *)b, *ldb);
}

template <>
void trmm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int *m, const int *n, const std::complex<double> *alpha,
          const std::complex<double> *a, const int *lda, std::complex<double> *b, const int *ldb) {
    cblas_ztrmm_wrapper(layout, side, uplo, transa, diag, *m, *n, (const void *)alpha,
                        (const void *)a, *lda, (void *)b, *ldb);
}

template <typename fp>
static void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
                 CBLAS_DIAG diag, const int *m, const int *n, const fp *alpha, const fp *a,
                 const int *lda, fp *b, const int *ldb);

template <>
void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int *m, const int *n, const float *alpha, const float *a,
          const int *lda, float *b, const int *ldb) {
    cblas_strsm_wrapper(layout, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int *m, const int *n, const double *alpha, const double *a,
          const int *lda, double *b, const int *ldb) {
    cblas_dtrsm_wrapper(layout, side, uplo, transa, diag, *m, *n, *alpha, a, *lda, b, *ldb);
}

template <>
void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int *m, const int *n, const std::complex<float> *alpha,
          const std::complex<float> *a, const int *lda, std::complex<float> *b, const int *ldb) {
    cblas_ctrsm_wrapper(layout, side, uplo, transa, diag, *m, *n, (const void *)alpha,
                        (const void *)a, *lda, (void *)b, *ldb);
}

template <>
void trsm(CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transa,
          CBLAS_DIAG diag, const int *m, const int *n, const std::complex<double> *alpha,
          const std::complex<double> *a, const int *lda, std::complex<double> *b, const int *ldb) {
    cblas_ztrsm_wrapper(layout, side, uplo, transa, diag, *m, *n, (const void *)alpha,
                        (const void *)a, *lda, (void *)b, *ldb);
}

/* Level 2 */

template <typename fp>
static void gemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int *m, const int *n,
                 const fp *alpha, const fp *a, const int *lda, const fp *x, const int *incx,
                 const fp *beta, fp *y, const int *incy);

template <>
void gemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int *m, const int *n,
          const float *alpha, const float *a, const int *lda, const float *x, const int *incx,
          const float *beta, float *y, const int *incy) {
    cblas_sgemv_wrapper(layout, trans, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void gemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int *m, const int *n,
          const double *alpha, const double *a, const int *lda, const double *x, const int *incx,
          const double *beta, double *y, const int *incy) {
    cblas_dgemv_wrapper(layout, trans, *m, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void gemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int *m, const int *n,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *x, const int *incx, const std::complex<float> *beta,
          std::complex<float> *y, const int *incy) {
    cblas_cgemv_wrapper(layout, trans, *m, *n, (const void *)alpha, (const void *)a, *lda,
                        (const void *)x, *incx, (const void *)beta, (void *)y, *incy);
}

template <>
void gemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int *m, const int *n,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *x, const int *incx, const std::complex<double> *beta,
          std::complex<double> *y, const int *incy) {
    cblas_zgemv_wrapper(layout, trans, *m, *n, (const void *)alpha, (const void *)a, *lda,
                        (const void *)x, *incx, (const void *)beta, (void *)y, *incy);
}

template <typename fp>
static void gbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int *m, const int *n, int *kl,
                 int *ku, const fp *alpha, const fp *a, const int *lda, const fp *x,
                 const int *incx, const fp *beta, fp *y, const int *incy);

template <>
void gbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int *m, const int *n, int *kl, int *ku,
          const float *alpha, const float *a, const int *lda, const float *x, const int *incx,
          const float *beta, float *y, const int *incy) {
    cblas_sgbmv_wrapper(layout, trans, *m, *n, *kl, *ku, *alpha, a, *lda, x, *incx, *beta, y,
                        *incy);
}

template <>
void gbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int *m, const int *n, int *kl, int *ku,
          const double *alpha, const double *a, const int *lda, const double *x, const int *incx,
          const double *beta, double *y, const int *incy) {
    cblas_dgbmv_wrapper(layout, trans, *m, *n, *kl, *ku, *alpha, a, *lda, x, *incx, *beta, y,
                        *incy);
}

template <>
void gbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int *m, const int *n, int *kl, int *ku,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *x, const int *incx, const std::complex<float> *beta,
          std::complex<float> *y, const int *incy) {
    cblas_cgbmv_wrapper(layout, trans, *m, *n, *kl, *ku, (const void *)alpha, (const void *)a, *lda,
                        (const void *)x, *incx, (const void *)beta, (void *)y, *incy);
}

template <>
void gbmv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, const int *m, const int *n, int *kl, int *ku,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *x, const int *incx, const std::complex<double> *beta,
          std::complex<double> *y, const int *incy) {
    cblas_zgbmv_wrapper(layout, trans, *m, *n, *kl, *ku, (const void *)alpha, (const void *)a, *lda,
                        (const void *)x, *incx, (const void *)beta, (void *)y, *incy);
}

template <typename fp>
static void ger(CBLAS_LAYOUT layout, const int *m, const int *n, const fp *alpha, const fp *x,
                const int *incx, const fp *y, const int *incy, fp *a, const int *lda);

template <>
void ger(CBLAS_LAYOUT layout, const int *m, const int *n, const float *alpha, const float *x,
         const int *incx, const float *y, const int *incy, float *a, const int *lda) {
    cblas_sger_wrapper(layout, *m, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <>
void ger(CBLAS_LAYOUT layout, const int *m, const int *n, const double *alpha, const double *x,
         const int *incx, const double *y, const int *incy, double *a, const int *lda) {
    cblas_dger_wrapper(layout, *m, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <typename fp>
static void gerc(CBLAS_LAYOUT layout, const int *m, const int *n, const fp *alpha, const fp *x,
                 const int *incx, const fp *y, const int *incy, fp *a, const int *lda);

template <>
void gerc(CBLAS_LAYOUT layout, const int *m, const int *n, const std::complex<float> *alpha,
          const std::complex<float> *x, const int *incx, const std::complex<float> *y,
          const int *incy, std::complex<float> *a, const int *lda) {
    cblas_cgerc_wrapper(layout, *m, *n, (const void *)alpha, (const void *)x, *incx,
                        (const void *)y, *incy, (void *)a, *lda);
}

template <>
void gerc(CBLAS_LAYOUT layout, const int *m, const int *n, const std::complex<double> *alpha,
          const std::complex<double> *x, const int *incx, const std::complex<double> *y,
          const int *incy, std::complex<double> *a, const int *lda) {
    cblas_zgerc_wrapper(layout, *m, *n, (const void *)alpha, (const void *)x, *incx,
                        (const void *)y, *incy, (void *)a, *lda);
}

template <typename fp>
static void geru(CBLAS_LAYOUT layout, const int *m, const int *n, const fp *alpha, const fp *x,
                 const int *incx, const fp *y, const int *incy, fp *a, const int *lda);

template <>
void geru(CBLAS_LAYOUT layout, const int *m, const int *n, const std::complex<float> *alpha,
          const std::complex<float> *x, const int *incx, const std::complex<float> *y,
          const int *incy, std::complex<float> *a, const int *lda) {
    cblas_cgeru_wrapper(layout, *m, *n, (const void *)alpha, (const void *)x, *incx,
                        (const void *)y, *incy, (void *)a, *lda);
}

template <>
void geru(CBLAS_LAYOUT layout, const int *m, const int *n, const std::complex<double> *alpha,
          const std::complex<double> *x, const int *incx, const std::complex<double> *y,
          const int *incy, std::complex<double> *a, const int *lda) {
    cblas_zgeru_wrapper(layout, *m, *n, (const void *)alpha, (const void *)x, *incx,
                        (const void *)y, *incy, (void *)a, *lda);
}

template <typename fp>
static void hbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const int *k,
                 const fp *alpha, const fp *a, const int *lda, const fp *x, const int *incx,
                 const fp *beta, fp *y, const int *incy);

template <>
void hbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const int *k,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *x, const int *incx, const std::complex<float> *beta,
          std::complex<float> *y, const int *incy) {
    cblas_chbmv_wrapper(layout, upper_lower, *n, *k, (const void *)alpha, (const void *)a, *lda,
                        (const void *)x, *incx, (const void *)beta, (void *)y, *incy);
}

template <>
void hbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const int *k,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *x, const int *incx, const std::complex<double> *beta,
          std::complex<double> *y, const int *incy) {
    cblas_zhbmv_wrapper(layout, upper_lower, *n, *k, (const void *)alpha, (const void *)a, *lda,
                        (const void *)x, *incx, (const void *)beta, (void *)y, *incy);
}

template <typename fp>
static void hemv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp *alpha,
                 const fp *a, const int *lda, const fp *x, const int *incx, const fp *beta, fp *y,
                 const int *incy);

template <>
void hemv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n,
          const std::complex<float> *alpha, const std::complex<float> *a, const int *lda,
          const std::complex<float> *x, const int *incx, const std::complex<float> *beta,
          std::complex<float> *y, const int *incy) {
    cblas_chemv_wrapper(layout, upper_lower, *n, (const void *)alpha, (const void *)a, *lda,
                        (const void *)x, *incx, (const void *)beta, (void *)y, *incy);
}

template <>
void hemv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n,
          const std::complex<double> *alpha, const std::complex<double> *a, const int *lda,
          const std::complex<double> *x, const int *incx, const std::complex<double> *beta,
          std::complex<double> *y, const int *incy) {
    cblas_zhemv_wrapper(layout, upper_lower, *n, (const void *)alpha, (const void *)a, *lda,
                        (const void *)x, *incx, (const void *)beta, (void *)y, *incy);
}

template <typename fp_scalar, typename fp_data>
static void her(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp_scalar *alpha,
                const fp_data *x, const int *incx, fp_data *a, const int *lda);

template <>
void her(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const float *alpha,
         const std::complex<float> *x, const int *incx, std::complex<float> *a, const int *lda) {
    cblas_cher_wrapper(layout, upper_lower, *n, *alpha, (const void *)x, *incx, (void *)a, *lda);
}

template <>
void her(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const double *alpha,
         const std::complex<double> *x, const int *incx, std::complex<double> *a, const int *lda) {
    cblas_zher_wrapper(layout, upper_lower, *n, *alpha, (const void *)x, *incx, (void *)a, *lda);
}

template <typename fp>
static void her2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp *alpha,
                 const fp *x, const int *incx, const fp *y, const int *incy, fp *a, const int *lda);

template <>
void her2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n,
          const std::complex<float> *alpha, const std::complex<float> *x, const int *incx,
          const std::complex<float> *y, const int *incy, std::complex<float> *a, const int *lda) {
    cblas_cher2_wrapper(layout, upper_lower, *n, (const void *)alpha, (const void *)x, *incx,
                        (const void *)y, *incy, (void *)a, *lda);
}

template <>
void her2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n,
          const std::complex<double> *alpha, const std::complex<double> *x, const int *incx,
          const std::complex<double> *y, const int *incy, std::complex<double> *a, const int *lda) {
    cblas_zher2_wrapper(layout, upper_lower, *n, (const void *)alpha, (const void *)x, *incx,
                        (const void *)y, *incy, (void *)a, *lda);
}

template <typename fp>
static void hpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp *alpha,
                 const fp *a, const fp *x, const int *incx, const fp *beta, fp *y, const int *incy);

template <>
void hpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n,
          const std::complex<float> *alpha, const std::complex<float> *a,
          const std::complex<float> *x, const int *incx, const std::complex<float> *beta,
          std::complex<float> *y, const int *incy) {
    cblas_chpmv_wrapper(layout, upper_lower, *n, (const void *)alpha, (const void *)a,
                        (const void *)x, *incx, (const void *)beta, (void *)y, *incy);
}

template <>
void hpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n,
          const std::complex<double> *alpha, const std::complex<double> *a,
          const std::complex<double> *x, const int *incx, const std::complex<double> *beta,
          std::complex<double> *y, const int *incy) {
    cblas_zhpmv_wrapper(layout, upper_lower, *n, (const void *)alpha, (const void *)a,
                        (const void *)x, *incx, (const void *)beta, (void *)y, *incy);
}

template <typename fp_scalar, typename fp_data>
static void hpr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp_scalar *alpha,
                const fp_data *x, const int *incx, fp_data *a);

template <>
void hpr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const float *alpha,
         const std::complex<float> *x, const int *incx, std::complex<float> *a) {
    cblas_chpr_wrapper(layout, upper_lower, *n, *alpha, (const void *)x, *incx, (void *)a);
}

template <>
void hpr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const double *alpha,
         const std::complex<double> *x, const int *incx, std::complex<double> *a) {
    cblas_zhpr_wrapper(layout, upper_lower, *n, *alpha, (const void *)x, *incx, (void *)a);
}

template <typename fp>
static void hpr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp *alpha,
                 const fp *x, const int *incx, const fp *y, const int *incy, fp *a);

template <>
void hpr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n,
          const std::complex<float> *alpha, const std::complex<float> *x, const int *incx,
          const std::complex<float> *y, const int *incy, std::complex<float> *a) {
    cblas_chpr2_wrapper(layout, upper_lower, *n, (const void *)alpha, (const void *)x, *incx,
                        (const void *)y, *incy, (void *)a);
}

template <>
void hpr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n,
          const std::complex<double> *alpha, const std::complex<double> *x, const int *incx,
          const std::complex<double> *y, const int *incy, std::complex<double> *a) {
    cblas_zhpr2_wrapper(layout, upper_lower, *n, (const void *)alpha, (const void *)x, *incx,
                        (const void *)y, *incy, (void *)a);
}

template <typename fp>
static void sbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const int *k,
                 const fp *alpha, const fp *a, const int *lda, const fp *x, const int *incx,
                 const fp *beta, fp *y, const int *incy);

template <>
void sbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const int *k,
          const float *alpha, const float *a, const int *lda, const float *x, const int *incx,
          const float *beta, float *y, const int *incy) {
    cblas_ssbmv_wrapper(layout, upper_lower, *n, *k, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void sbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const int *k,
          const double *alpha, const double *a, const int *lda, const double *x, const int *incx,
          const double *beta, double *y, const int *incy) {
    cblas_dsbmv_wrapper(layout, upper_lower, *n, *k, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <typename fp>
static void symv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp *alpha,
                 const fp *a, const int *lda, const fp *x, const int *incx, const fp *beta, fp *y,
                 const int *incy);

template <>
void symv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const float *alpha,
          const float *a, const int *lda, const float *x, const int *incx, const float *beta,
          float *y, const int *incy) {
    cblas_ssymv_wrapper(layout, upper_lower, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <>
void symv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const double *alpha,
          const double *a, const int *lda, const double *x, const int *incx, const double *beta,
          double *y, const int *incy) {
    cblas_dsymv_wrapper(layout, upper_lower, *n, *alpha, a, *lda, x, *incx, *beta, y, *incy);
}

template <typename fp>
static void syr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp *alpha,
                const fp *x, const int *incx, fp *a, const int *lda);

template <>
void syr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const float *alpha,
         const float *x, const int *incx, float *a, const int *lda) {
    cblas_ssyr_wrapper(layout, upper_lower, *n, *alpha, x, *incx, a, *lda);
}

template <>
void syr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const double *alpha,
         const double *x, const int *incx, double *a, const int *lda) {
    cblas_dsyr_wrapper(layout, upper_lower, *n, *alpha, x, *incx, a, *lda);
}

template <typename fp>
static void syr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp *alpha,
                 const fp *x, const int *incx, const fp *y, const int *incy, fp *a, const int *lda);

template <>
void syr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const float *alpha,
          const float *x, const int *incx, const float *y, const int *incy, float *a,
          const int *lda) {
    cblas_ssyr2_wrapper(layout, upper_lower, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <>
void syr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const double *alpha,
          const double *x, const int *incx, const double *y, const int *incy, double *a,
          const int *lda) {
    cblas_dsyr2_wrapper(layout, upper_lower, *n, *alpha, x, *incx, y, *incy, a, *lda);
}

template <typename fp>
static void spmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp *alpha,
                 const fp *a, const fp *x, const int *incx, const fp *beta, fp *y, const int *incy);

template <>
void spmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const float *alpha,
          const float *a, const float *x, const int *incx, const float *beta, float *y,
          const int *incy) {
    cblas_sspmv_wrapper(layout, upper_lower, *n, *alpha, a, x, *incx, *beta, y, *incy);
}

template <>
void spmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const double *alpha,
          const double *a, const double *x, const int *incx, const double *beta, double *y,
          const int *incy) {
    cblas_dspmv_wrapper(layout, upper_lower, *n, *alpha, a, x, *incx, *beta, y, *incy);
}

template <typename fp>
static void spr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp *alpha,
                const fp *x, const int *incx, fp *a);

template <>
void spr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const float *alpha,
         const float *x, const int *incx, float *a) {
    cblas_sspr_wrapper(layout, upper_lower, *n, *alpha, x, *incx, a);
}

template <>
void spr(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const double *alpha,
         const double *x, const int *incx, double *a) {
    cblas_dspr_wrapper(layout, upper_lower, *n, *alpha, x, *incx, a);
}

template <typename fp>
static void spr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const fp *alpha,
                 const fp *x, const int *incx, const fp *y, const int *incy, fp *a);

template <>
void spr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const float *alpha,
          const float *x, const int *incx, const float *y, const int *incy, float *a) {
    cblas_sspr2_wrapper(layout, upper_lower, *n, *alpha, x, *incx, y, *incy, a);
}

template <>
void spr2(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, const int *n, const double *alpha,
          const double *x, const int *incx, const double *y, const int *incy, double *a) {
    cblas_dspr2_wrapper(layout, upper_lower, *n, *alpha, x, *incx, y, *incy, a);
}

template <typename fp>
static void tbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int *n, const int *k, const fp *a, const int *lda,
                 fp *x, const int *incx);

template <>
void tbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const int *k, const float *a, const int *lda, float *x, const int *incx) {
    cblas_stbmv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const int *k, const double *a, const int *lda, double *x, const int *incx) {
    cblas_dtbmv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const int *k, const std::complex<float> *a, const int *lda,
          std::complex<float> *x, const int *incx) {
    cblas_ctbmv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, (const void *)a, *lda,
                        (void *)x, *incx);
}

template <>
void tbmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const int *k, const std::complex<double> *a, const int *lda,
          std::complex<double> *x, const int *incx) {
    cblas_ztbmv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, (const void *)a, *lda,
                        (void *)x, *incx);
}

template <typename fp>
static void tbsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int *n, const int *k, const fp *a, const int *lda,
                 fp *x, const int *incx);

template <>
void tbsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const int *k, const float *a, const int *lda, float *x, const int *incx) {
    cblas_stbsv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const int *k, const double *a, const int *lda, double *x, const int *incx) {
    cblas_dtbsv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, a, *lda, x, *incx);
}

template <>
void tbsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const int *k, const std::complex<float> *a, const int *lda,
          std::complex<float> *x, const int *incx) {
    cblas_ctbsv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, (const void *)a, *lda,
                        (void *)x, *incx);
}

template <>
void tbsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const int *k, const std::complex<double> *a, const int *lda,
          std::complex<double> *x, const int *incx) {
    cblas_ztbsv_wrapper(layout, upper_lower, trans, unit_diag, *n, *k, (const void *)a, *lda,
                        (void *)x, *incx);
}

template <typename fp>
static void tpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int *n, const fp *a, fp *x, const int *incx);

template <>
void tpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const float *a, float *x, const int *incx) {
    cblas_stpmv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const double *a, double *x, const int *incx) {
    cblas_dtpmv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const std::complex<float> *a, std::complex<float> *x, const int *incx) {
    cblas_ctpmv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void *)a, (void *)x,
                        *incx);
}

template <>
void tpmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const std::complex<double> *a, std::complex<double> *x, const int *incx) {
    cblas_ztpmv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void *)a, (void *)x,
                        *incx);
}

template <typename fp>
static void tpsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int *n, const fp *a, fp *x, const int *incx);

template <>
void tpsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const float *a, float *x, const int *incx) {
    cblas_stpsv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const double *a, double *x, const int *incx) {
    cblas_dtpsv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, x, *incx);
}

template <>
void tpsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const std::complex<float> *a, std::complex<float> *x, const int *incx) {
    cblas_ctpsv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void *)a, (void *)x,
                        *incx);
}

template <>
void tpsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const std::complex<double> *a, std::complex<double> *x, const int *incx) {
    cblas_ztpsv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void *)a, (void *)x,
                        *incx);
}

template <typename fp>
static void trmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int *n, const fp *a, const int *lda, fp *x,
                 const int *incx);

template <>
void trmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const float *a, const int *lda, float *x, const int *incx) {
    cblas_strmv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const double *a, const int *lda, double *x, const int *incx) {
    cblas_dtrmv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const std::complex<float> *a, const int *lda, std::complex<float> *x,
          const int *incx) {
    cblas_ctrmv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void *)a, *lda, (void *)x,
                        *incx);
}

template <>
void trmv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const std::complex<double> *a, const int *lda, std::complex<double> *x,
          const int *incx) {
    cblas_ztrmv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void *)a, *lda, (void *)x,
                        *incx);
}

template <typename fp>
static void trsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans,
                 CBLAS_DIAG unit_diag, const int *n, const fp *a, const int *lda, fp *x,
                 const int *incx);

template <>
void trsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const float *a, const int *lda, float *x, const int *incx) {
    cblas_strsv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const double *a, const int *lda, double *x, const int *incx) {
    cblas_dtrsv_wrapper(layout, upper_lower, trans, unit_diag, *n, a, *lda, x, *incx);
}

template <>
void trsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const std::complex<float> *a, const int *lda, std::complex<float> *x,
          const int *incx) {
    cblas_ctrsv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void *)a, *lda, (void *)x,
                        *incx);
}

template <>
void trsv(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE trans, CBLAS_DIAG unit_diag,
          const int *n, const std::complex<double> *a, const int *lda, std::complex<double> *x,
          const int *incx) {
    cblas_ztrsv_wrapper(layout, upper_lower, trans, unit_diag, *n, (const void *)a, *lda, (void *)x,
                        *incx);
}

/* Level 1 */

template <typename fp_data, typename fp_res>
static fp_res asum(const int *n, const fp_data *x, const int *incx);

template <>
float asum(const int *n, const float *x, const int *incx) {
    return cblas_sasum_wrapper(*n, x, *incx);
}

template <>
double asum(const int *n, const double *x, const int *incx) {
    return cblas_dasum_wrapper(*n, x, *incx);
}

template <>
float asum(const int *n, const std::complex<float> *x, const int *incx) {
    return cblas_scasum_wrapper(*n, (const void *)x, *incx);
}

template <>
double asum(const int *n, const std::complex<double> *x, const int *incx) {
    return cblas_dzasum_wrapper(*n, (const void *)x, *incx);
}

template <typename fp>
static void axpy(const int *n, const fp *alpha, const fp *x, const int *incx, fp *y,
                 const int *incy);

template <>
void axpy(const int *n, const float *alpha, const float *x, const int *incx, float *y,
          const int *incy) {
    cblas_saxpy_wrapper(*n, *alpha, x, *incx, y, *incy);
}

template <>
void axpy(const int *n, const double *alpha, const double *x, const int *incx, double *y,
          const int *incy) {
    cblas_daxpy_wrapper(*n, *alpha, x, *incx, y, *incy);
}

template <>
void axpy(const int *n, const std::complex<float> *alpha, const std::complex<float> *x,
          const int *incx, std::complex<float> *y, const int *incy) {
    cblas_caxpy_wrapper(*n, (const void *)alpha, (const void *)x, *incx, (void *)y, *incy);
}

template <>
void axpy(const int *n, const std::complex<double> *alpha, const std::complex<double> *x,
          const int *incx, std::complex<double> *y, const int *incy) {
    cblas_zaxpy_wrapper(*n, (const void *)alpha, (const void *)x, *incx, (void *)y, *incy);
}

template <typename fp>
static void copy(const int *n, const fp *x, const int *incx, fp *y, const int *incy);

template <>
void copy(const int *n, const float *x, const int *incx, float *y, const int *incy) {
    cblas_scopy_wrapper(*n, x, *incx, y, *incy);
}
template <>
void copy(const int *n, const double *x, const int *incx, double *y, const int *incy) {
    cblas_dcopy_wrapper(*n, x, *incx, y, *incy);
}
template <>
void copy(const int *n, const std::complex<float> *x, const int *incx, std::complex<float> *y,
          const int *incy) {
    cblas_ccopy_wrapper(*n, (const void *)x, *incx, (void *)y, *incy);
}
template <>
void copy(const int *n, const std::complex<double> *x, const int *incx, std::complex<double> *y,
          const int *incy) {
    cblas_zcopy_wrapper(*n, (const void *)x, *incx, (void *)y, *incy);
}

template <typename fp, typename fp_res>
static fp_res dot(const int *n, const fp *x, const int *incx, const fp *y, const int *incy);

template <>
float dot(const int *n, const float *x, const int *incx, const float *y, const int *incy) {
    return cblas_sdot_wrapper(*n, x, *incx, y, *incy);
}

template <>
double dot(const int *n, const double *x, const int *incx, const double *y, const int *incy) {
    return cblas_ddot_wrapper(*n, x, *incx, y, *incy);
}

template <>
double dot(const int *n, const float *x, const int *incx, const float *y, const int *incy) {
    return cblas_dsdot_wrapper(*n, x, *incx, y, *incy);
}

static float sdsdot(const int *n, const float *sb, const float *x, const int *incx, const float *y,
                    const int *incy) {
    return cblas_sdsdot_wrapper(*n, *sb, x, *incx, y, *incy);
}

template <typename fp, typename fp_res>
static fp_res nrm2(const int *n, const fp *x, const int *incx);

template <>
float nrm2(const int *n, const float *x, const int *incx) {
    return cblas_snrm2_wrapper(*n, x, *incx);
}

template <>
double nrm2(const int *n, const double *x, const int *incx) {
    return cblas_dnrm2_wrapper(*n, x, *incx);
}

template <>
float nrm2(const int *n, const std::complex<float> *x, const int *incx) {
    return cblas_scnrm2_wrapper(*n, (const void *)x, *incx);
}

template <>
double nrm2(const int *n, const std::complex<double> *x, const int *incx) {
    return cblas_dznrm2_wrapper(*n, (const void *)x, *incx);
}

template <typename fp, typename fp_scalar>
static void rot(const int *n, fp *x, const int *incx, fp *y, const int *incy, const fp_scalar *c,
                const fp_scalar *s);

template <>
void rot(const int *n, float *x, const int *incx, float *y, const int *incy, const float *c,
         const float *s) {
    cblas_srot_wrapper(*n, x, *incx, y, *incy, *c, *s);
}

template <>
void rot(const int *n, double *x, const int *incx, double *y, const int *incy, const double *c,
         const double *s) {
    cblas_drot_wrapper(*n, x, *incx, y, *incy, *c, *s);
}

template <>
void rot(const int *n, std::complex<float> *x, const int *incx, std::complex<float> *y,
         const int *incy, const float *c, const float *s) {
    csrot_wrapper(n, (void *)x, incx, (void *)y, incy, c, s);
}

template <>
void rot(const int *n, std::complex<double> *x, const int *incx, std::complex<double> *y,
         const int *incy, const double *c, const double *s) {
    zdrot_wrapper(n, (void *)x, incx, (void *)y, incy, c, s);
}

template <typename fp, typename fp_c>
static void rotg(fp *a, fp *b, fp_c *c, fp *s);

template <>
void rotg(float *a, float *b, float *c, float *s) {
    cblas_srotg_wrapper(a, b, c, s);
}

template <>
void rotg(double *a, double *b, double *c, double *s) {
    cblas_drotg_wrapper(a, b, c, s);
}

template <>
void rotg(std::complex<float> *a, std::complex<float> *b, float *c, std::complex<float> *s) {
    crotg_wrapper((void *)a, (void *)b, c, (void *)s);
}

template <>
void rotg(std::complex<double> *a, std::complex<double> *b, double *c, std::complex<double> *s) {
    zrotg_wrapper((void *)a, (void *)b, c, (void *)s);
}

template <typename fp>
static void rotm(const int *n, fp *x, const int *incx, fp *y, const int *incy, const fp *param);

template <>
void rotm(const int *n, float *x, const int *incx, float *y, const int *incy, const float *param) {
    cblas_srotm_wrapper(*n, x, *incx, y, *incy, param);
}

template <>
void rotm(const int *n, double *x, const int *incx, double *y, const int *incy,
          const double *param) {
    cblas_drotm_wrapper(*n, x, *incx, y, *incy, param);
}

template <typename fp>
static void rotmg(fp *d1, fp *d2, fp *x1, fp *y1, fp *param);

template <>
void rotmg(float *d1, float *d2, float *x1, float *y1, float *param) {
    cblas_srotmg_wrapper(d1, d2, x1, *y1, param);
}

template <>
void rotmg(double *d1, double *d2, double *x1, double *y1, double *param) {
    cblas_drotmg_wrapper(d1, d2, x1, *y1, param);
}

template <typename fp_scalar, typename fp_data>
static void scal(const int *n, const fp_scalar *alpha, fp_data *x, const int *incx);

template <>
void scal(const int *n, const float *alpha, float *x, const int *incx) {
    cblas_sscal_wrapper(*n, *alpha, x, *incx);
}
template <>
void scal(const int *n, const double *alpha, double *x, const int *incx) {
    cblas_dscal_wrapper(*n, *alpha, x, *incx);
}
template <>
void scal(const int *n, const std::complex<float> *alpha, std::complex<float> *x, const int *incx) {
    cblas_cscal_wrapper(*n, (const void *)alpha, (void *)x, *incx);
}
template <>
void scal(const int *n, const std::complex<double> *alpha, std::complex<double> *x,
          const int *incx) {
    cblas_zscal_wrapper(*n, (const void *)alpha, (void *)x, *incx);
}
template <>
void scal(const int *n, const float *alpha, std::complex<float> *x, const int *incx) {
    cblas_csscal_wrapper(*n, *alpha, (void *)x, *incx);
}
template <>
void scal(const int *n, const double *alpha, std::complex<double> *x, const int *incx) {
    cblas_zdscal_wrapper(*n, *alpha, (void *)x, *incx);
}

template <typename fp>
static void swap(const int *n, fp *x, const int *incx, fp *y, const int *incy);

template <>
void swap(const int *n, float *x, const int *incx, float *y, const int *incy) {
    cblas_sswap_wrapper(*n, x, *incx, y, *incy);
}

template <>
void swap(const int *n, double *x, const int *incx, double *y, const int *incy) {
    cblas_dswap_wrapper(*n, x, *incx, y, *incy);
}

template <>
void swap(const int *n, std::complex<float> *x, const int *incx, std::complex<float> *y,
          const int *incy) {
    cblas_cswap_wrapper(*n, (void *)x, *incx, (void *)y, *incy);
}

template <>
void swap(const int *n, std::complex<double> *x, const int *incx, std::complex<double> *y,
          const int *incy) {
    cblas_zswap_wrapper(*n, (void *)x, *incx, (void *)y, *incy);
}

template <typename fp>
static void dotc(fp *pres, const int *n, const fp *x, const int *incx, const fp *y,
                 const int *incy);

template <>
void dotc(std::complex<float> *pres, const int *n, const std::complex<float> *x, const int *incx,
          const std::complex<float> *y, const int *incy) {
    cblas_cdotc_sub_wrapper(*n, (const void *)x, *incx, (const void *)y, *incy, (void *)pres);
}

template <>
void dotc(std::complex<double> *pres, const int *n, const std::complex<double> *x, const int *incx,
          const std::complex<double> *y, const int *incy) {
    cblas_zdotc_sub_wrapper(*n, (const void *)x, *incx, (const void *)y, *incy, (void *)pres);
}

template <typename fp>
static void dotu(fp *pres, const int *n, const fp *x, const int *incx, const fp *y,
                 const int *incy);

template <>
void dotu(std::complex<float> *pres, const int *n, const std::complex<float> *x, const int *incx,
          const std::complex<float> *y, const int *incy) {
    cblas_cdotu_sub_wrapper(*n, (const void *)x, *incx, (const void *)y, *incy, (void *)pres);
}

template <>
void dotu(std::complex<double> *pres, const int *n, const std::complex<double> *x, const int *incx,
          const std::complex<double> *y, const int *incy) {
    cblas_zdotu_sub_wrapper(*n, (const void *)x, *incx, (const void *)y, *incy, (void *)pres);
}

template <typename fp>
static int iamax(const int *n, const fp *x, const int *incx);

template <>
int iamax(const int *n, const float *x, const int *incx) {
    return cblas_isamax_wrapper(*n, x, *incx);
}

template <>
int iamax(const int *n, const double *x, const int *incx) {
    return cblas_idamax_wrapper(*n, x, *incx);
}

template <>
int iamax(const int *n, const std::complex<float> *x, const int *incx) {
    return cblas_icamax_wrapper(*n, (const void *)x, *incx);
}

template <>
int iamax(const int *n, const std::complex<double> *x, const int *incx) {
    return cblas_izamax_wrapper(*n, (const void *)x, *incx);
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
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < *n; ++logical_i) {
        int i = logical_i * std::abs(*incx);
        auto curr_val = abs_val(x[i]);
        if (sycl::isnan(curr_val))
            return logical_i;
        if (curr_val < min_val) {
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
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < *n; ++logical_i) {
        int i = logical_i * std::abs(*incx);
        auto curr_val = abs_val(x[i]);
        if (sycl::isnan(curr_val))
            return logical_i;
        if (curr_val < min_val) {
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
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < *n; ++logical_i) {
        int i = logical_i * std::abs(*incx);
        auto curr_val = abs_val(x[i]);
        if (sycl::isnan(curr_val))
            return logical_i;
        if (curr_val < min_val) {
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
    int min_idx = 0;
    auto min_val = abs_val(x[0]);
    if (sycl::isnan(min_val))
        return 0;

    for (int logical_i = 1; logical_i < *n; ++logical_i) {
        int i = logical_i * std::abs(*incx);
        auto curr_val = abs_val(x[i]);
        if (sycl::isnan(curr_val))
            return logical_i;
        if (curr_val < min_val) {
            min_idx = logical_i;
            min_val = curr_val;
        }
    }
    return min_idx;
}

/* Extensions */

template <typename fp>
static void axpby(const int *n, const fp *alpha, const fp *x, const int *incx, const fp *beta,
                  fp *y, const int *incy);

template <>
void axpby(const int *n, const float *alpha, const float *x, const int *incx, const float *beta,
           float *y, const int *incy) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    int idx = (*incx) > 0 ? 0 : (1 - *n) * (*incx);
    int idy = (*incy) > 0 ? 0 : (1 - *n) * (*incy);
    for (int i = 0; i < *n; i++)
        y[idy + i * (*incy)] = *alpha * x[idx + i * (*incx)] + (*beta) * y[idy + i * (*incy)];
}

template <>
void axpby(const int *n, const double *alpha, const double *x, const int *incx, const double *beta,
           double *y, const int *incy) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    int idx = (*incx) > 0 ? 0 : (1 - *n) * (*incx);
    int idy = (*incy) > 0 ? 0 : (1 - *n) * (*incy);
    for (int i = 0; i < *n; i++)
        y[idy + i * (*incy)] = *alpha * x[idx + i * (*incx)] + (*beta) * y[idy + i * (*incy)];
}

template <>
void axpby(const int *n, const std::complex<float> *alpha, const std::complex<float> *x,
           const int *incx, const std::complex<float> *beta, std::complex<float> *y,
           const int *incy) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    int idx = (*incx) > 0 ? 0 : (1 - *n) * (*incx);
    int idy = (*incy) > 0 ? 0 : (1 - *n) * (*incy);
    for (int i = 0; i < *n; i++)
        y[idy + i * (*incy)] = *alpha * x[idx + i * (*incx)] + (*beta) * y[idy + i * (*incy)];
}

template <>
void axpby(const int *n, const std::complex<double> *alpha, const std::complex<double> *x,
           const int *incx, const std::complex<double> *beta, std::complex<double> *y,
           const int *incy) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    int idx = (*incx) > 0 ? 0 : (1 - *n) * (*incx);
    int idy = (*incy) > 0 ? 0 : (1 - *n) * (*incy);
    for (int i = 0; i < *n; i++)
        y[idy + i * (*incy)] = *alpha * x[idx + i * (*incx)] + (*beta) * y[idy + i * (*incy)];
}

template <typename fps, typename fpa, typename fpb, typename fpc>
static void gemm_bias(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
                      CBLAS_OFFSET offsetc, const int *m, const int *n, const int *k,
                      const fps *alpha, const fpa *a, const int *lda, const fpa *ao, const fpb *b,
                      const int *ldb, const fpb *bo, const fps *beta, fpc *c, const int *ldc,
                      const fpc *co);

template <>
void gemm_bias(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
               CBLAS_OFFSET offsetc, const int *m, const int *n, const int *k, const float *alpha,
               const int8_t *a, const int *lda, const int8_t *ao, const int8_t *b, const int *ldb,
               const int8_t *bo, const float *beta, int32_t *c, const int *ldc, const int32_t *co) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizea, sizeb, sizec;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
        sizec = *ldc * *n;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
        sizec = *ldc * *m;
    }
    double *ad = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizea);
    double *bd = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizeb);
    double *cd = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizec);
    double alphad = *alpha;
    double betad = *beta;
    double aod = *ao;
    double bod = *bo;
    copy_mat(a, layout, transa, *m, *k, *lda, aod, ad);
    copy_mat(b, layout, transb, *k, *n, *ldb, bod, bd);
    copy_mat(c, layout, CblasNoTrans, *m, *n, *ldc, 0.0, cd);
    cblas_dgemm_wrapper(layout, transa, transb, *m, *n, *k, alphad, ad, *lda, bd, *ldb, betad, cd,
                        *ldc);
    copy_mat(cd, layout, *m, *n, *ldc, offsetc, co, c);
    oneapi::mkl::aligned_free(ad);
    oneapi::mkl::aligned_free(bd);
    oneapi::mkl::aligned_free(cd);
}

template <>
void gemm_bias(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
               CBLAS_OFFSET offsetc, const int *m, const int *n, const int *k, const float *alpha,
               const int8_t *a, const int *lda, const int8_t *ao, const uint8_t *b, const int *ldb,
               const uint8_t *bo, const float *beta, int32_t *c, const int *ldc,
               const int32_t *co) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizea, sizeb, sizec;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
        sizec = *ldc * *n;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
        sizec = *ldc * *m;
    }
    double *ad = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizea);
    double *bd = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizeb);
    double *cd = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizec);
    double alphad = *alpha;
    double betad = *beta;
    double aod = *ao;
    double bod = *bo;
    copy_mat(a, layout, transa, *m, *k, *lda, aod, ad);
    copy_mat(b, layout, transb, *k, *n, *ldb, bod, bd);
    copy_mat(c, layout, CblasNoTrans, *m, *n, *ldc, 0.0, cd);
    cblas_dgemm_wrapper(layout, transa, transb, *m, *n, *k, alphad, ad, *lda, bd, *ldb, betad, cd,
                        *ldc);
    copy_mat(cd, layout, *m, *n, *ldc, offsetc, co, c);
    oneapi::mkl::aligned_free(ad);
    oneapi::mkl::aligned_free(bd);
    oneapi::mkl::aligned_free(cd);
}

template <>
void gemm_bias(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
               CBLAS_OFFSET offsetc, const int *m, const int *n, const int *k, const float *alpha,
               const uint8_t *a, const int *lda, const uint8_t *ao, const int8_t *b, const int *ldb,
               const int8_t *bo, const float *beta, int32_t *c, const int *ldc, const int32_t *co) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizea, sizeb, sizec;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
        sizec = *ldc * *n;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
        sizec = *ldc * *m;
    }
    double *ad = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizea);
    double *bd = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizeb);
    double *cd = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizec);
    double alphad = *alpha;
    double betad = *beta;
    double aod = *ao;
    double bod = *bo;
    copy_mat(a, layout, transa, *m, *k, *lda, aod, ad);
    copy_mat(b, layout, transb, *k, *n, *ldb, bod, bd);
    copy_mat(c, layout, CblasNoTrans, *m, *n, *ldc, 0.0, cd);
    cblas_dgemm_wrapper(layout, transa, transb, *m, *n, *k, alphad, ad, *lda, bd, *ldb, betad, cd,
                        *ldc);
    copy_mat(cd, layout, *m, *n, *ldc, offsetc, co, c);
    oneapi::mkl::aligned_free(ad);
    oneapi::mkl::aligned_free(bd);
    oneapi::mkl::aligned_free(cd);
}

template <>
void gemm_bias(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb,
               CBLAS_OFFSET offsetc, const int *m, const int *n, const int *k, const float *alpha,
               const uint8_t *a, const int *lda, const uint8_t *ao, const uint8_t *b,
               const int *ldb, const uint8_t *bo, const float *beta, int32_t *c, const int *ldc,
               const int32_t *co) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizea, sizeb, sizec;
    if (layout == CblasColMajor) {
        sizea = (transa == CblasNoTrans) ? *lda * *k : *lda * *m;
        sizeb = (transb == CblasNoTrans) ? *ldb * *n : *ldb * *k;
        sizec = *ldc * *n;
    }
    else {
        sizea = (transa == CblasNoTrans) ? *lda * *m : *lda * *k;
        sizeb = (transb == CblasNoTrans) ? *ldb * *k : *ldb * *n;
        sizec = *ldc * *m;
    }
    double *ad = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizea);
    double *bd = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizeb);
    double *cd = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizec);
    double alphad = *alpha;
    double betad = *beta;
    double aod = *ao;
    double bod = *bo;
    copy_mat(a, layout, transa, *m, *k, *lda, aod, ad);
    copy_mat(b, layout, transb, *k, *n, *ldb, bod, bd);
    copy_mat(c, layout, CblasNoTrans, *m, *n, *ldc, 0.0, cd);
    cblas_dgemm_wrapper(layout, transa, transb, *m, *n, *k, alphad, ad, *lda, bd, *ldb, betad, cd,
                        *ldc);
    copy_mat(cd, layout, *m, *n, *ldc, offsetc, co, c);
    oneapi::mkl::aligned_free(ad);
    oneapi::mkl::aligned_free(bd);
    oneapi::mkl::aligned_free(cd);
}

template <typename fp>
static void gemmt(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa,
                  CBLAS_TRANSPOSE transb, const int *n, const int *k, const fp *alpha, const fp *a,
                  const int *lda, const fp *b, const int *ldb, const fp *beta, fp *c,
                  const int *ldc);

template <>
void gemmt(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa,
           CBLAS_TRANSPOSE transb, const int *n, const int *k, const float *alpha, const float *a,
           const int *lda, const float *b, const int *ldb, const float *beta, float *c,
           const int *ldc) {
    // Not supported in NETLIB. SGEMM is used as reference.
    int sizec;
    sizec = *ldc * *n;
    float *cf = (float *)oneapi::mkl::aligned_alloc(64, sizeof(float) * sizec);
    update_c(c, layout, upper_lower, *n, *n, *ldc, cf);
    cblas_sgemm_wrapper(layout, transa, transb, *n, *n, *k, *alpha, a, *lda, b, *ldb, *beta, cf,
                        *ldc);
    update_c(cf, layout, upper_lower, *n, *n, *ldc, c);
    oneapi::mkl::aligned_free(cf);
}

template <>
void gemmt(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa,
           CBLAS_TRANSPOSE transb, const int *n, const int *k, const double *alpha, const double *a,
           const int *lda, const double *b, const int *ldb, const double *beta, double *c,
           const int *ldc) {
    // Not supported in NETLIB. DGEMM is used as reference.
    int sizec;
    sizec = *ldc * *n;
    double *cf = (double *)oneapi::mkl::aligned_alloc(64, sizeof(double) * sizec);
    update_c(c, layout, upper_lower, *n, *n, *ldc, cf);
    cblas_dgemm_wrapper(layout, transa, transb, *n, *n, *k, *alpha, a, *lda, b, *ldb, *beta, cf,
                        *ldc);
    update_c(cf, layout, upper_lower, *n, *n, *ldc, c);
    oneapi::mkl::aligned_free(cf);
}

template <>
void gemmt(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa,
           CBLAS_TRANSPOSE transb, const int *n, const int *k, const std::complex<float> *alpha,
           const std::complex<float> *a, const int *lda, const std::complex<float> *b,
           const int *ldb, const std::complex<float> *beta, std::complex<float> *c,
           const int *ldc) {
    // Not supported in NETLIB. CGEMM is used as reference.
    int sizec;
    sizec = *ldc * *n;
    std::complex<float> *cf =
        (std::complex<float> *)oneapi::mkl::aligned_alloc(64, sizeof(std::complex<float>) * sizec);
    update_c(c, layout, upper_lower, *n, *n, *ldc, cf);
    cblas_cgemm_wrapper(layout, transa, transb, *n, *n, *k, alpha, a, *lda, b, *ldb, beta, cf,
                        *ldc);
    update_c(cf, layout, upper_lower, *n, *n, *ldc, c);
    oneapi::mkl::aligned_free(cf);
}

template <>
void gemmt(CBLAS_LAYOUT layout, CBLAS_UPLO upper_lower, CBLAS_TRANSPOSE transa,
           CBLAS_TRANSPOSE transb, const int *n, const int *k, const std::complex<double> *alpha,
           const std::complex<double> *a, const int *lda, const std::complex<double> *b,
           const int *ldb, const std::complex<double> *beta, std::complex<double> *c,
           const int *ldc) {
    // Not supported in NETLIB. ZGEMM is used as reference.
    int sizec;
    sizec = *ldc * *n;
    std::complex<double> *cf = (std::complex<double> *)oneapi::mkl::aligned_alloc(
        64, sizeof(std::complex<double>) * sizec);
    update_c(c, layout, upper_lower, *n, *n, *ldc, cf);
    cblas_zgemm_wrapper(layout, transa, transb, *n, *n, *k, alpha, a, *lda, b, *ldb, beta, cf,
                        *ldc);
    update_c(cf, layout, upper_lower, *n, *n, *ldc, c);
    oneapi::mkl::aligned_free(cf);
}

template <typename fp>
static void dgmm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, const int *m, const int *n,
                 const fp *a, const int *lda, const fp *x, const int *incx, fp *c, const int *ldc);

template <>
void dgmm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, const int *m, const int *n, const float *a,
          const int *lda, const float *x, const int *incx, float *c, const int *ldc) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    float tmp;
    int size_x = (left_right == CblasLeft) ? *m : *n;
    int idx = (*incx) > 0 ? 0 : (1 - size_x) * (*incx);

    if (left_right == CblasRight) {
        for (int i = 0; i < *n; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *m; j++) {
                if (layout == CblasColMajor)
                    c[j + i * (*ldc)] = tmp * a[j + i * (*lda)];
                else
                    c[i + j * (*ldc)] = tmp * a[i + j * (*lda)];
            }
        }
    }
    else {
        for (int i = 0; i < *m; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *n; j++) {
                if (layout == CblasColMajor)
                    c[i + j * (*ldc)] = tmp * a[i + j * (*lda)];
                else
                    c[j + i * (*ldc)] = tmp * a[j + i * (*lda)];
            }
        }
    }
}

template <>
void dgmm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, const int *m, const int *n, const double *a,
          const int *lda, const double *x, const int *incx, double *c, const int *ldc) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    double tmp;
    int size_x = (left_right == CblasLeft) ? *m : *n;
    int idx = (*incx) > 0 ? 0 : (1 - size_x) * (*incx);

    if (left_right == CblasRight) {
        for (int i = 0; i < *n; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *m; j++) {
                if (layout == CblasColMajor)
                    c[j + i * (*ldc)] = tmp * a[j + i * (*lda)];
                else
                    c[i + j * (*ldc)] = tmp * a[i + j * (*lda)];
            }
        }
    }
    else {
        for (int i = 0; i < *m; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *n; j++) {
                if (layout == CblasColMajor)
                    c[i + j * (*ldc)] = tmp * a[i + j * (*lda)];
                else
                    c[j + i * (*ldc)] = tmp * a[j + i * (*lda)];
            }
        }
    }
}

template <>
void dgmm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, const int *m, const int *n,
          const std::complex<float> *a, const int *lda, const std::complex<float> *x,
          const int *incx, std::complex<float> *c, const int *ldc) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    std::complex<float> tmp;
    int size_x = (left_right == CblasLeft) ? *m : *n;
    int idx = (*incx) > 0 ? 0 : (1 - size_x) * (*incx);

    if (left_right == CblasRight) {
        for (int i = 0; i < *n; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *m; j++) {
                if (layout == CblasColMajor) {
                    c[j + i * (*ldc)] =
                        std::complex<float>((tmp.real() * a[j + i * (*lda)].real() -
                                             tmp.imag() * a[j + i * (*lda)].imag()),
                                            (tmp.real() * a[j + i * (*lda)].imag() +
                                             tmp.imag() * a[j + i * (*lda)].real()));
                }
                else {
                    c[i + j * (*ldc)] =
                        std::complex<float>((tmp.real() * a[i + j * (*lda)].real() -
                                             tmp.imag() * a[i + j * (*lda)].imag()),
                                            (tmp.real() * a[i + j * (*lda)].imag() +
                                             tmp.imag() * a[i + j * (*lda)].real()));
                }
            }
        }
    }
    else {
        for (int i = 0; i < *m; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *n; j++) {
                if (layout == CblasColMajor) {
                    c[i + j * (*ldc)] =
                        std::complex<float>((tmp.real() * a[i + j * (*lda)].real() -
                                             tmp.imag() * a[i + j * (*lda)].imag()),
                                            (tmp.real() * a[i + j * (*lda)].imag() +
                                             tmp.imag() * a[i + j * (*lda)].real()));
                }
                else {
                    c[j + i * (*ldc)] =
                        std::complex<float>((tmp.real() * a[j + i * (*lda)].real() -
                                             tmp.imag() * a[j + i * (*lda)].imag()),
                                            (tmp.real() * a[j + i * (*lda)].imag() +
                                             tmp.imag() * a[j + i * (*lda)].real()));
                }
            }
        }
    }
}

template <>
void dgmm(CBLAS_LAYOUT layout, CBLAS_SIDE left_right, const int *m, const int *n,
          const std::complex<double> *a, const int *lda, const std::complex<double> *x,
          const int *incx, std::complex<double> *c, const int *ldc) {
    // Not supported in NETLIB. Reference C++ implementation is used.
    std::complex<double> tmp;
    int size_x = (left_right == CblasLeft) ? *m : *n;
    int idx = (*incx) > 0 ? 0 : (1 - size_x) * (*incx);

    if (left_right == CblasRight) {
        for (int i = 0; i < *n; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *m; j++) {
                if (layout == CblasColMajor) {
                    c[j + i * (*ldc)] =
                        std::complex<double>((tmp.real() * a[j + i * (*lda)].real() -
                                              tmp.imag() * a[j + i * (*lda)].imag()),
                                             (tmp.real() * a[j + i * (*lda)].imag() +
                                              tmp.imag() * a[j + i * (*lda)].real()));
                }
                else {
                    c[i + j * (*ldc)] =
                        std::complex<double>((tmp.real() * a[i + j * (*lda)].real() -
                                              tmp.imag() * a[i + j * (*lda)].imag()),
                                             (tmp.real() * a[i + j * (*lda)].imag() +
                                              tmp.imag() * a[i + j * (*lda)].real()));
                }
            }
        }
    }
    else {
        for (int i = 0; i < *m; i++) {
            tmp = x[idx + i * (*incx)];
            for (int j = 0; j < *n; j++) {
                if (layout == CblasColMajor) {
                    c[i + j * (*ldc)] =
                        std::complex<double>((tmp.real() * a[i + j * (*lda)].real() -
                                              tmp.imag() * a[i + j * (*lda)].imag()),
                                             (tmp.real() * a[i + j * (*lda)].imag() +
                                              tmp.imag() * a[i + j * (*lda)].real()));
                }
                else {
                    c[j + i * (*ldc)] =
                        std::complex<double>((tmp.real() * a[j + i * (*lda)].real() -
                                              tmp.imag() * a[j + i * (*lda)].imag()),
                                             (tmp.real() * a[j + i * (*lda)].imag() +
                                              tmp.imag() * a[j + i * (*lda)].real()));
                }
            }
        }
    }
}

// std::conj can take a real type as input, but still returns a complex type.
// This version always returns the same type it has as input
template <typename fp>
fp sametype_conj(fp x) {
    if constexpr (std::is_same_v<fp, std::complex<float>> ||
                  std::is_same_v<fp, std::complex<double>>) {
        return std::conj(x);
    }
    else {
        return x;
    }
}

template <typename fp>
void omatcopy_ref(oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int64_t m, int64_t n,
                  fp alpha, fp *A, int64_t lda, fp *B, int64_t ldb) {
    int64_t logical_m, logical_n;
    if (layout == oneapi::mkl::layout::column_major) {
        logical_m = m;
        logical_n = n;
    }
    else {
        logical_m = n;
        logical_n = m;
    }
    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                B[j * ldb + i] = alpha * A[j * lda + i];
            }
        }
    }
    else if (trans == oneapi::mkl::transpose::trans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                B[i * ldb + j] = alpha * A[j * lda + i];
            }
        }
    }
    else {
        // conjtrans
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                B[i * ldb + j] = alpha * sametype_conj(A[j * lda + i]);
            }
        }
    }
}

template <typename fp>
void imatcopy_ref(oneapi::mkl::layout layout, oneapi::mkl::transpose trans, int64_t m, int64_t n,
                  fp alpha, fp *A, int64_t lda, int64_t ldb) {
    int64_t logical_m, logical_n;
    if (layout == oneapi::mkl::layout::column_major) {
        logical_m = m;
        logical_n = n;
    }
    else {
        logical_m = n;
        logical_n = m;
    }
    std::vector<fp> temp(m * n);
    int64_t ld_temp = (trans == oneapi::mkl::transpose::nontrans ? logical_m : logical_n);

    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                temp[j * ld_temp + i] = alpha * A[j * lda + i];
            }
        }
    }
    else if (trans == oneapi::mkl::transpose::trans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                temp[i * ld_temp + j] = alpha * A[j * lda + i];
            }
        }
    }
    else {
        // conjtrans
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                temp[i * ld_temp + j] = alpha * sametype_conj(A[j * lda + i]);
            }
        }
    }

    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                A[j * ldb + i] = temp[j * ld_temp + i];
            }
        }
    }
    else {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                A[i * ldb + j] = temp[i * ld_temp + j];
            }
        }
    }
}

template <typename fp>
void omatadd_ref(oneapi::mkl::layout layout, oneapi::mkl::transpose transa,
                 oneapi::mkl::transpose transb, int64_t m, int64_t n, fp alpha, fp *A, int64_t lda,
                 fp beta, fp *B, int64_t ldb, fp *C, int64_t ldc) {
    int64_t logical_m, logical_n;
    if (layout == oneapi::mkl::layout::column_major) {
        logical_m = m;
        logical_n = n;
    }
    else {
        logical_m = n;
        logical_n = m;
    }

    for (int64_t j = 0; j < logical_n; j++) {
        for (int64_t i = 0; i < logical_m; i++) {
            C[j * ldc + i] = 0.0;
        }
    }

    if (transa == oneapi::mkl::transpose::nontrans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += alpha * A[j * lda + i];
            }
        }
    }
    else if (transa == oneapi::mkl::transpose::trans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += alpha * A[i * lda + j];
            }
        }
    }
    else {
        // conjtrans
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += alpha * sametype_conj(A[i * lda + j]);
            }
        }
    }

    if (transb == oneapi::mkl::transpose::nontrans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += beta * B[j * ldb + i];
            }
        }
    }
    else if (transa == oneapi::mkl::transpose::trans) {
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += beta * B[i * ldb + j];
            }
        }
    }
    else {
        // conjtrans
        for (int64_t j = 0; j < logical_n; j++) {
            for (int64_t i = 0; i < logical_m; i++) {
                C[j * ldc + i] += beta * sametype_conj(B[i * ldb + j]);
            }
        }
    }
}

#endif /* header guard */
