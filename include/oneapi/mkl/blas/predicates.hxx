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

// Buffer APIs

inline void herk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, float alpha,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              float beta, sycl::buffer<std::complex<float>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void herk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, float alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               float beta, sycl::buffer<std::complex<float>, 1> &c,
                               std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void herk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, double alpha,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              double beta, sycl::buffer<std::complex<double>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void herk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, double alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               double beta, sycl::buffer<std::complex<double>, 1> &c,
                               std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, float alpha,
                              sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, float alpha,
                               sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, double alpha,
                              sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, double alpha,
                               sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, float alpha,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, float alpha,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, double alpha,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, double alpha,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
                              std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
                               std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
                              std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
                               std::int64_t lda, sycl::buffer<double, 1> &x,
                               std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
                              sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
                               sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
                              sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
                               sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1> &a,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &a,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1> &a,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &a,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             sycl::buffer<float, 1> &x, std::int64_t incx,
                             sycl::buffer<float, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              sycl::buffer<float, 1> &x, std::int64_t incx,
                              sycl::buffer<float, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             sycl::buffer<double, 1> &x, std::int64_t incx,
                             sycl::buffer<double, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                    sycl::buffer<float, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, sycl::buffer<float, 1> &b,
                                    std::int64_t ldb, std::int64_t stride_b, float beta,
                                    sycl::buffer<float, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                     sycl::buffer<float, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, sycl::buffer<float, 1> &b,
                                     std::int64_t ldb, std::int64_t stride_b, float beta,
                                     sycl::buffer<float, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                                    sycl::buffer<double, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, sycl::buffer<double, 1> &b,
                                    std::int64_t ldb, std::int64_t stride_b, double beta,
                                    sycl::buffer<double, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                                     sycl::buffer<double, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, sycl::buffer<double, 1> &b,
                                     std::int64_t ldb, std::int64_t stride_b, double beta,
                                     sycl::buffer<double, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k,
                                    std::complex<float> alpha,
                                    sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a,
                                    sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                    std::int64_t stride_b, std::complex<float> beta,
                                    sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k,
                                     std::complex<float> alpha,
                                     sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::complex<float> beta,
                                     sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k,
                                    std::complex<double> alpha,
                                    sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a,
                                    sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                    std::int64_t stride_b, std::complex<double> beta,
                                    sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k,
                                     std::complex<double> alpha,
                                     sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::complex<double> beta,
                                     sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k,
                                    sycl::half alpha, sycl::buffer<sycl::half, 1> &a,
                                    std::int64_t lda, std::int64_t stride_a,
                                    sycl::buffer<sycl::half, 1> &b, std::int64_t ldb,
                                    std::int64_t stride_b, sycl::half beta,
                                    sycl::buffer<sycl::half, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k,
                                     sycl::half alpha, sycl::buffer<sycl::half, 1> &a,
                                     std::int64_t lda, std::int64_t stride_a,
                                     sycl::buffer<sycl::half, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, sycl::half beta,
                                     sycl::buffer<sycl::half, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, float alpha,
                              sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
                              sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, float alpha,
                               sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
                               sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, double alpha,
                              sycl::buffer<double, 1> &a, std::int64_t lda, double beta,
                              sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, double alpha,
                               sycl::buffer<double, 1> &a, std::int64_t lda, double beta,
                               sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                    std::int64_t n, std::int64_t k, float alpha,
                                    sycl::buffer<float, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, float beta,
                                    sycl::buffer<float, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, float alpha,
                                     sycl::buffer<float, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, float beta,
                                     sycl::buffer<float, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                    std::int64_t n, std::int64_t k, double alpha,
                                    sycl::buffer<double, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, double beta,
                                    sycl::buffer<double, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, double alpha,
                                     sycl::buffer<double, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, double beta,
                                     sycl::buffer<double, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                    std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                    sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, std::complex<float> beta,
                                    sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                     sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, std::complex<float> beta,
                                     sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                    std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                    sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, std::complex<double> beta,
                                    sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                     sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, std::complex<double> beta,
                                     sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hbmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hbmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hbmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hbmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(sycl::queue &queue, std::int64_t n,
                             sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                             sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                             float c, float s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              float c, float s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(sycl::queue &queue, std::int64_t n,
                             sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                             sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                             double c, double s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              double c, double s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x,
                             std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
                             float c, float s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x,
                              std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
                              float c, float s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x,
                             std::int64_t incx, sycl::buffer<double, 1> &y, std::int64_t incy,
                             double c, double s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &y, std::int64_t incy, double c,
                              double s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(sycl::queue &queue, std::int64_t n, float alpha,
                              sycl::buffer<float, 1> &x, std::int64_t incx,
                              sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(sycl::queue &queue, std::int64_t n, float alpha,
                               sycl::buffer<float, 1> &x, std::int64_t incx,
                               sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(sycl::queue &queue, std::int64_t n, double alpha,
                              sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(sycl::queue &queue, std::int64_t n, double alpha,
                               sycl::buffer<double, 1> &x, std::int64_t incx,
                               sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t n, float alpha,
                                    sycl::buffer<float, 1> &x, std::int64_t incx,
                                    std::int64_t stridex, sycl::buffer<float, 1> &y,
                                    std::int64_t incy, std::int64_t stridey,
                                    std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t n, float alpha,
                                     sycl::buffer<float, 1> &x, std::int64_t incx,
                                     std::int64_t stridex, sycl::buffer<float, 1> &y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t n, double alpha,
                                    sycl::buffer<double, 1> &x, std::int64_t incx,
                                    std::int64_t stridex, sycl::buffer<double, 1> &y,
                                    std::int64_t incy, std::int64_t stridey,
                                    std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t n, double alpha,
                                     sycl::buffer<double, 1> &x, std::int64_t incx,
                                     std::int64_t stridex, sycl::buffer<double, 1> &y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t n,
                                    std::complex<float> alpha,
                                    sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    std::int64_t stridex,
                                    sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                    std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t n,
                                     std::complex<float> alpha,
                                     sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                     std::int64_t stridex,
                                     sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                     std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t n,
                                    std::complex<double> alpha,
                                    sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    std::int64_t stridex,
                                    sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                    std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t n,
                                     std::complex<double> alpha,
                                     sycl::buffer<std::complex<double>, 1> &x,
                                     std::int64_t incx, std::int64_t stridex,
                                     sycl::buffer<std::complex<double>, 1> &y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpby_precondition(sycl::queue &queue, std::int64_t n, float alpha,
                               sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                               sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpby_postcondition(sycl::queue &queue, std::int64_t n, float alpha,
                                sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                                sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpby_precondition(sycl::queue &queue, std::int64_t n, double alpha,
                               sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                               sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpby_postcondition(sycl::queue &queue, std::int64_t n, double alpha,
                                sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                                sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpby_precondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpby_postcondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                                sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                std::complex<float> beta,
                                sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpby_precondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpby_postcondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                                sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                std::complex<double> beta,
                                sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gerc_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gerc_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gerc_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gerc_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, float alpha,
                               sycl::buffer<float, 1> &a, std::int64_t lda,
                               sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                               sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, float alpha,
                                sycl::buffer<float, 1> &a, std::int64_t lda,
                                sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, double alpha,
                               sycl::buffer<double, 1> &a, std::int64_t lda,
                               sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                               sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, double alpha,
                                sycl::buffer<double, 1> &a, std::int64_t lda,
                                sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                std::complex<float> beta,
                                sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                std::complex<double> beta,
                                sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, float alpha, sycl::buffer<float, 1> &a,
                              std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx,
                              float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, float alpha, sycl::buffer<float, 1> &a,
                               std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx,
                               float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
                              std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx,
                              double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
                               std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx,
                               double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                    std::int64_t n, float alpha, sycl::buffer<float, 1> &a,
                                    std::int64_t lda, std::int64_t stridea,
                                    sycl::buffer<float, 1> &x, std::int64_t incx,
                                    std::int64_t stridex, float beta, sycl::buffer<float, 1> &y,
                                    std::int64_t incy, std::int64_t stridey,
                                    std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                     std::int64_t n, float alpha, sycl::buffer<float, 1> &a,
                                     std::int64_t lda, std::int64_t stridea,
                                     sycl::buffer<float, 1> &x, std::int64_t incx,
                                     std::int64_t stridex, float beta,
                                     sycl::buffer<float, 1> &y, std::int64_t incy,
                                     std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                    std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
                                    std::int64_t lda, std::int64_t stridea,
                                    sycl::buffer<double, 1> &x, std::int64_t incx,
                                    std::int64_t stridex, double beta,
                                    sycl::buffer<double, 1> &y, std::int64_t incy,
                                    std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                     std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
                                     std::int64_t lda, std::int64_t stridea,
                                     sycl::buffer<double, 1> &x, std::int64_t incx,
                                     std::int64_t stridex, double beta,
                                     sycl::buffer<double, 1> &y, std::int64_t incy,
                                     std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                    std::int64_t n, std::complex<float> alpha,
                                    sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    std::int64_t stridea,
                                    sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    std::int64_t stridex, std::complex<float> beta,
                                    sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                    std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                     std::int64_t n, std::complex<float> alpha,
                                     sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     std::int64_t stridea,
                                     sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                     std::int64_t stridex, std::complex<float> beta,
                                     sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                     std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                    std::int64_t n, std::complex<double> alpha,
                                    sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    std::int64_t stridea,
                                    sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    std::int64_t stridex, std::complex<double> beta,
                                    sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                    std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(
    sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
    std::int64_t stridea, sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
    std::int64_t stridex, std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &y,
    std::int64_t incy, std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side left_right, std::int64_t m,
                                    std::int64_t n, sycl::buffer<float, 1> &a, std::int64_t lda,
                                    std::int64_t stridea, sycl::buffer<float, 1> &x,
                                    std::int64_t incx, std::int64_t stridex,
                                    sycl::buffer<float, 1> &c, std::int64_t ldc,
                                    std::int64_t stridec, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side left_right, std::int64_t m,
                                     std::int64_t n, sycl::buffer<float, 1> &a,
                                     std::int64_t lda, std::int64_t stridea,
                                     sycl::buffer<float, 1> &x, std::int64_t incx,
                                     std::int64_t stridex, sycl::buffer<float, 1> &c,
                                     std::int64_t ldc, std::int64_t stridec,
                                     std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side left_right, std::int64_t m,
                                    std::int64_t n, sycl::buffer<double, 1> &a,
                                    std::int64_t lda, std::int64_t stridea,
                                    sycl::buffer<double, 1> &x, std::int64_t incx,
                                    std::int64_t stridex, sycl::buffer<double, 1> &c,
                                    std::int64_t ldc, std::int64_t stridec,
                                    std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side left_right, std::int64_t m,
                                     std::int64_t n, sycl::buffer<double, 1> &a,
                                     std::int64_t lda, std::int64_t stridea,
                                     sycl::buffer<double, 1> &x, std::int64_t incx,
                                     std::int64_t stridex, sycl::buffer<double, 1> &c,
                                     std::int64_t ldc, std::int64_t stridec,
                                     std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side left_right, std::int64_t m,
                                    std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
                                    std::int64_t lda, std::int64_t stridea,
                                    sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    std::int64_t stridex,
                                    sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                    std::int64_t stridec, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side left_right, std::int64_t m,
                                     std::int64_t n, sycl::buffer<std::complex<float>, 1> &a,
                                     std::int64_t lda, std::int64_t stridea,
                                     sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                     std::int64_t stridex,
                                     sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                     std::int64_t stridec, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side left_right, std::int64_t m,
                                    std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
                                    std::int64_t lda, std::int64_t stridea,
                                    sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    std::int64_t stridex,
                                    sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                    std::int64_t stridec, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side left_right, std::int64_t m,
                                     std::int64_t n, sycl::buffer<std::complex<double>, 1> &a,
                                     std::int64_t lda, std::int64_t stridea,
                                     sycl::buffer<std::complex<double>, 1> &x,
                                     std::int64_t incx, std::int64_t stridex,
                                     sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                     std::int64_t stridec, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                             sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                             sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, sycl::buffer<std::complex<double>, 1> &x,
                              std::int64_t incx, sycl::buffer<std::complex<double>, 1> &a,
                              std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                             sycl::buffer<std::complex<float>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                             sycl::buffer<std::complex<double>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, sycl::buffer<std::complex<double>, 1> &x,
                              std::int64_t incx, sycl::buffer<std::complex<double>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_bias_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                   offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                   float alpha, sycl::buffer<int8_t, 1> &a, std::int64_t lda,
                                   int8_t ao, sycl::buffer<uint8_t, 1> &b, std::int64_t ldb,
                                   uint8_t bo, float beta, sycl::buffer<int32_t, 1> &c,
                                   std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_bias_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, sycl::buffer<int8_t, 1> &a, std::int64_t lda,
                                    int8_t ao, sycl::buffer<uint8_t, 1> &b, std::int64_t ldb,
                                    uint8_t bo, float beta, sycl::buffer<int32_t, 1> &c,
                                    std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_bias_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                   offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                   float alpha, sycl::buffer<int8_t, 1> &a, std::int64_t lda,
                                   int8_t ao, sycl::buffer<int8_t, 1> &b, std::int64_t ldb,
                                   int8_t bo, float beta, sycl::buffer<int32_t, 1> &c,
                                   std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_bias_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, sycl::buffer<int8_t, 1> &a, std::int64_t lda,
                                    int8_t ao, sycl::buffer<int8_t, 1> &b, std::int64_t ldb,
                                    int8_t bo, float beta, sycl::buffer<int32_t, 1> &c,
                                    std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_bias_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                   offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                   float alpha, sycl::buffer<uint8_t, 1> &a, std::int64_t lda,
                                   uint8_t ao, sycl::buffer<int8_t, 1> &b, std::int64_t ldb,
                                   int8_t bo, float beta, sycl::buffer<int32_t, 1> &c,
                                   std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_bias_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, sycl::buffer<uint8_t, 1> &a, std::int64_t lda,
                                    uint8_t ao, sycl::buffer<int8_t, 1> &b, std::int64_t ldb,
                                    int8_t bo, float beta, sycl::buffer<int32_t, 1> &c,
                                    std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_bias_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                   offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                   float alpha, sycl::buffer<uint8_t, 1> &a, std::int64_t lda,
                                   uint8_t ao, sycl::buffer<uint8_t, 1> &b, std::int64_t ldb,
                                   uint8_t bo, float beta, sycl::buffer<int32_t, 1> &c,
                                   std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_bias_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, sycl::buffer<uint8_t, 1> &a, std::int64_t lda,
                                    uint8_t ao, sycl::buffer<uint8_t, 1> &b, std::int64_t ldb,
                                    uint8_t bo, float beta, sycl::buffer<int32_t, 1> &c,
                                    std::int64_t ldc, sycl::buffer<int32_t, 1> &co) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<float, 1> &x, std::int64_t incx,
                               sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(sycl::queue &queue, std::int64_t n,
                                sycl::buffer<float, 1> &x, std::int64_t incx,
                                sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<double, 1> &x, std::int64_t incx,
                               sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(sycl::queue &queue, std::int64_t n,
                                sycl::buffer<double, 1> &x, std::int64_t incx,
                                sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(sycl::queue &queue, std::int64_t n,
                                sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(sycl::queue &queue, std::int64_t n,
                                sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &a,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &a,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              sycl::buffer<float, 1> &a, sycl::buffer<float, 1> &x,
                              std::int64_t incx, float beta, sycl::buffer<float, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, sycl::buffer<float, 1> &a,
                               sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                               sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, sycl::buffer<double, 1> &a,
                              sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                              sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, sycl::buffer<double, 1> &a,
                               sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                               sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotmg_precondition(sycl::queue &queue, sycl::buffer<float, 1> &d1,
                               sycl::buffer<float, 1> &d2, sycl::buffer<float, 1> &x1,
                               float y1, sycl::buffer<float, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotmg_postcondition(sycl::queue &queue, sycl::buffer<float, 1> &d1,
                                sycl::buffer<float, 1> &d2, sycl::buffer<float, 1> &x1,
                                float y1, sycl::buffer<float, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotmg_precondition(sycl::queue &queue, sycl::buffer<double, 1> &d1,
                               sycl::buffer<double, 1> &d2, sycl::buffer<double, 1> &x1,
                               double y1, sycl::buffer<double, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotmg_postcondition(sycl::queue &queue, sycl::buffer<double, 1> &d1,
                                sycl::buffer<double, 1> &d2, sycl::buffer<double, 1> &x1,
                                double y1, sycl::buffer<double, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x,
                              std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<float, 1> &x, std::int64_t incx,
                               sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<double, 1> &x, std::int64_t incx,
                               sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void geru_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void geru_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void geru_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void geru_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x,
                              std::int64_t incx, sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<float, 1> &x, std::int64_t incx,
                               sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<double, 1> &x, std::int64_t incx,
                               sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k, float alpha,
                               sycl::buffer<float, 1> &a, std::int64_t lda,
                               sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                               sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k, float alpha,
                                sycl::buffer<float, 1> &a, std::int64_t lda,
                                sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k, double alpha,
                               sycl::buffer<double, 1> &a, std::int64_t lda,
                               sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                               sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k, double alpha,
                                sycl::buffer<double, 1> &a, std::int64_t lda,
                                sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k,
                               std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k,
                                std::complex<float> alpha,
                                sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                std::complex<float> beta,
                                sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k,
                               std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k,
                                std::complex<double> alpha,
                                sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                std::complex<double> beta,
                                sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              sycl::buffer<float, 1> &a, std::int64_t lda,
                              sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                              sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               sycl::buffer<float, 1> &a, std::int64_t lda,
                               sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                               sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                              sycl::buffer<double, 1> &a, std::int64_t lda,
                              sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                              sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                               sycl::buffer<double, 1> &a, std::int64_t lda,
                               sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                               sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                              std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                              std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                              sycl::buffer<sycl::half, 1> &a, std::int64_t lda,
                              sycl::buffer<sycl::half, 1> &b, std::int64_t ldb, sycl::half beta,
                              sycl::buffer<sycl::half, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                               sycl::buffer<sycl::half, 1> &a, std::int64_t lda,
                               sycl::buffer<sycl::half, 1> &b, std::int64_t ldb,
                               sycl::half beta, sycl::buffer<sycl::half, 1> &c,
                               std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              sycl::buffer<sycl::half, 1> &a, std::int64_t lda,
                              sycl::buffer<sycl::half, 1> &b, std::int64_t ldb, float beta,
                              sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               sycl::buffer<sycl::half, 1> &a, std::int64_t lda,
                               sycl::buffer<sycl::half, 1> &b, std::int64_t ldb, float beta,
                               sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              sycl::buffer<bfloat16, 1> &a, std::int64_t lda,
                              sycl::buffer<bfloat16, 1> &b, std::int64_t ldb, float beta,
                              sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               sycl::buffer<bfloat16, 1> &a, std::int64_t lda,
                               sycl::buffer<bfloat16, 1> &b, std::int64_t ldb, float beta,
                               sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              sycl::buffer<float, 1> &x, std::int64_t incx,
                              sycl::buffer<float, 1> &y, std::int64_t incy,
                              sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, sycl::buffer<float, 1> &x, std::int64_t incx,
                               sycl::buffer<float, 1> &y, std::int64_t incy,
                               sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &y, std::int64_t incy,
                              sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, sycl::buffer<double, 1> &x, std::int64_t incx,
                               sycl::buffer<double, 1> &y, std::int64_t incy,
                               sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void ger_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                             sycl::buffer<float, 1> &x, std::int64_t incx,
                             sycl::buffer<float, 1> &y, std::int64_t incy,
                             sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void ger_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                              sycl::buffer<float, 1> &x, std::int64_t incx,
                              sycl::buffer<float, 1> &y, std::int64_t incy,
                              sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void ger_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                             sycl::buffer<double, 1> &x, std::int64_t incx,
                             sycl::buffer<double, 1> &y, std::int64_t incy,
                             sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void ger_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                              sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &y, std::int64_t incy,
                              sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
                              sycl::buffer<float, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
                               sycl::buffer<float, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
                              sycl::buffer<double, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
                               sycl::buffer<double, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dotu_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<float>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dotu_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<float>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dotu_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<double>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dotu_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<double>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hemm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                              std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hemm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hemm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                              std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hemm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<float>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<float>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<double>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<double>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
                              sycl::buffer<float, 1> &a, std::int64_t lda,
                              sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                              sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
                               sycl::buffer<float, 1> &a, std::int64_t lda,
                               sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                               sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                              sycl::buffer<double, 1> &a, std::int64_t lda,
                              sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                              sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                               sycl::buffer<double, 1> &a, std::int64_t lda,
                               sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                               sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku,
                              std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku,
                               std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku,
                              std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku,
                               std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              sycl::buffer<float, 1> &a, std::int64_t lda,
                              sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               sycl::buffer<float, 1> &a, std::int64_t lda,
                               sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              sycl::buffer<double, 1> &a, std::int64_t lda,
                              sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               sycl::buffer<double, 1> &a, std::int64_t lda,
                               sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, float alpha,
                              sycl::buffer<float, 1> &a, std::int64_t lda,
                              sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                              sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, float alpha,
                               sycl::buffer<float, 1> &a, std::int64_t lda,
                               sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                               sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, double alpha,
                              sycl::buffer<double, 1> &a, std::int64_t lda,
                              sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                              sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, double alpha,
                               sycl::buffer<double, 1> &a, std::int64_t lda,
                               sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                               sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                              std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                              std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dotc_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<float>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dotc_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<float>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dotc_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              sycl::buffer<std::complex<double>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dotc_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               sycl::buffer<std::complex<double>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             sycl::buffer<float, 1> &x, std::int64_t incx,
                             sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              sycl::buffer<float, 1> &x, std::int64_t incx,
                              sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             sycl::buffer<double, 1> &x, std::int64_t incx,
                             sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
                              sycl::buffer<float, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
                               sycl::buffer<float, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
                              sycl::buffer<double, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
                               sycl::buffer<double, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              sycl::buffer<float, 1> &a, std::int64_t lda,
                              sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                              sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
                               sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                               sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
                              sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                              sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
                               sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                               sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
                              sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
                               sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
                              sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
                               sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1> &a,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &a,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1> &a,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &a,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
                              std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, sycl::buffer<float, 1> &a,
                               std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
                              std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, sycl::buffer<double, 1> &a,
                               std::int64_t lda, sycl::buffer<double, 1> &x,
                               std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x,
                              std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<float, 1> &x, std::int64_t incx,
                               sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<double, 1> &x, std::int64_t incx,
                               sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t n,
                                    sycl::buffer<float, 1> &x, std::int64_t incx,
                                    std::int64_t stridex, sycl::buffer<float, 1> &y,
                                    std::int64_t incy, std::int64_t stridey,
                                    std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t n,
                                     sycl::buffer<float, 1> &x, std::int64_t incx,
                                     std::int64_t stridex, sycl::buffer<float, 1> &y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t n,
                                    sycl::buffer<double, 1> &x, std::int64_t incx,
                                    std::int64_t stridex, sycl::buffer<double, 1> &y,
                                    std::int64_t incy, std::int64_t stridey,
                                    std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t n,
                                     sycl::buffer<double, 1> &x, std::int64_t incx,
                                     std::int64_t stridex, sycl::buffer<double, 1> &y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t n,
                                    sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                    std::int64_t stridex,
                                    sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                    std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t n,
                                     sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                     std::int64_t stridex,
                                     sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                                     std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t n,
                                    sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                    std::int64_t stridex,
                                    sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                                    std::int64_t stridey, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t n,
                                     sycl::buffer<std::complex<double>, 1> &x,
                                     std::int64_t incx, std::int64_t stridex,
                                     sycl::buffer<std::complex<double>, 1> &y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hemv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hemv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               std::complex<float> beta,
                               sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hemv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              std::complex<double> beta,
                              sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hemv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               std::complex<double> beta,
                               sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<float, 1> &x, std::int64_t incx,
                               sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(sycl::queue &queue, std::int64_t n,
                                sycl::buffer<float, 1> &x, std::int64_t incx,
                                sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<double, 1> &x, std::int64_t incx,
                               sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(sycl::queue &queue, std::int64_t n,
                                sycl::buffer<double, 1> &x, std::int64_t incx,
                                sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(sycl::queue &queue, std::int64_t n,
                                sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(sycl::queue &queue, std::int64_t n,
                                sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void sbmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, float alpha, sycl::buffer<float, 1> &a,
                              std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx,
                              float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void sbmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, float alpha, sycl::buffer<float, 1> &a,
                               std::int64_t lda, sycl::buffer<float, 1> &x, std::int64_t incx,
                               float beta, sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void sbmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, double alpha, sycl::buffer<double, 1> &a,
                              std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx,
                              double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void sbmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, double alpha, sycl::buffer<double, 1> &a,
                               std::int64_t lda, sycl::buffer<double, 1> &x, std::int64_t incx,
                               double beta, sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x,
                              std::int64_t incx, sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<float, 1> &x, std::int64_t incx,
                               sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<double, 1> &x, std::int64_t incx,
                               sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              sycl::buffer<float, 1> &a, std::int64_t lda,
                              sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               sycl::buffer<float, 1> &a, std::int64_t lda,
                               sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              sycl::buffer<double, 1> &a, std::int64_t lda,
                              sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               sycl::buffer<double, 1> &a, std::int64_t lda,
                               sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              sycl::buffer<float, 1> &x, std::int64_t incx,
                              sycl::buffer<float, 1> &y, std::int64_t incy,
                              sycl::buffer<float, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, sycl::buffer<float, 1> &x, std::int64_t incx,
                               sycl::buffer<float, 1> &y, std::int64_t incy,
                               sycl::buffer<float, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &y, std::int64_t incy,
                              sycl::buffer<double, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, sycl::buffer<double, 1> &x, std::int64_t incx,
                               sycl::buffer<double, 1> &y, std::int64_t incy,
                               sycl::buffer<double, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, sycl::buffer<float, 1> &b,
                                    std::int64_t ldb, std::int64_t stride_b,
                                    std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, float alpha, sycl::buffer<float, 1> &a,
                                     std::int64_t lda, std::int64_t stride_a,
                                     sycl::buffer<float, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, sycl::buffer<double, 1> &b,
                                    std::int64_t ldb, std::int64_t stride_b,
                                    std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
                                     std::int64_t lda, std::int64_t stride_a,
                                     sycl::buffer<double, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    std::complex<float> alpha,
                                    sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a,
                                    sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                    std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, std::complex<float> alpha,
                                     sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    std::complex<double> alpha,
                                    sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a,
                                    sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                    std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, std::complex<double> alpha,
                                     sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotm_precondition(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x,
                              std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
                              sycl::buffer<float, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotm_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<float, 1> &x, std::int64_t incx,
                               sycl::buffer<float, 1> &y, std::int64_t incy,
                               sycl::buffer<float, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotm_precondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &y, std::int64_t incy,
                              sycl::buffer<double, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotm_postcondition(sycl::queue &queue, std::int64_t n,
                               sycl::buffer<double, 1> &x, std::int64_t incx,
                               sycl::buffer<double, 1> &y, std::int64_t incy,
                               sycl::buffer<double, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dot_precondition(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x,
                             std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
                             sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dot_postcondition(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x,
                              std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
                              sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dot_precondition(sycl::queue &queue, std::int64_t n, sycl::buffer<double, 1> &x,
                             std::int64_t incx, sycl::buffer<double, 1> &y, std::int64_t incy,
                             sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dot_postcondition(sycl::queue &queue, std::int64_t n,
                              sycl::buffer<double, 1> &x, std::int64_t incx,
                              sycl::buffer<double, 1> &y, std::int64_t incy,
                              sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dot_precondition(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x,
                             std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
                             sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dot_postcondition(sycl::queue &queue, std::int64_t n, sycl::buffer<float, 1> &x,
                              std::int64_t incx, sycl::buffer<float, 1> &y, std::int64_t incy,
                              sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void sdsdot_precondition(sycl::queue &queue, std::int64_t n, float sb,
                                sycl::buffer<float, 1> &x, std::int64_t incx,
                                sycl::buffer<float, 1> &y, std::int64_t incy,
                                sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void sdsdot_postcondition(sycl::queue &queue, std::int64_t n, float sb,
                                 sycl::buffer<float, 1> &x, std::int64_t incx,
                                 sycl::buffer<float, 1> &y, std::int64_t incy,
                                 sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<float> alpha,
                               sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               float beta, sycl::buffer<std::complex<float>, 1> &c,
                               std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                float beta, sycl::buffer<std::complex<float>, 1> &c,
                                std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<double> alpha,
                               sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               double beta, sycl::buffer<std::complex<double>, 1> &c,
                               std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                double beta, sycl::buffer<std::complex<double>, 1> &c,
                                std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(sycl::queue &queue, sycl::buffer<float, 1> &a,
                              sycl::buffer<float, 1> &b, sycl::buffer<float, 1> &c,
                              sycl::buffer<float, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(sycl::queue &queue, sycl::buffer<float, 1> &a,
                               sycl::buffer<float, 1> &b, sycl::buffer<float, 1> &c,
                               sycl::buffer<float, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(sycl::queue &queue, sycl::buffer<double, 1> &a,
                              sycl::buffer<double, 1> &b, sycl::buffer<double, 1> &c,
                              sycl::buffer<double, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(sycl::queue &queue, sycl::buffer<double, 1> &a,
                               sycl::buffer<double, 1> &b, sycl::buffer<double, 1> &c,
                               sycl::buffer<double, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(sycl::queue &queue, sycl::buffer<std::complex<float>, 1> &a,
                              sycl::buffer<std::complex<float>, 1> &b,
                              sycl::buffer<float, 1> &c,
                              sycl::buffer<std::complex<float>, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(sycl::queue &queue, sycl::buffer<std::complex<float>, 1> &a,
                               sycl::buffer<std::complex<float>, 1> &b,
                               sycl::buffer<float, 1> &c,
                               sycl::buffer<std::complex<float>, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(sycl::queue &queue, sycl::buffer<std::complex<double>, 1> &a,
                              sycl::buffer<std::complex<double>, 1> &b,
                              sycl::buffer<double, 1> &c,
                              sycl::buffer<std::complex<double>, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(sycl::queue &queue, sycl::buffer<std::complex<double>, 1> &a,
                               sycl::buffer<std::complex<double>, 1> &b,
                               sycl::buffer<double, 1> &c,
                               sycl::buffer<std::complex<double>, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, float alpha, sycl::buffer<float, 1> &a,
                                        std::int64_t lda, std::int64_t stride_a,
                                        sycl::buffer<float, 1> &b, std::int64_t ldb,
                                        std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, float alpha, sycl::buffer<float, 1> &a,
                                         std::int64_t lda, std::int64_t stride_a,
                                         sycl::buffer<float, 1> &b, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
                                        std::int64_t lda, std::int64_t stride_a,
                                        sycl::buffer<double, 1> &b, std::int64_t ldb,
                                        std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
                                         std::int64_t lda, std::int64_t stride_a,
                                         sycl::buffer<double, 1> &b, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, std::complex<float> alpha,
                                        sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                        std::int64_t stride_a,
                                        sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                        std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<float> alpha,
                                         sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                         std::int64_t stride_a,
                                         sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, std::complex<double> alpha,
                                        sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                        std::int64_t stride_a,
                                        sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                        std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<double> alpha,
                                         sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                         std::int64_t stride_a,
                                         sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                         std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, float alpha, sycl::buffer<float, 1> &ab,
                                        std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                                        std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, float alpha, sycl::buffer<float, 1> &ab,
                                         std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, double alpha, sycl::buffer<double, 1> &ab,
                                        std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                                        std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, double alpha, sycl::buffer<double, 1> &ab,
                                         std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, std::complex<float> alpha,
                                        sycl::buffer<std::complex<float>, 1> &ab, std::int64_t lda,
                                        std::int64_t ldb, std::int64_t stride,
                                        std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<float> alpha,
                                         sycl::buffer<std::complex<float>, 1> &ab, std::int64_t lda,
                                         std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, std::complex<double> alpha,
                                        sycl::buffer<std::complex<double>, 1> &ab, std::int64_t lda,
                                        std::int64_t ldb, std::int64_t stride,
                                        std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<double> alpha,
                                         sycl::buffer<std::complex<double>, 1> &ab,
                                         std::int64_t lda, std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                       std::int64_t m, std::int64_t n, float alpha,
                                       sycl::buffer<float, 1> &a, std::int64_t lda,
                                       std::int64_t stride_a, float beta, sycl::buffer<float, 1> &b,
                                       std::int64_t ldb, std::int64_t stride_b,
                                       sycl::buffer<float, 1> &c, std::int64_t ldc,
                                       std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_postcondition(
    sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    float alpha, sycl::buffer<float, 1> &a, std::int64_t lda, std::int64_t stride_a, float beta,
    sycl::buffer<float, 1> &b, std::int64_t ldb, std::int64_t stride_b, sycl::buffer<float, 1> &c,
    std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_precondition(
    sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    double alpha, sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a, double beta,
    sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b, sycl::buffer<double, 1> &c,
    std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_postcondition(
    sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    double alpha, sycl::buffer<double, 1> &a, std::int64_t lda, std::int64_t stride_a, double beta,
    sycl::buffer<double, 1> &b, std::int64_t ldb, std::int64_t stride_b, sycl::buffer<double, 1> &c,
    std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                       std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                       sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                       std::int64_t stride_a, std::complex<float> beta,
                                       sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                       std::int64_t stride_b,
                                       sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                       std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                        std::int64_t m, std::int64_t n, std::complex<float> alpha,
                                        sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                        std::int64_t stride_a, std::complex<float> beta,
                                        sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                        std::int64_t stride_b,
                                        sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                        std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                       std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                       sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                       std::int64_t stride_a, std::complex<double> beta,
                                       sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                       std::int64_t stride_b,
                                       sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                       std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                        std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                        sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                        std::int64_t stride_a, std::complex<double> beta,
                                        sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                        std::int64_t stride_b,
                                        sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                        std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

// USM APIs

inline void herk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, float alpha,
                              const std::complex<float> *a, std::int64_t lda, float beta,
                              std::complex<float> *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void herk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, float alpha,
                               const std::complex<float> *a, std::int64_t lda, float beta,
                               std::complex<float> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void herk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, double alpha,
                              const std::complex<double> *a, std::int64_t lda, double beta,
                              std::complex<double> *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void herk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, double alpha,
                               const std::complex<double> *a, std::int64_t lda, double beta,
                               std::complex<double> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, float alpha, float *x,
                              std::int64_t incx, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, float alpha, float *x,
                               std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, double alpha, double *x,
                              std::int64_t incx, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, double alpha, double *x,
                               std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                              std::complex<float> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                               std::complex<float> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                              std::complex<double> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                               std::complex<double> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, float alpha,
                              std::complex<float> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, float alpha,
                               std::complex<float> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(sycl::queue &queue, std::int64_t n, double alpha,
                              std::complex<double> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(sycl::queue &queue, std::int64_t n, double alpha,
                               std::complex<double> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const float *a, std::int64_t lda,
                              float *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const float *a, std::int64_t lda,
                               float *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const double *a, std::int64_t lda,
                              double *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const double *a, std::int64_t lda,
                               double *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const std::complex<float> *a,
                              std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<float> *a,
                               std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const std::complex<double> *a,
                              std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<double> *a,
                               std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const float *a, float *x,
                              std::int64_t incx, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const float *a, float *x,
                               std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const double *a, double *x,
                              std::int64_t incx, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const double *a, double *x,
                               std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const std::complex<float> *a,
                              std::complex<float> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<float> *a,
                               std::complex<float> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const std::complex<double> *a,
                              std::complex<double> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<double> *a,
                               std::complex<double> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             const float *x, std::int64_t incx, float *a,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              const float *x, std::int64_t incx, float *a,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             const double *x, std::int64_t incx, double *a,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, const double *x, std::int64_t incx, double *a,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose *transa, transpose *transb,
                                    std::int64_t *m, std::int64_t *n, std::int64_t *k, float *alpha,
                                    const float **a, std::int64_t *lda, const float **b,
                                    std::int64_t *ldb, float *beta, float **c, std::int64_t *ldc,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose *transa, transpose *transb,
                                     std::int64_t *m, std::int64_t *n, std::int64_t *k,
                                     float *alpha, const float **a, std::int64_t *lda,
                                     const float **b, std::int64_t *ldb, float *beta, float **c,
                                     std::int64_t *ldc, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose *transa, transpose *transb,
                                    std::int64_t *m, std::int64_t *n, std::int64_t *k,
                                    double *alpha, const double **a, std::int64_t *lda,
                                    const double **b, std::int64_t *ldb, double *beta, double **c,
                                    std::int64_t *ldc, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose *transa, transpose *transb,
                                     std::int64_t *m, std::int64_t *n, std::int64_t *k,
                                     double *alpha, const double **a, std::int64_t *lda,
                                     const double **b, std::int64_t *ldb, double *beta, double **c,
                                     std::int64_t *ldc, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose *transa, transpose *transb,
                                    std::int64_t *m, std::int64_t *n, std::int64_t *k,
                                    std::complex<float> *alpha, const std::complex<float> **a,
                                    std::int64_t *lda, const std::complex<float> **b,
                                    std::int64_t *ldb, std::complex<float> *beta,
                                    std::complex<float> **c, std::int64_t *ldc,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose *transa, transpose *transb,
                                     std::int64_t *m, std::int64_t *n, std::int64_t *k,
                                     std::complex<float> *alpha, const std::complex<float> **a,
                                     std::int64_t *lda, const std::complex<float> **b,
                                     std::int64_t *ldb, std::complex<float> *beta,
                                     std::complex<float> **c, std::int64_t *ldc,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose *transa, transpose *transb,
                                    std::int64_t *m, std::int64_t *n, std::int64_t *k,
                                    std::complex<double> *alpha, const std::complex<double> **a,
                                    std::int64_t *lda, const std::complex<double> **b,
                                    std::int64_t *ldb, std::complex<double> *beta,
                                    std::complex<double> **c, std::int64_t *ldc,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose *transa, transpose *transb,
                                     std::int64_t *m, std::int64_t *n, std::int64_t *k,
                                     std::complex<double> *alpha, const std::complex<double> **a,
                                     std::int64_t *lda, const std::complex<double> **b,
                                     std::int64_t *ldb, std::complex<double> *beta,
                                     std::complex<double> **c, std::int64_t *ldc,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose *transa, transpose *transb,
                                    std::int64_t *m, std::int64_t *n, std::int64_t *k,
                                    sycl::half *alpha, const sycl::half **a, std::int64_t *lda,
                                    const sycl::half **b, std::int64_t *ldb, sycl::half *beta,
                                    sycl::half **c, std::int64_t *ldc, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose *transa, transpose *transb,
                                     std::int64_t *m, std::int64_t *n, std::int64_t *k,
                                     sycl::half *alpha, const sycl::half **a, std::int64_t *lda,
                                     const sycl::half **b, std::int64_t *ldb, sycl::half *beta,
                                     sycl::half **c, std::int64_t *ldc, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                    const float *a, std::int64_t lda, std::int64_t stride_a,
                                    const float *b, std::int64_t ldb, std::int64_t stride_b,
                                    float beta, float *c, std::int64_t ldc, std::int64_t stride_c,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                     const float *a, std::int64_t lda, std::int64_t stride_a,
                                     const float *b, std::int64_t ldb, std::int64_t stride_b,
                                     float beta, float *c, std::int64_t ldc, std::int64_t stride_c,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                                    const double *a, std::int64_t lda, std::int64_t stride_a,
                                    const double *b, std::int64_t ldb, std::int64_t stride_b,
                                    double beta, double *c, std::int64_t ldc, std::int64_t stride_c,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                                     const double *a, std::int64_t lda, std::int64_t stride_a,
                                     const double *b, std::int64_t ldb, std::int64_t stride_b,
                                     double beta, double *c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(
    sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    std::int64_t stride_a, const std::complex<float> *b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<float> beta, std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(
    sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    std::int64_t stride_a, const std::complex<float> *b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<float> beta, std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(
    sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    std::int64_t stride_a, const std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<double> beta, std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(
    sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    std::int64_t stride_a, const std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<double> beta, std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k,
                                    sycl::half alpha, const sycl::half *a, std::int64_t lda,
                                    std::int64_t stride_a, const sycl::half *b, std::int64_t ldb,
                                    std::int64_t stride_b, sycl::half beta, sycl::half *c,
                                    std::int64_t ldc, std::int64_t stride_c,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k,
                                     sycl::half alpha, const sycl::half *a, std::int64_t lda,
                                     std::int64_t stride_a, const sycl::half *b, std::int64_t ldb,
                                     std::int64_t stride_b, sycl::half beta, sycl::half *c,
                                     std::int64_t ldc, std::int64_t stride_c,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, float alpha, const float *a,
                              std::int64_t lda, float beta, float *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, float alpha, const float *a,
                               std::int64_t lda, float beta, float *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, double alpha, const double *a,
                              std::int64_t lda, double beta, double *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, double alpha, const double *a,
                               std::int64_t lda, double beta, double *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, std::complex<float> alpha,
                              const std::complex<float> *a, std::int64_t lda,
                              std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<float> alpha,
                               const std::complex<float> *a, std::int64_t lda,
                               std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, std::complex<double> alpha,
                              const std::complex<double> *a, std::int64_t lda,
                              std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<double> alpha,
                               const std::complex<double> *a, std::int64_t lda,
                               std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo *upper_lower, transpose *trans,
                                    std::int64_t *n, std::int64_t *k, float *alpha, const float **a,
                                    std::int64_t *lda, float *beta, float **c, std::int64_t *ldc,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo *upper_lower, transpose *trans,
                                     std::int64_t *n, std::int64_t *k, float *alpha,
                                     const float **a, std::int64_t *lda, float *beta, float **c,
                                     std::int64_t *ldc, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo *upper_lower, transpose *trans,
                                    std::int64_t *n, std::int64_t *k, double *alpha,
                                    const double **a, std::int64_t *lda, double *beta, double **c,
                                    std::int64_t *ldc, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo *upper_lower, transpose *trans,
                                     std::int64_t *n, std::int64_t *k, double *alpha,
                                     const double **a, std::int64_t *lda, double *beta, double **c,
                                     std::int64_t *ldc, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo *upper_lower, transpose *trans,
                                    std::int64_t *n, std::int64_t *k, std::complex<float> *alpha,
                                    const std::complex<float> **a, std::int64_t *lda,
                                    std::complex<float> *beta, std::complex<float> **c,
                                    std::int64_t *ldc, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo *upper_lower, transpose *trans,
                                     std::int64_t *n, std::int64_t *k, std::complex<float> *alpha,
                                     const std::complex<float> **a, std::int64_t *lda,
                                     std::complex<float> *beta, std::complex<float> **c,
                                     std::int64_t *ldc, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo *upper_lower, transpose *trans,
                                    std::int64_t *n, std::int64_t *k, std::complex<double> *alpha,
                                    const std::complex<double> **a, std::int64_t *lda,
                                    std::complex<double> *beta, std::complex<double> **c,
                                    std::int64_t *ldc, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo *upper_lower, transpose *trans,
                                     std::int64_t *n, std::int64_t *k, std::complex<double> *alpha,
                                     const std::complex<double> **a, std::int64_t *lda,
                                     std::complex<double> *beta, std::complex<double> **c,
                                     std::int64_t *ldc, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                    std::int64_t n, std::int64_t k, float alpha, const float *a,
                                    std::int64_t lda, std::int64_t stride_a, float beta, float *c,
                                    std::int64_t ldc, std::int64_t stride_c,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, float alpha, const float *a,
                                     std::int64_t lda, std::int64_t stride_a, float beta, float *c,
                                     std::int64_t ldc, std::int64_t stride_c,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                    std::int64_t n, std::int64_t k, double alpha, const double *a,
                                    std::int64_t lda, std::int64_t stride_a, double beta, double *c,
                                    std::int64_t ldc, std::int64_t stride_c,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, double alpha, const double *a,
                                     std::int64_t lda, std::int64_t stride_a, double beta,
                                     double *c, std::int64_t ldc, std::int64_t stride_c,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                    std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                    const std::complex<float> *a, std::int64_t lda,
                                    std::int64_t stride_a, std::complex<float> beta,
                                    std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                     const std::complex<float> *a, std::int64_t lda,
                                     std::int64_t stride_a, std::complex<float> beta,
                                     std::complex<float> *c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_batch_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                    std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                    const std::complex<double> *a, std::int64_t lda,
                                    std::int64_t stride_a, std::complex<double> beta,
                                    std::complex<double> *c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_batch_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                     std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                     const std::complex<double> *a, std::int64_t lda,
                                     std::int64_t stride_a, std::complex<double> beta,
                                     std::complex<double> *c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha, const std::complex<float> *x,
                              std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                              std::complex<float> *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float> *x,
                               std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                               std::complex<float> *a, std::int64_t lda,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha, const std::complex<double> *x,
                              std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                              std::complex<double> *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double> *x,
                               std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                               std::complex<double> *a, std::int64_t lda,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hbmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, std::complex<float> alpha,
                              const std::complex<float> *a, std::int64_t lda,
                              const std::complex<float> *x, std::int64_t incx,
                              std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hbmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, std::complex<float> alpha,
                               const std::complex<float> *a, std::int64_t lda,
                               const std::complex<float> *x, std::int64_t incx,
                               std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hbmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, std::complex<double> alpha,
                              const std::complex<double> *a, std::int64_t lda,
                              const std::complex<double> *x, std::int64_t incx,
                              std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hbmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, std::complex<double> alpha,
                               const std::complex<double> *a, std::int64_t lda,
                               const std::complex<double> *x, std::int64_t incx,
                               std::complex<double> beta, std::complex<double> *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                             std::int64_t incx, std::complex<float> *y, std::int64_t incy, float c,
                             float s, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                              std::int64_t incx, std::complex<float> *y, std::int64_t incy, float c,
                              float s, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                             std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                             double c, double s, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                              std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                              double c, double s,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx,
                             float *y, std::int64_t incy, float c, float s,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx,
                              float *y, std::int64_t incy, float c, float s,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                             double *y, std::int64_t incy, double c, double s,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                              double *y, std::int64_t incy, double c, double s,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                              std::int64_t incx, float *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                               std::int64_t incx, float *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(sycl::queue &queue, std::int64_t n, double alpha, const double *x,
                              std::int64_t incx, double *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(sycl::queue &queue, std::int64_t n, double alpha,
                               const double *x, std::int64_t incx, double *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                              const std::complex<float> *x, std::int64_t incx,
                              std::complex<float> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                               const std::complex<float> *x, std::int64_t incx,
                               std::complex<float> *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                              const std::complex<double> *x, std::int64_t incx,
                              std::complex<double> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                               const std::complex<double> *x, std::int64_t incx,
                               std::complex<double> *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t *n, float *alpha,
                                    const float **x, std::int64_t *incx, float **y,
                                    std::int64_t *incy, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t *n, float *alpha,
                                     const float **x, std::int64_t *incx, float **y,
                                     std::int64_t *incy, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t *n, double *alpha,
                                    const double **x, std::int64_t *incx, double **y,
                                    std::int64_t *incy, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t *n, double *alpha,
                                     const double **x, std::int64_t *incx, double **y,
                                     std::int64_t *incy, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t *n,
                                    std::complex<float> *alpha, const std::complex<float> **x,
                                    std::int64_t *incx, std::complex<float> **y, std::int64_t *incy,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t *n,
                                     std::complex<float> *alpha, const std::complex<float> **x,
                                     std::int64_t *incx, std::complex<float> **y,
                                     std::int64_t *incy, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t *n,
                                    std::complex<double> *alpha, const std::complex<double> **x,
                                    std::int64_t *incx, std::complex<double> **y,
                                    std::int64_t *incy, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t *n,
                                     std::complex<double> *alpha, const std::complex<double> **x,
                                     std::int64_t *incx, std::complex<double> **y,
                                     std::int64_t *incy, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t n, float alpha,
                                    const float *x, std::int64_t incx, std::int64_t stridex,
                                    float *y, std::int64_t incy, std::int64_t stridey,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t n, float alpha,
                                     const float *x, std::int64_t incx, std::int64_t stridex,
                                     float *y, std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t n, double alpha,
                                    const double *x, std::int64_t incx, std::int64_t stridex,
                                    double *y, std::int64_t incy, std::int64_t stridey,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t n, double alpha,
                                     const double *x, std::int64_t incx, std::int64_t stridex,
                                     double *y, std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t n,
                                    std::complex<float> alpha, const std::complex<float> *x,
                                    std::int64_t incx, std::int64_t stridex, std::complex<float> *y,
                                    std::int64_t incy, std::int64_t stridey,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t n,
                                     std::complex<float> alpha, const std::complex<float> *x,
                                     std::int64_t incx, std::int64_t stridex,
                                     std::complex<float> *y, std::int64_t incy,
                                     std::int64_t stridey, std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_batch_precondition(sycl::queue &queue, std::int64_t n,
                                    std::complex<double> alpha, const std::complex<double> *x,
                                    std::int64_t incx, std::int64_t stridex,
                                    std::complex<double> *y, std::int64_t incy,
                                    std::int64_t stridey, std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_batch_postcondition(sycl::queue &queue, std::int64_t n,
                                     std::complex<double> alpha, const std::complex<double> *x,
                                     std::int64_t incx, std::int64_t stridex,
                                     std::complex<double> *y, std::int64_t incy,
                                     std::int64_t stridey, std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpby_precondition(sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                               std::int64_t incx, const float beta, float *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpby_postcondition(sycl::queue &queue, std::int64_t n, float alpha, const float *x,
                                std::int64_t incx, const float beta, float *y, std::int64_t incy,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpby_precondition(sycl::queue &queue, std::int64_t n, double alpha,
                               const double *x, std::int64_t incx, const double beta, double *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpby_postcondition(sycl::queue &queue, std::int64_t n, double alpha,
                                const double *x, std::int64_t incx, const double beta, double *y,
                                std::int64_t incy,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpby_precondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                               const std::complex<float> *x, std::int64_t incx,
                               const std::complex<float> beta, std::complex<float> *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpby_postcondition(sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                                const std::complex<float> *x, std::int64_t incx,
                                const std::complex<float> beta, std::complex<float> *y,
                                std::int64_t incy,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void axpby_precondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                               const std::complex<double> *x, std::int64_t incx,
                               const std::complex<double> beta, std::complex<double> *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void axpby_postcondition(sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                                const std::complex<double> *x, std::int64_t incx,
                                const std::complex<double> beta, std::complex<double> *y,
                                std::int64_t incy,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gerc_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha, const std::complex<float> *x,
                              std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                              std::complex<float> *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gerc_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float> *x,
                               std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                               std::complex<float> *a, std::int64_t lda,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gerc_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha, const std::complex<double> *x,
                              std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                              std::complex<double> *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gerc_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double> *x,
                               std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                               std::complex<double> *a, std::int64_t lda,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, float alpha, const float *a,
                               std::int64_t lda, const float *b, std::int64_t ldb, float beta,
                               float *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, float alpha, const float *a,
                                std::int64_t lda, const float *b, std::int64_t ldb, float beta,
                                float *c, std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, double alpha, const double *a,
                               std::int64_t lda, const double *b, std::int64_t ldb, double beta,
                               double *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, double alpha, const double *a,
                                std::int64_t lda, const double *b, std::int64_t ldb, double beta,
                                double *c, std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<float> alpha,
                               const std::complex<float> *a, std::int64_t lda,
                               const std::complex<float> *b, std::int64_t ldb,
                               std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                const std::complex<float> *a, std::int64_t lda,
                                const std::complex<float> *b, std::int64_t ldb,
                                std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<double> alpha,
                               const std::complex<double> *a, std::int64_t lda,
                               const std::complex<double> *b, std::int64_t ldb,
                               std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                const std::complex<double> *a, std::int64_t lda,
                                const std::complex<double> *b, std::int64_t ldb,
                                std::complex<double> beta, std::complex<double> *c,
                                std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, float alpha, const float *a, std::int64_t lda,
                              const float *x, std::int64_t incx, float beta, float *y,
                              std::int64_t incy, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, float alpha, const float *a, std::int64_t lda,
                               const float *x, std::int64_t incx, float beta, float *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, double alpha, const double *a, std::int64_t lda,
                              const double *x, std::int64_t incx, double beta, double *y,
                              std::int64_t incy, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, double alpha, const double *a, std::int64_t lda,
                               const double *x, std::int64_t incx, double beta, double *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::complex<float> alpha,
                              const std::complex<float> *a, std::int64_t lda,
                              const std::complex<float> *x, std::int64_t incx,
                              std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::complex<float> alpha,
                               const std::complex<float> *a, std::int64_t lda,
                               const std::complex<float> *x, std::int64_t incx,
                               std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::complex<double> alpha,
                              const std::complex<double> *a, std::int64_t lda,
                              const std::complex<double> *x, std::int64_t incx,
                              std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::complex<double> alpha,
                               const std::complex<double> *a, std::int64_t lda,
                               const std::complex<double> *x, std::int64_t incx,
                               std::complex<double> beta, std::complex<double> *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                    std::int64_t n, float alpha, const float *a, std::int64_t lda,
                                    std::int64_t stridea, const float *x, std::int64_t incx,
                                    std::int64_t stridex, float beta, float *y, std::int64_t incy,
                                    std::int64_t stridey, std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                     std::int64_t n, float alpha, const float *a, std::int64_t lda,
                                     std::int64_t stridea, const float *x, std::int64_t incx,
                                     std::int64_t stridex, float beta, float *y, std::int64_t incy,
                                     std::int64_t stridey, std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                    std::int64_t n, double alpha, const double *a, std::int64_t lda,
                                    std::int64_t stridea, const double *x, std::int64_t incx,
                                    std::int64_t stridex, double beta, double *y, std::int64_t incy,
                                    std::int64_t stridey, std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                     std::int64_t n, double alpha, const double *a,
                                     std::int64_t lda, std::int64_t stridea, const double *x,
                                     std::int64_t incx, std::int64_t stridex, double beta,
                                     double *y, std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(
    sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda, std::int64_t stridea,
    const std::complex<float> *x, std::int64_t incx, std::int64_t stridex, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(
    sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda, std::int64_t stridea,
    const std::complex<float> *x, std::int64_t incx, std::int64_t stridex, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy, std::int64_t stridey, std::int64_t batch_size,
    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(
    sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    std::int64_t stridea, const std::complex<double> *x, std::int64_t incx, std::int64_t stridex,
    std::complex<double> beta, std::complex<double> *y, std::int64_t incy, std::int64_t stridey,
    std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(
    sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    std::int64_t stridea, const std::complex<double> *x, std::int64_t incx, std::int64_t stridex,
    std::complex<double> beta, std::complex<double> *y, std::int64_t incy, std::int64_t stridey,
    std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(sycl::queue &queue, transpose *trans, std::int64_t *m,
                                    std::int64_t *n, float *alpha, const float **a,
                                    std::int64_t *lda, const float **x, std::int64_t *incx,
                                    float *beta, float **y, std::int64_t *incy,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(sycl::queue &queue, transpose *trans, std::int64_t *m,
                                     std::int64_t *n, float *alpha, const float **a,
                                     std::int64_t *lda, const float **x, std::int64_t *incx,
                                     float *beta, float **y, std::int64_t *incy,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(sycl::queue &queue, transpose *trans, std::int64_t *m,
                                    std::int64_t *n, double *alpha, const double **a,
                                    std::int64_t *lda, const double **x, std::int64_t *incx,
                                    double *beta, double **y, std::int64_t *incy,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(sycl::queue &queue, transpose *trans, std::int64_t *m,
                                     std::int64_t *n, double *alpha, const double **a,
                                     std::int64_t *lda, const double **x, std::int64_t *incx,
                                     double *beta, double **y, std::int64_t *incy,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(sycl::queue &queue, transpose *trans, std::int64_t *m,
                                    std::int64_t *n, std::complex<float> *alpha,
                                    const std::complex<float> **a, std::int64_t *lda,
                                    const std::complex<float> **x, std::int64_t *incx,
                                    std::complex<float> *beta, std::complex<float> **y,
                                    std::int64_t *incy, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(sycl::queue &queue, transpose *trans, std::int64_t *m,
                                     std::int64_t *n, std::complex<float> *alpha,
                                     const std::complex<float> **a, std::int64_t *lda,
                                     const std::complex<float> **x, std::int64_t *incx,
                                     std::complex<float> *beta, std::complex<float> **y,
                                     std::int64_t *incy, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_batch_precondition(sycl::queue &queue, transpose *trans, std::int64_t *m,
                                    std::int64_t *n, std::complex<double> *alpha,
                                    const std::complex<double> **a, std::int64_t *lda,
                                    const std::complex<double> **x, std::int64_t *incx,
                                    std::complex<double> *beta, std::complex<double> **y,
                                    std::int64_t *incy, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_batch_postcondition(sycl::queue &queue, transpose *trans, std::int64_t *m,
                                     std::int64_t *n, std::complex<double> *alpha,
                                     const std::complex<double> **a, std::int64_t *lda,
                                     const std::complex<double> **x, std::int64_t *incx,
                                     std::complex<double> *beta, std::complex<double> **y,
                                     std::int64_t *incy, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side left_right, std::int64_t m,
                                    std::int64_t n, const float *a, std::int64_t lda,
                                    std::int64_t stridea, const float *x, std::int64_t incx,
                                    std::int64_t stridex, float *c, std::int64_t ldc,
                                    std::int64_t stridec, std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side left_right, std::int64_t m,
                                     std::int64_t n, const float *a, std::int64_t lda,
                                     std::int64_t stridea, const float *x, std::int64_t incx,
                                     std::int64_t stridex, float *c, std::int64_t ldc,
                                     std::int64_t stridec, std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side left_right, std::int64_t m,
                                    std::int64_t n, const double *a, std::int64_t lda,
                                    std::int64_t stridea, const double *x, std::int64_t incx,
                                    std::int64_t stridex, double *c, std::int64_t ldc,
                                    std::int64_t stridec, std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side left_right, std::int64_t m,
                                     std::int64_t n, const double *a, std::int64_t lda,
                                     std::int64_t stridea, const double *x, std::int64_t incx,
                                     std::int64_t stridex, double *c, std::int64_t ldc,
                                     std::int64_t stridec, std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side left_right, std::int64_t m,
                                    std::int64_t n, const std::complex<float> *a, std::int64_t lda,
                                    std::int64_t stridea, const std::complex<float> *x,
                                    std::int64_t incx, std::int64_t stridex, std::complex<float> *c,
                                    std::int64_t ldc, std::int64_t stridec, std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side left_right, std::int64_t m,
                                     std::int64_t n, const std::complex<float> *a, std::int64_t lda,
                                     std::int64_t stridea, const std::complex<float> *x,
                                     std::int64_t incx, std::int64_t stridex,
                                     std::complex<float> *c, std::int64_t ldc, std::int64_t stridec,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side left_right, std::int64_t m,
                                    std::int64_t n, const std::complex<double> *a, std::int64_t lda,
                                    std::int64_t stridea, const std::complex<double> *x,
                                    std::int64_t incx, std::int64_t stridex,
                                    std::complex<double> *c, std::int64_t ldc, std::int64_t stridec,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side left_right, std::int64_t m,
                                     std::int64_t n, const std::complex<double> *a,
                                     std::int64_t lda, std::int64_t stridea,
                                     const std::complex<double> *x, std::int64_t incx,
                                     std::int64_t stridex, std::complex<double> *c,
                                     std::int64_t ldc, std::int64_t stridec,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side *left_right, std::int64_t *m,
                                    std::int64_t *n, const float **a, std::int64_t *lda,
                                    const float **x, std::int64_t *incx, float **c,
                                    std::int64_t *ldc, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side *left_right, std::int64_t *m,
                                     std::int64_t *n, const float **a, std::int64_t *lda,
                                     const float **x, std::int64_t *incx, float **c,
                                     std::int64_t *ldc, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side *left_right, std::int64_t *m,
                                    std::int64_t *n, const double **a, std::int64_t *lda,
                                    const double **x, std::int64_t *incx, double **c,
                                    std::int64_t *ldc, std::int64_t group_count,
                                    std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side *left_right, std::int64_t *m,
                                     std::int64_t *n, const double **a, std::int64_t *lda,
                                     const double **x, std::int64_t *incx, double **c,
                                     std::int64_t *ldc, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side *left_right, std::int64_t *m,
                                    std::int64_t *n, const std::complex<float> **a,
                                    std::int64_t *lda, const std::complex<float> **x,
                                    std::int64_t *incx, std::complex<float> **c, std::int64_t *ldc,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side *left_right, std::int64_t *m,
                                     std::int64_t *n, const std::complex<float> **a,
                                     std::int64_t *lda, const std::complex<float> **x,
                                     std::int64_t *incx, std::complex<float> **c, std::int64_t *ldc,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_precondition(sycl::queue &queue, side *left_right, std::int64_t *m,
                                    std::int64_t *n, const std::complex<double> **a,
                                    std::int64_t *lda, const std::complex<double> **x,
                                    std::int64_t *incx, std::complex<double> **c, std::int64_t *ldc,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dgmm_batch_postcondition(sycl::queue &queue, side *left_right, std::int64_t *m,
                                     std::int64_t *n, const std::complex<double> **a,
                                     std::int64_t *lda, const std::complex<double> **x,
                                     std::int64_t *incx, std::complex<double> **c,
                                     std::int64_t *ldc, std::int64_t group_count,
                                     std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             const std::complex<float> *x, std::int64_t incx,
                             std::complex<float> *a, std::int64_t lda,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              const std::complex<float> *x, std::int64_t incx,
                              std::complex<float> *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             const std::complex<double> *x, std::int64_t incx,
                             std::complex<double> *a, std::int64_t lda,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, const std::complex<double> *x, std::int64_t incx,
                              std::complex<double> *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             const std::complex<float> *x, std::int64_t incx,
                             std::complex<float> *a,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              const std::complex<float> *x, std::int64_t incx,
                              std::complex<float> *a,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             const std::complex<double> *x, std::int64_t incx,
                             std::complex<double> *a,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, const std::complex<double> *x, std::int64_t incx,
                              std::complex<double> *a,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(sycl::queue &queue, std::int64_t n, const float *x,
                               std::int64_t incx, std::int64_t *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(sycl::queue &queue, std::int64_t n, const float *x,
                                std::int64_t incx, std::int64_t *result,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(sycl::queue &queue, std::int64_t n, const double *x,
                               std::int64_t incx, std::int64_t *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(sycl::queue &queue, std::int64_t n, const double *x,
                                std::int64_t incx, std::int64_t *result,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                               std::int64_t incx, std::int64_t *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(sycl::queue &queue, std::int64_t n,
                                const std::complex<float> *x, std::int64_t incx,
                                std::int64_t *result,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(sycl::queue &queue, std::int64_t n,
                               const std::complex<double> *x, std::int64_t incx,
                               std::int64_t *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(sycl::queue &queue, std::int64_t n,
                                const std::complex<double> *x, std::int64_t incx,
                                std::int64_t *result,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha, const std::complex<float> *a,
                              const std::complex<float> *x, std::int64_t incx,
                              std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float> *a,
                               const std::complex<float> *x, std::int64_t incx,
                               std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha, const std::complex<double> *a,
                              const std::complex<double> *x, std::int64_t incx,
                              std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double> *a,
                               const std::complex<double> *x, std::int64_t incx,
                               std::complex<double> beta, std::complex<double> *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              const float *a, const float *x, std::int64_t incx, float beta,
                              float *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, const float *a, const float *x, std::int64_t incx,
                               float beta, float *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, const double *a, const double *x, std::int64_t incx,
                              double beta, double *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, const double *a, const double *x, std::int64_t incx,
                               double beta, double *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotmg_precondition(sycl::queue &queue, float *d1, float *d2, float *x1, float y1,
                               float *param, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotmg_postcondition(sycl::queue &queue, float *d1, float *d2, float *x1, float y1,
                                float *param, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotmg_precondition(sycl::queue &queue, double *d1, double *d2, double *x1,
                               double y1, double *param,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotmg_postcondition(sycl::queue &queue, double *d1, double *d2, double *x1,
                                double y1, double *param,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx,
                              float *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx,
                               float *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                              double *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                               double *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                              std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(sycl::queue &queue, std::int64_t n, std::complex<float> *x,
                               std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                              std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(sycl::queue &queue, std::int64_t n, std::complex<double> *x,
                               std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void geru_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha, const std::complex<float> *x,
                              std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                              std::complex<float> *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void geru_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float> *x,
                               std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                               std::complex<float> *a, std::int64_t lda,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void geru_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha, const std::complex<double> *x,
                              std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                              std::complex<double> *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void geru_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double> *x,
                               std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                               std::complex<double> *a, std::int64_t lda,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                              std::int64_t incx, float *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                               std::int64_t incx, float *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                              std::int64_t incx, double *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(sycl::queue &queue, std::int64_t n,
                               const std::complex<double> *x, std::int64_t incx, double *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(sycl::queue &queue, std::int64_t n, const float *x,
                              std::int64_t incx, float *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(sycl::queue &queue, std::int64_t n, const float *x,
                               std::int64_t incx, float *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(sycl::queue &queue, std::int64_t n, const double *x,
                              std::int64_t incx, double *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(sycl::queue &queue, std::int64_t n, const double *x,
                               std::int64_t incx, double *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k, float alpha,
                               const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
                               float beta, float *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k, float alpha,
                                const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
                                float beta, float *c, std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k, double alpha,
                               const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                               double beta, double *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k, double alpha,
                                const double *a, std::int64_t lda, const double *b,
                                std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k,
                               std::complex<float> alpha, const std::complex<float> *a,
                               std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                               std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k,
                                std::complex<float> alpha, const std::complex<float> *a,
                                std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                                std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k,
                               std::complex<double> alpha, const std::complex<double> *a,
                               std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                               std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k,
                                std::complex<double> alpha, const std::complex<double> *a,
                                std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                                std::complex<double> beta, std::complex<double> *c,
                                std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
                              float beta, float *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
                               float beta, float *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                              const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                              double beta, double *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                               const double *a, std::int64_t lda, const double *b, std::int64_t ldb,
                               double beta, double *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<float> alpha, const std::complex<float> *a,
                              std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                              std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<float> alpha, const std::complex<float> *a,
                               std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                               std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<double> alpha, const std::complex<double> *a,
                              std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                              std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<double> alpha, const std::complex<double> *a,
                               std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                               std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                              const sycl::half *a, std::int64_t lda, const sycl::half *b,
                              std::int64_t ldb, sycl::half beta, sycl::half *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, sycl::half alpha,
                               const sycl::half *a, std::int64_t lda, const sycl::half *b,
                               std::int64_t ldb, sycl::half beta, sycl::half *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              const sycl::half *a, std::int64_t lda, const sycl::half *b,
                              std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               const sycl::half *a, std::int64_t lda, const sycl::half *b,
                               std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              const bfloat16 *a, std::int64_t lda, const bfloat16 *b,
                              std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               const bfloat16 *a, std::int64_t lda, const bfloat16 *b,
                               std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_bias_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                   offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                   float alpha, const std::int8_t *a, std::int64_t lda,
                                   std::int8_t ao, const std::uint8_t *b, std::int64_t ldb,
                                   std::uint8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                                   const std::int32_t *co,
                                   const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_bias_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, const std::int8_t *a, std::int64_t lda,
                                    std::int8_t ao, const std::uint8_t *b, std::int64_t ldb,
                                    std::uint8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                                    const std::int32_t *co,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_bias_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                   offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                   float alpha, const std::int8_t *a, std::int64_t lda,
                                   std::int8_t ao, const std::int8_t *b, std::int64_t ldb,
                                   std::int8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                                   const std::int32_t *co,
                                   const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_bias_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, const std::int8_t *a, std::int64_t lda,
                                    std::int8_t ao, const std::int8_t *b, std::int64_t ldb,
                                    std::int8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                                    const std::int32_t *co,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_bias_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                   offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                   float alpha, const std::uint8_t *a, std::int64_t lda,
                                   std::uint8_t ao, const std::int8_t *b, std::int64_t ldb,
                                   std::int8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                                   const std::int32_t *co,
                                   const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_bias_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, const std::uint8_t *a, std::int64_t lda,
                                    std::uint8_t ao, const std::int8_t *b, std::int64_t ldb,
                                    std::int8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                                    const std::int32_t *co,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_bias_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                   offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                   float alpha, const std::uint8_t *a, std::int64_t lda,
                                   std::uint8_t ao, const std::uint8_t *b, std::int64_t ldb,
                                   std::uint8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                                   const std::int32_t *co,
                                   const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_bias_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                    offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                    float alpha, const std::uint8_t *a, std::int64_t lda,
                                    std::uint8_t ao, const std::uint8_t *b, std::int64_t ldb,
                                    std::uint8_t bo, float beta, std::int32_t *c, std::int64_t ldc,
                                    const std::int32_t *co,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              const float *x, std::int64_t incx, const float *y, std::int64_t incy,
                              float *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, const float *x, std::int64_t incx, const float *y,
                               std::int64_t incy, float *a, std::int64_t lda,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, const double *x, std::int64_t incx, const double *y,
                              std::int64_t incy, double *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, const double *x, std::int64_t incx, const double *y,
                               std::int64_t incy, double *a, std::int64_t lda,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void ger_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                             const float *x, std::int64_t incx, const float *y, std::int64_t incy,
                             float *a, std::int64_t lda,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void ger_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                              const float *x, std::int64_t incx, const float *y, std::int64_t incy,
                              float *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void ger_precondition(sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                             const double *x, std::int64_t incx, const double *y, std::int64_t incy,
                             double *a, std::int64_t lda,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void ger_postcondition(sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                              const double *x, std::int64_t incx, const double *y,
                              std::int64_t incy, double *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              float alpha, const float *a, std::int64_t lda, float *b,
                              std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               float alpha, const float *a, std::int64_t lda, float *b,
                               std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              double alpha, const double *a, std::int64_t lda, double *b,
                              std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               double alpha, const double *a, std::int64_t lda, double *b,
                               std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha, const std::complex<float> *a,
                              std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float> *a,
                               std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha, const std::complex<double> *a,
                              std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double> *a,
                               std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    float alpha, const float *a, std::int64_t lda,
                                    std::int64_t stride_a, float *b, std::int64_t ldb,
                                    std::int64_t stride_b, std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, float alpha, const float *a, std::int64_t lda,
                                     std::int64_t stride_a, float *b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    double alpha, const double *a, std::int64_t lda,
                                    std::int64_t stride_a, double *b, std::int64_t ldb,
                                    std::int64_t stride_b, std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, double alpha, const double *a,
                                     std::int64_t lda, std::int64_t stride_a, double *b,
                                     std::int64_t ldb, std::int64_t stride_b,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    std::complex<float> alpha, const std::complex<float> *a,
                                    std::int64_t lda, std::int64_t stride_a, std::complex<float> *b,
                                    std::int64_t ldb, std::int64_t stride_b,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, std::complex<float> alpha,
                                     const std::complex<float> *a, std::int64_t lda,
                                     std::int64_t stride_a, std::complex<float> *b,
                                     std::int64_t ldb, std::int64_t stride_b,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    std::complex<double> alpha, const std::complex<double> *a,
                                    std::int64_t lda, std::int64_t stride_a,
                                    std::complex<double> *b, std::int64_t ldb,
                                    std::int64_t stride_b, std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, std::complex<double> alpha,
                                     const std::complex<double> *a, std::int64_t lda,
                                     std::int64_t stride_a, std::complex<double> *b,
                                     std::int64_t ldb, std::int64_t stride_b,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side *left_right, uplo *upper_lower,
                                    transpose *trans, diag *unit_diag, std::int64_t *m,
                                    std::int64_t *n, float *alpha, const float **a,
                                    std::int64_t *lda, float **b, std::int64_t *ldb,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side *left_right, uplo *upper_lower,
                                     transpose *trans, diag *unit_diag, std::int64_t *m,
                                     std::int64_t *n, float *alpha, const float **a,
                                     std::int64_t *lda, float **b, std::int64_t *ldb,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side *left_right, uplo *upper_lower,
                                    transpose *trans, diag *unit_diag, std::int64_t *m,
                                    std::int64_t *n, double *alpha, const double **a,
                                    std::int64_t *lda, double **b, std::int64_t *ldb,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side *left_right, uplo *upper_lower,
                                     transpose *trans, diag *unit_diag, std::int64_t *m,
                                     std::int64_t *n, double *alpha, const double **a,
                                     std::int64_t *lda, double **b, std::int64_t *ldb,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side *left_right, uplo *upper_lower,
                                    transpose *trans, diag *unit_diag, std::int64_t *m,
                                    std::int64_t *n, std::complex<float> *alpha,
                                    const std::complex<float> **a, std::int64_t *lda,
                                    std::complex<float> **b, std::int64_t *ldb,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side *left_right, uplo *upper_lower,
                                     transpose *trans, diag *unit_diag, std::int64_t *m,
                                     std::int64_t *n, std::complex<float> *alpha,
                                     const std::complex<float> **a, std::int64_t *lda,
                                     std::complex<float> **b, std::int64_t *ldb,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(sycl::queue &queue, side *left_right, uplo *upper_lower,
                                    transpose *trans, diag *unit_diag, std::int64_t *m,
                                    std::int64_t *n, std::complex<double> *alpha,
                                    const std::complex<double> **a, std::int64_t *lda,
                                    std::complex<double> **b, std::int64_t *ldb,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(sycl::queue &queue, side *left_right, uplo *upper_lower,
                                     transpose *trans, diag *unit_diag, std::int64_t *m,
                                     std::int64_t *n, std::complex<double> *alpha,
                                     const std::complex<double> **a, std::int64_t *lda,
                                     std::complex<double> **b, std::int64_t *ldb,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dotu_precondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                              std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                              std::complex<float> *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dotu_postcondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                               std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                               std::complex<float> *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dotu_precondition(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                              std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                              std::complex<double> *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dotu_postcondition(sycl::queue &queue, std::int64_t n,
                               const std::complex<double> *x, std::int64_t incx,
                               const std::complex<double> *y, std::int64_t incy,
                               std::complex<double> *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hemm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<float> alpha,
                              const std::complex<float> *a, std::int64_t lda,
                              const std::complex<float> *b, std::int64_t ldb,
                              std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hemm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<float> alpha,
                               const std::complex<float> *a, std::int64_t lda,
                               const std::complex<float> *b, std::int64_t ldb,
                               std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hemm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<double> alpha,
                              const std::complex<double> *a, std::int64_t lda,
                              const std::complex<double> *b, std::int64_t ldb,
                              std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hemm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<double> alpha,
                               const std::complex<double> *a, std::int64_t lda,
                               const std::complex<double> *b, std::int64_t ldb,
                               std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha, const std::complex<float> *x,
                              std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                              std::complex<float> *a,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float> *x,
                               std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                               std::complex<float> *a,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha, const std::complex<double> *x,
                              std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                              std::complex<double> *a,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double> *x,
                               std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                               std::complex<double> *a,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
                              const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                              float beta, float *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
                               const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                               float beta, float *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                              const double *a, std::int64_t lda, const double *x, std::int64_t incx,
                              double beta, double *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                               const double *a, std::int64_t lda, const double *x,
                               std::int64_t incx, double beta, double *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku,
                              std::complex<float> alpha, const std::complex<float> *a,
                              std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                              std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku,
                               std::complex<float> alpha, const std::complex<float> *a,
                               std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                               std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku,
                              std::complex<double> alpha, const std::complex<double> *a,
                              std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                              std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku,
                               std::complex<double> alpha, const std::complex<double> *a,
                               std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                               std::complex<double> beta, std::complex<double> *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
                              std::int64_t lda, float *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
                               std::int64_t lda, float *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
                              std::int64_t lda, double *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
                               std::int64_t lda, double *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              const std::complex<float> *a, std::int64_t lda,
                              std::complex<float> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               const std::complex<float> *a, std::int64_t lda,
                               std::complex<float> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              const std::complex<double> *a, std::int64_t lda,
                              std::complex<double> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               const std::complex<double> *a, std::int64_t lda,
                               std::complex<double> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, float alpha, const float *a,
                              std::int64_t lda, const float *b, std::int64_t ldb, float beta,
                              float *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, float alpha, const float *a,
                               std::int64_t lda, const float *b, std::int64_t ldb, float beta,
                               float *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, double alpha, const double *a,
                              std::int64_t lda, const double *b, std::int64_t ldb, double beta,
                              double *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, double alpha, const double *a,
                               std::int64_t lda, const double *b, std::int64_t ldb, double beta,
                               double *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<float> alpha,
                              const std::complex<float> *a, std::int64_t lda,
                              const std::complex<float> *b, std::int64_t ldb,
                              std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<float> alpha,
                               const std::complex<float> *a, std::int64_t lda,
                               const std::complex<float> *b, std::int64_t ldb,
                               std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<double> alpha,
                              const std::complex<double> *a, std::int64_t lda,
                              const std::complex<double> *b, std::int64_t ldb,
                              std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<double> alpha,
                               const std::complex<double> *a, std::int64_t lda,
                               const std::complex<double> *b, std::int64_t ldb,
                               std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dotc_precondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                              std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                              std::complex<float> *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dotc_postcondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                               std::int64_t incx, const std::complex<float> *y, std::int64_t incy,
                               std::complex<float> *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dotc_precondition(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                              std::int64_t incx, const std::complex<double> *y, std::int64_t incy,
                              std::complex<double> *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dotc_postcondition(sycl::queue &queue, std::int64_t n,
                               const std::complex<double> *x, std::int64_t incx,
                               const std::complex<double> *y, std::int64_t incy,
                               std::complex<double> *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             const float *x, std::int64_t incx, float *a, std::int64_t lda,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              const float *x, std::int64_t incx, float *a, std::int64_t lda,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void syr_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             const double *x, std::int64_t incx, double *a, std::int64_t lda,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void syr_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, const double *x, std::int64_t incx, double *a,
                              std::int64_t lda, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              float alpha, const float *a, std::int64_t lda, float *b,
                              std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               float alpha, const float *a, std::int64_t lda, float *b,
                               std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              double alpha, const double *a, std::int64_t lda, double *b,
                              std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               double alpha, const double *a, std::int64_t lda, double *b,
                               std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha, const std::complex<float> *a,
                              std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float> *a,
                               std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha, const std::complex<double> *a,
                              std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double> *a,
                               std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              const float *a, std::int64_t lda, const float *x, std::int64_t incx,
                              float beta, float *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, const float *a, std::int64_t lda, const float *x,
                               std::int64_t incx, float beta, float *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void symv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, const double *a, std::int64_t lda, const double *x,
                              std::int64_t incx, double beta, double *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void symv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, const double *a, std::int64_t lda, const double *x,
                               std::int64_t incx, double beta, double *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const float *a, float *x,
                              std::int64_t incx, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const float *a, float *x,
                               std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const double *a, double *x,
                              std::int64_t incx, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const double *a, double *x,
                               std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const std::complex<float> *a,
                              std::complex<float> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<float> *a,
                               std::complex<float> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const std::complex<double> *a,
                              std::complex<double> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<double> *a,
                               std::complex<double> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const float *a, std::int64_t lda,
                              float *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const float *a, std::int64_t lda,
                               float *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const double *a, std::int64_t lda,
                              double *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const double *a, std::int64_t lda,
                               double *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const std::complex<float> *a,
                              std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<float> *a,
                               std::int64_t lda, std::complex<float> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, const std::complex<double> *a,
                              std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, const std::complex<double> *a,
                               std::int64_t lda, std::complex<double> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(sycl::queue &queue, std::int64_t n, const float *x,
                              std::int64_t incx, float *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(sycl::queue &queue, std::int64_t n, const float *x,
                               std::int64_t incx, float *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(sycl::queue &queue, std::int64_t n, const double *x,
                              std::int64_t incx, double *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(sycl::queue &queue, std::int64_t n, const double *x,
                               std::int64_t incx, double *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                              std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                               std::int64_t incx, std::complex<float> *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                              std::int64_t incx, std::complex<double> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(sycl::queue &queue, std::int64_t n,
                               const std::complex<double> *x, std::int64_t incx,
                               std::complex<double> *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t *n, const float **x,
                                    std::int64_t *incx, float **y, std::int64_t *incy,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t *n, const float **x,
                                     std::int64_t *incx, float **y, std::int64_t *incy,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t *n, const double **x,
                                    std::int64_t *incx, double **y, std::int64_t *incy,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t *n, const double **x,
                                     std::int64_t *incx, double **y, std::int64_t *incy,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t *n,
                                    const std::complex<float> **x, std::int64_t *incx,
                                    std::complex<float> **y, std::int64_t *incy,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t *n,
                                     const std::complex<float> **x, std::int64_t *incx,
                                     std::complex<float> **y, std::int64_t *incy,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t *n,
                                    const std::complex<double> **x, std::int64_t *incx,
                                    std::complex<double> **y, std::int64_t *incy,
                                    std::int64_t group_count, std::int64_t *group_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t *n,
                                     const std::complex<double> **x, std::int64_t *incx,
                                     std::complex<double> **y, std::int64_t *incy,
                                     std::int64_t group_count, std::int64_t *group_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t n, const float *x,
                                    std::int64_t incx, std::int64_t stridex, float *y,
                                    std::int64_t incy, std::int64_t stridey,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t n, const float *x,
                                     std::int64_t incx, std::int64_t stridex, float *y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t n, const double *x,
                                    std::int64_t incx, std::int64_t stridex, double *y,
                                    std::int64_t incy, std::int64_t stridey,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t n, const double *x,
                                     std::int64_t incx, std::int64_t stridex, double *y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t n,
                                    const std::complex<float> *x, std::int64_t incx,
                                    std::int64_t stridex, std::complex<float> *y, std::int64_t incy,
                                    std::int64_t stridey, std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t n,
                                     const std::complex<float> *x, std::int64_t incx,
                                     std::int64_t stridex, std::complex<float> *y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_batch_precondition(sycl::queue &queue, std::int64_t n,
                                    const std::complex<double> *x, std::int64_t incx,
                                    std::int64_t stridex, std::complex<double> *y,
                                    std::int64_t incy, std::int64_t stridey,
                                    std::int64_t batch_size,
                                    const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_batch_postcondition(sycl::queue &queue, std::int64_t n,
                                     const std::complex<double> *x, std::int64_t incx,
                                     std::int64_t stridex, std::complex<double> *y,
                                     std::int64_t incy, std::int64_t stridey,
                                     std::int64_t batch_size,
                                     const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hemv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha, const std::complex<float> *a,
                              std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                              std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hemv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha, const std::complex<float> *a,
                               std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
                               std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void hemv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha, const std::complex<double> *a,
                              std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                              std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void hemv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha, const std::complex<double> *a,
                               std::int64_t lda, const std::complex<double> *x, std::int64_t incx,
                               std::complex<double> beta, std::complex<double> *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(sycl::queue &queue, std::int64_t n, const float *x,
                               std::int64_t incx, std::int64_t *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(sycl::queue &queue, std::int64_t n, const float *x,
                                std::int64_t incx, std::int64_t *result,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(sycl::queue &queue, std::int64_t n, const double *x,
                               std::int64_t incx, std::int64_t *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(sycl::queue &queue, std::int64_t n, const double *x,
                                std::int64_t incx, std::int64_t *result,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                               std::int64_t incx, std::int64_t *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(sycl::queue &queue, std::int64_t n,
                                const std::complex<float> *x, std::int64_t incx,
                                std::int64_t *result,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(sycl::queue &queue, std::int64_t n,
                               const std::complex<double> *x, std::int64_t incx,
                               std::int64_t *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(sycl::queue &queue, std::int64_t n,
                                const std::complex<double> *x, std::int64_t incx,
                                std::int64_t *result,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void sbmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, float alpha, const float *a, std::int64_t lda,
                              const float *x, std::int64_t incx, float beta, float *y,
                              std::int64_t incy, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void sbmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, float alpha, const float *a, std::int64_t lda,
                               const float *x, std::int64_t incx, float beta, float *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void sbmv_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, double alpha, const double *a, std::int64_t lda,
                              const double *x, std::int64_t incx, double beta, double *y,
                              std::int64_t incy, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void sbmv_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, double alpha, const double *a, std::int64_t lda,
                               const double *x, std::int64_t incx, double beta, double *y,
                               std::int64_t incy,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                              std::int64_t incx, float *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(sycl::queue &queue, std::int64_t n, const std::complex<float> *x,
                               std::int64_t incx, float *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(sycl::queue &queue, std::int64_t n, const std::complex<double> *x,
                              std::int64_t incx, double *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(sycl::queue &queue, std::int64_t n,
                               const std::complex<double> *x, std::int64_t incx, double *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(sycl::queue &queue, std::int64_t n, const float *x,
                              std::int64_t incx, float *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(sycl::queue &queue, std::int64_t n, const float *x,
                               std::int64_t incx, float *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(sycl::queue &queue, std::int64_t n, const double *x,
                              std::int64_t incx, double *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(sycl::queue &queue, std::int64_t n, const double *x,
                               std::int64_t incx, double *result,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
                              std::int64_t lda, float *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k, const float *a,
                               std::int64_t lda, float *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
                              std::int64_t lda, double *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k, const double *a,
                               std::int64_t lda, double *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              const std::complex<float> *a, std::int64_t lda,
                              std::complex<float> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               const std::complex<float> *a, std::int64_t lda,
                               std::complex<float> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              const std::complex<double> *a, std::int64_t lda,
                              std::complex<double> *x, std::int64_t incx,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               const std::complex<double> *a, std::int64_t lda,
                               std::complex<double> *x, std::int64_t incx,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              const float *x, std::int64_t incx, const float *y, std::int64_t incy,
                              float *a, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, const float *x, std::int64_t incx, const float *y,
                               std::int64_t incy, float *a,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void spr2_precondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, const double *x, std::int64_t incx, const double *y,
                              std::int64_t incy, double *a,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void spr2_postcondition(sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, const double *x, std::int64_t incx, const double *y,
                               std::int64_t incy, double *a,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotm_precondition(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx,
                              float *y, std::int64_t incy, float *param,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotm_postcondition(sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx,
                               float *y, std::int64_t incy, float *param,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotm_precondition(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                              double *y, std::int64_t incy, double *param,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotm_postcondition(sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx,
                               double *y, std::int64_t incy, double *param,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dot_precondition(sycl::queue &queue, std::int64_t n, const float *x,
                             std::int64_t incx, const float *y, std::int64_t incy, float *result,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dot_postcondition(sycl::queue &queue, std::int64_t n, const float *x,
                              std::int64_t incx, const float *y, std::int64_t incy, float *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dot_precondition(sycl::queue &queue, std::int64_t n, const double *x,
                             std::int64_t incx, const double *y, std::int64_t incy, double *result,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dot_postcondition(sycl::queue &queue, std::int64_t n, const double *x,
                              std::int64_t incx, const double *y, std::int64_t incy, double *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void dot_precondition(sycl::queue &queue, std::int64_t n, const float *x,
                             std::int64_t incx, const float *y, std::int64_t incy, double *result,
                             const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void dot_postcondition(sycl::queue &queue, std::int64_t n, const float *x,
                              std::int64_t incx, const float *y, std::int64_t incy, double *result,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void sdsdot_precondition(sycl::queue &queue, std::int64_t n, float sb, const float *x,
                                std::int64_t incx, const float *y, std::int64_t incy, float *result,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void sdsdot_postcondition(sycl::queue &queue, std::int64_t n, float sb, const float *x,
                                 std::int64_t incx, const float *y, std::int64_t incy,
                                 float *result, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<float> alpha,
                               const std::complex<float> *a, std::int64_t lda,
                               const std::complex<float> *b, std::int64_t ldb, float beta,
                               std::complex<float> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                const std::complex<float> *a, std::int64_t lda,
                                const std::complex<float> *b, std::int64_t ldb, float beta,
                                std::complex<float> *c, std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void her2k_precondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<double> alpha,
                               const std::complex<double> *a, std::int64_t lda,
                               const std::complex<double> *b, std::int64_t ldb, double beta,
                               std::complex<double> *c, std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void her2k_postcondition(sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                const std::complex<double> *a, std::int64_t lda,
                                const std::complex<double> *b, std::int64_t ldb, double beta,
                                std::complex<double> *c, std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(sycl::queue &queue, float *a, float *b, float *c, float *s,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(sycl::queue &queue, float *a, float *b, float *c, float *s,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(sycl::queue &queue, double *a, double *b, double *c, double *s,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(sycl::queue &queue, double *a, double *b, double *c, double *s,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(sycl::queue &queue, std::complex<float> *a,
                              std::complex<float> *b, float *c, std::complex<float> *s,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(sycl::queue &queue, std::complex<float> *a,
                               std::complex<float> *b, float *c, std::complex<float> *s,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(sycl::queue &queue, std::complex<double> *a,
                              std::complex<double> *b, double *c, std::complex<double> *s,
                              const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(sycl::queue &queue, std::complex<double> *a,
                               std::complex<double> *b, double *c, std::complex<double> *s,
                               const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, float alpha, const float *a,
                                        std::int64_t lda, std::int64_t stride_a, float *b,
                                        std::int64_t ldb, std::int64_t stride_b,
                                        std::int64_t batch_size,
                                        const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, float alpha, const float *a,
                                         std::int64_t lda, std::int64_t stride_a, float *b,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, double alpha, const double *a,
                                        std::int64_t lda, std::int64_t stride_a, double *b,
                                        std::int64_t ldb, std::int64_t stride_b,
                                        std::int64_t batch_size,
                                        const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, double alpha, const double *a,
                                         std::int64_t lda, std::int64_t stride_a, double *b,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, std::complex<float> alpha,
                                        const std::complex<float> *a, std::int64_t lda,
                                        std::int64_t stride_a, std::complex<float> *b,
                                        std::int64_t ldb, std::int64_t stride_b,
                                        std::int64_t batch_size,
                                        const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<float> alpha,
                                         const std::complex<float> *a, std::int64_t lda,
                                         std::int64_t stride_a, std::complex<float> *b,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, std::complex<double> alpha,
                                        const std::complex<double> *a, std::int64_t lda,
                                        std::int64_t stride_a, std::complex<double> *b,
                                        std::int64_t ldb, std::int64_t stride_b,
                                        std::int64_t batch_size,
                                        const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<double> alpha,
                                         const std::complex<double> *a, std::int64_t lda,
                                         std::int64_t stride_a, std::complex<double> *b,
                                         std::int64_t ldb, std::int64_t stride_b,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, float alpha, float *ab, std::int64_t lda,
                                        std::int64_t ldb, std::int64_t stride,
                                        std::int64_t batch_size,
                                        const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, float alpha, float *ab, std::int64_t lda,
                                         std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, double alpha, double *ab, std::int64_t lda,
                                        std::int64_t ldb, std::int64_t stride,
                                        std::int64_t batch_size,
                                        const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, double alpha, double *ab, std::int64_t lda,
                                         std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, std::complex<float> alpha,
                                        std::complex<float> *ab, std::int64_t lda, std::int64_t ldb,
                                        std::int64_t stride, std::int64_t batch_size,
                                        const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<float> alpha,
                                         std::complex<float> *ab, std::int64_t lda,
                                         std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_precondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                        std::int64_t n, std::complex<double> alpha,
                                        std::complex<double> *ab, std::int64_t lda,
                                        std::int64_t ldb, std::int64_t stride,
                                        std::int64_t batch_size,
                                        const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void imatcopy_batch_postcondition(sycl::queue &queue, transpose trans, std::int64_t m,
                                         std::int64_t n, std::complex<double> alpha,
                                         std::complex<double> *ab, std::int64_t lda,
                                         std::int64_t ldb, std::int64_t stride,
                                         std::int64_t batch_size,
                                         const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                       std::int64_t m, std::int64_t n, float alpha, const float *a,
                                       std::int64_t lda, std::int64_t stride_a, float beta,
                                       const float *b, std::int64_t ldb, std::int64_t stride_b,
                                       float *c, std::int64_t ldc, std::int64_t stride_c,
                                       std::int64_t batch_size,
                                       const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                        std::int64_t m, std::int64_t n, float alpha, const float *a,
                                        std::int64_t lda, std::int64_t stride_a, float beta,
                                        const float *b, std::int64_t ldb, std::int64_t stride_b,
                                        float *c, std::int64_t ldc, std::int64_t stride_c,
                                        std::int64_t batch_size,
                                        const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                       std::int64_t m, std::int64_t n, double alpha,
                                       const double *a, std::int64_t lda, std::int64_t stride_a,
                                       double beta, const double *b, std::int64_t ldb,
                                       std::int64_t stride_b, double *c, std::int64_t ldc,
                                       std::int64_t stride_c, std::int64_t batch_size,
                                       const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                        std::int64_t m, std::int64_t n, double alpha,
                                        const double *a, std::int64_t lda, std::int64_t stride_a,
                                        double beta, const double *b, std::int64_t ldb,
                                        std::int64_t stride_b, double *c, std::int64_t ldc,
                                        std::int64_t stride_c, std::int64_t batch_size,
                                        const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_precondition(
    sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    std::int64_t stride_a, std::complex<float> beta, const std::complex<float> *b, std::int64_t ldb,
    std::int64_t stride_b, std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_postcondition(
    sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    std::int64_t stride_a, std::complex<float> beta, const std::complex<float> *b, std::int64_t ldb,
    std::int64_t stride_b, std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_precondition(sycl::queue &queue, transpose transa, transpose transb,
                                       std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                       const std::complex<double> *a, std::int64_t lda,
                                       std::int64_t stride_a, std::complex<double> beta,
                                       const std::complex<double> *b, std::int64_t ldb,
                                       std::int64_t stride_b, std::complex<double> *c,
                                       std::int64_t ldc, std::int64_t stride_c,
                                       std::int64_t batch_size,
                                       const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add prechecks to queue here for input args.  */
#endif
}

inline void omatadd_batch_postcondition(sycl::queue &queue, transpose transa, transpose transb,
                                        std::int64_t m, std::int64_t n, std::complex<double> alpha,
                                        const std::complex<double> *a, std::int64_t lda,
                                        std::int64_t stride_a, std::complex<double> beta,
                                        const std::complex<double> *b, std::int64_t ldb,
                                        std::int64_t stride_b, std::complex<double> *c,
                                        std::int64_t ldc, std::int64_t stride_c,
                                        std::int64_t batch_size,
                                        const std::vector<sycl::event> &dependencies) {
#ifndef ONEMKL_DISABLE_PREDICATES
    /* add postchecks to queue here for input args.  */
#endif
}
