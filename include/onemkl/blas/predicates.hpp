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

#ifndef _ONEMKL_BLAS_PREDICATES_HPP_
#define _ONEMKL_BLAS_PREDICATES_HPP_

#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>

#include "onemkl/detail/exceptions.hpp"
#include "onemkl/types.hpp"

namespace onemkl {
namespace blas {

inline void herk_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, float alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void herk_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, float alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                               std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void herk_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, double alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void herk_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, double alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                               std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(cl::sycl::queue &queue, std::int64_t n, float alpha,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(cl::sycl::queue &queue, std::int64_t n, float alpha,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(cl::sycl::queue &queue, std::int64_t n, double alpha,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(cl::sycl::queue &queue, std::int64_t n, double alpha,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(cl::sycl::queue &queue, std::int64_t n, float alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(cl::sycl::queue &queue, std::int64_t n, float alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void scal_precondition(cl::sycl::queue &queue, std::int64_t n, double alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void scal_postcondition(cl::sycl::queue &queue, std::int64_t n, double alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                              std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                               std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                              std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                               std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                               std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              cl::sycl::buffer<std::complex<float>, 1> &a,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &a,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tpmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              cl::sycl::buffer<std::complex<double>, 1> &a,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tpmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &a,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void spr_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<float, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void spr_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<float, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void spr_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<double, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void spr_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<float, 1> &beta,
    cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc, std::int64_t group_count,
    cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<float, 1> &beta,
    cl::sycl::buffer<float, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc, std::int64_t group_count,
    cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<double, 1> &alpha, cl::sycl::buffer<double, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<double, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<double, 1> &beta,
    cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<double, 1> &alpha, cl::sycl::buffer<double, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<double, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<double, 1> &beta,
    cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<float>, 1> &alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<float>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<float>, 1> &beta,
    cl::sycl::buffer<std::complex<float>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<float>, 1> &alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<float>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<float>, 1> &beta,
    cl::sycl::buffer<std::complex<float>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<double>, 1> &alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<double>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<double>, 1> &beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<double>, 1> &alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<double>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<double>, 1> &beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                    cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                                    std::int64_t ldb, std::int64_t stride_b, float beta,
                                    cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                     cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                                     std::int64_t ldb, std::int64_t stride_b, float beta,
                                     cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                                    cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                                    std::int64_t ldb, std::int64_t stride_b, double beta,
                                    cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                                     cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                                     std::int64_t ldb, std::int64_t stride_b, double beta,
                                     cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k,
                                    std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a,
                                    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                    std::int64_t stride_b, std::complex<float> beta,
                                    cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k,
                                     std::complex<float> alpha,
                                     cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::complex<float> beta,
                                     cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_batch_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                    std::int64_t m, std::int64_t n, std::int64_t k,
                                    std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a,
                                    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                    std::int64_t stride_b, std::complex<double> beta,
                                    cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                    std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_batch_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                     std::int64_t m, std::int64_t n, std::int64_t k,
                                     std::complex<double> alpha,
                                     cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::complex<double> beta,
                                     cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                                     std::int64_t stride_c, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, float alpha,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
                              cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, float alpha,
                               cl::sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
                               cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, double alpha,
                              cl::sycl::buffer<double, 1> &a, std::int64_t lda, double beta,
                              cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, double alpha,
                               cl::sycl::buffer<double, 1> &a, std::int64_t lda, double beta,
                               cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syrk_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              std::int64_t n, std::int64_t k, std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syrk_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void her2_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void her2_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void her2_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void her2_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hbmv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hbmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hbmv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hbmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(cl::sycl::queue &queue, std::int64_t n,
                             cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                             float c, float s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              float c, float s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(cl::sycl::queue &queue, std::int64_t n,
                             cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                             double c, double s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              double c, double s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                             std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                             float c, float s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                              std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                              float c, float s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rot_precondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                             std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                             double c, double s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rot_postcondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy, double c,
                              double s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(cl::sycl::queue &queue, std::int64_t n, float alpha,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(cl::sycl::queue &queue, std::int64_t n, float alpha,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(cl::sycl::queue &queue, std::int64_t n, double alpha,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(cl::sycl::queue &queue, std::int64_t n, double alpha,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void axpy_precondition(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void axpy_postcondition(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gerc_precondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gerc_postcondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gerc_precondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gerc_postcondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, float alpha,
                               cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                               cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, float alpha,
                                cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, double alpha,
                               cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                               cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, double alpha,
                                cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                std::complex<float> beta,
                                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2k_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2k_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                std::complex<double> beta,
                                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                              std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                              float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                               std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                              std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                               std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemv_precondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemv_postcondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void her_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void her_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void her_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void her_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                              std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &a,
                              std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<std::complex<float>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<std::complex<double>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                              std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_ext_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                  std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                  cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<half, 1> &b, std::int64_t ldb, float beta,
                                  cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_ext_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                   std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                   cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                                   cl::sycl::buffer<half, 1> &b, std::int64_t ldb, float beta,
                                   cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_ext_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                  offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                  float alpha, cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda,
                                  int8_t ao, cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb,
                                  uint8_t bo, float beta, cl::sycl::buffer<int32_t, 1> &c,
                                  std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_ext_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                   offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                                   float alpha, cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda,
                                   int8_t ao, cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb,
                                   uint8_t bo, float beta, cl::sycl::buffer<int32_t, 1> &c,
                                   std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_ext_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                  std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                  cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                  cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_ext_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                   std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                                   cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                   cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                   cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_ext_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                  std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                                  cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                  cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_ext_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                   std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                                   cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                   cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                   cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_ext_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                  std::int64_t m, std::int64_t n, std::int64_t k,
                                  std::complex<float> alpha,
                                  cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                  std::complex<float> beta,
                                  cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_ext_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                   std::int64_t m, std::int64_t n, std::int64_t k,
                                   std::complex<float> alpha,
                                   cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                   cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                   std::complex<float> beta,
                                   cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_ext_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                  std::int64_t m, std::int64_t n, std::int64_t k,
                                  std::complex<double> alpha,
                                  cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                  std::complex<double> beta,
                                  cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_ext_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                   std::int64_t m, std::int64_t n, std::int64_t k,
                                   std::complex<double> alpha,
                                   cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                   cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                   std::complex<double> beta,
                                   cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_ext_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                  std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
                                  cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                                  cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                                  cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_ext_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                                   std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
                                   cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                                   cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                                   cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void iamin_precondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void iamin_postcondition(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hpmv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hpmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hpmv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hpmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void spmv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                              std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void spmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, cl::sycl::buffer<float, 1> &a,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                               cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void spmv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, cl::sycl::buffer<double, 1> &a,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void spmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, cl::sycl::buffer<double, 1> &a,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                               cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rotmg_precondition(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
                               cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1,
                               float y1, cl::sycl::buffer<float, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rotmg_postcondition(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
                                cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1,
                                float y1, cl::sycl::buffer<float, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rotmg_precondition(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
                               cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1,
                               double y1, cl::sycl::buffer<double, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rotmg_postcondition(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
                                cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1,
                                double y1, cl::sycl::buffer<double, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                              std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void swap_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void swap_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void geru_precondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void geru_postcondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void geru_precondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void geru_postcondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                              std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void nrm2_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void nrm2_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k, float alpha,
                               cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                               cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k, float alpha,
                                cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                                cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k, double alpha,
                               cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                               cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k, double alpha,
                                cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                                cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k,
                                std::complex<float> alpha,
                                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                std::complex<float> beta,
                                cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemmt_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                               transpose transb, std::int64_t n, std::int64_t k,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemmt_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                                transpose transb, std::int64_t n, std::int64_t k,
                                std::complex<double> alpha,
                                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                std::complex<double> beta,
                                cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                              cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                               cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                               cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                              cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                              cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                               cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                               cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                              std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gemm_precondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
                              cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                              cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gemm_postcondition(cl::sycl::queue &queue, transpose transa, transpose transb,
                               std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
                               cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                               cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syr2_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syr2_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void ger_precondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                             cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                             cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void ger_postcondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void ger_precondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                             cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                             cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void ger_postcondition(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void dotu_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<float>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void dotu_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<float>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void dotu_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<double>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void dotu_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<double>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hemm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hemm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hemm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                              std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hemm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr2_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<float>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr2_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<float>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hpr2_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<double>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hpr2_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<double>, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                              cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku, float alpha,
                               cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                               cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                              cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku, double alpha,
                               cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                               cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void gbmv_precondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                              std::int64_t n, std::int64_t kl, std::int64_t ku,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void gbmv_postcondition(cl::sycl::queue &queue, transpose trans, std::int64_t m,
                               std::int64_t n, std::int64_t kl, std::int64_t ku,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tbmv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tbmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, float alpha,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                              cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, float alpha,
                               cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                               cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, double alpha,
                              cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                              cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, double alpha,
                               cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                               cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                              std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void symm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              std::int64_t m, std::int64_t n, std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                              std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void symm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               std::int64_t m, std::int64_t n, std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void dotc_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<float>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void dotc_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<float>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void dotc_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<std::complex<double>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void dotc_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<std::complex<double>, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syr_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                             cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syr_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void syr_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                             cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                             cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void syr_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trmm_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trmm_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                               transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void symv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                              cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void symv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                               cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void symv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void symv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                               cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              cl::sycl::buffer<std::complex<float>, 1> &a,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &a,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tpsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              cl::sycl::buffer<std::complex<double>, 1> &a,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tpsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &a,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                              std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, cl::sycl::buffer<float, 1> &a,
                               std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                              std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, cl::sycl::buffer<double, 1> &a,
                               std::int64_t lda, cl::sycl::buffer<double, 1> &x,
                               std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                              std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void copy_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void copy_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hemv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                              std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hemv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               std::complex<float> beta,
                               cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void hemv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void hemv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               std::complex<double> beta,
                               cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void iamax_precondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void iamax_postcondition(cl::sycl::queue &queue, std::int64_t n,
                                cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<std::int64_t, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void sbmv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                              std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                              float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void sbmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                               std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               float beta, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void sbmv_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                              std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void sbmv_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                               std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               double beta, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                              std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void asum_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void asum_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void tbsv_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                              diag unit_diag, std::int64_t n, std::int64_t k,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void tbsv_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               diag unit_diag, std::int64_t n, std::int64_t k,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void spr2_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                              cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<float, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void spr2_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               float alpha, cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<float, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void spr2_precondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                              double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<double, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void spr2_postcondition(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                               double alpha, cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<double, 1> &a) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<float, 1> &alpha,
    cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb, std::int64_t group_count,
    cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<float, 1> &alpha,
    cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb, std::int64_t group_count,
    cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<double, 1> &alpha,
    cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<double, 1> &alpha,
    cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::complex<float>, 1> &alpha,
    cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::complex<float>, 1> &alpha,
    cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::complex<double>, 1> &alpha,
    cl::sycl::buffer<std::complex<double>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::complex<double>, 1> &alpha,
    cl::sycl::buffer<std::complex<double>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                                    std::int64_t ldb, std::int64_t stride_b,
                                    std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                                     std::int64_t lda, std::int64_t stride_a,
                                     cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                                    std::int64_t ldb, std::int64_t stride_b,
                                    std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                                     std::int64_t lda, std::int64_t stride_a,
                                     cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    std::complex<float> alpha,
                                    cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a,
                                    cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                    std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, std::complex<float> alpha,
                                     cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void trsm_batch_precondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                    transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                                    std::complex<double> alpha,
                                    cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                    std::int64_t stride_a,
                                    cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                    std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void trsm_batch_postcondition(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                                     transpose trans, diag unit_diag, std::int64_t m,
                                     std::int64_t n, std::complex<double> alpha,
                                     cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                     std::int64_t stride_a,
                                     cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                     std::int64_t stride_b, std::int64_t batch_size) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rotm_precondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                              std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<float, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rotm_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<float, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rotm_precondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<double, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rotm_postcondition(cl::sycl::queue &queue, std::int64_t n,
                               cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                               cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                               cl::sycl::buffer<double, 1> &param) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void dot_precondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                             std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                             cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void dot_postcondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                              std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void dot_precondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                             std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                             cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void dot_postcondition(cl::sycl::queue &queue, std::int64_t n,
                              cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                              cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void dot_precondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                             std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                             cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void dot_postcondition(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                              std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                              cl::sycl::buffer<double, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void sdsdot_precondition(cl::sycl::queue &queue, std::int64_t n, float sb,
                                cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void sdsdot_postcondition(cl::sycl::queue &queue, std::int64_t n, float sb,
                                 cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                                 cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                                 cl::sycl::buffer<float, 1> &result) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void her2k_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<float> alpha,
                               cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                               float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                               std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void her2k_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<float> alpha,
                                cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                                cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                                float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                                std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void her2k_precondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                               std::int64_t n, std::int64_t k, std::complex<double> alpha,
                               cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                               cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                               double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                               std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void her2k_postcondition(cl::sycl::queue &queue, uplo upper_lower, transpose trans,
                                std::int64_t n, std::int64_t k, std::complex<double> alpha,
                                cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                                cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                                double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                                std::int64_t ldc) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
                              cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
                              cl::sycl::buffer<float, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
                               cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
                               cl::sycl::buffer<float, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
                              cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
                              cl::sycl::buffer<double, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
                               cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
                               cl::sycl::buffer<double, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
                              cl::sycl::buffer<std::complex<float>, 1> &b,
                              cl::sycl::buffer<float, 1> &c,
                              cl::sycl::buffer<std::complex<float>, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
                               cl::sycl::buffer<std::complex<float>, 1> &b,
                               cl::sycl::buffer<float, 1> &c,
                               cl::sycl::buffer<std::complex<float>, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

inline void rotg_precondition(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
                              cl::sycl::buffer<std::complex<double>, 1> &b,
                              cl::sycl::buffer<double, 1> &c,
                              cl::sycl::buffer<std::complex<double>, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add prechecks to queue here for input args.  */
#endif
}

inline void rotg_postcondition(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
                               cl::sycl::buffer<std::complex<double>, 1> &b,
                               cl::sycl::buffer<double, 1> &c,
                               cl::sycl::buffer<std::complex<double>, 1> &s) {
#ifndef ONEMKL_DISABLE_PREDICATES
        /* add postchecks to queue here for input args.  */
#endif
}

} //namespace blas
} //namespace onemkl

#endif //_ONEMKL_BLAS_PREDICATES_HPP_
