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

#ifndef _ONEMKL_BLAS_HPP_
#define _ONEMKL_BLAS_HPP_

#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>

#include "onemkl/types.hpp"

#include "onemkl/detail/backends_selector.hpp"

#include "onemkl/blas/predicates.hpp"

#include "onemkl/blas/detail/blas_loader.hpp"
#include "onemkl/blas/detail/cublas/blas_ct.hpp"
#include "onemkl/blas/detail/mklcpu/blas_ct.hpp"
#include "onemkl/blas/detail/mklgpu/blas_ct.hpp"

namespace onemkl {
namespace blas {

static inline void asum(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    detail::asum(select_backend(queue), queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

static inline void asum(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    detail::asum(select_backend(queue), queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

static inline void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    detail::asum(select_backend(queue), queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

static inline void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    detail::asum(select_backend(queue), queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

static inline void axpy(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    detail::axpy(select_backend(queue), queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

static inline void axpy(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    detail::axpy(select_backend(queue), queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

static inline void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    detail::axpy(select_backend(queue), queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

static inline void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    detail::axpy(select_backend(queue), queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

static inline void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    detail::copy(select_backend(queue), queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

static inline void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    detail::copy(select_backend(queue), queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

static inline void copy(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    detail::copy(select_backend(queue), queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

static inline void copy(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    detail::copy(select_backend(queue), queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

static inline void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    detail::dot(select_backend(queue), queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    detail::dot(select_backend(queue), queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    detail::dot(select_backend(queue), queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dotc(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result) {
    dotc_precondition(queue, n, x, incx, y, incy, result);
    detail::dotc(select_backend(queue), queue, n, x, incx, y, incy, result);
    dotc_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dotc(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result) {
    dotc_precondition(queue, n, x, incx, y, incy, result);
    detail::dotc(select_backend(queue), queue, n, x, incx, y, incy, result);
    dotc_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dotu(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result) {
    dotu_precondition(queue, n, x, incx, y, incy, result);
    detail::dotu(select_backend(queue), queue, n, x, incx, y, incy, result);
    dotu_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dotu(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result) {
    dotu_precondition(queue, n, x, incx, y, incy, result);
    detail::dotu(select_backend(queue), queue, n, x, incx, y, incy, result);
    dotu_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    detail::gbmv(select_backend(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                 incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    detail::gbmv(select_backend(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                 incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    detail::gbmv(select_backend(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                 incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    detail::gbmv(select_backend(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                 incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                        float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, half alpha, cl::sycl::buffer<half, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                        cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm_batch(cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
                              cl::sycl::buffer<transpose, 1> &transb,
                              cl::sycl::buffer<std::int64_t, 1> &m,
                              cl::sycl::buffer<std::int64_t, 1> &n,
                              cl::sycl::buffer<std::int64_t, 1> &k,
                              cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
                              cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
                              cl::sycl::buffer<std::int64_t, 1> &ldb,
                              cl::sycl::buffer<float, 1> &beta, cl::sycl::buffer<float, 1> &c,
                              cl::sycl::buffer<std::int64_t, 1> &ldc, std::int64_t group_count,
                              cl::sycl::buffer<std::int64_t, 1> &group_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size);
    detail::gemm_batch(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                       beta, c, ldc, group_count, group_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size);
}

static inline void gemm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<double, 1> &alpha, cl::sycl::buffer<double, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<double, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<double, 1> &beta,
    cl::sycl::buffer<double, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size);
    detail::gemm_batch(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                       beta, c, ldc, group_count, group_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size);
}

static inline void gemm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<float>, 1> &alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<float>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<float>, 1> &beta,
    cl::sycl::buffer<std::complex<float>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size);
    detail::gemm_batch(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                       beta, c, ldc, group_count, group_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size);
}

static inline void gemm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<transpose, 1> &transa,
    cl::sycl::buffer<transpose, 1> &transb, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::int64_t, 1> &k,
    cl::sycl::buffer<std::complex<double>, 1> &alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
    cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<std::complex<double>, 1> &b,
    cl::sycl::buffer<std::int64_t, 1> &ldb, cl::sycl::buffer<std::complex<double>, 1> &beta,
    cl::sycl::buffer<std::complex<double>, 1> &c, cl::sycl::buffer<std::int64_t, 1> &ldc,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size);
    detail::gemm_batch(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                       beta, c, ldc, group_count, group_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size);
}

static inline void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                              cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, float beta,
                              cl::sycl::buffer<float, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
    detail::gemm_batch(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                              cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, double beta,
                              cl::sycl::buffer<double, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
    detail::gemm_batch(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::complex<float> beta,
                              cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
    detail::gemm_batch(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_batch(cl::sycl::queue &queue, transpose transa, transpose transb,
                              std::int64_t m, std::int64_t n, std::int64_t k,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::complex<double> beta,
                              cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc,
                              std::int64_t stride_c, std::int64_t batch_size) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size);
    detail::gemm_batch(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                            cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<half, 1> &b, std::int64_t ldb, float beta,
                            cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm_ext(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                     beta, c, ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                            float alpha, cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda,
                            int8_t ao, cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb,
                            uint8_t bo, float beta, cl::sycl::buffer<int32_t, 1> &c,
                            std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
    gemm_ext_precondition(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo,
                          beta, c, ldc, co);
    detail::gemm_ext(select_backend(queue), queue, transa, transb, offsetc, m, n, k, alpha, a, lda,
                     ao, b, ldb, bo, beta, c, ldc, co);
    gemm_ext_postcondition(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo,
                           beta, c, ldc, co);
}

static inline void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k, float alpha,
                            cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                            cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm_ext(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                     beta, c, ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k, double alpha,
                            cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                            cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm_ext(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                     beta, c, ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k,
                            std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                            std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                            std::int64_t ldb, std::complex<float> beta,
                            cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm_ext(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                     beta, c, ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k,
                            std::complex<double> alpha,
                            cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                            std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                            std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm_ext(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                     beta, c, ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm_ext(cl::sycl::queue &queue, transpose transa, transpose transb,
                            std::int64_t m, std::int64_t n, std::int64_t k, half alpha,
                            cl::sycl::buffer<half, 1> &a, std::int64_t lda,
                            cl::sycl::buffer<half, 1> &b, std::int64_t ldb, half beta,
                            cl::sycl::buffer<half, 1> &c, std::int64_t ldc) {
    gemm_ext_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm_ext(select_backend(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                     beta, c, ldc);
    gemm_ext_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                         transpose transb, std::int64_t n, std::int64_t k, float alpha,
                         cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc);
    detail::gemmt(select_backend(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                  ldb, beta, c, ldc);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

static inline void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                         transpose transb, std::int64_t n, std::int64_t k, double alpha,
                         cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                         cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc);
    detail::gemmt(select_backend(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                  ldb, beta, c, ldc);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

static inline void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                         transpose transb, std::int64_t n, std::int64_t k,
                         std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b,
                         std::int64_t ldb, std::complex<float> beta,
                         cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc);
    detail::gemmt(select_backend(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                  ldb, beta, c, ldc);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

static inline void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                         transpose transb, std::int64_t n, std::int64_t k,
                         std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b,
                         std::int64_t ldb, std::complex<double> beta,
                         cl::sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc);
    detail::gemmt(select_backend(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                  ldb, beta, c, ldc);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::gemv(select_backend(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::gemv(select_backend(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::gemv(select_backend(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::gemv(select_backend(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::ger(select_backend(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::ger(select_backend(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::gerc(select_backend(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::gerc(select_backend(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::geru(select_backend(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::geru(select_backend(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    detail::hbmv(select_backend(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                 incy);
    hbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    detail::hbmv(select_backend(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                 incy);
    hbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc) {
    hemm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::hemm(select_backend(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
    hemm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void hemm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc) {
    hemm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::hemm(select_backend(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
    hemm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::hemv(select_backend(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::hemv(select_backend(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    detail::her(select_backend(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    detail::her(select_backend(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    detail::her2(select_backend(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    detail::her2(select_backend(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<float> alpha,
                         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    her2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::her2k(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
    her2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<double> alpha,
                         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                         double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                         std::int64_t ldc) {
    her2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::her2k(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
    her2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::herk(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, double alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::herk(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    detail::hpmv(select_backend(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    detail::hpmv(select_backend(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &a) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    detail::hpr(select_backend(queue), queue, upper_lower, n, alpha, x, incx, a);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

static inline void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &a) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    detail::hpr(select_backend(queue), queue, upper_lower, n, alpha, x, incx, a);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

static inline void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    detail::hpr2(select_backend(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    detail::hpr2(select_backend(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    detail::iamax(select_backend(queue), queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

static inline void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    detail::iamax(select_backend(queue), queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

static inline void iamax(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    detail::iamax(select_backend(queue), queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

static inline void iamax(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    detail::iamax(select_backend(queue), queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

static inline void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    detail::iamin(select_backend(queue), queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

static inline void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    detail::iamin(select_backend(queue), queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

static inline void iamin(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    detail::iamin(select_backend(queue), queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

static inline void iamin(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    detail::iamin(select_backend(queue), queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

static inline void nrm2(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    detail::nrm2(select_backend(queue), queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

static inline void nrm2(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    detail::nrm2(select_backend(queue), queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

static inline void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    detail::nrm2(select_backend(queue), queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

static inline void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    detail::nrm2(select_backend(queue), queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

static inline void rot(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
                       float s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    detail::rot(select_backend(queue), queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

static inline void rot(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c,
                       double s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    detail::rot(select_backend(queue), queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

static inline void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c,
                       float s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    detail::rot(select_backend(queue), queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

static inline void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       double c, double s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    detail::rot(select_backend(queue), queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
                        cl::sycl::buffer<float, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    detail::rotg(select_backend(queue), queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
                        cl::sycl::buffer<double, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    detail::rotg(select_backend(queue), queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
                        cl::sycl::buffer<std::complex<float>, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    detail::rotg(select_backend(queue), queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &b,
                        cl::sycl::buffer<double, 1> &c,
                        cl::sycl::buffer<std::complex<double>, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    detail::rotg(select_backend(queue), queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

static inline void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &param) {
    rotm_precondition(queue, n, x, incx, y, incy, param);
    detail::rotm(select_backend(queue), queue, n, x, incx, y, incy, param);
    rotm_postcondition(queue, n, x, incx, y, incy, param);
}

static inline void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &param) {
    rotm_precondition(queue, n, x, incx, y, incy, param);
    detail::rotm(select_backend(queue), queue, n, x, incx, y, incy, param);
    rotm_postcondition(queue, n, x, incx, y, incy, param);
}

static inline void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
                         cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1, float y1,
                         cl::sycl::buffer<float, 1> &param) {
    rotmg_precondition(queue, d1, d2, x1, y1, param);
    detail::rotmg(select_backend(queue), queue, d1, d2, x1, y1, param);
    rotmg_postcondition(queue, d1, d2, x1, y1, param);
}

static inline void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
                         cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1,
                         double y1, cl::sycl::buffer<double, 1> &param) {
    rotmg_precondition(queue, d1, d2, x1, y1, param);
    detail::rotmg(select_backend(queue), queue, d1, d2, x1, y1, param);
    rotmg_postcondition(queue, d1, d2, x1, y1, param);
}

static inline void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    detail::sbmv(select_backend(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                 incy);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    detail::sbmv(select_backend(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                 incy);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(select_backend(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(select_backend(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(select_backend(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(select_backend(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(select_backend(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(select_backend(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb,
                          cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                          cl::sycl::buffer<float, 1> &result) {
    sdsdot_precondition(queue, n, sb, x, incx, y, incy, result);
    detail::sdsdot(select_backend(queue), queue, n, sb, x, incx, y, incy, result);
    sdsdot_postcondition(queue, n, sb, x, incx, y, incy, result);
}

static inline void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
                        std::int64_t incy) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    detail::spmv(select_backend(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y,
                        std::int64_t incy) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    detail::spmv(select_backend(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &a) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    detail::spr(select_backend(queue), queue, upper_lower, n, alpha, x, incx, a);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

static inline void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &a) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    detail::spr(select_backend(queue), queue, upper_lower, n, alpha, x, incx, a);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

static inline void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    detail::spr2(select_backend(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    detail::spr2(select_backend(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    detail::swap(select_backend(queue), queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

static inline void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    detail::swap(select_backend(queue), queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

static inline void swap(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    detail::swap(select_backend(queue), queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

static inline void swap(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    detail::swap(select_backend(queue), queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

static inline void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                        float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::symm(select_backend(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                        double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::symm(select_backend(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::symm(select_backend(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::symm(select_backend(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::symv(select_backend(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::symv(select_backend(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    detail::syr(select_backend(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    detail::syr(select_backend(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    detail::syr2(select_backend(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    detail::syr2(select_backend(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                         float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::syr2k(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                         double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::syr2k(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<float> alpha,
                         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
                         std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                         std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::syr2k(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<double> alpha,
                         cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
                         std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                         std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::syr2k(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, float beta, cl::sycl::buffer<float, 1> &c,
                        std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::syrk(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, double beta, cl::sycl::buffer<double, 1> &c,
                        std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::syrk(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::syrk(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::syrk(select_backend(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                 incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                 incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                 incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                 incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                 incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                 incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                 incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x,
                 incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    detail::trmm(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

static inline void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    detail::trmm(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

static inline void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    detail::trmm(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

static inline void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    detail::trmm(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trmv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    detail::trsm(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

static inline void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    detail::trsm(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

static inline void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    detail::trsm(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

static inline void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    detail::trsm(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

static inline void trsm_batch(cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
                              cl::sycl::buffer<uplo, 1> &upper_lower,
                              cl::sycl::buffer<transpose, 1> &trans,
                              cl::sycl::buffer<diag, 1> &unit_diag,
                              cl::sycl::buffer<std::int64_t, 1> &m,
                              cl::sycl::buffer<std::int64_t, 1> &n,
                              cl::sycl::buffer<float, 1> &alpha, cl::sycl::buffer<float, 1> &a,
                              cl::sycl::buffer<std::int64_t, 1> &lda, cl::sycl::buffer<float, 1> &b,
                              cl::sycl::buffer<std::int64_t, 1> &ldb, std::int64_t group_count,
                              cl::sycl::buffer<std::int64_t, 1> &group_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            b, ldb, group_count, group_size);
    detail::trsm_batch(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m,
                       n, alpha, a, lda, b, ldb, group_count, group_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             b, ldb, group_count, group_size);
}

static inline void trsm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<double, 1> &alpha,
    cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            b, ldb, group_count, group_size);
    detail::trsm_batch(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m,
                       n, alpha, a, lda, b, ldb, group_count, group_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             b, ldb, group_count, group_size);
}

static inline void trsm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::complex<float>, 1> &alpha,
    cl::sycl::buffer<std::complex<float>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            b, ldb, group_count, group_size);
    detail::trsm_batch(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m,
                       n, alpha, a, lda, b, ldb, group_count, group_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             b, ldb, group_count, group_size);
}

static inline void trsm_batch(
    cl::sycl::queue &queue, cl::sycl::buffer<side, 1> &left_right,
    cl::sycl::buffer<uplo, 1> &upper_lower, cl::sycl::buffer<transpose, 1> &trans,
    cl::sycl::buffer<diag, 1> &unit_diag, cl::sycl::buffer<std::int64_t, 1> &m,
    cl::sycl::buffer<std::int64_t, 1> &n, cl::sycl::buffer<std::complex<double>, 1> &alpha,
    cl::sycl::buffer<std::complex<double>, 1> &a, cl::sycl::buffer<std::int64_t, 1> &lda,
    cl::sycl::buffer<std::complex<double>, 1> &b, cl::sycl::buffer<std::int64_t, 1> &ldb,
    std::int64_t group_count, cl::sycl::buffer<std::int64_t, 1> &group_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            b, ldb, group_count, group_size);
    detail::trsm_batch(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m,
                       n, alpha, a, lda, b, ldb, group_count, group_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             b, ldb, group_count, group_size);
}

static inline void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            stride_a, b, ldb, stride_b, batch_size);
    detail::trsm_batch(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m,
                       n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             stride_a, b, ldb, stride_b, batch_size);
}

static inline void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<double, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            stride_a, b, ldb, stride_b, batch_size);
    detail::trsm_batch(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m,
                       n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             stride_a, b, ldb, stride_b, batch_size);
}

static inline void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<float> alpha,
                              cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<float>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            stride_a, b, ldb, stride_b, batch_size);
    detail::trsm_batch(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m,
                       n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             stride_a, b, ldb, stride_b, batch_size);
}

static inline void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              std::complex<double> alpha,
                              cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<std::complex<double>, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            stride_a, b, ldb, stride_b, batch_size);
    detail::trsm_batch(select_backend(queue), queue, left_right, upper_lower, trans, unit_diag, m,
                       n, alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             stride_a, b, ldb, stride_b, batch_size);
}

static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trsv(select_backend(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

} //namespace blas
} //namespace onemkl

#endif //_ONEMKL_BLAS_LOADER_HPP_
