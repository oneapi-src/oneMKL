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

static inline void asum(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    detail::asum(get_device_id(queue), queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

static inline void asum(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    detail::asum(get_device_id(queue), queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

static inline void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    detail::asum(get_device_id(queue), queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

static inline void asum(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    asum_precondition(queue, n, x, incx, result);
    detail::asum(get_device_id(queue), queue, n, x, incx, result);
    asum_postcondition(queue, n, x, incx, result);
}

static inline void axpy(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

static inline void axpy(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

static inline void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

static inline void axpy(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy);
    detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy);
}

static inline void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    detail::copy(get_device_id(queue), queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

static inline void copy(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    detail::copy(get_device_id(queue), queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

static inline void copy(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    detail::copy(get_device_id(queue), queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

static inline void copy(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    copy_precondition(queue, n, x, incx, y, incy);
    detail::copy(get_device_id(queue), queue, n, x, incx, y, incy);
    copy_postcondition(queue, n, x, incx, y, incy);
}

static inline void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &result) {
    dot_precondition(queue, n, x, incx, y, incy, result);
    detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result);
    dot_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dotc(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result) {
    dotc_precondition(queue, n, x, incx, y, incy, result);
    detail::dotc(get_device_id(queue), queue, n, x, incx, y, incy, result);
    dotc_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dotc(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result) {
    dotc_precondition(queue, n, x, incx, y, incy, result);
    detail::dotc(get_device_id(queue), queue, n, x, incx, y, incy, result);
    dotc_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dotu(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<float>, 1> &result) {
    dotu_precondition(queue, n, x, incx, y, incy, result);
    detail::dotu(get_device_id(queue), queue, n, x, incx, y, incy, result);
    dotu_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void dotu(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<std::complex<double>, 1> &result) {
    dotu_precondition(queue, n, x, incx, y, incy, result);
    detail::dotu(get_device_id(queue), queue, n, x, incx, y, incy, result);
    dotu_postcondition(queue, n, x, incx, y, incy, result);
}

static inline void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                 incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gbmv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::int64_t kl, std::int64_t ku, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
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
    detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
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
    detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y,
                 incy);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                        float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
                        cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
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
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
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
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, cl::sycl::half alpha, cl::sycl::buffer<cl::sycl::half, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<cl::sycl::half, 1> &b, std::int64_t ldb, cl::sycl::half beta,
                        cl::sycl::buffer<cl::sycl::half, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void gemm(cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                        std::int64_t n, std::int64_t k, float alpha, cl::sycl::buffer<cl::sycl::half, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<cl::sycl::half, 1> &b, std::int64_t ldb,
                        float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta,
                 c, ldc);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
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
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
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
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
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
    detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda,
                       stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_size);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size);
}

static inline void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                             offset offsetc, std::int64_t m, std::int64_t n, std::int64_t k,
                             float alpha, cl::sycl::buffer<int8_t, 1> &a, std::int64_t lda,
                             int8_t ao, cl::sycl::buffer<uint8_t, 1> &b, std::int64_t ldb,
                             uint8_t bo, float beta, cl::sycl::buffer<int32_t, 1> &c,
                             std::int64_t ldc, cl::sycl::buffer<int32_t, 1> &co) {
    gemm_bias_precondition(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo,
                           beta, c, ldc, co);
    detail::gemm_bias(get_device_id(queue), queue, transa, transb, offsetc, m, n, k, alpha, a, lda,
                      ao, b, ldb, bo, beta, c, ldc, co);
    gemm_bias_postcondition(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo,
                            beta, c, ldc, co);
}

static inline void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa,
                         transpose transb, std::int64_t n, std::int64_t k, float alpha,
                         cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc);
    detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
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
    detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
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
    detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
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
    detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha, a, lda, b,
                  ldb, beta, c, ldc);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc);
}

static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void gemv(cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::ger(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void ger(cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::ger(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::gerc(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void gerc(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::gerc(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::geru(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void geru(cl::sycl::queue &queue, std::int64_t m, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
    detail::geru(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda);
}

static inline void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    detail::hbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                 incy);
    hbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void hbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    detail::hbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
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
    detail::hemm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
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
    detail::hemm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
    hemm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, std::complex<float> beta,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::hemv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void hemv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, std::complex<double> beta,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::hemv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    detail::her(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void her(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    detail::her(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    detail::her2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void her2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    detail::her2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void her2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, std::complex<float> alpha,
                         cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                         cl::sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
                         cl::sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    her2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::her2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
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
    detail::her2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
    her2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, float alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, float beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::herk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void herk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, double alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, double beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::herk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    detail::hpmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void hpmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    detail::hpmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &a) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    detail::hpr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

static inline void hpr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &a) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    detail::hpr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

static inline void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<float>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<float>, 1> &a) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    detail::hpr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void hpr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n,
                        std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<std::complex<double>, 1> &y,
                        std::int64_t incy, cl::sycl::buffer<std::complex<double>, 1> &a) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    detail::hpr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    detail::iamax(get_device_id(queue), queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

static inline void iamax(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    detail::iamax(get_device_id(queue), queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

static inline void iamax(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    detail::iamax(get_device_id(queue), queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

static inline void iamax(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result) {
    iamax_precondition(queue, n, x, incx, result);
    detail::iamax(get_device_id(queue), queue, n, x, incx, result);
    iamax_postcondition(queue, n, x, incx, result);
}

static inline void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    detail::iamin(get_device_id(queue), queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

static inline void iamin(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                         std::int64_t incx, cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    detail::iamin(get_device_id(queue), queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

static inline void iamin(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    detail::iamin(get_device_id(queue), queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

static inline void iamin(cl::sycl::queue &queue, std::int64_t n,
                         cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                         cl::sycl::buffer<std::int64_t, 1> &result) {
    iamin_precondition(queue, n, x, incx, result);
    detail::iamin(get_device_id(queue), queue, n, x, incx, result);
    iamin_postcondition(queue, n, x, incx, result);
}

static inline void nrm2(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    detail::nrm2(get_device_id(queue), queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

static inline void nrm2(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    detail::nrm2(get_device_id(queue), queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

static inline void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    detail::nrm2(get_device_id(queue), queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

static inline void nrm2(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &result) {
    nrm2_precondition(queue, n, x, incx, result);
    detail::nrm2(get_device_id(queue), queue, n, x, incx, result);
    nrm2_postcondition(queue, n, x, incx, result);
}

static inline void rot(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy, float c,
                       float s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

static inline void rot(cl::sycl::queue &queue, std::int64_t n,
                       cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy, double c,
                       double s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

static inline void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy, float c,
                       float s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

static inline void rot(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                       std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                       double c, double s) {
    rot_precondition(queue, n, x, incx, y, incy, c, s);
    detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s);
    rot_postcondition(queue, n, x, incx, y, incy, c, s);
}

static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &b, cl::sycl::buffer<float, 1> &c,
                        cl::sycl::buffer<float, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    detail::rotg(get_device_id(queue), queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &b, cl::sycl::buffer<double, 1> &c,
                        cl::sycl::buffer<double, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    detail::rotg(get_device_id(queue), queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &b, cl::sycl::buffer<float, 1> &c,
                        cl::sycl::buffer<std::complex<float>, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    detail::rotg(get_device_id(queue), queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

static inline void rotg(cl::sycl::queue &queue, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &b,
                        cl::sycl::buffer<double, 1> &c,
                        cl::sycl::buffer<std::complex<double>, 1> &s) {
    rotg_precondition(queue, a, b, c, s);
    detail::rotg(get_device_id(queue), queue, a, b, c, s);
    rotg_postcondition(queue, a, b, c, s);
}

static inline void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &param) {
    rotm_precondition(queue, n, x, incx, y, incy, param);
    detail::rotm(get_device_id(queue), queue, n, x, incx, y, incy, param);
    rotm_postcondition(queue, n, x, incx, y, incy, param);
}

static inline void rotm(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &param) {
    rotm_precondition(queue, n, x, incx, y, incy, param);
    detail::rotm(get_device_id(queue), queue, n, x, incx, y, incy, param);
    rotm_postcondition(queue, n, x, incx, y, incy, param);
}

static inline void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<float, 1> &d1,
                         cl::sycl::buffer<float, 1> &d2, cl::sycl::buffer<float, 1> &x1, float y1,
                         cl::sycl::buffer<float, 1> &param) {
    rotmg_precondition(queue, d1, d2, x1, y1, param);
    detail::rotmg(get_device_id(queue), queue, d1, d2, x1, y1, param);
    rotmg_postcondition(queue, d1, d2, x1, y1, param);
}

static inline void rotmg(cl::sycl::queue &queue, cl::sycl::buffer<double, 1> &d1,
                         cl::sycl::buffer<double, 1> &d2, cl::sycl::buffer<double, 1> &x1,
                         double y1, cl::sycl::buffer<double, 1> &param) {
    rotmg_precondition(queue, d1, d2, x1, y1, param);
    detail::rotmg(get_device_id(queue), queue, d1, d2, x1, y1, param);
    rotmg_postcondition(queue, d1, d2, x1, y1, param);
}

static inline void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    detail::sbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                 incy);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void sbmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
                        double alpha, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
    detail::sbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y,
                 incy);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, float alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void scal(cl::sycl::queue &queue, std::int64_t n, double alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    scal_precondition(queue, n, alpha, x, incx);
    detail::scal(get_device_id(queue), queue, n, alpha, x, incx);
    scal_postcondition(queue, n, alpha, x, incx);
}

static inline void sdsdot(cl::sycl::queue &queue, std::int64_t n, float sb,
                          cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                          cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                          cl::sycl::buffer<float, 1> &result) {
    sdsdot_precondition(queue, n, sb, x, incx, y, incy, result);
    detail::sdsdot(get_device_id(queue), queue, n, sb, x, incx, y, incy, result);
    sdsdot_postcondition(queue, n, sb, x, incx, y, incy, result);
}

static inline void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, float beta, cl::sycl::buffer<float, 1> &y,
                        std::int64_t incy) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    detail::spmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void spmv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, double beta, cl::sycl::buffer<double, 1> &y,
                        std::int64_t incy) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    detail::spmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy);
}

static inline void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &a) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    detail::spr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

static inline void spr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &a) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a);
    detail::spr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a);
}

static inline void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    detail::spr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void spr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
    detail::spr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a);
}

static inline void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<float, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    detail::swap(get_device_id(queue), queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

static inline void swap(cl::sycl::queue &queue, std::int64_t n, cl::sycl::buffer<double, 1> &x,
                        std::int64_t incx, cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    detail::swap(get_device_id(queue), queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

static inline void swap(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<float>, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    detail::swap(get_device_id(queue), queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

static inline void swap(cl::sycl::queue &queue, std::int64_t n,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<std::complex<double>, 1> &y, std::int64_t incy) {
    swap_precondition(queue, n, x, incx, y, incy);
    detail::swap(get_device_id(queue), queue, n, x, incx, y, incy);
    swap_postcondition(queue, n, x, incx, y, incy);
}

static inline void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                        float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void symm(cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                        std::int64_t n, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                        double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
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
    detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
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
    detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                 beta, c, ldc);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx, float beta,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::symv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void symv(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx, double beta,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
    detail::symv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx, beta, y,
                 incy);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy);
}

static inline void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                       cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    detail::syr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void syr(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                       cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                       cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda);
    detail::syr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda);
}

static inline void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<float, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    detail::syr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void syr2(cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx,
                        cl::sycl::buffer<double, 1> &y, std::int64_t incy,
                        cl::sycl::buffer<double, 1> &a, std::int64_t lda) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    detail::syr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda);
}

static inline void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<float, 1> &b, std::int64_t ldb,
                         float beta, cl::sycl::buffer<float, 1> &c, std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void syr2k(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                         std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                         std::int64_t lda, cl::sycl::buffer<double, 1> &b, std::int64_t ldb,
                         double beta, cl::sycl::buffer<double, 1> &c, std::int64_t ldc) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
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
    detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
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
    detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb,
                  beta, c, ldc);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, float alpha, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, float beta, cl::sycl::buffer<float, 1> &c,
                        std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, double alpha, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, double beta, cl::sycl::buffer<double, 1> &c,
                        std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, std::complex<float> alpha,
                        cl::sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
                        std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c,
                        std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void syrk(cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                        std::int64_t k, std::complex<double> alpha,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c,
                        std::int64_t ldc) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
    detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda, beta, c,
                 ldc);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc);
}

static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<float, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<double, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tbsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, std::int64_t k,
                        cl::sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx);
}

static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        cl::sycl::buffer<std::complex<float>, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void tpsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        cl::sycl::buffer<std::complex<double>, 1> &x, std::int64_t incx) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
    detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x, incx);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx);
}

static inline void trmm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
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
    detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
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
    detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
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
    detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trmv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsm(cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                        diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
                        cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &b, std::int64_t ldb) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                      ldb);
    detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
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
    detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
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
    detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
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
    detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                 alpha, a, lda, b, ldb);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb);
}

static inline void trsm_batch(cl::sycl::queue &queue, side left_right, uplo upper_lower,
                              transpose trans, diag unit_diag, std::int64_t m, std::int64_t n,
                              float alpha, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                              std::int64_t stride_a, cl::sycl::buffer<float, 1> &b,
                              std::int64_t ldb, std::int64_t stride_b, std::int64_t batch_size) {
    trsm_batch_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                            stride_a, b, ldb, stride_b, batch_size);
    detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                       alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
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
    detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                       alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
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
    detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                       alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
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
    detail::trsm_batch(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag, m, n,
                       alpha, a, lda, stride_a, b, ldb, stride_b, batch_size);
    trsm_batch_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda,
                             stride_a, b, ldb, stride_b, batch_size);
}

static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<float, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<float, 1> &x, std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<double, 1> &a, std::int64_t lda,
                        cl::sycl::buffer<double, 1> &x, std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<float>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &x,
                        std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

static inline void trsv(cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag,
                        std::int64_t n, cl::sycl::buffer<std::complex<double>, 1> &a,
                        std::int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &x,
                        std::int64_t incx) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx);
}

// USM APIs

static inline cl::sycl::event asum(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
    float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    asum_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::asum(get_device_id(queue), queue, n, x, incx, result, dependencies);
    asum_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event asum(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
    double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    asum_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::asum(get_device_id(queue), queue, n, x, incx, result, dependencies);
    asum_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event asum(
    cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    asum_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::asum(get_device_id(queue), queue, n, x, incx, result, dependencies);
    asum_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event asum(
    cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    asum_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::asum(get_device_id(queue), queue, n, x, incx, result, dependencies);
    asum_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event axpy(
    cl::sycl::queue &queue, std::int64_t n, float alpha, const float *x, std::int64_t incx,
    float *y, std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy, dependencies);
    auto done = detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy, dependencies);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event axpy(
    cl::sycl::queue &queue, std::int64_t n, double alpha, const double *x, std::int64_t incx,
    double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy, dependencies);
    auto done = detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy, dependencies);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event axpy(
    cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha, const std::complex<float> *x,
    std::int64_t incx, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy, dependencies);
    auto done = detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy, dependencies);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event axpy(
    cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    axpy_precondition(queue, n, alpha, x, incx, y, incy, dependencies);
    auto done = detail::axpy(get_device_id(queue), queue, n, alpha, x, incx, y, incy, dependencies);
    axpy_postcondition(queue, n, alpha, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event axpy_batch(
    cl::sycl::queue &queue, std::int64_t *n, float *alpha, const float **x, std::int64_t *incx,
    float **y, std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    axpy_batch_precondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, y, incy,
                                   group_count, group_size, dependencies);
    axpy_batch_postcondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

static inline cl::sycl::event axpy_batch(
    cl::sycl::queue &queue, std::int64_t *n, double *alpha, const double **x, std::int64_t *incx,
    double **y, std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    axpy_batch_precondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, y, incy,
                                   group_count, group_size, dependencies);
    axpy_batch_postcondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

static inline cl::sycl::event axpy_batch(
    cl::sycl::queue &queue, std::int64_t *n, std::complex<float> *alpha,
    const std::complex<float> **x, std::int64_t *incx, std::complex<float> **y, std::int64_t *incy,
    std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    axpy_batch_precondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, y, incy,
                                   group_count, group_size, dependencies);
    axpy_batch_postcondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

static inline cl::sycl::event axpy_batch(
    cl::sycl::queue &queue, std::int64_t *n, std::complex<double> *alpha,
    const std::complex<double> **x, std::int64_t *incx, std::complex<double> **y,
    std::int64_t *incy, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    axpy_batch_precondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                            dependencies);
    auto done = detail::axpy_batch(get_device_id(queue), queue, n, alpha, x, incx, y, incy,
                                   group_count, group_size, dependencies);
    axpy_batch_postcondition(queue, n, alpha, x, incx, y, incy, group_count, group_size,
                             dependencies);
    return done;
}

static inline cl::sycl::event copy(
    cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    copy_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = detail::copy(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    copy_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event copy(
    cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    copy_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = detail::copy(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    copy_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event copy(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    copy_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = detail::copy(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    copy_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event copy(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    copy_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = detail::copy(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    copy_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event dot(
    cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, const float *y,
    std::int64_t incy, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    dot_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done = detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    dot_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline cl::sycl::event dot(
    cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, const double *y,
    std::int64_t incy, double *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    dot_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done = detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    dot_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline cl::sycl::event dot(
    cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, const float *y,
    std::int64_t incy, double *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    dot_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done = detail::dot(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    dot_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline cl::sycl::event dotc(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
    const std::complex<float> *y, std::int64_t incy, std::complex<float> *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    dotc_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done =
        detail::dotc(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    dotc_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline cl::sycl::event dotc(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
    const std::complex<double> *y, std::int64_t incy, std::complex<double> *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    dotc_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done =
        detail::dotc(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    dotc_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline cl::sycl::event dotu(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
    const std::complex<float> *y, std::int64_t incy, std::complex<float> *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    dotu_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done =
        detail::dotu(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    dotu_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline cl::sycl::event dotu(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
    const std::complex<double> *y, std::int64_t incy, std::complex<double> *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    dotu_precondition(queue, n, x, incx, y, incy, result, dependencies);
    auto done =
        detail::dotu(get_device_id(queue), queue, n, x, incx, y, incy, result, dependencies);
    dotu_postcondition(queue, n, x, incx, y, incy, result, dependencies);
    return done;
}

static inline cl::sycl::event gbmv(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
    std::int64_t ku, float alpha, const float *a, std::int64_t lda, const float *x,
    std::int64_t incx, float beta, float *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy, dependencies);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

static inline cl::sycl::event gbmv(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
    std::int64_t ku, double alpha, const double *a, std::int64_t lda, const double *x,
    std::int64_t incx, double beta, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy, dependencies);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

static inline cl::sycl::event gbmv(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
    std::int64_t ku, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy, dependencies);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

static inline cl::sycl::event gbmv(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, std::int64_t kl,
    std::int64_t ku, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gbmv_precondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = detail::gbmv(get_device_id(queue), queue, trans, m, n, kl, ku, alpha, a, lda, x,
                             incx, beta, y, incy, dependencies);
    gbmv_postcondition(queue, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

static inline cl::sycl::event gemm(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
    float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc, dependencies);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

static inline cl::sycl::event gemm(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, double alpha, const double *a, std::int64_t lda, const double *b,
    std::int64_t ldb, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc, dependencies);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

static inline cl::sycl::event gemm(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc, dependencies);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

static inline cl::sycl::event gemm(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = detail::gemm(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                             ldb, beta, c, ldc, dependencies);
    gemm_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

static inline cl::sycl::event gemm_batch(
    cl::sycl::queue &queue, transpose *transa, transpose *transb, std::int64_t *m, std::int64_t *n,
    std::int64_t *k, float *alpha, const float **a, std::int64_t *lda, const float **b,
    std::int64_t *ldb, float *beta, float **c, std::int64_t *ldc, std::int64_t group_count,
    std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size, dependencies);
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

static inline cl::sycl::event gemm_batch(
    cl::sycl::queue &queue, transpose *transa, transpose *transb, std::int64_t *m, std::int64_t *n,
    std::int64_t *k, double *alpha, const double **a, std::int64_t *lda, const double **b,
    std::int64_t *ldb, double *beta, double **c, std::int64_t *ldc, std::int64_t group_count,
    std::int64_t *group_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size, dependencies);
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

static inline cl::sycl::event gemm_batch(
    cl::sycl::queue &queue, transpose *transa, transpose *transb, std::int64_t *m, std::int64_t *n,
    std::int64_t *k, std::complex<float> *alpha, const std::complex<float> **a, std::int64_t *lda,
    const std::complex<float> **b, std::int64_t *ldb, std::complex<float> *beta,
    std::complex<float> **c, std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size, dependencies);
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

static inline cl::sycl::event gemm_batch(
    cl::sycl::queue &queue, transpose *transa, transpose *transb, std::int64_t *m, std::int64_t *n,
    std::int64_t *k, std::complex<double> *alpha, const std::complex<double> **a, std::int64_t *lda,
    const std::complex<double> **b, std::int64_t *ldb, std::complex<double> *beta,
    std::complex<double> **c, std::int64_t *ldc, std::int64_t group_count, std::int64_t *group_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                            group_count, group_size, dependencies);
    auto done =
        detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc, group_count, group_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                             group_count, group_size, dependencies);
    return done;
}

static inline cl::sycl::event gemm_batch(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, float alpha, const float *a, std::int64_t lda, std::int64_t stride_a,
    const float *b, std::int64_t ldb, std::int64_t stride_b, float beta, float *c, std::int64_t ldc,
    std::int64_t stride_c, std::int64_t batch_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

static inline cl::sycl::event gemm_batch(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, double alpha, const double *a, std::int64_t lda, std::int64_t stride_a,
    const double *b, std::int64_t ldb, std::int64_t stride_b, double beta, double *c,
    std::int64_t ldc, std::int64_t stride_c, std::int64_t batch_size,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

static inline cl::sycl::event gemm_batch(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    std::int64_t stride_a, const std::complex<float> *b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<float> beta, std::complex<float> *c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

static inline cl::sycl::event gemm_batch(
    cl::sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    std::int64_t stride_a, const std::complex<double> *b, std::int64_t ldb, std::int64_t stride_b,
    std::complex<double> beta, std::complex<double> *c, std::int64_t ldc, std::int64_t stride_c,
    std::int64_t batch_size, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemm_batch_precondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                            stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    auto done = detail::gemm_batch(get_device_id(queue), queue, transa, transb, m, n, k, alpha, a,
                                   lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                                   batch_size, dependencies);
    gemm_batch_postcondition(queue, transa, transb, m, n, k, alpha, a, lda, stride_a, b, ldb,
                             stride_b, beta, c, ldc, stride_c, batch_size, dependencies);
    return done;
}

static inline cl::sycl::event gemmt(
    cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
    std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *b, std::int64_t ldb,
    float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc, dependencies);
    auto done = detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha,
                              a, lda, b, ldb, beta, c, ldc, dependencies);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, dependencies);
    return done;
}

static inline cl::sycl::event gemmt(
    cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
    std::int64_t k, double alpha, const double *a, std::int64_t lda, const double *b,
    std::int64_t ldb, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc, dependencies);
    auto done = detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha,
                              a, lda, b, ldb, beta, c, ldc, dependencies);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, dependencies);
    return done;
}

static inline cl::sycl::event gemmt(
    cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
    std::int64_t k, std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc, dependencies);
    auto done = detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha,
                              a, lda, b, ldb, beta, c, ldc, dependencies);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, dependencies);
    return done;
}

static inline cl::sycl::event gemmt(
    cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, std::int64_t n,
    std::int64_t k, std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemmt_precondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                       ldc, dependencies);
    auto done = detail::gemmt(get_device_id(queue), queue, upper_lower, transa, transb, n, k, alpha,
                              a, lda, b, ldb, beta, c, ldc, dependencies);
    gemmt_postcondition(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, dependencies);
    return done;
}

static inline cl::sycl::event gemv(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, float alpha,
    const float *a, std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta,
                             y, incy, dependencies);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event gemv(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, double alpha,
    const double *a, std::int64_t lda, const double *x, std::int64_t incx, double beta, double *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta,
                             y, incy, dependencies);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event gemv(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta,
                             y, incy, dependencies);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event gemv(
    cl::sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gemv_precondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = detail::gemv(get_device_id(queue), queue, trans, m, n, alpha, a, lda, x, incx, beta,
                             y, incy, dependencies);
    gemv_postcondition(queue, trans, m, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event ger(
    cl::sycl::queue &queue, std::int64_t m, std::int64_t n, float alpha, const float *x,
    std::int64_t incx, const float *y, std::int64_t incy, float *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = detail::ger(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                            dependencies);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event ger(
    cl::sycl::queue &queue, std::int64_t m, std::int64_t n, double alpha, const double *x,
    std::int64_t incx, const double *y, std::int64_t incy, double *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    ger_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = detail::ger(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                            dependencies);
    ger_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event gerc(
    cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
    std::int64_t incy, std::complex<float> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = detail::gerc(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                             dependencies);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event gerc(
    cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
    std::int64_t incy, std::complex<double> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    gerc_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = detail::gerc(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                             dependencies);
    gerc_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event geru(
    cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
    std::int64_t incy, std::complex<float> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = detail::geru(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                             dependencies);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event geru(
    cl::sycl::queue &queue, std::int64_t m, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
    std::int64_t incy, std::complex<double> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    geru_precondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = detail::geru(get_device_id(queue), queue, m, n, alpha, x, incx, y, incy, a, lda,
                             dependencies);
    geru_postcondition(queue, m, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event hbmv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> beta,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = detail::hbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    hbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

static inline cl::sycl::event hbmv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> beta,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = detail::hbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    hbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

static inline cl::sycl::event hemm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hemm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = detail::hemm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    hemm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

static inline cl::sycl::event hemm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hemm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = detail::hemm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    hemm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

static inline cl::sycl::event hemv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *a, std::int64_t lda, const std::complex<float> *x, std::int64_t incx,
    std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = detail::hemv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event hemv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *a, std::int64_t lda, const std::complex<double> *x,
    std::int64_t incx, std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hemv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = detail::hemv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    hemv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event her(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done = detail::her(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda,
                            dependencies);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event her(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    her_precondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done = detail::her(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda,
                            dependencies);
    her_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event her2(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
    std::int64_t incy, std::complex<float> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = detail::her2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, lda, dependencies);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event her2(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
    std::int64_t incy, std::complex<double> *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    her2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = detail::her2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, lda, dependencies);
    her2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event her2k(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, float beta, std::complex<float> *c,
    std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    her2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = detail::her2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    her2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

static inline cl::sycl::event her2k(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, double beta, std::complex<double> *c,
    std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    her2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = detail::her2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    her2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

static inline cl::sycl::event herk(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    float alpha, const std::complex<float> *a, std::int64_t lda, float beta, std::complex<float> *c,
    std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = detail::herk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

static inline cl::sycl::event herk(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    double alpha, const std::complex<double> *a, std::int64_t lda, double beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    herk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = detail::herk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    herk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

static inline cl::sycl::event hpmv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *a, const std::complex<float> *x, std::int64_t incx,
    std::complex<float> beta, std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    auto done = detail::hpmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta,
                             y, incy, dependencies);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event hpmv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *a, const std::complex<double> *x, std::int64_t incx,
    std::complex<double> beta, std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hpmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    auto done = detail::hpmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta,
                             y, incy, dependencies);
    hpmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event hpr(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha,
    const std::complex<float> *x, std::int64_t incx, std::complex<float> *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    auto done =
        detail::hpr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, dependencies);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

static inline cl::sycl::event hpr(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha,
    const std::complex<double> *x, std::int64_t incx, std::complex<double> *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hpr_precondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    auto done =
        detail::hpr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, dependencies);
    hpr_postcondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

static inline cl::sycl::event hpr2(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<float> alpha,
    const std::complex<float> *x, std::int64_t incx, const std::complex<float> *y,
    std::int64_t incy, std::complex<float> *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    auto done = detail::hpr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, dependencies);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    return done;
}

static inline cl::sycl::event hpr2(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::complex<double> alpha,
    const std::complex<double> *x, std::int64_t incx, const std::complex<double> *y,
    std::int64_t incy, std::complex<double> *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    hpr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    auto done = detail::hpr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, dependencies);
    hpr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    return done;
}

static inline cl::sycl::event iamax(
    cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    iamax_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::iamax(get_device_id(queue), queue, n, x, incx, result, dependencies);
    iamax_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event iamax(
    cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    iamax_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::iamax(get_device_id(queue), queue, n, x, incx, result, dependencies);
    iamax_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event iamax(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    iamax_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::iamax(get_device_id(queue), queue, n, x, incx, result, dependencies);
    iamax_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event iamax(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    iamax_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::iamax(get_device_id(queue), queue, n, x, incx, result, dependencies);
    iamax_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event iamin(
    cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, std::int64_t *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    iamin_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::iamin(get_device_id(queue), queue, n, x, incx, result, dependencies);
    iamin_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event iamin(
    cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    iamin_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::iamin(get_device_id(queue), queue, n, x, incx, result, dependencies);
    iamin_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event iamin(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    iamin_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::iamin(get_device_id(queue), queue, n, x, incx, result, dependencies);
    iamin_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event iamin(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
    std::int64_t *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    iamin_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::iamin(get_device_id(queue), queue, n, x, incx, result, dependencies);
    iamin_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event nrm2(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<float> *x, std::int64_t incx,
    float *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    nrm2_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::nrm2(get_device_id(queue), queue, n, x, incx, result, dependencies);
    nrm2_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event nrm2(
    cl::sycl::queue &queue, std::int64_t n, const std::complex<double> *x, std::int64_t incx,
    double *result, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    nrm2_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::nrm2(get_device_id(queue), queue, n, x, incx, result, dependencies);
    nrm2_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event nrm2(
    cl::sycl::queue &queue, std::int64_t n, const float *x, std::int64_t incx, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    nrm2_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::nrm2(get_device_id(queue), queue, n, x, incx, result, dependencies);
    nrm2_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event nrm2(
    cl::sycl::queue &queue, std::int64_t n, const double *x, std::int64_t incx, double *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    nrm2_precondition(queue, n, x, incx, result, dependencies);
    auto done = detail::nrm2(get_device_id(queue), queue, n, x, incx, result, dependencies);
    nrm2_postcondition(queue, n, x, incx, result, dependencies);
    return done;
}

static inline cl::sycl::event rot(
    cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x, std::int64_t incx,
    std::complex<float> *y, std::int64_t incy, float c, float s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rot_precondition(queue, n, x, incx, y, incy, c, s, dependencies);
    auto done = detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s, dependencies);
    rot_postcondition(queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

static inline cl::sycl::event rot(
    cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x, std::int64_t incx,
    std::complex<double> *y, std::int64_t incy, double c, double s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rot_precondition(queue, n, x, incx, y, incy, c, s, dependencies);
    auto done = detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s, dependencies);
    rot_postcondition(queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

static inline cl::sycl::event rot(
    cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
    std::int64_t incy, float c, float s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rot_precondition(queue, n, x, incx, y, incy, c, s, dependencies);
    auto done = detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s, dependencies);
    rot_postcondition(queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

static inline cl::sycl::event rot(
    cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
    std::int64_t incy, double c, double s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rot_precondition(queue, n, x, incx, y, incy, c, s, dependencies);
    auto done = detail::rot(get_device_id(queue), queue, n, x, incx, y, incy, c, s, dependencies);
    rot_postcondition(queue, n, x, incx, y, incy, c, s, dependencies);
    return done;
}

static inline cl::sycl::event rotg(
    cl::sycl::queue &queue, float *a, float *b, float *c, float *s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rotg_precondition(queue, a, b, c, s, dependencies);
    auto done = detail::rotg(get_device_id(queue), queue, a, b, c, s, dependencies);
    rotg_postcondition(queue, a, b, c, s, dependencies);
    return done;
}

static inline cl::sycl::event rotg(
    cl::sycl::queue &queue, double *a, double *b, double *c, double *s,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rotg_precondition(queue, a, b, c, s, dependencies);
    auto done = detail::rotg(get_device_id(queue), queue, a, b, c, s, dependencies);
    rotg_postcondition(queue, a, b, c, s, dependencies);
    return done;
}

static inline cl::sycl::event rotg(
    cl::sycl::queue &queue, std::complex<float> *a, std::complex<float> *b, float *c,
    std::complex<float> *s, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rotg_precondition(queue, a, b, c, s, dependencies);
    auto done = detail::rotg(get_device_id(queue), queue, a, b, c, s, dependencies);
    rotg_postcondition(queue, a, b, c, s, dependencies);
    return done;
}

static inline cl::sycl::event rotg(
    cl::sycl::queue &queue, std::complex<double> *a, std::complex<double> *b, double *c,
    std::complex<double> *s, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rotg_precondition(queue, a, b, c, s, dependencies);
    auto done = detail::rotg(get_device_id(queue), queue, a, b, c, s, dependencies);
    rotg_postcondition(queue, a, b, c, s, dependencies);
    return done;
}

static inline cl::sycl::event rotm(
    cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
    std::int64_t incy, float *param,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rotm_precondition(queue, n, x, incx, y, incy, param, dependencies);
    auto done = detail::rotm(get_device_id(queue), queue, n, x, incx, y, incy, param, dependencies);
    rotm_postcondition(queue, n, x, incx, y, incy, param, dependencies);
    return done;
}

static inline cl::sycl::event rotm(
    cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
    std::int64_t incy, double *param,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rotm_precondition(queue, n, x, incx, y, incy, param, dependencies);
    auto done = detail::rotm(get_device_id(queue), queue, n, x, incx, y, incy, param, dependencies);
    rotm_postcondition(queue, n, x, incx, y, incy, param, dependencies);
    return done;
}

static inline cl::sycl::event rotmg(
    cl::sycl::queue &queue, float *d1, float *d2, float *x1, float y1, float *param,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rotmg_precondition(queue, d1, d2, x1, y1, param, dependencies);
    auto done = detail::rotmg(get_device_id(queue), queue, d1, d2, x1, y1, param, dependencies);
    rotmg_postcondition(queue, d1, d2, x1, y1, param, dependencies);
    return done;
}

static inline cl::sycl::event rotmg(
    cl::sycl::queue &queue, double *d1, double *d2, double *x1, double y1, double *param,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    rotmg_precondition(queue, d1, d2, x1, y1, param, dependencies);
    auto done = detail::rotmg(get_device_id(queue), queue, d1, d2, x1, y1, param, dependencies);
    rotmg_postcondition(queue, d1, d2, x1, y1, param, dependencies);
    return done;
}

static inline cl::sycl::event sbmv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, float alpha,
    const float *a, std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = detail::sbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

static inline cl::sycl::event sbmv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, std::int64_t k, double alpha,
    const double *a, std::int64_t lda, const double *x, std::int64_t incx, double beta, double *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    sbmv_precondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                      dependencies);
    auto done = detail::sbmv(get_device_id(queue), queue, upper_lower, n, k, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    sbmv_postcondition(queue, upper_lower, n, k, alpha, a, lda, x, incx, beta, y, incy,
                       dependencies);
    return done;
}

static inline cl::sycl::event scal(
    cl::sycl::queue &queue, std::int64_t n, float alpha, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event scal(
    cl::sycl::queue &queue, std::int64_t n, double alpha, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event scal(
    cl::sycl::queue &queue, std::int64_t n, std::complex<float> alpha, std::complex<float> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event scal(
    cl::sycl::queue &queue, std::int64_t n, std::complex<double> alpha, std::complex<double> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event scal(
    cl::sycl::queue &queue, std::int64_t n, float alpha, std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event scal(
    cl::sycl::queue &queue, std::int64_t n, double alpha, std::complex<double> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    scal_precondition(queue, n, alpha, x, incx, dependencies);
    auto done = detail::scal(get_device_id(queue), queue, n, alpha, x, incx, dependencies);
    scal_postcondition(queue, n, alpha, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event sdsdot(
    cl::sycl::queue &queue, std::int64_t n, float sb, const float *x, std::int64_t incx,
    const float *y, std::int64_t incy, float *result,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    sdsdot_precondition(queue, n, sb, x, incx, y, incy, result, dependencies);
    auto done =
        detail::sdsdot(get_device_id(queue), queue, n, sb, x, incx, y, incy, result, dependencies);
    sdsdot_postcondition(queue, n, sb, x, incx, y, incy, result, dependencies);
    return done;
}

static inline cl::sycl::event spmv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha, const float *a,
    const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    auto done = detail::spmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta,
                             y, incy, dependencies);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event spmv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha, const double *a,
    const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    spmv_precondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    auto done = detail::spmv(get_device_id(queue), queue, upper_lower, n, alpha, a, x, incx, beta,
                             y, incy, dependencies);
    spmv_postcondition(queue, upper_lower, n, alpha, a, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event spr(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha, const float *x,
    std::int64_t incx, float *a, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    auto done =
        detail::spr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, dependencies);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

static inline cl::sycl::event spr(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha, const double *x,
    std::int64_t incx, double *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    spr_precondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    auto done =
        detail::spr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, dependencies);
    spr_postcondition(queue, upper_lower, n, alpha, x, incx, a, dependencies);
    return done;
}

static inline cl::sycl::event spr2(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha, const float *x,
    std::int64_t incx, const float *y, std::int64_t incy, float *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    auto done = detail::spr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, dependencies);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    return done;
}

static inline cl::sycl::event spr2(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha, const double *x,
    std::int64_t incx, const double *y, std::int64_t incy, double *a,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    spr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    auto done = detail::spr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, dependencies);
    spr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, dependencies);
    return done;
}

static inline cl::sycl::event swap(
    cl::sycl::queue &queue, std::int64_t n, float *x, std::int64_t incx, float *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    swap_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = detail::swap(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    swap_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event swap(
    cl::sycl::queue &queue, std::int64_t n, double *x, std::int64_t incx, double *y,
    std::int64_t incy, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    swap_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = detail::swap(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    swap_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event swap(
    cl::sycl::queue &queue, std::int64_t n, std::complex<float> *x, std::int64_t incx,
    std::complex<float> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    swap_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = detail::swap(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    swap_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event swap(
    cl::sycl::queue &queue, std::int64_t n, std::complex<double> *x, std::int64_t incx,
    std::complex<double> *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    swap_precondition(queue, n, x, incx, y, incy, dependencies);
    auto done = detail::swap(get_device_id(queue), queue, n, x, incx, y, incy, dependencies);
    swap_postcondition(queue, n, x, incx, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event symm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
    float alpha, const float *a, std::int64_t lda, const float *b, std::int64_t ldb, float beta,
    float *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

static inline cl::sycl::event symm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
    double alpha, const double *a, std::int64_t lda, const double *b, std::int64_t ldb, double beta,
    double *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

static inline cl::sycl::event symm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

static inline cl::sycl::event symm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    symm_precondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                      dependencies);
    auto done = detail::symm(get_device_id(queue), queue, left_right, upper_lower, m, n, alpha, a,
                             lda, b, ldb, beta, c, ldc, dependencies);
    symm_postcondition(queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    return done;
}

static inline cl::sycl::event symv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha, const float *a,
    std::int64_t lda, const float *x, std::int64_t incx, float beta, float *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = detail::symv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event symv(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha, const double *a,
    std::int64_t lda, const double *x, std::int64_t incx, double beta, double *y, std::int64_t incy,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    symv_precondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    auto done = detail::symv(get_device_id(queue), queue, upper_lower, n, alpha, a, lda, x, incx,
                             beta, y, incy, dependencies);
    symv_postcondition(queue, upper_lower, n, alpha, a, lda, x, incx, beta, y, incy, dependencies);
    return done;
}

static inline cl::sycl::event syr(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha, const float *x,
    std::int64_t incx, float *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done = detail::syr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda,
                            dependencies);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event syr(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha, const double *x,
    std::int64_t incx, double *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syr_precondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    auto done = detail::syr(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, a, lda,
                            dependencies);
    syr_postcondition(queue, upper_lower, n, alpha, x, incx, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event syr2(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, float alpha, const float *x,
    std::int64_t incx, const float *y, std::int64_t incy, float *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = detail::syr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, lda, dependencies);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event syr2(
    cl::sycl::queue &queue, uplo upper_lower, std::int64_t n, double alpha, const double *x,
    std::int64_t incx, const double *y, std::int64_t incy, double *a, std::int64_t lda,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syr2_precondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    auto done = detail::syr2(get_device_id(queue), queue, upper_lower, n, alpha, x, incx, y, incy,
                             a, lda, dependencies);
    syr2_postcondition(queue, upper_lower, n, alpha, x, incx, y, incy, a, lda, dependencies);
    return done;
}

static inline cl::sycl::event syr2k(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    float alpha, const float *a, std::int64_t lda, const float *b, std::int64_t ldb, float beta,
    float *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

static inline cl::sycl::event syr2k(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    double alpha, const double *a, std::int64_t lda, const double *b, std::int64_t ldb, double beta,
    double *c, std::int64_t ldc, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

static inline cl::sycl::event syr2k(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    const std::complex<float> *b, std::int64_t ldb, std::complex<float> beta,
    std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

static inline cl::sycl::event syr2k(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    const std::complex<double> *b, std::int64_t ldb, std::complex<double> beta,
    std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syr2k_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                       dependencies);
    auto done = detail::syr2k(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                              b, ldb, beta, c, ldc, dependencies);
    syr2k_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                        dependencies);
    return done;
}

static inline cl::sycl::event syrk(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    float alpha, const float *a, std::int64_t lda, float beta, float *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

static inline cl::sycl::event syrk(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    double alpha, const double *a, std::int64_t lda, double beta, double *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

static inline cl::sycl::event syrk(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<float> alpha, const std::complex<float> *a, std::int64_t lda,
    std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

static inline cl::sycl::event syrk(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
    std::complex<double> alpha, const std::complex<double> *a, std::int64_t lda,
    std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    syrk_precondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    auto done = detail::syrk(get_device_id(queue), queue, upper_lower, trans, n, k, alpha, a, lda,
                             beta, c, ldc, dependencies);
    syrk_postcondition(queue, upper_lower, trans, n, k, alpha, a, lda, beta, c, ldc, dependencies);
    return done;
}

static inline cl::sycl::event tbmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, const float *a, std::int64_t lda, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tbmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, const double *a, std::int64_t lda, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tbmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tbmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tbmv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = detail::tbmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    tbmv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tbsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, const float *a, std::int64_t lda, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tbsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, const double *a, std::int64_t lda, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tbsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, const std::complex<float> *a, std::int64_t lda, std::complex<float> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tbsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    std::int64_t k, const std::complex<double> *a, std::int64_t lda, std::complex<double> *x,
    std::int64_t incx, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tbsv_precondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    auto done = detail::tbsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, k, a,
                             lda, x, incx, dependencies);
    tbsv_postcondition(queue, upper_lower, trans, unit_diag, n, k, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tpmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const float *a, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tpmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const double *a, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tpmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const std::complex<float> *a, std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tpmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const std::complex<double> *a, std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tpmv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = detail::tpmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    tpmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tpsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const float *a, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tpsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const double *a, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tpsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const std::complex<float> *a, std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event tpsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const std::complex<double> *a, std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    tpsv_precondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    auto done = detail::tpsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, x,
                             incx, dependencies);
    tpsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event trmm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, float alpha, const float *a, std::int64_t lda, float *b,
    std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

static inline cl::sycl::event trmm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, double alpha, const double *a, std::int64_t lda, double *b,
    std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

static inline cl::sycl::event trmm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
    std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

static inline cl::sycl::event trmm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
    std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trmm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = detail::trmm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    trmm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

static inline cl::sycl::event trmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const float *a, std::int64_t lda, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event trmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const double *a, std::int64_t lda, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event trmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const std::complex<float> *a, std::int64_t lda, std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event trmv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const std::complex<double> *a, std::int64_t lda, std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trmv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = detail::trmv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    trmv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event trsm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, float alpha, const float *a, std::int64_t lda, float *b,
    std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

static inline cl::sycl::event trsm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, double alpha, const double *a, std::int64_t lda, double *b,
    std::int64_t ldb, const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

static inline cl::sycl::event trsm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
    std::int64_t lda, std::complex<float> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

static inline cl::sycl::event trsm(
    cl::sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
    std::int64_t m, std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
    std::int64_t lda, std::complex<double> *b, std::int64_t ldb,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trsm_precondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b, ldb,
                      dependencies);
    auto done = detail::trsm(get_device_id(queue), queue, left_right, upper_lower, trans, unit_diag,
                             m, n, alpha, a, lda, b, ldb, dependencies);
    trsm_postcondition(queue, left_right, upper_lower, trans, unit_diag, m, n, alpha, a, lda, b,
                       ldb, dependencies);
    return done;
}

static inline cl::sycl::event trsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const float *a, std::int64_t lda, float *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event trsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const double *a, std::int64_t lda, double *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event trsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const std::complex<float> *a, std::int64_t lda, std::complex<float> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}

static inline cl::sycl::event trsv(
    cl::sycl::queue &queue, uplo upper_lower, transpose trans, diag unit_diag, std::int64_t n,
    const std::complex<double> *a, std::int64_t lda, std::complex<double> *x, std::int64_t incx,
    const cl::sycl::vector_class<cl::sycl::event> &dependencies = {}) {
    trsv_precondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    auto done = detail::trsv(get_device_id(queue), queue, upper_lower, trans, unit_diag, n, a, lda,
                             x, incx, dependencies);
    trsv_postcondition(queue, upper_lower, trans, unit_diag, n, a, lda, x, incx, dependencies);
    return done;
}
