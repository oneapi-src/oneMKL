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

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               int64_t m, int64_t n, int64_t k, float alpha, cl::sycl::buffer<int8_t, 1> &a,
               int64_t lda, int8_t ao, cl::sycl::buffer<int8_t, 1> &b, int64_t ldb, int8_t bo,
               float beta, cl::sycl::buffer<int32_t, 1> &c, int64_t ldc,
               cl::sycl::buffer<int32_t, 1> &co) {
    ::oneapi::mkl::gpu::gemm_s8s8s32_sycl(&queue, MAJOR, mkl_convert(transa), mkl_convert(transb),
                                          mkl_convert(offsetc), m, n, k, alpha, &a, lda, ao, &b,
                                          ldb, bo, beta, &c, ldc, &co);
}

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               int64_t m, int64_t n, int64_t k, float alpha, cl::sycl::buffer<int8_t, 1> &a,
               int64_t lda, int8_t ao, cl::sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo,
               float beta, cl::sycl::buffer<int32_t, 1> &c, int64_t ldc,
               cl::sycl::buffer<int32_t, 1> &co) {
    ::oneapi::mkl::gpu::gemm_s8u8s32_sycl(&queue, MAJOR, mkl_convert(transa), mkl_convert(transb),
                                          mkl_convert(offsetc), m, n, k, alpha, &a, lda, ao, &b,
                                          ldb, bo, beta, &c, ldc, &co);
}

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               int64_t m, int64_t n, int64_t k, float alpha, cl::sycl::buffer<uint8_t, 1> &a,
               int64_t lda, uint8_t ao, cl::sycl::buffer<int8_t, 1> &b, int64_t ldb, int8_t bo,
               float beta, cl::sycl::buffer<int32_t, 1> &c, int64_t ldc,
               cl::sycl::buffer<int32_t, 1> &co) {
    ::oneapi::mkl::gpu::gemm_u8s8s32_sycl(&queue, MAJOR, mkl_convert(transa), mkl_convert(transb),
                                          mkl_convert(offsetc), m, n, k, alpha, &a, lda, ao, &b,
                                          ldb, bo, beta, &c, ldc, &co);
}

void gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
               int64_t m, int64_t n, int64_t k, float alpha, cl::sycl::buffer<uint8_t, 1> &a,
               int64_t lda, uint8_t ao, cl::sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo,
               float beta, cl::sycl::buffer<int32_t, 1> &c, int64_t ldc,
               cl::sycl::buffer<int32_t, 1> &co) {
    ::oneapi::mkl::gpu::gemm_u8u8s32_sycl(&queue, MAJOR, mkl_convert(transa), mkl_convert(transb),
                                          mkl_convert(offsetc), m, n, k, alpha, &a, lda, ao, &b,
                                          ldb, bo, beta, &c, ldc, &co);
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, float alpha, cl::sycl::buffer<float, 1> &a, int64_t lda,
           cl::sycl::buffer<float, 1> &b, int64_t ldb, float beta, cl::sycl::buffer<float, 1> &c,
           int64_t ldc) {
    ::oneapi::mkl::gpu::sgemmt(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(transa),
                               mkl_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, double alpha, cl::sycl::buffer<double, 1> &a, int64_t lda,
           cl::sycl::buffer<double, 1> &b, int64_t ldb, double beta, cl::sycl::buffer<double, 1> &c,
           int64_t ldc) {
    ::oneapi::mkl::gpu::dgemmt(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(transa),
                               mkl_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<double> alpha, cl::sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
           std::complex<double> beta, cl::sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    ::oneapi::mkl::gpu::zgemmt(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(transa),
                               mkl_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<float> alpha, cl::sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, cl::sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
           std::complex<float> beta, cl::sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    ::oneapi::mkl::gpu::cgemmt(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(transa),
                               mkl_convert(transb), n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// USM APIs

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                          const int8_t *a, int64_t lda, int8_t ao, const int8_t *b, int64_t ldb,
                          int8_t bo, float beta, int32_t *c, int64_t ldc, const int32_t *co,
                          const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::gemm_s8s8s32_sycl(
        &queue, MAJOR, mkl_convert(transa), mkl_convert(transb), mkl_convert(offsetc), m, n, k,
        alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co, dependencies);
}

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                          const int8_t *a, int64_t lda, int8_t ao, const uint8_t *b, int64_t ldb,
                          uint8_t bo, float beta, int32_t *c, int64_t ldc, const int32_t *co,
                          const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::gemm_s8u8s32_sycl(
        &queue, MAJOR, mkl_convert(transa), mkl_convert(transb), mkl_convert(offsetc), m, n, k,
        alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co, dependencies);
}

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                          const uint8_t *a, int64_t lda, uint8_t ao, const int8_t *b, int64_t ldb,
                          int8_t bo, float beta, int32_t *c, int64_t ldc, const int32_t *co,
                          const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::gemm_u8s8s32_sycl(
        &queue, MAJOR, mkl_convert(transa), mkl_convert(transb), mkl_convert(offsetc), m, n, k,
        alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co, dependencies);
}

cl::sycl::event gemm_bias(cl::sycl::queue &queue, transpose transa, transpose transb,
                          offset offsetc, int64_t m, int64_t n, int64_t k, float alpha,
                          const uint8_t *a, int64_t lda, uint8_t ao, const uint8_t *b, int64_t ldb,
                          uint8_t bo, float beta, int32_t *c, int64_t ldc, const int32_t *co,
                          const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::gemm_u8u8s32_sycl(
        &queue, MAJOR, mkl_convert(transa), mkl_convert(transb), mkl_convert(offsetc), m, n, k,
        alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, float alpha, const float *a, int64_t lda,
                      const float *b, int64_t ldb, float beta, float *c, int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemmt_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                           mkl_convert(transa), mkl_convert(transb), n, k, alpha, a,
                                           lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, double alpha, const double *a, int64_t lda,
                      const double *b, int64_t ldb, double beta, double *c, int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemmt_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                           mkl_convert(transa), mkl_convert(transb), n, k, alpha, a,
                                           lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      int64_t lda, const std::complex<float> *b, int64_t ldb,
                      std::complex<float> beta, std::complex<float> *c, int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemmt_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                           mkl_convert(transa), mkl_convert(transb), n, k, alpha, a,
                                           lda, b, ldb, beta, c, ldc, dependencies);
}

cl::sycl::event gemmt(cl::sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                      int64_t n, int64_t k, std::complex<double> alpha,
                      const std::complex<double> *a, int64_t lda, const std::complex<double> *b,
                      int64_t ldb, std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                      const std::vector<cl::sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemmt_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                           mkl_convert(transa), mkl_convert(transb), n, k, alpha, a,
                                           lda, b, ldb, beta, c, ldc, dependencies);
}
