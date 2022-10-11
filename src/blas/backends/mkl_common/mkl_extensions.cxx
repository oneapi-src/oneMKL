/*******************************************************************************
* Copyright 2022 Intel Corporation
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

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1> &a, int64_t lda,
               int8_t ao, sycl::buffer<int8_t, 1> &b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    blas_major::gemm_bias(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo,
                          beta, c, ldc, co);
}

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<int8_t, 1> &a, int64_t lda,
               int8_t ao, sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    blas_major::gemm_bias(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo,
                          beta, c, ldc, co);
}

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1> &a, int64_t lda,
               uint8_t ao, sycl::buffer<int8_t, 1> &b, int64_t ldb, int8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    blas_major::gemm_bias(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo,
                          beta, c, ldc, co);
}

void gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc, int64_t m,
               int64_t n, int64_t k, float alpha, sycl::buffer<uint8_t, 1> &a, int64_t lda,
               uint8_t ao, sycl::buffer<uint8_t, 1> &b, int64_t ldb, uint8_t bo, float beta,
               sycl::buffer<int32_t, 1> &c, int64_t ldc, sycl::buffer<int32_t, 1> &co) {
    blas_major::gemm_bias(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb, bo,
                          beta, c, ldc, co);
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, float alpha, sycl::buffer<float, 1> &a, int64_t lda,
           sycl::buffer<float, 1> &b, int64_t ldb, float beta, sycl::buffer<float, 1> &c,
           int64_t ldc) {
    blas_major::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc);
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, double alpha, sycl::buffer<double, 1> &a, int64_t lda,
           sycl::buffer<double, 1> &b, int64_t ldb, double beta, sycl::buffer<double, 1> &c,
           int64_t ldc) {
    blas_major::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc);
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<double>, 1> &b, int64_t ldb,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c, int64_t ldc) {
    blas_major::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc);
}

void gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb, int64_t n,
           int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
           int64_t lda, sycl::buffer<std::complex<float>, 1> &b, int64_t ldb,
           std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c, int64_t ldc) {
    blas_major::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c,
                      ldc);
}

// USM APIs

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const int8_t *a, int64_t lda,
                      int8_t ao, const int8_t *b, int64_t ldb, int8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    return blas_major::gemm_bias(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb,
                                 bo, beta, c, ldc, co, dependencies);
}

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const int8_t *a, int64_t lda,
                      int8_t ao, const uint8_t *b, int64_t ldb, uint8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    return blas_major::gemm_bias(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb,
                                 bo, beta, c, ldc, co, dependencies);
}

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const uint8_t *a, int64_t lda,
                      uint8_t ao, const int8_t *b, int64_t ldb, int8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    return blas_major::gemm_bias(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb,
                                 bo, beta, c, ldc, co, dependencies);
}

sycl::event gemm_bias(sycl::queue &queue, transpose transa, transpose transb, offset offsetc,
                      int64_t m, int64_t n, int64_t k, float alpha, const uint8_t *a, int64_t lda,
                      uint8_t ao, const uint8_t *b, int64_t ldb, uint8_t bo, float beta, int32_t *c,
                      int64_t ldc, const int32_t *co,
                      const std::vector<sycl::event> &dependencies) {
    return blas_major::gemm_bias(queue, transa, transb, offsetc, m, n, k, alpha, a, lda, ao, b, ldb,
                                 bo, beta, c, ldc, co, dependencies);
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, float alpha, const float *a, int64_t lda, const float *b,
                  int64_t ldb, float beta, float *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    return blas_major::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta,
                             c, ldc, dependencies);
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, double alpha, const double *a, int64_t lda, const double *b,
                  int64_t ldb, double beta, double *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    return blas_major::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta,
                             c, ldc, dependencies);
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                  int64_t lda, const std::complex<float> *b, int64_t ldb, std::complex<float> beta,
                  std::complex<float> *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    return blas_major::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta,
                             c, ldc, dependencies);
}

sycl::event gemmt(sycl::queue &queue, uplo upper_lower, transpose transa, transpose transb,
                  int64_t n, int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                  int64_t lda, const std::complex<double> *b, int64_t ldb,
                  std::complex<double> beta, std::complex<double> *c, int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    return blas_major::gemmt(queue, upper_lower, transa, transb, n, k, alpha, a, lda, b, ldb, beta,
                             c, ldc, dependencies);
}
