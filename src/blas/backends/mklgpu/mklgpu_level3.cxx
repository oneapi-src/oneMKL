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

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, float alpha, sycl::buffer<float, 1> &a,
          std::int64_t lda, sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          sycl::buffer<float, 1> &c, std::int64_t ldc) {
    blas_major::gemm(queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, double alpha, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::dgemm(queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k,
                              alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::cgemm(queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k,
                              alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::zgemm(queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k,
                              alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1> &a,
          std::int64_t lda, sycl::buffer<sycl::half, 1> &b, std::int64_t ldb, sycl::half beta,
          sycl::buffer<sycl::half, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::hgemm(queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n, k,
                              alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, float alpha, sycl::buffer<sycl::half, 1> &a,
          std::int64_t lda, sycl::buffer<sycl::half, 1> &b, std::int64_t ldb, float beta,
          sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::gemm_f16f16f32(queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m, n,
                                       k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
          std::int64_t n, std::int64_t k, float alpha, sycl::buffer<bfloat16, 1> &a,
          std::int64_t lda, sycl::buffer<bfloat16, 1> &b, std::int64_t ldb, float beta,
          sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::gemm_bf16bf16f32(queue, MAJOR, mkl_convert(transa), mkl_convert(transb), m,
                                         n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
          sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::ssymm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), m, n,
                              alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
          sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
          sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::dsymm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), m, n,
                              alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::csymm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), m, n,
                              alpha, a, lda, b, ldb, beta, c, ldc);
}

void symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    ::oneapi::mkl::gpu::zsymm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), m, n,
                              alpha, a, lda, b, ldb, beta, c, ldc);
}

void hemm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::chemm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), m, n,
                              alpha, a, lda, b, ldb, beta, c, ldc);
}

void hemm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    ::oneapi::mkl::gpu::zhemm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), m, n,
                              alpha, a, lda, b, ldb, beta, c, ldc);
}

void syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          float alpha, sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
          sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::ssyrk(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                              alpha, a, lda, beta, c, ldc);
}

void syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          double alpha, sycl::buffer<double, 1> &a, std::int64_t lda, double beta,
          sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::dsyrk(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                              alpha, a, lda, beta, c, ldc);
}

void syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::csyrk(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                              alpha, a, lda, beta, c, ldc);
}

void syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c,
          std::int64_t ldc) {
    ::oneapi::mkl::gpu::zsyrk(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                              alpha, a, lda, beta, c, ldc);
}

void herk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          float alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, float beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::cherk(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                              alpha, a, lda, beta, c, ldc);
}

void herk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          double alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::zherk(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                              alpha, a, lda, beta, c, ldc);
}

void syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
           sycl::buffer<float, 1> &b, std::int64_t ldb, float beta,
           sycl::buffer<float, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::ssyr2k(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
}

void syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
           sycl::buffer<double, 1> &b, std::int64_t ldb, double beta,
           sycl::buffer<double, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::dsyr2k(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
}

void syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
           std::int64_t lda, sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
           std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c,
           std::int64_t ldc) {
    ::oneapi::mkl::gpu::csyr2k(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
}

void syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
           std::int64_t lda, sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c,
           std::int64_t ldc) {
    ::oneapi::mkl::gpu::zsyr2k(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
}

void her2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
           std::int64_t lda, sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
           float beta, sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::cher2k(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
}

void her2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
           std::int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
           std::int64_t lda, sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
           double beta, sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc) {
    ::oneapi::mkl::gpu::zher2k(queue, MAJOR, mkl_convert(upper_lower), mkl_convert(trans), n, k,
                               alpha, a, lda, b, ldb, beta, c, ldc);
}

void trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &b,
          std::int64_t ldb) {
    ::oneapi::mkl::gpu::strmm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                              mkl_convert(transa), mkl_convert(unit_diag), m, n, alpha, a, lda, b,
                              ldb);
}

void trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &b,
          std::int64_t ldb) {
    ::oneapi::mkl::gpu::dtrmm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                              mkl_convert(transa), mkl_convert(unit_diag), m, n, alpha, a, lda, b,
                              ldb);
}

void trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    ::oneapi::mkl::gpu::ctrmm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                              mkl_convert(transa), mkl_convert(unit_diag), m, n, alpha, a, lda, b,
                              ldb);
}

void trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    ::oneapi::mkl::gpu::ztrmm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                              mkl_convert(transa), mkl_convert(unit_diag), m, n, alpha, a, lda, b,
                              ldb);
}

void trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, std::int64_t m, std::int64_t n, float alpha,
          sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &b,
          std::int64_t ldb) {
    ::oneapi::mkl::gpu::strsm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                              mkl_convert(transa), mkl_convert(unit_diag), m, n, alpha, a, lda, b,
                              ldb);
}

void trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, std::int64_t m, std::int64_t n, double alpha,
          sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &b,
          std::int64_t ldb) {
    ::oneapi::mkl::gpu::dtrsm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                              mkl_convert(transa), mkl_convert(unit_diag), m, n, alpha, a, lda, b,
                              ldb);
}

void trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb) {
    ::oneapi::mkl::gpu::ctrsm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                              mkl_convert(transa), mkl_convert(unit_diag), m, n, alpha, a, lda, b,
                              ldb);
}

void trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
          diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb) {
    ::oneapi::mkl::gpu::ztrsm(queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower),
                              mkl_convert(transa), mkl_convert(unit_diag), m, n, alpha, a, lda, b,
                              ldb);
}

// USM APIs

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                     const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::sgemm_sycl(&queue, MAJOR, mkl_convert(transa), mkl_convert(transb),
                                          m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                                          dependencies);
}

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, double alpha, const double *a,
                     std::int64_t lda, const double *b, std::int64_t ldb, double beta, double *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dgemm_sycl(&queue, MAJOR, mkl_convert(transa), mkl_convert(transb),
                                          m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                                          dependencies);
}

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                     std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cgemm_sycl(&queue, MAJOR, mkl_convert(transa), mkl_convert(transb),
                                          m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                                          dependencies);
}

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                     std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zgemm_sycl(&queue, MAJOR, mkl_convert(transa), mkl_convert(transb),
                                          m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                                          dependencies);
}

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, sycl::half alpha, const sycl::half *a,
                     std::int64_t lda, const sycl::half *b, std::int64_t ldb, sycl::half beta,
                     sycl::half *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::hgemm_sycl(&queue, MAJOR, mkl_convert(transa), mkl_convert(transb),
                                          m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                                          dependencies);
}

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, float alpha, const sycl::half *a,
                     std::int64_t lda, const sycl::half *b, std::int64_t ldb, float beta, float *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::gemm_f16f16f32_sycl(&queue, MAJOR, mkl_convert(transa),
                                                   mkl_convert(transb), m, n, k, alpha, a, lda, b,
                                                   ldb, beta, c, ldc, dependencies);
}

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                     std::int64_t n, std::int64_t k, float alpha, const bfloat16 *a,
                     std::int64_t lda, const bfloat16 *b, std::int64_t ldb, float beta, float *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::gemm_bf16bf16f32_sycl(&queue, MAJOR, mkl_convert(transa),
                                                     mkl_convert(transb), m, n, k, alpha, a, lda, b,
                                                     ldb, beta, c, ldc, dependencies);
}

sycl::event symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *b,
                     std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssymm_sycl(&queue, MAJOR, mkl_convert(left_right),
                                          mkl_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                          beta, c, ldc, dependencies);
}

sycl::event symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, double alpha, const double *a, std::int64_t lda,
                     const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsymm_sycl(&queue, MAJOR, mkl_convert(left_right),
                                          mkl_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                          beta, c, ldc, dependencies);
}

sycl::event symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                     std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csymm_sycl(&queue, MAJOR, mkl_convert(left_right),
                                          mkl_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                          beta, c, ldc, dependencies);
}

sycl::event symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                     std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zsymm_sycl(&queue, MAJOR, mkl_convert(left_right),
                                          mkl_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                          beta, c, ldc, dependencies);
}

sycl::event hemm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                     std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::chemm_sycl(&queue, MAJOR, mkl_convert(left_right),
                                          mkl_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                          beta, c, ldc, dependencies);
}

sycl::event hemm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                     std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                     std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zhemm_sycl(&queue, MAJOR, mkl_convert(left_right),
                                          mkl_convert(upper_lower), m, n, alpha, a, lda, b, ldb,
                                          beta, c, ldc, dependencies);
}

sycl::event syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, float alpha, const float *a, std::int64_t lda, float beta,
                     float *c, std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyrk_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                          mkl_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                          dependencies);
}

sycl::event syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, double alpha, const double *a, std::int64_t lda, double beta,
                     double *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyrk_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                          mkl_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                          dependencies);
}

sycl::event syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                     std::int64_t lda, std::complex<float> beta, std::complex<float> *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csyrk_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                          mkl_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                          dependencies);
}

sycl::event syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                     std::int64_t lda, std::complex<double> beta, std::complex<double> *c,
                     std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zsyrk_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                          mkl_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                          dependencies);
}

sycl::event herk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, float alpha, const std::complex<float> *a, std::int64_t lda,
                     float beta, std::complex<float> *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cherk_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                          mkl_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                          dependencies);
}

sycl::event herk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                     std::int64_t k, double alpha, const std::complex<double> *a, std::int64_t lda,
                     double beta, std::complex<double> *c, std::int64_t ldc,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zherk_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                          mkl_convert(trans), n, k, alpha, a, lda, beta, c, ldc,
                                          dependencies);
}

sycl::event syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *b,
                      std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ssyr2k_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                           mkl_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c,
                                           ldc, dependencies);
}

sycl::event syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, double alpha, const double *a, std::int64_t lda,
                      const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dsyr2k_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                           mkl_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c,
                                           ldc, dependencies);
}

sycl::event syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                      std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::csyr2k_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                           mkl_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c,
                                           ldc, dependencies);
}

sycl::event syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                      std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                      std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zsyr2k_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                           mkl_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c,
                                           ldc, dependencies);
}

sycl::event her2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                      std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, float beta,
                      std::complex<float> *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::cher2k_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                           mkl_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c,
                                           ldc, dependencies);
}

sycl::event her2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                      std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                      std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                      double beta, std::complex<double> *c, std::int64_t ldc,
                      const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::zher2k_sycl(&queue, MAJOR, mkl_convert(upper_lower),
                                           mkl_convert(trans), n, k, alpha, a, lda, b, ldb, beta, c,
                                           ldc, dependencies);
}

sycl::event trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, std::int64_t m, std::int64_t n, float alpha, const float *a,
                     std::int64_t lda, float *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strmm_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(transa),
        mkl_convert(unit_diag), m, n, alpha, a, lda, b, ldb, dependencies);
}

sycl::event trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, std::int64_t m, std::int64_t n, double alpha, const double *a,
                     std::int64_t lda, double *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrmm_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(transa),
        mkl_convert(unit_diag), m, n, alpha, a, lda, b, ldb, dependencies);
}

sycl::event trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *b,
                     std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrmm_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(transa),
        mkl_convert(unit_diag), m, n, alpha, a, lda, b, ldb, dependencies);
}

sycl::event trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *b,
                     std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrmm_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(transa),
        mkl_convert(unit_diag), m, n, alpha, a, lda, b, ldb, dependencies);
}

sycl::event trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, std::int64_t m, std::int64_t n, float alpha, const float *a,
                     std::int64_t lda, float *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::strsm_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(transa),
        mkl_convert(unit_diag), m, n, alpha, a, lda, b, ldb, dependencies);
}

sycl::event trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, std::int64_t m, std::int64_t n, double alpha, const double *a,
                     std::int64_t lda, double *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::dtrsm_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(transa),
        mkl_convert(unit_diag), m, n, alpha, a, lda, b, ldb, dependencies);
}

sycl::event trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                     const std::complex<float> *a, std::int64_t lda, std::complex<float> *b,
                     std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ctrsm_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(transa),
        mkl_convert(unit_diag), m, n, alpha, a, lda, b, ldb, dependencies);
}

sycl::event trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose transa,
                     diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                     const std::complex<double> *a, std::int64_t lda, std::complex<double> *b,
                     std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
    return ::oneapi::mkl::gpu::ztrsm_sycl(
        &queue, MAJOR, mkl_convert(left_right), mkl_convert(upper_lower), mkl_convert(transa),
        mkl_convert(unit_diag), m, n, alpha, a, lda, b, ldb, dependencies);
}
