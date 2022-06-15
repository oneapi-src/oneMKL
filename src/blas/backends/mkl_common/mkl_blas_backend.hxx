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

/// level3,  buffer

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &b, std::int64_t ldb, float beta, sycl::buffer<float, 1> &c,
          std::int64_t ldc);

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, double alpha, sycl::buffer<double, 1> &a, std::int64_t lda,
          sycl::buffer<double, 1> &b, std::int64_t ldb, double beta, sycl::buffer<double, 1> &c,
          std::int64_t ldc);

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb,
          std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a,
          std::int64_t lda, sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, sycl::half alpha, sycl::buffer<sycl::half, 1> &a, std::int64_t lda,
          sycl::buffer<sycl::half, 1> &b, std::int64_t ldb, sycl::half beta,
          sycl::buffer<sycl::half, 1> &c, std::int64_t ldc);

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<sycl::half, 1> &a, std::int64_t lda,
          sycl::buffer<sycl::half, 1> &b, std::int64_t ldb, float beta, sycl::buffer<float, 1> &c,
          std::int64_t ldc);

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<bfloat16, 1> &a, std::int64_t lda,
          sycl::buffer<bfloat16, 1> &b, std::int64_t ldb, float beta, sycl::buffer<float, 1> &c,
          std::int64_t ldc);

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<bfloat16, 1> &a, std::int64_t lda,
          sycl::buffer<bfloat16, 1> &b, std::int64_t ldb, float beta, sycl::buffer<bfloat16, 1> &c,
          std::int64_t ldc);

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<std::int8_t, 1> &a, std::int64_t lda,
          sycl::buffer<std::int8_t, 1> &b, std::int64_t ldb, float beta,
          sycl::buffer<std::int32_t, 1> &c, std::int64_t ldc);

void gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
          std::int64_t k, float alpha, sycl::buffer<std::int8_t, 1> &a, std::int64_t lda,
          sycl::buffer<std::int8_t, 1> &b, std::int64_t ldb, float beta, sycl::buffer<float, 1> &c,
          std::int64_t ldc);

void symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          float alpha, sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &b,
          std::int64_t ldb, float beta, sycl::buffer<float, 1> &c, std::int64_t ldc);

void symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          double alpha, sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &b,
          std::int64_t ldb, double beta, sycl::buffer<double, 1> &c, std::int64_t ldc);

void symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

void hemm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void hemm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m, std::int64_t n,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

void syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          float alpha, sycl::buffer<float, 1> &a, std::int64_t lda, float beta,
          sycl::buffer<float, 1> &c, std::int64_t ldc);

void syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          double alpha, sycl::buffer<double, 1> &a, std::int64_t lda, double beta,
          sycl::buffer<double, 1> &c, std::int64_t ldc);

void syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          std::complex<float> beta, sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          std::complex<double> beta, sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

void herk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          float alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda, float beta,
          sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void herk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
          double alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda, double beta,
          sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

void syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           float alpha, sycl::buffer<float, 1> &a, std::int64_t lda, sycl::buffer<float, 1> &b,
           std::int64_t ldb, float beta, sycl::buffer<float, 1> &c, std::int64_t ldc);

void syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           double alpha, sycl::buffer<double, 1> &a, std::int64_t lda, sycl::buffer<double, 1> &b,
           std::int64_t ldb, double beta, sycl::buffer<double, 1> &c, std::int64_t ldc);

void syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, std::complex<float> beta,
           sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, std::complex<double> beta,
           sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

void her2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           std::complex<float> alpha, sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb, float beta,
           sycl::buffer<std::complex<float>, 1> &c, std::int64_t ldc);

void her2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n, std::int64_t k,
           std::complex<double> alpha, sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb, double beta,
           sycl::buffer<std::complex<double>, 1> &c, std::int64_t ldc);

void trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &b, std::int64_t ldb);

void trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &b, std::int64_t ldb);

void trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);

void trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

void trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, float alpha, sycl::buffer<float, 1> &a, std::int64_t lda,
          sycl::buffer<float, 1> &b, std::int64_t ldb);

void trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, double alpha, sycl::buffer<double, 1> &a,
          std::int64_t lda, sycl::buffer<double, 1> &b, std::int64_t ldb);

void trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, std::complex<float> alpha,
          sycl::buffer<std::complex<float>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<float>, 1> &b, std::int64_t ldb);

void trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans, diag unit_diag,
          std::int64_t m, std::int64_t n, std::complex<double> alpha,
          sycl::buffer<std::complex<double>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<double>, 1> &b, std::int64_t ldb);

// level 3, USM

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, float alpha, const float *a, std::int64_t lda,
                 const float *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, double alpha, const double *a, std::int64_t lda,
                 const double *b, std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, std::complex<float> alpha,
                 const std::complex<float> *a, std::int64_t lda, const std::complex<float> *b,
                 std::int64_t ldb, std::complex<float> beta, std::complex<float> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies = {});

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, std::complex<double> alpha,
                 const std::complex<double> *a, std::int64_t lda, const std::complex<double> *b,
                 std::int64_t ldb, std::complex<double> beta, std::complex<double> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies = {});

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, sycl::half alpha, const sycl::half *a,
                 std::int64_t lda, const sycl::half *b, std::int64_t ldb, sycl::half beta,
                 sycl::half *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, float alpha, const sycl::half *a, std::int64_t lda,
                 const sycl::half *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, float alpha, const bfloat16 *a, std::int64_t lda,
                 const bfloat16 *b, std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, float alpha, const bfloat16 *a, std::int64_t lda,
                 const bfloat16 *b, std::int64_t ldb, float beta, bfloat16 *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, float alpha, const std::int8_t *a,
                 std::int64_t lda, const std::int8_t *b, std::int64_t ldb, float beta,
                 std::int32_t *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event gemm(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                 std::int64_t n, std::int64_t k, float alpha, const std::int8_t *a,
                 std::int64_t lda, const std::int8_t *b, std::int64_t ldb, float beta, float *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies = {});

sycl::event symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, float alpha, const float *a, std::int64_t lda, const float *b,
                 std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, double alpha, const double *a, std::int64_t lda, const double *b,
                 std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                 std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                 std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event symm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                 std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                 std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event hemm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, std::complex<float> alpha, const std::complex<float> *a,
                 std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                 std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event hemm(sycl::queue &queue, side left_right, uplo upper_lower, std::int64_t m,
                 std::int64_t n, std::complex<double> alpha, const std::complex<double> *a,
                 std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                 std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, float alpha, const float *a, std::int64_t lda, float beta,
                 float *c, std::int64_t ldc, const std::vector<sycl::event> &dependencies = {});

sycl::event syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, double alpha, const double *a, std::int64_t lda, double beta,
                 double *c, std::int64_t ldc, const std::vector<sycl::event> &dependencies = {});

sycl::event syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                 std::int64_t lda, std::complex<float> beta, std::complex<float> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies = {});

sycl::event syrk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                 std::int64_t lda, std::complex<double> beta, std::complex<double> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies = {});

sycl::event herk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, float alpha, const std::complex<float> *a, std::int64_t lda,
                 float beta, std::complex<float> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event herk(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                 std::int64_t k, double alpha, const std::complex<double> *a, std::int64_t lda,
                 double beta, std::complex<double> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, float alpha, const float *a, std::int64_t lda, const float *b,
                  std::int64_t ldb, float beta, float *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies = {});

sycl::event syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, double alpha, const double *a, std::int64_t lda, const double *b,
                  std::int64_t ldb, double beta, double *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies = {});

sycl::event syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                  std::int64_t lda, const std::complex<float> *b, std::int64_t ldb,
                  std::complex<float> beta, std::complex<float> *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies = {});

sycl::event syr2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                  std::int64_t lda, const std::complex<double> *b, std::int64_t ldb,
                  std::complex<double> beta, std::complex<double> *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies = {});

sycl::event her2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, std::complex<float> alpha, const std::complex<float> *a,
                  std::int64_t lda, const std::complex<float> *b, std::int64_t ldb, float beta,
                  std::complex<float> *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies = {});

sycl::event her2k(sycl::queue &queue, uplo upper_lower, transpose trans, std::int64_t n,
                  std::int64_t k, std::complex<double> alpha, const std::complex<double> *a,
                  std::int64_t lda, const std::complex<double> *b, std::int64_t ldb, double beta,
                  std::complex<double> *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies = {});

sycl::event trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, float alpha, const float *a,
                 std::int64_t lda, float *b, std::int64_t ldb,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, double alpha, const double *a,
                 std::int64_t lda, double *b, std::int64_t ldb,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float> *a, std::int64_t lda, std::complex<float> *b,
                 std::int64_t ldb, const std::vector<sycl::event> &dependencies = {});

sycl::event trmm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double> *a, std::int64_t lda, std::complex<double> *b,
                 std::int64_t ldb, const std::vector<sycl::event> &dependencies = {});

sycl::event trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, float alpha, const float *a,
                 std::int64_t lda, float *b, std::int64_t ldb,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, double alpha, const double *a,
                 std::int64_t lda, double *b, std::int64_t ldb,
                 const std::vector<sycl::event> &dependencies = {});

sycl::event trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, std::complex<float> alpha,
                 const std::complex<float> *a, std::int64_t lda, std::complex<float> *b,
                 std::int64_t ldb, const std::vector<sycl::event> &dependencies = {});

sycl::event trsm(sycl::queue &queue, side left_right, uplo upper_lower, transpose trans,
                 diag unit_diag, std::int64_t m, std::int64_t n, std::complex<double> alpha,
                 const std::complex<double> *a, std::int64_t lda, std::complex<double> *b,
                 std::int64_t ldb, const std::vector<sycl::event> &dependencies = {});
