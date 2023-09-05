/*******************************************************************************
* Copyright Codeplay Software
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

void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, real_t alpha, sycl::buffer<real_t, 1> &a,
          std::int64_t lda, sycl::buffer<real_t, 1> &b, std::int64_t ldb, real_t beta,
          sycl::buffer<real_t, 1> &c, std::int64_t ldc) {
    CALL_PORTBLAS_FN(::blas::_gemm, queue, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                     ldc);
}

void gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
          std::int64_t m, std::int64_t n, std::int64_t k, std::complex<real_t> alpha,
          sycl::buffer<std::complex<real_t>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<real_t>, 1> &b, std::int64_t ldb, std::complex<real_t> beta,
          sycl::buffer<std::complex<real_t>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "gemm", " for complex");
}

void symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, real_t alpha, sycl::buffer<real_t, 1> &a,
          std::int64_t lda, sycl::buffer<real_t, 1> &b, std::int64_t ldb, real_t beta,
          sycl::buffer<real_t, 1> &c, std::int64_t ldc) {
    CALL_PORTBLAS_FN(::blas::_symm, queue, left_right, upper_lower, m, n, alpha, a, lda, b, ldb,
                     beta, c, ldc);
}

void symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<real_t> alpha,
          sycl::buffer<std::complex<real_t>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<real_t>, 1> &b, std::int64_t ldb, std::complex<real_t> beta,
          sycl::buffer<std::complex<real_t>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "symm", "");
}

void hemm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          std::int64_t m, std::int64_t n, std::complex<real_t> alpha,
          sycl::buffer<std::complex<real_t>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<real_t>, 1> &b, std::int64_t ldb, std::complex<real_t> beta,
          sycl::buffer<std::complex<real_t>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "hemm", "");
}

void syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, real_t alpha, sycl::buffer<real_t, 1> &a,
          std::int64_t lda, real_t beta, sycl::buffer<real_t, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "syrk", "");
}

void syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, std::complex<real_t> alpha,
          sycl::buffer<std::complex<real_t>, 1> &a, std::int64_t lda, std::complex<real_t> beta,
          sycl::buffer<std::complex<real_t>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "syrk", "");
}

void herk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
          std::int64_t n, std::int64_t k, real_t alpha, sycl::buffer<std::complex<real_t>, 1> &a,
          std::int64_t lda, real_t beta, sycl::buffer<std::complex<real_t>, 1> &c,
          std::int64_t ldc) {
    throw unimplemented("blas", "herk", "");
}

void syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, real_t alpha, sycl::buffer<real_t, 1> &a,
           std::int64_t lda, sycl::buffer<real_t, 1> &b, std::int64_t ldb, real_t beta,
           sycl::buffer<real_t, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "syr2k", "");
}

void syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<real_t> alpha,
           sycl::buffer<std::complex<real_t>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<real_t>, 1> &b, std::int64_t ldb, std::complex<real_t> beta,
           sycl::buffer<std::complex<real_t>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "syr2k", "");
}

void her2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
           std::int64_t n, std::int64_t k, std::complex<real_t> alpha,
           sycl::buffer<std::complex<real_t>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<real_t>, 1> &b, std::int64_t ldb, real_t beta,
           sycl::buffer<std::complex<real_t>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "her2k", "");
}

void trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          real_t alpha, sycl::buffer<real_t, 1> &a, std::int64_t lda, sycl::buffer<real_t, 1> &b,
          std::int64_t ldb) {
    throw unimplemented("blas", "trmm", "");
}

void trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<real_t>, 1> &b, std::int64_t ldb) {
    throw unimplemented("blas", "trmm", "");
}

void trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          real_t alpha, sycl::buffer<real_t, 1> &a, std::int64_t lda, sycl::buffer<real_t, 1> &b,
          std::int64_t ldb) {
    CALL_PORTBLAS_FN(::blas::_trsm, queue, left_right, upper_lower, trans, unit_diag, m, n, alpha,
                     a, lda, b, ldb);
}

void trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
          oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m, std::int64_t n,
          std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1> &a, std::int64_t lda,
          sycl::buffer<std::complex<real_t>, 1> &b, std::int64_t ldb) {
    throw unimplemented("blas", "trsm", " for complex");
}

void gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, real_t alpha,
           sycl::buffer<real_t, 1> &a, std::int64_t lda, sycl::buffer<real_t, 1> &b,
           std::int64_t ldb, real_t beta, sycl::buffer<real_t, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "gemmt", "");
}

void gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
           oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k,
           std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1> &a, std::int64_t lda,
           sycl::buffer<std::complex<real_t>, 1> &b, std::int64_t ldb, std::complex<real_t> beta,
           sycl::buffer<std::complex<real_t>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "gemmt", "");
}

void omatcopy(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, real_t alpha,
              sycl::buffer<real_t, 1> &a, std::int64_t lda, sycl::buffer<real_t, 1> &b,
              std::int64_t ldb) {
    CALL_PORTBLAS_FN(::blas::_omatcopy, queue, trans, m, n, alpha, a, lda, b, ldb);
}

void omatcopy(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
              std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1> &a,
              std::int64_t lda, sycl::buffer<std::complex<real_t>, 1> &b, std::int64_t ldb) {
    throw unimplemented("blas", "omatcopy", "");
}

void imatcopy(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n, real_t alpha,
              sycl::buffer<real_t, 1> &ab, std::int64_t lda, std::int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "");
}

void imatcopy(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
              std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1> &ab,
              std::int64_t lda, std::int64_t ldb) {
    throw unimplemented("blas", "imatcopy", "");
}

void omatadd(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
             real_t alpha, sycl::buffer<real_t, 1> &a, std::int64_t lda, real_t beta,
             sycl::buffer<real_t, 1> &b, std::int64_t ldb, sycl::buffer<real_t, 1> &c,
             std::int64_t ldc) {
    CALL_PORTBLAS_FN(::blas::_omatadd, queue, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c,
                     ldc);
}

void omatadd(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m, std::int64_t n,
             std::complex<real_t> alpha, sycl::buffer<std::complex<real_t>, 1> &a, std::int64_t lda,
             std::complex<real_t> beta, sycl::buffer<std::complex<real_t>, 1> &b, std::int64_t ldb,
             sycl::buffer<std::complex<real_t>, 1> &c, std::int64_t ldc) {
    throw unimplemented("blas", "omatadd", "");
}

// USM APIs

sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, real_t alpha, const real_t *a,
                 std::int64_t lda, const real_t *b, std::int64_t ldb, real_t beta, real_t *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", " for USM");
}

sycl::event gemm(sycl::queue &queue, oneapi::mkl::transpose transa, oneapi::mkl::transpose transb,
                 std::int64_t m, std::int64_t n, std::int64_t k, std::complex<real_t> alpha,
                 const std::complex<real_t> *a, std::int64_t lda, const std::complex<real_t> *b,
                 std::int64_t ldb, std::complex<real_t> beta, std::complex<real_t> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemm", " for USM");
}

sycl::event symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 std::int64_t m, std::int64_t n, real_t alpha, const real_t *a, std::int64_t lda,
                 const real_t *b, std::int64_t ldb, real_t beta, real_t *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "symm", " for USM");
}

sycl::event symm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 std::int64_t m, std::int64_t n, std::complex<real_t> alpha,
                 const std::complex<real_t> *a, std::int64_t lda, const std::complex<real_t> *b,
                 std::int64_t ldb, std::complex<real_t> beta, std::complex<real_t> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "symm", " for USM");
}

sycl::event hemm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 std::int64_t m, std::int64_t n, std::complex<real_t> alpha,
                 const std::complex<real_t> *a, std::int64_t lda, const std::complex<real_t> *b,
                 std::int64_t ldb, std::complex<real_t> beta, std::complex<real_t> *c,
                 std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "hemm", " for USM");
}

sycl::event syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t k, real_t alpha, const real_t *a, std::int64_t lda,
                 real_t beta, real_t *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk", " for USM");
}

sycl::event syrk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t k, std::complex<real_t> alpha,
                 const std::complex<real_t> *a, std::int64_t lda, std::complex<real_t> beta,
                 std::complex<real_t> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syrk", " for USM");
}

sycl::event herk(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                 std::int64_t n, std::int64_t k, real_t alpha, const std::complex<real_t> *a,
                 std::int64_t lda, real_t beta, std::complex<real_t> *c, std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "herk", " for USM");
}

sycl::event syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t k, real_t alpha, const real_t *a, std::int64_t lda,
                  const real_t *b, std::int64_t ldb, real_t beta, real_t *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr2k", " for USM");
}

sycl::event syr2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t k, std::complex<real_t> alpha,
                  const std::complex<real_t> *a, std::int64_t lda, const std::complex<real_t> *b,
                  std::int64_t ldb, std::complex<real_t> beta, std::complex<real_t> *c,
                  std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "syr2k", " for USM");
}

sycl::event her2k(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans,
                  std::int64_t n, std::int64_t k, std::complex<real_t> alpha,
                  const std::complex<real_t> *a, std::int64_t lda, const std::complex<real_t> *b,
                  std::int64_t ldb, real_t beta, std::complex<real_t> *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "her2k", " for USM");
}

sycl::event trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, real_t alpha, const real_t *a, std::int64_t lda, real_t *b,
                 std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trmm", " for USM");
}

sycl::event trmm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, std::complex<real_t> alpha, const std::complex<real_t> *a,
                 std::int64_t lda, std::complex<real_t> *b, std::int64_t ldb,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trmm", " for USM");
}

sycl::event trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, real_t alpha, const real_t *a, std::int64_t lda, real_t *b,
                 std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm", " for USM");
}

sycl::event trsm(sycl::queue &queue, oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower,
                 oneapi::mkl::transpose trans, oneapi::mkl::diag unit_diag, std::int64_t m,
                 std::int64_t n, std::complex<real_t> alpha, const std::complex<real_t> *a,
                 std::int64_t lda, std::complex<real_t> *b, std::int64_t ldb,
                 const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "trsm", " for USM");
}

sycl::event gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                  oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k, real_t alpha,
                  const real_t *a, std::int64_t lda, const real_t *b, std::int64_t ldb, real_t beta,
                  real_t *c, std::int64_t ldc, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", " for USM");
}

sycl::event gemmt(sycl::queue &queue, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose transa,
                  oneapi::mkl::transpose transb, std::int64_t n, std::int64_t k,
                  std::complex<real_t> alpha, const std::complex<real_t> *a, std::int64_t lda,
                  const std::complex<real_t> *b, std::int64_t ldb, std::complex<real_t> beta,
                  std::complex<real_t> *c, std::int64_t ldc,
                  const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "gemmt", " for USM");
}

sycl::event omatcopy(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     real_t alpha, const real_t *a, std::int64_t lda, real_t *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy", "");
}

sycl::event omatcopy(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::complex<real_t> alpha, const std::complex<real_t> *a, std::int64_t lda,
                     std::complex<real_t> *b, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy", "");
}

sycl::event imatcopy(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     real_t alpha, real_t *ab, std::int64_t lda, std::int64_t ldb,
                     const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy", "");
}

sycl::event imatcopy(sycl::queue &queue, transpose trans, std::int64_t m, std::int64_t n,
                     std::complex<real_t> alpha, std::complex<real_t> *ab, std::int64_t lda,
                     std::int64_t ldb, const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy", "");
}

sycl::event omatadd(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                    std::int64_t n, real_t alpha, const real_t *a, std::int64_t lda, real_t beta,
                    const real_t *b, std::int64_t ldb, real_t *c, std::int64_t ldc,
                    const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd", "");
}

sycl::event omatadd(sycl::queue &queue, transpose transa, transpose transb, std::int64_t m,
                    std::int64_t n, std::complex<real_t> alpha, const std::complex<real_t> *a,
                    std::int64_t lda, std::complex<real_t> beta, const std::complex<real_t> *b,
                    std::int64_t ldb, std::complex<real_t> *c, std::int64_t ldc,
                    const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatadd", "");
}
sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           real_t *alpha, const real_t **a, int64_t *lda, real_t **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "");
}

sycl::event omatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<real_t> *alpha, const std::complex<real_t> **a,
                           int64_t *lda, std::complex<real_t> **b, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "omatcopy_batch", "");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           real_t *alpha, real_t **ab, int64_t *lda, int64_t *ldb,
                           int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "");
}

sycl::event imatcopy_batch(sycl::queue &queue, transpose *trans, int64_t *m, int64_t *n,
                           std::complex<real_t> *alpha, std::complex<real_t> **ab, int64_t *lda,
                           int64_t *ldb, int64_t group_count, int64_t *groupsize,
                           const std::vector<sycl::event> &dependencies) {
    throw unimplemented("blas", "imatcopy_batch", "");
}
